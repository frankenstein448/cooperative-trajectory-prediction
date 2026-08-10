"""
four_stream_fusion.py
Full real integration: ego (LiDAR+camera) + RSU/CAV coop (real neighbor
lookup) + V2X (CAM/CPM) + HD map (RGCN + cross-attention) -> fusion ->
trajectory head.

Verified tuple: agent_id=4, scene=100, frame=6 (Town05).
Run in gpu-test-venv.
"""
import torch
import torch.nn as nn

from ego_stream import EgoStream
from ego_sensor_encoder import load_semantic_lidar
from ego_camera_encoder import load_image
from coop_sensor_stream import CoopSensorStream
from v2x_message_stream import V2XMetadataLoader, V2XMessageStream
from rgcn_map_encoder import load_graph, RGCNMapEncoder, get_nearby_lane_nodes
from map_cross_attention_fusion import MapCrossAttentionFusion, gather_nearby_lane_embeddings
from coord_transform import v2xsim_to_carla

EMBED_DIM = 128
N_MODES = 6
HORIZON = 30
AGENT_ID, SCENE, FRAME = 4, 100, 6
TOWN = "Town05"


class TrajectoryHead(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_modes=N_MODES, horizon=HORIZON):
        super().__init__()
        self.n_modes = n_modes
        self.horizon = horizon
        self.traj_mlp = nn.Linear(embed_dim, n_modes * horizon * 2)
        self.conf_mlp = nn.Linear(embed_dim, n_modes)

    def forward(self, x):
        n = x.shape[0]
        trajs = self.traj_mlp(x).view(n, self.n_modes, self.horizon, 2)
        conf_logits = self.conf_mlp(x)  # raw logits -- softmax applied only where probabilities are needed
        return trajs, conf_logits


class FourStreamFusion(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_modes=N_MODES):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.traj_head = TrajectoryHead(embed_dim, n_modes)

    def forward(self, ego, coop, v2x, map_):
        x = torch.cat([ego, coop, v2x, map_], dim=-1)  # [N, 4*embed_dim]
        fused = self.fuse(x)
        trajs, conf = self.traj_head(fused)
        return fused, trajs, conf


if __name__ == "__main__":
    # ---------- Ego stream (real) ----------
    lidar_path = f"/home/karra/V2X-Sim-2.0/sweeps/SEMLIDAR_TOP_id_{AGENT_ID}/scene_{SCENE}_{FRAME:06d}.pcd.bin"
    points = load_semantic_lidar(lidar_path)

    cam_base = "/home/karra/V2X-Sim-2.0/sweeps/{view}_id_" + str(AGENT_ID) + "/scene_" + str(SCENE) + f"_{FRAME:06d}.jpg"
    views = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    images = [load_image(cam_base.format(view=v)) for v in views]

    ego_encoder = EgoStream()
    ego_embed = ego_encoder(points, images).unsqueeze(0)  # [1, 128]
    del points, images  # free raw sensor data, only need the embedding now

    # ---------- Coop stream (real neighbors) ----------
    from find_coop_neighbors import find_coop_neighbors
    from v2x_message_stream import V2XMetadataLoader as _NeighborLoader  # reuse below

    neighbor_loader = _NeighborLoader()
    neighbor_ids = find_coop_neighbors(neighbor_loader, ego_agent_id=AGENT_ID, scene=SCENE, frame=FRAME)

    coop_points, coop_images = [], []
    for nid in neighbor_ids:
        n_lidar_path = f"/home/karra/V2X-Sim-2.0/sweeps/SEMLIDAR_TOP_id_{nid}/scene_{SCENE}_{FRAME:06d}.pcd.bin"
        coop_points.append(load_semantic_lidar(n_lidar_path))

        n_cam_base = "/home/karra/V2X-Sim-2.0/sweeps/{view}_id_" + str(nid) + "/scene_" + str(SCENE) + f"_{FRAME:06d}.jpg"
        coop_images.append([load_image(n_cam_base.format(view=v)) for v in views])

    coop_encoder = CoopSensorStream()
    coop_embed = coop_encoder(coop_points, coop_images).unsqueeze(0)  # [1, 128]
    print(f"Coop neighbors used: {neighbor_ids}")
     # ---------- Map stream (real) ----------
    data, id_to_idx, mean, std = load_graph(f"topology/{TOWN}_topology.pkl")

    # ---------- V2X stream (real) ----------
    loader = V2XMetadataLoader()
    cam, _ = loader.get_cam(AGENT_ID, SCENE, FRAME)
    cpm_objects = loader.get_cpm(AGENT_ID, SCENE, FRAME)
    v2x_encoder = V2XMessageStream()
    v2x_embed = v2x_encoder(cam, cpm_objects).unsqueeze(0)  # [1, 128]

    # ---------- Map stream (real) ----------
    data, id_to_idx, mean, std = load_graph(f"topology/{TOWN}_topology.pkl")
    map_encoder = RGCNMapEncoder(in_dim=2, hidden_dim=64, out_dim=128, num_relations=3)
    lane_embeds = map_encoder(data.x, data.edge_index, data.edge_type)  # [num_nodes, 128]

    ego_x, ego_y = cam[0].item(), cam[1].item()  # V2X-Sim world frame
    carla_x, carla_y = v2xsim_to_carla(ego_x, ego_y)
    agent_pos = torch.tensor([[carla_x, carla_y]], dtype=torch.float32)
    agent_pos_norm = (agent_pos - mean) / std  # match load_graph's normalization

    k = 5
    nearby_idx = get_nearby_lane_nodes(agent_pos_norm, data.x, k=k)  # [k] -- collapses batch dim for B=1
    nearby_embeds = gather_nearby_lane_embeddings(lane_embeds, nearby_idx)  # [k, 128]
    nearby_embeds = nearby_embeds.unsqueeze(0)  # [1, k, 128] -- restore batch dim for MapCrossAttentionFusion
    print(f"agent_pos_norm shape: {agent_pos_norm.shape}")
    print(f"nearby_idx shape: {nearby_idx.shape}")
    print(f"nearby_embeds shape: {nearby_embeds.shape}")
    print(f"ego_embed shape going into fusion: {ego_embed.shape}")
    map_fusion = MapCrossAttentionFusion(agent_dim=EMBED_DIM, lane_dim=128, d_model=128)
    map_embed = map_fusion(ego_embed, nearby_embeds)  # [1, 128] -- queried with ego embedding

    # ---------- Four-stream fusion ----------
    fusion = FourStreamFusion()
    fused, trajs, conf = fusion(ego_embed, coop_embed, v2x_embed, map_embed)

    print(f"Ego embed:    {ego_embed.shape}")
    print(f"Coop embed:   {coop_embed.shape}")
    print(f"V2X embed:    {v2x_embed.shape}")
    print(f"Map embed:    {map_embed.shape}")
    print(f"Fused:        {fused.shape}")
    print(f"Trajectories: {trajs.shape}")
    print(f"Confidence:   {conf.shape}")