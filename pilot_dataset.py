"""
pilot_dataset.py
PyTorch Dataset scaled to all Town05 scenes available in our LiDAR data
(38 scenes, agents 1-4, camera-compatible). Town03/Town04 scenes excluded --
coordinate transform (coord_transform.py) is currently Town05-only.

All model objects (encoders, RGCN, map graph) are built ONCE in __init__ and
reused across every sample. Frozen-encoder training: no gradients tracked
during data loading (see torch.no_grad() in __getitem__).

Run in gpu-test-venv.
"""
import re
import glob
import torch
from torch.utils.data import Dataset

from v2x_message_stream import V2XMetadataLoader, V2XMessageStream
from ground_truth_extractor import get_future_trajectory, HORIZON
from ego_stream import EgoStream
from ego_sensor_encoder import load_semantic_lidar
from ego_camera_encoder import load_image
from coop_sensor_stream import CoopSensorStream
from find_coop_neighbors import find_coop_neighbors
from rgcn_map_encoder import load_graph, RGCNMapEncoder, get_nearby_lane_nodes
from map_cross_attention_fusion import MapCrossAttentionFusion, gather_nearby_lane_embeddings
from coord_transform import v2xsim_to_carla

SWEEPS_ROOT = "/home/karra/V2X-Sim-2.0/sweeps"
CAMERA_COMPATIBLE_AGENTS = [1, 2, 3, 4]
TOWN = "Town05"
EMBED_DIM = 128
VIEWS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scenes confirmed as Town05 (from scene.json descriptions), intersected with
# scenes we actually have LiDAR data for (54 total available, 38 are Town05).
TOWN05_SCENES = [
    3, 14, 15, 16, 21, 22, 27, 32, 34, 35, 36, 37, 38, 39, 41, 42, 43, 45, 46,
    49, 50, 51, 52, 55, 59, 62, 65, 68, 71, 72, 73, 76, 78, 86, 89, 91, 95, 100,
]


def _available_frames(agent_id: int, scene: int) -> set:
    pattern = f"{SWEEPS_ROOT}/SEMLIDAR_TOP_id_{agent_id}/scene_{scene}_*.pcd.bin"
    frames = set()
    for path in glob.glob(pattern):
        m = re.search(r"scene_\d+_(\d+)\.pcd\.bin", path)
        if m:
            frames.add(int(m.group(1)))
    return frames


class PilotDataset(Dataset):
    def __init__(self, scenes=TOWN05_SCENES, agent_ids=CAMERA_COMPATIBLE_AGENTS, horizon: int = HORIZON):
        self.horizon = horizon

        # ---------- built ONCE, shared across all samples ----------
        self.loader = V2XMetadataLoader()

        self.ego_encoder = EgoStream().to(DEVICE)
        self.coop_encoder = CoopSensorStream().to(DEVICE)
        self.v2x_encoder = V2XMessageStream().to(DEVICE)

        self.data, self.id_to_idx, self.mean, self.std = load_graph(f"topology/{TOWN}_topology.pkl")
        self.data.x = self.data.x.to(DEVICE)
        self.data.edge_index = self.data.edge_index.to(DEVICE)
        self.data.edge_type = self.data.edge_type.to(DEVICE)

        self.map_encoder = RGCNMapEncoder(in_dim=2, hidden_dim=64, out_dim=128, num_relations=3).to(DEVICE)
        self.lane_embeds = self.map_encoder(self.data.x, self.data.edge_index, self.data.edge_type)
        self.map_fusion = MapCrossAttentionFusion(agent_dim=EMBED_DIM, lane_dim=128, d_model=128).to(DEVICE)

        self.mean = self.mean.to(DEVICE) if torch.is_tensor(self.mean) else self.mean
        self.std = self.std.to(DEVICE) if torch.is_tensor(self.std) else self.std
        # ------------------------------------------------------------

        self.samples = []  # (agent_id, scene, frame)
        for scene in scenes:
            for agent_id in agent_ids:
                frames = _available_frames(agent_id, scene)
                for frame in sorted(frames):
                    if all((frame + step) in frames for step in range(1, horizon + 1)):
                        self.samples.append((agent_id, scene, frame))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        agent_id, scene, frame = self.samples[idx]

        with torch.no_grad():
            lidar_path = f"{SWEEPS_ROOT}/SEMLIDAR_TOP_id_{agent_id}/scene_{scene}_{frame:06d}.pcd.bin"
            points = load_semantic_lidar(lidar_path).to(DEVICE)

            cam_base = SWEEPS_ROOT + "/{view}_id_" + str(agent_id) + "/scene_" + str(scene) + f"_{frame:06d}.jpg"
            images = [load_image(cam_base.format(view=v)).to(DEVICE) for v in VIEWS]

            ego_embed = self.ego_encoder(points, images).unsqueeze(0)

            cam, _ = self.loader.get_cam(agent_id, scene, frame)
            cpm_objects = self.loader.get_cpm(agent_id, scene, frame)
            cam = cam.to(DEVICE)
            cpm_objects = cpm_objects.to(DEVICE)
            v2x_embed = self.v2x_encoder(cam, cpm_objects).unsqueeze(0)

            neighbor_ids = find_coop_neighbors(self.loader, ego_agent_id=agent_id, scene=scene, frame=frame,
                                                candidate_ids=[a for a in CAMERA_COMPATIBLE_AGENTS if a != agent_id])
            coop_points, coop_images = [], []
            for nid in neighbor_ids:
                n_lidar_path = f"{SWEEPS_ROOT}/SEMLIDAR_TOP_id_{nid}/scene_{scene}_{frame:06d}.pcd.bin"
                coop_points.append(load_semantic_lidar(n_lidar_path).to(DEVICE))
                n_cam_base = SWEEPS_ROOT + "/{view}_id_" + str(nid) + "/scene_" + str(scene) + f"_{frame:06d}.jpg"
                coop_images.append([load_image(n_cam_base.format(view=v)).to(DEVICE) for v in VIEWS])
            coop_embed = self.coop_encoder(coop_points, coop_images).unsqueeze(0)

            ego_x, ego_y = cam[0].item(), cam[1].item()
            carla_x, carla_y = v2xsim_to_carla(ego_x, ego_y)
            agent_pos = torch.tensor([[carla_x, carla_y]], dtype=torch.float32, device=DEVICE)
            agent_pos_norm = (agent_pos - self.mean) / self.std
            nearby_idx = get_nearby_lane_nodes(agent_pos_norm, self.data.x, k=5)
            nearby_embeds = gather_nearby_lane_embeddings(self.lane_embeds, nearby_idx).unsqueeze(0)
            map_embed = self.map_fusion(ego_embed, nearby_embeds)

            gt_traj, gt_mask = get_future_trajectory(self.loader, agent_id, scene, frame, self.horizon)

        return {
            "agent_id": agent_id,
            "scene": scene,
            "frame": frame,
            "ego_embed": ego_embed.squeeze(0).cpu(),
            "coop_embed": coop_embed.squeeze(0).cpu(),
            "v2x_embed": v2x_embed.squeeze(0).cpu(),
            "map_embed": map_embed.squeeze(0).cpu(),
            "gt_traj": gt_traj,
            "gt_mask": gt_mask,
        }


if __name__ == "__main__":
    import time

    print(f"Using device: {DEVICE}")
    print("Building dataset (one-time setup)...")
    t0 = time.time()
    dataset = PilotDataset()
    print(f"Setup took {time.time() - t0:.1f}s, {len(dataset)} samples across {len(TOWN05_SCENES)} scenes")

    print("\nFetching 3 samples...")
    for i in range(3):
        t0 = time.time()
        sample = dataset[i]
        elapsed = time.time() - t0
        print(f"  sample {i} (agent {sample['agent_id']}, scene {sample['scene']}, frame {sample['frame']}): {elapsed:.2f}s")