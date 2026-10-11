"""
end_to_end_model.py
Full trainable model: all four stream encoders + fusion + trajectory head.
Unlike the pilot phase, encoders are NOT frozen -- gradients flow through
LiDAR/camera/map encoders too.

use_map=True (default): full four-stream model, as trained/evaluated so far.
use_map=False: map stream fully removed (not zeroed) -- for the ablation
study isolating the HD map fusion's contribution. fuse layer's input dim
shrinks accordingly (3*embed_dim instead of 4*embed_dim), so the ablated
model isn't just carrying dead weights for a disabled input.

history_frames=H > 0: batch["history"] [B, H, 2] (observed past positions) is
encoded by history_encoder.py (gru or transformer) to [B, 128] and
concatenated with the CAM embedding inside V2XMessageStream. history_frames=0
(default) builds exactly the model without it.

Map stream is per-town: each town has its own topology graph, so RGCN
output is computed once per town present in a batch, and each sample's
nearby-lane lookup uses its OWN town's lane embeddings. Mixed-town batches
are the normal case (see pilot_dataset.py).

Why the per-sample loops for ego/coop/CPM: LiDAR point clouds and coop
neighbor counts vary in size per sample -- can't be stacked into one batched
tensor. Looping over the batch (size 8) for these specific sub-encoders is
simple and correct; CAM is naturally fixed-size and stays batched.

Run in gpu-test-venv.
"""
import torch
import torch.nn as nn

from ego_stream import EgoStream
from coop_sensor_stream import CoopSensorStream
from v2x_message_stream import V2XMessageStream
from rgcn_map_encoder import load_graph, RGCNMapEncoder, get_nearby_lane_nodes
from map_cross_attention_fusion import MapCrossAttentionFusion
from history_encoder import build_history_encoder

EMBED_DIM = 128
N_MODES = 6
HORIZON = 30
TOWNS = ["Town03", "Town04", "Town05"]


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
        conf_logits = self.conf_mlp(x)  # raw logits -- softmax applied only where needed
        return trajs, conf_logits


class EndToEndFusion(nn.Module):
    def __init__(self, device, embed_dim=EMBED_DIM, n_modes=N_MODES, use_map=True,
                 history_frames=0, history_encoder="gru"):
        super().__init__()
        self.device = device
        self.use_map = use_map
        # history_frames=0: no history encoder, and V2XMessageStream is the original
        # one -- the model (and its state_dict) is identical to run D's.
        self.history_frames = history_frames

        self.ego_encoder = EgoStream(out_dim=embed_dim)
        self.coop_encoder = CoopSensorStream(out_dim=embed_dim)
        self.v2x_encoder = V2XMessageStream(out_dim=embed_dim, use_history=history_frames > 0)
        if history_frames > 0:
            self.history_encoder = build_history_encoder(history_encoder, embed_dim, history_frames)

        if self.use_map:
            # Per-town graphs -- registered as buffers with per-town names so
            # .to(device) moves all of them automatically.
            for town in TOWNS:
                data, id_to_idx, mean, std = load_graph(f"topology/{town}_topology.pkl")
                self.register_buffer(f"graph_x_{town}", data.x)
                self.register_buffer(f"graph_edge_index_{town}", data.edge_index)
                self.register_buffer(f"graph_edge_type_{town}", data.edge_type)

            self.map_encoder = RGCNMapEncoder(in_dim=2, hidden_dim=64, out_dim=128, num_relations=3)
            self.map_fusion = MapCrossAttentionFusion(agent_dim=embed_dim, lane_dim=128, d_model=128)

        n_streams = 4 if use_map else 3
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * n_streams, embed_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.traj_head = TrajectoryHead(embed_dim, n_modes)

    def forward(self, batch):
        B = len(batch["points"])

        # ---------- Ego: per-sample loop (variable-size LiDAR + fixed camera views) ----------
        ego_embeds = []
        for i in range(B):
            points = batch["points"][i].to(self.device)
            images = [img.to(self.device) for img in batch["images"][i]]
            ego_embeds.append(self.ego_encoder(points, images))
        ego_embed = torch.stack(ego_embeds, dim=0)  # [B, 128]

        # ---------- Coop: per-sample loop (variable neighbor count) ----------
        coop_embeds = []
        for i in range(B):
            coop_points = [p.to(self.device) for p in batch["coop_points"][i]]
            coop_images = [[img.to(self.device) for img in imgs] for imgs in batch["coop_images"][i]]
            coop_embeds.append(self.coop_encoder(coop_points, coop_images))
        coop_embed = torch.stack(coop_embeds, dim=0)  # [B, 128]

        # ---------- History: fixed-size [B, H, 2], encoded batched, joins the CAM embedding below ----------
        history_embed = None
        if self.history_frames > 0:
            history_embed = self.history_encoder(batch["history"].to(self.device))  # [B, 128]

        # ---------- V2X: CAM is fixed-size but CPM varies -> loop for consistency/simplicity ----------
        v2x_embeds = []
        for i in range(B):
            cam = batch["cam"][i].to(self.device)
            cpm_objects = batch["cpm_objects"][i].to(self.device)
            hist_i = history_embed[i] if history_embed is not None else None
            v2x_embeds.append(self.v2x_encoder(cam, cpm_objects, hist_i))
        v2x_embed = torch.stack(v2x_embeds, dim=0)  # [B, 128]

        streams = [ego_embed, coop_embed, v2x_embed]

        # ---------- Map: per-town graph, per-sample lookup into ITS OWN town's embeddings ----------
        if self.use_map:
            towns_in_batch = set(batch["town"])
            lane_embeds_by_town = {}
            for town in towns_in_batch:
                x = getattr(self, f"graph_x_{town}")
                edge_index = getattr(self, f"graph_edge_index_{town}")
                edge_type = getattr(self, f"graph_edge_type_{town}")
                lane_embeds_by_town[town] = self.map_encoder(x, edge_index, edge_type)  # [num_nodes_town, 128]

            nearby_idx = batch["nearby_idx"].to(self.device)  # [B, k]
            nearby_embeds = torch.stack([
                lane_embeds_by_town[batch["town"][i]][nearby_idx[i]]
                for i in range(B)
            ], dim=0)  # [B, k, 128]
            map_embed = self.map_fusion(ego_embed, nearby_embeds)  # [B, 128]
            streams.append(map_embed)

        # ---------- Fusion + trajectory head ----------
        x = torch.cat(streams, dim=-1)  # [B, n_streams*128]
        fused = self.fuse(x)
        trajs, conf_logits = self.traj_head(fused)
        return fused, trajs, conf_logits