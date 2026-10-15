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

use_cpm=False / use_coop=False (V2X ablation): the CPM encoder / the cooperative
sensor stream is removed the same way as the map, not zeroed. All defaults
(use_map=True, history_frames=0, use_cpm=True, use_coop=True, predict_scale=False)
give run D's model.

map_version="v2" (runs/DECISION_RULES_MAPV2_EGO.md; with use_map): the map stream uses the env task's lane-piece
tokens (env_map.py, batch["map_tokens"] / "map_pt_mask" / "map_piece_mask", ego frame) instead of the RGCN lane
graph: env_model.PolylineEncoder per piece, plus a learned "no lane" token that is always present, then the same
MapCrossAttentionFusion (query = ego embedding) -> [B, 128] map stream. No graph buffers, no RGCN; the fuse layer
is unchanged (4 streams). map_version="v1" (default) builds exactly the RGCN model.

predict_scale=True (only for --loss laplace_mixture): the trajectory head gets one
extra Linear (scale_mlp, same shape as traj_mlp) giving a per-mode, per-step, per-
coordinate Laplace scale b = softplus(raw) + LAPLACE_SCALE_FLOOR, and forward()
returns a 4th output, scales [B, n_modes, horizon, 2]. Every other parameter and
state_dict key is D's.

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
import torch.nn.functional as F

from ego_stream import EgoStream
from coop_sensor_stream import CoopSensorStream
from v2x_message_stream import V2XMessageStream
from rgcn_map_encoder import load_graph, RGCNMapEncoder, get_nearby_lane_nodes
from map_cross_attention_fusion import MapCrossAttentionFusion
from history_encoder import build_history_encoder
from trajectory_loss import LAPLACE_SCALE_FLOOR

EMBED_DIM = 128
N_MODES = 6
HORIZON = 30
TOWNS = ["Town03", "Town04", "Town05"]


class TrajectoryHead(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_modes=N_MODES, horizon=HORIZON, predict_scale=False):
        super().__init__()
        self.n_modes = n_modes
        self.horizon = horizon
        self.predict_scale = predict_scale
        self.traj_mlp = nn.Linear(embed_dim, n_modes * horizon * 2)
        self.conf_mlp = nn.Linear(embed_dim, n_modes)
        if predict_scale:
            # forked RNG: the scale head's init does not shift the global stream, so with the same
            # --seed the rest of the run (data shuffling) matches D's run exactly
            with torch.random.fork_rng(devices=[]):
                self.scale_mlp = nn.Linear(embed_dim, n_modes * horizon * 2)

    def forward(self, x):
        n = x.shape[0]
        trajs = self.traj_mlp(x).view(n, self.n_modes, self.horizon, 2)
        conf_logits = self.conf_mlp(x)  # raw logits -- softmax applied only where needed
        if not self.predict_scale:
            return trajs, conf_logits
        scales = F.softplus(self.scale_mlp(x)).view(n, self.n_modes, self.horizon, 2) + LAPLACE_SCALE_FLOOR
        return trajs, conf_logits, scales


class EndToEndFusion(nn.Module):
    def __init__(self, device, embed_dim=EMBED_DIM, n_modes=N_MODES, use_map=True,
                 history_frames=0, history_encoder="gru", use_cpm=True, use_coop=True,
                 predict_scale=False, map_version="v1"):
        super().__init__()
        self.device = device
        self.use_map = use_map
        assert map_version in ("v1", "v2") and (use_map or map_version == "v1"), \
            f"map_version={map_version!r} (v2 needs use_map)"
        self.map_version = map_version
        # V2X ablation, by removal like use_map: use_cpm=False drops V2XMessageStream's
        # cpm_enc (its fuse input shrinks); use_coop=False drops coop_encoder (the fuse
        # below loses a stream). The ego's own CAM is always kept.
        self.use_cpm = use_cpm
        self.use_coop = use_coop
        # history_frames=0: no history encoder, and V2XMessageStream is the original
        # one -- the model (and its state_dict) is identical to run D's.
        self.history_frames = history_frames

        self.ego_encoder = EgoStream(out_dim=embed_dim)
        if use_coop:
            self.coop_encoder = CoopSensorStream(out_dim=embed_dim)
        self.v2x_encoder = V2XMessageStream(out_dim=embed_dim, use_history=history_frames > 0, use_cpm=use_cpm)
        if history_frames > 0:
            self.history_encoder = build_history_encoder(history_encoder, embed_dim, history_frames)

        if self.use_map and map_version == "v2":
            from env_model import PolylineEncoder  # env_model imports this module; import here, not at the top
            self.poly_encoder = PolylineEncoder(d=128)
            self.no_lane = nn.Parameter(torch.zeros(128))  # as env_model's "no lane" token
            self.map_fusion = MapCrossAttentionFusion(agent_dim=embed_dim, lane_dim=128, d_model=128)
        elif self.use_map:
            # Per-town graphs -- registered as buffers with per-town names so
            # .to(device) moves all of them automatically.
            for town in TOWNS:
                data, id_to_idx, mean, std = load_graph(f"topology/{town}_topology.pkl")
                self.register_buffer(f"graph_x_{town}", data.x)
                self.register_buffer(f"graph_edge_index_{town}", data.edge_index)
                self.register_buffer(f"graph_edge_type_{town}", data.edge_type)

            self.map_encoder = RGCNMapEncoder(in_dim=2, hidden_dim=64, out_dim=128, num_relations=3)
            self.map_fusion = MapCrossAttentionFusion(agent_dim=embed_dim, lane_dim=128, d_model=128)

        n_streams = 2 + int(use_coop) + int(use_map)  # ego + v2x always
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * n_streams, embed_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.traj_head = TrajectoryHead(embed_dim, n_modes, predict_scale=predict_scale)

    def forward(self, batch):
        B = len(batch["points"])

        # ---------- Ego: per-sample loop (variable-size LiDAR + fixed camera views) ----------
        ego_embeds = []
        for i in range(B):
            points = batch["points"][i].to(self.device)
            images = [img.to(self.device) for img in batch["images"][i]]
            ego_embeds.append(self.ego_encoder(points, images))
        ego_embed = torch.stack(ego_embeds, dim=0)  # [B, 128]

        streams = [ego_embed]

        # ---------- Coop: per-sample loop (variable neighbor count) ----------
        if self.use_coop:
            coop_embeds = []
            for i in range(B):
                coop_points = [p.to(self.device) for p in batch["coop_points"][i]]
                coop_images = [[img.to(self.device) for img in imgs] for imgs in batch["coop_images"][i]]
                coop_embeds.append(self.coop_encoder(coop_points, coop_images))
            streams.append(torch.stack(coop_embeds, dim=0))  # [B, 128]

        # ---------- History: fixed-size [B, H, 2], encoded batched, joins the CAM embedding below ----------
        history_embed = None
        if self.history_frames > 0:
            history_embed = self.history_encoder(batch["history"].to(self.device))  # [B, 128]

        # ---------- V2X: CAM is fixed-size but CPM varies -> loop for consistency/simplicity ----------
        v2x_embeds = []
        for i in range(B):
            cam = batch["cam"][i].to(self.device)
            cpm_objects = batch["cpm_objects"][i].to(self.device) if self.use_cpm else None
            hist_i = history_embed[i] if history_embed is not None else None
            v2x_embeds.append(self.v2x_encoder(cam, cpm_objects, hist_i))
        streams.append(torch.stack(v2x_embeds, dim=0))  # [B, 128]

        # ---------- Map: per-town graph, per-sample lookup into ITS OWN town's embeddings ----------
        if self.use_map and self.map_version == "v2":
            pieces = self.poly_encoder(batch["map_tokens"].to(self.device),
                                       batch["map_pt_mask"].to(self.device))                    # [B, K, 128]
            tokens = torch.cat([self.no_lane.view(1, 1, -1).expand(B, 1, -1), pieces], dim=1)   # [B, K + 1, 128]
            real = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=self.device),
                              batch["map_piece_mask"].to(self.device)], dim=1)                  # [B, K + 1]
            streams.append(self.map_fusion(ego_embed, tokens, key_padding_mask=~real))         # [B, 128]
        elif self.use_map:
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
        # (trajs, conf_logits) or, with predict_scale, (trajs, conf_logits, scales)
        return (fused, *self.traj_head(fused))