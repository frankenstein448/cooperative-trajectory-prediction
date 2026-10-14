"""
env_model.py
EnvFusion: per-target trajectory prediction for the env task (runs/DESIGN_ENV.md section 4). d = 128,
4 heads, pre-LayerNorm, dropout 0.1.

1. Scene tokens: D's EgoStream (ego LiDAR + 3 cameras) -> 1 token; D's CAMEncoder on the ego's CAM -> 1 token;
   with v2x: D's CoopSensorStream (mean over the coop CAVs) -> 1 token, or a learned "no coop" token when
   no coop CAV is in range. Each gets a learned type embedding. (D's pooled encoders, unchanged.)
2. Target query: GRU history encoder (history_encoder.HistoryEncoder, D's GRU) on t_hist + MLP(t_state), summed.
3. N_BLOCKS blocks, each (pre-LN, residual):
   a. interaction: attention over the other known targets and the ego (an agent at the ego-frame origin,
      token = Linear(CAM embedding)). The pose of key j in query i's frame (dx, dy, cos, sin dyaw, distance)
      goes through an MLP and is added to j's key and value: agent-centric, no absolute coordinates.
   b. map (use_map only): attention over the target's own lane-piece tokens (env_map.py; PolylineEncoder:
      per-point MLP, masked max over the piece's points) plus a learned "no lane" token that is always present.
   c. scene: attention over the scene tokens.
   d. FFN.
4. Head: D's TrajectoryHead(128, 6 modes, 30 steps) per target -> trajs [B, T, 6, 30, 2] (target's own frame)
   and mode logits [B, T, 6].
use_map=False removes the polyline encoder, the "no lane" token and every map attention (not zeroed).
env_wta_loss: D's WinnerTakesAllLoss on the real targets of the batch, flattened to [N, ...].
"""
import math

import torch
import torch.nn as nn

from ego_stream import EgoStream
from coop_sensor_stream import CoopSensorStream
from v2x_message_stream import CAMEncoder
from history_encoder import HistoryEncoder
from end_to_end_model import TrajectoryHead
from env_dataset import STATE_DIM, POS_SCALE_M
from env_map import N_POINT_FEATS

D_MODEL = 128
N_HEADS = 4
N_BLOCKS = 2
DROPOUT = 0.1
PAIR_DIM = 5
SCENE_TYPES = ("ego_sensors", "ego_cam", "coop")


def rel_pose_features(pose_q, pose_k):
    """pose_q [B, T, 3], pose_k [B, J, 3] (x, y, yaw in one shared frame) -> [B, T, J, PAIR_DIM]:
    key j in query i's frame: dx / 50, dy / 50, cos(dyaw), sin(dyaw), distance / 50."""
    dx = pose_k[:, None, :, 0] - pose_q[:, :, None, 0]
    dy = pose_k[:, None, :, 1] - pose_q[:, :, None, 1]
    c, s = torch.cos(pose_q[..., 2])[..., None], torch.sin(pose_q[..., 2])[..., None]
    lx, ly = dx * c + dy * s, -dx * s + dy * c
    dyaw = pose_k[:, None, :, 2] - pose_q[:, :, None, 2]
    return torch.stack([lx / POS_SCALE_M, ly / POS_SCALE_M, torch.cos(dyaw), torch.sin(dyaw),
                        torch.hypot(lx, ly) / POS_SCALE_M], dim=-1)


class RelPoseAttention(nn.Module):
    """Multi-head attention whose keys / values get an additive term from pairwise relative-pose features."""
    def __init__(self, d=D_MODEL, n_heads=N_HEADS, pair_dim=PAIR_DIM, dropout=DROPOUT):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.pair = nn.Sequential(nn.Linear(pair_dim, d), nn.ReLU(), nn.Linear(d, 2 * d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x_q, x_k, pair, key_mask):
        """x_q [B, T, d], x_k [B, J, d], pair [B, T, J, pair_dim], key_mask [B, J] (True = attend) -> [B, T, d]."""
        B, T, d = x_q.shape
        J = x_k.shape[1]
        pk, pv = self.pair(pair).chunk(2, dim=-1)                              # [B, T, J, d] each
        q = self.q(x_q).view(B, T, self.h, self.dh)
        k = (self.k(x_k)[:, None] + pk).view(B, T, J, self.h, self.dh)
        v = (self.v(x_k)[:, None] + pv).view(B, T, J, self.h, self.dh)
        att = torch.einsum("bthd,btjhd->bthj", q, k) / math.sqrt(self.dh)
        att = att.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        att = self.drop(torch.softmax(att, dim=-1))
        out = torch.einsum("bthj,btjhd->bthd", att, v).reshape(B, T, d)
        return self.o(out)


class PolylineEncoder(nn.Module):
    """Lane piece [P, N_POINT_FEATS] -> [d]: per-point MLP, max over the piece's real points."""
    def __init__(self, d=D_MODEL, in_dim=N_POINT_FEATS, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, d), nn.ReLU(),
                                 nn.Linear(d, d))

    def forward(self, feats, point_mask):
        """feats [..., P, F], point_mask [..., P] -> [..., d] (zeros for a piece without points)."""
        x = self.mlp(feats).masked_fill(~point_mask[..., None], float("-inf"))
        x = x.max(dim=-2).values
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))


class EnvBlock(nn.Module):
    def __init__(self, d=D_MODEL, n_heads=N_HEADS, use_map=True, dropout=DROPOUT):
        super().__init__()
        self.use_map = use_map
        self.ln_int = nn.LayerNorm(d)
        self.ln_key = nn.LayerNorm(d)
        self.inter = RelPoseAttention(d, n_heads, dropout=dropout)
        if use_map:
            self.ln_map = nn.LayerNorm(d)
            self.map_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln_scene = nn.LayerNorm(d)
        self.scene_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, ego_tok, pair, key_mask, scene, map_tok=None, map_mask=None):
        """h [B, T, d] targets; ego_tok [B, 1, d]; pair [B, T, T + 1, PAIR_DIM]; key_mask [B, T + 1];
        scene [B, S, d]; map_tok [B, T, K + 1, d], map_mask [B, T, K + 1] (True = real)."""
        B, T, d = h.shape
        x = self.ln_int(h)
        keys = self.ln_key(torch.cat([h, ego_tok], dim=1))
        h = h + self.drop(self.inter(x, keys, pair, key_mask))
        if self.use_map:
            q = self.ln_map(h).reshape(B * T, 1, d)
            kv = map_tok.reshape(B * T, -1, d)
            out, _ = self.map_attn(q, kv, kv, key_padding_mask=~map_mask.reshape(B * T, -1), need_weights=False)
            h = h + self.drop(out.reshape(B, T, d))
        out, _ = self.scene_attn(self.ln_scene(h), scene, scene, need_weights=False)
        h = h + self.drop(out)
        return h + self.drop(self.ffn(self.ln_ffn(h)))


class EnvFusion(nn.Module):
    def __init__(self, device, v2x=False, use_map=True, d=D_MODEL, n_blocks=N_BLOCKS, n_heads=N_HEADS):
        super().__init__()
        self.device, self.v2x, self.use_map = device, v2x, use_map
        self.ego_encoder = EgoStream(out_dim=d)
        self.cam_encoder = CAMEncoder(out_dim=d)
        if v2x:
            self.coop_encoder = CoopSensorStream(out_dim=d)
            self.no_coop = nn.Parameter(torch.zeros(d))
        self.scene_type = nn.Embedding(len(SCENE_TYPES), d)
        self.ego_agent = nn.Linear(d, d)
        self.hist_encoder = HistoryEncoder(out_dim=d)
        self.state_mlp = nn.Sequential(nn.Linear(STATE_DIM, d), nn.ReLU(), nn.Linear(d, d))
        if use_map:
            self.poly_encoder = PolylineEncoder(d)
            self.no_lane = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(EnvBlock(d, n_heads, use_map) for _ in range(n_blocks))
        self.ln_out = nn.LayerNorm(d)
        self.traj_head = TrajectoryHead(d)

    # ---------- scene part: D's per-sample encoders ----------
    def scene_tokens(self, batch):
        """-> scene [B, S, d] (S = 2, or 3 with v2x), cam_embed [B, d]."""
        B = len(batch["points"])
        ego = torch.stack([self.ego_encoder(batch["points"][i].to(self.device),
                                            [img.to(self.device) for img in batch["images"][i]]) for i in range(B)])
        cam = self.cam_encoder(torch.stack([c.to(self.device) for c in batch["cam"]]))
        toks = [ego + self.scene_type.weight[0], cam + self.scene_type.weight[1]]
        if self.v2x:
            coop = []
            for i in range(B):
                pts = batch["coop_points"][i]
                if len(pts) == 0:
                    coop.append(self.no_coop)
                else:
                    coop.append(self.coop_encoder([p.to(self.device) for p in pts],
                                                  [[im.to(self.device) for im in ims] for ims in batch["coop_images"][i]]))
            toks.append(torch.stack(coop) + self.scene_type.weight[2])
        return torch.stack(toks, dim=1), cam

    # ---------- target part: plain tensors (unit-testable) ----------
    def core(self, scene, cam_embed, t_hist, t_state, t_pose, target_mask, t_map=None, t_map_pt_mask=None,
             t_map_piece_mask=None):
        """scene [B, S, d], cam_embed [B, d], t_hist [B, T, 8, 2], t_state [B, T, STATE_DIM], t_pose [B, T, 3],
        target_mask [B, T]; map tensors [B, T, K, P, F] / [B, T, K, P] / [B, T, K] (use_map)
        -> trajs [B, T, 6, 30, 2], logits [B, T, 6] (padded targets give finite, unused outputs)."""
        B, T = target_mask.shape
        d = scene.shape[-1]
        h = self.hist_encoder(t_hist.reshape(B * T, *t_hist.shape[2:])).view(B, T, d) + self.state_mlp(t_state)
        ego_tok = self.ego_agent(cam_embed)[:, None]                                       # [B, 1, d]
        ego_pose = torch.zeros(B, 1, 3, device=t_pose.device, dtype=t_pose.dtype)
        pair = rel_pose_features(t_pose, torch.cat([t_pose, ego_pose], dim=1))              # [B, T, T + 1, 5]
        key_mask = torch.cat([target_mask, torch.ones(B, 1, dtype=torch.bool, device=target_mask.device)], dim=1)
        map_tok = map_mask = None
        if self.use_map:
            K = t_map.shape[2]
            pieces = self.poly_encoder(t_map, t_map_pt_mask)                               # [B, T, K, d]
            map_tok = torch.cat([self.no_lane.expand(B, T, 1, d), pieces], dim=2)          # [B, T, K + 1, d]
            map_mask = torch.cat([torch.ones(B, T, 1, dtype=torch.bool, device=t_map.device), t_map_piece_mask], dim=2)
            assert map_mask.shape[2] == K + 1
        for blk in self.blocks:
            h = blk(h, ego_tok, pair, key_mask, scene, map_tok, map_mask)
        trajs, logits = self.traj_head(self.ln_out(h).reshape(B * T, d))
        return trajs.view(B, T, *trajs.shape[1:]), logits.view(B, T, -1)

    def forward(self, batch):
        scene, cam = self.scene_tokens(batch)
        dev = lambda k: batch[k].to(self.device)
        maps = {k: dev(k) for k in ("t_map", "t_map_pt_mask", "t_map_piece_mask")} if self.use_map else {}
        return self.core(scene, cam, dev("t_hist"), dev("t_state"), dev("t_pose"), dev("target_mask"), **maps)


def env_wta_loss(loss_fn, trajs, logits, t_fut, t_fut_mask, target_mask):
    """D's loss (e.g. trajectory_loss.WinnerTakesAllLoss) on the real targets only, flattened to [N, ...]:
    each target weighs the same; padded targets get no gradient."""
    m = target_mask
    assert m.any(), "batch without a known target"
    return loss_fn(trajs[m], logits[m], t_fut[m], t_fut_mask[m])
