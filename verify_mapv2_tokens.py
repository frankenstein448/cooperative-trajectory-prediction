"""
verify_mapv2_tokens.py
Checks of the ego-task map v2 input (runs/DECISION_RULES_MAPV2_EGO.md) before any C_Dmapv2 training. Uses the
train + tuning samples of split v2 only (test samples are never read). No training; prints RESULT: PASS / FAIL.

1. Frame: tokens are in the frame of gt_traj. For every moving train + tuning ego sample (raw CAM speed
   > MOVING_M_PER_FRAME), the nearest lane point (over all real points of all pieces): distance and the angle
   between the lane direction there and the ego heading (atan2(sin d, cos d) of the token features).
   Pass: median distance <= 1.0 m AND p95 <= 2.5 m (the map-alignment bar of runs/DECISION_RULES_CORRECTED.md)
   AND median |angle| <= 5 deg. Also: the first valid ground-truth step of moving samples points along +x
   (median |atan2(y, x)| <= 5 deg), i.e. x is the ego's heading, as in the tokens.
2. Coverage: share of samples with no lane piece within the radius (reported; only the "no lane" token then).
3. Dataset: PilotDataset(map_version="v2")[i] returns map_tokens / map_pt_mask / map_piece_mask equal to
   map_tokens() and to env_map.LanePolylines.tokens at the ego pose, with the shapes of env_map; every other key
   equals map_version v1's sample; pilot_collate_fn stacks them to [B, ...].
4. Model: EndToEndFusion(map_version="v2") has no graph buffers or RGCN, has poly_encoder / no_lane / map_fusion,
   and the same fuse / trajectory head shapes as v1; a forward + backward on a real batch gives finite outputs
   and nonzero gradients for poly_encoder, no_lane and map_fusion; a batch with every piece masked gives finite
   outputs.

Run in gpu-test-venv:  python verify_mapv2_tokens.py --run-dir runs/C_D_seed0 2>&1 | tee verify_mapv2_tokens.log
(--run-dir: a C_D run, for its flags and cam_norm.json only.)
"""
import os
import json
import math
import random
import argparse

import numpy as np
import torch

from pilot_dataset import PilotDataset, pilot_collate_fn
from end_to_end_model import EndToEndFusion
from env_map import RADIUS_M, K_PIECES, MAX_PIECE_POINTS, N_POINT_FEATS
from scene_split import three_way_scene_split
from ground_truth_extractor import get_future_trajectory  # the gt_traj of PilotDataset.__getitem__
from v2x_message_stream import quaternion_to_yaw

MOVING_M_PER_FRAME = 0.5
N_DATASET_CHECK = 20

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
args = ap.parse_args()
cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
cam_norm = json.load(open(os.path.join(args.run_dir, "cam_norm.json")))
kw = dict(use_image_cache=cfg["use_image_cache"], require_prev_frame=cfg["require_prev_frame"],
          use_speed=cfg["use_speed"], cam_norm=cam_norm, drop_cam_xy=cfg["drop_cam_xy"],
          use_cpm=not cfg["no_cpm"], use_coop=not cfg["no_coop"], map_transform="mirror")
ds2 = PilotDataset(**kw, map_version="v2")
ds1 = PilotDataset(**kw)
train, tuning, _ = three_way_scene_split(ds2, seed=42, split_version="v2")
ids = sorted(train.indices + tuning.indices)
fails = []


def equal(x, y):
    """Recursive equality of sample values (tensors, lists of tensors, ints / strings)."""
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and torch.equal(x, y)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(equal(u, v) for u, v in zip(x, y))
    return x == y


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        fails.append(msg)


# ---------- 1 / 2: frame and coverage ----------
dist, ang, gt_ang, empty = [], [], [], 0
for i in ids:
    agent, scene, frame, town = ds2.samples[i]
    feats, pt_mask, piece_mask = ds2.map_tokens(agent, scene, frame, town)
    if not piece_mask.any():
        empty += 1
        continue
    if ds2.get_cam_features(agent, scene, frame)[3].item() <= MOVING_M_PER_FRAME:
        continue
    p = feats[pt_mask]                                       # [n, 6]
    d = np.hypot(p[:, 0], p[:, 1]) * RADIUS_M
    j = int(d.argmin())
    dist.append(d[j])
    ang.append(abs(math.degrees(math.atan2(p[j, 3], p[j, 2]))))
dist, ang = np.array(dist), np.array(ang)
print(f"1. Frame: {len(dist)} moving train + tuning samples (speed > {MOVING_M_PER_FRAME} m/frame)")
print(f"   nearest lane point: median {np.median(dist):.2f} m, p95 {np.percentile(dist, 95):.2f} m; "
      f"|lane direction - ego heading| median {np.median(ang):.2f} deg, p95 {np.percentile(ang, 95):.2f} deg")
check(np.median(dist) <= 1.0 and np.percentile(dist, 95) <= 2.5, "nearest lane: median <= 1.0 m AND p95 <= 2.5 m")
check(np.median(ang) <= 5.0, "median |lane direction - ego heading| <= 5 deg")

rng = random.Random(0)
moving = [i for i in rng.sample(ids, 2000)
          if ds2.get_cam_features(*ds2.samples[i][:3])[3].item() > MOVING_M_PER_FRAME]
for i in moving:
    agent, scene, frame, _ = ds2.samples[i]
    gt, gm = get_future_trajectory(ds2.loader, agent, scene, frame, ds2.horizon)
    if gm[0]:
        gt_ang.append(abs(math.degrees(math.atan2(gt[0, 1].item(), gt[0, 0].item()))))
print(f"   first GT step direction, {len(gt_ang)} moving samples: median |angle to +x| {np.median(gt_ang):.2f} deg")
check(np.median(gt_ang) <= 5.0, "gt_traj frame: x along the ego heading (median <= 5 deg)")
print(f"2. Coverage: {empty} / {len(ids)} train + tuning samples ({100 * empty / len(ids):.2f}%) have no lane piece "
      f"within {RADIUS_M:g} m")

# ---------- 3: dataset ----------
print(f"3. Dataset: {N_DATASET_CHECK} random train + tuning samples")
sel = rng.sample(ids, N_DATASET_CHECK)
ok_tok, ok_rest = True, True
for i in sel:
    a, b = ds2[i], ds1[i]
    agent, scene, frame, town = ds2.samples[i]
    pose = ds2.loader.ego_pose[ds2.loader.find_sample_data(agent, scene, frame)["ego_pose_token"]]
    ref = ds2.lanes[town].tokens(np.array(pose["translation"][:2], dtype=np.float64),
                                 quaternion_to_yaw(pose["rotation"]))
    for k, r in zip(("map_tokens", "map_pt_mask", "map_piece_mask"), ref):
        ok_tok &= torch.equal(a[k], torch.from_numpy(r))
    ok_tok &= (tuple(a["map_tokens"].shape) == (K_PIECES, MAX_PIECE_POINTS, N_POINT_FEATS)
               and tuple(a["map_pt_mask"].shape) == (K_PIECES, MAX_PIECE_POINTS)
               and tuple(a["map_piece_mask"].shape) == (K_PIECES,))
    ok_rest &= all(equal(a[k], b[k]) for k in b)
    ok_rest &= set(a) - set(b) == {"map_tokens", "map_pt_mask", "map_piece_mask"}
check(ok_tok, "map tokens = LanePolylines.tokens at the ego pose, env_map shapes")
check(ok_rest, "every other key equals map_version v1's sample")
batch = pilot_collate_fn([ds2[i] for i in sel[:4]])
check(tuple(batch["map_tokens"].shape) == (4, K_PIECES, MAX_PIECE_POINTS, N_POINT_FEATS)
      and tuple(batch["map_piece_mask"].shape) == (4, K_PIECES), "collate stacks the map tensors to [B, ...]")

# ---------- 4: model ----------
print("4. Model")
torch.manual_seed(0)
m2 = EndToEndFusion(device="cpu", map_version="v2")
m1 = EndToEndFusion(device="cpu")
k2, k1 = set(m2.state_dict()), set(m1.state_dict())
check(not any(k.startswith(("graph_", "map_encoder.")) for k in k2), "v2: no graph buffers, no RGCN")
check(any(k.startswith("poly_encoder.") for k in k2) and "no_lane" in k2 and any(k.startswith("map_fusion.") for k in k2),
      "v2: poly_encoder, no_lane, map_fusion present")
same_shape = all(m1.state_dict()[k].shape == m2.state_dict()[k].shape
                 for k in k1 & k2 if k.startswith(("fuse.", "traj_head.", "map_fusion.")))
check(same_shape and {k for k in k1 if k.startswith(("fuse.", "traj_head."))} <= k2,
      "v2: fuse, trajectory head and map_fusion shapes equal v1's")
m2.train()
_, trajs, conf = m2(batch)
(trajs.square().mean() + conf.square().mean()).backward()
check(bool(torch.isfinite(trajs).all() and torch.isfinite(conf).all()), "forward on a real batch: finite outputs")
grads = {n: p.grad for n, p in m2.named_parameters()}
nz = lambda pre: any(g is not None and g.abs().sum() > 0 for n, g in grads.items() if n.startswith(pre))
check(nz("poly_encoder.") and nz("no_lane") and nz("map_fusion."), "nonzero gradients: poly_encoder, no_lane, map_fusion")
empty_batch = dict(batch, map_piece_mask=torch.zeros_like(batch["map_piece_mask"]),
                   map_pt_mask=torch.zeros_like(batch["map_pt_mask"]))
m2.eval()
with torch.no_grad():
    _, t_e, c_e = m2(empty_batch)
check(bool(torch.isfinite(t_e).all() and torch.isfinite(c_e).all()), "every piece masked: finite outputs (no-lane token)")

print("RESULT: " + ("PASS" if not fails else f"FAIL ({len(fails)}): {fails}"))
