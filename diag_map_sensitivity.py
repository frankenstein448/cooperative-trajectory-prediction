"""
diag_map_sensitivity.py
Read-only diagnostic: how much does a trained full (use_map=True) model's
prediction depend on the map input? On the tuning split, every batch is run
twice:
  - real:    the sample's own nearby_idx (its K nearest lane nodes)
  - swapped: nearby_idx of a partner sample from ANOTHER scene of the SAME
             town (same town graph, so only the lane location changes)
Everything else in the batch (points, images, coop, cam, cpm) is untouched.

The map input is entirely nearby_idx + the town's graph (end_to_end_model.py
looks lane embeddings up by batch["nearby_idx"] in batch["town"]'s graph), so
swapping nearby_idx is the whole map input.

Partner pool: all NON-TEST samples (train + tuning) of the same town whose
scene differs from the sample's own. Town03 and Town04 each have a single
tuning scene, so a tuning-only pool would have no partner there. Test-split
samples are never read. Pairing is fixed: np.random.default_rng(0), one draw
per tuning sample in tuning-split order. The partner's nearby_idx is computed
with the same lookup PilotDataset.__getitem__ uses (raw cam x,y ->
v2xsim_to_carla -> per-town normalization -> get_nearby_lane_nodes), without
loading the partner's LiDAR/images. That helper is checked against the real
batch's nearby_idx on every tuning sample.

Reports, per group (stationary / moving / ALL; stationary = RAW speed 0 via
motion_utils.get_cv_velocity, same as diag_speed_sensitivity.py): mean L2
change of the top-1 trajectory (masked mean over valid steps), argmax
mode-flip fraction, ADE1 real vs swapped and its mean change, and the same
for minADE6. Also reports how far apart the swapped lane locations are and
how often the swapped nearby_idx happens to equal the real one.
"""
import argparse
import json
import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from pilot_dataset import PilotDataset, pilot_collate_fn, load_history_settings, load_v2x_settings, K_NEAREST_LANES
from rgcn_map_encoder import get_nearby_lane_nodes
from coord_transform import v2xsim_to_carla
from motion_utils import get_cv_velocity
from visualize_predictions import load_model
from scene_split import three_way_scene_split

PAIRING_SEED = 0

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--ckpt", default=None, help="defaults to <run-dir>/checkpoint_epoch12.pt")
args = parser.parse_args()

ckpt_path = args.ckpt or os.path.join(args.run_dir, "checkpoint_epoch12.pt")
config_path = os.path.join(args.run_dir, "config.json")
with open(config_path) as f:
    run_config = json.load(f)
assert not run_config.get("no_map", False), f"{config_path} has no_map=True -- this diagnostic needs a map model"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cam_norm = None
if run_config.get("normalize_cam"):
    cam_norm_path = os.path.join(args.run_dir, "cam_norm.json")
    if not os.path.exists(cam_norm_path):
        raise FileNotFoundError(f"run config says normalize_cam=True but no cam_norm.json at {cam_norm_path}")
    with open(cam_norm_path) as f:
        cam_norm = json.load(f)
drop_cam_xy = run_config.get("drop_cam_xy", False)
assert not drop_cam_xy or cam_norm is not None, f"{config_path}: drop_cam_xy=True requires normalize_cam=True"

dataset = PilotDataset(use_image_cache=run_config["use_image_cache"],
                        require_prev_frame=run_config["require_prev_frame"],
                        use_speed=run_config["use_speed"], cam_norm=cam_norm, drop_cam_xy=drop_cam_xy,
                        **load_history_settings(args.run_dir, run_config), **load_v2x_settings(run_config))
train_set, tuning_set, _ = three_way_scene_split(dataset, seed=42)  # 42 matches train.py's SCENE_SPLIT_SEED
loader = DataLoader(tuning_set, batch_size=8, shuffle=False, collate_fn=pilot_collate_fn)

model = load_model(ckpt_path, use_map=True, history_frames=dataset.history_frames,
                   history_encoder=run_config.get("history_encoder") or "gru",
                   **load_v2x_settings(run_config)).to(DEVICE).eval()
print(f"ckpt={ckpt_path}")
print(f"run config: use_speed={run_config['use_speed']} use_image_cache={run_config['use_image_cache']} "
      f"require_prev_frame={run_config['require_prev_frame']} normalize_cam={cam_norm is not None} "
      f"drop_cam_xy={drop_cam_xy}")


def lane_lookup(idx):
    """Same nearest-lane lookup as PilotDataset.__getitem__, from RAW cam x,y,
    without loading sensors. Returns (nearby_idx [k], carla_xy)."""
    agent_id, scene, frame, town = dataset.samples[idx]
    graph = dataset.town_graphs[town]
    cam_raw = dataset.get_cam_features(agent_id, scene, frame)
    carla_x, carla_y = v2xsim_to_carla(cam_raw[0].item(), cam_raw[1].item(), town)
    agent_pos = torch.tensor([[carla_x, carla_y]], dtype=torch.float32)
    agent_pos_norm = (agent_pos - graph["mean"]) / graph["std"]
    return get_nearby_lane_nodes(agent_pos_norm, graph["data"].x, k=K_NEAREST_LANES), (carla_x, carla_y)


def raw_speed(agent_id, scene, frame):
    ok, vx, vy = get_cv_velocity(dataset.loader, agent_id, scene, frame)
    assert ok, f"require_prev_frame guarantees frame-1 exists, but get_cv_velocity failed for ({agent_id},{scene},{frame})"
    return math.hypot(vx, vy)


# ---------- fixed pairing: tuning sample -> non-test partner, same town, other scene ----------
pool_by_town = {}
for i in sorted(train_set.indices + tuning_set.indices):
    _, scene, _, town = dataset.samples[i]
    pool_by_town.setdefault(town, []).append(i)

rng = np.random.default_rng(PAIRING_SEED)
partner_of = []
for i in tuning_set.indices:
    _, scene, _, town = dataset.samples[i]
    cands = [j for j in pool_by_town[town] if dataset.samples[j][1] != scene]
    assert cands, f"no other-scene partner for tuning idx {i} ({town}, scene {scene})"
    partner_of.append(cands[rng.integers(len(cands))])
print(f"Paired {len(partner_of)} tuning samples with non-test partners (same town, other scene), "
      f"seed {PAIRING_SEED}")

# ---------- real vs swapped forward passes ----------
l2_changes, mode_changed, stationary_mask = [], [], []
ade1_real, ade1_swap, ade6_real, ade6_swap = [], [], [], []
lane_dist, same_idx = [], []


def ade_stats(trajs, conf, gt, mask, cnt):
    """Returns per-sample ADE1 (top-scored mode), minADE6, and the top-1 index."""
    dist = torch.norm(trajs - gt.unsqueeze(1), dim=-1)                 # [B, K, T]
    ade_modes = (dist * mask.unsqueeze(1)).sum(-1) / cnt.unsqueeze(1)  # [B, K]
    top = conf.argmax(-1)                                              # [B]
    return ade_modes.gather(1, top.unsqueeze(1)).squeeze(1), ade_modes.min(1).values, top


pos = 0  # position in tuning_set, to index partner_of
with torch.no_grad():
    for batch in loader:
        B = len(batch["agent_id"])
        gt = batch["gt_traj"].to(DEVICE)
        mask = batch["gt_mask"].to(DEVICE).float()
        cnt = mask.sum(-1).clamp(min=1)

        real_idx_list, swap_idx_list = [], []
        for b in range(B):
            own = tuning_set.indices[pos + b]
            own_idx, own_xy = lane_lookup(own)
            assert torch.equal(own_idx, batch["nearby_idx"][b]), \
                f"lane_lookup disagrees with PilotDataset's nearby_idx for idx {own}"
            swap_idx, swap_xy = lane_lookup(partner_of[pos + b])
            swap_idx_list.append(swap_idx)
            lane_dist.append(math.hypot(own_xy[0] - swap_xy[0], own_xy[1] - swap_xy[1]))
            same_idx.append(torch.equal(own_idx, swap_idx))
        pos += B

        stationary_mask.append(torch.tensor([raw_speed(batch["agent_id"][b], batch["scene"][b], batch["frame"][b]) == 0.0
                                             for b in range(B)]))

        batch_swap = dict(batch)
        batch_swap["nearby_idx"] = torch.stack(swap_idx_list, dim=0)

        _, trajs_r, conf_r = model(batch)
        _, trajs_s, conf_s = model(batch_swap)

        a1r, a6r, top_r = ade_stats(trajs_r, conf_r, gt, mask, cnt)
        a1s, a6s, top_s = ade_stats(trajs_s, conf_s, gt, mask, cnt)

        idx = torch.arange(B, device=trajs_r.device)
        d = torch.norm(trajs_r[idx, top_r] - trajs_s[idx, top_s], dim=-1)  # [B, T]
        l2_changes.append(((d * mask).sum(-1) / cnt).cpu())
        mode_changed.append((top_r != top_s).cpu())
        ade1_real.append(a1r.cpu()); ade1_swap.append(a1s.cpu())
        ade6_real.append(a6r.cpu()); ade6_swap.append(a6s.cpu())

assert pos == len(tuning_set), f"consumed {pos} of {len(tuning_set)} tuning samples"
l2_changes = torch.cat(l2_changes)
mode_changed = torch.cat(mode_changed)
stationary_mask = torch.cat(stationary_mask)
ade1_real, ade1_swap = torch.cat(ade1_real), torch.cat(ade1_swap)
ade6_real, ade6_swap = torch.cat(ade6_real), torch.cat(ade6_swap)
lane_dist = torch.tensor(lane_dist)
same_idx = torch.tensor(same_idx)

print(f"\nSwap geometry: partner CARLA-position distance mean={lane_dist.mean():.1f} m "
      f"median={lane_dist.median():.1f} m; swapped nearby_idx identical to real in "
      f"{int(same_idx.sum())}/{len(same_idx)} samples")
print(f"\nOver {len(l2_changes)} tuning samples (real map input vs map input from another scene, same town):")
for label, m in (("stationary", stationary_mask), ("moving", ~stationary_mask),
                 ("ALL", torch.ones_like(stationary_mask))):
    n = int(m.sum())
    if not n:
        print(f"  [{label:10s}] n=    0")
        continue
    print(f"  [{label:10s}] n={n:5d}  top-1 L2 change={l2_changes[m].mean():.4f} m  "
          f"mode-flip frac={mode_changed[m].float().mean():.4f}  "
          f"ADE1 real={ade1_real[m].mean():.3f} swapped={ade1_swap[m].mean():.3f} "
          f"(delta={(ade1_swap[m] - ade1_real[m]).mean():+.3f} m)  "
          f"minADE6 real={ade6_real[m].mean():.3f} swapped={ade6_swap[m].mean():.3f} "
          f"(delta={(ade6_swap[m] - ade6_real[m]).mean():+.3f} m)")
