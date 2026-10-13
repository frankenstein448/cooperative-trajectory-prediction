"""
diag_history_sensitivity.py
Read-only diagnostic, like diag_speed_sensitivity.py but for the observed-history
input: loads a history_frames > 0 checkpoint and, on the tuning split, replaces
the history per group:
  - moving samples (frame-1 -> frame displacement > 0): all-zero (parked) history
  - stationary samples (displacement == 0):  straight constant-velocity history at
    the median RAW moving speed of the tuning split, along the current heading,
    i.e. history[k] = (-(H - k) * v, 0) for k = 0 .. H-1 (so history[-1] = (-v, 0))
Classification uses motion_utils.get_cv_velocity (as the speed probe and
eval_scenes.py do). Perturbations are built in RAW metres and only then
normalized with the run's own history_norm.json. Everything else in the batch
-- including cam, whose speed field is left at its real value -- is untouched,
so this isolates what the model takes from the history input itself.

Reports, per group: mean top-1 final-step travel (real / perturbed) and its
change, mean L2 change of the top-1 trajectory, and the fraction of samples whose
argmax mode flips.
"""
import argparse
import json
import os
import math
import torch
from torch.utils.data import DataLoader

from coord_transform import map_transform_of
from pilot_dataset import PilotDataset, pilot_collate_fn, load_history_settings, load_v2x_settings
from motion_utils import get_cv_velocity
from visualize_predictions import load_model
from scene_split import three_way_scene_split

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", default="runs/H8_seed0")
parser.add_argument("--ckpt", default=None, help="defaults to <run-dir>/checkpoint_epoch12.pt")
args = parser.parse_args()

ckpt_path = args.ckpt or os.path.join(args.run_dir, "checkpoint_epoch12.pt")
config_path = os.path.join(args.run_dir, "config.json")
with open(config_path) as f:
    run_config = json.load(f)
assert run_config.get("history_frames", 0) > 0, f"{config_path} has no history input -- this diagnostic needs one"

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
use_map = not run_config.get("no_map", False)
history_kwargs = load_history_settings(args.run_dir, run_config)  # errors if history_norm.json is missing
H = history_kwargs["history_frames"]
hist_mean = torch.tensor(history_kwargs["history_norm"]["mean"], dtype=torch.float32)
hist_std = torch.tensor(history_kwargs["history_norm"]["std"], dtype=torch.float32)

dataset = PilotDataset(use_image_cache=run_config["use_image_cache"],
                        require_prev_frame=run_config["require_prev_frame"],
                        use_speed=run_config["use_speed"], cam_norm=cam_norm, drop_cam_xy=drop_cam_xy,
                        **history_kwargs, **load_v2x_settings(run_config),
                        map_transform=map_transform_of(run_config))
_, tuning_set, _ = three_way_scene_split(dataset, seed=42)  # tuning part; 42 matches train.py's SCENE_SPLIT_SEED
loader = DataLoader(tuning_set, batch_size=8, shuffle=False, collate_fn=pilot_collate_fn)

model = load_model(ckpt_path, use_map=use_map, history_frames=H,
                   history_encoder=run_config["history_encoder"],
                   **load_v2x_settings(run_config)).to(DEVICE).eval()
print(f"ckpt={ckpt_path}")
print(f"run config: history_frames={H} history_encoder={run_config['history_encoder']} "
      f"use_speed={run_config['use_speed']} normalize_cam={cam_norm is not None} "
      f"drop_cam_xy={drop_cam_xy} use_map={use_map}")


def raw_speed(agent_id, scene, frame):
    ok, vx, vy = get_cv_velocity(dataset.loader, agent_id, scene, frame)
    assert ok, f"history filter guarantees frame-1 exists, but get_cv_velocity failed for ({agent_id},{scene},{frame})"
    return math.hypot(vx, vy)


# ---------- pass 1 (data only): RAW moving-speed median ----------
moving_speeds_raw = []
for i in tuning_set.indices:
    r = raw_speed(*dataset.samples[i][:3])
    if r > 0:
        moving_speeds_raw.append(r)
median_speed_raw = torch.tensor(moving_speeds_raw).median().item()
print(f"\nMoving samples in tuning split: {len(moving_speeds_raw)} / {len(tuning_set)}  "
      f"median RAW speed = {median_speed_raw:.4f} m/frame")

parked_raw = torch.zeros(H, 2)
cv_raw = torch.zeros(H, 2)
cv_raw[:, 0] = -median_speed_raw * torch.arange(H, 0, -1, dtype=torch.float32)  # -(H-k) * v
print(f"Stationary replacement history (RAW, m): x = {[round(x, 3) for x in cv_raw[:, 0].tolist()]}, y = 0")
parked_norm = (parked_raw - hist_mean) / hist_std
cv_norm = (cv_raw - hist_mean) / hist_std


# ---------- pass 2 (model forward, real vs perturbed) ----------
l2_changes, mode_changed, stationary_mask = [], [], []
disp_real_list, disp_pert_list = [], []

with torch.no_grad():
    for batch in loader:
        gt_mask = batch["gt_mask"].to(DEVICE).float()              # [B, T]
        cnt = gt_mask.sum(-1).clamp(min=1)
        T = gt_mask.shape[-1]
        last = (gt_mask * (torch.arange(T, device=DEVICE).float() + 1)).argmax(-1)  # [B]

        stationary = torch.tensor([raw_speed(batch["agent_id"][i], batch["scene"][i], batch["frame"][i]) == 0.0
                                   for i in range(len(batch["agent_id"]))])
        stationary_mask.append(stationary)

        batch2 = dict(batch)  # shallow copy; only "history" is replaced
        batch2["history"] = torch.stack([cv_norm if s else parked_norm for s in stationary.tolist()])

        out_real = model(batch)
        out_pert = model(batch2)
        trajs_real, conf_real = out_real[1], out_real[2]
        trajs_pert, conf_pert = out_pert[1], out_pert[2]

        top_real = conf_real.argmax(-1)                             # [B]
        top_pert = conf_pert.argmax(-1)
        B = trajs_real.shape[0]
        idx = torch.arange(B, device=trajs_real.device)
        top1_real = trajs_real[idx, top_real]                       # [B, T, 2]
        top1_pert = trajs_pert[idx, top_pert]

        dist = torch.norm(top1_real - top1_pert, dim=-1)            # [B, T]
        l2_changes.append(((dist * gt_mask).sum(-1) / cnt).cpu())
        mode_changed.append((top_real != top_pert).cpu())
        disp_real_list.append(torch.norm(top1_real[idx, last], dim=-1).cpu())
        disp_pert_list.append(torch.norm(top1_pert[idx, last], dim=-1).cpu())

l2_changes = torch.cat(l2_changes)
mode_changed = torch.cat(mode_changed)
stationary_mask = torch.cat(stationary_mask)
moving_mask = ~stationary_mask
disp_real_all = torch.cat(disp_real_list)
disp_pert_all = torch.cat(disp_pert_list)

print(f"\nOver {len(l2_changes)} tuning samples "
      f"(moving: history -> all-zero; stationary: history -> constant velocity at {median_speed_raw:.4f} m/frame):")
for label, mask in (("stationary", stationary_mask), ("moving", moving_mask)):
    n = int(mask.sum())
    if not n:
        print(f"  [{label:10s}] n=    0")
        continue
    d_real, d_pert = disp_real_all[mask].mean(), disp_pert_all[mask].mean()
    print(f"  [{label:10s}] n={n:5d}  final-step travel: real={d_real:.4f} m, perturbed={d_pert:.4f} m, "
          f"change={d_pert - d_real:+.4f} m  |  mean L2 change={l2_changes[mask].mean():.4f} m  "
          f"mode-flip frac={mode_changed[mask].float().mean():.4f}")
