"""
diag_speed_sensitivity.py
Read-only diagnostic: loads a use_speed=True checkpoint and, on the tuning
split, perturbs speed per group instead of just zeroing it -- zeroing
stationary samples (already 0) would be a no-op by construction:
  - moving samples:     real RAW speed -> 0
  - stationary samples: 0 -> median RAW speed of the moving samples in the split
Classification and perturbation always work in RAW speed space (via
motion_utils.get_cv_velocity), never by reading cam[3] off the batch --
cam[3] is only raw when normalize_cam is off; when it's on, cam[3] is
z-scored and reading it directly would misclassify everything. The raw
perturbed value is converted to whatever the model actually sees (raw, or
normalized via the run's own cam_norm.json) only as the last step, so a
run with normalize_cam can be compared fairly against one without.

Reports, per group: mean L2 change of the top-1 trajectory, fraction whose
argmax mode flips, and mean top-1 final-step displacement in both passes.
Also reports mean |cam| per feature over the tuning split (whatever the
model actually receives -- raw or normalized).
"""
import argparse
import json
import os
import math
import torch
from torch.utils.data import DataLoader

from pilot_dataset import PilotDataset, pilot_collate_fn, load_history_settings, load_v2x_settings
from motion_utils import get_cv_velocity
from visualize_predictions import load_model
from scene_split import three_way_scene_split

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", default="runs/B_seed0")
parser.add_argument("--ckpt", default=None, help="defaults to <run-dir>/checkpoint_epoch12.pt")
args = parser.parse_args()

ckpt_path = args.ckpt or os.path.join(args.run_dir, "checkpoint_epoch12.pt")
config_path = os.path.join(args.run_dir, "config.json")
with open(config_path) as f:
    run_config = json.load(f)
assert run_config["use_speed"], f"{config_path} has use_speed=False -- this diagnostic needs a speed-trained model"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cam_norm = None
if run_config.get("normalize_cam"):
    cam_norm_path = os.path.join(args.run_dir, "cam_norm.json")
    if not os.path.exists(cam_norm_path):
        raise FileNotFoundError(f"run config says normalize_cam=True but no cam_norm.json at {cam_norm_path}")
    with open(cam_norm_path) as f:
        cam_norm = json.load(f)
drop_cam_xy = run_config.get("drop_cam_xy", False)  # absent in pre-D configs
assert not drop_cam_xy or cam_norm is not None, f"{config_path}: drop_cam_xy=True requires normalize_cam=True"
use_map = not run_config.get("no_map", False)  # absent in pre-E configs

dataset = PilotDataset(use_image_cache=run_config["use_image_cache"],
                        require_prev_frame=run_config["require_prev_frame"],
                        use_speed=run_config["use_speed"], cam_norm=cam_norm, drop_cam_xy=drop_cam_xy,
                        **load_history_settings(args.run_dir, run_config), **load_v2x_settings(run_config))
_, tuning_set, _ = three_way_scene_split(dataset, seed=42)  # tuning part; 42 matches train.py's SCENE_SPLIT_SEED
loader = DataLoader(tuning_set, batch_size=8, shuffle=False, collate_fn=pilot_collate_fn)

model = load_model(ckpt_path, use_map=use_map, history_frames=dataset.history_frames,
                   history_encoder=run_config.get("history_encoder") or "gru",
                   **load_v2x_settings(run_config)).to(DEVICE).eval()
print(f"ckpt={ckpt_path}")
print(f"run config: use_speed={run_config['use_speed']} use_image_cache={run_config['use_image_cache']} "
      f"require_prev_frame={run_config['require_prev_frame']} normalize_cam={cam_norm is not None} "
      f"drop_cam_xy={drop_cam_xy} use_map={use_map}")


def raw_speed(agent_id, scene, frame):
    ok, vx, vy = get_cv_velocity(dataset.loader, agent_id, scene, frame)
    assert ok, f"require_prev_frame guarantees frame-1 exists, but get_cv_velocity failed for ({agent_id},{scene},{frame})"
    return math.hypot(vx, vy)


def encode_speed(raw, cam_norm):
    """RAW speed -> whatever the model actually sees: raw itself, or normalized
    via the same per-feature transform the dataset applies to real samples."""
    if cam_norm is None:
        return raw
    mean3, std3 = cam_norm["mean"][3], cam_norm["std"][3]
    return (raw - mean3) / std3


# ---------- pass 1 (data only, no model): mean |cam|, and the RAW moving-speed median ----------
cam_abs_sum, n_cam = torch.zeros(4), 0
moving_speeds_raw = []
for batch in loader:
    cam_stack = torch.stack(batch["cam"])  # [B, 4] -- raw if cam_norm is off, normalized if on
    cam_abs_sum += cam_stack.abs().sum(0)
    n_cam += cam_stack.shape[0]
    for i in range(len(batch["agent_id"])):
        r = raw_speed(batch["agent_id"][i], batch["scene"][i], batch["frame"][i])
        if r > 0:
            moving_speeds_raw.append(r)

cam_mean_abs = cam_abs_sum / n_cam
median_speed_raw = torch.tensor(moving_speeds_raw).median().item()
print(f"\nMoving samples in tuning split: {len(moving_speeds_raw)} / {n_cam}  "
      f"median RAW speed = {median_speed_raw:.4f} m/frame")
print(f"Mean |cam| over tuning split (x, y, heading, speed), as seen by the model: "
      f"{cam_mean_abs[0]:.3f}, {cam_mean_abs[1]:.3f}, {cam_mean_abs[2]:.3f}, {cam_mean_abs[3]:.3f}")


def perturb_batch(batch, raw_speeds, median_speed_raw, cam_norm):
    """Shallow-copies batch, replacing cam with a copy where index 3 (speed)
    is swapped per sample in RAW space: moving (raw>0) -> 0; stationary
    (raw==0) -> median_speed_raw -- then encoded into whatever the model
    actually sees. Everything else (points, images, coop, cpm, nearby_idx,
    gt_traj, gt_mask) is untouched."""
    batch2 = dict(batch)
    new_cam = []
    for cam, r in zip(batch["cam"], raw_speeds):
        cam2 = cam.clone()
        new_raw = median_speed_raw if r == 0.0 else 0.0
        cam2[3] = encode_speed(new_raw, cam_norm)
        new_cam.append(cam2)
    batch2["cam"] = new_cam
    return batch2


# ---------- pass 2 (model forward, real vs perturbed) ----------
l2_changes, mode_changed, stationary_mask = [], [], []
disp_real_list, disp_pert_list = [], []

with torch.no_grad():
    for batch in loader:
        gt_mask = batch["gt_mask"].to(DEVICE).float()              # [B, T]
        cnt = gt_mask.sum(-1).clamp(min=1)
        T = gt_mask.shape[-1]
        last = (gt_mask * (torch.arange(T, device=DEVICE).float() + 1)).argmax(-1)  # [B]

        raw_speeds = [raw_speed(batch["agent_id"][i], batch["scene"][i], batch["frame"][i])
                      for i in range(len(batch["agent_id"]))]
        stationary_mask.append(torch.tensor([r == 0.0 for r in raw_speeds]))

        out_real = model(batch)
        out_pert = model(perturb_batch(batch, raw_speeds, median_speed_raw, cam_norm))
        trajs_real, conf_real = out_real[1], out_real[2]
        trajs_pert, conf_pert = out_pert[1], out_pert[2]

        top_real = conf_real.argmax(-1)                             # [B]
        top_pert = conf_pert.argmax(-1)
        B = trajs_real.shape[0]
        idx = torch.arange(B, device=trajs_real.device)
        top1_real = trajs_real[idx, top_real]                       # [B, T, 2]
        top1_pert = trajs_pert[idx, top_pert]

        dist = torch.norm(top1_real - top1_pert, dim=-1)            # [B, T]
        per_sample_l2 = (dist * gt_mask).sum(-1) / cnt               # [B]

        disp_real = torch.norm(top1_real[idx, last], dim=-1)        # [B]
        disp_pert = torch.norm(top1_pert[idx, last], dim=-1)        # [B]

        l2_changes.append(per_sample_l2.cpu())
        mode_changed.append((top_real != top_pert).cpu())
        disp_real_list.append(disp_real.cpu())
        disp_pert_list.append(disp_pert.cpu())

l2_changes = torch.cat(l2_changes)
mode_changed = torch.cat(mode_changed)
stationary_mask = torch.cat(stationary_mask)
moving_mask = ~stationary_mask
disp_real_all = torch.cat(disp_real_list)
disp_pert_all = torch.cat(disp_pert_list)

print(f"\nOver {len(l2_changes)} tuning samples "
      f"(moving: real RAW speed -> 0; stationary: 0 -> median RAW moving speed {median_speed_raw:.4f}):")
for label, mask in (("stationary", stationary_mask), ("moving", moving_mask), ("ALL", torch.ones_like(stationary_mask))):
    n = int(mask.sum())
    if not n:
        print(f"  [{label:10s}] n=    0")
        continue
    print(f"  [{label:10s}] n={n:5d}  mean L2 change={l2_changes[mask].mean():.4f} m  "
          f"mode-flip frac={mode_changed[mask].float().mean():.4f}  "
          f"final-step disp: real={disp_real_all[mask].mean():.4f} m, perturbed={disp_pert_all[mask].mean():.4f} m")
