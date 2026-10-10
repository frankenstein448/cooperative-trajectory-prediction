import argparse
import os
import json
import statistics
import torch
from torch.utils.data import DataLoader

from pilot_dataset import PilotDataset, pilot_collate_fn
from motion_utils import get_cv_velocity
from visualize_predictions import load_model
from scene_split import scene_split, three_way_scene_split

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--no-map", action="store_true")
parser.add_argument("--use-image-cache", action="store_true")
parser.add_argument("--run-dir", default=None,
                     help="read <run-dir>/config.json for use_image_cache/require_prev_frame/use_speed and "
                          "evaluate on three_way_scene_split's tuning part, instead of the old val split")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.run_dir:
    config_path = os.path.join(args.run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"--run-dir given but no config.json at {config_path}")
    with open(config_path) as f:
        run_config = json.load(f)
    if args.use_image_cache and not run_config["use_image_cache"]:
        raise ValueError(f"--use-image-cache conflicts with {config_path}'s "
                          f"use_image_cache={run_config['use_image_cache']}")
    use_image_cache = run_config["use_image_cache"]
    require_prev_frame = run_config["require_prev_frame"]
    use_speed = run_config["use_speed"]
    cam_norm = None
    if run_config.get("normalize_cam"):
        cam_norm_path = os.path.join(args.run_dir, "cam_norm.json")
        if not os.path.exists(cam_norm_path):
            raise FileNotFoundError(f"run config says normalize_cam=True but no cam_norm.json at {cam_norm_path}")
        with open(cam_norm_path) as f:
            cam_norm = json.load(f)
    drop_cam_xy = run_config.get("drop_cam_xy", False)  # absent in pre-D configs
    assert not drop_cam_xy or cam_norm is not None, f"{config_path}: drop_cam_xy=True requires normalize_cam=True"
    print(f"--run-dir {args.run_dir}: use_image_cache={use_image_cache}, "
          f"require_prev_frame={require_prev_frame}, use_speed={use_speed}, "
          f"normalize_cam={cam_norm is not None}, drop_cam_xy={drop_cam_xy} (from config.json)")
else:
    use_image_cache = args.use_image_cache
    require_prev_frame = False
    use_speed = False
    cam_norm = None
    drop_cam_xy = False

dataset = PilotDataset(use_image_cache=use_image_cache, require_prev_frame=require_prev_frame,
                        use_speed=use_speed, cam_norm=cam_norm, drop_cam_xy=drop_cam_xy)
if args.run_dir:
    _, val_set, _ = three_way_scene_split(dataset, seed=42)  # tuning part; 42 matches train.py's SCENE_SPLIT_SEED
else:
    _, val_set = scene_split(dataset, 0.15, 42)  # unchanged old path
keys = [(dataset.samples[i][3], dataset.samples[i][1]) for i in val_set.indices]
loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=pilot_collate_fn)
model = load_model(args.ckpt, use_map=not args.no_map).to(DEVICE).eval()
print(f"ckpt={args.ckpt}  use_map={not args.no_map}")

cols = {k: [] for k in ("ade6", "fde6", "ade1", "fde1", "zade", "zfde", "valid", "first",
                         "cvade", "cvfde", "cv1", "cv_ok", "v_norm")}
score_idx = None
with torch.no_grad():
    for batch in loader:
        gt = batch["gt_traj"].to(DEVICE)                                  # [B, T, 2]
        mask = batch["gt_mask"].to(DEVICE).float()                        # [B, T]
        out = model(batch)
        if not (isinstance(out, (tuple, list)) and len(out) == 3):
            raise RuntimeError(f"Expected model to return 3 outputs, got {type(out)}")
        trajs = out[1]                                                    # [B, K, T, 2]
        B, K = trajs.shape[:2]
        if trajs.shape[2:] != gt.shape[1:]:
            raise RuntimeError(f"trajs {tuple(trajs.shape)} does not match gt {tuple(gt.shape)}")
        if score_idx is None:
            cands = [i for i in (0, 2) if torch.is_tensor(out[i]) and tuple(out[i].shape) == (B, K)]
            if len(cands) != 1:
                shapes = [tuple(o.shape) if torch.is_tensor(o) else type(o).__name__ for o in out]
                raise RuntimeError(f"Cannot identify mode scores uniquely; output shapes: {shapes}")
            score_idx = cands[0]
            print(f"Using model output[{score_idx}] as mode scores (shape {tuple(out[score_idx].shape)})")
        scores = out[score_idx]

        T = mask.shape[-1]
        cnt = mask.sum(-1).clamp(min=1)                                   # [B]
        last = (mask * (torch.arange(T, device=DEVICE).float() + 1)).argmax(-1)  # last valid step [B]
        dist = torch.norm(trajs - gt.unsqueeze(1), dim=-1)                # [B, K, T]
        ade_modes = (dist * mask.unsqueeze(1)).sum(-1) / cnt.unsqueeze(1) # [B, K]
        fde_modes = dist.gather(-1, last.view(B, 1, 1).expand(B, K, 1)).squeeze(-1)  # [B, K]
        top = scores.argmax(-1, keepdim=True)                             # [B, 1]
        gt_norm = torch.norm(gt, dim=-1)                                  # [B, T]

        # ---- constant-velocity baseline: v * t, v = frame-1 -> frame displacement ----
        cv_ok = torch.zeros(B, dtype=torch.bool)
        v = torch.zeros(B, 2)
        for i in range(B):
            ok, vx, vy = get_cv_velocity(dataset.loader, batch["agent_id"][i], batch["scene"][i], batch["frame"][i])
            cv_ok[i] = ok
            if ok:
                v[i, 0], v[i, 1] = vx, vy
        steps = torch.arange(1, T + 1, device=DEVICE).float().view(1, T, 1)   # [1, T, 1]
        cv_traj = steps * v.to(DEVICE).view(B, 1, 2)                          # [B, T, 2]
        dist_cv = torch.norm(cv_traj - gt, dim=-1)                           # [B, T]

        cols["ade6"].append(ade_modes.min(1).values.cpu())
        cols["fde6"].append(fde_modes.min(1).values.cpu())
        cols["ade1"].append(ade_modes.gather(1, top).squeeze(1).cpu())
        cols["fde1"].append(fde_modes.gather(1, top).squeeze(1).cpu())
        cols["zade"].append(((gt_norm * mask).sum(-1) / cnt).cpu())
        cols["zfde"].append(gt_norm.gather(1, last.unsqueeze(1)).squeeze(1).cpu())
        cols["valid"].append((mask.sum(-1) > 0).cpu())
        cols["first"].append(gt_norm[:, 0].cpu())
        cols["cvade"].append(((dist_cv * mask).sum(-1) / cnt).cpu())
        cols["cvfde"].append(dist_cv.gather(-1, last.unsqueeze(1)).squeeze(1).cpu())
        cols["cv1"].append(dist_cv[:, 0].cpu())
        cols["cv_ok"].append(cv_ok.cpu())
        cols["v_norm"].append(v.norm(dim=-1).cpu())

c = {k: torch.cat(v) for k, v in cols.items()}
assert len(c["ade6"]) == len(keys), f"{len(c['ade6'])} predictions vs {len(keys)} keys"
valid = c["valid"]
cv_ok = c["cv_ok"].bool()
stationary = cv_ok & (c["v_norm"] == 0)  # speed == 0 via motion_utils.get_cv_velocity
moving = cv_ok & (c["v_norm"] > 0)


def print_group(label, mask):
    n = int(mask.sum())
    if not n:
        print(f"    {label:14s}: n=    0")
        return
    print(f"    {label:14s}: n={n:5d} minADE6={c['ade6'][mask].mean():7.3f}m ADE1={c['ade1'][mask].mean():7.3f}m "
          f"cvADE={c['cvade'][mask].mean():7.3f}m stayADE={c['zade'][mask].mean():7.3f}m")


print(f"Sanity: mean distance of first GT point from origin = {c['first'].mean():.2f} m "
      f"(should be small, near the per-step motion, if gt_traj is relative to the agent)")
print(f"Sanity: mean CV error at step 1 = {c['cv1'][cv_ok].mean():.3f} m (should be well below 0.40 m)")
hdr = (f"\n{'town':7s} {'scene':>5s} {'n':>5s} {'minADE6':>8s} {'minFDE6':>8s} "
       f"{'ADE1':>8s} {'FDE1':>8s} {'stayADE':>8s} {'stayFDE':>8s}")
print(hdr)
row = lambda s: " ".join(f"{c[k][s].mean():7.3f}m" for k in ("ade6", "fde6", "ade1", "fde1", "zade", "zfde"))
per_scene = []
for key in sorted(set(keys)):
    scene_mask = torch.tensor([k == key for k in keys])
    sel = scene_mask & valid
    n = int(sel.sum())
    if n:
        print(f"{key[0]:7s} {key[1]:5d} {n:5d} {row(sel)}")
        per_scene.append({k: float(c[k][sel].mean()) for k in ("ade6", "ade1", "zade")})
    sel_cv = sel & cv_ok
    n_cv = int(sel_cv.sum())
    n_excl = int((scene_mask & ~cv_ok).sum())
    if n_cv:
        print(f"    CV: n={n_cv:5d} excluded={n_excl:3d} (no frame-1)  "
              f"cvADE={c['cvade'][sel_cv].mean():7.3f}m cvFDE={c['cvfde'][sel_cv].mean():7.3f}m  "
              f"| model[cv_ok] minADE6={c['ade6'][sel_cv].mean():7.3f}m ADE1={c['ade1'][sel_cv].mean():7.3f}m")
    else:
        print(f"    CV: n=    0 excluded={n_excl:3d} (no frame-1)  cvADE=    n/a   cvFDE=    n/a")
    print_group("stationary", sel & stationary)
    print_group("moving", sel & moving)
print(f"{'ALL':7s} {'':5s} {int(valid.sum()):5d} {row(valid)}")
sel_cv_all = valid & cv_ok
print(f"{'CV':7s} {'':5s} {int(sel_cv_all.sum()):5d} "
      f"cvADE={c['cvade'][sel_cv_all].mean():7.3f}m cvFDE={c['cvfde'][sel_cv_all].mean():7.3f}m  "
      f"| model[cv_ok] minADE6={c['ade6'][sel_cv_all].mean():7.3f}m ADE1={c['ade1'][sel_cv_all].mean():7.3f}m  "
      f"(excluded {int((~cv_ok).sum())} of {len(valid)} samples, no frame-1)")
print_group("stationary (ALL)", valid & stationary)
print_group("moving (ALL)", valid & moving)

print(f"\nMedian of per-scene minADE6: {statistics.median(s['ade6'] for s in per_scene):.3f} m   "
      f"ADE1: {statistics.median(s['ade1'] for s in per_scene):.3f} m")
wins = sum(s["ade1"] < s["zade"] for s in per_scene)
print(f"Scenes where top-1 ADE beats stay-put: {wins}/{len(per_scene)}")
