import argparse
import os
import sys
import json
import statistics
import subprocess
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from pilot_dataset import PilotDataset, pilot_collate_fn, load_history_settings
from motion_utils import get_cv_velocity
from visualize_predictions import load_model
from scene_split import scene_split, three_way_scene_split
from sensor_degradation import DEGRADE_MODES, degrade_config, fallback_counts, reset_fallback_counts

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--no-map", action="store_true")
parser.add_argument("--use-image-cache", action="store_true")
parser.add_argument("--run-dir", default=None,
                     help="read <run-dir>/config.json for use_image_cache/require_prev_frame/use_speed/no_map etc. and "
                          "evaluate on three_way_scene_split's tuning (or --split test) part, instead of the old val split")
parser.add_argument("--split", choices=["tuning", "test"], default="tuning",
                     help="which three_way_scene_split part to evaluate (requires --run-dir). "
                          "test requires --confirm-final-test and is logged to runs/TEST_EVALS.log")
parser.add_argument("--confirm-final-test", action="store_true",
                     help="required for --split test: acknowledges this is a final, logged test-set evaluation")
parser.add_argument("--degrade", choices=DEGRADE_MODES, default=None,
                     help="evaluate under this sensor degradation (sensor_degradation.py), whatever the run was "
                          "trained with. Tuning split only; requires --run-dir; output is also written to "
                          "eval_<run>_tuning_<degrade>.log")
args = parser.parse_args()

TEST_EVALS_LOG = "runs/TEST_EVALS.log"
if args.split == "test":
    if not args.run_dir:
        parser.error("--split test requires --run-dir (the test split only exists in three_way_scene_split)")
    if not args.confirm_final_test:
        parser.error("--split test requires --confirm-final-test")
    if args.degrade:
        parser.error("--degrade is exploratory and tuning-split only; refusing --split test")
if args.degrade:
    if not args.run_dir:
        parser.error("--degrade requires --run-dir (the log is named after the run)")
    run_name = os.path.basename(os.path.normpath(args.run_dir))
    degrade_log = f"eval_{run_name}_tuning_{args.degrade}.log"
    if os.path.exists(degrade_log):
        parser.error(f"{degrade_log} already exists; refusing to overwrite it")


    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)
                st.flush()

        def flush(self):
            for st in self.streams:
                st.flush()


    sys.stdout = _Tee(sys.__stdout__, open(degrade_log, "w"))
    print(f"Writing this output to {degrade_log}")
    print(f"--degrade {args.degrade}: {degrade_config(args.degrade)}")

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
    run_no_map = run_config.get("no_map", False)        # absent in pre-E configs
    if args.no_map and not run_no_map:
        raise ValueError(f"--no-map conflicts with {config_path}'s no_map={run_no_map}")
    use_map = not run_no_map
    assert not drop_cam_xy or cam_norm is not None, f"{config_path}: drop_cam_xy=True requires normalize_cam=True"
    print(f"--run-dir {args.run_dir}: use_image_cache={use_image_cache}, "
          f"require_prev_frame={require_prev_frame}, use_speed={use_speed}, "
          f"normalize_cam={cam_norm is not None}, drop_cam_xy={drop_cam_xy}, use_map={use_map} (from config.json)")
    history_kwargs = load_history_settings(args.run_dir, run_config)  # errors if history_norm.json is missing
    history_encoder = run_config.get("history_encoder") or "gru"     # absent in pre-history configs
    print(f"history_frames={history_kwargs['history_frames']} "
          f"(encoder={history_encoder if history_kwargs['history_frames'] else None}), "
          f"require_history_frames={history_kwargs['require_history_frames']} (from config.json)")
    trained_degrade = run_config.get("degrade")  # absent in pre-degradation configs
    print(f"Trained with degrade={trained_degrade['mode'] if trained_degrade else None}; "
          f"evaluating with degrade={args.degrade}")
else:
    use_image_cache = args.use_image_cache
    require_prev_frame = False
    use_speed = False
    cam_norm = None
    drop_cam_xy = False
    use_map = not args.no_map
    history_kwargs = {}
    history_encoder = "gru"

dataset = PilotDataset(use_image_cache=use_image_cache, require_prev_frame=require_prev_frame,
                        use_speed=use_speed, cam_norm=cam_norm, drop_cam_xy=drop_cam_xy, degrade=args.degrade,
                        **history_kwargs)
if args.run_dir:
    _, tuning_set, test_set = three_way_scene_split(dataset, seed=42)  # 42 matches train.py's SCENE_SPLIT_SEED
    val_set = test_set if args.split == "test" else tuning_set
    print(f"Evaluating on the {args.split.upper()} split ({len(val_set)} samples)")
    if args.split == "test":
        # Logged BEFORE evaluating, so an attempt is on record even if it crashes partway.
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
        except Exception as e:
            commit = f"<unavailable: {e}>"
        with open(TEST_EVALS_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')}\trun_dir={args.run_dir}\t"
                    f"git_commit={commit}\tckpt={args.ckpt}\n")
        print(f"Appended this test evaluation to {TEST_EVALS_LOG}")
else:
    _, val_set = scene_split(dataset, 0.15, 42)  # unchanged old path
keys = [(dataset.samples[i][3], dataset.samples[i][1]) for i in val_set.indices]
loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=pilot_collate_fn)
model = load_model(args.ckpt, use_map=use_map, history_frames=dataset.history_frames,
                   history_encoder=history_encoder).to(DEVICE).eval()
print(f"ckpt={args.ckpt}  use_map={use_map}")

cols = {k: [] for k in ("ade6", "fde6", "ade1", "fde1", "zade", "zfde", "valid", "first",
                         "cvade", "cvfde", "cv1", "cv_ok", "v_norm")}
score_idx = None
n_sensors = 0  # ego + coop LiDARs seen, to cross-check the fallback counter
reset_fallback_counts()
with torch.no_grad():
    for batch in loader:
        n_sensors += len(batch["points"]) + sum(len(cp) for cp in batch["coop_points"])
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

if args.degrade:
    fb = fallback_counts()
    # loader has no workers, so the counter saw every load; a mismatch means it didn't
    assert fb["sensors"] == n_sensors, f"fallback counter saw {fb['sensors']} sensors, batches had {n_sensors}"
    print(f"Empty-LiDAR fallback ({args.degrade}): {fb['fallback']}/{fb['sensors']} sensors (ego + coop)")

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
