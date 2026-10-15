import argparse
import os
import sys
import json
import statistics
import subprocess
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

from pilot_dataset import PilotDataset, pilot_collate_fn, load_history_settings, load_v2x_settings, load_map_settings
from motion_utils import get_cv_velocity
from visualize_predictions import load_model
from scene_split import scene_split, three_way_scene_split, split_version_of
from sensor_degradation import DEGRADE_MODES, degrade_config, fallback_counts, reset_fallback_counts
from safety_metrics import (LaneCentrelines, raw_pose, agent_to_carla,
                            MISS_THRESHOLD_M, OFFROAD_THRESHOLD_M, GT_OFFROAD_UNRELIABLE)
from trajectory_loss import loss_predicts_scale
from coord_transform import map_transform_of
from ranking_metrics import brier_min_fde, mode_diversity, laplace_coverage, COVERAGE_LEVELS
import final_test_guard
from scene_conditions import (heading_change_deg, JunctionAreas, TURN_THRESHOLD_DEG, MIN_STEP_M,
                              JUNCTION_RADIUS_M)

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
parser.add_argument("--metrics-only-reeval", action="store_true",
                     help="with --split test: re-run an evaluation already in runs/TEST_EVALS.log (same run-dir and "
                          "ckpt) only to add metrics (miss / off-road); refused if no such earlier entry exists, "
                          "and recorded in the log as a metrics-only re-evaluation")
parser.add_argument("--confirmatory-turning-test", action="store_true",
                     help="with --split test: the confirmatory condition evaluation of runs/DECISION_RULES_TURNING_TEST.md; "
                          "output (stdout and stderr) is written to eval_<run>_test_cond.log and the evaluation is recorded "
                          "in the log with that purpose. A re-run is allowed only if every earlier attempt's log has no "
                          "COND line; the earlier log is then kept as eval_<run>_test_cond_attempt<N>.log")
parser.add_argument("--final-test", action="store_true",
                     help="with --split test: the final test evaluation of runs/DECISION_RULES_FINAL_TEST.md (split v2, "
                          "map_transform mirror runs only); output (stdout and stderr) is written to "
                          "eval_<run>_test_final.log and recorded in runs/TEST_EVALS.log with purpose 'final test "
                          "evaluation (split v2)'. A re-run is allowed only if every earlier attempt's log has no result "
                          "line (final_test_guard.py); the earlier log is then kept as eval_<run>_test_final_attempt<N>.log")
args = parser.parse_args()

TEST_EVALS_LOG = "runs/TEST_EVALS.log"
CONFIRMATORY_TURNING_PURPOSE = "confirmatory condition evaluation (turning hypothesis from tuning)"
CONFIRMATORY_RERUN_SUFFIX = " (re-run after crash, no results seen)"


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


def logged_test_evals():
    """(run_dir, ckpt, purpose) of every entry in TEST_EVALS_LOG, paths normalised."""
    entries = []
    if os.path.exists(TEST_EVALS_LOG):
        with open(TEST_EVALS_LOG) as f:
            for line in f:
                fields = dict(kv.split("=", 1) for kv in line.rstrip("\n").split("\t")[1:] if "=" in kv)
                entries.append((os.path.normpath(fields.get("run_dir", "")), os.path.normpath(fields.get("ckpt", "")),
                                fields.get("purpose", "")))
    return entries


if args.metrics_only_reeval and args.split != "test":
    parser.error("--metrics-only-reeval is for re-running an already-logged test evaluation (--split test)")
if args.confirmatory_turning_test and args.split != "test":
    parser.error("--confirmatory-turning-test is a test-split evaluation (--split test)")
if args.confirmatory_turning_test and args.metrics_only_reeval:
    parser.error("--confirmatory-turning-test and --metrics-only-reeval are exclusive")
if args.final_test:
    if args.split != "test":
        parser.error("--final-test is a test-split evaluation (--split test)")
    if args.confirmatory_turning_test or args.metrics_only_reeval:
        parser.error("--final-test excludes --confirmatory-turning-test and --metrics-only-reeval")
if args.split == "test":
    if not args.run_dir:
        parser.error("--split test requires --run-dir (the test split only exists in three_way_scene_split)")
    if not args.confirm_final_test:
        parser.error("--split test requires --confirm-final-test")
    if args.degrade:
        parser.error("--degrade is exploratory and tuning-split only; refusing --split test")
    this_eval = (os.path.normpath(args.run_dir), os.path.normpath(args.ckpt))
    if args.metrics_only_reeval:
        # only an original evaluation counts as "already logged"
        logged = [(r, c) for r, c, p in logged_test_evals() if not p.startswith("metrics-only")]
        if this_eval not in logged:
            parser.error(f"--metrics-only-reeval: no earlier evaluation of run_dir={args.run_dir} ckpt={args.ckpt} "
                         f"in {TEST_EVALS_LOG}; refusing a new test evaluation under this flag")
    if args.confirmatory_turning_test:
        # runs/DECISION_RULES_TURNING_TEST.md: one evaluation per checkpoint; a re-run only if no earlier
        # attempt got as far as any COND line (crashed before any condition result existed)
        cond_log = f"eval_{os.path.basename(os.path.normpath(args.run_dir))}_test_cond.log"
        stem = cond_log[:-len(".log")]
        n_entries = sum(1 for r, c, p in logged_test_evals()
                        if (r, c) == this_eval and p.startswith(CONFIRMATORY_TURNING_PURPOSE))
        old_logs = sorted(f for f in os.listdir(".") if f.startswith(stem + "_attempt") and f.endswith(".log"))
        old_logs += [cond_log] if os.path.exists(cond_log) else []
        if n_entries > len(old_logs):
            parser.error(f"--confirmatory-turning-test: {n_entries} earlier confirmatory entries for run_dir={args.run_dir} "
                         f"ckpt={args.ckpt} in {TEST_EVALS_LOG} but only {len(old_logs)} attempt logs ({old_logs}); "
                         f"cannot verify that no condition result was seen -- refusing")
        for f_old in old_logs:
            with open(f_old) as fh:
                if any(line.startswith("COND") for line in fh):
                    parser.error(f"--confirmatory-turning-test: {f_old} contains COND lines (condition results); "
                                 f"that attempt is final -- refusing a re-run")
        rerun = bool(old_logs) or n_entries > 0
        if os.path.exists(cond_log):
            attempt = f"{stem}_attempt{len(old_logs)}.log"  # attempts numbered from 1 in order
            assert not os.path.exists(attempt), f"{attempt} already exists"
            os.rename(cond_log, attempt)
        log_fh = open(cond_log, "x")
        sys.stdout = _Tee(sys.__stdout__, log_fh)
        sys.stderr = _Tee(sys.__stderr__, log_fh)
        print(f"Writing this output to {cond_log}")
        if rerun:
            kept = sorted(f for f in os.listdir(".") if f.startswith(stem + "_attempt") and f.endswith(".log"))
            print(f"Confirmatory RE-RUN after crash: earlier attempt logs {kept} have no COND line; "
                  f"{n_entries} earlier confirmatory entries in {TEST_EVALS_LOG}")
    with open(os.path.join(args.run_dir, "config.json")) as fh:
        _cfg = json.load(fh)
    if split_version_of(_cfg) == "v2" and not args.final_test:
        parser.error("the split v2 test split is evaluated only under runs/DECISION_RULES_FINAL_TEST.md (--final-test)")
    if args.final_test:
        # runs/DECISION_RULES_FINAL_TEST.md: split v2 / mirror runs only; one evaluation per checkpoint
        if (split_version_of(_cfg), map_transform_of(_cfg)) != ("v2", "mirror"):
            parser.error(f"--final-test: {args.run_dir} is not a split v2 / map_transform mirror run")
        final_rerun = final_test_guard.guard(f"eval_{os.path.basename(os.path.normpath(args.run_dir))}_test_final.log",
                                             args.run_dir, args.ckpt, final_test_guard.is_scenes_result,
                                             parser.error)
if args.degrade:
    if not args.run_dir:
        parser.error("--degrade requires --run-dir (the log is named after the run)")
    run_name = os.path.basename(os.path.normpath(args.run_dir))
    degrade_log = f"eval_{run_name}_tuning_{args.degrade}.log"
    if os.path.exists(degrade_log):
        parser.error(f"{degrade_log} already exists; refusing to overwrite it")
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
    v2x_kwargs = load_v2x_settings(run_config)                        # absent in pre-ablation configs = both on
    print(f"use_cpm={v2x_kwargs['use_cpm']}, use_coop={v2x_kwargs['use_coop']} (from config.json)")
    map_kwargs = load_map_settings(run_config)                        # absent = v1 (RGCN lane graph)
    if map_kwargs["map_version"] != "v1":
        print(f"map_version={map_kwargs['map_version']} (from config.json)")
    print(f"history_frames={history_kwargs['history_frames']} "
          f"(encoder={history_encoder if history_kwargs['history_frames'] else None}), "
          f"require_history_frames={history_kwargs['require_history_frames']} (from config.json)")
    run_loss = run_config.get("loss", "wta")  # absent in pre-loss-option configs
    predict_scale = loss_predicts_scale(run_loss)
    print(f"loss={run_loss}, predict_scale={predict_scale} (from config.json)")
    trained_degrade = run_config.get("degrade")  # absent in pre-degradation configs
    map_transform = map_transform_of(run_config)  # absent in pre-mirror configs = legacy
    split_version = split_version_of(run_config)  # absent in pre-v2 configs = v1
    print(f"map_transform={map_transform}, split_version={split_version} (from config.json)")
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
    v2x_kwargs = {}
    map_kwargs = {}
    history_encoder = "gru"
    predict_scale = False
    map_transform = "legacy"

dataset = PilotDataset(use_image_cache=use_image_cache, require_prev_frame=require_prev_frame,
                        use_speed=use_speed, cam_norm=cam_norm, drop_cam_xy=drop_cam_xy, degrade=args.degrade,
                        **history_kwargs, **v2x_kwargs, **map_kwargs, map_transform=map_transform)
if args.run_dir:
    _, tuning_set, test_set = three_way_scene_split(dataset, seed=42,  # 42 matches train.py's SCENE_SPLIT_SEED
                                                    split_version=split_version)
    val_set = test_set if args.split == "test" else tuning_set
    print(f"Evaluating on the {args.split.upper()} split ({len(val_set)} samples)")
    if args.split == "test":
        # Logged BEFORE evaluating, so an attempt is on record even if it crashes partway.
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
        except Exception as e:
            commit = f"<unavailable: {e}>"
        purpose = ("\tpurpose=metrics-only re-evaluation (adds miss/off-road; same checkpoint as the earlier "
                   "logged evaluation)") if args.metrics_only_reeval else ""
        if args.confirmatory_turning_test:
            purpose = f"\tpurpose={CONFIRMATORY_TURNING_PURPOSE}" + (CONFIRMATORY_RERUN_SUFFIX if rerun else "")
        if args.final_test:
            final_test_guard.record(args.run_dir, args.ckpt, final_rerun, split_version)
        else:
            with open(TEST_EVALS_LOG, "a") as f:
                f.write(f"{datetime.now().isoformat(timespec='seconds')}\trun_dir={args.run_dir}\t"
                        f"git_commit={commit}\tckpt={args.ckpt}{purpose}"
                        + (f"\tsplit_version={split_version}" if split_version != "v1" else "") + "\n")
            print(f"Appended this test evaluation to {TEST_EVALS_LOG}")
else:
    _, val_set = scene_split(dataset, 0.15, 42)  # unchanged old path
keys = [(dataset.samples[i][3], dataset.samples[i][1]) for i in val_set.indices]
loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=pilot_collate_fn)
model = load_model(args.ckpt, use_map=use_map, history_frames=dataset.history_frames,
                   history_encoder=history_encoder, predict_scale=predict_scale, **v2x_kwargs,
                   **map_kwargs).to(DEVICE).eval()
print(f"ckpt={args.ckpt}  use_map={use_map}")

cols = {k: [] for k in ("ade6", "fde6", "ade1", "fde1", "zade", "zfde", "valid", "first",
                         "cvade", "cvfde", "cv1", "cv_ok", "v_norm", "or1", "or6", "orgt", "bfde", "div",
                         "hchg", "nsteps", "jdist")}
if predict_scale:
    for lvl in COVERAGE_LEVELS:
        cols[f"cov{lvl}"], cols[f"covxy{lvl}"] = [], []
    cols["npts"] = []
lanes = LaneCentrelines(sorted(dataset.town_graphs), map_transform)
junctions = JunctionAreas(sorted(dataset.town_graphs))
score_idx = None
n_sensors = 0  # ego + coop LiDARs seen, to cross-check the fallback counter
reset_fallback_counts()
with torch.no_grad():
    for batch in loader:
        n_sensors += len(batch["points"]) + sum(len(cp) for cp in batch.get("coop_points", []))
        gt = batch["gt_traj"].to(DEVICE)                                  # [B, T, 2]
        mask = batch["gt_mask"].to(DEVICE).float()                        # [B, T]
        out = model(batch)
        n_out = 4 if predict_scale else 3  # (fused, trajs, scores[, scales])
        if not (isinstance(out, (tuple, list)) and len(out) == n_out):
            raise RuntimeError(f"Expected model to return {n_out} outputs, got {type(out)}")
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

        # ---- off-road: fraction of valid-step points > OFFROAD_THRESHOLD_M from the nearest lane centre ----
        trajs_np, gt_np = trajs.cpu().numpy(), gt.cpu().numpy()
        valid_np = mask.cpu().numpy().astype(bool)                          # [B, T]
        top_np = top.squeeze(1).cpu().numpy()
        or1, or6, orgt = torch.zeros(B), torch.zeros(B), torch.zeros(B)
        hchg, nsteps, jdist = torch.zeros(B), torch.zeros(B), torch.zeros(B)  # COND block (scene_conditions.py)
        for i in range(B):
            town = batch["town"][i]
            pose = raw_pose(dataset.loader, batch["agent_id"][i], batch["scene"][i], batch["frame"][i])
            off_modes = lanes.offroad(town, agent_to_carla(trajs_np[i], pose, town, map_transform))  # [K, T]
            off_gt = lanes.offroad(town, agent_to_carla(gt_np[i], pose, town, map_transform))       # [T]
            vm = valid_np[i]  # not `v`: that is the CV velocity above, still used for v_norm
            nv = max(int(vm.sum()), 1)
            or1[i] = off_modes[top_np[i]][vm].sum() / nv
            or6[i] = off_modes[:, vm].sum() / (nv * K)
            orgt[i] = off_gt[vm].sum() / nv
            hchg[i], nsteps[i] = heading_change_deg(gt_np[i], vm)
            jdist[i] = junctions.distance(town, agent_to_carla(np.zeros(2), pose, town, map_transform))

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
        cols["or1"].append(or1)
        cols["or6"].append(or6)
        cols["orgt"].append(orgt)
        cols["hchg"].append(hchg)
        cols["nsteps"].append(nsteps)
        cols["jdist"].append(jdist)

        # ---- mode ranking metrics (ranking_metrics.py) ----
        cols["bfde"].append(brier_min_fde(fde_modes, scores).cpu())
        cols["div"].append(mode_diversity(trajs).cpu())
        if predict_scale:
            pt_hits, xy_hits, npts = laplace_coverage(trajs, out[3], gt, mask, top.squeeze(1))
            for lvl in COVERAGE_LEVELS:
                cols[f"cov{lvl}"].append(pt_hits[lvl].cpu())
                cols[f"covxy{lvl}"].append(xy_hits[lvl].cpu())
            cols["npts"].append(npts.cpu())

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

# ---------- safety metrics (evaluation only). Every line starts with "SAFE", so analyze_seeds.py's
# ^Town / ^ALL row parsers never see them. ----------
miss1 = c["fde1"] > MISS_THRESHOLD_M
miss6 = c["fde6"] > MISS_THRESHOLD_M  # fde6 = best-of-6 FDE
towns_of = [k[0] for k in keys]
town_sel = {t: torch.tensor([x == t for x in towns_of]) & valid for t in sorted(set(towns_of))}
gt_offroad = {t: float(c["orgt"][m].mean()) for t, m in town_sel.items() if m.any()}
unreliable = {t for t, r in gt_offroad.items() if r > GT_OFFROAD_UNRELIABLE}

print(f"\nSAFETY  miss: FDE > {MISS_THRESHOLD_M} m (top-1 / best-of-6); off-road: fraction of valid-step points "
      f"> {OFFROAD_THRESHOLD_M} m from the nearest lane-centre waypoint (top-1 / all 6 modes / ground truth)")
print("SAFE control: ground-truth off-road rate per town (off-road numbers flagged UNRELIABLE above "
      f"{GT_OFFROAD_UNRELIABLE:.0%}):")
for t, r in gt_offroad.items():
    print(f"SAFE   {t}: GT off-road {r:.4f}" + ("  -> UNRELIABLE" if t in unreliable else ""))
print(f"SAFE {'group':24s} {'n':>5s} {'miss1':>7s} {'miss6':>7s} {'offr1':>7s} {'offr6':>7s} {'offrGT':>7s}")


def safety_row(label, sel, towns):
    n = int(sel.sum())
    if not n:
        print(f"SAFE {label:24s} {0:5d}")
        return
    flag = sorted(towns & unreliable)
    print(f"SAFE {label:24s} {n:5d} {miss1[sel].float().mean():7.4f} {miss6[sel].float().mean():7.4f} "
          f"{c['or1'][sel].mean():7.4f} {c['or6'][sel].mean():7.4f} {c['orgt'][sel].mean():7.4f}"
          + (f"  off-road UNRELIABLE ({', '.join(flag)})" if flag else ""))


def safety_block(label, sel, towns):
    safety_row(label, sel, towns)
    safety_row(f"  stationary", sel & stationary, towns)
    safety_row(f"  moving", sel & moving, towns)


for key in sorted(set(keys)):
    safety_block(f"{key[0]} {key[1]}", torch.tensor([k == key for k in keys]) & valid, {key[0]})
for t, sel in town_sel.items():
    safety_block(f"{t} (town)", sel, {t})
safety_block("ALL", valid, set(town_sel))

# ---------- mode ranking metrics (evaluation only). Every line starts with "RANK", so the ^Town / ^ALL /
# ^SAFE parsers never see them. ----------
cov_hdr = "".join(f" {f'pt{l}':>7s} {f'xy{l}':>7s}" for l in COVERAGE_LEVELS) if predict_scale else ""
print(f"\nRANKING  brierFDE = minFDE6 + (1 - p)^2, p = softmax score of the min-FDE mode; div = mean pairwise "
      f"distance between the {K} modes at the final horizon step"
      + ("; calibration of the top mode's Laplace: pt<L> = fraction of valid GT points inside the joint box "
         "of per-coordinate sqrt(L%) central intervals (nominal L%), xy<L> = fraction of valid GT coordinates "
         "inside the per-coordinate L% central interval (nominal L%)" if predict_scale else ""))
print(f"RANK {'group':24s} {'n':>5s} {'minFDE6':>8s} {'brierFDE':>8s} {'div':>8s}{cov_hdr}")


def rank_row(label, sel):
    n = int(sel.sum())
    if not n:
        print(f"RANK {label:24s} {0:5d}")
        return
    cov = ""
    if predict_scale:
        npts = c["npts"][sel].sum().clamp(min=1)
        cov = "".join(f" {c[f'cov{l}'][sel].sum() / npts:7.4f} {c[f'covxy{l}'][sel].sum() / (2 * npts):7.4f}"
                      for l in COVERAGE_LEVELS)
    print(f"RANK {label:24s} {n:5d} {c['fde6'][sel].mean():8.3f} {c['bfde'][sel].mean():8.3f} "
          f"{c['div'][sel].mean():8.3f}{cov}")


for key in sorted(set(keys)):
    rank_row(f"{key[0]} {key[1]}", torch.tensor([k == key for k in keys]) & valid)
rank_row("ALL", valid)

# ---------- condition-based evaluation (evaluation only; runs/DECISION_RULES_CONDITIONS.md, labels from
# ground truth in scene_conditions.py). Every line starts with "COND", so the ^Town / ^ALL / ^SAFE / ^RANK
# parsers never see them. ----------
turning = moving & (c["hchg"].abs() > TURN_THRESHOLD_DEG)
straight = moving & ~turning
near_junction = c["jdist"] <= JUNCTION_RADIUS_M
cond_masks = (("stationary", stationary), ("moving", moving), ("turning", turning), ("straight", straight),
              ("near_junction", near_junction), ("not_near_junction", ~near_junction))
print(f"\nCONDITIONS  from ground truth: stationary / moving = frame-1 -> frame displacement = 0 / > 0; "
      f"turning = moving and |heading change over the GT| > {TURN_THRESHOLD_DEG:g} deg (path direction, first vs "
      f"last step >= {MIN_STEP_M:g} m); straight = moving and not turning; near_junction = current position within "
      f"{JUNCTION_RADIUS_M:g} m of an in-junction lane-centre waypoint (CARLA OpenDRIVE, junction_waypoints_*.npy)")
n_mov = int((valid & moving).sum())
n_short = int((valid & moving & (c["nsteps"] < 2)).sum())
print(f"COND note: {n_short} of {n_mov} moving samples have < 2 GT steps >= {MIN_STEP_M:g} m "
      f"(heading change 0 -> straight)")
print(f"COND {'town':7s} {'condition':18s} {'n':>5s} {'minADE6':>8s} {'ADE1':>8s} {'minFDE6':>8s}")


def cond_row(town, label, sel):
    n = int(sel.sum())
    if not n:
        print(f"COND {town:7s} {label:18s} {0:5d}")
        return
    print(f"COND {town:7s} {label:18s} {n:5d} {c['ade6'][sel].mean():8.3f} {c['ade1'][sel].mean():8.3f} "
          f"{c['fde6'][sel].mean():8.3f}")


for town, tsel in [("ALL", valid)] + list(town_sel.items()):
    for label, m in cond_masks:
        cond_row(town, label, tsel & m)
