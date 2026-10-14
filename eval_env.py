"""
eval_env.py
Evaluation of an env-task run (train.py --task env; runs/DESIGN_ENV.md Stage 3, runs/DECISION_RULES_ENV.md).

Per known target of the TUNING split (split_version from the run's config; or the TEST split, see below): minADE6, ADE1, minFDE6, FDE1, miss
rates (top-1 / best-of-6, FDE > 2 m), and the constant-velocity and stay-put baselines (env_metrics.py). Rows by
tuning scene and ALL, for the subsets of env_metrics.SUBSETS: visibility class (a) / (b), condition
(moving / stationary / turning / straight), category (car / two-wheeler). Coverage: the model's known targets /
all kept targets (a + b + c) over every D sample of the split's scenes, with n_removed as the denominator note.

Writes <run-dir>/eval_env_<split>_<ckpt stem>.npz (per-target arrays, read by analyze_env.py) and prints the
tables (ENV / COV lines).

--split test --confirm-final-test: the final test evaluation of runs/DECISION_RULES_FINAL_TEST.md (split v2 runs
only). Output (stdout and stderr) goes to eval_env_<run>_test_final.log; the evaluation is recorded in
runs/TEST_EVALS.log with purpose "final test evaluation (split v2)" before it starts; a re-run is allowed only if
every earlier attempt's log has no ENV / COV line (final_test_guard.py). An existing test npz is never overwritten.

Run in gpu-test-venv:  python eval_env.py --run-dir runs/E_V2X_seed0 --ckpt runs/E_V2X_seed0/checkpoint_epoch12.pt
"""
import os
import json
import argparse
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader

import final_test_guard

from env_metrics import (displacement_metrics, baseline_metrics, cv_prediction, condition_labels, subset_masks,
                         coverage, SUBSETS, METRICS, BASELINE_METRICS)

PER_TARGET = ("tid", "scene", "town", "vis", "cat4", "moving", "turning", "straight", "stationary") + METRICS + \
    tuple(f"cv_{m}" for m in BASELINE_METRICS) + tuple(f"stay_{m}" for m in BASELINE_METRICS)


def collect(batch, trajs, logits):
    """One batch -> per-target numpy arrays (real targets only, batch order). trajs [B, T, 6, 30, 2],
    logits [B, T, 6] (any device); batch from env_dataset.env_collate_fn."""
    tm = batch["target_mask"].cpu().numpy()
    take = lambda k: batch[k].cpu().numpy()[tm]
    fut, mask = take("t_fut"), take("t_fut_mask")
    out = {"tid": take("t_tid"), "vis": take("t_vis"), "cat4": take("t_cat4")}
    nt = tm.sum(1)
    out["scene"] = np.repeat(np.asarray(batch["scene"]), nt)
    out["town"] = np.repeat(np.asarray(batch["town"]), nt)
    out.update(condition_labels(fut, mask, take("t_moving")))
    out.update(displacement_metrics(trajs.detach().cpu().numpy()[tm], logits.detach().cpu().numpy()[tm], fut, mask))
    for name, pred in (("cv", cv_prediction(take("t_cv_vel"), fut.shape[1])), ("stay", np.zeros_like(fut))):
        for k, v in baseline_metrics(pred, fut, mask).items():
            out[f"{name}_{k}"] = v
    return out


def concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in PER_TARGET}


def table_lines(t, scenes):
    """ENV lines: per scene and ALL, per subset: n, the model's METRICS, then CV and stay-put (ade fde miss)."""
    lines = [f"ENV {'scene':>9s} {'subset':19s} {'n':>6s} | " + " ".join(f"{m:>7s}" for m in METRICS)
             + " | cv: ade fde miss | stay: ade fde miss"]
    masks = subset_masks(t)
    keys = [("ALL", np.ones(len(t["tid"]), bool))] + [(f"{tw[-2:]}#{s}", np.asarray(t["scene"]) == s) for tw, s in scenes]
    for label, sel in keys:
        for sub in SUBSETS:
            m = sel & masks[sub]
            if not m.any():
                lines.append(f"ENV {label:>9s} {sub:19s} {0:6d}")
                continue
            mean = lambda k: float(np.mean(t[k][m]))
            lines.append(f"ENV {label:>9s} {sub:19s} {int(m.sum()):6d} | " + " ".join(f"{mean(k):7.3f}" for k in METRICS)
                         + " | " + " ".join(f"{mean('cv_' + k):6.3f}" for k in BASELINE_METRICS)
                         + " | " + " ".join(f"{mean('stay_' + k):6.3f}" for k in BASELINE_METRICS))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["tuning", "test"], default="tuning")
    ap.add_argument("--confirm-final-test", action="store_true",
                    help="required for --split test: the final, logged test evaluation (runs/DECISION_RULES_FINAL_TEST.md)")
    args = ap.parse_args()
    final_rerun = False
    if args.split == "test":
        if not args.confirm_final_test:
            ap.error("--split test requires --confirm-final-test (runs/DECISION_RULES_FINAL_TEST.md)")
        with open(os.path.join(args.run_dir, "config.json")) as f:
            c0 = json.load(f)
        if (c0.get("split_version"), c0.get("map_transform")) != ("v2", "mirror"):
            ap.error(f"{args.run_dir}: the final test is for split v2 / mirror runs only")
        npz = os.path.join(args.run_dir, f"eval_env_test_{os.path.splitext(os.path.basename(args.ckpt))[0]}.npz")
        if os.path.exists(npz):
            ap.error(f"{npz} exists: this checkpoint's test evaluation produced results; refusing")
        run = os.path.basename(os.path.normpath(args.run_dir))
        final_rerun = final_test_guard.guard(f"eval_env_{run}_test_final.log", args.run_dir, args.ckpt,
                                             final_test_guard.is_env_result, ap.error)
    elif args.confirm_final_test:
        ap.error("--confirm-final-test is for --split test")

    from env_dataset import EnvDataset, env_collate_fn
    from env_model import EnvFusion
    from scene_split import three_way_scene_split, split_version_of

    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)
    assert cfg.get("task") == "env", f"{args.run_dir} is not an env-task run"
    split_version = split_version_of(cfg)
    print(f"--run-dir {args.run_dir}: v2x={cfg['v2x']}, no_map={cfg['no_map']}, map_transform={cfg['map_transform']}, "
          f"split_version={split_version} (from config.json)")
    with open(os.path.join(args.run_dir, "cam_norm.json")) as f:
        cam_norm = json.load(f)
    with open(os.path.join(args.run_dir, "env_norm.json")) as f:
        env_norm = json.load(f)
    ds = EnvDataset(v2x=cfg["v2x"], use_map=not cfg["no_map"], map_transform=cfg["map_transform"], env_norm=env_norm,
                    use_image_cache=cfg["use_image_cache"], use_speed=cfg["use_speed"], drop_cam_xy=cfg["drop_cam_xy"])
    ds.cam_norm = cam_norm
    assert ds.cache["manifest"]["git_commit"] == cfg["env_cache_commit"], "env cache differs from the run's"
    _, tuning, test = three_way_scene_split(ds, seed=42, split_version=split_version)
    part = test if args.split == "test" else tuning
    scenes = sorted({(ds.samples[i][3], ds.samples[i][1]) for i in part.indices})
    print(f"Evaluating on the {args.split.upper()} split: {len(part)} samples, scenes {[s for _, s in scenes]}")
    if args.split == "test":
        final_test_guard.record(args.run_dir, args.ckpt, final_rerun, split_version)  # before evaluating

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnvFusion(device=device, v2x=cfg["v2x"] == "on", use_map=not cfg["no_map"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state_dict"])
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in DataLoader(part, batch_size=8, shuffle=False, num_workers=2, collate_fn=env_collate_fn):
            trajs, logits = model(batch)
            parts.append(collect(batch, trajs, logits))
    t = concat(parts)
    assert len(np.unique(t["tid"])) == len(t["tid"]), "a target was evaluated twice"

    for line in table_lines(t, scenes):
        print(line)
    c = ds.cache
    split_rows = np.nonzero(np.isin(c["sample_scene"], [s for _, s in scenes]))[0]  # every D sample of these scenes
    print(f"\nCOV coverage over all {len(split_rows)} D samples of the {args.split} scenes "
          f"(predicted = known targets of v2x={cfg['v2x']}; all = kept targets a+b+c; removed = window-filtered):")
    for label, rows in [("ALL", split_rows)] + [(f"{tw[-2:]}#{s}", split_rows[c["sample_scene"][split_rows] == s])
                                                for tw, s in scenes]:
        cv = coverage(c, rows, cfg["v2x"])
        print(f"COV {label:>9s} samples={cv['samples']:5d} a={cv['a']:6d} b={cv['b']:6d} c={cv['c']:6d} "
              f"all={cv['all_kept']:6d} removed={cv['removed']:5d} predicted={cv['predicted']:6d} "
              f"coverage={cv['coverage']:.4f} (incl. removed {cv['coverage_incl_removed']:.4f})")
    n_eval = len(t["tid"])
    n_cov = coverage(c, split_rows, cfg["v2x"])["predicted"]
    assert n_eval == n_cov, f"evaluated {n_eval} targets, coverage counts {n_cov} predicted ones"

    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    out = os.path.join(args.run_dir, f"eval_env_{args.split}_{stem}.npz")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    np.savez(out, **t, meta=json.dumps({"run_dir": args.run_dir, "ckpt": args.ckpt, "split": args.split,
                                        "v2x": cfg["v2x"], "no_map": cfg["no_map"], "split_version": split_version,
                                        "map_transform": cfg["map_transform"], "git_commit": commit,
                                        "scenes": [[tw, s] for tw, s in scenes],
                                        "coverage": coverage(c, split_rows, cfg["v2x"])}))
    print(f"\nwrote {out} ({n_eval} targets)")


if __name__ == "__main__":
    main()
