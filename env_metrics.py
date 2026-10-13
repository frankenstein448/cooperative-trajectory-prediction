"""
env_metrics.py
Per-target metrics, labels, subsets, coverage and the decision-rule computation for the env task (Stage 3 / 4,
runs/DESIGN_ENV.md, runs/DECISION_RULES_ENV.md). Pure numpy: used by eval_env.py and analyze_env.py and
unit-tested on synthetic arrays (test_env_stage2.py).

Per target (all positions in the target's own frame, metres):
  minADE6 / minFDE6     best of the 6 modes; ADE1 / FDE1 of the top-scored mode
  miss1 / miss6         FDE1 / minFDE6 > MISS_THRESHOLD_M (2 m, as Part 1's safety metrics)
  cv_*                  constant velocity: the frame-1 -> frame displacement repeated for 30 steps
  stay_*                stay-put: the current position for all 30 steps
Labels (Part 1 definitions, scene_conditions.py, applied to the target's own-frame ground truth):
  moving = nonzero frame-1 -> frame displacement (cache speed > 0); stationary = not moving;
  turning = moving AND |heading change over the 30-step GT| > 30 deg; straight = moving AND not turning.
  car = car or truck (cat4 0, 1; V2X-Sim has no truck, police cars are cars); two_wheeler = bicycle or motorcycle.
"""
import statistics as st

import numpy as np

from scene_conditions import heading_change_deg, TURN_THRESHOLD_DEG

MISS_THRESHOLD_M = 2.0     # = safety_metrics.MISS_THRESHOLD_M
METRICS = ("minade6", "ade1", "minfde6", "fde1", "miss1", "miss6")
BASELINE_METRICS = ("ade", "fde", "miss")


def displacement_metrics(trajs, logits, fut, mask):
    """trajs [N, K, T, 2], logits [N, K], fut [N, T, 2], mask [N, T] bool (>= 1 valid step per target)
    -> {metric: [N]} (METRICS)."""
    trajs, fut, mask = np.asarray(trajs, np.float64), np.asarray(fut, np.float64), np.asarray(mask, bool)
    d = np.linalg.norm(trajs - fut[:, None], axis=-1)                       # [N, K, T]
    nv = mask.sum(-1)
    assert (nv > 0).all(), "a target without any valid future step"
    ade = (d * mask[:, None]).sum(-1) / nv[:, None]                          # [N, K]
    last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)              # last valid step
    fde = d[np.arange(len(d)), :, last]                                      # [N, K]
    top = np.argmax(np.asarray(logits), axis=1)
    i = np.arange(len(d))
    out = {"minade6": ade.min(1), "ade1": ade[i, top], "minfde6": fde.min(1), "fde1": fde[i, top]}
    out["miss1"] = (out["fde1"] > MISS_THRESHOLD_M).astype(np.float64)
    out["miss6"] = (out["minfde6"] > MISS_THRESHOLD_M).astype(np.float64)
    return out


def baseline_metrics(pred, fut, mask):
    """Single-hypothesis prediction pred [N, T, 2] -> {"ade", "fde", "miss"} [N]."""
    m = displacement_metrics(np.asarray(pred)[:, None], np.zeros((len(pred), 1)), fut, mask)
    return {"ade": m["ade1"], "fde": m["fde1"], "miss": m["miss1"]}


def cv_prediction(vel, horizon=30):
    """vel [N, 2] (metres per frame, own frame) -> [N, horizon, 2]: k * vel for k = 1 .. horizon."""
    return np.asarray(vel, np.float64)[:, None, :] * np.arange(1, horizon + 1)[None, :, None]


def condition_labels(fut, mask, moving):
    """-> dict of bool [N]: moving, stationary, turning, straight (module docstring)."""
    moving = np.asarray(moving, bool)
    hc = np.array([abs(heading_change_deg(f, m)[0]) for f, m in zip(np.asarray(fut), np.asarray(mask))])
    turning = moving & (hc > TURN_THRESHOLD_DEG)
    return {"moving": moving, "stationary": ~moving, "turning": turning, "straight": moving & ~turning}


SUBSETS = ("all", "moving", "stationary", "turning", "straight", "a", "b", "a_moving", "b_moving",
           "car", "two_wheeler", "car_moving", "two_wheeler_moving")


def subset_masks(t):
    """t: per-target arrays with vis, cat4, moving, turning, straight -> {subset: bool [N]} (SUBSETS)."""
    vis, cat, mov = np.asarray(t["vis"]), np.asarray(t["cat4"]), np.asarray(t["moving"], bool)
    car, tw = cat <= 1, cat >= 2
    m = {"all": np.ones(len(vis), bool), "moving": mov, "stationary": ~mov,
         "turning": np.asarray(t["turning"], bool), "straight": np.asarray(t["straight"], bool),
         "a": vis == 0, "b": vis == 1, "a_moving": (vis == 0) & mov, "b_moving": (vis == 1) & mov,
         "car": car, "two_wheeler": tw, "car_moving": car & mov, "two_wheeler_moving": tw & mov}
    return {k: m[k] for k in SUBSETS}


def coverage(cache, rows, v2x):
    """Coverage over the given cache sample rows (all of a split's D samples, including those without a class-(a)
    target): predicted = the model's known targets; all = kept targets (a + b + c). n_removed (vehicles within
    50 m dropped by the window filter: too short a track, or a step > 10 m) is the denominator note."""
    vis = []
    removed = 0
    for r in rows:
        lo, hi = int(cache["sample_ptr"][r]), int(cache["sample_ptr"][r + 1])
        vis.append(np.asarray(cache["vis"][lo:hi]))
        removed += int(cache["n_removed"][r])
    vis = np.concatenate(vis) if vis else np.zeros(0, int)
    n = {k: int((vis == i).sum()) for i, k in enumerate("abc")}
    total = len(vis)
    predicted = n["a"] + (n["b"] if v2x == "on" else 0)
    return {"samples": len(rows), "a": n["a"], "b": n["b"], "c": n["c"], "all_kept": total, "removed": removed,
            "predicted": predicted, "coverage": predicted / max(total, 1),
            "coverage_incl_removed": predicted / max(total + removed, 1)}


def scene_values(t, metric, mask, scenes):
    """Mean of t[metric] over targets in `mask`, per scene (None if a scene has none) and pooled ('ALL')."""
    vals, sc = np.asarray(t[metric], np.float64), np.asarray(t["scene"])
    out = {s: (float(vals[mask & (sc == s)].mean()) if (mask & (sc == s)).any() else None) for s in scenes}
    out["ALL"] = float(vals[mask].mean()) if mask.any() else None
    return out


def rule(with_runs, without_runs, scenes, min_scene_wins=7):
    """Part 1's rule form on per-run {scene: value, 'ALL': value} dicts (lower = better):
    full form (every scene has samples): 'with' helps iff its seed-mean is lower in >= min_scene_wins of the
    scenes AND (seed-mean ALL without - with) > 2 x max(seed std with, seed std without);
    pooled form (some scene has no sample): iff seed-mean ALL with < without AND the same gap condition.
    A deterministic baseline enters as `without_runs` with identical values per run (seed std 0)."""
    m = lambda runs, k: st.mean(r[k] for r in runs)
    sd = lambda runs, k: st.stdev([r[k] for r in runs]) if len(runs) > 1 else 0.0
    has = [s for s in scenes if all(r[s] is not None for r in with_runs + without_runs)]
    full = len(has) == len(scenes)
    wins = sum(m(with_runs, s) < m(without_runs, s) for s in has)
    mw, mo = m(with_runs, "ALL"), m(without_runs, "ALL")
    gap, thr = mo - mw, 2 * max(sd(with_runs, "ALL"), sd(without_runs, "ALL"))
    ok = (wins >= min_scene_wins if full else mw < mo) and gap > thr
    return {"form": "full" if full else "pooled", "scenes_with_samples": len(has), "wins": wins, "with": mw,
            "without": mo, "sd_with": sd(with_runs, "ALL"), "sd_without": sd(without_runs, "ALL"), "gap": gap,
            "threshold": thr, "helps": bool(ok)}
