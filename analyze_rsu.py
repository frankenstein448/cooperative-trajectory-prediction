"""
analyze_rsu.py
Applies runs/DECISION_RULES_RSU.md to the per-target evaluations (eval_env.py) of E_V2XRSU (train.py --task env
--v2x on --use-rsu) and E_V2X, seeds 0-2, epoch-12 checkpoints, split v2.

  H1 coverage (deterministic): kept targets in V and in V + R (eval meta coverage); the RSU adds coverage if
     V + R exceeds V by >= 2 percentage points. (Already decided on the tuning split and recorded in the rules
     file; recomputed here from the evaluation as a consistency check.)
  H2 RSU-only vs constant velocity: moving targets in R; metric top-1 ADE1 vs the constant-velocity ADE.
     Eligible scenes: >= 20 moving R targets (ground-truth labels; identical for every seed). Fewer than 6 eligible
     scenes -> "not testable". Helps iff the seed-mean ADE1 is lower than constant velocity in at least
     ceil(7/8 x eligible) eligible scenes AND the pooled gap (constant velocity - seed-mean ADE1, over every moving
     R target of the split) exceeds 2 x the model's seed std of the pooled ADE1 (the baseline's seed std is 0).
  H3 E_V2XRSU vs E_V2X on moving targets in V (E_V2XRSU's V targets must be exactly E_V2X's targets): minADE6.
     Helps iff the E_V2XRSU seed mean is lower in >= 7 of the 8 scenes AND the pooled gap exceeds 2 x the larger
     seed std of the two models; otherwise "no detectable effect" (either direction).
  (env_metrics.rule implements both forms; a scene without targets of the set switches it to the pooled form.)
Descriptive only: ALL and stationary targets; ADE1 on V; minADE6 on R; per-scene tables; H2 / H3 sets split by
zero-width (annotated box width 0, read from the env target cache by target index) vs width > 0.

Run in gpu-test-venv:  python analyze_rsu.py [--split tuning|test]
"""
import os
import json
import math
import argparse
import statistics as st

import numpy as np

from env_metrics import subset_masks, rsu_subset_masks, scene_values, rule, METRICS
from env_dataset import ENV_CACHE_DIR

H1_MIN_GAIN_PP = 2.0
H2_MIN_TARGETS, H2_MIN_SCENES = 20, 6
H3_MIN_WINS = 7


def load(path):
    z = np.load(path, allow_pickle=False)
    t = {k: z[k] for k in z.files if k != "meta"}
    return t, json.loads(str(z["meta"]))


def masks_of(t):
    m = subset_masks(t)
    m.update(rsu_subset_masks(t))
    return m


def same_across_seeds(ts, name):
    """Every seed evaluated the same targets with the same labels; returns seed 0's arrays sorted by tid."""
    o0 = np.argsort(ts[0]["tid"])
    for t in ts[1:]:
        o = np.argsort(t["tid"])
        for k in ("tid", "vis", "moving", "stationary", "scene", "cv_ade"):
            assert np.array_equal(t[k][o], ts[0][k][o0]), f"{name}: {k} differs between seeds"


def seed_table(label, ts, sub):
    vals = []
    for t in ts:
        m = masks_of(t)[sub]
        vals.append([float(t[k][m].mean()) if m.any() else float("nan") for k in METRICS])
    arr = np.array(vals)
    n = int(masks_of(ts[0])[sub].sum())
    cells = [f"{arr[:, i].mean():7.3f}+/-{arr[:, i].std(ddof=1) if len(arr) > 1 else 0.0:5.3f}"
             if np.isfinite(arr[:, i]).all() else f"{'n/a':>15s}" for i in range(len(METRICS))]
    print(f"  {label:9s} {sub:19s} n={n:6d} " + " ".join(f"{k}={c}" for k, c in zip(METRICS, cells)))


def compare(label, w_ts, w_metric, w_mask, o_ts, o_metric, o_mask, scenes, min_wins):
    """Rule on per-seed scene values; w_mask / o_mask: t -> bool mask."""
    w = [scene_values(t, w_metric, w_mask(t), scenes) for t in w_ts]
    o = [scene_values(t, o_metric, o_mask(t), scenes) for t in o_ts]
    r = rule(w, o, scenes, min_scene_wins=min_wins)
    print(f"\n{label}")
    print("  per scene seed means (with / without): " + "  ".join(
        f"#{s}: " + (f"{st.mean(x[s] for x in w):.3f}/{st.mean(x[s] for x in o):.3f}"
                     if all(x[s] is not None for x in w + o) else "n/a") for s in scenes))
    print(f"  form={r['form']} (scenes with targets {r['scenes_with_samples']}/{len(scenes)})  wins={r['wins']} "
          f"(need {min_wins})  pooled {r['with']:.3f} (sd {r['sd_with']:.3f}) vs {r['without']:.3f} "
          f"(sd {r['sd_without']:.3f})  gap {r['gap']:+.3f} > {r['threshold']:.3f}?")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tuning", "test"], default="tuning")
    ap.add_argument("--rsu-prefix", default="E_V2XRSU")
    ap.add_argument("--ref-prefix", default="E_V2X")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    npz = f"eval_env_{args.split}_checkpoint_epoch12.npz"
    R, V2X, metas = [], [], {}
    for prefix, out, rsu in ((args.rsu_prefix, R, True), (args.ref_prefix, V2X, False)):
        for s in seeds:
            name = f"{prefix}_seed{s}"
            t, m = load(f"runs/{name}/{npz}")
            assert (m["split"], m["split_version"], m["map_transform"], m["v2x"], m["no_map"]) == \
                (args.split, "v2", "mirror", "on", False), f"{name}: {m}"
            assert m.get("use_rsu", False) == rsu, f"{name}: use_rsu={m.get('use_rsu')}"
            out.append(t)
            metas[name] = m
    scene_lists = {json.dumps(m["scenes"]) for m in metas.values()}
    assert len(scene_lists) == 1, "scene lists differ between runs"
    scenes = [s for _, s in json.loads(scene_lists.pop())]
    same_across_seeds(R, args.rsu_prefix)
    same_across_seeds(V2X, args.ref_prefix)
    r0, v0 = R[0], V2X[0]
    assert set(np.unique(r0["vis"])) <= {0, 1, 2} and set(np.unique(v0["vis"])) <= {0, 1}
    assert np.array_equal(np.sort(r0["tid"][r0["vis"] <= 1]), np.sort(v0["tid"])), \
        f"{args.rsu_prefix}'s V targets are not exactly {args.ref_prefix}'s targets"
    zw_all = np.load(os.path.join(ENV_CACHE_DIR, "env_targets.npz"))["size"][:, 0] == 0
    zw = lambda t: zw_all[t["tid"]]
    print(f"runs: {args.rsu_prefix} / {args.ref_prefix} seeds {seeds}; {args.split.upper()} split v2, scenes {scenes}")

    m = masks_of(r0)
    print(f"\nTarget counts ({args.rsu_prefix}): V {int(m['v'].sum())}, R {int(m['r'].sum())}; moving V "
          f"{int(m['v_moving'].sum())}, moving R {int(m['r_moving'].sum())}; zero-width: V {int((m['v'] & zw(r0)).sum())}, "
          f"R {int((m['r'] & zw(r0)).sum())}")
    print("  moving R per scene: " + "  ".join(f"#{s}: {int((m['r_moving'] & (r0['scene'] == s)).sum())}" for s in scenes))
    print("  moving V per scene: " + "  ".join(f"#{s}: {int((m['v_moving'] & (r0['scene'] == s)).sum())}" for s in scenes))

    # ---------- H1 ----------
    print("\nH1 coverage (eval meta; all D samples of the split's scenes):")
    covs = {json.dumps(metas[f"{args.rsu_prefix}_seed{s}"]["coverage"], sort_keys=True) for s in seeds}
    assert len(covs) == 1, "coverage differs between seeds"
    c = json.loads(covs.pop())
    cov_v, cov_vr = (c["a"] + c["b"]) / c["all_kept"], c["predicted"] / c["all_kept"]
    assert c["predicted"] == c["a"] + c["b"] + c["r"]
    gain = 100 * (cov_vr - cov_v)
    h1 = "the RSU adds coverage" if gain >= H1_MIN_GAIN_PP else "no coverage gain >= 2 pp"
    print(f"  kept {c['all_kept']}: V {c['a'] + c['b']} ({100 * cov_v:.2f}%), V+R {c['predicted']} ({100 * cov_vr:.2f}%), "
          f"gain {gain:+.2f} pp (bar {H1_MIN_GAIN_PP} pp) -> {h1}")

    # ---------- H2 ----------
    print(f"\nH2: {args.rsu_prefix} ADE1 vs constant velocity ADE, moving R targets")
    n_per = {s: int((m["r_moving"] & (r0["scene"] == s)).sum()) for s in scenes}
    eligible = [s for s in scenes if n_per[s] >= H2_MIN_TARGETS]
    print(f"  eligible scenes (>= {H2_MIN_TARGETS} moving R targets): {eligible} ({len(eligible)}; need >= {H2_MIN_SCENES})")
    if len(eligible) < H2_MIN_SCENES:
        h2 = "not testable"
    else:
        k = math.ceil(7 / 8 * len(eligible))
        r = compare(f"  H2 rule over the {len(eligible)} eligible scenes (wins needed ceil(7/8 x {len(eligible)}) = {k})",
                    R, "ade1", lambda t: masks_of(t)["r_moving"], R, "cv_ade", lambda t: masks_of(t)["r_moving"],
                    eligible, k)
        h2 = "HELPS (better than constant velocity)" if r["helps"] else "no detectable effect"
    print(f"  -> H2: {h2}")

    # ---------- H3 ----------
    print(f"\nH3: {args.rsu_prefix} vs {args.ref_prefix} minADE6, moving V targets")
    r = compare(f"  H3 rule over the {len(scenes)} scenes", R, "minade6", lambda t: masks_of(t)["v_moving"],
                V2X, "minade6", lambda t: masks_of(t)["moving"], scenes, H3_MIN_WINS)
    h3 = "HELPS / better" if r["helps"] else "no detectable effect"
    print(f"  -> H3: {h3}")

    # ---------- descriptive ----------
    print("\nDESCRIPTIVE (no verdicts). Seed mean +/- std, pooled over the split:")
    for sub in ("all", "moving", "stationary", "v", "v_moving", "r", "r_moving"):
        seed_table(args.rsu_prefix, R, sub)
    for sub in ("all", "moving", "stationary"):
        seed_table(args.ref_prefix, V2X, sub)
    for sub in ("r_moving", "v_moving"):
        mm = masks_of(r0)[sub]
        print(f"  baseline  {sub:19s} cv ade={r0['cv_ade'][mm].mean():.3f} fde={r0['cv_fde'][mm].mean():.3f} "
              f"miss={r0['cv_miss'][mm].mean():.3f} | stay ade={r0['stay_ade'][mm].mean():.3f}")
    print("\nDESCRIPTIVE: H2 / H3 sets split by zero-width (pooled seed mean +/- std; no verdict):")
    pooled = lambda ts, metric, sel: [float(t[metric][sel(t)].mean()) for t in ts]
    fmt = lambda v: f"{st.mean(v):.3f}+/-{st.stdev(v) if len(v) > 1 else 0.0:.3f}"
    for zname, zsel in (("width > 0", lambda t: ~zw(t)), ("zero-width", zw)):
        r_sel = lambda t: masks_of(t)["r_moving"] & zsel(t)
        v_sel = lambda t: masks_of(t)["v_moving"] & zsel(t)
        o_sel = lambda t: masks_of(t)["moving"] & zsel(t)
        n2, n3 = int(r_sel(r0).sum()), int(v_sel(r0).sum())
        h2_txt = (f"ADE1 {fmt(pooled(R, 'ade1', r_sel))} vs CV {float(r0['cv_ade'][r_sel(r0)].mean()):.3f}"
                  if n2 else "n/a")
        h3_txt = (f"minADE6 {fmt(pooled(R, 'minade6', v_sel))} vs {args.ref_prefix} {fmt(pooled(V2X, 'minade6', o_sel))}"
                  if n3 else "n/a")
        print(f"  {zname:10s} H2 set (moving R, n={n2}): {h2_txt} | H3 set (moving V, n={n3}): {h3_txt}")
    print("\nDESCRIPTIVE: minADE6 on moving R, ADE1 on moving V, per scene (seed means):")
    for label, ts, metric, sub in ((args.rsu_prefix, R, "minade6", "r_moving"), (args.rsu_prefix, R, "ade1", "v_moving"),
                                   (args.ref_prefix, V2X, "ade1", "moving")):
        vals = [scene_values(t, metric, masks_of(t)[sub], scenes) for t in ts]
        print(f"  {label:9s} {metric:7s} {sub:9s} ALL {st.mean(v['ALL'] for v in vals):.3f}  " + "  ".join(
            f"#{s}: " + (f"{st.mean(v[s] for v in vals):.3f}" if all(v[s] is not None for v in vals) else "n/a")
            for s in scenes))

    print("\nVERDICTS:")
    print(f"  H1 coverage                         {h1} ({gain:+.2f} pp)")
    print(f"  H2 RSU-only vs constant velocity    {h2}")
    print(f"  H3 E_V2X+RSU vs E_V2X (moving V)    {h3}")


if __name__ == "__main__":
    main()
