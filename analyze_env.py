"""
analyze_env.py
Applies runs/DECISION_RULES_ENV.md to the env runs' per-target evaluations (eval_env.py ->
runs/<run>/eval_env_tuning_checkpoint_epoch12.npz). All rules use MOVING targets and the tuning split (v2),
and Part 1's rule form (env_metrics.rule): full form = seed-mean lower in >= 7 of the 8 tuning scenes AND ALL gap
> 2 x the larger seed std; pooled form (if any tuning scene has no target of the subset) = seed-mean ALL lower AND
the same gap condition. Which form applies is decided by the ground-truth labels alone (identical for every run).

  (1) V2X on (a):   E_V2X vs E_EgoOnly, minADE6 on moving class-(a) targets.
  (2) V2X on (b):   E_V2X ADE1 (its top-scored mode) vs the constant-velocity baseline's ADE on moving class-(b)
                    targets (the baseline is deterministic: its seed std is 0). E_V2X minADE6 vs the same baseline
                    is printed as a descriptive comparison only (best of 6 vs a single prediction).
  (3) Map:          E_V2X vs E_V2Xnomap, minADE6 on moving targets ((a) + (b)); secondary: on turning targets.
Everything else (other metrics, minFDE6, miss rates, stationary / straight / category rows, baselines, coverage) is
descriptive.

Run in gpu-test-venv:  python analyze_env.py [--prefix E_] [--npz eval_env_tuning_checkpoint_epoch12.npz]
"""
import json
import argparse
import statistics as st

import numpy as np

from env_metrics import subset_masks, scene_values, rule, METRICS, SUBSETS

GROUPS = ("EgoOnly", "V2X", "V2Xnomap")
EXPECT = {"EgoOnly": ("off", False), "V2X": ("on", False), "V2Xnomap": ("on", True)}  # (v2x, no_map)
RULES = (  # (label, with group, with metric, without group, without metric, subset)
    ("(1) V2X on (a)", "V2X", "minade6", "EgoOnly", "minade6", "a_moving"),
    ("(2) V2X on (b) vs constant velocity", "V2X", "ade1", "V2X", "cv_ade", "b_moving"),
    ("(3) map, moving targets", "V2X", "minade6", "V2Xnomap", "minade6", "moving"),
    ("(3b) map, turning targets (secondary)", "V2X", "minade6", "V2Xnomap", "minade6", "turning"),
)
DESCRIPTIVE_COMPARISONS = (  # same computation, no verdict
    ("(2-desc) V2X minADE6 on (b) vs constant velocity", "V2X", "minade6", "V2X", "cv_ade", "b_moving"),
)
METRIC_NAME = {"minade6": "minADE6", "ade1": "ADE1", "cv_ade": "constant velocity ADE"}


def load(path):
    z = np.load(path, allow_pickle=False)
    t = {k: z[k] for k in z.files if k != "meta"}
    return t, json.loads(str(z["meta"]))


def check_runs(runs, metas):
    """Same split / scenes / transform everywhere; each config's runs have identical target lists;
    EgoOnly's targets are exactly V2X's class-(a) targets."""
    m0 = next(iter(metas.values()))
    for name, m in metas.items():
        assert m["split"] == "tuning" and m["split_version"] == "v2" and m["map_transform"] == "mirror", name
        assert m["scenes"] == m0["scenes"], f"{name}: tuning scenes differ"
    by_group = {}
    for name, t in runs.items():
        by_group.setdefault(name.rsplit("_seed", 1)[0], []).append(t)
    for g, ts in by_group.items():
        order = np.argsort(ts[0]["tid"])
        for t in ts[1:]:
            o = np.argsort(t["tid"])
            assert np.array_equal(t["tid"][o], ts[0]["tid"][order]), f"{g}: target lists differ between seeds"
            for k in ("vis", "moving", "turning", "scene"):
                assert np.array_equal(t[k][o], ts[0][k][order]), f"{g}: {k} differs between seeds"
    return by_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="E_")
    ap.add_argument("--npz", default="eval_env_tuning_checkpoint_epoch12.npz")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    runs, metas = {}, {}
    for g in GROUPS:
        for s in seeds:
            name = f"{args.prefix}{g}_seed{s}"
            runs[name], metas[name] = load(f"runs/{name}/{args.npz}")
            assert (metas[name]["v2x"], metas[name]["no_map"]) == EXPECT[g], f"{name}: wrong config {metas[name]}"
    by_group = check_runs(runs, metas)
    G = {g: by_group[f"{args.prefix}{g}"] for g in GROUPS}
    ego_tid, v2x_tid = np.sort(G["EgoOnly"][0]["tid"]), G["V2X"][0]["tid"]
    assert np.array_equal(ego_tid, np.sort(v2x_tid[G["V2X"][0]["vis"] == 0])), "EgoOnly targets != V2X's (a) targets"
    scenes = [s for _, s in next(iter(metas.values()))["scenes"]]
    print(f"runs: {', '.join(runs)}; tuning scenes {scenes}")

    ref = G["V2X"][0]
    masks = subset_masks(ref)
    print("\nTarget counts (V2X-env's known targets = (a) + (b); EgoOnly-env predicts (a) only):")
    for sub in SUBSETS:
        per = [int((masks[sub] & (ref["scene"] == s)).sum()) for s in scenes]
        print(f"  {sub:19s} ALL {int(masks[sub].sum()):6d}  per scene {per}")

    print("\nSeed mean +/- std, ALL, per subset (descriptive):")
    for g in GROUPS:
        for sub in ("all", "moving", "stationary", "turning", "straight", "a_moving", "b_moving", "car_moving",
                    "two_wheeler_moving"):
            vals = []
            for t in G[g]:
                m = subset_masks(t)[sub]
                vals.append([float(t[k][m].mean()) if m.any() else float("nan") for k in METRICS])
            arr = np.array(vals)                                   # [seeds, metrics]
            cells = [f"{arr[:, i].mean():7.3f}+/-{arr[:, i].std(ddof=1) if len(arr) > 1 else 0.0:5.3f}"
                     if np.isfinite(arr[:, i]).all() else f"{'n/a':>15s}" for i in range(len(METRICS))]
            print(f"  {g:9s} {sub:19s} " + " ".join(f"{m}={c}" for m, c in zip(METRICS, cells)))
    for sub in ("moving", "a_moving", "b_moving", "turning"):
        m = masks[sub]
        print(f"  baselines {sub:19s} cv ade={ref['cv_ade'][m].mean():.3f} fde={ref['cv_fde'][m].mean():.3f} "
              f"miss={ref['cv_miss'][m].mean():.3f} | stay ade={ref['stay_ade'][m].mean():.3f}")

    def compare(label, gw, mw, go, mo, sub):
        w = [scene_values(t, mw, subset_masks(t)[sub], scenes) for t in G[gw]]
        o = [scene_values(t, mo, subset_masks(t)[sub], scenes) for t in G[go]]
        r = rule(w, o, scenes)
        a, b = f"{gw} {METRIC_NAME[mw]}", f"{go} {METRIC_NAME[mo]}"
        print(f"\n{label}: {a} vs {b}, subset {sub}")
        print(f"  per scene seed means ({a} / {b}): " + "  ".join(
            f"#{s}: " + (f"{st.mean(x[s] for x in w):.3f}/{st.mean(x[s] for x in o):.3f}"
                         if all(x[s] is not None for x in w + o) else "n/a") for s in scenes))
        print(f"  form={r['form']} (scenes with samples {r['scenes_with_samples']}/8)  wins={r['wins']}  "
              f"ALL {r['with']:.3f} (sd {r['sd_with']:.3f}) vs {r['without']:.3f} (sd {r['sd_without']:.3f})  "
              f"gap {r['gap']:+.3f} > {r['threshold']:.3f}?")
        return r

    print("\nRULES (runs/DECISION_RULES_ENV.md; moving targets; lower is better)")
    verdicts = []
    for spec in RULES:
        r = compare(*spec)
        verdicts.append((spec[0], "HELPS / better" if r["helps"] else "no detectable effect"))
        print(f"  -> {verdicts[-1][1]}")
    print("\nDESCRIPTIVE comparisons (same computation, NO verdict):")
    for spec in DESCRIPTIVE_COMPARISONS:
        r = compare(*spec)
        print(f"  -> descriptive only (the rule form would say: {'lower' if r['helps'] else 'not lower'})")
    print("\nVERDICTS:")
    for label, v in verdicts:
        print(f"  {label:42s} {v}")
    print("\nCoverage per model (all D samples of the tuning scenes; predicted / kept a+b+c; incl. removed):")
    for g in GROUPS:
        covs = {json.dumps(metas[f"{args.prefix}{g}_seed{s}"]["coverage"], sort_keys=True) for s in seeds}
        assert len(covs) == 1, f"{g}: coverage differs between seeds"
        c = json.loads(covs.pop())
        print(f"  {g:9s} predicted {c['predicted']:6d} of {c['all_kept']:6d} (a {c['a']}, b {c['b']}, c {c['c']}): "
              f"coverage {c['coverage']:.4f}; with the {c['removed']} window-filtered vehicles in the denominator "
              f"{c['coverage_incl_removed']:.4f}")


if __name__ == "__main__":
    main()
