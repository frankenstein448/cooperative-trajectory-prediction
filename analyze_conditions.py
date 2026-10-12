"""Applies runs/DECISION_RULES_CONDITIONS.md to the condition rows (eval_scenes.py's COND block) of
the D / Dnomap / EgoOnly epoch-12 tuning re-evaluations (run_condition_evals.sh -> eval_<run>_<suffix>.log).

For each condition in scene_conditions.CONDITIONS (turning, straight, stationary, near_junction):
  map helps  only if seed-mean minADE6 of D < Dnomap's AND (Dnomap - D) > 2 x max(seed std D, seed std Dnomap)
  V2X helps  only if seed-mean minADE6 of D < EgoOnly's AND (EgoOnly - D) > 2 x max(seed std D, seed std EgoOnly)
on the pooled (COND ALL) rows. ADE1, minFDE6 and per-town rows are reported, not part of the rule."""
import re
import argparse
import statistics as st

from scene_conditions import CONDITIONS

parser = argparse.ArgumentParser()
parser.add_argument("--suffix", default="e12_cond", help="log suffix: eval_<run>_<suffix>.log")
parser.add_argument("--expect-n", type=int, default=2208, help="ALL sample count every log must show")
args = parser.parse_args()

GROUPS = {"D": ["D_seed0", "D_seed1", "D_seed2"],
          "Dnomap": ["Dnomap_seed0", "Dnomap_seed1", "Dnomap_seed2"],
          "EgoOnly": ["EgoOnly_seed0", "EgoOnly_seed1", "EgoOnly_seed2"]}
COMPARISONS = (("map", "D", "Dnomap"), ("V2X", "D", "EgoOnly"))  # (effect, with, without)
COND = re.compile(r"^COND (\S+)\s+(\S+)\s+(\d+)(?:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+))?\s*$")
ALL = re.compile(r"^ALL\s+(\d+)\s+[\d.]+m", re.M)
STAT_ALL = re.compile(r"^    stationary \(ALL\): n=\s*(\d+)", re.M)
METRICS = ("minADE6", "ADE1", "minFDE6")


def parse(name):
    path = f"eval_{name}_{args.suffix}.log"
    with open(path) as f:
        text = f.read()
    rows = {}  # (town, condition) -> (n, minADE6, ADE1, minFDE6) or (0, None, None, None)
    for line in text.splitlines():
        m = COND.match(line)
        if m and m.group(1) != "town":
            vals = tuple(float(m.group(k)) for k in (4, 5, 6)) if m.group(4) else (None,) * 3
            rows[(m.group(1), m.group(2))] = (int(m.group(3)),) + vals
    assert rows, f"{path}: no COND rows"
    n_all = int(ALL.search(text).group(1))
    assert n_all == args.expect_n, f"{path}: ALL n = {n_all} != {args.expect_n}"
    n = lambda cond, town="ALL": rows[(town, cond)][0]
    for town in sorted({t for t, _ in rows}):
        assert n("stationary", town) + n("moving", town) == (n_all if town == "ALL" else
               n("near_junction", town) + n("not_near_junction", town)), f"{path} {town}: stationary + moving"
        assert n("turning", town) + n("straight", town) == n("moving", town), f"{path} {town}: turning + straight"
    assert n("near_junction") + n("not_near_junction") == n_all, f"{path}: near + not-near != ALL"
    assert n("stationary") == int(STAT_ALL.search(text).group(1)), f"{path}: COND stationary n != stationary (ALL)"
    return rows


runs = {name: parse(name) for names in GROUPS.values() for name in names}
first = next(iter(runs.values()))
for name, rows in runs.items():
    assert {k: v[0] for k, v in rows.items()} == {k: v[0] for k, v in first.items()}, \
        f"{name}: condition sample counts differ from {next(iter(runs))} (labels must not depend on the model)"
towns = sorted({t for t, _ in first if t != "ALL"})
conds_all = [c for t, c in first if t == "ALL"]


def seed_stats(group, town, cond, k):
    vals = [runs[n][(town, cond)][k] for n in GROUPS[group]]
    if any(v is None for v in vals):
        return None, None, vals
    return st.mean(vals), st.stdev(vals), vals


print(f"Condition sample counts (identical in all {len(runs)} logs):")
for town in ["ALL"] + towns:
    print(f"  {town:7s} " + "  ".join(f"{c}={first[(town, c)][0]}" for c in conds_all))

print("\nPer run, pooled over the tuning split, minADE6 / ADE1 / minFDE6:")
for cond in conds_all:
    print(f"  {cond} (n={first[('ALL', cond)][0]})")
    for group, names in GROUPS.items():
        for name in names:
            r = runs[name][("ALL", cond)]
            print(f"    {name:14s} " + ("  ".join(f"{x:7.3f}" for x in r[1:]) if r[1] is not None else "n/a"))

print("\nSeed mean +/- std (pooled tuning split; descriptive except minADE6 of the rule's conditions):")
print(f"  {'condition':18s} {'group':8s} " + " ".join(f"{m:>17s}" for m in METRICS))
for cond in conds_all:
    for group in GROUPS:
        cells = []
        for k in (1, 2, 3):
            m, s, _ = seed_stats(group, "ALL", cond, k)
            cells.append(f"{m:8.3f} +/- {s:5.3f}" if m is not None else f"{'n/a':>17s}")
        print(f"  {cond:18s} {group:8s} " + " ".join(cells))

print("\nPer town, seed-mean minADE6 (descriptive only):")
print(f"  {'town':7s} {'condition':18s} {'n':>5s} " + " ".join(f"{g:>8s}" for g in GROUPS))
for town in towns:
    for cond in conds_all:
        means = [seed_stats(g, town, cond, 1)[0] for g in GROUPS]
        print(f"  {town:7s} {cond:18s} {first[(town, cond)][0]:5d} "
              + " ".join(f"{m:8.3f}" if m is not None else f"{'n/a':>8s}" for m in means))

print("\nRULE (runs/DECISION_RULES_CONDITIONS.md): helps on a condition only if seed-mean minADE6 with < without "
      "AND (without - with) > 2 x max(seed std with, seed std without); pooled tuning split")
verdicts = []
for effect, with_g, without_g in COMPARISONS:
    print(f"\n{effect}: {with_g} (with) vs {without_g} (without)")
    for cond in CONDITIONS:
        mw, sw, _ = seed_stats(with_g, "ALL", cond, 1)
        mo, so, _ = seed_stats(without_g, "ALL", cond, 1)
        if mw is None or mo is None:
            print(f"  {cond:14s} n=0 -> not evaluable")
            verdicts.append((effect, cond, "not evaluable (n=0)"))
            continue
        gap, thr = mo - mw, 2 * max(sw, so)
        c1, c2 = mw < mo, gap > thr
        v = f"{effect} HELPS" if (c1 and c2) else "no detectable effect"
        print(f"  {cond:14s} n={first[('ALL', cond)][0]:5d}  {with_g} {mw:.3f} +/- {sw:.3f}  "
              f"{without_g} {mo:.3f} +/- {so:.3f}  lower? {c1}  gap {gap:+.3f} > {thr:.3f}? {c2}  -> {v}")
        verdicts.append((effect, cond, v))

print("\nVERDICTS (4 conditions x 2 comparisons, no multiple-comparison correction):")
for effect, cond, v in verdicts:
    print(f"  {effect:4s} {cond:14s} {v}")
