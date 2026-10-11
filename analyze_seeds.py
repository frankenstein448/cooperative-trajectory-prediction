"""Applies the pre-registered map rule to the six seed runs (tuning split, epoch 12).
Map 'helps' only if: seed-averaged full minADE6 is lower in >= 7 of 8 scenes AND
the ALL minADE6 gap > 2 x the larger of the two seed standard deviations.

--suffix picks the logs, eval_<run>_<suffix>.log (default e12 = clean tuning,
epoch 12; e.g. tuning_fog for eval_scenes.py --degrade fog). For any suffix
other than e12, also prints each model's ALL increase over its clean e12 log
(skipped if a run has none). --full / --nomap pick the runs (default D / Dnomap).

--rule history (runs/DECISION_RULES_HISTORY.md) reuses the full / no-map labels
for history (H8) / control (Dh), and applies that rule instead: adopt history only
if seed-mean ALL ADE1 AND ALL minADE6 are both lower than the control's AND
seed-mean ADE1 is lower in >= 6 of 8 tuning scenes. --expect-n is the ALL sample
count every log must show (2208 for D's sample filter, 1984 with 8 frames of history)."""
import os
import re
import argparse
import statistics as st

parser = argparse.ArgumentParser()
parser.add_argument("--suffix", default="e12")
parser.add_argument("--full", default="D_seed0,D_seed1,D_seed2", help="comma-separated full-model run names")
parser.add_argument("--nomap", default="Dnomap_seed0,Dnomap_seed1,Dnomap_seed2",
                    help="comma-separated no-map run names")
parser.add_argument("--rule", choices=["map", "history"], default="map",
                    help="map: the map rule (default). history: runs/DECISION_RULES_HISTORY.md, "
                         "--full = history runs, --nomap = control runs")
parser.add_argument("--expect-n", type=int, default=2208, help="ALL sample count every log must show")
args = parser.parse_args()
CLEAN_SUFFIX = "e12"

FULL = args.full.split(",")
NOMAP = args.nomap.split(",")
SCENE = re.compile(r"^(Town\d\d)\s+(\d+)\s+(\d+)\s+([\d.]+)m\s+([\d.]+)m\s+([\d.]+)m")
ALL = re.compile(r"^ALL\s+(\d+)\s+([\d.]+)m\s+([\d.]+)m\s+([\d.]+)m")


def parse(name, suffix=args.suffix):
    path = f"eval_{name}_{suffix}.log"
    scenes, allrow = {}, None
    with open(path) as f:
        text = f.read()
    for line in text.splitlines():
        m = SCENE.match(line)
        if m:
            scenes[(m.group(1), int(m.group(2)))] = (float(m.group(4)), float(m.group(6)))  # minADE6, ADE1
        m = ALL.match(line)
        if m:
            allrow = (float(m.group(2)), float(m.group(4)))
    assert len(scenes) == 8, f"{path}: expected 8 scenes, found {len(scenes)}"
    assert allrow is not None, f"{path}: no ALL row"
    assert int(re.search(r"^ALL\s+(\d+)", text, re.M).group(1)) == args.expect_n, f"{path}: ALL n != {args.expect_n}"
    return scenes, allrow


runs = {n: parse(n) for n in FULL + NOMAP}
assert all(set(runs[n][0]) == set(runs[FULL[0]][0]) for n in runs), "scene sets differ between runs"


def summary(names, idx):
    vals = [runs[n][1][idx] for n in names]
    return st.mean(vals), st.stdev(vals), vals


print("Per run, ALL (minADE6, ADE1):")
for n in FULL + NOMAP:
    print(f"  {n:14s} {runs[n][1][0]:.3f}  {runs[n][1][1]:.3f}")

for idx, label in ((0, "minADE6"), (1, "ADE1")):
    mf, sf, _ = summary(FULL, idx)
    mn, sn, _ = summary(NOMAP, idx)
    print(f"\nALL {label}: full {mf:.3f} +/- {sf:.3f}   no-map {mn:.3f} +/- {sn:.3f}   gap (no-map - full) {mn - mf:+.3f}")

print("\nPer scene, seed-averaged (positive diff = map helps):")
print(f"{'scene':12s} {'full m6':>8s} {'nomap m6':>9s} {'diff':>7s} | {'full a1':>8s} {'nomap a1':>9s} {'diff':>7s}")
wins = {0: 0, 1: 0}
for key in sorted(runs[FULL[0]][0]):
    row = []
    for idx in (0, 1):
        f = st.mean(runs[n][0][key][idx] for n in FULL)
        m = st.mean(runs[n][0][key][idx] for n in NOMAP)
        wins[idx] += f < m
        row.append((f, m, m - f))
    print(f"{key[0]} #{key[1]:<4d} {row[0][0]:8.3f} {row[0][1]:9.3f} {row[0][2]:+7.3f} | "
          f"{row[1][0]:8.3f} {row[1][1]:9.3f} {row[1][2]:+7.3f}")
print(f"Full better in {wins[0]}/8 scenes (minADE6), {wins[1]}/8 scenes (ADE1)")

if args.rule == "map":
    mf, sf, _ = summary(FULL, 0)
    mn, sn, _ = summary(NOMAP, 0)
    gap, thr = mn - mf, 2 * max(sf, sn)
    c1, c2 = wins[0] >= 7, gap > thr
    print(f"\nRULE (minADE6): scenes {wins[0]}/8 >= 7? {c1}   gap {gap:+.3f} > 2 x max std {thr:.3f}? {c2}")
    print("VERDICT:", "map effect detected" if (c1 and c2) else "no detectable map effect under the scene-held-out protocol")
else:
    # "full" = history runs, "no-map" = control runs (labels reused)
    h_ade1, c_ade1 = summary(FULL, 1)[0], summary(NOMAP, 1)[0]
    h_m6, c_m6 = summary(FULL, 0)[0], summary(NOMAP, 0)[0]
    c1, c2, c3 = h_ade1 < c_ade1, h_m6 < c_m6, wins[1] >= 6
    print(f"\nHISTORY RULE (full = history, no-map = control):")
    print(f"  ALL ADE1    history {h_ade1:.3f} < control {c_ade1:.3f}? {c1}")
    print(f"  ALL minADE6 history {h_m6:.3f} < control {c_m6:.3f}? {c2}")
    print(f"  ADE1 lower in {wins[1]}/8 scenes >= 6? {c3}")
    print("VERDICT:", "ADOPT history" if (c1 and c2 and c3) else "do NOT adopt history")

missing_clean = [n for n in FULL + NOMAP if not os.path.exists(f"eval_{n}_{CLEAN_SUFFIX}.log")]
if args.suffix != CLEAN_SUFFIX and missing_clean:
    # e.g. fog-trained runs, which have no clean eval
    print(f"\nIncrease vs clean ({CLEAN_SUFFIX}) skipped: no clean log for {', '.join(missing_clean)}")
elif args.suffix != CLEAN_SUFFIX:
    clean = {n: parse(n, CLEAN_SUFFIX) for n in FULL + NOMAP}
    print(f"\nIncrease in ALL vs clean ({CLEAN_SUFFIX}), {args.suffix} - clean, seed mean +/- std "
          f"(per seed in brackets):")
    for label, names in (("full", FULL), ("no-map", NOMAP)):
        for idx, metric in ((0, "minADE6"), (1, "ADE1")):
            inc = [runs[n][1][idx] - clean[n][1][idx] for n in names]
            print(f"  {label:7s} {metric:8s} clean {st.mean(clean[n][1][idx] for n in names):.3f} -> "
                  f"{st.mean(runs[n][1][idx] for n in names):.3f}   increase {st.mean(inc):+.3f} +/- {st.stdev(inc):.3f}  "
                  f"[{', '.join(f'{x:+.3f}' for x in inc)}]")
