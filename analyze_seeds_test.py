"""Applies the pre-registered map rule to the six seed runs (tuning split, epoch 12).
Map 'helps' only if: seed-averaged full minADE6 is lower in >= 7 of 8 scenes AND
the ALL minADE6 gap > 2 x the larger of the two seed standard deviations."""
import re
import statistics as st

FULL = ["D_seed0", "D_seed1", "D_seed2"]
NOMAP = ["Dnomap_seed0", "Dnomap_seed1", "Dnomap_seed2"]
SCENE = re.compile(r"^(Town\d\d)\s+(\d+)\s+(\d+)\s+([\d.]+)m\s+([\d.]+)m\s+([\d.]+)m")
ALL = re.compile(r"^ALL\s+(\d+)\s+([\d.]+)m\s+([\d.]+)m\s+([\d.]+)m")


def parse(name):
    path = f"eval_{name}_test.log"
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
    assert int(re.search(r"^ALL\s+(\d+)", text, re.M).group(1)) == 2208, f"{path}: ALL n != 2208"
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

mf, sf, _ = summary(FULL, 0)
mn, sn, _ = summary(NOMAP, 0)
gap, thr = mn - mf, 2 * max(sf, sn)
c1, c2 = wins[0] >= 7, gap > thr
print(f"\nRULE (minADE6): scenes {wins[0]}/8 >= 7? {c1}   gap {gap:+.3f} > 2 x max std {thr:.3f}? {c2}")
print("VERDICT:", "map effect detected" if (c1 and c2) else "no detectable map effect under the scene-held-out protocol")
