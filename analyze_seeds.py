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
count every log must show (2208 for D's sample filter, 1984 with 8 frames of history).

--rule component (runs/DECISION_RULES_V2X.md) is the map rule's computation, with
--full = the full model's runs and --nomap = an ablation's runs (labels reused):
the removed component helps only if the full model's seed-mean minADE6 is lower in
>= 7 of 8 tuning scenes AND the ALL gap (ablation - full) > 2 x the larger seed std.

--rule loss (runs/DECISION_RULES_LOSS.md): --full = laplace_mixture runs (L), --nomap =
D runs. Adopt laplace_mixture only if seed-mean ALL ADE1 is lower than D's AND seed-mean
ADE1 is lower in >= 6 of 8 tuning scenes AND seed-mean ALL minADE6 is at most 0.1 m
above D's AND seed-mean mode diversity (eval_scenes.py's RANK ALL row) is >= 50% of D's.
Per-run and seed mean +/- std of ADE1, minADE6, minFDE6, Brier-minFDE, diversity and
(L only) calibration are printed; ADE1 seed std is reported, not a criterion."""
import os
import re
import argparse
import statistics as st

parser = argparse.ArgumentParser()
parser.add_argument("--suffix", default="e12")
parser.add_argument("--full", default="D_seed0,D_seed1,D_seed2", help="comma-separated full-model run names")
parser.add_argument("--nomap", default="Dnomap_seed0,Dnomap_seed1,Dnomap_seed2",
                    help="comma-separated no-map run names")
parser.add_argument("--rule", choices=["map", "history", "component", "loss"], default="map",
                    help="map: the map rule (default). history: runs/DECISION_RULES_HISTORY.md, "
                         "--full = history runs, --nomap = control runs. component: runs/DECISION_RULES_V2X.md, "
                         "the map rule with --full = full-model runs, --nomap = ablation runs. loss: "
                         "runs/DECISION_RULES_LOSS.md, --full = laplace_mixture runs, --nomap = D runs. "
                         "A run name may carry its own log suffix, name:suffix")
parser.add_argument("--full-suffix", default=None,
                    help="log suffix for the --full runs only (default: --suffix), e.g. e12_safety to read the "
                         "full model's safety metrics from run_safety_reeval.sh's logs (same accuracy rows as e12)")
parser.add_argument("--expect-n", type=int, default=2208, help="ALL sample count every log must show")
args = parser.parse_args()
CLEAN_SUFFIX = "e12"

SUFFIX_OF = {}  # run -> log suffix, from a "name:suffix" entry in --full / --nomap


def run_names(spec):
    names = []
    for entry in spec.split(","):
        name, _, suffix = entry.partition(":")
        if suffix:
            SUFFIX_OF[name] = suffix
        names.append(name)
    return names


FULL = run_names(args.full)
NOMAP = run_names(args.nomap)
SCENE = re.compile(r"^(Town\d\d)\s+(\d+)\s+(\d+)\s+([\d.]+)m\s+([\d.]+)m\s+([\d.]+)m")
ALL = re.compile(r"^ALL\s+(\d+)\s+([\d.]+)m\s+([\d.]+)m\s+([\d.]+)m")
SAFE_ALL = re.compile(r"^SAFE (ALL|  stationary|  moving)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
RANK_ALL = re.compile(r"^RANK ALL\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)((?:\s+[\d.]+)*)\s*$")
SAFETY = {}   # run -> {group: (miss1, miss6, offr1, offr6, offrGT)}, filled by parse() for the main suffix
RANKING = {}  # run -> (minFDE6, brierFDE, div, (pt50, xy50, pt90, xy90) or ()), from eval_scenes.py's RANK ALL row


def parse(name, suffix=args.suffix, record_safety=False):
    suffix = SUFFIX_OF.get(name, suffix)  # a per-run "name:suffix" overrides --suffix / --full-suffix
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
    if record_safety:
        # the ALL block is the last one with these labels in eval_scenes.py's SAFE output
        safe = {}
        for line in text.splitlines():
            m = SAFE_ALL.match(line)
            if m:
                safe[m.group(1).strip()] = tuple(float(m.group(k)) for k in range(2, 7))
        if safe:
            SAFETY[name] = safe
        for line in text.splitlines():
            m = RANK_ALL.match(line)
            if m:
                RANKING[name] = (float(m.group(1)), float(m.group(2)), float(m.group(3)),
                                 tuple(float(x) for x in m.group(4).split()))
    return scenes, allrow


runs = {n: parse(n, args.full_suffix or args.suffix, record_safety=True) for n in FULL}
runs.update({n: parse(n, record_safety=True) for n in NOMAP})
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

if args.rule in ("map", "component"):
    mf, sf, _ = summary(FULL, 0)
    mn, sn, _ = summary(NOMAP, 0)
    gap, thr = mn - mf, 2 * max(sf, sn)
    c1, c2 = wins[0] >= 7, gap > thr
    print(f"\nRULE (minADE6): scenes {wins[0]}/8 >= 7? {c1}   gap {gap:+.3f} > 2 x max std {thr:.3f}? {c2}")
    if args.rule == "map":
        print("VERDICT:", "map effect detected" if (c1 and c2) else "no detectable map effect under the scene-held-out protocol")
    else:
        print(f"(component rule: full = {','.join(FULL)}, ablation = {','.join(NOMAP)})")
        print("VERDICT:", "removed component HELPS (full model better)" if (c1 and c2)
              else "no detectable effect of the removed component under the scene-held-out protocol")
elif args.rule == "history":
    # "full" = history runs, "no-map" = control runs (labels reused)
    h_ade1, c_ade1 = summary(FULL, 1)[0], summary(NOMAP, 1)[0]
    h_m6, c_m6 = summary(FULL, 0)[0], summary(NOMAP, 0)[0]
    c1, c2, c3 = h_ade1 < c_ade1, h_m6 < c_m6, wins[1] >= 6
    print(f"\nHISTORY RULE (full = history, no-map = control):")
    print(f"  ALL ADE1    history {h_ade1:.3f} < control {c_ade1:.3f}? {c1}")
    print(f"  ALL minADE6 history {h_m6:.3f} < control {c_m6:.3f}? {c2}")
    print(f"  ADE1 lower in {wins[1]}/8 scenes >= 6? {c3}")
    print("VERDICT:", "ADOPT history" if (c1 and c2 and c3) else "do NOT adopt history")
elif args.rule == "loss":
    # "full" = laplace_mixture runs (L), "no-map" = baseline runs (D) (labels reused)
    missing = [n for n in FULL + NOMAP if n not in RANKING]
    assert not missing, f"no RANK ALL row (ranking metrics) in the logs of {', '.join(missing)}"
    print("\nRanking metrics, ALL, per run (minFDE6, brierFDE, div"
          + (", calibration pt50 xy50 pt90 xy90 of the top mode" if RANKING[FULL[0]][3] else "") + "):")
    for n in FULL + NOMAP:
        r = RANKING[n]
        print(f"  {n:14s} {r[0]:.3f}  {r[1]:.3f}  {r[2]:.3f}" + "".join(f"  {x:.4f}" for x in r[3]))
    rank = lambda names, k: [RANKING[n][k] for n in names]
    for label, names in (("L", FULL), ("D", NOMAP)):
        cols = [("ADE1", [runs[n][1][1] for n in names]), ("minADE6", [runs[n][1][0] for n in names]),
                ("minFDE6", rank(names, 0)), ("brierFDE", rank(names, 1)), ("div", rank(names, 2))]
        if RANKING[names[0]][3]:
            cols += [(f"cal {c}", [RANKING[n][3][j] for n in names])
                     for j, c in enumerate(("pt50", "xy50", "pt90", "xy90"))]
        print(f"  {label} seed mean +/- std: " + "   ".join(f"{c} {st.mean(v):.3f} +/- {st.stdev(v):.3f}" for c, v in cols))
    l_ade1, l_ade1_sd, _ = summary(FULL, 1)
    d_ade1, d_ade1_sd, _ = summary(NOMAP, 1)
    l_m6, d_m6 = summary(FULL, 0)[0], summary(NOMAP, 0)[0]
    l_div, d_div = st.mean(rank(FULL, 2)), st.mean(rank(NOMAP, 2))
    c1, c2 = l_ade1 < d_ade1, wins[1] >= 6
    c3, c4 = l_m6 - d_m6 <= 0.1, l_div >= 0.5 * d_div
    print(f"\nLOSS RULE (full = laplace_mixture L, no-map = D):")
    print(f"  ALL ADE1    L {l_ade1:.3f} < D {d_ade1:.3f}? {c1}")
    print(f"  ADE1 lower in {wins[1]}/8 scenes >= 6? {c2}")
    print(f"  ALL minADE6 L {l_m6:.3f} - D {d_m6:.3f} = {l_m6 - d_m6:+.3f} <= +0.100? {c3}")
    print(f"  ALL div     L {l_div:.3f} >= 0.5 x D {d_div:.3f} = {0.5 * d_div:.3f}? {c4}")
    print(f"  (reported, not a criterion) ADE1 seed std: L {l_ade1_sd:.3f}  D {d_ade1_sd:.3f}")
    print("VERDICT:", "ADOPT laplace_mixture" if (c1 and c2 and c3 and c4) else "do NOT adopt laplace_mixture")

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

if all(n in SAFETY for n in FULL + NOMAP):
    print("\nSafety, ALL rows, seed mean (descriptive only, not part of any rule; "
          "see each log's SAFE block for per-scene/per-town rows and GT-control UNRELIABLE flags):")
    print(f"  {'':9s} {'group':11s} {'miss1':>7s} {'miss6':>7s} {'offr1':>7s} {'offr6':>7s} {'offrGT':>7s}")
    for label, names in (("full", FULL), ("no-map", NOMAP)):
        for group in ("ALL", "stationary", "moving"):
            vals = [SAFETY[n][group] for n in names]
            print(f"  {label:9s} {group:11s} " + " ".join(f"{st.mean(v[k] for v in vals):7.4f}" for k in range(5)))
else:
    missing = [n for n in FULL + NOMAP if n not in SAFETY]
    print(f"\nSafety summary skipped: no SAFE rows in the logs of {', '.join(missing)}")
