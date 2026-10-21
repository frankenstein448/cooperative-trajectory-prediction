"""check_scene_overlap.py (read-only): scene pairs sharing exact moving-vehicle positions (same SUMO log, overlapping in time), with split and log.
Run in gpu-test-venv. Found 2026-10-13: 35 cross-split pairs, scenes 49 (test) and 72 (train) identical.
--split-version v2 (scene_split.py) checks the leak-free split; --threshold sets the listed share (default 0.05);
the script ends with the number of cross-split pairs above the threshold."""
import os, re, json, sys, argparse, collections
HERE = os.path.dirname(os.path.abspath(__file__))   # this checkout: code and test_scenes.json
sys.path.insert(0, HERE)
import paths
from pilot_dataset import ALL_SCENES, SCENE_TO_TOWN
from scene_split import KNOWN_TUNING_SCENES, v2_scene_sets
ap = argparse.ArgumentParser()
ap.add_argument("--split-version", choices=["v1", "v2"], default="v1")
ap.add_argument("--threshold", type=float, default=0.05)
args = ap.parse_args()
R = paths.V2X_META_DIR
jl = lambda n: json.load(open(f"{R}/{n}"))
test = {t: set(v) for t, v in json.load(open(os.path.join(HERE, "test_scenes.json"))).items()}
sp = lambda s: "tuning" if s in KNOWN_TUNING_SCENES.get(SCENE_TO_TOWN[s], ()) else "test" if s in test.get(SCENE_TO_TOWN[s], ()) else "train"
if args.split_version == "v2":
    V2 = v2_scene_sets()
    sp = lambda s: "tuning" if s in V2[SCENE_TO_TOWN[s]][0] else "test" if s in V2[SCENE_TO_TOWN[s]][1] else "train"
print(f"split_version={args.split_version}, threshold={args.threshold:.0%}")
ann = jl("sample_annotation.json"); smp = {r["token"]: r for r in jl("sample.json")}
scs = {r["token"]: r for r in jl("scene.json")}; logs = {r["token"]: r["logfile"] for r in jl("log.json")}
scn = {t: int(re.search(r"(\d+)$", r["name"]).group(1)) for t, r in scs.items()}
log_of = {scn[t]: logs[r["log_token"]] for t, r in scs.items()}
bt = {a["token"]: a for a in ann}; used = set(ALL_SCENES)
mv = collections.defaultdict(set); inst = collections.defaultdict(set)
for a in ann:
    s = scn[smp[a["sample_token"]]["scene_token"]]
    if s in used and a["prev"]:
        p, q = a["translation"], bt[a["prev"]]["translation"]
        if (p[0], p[1]) != (q[0], q[1]) and abs(p[0]) < 1e6:
            mv[s].add((round(p[0], 3), round(p[1], 3)))
    if s in used:
        inst[s].add(a["instance_token"])
print(f"scene pairs sharing > {args.threshold:.0%} of exact MOVING positions (share of the smaller scene), with split and log:")
S = sorted(used); cross = 0
for i, a in enumerate(S):
    for b in S[i + 1:]:
        j = len(mv[a] & mv[b]) / max(1, min(len(mv[a]), len(mv[b])))
        if j > args.threshold:
            x = sp(a) != sp(b)
            cross += x
            print(f"  {a:3d} ({sp(a):6s}) - {b:3d} ({sp(b):6s}) {j:.3f}  logs {log_of[a]} / {log_of[b]}  shared instance tokens {len(inst[a] & inst[b])}"
                  + ("   <-- CROSS-SPLIT" if x else ""))
print(f"cross-split pairs above {args.threshold:.0%} (split {args.split_version}): {cross}")
print("scene descriptions / nbr_samples of 49 and 72:", [(scn[t], r["description"], r["nbr_samples"], r["first_sample_token"][:8]) for t, r in scs.items() if scn[t] in (49, 72)])
