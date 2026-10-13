"""check_scene_overlap.py (read-only): scene pairs sharing exact moving-vehicle positions (same SUMO log, overlapping in time), with split and log.
Run in gpu-test-venv. Found 2026-10-13: 35 cross-split pairs, scenes 49 (test) and 72 (train) identical."""
import re, json, sys, collections
sys.path.insert(0, "/home/karra/hd-maps-dev")
from pilot_dataset import ALL_SCENES, SCENE_TO_TOWN
from scene_split import KNOWN_TUNING_SCENES
R = "/home/karra/V2X-Sim-2.0/v2.0"
jl = lambda n: json.load(open(f"{R}/{n}"))
test = {t: set(v) for t, v in json.load(open("/home/karra/hd-maps-dev/test_scenes.json")).items()}
sp = lambda s: "tuning" if s in KNOWN_TUNING_SCENES.get(SCENE_TO_TOWN[s], ()) else "test" if s in test.get(SCENE_TO_TOWN[s], ()) else "train"
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
print("scene pairs sharing > 5% of exact MOVING positions (share of the smaller scene), with split and log:")
S = sorted(used); cross = 0
for i, a in enumerate(S):
    for b in S[i + 1:]:
        j = len(mv[a] & mv[b]) / max(1, min(len(mv[a]), len(mv[b])))
        if j > 0.05:
            x = sp(a) != sp(b)
            cross += x
            print(f"  {a:3d} ({sp(a):6s}) - {b:3d} ({sp(b):6s}) {j:.3f}  logs {log_of[a]} / {log_of[b]}  shared instance tokens {len(inst[a] & inst[b])}"
                  + ("   <-- CROSS-SPLIT" if x else ""))
print(f"cross-split pairs: {cross}")
print("scene descriptions / nbr_samples of 49 and 72:", [(scn[t], r["description"], r["nbr_samples"], r["first_sample_token"][:8]) for t, r in scs.items() if scn[t] in (49, 72)])
