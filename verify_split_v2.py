"""
verify_split_v2.py
Checks for the versioned scene split (scene_split.SPLIT_VERSIONS = v1 / v2), before any v2 run:

1. v1 is unchanged: three_way_scene_split(dataset) (default) and split_version="v1" give exactly the index
   lists of the function at the last commit before the option, on D's sample list.
2. v2: the scene sets equal scene_split_proposal.json (approved), the three splits are disjoint and cover all
   54 scenes, sample counts per split and town; 8 tuning scenes with 2,208 samples (analyze_seeds.py's
   7-of-8 rule and --expect-n 2208 apply unchanged).
3. Configs: runs/D_seed0 (no key) -> v1; train.py --split-version default v2, written to config.json and
   passed to three_way_scene_split; eval_scenes.py reads it from the run's config.
Cross-split overlap of v2 is checked separately: check_scene_overlap.py --split-version v2 --threshold 0.01.

Run in gpu-test-venv:  python verify_split_v2.py
"""
import re
import json
import subprocess
import collections

import scene_split as ss
from pilot_dataset import PilotDataset

fails = []


def check(ok, msg):
    print(("  OK   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


revs = subprocess.run(["git", "log", "--format=%H", "--", "scene_split.py"], capture_output=True, text=True).stdout.split()
BASE = next(r for r in ["HEAD"] + revs if "SPLIT_VERSIONS" not in subprocess.run(
    ["git", "show", f"{r}:scene_split.py"], capture_output=True, text=True).stdout)
old = {}
exec(subprocess.run(["git", "show", f"{BASE}:scene_split.py"], capture_output=True, text=True, check=True).stdout, old)
print(f"v1 reference: scene_split.py at {BASE[:7]}")

ds = PilotDataset(require_prev_frame=True, use_speed=True, use_cpm=False, use_coop=False)
print(f"D sample list: {len(ds)} samples")

print("\n1. v1 unchanged")
ref = [s.indices for s in old["three_way_scene_split"](ds, seed=42)]
for label, kw in (("default", {}), ("split_version='v1'", {"split_version": "v1"})):
    got = [s.indices for s in ss.three_way_scene_split(ds, seed=42, **kw)]
    check(got == ref, f"{label}: train/tuning/test index lists identical to {BASE[:7]} "
                      f"({len(ref[0])}/{len(ref[1])}/{len(ref[2])})")

print("\n2. v2")
tr, tu, te = ss.three_way_scene_split(ds, seed=42, split_version="v2")
scenes = {k: {(ds.samples[i][3], ds.samples[i][1]) for i in sub.indices} for k, sub in
          (("train", tr), ("tuning", tu), ("test", te))}
with open("scene_split_proposal.json") as f:
    prop = json.load(f)
want = {k: {(t, s) for t, v in prop.items() for s in v[k]} for k in ("train", "tuning", "test")}
check(scenes == want, "v2 scene sets == scene_split_proposal.json (approved)")
allsc = {(s[3], s[1]) for s in ds.samples}
check(not (scenes["train"] & scenes["tuning"]) and not (scenes["train"] & scenes["test"])
      and not (scenes["tuning"] & scenes["test"]) and set().union(*scenes.values()) == allsc and len(allsc) == 54,
      "v2 splits disjoint and cover all 54 scenes")
check(len(tr) + len(tu) + len(te) == len(ds), f"every sample in exactly one split ({len(tr)}+{len(tu)}+{len(te)} = {len(ds)})")
for k, sub in (("train", tr), ("tuning", tu), ("test", te)):
    c = collections.Counter(ds.samples[i][3] for i in sub.indices)
    print(f"  {k:6s} {len(sub):6d} samples ({len(sub) / len(ds):.3f})  " + "  ".join(f"{t} {c[t]}" for t in sorted(c))
          + f"  scenes {len(scenes[k])}")
check(len(scenes["tuning"]) == 8 and len(tu) == 2208, "v2 tuning: 8 scenes, 2,208 samples (7-of-8 rule, --expect-n 2208)")
old_test = {(t, s) for t, v in json.load(open("test_scenes.json")).items() for s in v}
print(f"  v1 test scenes in v2 test: {sorted(s for _, s in scenes['test'] & old_test)}; "
      f"v1 tuning scenes in v2 test: {sorted(s for t, s in scenes['test'] if s in ss.KNOWN_TUNING_SCENES.get(t, ()))}")

print("\n3. Configs")
with open("runs/D_seed0/config.json") as f:
    dcfg = json.load(f)
check("split_version" not in dcfg and ss.split_version_of(dcfg) == "v1", "runs/D_seed0 config (no key) -> v1")
src = open("train.py").read()
check(re.search(r'"--split-version", choices=SPLIT_VERSIONS, default="v2"', src) is not None
      and '"split_version": args.split_version' in src and "split_version=args.split_version" in src,
      "train.py: --split-version default v2, written to config.json, passed to three_way_scene_split")
src = open("eval_scenes.py").read()
check("split_version = split_version_of(run_config)" in src and "split_version=split_version)" in src,
      "eval_scenes.py: split_version from the run's config")
print("\nRESULT: " + ("PASS" if not fails else f"FAIL {fails}"))
