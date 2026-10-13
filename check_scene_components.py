"""
check_scene_components.py
Read-only proposal for a leak-free scene split (writes nothing but stdout and scene_split_proposal.json).

1. Scenes per source log per town (scene.json -> log.json).
2. Connected components of the 54 used scenes, linked when they share exact moving-vehicle positions
   (as check_scene_overlap.py: rounded to 1 mm, nonzero frame-1 -> frame step) in > LINK_SHARE of the
   smaller scene's moving positions.
3. Proposal: whole components assigned to train / tuning / test per town, about 70/15/15 by D samples,
   every town with >= 1 tuning and >= 1 test component. Exhaustive over assignments per town; the cost is the
   squared deviation from the target shares, with a small penalty per old TUNING scene in the new test
   split (those scenes drove past decisions) and ties broken toward the smallest component ids, so the result is
   deterministic. Cross-split overlap of the proposal is recomputed at every pair level (any shared moving
   position at all, not only > LINK_SHARE).

Run in gpu-test-venv:  python check_scene_components.py
"""
import re
import json
import itertools
import collections

import numpy as np

from pilot_dataset import ALL_SCENES, SCENE_TO_TOWN
from scene_split import KNOWN_TUNING_SCENES

ROOT = "/home/karra/V2X-Sim-2.0/v2.0"
LINK_SHARE = 0.01
TARGET = {"train": 0.70, "tuning": 0.15, "test": 0.15}
OLD_TUNING_IN_TEST_PENALTY = 0.002
SAMPLES_PER_SCENE_CACHE = "/home/karra/hd_cache/env_targets/env_targets.npz"


def jl(name):
    with open(f"{ROOT}/{name}") as f:
        return json.load(f)


def main():
    with open("test_scenes.json") as f:
        old_test = {t: set(v) for t, v in json.load(f).items()}
    old_split = lambda s: ("tuning" if s in KNOWN_TUNING_SCENES.get(SCENE_TO_TOWN[s], ())
                           else "test" if s in old_test.get(SCENE_TO_TOWN[s], ()) else "train")
    z = np.load(SAMPLES_PER_SCENE_CACHE)
    n_samples = collections.Counter(z["sample_scene"].tolist())

    print("loading raw JSON (once) ...", flush=True)
    ann = jl("sample_annotation.json")
    smp = {r["token"]: r for r in jl("sample.json")}
    scs = {r["token"]: r for r in jl("scene.json")}
    logs = {r["token"]: r["logfile"] for r in jl("log.json")}
    scn = {t: int(re.search(r"(\d+)$", r["name"]).group(1)) for t, r in scs.items()}
    log_of = {scn[t]: logs[r["log_token"]] for t, r in scs.items()}
    bt = {a["token"]: a for a in ann}
    used = sorted(ALL_SCENES)
    mv = collections.defaultdict(set)
    for a in ann:
        s = scn[smp[a["sample_token"]]["scene_token"]]
        if s in used and a["prev"]:
            p, q = a["translation"], bt[a["prev"]]["translation"]
            if (p[0], p[1]) != (q[0], q[1]) and abs(p[0]) < 1e6:
                mv[s].add((round(p[0], 3), round(p[1], 3)))
    del ann, bt

    print("\n1. Scenes per source log (used scenes; D samples per scene in brackets)")
    for town in ("Town03", "Town04", "Town05"):
        for lg in sorted({log_of[s] for s in used if SCENE_TO_TOWN[s] == town}):
            ss = [s for s in used if log_of[s] == lg]
            print(f"  {town} {lg:26s} {len(ss):2d} scenes: " + " ".join(f"{s}({old_split(s)[:2]})" for s in ss))
    print("  (tr/tu/te = old split)")

    share = {}
    for a, b in itertools.combinations(used, 2):
        inter = len(mv[a] & mv[b])
        share[(a, b)] = share[(b, a)] = inter / max(1, min(len(mv[a]), len(mv[b])))
    parent = {s: s for s in used}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (a, b), v in share.items():
        if a < b and v > LINK_SHARE:
            parent[find(a)] = find(b)
    comps = collections.defaultdict(list)
    for s in used:
        comps[find(s)].append(s)
    comps = sorted((sorted(v) for v in comps.values()), key=lambda c: (SCENE_TO_TOWN[c[0]], c[0]))
    print(f"\n2. Connected components (link: shared moving positions > {LINK_SHARE:.0%} of the smaller scene): {len(comps)}")
    for i, c in enumerate(comps):
        lgs = sorted({log_of[s] for s in c})
        mx = max((share[(a, b)] for a, b in itertools.combinations(c, 2)), default=0.0)
        print(f"  C{i:02d} {SCENE_TO_TOWN[c[0]]} {len(c):2d} scenes {sum(n_samples[s] for s in c):5d} samples  scenes {c}  "
              f"logs {lgs}  max pair share {mx:.3f}  old splits {dict(collections.Counter(old_split(s) for s in c))}")
    weak = sorted((v, a, b) for (a, b), v in share.items() if a < b and 0 < v <= LINK_SHARE)
    print(f"  pairs with some shared moving positions but <= {LINK_SHARE:.0%} (not linked): {len(weak)}"
          + (f", largest {weak[-1][0]:.4f} ({weak[-1][1]}, {weak[-1][2]})" if weak else ""))

    print("\n3. Proposal (whole components, per town, ~70/15/15 by samples)")
    proposal, report = {}, []
    for town in ("Town03", "Town04", "Town05"):
        tc = [c for c in comps if SCENE_TO_TOWN[c[0]] == town]
        sizes = [sum(n_samples[s] for s in c) for c in tc]
        total = sum(sizes)
        best = None
        for assign in itertools.product(("train", "tuning", "test"), repeat=len(tc)):
            if "tuning" not in assign or "test" not in assign or "train" not in assign:
                continue
            frac = {k: sum(sz for sz, a in zip(sizes, assign) if a == k) / total for k in TARGET}
            cost = sum((frac[k] - TARGET[k]) ** 2 for k in TARGET)
            cost += OLD_TUNING_IN_TEST_PENALTY * sum(old_split(s) == "tuning" for c, a in zip(tc, assign) if a == "test" for s in c)
            key = (round(cost, 9), assign)
            if best is None or key < best[0]:
                best = (key, assign, frac)
        if best is None:
            print(f"  {town}: {len(tc)} component(s) -- cannot give train, tuning AND test whole components")
            continue
        _, assign, frac = best
        for c, a in zip(tc, assign):
            for s in c:
                proposal[s] = a
        print(f"  {town}: {len(tc)} components, samples {total}: " + ", ".join(f"{k} {frac[k]:.3f}" for k in TARGET))
        for k in TARGET:
            ss = sorted(s for s in proposal if SCENE_TO_TOWN[s] == town and proposal[s] == k)
            print(f"    {k:6s}: {ss}  (old: {dict(collections.Counter(old_split(s) for s in ss))})")

    if len(proposal) == len(used):
        tot = collections.Counter()
        for s, k in proposal.items():
            tot[k] += n_samples[s]
        print("  overall samples: " + ", ".join(f"{k} {tot[k]} ({tot[k] / sum(tot.values()):.3f})" for k in TARGET))
        cross = [(a, b, share[(a, b)]) for a, b in itertools.combinations(used, 2)
                 if proposal[a] != proposal[b] and share[(a, b)] > 0]
        print(f"  cross-split pairs with ANY shared moving position: {len(cross)}"
              + (f", max share {max(v for *_, v in cross):.4f}: {cross[:10]}" if cross else ""))
        old_tu_in_test = sorted(s for s, k in proposal.items() if k == "test" and old_split(s) == "tuning")
        old_te_in_train = sorted(s for s, k in proposal.items() if k == "train" and old_split(s) == "test")
        print(f"  old tuning scenes in the new test split: {old_tu_in_test}; old test scenes in the new train split: "
              f"{old_te_in_train}")
        with open("scene_split_proposal.json", "w") as f:
            json.dump({t: {k: sorted(s for s in proposal if SCENE_TO_TOWN[s] == t and proposal[s] == k) for k in TARGET}
                       for t in ("Town03", "Town04", "Town05")}, f, indent=2)
        print("  wrote scene_split_proposal.json (a proposal only; test_scenes.json is untouched)")
    print("\nDONE")


if __name__ == "__main__":
    main()
