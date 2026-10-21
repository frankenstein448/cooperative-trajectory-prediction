"""
verify_env_targets.py
Checks ENV_CACHE_DIR/env_targets.npz (build_env_target_cache.py) against the raw V2X-Sim JSON,
re-derived independently of the builder (no V2XMetadataLoader, no chain building: windows are
walked along the prev/next pointers of the annotation itself; ego / CAV poses from every
sample_data record of that agent and frame, which must agree).

1. 50 random targets (seed 0): every stored field must match exactly (tokens, category, vis, nlp,
   coop mask, is_cav; floats compared after the same float32 cast), plus the full kept-target
   set and n_removed of each of their samples.
2. Counts and class shares, and manifest builder_dirty=False, against the scoping run (env_scoping.log / env_followup.log, 2026-10-13).

Run in gpu-test-venv:  python verify_env_targets.py
"""
import os
import re
import json
import math
import collections

import numpy as np

import paths
from build_env_target_cache import ENV_CACHE_DIR, CAT4_OF, load_test_scenes

ROOT = paths.V2X_META_DIR
N_CHECK = 50
# scoping numbers (env_scoping.log, env_followup.log)
SCOPING_BEFORE = {"train": 298747, "tuning": 63372, "test": 68270}
SCOPING_KEPT = {"train": 267669, "tuning": 56809, "test": 61423}
SCOPING_KEPT_SHARES = (0.619, 0.204, 0.178)  # nlp pipe, after the filter
SCOPING_SAMPLES = {"train": 10488, "tuning": 2208, "test": 2208}


def jl(name):
    with open(f"{ROOT}/{name}") as f:
        return json.load(f)


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rot(dx, dy, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return dx * c - dy * s, dx * s + dy * c


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    C = dict(np.load(os.path.join(ENV_CACHE_DIR, "env_targets.npz")))
    with open(os.path.join(ENV_CACHE_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    n = len(C["sample"])
    print(f"cache: {n} targets, {len(C['sample_ptr']) - 1} samples, manifest commit {manifest['git_commit'][:7]}")
    fails = []
    print(f"  manifest builder_dirty={manifest['builder_dirty']}, frame_rate_hz={manifest.get('frame_rate_hz')}, "
          f"horizon_s={manifest.get('horizon_s')}, hist_s={manifest.get('hist_s')}")
    if manifest["builder_dirty"]:
        fails.append("cache built from an uncommitted builder")

    # ---------- 2. counts vs scoping (no metadata needed) ----------
    sp_t = C["sample_split"][C["sample"]]
    for sp in ("train", "tuning", "test"):
        ss = C["sample_split"] == sp
        kept, rem = int((sp_t == sp).sum()), int(C["n_removed"][ss].sum())
        ok = (ss.sum() == SCOPING_SAMPLES[sp] and kept + rem == SCOPING_BEFORE[sp] and kept == SCOPING_KEPT[sp])
        print(f"  {sp:7s} samples {ss.sum()} (scoping {SCOPING_SAMPLES[sp]}), before {kept + rem} "
              f"(scoping {SCOPING_BEFORE[sp]}), kept {kept} (scoping {SCOPING_KEPT[sp]})  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"counts {sp}")
    shares = tuple(round(float(np.mean(C["vis"] == k)), 3) for k in range(3))
    ok = shares == SCOPING_KEPT_SHARES
    print(f"  kept class shares a/b/c {shares} vs scoping {SCOPING_KEPT_SHARES}  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        fails.append("class shares")
    ptr = C["sample_ptr"]
    assert ptr[0] == 0 and ptr[-1] == n and (np.diff(ptr) >= 0).all()
    assert (C["sample"] == np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))).all(), "targets not contiguous per sample"
    test_scenes = load_test_scenes()
    from build_env_target_cache import split_of
    assert all(split_of(int(s), test_scenes) == sp for s, sp in zip(C["sample_scene"], C["sample_split"]))

    # ---------- 1. raw re-derivation ----------
    print("loading raw JSON (once) ...", flush=True)
    ann = {a["token"]: a for a in jl("sample_annotation.json")}
    by_sample = collections.defaultdict(list)
    for a in ann.values():
        by_sample[a["sample_token"]].append(a)
    frame_of = {r["token"]: r["timestamp"] for r in jl("sample.json")}
    catn = {c["token"]: c["name"] for c in jl("category.json")}
    icat = {i["token"]: catn[i["category_token"]] for i in jl("instance.json")}
    poses = {p["token"]: p for p in jl("ego_pose.json")}
    recs = collections.defaultdict(list)
    for r in jl("sample_data.json"):
        m = re.search(r"id_(\d+)/scene_(\d+)_(\d+)\.", r["filename"])
        if m:
            recs[tuple(int(g) for g in m.groups())].append(r)

    def agent_pose(a, s, f):
        rs = recs.get((a, s, f))
        if not rs:
            return None
        ps = {(tuple(poses[r["ego_pose_token"]]["translation"][:2]), tuple(poses[r["ego_pose_token"]]["rotation"]))
              for r in rs}
        assert len(ps) == 1, f"agent {a} scene {s} frame {f}: sample_data records disagree on the ego pose"
        (xy, q), = ps
        st = {r["sample_token"] for r in rs}
        assert len(st) == 1
        return np.array(xy), yaw_of(q), st.pop()

    def walk(tok, key, k):
        out, t = [], tok
        for _ in range(k):
            t = ann[t][key]
            if not t:
                return None
            out.append(t)
        return out

    rng = np.random.default_rng(0)
    pick = rng.choice(n, size=N_CHECK, replace=False)
    max_diff = 0.0
    n_set_ok = 0
    for idx in pick:
        si = int(C["sample"][idx])
        a, s, f = int(C["sample_agent"][si]), int(C["sample_scene"][si]), int(C["sample_frame"][si])
        exy, eyaw, stok = agent_pose(a, s, f)
        assert frame_of[stok] == f
        cav = {b: agent_pose(b, s, f) for b in (1, 2, 3, 4) if b != a}
        cav = {b: v for b, v in cav.items() if v is not None}
        coop = [b for b, v in cav.items() if np.hypot(*(v[0] - exy)) <= 200.0]
        near = sorted((math.hypot(x["translation"][0] - exy[0], x["translation"][1] - exy[1]), x["token"])
                      for x in by_sample[stok])
        near = [t for t in near if t[0] <= 50.0]
        if near and near[0][0] <= 1.0:
            near = near[1:]
        kept, nrem = {}, 0
        for d, tok in near:
            x = ann[tok]
            if not icat[x["instance_token"]].startswith("vehicle."):
                continue
            hist = walk(tok, "prev", 8)
            fut = walk(tok, "next", 30)
            if hist is None or fut is None:
                nrem += 1
                continue
            toks = hist[::-1] + [tok] + fut
            fr = [frame_of[ann[t]["sample_token"]] for t in toks]
            xy = np.array([ann[t]["translation"][:2] for t in toks])
            if fr != list(range(f - 8, f + 31)) or not np.isfinite(xy).all() or \
                    not (np.hypot(*np.diff(xy, axis=0).T) <= 10.0).all():
                nrem += 1
                continue
            kept[tok] = (d, xy)
        cached = list(C["ann"][ptr[si]:ptr[si + 1]])
        set_ok = sorted(cached) == sorted(kept) and nrem == int(C["n_removed"][si])
        n_set_ok += set_ok
        if not set_ok:
            fails.append(f"target set of sample {si}")

        tok = str(C["ann"][idx])
        x = ann[tok]
        d, xy = kept[tok]
        p0, yaw = xy[8], yaw_of(x["rotation"])
        nlp = list(x["num_lidar_pts"]) + [-1] * (6 - len(x["num_lidar_pts"]))
        is_cav = next((b for b, v in cav.items() if np.hypot(*(p0 - v[0])) <= 1.0), 0)
        vis = 0 if nlp[a] >= 5 else (1 if any(b != is_cav and nlp[b] >= 5 for b in coop) else 2)
        cat = icat[x["instance_token"]]
        exp = dict(
            inst=x["instance_token"], category=cat, cat4=CAT4_OF[cat], two_wheeler=CAT4_OF[cat] >= 2, vis=vis,
            nlp=nlp, coop_mask=sum(1 << b for b in coop), is_cav=is_cav, sample=si,
            xy_world=p0, yaw_world=yaw, speed=math.hypot(*(p0 - xy[7])), size=x["size"], dist=d,
            cur_ego=rot(*(p0 - exy), eyaw), yaw_ego=wrap(yaw - eyaw),
            hist_ego=[rot(*(q - exy), eyaw) for q in xy[:8]], fut_ego=[rot(*(q - exy), eyaw) for q in xy[9:]],
            hist_own=[rot(*(q - p0), yaw) for q in xy[:8]], fut_own=[rot(*(q - p0), yaw) for q in xy[9:]],
            fut_mask=[True] * 30)
        bad = []
        for k, v in exp.items():
            got = C[k][idx]
            want = np.asarray(v).astype(got.dtype) if got.dtype.kind not in "US" else np.asarray(v, dtype=str)
            if not np.array_equal(got, want):
                bad.append(k)
            if got.dtype.kind == "f":
                max_diff = max(max_diff, float(np.max(np.abs(got.astype(np.float64) - np.asarray(v, dtype=np.float64)))))
        if bad:
            fails.append(f"target {idx}: {bad}")
            print(f"  target {idx} (sample {si}) MISMATCH in {bad}")
    print(f"\n{N_CHECK} random targets: fields exact {N_CHECK - sum(1 for x in fails if x.startswith('target '))}/{N_CHECK}; "
          f"their samples' kept sets + n_removed exact {n_set_ok}/{N_CHECK}; "
          f"max |cache float32 - float64 re-derivation| {max_diff:.2e} m/rad")
    print(f"vis of the 50: " + ", ".join(f"{'abc'[k]}={int(np.sum(C['vis'][pick] == k))}" for k in range(3)))
    print("\nRESULT: " + ("PASS" if not fails else f"FAIL {fails}"))


if __name__ == "__main__":
    main()
