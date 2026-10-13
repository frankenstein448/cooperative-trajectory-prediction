"""
build_env_target_cache.py
Offline target cache for Part 2 (environment prediction: OTHER vehicles around the ego).
Read-only on the dataset; writes ENV_CACHE_DIR/{env_targets.npz, manifest.json}.

Samples: exactly D's ego samples (PilotDataset(require_prev_frame=True): 30 future frames and
frame-1 present), in PilotDataset order, split by three_way_scene_split's scenes
(KNOWN_TUNING_SCENES / test_scenes.json). Nothing is evaluated here.

Targets per sample: vehicle.* annotations within RADIUS_M of the ego position at the current
frame (ego pose = loader.find_sample_data, as get_cam), excluding the ego's own box (the
nearest annotation, if within EGO_MATCH_M). Kept only with a full window along the instance's
prev/next chain: HIST frames before and HORIZON after, every position finite and every
per-frame step (frame-HIST -> ... -> frame+HORIZON, 38 steps) <= MAX_STEP_M.

Visibility (num_lidar_pts[agent] >= VIS_PTS; list index = agent id; RSU never used):
  0 = (a) the ego sees it; 1 = (b) not (a), but a coop CAV sees it: a CAV in 1-4, not the ego,
  not the target itself, within COOP_RADIUS_M of the ego (find_coop_neighbors' set); 2 = (c) neither.

Per target (float32 unless noted). Positions are metres, speed is metres per frame (as D's cam):
  own frame = origin at the target's current position, x along its box yaw;
  ego frame = origin at the ego's current position, x along the ego's heading (D's convention).
  hist_* [8, 2] = frames frame-8 .. frame-1, fut_* [30, 2] = frames frame+1 .. frame+30, fut_mask [30].
  cur_ego [2], yaw_ego (box yaw - ego heading, wrapped), yaw_world, xy_world (float64),
  speed (|frame-1 -> frame|), size [3] (w, l, h), dist (to the ego), category / cat4 / two_wheeler,
  vis, nlp [6] (int16, -1 = absent), coop_mask (bit b = CAV b in the coop set), is_cav (agent id or 0),
  inst / ann tokens (str), sample (int32 index into the sample arrays).
Samples are contiguous: targets of sample s are [sample_ptr[s], sample_ptr[s + 1]).
Per sample also n_removed (vehicle targets within RADIUS_M dropped by the window filter).

Frame rate: V2X-Sim is recorded at 5 Hz, 100 frames per 20 s scene (V2X-Sim paper, arXiv 2202.08449;
local timestamps are frame indices), so HORIZON = 30 frames = 6 s, HIST = 8 frames = 1.6 s, and 1 m/frame
= 5 m/s. Steps of 5-10 m/frame (100-180 km/h) are real fast driving and are kept (see the build log).

Run in gpu-test-venv:  python build_env_target_cache.py
"""
import os
import json
import math
import subprocess
import collections

import numpy as np

from v2x_message_stream import V2XMetadataLoader, V2X_ROOT, quaternion_to_yaw
from pilot_dataset import _available_frames, ALL_SCENES, SCENE_TO_TOWN, CAMERA_COMPATIBLE_AGENTS
from scene_split import KNOWN_TUNING_SCENES
from ground_truth_extractor import HORIZON

ENV_CACHE_DIR = os.path.expanduser("~/hd_cache/env_targets")
RADIUS_M = 50.0
EGO_MATCH_M = 1.0
HIST = 8
MAX_STEP_M = 10.0
VIS_PTS = 5
FRAME_RATE_HZ = 5.0     # V2X-Sim paper (arXiv 2202.08449)
COOP_RADIUS_M = 200.0     # find_coop_neighbors.COOP_RADIUS_M
COOP_CAVS = (1, 2, 3, 4)  # pilot_dataset.CAMERA_COMPATIBLE_AGENTS, the coop stream's candidates
SPLITS = ("train", "tuning", "test")
TOWNS = ("Town03", "Town04", "Town05")
CAT4 = ("car", "truck", "bicycle", "motorcycle")
# category name -> CAT4 index. V2X-Sim has no truck category; police cars count as cars.
CAT4_OF = {"vehicle.car": 0, "vehicle.emergency.police": 0, "vehicle.truck": 1,
           "vehicle.bicycle": 2, "vehicle.motorcycle": 3}


def load_test_scenes(path="test_scenes.json"):
    with open(path) as f:
        return {t: set(v) for t, v in json.load(f).items()}


def split_of(scene, test_scenes):
    town = SCENE_TO_TOWN[scene]
    if scene in KNOWN_TUNING_SCENES.get(town, ()):
        return "tuning"
    return "test" if scene in test_scenes.get(town, ()) else "train"


def d_samples():
    """PilotDataset(require_prev_frame=True).samples, without building the dataset."""
    out = []
    for scene in ALL_SCENES:
        for agent_id in CAMERA_COMPATIBLE_AGENTS:
            frames = _available_frames(agent_id, scene)
            for frame in sorted(frames):
                if all(frame + k in frames for k in range(1, HORIZON + 1)) and (frame - 1) in frames:
                    out.append((agent_id, scene, frame))
    return out


def to_frame(xy, origin, yaw):
    """xy [..., 2] world -> frame at origin [2] with heading yaw (rotate by -yaw), D's convention."""
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx, dy = xy[..., 0] - origin[0], xy[..., 1] - origin[1]
    return np.stack([dx * c - dy * s, dx * s + dy * c], axis=-1)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def jload(name):
    with open(f"{V2X_ROOT}/{name}") as f:
        return json.load(f)


def main():
    test_scenes = load_test_scenes()
    samples = d_samples()
    sample_split = [split_of(s, test_scenes) for _, s, _ in samples]
    print(f"ego samples (D filter): {len(samples)}  "
          + "  ".join(f"{sp}={sample_split.count(sp)}" for sp in SPLITS))

    print("loading V2XMetadataLoader (once) ...", flush=True)
    L = V2XMetadataLoader()
    ann = {a["token"]: a for a in L.sample_annotation}
    frame_of_sample = {r["token"]: r["timestamp"] for r in jload("sample.json")}  # timestamp == frame index
    cat_name = {c["token"]: c["name"] for c in jload("category.json")}
    inst_cat = {i["token"]: cat_name[i["category_token"]] for i in jload("instance.json")}

    def pose(agent_id, scene, frame):
        p = L.ego_pose[L.find_sample_data(agent_id, scene, frame)["ego_pose_token"]]
        return np.array(p["translation"][:2], dtype=np.float64), quaternion_to_yaw(p["rotation"])

    # every vehicle chain that contains an annotation we touch, built lazily from any of its tokens
    chains = {}       # instance -> dict(toks, frames, xy)
    pos_in_chain = {}  # annotation token -> index in its chain

    def chain(tok):
        inst = ann[tok]["instance_token"]
        if inst not in chains:
            t = tok
            while ann[t]["prev"]:
                t = ann[t]["prev"]
            toks = []
            while t:
                toks.append(t)
                t = ann[t]["next"]
            frames = np.array([frame_of_sample[ann[t]["sample_token"]] for t in toks], dtype=np.int64)
            xy = np.array([ann[t]["translation"][:2] for t in toks], dtype=np.float64)
            chains[inst] = dict(toks=toks, frames=frames, xy=xy)
            for i, t in enumerate(toks):
                pos_in_chain[t] = i
        return chains[inst], pos_in_chain[tok]

    T = collections.defaultdict(list)
    sample_ptr = [0]
    n_removed = []
    removed_why = collections.Counter()
    ego_match_d = []
    yaw_vs_motion, yaw_vs_cavpose = [], []
    kept_steps = collections.defaultdict(set)  # instance -> step indices j (step j = xy[j] -> xy[j+1]) in kept windows
    for si, (a, s, f) in enumerate(samples):
        exy, eyaw = pose(a, s, f)
        rec = L.find_sample_data(a, s, f)
        near = []
        for an in L.anno_by_sample.get(rec["sample_token"], []):
            d = math.hypot(an["translation"][0] - exy[0], an["translation"][1] - exy[1])
            if d <= RADIUS_M:
                near.append((d, an["token"]))
        near.sort()
        if near and near[0][0] <= EGO_MATCH_M:
            ego_match_d.append(near[0][0])
            near = near[1:]
        else:
            ego_match_d.append(np.inf)
        cav_xy = {}
        for b in COOP_CAVS:
            if b == a:
                continue
            try:
                cav_xy[b] = pose(b, s, f)
            except ValueError:
                continue
        coop = [b for b, (bxy, _) in cav_xy.items() if math.hypot(*(bxy - exy)) <= COOP_RADIUS_M]
        coop_mask = sum(1 << b for b in coop)
        nrem = 0
        for d, tok in near:
            an = ann[tok]
            cat = inst_cat[an["instance_token"]]
            if not cat.startswith("vehicle."):
                continue
            c, i = chain(tok)
            n = len(c["toks"])
            if i - HIST < 0 or i + HORIZON >= n or c["frames"][i + HORIZON] - c["frames"][i - HIST] != HIST + HORIZON:
                nrem += 1
                removed_why["short window (history or future)"] += 1
                continue
            win = c["xy"][i - HIST:i + HORIZON + 1]
            if not np.isfinite(win).all():
                nrem += 1
                removed_why["non-finite position"] += 1
                continue
            steps = np.hypot(*np.diff(win, axis=0).T)
            if not (steps <= MAX_STEP_M).all():
                nrem += 1
                removed_why[f"step > {MAX_STEP_M:g} m"] += 1
                continue
            kept_steps[an["instance_token"]].update(range(i - HIST, i + HORIZON))

            p0 = c["xy"][i]
            yaw = quaternion_to_yaw(an["rotation"])
            nlp = list(an.get("num_lidar_pts") or [])
            nlp = [nlp[j] if j < len(nlp) else -1 for j in range(6)]
            is_cav = next((b for b, (bxy, _) in cav_xy.items() if math.hypot(*(p0 - bxy)) <= EGO_MATCH_M), 0)
            seen_by_coop = any(b != is_cav and nlp[b] >= VIS_PTS for b in coop)
            vis = 0 if nlp[a] >= VIS_PTS else (1 if seen_by_coop else 2)
            hist_w, fut_w = c["xy"][i - HIST:i], c["xy"][i + 1:i + HORIZON + 1]
            speed = math.hypot(*(p0 - c["xy"][i - 1]))
            if speed > 0.5:
                mv = p0 - c["xy"][i - 1]
                yaw_vs_motion.append((math.degrees(abs(wrap(yaw - math.atan2(mv[1], mv[0])))), cat))
            if is_cav:
                yaw_vs_cavpose.append(math.degrees(abs(wrap(yaw - cav_xy[is_cav][1]))))

            T["sample"].append(si)
            T["inst"].append(an["instance_token"]); T["ann"].append(tok)
            T["category"].append(cat); T["cat4"].append(CAT4_OF[cat]); T["two_wheeler"].append(CAT4_OF[cat] >= 2)
            T["vis"].append(vis); T["nlp"].append(nlp); T["coop_mask"].append(coop_mask); T["is_cav"].append(is_cav)
            T["xy_world"].append(p0); T["yaw_world"].append(yaw); T["speed"].append(speed)
            T["size"].append(an["size"]); T["dist"].append(d)
            T["cur_ego"].append(to_frame(p0, exy, eyaw)); T["yaw_ego"].append(wrap(yaw - eyaw))
            T["hist_ego"].append(to_frame(hist_w, exy, eyaw)); T["fut_ego"].append(to_frame(fut_w, exy, eyaw))
            T["hist_own"].append(to_frame(hist_w, p0, yaw)); T["fut_own"].append(to_frame(fut_w, p0, yaw))
            T["fut_mask"].append(np.ones(HORIZON, bool))
        n_removed.append(nrem)
        sample_ptr.append(len(T["sample"]))
        if (si + 1) % 2000 == 0:
            print(f"  {si + 1}/{len(samples)} samples, {len(T['sample'])} targets", flush=True)

    f32 = ("speed", "yaw_world", "yaw_ego", "size", "dist", "cur_ego", "hist_ego", "fut_ego", "hist_own", "fut_own")
    arr = {}
    for k, v in T.items():
        if k in f32:
            arr[k] = np.asarray(v, dtype=np.float32)
        elif k in ("inst", "ann", "category"):
            arr[k] = np.asarray(v, dtype=str)
        elif k == "xy_world":
            arr[k] = np.asarray(v, dtype=np.float64)
        elif k == "nlp":
            arr[k] = np.asarray(v, dtype=np.int16)
        elif k in ("fut_mask", "two_wheeler"):
            arr[k] = np.asarray(v, dtype=bool)
        else:
            arr[k] = np.asarray(v, dtype=np.int32)
    smp = np.asarray(samples, dtype=np.int32)
    arr.update(sample_agent=smp[:, 0], sample_scene=smp[:, 1], sample_frame=smp[:, 2],
               sample_split=np.asarray(sample_split, dtype=str),
               sample_town=np.asarray([SCENE_TO_TOWN[s] for _, s, _ in samples], dtype=str),
               sample_ptr=np.asarray(sample_ptr, dtype=np.int64), n_removed=np.asarray(n_removed, dtype=np.int32))
    os.makedirs(ENV_CACHE_DIR, exist_ok=True)
    np.savez(os.path.join(ENV_CACHE_DIR, "env_targets.npz"), **arr)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "build_env_target_cache.py"],
                                capture_output=True, text=True).stdout.strip())
    manifest = dict(git_commit=commit, builder_dirty=dirty, n_samples=len(samples), n_targets=int(len(arr["sample"])),
                    radius_m=RADIUS_M, ego_match_m=EGO_MATCH_M, hist=HIST, horizon=HORIZON, max_step_m=MAX_STEP_M,
                    vis_pts=VIS_PTS, coop_radius_m=COOP_RADIUS_M, coop_cavs=list(COOP_CAVS), cat4=list(CAT4),
                    units="metres; speed metres per frame",
                    frame_rate_hz=FRAME_RATE_HZ, horizon_s=HORIZON / FRAME_RATE_HZ, hist_s=HIST / FRAME_RATE_HZ,
                    frame_rate_source="V2X-Sim paper, arXiv 2202.08449: 5 Hz, 100 frames per 20 s scene")
    with open(os.path.join(ENV_CACHE_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nsaved {ENV_CACHE_DIR}/env_targets.npz ({os.path.getsize(os.path.join(ENV_CACHE_DIR, 'env_targets.npz')) / 1e6:.0f} MB)"
          f", manifest commit {commit[:7]} (builder dirty={dirty})")
    report(arr, removed_why, np.asarray(ego_match_d), yaw_vs_motion, yaw_vs_cavpose, chains, kept_steps)


def report(arr, removed_why, ego_match_d, yaw_vs_motion, yaw_vs_cavpose, chains, kept_steps):
    sp_t = arr["sample_split"][arr["sample"]]
    tw_t = arr["sample_town"][arr["sample"]]
    nrem = arr["n_removed"]
    sp_s, tw_s = arr["sample_split"], arr["sample_town"]

    print(f"\nego self-match: {np.isfinite(ego_match_d).mean():.4f} of samples, max distance "
          f"{ego_match_d[np.isfinite(ego_match_d)].max():.3f} m")
    print("\nFilter (vehicle.* within 50 m, ego excluded): before / removed / kept")
    print(f"  {'split':7s} {'town':7s} {'before':>8s} {'removed':>8s} {'kept':>8s} {'removed %':>9s}")
    for sp in SPLITS + ("ALL",):
        for tw in TOWNS + ("ALL",):
            ss = ((sp_s == sp) | (sp == "ALL")) & ((tw_s == tw) | (tw == "ALL"))
            st = ((sp_t == sp) | (sp == "ALL")) & ((tw_t == tw) | (tw == "ALL"))
            k, r = int(st.sum()), int(nrem[ss].sum())
            print(f"  {sp:7s} {tw:7s} {k + r:8d} {r:8d} {k:8d} {100 * r / max(k + r, 1):8.2f}%")
    print("  removed by reason (first failing check): " + ", ".join(f"{k}={v}" for k, v in removed_why.items()))

    print("\nVisibility (num_lidar_pts >= 5; coop = CAVs 1-4 within 200 m; RSU excluded), kept targets")
    print(f"  {'split':7s} {'town':7s} {'targets':>8s} {'(a) ego':>8s} {'(b) coop':>8s} {'(c) none':>8s}")
    for sp in SPLITS + ("ALL",):
        for tw in TOWNS + ("ALL",):
            st = ((sp_t == sp) | (sp == "ALL")) & ((tw_t == tw) | (tw == "ALL"))
            v = arr["vis"][st]
            print(f"  {sp:7s} {tw:7s} {st.sum():8d} " + " ".join(f"{np.mean(v == k):8.3f}" for k in range(3)))

    print("\nCategory (kept targets): " + ", ".join(f"{k}={v}" for k, v in collections.Counter(arr["category"]).most_common()))
    for sp in SPLITS + ("ALL",):
        st = (sp_t == sp) | (sp == "ALL")
        c4 = arr["cat4"][st]
        print(f"  {sp:7s} " + "  ".join(f"{CAT4[k]} {np.mean(c4 == k):.3f}" for k in range(4))
              + f"  | two-wheeler share by vis a/b/c: "
              + " ".join(f"{np.mean(arr['two_wheeler'][st & (arr['vis'] == k)]):.3f}" for k in range(3)))

    print(f"\ntargets that are a CAV 1-4 (pose within {EGO_MATCH_M} m): "
          + ", ".join(f"{b}={int((arr['is_cav'] == b).sum())}" for b in COOP_CAVS))
    sp = arr["speed"]
    print(f"current speed (m/frame): stationary (==0) {np.mean(sp == 0):.3f}, p50 {np.median(sp):.2f}, "
          f"p90 {np.percentile(sp, 90):.2f}, max {sp.max():.2f}")
    print(f"distance to ego: p50 {np.median(arr['dist']):.1f} m, max {arr['dist'].max():.2f} m; "
          f"min {arr['dist'].min():.2f} m")

    ym = np.array([d for d, _ in yaw_vs_motion])
    bins = [0, 5, 15, 45, 135, 180.01]
    h, _ = np.histogram(ym, bins=bins)
    print(f"\nbox yaw vs motion direction (frame-1 -> frame, speed > 0.5 m/frame, n={len(ym)}): "
          + "  ".join(f"[{bins[i]:g},{bins[i + 1]:.0f}):{h[i] / len(ym):.4f}" for i in range(len(h))))
    for cat in sorted({c for _, c in yaw_vs_motion}):
        yc = np.array([d for d, c in yaw_vs_motion if c == cat])
        print(f"  {cat:26s} n={len(yc):7d}  median {np.median(yc):.2f} deg, p99 {np.percentile(yc, 99):.1f}, >45 deg {np.mean(yc > 45):.4f}")
    yc = np.asarray(yaw_vs_cavpose)
    print(f"box yaw vs the CAV's own ego_pose yaw (CAV targets, n={len(yc)}): median {np.median(yc):.3f} deg, "
          f"max {yc.max():.2f} deg")
    print(f"own-frame sanity: |hist_own[-1] + (speed, 0)| p99 over moving targets = "
          f"{np.percentile(np.hypot(arr['hist_own'][:, -1, 0] + sp, arr['hist_own'][:, -1, 1])[sp > 0.5], 99):.3f} m "
          "(0 iff the box yaw equals the motion direction)")

    # ---- 5-10 m steps: unique (track, step) pairs inside kept windows ----
    print("\nPer-frame steps inside kept windows (unique track steps): 5-10 m share per town, and their neighbours")
    town_of_inst = {}
    for inst, smp in zip(arr["inst"], arr["sample"]):
        town_of_inst.setdefault(inst, arr["sample_town"][smp])
    rows = collections.defaultdict(list)
    for inst, js in kept_steps.items():
        st = np.hypot(*np.diff(chains[inst]["xy"], axis=0).T)
        xy = chains[inst]["xy"]
        js = sorted(js)
        for j in js:
            s = st[j]
            nb = [st[k] for k in (j - 1, j + 1) if 0 <= k < len(st)]
            hd = []
            for k in (j - 1, j + 1):
                if 0 <= k < len(st) and st[k] > 0.5 and s > 0.5:
                    v0, v1 = xy[j + 1] - xy[j], xy[k + 1] - xy[k]
                    hd.append(abs(math.degrees(wrap(math.atan2(v1[1], v1[0]) - math.atan2(v0[1], v0[0])))))
            rows[town_of_inst[inst]].append((s, max(abs(s - x) for x in nb) if nb else np.nan,
                                             min(nb) if nb else np.nan, max(hd) if hd else np.nan))
    for tw in TOWNS:
        r = np.array(rows[tw])
        big = (r[:, 0] >= 5) & (r[:, 0] <= MAX_STEP_M)
        rb = r[big]
        print(f"  {tw}: {len(r)} steps, 5-10 m {big.sum()} ({100 * big.mean():.2f}%), steps 0 m {np.mean(r[:, 0] == 0):.3f}")
        if not len(rb):
            continue
        dmax, nmin, hmax = rb[:, 1], rb[:, 2], rb[:, 3]
        print(f"    max |step - neighbour step|: <=0.1 m {np.mean(dmax <= 0.1):.3f}, <=0.5 m {np.mean(dmax <= 0.5):.3f}, "
              f"<=2 m {np.mean(dmax <= 2):.3f}, >2 m {np.mean(dmax > 2):.3f}")
        print(f"    both neighbours also >= 4 m: {np.mean(nmin >= 4):.3f}; isolated spike (both neighbours "
              f"< 2.5 m): {np.mean(nmin < 2.5):.3f}; max heading change to a neighbour step: "
              f"<=5 deg {np.nanmean(hmax <= 5):.3f}, >30 deg {np.nanmean(hmax > 30):.3f}")
        print(f"    speed implied: p50 {np.median(rb[:, 0]) * 5 * 3.6:.0f} km/h, max {rb[:, 0].max() * 5 * 3.6:.0f} km/h (5 Hz)")
    print("\nDONE")


if __name__ == "__main__":
    main()
