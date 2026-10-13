"""
check_map_alignment.py
Read-only map-alignment diagnostic (runs/DESIGN_ENV.md section 9.1). Changes nothing that any model uses.

Points: every MOVING vehicle.* annotation (CAVs included) in the 54 used scenes. Moving = nonzero
displacement from the previous annotation of the same instance (frame-1 -> frame), the step <= 10 m and
both positions finite and not the sentinel (|x| < 1e6). Lane waypoints: dense_waypoints_<town>.npy.

Per scene:
  - source log (scene.json -> log.json) and town from the scene description and the log, checked against
    pilot_dataset.SCENE_TO_TOWN;
  - distance of each point to the nearest waypoint under the current per-town transform
    (coord_transform.TOWN_TRANSFORMS): median, p95 (all moving vehicles; CAVs alone for reference);
  - the parameter-free y-mirror (x, y) -> (x, -y) (CARLA is left-handed; a first test on 9 scenes gave
    medians 0.54-0.65 m with it, vs 3-15 m under the per-town rotations);
  - a per-scene refit of (theta, tx, ty), the existing grid-search approach (grid_search_transform.py:
    coarse global grid, then a local refinement), run for both handednesses: rotation only (translation
    window around the per-town (tx, ty)) and y-mirror then rotation (window around (0, 0)); the lower
    objective wins and its `mirror` flag is reported. Carla = R(theta) @ M @ p + t, M = diag(1, -1) if mirror. Objective: robust truncated mean of min(d, TRUNC_M) on a
    random subsample, scored on a 0.2 m Euclidean distance-transform raster of the waypoints (lookup instead
    of a KD-tree query per candidate). The final metrics are exact KD-tree distances on all points;
  - pass = median <= 1.0 m AND p95 <= 2.5 m (fixed before running), for the old transform, the y-mirror and
    the per-scene refit ("new").
Scenes failing both are listed for exclusion. Part 1 impact: tuning / test scenes failing the old transform.
Writes stdout and map_alignment_per_scene.json (per-scene numbers and refit transforms; nothing reads it yet).

Run in gpu-test-venv:  python check_map_alignment.py
"""
import re
import json
import collections
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

from coord_transform import TOWN_TRANSFORMS
from pilot_dataset import ALL_SCENES, SCENE_TO_TOWN
from scene_split import KNOWN_TUNING_SCENES

ROOT = "/home/karra/V2X-Sim-2.0/v2.0"
PASS_MEDIAN_M, PASS_P95_M = 1.0, 2.5
MAX_STEP_M, SENTINEL = 10.0, 1e6
TRUNC_M = 3.0
RES = 0.2
N_COARSE, N_FINE = 800, 4000
COARSE_THETA = np.arange(0.0, 360.0, 2.0)
COARSE_T = np.arange(-60.0, 60.01, 2.0)          # around the per-town (tx, ty)
FINE_THETA = np.arange(-3.0, 3.001, 0.1)         # around the coarse best
FINE_T = np.arange(-3.0, 3.001, 0.1)
OUT_JSON = "map_alignment_per_scene.json"


def jl(name):
    with open(f"{ROOT}/{name}") as f:
        return json.load(f)


def rot(pts, theta_deg):
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return pts @ np.array([[c, -s], [s, c]]).T


class Field:
    """Distance (m) to the nearest waypoint on a RES raster; outside the raster = TRUNC_M."""
    def __init__(self, wp):
        self.x0, self.y0 = wp.min(0) - 150.0
        nx, ny = ((wp.max(0) + 150.0 - wp.min(0) + 150.0) / RES).astype(int) + 1
        occ = np.ones((nx, ny), bool)
        ix = ((wp[:, 0] - self.x0) / RES).round().astype(int)
        iy = ((wp[:, 1] - self.y0) / RES).round().astype(int)
        occ[ix, iy] = False
        self.d = np.minimum(distance_transform_edt(occ) * RES, TRUNC_M).astype(np.float32)
        self.nx, self.ny = nx, ny

    def score(self, prot, txs, tys):
        """prot [n, 2] rotated points -> [len(txs), len(tys)] truncated mean distance."""
        ix = np.floor((prot[None, :, 0] + txs[:, None] - self.x0) / RES + 0.5).astype(np.int64)  # [ntx, n]
        iy = np.floor((prot[None, :, 1] + tys[:, None] - self.y0) / RES + 0.5).astype(np.int64)  # [nty, n]
        okx = (ix >= 0) & (ix < self.nx)
        oky = (iy >= 0) & (iy < self.ny)
        ixc, iyc = np.clip(ix, 0, self.nx - 1), np.clip(iy, 0, self.ny - 1)
        v = self.d[ixc[:, None, :], iyc[None, :, :]]                       # [ntx, nty, n]
        v = np.where(okx[:, None, :] & oky[None, :, :], v, TRUNC_M)
        return v.mean(-1)


def search(field, pts, thetas, txs, tys):
    best = (np.inf, None, None, None)
    for th in thetas:
        sc = field.score(rot(pts, th), txs, tys)
        i, j = np.unravel_index(np.argmin(sc), sc.shape)
        if sc[i, j] < best[0]:
            best = (float(sc[i, j]), float(th), float(txs[i]), float(tys[j]))
    return best


MIRROR = np.array([1.0, -1.0])


def stats(tree, pts, theta, tx, ty, mirror=False):
    d, _ = tree.query(rot(pts * MIRROR if mirror else pts, theta) + np.array([tx, ty]))
    med, p95 = float(np.median(d)), float(np.percentile(d, 95))
    return dict(median=med, p95=p95, ok=bool(med <= PASS_MEDIAN_M and p95 <= PASS_P95_M))


def fit_scene(job):
    s, meta, P, cav = job
    rng = np.random.default_rng(s)
    town = meta["town"]
    tree, field = TREES[town], FIELDS[town]
    old = TOWN_TRANSFORMS[town]
    sub_c = P[rng.choice(len(P), min(N_COARSE, len(P)), replace=False)]
    sub_f = P[rng.choice(len(P), min(N_FINE, len(P)), replace=False)]
    fits = []
    for mirror, (cx, cy) in ((False, (old["tx"], old["ty"])), (True, (0.0, 0.0))):
        mc, mf = (sub_c * MIRROR, sub_f * MIRROR) if mirror else (sub_c, sub_f)
        coarse = search(field, mc, COARSE_THETA, cx + COARSE_T, cy + COARSE_T)
        fine = search(field, mf, coarse[1] + FINE_THETA, coarse[2] + FINE_T, coarse[3] + FINE_T)
        fits.append((fine[0], mirror, fine, coarse[0]))
    obj, mirror, fine, cobj = min(fits, key=lambda f: f[0])
    th, tx, ty = fine[1] % 360, fine[2], fine[3]
    ev = lambda pts, *a: stats(tree, pts, *a) if len(pts) else None
    return s, dict(**meta, n_points=int(len(P)), n_cav=int(cav.sum()),
                   old=ev(P, old["theta"], old["tx"], old["ty"]), old_cav=ev(P[cav], old["theta"], old["tx"], old["ty"]),
                   mirror=ev(P, 0.0, 0.0, 0.0, True), mirror_cav=ev(P[cav], 0.0, 0.0, 0.0, True),
                   refit=dict(mirror=mirror, theta=round(th, 3), tx=round(tx, 3), ty=round(ty, 3), objective=round(obj, 4),
                              coarse_objective=round(cobj, 4),
                              objective_other_handedness=round([f[0] for f in fits if f[1] != mirror][0], 4)),
                   new=ev(P, th, tx, ty, mirror), new_cav=ev(P[cav], th, tx, ty, mirror))


def main():
    with open("test_scenes.json") as f:
        test = {t: set(v) for t, v in json.load(f).items()}

    def split_of(s):
        t = SCENE_TO_TOWN[s]
        return "tuning" if s in KNOWN_TUNING_SCENES.get(t, ()) else "test" if s in test.get(t, ()) else "train"

    print("loading raw JSON (once) ...", flush=True)
    ann = jl("sample_annotation.json")
    samples = {r["token"]: r for r in jl("sample.json")}
    scenes = {r["token"]: r for r in jl("scene.json")}
    logs = {r["token"]: r for r in jl("log.json")}
    catn = {c["token"]: c["name"] for c in jl("category.json")}
    icat = {i["token"]: catn[i["category_token"]] for i in jl("instance.json")}
    by_tok = {a["token"]: a for a in ann}
    cav_tokens = set()  # CAV pose positions, to tag CAV annotations (<= 1 m from a CAV pose of that sample)
    poses = {p["token"]: p for p in jl("ego_pose.json")}
    cav_xy = collections.defaultdict(list)
    for r in jl("sample_data.json"):
        m = re.search(r"SEMLIDAR_TOP_id_(\d+)/scene_(\d+)_(\d+)\.", r["filename"])
        if m and int(m.group(1)) >= 1:
            cav_xy[r["sample_token"]].append(poses[r["ego_pose_token"]]["translation"][:2])

    scene_num = {tok: int(re.search(r"(\d+)$", s["name"]).group(1)) for tok, s in scenes.items()}
    used = set(ALL_SCENES)
    pts = collections.defaultdict(list)
    is_cav = collections.defaultdict(list)
    for a in ann:
        if not a["prev"] or not icat[a["instance_token"]].startswith("vehicle."):
            continue
        s = scene_num[samples[a["sample_token"]]["scene_token"]]
        if s not in used:
            continue
        p = np.array(a["translation"][:2])
        q = np.array(by_tok[a["prev"]]["translation"][:2])
        if not (np.isfinite(p).all() and np.isfinite(q).all()) or np.abs(p).max() > SENTINEL or np.abs(q).max() > SENTINEL:
            continue
        step = np.hypot(*(p - q))
        if step == 0 or step > MAX_STEP_M:
            continue
        pts[s].append(p)
        is_cav[s].append(any(np.hypot(*(p - np.array(c))) <= 1.0 for c in cav_xy[a["sample_token"]]))
    del ann, by_tok

    global TREES, FIELDS
    TREES, FIELDS = {}, {}
    for t in ("Town03", "Town04", "Town05"):
        wp = np.load(f"dense_waypoints_{t.lower()}.npy")
        TREES[t], FIELDS[t] = cKDTree(wp), Field(wp)
    print(f"rasters built ({RES} m); fitting {len(used)} scenes", flush=True)

    tok_of = {v: k for k, v in scene_num.items()}
    jobs = []
    for s in ALL_SCENES:
        sc = scenes[tok_of[s]]
        lg = logs[sc["log_token"]]
        desc_town = re.search(r"town\s*(\d+)", sc["description"], re.I)
        meta = dict(town=SCENE_TO_TOWN[s], split=split_of(s), log=lg["logfile"], log_location=lg["location"],
                    description=sc["description"], desc_town=int(desc_town.group(1)) if desc_town else None)
        jobs.append((s, meta, np.asarray(pts[s]), np.asarray(is_cav[s])))
    out = {}
    with Pool(6) as pool:
        for s, r in pool.imap_unordered(fit_scene, jobs):
            out[s] = r
            o, m, n = r["old"], r["mirror"], r["new"]
            print(f"  scene {s:3d} {r['town']} {r['split']:6s} n={r['n_points']:6d}  old {o['median']:6.2f}/{o['p95']:6.2f} "
                  f"{'PASS' if o['ok'] else 'fail'}  mirror {m['median']:5.2f}/{m['p95']:5.2f} {'PASS' if m['ok'] else 'fail'}  "
                  f"new {n['median']:5.2f}/{n['p95']:5.2f} {'PASS' if n['ok'] else 'fail'}  refit mirror={r['refit']['mirror']} "
                  f"theta {r['refit']['theta']:7.2f} t ({r['refit']['tx']:7.2f},{r['refit']['ty']:7.2f})", flush=True)
    with open(OUT_JSON, "w") as f:
        json.dump({str(k): v for k, v in out.items()}, f, indent=1)

    # ---------------- report ----------------
    pf = lambda x: "PASS" if x["ok"] else "fail"
    nan = dict(median=np.nan, p95=np.nan, ok=False)
    print(f"\nTown check (SCENE_TO_TOWN vs scene description vs log location):")
    bad = [s for s, r in out.items() if not (int(r["town"][-1]) == r["desc_town"] == r["log_location"])]
    print("  all consistent" if not bad else f"  INCONSISTENT scenes: {bad}")

    print(f"\nPer scene, grouped by source log: median / p95 (m) over all moving vehicles; 'CAV' = moving CAVs only "
          f"(the positions D's map lookup uses). Pass = median <= {PASS_MEDIAN_M} AND p95 <= {PASS_P95_M}.")
    print(f"  {'log':26s} {'scene':>5s} {'split':6s} {'n':>6s} | {'old':>13s} {'':4s} {'old CAV':>13s} | "
          f"{'y-mirror':>13s} {'':4s} {'mir CAV':>13s} | {'refit':>13s} {'':4s} | mirror theta, tx, ty")
    for lg in sorted({r["log"] for r in out.values()}):
        for s in sorted(k for k, r in out.items() if r["log"] == lg):
            r = out[s]
            oc, mc = r["old_cav"] or nan, r["mirror_cav"] or nan
            f = r["refit"]
            print(f"  {lg:26s} {s:5d} {r['split']:6s} {r['n_points']:6d} | "
                  f"{r['old']['median']:6.2f}/{r['old']['p95']:6.2f} {pf(r['old']):4s} {oc['median']:6.2f}/{oc['p95']:6.2f} | "
                  f"{r['mirror']['median']:6.2f}/{r['mirror']['p95']:6.2f} {pf(r['mirror']):4s} {mc['median']:6.2f}/{mc['p95']:6.2f} | "
                  f"{r['new']['median']:6.2f}/{r['new']['p95']:6.2f} {pf(r['new']):4s} | "
                  f"{str(f['mirror']):5s} {f['theta']:7.2f} {f['tx']:7.2f} {f['ty']:7.2f}")

    print("\nRefit per log: handedness chosen, and spread of the refit over its scenes "
          "(theta wrapped to (-180, 180]); objective of the other handedness for contrast")
    for lg in sorted({r["log"] for r in out.values()}):
        rs = [r["refit"] for r in out.values() if r["log"] == lg]
        th = [((x["theta"] + 180) % 360) - 180 for x in rs]
        rng_ = lambda v: max(v) - min(v)
        print(f"  {lg:26s} scenes {len(rs):2d}: mirror {collections.Counter(x['mirror'] for x in rs)}; theta "
              f"{min(th):6.2f}..{max(th):6.2f} deg, tx {min(x['tx'] for x in rs):6.2f}..{max(x['tx'] for x in rs):6.2f}, "
              f"ty {min(x['ty'] for x in rs):6.2f}..{max(x['ty'] for x in rs):6.2f} m; objective "
              f"{np.mean([x['objective'] for x in rs]):.3f} vs other handedness {np.mean([x['objective_other_handedness'] for x in rs]):.3f}")

    print()
    for split in ("train", "tuning", "test", None):
        sel = [s for s, r in out.items() if split is None or r["split"] == split]
        cnt = lambda k: sum(out[s][k]["ok"] for s in sel)
        print(f"{split or 'ALL':6s}: {len(sel):2d} scenes, pass old {cnt('old'):2d}, pass y-mirror {cnt('mirror'):2d}, "
              f"pass per-scene refit {cnt('new'):2d}")
    for k, lab in (("old", "old"), ("mirror", "y-mirror"), ("new", "refit")):
        allp = np.concatenate([[out[s][k]["median"]] for s in out])
        print(f"  {lab:8s}: scene medians {np.min(allp):.2f}..{np.max(allp):.2f} m, scene p95s "
              f"{min(out[s][k]['p95'] for s in out):.2f}..{max(out[s][k]['p95'] for s in out):.2f} m")
    excl = sorted(s for s, r in out.items() if not r["old"]["ok"] and not r["new"]["ok"])
    print(f"\nScenes failing BOTH the old transform and the per-scene refit (listed for exclusion): {excl}")
    for s in excl:
        r = out[s]
        print(f"  scene {s} ({r['town']}, {r['split']}, {r['log']}): refit {r['new']['median']:.2f}/{r['new']['p95']:.2f}, "
              f"y-mirror {r['mirror']['median']:.2f}/{r['mirror']['p95']:.2f}"
              + (f", CAVs under refit {r['new_cav']['median']:.2f}/{r['new_cav']['p95']:.2f}" if r["new_cav"] else ""))
    print("\nPart 1 impact: tuning / test scenes failing under the OLD per-town transform (used by D's map lookup):")
    for split in ("tuning", "test"):
        f_ = sorted(s for s, r in out.items() if r["split"] == split and not r["old"]["ok"])
        print(f"  {split}: {len(f_)} of {sum(r['split'] == split for r in out.values())} fail: {f_}")
        for s in f_:
            r = out[s]
            print(f"    scene {s} ({r['town']}): old {r['old']['median']:.2f}/{r['old']['p95']:.2f}"
                  + (f"; ego-type positions (CAVs) {r['old_cav']['median']:.2f}/{r['old_cav']['p95']:.2f}" if r["old_cav"] else ""))
    print(f"\nwrote {OUT_JSON}\nDONE")

if __name__ == "__main__":
    main()
