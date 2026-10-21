"""
verify_map_transform.py
Checks for the versioned map transform (coord_transform.MAP_TRANSFORMS = legacy / mirror), before any run:

1. Legacy is unchanged: coord_transform's v2xsim_to_carla / _batch (default kind and kind="legacy") and
   safety_metrics.agent_to_carla equal, bit for bit, the functions at the last commit before the option on 10,000
   random points per town; PilotDataset.__getitem__'s nearby_idx (default = legacy) equals that commit's formula on
   50 samples (D's sample filter), and with map_transform="mirror" equals the mirror formula.
2. Configs: runs/D_seed0/config.json (no key) -> legacy; {"map_transform": "mirror"} -> mirror; train.py's
   --map-transform default is mirror and is written to config.json.
3. Mirror: positions and headings round-trip exactly (carla_to_v2xsim_batch, carla_yaw_to_v2xsim); legacy
   round-trips to ~1e-9.
4. Headings through the API on the env target cache (moving targets): lane yaw at the nearest lane point
   (lane_polylines_<town>.npz) vs v2xsim_yaw_to_carla(box yaw), and the lane direction taken back with
   carla_yaw_to_v2xsim vs the box yaw; median |difference| should be small for mirror (and is not for legacy).

Run in gpu-test-venv:  python verify_map_transform.py
"""
import re
import json
import subprocess

import numpy as np
import torch
from scipy.spatial import cKDTree

import coord_transform as ct
import paths
import safety_metrics
from rgcn_map_encoder import get_nearby_lane_nodes

TOWNS = ("Town03", "Town04", "Town05")
fails = []


def check(ok, msg):
    print(("  OK   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


def old_module(rev, path):
    g = {}
    exec(subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True, check=True).stdout, g)
    return g


def last_rev_without(path, marker):
    """Latest commit whose version of `path` does not contain `marker` (i.e. before the option)."""
    revs = subprocess.run(["git", "log", "--format=%H", "--", path], capture_output=True, text=True).stdout.split()
    return next(r for r in ["HEAD"] + revs if marker not in subprocess.run(
        ["git", "show", f"{r}:{path}"], capture_output=True, text=True).stdout)


BASE = last_rev_without("coord_transform.py", "MAP_TRANSFORMS")
BASE_SM = last_rev_without("safety_metrics.py", 'kind="legacy"')
print(f"legacy reference: coord_transform.py at {BASE[:7]}, safety_metrics.py at {BASE_SM[:7]}")
old_ct = old_module(BASE, "coord_transform.py")
old_sm_src = subprocess.run(["git", "show", f"{BASE_SM}:safety_metrics.py"], capture_output=True, text=True).stdout

print("\n1. Legacy unchanged")
rng = np.random.default_rng(0)
for t in TOWNS:
    pts = rng.uniform(-400, 400, size=(10000, 2))
    new_b = ct.v2xsim_to_carla_batch(pts, t)
    new_bl = ct.v2xsim_to_carla_batch(pts, t, "legacy")
    old_b = old_ct["v2xsim_to_carla_batch"](pts, t)
    new_s = np.array([ct.v2xsim_to_carla(x, y, t) for x, y in pts[:2000]])
    old_s = np.array([old_ct["v2xsim_to_carla"](x, y, t) for x, y in pts[:2000]])
    check(np.array_equal(new_b, old_b) and np.array_equal(new_bl, old_b) and np.array_equal(new_s, old_s),
          f"{t}: v2xsim_to_carla(_batch) legacy/default bit-identical to {BASE[:7]}")
    pose = (12.3, -45.6, 0.7)
    local = rng.normal(0, 20, size=(6, 30, 2))
    g = {"OLD_BATCH": old_ct["v2xsim_to_carla_batch"]}
    exec(old_sm_src.replace("from coord_transform import v2xsim_to_carla_batch",
                            "v2xsim_to_carla_batch = OLD_BATCH"), g)
    check(np.array_equal(safety_metrics.agent_to_carla(local, pose, t), g["agent_to_carla"](local, pose, t)),
          f"{t}: safety_metrics.agent_to_carla default bit-identical to {BASE[:7]}")

from pilot_dataset import PilotDataset  # noqa: E402  (loads the metadata once)
ds = PilotDataset(use_image_cache=True, require_prev_frame=True, use_speed=True, use_cpm=False, use_coop=False)
check(ds.map_transform == "legacy", "PilotDataset default map_transform is legacy")
idx = np.linspace(0, len(ds) - 1, 50).astype(int)
ok_leg = ok_mir = 0
n_diff = 0
for i in idx:
    agent_id, scene, frame, town = ds.samples[i]
    g = ds.town_graphs[town]
    cam = ds.get_cam_features(agent_id, scene, frame)
    exp = {}
    for kind, fn in (("legacy", lambda x, y: old_ct["v2xsim_to_carla"](x, y, town)), ("mirror", lambda x, y: (x, -y))):
        cx, cy = fn(cam[0].item(), cam[1].item())
        pos = (torch.tensor([[cx, cy]], dtype=torch.float32) - g["mean"]) / g["std"]
        exp[kind] = get_nearby_lane_nodes(pos, g["data"].x, k=5)
    ds.map_transform = "legacy"
    got_l = ds[i]["nearby_idx"]
    ds.map_transform = "mirror"
    got_m = ds[i]["nearby_idx"]
    ok_leg += torch.equal(got_l, exp["legacy"])
    ok_mir += torch.equal(got_m, exp["mirror"])
    n_diff += not torch.equal(got_l, got_m)
ds.map_transform = "legacy"
check(ok_leg == len(idx), f"PilotDataset nearby_idx, legacy == {BASE[:7]} formula: {ok_leg}/{len(idx)}")
check(ok_mir == len(idx), f"PilotDataset nearby_idx, mirror == mirror formula: {ok_mir}/{len(idx)}")
print(f"  (legacy and mirror nearby nodes differ on {n_diff}/{len(idx)} samples)")

print("\n2. Configs")
with open("runs/D_seed0/config.json") as f:
    dcfg = json.load(f)
check("map_transform" not in dcfg and ct.map_transform_of(dcfg) == "legacy", "runs/D_seed0 config (no key) -> legacy")
check(ct.map_transform_of({"map_transform": "mirror"}) == "mirror", "{'map_transform': 'mirror'} -> mirror")
src = open("train.py").read()
check(re.search(r'"--map-transform", choices=MAP_TRANSFORMS, default="mirror"', src) is not None
      and '"map_transform": args.map_transform' in src and "map_transform=args.map_transform" in src,
      "train.py: --map-transform default mirror, written to config.json, passed to PilotDataset")
src = open("eval_scenes.py").read()
check("map_transform = map_transform_of(run_config)" in src and src.count("pose, town, map_transform)") == 3
      and "map_transform=map_transform)" in src, "eval_scenes.py: map_transform from config -> dataset, off-road, junctions")

print("\n3. Round trips")
pts = rng.uniform(-400, 400, size=(10000, 2))
yaw = rng.uniform(-np.pi, np.pi, size=10000)
for t in TOWNS:
    m = ct.carla_to_v2xsim_batch(ct.v2xsim_to_carla_batch(pts, t, "mirror"), t, "mirror")
    l = ct.carla_to_v2xsim_batch(ct.v2xsim_to_carla_batch(pts, t, "legacy"), t, "legacy")
    ym = ct.carla_yaw_to_v2xsim(ct.v2xsim_yaw_to_carla(yaw, t, "mirror"), t, "mirror")
    check(np.array_equal(m, pts) and np.array_equal(ym, yaw) and np.abs(l - pts).max() < 1e-9,
          f"{t}: mirror position/yaw round trip exact; legacy max error {np.abs(l - pts).max():.1e}")

print("\n4. Lane heading vs mirrored target heading (env cache, moving targets)")
z = np.load(paths.ENV_TARGETS_NPZ)
town_t = z["sample_town"][z["sample"]]
mov = z["speed"] > 0
wrapd = lambda a: np.abs((np.degrees(a) + 180) % 360 - 180)
for t in TOWNS:
    lp = np.load(f"lane_polylines_{t.lower()}.npz")
    tree, lyaw = cKDTree(lp["points"]), np.radians(lp["yaw_deg"])
    sel = (town_t == t) & mov
    xy, yw = z["xy_world"][sel], z["yaw_world"][sel].astype(np.float64)
    for kind in ("mirror", "legacy"):
        d, j = tree.query(ct.v2xsim_to_carla_batch(xy, t, kind))
        fwd = wrapd(lyaw[j] - ct.v2xsim_yaw_to_carla(yw, t, kind))
        back = wrapd(ct.carla_yaw_to_v2xsim(lyaw[j], t, kind) - yw)
        line = (f"{t} {kind:6s} n={sel.sum():6d}: distance median {np.median(d):.2f} m; heading |d| median "
                f"{np.median(fwd):.2f} deg (lane->v2x back {np.median(back):.2f}), > 90 deg {np.mean(fwd > 90):.3f}")
        if kind == "mirror":
            check(np.median(fwd) < 1.0 and np.allclose(fwd, back, atol=1e-6) and np.median(d) <= 1.0, line)
        else:
            print("  info " + line)

print("\nRESULT: " + ("PASS" if not fails else f"FAIL {fails}"))
