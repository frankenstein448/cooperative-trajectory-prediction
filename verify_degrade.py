"""
verify_degrade.py
For 50 random dataset indices, under run D's pipeline (runs/D_seed0 config +
cam_norm), compares each --degrade mode against the clean sample. Asserts:
  - every non-sensor field (cam = V2X CAM, cpm_objects, nearby_idx = map, gt, ids)
    is identical to clean, and the coop neighbor count is unchanged;
  - only points / images / coop_points / coop_images change, and each changes as
    specified (ego AND coop):
      lidar_range: points == clean points with hypot(x,y) <= 20 (images untouched)
      cam_fog:     images in [0,1], residual vs 0.4*clean+0.3 has mean ~0, std ~0.05 (points untouched)
      fog:         points == lidar_range's, images == cam_fog's (bit-identical)
      sensors_off: all sensor tensors zero, shapes unchanged
  - two loads of the same index give identical tensors (deterministic).
Also prints the clean LiDAR horizontal-range distribution (backs the assumption
that .pcd.bin points are in the agent's own sensor frame, and shows how much the
20 m cut removes) and, per mode, how many sensors hit the empty-LiDAR fallback. Read-only, no training. A single PilotDataset
instance is used, toggling dataset.degrade between calls.
"""
import json
import random
import torch

from pilot_dataset import PilotDataset
from sensor_degradation import (DEGRADE_MODES, LIDAR_RANGE_M, FOG_CONTRAST, FOG_OFFSET, FOG_NOISE_STD,
                                fallback_counts, reset_fallback_counts)
from coord_transform import warn_legacy_script
warn_legacy_script("verify_degrade.py")

random.seed(0)
N = 50
RUN_DIR = "runs/D_seed0"
SENSOR_KEYS = ("points", "images", "coop_points", "coop_images")

with open(f"{RUN_DIR}/config.json") as f:
    config = json.load(f)
with open(f"{RUN_DIR}/cam_norm.json") as f:
    cam_norm = json.load(f)

dataset = PilotDataset(use_image_cache=config["use_image_cache"],
                        require_prev_frame=config["require_prev_frame"],
                        use_speed=config["use_speed"], cam_norm=cam_norm,
                        drop_cam_xy=config["drop_cam_xy"])
idxs = random.sample(range(len(dataset)), N)


def same(a, b):
    """Deep equality over tensors / lists / scalars."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def load_all(mode):
    dataset.degrade = mode
    return [dataset[i] for i in idxs]


def agents(item):
    """[(points, images)] for the ego followed by every coop neighbor."""
    return [(item["points"], item["images"])] + list(zip(item["coop_points"], item["coop_images"]))


def in_range(points):
    kept = points[torch.hypot(points[:, 0], points[:, 1]) <= LIDAR_RANGE_M]
    return kept if len(kept) else points.new_zeros(1, points.shape[1])


clean = load_all(None)
items, fallbacks = {}, {}
for mode in DEGRADE_MODES:
    reset_fallback_counts()
    items[mode] = load_all(mode)
    fallbacks[mode] = fallback_counts()

n_coop = sum(len(c["coop_points"]) for c in clean)
n_sensors = N + n_coop
assert n_coop > 0, "no sample in the 50 has a coop neighbor -- coop degradation is untested"
for label, pts in (("ego", [c["points"] for c in clean]),
                   ("ego + coop", [p for c in clean for p, _ in agents(c)])):
    r = torch.cat([torch.hypot(p[:, 0], p[:, 1]) for p in pts])
    q = torch.quantile(r[torch.randperm(len(r), generator=torch.Generator().manual_seed(0))[:1_000_000]],
                       torch.tensor([0.1, 0.5, 0.9]))  # quantile() caps input size; 1M-point subsample
    print(f"Clean LiDAR horizontal range hypot(x,y), {label} ({len(pts)} sensors, {len(r)} points): "
          f"p10 {q[0]:.1f} m, median {q[1]:.1f} m, p90 {q[2]:.1f} m, max {r.max():.1f} m; "
          f"{(r > LIDAR_RANGE_M).float().mean():.1%} of points beyond {LIDAR_RANGE_M:g} m")
print(f"{n_coop} coop neighbors across the {N} samples ({n_sensors} sensors)")
for mode in DEGRADE_MODES:
    fb = fallbacks[mode]
    assert fb["sensors"] == n_sensors, f"{mode}: counted {fb['sensors']} sensors, expected {n_sensors}"
    print(f"Empty-LiDAR fallback, {mode:11s}: {fb['fallback']}/{fb['sensors']} sensors (ego + coop)")

for mode, degraded in items.items():
    n_points_dropped, residuals = 0, []
    for idx, c, d in zip(idxs, clean, degraded):
        assert c.keys() == d.keys(), f"{mode} idx {idx}: returned keys differ"
        for k in c:
            if k not in SENSOR_KEYS:
                assert same(c[k], d[k]), f"{mode} idx {idx}: non-sensor field {k} differs from clean"
        assert len(c["coop_points"]) == len(d["coop_points"]) == len(d["coop_images"]), \
            f"{mode} idx {idx}: coop neighbor count changed"

        for (cp, ci), (dp, di) in zip(agents(c), agents(d)):
            if mode in ("lidar_range", "fog"):
                assert torch.equal(dp, in_range(cp)), f"{mode} idx {idx}: points != clean points within range"
                n_points_dropped += len(cp) - len(dp)
            elif mode == "cam_fog":
                assert torch.equal(dp, cp), f"{mode} idx {idx}: points changed"
            if mode in ("cam_fog", "fog"):
                for cv, dv in zip(ci, di):
                    assert dv.shape == cv.shape and dv.min() >= 0 and dv.max() <= 1, f"{mode} idx {idx}: bad image"
                    expected = FOG_CONTRAST * cv + FOG_OFFSET
                    unclamped = (dv > 0) & (dv < 1)
                    residuals.append((dv - expected)[unclamped])
            elif mode == "lidar_range":
                assert same(di, ci), f"{mode} idx {idx}: images changed"
            if mode == "sensors_off":
                assert dp.shape == cp.shape and not dp.any(), f"{mode} idx {idx}: points not zero / reshaped"
                assert all(dv.shape == cv.shape and not dv.any() for cv, dv in zip(ci, di)), \
                    f"{mode} idx {idx}: images not zero / reshaped"

    if mode == "fog":
        for idx, f_, lr, cf in zip(idxs, degraded, items["lidar_range"], items["cam_fog"]):
            assert same(f_["points"], lr["points"]) and same(f_["coop_points"], lr["coop_points"]), \
                f"fog idx {idx}: points differ from lidar_range"
            assert same(f_["images"], cf["images"]) and same(f_["coop_images"], cf["coop_images"]), \
                f"fog idx {idx}: images differ from cam_fog"

    msg = f"OK {mode:11s}: only sensor tensors differ from clean in {N}/{N} samples (ego + {n_coop} coop)"
    if mode in ("lidar_range", "fog"):
        assert n_points_dropped > 0, f"{mode}: no LiDAR point was beyond {LIDAR_RANGE_M} m in any sample"
        msg += f"; {n_points_dropped} LiDAR points dropped"
    if mode in ("cam_fog", "fog"):
        r = torch.cat(residuals)
        assert abs(r.mean()) < 0.01 and abs(r.std() - FOG_NOISE_STD) < 0.01, \
            f"{mode}: residual mean {r.mean():.4f} std {r.std():.4f}, expected ~0 / ~{FOG_NOISE_STD}"
        msg += f"; image residual mean {r.mean():+.4f} std {r.std():.4f} (unclamped pixels)"
    print(msg)

for mode in DEGRADE_MODES:
    dataset.degrade = mode
    for i in idxs[:10]:
        a, b = dataset[i], dataset[i]
        assert all(same(a[k], b[k]) for k in SENSOR_KEYS), f"{mode} idx {i}: two loads differ"
print("OK deterministic: two loads of 10 indices are identical under every mode")
