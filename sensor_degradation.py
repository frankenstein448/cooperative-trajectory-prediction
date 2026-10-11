"""
sensor_degradation.py
Load-time sensor degradation for the "does the map help when sensors degrade?"
study. Applied by PilotDataset to EVERY agent's raw sensor tensors (ego and
each coop neighbor) right after loading -- never to cam/cpm (V2X messages),
nearby_idx (map) or gt.

Deterministic: all randomness (only cam_fog's noise) comes from a generator
seeded by the SENSOR's (agent_id, scene, frame), so the same camera at the same
frame gets the same noise whether it appears as ego or as someone's coop
neighbor, in any DataLoader worker, across train/eval and across modes
(fog's images are bit-identical to cam_fog's, fog's points to lidar_range's).

Modes (None = off):
  lidar_range  drop LiDAR points with horizontal range hypot(x, y) > LIDAR_RANGE_M.
               .pcd.bin points are in that agent's own sensor frame, so this is
               distance from that agent's LiDAR.
  cam_fog      image = FOG_CONTRAST * image + FOG_OFFSET + N(0, FOG_NOISE_STD), clamped to [0, 1]
  fog          lidar_range + cam_fog
  sensors_off  LiDAR points and images set to zero, shapes kept
"""
import zlib
import torch

DEGRADE_MODES = ("lidar_range", "cam_fog", "fog", "sensors_off")

LIDAR_RANGE_M = 20.0
FOG_CONTRAST = 0.4
FOG_OFFSET = 0.3
FOG_NOISE_STD = 0.05

# Empty-LiDAR fallback counter: every degrade_points call is one sensor (ego or
# coop); "fallback" counts those left with no point within range. Per process --
# DataLoader workers each keep their own copy, so read it only where loading
# happens in the main process (eval_scenes.py, verify_degrade.py).
_counts = {"sensors": 0, "fallback": 0}


def fallback_counts():
    return dict(_counts)


def reset_fallback_counts():
    _counts.update(sensors=0, fallback=0)


def sensor_seed(agent_id, scene, frame):
    """Stable across processes (unlike hash()), so every DataLoader worker agrees."""
    return zlib.crc32(f"{agent_id}_{scene}_{frame}".encode())


def degrade_points(points, mode):
    """points: [N, 5] in the sensor frame (x, y, z, ...)."""
    _counts["sensors"] += 1
    if mode in ("lidar_range", "fog"):
        keep = torch.hypot(points[:, 0], points[:, 1]) <= LIDAR_RANGE_M
        if not keep.any():
            _counts["fallback"] += 1
            # PointNet max-pools over points, so it needs >= 1; a single all-zero
            # point is what sensors_off would feed for this agent.
            return points.new_zeros(1, points.shape[1])
        return points[keep]
    if mode == "sensors_off":
        return torch.zeros_like(points)
    return points


def degrade_images(images, mode, agent_id, scene, frame):
    """images: list of [3, H, W] float tensors in [0, 1] (one per view)."""
    if mode in ("cam_fog", "fog"):
        g = torch.Generator().manual_seed(sensor_seed(agent_id, scene, frame))
        return [(FOG_CONTRAST * img + FOG_OFFSET
                 + FOG_NOISE_STD * torch.randn(img.shape, generator=g)).clamp_(0.0, 1.0)
                for img in images]
    if mode == "sensors_off":
        return [torch.zeros_like(img) for img in images]
    return images


def degrade_config(mode):
    """Everything that defines the degradation, for config.json."""
    if mode is None:
        return None
    return {"mode": mode, "lidar_range_m": LIDAR_RANGE_M, "fog_contrast": FOG_CONTRAST,
            "fog_offset": FOG_OFFSET, "fog_noise_std": FOG_NOISE_STD,
            "seed": "zlib.crc32(f'{agent_id}_{scene}_{frame}') of the sensor's agent"}
