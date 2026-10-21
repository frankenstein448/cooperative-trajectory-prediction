"""
coord_transform.py
V2X-Sim world coordinates -> CARLA world frame, two versions (map_transform below):
  mirror (default for new runs): CARLA = (x, -y), yaw_carla = -yaw_v2x.
  legacy (Part 1, Appendix G): per-town rotation + translation from the transform search, TOWN_TRANSFORMS
  (Town04 = coarse-search candidate A of refine_town04_theta.py; Town05 = the "refit" of score_transforms.py):

  Town03: theta=194.75, tx=-34.25, ty=6.00
  Town04: theta=260.50, tx=37.75,  ty=-41.25
  Town05: theta=85.60,  tx=-52.50, ty=-30.71

Earlier versions of this docstring listed intermediate candidates (Town04 theta=296.50, Town05 theta=70.5) with
their residuals; those values are not the legacy transform.
"""
import numpy as np

TOWN_TRANSFORMS = {
    "Town03": {"theta": 194.75, "tx": -34.25, "ty": 6.00},
    "Town04": {"theta": 260.50, "tx": 37.75, "ty": -41.25},
    "Town05": {"theta": 85.60,   "tx": -52.50, "ty": -30.71},
}


# map_transform (recorded in a run's config.json; absent = "legacy", so old runs reproduce exactly):
#   legacy: the per-town rotations above (TOWN_TRANSFORMS). check_map_alignment.py (2026-10-13) found
#           them wrong away from where they were fitted: no scene passes median <= 1.0 m & p95 <= 2.5 m.
#   mirror: CARLA = (x, -y) of V2X-Sim (CARLA is left-handed), heading yaw_carla = -yaw_v2x. All 54 used
#           scenes pass; lane yaw vs mirrored heading median 0.03-0.11 deg (check_map_alignment_mirror.log).
# New runs default to mirror (train.py --map-transform).
MAP_TRANSFORMS = ("legacy", "mirror")


def map_transform_of(run_config):
    """A run's map transform from its config.json dict; configs without the key are legacy."""
    kind = run_config.get("map_transform", "legacy")
    assert kind in MAP_TRANSFORMS, f"unknown map_transform {kind!r}"
    return kind


def v2xsim_to_carla(x, y, town, kind="legacy"):
    if kind == "mirror":
        return x, -y
    assert kind == "legacy", f"unknown map_transform {kind!r}"
    p = TOWN_TRANSFORMS[town]
    theta = np.radians(p["theta"])
    c, s = np.cos(theta), np.sin(theta)
    return x * c - y * s + p["tx"], x * s + y * c + p["ty"]


def v2xsim_to_carla_batch(pts, town, kind="legacy"):
    """pts: (N, 2) array of (x, y) in V2X-Sim world coordinates."""
    if kind == "mirror":
        return np.asarray(pts) * np.array([1.0, -1.0])
    assert kind == "legacy", f"unknown map_transform {kind!r}"
    p = TOWN_TRANSFORMS[town]
    theta = np.radians(p["theta"])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([p["tx"], p["ty"]])


def carla_to_v2xsim_batch(pts, town, kind="legacy"):
    """Inverse of v2xsim_to_carla_batch: (N, 2) CARLA world -> V2X-Sim world."""
    if kind == "mirror":
        return np.asarray(pts) * np.array([1.0, -1.0])
    assert kind == "legacy", f"unknown map_transform {kind!r}"
    p = TOWN_TRANSFORMS[town]
    theta = np.radians(p["theta"])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (np.asarray(pts) - np.array([p["tx"], p["ty"]])) @ R


def v2xsim_yaw_to_carla(yaw, town, kind="legacy"):
    """Heading (radians) in V2X-Sim -> CARLA: mirror -> -yaw; legacy -> yaw + theta (the rotation's own rule)."""
    if kind == "mirror":
        return -np.asarray(yaw)
    assert kind == "legacy", f"unknown map_transform {kind!r}"
    return np.asarray(yaw) + np.radians(TOWN_TRANSFORMS[town]["theta"])


def carla_yaw_to_v2xsim(yaw, town, kind="legacy"):
    """Inverse of v2xsim_yaw_to_carla, e.g. a CARLA lane direction (radians) -> V2X-Sim."""
    if kind == "mirror":
        return -np.asarray(yaw)
    assert kind == "legacy", f"unknown map_transform {kind!r}"
    return np.asarray(yaw) - np.radians(TOWN_TRANSFORMS[town]["theta"])


def warn_legacy_script(script):
    """Printed by scripts that predate map_transform / split_version and always use legacy + v1."""
    bar = "!" * 100
    print(f"{bar}\n!! {script}: LEGACY map transform (coord_transform.TOWN_TRANSFORMS, wrong away from its fit area) "
          f"/ v1 split (leaky,\n!! scene_split.py). Not for new results; new runs use train.py / eval_scenes.py "
          f"(map_transform mirror, split_version v2).\n{bar}", flush=True)
