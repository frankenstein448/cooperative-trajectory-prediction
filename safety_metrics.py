"""
safety_metrics.py
Geometry for eval_scenes.py's safety metrics (evaluation only, nothing here is trained on):

  - miss rate: FDE > MISS_THRESHOLD_M (top-1, and best-of-6), computed in eval_scenes.py
  - off-road rate: fraction of trajectory points farther than OFFROAD_THRESHOLD_M from
    the nearest lane-centreline point of that town.

Lane centrelines are the dense CARLA waypoints (dense_waypoints_town0X.npy, built by
dense_waypoints.py with carla_map.generate_waypoints(distance=2.0): lane centres,
every lane, ~2 m apart). Distance is to the nearest waypoint, not to the polyline
between them, so it overestimates the true centreline distance by at most ~1 m
(half the spacing) for a point lying on a lane. The ground-truth control in
eval_scenes.py measures how much this, and the V2X-Sim -> CARLA alignment
(coord_transform.py, median residuals ~0.1-1 m), inflate the rate.

Trajectories are agent-centric (ground_truth_extractor convention: offset from the
current position, rotated by -heading); they are taken back to V2X-Sim world
coordinates with the agent's RAW pose (never the normalized cam), then into CARLA
with coord_transform's map transform (`kind`: legacy = TOWN_TRANSFORMS, mirror = (x, -y)).
"""
import math
import numpy as np
from scipy.spatial import cKDTree

from coord_transform import v2xsim_to_carla_batch
from v2x_message_stream import quaternion_to_yaw

MISS_THRESHOLD_M = 2.0
OFFROAD_THRESHOLD_M = 2.5      # fixed before any off-road number was seen
GT_OFFROAD_UNRELIABLE = 0.05   # a town whose GT off-road rate exceeds this gets its off-road numbers flagged


class LaneCentrelines:
    """One KD-tree per town over dense_waypoints_<town>.npy (CARLA x, y)."""
    def __init__(self, towns):
        self.trees = {t: cKDTree(np.load(f"dense_waypoints_{t.lower()}.npy")) for t in towns}

    def offroad(self, town, pts_carla):
        """pts_carla: [..., 2] -> bool [...] (True = farther than OFFROAD_THRESHOLD_M from every waypoint)."""
        d, _ = self.trees[town].query(pts_carla.reshape(-1, 2))
        return (d > OFFROAD_THRESHOLD_M).reshape(pts_carla.shape[:-1])


def raw_pose(loader, agent_id, scene, frame):
    """(x, y, heading) at `frame` in V2X-Sim world coordinates, from ego_pose."""
    pose = loader.ego_pose[loader.find_sample_data(agent_id, scene, frame)["ego_pose_token"]]
    x, y, _ = pose["translation"]
    return x, y, quaternion_to_yaw(pose["rotation"])


def agent_to_carla(local_xy, pose, town, kind="legacy"):
    """local_xy: [..., 2] agent-centric (rotated by -heading) -> [..., 2] CARLA world.
    Undoes ground_truth_extractor's rotation (rotate by +heading), adds the current
    position, then applies coord_transform's map transform `kind` (legacy = the per-town rigid transform)."""
    x, y, h = pose
    c, s = math.cos(h), math.sin(h)
    lx, ly = local_xy[..., 0], local_xy[..., 1]
    world = np.stack([x + lx * c - ly * s, y + lx * s + ly * c], axis=-1)
    return v2xsim_to_carla_batch(world.reshape(-1, 2), town, kind).reshape(world.shape)
