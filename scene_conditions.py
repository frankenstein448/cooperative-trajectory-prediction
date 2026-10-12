"""
scene_conditions.py
Ground-truth condition labels for eval_scenes.py's COND block (evaluation only,
runs/DECISION_RULES_CONDITIONS.md). Fixed before any condition number was seen:

  - stationary:    frame-1 -> frame displacement = 0 (motion_utils.get_cv_velocity, as before)
  - moving:        frame-1 -> frame displacement > 0
  - turning:       moving AND |heading change over the 30-step GT| > TURN_THRESHOLD_DEG.
                   The GT path is the current position (origin) followed by the valid GT
                   points in order; its steps are the displacements between successive
                   points. Steps shorter than MIN_STEP_M are ignored. Heading change =
                   direction of the LAST remaining step - direction of the FIRST, wrapped
                   to (-180, 180]. With fewer than 2 remaining steps the change is 0.
  - straight:      moving AND not turning
  - near_junction: the agent's current position (CARLA frame, coord_transform) is within
                   JUNCTION_RADIUS_M of the nearest in-junction lane-centre waypoint
                   (junction_waypoints_<town>.npy, export_junctions.py: CARLA
                   OpenDRIVE, wp.is_junction, 1 m spacing). Independent of motion.
"""
import math
import numpy as np
from scipy.spatial import cKDTree

TURN_THRESHOLD_DEG = 30.0
MIN_STEP_M = 0.5
JUNCTION_RADIUS_M = 20.0
CONDITIONS = ("turning", "straight", "stationary", "near_junction")  # the rule's conditions


def heading_change_deg(gt_xy, valid):
    """gt_xy: [T, 2] agent-centric GT, valid: [T] bool -> (signed heading change in degrees, steps used)."""
    pts = np.concatenate([np.zeros((1, 2)), np.asarray(gt_xy, dtype=np.float64)[np.asarray(valid, dtype=bool)]])
    steps = np.diff(pts, axis=0)
    steps = steps[np.linalg.norm(steps, axis=1) >= MIN_STEP_M]
    if len(steps) < 2:
        return 0.0, len(steps)
    a0 = math.atan2(steps[0, 1], steps[0, 0])
    a1 = math.atan2(steps[-1, 1], steps[-1, 0])
    d = math.degrees(a1 - a0)
    d = (d + 180.0) % 360.0 - 180.0  # -> [-180, 180)
    return (180.0 if d == -180.0 else d), len(steps)


class JunctionAreas:
    """One KD-tree per town over junction_waypoints_<town>.npy (CARLA x, y; column 2 = junction id)."""
    def __init__(self, towns):
        self.trees = {t: cKDTree(np.load(f"junction_waypoints_{t.lower()}.npy")[:, :2]) for t in towns}

    def distance(self, town, xy_carla):
        """xy_carla: [2] -> distance (m) to the nearest in-junction lane-centre waypoint."""
        d, _ = self.trees[town].query(np.asarray(xy_carla, dtype=np.float64).reshape(2))
        return float(d)
