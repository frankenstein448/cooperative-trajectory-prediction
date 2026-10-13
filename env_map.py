"""
env_map.py
Map v2 tokens for the env task (runs/DESIGN_ENV.md): lane polyline pieces near a target, in the TARGET's
own frame, with no absolute coordinates.

Source: lane_polylines_<town>.npz (export_lane_polylines.py: CARLA generate_waypoints(2.0) driving lanes, plus
Town03's Parking lanes; points grouped into pieces of <= MAX_PIECE_POINTS points in travel order). The lane
points and directions are taken from CARLA back to V2X-Sim with the run's map transform
(coord_transform.carla_to_v2xsim_batch / carla_yaw_to_v2xsim; mirror: (x, -y), yaw -> -yaw), so they live in
the same frame as the env cache's target states (V2X-Sim world, box yaw). A target's tokens are expressed
in its own frame: origin at its current position, x along its box yaw -- the frame of fut_own / hist_own.

Per target: every piece with at least one point within RADIUS_M of the target, nearest first (by the
piece's nearest point), at most K_PIECES. Per point, N_POINT_FEATS = 6 features:
  x / RADIUS_M, y / RADIUS_M        position in the target's frame
  cos(d), sin(d)                    d = lane direction at the point (waypoint yaw) - target yaw
  is_junction, is_parking           lane flags (0 / 1)
Returned arrays: feats [K_PIECES, MAX_PIECE_POINTS, 6] float32, point_mask [K_PIECES, MAX_PIECE_POINTS] bool,
piece_mask [K_PIECES] bool (True = real). Padding is zero and masked.
"""
import math

import numpy as np
from scipy.spatial import cKDTree

from coord_transform import carla_to_v2xsim_batch, carla_yaw_to_v2xsim

RADIUS_M = 30.0
K_PIECES = 48
MAX_PIECE_POINTS = 11     # export_lane_polylines.py: pieces of 10, a short tail merged into the previous one
N_POINT_FEATS = 6


def piece_tokens(pts, yaw, is_junction, is_parking, piece_points, piece_of_point, tree, xy, heading,
                 radius=RADIUS_M, k_pieces=K_PIECES, max_points=MAX_PIECE_POINTS):
    """Core, frame-agnostic: lane points pts [N, 2] and directions yaw [N] (radians) in the SAME world frame
    as the target position xy [2] and heading (radians). piece_points: [P, max_points] point indices, -1 =
    padding; piece_of_point: [N]; tree: cKDTree(pts). Returns (feats, point_mask, piece_mask)."""
    feats = np.zeros((k_pieces, max_points, N_POINT_FEATS), dtype=np.float32)
    point_mask = np.zeros((k_pieces, max_points), dtype=bool)
    piece_mask = np.zeros(k_pieces, dtype=bool)
    near = np.asarray(tree.query_ball_point(np.asarray(xy, dtype=np.float64), radius), dtype=np.int64)
    if len(near) == 0:
        return feats, point_mask, piece_mask
    d = np.hypot(pts[near, 0] - xy[0], pts[near, 1] - xy[1])
    pieces = piece_of_point[near]
    best = {}
    for p, dist in zip(pieces.tolist(), d.tolist()):
        if dist < best.get(p, math.inf):
            best[p] = dist
    chosen = sorted(best, key=lambda p: (best[p], p))[:k_pieces]
    c, s = math.cos(-heading), math.sin(-heading)
    for slot, p in enumerate(chosen):
        idx = piece_points[p]
        idx = idx[idx >= 0]
        dx, dy = pts[idx, 0] - xy[0], pts[idx, 1] - xy[1]
        rel = yaw[idx] - heading
        n = len(idx)
        feats[slot, :n, 0] = (dx * c - dy * s) / radius
        feats[slot, :n, 1] = (dx * s + dy * c) / radius
        feats[slot, :n, 2] = np.cos(rel)
        feats[slot, :n, 3] = np.sin(rel)
        feats[slot, :n, 4] = is_junction[idx]
        feats[slot, :n, 5] = is_parking[idx]
        point_mask[slot, :n] = True
        piece_mask[slot] = True
    return feats, point_mask, piece_mask


class LanePolylines:
    """One town's lane pieces, converted to V2X-Sim coordinates with map transform `kind`."""
    def __init__(self, town, kind="mirror", data=None, max_points=MAX_PIECE_POINTS):
        z = data if data is not None else np.load(f"lane_polylines_{town.lower()}.npz")
        self.town, self.kind, self.max_points = town, kind, max_points
        self.pts = carla_to_v2xsim_batch(np.asarray(z["points"], dtype=np.float64), town, kind)
        self.yaw = carla_yaw_to_v2xsim(np.radians(np.asarray(z["yaw_deg"], dtype=np.float64)), town, kind)
        self.is_junction = np.asarray(z["is_junction"], dtype=np.float32)
        self.is_parking = (np.asarray(z["lane_type"]) == "Parking").astype(np.float32)
        order, ptr = np.asarray(z["order"]), np.asarray(z["piece_ptr"])
        n_pieces = len(ptr) - 1
        assert np.diff(ptr).max() <= max_points, f"{town}: a piece has more than {max_points} points"
        self.piece_points = np.full((n_pieces, max_points), -1, dtype=np.int64)
        self.piece_of_point = np.full(len(self.pts), -1, dtype=np.int64)
        for p in range(n_pieces):
            idx = order[ptr[p]:ptr[p + 1]]
            self.piece_points[p, :len(idx)] = idx
            self.piece_of_point[idx] = p
        assert (self.piece_of_point >= 0).all(), f"{town}: a lane point belongs to no piece"
        self.tree = cKDTree(self.pts)

    def tokens(self, xy, heading, radius=RADIUS_M, k_pieces=K_PIECES):
        """Target at xy [2] (V2X-Sim world) with box heading (radians) -> (feats, point_mask, piece_mask)."""
        return piece_tokens(self.pts, self.yaw, self.is_junction, self.is_parking, self.piece_points,
                            self.piece_of_point, self.tree, xy, heading, radius, k_pieces, self.max_points)
