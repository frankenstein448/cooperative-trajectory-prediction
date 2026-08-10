"""
Coarse-to-fine grid search for the rigid transform (rotation + translation)
that aligns V2X-Sim Town05 agent positions onto the real CARLA lane network.

Two-stage pipeline:
  1. extract_agent_points()  -- pull ALL agent world positions for given
     Town05 scenes from ego_pose.json (generalizes the single-point trace
     you already validated).
  2. grid_search_transform() -- coarse global search scored against a
     KD-tree of the dense CARLA waypoint cloud, then local refinement.

Run in carla-venv (needs numpy, scipy). No CARLA server connection required
for the search itself -- only for producing dense_waypoints.py's output,
which you already have.
"""

import json
import glob
import re
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# 1. Extract all agent positions for one or more Town05 scenes
# ---------------------------------------------------------------------------

def extract_agent_points(v2x_sim_root, scene_ids):
    """
    Walk sample_data.json for the given scene numbers, resolve each
    ego_pose_token -> ego_pose.json translation, return an (N,2) array
    of world-frame (x, y) agent positions.

    scene_ids: list of int scene numbers to include (more scenes = better
    shape signal for the grid search).
    """
    with open(f"{v2x_sim_root}/v2.0/sample_data.json") as f:
        sample_data = json.load(f)
    with open(f"{v2x_sim_root}/v2.0/ego_pose.json") as f:
        ego_pose = json.load(f)
    ego_pose_by_token = {e["token"]: e for e in ego_pose}

    pattern = re.compile(r"sweeps/LIDAR_TOP_id_(\d+)/scene_(\d+)_(\d+)\.pcd\.bin")

    points = []
    for sd in sample_data:
        m = pattern.search(sd.get("filename", ""))
        if not m:
            continue
        scene_num = int(m.group(2))
        if scene_num not in scene_ids:
            continue
        pose = ego_pose_by_token.get(sd["ego_pose_token"])
        if pose is None:
            continue
        x, y, _z = pose["translation"]
        points.append((x, y))

    pts = np.array(points, dtype=np.float64)
    print(f"Extracted {len(pts)} agent points from scenes {scene_ids}")
    return pts


# ---------------------------------------------------------------------------
# 2. Coarse-to-fine grid search against the CARLA lane point cloud
# ---------------------------------------------------------------------------

def apply_transform(pts, theta_deg, tx, ty):
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([tx, ty])


def score_transform(agent_pts, tree, theta_deg, tx, ty):
    transformed = apply_transform(agent_pts, theta_deg, tx, ty)
    dists, _ = tree.query(transformed)
    return np.median(dists)  # robust to off-centerline agents


def grid_search(agent_pts, carla_pts,
                 theta_range=(0, 360, 5),
                 t_center=(0, 0), t_half_width=40, t_step=2):
    """
    theta_range: (start, stop, step) in degrees.
    t_center: (tx, ty) guess to center the translation search on -- use
        your known ~18m offset direction here if you have a rough idea,
        otherwise (0, 0) and widen t_half_width.
    t_half_width / t_step: translation search window and step (meters).
    """
    tree = cKDTree(carla_pts)

    thetas = np.arange(*theta_range)
    txs = np.arange(t_center[0] - t_half_width, t_center[0] + t_half_width + 1, t_step)
    tys = np.arange(t_center[1] - t_half_width, t_center[1] + t_half_width + 1, t_step)

    best = None
    total = len(thetas) * len(txs) * len(tys)
    print(f"Coarse search: {total} candidates")
    count = 0
    for theta in thetas:
        for tx in txs:
            for ty in tys:
                count += 1
                s = score_transform(agent_pts, tree, theta, tx, ty)
                if best is None or s < best[0]:
                    best = (s, theta, tx, ty)
        if theta % 30 == 0:
            print(f"  ...{count}/{total}, best so far: {best}")

    print(f"Coarse best: score={best[0]:.2f}m theta={best[1]}° tx={best[2]} ty={best[3]}")
    return best


def refine(agent_pts, carla_pts, coarse_best,
           theta_half_width=5, theta_step=0.5,
           t_half_width=3, t_step=0.25):
    tree = cKDTree(carla_pts)
    _, theta0, tx0, ty0 = coarse_best

    thetas = np.arange(theta0 - theta_half_width, theta0 + theta_half_width + 0.01, theta_step)
    txs = np.arange(tx0 - t_half_width, tx0 + t_half_width + 0.01, t_step)
    tys = np.arange(ty0 - t_half_width, ty0 + t_half_width + 0.01, t_step)

    best = None
    for theta in thetas:
        for tx in txs:
            for ty in tys:
                s = score_transform(agent_pts, tree, theta, tx, ty)
                if best is None or s < best[0]:
                    best = (s, theta, tx, ty)

    print(f"Refined best: score={best[0]:.3f}m theta={best[1]}° tx={best[2]:.2f} ty={best[3]:.2f}")
    return best


# ---------------------------------------------------------------------------
# 3. Optional: ICP polish seeded from the refined transform
# ---------------------------------------------------------------------------

def icp_polish(agent_pts, carla_pts, seed, max_iters=50, tol=1e-4):
    """Standard ICP: alternate nearest-neighbor correspondence + Kabsch fit,
    seeded from a transform that's already close (from grid search)."""
    tree = cKDTree(carla_pts)
    _, theta0, tx0, ty0 = seed
    current = apply_transform(agent_pts, theta0, tx0, ty0)

    R_total = np.array([[np.cos(np.radians(theta0)), -np.sin(np.radians(theta0))],
                         [np.sin(np.radians(theta0)),  np.cos(np.radians(theta0))]])
    t_total = np.array([tx0, ty0])

    prev_score = np.inf
    for i in range(max_iters):
        dists, idx = tree.query(current)
        matched = carla_pts[idx]

        # Kabsch on current correspondences
        centroid_a = current.mean(axis=0)
        centroid_b = matched.mean(axis=0)
        H = (current - centroid_a).T @ (matched - centroid_b)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_b - R @ centroid_a

        current = current @ R.T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        score = np.median(dists)
        if abs(prev_score - score) < tol:
            break
        prev_score = score

    theta_final = np.degrees(np.arctan2(R_total[1, 0], R_total[0, 0]))
    print(f"ICP final: score={score:.3f}m theta={theta_final:.2f}° t={t_total}")
    return score, theta_final, t_total[0], t_total[1]


if __name__ == "__main__":
    V2X_SIM_ROOT = "/home/karra/V2X-Sim-2.0"
    CARLA_POINTS_NPY = "dense_waypoints_town05.npy" # from your dense_waypoints.py output

    # Adjust scene_ids to whichever Town05 scenes you have (more = better).
    agent_pts = extract_agent_points(V2X_SIM_ROOT, scene_ids=[77])
    carla_pts = np.load(CARLA_POINTS_NPY)

    coarse_best = grid_search(agent_pts, carla_pts, t_center=(0, 0), t_half_width=40, t_step=2)
    refined = refine(agent_pts, carla_pts, coarse_best)
    final = icp_polish(agent_pts, carla_pts, refined)