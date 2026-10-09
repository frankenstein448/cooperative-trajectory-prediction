"""
refine_town04_theta.py
Finer theta search around the two ambiguous candidates from the coarse search:
  A: theta=260.5, tx=37.75, ty=-41.25 (1.25m median)
  B: theta=295.0, tx=-38.00, ty=79.00 (1.11m median)
The coarse search only tested theta at 5-degree steps, too coarse to distinguish
a real optimum from a scoring artifact caused by Town04's parallel lanes.
"""
import numpy as np
from scipy.spatial import cKDTree
from grid_search_transform import extract_agent_points, apply_transform

V2X_SIM_ROOT = "/home/karra/V2X-Sim-2.0"
CARLA_POINTS_NPY = "dense_waypoints_town04.npy"
SCENE_IDS = [6, 9, 20, 26, 30, 92, 96, 97, 98]


def fine_search(agent_pts, tree, theta_center, t_center,
                 theta_half_width=15, theta_step=0.25,
                 t_half_width=10, t_step=1):
    best = (np.inf, None, None, None)
    thetas = np.arange(theta_center - theta_half_width, theta_center + theta_half_width + theta_step, theta_step)
    txs = np.arange(t_center[0] - t_half_width, t_center[0] + t_half_width + t_step, t_step)
    tys = np.arange(t_center[1] - t_half_width, t_center[1] + t_half_width + t_step, t_step)
    total = len(thetas) * len(txs) * len(tys)
    count = 0
    for theta in thetas:
        for tx in txs:
            for ty in tys:
                count += 1
                transformed = apply_transform(agent_pts, theta, tx, ty)
                dists, _ = tree.query(transformed)
                score = np.median(dists)
                if score < best[0]:
                    best = (score, theta, tx, ty)
                if count % 20000 == 0:
                    print(f"  ...{count}/{total}, best so far: {best}")
    return best


if __name__ == "__main__":
    agent_pts = extract_agent_points(V2X_SIM_ROOT, scene_ids=SCENE_IDS)
    carla_pts = np.load(CARLA_POINTS_NPY)
    tree = cKDTree(carla_pts)

    print("=== Candidate A: theta=260.5 ===")
    best_a = fine_search(agent_pts, tree, theta_center=260.5, t_center=(37.75, -41.25))
    print(f"Best A: score={best_a[0]:.3f}m theta={best_a[1]:.2f} tx={best_a[2]:.2f} ty={best_a[3]:.2f}")

    print("\n=== Candidate B: theta=295.0 ===")
    best_b = fine_search(agent_pts, tree, theta_center=295.0, t_center=(-38.00, 79.00))
    print(f"Best B: score={best_b[0]:.3f}m theta={best_b[1]:.2f} tx={best_b[2]:.2f} ty={best_b[3]:.2f}")

    winner = best_a if best_a[0] < best_b[0] else best_b
    print(f"\nWinner: score={winner[0]:.3f}m theta={winner[1]:.2f} tx={winner[2]:.2f} ty={winner[3]:.2f}")