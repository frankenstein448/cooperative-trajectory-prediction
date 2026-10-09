"""
refine_town03_theta.py
Finer theta search around Town03's coarse-search candidate:
  theta=195.0, tx=-34.25, ty=6.00 (1.072m median)
Same rationale as Town04: the coarse search only tested theta at 5-degree
steps, so we don't yet know if a nearby finer angle scores better.
"""
import numpy as np
from scipy.spatial import cKDTree
from grid_search_transform import extract_agent_points, apply_transform

V2X_SIM_ROOT = "/home/karra/V2X-Sim-2.0"
CARLA_POINTS_NPY = "dense_waypoints_town03.npy"
SCENE_IDS = [4, 5, 64, 77, 79, 80, 85]


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

    print("=== Refining around theta=195.0 ===")
    best = fine_search(agent_pts, tree, theta_center=195.0, t_center=(-34.25, 6.00))
    print(f"Best: score={best[0]:.3f}m theta={best[1]:.2f} tx={best[2]:.2f} ty={best[3]:.2f}")