"""
check_town03_residuals.py
Residual distribution check for the resolved Town03 transform:
  theta=194.75, tx=-34.25, ty=6.00 (median 1.023m)
"""
import numpy as np
from scipy.spatial import cKDTree
from grid_search_transform import extract_agent_points, apply_transform

V2X_SIM_ROOT = "/home/karra/V2X-Sim-2.0"
CARLA_POINTS_NPY = "dense_waypoints_town03.npy"
SCENE_IDS = [4, 5, 64, 77, 79, 80, 85]

THETA, TX, TY = 194.75, -34.25, 6.00

agent_pts = extract_agent_points(V2X_SIM_ROOT, scene_ids=SCENE_IDS)
carla_pts = np.load(CARLA_POINTS_NPY)
tree = cKDTree(carla_pts)

transformed = apply_transform(agent_pts, THETA, TX, TY)
dists, _ = tree.query(transformed)

print(f"Median: {np.median(dists):.3f}m  Mean: {np.mean(dists):.3f}m  Min: {np.min(dists):.3f}m  Max: {np.max(dists):.3f}m")
print()
for threshold in [0.5, 1, 2, 5, 10, 20, 50]:
    pct = (dists <= threshold).mean() * 100
    print(f"  <= {threshold}m: {pct:.1f}%")