"""
check_town04_residuals.py
Diagnostic: apply the current best-found Town04 transform to all extracted
agent points, and look at the FULL residual distribution -- not just the
median -- to tell whether this is "one good transform + some noisy outliers"
(fine) or "no single transform fits well" (a real problem).
"""
import numpy as np
from scipy.spatial import cKDTree
from grid_search_transform import extract_agent_points, apply_transform

V2X_SIM_ROOT = "/home/karra/V2X-Sim-2.0"
CARLA_POINTS_NPY = "dense_waypoints_town04.npy"

# current best from the widened search
THETA, TX, TY = 295.0, -38.00, 79.00

agent_pts = extract_agent_points(V2X_SIM_ROOT, scene_ids=[6, 9, 20, 26, 30, 92, 96, 97, 98])
carla_pts = np.load(CARLA_POINTS_NPY)
tree = cKDTree(carla_pts)

transformed = apply_transform(agent_pts, THETA, TX, TY)
dists, _ = tree.query(transformed)

print(f"Total points: {len(dists)}")
print(f"Median: {np.median(dists):.3f}m")
print(f"Mean: {np.mean(dists):.3f}m")
print(f"Min: {np.min(dists):.3f}m")
print(f"Max: {np.max(dists):.3f}m")
print()
print("Percentage of points within a given distance:")
for threshold in [0.5, 1, 2, 5, 10, 20, 50]:
    pct = (dists <= threshold).mean() * 100
    print(f"  <= {threshold}m: {pct:.1f}%")

print()
print("Distribution by percentile:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: {np.percentile(dists, p):.2f}m")