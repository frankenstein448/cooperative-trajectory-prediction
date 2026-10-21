"""
export_junctions.py
Junction areas for eval_scenes.py's near_junction condition (evaluation only).

Builds each town's carla.Map OFFLINE from CARLA 0.9.15's OpenDRIVE file
(carla.Map(name, xodr_text); no server, no load_world) and saves the lane-centre
waypoints that lie inside a junction (wp.is_junction, i.e. on an OpenDRIVE
connecting road with junction != -1), every JUNCTION_SPACING_M metres along every
lane, as junction_waypoints_<town>.npy: [N, 3] = CARLA x, y, junction id.

Check: the same offline map's generate_waypoints(2.0) must reproduce
dense_waypoints_<town>.npy (built from the live server by dense_waypoints.py)
exactly, so this is the map the off-road metric and the RGCN topology use.

Needs the carla package, i.e. run with ~/carla-venv/bin/python (not gpu-test-venv):
    ~/carla-venv/bin/python export_junctions.py
"""
import os
import numpy as np
import carla
from scipy.spatial import cKDTree

import paths

XODR_DIR = paths.CARLA_XODR_DIR
TOWNS = ["Town03", "Town04", "Town05"]
JUNCTION_SPACING_M = 1.0

for town in TOWNS:
    with open(os.path.join(XODR_DIR, f"{town}.xodr")) as f:
        carla_map = carla.Map(town, f.read())

    xy = lambda wps: np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
    ref = np.load(f"dense_waypoints_{town.lower()}.npy")
    dense = xy(carla_map.generate_waypoints(2.0))
    d_fwd, _ = cKDTree(ref).query(dense)
    d_back, _ = cKDTree(dense).query(ref)
    if len(dense) != len(ref) or d_fwd.max() > 1e-3 or d_back.max() > 1e-3:
        raise RuntimeError(f"{town}: offline map does not reproduce dense_waypoints_{town.lower()}.npy "
                           f"({len(dense)} vs {len(ref)} points, max distance {max(d_fwd.max(), d_back.max()):.3f} m)")

    jwps = [w for w in carla_map.generate_waypoints(JUNCTION_SPACING_M) if w.is_junction]
    out = np.concatenate([xy(jwps), np.array([[w.get_junction().id] for w in jwps], dtype=np.float64)], axis=1)
    np.save(f"junction_waypoints_{town.lower()}.npy", out)
    print(f"{town}: offline map matches dense_waypoints ({len(ref)} points); "
          f"{len(out)} in-junction waypoints, {len(np.unique(out[:, 2]))} junctions "
          f"-> junction_waypoints_{town.lower()}.npy")
