"""
dense_waypoints.py
Generates a dense waypoint cloud for a given town, used as the reference
point set for grid_search_transform.py's coordinate alignment.

IMPORTANT: does NOT use client.load_world() -- that was found unreliable
(silently defaults to the wrong town in some cases). CARLA must already be
running with the correct town loaded via startup argument. This script
verifies the loaded map matches before saving anything.
"""
import sys
import carla
import matplotlib.pyplot as plt
import numpy as np

EXPECTED_TOWN = sys.argv[1] if len(sys.argv) > 1 else "Town05"  # e.g. python dense_waypoints.py Town03

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()  # NOT load_world() -- just connect to whatever's already running
carla_map = world.get_map()

# Safety guard: confirm the live server actually has the town we expect loaded
loaded_name = carla_map.name.split("/")[-1]  # e.g. "Town03" from "Carla/Maps/Town03"
if EXPECTED_TOWN not in loaded_name:
    raise RuntimeError(
        f"Loaded map is '{loaded_name}', but expected '{EXPECTED_TOWN}'. "
        f"Make sure CARLA was launched with '{EXPECTED_TOWN}' as a startup argument, "
        f"not via load_world()."
    )
print(f"Confirmed: CARLA has {loaded_name} loaded, matches expected {EXPECTED_TOWN}")

waypoints = carla_map.generate_waypoints(distance=2.0)
xs = [wp.transform.location.x for wp in waypoints]
ys = [wp.transform.location.y for wp in waypoints]
pts = np.array(list(zip(xs, ys)), dtype=np.float64)
np.save(f"dense_waypoints_{EXPECTED_TOWN.lower()}.npy", pts)

plt.figure(figsize=(10, 10))
plt.scatter(xs, ys, s=1, c="black")
plt.gca().set_aspect("equal")
plt.title(f"{EXPECTED_TOWN} dense waypoints (live CARLA)")
plt.savefig(f"{EXPECTED_TOWN}_dense_waypoints.png", dpi=150)
print(f"Plotted {len(waypoints)} waypoints, saved dense_waypoints_{EXPECTED_TOWN.lower()}.npy")