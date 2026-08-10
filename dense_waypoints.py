import carla
import matplotlib.pyplot as plt
import numpy as np

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.load_world('Town05')
carla_map = world.get_map()

waypoints = carla_map.generate_waypoints(distance=2.0)
xs = [wp.transform.location.x for wp in waypoints]
ys = [wp.transform.location.y for wp in waypoints]
pts = np.array(list(zip(xs, ys)), dtype=np.float64)
np.save("dense_waypoints_town05.npy", pts)

plt.figure(figsize=(10, 10))
plt.scatter(xs, ys, s=1, c="black")
plt.gca().set_aspect("equal")
plt.title("Town05 dense waypoints (live CARLA)")
plt.savefig("Town05_dense_waypoints.png", dpi=150)
print(f"Plotted {len(waypoints)} waypoints")
