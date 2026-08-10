import numpy as np
import carla
from coord_transform import v2xsim_to_carla_batch
from grid_search_transform import extract_agent_points

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.load_world('Town05')
carla_map = world.get_map()

agent_pts = extract_agent_points("/home/karra/V2X-Sim-2.0", scene_ids=[77])
transformed = v2xsim_to_carla_batch(agent_pts)

dists = []
for x, y in transformed:
    wp = carla_map.get_waypoint(carla.Location(x=x, y=y, z=0),
                                 project_to_road=True, lane_type=carla.LaneType.Driving)
    loc = wp.transform.location
    dists.append(((loc.x - x)**2 + (loc.y - y)**2) ** 0.5)

dists = np.array(dists)
print(f"median: {np.median(dists):.2f}m")
print(f"mean:   {np.mean(dists):.2f}m")
print(f"p90:    {np.percentile(dists, 90):.2f}m")
print(f"max:    {np.max(dists):.2f}m")
print(f"fraction under 2m: {(dists < 2).mean()*100:.1f}%")
