import carla

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.load_world('Town05')
carla_map = world.get_map()

candidates = {
    "original":      (87.72, 105.93),
    "y_flip":        (87.72, -105.93),
    "x_flip":        (-87.72, 105.93),
    "xy_flip":       (-87.72, -105.93),
    "xy_swap":       (105.93, 87.72),
    "xy_swap_yflip": (105.93, -87.72),
}

for name, (x, y) in candidates.items():
    loc = carla.Location(x=x, y=y, z=0.0)
    wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    dist = loc.distance(wp.transform.location)
    print(f"{name:16s} ({x:8.2f}, {y:8.2f}) -> nearest lane dist = {dist:.2f}  road={wp.road_id} lane={wp.lane_id} junction={wp.is_junction}")
