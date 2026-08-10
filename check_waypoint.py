import carla

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)

world = client.load_world('Town05')
print("Loaded map:", world.get_map().name)
assert "Town05" in world.get_map().name, f"Wrong map loaded: {world.get_map().name}"

carla_map = world.get_map()
loc = carla.Location(x=87.72, y=105.93, z=0.0)

wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
print("Nearest waypoint location:", wp.transform.location)
print("Distance to agent pos:", loc.distance(wp.transform.location))
print("Is junction:", wp.is_junction)
print("Road id:", wp.road_id, "Lane id:", wp.lane_id)