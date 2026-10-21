import carla

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.load_world('Town05')
carla_map = world.get_map()

x, y = -109.57, 122.80
wp = carla_map.get_waypoint(carla.Location(x=x, y=y, z=0),
                             project_to_road=True, lane_type=carla.LaneType.Driving)
loc = wp.transform.location
dist = ((loc.x - x)**2 + (loc.y - y)**2) ** 0.5
print(f"Nearest lane distance: {dist:.2f}m")
