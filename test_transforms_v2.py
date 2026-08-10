import carla
import math

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.load_world('Town05')
carla_map = world.get_map()

# movement direction from your 5 frames: dx=87.46-87.39=0.07, dy=98.71-96.75=1.96 over 4 frames
move_dir = math.degrees(math.atan2(1.96, 0.07))  # ~ mostly +y direction

candidates = {
    "original":      (87.72, 105.93),
    "y_flip":        (87.72, -105.93),
    "x_flip":        (-87.72, 105.93),
    "xy_flip":       (-87.72, -105.93),
    "xy_swap":       (105.93, 87.72),
    "xy_swap_yflip": (105.93, -87.72),
}

print(f"observed movement heading: {move_dir:.1f} deg\n")
for name, (x, y) in candidates.items():
    loc = carla.Location(x=x, y=y, z=0.0)
    wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    dist = loc.distance(wp.transform.location)
    lane_yaw = wp.transform.rotation.yaw % 360
    diff = min(abs(lane_yaw - move_dir), 360 - abs(lane_yaw - move_dir))
    diff_rev = min(abs((lane_yaw + 180) % 360 - move_dir), 360 - abs((lane_yaw + 180) % 360 - move_dir))
    heading_match = min(diff, diff_rev)
    print(f"{name:16s} dist={dist:6.2f}  lane_yaw={lane_yaw:6.1f}  heading_diff={heading_match:5.1f}")
