import carla
import pickle
import os

os.makedirs("topology", exist_ok=True)

town = 'Town04'

print("connecting...")
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)

print(f"loading {town}...")
client.load_world(town)
world = client.get_world()
carla_map = world.get_map()

print("getting topology...")
topology = carla_map.get_topology()
print(f"got {len(topology)} topology edges")

edges = []
seen = set()

def add_edge(wp_a, wp_b, relation):
    edges.append({
        "start_id": wp_a.id, "end_id": wp_b.id,
        "start_pos": (wp_a.transform.location.x, wp_a.transform.location.y),
        "end_pos": (wp_b.transform.location.x, wp_b.transform.location.y),
        "road_id": wp_a.road_id, "lane_id": wp_a.lane_id,
        "relation": relation,  # 0=successor, 1=left, 2=right
    })

for i, (wp_start, wp_end) in enumerate(topology):
    if i % 50 == 0:
        print(f"processing edge {i}/{len(topology)}")
    add_edge(wp_start, wp_end, 0)
    for wp in (wp_start, wp_end):
        if wp.id in seen:
            continue
        seen.add(wp.id)
        if wp.is_junction:
            continue
        print(f"  checking left/right for wp {wp.id}")
        left = wp.get_left_lane()
        if left is not None and left.lane_type == carla.LaneType.Driving:
            add_edge(wp, left, 1)
        right = wp.get_right_lane()
        if right is not None and right.lane_type == carla.LaneType.Driving:
            add_edge(wp, right, 2)

print(f"saving {len(edges)} total edges...")
with open(f"topology/{town}_topology.pkl", "wb") as f:
    pickle.dump(edges, f)

print(f"{town}: saved {len(edges)} edges")