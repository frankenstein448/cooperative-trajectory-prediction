import json
import numpy as np
import carla

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.load_world('Town05')
carla_map = world.get_map()

with open('/home/karra/V2X-Sim-2.0/v2.0/sample_data.json') as f:
    sample_data = json.load(f)
with open('/home/karra/V2X-Sim-2.0/v2.0/ego_pose.json') as f:
    ego_pose = json.load(f)

ego_pose_by_token = {e['token']: e for e in ego_pose}

# pull many LIDAR_TOP_id_1 positions across scene 77
matches = [d for d in sample_data if 'LIDAR_TOP_id_1' in d.get('filename', '') and 'scene_77' in d.get('filename', '')]
matches = sorted(matches, key=lambda d: d['filename'])

src_pts = []  # raw V2X-Sim positions
dst_pts = []  # nearest CARLA lane point for each

for d in matches:
    pose = ego_pose_by_token[d['ego_pose_token']]
    x, y, z = pose['translation']
    loc = carla.Location(x=x, y=y, z=0.0)
    wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    dist = loc.distance(wp.transform.location)
    if dist < 30:  # discard wild outliers
        src_pts.append([x, y])
        dst_pts.append([wp.transform.location.x, wp.transform.location.y])

src = np.array(src_pts)
dst = np.array(dst_pts)
print(f"Using {len(src)} point pairs")

# Kabsch algorithm: find best rotation + translation mapping src -> dst
src_mean = src.mean(axis=0)
dst_mean = dst.mean(axis=0)
src_c = src - src_mean
dst_c = dst - dst_mean

H = src_c.T @ dst_c
U, S, Vt = np.linalg.svd(H)
R = Vt.T @ U.T
if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1
    R = Vt.T @ U.T
t = dst_mean - R @ src_mean

print("Rotation matrix:\n", R)
print("Translation:", t)
print("Rotation angle (deg):", np.degrees(np.arctan2(R[1, 0], R[0, 0])))

# check residual error after applying fitted transform
transformed = (R @ src.T).T + t
residuals = np.linalg.norm(transformed - dst, axis=1)
print("Mean residual after fit:", residuals.mean())
print("Max residual after fit:", residuals.max())
