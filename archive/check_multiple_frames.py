import json

with open('/home/karra/V2X-Sim-2.0/v2.0/sample_data.json') as f:
    sample_data = json.load(f)
with open('/home/karra/V2X-Sim-2.0/v2.0/ego_pose.json') as f:
    ego_pose = json.load(f)

ego_pose_by_token = {e['token']: e for e in ego_pose}

# grab a handful of different frames for scene 77, agent 1
matches = [d for d in sample_data if 'id_1' in d.get('filename', '') and 'scene_77' in d.get('filename', '')]
matches = sorted(matches, key=lambda d: d['filename'])[:5]

for d in matches:
    pose = ego_pose_by_token[d['ego_pose_token']]
    x, y, z = pose['translation']
    print(f"{d['filename']}: world pos = ({x:.2f}, {y:.2f}, {z:.2f})")
