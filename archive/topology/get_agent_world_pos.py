import json

with open('/home/karra/V2X-Sim-2.0/v2.0/sample_data.json') as f:
    sample_data = json.load(f)

with open('/home/karra/V2X-Sim-2.0/v2.0/ego_pose.json') as f:
    ego_pose_list = json.load(f)
ego_pose_by_token = {e['token']: e for e in ego_pose_list}

# search for LIDAR_TOP_id_1 entries matching scene 77, frame 23
# first, let's see what scene naming/frame numbering actually looks like near this range
candidates = [sd for sd in sample_data if 'LIDAR_TOP_id_1' in sd['filename'] and 'scene_77_' in sd['filename']]
print(f"Found {len(candidates)} LIDAR_TOP_id_1 entries for scene_77")
for c in candidates[:5]:
    print(f"  {c['filename']}  ego_pose_token={c['ego_pose_token']}")

# try to find frame 23 specifically (check both 0-indexed and 1-indexed possibilities)
for frame_guess in [23, 22, 24]:
    fname_guess = f"scene_77_{frame_guess:06d}"
    match = [c for c in candidates if fname_guess in c['filename']]
    if match:
        token = match[0]['ego_pose_token']
        pose = ego_pose_by_token[token]
        print(f"\nMatch for frame_guess={frame_guess}: {match[0]['filename']}")
        print(f"  translation (world x,y,z): {pose['translation']}")