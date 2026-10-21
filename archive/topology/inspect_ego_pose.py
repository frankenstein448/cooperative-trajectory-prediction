import json

with open('/home/karra/V2X-Sim-2.0/v2.0/ego_pose.json') as f:
    ego_pose = json.load(f)

print(f"Total ego_pose entries: {len(ego_pose)}")
print("First entry:")
print(json.dumps(ego_pose[0], indent=2))

with open('/home/karra/V2X-Sim-2.0/v2.0/scene.json') as f:
    scenes = json.load(f)
print(f"\nTotal scenes: {len(scenes)}")
print("First scene entry:")
print(json.dumps(scenes[0], indent=2))

with open('/home/karra/V2X-Sim-2.0/v2.0/sample.json') as f:
    samples = json.load(f)
print(f"\nTotal samples: {len(samples)}")
print("First sample entry:")
print(json.dumps(samples[0], indent=2))