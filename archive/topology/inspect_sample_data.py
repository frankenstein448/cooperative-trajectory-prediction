import json

with open('/home/karra/V2X-Sim-2.0/v2.0/sample_data.json') as f:
    sample_data = json.load(f)

print(f"Total sample_data entries: {len(sample_data)}")
print("First entry:")
print(json.dumps(sample_data[0], indent=2))
print("\nEntries for the same sample_token as first entry (to see channels per sample):")
first_sample_token = sample_data[0]['sample_token']
matches = [sd for sd in sample_data if sd['sample_token'] == first_sample_token]
print(f"Found {len(matches)} sample_data entries for this sample_token")
for m in matches[:10]:
    print(f"  channel={m.get('channel')}, is_key_frame={m.get('is_key_frame')}, ego_pose_token={m.get('ego_pose_token')}")