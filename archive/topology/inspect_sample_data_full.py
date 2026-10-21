import json

with open('/home/karra/V2X-Sim-2.0/v2.0/sample_data.json') as f:
    sample_data = json.load(f)

print("Full first entry (all fields):")
print(json.dumps(sample_data[0], indent=2))

print("\nFilenames of first 15 entries for same sample_token:")
first_sample_token = sample_data[0]['sample_token']
matches = [sd for sd in sample_data if sd['sample_token'] == first_sample_token]
for m in matches[:15]:
    print(f"  {m.get('filename')}")