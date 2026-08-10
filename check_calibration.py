import json

with open('/home/karra/V2X-Sim-2.0/v2.0/sample_data.json') as f:
    sample_data = json.load(f)
with open('/home/karra/V2X-Sim-2.0/v2.0/calibrated_sensor.json') as f:
    calib = json.load(f)

calib_by_token = {c['token']: c for c in calib}

d = next(x for x in sample_data if x['filename'] == 'sweeps/BEV_TOP_id_1/scene_77_000005.npz')
cs = calib_by_token[d['calibrated_sensor_token']]
print("sensor translation (relative to vehicle):", cs['translation'])
print("sensor rotation (quaternion):", cs['rotation'])
