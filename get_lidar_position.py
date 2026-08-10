import json

with open('/home/karra/V2X-Sim-2.0/v2.0/sample_data.json') as f:
    sample_data = json.load(f)
with open('/home/karra/V2X-Sim-2.0/v2.0/ego_pose.json') as f:
    ego_pose = json.load(f)
with open('/home/karra/V2X-Sim-2.0/v2.0/calibrated_sensor.json') as f:
    calib = json.load(f)

ego_pose_by_token = {e['token']: e for e in ego_pose}
calib_by_token = {c['token']: c for c in calib}

d = next(x for x in sample_data if x['filename'] == 'sweeps/LIDAR_TOP_id_1/scene_77_000005.pcd.bin')
pose = ego_pose_by_token[d['ego_pose_token']]
cs = calib_by_token[d['calibrated_sensor_token']]

print("ego_pose translation:", pose['translation'])
print("sensor calibration translation:", cs['translation'])
