"""
profile_getitem.py
Single-process timing breakdown of PilotDataset.__getitem__, mirroring its
exact structure (pilot_dataset.py:85-133), over 100 random train-split
samples. Read-only, no training, no model forward/backward involved.

USE_IMAGE_CACHE toggles whether the "jpeg" bucket goes through the on-disk
image cache (dataset._load_images) or decodes the original JPEGs.
"""
import random
import time
import torch

from pilot_dataset import PilotDataset, SWEEPS_ROOT, K_NEAREST_LANES, CAMERA_COMPATIBLE_AGENTS
from ego_sensor_encoder import load_semantic_lidar
from find_coop_neighbors import find_coop_neighbors
from rgcn_map_encoder import get_nearby_lane_nodes
from coord_transform import v2xsim_to_carla
from ground_truth_extractor import get_future_trajectory
from scene_split import scene_split
from coord_transform import warn_legacy_script
warn_legacy_script("profile_getitem.py")

random.seed(0)
N = 100
USE_IMAGE_CACHE = True

dataset = PilotDataset(use_image_cache=USE_IMAGE_CACHE)
train_set, _ = scene_split(dataset, 0.15, 42)
idxs = random.sample(list(train_set.indices), N)

totals = {"jpeg": 0.0, "lidar": 0.0, "neighbor_search": 0.0, "meta_gt": 0.0}
n_jpeg_files = 0
n_lidar_files = 0
n_agents_list = []

for idx in idxs:
    agent_id, scene, frame, town = dataset.samples[idx]
    graph = dataset.town_graphs[town]

    t0 = time.perf_counter()
    lidar_path = f"{SWEEPS_ROOT}/SEMLIDAR_TOP_id_{agent_id}/scene_{scene}_{frame:06d}.pcd.bin"
    points = load_semantic_lidar(lidar_path)
    t1 = time.perf_counter(); totals["lidar"] += t1 - t0; n_lidar_files += 1

    images = dataset._load_images(agent_id, scene, frame)
    t2 = time.perf_counter(); totals["jpeg"] += t2 - t1; n_jpeg_files += len(images)

    cam, _ = dataset.loader.get_cam(agent_id, scene, frame)
    cpm_objects = dataset.loader.get_cpm(agent_id, scene, frame)
    t3 = time.perf_counter(); totals["meta_gt"] += t3 - t2

    neighbor_ids = find_coop_neighbors(dataset.loader, ego_agent_id=agent_id, scene=scene, frame=frame,
                                        candidate_ids=[a for a in CAMERA_COMPATIBLE_AGENTS if a != agent_id])
    t4 = time.perf_counter(); totals["neighbor_search"] += t4 - t3

    n_agents = 1
    for nid in neighbor_ids:
        n_lidar_path = f"{SWEEPS_ROOT}/SEMLIDAR_TOP_id_{nid}/scene_{scene}_{frame:06d}.pcd.bin"
        ta = time.perf_counter()
        load_semantic_lidar(n_lidar_path)
        tb = time.perf_counter(); totals["lidar"] += tb - ta; n_lidar_files += 1

        n_images = dataset._load_images(nid, scene, frame)
        tc = time.perf_counter(); totals["jpeg"] += tc - tb; n_jpeg_files += len(n_images)
        n_agents += 1
    n_agents_list.append(n_agents)

    t5 = time.perf_counter()
    ego_x, ego_y = cam[0].item(), cam[1].item()
    carla_x, carla_y = v2xsim_to_carla(ego_x, ego_y, town)
    agent_pos = torch.tensor([[carla_x, carla_y]], dtype=torch.float32)
    agent_pos_norm = (agent_pos - graph["mean"]) / graph["std"]
    nearby_idx = get_nearby_lane_nodes(agent_pos_norm, graph["data"].x, k=K_NEAREST_LANES)
    gt_traj, gt_mask = get_future_trajectory(dataset.loader, agent_id, scene, frame, dataset.horizon)
    t6 = time.perf_counter(); totals["meta_gt"] += t6 - t5

n = len(idxs)
total_all = sum(totals.values())
print(f"Over {n} random train samples (mean {sum(n_agents_list)/n:.2f} agents/sample, "
      f"{n_jpeg_files} jpeg files, {n_lidar_files} lidar files total):\n")
for k, v in totals.items():
    print(f"  {k:16s}: {v/n*1000:7.2f} ms/sample  ({100*v/total_all:5.1f}% of accounted time)")
print(f"  {'sum':16s}: {total_all/n*1000:7.2f} ms/sample")
print(f"\n  jpeg per-file:  {totals['jpeg']/n_jpeg_files*1000:6.3f} ms/file  ({n_jpeg_files} files)")
print(f"  lidar per-file: {totals['lidar']/n_lidar_files*1000:6.3f} ms/file  ({n_lidar_files} files)")
