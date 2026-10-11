"""
pilot_dataset.py
Raw data loading ONLY -- no model/encoder objects, no forward passes.
For end-to-end training: encoders live in end_to_end_model.py so gradients
can flow through them. This file just reads files off disk and does the
non-trainable geometric prep (coordinate transform, nearest-lane lookup).

Requires a custom collate_fn (pilot_collate_fn below) since LiDAR point
counts and coop neighbor counts vary per sample -- can't be stacked directly.

Run in gpu-test-venv.
"""
import re
import glob
import os
import json
import math
import torch
import numpy as np
from torch.utils.data import Dataset

from v2x_message_stream import V2XMetadataLoader
from ground_truth_extractor import get_future_trajectory, HORIZON
from ego_sensor_encoder import load_semantic_lidar
from ego_camera_encoder import load_image, IMG_SIZE
from find_coop_neighbors import find_coop_neighbors
from rgcn_map_encoder import load_graph, get_nearby_lane_nodes
from coord_transform import v2xsim_to_carla
from motion_utils import get_cv_velocity
from sensor_degradation import DEGRADE_MODES, degrade_points, degrade_images

SWEEPS_ROOT = "/home/karra/V2X-Sim-2.0/sweeps"
CAMERA_COMPATIBLE_AGENTS = [1, 2, 3, 4]
VIEWS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
K_NEAREST_LANES = 5

# Camera image cache (built by build_image_cache.py): one .npy per
# (agent_id, scene, frame), holding all 3 views as uint8 [3, 3, 224, 224]
# (decoded + resized, before ToTensor's float/255 conversion).
IMAGE_CACHE_DIR = os.path.expanduser("~/hd_cache/images")

TOWN03_SCENES = [4, 5, 64, 77, 79, 80, 85]
TOWN04_SCENES = [6, 9, 20, 26, 30, 92, 96, 97, 98]
TOWN05_SCENES = [
    3, 14, 15, 16, 21, 22, 27, 32, 34, 35, 36, 37, 38, 39, 41, 42, 43, 45, 46,
    49, 50, 51, 52, 55, 59, 62, 65, 68, 71, 72, 73, 76, 78, 86, 89, 91, 95, 100,
]

# scene -> town lookup, built once at import time
SCENE_TO_TOWN = {}
for _s in TOWN03_SCENES:
    SCENE_TO_TOWN[_s] = "Town03"
for _s in TOWN04_SCENES:
    SCENE_TO_TOWN[_s] = "Town04"
for _s in TOWN05_SCENES:
    SCENE_TO_TOWN[_s] = "Town05"

ALL_SCENES = TOWN03_SCENES + TOWN04_SCENES + TOWN05_SCENES


def _available_frames(agent_id: int, scene: int) -> set:
    pattern = f"{SWEEPS_ROOT}/SEMLIDAR_TOP_id_{agent_id}/scene_{scene}_*.pcd.bin"
    frames = set()
    for path in glob.glob(pattern):
        m = re.search(r"scene_\d+_(\d+)\.pcd\.bin", path)
        if m:
            frames.add(int(m.group(1)))
    return frames


class PilotDataset(Dataset):
    def __init__(self, scenes=ALL_SCENES, agent_ids=CAMERA_COMPATIBLE_AGENTS, horizon: int = HORIZON,
                 use_image_cache: bool = False, require_prev_frame: bool = False, use_speed: bool = False,
                 cam_norm: dict = None, drop_cam_xy: bool = False, degrade: str = None):
        if use_speed:
            assert require_prev_frame, "use_speed=True requires require_prev_frame=True (frame-1 must exist)"

        self.horizon = horizon
        self.use_image_cache = use_image_cache
        self.require_prev_frame = require_prev_frame
        self.use_speed = use_speed
        # Settable post-construction too (train.py computes train-split stats
        # AFTER building the dataset/split, then sets dataset.cam_norm directly
        # rather than re-constructing a second dataset). {"mean": [4], "std": [4]}.
        # Applied ONLY to the returned sample's "cam" -- every lookup inside
        # __getitem__ (nearest-lane, etc.) must keep using the raw cam.
        self.cam_norm = cam_norm
        # Zeroes the returned sample's cam[0:2] (x,y) AFTER normalization, so the
        # model sees no absolute position. Requires cam_norm (checked in
        # __getitem__, since train.py sets cam_norm post-construction). Lookups
        # keep using cam_raw, exactly as with cam_norm alone.
        self.drop_cam_xy = drop_cam_xy
        # Sensor degradation (sensor_degradation.py), applied to every agent's
        # raw LiDAR/images as they are loaded -- ego and coop alike. cam, cpm,
        # nearby_idx and gt are never touched. Settable post-construction.
        assert degrade is None or degrade in DEGRADE_MODES, f"unknown degrade={degrade!r}"
        self.degrade = degrade
        self.loader = V2XMetadataLoader()

        if use_image_cache:
            manifest_path = os.path.join(IMAGE_CACHE_DIR, "manifest.json")
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(
                    f"use_image_cache=True but no manifest at {manifest_path} -- "
                    f"build the cache with build_image_cache.py first.")
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert manifest["img_size"] == IMG_SIZE, (
                f"Image cache manifest img_size={manifest['img_size']} does not match "
                f"expected {IMG_SIZE} (ego_camera_encoder.IMG_SIZE)")
        print(f"PilotDataset: use_image_cache={use_image_cache}"
              + (f" (dir={IMAGE_CACHE_DIR})" if use_image_cache else "")
              + f", require_prev_frame={require_prev_frame}, use_speed={use_speed}, "
              + f"cam_norm={'on' if cam_norm is not None else 'off'}, drop_cam_xy={drop_cam_xy}, "
              + f"degrade={degrade}")

        # graph kept here ONLY for the non-trainable nearest-node lookup.
        # Now per-town: each town has its own topology + normalization stats,
        # loaded once and cached, not per-sample.
        self.town_graphs = {}
        for town in sorted(set(SCENE_TO_TOWN[s] for s in scenes)):
            data, id_to_idx, mean, std = load_graph(f"topology/{town}_topology.pkl")
            self.town_graphs[town] = {"data": data, "id_to_idx": id_to_idx, "mean": mean, "std": std}

        self.samples = []
        for scene in scenes:
            town = SCENE_TO_TOWN[scene]
            for agent_id in agent_ids:
                frames = _available_frames(agent_id, scene)
                for frame in sorted(frames):
                    if not all((frame + step) in frames for step in range(1, horizon + 1)):
                        continue
                    if require_prev_frame and (frame - 1) not in frames:
                        continue
                    self.samples.append((agent_id, scene, frame, town))

    def __len__(self):
        return len(self.samples)

    def get_cam_features(self, agent_id, scene, frame):
        """Returns the RAW [4] (x,y,heading,speed) cam tensor, before any
        normalization -- speed is motion_utils-derived if self.use_speed,
        else get_cam's own (always-0) speed field. Factored out so train.py
        can compute cam_norm stats over the train split without loading
        images/LiDAR, and so __getitem__ can keep a raw copy for lookups."""
        cam, _ = self.loader.get_cam(agent_id, scene, frame)
        if self.use_speed:
            ok, vx, vy = get_cv_velocity(self.loader, agent_id, scene, frame)
            assert ok, (f"require_prev_frame guarantees frame-1 exists, but get_cv_velocity found none "
                        f"for (agent={agent_id}, scene={scene}, frame={frame})")
            cam[3] = math.hypot(vx, vy)  # metres per frame, from frame-1 -> frame displacement
        return cam

    def _load_images(self, agent_id, scene, frame):
        """Returns VIEWS as [3,224,224] float tensors in [0,1], identical to
        load_image()'s ToTensor() output. Reads the on-disk cache if
        self.use_image_cache is True; a missing key is a hard error, never a
        silent fallback to decoding the original JPEGs."""
        if not self.use_image_cache:
            cam_base = SWEEPS_ROOT + "/{view}_id_" + str(agent_id) + "/scene_" + str(scene) + f"_{frame:06d}.jpg"
            return [load_image(cam_base.format(view=v)) for v in VIEWS]

        key = (agent_id, scene, frame)
        cache_path = os.path.join(IMAGE_CACHE_DIR, f"{agent_id}_{scene}_{frame:06d}.npy")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Image cache miss for key {key}: {cache_path} not found. "
                f"use_image_cache=True requires every referenced key to be pre-built by "
                f"build_image_cache.py -- refusing to fall back to decoding from disk.")
        arr = np.load(cache_path)  # [3 views, 3, 224, 224] uint8, CHW per view
        assert arr.shape == (3, 3, IMG_SIZE, IMG_SIZE), (
            f"Image cache shape mismatch for key {key}: got {arr.shape}, "
            f"expected (3, 3, {IMG_SIZE}, {IMG_SIZE})")
        assert arr.dtype == np.uint8, f"Image cache dtype mismatch for key {key}: got {arr.dtype}, expected uint8"
        return [torch.from_numpy(arr[v]).float().div(255) for v in range(arr.shape[0])]

    def _load_sensors(self, agent_id, scene, frame):
        """One agent's (points, images), with self.degrade applied. Used for the
        ego AND every coop neighbor, so degradation can't reach one but not the other."""
        lidar_path = f"{SWEEPS_ROOT}/SEMLIDAR_TOP_id_{agent_id}/scene_{scene}_{frame:06d}.pcd.bin"
        points = load_semantic_lidar(lidar_path)
        images = self._load_images(agent_id, scene, frame)
        if self.degrade is not None:
            points = degrade_points(points, self.degrade)
            images = degrade_images(images, self.degrade, agent_id, scene, frame)
        return points, images

    def __getitem__(self, idx):
        agent_id, scene, frame, town = self.samples[idx]
        graph = self.town_graphs[town]

        points, images = self._load_sensors(agent_id, scene, frame)

        # RAW cam -- used for every lookup below (nearest-lane, etc.). Normalization,
        # if any, is applied to a separate copy only at the very end of this method,
        # right before building the returned dict -- never to this variable.
        cam_raw = self.get_cam_features(agent_id, scene, frame)

        cpm_objects = self.loader.get_cpm(agent_id, scene, frame)

        neighbor_ids = find_coop_neighbors(self.loader, ego_agent_id=agent_id, scene=scene, frame=frame,
                                            candidate_ids=[a for a in CAMERA_COMPATIBLE_AGENTS if a != agent_id])
        coop_points, coop_images = [], []
        for nid in neighbor_ids:
            n_points, n_images = self._load_sensors(nid, scene, frame)
            coop_points.append(n_points)
            coop_images.append(n_images)

        # non-trainable geometry: coordinate transform + nearest-lane lookup
        # -- now town-aware, using this sample's own town's transform + graph.
        # Uses cam_raw, NOT the (possibly normalized) cam built below -- this
        # lookup needs real V2X-Sim world coordinates, not z-scored values.
        ego_x, ego_y = cam_raw[0].item(), cam_raw[1].item()
        carla_x, carla_y = v2xsim_to_carla(ego_x, ego_y, town)
        agent_pos = torch.tensor([[carla_x, carla_y]], dtype=torch.float32)
        agent_pos_norm = (agent_pos - graph["mean"]) / graph["std"]
        nearby_idx = get_nearby_lane_nodes(agent_pos_norm, graph["data"].x, k=K_NEAREST_LANES)  # [k] for single agent
        if nearby_idx.dim() == 1:
            nearby_idx = nearby_idx  # already [k], fine as-is per-sample -- collate stacks to [B, k]

        gt_traj, gt_mask = get_future_trajectory(self.loader, agent_id, scene, frame, self.horizon)

        # cam normalization happens here, as the LAST step, on a copy -- cam_raw
        # above (used for the lookup) is never touched.
        cam = cam_raw
        if self.cam_norm is not None:
            mean = torch.tensor(self.cam_norm["mean"], dtype=torch.float32)
            std = torch.tensor(self.cam_norm["std"], dtype=torch.float32)
            cam = (cam_raw - mean) / std
        if self.drop_cam_xy:
            assert self.cam_norm is not None, "drop_cam_xy=True requires cam_norm (--normalize-cam)"
            cam = cam.clone()  # never write into cam_raw
            cam[0:2] = 0.0

        return {
            "agent_id": agent_id,
            "scene": scene,
            "frame": frame,
            "town": town,
            "points": points,
            "images": images,
            "cam": cam,
            "cpm_objects": cpm_objects,
            "coop_points": coop_points,
            "coop_images": coop_images,
            "nearby_idx": nearby_idx,
            "gt_traj": gt_traj,
            "gt_mask": gt_mask,
        }


def pilot_collate_fn(batch):
    """
    Custom collate: variable-size items (points, images, coop_points, coop_images,
    cpm_objects) stay as Python lists across the batch. Fixed-size items (cam,
    nearby_idx, gt_traj, gt_mask) get stacked into normal batched tensors.
    """
    return {
        "agent_id": [b["agent_id"] for b in batch],
        "scene": [b["scene"] for b in batch],
        "frame": [b["frame"] for b in batch],
        "town": [b["town"] for b in batch],
        "points": [b["points"] for b in batch],
        "images": [b["images"] for b in batch],
        "cam": [b["cam"] for b in batch],                      # kept as list; small, looped in model anyway
        "cpm_objects": [b["cpm_objects"] for b in batch],
        "coop_points": [b["coop_points"] for b in batch],
        "coop_images": [b["coop_images"] for b in batch],
        "nearby_idx": torch.stack([b["nearby_idx"] for b in batch], dim=0),  # [B, k] -- fixed size, safe to stack
        "gt_traj": torch.stack([b["gt_traj"] for b in batch], dim=0),
        "gt_mask": torch.stack([b["gt_mask"] for b in batch], dim=0),
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import time

    print("Building dataset...")
    t0 = time.time()
    dataset = PilotDataset()
    print(f"Setup took {time.time() - t0:.1f}s, {len(dataset)} samples")
    for town in dataset.town_graphs:
        n = sum(1 for s in dataset.samples if s[3] == town)
        print(f"  {town}: {n} samples")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=pilot_collate_fn)
    batch = next(iter(loader))
    print(f"Batch nearby_idx shape: {batch['nearby_idx'].shape}")
    print(f"Batch gt_traj shape: {batch['gt_traj'].shape}")
    print(f"Batch towns: {batch['town']}")
    print(f"Points in batch (list of {len(batch['points'])} tensors), first shape: {batch['points'][0].shape}")