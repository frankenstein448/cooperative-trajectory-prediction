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

from v2x_message_stream import V2XMetadataLoader, quaternion_to_yaw
from ground_truth_extractor import get_future_trajectory, HORIZON
from ego_sensor_encoder import load_semantic_lidar
from ego_camera_encoder import load_image, IMG_SIZE
from find_coop_neighbors import find_coop_neighbors
from rgcn_map_encoder import load_graph, get_nearby_lane_nodes
import paths
from coord_transform import v2xsim_to_carla, MAP_TRANSFORMS
from motion_utils import get_cv_velocity
from sensor_degradation import DEGRADE_MODES, degrade_points, degrade_images

SWEEPS_ROOT = paths.V2X_SWEEPS_DIR
CAMERA_COMPATIBLE_AGENTS = [1, 2, 3, 4]
VIEWS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
K_NEAREST_LANES = 5

# Camera image cache (built by build_image_cache.py): one .npy per
# (agent_id, scene, frame), holding all 3 views as uint8 [3, 3, 224, 224]
# (decoded + resized, before ToTensor's float/255 conversion).
IMAGE_CACHE_DIR = paths.IMAGE_CACHE_DIR

# RSU (agent 0; env task use_rsu, runs/DECISION_RULES_RSU.md): SEMLIDAR_TOP_id_0 + 4 cameras CAM_id_0_<k>.
# Its image cache (build_rsu_image_cache.py) is separate: one .npy per (scene, frame), uint8 [4, 3, 224, 224].
RSU_AGENT_ID = 0
RSU_VIEWS = ["CAM_id_0_0", "CAM_id_0_1", "CAM_id_0_2", "CAM_id_0_3"]
RSU_IMAGE_CACHE_DIR = paths.RSU_IMAGE_CACHE_DIR

# Ego-task map input: v1 = RGCN lane graph (topology/*.pkl, nearby_idx); v2 = the env task's lane-piece tokens
# (env_map.py; runs/DECISION_RULES_MAPV2_EGO.md).
MAP_VERSIONS = ("v1", "v2")


def rsu_cache_path(scene, frame):
    return os.path.join(RSU_IMAGE_CACHE_DIR, f"{RSU_AGENT_ID}_{scene}_{frame:06d}.npy")


def load_rsu_images(scene, frame, use_image_cache):
    """The RSU's 4 views as [3,224,224] float tensors in [0,1] (load_image's output). From the RSU cache if
    use_image_cache: a missing key is a hard error, never a fallback to the JPEGs."""
    if not use_image_cache:
        return [load_image(f"{SWEEPS_ROOT}/{v}/scene_{scene}_{frame:06d}.jpg") for v in RSU_VIEWS]
    path = rsu_cache_path(scene, frame)
    if not os.path.exists(path):
        raise FileNotFoundError(f"RSU image cache miss for (scene {scene}, frame {frame}): {path} not found; "
                                f"build it with build_rsu_image_cache.py")
    arr = np.load(path)
    assert arr.shape == (len(RSU_VIEWS), 3, IMG_SIZE, IMG_SIZE), (
        f"RSU image cache shape mismatch for {path}: got {arr.shape}, expected ({len(RSU_VIEWS)}, 3, {IMG_SIZE}, {IMG_SIZE})")
    assert arr.dtype == np.uint8, f"RSU image cache dtype mismatch for {path}: got {arr.dtype}, expected uint8"
    return [torch.from_numpy(arr[v]).float().div(255) for v in range(arr.shape[0])]

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
                 cam_norm: dict = None, drop_cam_xy: bool = False, degrade: str = None,
                 history_frames: int = 0, require_history_frames: int = 0, history_norm: dict = None,
                 use_cpm: bool = True, use_coop: bool = True, map_transform: str = "legacy", use_rsu: bool = False,
                 map_version: str = "v1"):
        if use_speed:
            assert require_prev_frame, "use_speed=True requires require_prev_frame=True (frame-1 must exist)"
        assert history_frames >= 0 and require_history_frames >= 0
        assert not (history_frames and require_history_frames), (
            "history_frames already implies its own frame filter; require_history_frames is for the "
            "no-history control only")

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
        # history_frames=H > 0: keep only samples whose agent has frames frame-H .. frame,
        # and return "history" [H, 2] (see get_history). require_history_frames=N > 0:
        # the same filter with N, but no history returned -- the control for H=N.
        self.history_frames = history_frames
        self.require_history_frames = require_history_frames
        # V2X ablation (EndToEndFusion's use_cpm / use_coop, same names): use_cpm=False
        # skips get_cpm and returns no "cpm_objects"; use_coop=False skips the neighbour
        # search and neighbour sensor loading and returns no "coop_points"/"coop_images".
        # The sample list is unaffected. The ego's own CAM is always returned.
        self.use_cpm = use_cpm
        self.use_coop = use_coop
        # V2X-Sim -> CARLA for the nearest-lane lookup (coord_transform.MAP_TRANSFORMS). Default legacy, so
        # every existing caller is unchanged; train.py / eval_scenes.py pass the run's map_transform.
        assert map_transform in MAP_TRANSFORMS, f"unknown map_transform={map_transform!r}"
        self.map_transform = map_transform
        # use_rsu=True (env task, runs/DECISION_RULES_RSU.md): also return the RSU's (agent 0) SEMLIDAR points and
        # its 4 camera views as "rsu_points" / "rsu_images". Default off: nothing else changes.
        assert not (use_rsu and degrade), "use_rsu does not support degrade"
        self.use_rsu = use_rsu
        # map_version="v2" (ego task, runs/DECISION_RULES_MAPV2_EGO.md): also return the env task's map v2 tokens
        # (env_map.py) at the ego's pose, in the ego frame of gt_traj: "map_tokens" [K_PIECES, MAX_PIECE_POINTS,
        # N_POINT_FEATS], "map_pt_mask", "map_piece_mask". Default v1: nothing else changes (nearby_idx is
        # returned either way).
        assert map_version in MAP_VERSIONS, f"unknown map_version={map_version!r}"
        self.map_version = map_version
        self.lanes = None
        if map_version == "v2":
            assert map_transform == "mirror", "map_version v2 requires map_transform mirror"
            from env_map import LanePolylines
            self.lanes = {t: LanePolylines(t, map_transform) for t in ("Town03", "Town04", "Town05")}
        # Like cam_norm: {"mean": [2], "std": [2]} per coordinate, from the train
        # split, settable post-construction, applied to the returned copy only.
        self.history_norm = history_norm
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
            if use_rsu:
                with open(os.path.join(RSU_IMAGE_CACHE_DIR, "manifest.json")) as f:
                    rsu_manifest = json.load(f)
                assert (rsu_manifest["img_size"], rsu_manifest["views"]) == (IMG_SIZE, RSU_VIEWS), (
                    f"RSU image cache manifest {rsu_manifest} does not match IMG_SIZE={IMG_SIZE}, RSU_VIEWS={RSU_VIEWS}")
        print(f"PilotDataset: use_image_cache={use_image_cache}"
              + (f" (dir={IMAGE_CACHE_DIR})" if use_image_cache else "")
              + f", require_prev_frame={require_prev_frame}, use_speed={use_speed}, "
              + f"cam_norm={'on' if cam_norm is not None else 'off'}, drop_cam_xy={drop_cam_xy}, "
              + f"degrade={degrade}, history_frames={history_frames}, "
              + f"require_history_frames={require_history_frames}, "
              + f"history_norm={'on' if history_norm is not None else 'off'}, "
              + f"use_cpm={use_cpm}, use_coop={use_coop}, map_transform={map_transform}"
              + (f", use_rsu=True (dir={RSU_IMAGE_CACHE_DIR})" if use_rsu else "")
              + (", map_version=v2" if map_version == "v2" else ""))

        # graph kept here ONLY for the non-trainable nearest-node lookup.
        # Now per-town: each town has its own topology + normalization stats,
        # loaded once and cached, not per-sample.
        self.town_graphs = {}
        for town in sorted(set(SCENE_TO_TOWN[s] for s in scenes)):
            data, id_to_idx, mean, std = load_graph(f"topology/{town}_topology.pkl")
            self.town_graphs[town] = {"data": data, "id_to_idx": id_to_idx, "mean": mean, "std": std}

        n_past = max(history_frames, require_history_frames)
        self.samples = []
        n_dropped_history = 0
        for scene in scenes:
            town = SCENE_TO_TOWN[scene]
            for agent_id in agent_ids:
                frames = _available_frames(agent_id, scene)
                for frame in sorted(frames):
                    if not all((frame + step) in frames for step in range(1, horizon + 1)):
                        continue
                    if require_prev_frame and (frame - 1) not in frames:
                        continue
                    if n_past and not all((frame - step) in frames for step in range(1, n_past + 1)):
                        n_dropped_history += 1
                        continue
                    self.samples.append((agent_id, scene, frame, town))
        if n_past:
            print(f"PilotDataset: dropped {n_dropped_history} samples without frames frame-{n_past} .. frame "
                  f"({len(self.samples)} kept)")

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

    def map_tokens(self, agent_id, scene, frame, town):
        """map_version v2: env_map tokens around the agent, in the frame of gt_traj (origin at the ego_pose
        translation, x along quaternion_to_yaw(ego_pose rotation), as ground_truth_extractor.get_future_trajectory).
        -> (feats, point_mask, piece_mask) numpy arrays."""
        pose = self.loader.ego_pose[self.loader.find_sample_data(agent_id, scene, frame)["ego_pose_token"]]
        x, y, _ = pose["translation"]
        return self.lanes[town].tokens(np.array([x, y], dtype=np.float64), quaternion_to_yaw(pose["rotation"]))

    def get_history(self, agent_id, scene, frame):
        """Returns the RAW [H, 2] history: positions at frame-H .. frame-1 minus the
        position at `frame`, rotated into the heading at `frame` -- the convention
        of ground_truth_extractor.get_future_trajectory / motion_utils.get_cv_velocity,
        so history[-1] == -(get_cv_velocity's vx, vy). A parked agent gives all
        zeros. Factored out (like get_cam_features) so train.py can compute
        history_norm over the train split without loading sensors."""
        H = self.history_frames
        cur_pose = self.loader.ego_pose[self.loader.find_sample_data(agent_id, scene, frame)["ego_pose_token"]]
        cur_x, cur_y, _ = cur_pose["translation"]
        cur_heading = quaternion_to_yaw(cur_pose["rotation"])
        cos_h, sin_h = math.cos(-cur_heading), math.sin(-cur_heading)  # inverse rotation

        history = torch.zeros((H, 2), dtype=torch.float32)
        for k in range(H):
            past_frame = frame - H + k
            # the sample filter guarantees these frames exist; find_sample_data raises if not
            pose = self.loader.ego_pose[self.loader.find_sample_data(agent_id, scene, past_frame)["ego_pose_token"]]
            px, py, _ = pose["translation"]
            dx, dy = px - cur_x, py - cur_y
            history[k, 0] = dx * cos_h - dy * sin_h
            history[k, 1] = dx * sin_h + dy * cos_h
        return history

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

        extra = {}
        if self.use_cpm:
            extra["cpm_objects"] = self.loader.get_cpm(agent_id, scene, frame)

        if self.use_coop:
            neighbor_ids = find_coop_neighbors(self.loader, ego_agent_id=agent_id, scene=scene, frame=frame,
                                                candidate_ids=[a for a in CAMERA_COMPATIBLE_AGENTS if a != agent_id])
            coop_points, coop_images = [], []
            for nid in neighbor_ids:
                n_points, n_images = self._load_sensors(nid, scene, frame)
                coop_points.append(n_points)
                coop_images.append(n_images)
            extra["coop_points"] = coop_points
            extra["coop_images"] = coop_images

        if self.lanes is not None:
            feats, pt_mask, piece_mask = self.map_tokens(agent_id, scene, frame, town)
            extra["map_tokens"] = torch.from_numpy(feats)
            extra["map_pt_mask"] = torch.from_numpy(pt_mask)
            extra["map_piece_mask"] = torch.from_numpy(piece_mask)

        if self.use_rsu:
            extra["rsu_points"] = load_semantic_lidar(
                f"{SWEEPS_ROOT}/SEMLIDAR_TOP_id_{RSU_AGENT_ID}/scene_{scene}_{frame:06d}.pcd.bin")
            extra["rsu_images"] = load_rsu_images(scene, frame, self.use_image_cache)

        # non-trainable geometry: coordinate transform + nearest-lane lookup
        # -- now town-aware, using this sample's own town's transform + graph.
        # Uses cam_raw, NOT the (possibly normalized) cam built below -- this
        # lookup needs real V2X-Sim world coordinates, not z-scored values.
        ego_x, ego_y = cam_raw[0].item(), cam_raw[1].item()
        carla_x, carla_y = v2xsim_to_carla(ego_x, ego_y, town, self.map_transform)
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

        if self.history_frames > 0:
            history = self.get_history(agent_id, scene, frame)
            if self.history_norm is not None:
                mean = torch.tensor(self.history_norm["mean"], dtype=torch.float32)
                std = torch.tensor(self.history_norm["std"], dtype=torch.float32)
                history = (history - mean) / std  # new tensor; per coordinate, broadcast over H
            extra["history"] = history

        return {
            "agent_id": agent_id,
            "scene": scene,
            "frame": frame,
            "town": town,
            "points": points,
            "images": images,
            "cam": cam,
            "nearby_idx": nearby_idx,
            "gt_traj": gt_traj,
            "gt_mask": gt_mask,
            **extra,
        }


def load_history_settings(run_dir, run_config):
    """Dataset kwargs for the history experiment, from a run's config.json:
    history_frames, require_history_frames, and history_norm loaded from
    <run_dir>/history_norm.json -- a hard error if history is on and the file is
    missing. Configs from before the history experiment give all-off."""
    history_frames = run_config.get("history_frames", 0)
    history_norm = None
    if history_frames > 0:
        path = os.path.join(run_dir, "history_norm.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"run config says history_frames={history_frames} but no history_norm.json at {path}")
        with open(path) as f:
            history_norm = json.load(f)
    return {"history_frames": history_frames,
            "require_history_frames": run_config.get("require_history_frames", 0),
            "history_norm": history_norm}


def load_map_settings(run_config):
    """map_version from a run's config.json (absent = v1, the RGCN lane graph). Same kwarg name for PilotDataset
    and EndToEndFusion."""
    v = run_config.get("map_version", "v1")
    assert v in MAP_VERSIONS, f"unknown map_version {v!r}"
    return {"map_version": v}


def load_v2x_settings(run_config):
    """use_cpm / use_coop from a run's config.json (keys no_cpm / no_coop; absent in
    pre-ablation configs = both on). Same kwarg names for PilotDataset and EndToEndFusion."""
    return {"use_cpm": not run_config.get("no_cpm", False),
            "use_coop": not run_config.get("no_coop", False)}


def pilot_collate_fn(batch):
    """
    Custom collate: variable-size items (points, images, coop_points, coop_images,
    cpm_objects) stay as Python lists across the batch. Fixed-size items (cam,
    nearby_idx, gt_traj, gt_mask) get stacked into normal batched tensors.
    """
    extra = {}
    if "history" in batch[0]:
        extra["history"] = torch.stack([b["history"] for b in batch], dim=0)  # [B, H, 2]
    for key in ("cpm_objects", "coop_points", "coop_images", "rsu_points", "rsu_images"):  # absent when off
        if key in batch[0]:
            extra[key] = [b[key] for b in batch]
    for key in ("map_tokens", "map_pt_mask", "map_piece_mask"):  # map_version v2 only; fixed size
        if key in batch[0]:
            extra[key] = torch.stack([b[key] for b in batch], dim=0)
    return {
        **extra,
        "agent_id": [b["agent_id"] for b in batch],
        "scene": [b["scene"] for b in batch],
        "frame": [b["frame"] for b in batch],
        "town": [b["town"] for b in batch],
        "points": [b["points"] for b in batch],
        "images": [b["images"] for b in batch],
        "cam": [b["cam"] for b in batch],                      # kept as list; small, looped in model anyway
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