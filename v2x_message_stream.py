"""
v2x_message_stream.py
V2X message stream: CAM (ego's own state) + CPM (detected nearby objects),
built from V2X-Sim's nuScenes-style metadata (no raw sensor data involved).
Run in gpu-test-venv.
"""
import json
import re
import math
import torch
import torch.nn as nn

EMBED_DIM = 128
V2X_ROOT = "/home/karra/V2X-Sim-2.0/v2.0"
CPM_RADIUS_M = 50.0  # only include objects within this range of ego


def _load_json(name):
    with open(f"{V2X_ROOT}/{name}") as f:
        return json.load(f)


def _index_by(records, key):
    return {r[key]: r for r in records}


def quaternion_to_yaw(q):
    """q: [w, x, y, z] (nuScenes convention) -> yaw in radians."""
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class V2XMetadataLoader:
    """Joins sample_data -> ego_pose / sample -> sample_annotation."""
    def __init__(self):
        self.sample_data = _load_json("sample_data.json")
        # Build a fast lookup index: (agent_id, scene, frame) -> record
        # Avoids a full linear scan through sample_data on every find_sample_data() call.
        self._sample_data_index = {}
        for rec in self.sample_data:
            m = re.search(r"id_(\d+)/scene_(\d+)_(\d+)\.", rec["filename"])
            if m:
                agent_id, scene, frame = int(m.group(1)), int(m.group(2)), int(m.group(3))
                self._sample_data_index[(agent_id, scene, frame)] = rec
        self.ego_pose = _index_by(_load_json("ego_pose.json"), "token")
        self.sample_annotation = _load_json("sample_annotation.json")
        # group annotations by sample_token for fast lookup
        self.anno_by_sample = {}
        for a in self.sample_annotation:
            self.anno_by_sample.setdefault(a["sample_token"], []).append(a)

    def find_sample_data(self, agent_id, scene, frame):
        """Match V2X-Sim's filename convention to find the right record."""
        key = (agent_id, scene, frame)
        if key not in self._sample_data_index:
            raise ValueError(f"No sample_data match for agent {agent_id}, scene {scene}, frame {frame}")
        return self._sample_data_index[key]

    def get_cam(self, agent_id, scene, frame, prev_frame=None):
        """Returns [x, y, heading, speed] for this agent at this frame."""
        rec = self.find_sample_data(agent_id, scene, frame)
        pose = self.ego_pose[rec["ego_pose_token"]]
        x, y, _ = pose["translation"]
        heading = quaternion_to_yaw(pose["rotation"])

        speed = 0.0
        if prev_frame is not None:
            prev_rec = self.find_sample_data(agent_id, scene, prev_frame)
            prev_pose = self.ego_pose[prev_rec["ego_pose_token"]]
            px, py, _ = prev_pose["translation"]
            dt = (pose["timestamp"] - prev_pose["timestamp"]) / 1e6  # microseconds -> seconds
            if dt > 0:
                speed = math.hypot(x - px, y - py) / dt

        return torch.tensor([x, y, heading, speed], dtype=torch.float32), rec

    def get_cpm(self, agent_id, scene, frame):
        """Returns [M, 7] tensor: (rel_x, rel_y, rel_z, size_x, size_y, size_z, yaw) per nearby object."""
        cam, rec = self.get_cam(agent_id, scene, frame)
        ego_x, ego_y = cam[0].item(), cam[1].item()
        sample_token = rec["sample_token"]

        objects = []
        for anno in self.anno_by_sample.get(sample_token, []):
            ox, oy, oz = anno["translation"]
            dist = math.hypot(ox - ego_x, oy - ego_y)
            if dist <= CPM_RADIUS_M:
                sx, sy, sz = anno["size"]
                yaw = quaternion_to_yaw(anno["rotation"])
                objects.append([ox - ego_x, oy - ego_y, oz, sx, sy, sz, yaw])

        if len(objects) == 0:
            return torch.zeros((0, 7), dtype=torch.float32)
        return torch.tensor(objects, dtype=torch.float32)


class CAMEncoder(nn.Module):
    """CAM: [4] (x,y,heading,speed) -> [128]."""
    def __init__(self, out_dim=EMBED_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, cam: torch.Tensor) -> torch.Tensor:
        return self.mlp(cam)


class CPMEncoder(nn.Module):
    """CPM: [M, 7] variable-size object list -> [128], PointNet-style (per-object MLP + max-pool)."""
    def __init__(self, in_dim=7, hidden_dim=64, out_dim=EMBED_DIM):
        super().__init__()
        self.obj_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.out_mlp = nn.Linear(hidden_dim, out_dim)
        self.empty_embed = nn.Parameter(torch.zeros(out_dim))  # learned "no objects" embedding

    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        if objects.shape[0] == 0:
            return self.empty_embed
        feats = self.obj_mlp(objects)       # [M, 64]
        pooled, _ = feats.max(dim=0)          # [64]
        return self.out_mlp(pooled)            # [128]


class V2XMessageStream(nn.Module):
    """CAM + CPM -> concat -> project to [128].

    use_history=True: a history embedding (same width as the CAM embedding,
    computed batched by EndToEndFusion's history encoder) is concatenated right
    after the CAM embedding. use_cpm=False: cpm_enc is removed (not fed an empty
    object list) and the CPM embedding drops out of the concat. fuse's input is
    out_dim * (1 + use_history + use_cpm). With use_history=False, use_cpm=True
    the module is exactly the original one."""
    def __init__(self, out_dim=EMBED_DIM, use_history=False, use_cpm=True):
        super().__init__()
        self.use_history = use_history
        self.use_cpm = use_cpm
        self.cam_enc = CAMEncoder(out_dim=out_dim)
        if use_cpm:
            self.cpm_enc = CPMEncoder(out_dim=out_dim)
        self.fuse = nn.Linear(out_dim * (1 + int(use_history) + int(use_cpm)), out_dim)

    def forward(self, cam: torch.Tensor, cpm_objects: torch.Tensor = None,
                history_embed: torch.Tensor = None) -> torch.Tensor:
        assert (history_embed is not None) == self.use_history, (
            f"use_history={self.use_history} but history_embed is {'given' if history_embed is not None else 'None'}")
        assert (cpm_objects is not None) == self.use_cpm, (
            f"use_cpm={self.use_cpm} but cpm_objects is {'given' if cpm_objects is not None else 'None'}")
        parts = [self.cam_enc(cam)]
        if self.use_history:
            parts.append(history_embed)
        if self.use_cpm:
            parts.append(self.cpm_enc(cpm_objects))
        concat = torch.cat(parts, dim=0)
        return self.fuse(concat)


if __name__ == "__main__":
    loader = V2XMetadataLoader()

    agent_id, scene, frame = 4, 89, 100
    cam, _ = loader.get_cam(agent_id, scene, frame)
    cpm_objects = loader.get_cpm(agent_id, scene, frame)
    print(f"CAM: {cam}")
    print(f"CPM objects found: {cpm_objects.shape}")

    stream = V2XMessageStream()
    embedding = stream(cam, cpm_objects)
    print(f"V2X message embedding shape: {embedding.shape}")