"""
env_dataset.py
Dataset for the env task (runs/DESIGN_ENV.md, Part 2 Stage 2): predict the OTHER vehicles around the ego.

EnvDataset wraps PilotDataset (the ego's sensors and CAM, exactly as D loads them; CPM always off -- D's CPM
holds every GT box, including uncovered class (c), so it would leak; cooperative sensors only with
v2x="on") and joins the env target cache (build_env_target_cache.py, ~/hd_cache/env_targets) by sample row.
The cache's own sample_split field is the v1 split and is NOT used: splits come from
scene_split.three_way_scene_split(dataset, split_version=...) on dataset.samples, like every other run.

Samples: D's sample list (require_prev_frame) minus samples without a class-(a) target -- 13,384 samples,
the same list for every model (every sample with a (b) target also has an (a) target).
Known targets (inputs and prediction targets): v2x="off": vis == 0 (a); v2x="on": vis <= 1 ((a) and (b)).
Class (c) is never loaded, except with use_rsu (v2x="on" only, runs/DECISION_RULES_RSU.md): then also the (c)
targets the RSU sees (num_lidar_pts[0] >= RSU_VIS_PTS) -- set R; their t_vis stays 2, so an evaluation splits
V (t_vis <= 1) from R (t_vis == 2). The sample list is unchanged (it depends on class (a) only).

Ego CAM: the env task keeps only the ego's speed. After PilotDataset's cam normalization, env_cam() zeroes
x, y (as D's drop_cam_xy) AND the absolute world heading (DESIGN_ENV.md section 10), so no world-frame
quantity reaches the model. EnvDataset requires drop_cam_xy=True and a cam_norm (set by train.py).

Per known target (T per sample, padded per batch by env_collate_fn, target_mask marks real ones):
  t_hist  [T, 8, 2]   history in the target's own frame (hist_own), z-scored per coordinate with env_norm
                      (train-split stats over the model's known targets, saved as env_norm.json)
  t_state [T, 11]     cur_ego / 50 (2), cos / sin yaw_ego (2), speed / 5 (m per frame), size w, l / 5 (2),
                      category one-hot (4: car, truck, bicycle, motorcycle)
  t_pose  [T, 3]      x, y (m), yaw (rad) in the ego frame (raw; the model builds relative poses from it)
  t_fut   [T, 30, 2], t_fut_mask [T, 30]   prediction target, own frame (fut_own)
  t_vis, t_cat4, t_moving [T]              for evaluation splits (not model inputs)
  t_tid [T] (cache target index), t_cv_vel [T, 2] (frame-1 -> frame displacement, own frame, raw m per frame:
                      -hist_own[-1]; the constant-velocity baseline)   for evaluation (not model inputs)
  t_map, t_map_pt_mask, t_map_piece_mask   map v2 tokens (env_map.py), only with use_map
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from env_map import LanePolylines, K_PIECES, MAX_PIECE_POINTS, N_POINT_FEATS

ENV_CACHE_DIR = os.path.expanduser("~/hd_cache/env_targets")
TOWNS = ("Town03", "Town04", "Town05")
POS_SCALE_M = 50.0     # target radius around the ego
SPEED_SCALE = 5.0      # m per frame (5 Hz)
SIZE_SCALE_M = 5.0
STATE_DIM = 11
N_CAT4 = 4
EXPECTED_ENV_SAMPLES = 13384
RSU_VIS_PTS = 5        # = build_env_target_cache.VIS_PTS; num_lidar_pts index 0 = the RSU
CACHE_KEYS = ("sample_agent", "sample_scene", "sample_frame", "sample_town", "sample_ptr", "vis", "cat4", "speed",
              "size", "cur_ego", "yaw_ego", "hist_own", "fut_own", "fut_mask", "xy_world", "yaw_world", "n_removed")


CAM_KEEP = 3  # cam = (x, y, heading, speed): only index 3 (speed) is kept


def env_cam(cam):
    """Env-task ego CAM: a copy with x, y and the world heading zeroed; only the speed is kept."""
    out = cam.clone()
    out[:CAM_KEEP] = 0.0
    return out


def load_env_cache(cache_dir=ENV_CACHE_DIR):
    """The numeric arrays the env task needs (token strings are left out), plus the manifest."""
    z = np.load(os.path.join(cache_dir, "env_targets.npz"))
    cache = {k: z[k] for k in CACHE_KEYS}
    cache["rsu_vis"] = z["nlp"][:, 0] >= RSU_VIS_PTS  # the RSU sees the target (used only with use_rsu)
    with open(os.path.join(cache_dir, "manifest.json")) as f:
        cache["manifest"] = json.load(f)
    return cache


def known_mask(vis, v2x, rsu_vis=None):
    """Targets a model knows (input + prediction): v2x off -> class (a); on -> (a) and (b); with rsu_vis (the
    use_rsu model, v2x on only) also the (c) targets with rsu_vis (set R)."""
    assert v2x in ("off", "on"), f"v2x must be 'off' or 'on', got {v2x!r}"
    vis = np.asarray(vis)
    if rsu_vis is not None:
        assert v2x == "on", "use_rsu requires v2x on"
        return (vis <= 1) | ((vis == 2) & np.asarray(rsu_vis, bool))
    return vis == 0 if v2x == "off" else vis <= 1


def env_sample_rows(cache):
    """Cache sample rows with >= 1 class-(a) target: the common sample list of every env model."""
    ptr = np.asarray(cache["sample_ptr"])
    sample_of_target = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    n_a = np.bincount(sample_of_target[np.asarray(cache["vis"]) == 0], minlength=len(ptr) - 1)
    return np.nonzero(n_a > 0)[0]


def target_rows(cache, row, v2x, use_rsu=False):
    """Cache target indices of sample `row` known to the model, in cache order."""
    lo, hi = int(cache["sample_ptr"][row]), int(cache["sample_ptr"][row + 1])
    idx = np.arange(lo, hi)
    return idx[known_mask(cache["vis"][lo:hi], v2x, cache["rsu_vis"][lo:hi] if use_rsu else None)]


def compute_env_norm(cache, rows, v2x, use_rsu=False):
    """Per-coordinate mean / std of hist_own over the known targets of the given sample rows (train split)."""
    idx = np.concatenate([target_rows(cache, r, v2x, use_rsu) for r in rows])
    h = cache["hist_own"][idx].reshape(-1, 2).astype(np.float64)
    return {"mean": h.mean(0).tolist(), "std": (h.std(0, ddof=1) + 1e-6).tolist()}


def state_features(cache, idx):
    """[len(idx), STATE_DIM] float32, see the module docstring."""
    cur = cache["cur_ego"][idx].astype(np.float32)
    yaw = cache["yaw_ego"][idx].astype(np.float32)
    onehot = np.eye(N_CAT4, dtype=np.float32)[cache["cat4"][idx]]
    return np.concatenate([cur / POS_SCALE_M, np.cos(yaw)[:, None], np.sin(yaw)[:, None],
                           cache["speed"][idx, None].astype(np.float32) / SPEED_SCALE,
                           cache["size"][idx, :2].astype(np.float32) / SIZE_SCALE_M, onehot], axis=1)


def sample_features(cache, row, v2x, env_norm=None, lanes=None, use_rsu=False):
    """Env inputs / targets of one cache sample row (dict of tensors; T = known targets, may be 0)."""
    idx = target_rows(cache, row, v2x, use_rsu)
    hist = cache["hist_own"][idx].astype(np.float32)
    if env_norm is not None:
        hist = (hist - np.asarray(env_norm["mean"], dtype=np.float32)) / np.asarray(env_norm["std"], dtype=np.float32)
    pose = np.concatenate([cache["cur_ego"][idx], cache["yaw_ego"][idx, None]], axis=1).astype(np.float32)
    out = {
        "t_hist": torch.from_numpy(hist),
        "t_state": torch.from_numpy(state_features(cache, idx)),
        "t_pose": torch.from_numpy(pose),
        "t_fut": torch.from_numpy(cache["fut_own"][idx].astype(np.float32)),
        "t_fut_mask": torch.from_numpy(cache["fut_mask"][idx].astype(bool)),
        "t_vis": torch.from_numpy(cache["vis"][idx].astype(np.int64)),
        "t_cat4": torch.from_numpy(cache["cat4"][idx].astype(np.int64)),
        "t_moving": torch.from_numpy(cache["speed"][idx] > 0),
        "t_tid": torch.from_numpy(idx.astype(np.int64)),
        "t_cv_vel": torch.from_numpy(-cache["hist_own"][idx, -1].astype(np.float32)),
    }
    if lanes is not None:
        town = str(cache["sample_town"][row])
        toks = [lanes[town].tokens(cache["xy_world"][j], float(cache["yaw_world"][j])) for j in idx]
        k, p = K_PIECES, MAX_PIECE_POINTS
        if toks:
            out["t_map"] = torch.from_numpy(np.stack([t[0] for t in toks]))
            out["t_map_pt_mask"] = torch.from_numpy(np.stack([t[1] for t in toks]))
            out["t_map_piece_mask"] = torch.from_numpy(np.stack([t[2] for t in toks]))
        else:
            out["t_map"] = torch.zeros((0, k, p, N_POINT_FEATS))
            out["t_map_pt_mask"] = torch.zeros((0, k, p), dtype=torch.bool)
            out["t_map_piece_mask"] = torch.zeros((0, k), dtype=torch.bool)
    return out


ENV_KEYS = ("t_hist", "t_state", "t_pose", "t_fut", "t_fut_mask", "t_vis", "t_cat4", "t_moving", "t_tid", "t_cv_vel",
            "t_map", "t_map_pt_mask", "t_map_piece_mask")


def pad_targets(items):
    """Pads the per-sample env tensors to the batch's largest target count (at least 1):
    returns {key: [B, Tmax, ...]} plus target_mask [B, Tmax] (True = real target)."""
    t_max = max(1, max(it["t_hist"].shape[0] for it in items))
    out = {}
    for key in ENV_KEYS:
        if key not in items[0]:
            continue
        ref = items[0][key]
        padded = torch.zeros((len(items), t_max) + tuple(ref.shape[1:]), dtype=ref.dtype)
        for b, it in enumerate(items):
            padded[b, :it[key].shape[0]] = it[key]
        out[key] = padded
    out["target_mask"] = torch.zeros((len(items), t_max), dtype=torch.bool)
    for b, it in enumerate(items):
        out["target_mask"][b, :it["t_hist"].shape[0]] = True
    return out


def env_collate_fn(batch):
    """pilot_collate_fn for the ego / coop / CAM part, plus the padded env tensors."""
    from pilot_dataset import pilot_collate_fn
    base = pilot_collate_fn([{k: v for k, v in b.items() if k not in ENV_KEYS} for b in batch])
    base.update(pad_targets(batch))
    return base


class EnvDataset(Dataset):
    """PilotDataset sensors + env targets. Building it loads the V2X-Sim metadata (PilotDataset)."""
    def __init__(self, v2x, use_map=True, map_transform="mirror", cache_dir=ENV_CACHE_DIR, env_norm=None,
                 use_rsu=False, **pilot_kwargs):
        from pilot_dataset import PilotDataset
        for k in ("use_cpm", "use_coop", "require_prev_frame", "history_frames", "require_history_frames", "degrade"):
            assert k not in pilot_kwargs, f"EnvDataset sets / does not support {k}"
        assert pilot_kwargs.get("drop_cam_xy"), "the env task requires drop_cam_xy=True (with --normalize-cam)"
        self.v2x, self.use_map, self.map_transform, self.use_rsu = v2x, use_map, map_transform, use_rsu
        known_mask([0], v2x, [False] if use_rsu else None)  # validates v2x (and v2x on with use_rsu)
        self.cache = load_env_cache(cache_dir)
        self.pilot = PilotDataset(require_prev_frame=True, use_cpm=False, use_coop=(v2x == "on"),
                                  map_transform=map_transform, use_rsu=use_rsu, **pilot_kwargs)
        c = self.cache
        ps = self.pilot.samples
        assert len(ps) == len(c["sample_agent"]) and all(
            (a, s, f, t) == (c["sample_agent"][i], c["sample_scene"][i], c["sample_frame"][i], c["sample_town"][i])
            for i, (a, s, f, t) in enumerate(ps)), "env cache samples do not match PilotDataset's sample list"
        self.rows = env_sample_rows(c)
        assert len(self.rows) == EXPECTED_ENV_SAMPLES, f"{len(self.rows)} env samples, expected {EXPECTED_ENV_SAMPLES}"
        self.samples = [ps[r] for r in self.rows]  # what three_way_scene_split reads
        self.env_norm = env_norm
        self.lanes = {t: LanePolylines(t, map_transform) for t in TOWNS} if use_map else None
        print(f"EnvDataset: v2x={v2x}, use_map={use_map}, map_transform={map_transform}, "
              + ("use_rsu=True, " if use_rsu else "") + f"{len(self.rows)} samples, "
              f"cache commit {c['manifest']['git_commit'][:7]}")

    @property
    def cam_norm(self):
        return self.pilot.cam_norm

    @cam_norm.setter
    def cam_norm(self, value):
        self.pilot.cam_norm = value

    def get_cam_features(self, agent_id, scene, frame):
        return self.pilot.get_cam_features(agent_id, scene, frame)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        item = self.pilot[r]
        assert self.pilot.cam_norm is not None, "set cam_norm before loading env samples (train.py --normalize-cam)"
        item["cam"] = env_cam(item["cam"])
        item.update(sample_features(self.cache, r, self.v2x, self.env_norm, self.lanes, self.use_rsu))
        return item
