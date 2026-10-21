"""
build_image_cache.py
Builds the on-disk camera image cache consumed by pilot_dataset.py's
USE_IMAGE_CACHE path: one .npy per (agent_id, scene, frame) key, holding all
3 views as uint8 [3, 3, 224, 224] -- decoded + resized (T.Resize), before
ToTensor's float/255 conversion.

Keys are the union of every (ego, scene, frame) and (coop_neighbor, scene,
frame) tuple PilotDataset.__getitem__ actually references -- the same set
pilot_dataset.py's cache reader will look up.

Skips keys already present. Writes atomically (tempfile + os.replace) so an
interrupted build never leaves a half-written .npy behind.
"""
import os
import json
import subprocess
import tempfile
import numpy as np
from PIL import Image
import torchvision.transforms as T

import paths
from pilot_dataset import PilotDataset, SWEEPS_ROOT, VIEWS, CAMERA_COMPATIBLE_AGENTS
from find_coop_neighbors import find_coop_neighbors

CACHE_DIR = paths.IMAGE_CACHE_DIR
IMG_SIZE = 224
_resize = T.Resize((IMG_SIZE, IMG_SIZE))


def _git_commit():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir).decode().strip()
    except Exception as e:
        raise RuntimeError(f"Could not determine git commit for manifest: {e}")


def write_manifest(cache_dir):
    npy_files = [f for f in os.listdir(cache_dir) if f.endswith(".npy")]
    manifest = {
        "img_size": IMG_SIZE,
        "num_views": len(VIEWS),
        "channels": 3,
        "file_count": len(npy_files),
        "git_commit": _git_commit(),
    }
    manifest_path = os.path.join(cache_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {manifest_path}: {manifest}")


def collect_keys(dataset):
    keys = set()
    for agent_id, scene, frame, town in dataset.samples:
        keys.add((agent_id, scene, frame))
        neighbor_ids = find_coop_neighbors(dataset.loader, ego_agent_id=agent_id, scene=scene, frame=frame,
                                            candidate_ids=[a for a in CAMERA_COMPATIBLE_AGENTS if a != agent_id])
        for nid in neighbor_ids:
            keys.add((nid, scene, frame))
    return keys


def build_one(agent_id, scene, frame):
    views = []
    for v in VIEWS:
        path = f"{SWEEPS_ROOT}/{v}_id_{agent_id}/scene_{scene}_{frame:06d}.jpg"
        img = Image.open(path).convert("RGB")
        img = _resize(img)                        # same T.Resize as ego_camera_encoder.load_image
        arr = np.array(img, dtype=np.uint8)        # HWC uint8
        arr = arr.transpose(2, 0, 1)               # -> CHW uint8
        views.append(arr)
    return np.stack(views, axis=0)                 # [3 views, 3, 224, 224] uint8


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    dataset = PilotDataset()
    keys = collect_keys(dataset)
    total_bytes = len(keys) * 3 * 3 * IMG_SIZE * IMG_SIZE  # uint8 -> 1 byte/elem
    print(f"{len(keys)} unique (agent, scene, frame) keys referenced as ego or coop "
          f"across all {len(set((s for _, s, _, _ in dataset.samples)))} scenes")
    print(f"expected cache size: {total_bytes / 1024**3:.2f} GiB  "
          f"({total_bytes / len(keys) / 1024:.1f} KiB/key)")

    built, skipped = 0, 0
    for agent_id, scene, frame in sorted(keys):
        out_path = os.path.join(CACHE_DIR, f"{agent_id}_{scene}_{frame:06d}.npy")
        if os.path.exists(out_path):
            skipped += 1
            continue
        arr = build_one(agent_id, scene, frame)
        fd, tmp_path = tempfile.mkstemp(dir=CACHE_DIR, suffix=".npy.tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.save(f, arr)
            os.replace(tmp_path, out_path)  # atomic rename, same filesystem
        except Exception:
            os.remove(tmp_path)
            raise
        built += 1
        if built % 1000 == 0:
            print(f"  built {built}, skipped {skipped} so far (of {len(keys)} total)")

    print(f"Done: built {built}, skipped {skipped} (already present), total {len(keys)}")
    write_manifest(CACHE_DIR)


if __name__ == "__main__":
    main()
