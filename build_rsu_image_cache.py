"""
build_rsu_image_cache.py
Builds the RSU camera image cache read by pilot_dataset.load_rsu_images (env task --use-rsu,
runs/DECISION_RULES_RSU.md): one .npy per (scene, frame) in RSU_IMAGE_CACHE_DIR, named 0_<scene>_<frame>.npy,
holding the RSU's 4 views (pilot_dataset.RSU_VIEWS) as uint8 [4, 3, 224, 224] -- decoded + resized with the same
T.Resize as the CAV cache (build_image_cache.py), before ToTensor's float/255 conversion.

Keys: every (scene, frame) of the env target cache's samples (= D's sample list, which EnvDataset indexes), read
from ~/hd_cache/env_targets/env_targets.npz, so the V2X-Sim metadata is not loaded.

Skips keys already present (after checking their shape and dtype). Writes atomically (tempfile + os.replace), so
an interrupted build never leaves a half-written .npy behind. Writes manifest.json last.

Run in gpu-test-venv:  python build_rsu_image_cache.py
"""
import os
import json
import tempfile

import numpy as np
from PIL import Image

from build_image_cache import IMG_SIZE, _resize, _git_commit
from pilot_dataset import SWEEPS_ROOT, RSU_VIEWS, RSU_IMAGE_CACHE_DIR, rsu_cache_path
from env_dataset import ENV_CACHE_DIR

SHAPE = (len(RSU_VIEWS), 3, IMG_SIZE, IMG_SIZE)


def rsu_keys(cache_dir=ENV_CACHE_DIR):
    """Sorted unique (scene, frame) of the env target cache's samples."""
    z = np.load(os.path.join(cache_dir, "env_targets.npz"))
    return sorted(set(zip(z["sample_scene"].tolist(), z["sample_frame"].tolist())))


def build_one(scene, frame):
    views = []
    for v in RSU_VIEWS:
        img = Image.open(f"{SWEEPS_ROOT}/{v}/scene_{scene}_{frame:06d}.jpg").convert("RGB")
        img = _resize(img)                                       # same T.Resize as ego_camera_encoder.load_image
        views.append(np.array(img, dtype=np.uint8).transpose(2, 0, 1))  # HWC -> CHW uint8
    arr = np.stack(views, axis=0)
    assert arr.shape == SHAPE and arr.dtype == np.uint8, (arr.shape, arr.dtype)
    return arr


def main():
    os.makedirs(RSU_IMAGE_CACHE_DIR, exist_ok=True)
    keys = rsu_keys()
    print(f"{len(keys)} RSU (scene, frame) keys over {len({s for s, _ in keys})} scenes; expected size "
          f"{len(keys) * int(np.prod(SHAPE)) / 1024 ** 3:.2f} GiB")
    built = skipped = 0
    for scene, frame in keys:
        out_path = rsu_cache_path(scene, frame)
        if os.path.exists(out_path):
            arr = np.load(out_path)
            assert arr.shape == SHAPE and arr.dtype == np.uint8, f"{out_path}: {arr.shape} {arr.dtype}"
            skipped += 1
            continue
        arr = build_one(scene, frame)
        fd, tmp_path = tempfile.mkstemp(dir=RSU_IMAGE_CACHE_DIR, suffix=".npy.tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.save(f, arr)
            os.replace(tmp_path, out_path)  # atomic rename, same filesystem
        except Exception:
            os.remove(tmp_path)
            raise
        built += 1
        if built % 500 == 0:
            print(f"  built {built}, skipped {skipped} so far (of {len(keys)})", flush=True)
    n_files = len([f for f in os.listdir(RSU_IMAGE_CACHE_DIR) if f.endswith(".npy")])
    assert n_files == len(keys), f"{n_files} .npy files in {RSU_IMAGE_CACHE_DIR}, expected {len(keys)}"
    manifest = {"img_size": IMG_SIZE, "num_views": len(RSU_VIEWS), "views": RSU_VIEWS, "channels": 3,
                "shape": list(SHAPE), "dtype": "uint8", "file_count": n_files,
                "keys_source": "env_targets.npz sample_scene / sample_frame", "git_commit": _git_commit()}
    with open(os.path.join(RSU_IMAGE_CACHE_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done: built {built}, skipped {skipped}, total {len(keys)}; manifest {manifest}")


if __name__ == "__main__":
    main()
