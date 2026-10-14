"""
verify_rsu_image_cache.py
RSU image cache check (build_rsu_image_cache.py): the manifest matches (views, shape, dtype, file count = every
env sample's (scene, frame)), and for 50 random keys the 4 view tensors pilot_dataset.load_rsu_images returns are
torch.equal with use_image_cache=False (JPEG decode) vs True (cache), as verify_image_cache.py does for the CAVs.
Read-only, no training. Prints RESULT: PASS / FAIL (run_rsu_seeds.sh checks the committed log).

Run in gpu-test-venv:  python verify_rsu_image_cache.py 2>&1 | tee verify_rsu_image_cache.log
"""
import os
import json
import random

import torch

from pilot_dataset import RSU_VIEWS, RSU_IMAGE_CACHE_DIR, load_rsu_images
from build_rsu_image_cache import rsu_keys, SHAPE

N = 50


def main():
    keys = rsu_keys()
    with open(os.path.join(RSU_IMAGE_CACHE_DIR, "manifest.json")) as f:
        m = json.load(f)
    assert m["views"] == RSU_VIEWS and m["shape"] == list(SHAPE) and m["dtype"] == "uint8", m
    n_files = len([f for f in os.listdir(RSU_IMAGE_CACHE_DIR) if f.endswith(".npy")])
    assert m["file_count"] == n_files == len(keys), f"manifest {m['file_count']}, files {n_files}, keys {len(keys)}"
    print(f"manifest OK: {n_files} files = {len(keys)} env (scene, frame) keys, views {RSU_VIEWS}, shape {SHAPE} uint8")

    random.seed(0)
    for scene, frame in random.sample(keys, N):
        a = load_rsu_images(scene, frame, use_image_cache=False)
        b = load_rsu_images(scene, frame, use_image_cache=True)  # asserts shape / dtype
        assert len(a) == len(b) == len(RSU_VIEWS), f"({scene}, {frame}): view count"
        for v in range(len(RSU_VIEWS)):
            assert a[v].dtype == b[v].dtype == torch.float32 and a[v].shape == b[v].shape
            assert torch.equal(a[v], b[v]), f"({scene}, {frame}) view {RSU_VIEWS[v]} differs"
    print(f"OK: {N} random keys, all {len(RSU_VIEWS)} views torch.equal (JPEG decode vs cache)")
    print("RESULT: PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"RESULT: FAIL ({e})")
        raise
