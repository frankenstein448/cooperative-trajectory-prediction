"""
verify_image_cache.py
For 50 random dataset indices, asserts every image tensor __getitem__ returns
(ego's 3 views, and all 3 views for every coop neighbor) is torch.equal
between use_image_cache=False and use_image_cache=True, and that every other
field is unchanged. Read-only check, no training.
"""
import random
import torch

from pilot_dataset import PilotDataset
from coord_transform import warn_legacy_script
warn_legacy_script("verify_image_cache.py")

random.seed(0)
N = 50

ds_nocache = PilotDataset(use_image_cache=False)
ds_cache = PilotDataset(use_image_cache=True)
assert len(ds_nocache) == len(ds_cache) == len(ds_nocache.samples), "dataset sample lists diverged"
assert ds_nocache.samples == ds_cache.samples, "sample ordering diverged between the two instances"

idxs = random.sample(range(len(ds_nocache)), N)

for idx in idxs:
    a = ds_nocache[idx]
    b = ds_cache[idx]

    assert a["agent_id"] == b["agent_id"]
    assert a["scene"] == b["scene"]
    assert a["frame"] == b["frame"]
    assert a["town"] == b["town"]
    assert torch.equal(a["points"], b["points"]), f"idx {idx}: ego points differ"
    assert torch.equal(a["cam"], b["cam"]), f"idx {idx}: cam differs"
    assert torch.equal(a["cpm_objects"], b["cpm_objects"]), f"idx {idx}: cpm_objects differ"
    assert torch.equal(a["nearby_idx"], b["nearby_idx"]), f"idx {idx}: nearby_idx differs"
    assert torch.equal(a["gt_traj"], b["gt_traj"]), f"idx {idx}: gt_traj differs"
    assert torch.equal(a["gt_mask"], b["gt_mask"]), f"idx {idx}: gt_mask differs"

    assert len(a["images"]) == len(b["images"]) == 3, f"idx {idx}: ego view count mismatch"
    for v in range(3):
        assert torch.equal(a["images"][v], b["images"][v]), f"idx {idx}: ego view {v} differs"

    assert len(a["coop_points"]) == len(b["coop_points"]), f"idx {idx}: coop agent count mismatch"
    for i in range(len(a["coop_points"])):
        assert torch.equal(a["coop_points"][i], b["coop_points"][i]), f"idx {idx}: coop {i} points differ"

    assert len(a["coop_images"]) == len(b["coop_images"]), f"idx {idx}: coop image-agent count mismatch"
    for i in range(len(a["coop_images"])):
        assert len(a["coop_images"][i]) == len(b["coop_images"][i]) == 3, f"idx {idx}: coop {i} view count mismatch"
        for v in range(3):
            assert torch.equal(a["coop_images"][i][v], b["coop_images"][i][v]), \
                f"idx {idx}: coop {i} view {v} differs"

print(f"OK: {N} random samples (ego images, all coop images, and every other field) "
      f"are torch.equal with use_image_cache=False vs True.")
