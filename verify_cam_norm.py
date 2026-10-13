"""
verify_cam_norm.py
For 50 random dataset indices, asserts nearby_idx and cpm_objects are
identical between cam_norm off and on, and that cam differs (it must, by
construction, once cam_norm is actually applied) -- catches the lookup-uses-
normalized-cam bug class. Read-only, no training. A single PilotDataset
instance is used throughout, toggling dataset.cam_norm between calls, which
is exactly how train.py uses it (built once, cam_norm set after the split).
"""
import random
import torch

from pilot_dataset import PilotDataset
from coord_transform import warn_legacy_script
warn_legacy_script("verify_cam_norm.py")

random.seed(0)
N = 50

dataset = PilotDataset(require_prev_frame=True, use_speed=True)
idxs = random.sample(range(len(dataset)), N)

# Representative cam_norm -- computed the same way train.py does, just over
# a convenient slice rather than a real train split; only the mechanism is
# under test here, not the exact stats.
cam_values = torch.stack([dataset.get_cam_features(*dataset.samples[i][:3]) for i in idxs])
mean = cam_values.mean(0)
std = cam_values.std(0) + 1e-6
cam_norm = {"mean": mean.tolist(), "std": std.tolist()}
print(f"Test cam_norm: {cam_norm}")

dataset.cam_norm = None
items_off = [dataset[i] for i in idxs]
dataset.cam_norm = cam_norm
items_on = [dataset[i] for i in idxs]

n_cam_differs = 0
for idx, a, b in zip(idxs, items_off, items_on):
    assert torch.equal(a["nearby_idx"], b["nearby_idx"]), f"idx {idx}: nearby_idx differs with cam_norm on vs off"
    assert torch.equal(a["cpm_objects"], b["cpm_objects"]), f"idx {idx}: cpm_objects differs with cam_norm on vs off"
    assert torch.equal(a["gt_traj"], b["gt_traj"]), f"idx {idx}: gt_traj differs with cam_norm on vs off"
    assert torch.equal(a["gt_mask"], b["gt_mask"]), f"idx {idx}: gt_mask differs with cam_norm on vs off"
    assert torch.equal(a["points"], b["points"]), f"idx {idx}: points differs with cam_norm on vs off"
    if not torch.equal(a["cam"], b["cam"]):
        n_cam_differs += 1
    # sanity: the "on" cam should equal applying the transform to the "off" (raw) cam
    expected = (a["cam"] - mean) / std
    assert torch.allclose(b["cam"], expected, atol=1e-6), f"idx {idx}: cam_norm transform mismatch"

assert n_cam_differs == N, f"only {n_cam_differs}/{N} samples had cam actually change -- cam_norm may not be applied"
print(f"OK: {N}/{N} samples have identical nearby_idx/cpm_objects/gt_traj/gt_mask/points with cam_norm on vs off, "
      f"and cam differs in all {n_cam_differs}, matching the expected (raw-mean)/std transform exactly.")
