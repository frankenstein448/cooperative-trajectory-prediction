"""
verify_drop_cam_xy.py
For 50 random dataset indices, compares run C's pipeline (cam_norm from
runs/C_seed0/cam_norm.json, drop_cam_xy off) against run D's (same cam_norm,
drop_cam_xy on). Asserts every returned field is identical except cam[0:2],
which must be exactly 0 under D -- i.e. the flag touches nothing else, and
the nearest-lane lookup still sees raw x,y. Read-only, no training. A single
PilotDataset instance is used, toggling dataset.drop_cam_xy between calls.
"""
import json
import random
import torch

from pilot_dataset import PilotDataset

random.seed(0)
N = 50
C_RUN_DIR = "runs/C_seed0"

with open(f"{C_RUN_DIR}/config.json") as f:
    c_config = json.load(f)
with open(f"{C_RUN_DIR}/cam_norm.json") as f:
    cam_norm = json.load(f)

dataset = PilotDataset(use_image_cache=c_config["use_image_cache"],
                        require_prev_frame=c_config["require_prev_frame"],
                        use_speed=c_config["use_speed"], cam_norm=cam_norm)
idxs = random.sample(range(len(dataset)), N)


def same(a, b):
    """Deep equality over tensors / lists / scalars."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


dataset.drop_cam_xy = False
items_c = [dataset[i] for i in idxs]
dataset.drop_cam_xy = True
items_d = [dataset[i] for i in idxs]

n_xy_differs = 0
for idx, c, d in zip(idxs, items_c, items_d):
    assert c.keys() == d.keys(), f"idx {idx}: returned keys differ"
    for k in c:
        if k == "cam":
            continue
        assert same(c[k], d[k]), f"idx {idx}: {k} differs between C and D pipelines"
    assert torch.equal(c["cam"][2:], d["cam"][2:]), f"idx {idx}: cam[2:] (heading/speed) differs"
    assert torch.equal(d["cam"][0:2], torch.zeros(2)), f"idx {idx}: D's cam[0:2] is {d['cam'][0:2]}, expected 0"
    if not torch.equal(c["cam"][0:2], d["cam"][0:2]):
        n_xy_differs += 1

assert n_xy_differs == N, f"only {n_xy_differs}/{N} samples had cam[0:2] change -- drop_cam_xy may not be applied"
print(f"OK: {N}/{N} samples identical between C and D pipelines in every returned field "
      f"({', '.join(k for k in items_c[0] if k != 'cam')}) and cam[2:]; "
      f"cam[0:2] differs in all {n_xy_differs} and is exactly 0 under D.")
