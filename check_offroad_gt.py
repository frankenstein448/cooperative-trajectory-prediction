"""
check_offroad_gt.py
Ground-truth off-road control per town (eval_scenes.py's SAFE control, without a model): the fraction of
valid GT future points (30 steps, agent-centric GT taken to CARLA with the agent's raw pose) farther than
OFFROAD_THRESHOLD_M from the nearest lane-centre point. A town above GT_OFFROAD_UNRELIABLE gets its off-road
numbers flagged UNRELIABLE in eval_scenes.py.

Variants: legacy (TOWN_TRANSFORMS + dense_waypoints, what Part 1 used), mirror with driving lanes only
(dense_waypoints), and mirror with all exported lanes (lane_polylines, + Town03 Parking) = what
safety_metrics.LaneCentrelines(kind="mirror") uses. D's sample list; split v2 tuning (what eval_scenes.py
reports) and v2 train. The v2 test split is not touched. Ground truth only, no model.

Run in gpu-test-venv:  python check_offroad_gt.py
"""
import collections

import numpy as np
from scipy.spatial import cKDTree

from pilot_dataset import PilotDataset
from scene_split import three_way_scene_split
from ground_truth_extractor import get_future_trajectory
from safety_metrics import LaneCentrelines, raw_pose, agent_to_carla, OFFROAD_THRESHOLD_M, GT_OFFROAD_UNRELIABLE

ds = PilotDataset(require_prev_frame=True, use_speed=True, use_cpm=False, use_coop=False)
tr, tu, _ = three_way_scene_split(ds, seed=42, split_version="v2")
towns = sorted(ds.town_graphs)
variants = {
    "legacy (Part 1)": ("legacy", LaneCentrelines(towns, "legacy")),
    "mirror, driving lanes only": ("mirror", LaneCentrelines(towns, "legacy")),  # dense_waypoints trees
    "mirror, all lanes (used)": ("mirror", LaneCentrelines(towns, "mirror")),
}
print(f"OFFROAD_THRESHOLD_M={OFFROAD_THRESHOLD_M}, UNRELIABLE above {GT_OFFROAD_UNRELIABLE:.0%}")
for split, sub in (("v2 tuning", tu), ("v2 train", tr)):
    off = {k: collections.Counter() for k in variants}
    tot = collections.Counter()
    for i in sub.indices:
        agent_id, scene, frame, town = ds.samples[i]
        gt, mask = get_future_trajectory(ds.loader, agent_id, scene, frame)
        vm = mask.numpy().astype(bool)
        pose = raw_pose(ds.loader, agent_id, scene, frame)
        tot[town] += int(vm.sum())
        for k, (kind, lanes) in variants.items():
            off[k][town] += int(lanes.offroad(town, agent_to_carla(gt.numpy()[vm], pose, town, kind)).sum())
    print(f"\n{split}: {len(sub.indices)} samples; GT off-road rate per town")
    print(f"  {'variant':28s} " + " ".join(f"{t:>9s}" for t in towns) + f" {'ALL':>9s}")
    for k in variants:
        rates = [off[k][t] / max(tot[t], 1) for t in towns]
        allr = sum(off[k].values()) / max(sum(tot.values()), 1)
        print(f"  {k:28s} " + " ".join(f"{r:9.4f}" for r in rates) + f" {allr:9.4f}  UNRELIABLE: "
              + (", ".join(t for t, r in zip(towns, rates) if r > GT_OFFROAD_UNRELIABLE) or "none"))
print("\nDONE")
