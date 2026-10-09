import json
import os
import numpy as np
from collections import Counter
from torch.utils.data import Subset

KNOWN_TUNING_SCENES = {
    "Town03": {4},
    "Town04": {96},
    "Town05": {16, 41, 42, 51, 65, 73},
}


def scene_split(dataset, val_fraction=0.15, seed=42):
    """Hold out whole scenes: per town, at least one scene, about val_fraction of its scenes."""
    rng = np.random.default_rng(seed)
    by_town = {}
    for i, (agent_id, scene, frame, town) in enumerate(dataset.samples):
        by_town.setdefault(town, {}).setdefault(scene, []).append(i)
    train_idx, val_idx = [], []
    for town, scenes in sorted(by_town.items()):
        ids = sorted(scenes)
        n_val = max(1, round(len(ids) * val_fraction))
        held = set(rng.choice(ids, size=n_val, replace=False).tolist())
        for s in ids:
            (val_idx if s in held else train_idx).extend(scenes[s])
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def three_way_scene_split(dataset, tuning_fraction=0.15, seed=42, test_scenes_path="test_scenes.json"):
    """Train / tuning / test, holding out whole scenes, stratified by town.

    Tuning is obtained by calling scene_split() itself -- not a re-implementation
    of its draw -- so it is guaranteed identical to the existing train/val split,
    with no risk of RNG-state drift between the two. Test is a second, disjoint
    per-town draw from what's left over, using an independent generator
    (seed + 1) so it can't perturb or be perturbed by the tuning draw, with the
    same scene count per town as tuning (not a fraction). test_scenes_path is
    written once, on first call; on later calls the file is loaded and the
    deterministic draw is asserted to match it -- it is never silently
    redrawn or overwritten.
    """
    train_full, tuning = scene_split(dataset, tuning_fraction, seed)

    tuning_scenes = {}
    for i in tuning.indices:
        _, scene, _, town = dataset.samples[i]
        tuning_scenes.setdefault(town, set()).add(scene)
    for town, known in KNOWN_TUNING_SCENES.items():
        assert tuning_scenes.get(town, set()) == known, (
            f"tuning split drifted for {town}: got {sorted(tuning_scenes.get(town, set()))}, "
            f"expected {sorted(known)}")

    remaining_by_town = {}
    for i in train_full.indices:
        _, scene, _, town = dataset.samples[i]
        remaining_by_town.setdefault(town, {}).setdefault(scene, []).append(i)

    rng_test = np.random.default_rng(seed + 1)
    test_scenes = {}
    for town in sorted(remaining_by_town):
        n_test = len(tuning_scenes.get(town, set()))
        ids = sorted(remaining_by_town[town])
        held = set(rng_test.choice(ids, size=n_test, replace=False).tolist())
        test_scenes[town] = sorted(held)

    if os.path.exists(test_scenes_path):
        with open(test_scenes_path) as f:
            on_disk = {k: sorted(v) for k, v in json.load(f).items()}
        assert on_disk == test_scenes, (
            f"test_scenes.json on disk {on_disk} does not match the deterministic draw {test_scenes}; "
            "refusing to overwrite -- investigate before touching the test split.")
        test_scenes = on_disk
    else:
        with open(test_scenes_path, "w") as f:
            json.dump(test_scenes, f, indent=2)

    train_idx, test_idx = [], []
    for town, scene_to_idx in remaining_by_town.items():
        held = set(test_scenes.get(town, []))
        for scene, idxs in scene_to_idx.items():
            (test_idx if scene in held else train_idx).extend(idxs)

    return Subset(dataset, train_idx), tuning, Subset(dataset, test_idx)


if __name__ == "__main__":
    from pilot_dataset import PilotDataset
    ds = PilotDataset()
    tr, va = scene_split(ds)
    print("train", len(tr), "| val", len(va))
    print("val samples per town:", dict(Counter(ds.samples[i][3] for i in va.indices)))
    print("held-out scenes:", sorted({(ds.samples[i][3], ds.samples[i][1]) for i in va.indices}))

    tr3, tuning3, test3 = three_way_scene_split(ds)
    print("\nthree-way: train", len(tr3), "| tuning", len(tuning3), "| test", len(test3))
    print("test samples per town:", dict(Counter(ds.samples[i][3] for i in test3.indices)))
    print("test scenes:", sorted({(ds.samples[i][3], ds.samples[i][1]) for i in test3.indices}))