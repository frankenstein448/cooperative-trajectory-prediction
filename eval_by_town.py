"""
eval_by_town.py
minADE6 / minFDE6 on the held-out validation split (same SEED/split as
training), overall and per town.

Usage (gpu-test-venv, from ~/hd-maps):
    python eval_by_town.py --ckpt checkpoints_e2e/best_model.pt
    python eval_by_town.py --ckpt checkpoints_e2e_oldtransform/best_model.pt --old-transform
    python eval_by_town.py --ckpt checkpoints_e2e_no_map/best_model.pt --no-map

--old-transform overrides the Town05/Town04 constants in memory only
(coord_transform.py on disk is not modified). Use it to score a model under
the transform it was trained with, or to test how sensitive a model is to
the alignment at inference time.
"""
import argparse
import torch
from torch.utils.data import DataLoader, random_split

import coord_transform

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--no-map", action="store_true")
parser.add_argument("--old-transform", action="store_true")
parser.add_argument("--scene-split", action="store_true")
args = parser.parse_args()

if args.old_transform:
    coord_transform.TOWN_TRANSFORMS["Town05"].update(theta=70.5, tx=-39.00, ty=4.75)
    coord_transform.TOWN_TRANSFORMS["Town04"].update(theta=296.5, tx=-43.00, ty=73.00)

from pilot_dataset import PilotDataset, pilot_collate_fn
from visualize_predictions import load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
VAL_FRACTION = 0.15
SEED = 42


def per_sample_metrics(trajs, gt_traj, gt_mask):
    """Full-length per-sample tensors plus a validity mask (no filtering,
    so the output stays aligned with the sample order)."""
    B, K, T, _ = trajs.shape
    gt_exp = gt_traj.unsqueeze(1).expand(-1, K, -1, -1)
    mask_exp = gt_mask.unsqueeze(1).expand(-1, K, -1)
    dist = torch.norm(trajs - gt_exp, dim=-1)                    # [B, K, T]
    valid_counts = mask_exp.sum(dim=-1).clamp(min=1)
    ade_per_mode = (dist * mask_exp).sum(dim=-1) / valid_counts  # [B, K]
    best = ade_per_mode.argmin(dim=1)
    min_ade = ade_per_mode.gather(1, best.unsqueeze(1)).squeeze(1)
    last = gt_mask.sum(dim=-1).long().clamp(min=1) - 1
    fde_all = dist.gather(2, last.view(B, 1, 1).expand(-1, K, 1)).squeeze(-1)
    fde_same = fde_all.gather(1, best.unsqueeze(1)).squeeze(1)
    fde_indep = fde_all.min(dim=1).values
    valid = gt_mask.sum(dim=-1) > 0
    return min_ade, fde_same, fde_indep, valid


def main():
    ckpt = torch.load(args.ckpt, map_location="cpu")
    print(f"Checkpoint: {args.ckpt} | epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    print(f"map stream: {'off' if args.no_map else 'on'} | "
          f"transform: {'OLD (Town05/Town04 overridden)' if args.old_transform else 'current file'}")

    dataset = PilotDataset()
    if args.scene_split:
        from scene_split import scene_split
        _, val_set = scene_split(dataset, VAL_FRACTION, SEED)
    else:
        n_val = int(len(dataset) * VAL_FRACTION)
        _, val_set = random_split(dataset, [len(dataset) - n_val, n_val],
                                  generator=torch.Generator().manual_seed(SEED))
    towns = [dataset.samples[i][3] for i in val_set.indices]   # same order as the loader
    loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pilot_collate_fn)

    model = load_model(args.ckpt, use_map=not args.no_map).to(DEVICE).eval()

    ade, fs, fi, ok = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            gt_traj = batch["gt_traj"].to(DEVICE)
            gt_mask = batch["gt_mask"].to(DEVICE)
            _, trajs, _ = model(batch)
            a, s, i, v = per_sample_metrics(trajs, gt_traj, gt_mask)
            ade.append(a.cpu()); fs.append(s.cpu()); fi.append(i.cpu()); ok.append(v.cpu())
    ade, fs, fi, ok = (torch.cat(x) for x in (ade, fs, fi, ok))
    assert len(ade) == len(towns), "sample order mismatch"

    print(f"\n{'group':8s} {'n':>5s} {'minADE6':>9s} {'minFDE6(ind)':>13s} {'minFDE6(same)':>14s}")
    for name in ["ALL"] + sorted(set(towns)):
        sel = ok.clone()
        if name != "ALL":
            sel &= torch.tensor([t == name for t in towns])
        n = int(sel.sum())
        if n == 0:
            continue
        print(f"{name:8s} {n:5d} {ade[sel].mean():8.3f}m {fi[sel].mean():12.3f}m {fs[sel].mean():13.3f}m")


if __name__ == "__main__":
    main()