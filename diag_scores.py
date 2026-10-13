import argparse
import math
import torch
from torch.utils.data import DataLoader

from pilot_dataset import PilotDataset, pilot_collate_fn
from visualize_predictions import load_model
from scene_split import scene_split
from coord_transform import warn_legacy_script
warn_legacy_script("diag_scores.py")

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--no-map", action="store_true")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PilotDataset()
_, val_set = scene_split(dataset, 0.15, 42)
loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=pilot_collate_fn)
model = load_model(args.ckpt, use_map=not args.no_map).to(DEVICE).eval()
print(f"ckpt={args.ckpt}  use_map={not args.no_map}")

acc = {k: [] for k in ("best", "amax", "amin", "mean", "rowsum", "smin", "smax", "ent", "best_idx", "amax_idx")}
K = None
with torch.no_grad():
    for batch in loader:
        gt = batch["gt_traj"].to(DEVICE)
        mask = batch["gt_mask"].to(DEVICE).float()
        out = model(batch)
        if K is None:
            for i, o in enumerate(out):
                print(f"output[{i}]: {tuple(o.shape) if torch.is_tensor(o) else type(o).__name__}")
        trajs, s = out[1], out[2].float()
        K = trajs.shape[1]
        if tuple(s.shape) != tuple(trajs.shape[:2]):
            raise RuntimeError(f"scores {tuple(s.shape)} vs trajs {tuple(trajs.shape)}")
        cnt = mask.sum(-1).clamp(min=1)
        dist = torch.norm(trajs - gt.unsqueeze(1), dim=-1)
        ade = (dist * mask.unsqueeze(1)).sum(-1) / cnt.unsqueeze(1)   # [B, K]
        v = mask.sum(-1) > 0
        ade, s = ade[v], s[v]
        is_prob = bool((s >= 0).all()) and torch.allclose(s.sum(1), torch.ones_like(s[:, 0]), atol=1e-3)
        p = s if is_prob else s.softmax(1)
        acc["best"].append(ade.min(1).values.cpu())
        acc["amax"].append(ade.gather(1, s.argmax(1, keepdim=True)).squeeze(1).cpu())
        acc["amin"].append(ade.gather(1, s.argmin(1, keepdim=True)).squeeze(1).cpu())
        acc["mean"].append(ade.mean(1).cpu())
        acc["rowsum"].append(s.sum(1).cpu())
        acc["smin"].append(s.min(1).values.cpu())
        acc["smax"].append(s.max(1).values.cpu())
        acc["ent"].append((-(p * p.clamp_min(1e-12).log()).sum(1) / math.log(K)).cpu())
        acc["best_idx"].append(ade.argmin(1).cpu())
        acc["amax_idx"].append(s.argmax(1).cpu())

a = {k: torch.cat(v) for k, v in acc.items()}
n = len(a["best"])
print(f"\nvalid samples: {n}, K={K}")
print(f"score range [{a['smin'].min():.3f}, {a['smax'].max():.3f}], mean row sum {a['rowsum'].mean():.3f} "
      f"(1.0 = probabilities; otherwise logits or something else)")
print(f"normalized entropy of mode distribution: {a['ent'].mean():.3f} (1.0 = uniform, 0 = one-hot)")
print(f"\nADE best mode (minADE{K}): {a['best'].mean():.3f} m")
print(f"ADE argmax-score mode:     {a['amax'].mean():.3f} m")
print(f"ADE argmin-score mode:     {a['amin'].mean():.3f} m")
print(f"ADE averaged over modes:   {a['mean'].mean():.3f} m  (= expected ADE of a random mode)")
print(f"\nargmax picks best mode: {(a['amax_idx'] == a['best_idx']).float().mean():.3f}  (chance = {1/K:.3f})")
amin_hit = 0.0  # computed below from ADE equality
print(f"best-mode slot histogram:   {torch.bincount(a['best_idx'], minlength=K).tolist()}")
print(f"argmax-score slot histogram: {torch.bincount(a['amax_idx'], minlength=K).tolist()}")
