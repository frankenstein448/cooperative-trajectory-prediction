"""
eval_ablation.py
Evaluates the no-map ablation checkpoint on minADE6/minFDE6, using the
SAME val split as the full model (same SEED/VAL_FRACTION) for a direct
apples-to-apples comparison. Reports both FDE conventions, same as
eval_metrics.py, so results line up with what's already been reported.

Run in gpu-test-venv.
"""
import torch
from torch.utils.data import DataLoader, random_split

from pilot_dataset import PilotDataset, pilot_collate_fn
from end_to_end_model import EndToEndFusion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
VAL_FRACTION = 0.15
SEED = 42
CHECKPOINT_PATH = "checkpoints_e2e_no_map/best_model.pt"


def compute_minade_minfde(trajs, gt_traj, gt_mask):
    B, K, T, _ = trajs.shape
    gt_exp = gt_traj.unsqueeze(1).expand(-1, K, -1, -1)
    mask_exp = gt_mask.unsqueeze(1).expand(-1, K, -1)

    dist = torch.norm(trajs - gt_exp, dim=-1)
    valid_counts = mask_exp.sum(dim=-1).clamp(min=1)
    ade_per_mode = (dist * mask_exp).sum(dim=-1) / valid_counts

    best_mode_ade = ade_per_mode.argmin(dim=1)
    min_ade = ade_per_mode.gather(1, best_mode_ade.unsqueeze(1)).squeeze(1)

    last_valid_idx = gt_mask.sum(dim=-1).long().clamp(min=1) - 1
    fde_all_modes = dist.gather(2, last_valid_idx.view(B, 1, 1).expand(-1, K, 1)).squeeze(-1)

    min_fde_same_mode = fde_all_modes.gather(1, best_mode_ade.unsqueeze(1)).squeeze(1)
    min_fde_independent = fde_all_modes.min(dim=1).values

    valid_sample = gt_mask.sum(dim=-1) > 0
    return (min_ade[valid_sample], min_fde_same_mode[valid_sample], min_fde_independent[valid_sample])


def main():
    print(f"Using device: {DEVICE}")
    dataset = PilotDataset()
    n_val = int(len(dataset) * VAL_FRACTION)
    n_train = len(dataset) - n_val
    _, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    print(f"Val set: {len(val_set)} samples (same split as full model)")

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pilot_collate_fn)

    model = EndToEndFusion(device=DEVICE, use_map=False).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded ablation checkpoint: epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    all_ade, all_fde_same, all_fde_indep = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            gt_traj = batch["gt_traj"].to(DEVICE)
            gt_mask = batch["gt_mask"].to(DEVICE)
            _, trajs, conf = model(batch)
            ade, fde_same, fde_indep = compute_minade_minfde(trajs, gt_traj, gt_mask)
            all_ade.append(ade)
            all_fde_same.append(fde_same)
            all_fde_indep.append(fde_indep)

    all_ade = torch.cat(all_ade)
    all_fde_same = torch.cat(all_fde_same)
    all_fde_indep = torch.cat(all_fde_indep)

    print(f"\n{'='*55}")
    print(f"NO-MAP ABLATION RESULTS")
    print(f"minADE6:               {all_ade.mean().item():.3f}m  (full model: 1.297m)")
    print(f"minFDE6 (independent): {all_fde_indep.mean().item():.3f}m  (full model: 2.032m)")
    print(f"minFDE6 (same mode):   {all_fde_same.mean().item():.3f}m  (full model: 2.647m)")
    print(f"{'='*55}")
    print(f"Evaluated on {len(all_ade)} valid samples")


if __name__ == "__main__":
    main()