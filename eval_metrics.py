"""
eval_metrics.py
Computes minADE6 / minFDE6 against the held-out validation split (same
split as training, reproduced via same SEED + VAL_FRACTION + random_split).

Convention note: best mode is selected as the one minimizing AVERAGE
displacement error across the horizon (oracle metric -- ignores predicted
confidence, which is a separate calibration question). minFDE6 reports that
SAME mode's final-timestep error, not an independently-selected best-FDE
mode. If V2XPnP's 1.35m/2.36m used independent selection per metric, our
FDE will read slightly higher than a strict apples-to-apples comparison --
worth checking their paper/code before treating this as final.

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
CHECKPOINT_PATH = "checkpoints_e2e/best_model.pt"


def compute_minade_minfde(trajs, gt_traj, gt_mask):
    """
    Returns per-sample minADE (mode chosen by min avg ADE) and BOTH:
      - min_fde_same_mode: FDE of the ADE-argmin mode (current convention)
      - min_fde_independent: FDE of whichever mode independently minimizes FDE
    Reporting both makes the comparison against V2XPnP's published numbers
    transparent about which convention is being used.
    """
    B, K, T, _ = trajs.shape
    gt_exp = gt_traj.unsqueeze(1).expand(-1, K, -1, -1)
    mask_exp = gt_mask.unsqueeze(1).expand(-1, K, -1)

    dist = torch.norm(trajs - gt_exp, dim=-1)                    # [B, K, T]
    valid_counts = mask_exp.sum(dim=-1).clamp(min=1)             # [B, K]
    ade_per_mode = (dist * mask_exp).sum(dim=-1) / valid_counts  # [B, K]

    best_mode_ade = ade_per_mode.argmin(dim=1)
    min_ade = ade_per_mode.gather(1, best_mode_ade.unsqueeze(1)).squeeze(1)

    last_valid_idx = gt_mask.sum(dim=-1).long().clamp(min=1) - 1
    fde_all_modes = dist.gather(2, last_valid_idx.view(B, 1, 1).expand(-1, K, 1)).squeeze(-1)  # [B, K]

    min_fde_same_mode = fde_all_modes.gather(1, best_mode_ade.unsqueeze(1)).squeeze(1)
    min_fde_independent = fde_all_modes.min(dim=1).values  # independently best FDE mode

    valid_sample = gt_mask.sum(dim=-1) > 0
    return (min_ade[valid_sample], min_fde_same_mode[valid_sample], min_fde_independent[valid_sample])


def main():
    print(f"Using device: {DEVICE}")
    dataset = PilotDataset()
    n_val = int(len(dataset) * VAL_FRACTION)
    n_train = len(dataset) - n_val
    _, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    print(f"Val set: {len(val_set)} samples (same split as training)")

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pilot_collate_fn)

    model = EndToEndFusion(device=DEVICE).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

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

    print(f"\n{'='*50}")
    print(f"minADE6:              {all_ade.mean().item():.3f}m  (V2XPnP: 1.35m)")
    print(f"minFDE6 (same mode):  {all_fde_same.mean().item():.3f}m  (V2XPnP: 2.36m)")
    print(f"minFDE6 (independent):{all_fde_indep.mean().item():.3f}m")
    print(f"{'='*50}")
    print(f"Evaluated on {len(all_ade)} valid samples")


if __name__ == "__main__":
    from coord_transform import warn_legacy_script
    warn_legacy_script("eval_metrics.py")
    main()