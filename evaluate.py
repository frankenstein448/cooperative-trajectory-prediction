"""
evaluate.py
Computes ADE / FDE on the held-out validation split, using the trained
END-TO-END checkpoint (checkpoints_e2e/best_model.pt).

Reports two views:
  - min-ADE / min-FDE (over 6 modes)  -- best-case metric, standard in
    multimodal trajectory prediction papers, comparable to published baselines
  - Top-1 ADE / FDE (confidence-weighted) -- only the mode the model is most
    confident in

Uses the SAME train/val split (same seed) as train.py, so this evaluates on
samples the model never trained on.

Run in gpu-test-venv.
"""
import torch
from torch.utils.data import DataLoader, random_split

from pilot_dataset import PilotDataset, pilot_collate_fn
from end_to_end_model import EndToEndFusion

BATCH_SIZE = 8
VAL_FRACTION = 0.15
SEED = 42
NUM_WORKERS = 4
CHECKPOINT_PATH = "checkpoints_e2e/best_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_displacement(pred_traj, gt_traj, gt_mask):
    gt_expanded = gt_traj.unsqueeze(1)
    mask_expanded = gt_mask.unsqueeze(1).unsqueeze(-1)
    sq_err = ((pred_traj - gt_expanded) ** 2) * mask_expanded
    per_step_dist = sq_err.sum(dim=-1).sqrt()
    return per_step_dist


def main():
    print(f"Using device: {DEVICE}")

    dataset = PilotDataset()
    n_val = int(len(dataset) * VAL_FRACTION)
    n_train = len(dataset) - n_val
    _, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    print(f"Evaluating on {len(val_set)} held-out validation samples")

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=pilot_collate_fn)

    model = EndToEndFusion(device=DEVICE).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f} at save time)")

    all_min_ade, all_min_fde = [], []
    all_top1_ade, all_top1_fde = [], []

    with torch.no_grad():
        for batch in val_loader:
            gt_traj = batch["gt_traj"].to(DEVICE)
            gt_mask = batch["gt_mask"].to(DEVICE)

            fused, trajs, conf_logits = model(batch)
            per_step_dist = compute_displacement(trajs, gt_traj, gt_mask)

            valid_counts = gt_mask.sum(dim=-1).clamp(min=1)
            ade_per_mode = per_step_dist.sum(dim=-1) / valid_counts.unsqueeze(1)

            B, n_modes, horizon = per_step_dist.shape
            last_valid_idx = gt_mask.float().cumsum(dim=-1).argmax(dim=-1)
            fde_per_mode = per_step_dist[torch.arange(B).unsqueeze(1), torch.arange(n_modes).unsqueeze(0), last_valid_idx.unsqueeze(1)]

            min_ade = ade_per_mode.min(dim=-1).values
            min_fde = fde_per_mode.min(dim=-1).values

            top1_idx = conf_logits.argmax(dim=-1)
            top1_ade = ade_per_mode[torch.arange(B), top1_idx]
            top1_fde = fde_per_mode[torch.arange(B), top1_idx]

            all_min_ade.append(min_ade)
            all_min_fde.append(min_fde)
            all_top1_ade.append(top1_ade)
            all_top1_fde.append(top1_fde)

    min_ade = torch.cat(all_min_ade)
    min_fde = torch.cat(all_min_fde)
    top1_ade = torch.cat(all_top1_ade)
    top1_fde = torch.cat(all_top1_fde)

    print("\n=== Evaluation Results (End-to-End Model, held-out validation set) ===")
    print(f"min-ADE (best of 6 modes):  {min_ade.mean().item():.3f} m")
    print(f"min-FDE (best of 6 modes):  {min_fde.mean().item():.3f} m")
    print(f"Top-1 ADE (by confidence):  {top1_ade.mean().item():.3f} m")
    print(f"Top-1 FDE (by confidence):  {top1_fde.mean().item():.3f} m")

    gap_ade = top1_ade.mean().item() - min_ade.mean().item()
    print(f"\nConfidence-head gap (Top-1 ADE - min-ADE): {gap_ade:.3f} m")


if __name__ == "__main__":
    from coord_transform import warn_legacy_script
    warn_legacy_script("evaluate.py")
    main()