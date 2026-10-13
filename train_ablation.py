"""
train_ablation.py
Trains the no-map variant of EndToEndFusion for the map-stream ablation
study -- same data, same recipe, same seed/split as the full model, only
difference is use_map=False. Reuses run_epoch/save_checkpoint from train.py
so the training loop itself can't silently drift out of sync with the full
model's run.

NUM_WORKERS lowered to 2 here (vs train.py's 4) based on the DataLoader
warning seen in the Town05-only pilot run, where 4 workers on a 2-core
system roughly doubled epoch time from ~650s to ~1480s partway through.

Superseded for new runs by `train.py --no-map` (three-way scene split,
config.json, --run-dir-aware eval). Kept only to reproduce the old
checkpoints_scene_no_map run.

Run in gpu-test-venv.
"""
import os
import time
import torch
from torch.utils.data import DataLoader, random_split

from pilot_dataset import PilotDataset, pilot_collate_fn
from end_to_end_model import EndToEndFusion
from trajectory_loss import WinnerTakesAllLoss
from train import run_epoch, save_checkpoint  # reuse, don't duplicate

BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
VAL_FRACTION = 0.15
SEED = 42                # same seed -> same train/val split as the full model
NUM_WORKERS = 2
CHECKPOINT_EVERY = 5
CHECKPOINT_DIR = "checkpoints_scene_no_map"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Using device: {DEVICE}")
    print("Training ABLATION model: use_map=False")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataset = PilotDataset()
    print(f"Dataset size: {len(dataset)} samples")

    from scene_split import scene_split
    train_set, val_set = scene_split(dataset, VAL_FRACTION, SEED)
    n_train, n_val = len(train_set), len(val_set)
    print(f"Train: {n_train} samples | Val: {n_val} samples (same split as full model)")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, collate_fn=pilot_collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=pilot_collate_fn)

    model = EndToEndFusion(device=DEVICE, use_map=False).to(DEVICE)
    loss_fn = WinnerTakesAllLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,} (full model had 740,942)")

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_reg, train_cls = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_reg, val_cls = run_epoch(model, val_loader, loss_fn, optimizer=None)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{EPOCHS} ({elapsed:.1f}s) | "
              f"train_loss={train_loss:.4f} (reg={train_reg:.4f}, cls={train_cls:.4f}) | "
              f"val_loss={val_loss:.4f} (reg={val_reg:.4f}, cls={val_cls:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                             os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"  -> New best val_loss ({val_loss:.4f}), saved best_model.pt")

        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                             os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch}.pt"))
            print(f"  -> Periodic checkpoint saved (epoch {epoch})")

    print("\nAblation training complete.")
    torch.save(model.state_dict(), "e2e_model_no_map_scene.pt")
    print("Saved final model to e2e_model_no_map_scene.pt")


if __name__ == "__main__":
    from coord_transform import warn_legacy_script
    warn_legacy_script("train_ablation.py")
    main()