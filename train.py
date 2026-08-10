"""
train.py
Training loop with train/val split over PilotDataset (Town05, 38 scenes,
10,640 samples) -> FourStreamFusion -> WinnerTakesAllLoss -> optimizer step.
Stream encoders (ego/coop/v2x/map) stay frozen -- only FourStreamFusion's
fuse + trajectory head layers are trained.

Train/val split: 85% train, 15% val. Val loss is measured every epoch but
never backpropagated -- it's the honest signal for whether the model
generalizes rather than just memorizing the training samples.

Run in gpu-test-venv.
"""
import torch
from torch.utils.data import DataLoader, random_split

from pilot_dataset import PilotDataset, DEVICE
from four_stream_fusion import FourStreamFusion
from trajectory_loss import WinnerTakesAllLoss

BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-3
VAL_FRACTION = 0.15
SEED = 42


def run_epoch(model, loader, loss_fn, optimizer=None):
    """optimizer=None -> validation mode (no backward, no weight update)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_reg, total_cls, n_batches = 0.0, 0.0, 0.0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            ego = batch["ego_embed"].to(DEVICE)
            coop = batch["coop_embed"].to(DEVICE)
            v2x = batch["v2x_embed"].to(DEVICE)
            map_ = batch["map_embed"].to(DEVICE)
            gt_traj = batch["gt_traj"].to(DEVICE)
            gt_mask = batch["gt_mask"].to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            fused, trajs, conf = model(ego, coop, v2x, map_)
            loss, info = loss_fn(trajs, conf, gt_traj, gt_mask)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_reg += info["reg_loss"]
            total_cls += info["cls_loss"]
            n_batches += 1

    return total_loss / n_batches, total_reg / n_batches, total_cls / n_batches


def main():
    print(f"Using device: {DEVICE}")

    dataset = PilotDataset()
    print(f"Dataset size: {len(dataset)} samples")

    n_val = int(len(dataset) * VAL_FRACTION)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    print(f"Train: {n_train} samples | Val: {n_val} samples")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = FourStreamFusion().to(DEVICE)
    loss_fn = WinnerTakesAllLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        train_loss, train_reg, train_cls = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_reg, val_cls = run_epoch(model, val_loader, loss_fn, optimizer=None)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} (reg={train_reg:.4f}, cls={train_cls:.4f}) | "
              f"val_loss={val_loss:.4f} (reg={val_reg:.4f}, cls={val_cls:.4f})")

    print("\nTraining complete.")
    torch.save(model.state_dict(), "pilot_model.pt")
    print("Saved model to pilot_model.pt")


if __name__ == "__main__":
    main()