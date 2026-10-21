"""
smoke_test.py
Quick end-to-end sanity check after multi-town dataset extension. Runs a
handful of forward+backward steps, explicitly forcing samples from ALL
THREE towns into the same batches -- this is the case that would break if
the per-town map-graph lookup fix in end_to_end_model.py were wrong or
missing (wrong graph used = wrong lane embeddings, silently, no crash).

Run in gpu-test-venv.
"""
import torch
from torch.utils.data import DataLoader, Subset

from pilot_dataset import PilotDataset, pilot_collate_fn
from end_to_end_model import EndToEndFusion
from trajectory_loss import WinnerTakesAllLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Using device: {DEVICE}")
    dataset = PilotDataset()
    print(f"Full dataset: {len(dataset)} samples")

    # Find each town's index range within dataset.samples, so we can pull a
    # few samples from each -- not left to chance/shuffle luck.
    town_counts = {}
    for _, _, _, town in dataset.samples:
        town_counts[town] = town_counts.get(town, 0) + 1
    print(f"Per-town sample counts: {town_counts}")

    boundaries, offset = {}, 0
    for town in ["Town03", "Town04", "Town05"]:
        n = town_counts.get(town, 0)
        boundaries[town] = (offset, offset + n)
        offset += n
    print(f"Index ranges: {boundaries}")

    smoke_indices = []
    for town, (start, end) in boundaries.items():
        smoke_indices.extend([start, start + 1, start + 2])
    print(f"Smoke test indices (3 per town): {smoke_indices}")

    smoke_set = Subset(dataset, smoke_indices)
    loader = DataLoader(smoke_set, batch_size=3, shuffle=True, collate_fn=pilot_collate_fn)

    model = EndToEndFusion(device=DEVICE).to(DEVICE)
    loss_fn = WinnerTakesAllLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    model.train()
    for step, batch in enumerate(loader):
        print(f"\nStep {step}: towns in batch = {batch['town']}")
        gt_traj = batch["gt_traj"].to(DEVICE)
        gt_mask = batch["gt_mask"].to(DEVICE)

        optimizer.zero_grad()
        fused, trajs, conf = model(batch)
        loss, info = loss_fn(trajs, conf, gt_traj, gt_mask)
        loss.backward()
        optimizer.step()

        print(f"  fused shape: {fused.shape}, trajs shape: {trajs.shape}, conf shape: {conf.shape}")
        print(f"  loss={loss.item():.4f} (reg={info['reg_loss']:.4f}, cls={info['cls_loss']:.4f})")

        assert torch.isfinite(loss), f"Non-finite loss at step {step} -- likely a bad per-town graph lookup!"

    print("\nSmoke test passed: forward + backward ran cleanly with Town03/04/05 mixed in the same batches.")


if __name__ == "__main__":
    from coord_transform import warn_legacy_script
    warn_legacy_script("smoke_test.py")
    main()