"""
train.py
End-to-end training: all encoders (ego, coop, v2x, map) are now trainable,
not frozen. Uses EndToEndFusion (end_to_end_model.py) and PilotDataset's
custom collate_fn (variable-size LiDAR/camera/coop data per sample).

Run in gpu-test-venv.
"""
import os
import json
import time
import random
import argparse
import subprocess
import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader

from pilot_dataset import PilotDataset, pilot_collate_fn
from end_to_end_model import EndToEndFusion
from trajectory_loss import build_loss, RELAXED_WTA_EPS
from scene_split import three_way_scene_split

BATCH_SIZE = 8
LR = 1e-4          # lower than the frozen-encoder run (1e-3) -- more parameters now training, needs more caution
NUM_WORKERS = 2    # this box has 2 CPU cores; 4 workers measurably slowed earlier runs (see train_ablation.py)
SCENE_SPLIT_SEED = 42  # fixed protocol seed for three_way_scene_split -- NOT the experiment's --seed.
                       # Changing this would reshuffle which scenes are train/tuning/test; the tuning and
                       # test scenes are locked (see scene_split.py's KNOWN_TUNING_SCENES guard and
                       # test_scenes.json) and must stay identical across every run regardless of --seed.
REQUIRE_PREV_FRAME = True  # always on for this experiment -- drops the 216 scene/agent-first-frame samples
                           # that have no frame-1, so --use-speed and the baseline see the same dataset.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_commit():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir).decode().strip()
    except Exception as e:
        return f"<unavailable: {e}>"


def run_epoch(model, loader, loss_fn, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_reg, total_cls, n_batches = 0.0, 0.0, 0.0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            gt_traj = batch["gt_traj"].to(DEVICE)
            gt_mask = batch["gt_mask"].to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            fused, trajs, conf = model(batch)
            loss, info = loss_fn(trajs, conf, gt_traj, gt_mask)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_reg += info["reg_loss"]
            total_cls += info["cls_loss"]
            n_batches += 1

    return total_loss / n_batches, total_reg / n_batches, total_cls / n_batches


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                         help="seeds torch/numpy/random for model init and shuffling -- NOT the scene split, "
                              "which is always SCENE_SPLIT_SEED regardless of this value")
    parser.add_argument("--use-speed", action="store_true")
    parser.add_argument("--use-image-cache", action="store_true")
    parser.add_argument("--normalize-cam", action="store_true",
                         help="z-score cam (x,y,heading,speed) using train-split stats, saved to "
                              "<out-dir>/cam_norm.json; applied to the returned sample only, never to lookups")
    parser.add_argument("--drop-cam-xy", action="store_true",
                         help="after cam normalization, zero cam[0:2] (x,y) in the returned sample only; "
                              "lookups keep raw cam. Requires --normalize-cam")
    parser.add_argument("--loss", choices=["wta", "relaxed_wta"], default="wta",
                         help=f"wta: pure winner-takes-all (default). relaxed_wta: winner regression weight 1-eps, "
                              f"each other mode eps/(K-1), eps={RELAXED_WTA_EPS} fixed")
    parser.add_argument("--no-map", action="store_true",
                         help="EndToEndFusion(use_map=False): map stream removed, for the ablation "
                              "(replaces train_ablation.py for new runs)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dry-run", action="store_true",
                         help="build dataset/splits, write config.json, run 2 train + 2 val batches, "
                              "then exit without saving any checkpoint")
    args = parser.parse_args()
    assert not args.drop_cam_xy or args.normalize_cam, "--drop-cam-xy requires --normalize-cam"
    return args


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Using device: {DEVICE}")
    os.makedirs(args.out_dir, exist_ok=True)

    config = {
        "seed": args.seed,
        "use_speed": args.use_speed,
        "use_image_cache": args.use_image_cache,
        "normalize_cam": args.normalize_cam,
        "drop_cam_xy": args.drop_cam_xy,
        "loss": args.loss,
        "relaxed_wta_eps": RELAXED_WTA_EPS if args.loss == "relaxed_wta" else None,
        "no_map": args.no_map,
        "epochs": args.epochs,
        "out_dir": args.out_dir,
        "require_prev_frame": REQUIRE_PREV_FRAME,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "num_workers": NUM_WORKERS,
        "scene_split_seed": SCENE_SPLIT_SEED,
        "git_commit": git_commit(),
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote config: {config}")

    dataset = PilotDataset(use_image_cache=args.use_image_cache,
                            require_prev_frame=REQUIRE_PREV_FRAME,
                            use_speed=args.use_speed,
                            drop_cam_xy=args.drop_cam_xy)
    print(f"Dataset size: {len(dataset)} samples")

    train_set, tuning_set, test_set = three_way_scene_split(dataset, seed=SCENE_SPLIT_SEED)
    print(f"Train: {len(train_set)} samples | Tuning: {len(tuning_set)} samples "
          f"| Test: {len(test_set)} samples (never used here)")

    if args.normalize_cam:
        cam_values = torch.stack([dataset.get_cam_features(*dataset.samples[i][:3])
                                   for i in train_set.indices])
        mean = cam_values.mean(0)
        std = cam_values.std(0) + 1e-6
        cam_norm = {"mean": mean.tolist(), "std": std.tolist()}
        with open(os.path.join(args.out_dir, "cam_norm.json"), "w") as f:
            json.dump(cam_norm, f, indent=2)
        dataset.cam_norm = cam_norm  # settable post-construction; see PilotDataset.__init__
        print(f"Computed cam_norm from {len(train_set.indices)} train samples: {cam_norm}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, collate_fn=pilot_collate_fn)
    val_loader = DataLoader(tuning_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=pilot_collate_fn)

    model = EndToEndFusion(device=DEVICE, use_map=not args.no_map).to(DEVICE)
    loss_fn = build_loss(args.loss)
    print(f"Model use_map={not args.no_map}, loss={type(loss_fn).__name__}"
          + (f" (eps={loss_fn.eps})" if args.loss == "relaxed_wta" else ""))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    if args.dry_run:
        print("DRY RUN: 2 train batches + 2 val batches, no checkpoints will be saved")
        model.train()
        for i, batch in enumerate(train_loader):
            if i >= 2:
                break
            gt_traj = batch["gt_traj"].to(DEVICE)
            gt_mask = batch["gt_mask"].to(DEVICE)
            optimizer.zero_grad()
            fused, trajs, conf = model(batch)
            loss, info = loss_fn(trajs, conf, gt_traj, gt_mask)
            loss.backward()
            optimizer.step()
            print(f"  train batch {i}: loss={loss.item():.4f} reg={info['reg_loss']:.4f} cls={info['cls_loss']:.4f}")

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 2:
                    break
                gt_traj = batch["gt_traj"].to(DEVICE)
                gt_mask = batch["gt_mask"].to(DEVICE)
                fused, trajs, conf = model(batch)
                loss, info = loss_fn(trajs, conf, gt_traj, gt_mask)
                print(f"  val batch {i}: loss={loss.item():.4f} reg={info['reg_loss']:.4f} cls={info['cls_loss']:.4f}")

        print("DRY RUN complete -- no checkpoints saved, exiting.")
        return

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_reg, train_cls = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_reg, val_cls = run_epoch(model, val_loader, loss_fn, optimizer=None)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
              f"train_loss={train_loss:.4f} (reg={train_reg:.4f}, cls={train_cls:.4f}) | "
              f"val_loss={val_loss:.4f} (reg={val_reg:.4f}, cls={val_cls:.4f})")

        save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                         os.path.join(args.out_dir, f"checkpoint_epoch{epoch}.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                             os.path.join(args.out_dir, "best_model.pt"))
            print(f"  -> New best val_loss ({val_loss:.4f}), saved best_model.pt")

    print("\nTraining complete.")
    torch.save(model.state_dict(), os.path.join(args.out_dir, "final_model.pt"))
    print(f"Saved final model to {os.path.join(args.out_dir, 'final_model.pt')}")


if __name__ == "__main__":
    main()