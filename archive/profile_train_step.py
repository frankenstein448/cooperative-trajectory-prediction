"""
profile_train_step.py
Read-only timing over ~50 train-split batches: DataLoader wait, host->device
copy (via torch.profiler, since H2D calls are interleaved inside forward --
see end_to_end_model.py's per-sample .to(device) calls), forward, backward.
No optimizer.step() is ever called and nothing is saved -- this does not
train the model, only measures it.
"""
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader

from pilot_dataset import PilotDataset, pilot_collate_fn
from end_to_end_model import EndToEndFusion
from trajectory_loss import WinnerTakesAllLoss
from scene_split import scene_split
from coord_transform import warn_legacy_script
warn_legacy_script("profile_train_step.py")

N_BATCHES = 50
WARMUP = 5
BATCH_SIZE = 8
NUM_WORKERS = 2  # matches what we will actually train with on this 2-core box

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE.type == "cuda" else [])

dataset = PilotDataset(use_image_cache=True)
train_set, _ = scene_split(dataset, 0.15, 42)
loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=NUM_WORKERS, collate_fn=pilot_collate_fn)

model = EndToEndFusion(device=DEVICE).to(DEVICE)
loss_fn = WinnerTakesAllLoss()

wait_times = []
it = iter(loader)

with profile(activities=activities, record_shapes=False) as prof:
    for step in range(N_BATCHES):
        t0 = time.perf_counter()
        batch = next(it)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        wait_times.append(time.perf_counter() - t0)

        with record_function("step/forward"):
            gt_traj = batch["gt_traj"].to(DEVICE)
            gt_mask = batch["gt_mask"].to(DEVICE)
            fused, trajs, conf = model(batch)  # per-sample H2D copies happen inside here too

        with record_function("step/backward"):
            loss, info = loss_fn(trajs, conf, gt_traj, gt_mask)
            loss.backward()

        model.zero_grad(set_to_none=True)  # discard grads -- optimizer.step() is never called

wait_times = wait_times[WARMUP:]
print(f"DataLoader wait: mean {sum(wait_times)/len(wait_times)*1000:.1f} ms over {len(wait_times)} batches")

table = prof.key_averages()
print(table.table(sort_by="cpu_time_total", row_limit=20))

h2d_keys = ("aten::to", "aten::copy_", "cudaMemcpyAsync", "Memcpy")
h2d_us = sum(e.cpu_time_total for e in table if any(k in e.key for k in h2d_keys))
fwd_us = sum(e.cpu_time_total for e in table if e.key == "step/forward")   # includes nested H2D
bwd_us = sum(e.cpu_time_total for e in table if e.key == "step/backward")
print(f"\nhost->device copy (summed aten::to/copy_/memcpy, nested inside forward): {h2d_us/1000/N_BATCHES:.1f} ms/batch")
print(f"forward (record_function total, includes the H2D above):  {fwd_us/1000/N_BATCHES:.1f} ms/batch")
print(f"backward (record_function total):                         {bwd_us/1000/N_BATCHES:.1f} ms/batch")
