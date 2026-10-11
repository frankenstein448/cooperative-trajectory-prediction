"""
verify_history.py
Checks for the observed-history input (pilot_dataset.history_frames), before any training:

1. On 50 samples (evenly spaced over the H=8 dataset): history is [8, 2],
   history[-1] equals -(motion_utils.get_cv_velocity's frame-1 -> frame
   displacement, rotated into the current heading), and every parked agent
   (zero displacement) has an all-zero history -- reported, allowed.
2. Sample filter: how many samples H=8 drops, and that require_history_frames=8
   keeps exactly the same sample list.
3. Model: history off has the same parameter count as D and loads
   runs/D_seed0/checkpoint_epoch12.pt with strict=True; gru and transformer
   build and run a forward pass on a real 2-sample batch.

Run in gpu-test-venv.
"""
import torch

from pilot_dataset import PilotDataset, pilot_collate_fn
from motion_utils import get_cv_velocity
from end_to_end_model import EndToEndFusion

H = 8
N_CHECK = 50
D_CKPT = "runs/D_seed0/checkpoint_epoch12.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# one dataset (one metadata loader) in memory at a time -- each holds all of V2X-Sim's JSON metadata
import gc
base_samples = PilotDataset(use_image_cache=True, require_prev_frame=True, use_speed=True).samples
gc.collect()
ctrl = PilotDataset(use_image_cache=True, require_prev_frame=True, use_speed=True, require_history_frames=H)
ctrl_samples, ctrl_item_keys = ctrl.samples, set(ctrl[0])
del ctrl
gc.collect()
hist = PilotDataset(use_image_cache=True, require_prev_frame=True, use_speed=True, history_frames=H)
print(f"\nSamples: D filter {len(base_samples)}, history_frames={H} {len(hist)} "
      f"(dropped {len(base_samples) - len(hist)}), require_history_frames={H} {len(ctrl_samples)}")
assert hist.samples == ctrl_samples, "history_frames and require_history_frames must keep the same samples"
assert set(hist.samples) <= set(base_samples)
assert "history" not in ctrl_item_keys, "control must not return history"

# ---------- 1. history[-1] vs motion_utils ----------
idxs = [round(i * (len(hist) - 1) / (N_CHECK - 1)) for i in range(N_CHECK)]
max_err, n_parked = 0.0, 0
for i in idxs:
    agent_id, scene, frame, _ = hist.samples[i]
    h = hist.get_history(agent_id, scene, frame)
    assert h.shape == (H, 2), h.shape
    ok, vx, vy = get_cv_velocity(hist.loader, agent_id, scene, frame)
    assert ok
    err = (h[-1] - torch.tensor([-vx, -vy])).abs().max().item()
    max_err = max(max_err, err)
    if vx == 0.0 and vy == 0.0:
        n_parked += 1
        if not torch.all(h == 0):
            print(f"  note: ({agent_id},{scene},{frame}) is still at frame-1 -> frame, history not all-zero: "
                  f"max |history| = {h.abs().max():.2e} m")
        else:
            print(f"  parked ({agent_id},{scene},{frame}): all-zero history")
print(f"1. history[-1] == -(motion_utils displacement): max abs error over {N_CHECK} samples = {max_err:.2e} m; "
      f"{n_parked} with zero frame-1 -> frame displacement")
assert max_err < 1e-5, f"history[-1] mismatch: {max_err}"

# get_history ignores history_norm; __getitem__ returns the normalized copy
hist.history_norm = {"mean": [1.0, -2.0], "std": [2.0, 4.0]}
item = hist[idxs[0]]
raw = hist.get_history(*hist.samples[idxs[0]][:3])
assert torch.allclose(item["history"], (raw - torch.tensor([1.0, -2.0])) / torch.tensor([2.0, 4.0]))
print("   __getitem__ applies history_norm to the returned copy; get_history stays raw")
hist.history_norm = None

# ---------- 3. models ----------
n_params = lambda m: sum(p.numel() for p in m.parameters())
off = EndToEndFusion(device=DEVICE, use_map=True)
off_h0 = EndToEndFusion(device=DEVICE, use_map=True, history_frames=0, history_encoder="transformer")
sd = torch.load(D_CKPT, map_location="cpu")["model_state_dict"]
off_h0.load_state_dict(sd, strict=True)
n_d = sum(v.numel() for k, v in sd.items() if not k.startswith("graph_"))  # graph_* are buffers
assert n_params(off) == n_params(off_h0) == n_d, (n_params(off), n_params(off_h0), n_d)
print(f"3. history off: {n_params(off_h0):,} parameters == D's checkpoint ({n_d:,}); "
      f"loads {D_CKPT} with strict=True")

batch = pilot_collate_fn([hist[idxs[0]], hist[idxs[-1]]])
assert batch["history"].shape == (2, H, 2)
for kind in ("gru", "transformer"):
    m = EndToEndFusion(device=DEVICE, use_map=True, history_frames=H, history_encoder=kind).to(DEVICE)
    with torch.no_grad():
        _, trajs, conf = m(batch)
    extra = n_params(m) - n_d
    print(f"   history_frames={H}, {kind}: {n_params(m):,} parameters (+{extra:,} vs D), "
          f"forward ok trajs {tuple(trajs.shape)} conf {tuple(conf.shape)}")
print("\nALL HISTORY CHECKS PASSED")
