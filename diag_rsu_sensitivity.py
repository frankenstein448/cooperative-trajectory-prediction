"""
diag_rsu_sensitivity.py
Read-only diagnostic (no verdict; runs/DECISION_RULES_RSU.md is not changed by it): how much does a trained
E_V2X+RSU model (use_rsu=True) depend on its RSU input? On the TUNING split, every batch is run three times:
  - real:    the sample's own RSU input (SEMLIDAR_TOP_id_0 points + 4 RSU camera views of its scene / frame)
  - zeroed:  the same tensors set to 0 (points zeros_like: same point count, every coordinate / feature 0;
             images zeros_like, i.e. black in load_image's [0, 1] range)
  - swapped: the RSU input of a partner sample from ANOTHER scene of the SAME town (its own (scene, frame) files,
             loaded the way PilotDataset does: load_semantic_lidar + load_rsu_images)
Everything else in the batch (ego sensors, coop, CAM, targets, map) is untouched. The RSU input is only
batch["rsu_points"] / batch["rsu_images"] (env_model.EnvFusion.scene_tokens), and RSU points are in the RSU's own
sensor frame (not ego-relative), so replacing those two entries is the whole RSU input.

Partner pool: all NON-TEST env samples (train + tuning) of the same town whose scene differs from the sample's own
(Town03 and Town04 have one tuning scene each, as in diag_map_sensitivity.py). Test-split samples are never read.
Pairing is fixed: np.random.default_rng(0), one draw per tuning sample in tuning-split order. Checks: the batch's
(scene, frame) is the tuning sample's, the batch's RSU tensors equal the direct load of the sample's own files, and
the real pass's per-target ADE1 / minADE6 match <run-dir>/eval_env_tuning_<ckpt stem>.npz (eval_env.py) by tid.

Reports, per set (moving V = moving targets some CAV sees; moving R = moving RSU-only targets; env_metrics
rsu_subset_masks), for zeroed and swapped: ADE1 and minADE6 real vs perturbed and the mean change, top-1 shift
(L2 between the real and perturbed top-scored trajectories, masked mean over valid future steps, metres, target
frame) and mode-flip rate (argmax mode changes). ALL known targets as context.

Run in gpu-test-venv:  python diag_rsu_sensitivity.py --run-dir runs/E_V2XRSU_seed0 2>&1 | tee diag_rsu_E_V2XRSU_seed0.log
"""
import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from env_dataset import EnvDataset, env_collate_fn
from env_model import EnvFusion
from env_metrics import rsu_subset_masks, subset_masks
from eval_env import collect, concat
from ego_sensor_encoder import load_semantic_lidar
from pilot_dataset import SWEEPS_ROOT, RSU_AGENT_ID, load_rsu_images
from scene_split import three_way_scene_split, split_version_of

PAIRING_SEED = 0

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--ckpt", default=None, help="defaults to <run-dir>/checkpoint_epoch12.pt")
args = ap.parse_args()
ckpt = args.ckpt or os.path.join(args.run_dir, "checkpoint_epoch12.pt")

with open(os.path.join(args.run_dir, "config.json")) as f:
    cfg = json.load(f)
assert cfg.get("task") == "env" and cfg.get("use_rsu", False), f"{args.run_dir} is not a use_rsu env run"
with open(os.path.join(args.run_dir, "cam_norm.json")) as f:
    cam_norm = json.load(f)
with open(os.path.join(args.run_dir, "env_norm.json")) as f:
    env_norm = json.load(f)
ds = EnvDataset(v2x=cfg["v2x"], use_map=not cfg["no_map"], map_transform=cfg["map_transform"], env_norm=env_norm,
                use_image_cache=cfg["use_image_cache"], use_speed=cfg["use_speed"], drop_cam_xy=cfg["drop_cam_xy"],
                use_rsu=True)
ds.cam_norm = cam_norm
assert ds.cache["manifest"]["git_commit"] == cfg["env_cache_commit"], "env cache differs from the run's"
train, tuning, _ = three_way_scene_split(ds, seed=42, split_version=split_version_of(cfg))  # never iterated: test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnvFusion(device=device, v2x=cfg["v2x"] == "on", use_map=not cfg["no_map"], use_rsu=True).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
model.eval()
print(f"ckpt={ckpt}; tuning split ({split_version_of(cfg)}): {len(tuning)} samples")


def rsu_input(scene, frame):
    """The RSU's (points, images) for (scene, frame), as PilotDataset.__getitem__ loads them."""
    points = load_semantic_lidar(f"{SWEEPS_ROOT}/SEMLIDAR_TOP_id_{RSU_AGENT_ID}/scene_{scene}_{frame:06d}.pcd.bin")
    return points, load_rsu_images(scene, frame, cfg["use_image_cache"])


# ---------- fixed pairing: tuning sample -> non-test partner, same town, other scene ----------
pool_by_town = {}
for i in sorted(train.indices + tuning.indices):
    pool_by_town.setdefault(ds.samples[i][3], []).append(i)
rng = np.random.default_rng(PAIRING_SEED)
partner_of = []
for i in tuning.indices:
    _, scene, _, town = ds.samples[i]
    cands = [j for j in pool_by_town[town] if ds.samples[j][1] != scene]
    assert cands, f"no other-scene partner for tuning idx {i} ({town}, scene {scene})"
    partner_of.append(cands[rng.integers(len(cands))])
print(f"Paired {len(partner_of)} tuning samples with non-test partners (same town, other scene), seed {PAIRING_SEED}")


def top1(trajs, logits, tm):
    """-> top-scored trajectory [N, 30, 2] and its index [N] of the real targets (batch order, as collect())."""
    tr, lg = trajs.detach().cpu().numpy()[tm], logits.detach().cpu().numpy()[tm]
    top = lg.argmax(1)
    return tr[np.arange(len(tr)), top], top


parts = {k: [] for k in ("real", "zero", "swap")}
shift = {k: [] for k in ("zero", "swap")}
flip = {k: [] for k in ("zero", "swap")}
pos = 0
with torch.no_grad():
    for batch in DataLoader(tuning, batch_size=8, shuffle=False, num_workers=2, collate_fn=env_collate_fn):
        B = len(batch["scene"])
        own = [ds.samples[i] for i in tuning.indices[pos:pos + B]]
        assert [(s, f) for _, s, f, _ in own] == list(zip(batch["scene"], batch["frame"])), "batch / tuning order"
        if pos == 0 or pos % 400 < B:  # spot check: the batch's RSU tensors are the sample's own files
            p0, im0 = rsu_input(batch["scene"][0], batch["frame"][0])
            assert torch.equal(p0, batch["rsu_points"][0]) and all(
                torch.equal(a, b) for a, b in zip(im0, batch["rsu_images"][0])), "rsu_input disagrees with the batch"
        swap = [rsu_input(ds.samples[j][1], ds.samples[j][2]) for j in partner_of[pos:pos + B]]
        pos += B

        variants = {
            "real": batch,
            "zero": dict(batch, rsu_points=[torch.zeros_like(p) for p in batch["rsu_points"]],
                         rsu_images=[[torch.zeros_like(im) for im in ims] for ims in batch["rsu_images"]]),
            "swap": dict(batch, rsu_points=[p for p, _ in swap], rsu_images=[ims for _, ims in swap]),
        }
        tm = batch["target_mask"].cpu().numpy()
        fut_mask = batch["t_fut_mask"].cpu().numpy()[tm].astype(bool)
        tops = {}
        for name, b in variants.items():
            trajs, logits = model(b)
            parts[name].append(collect(b, trajs, logits))
            tops[name] = top1(trajs, logits, tm)
        for name in ("zero", "swap"):
            d = np.linalg.norm(tops[name][0] - tops["real"][0], axis=-1)            # [N, 30]
            shift[name].append((d * fut_mask).sum(-1) / fut_mask.sum(-1))
            flip[name].append(tops[name][1] != tops["real"][1])

assert pos == len(tuning), f"consumed {pos} of {len(tuning)} tuning samples"
t = {k: concat(v) for k, v in parts.items()}
shift = {k: np.concatenate(v) for k, v in shift.items()}
flip = {k: np.concatenate(v) for k, v in flip.items()}
for k in ("zero", "swap"):
    assert np.array_equal(t[k]["tid"], t["real"]["tid"]), f"{k}: targets differ from the real pass"

# real pass == eval_env.py's tuning evaluation of this checkpoint
stem = os.path.splitext(os.path.basename(ckpt))[0]
ref = np.load(os.path.join(args.run_dir, f"eval_env_tuning_{stem}.npz"))
o_ref, o = np.argsort(ref["tid"]), np.argsort(t["real"]["tid"])
assert np.array_equal(ref["tid"][o_ref], t["real"]["tid"][o]), "targets differ from eval_env.py's npz"
for k in ("ade1", "minade6"):
    dmax = float(np.abs(ref[k][o_ref] - t["real"][k][o]).max())
    print(f"real pass vs eval_env_tuning_{stem}.npz: max |{k} diff| = {dmax:.2e}")
    assert dmax < 1e-3, f"real pass {k} does not reproduce eval_env.py's"

masks = subset_masks(t["real"])
masks.update(rsu_subset_masks(t["real"]))
print(f"\nOver {len(tuning)} tuning samples, {len(t['real']['tid'])} known targets "
      f"(real RSU input vs zeroed / swapped from another scene of the same town):")
for label, sub in (("moving V", "v_moving"), ("moving R", "r_moving"), ("ALL", "all")):
    m = masks[sub]
    print(f"\n  [{label}] n={int(m.sum())}")
    if not m.any():
        continue
    r = t["real"]
    for name in ("zero", "swap"):
        p = t[name]
        print(f"    {name:4s}: ADE1 {r['ade1'][m].mean():.3f} -> {p['ade1'][m].mean():.3f} "
              f"(delta {(p['ade1'][m] - r['ade1'][m]).mean():+.3f})  "
              f"minADE6 {r['minade6'][m].mean():.3f} -> {p['minade6'][m].mean():.3f} "
              f"(delta {(p['minade6'][m] - r['minade6'][m]).mean():+.3f})  "
              f"top-1 shift {shift[name][m].mean():.3f} m (median {np.median(shift[name][m]):.3f})  "
              f"mode-flip {flip[name][m].mean():.4f}")
