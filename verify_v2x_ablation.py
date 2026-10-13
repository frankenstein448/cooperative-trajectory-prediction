"""
verify_v2x_ablation.py
Checks for the V2X ablation flags (use_cpm / use_coop), before any ablation training.
Loads PilotDataset once, with D's settings.

1. Model: all flags at default == D (parameter count, strict load of
   runs/D_seed0/checkpoint_epoch12.pt). Each ablation's parameter count is D's
   minus exactly the removed module(s) and the shrunk fuse input(s).
2. Dataset: with use_cpm / use_coop off, the returned sample lacks exactly
   cpm_objects / coop_points+coop_images, and every other field is identical
   (20 samples). Each config runs a forward pass on a real 2-sample batch.
3. Own box in CPM: for 50 samples, the CPM object nearest the target agent
   (rel_x, rel_y = annotation minus the agent's own position, world frame).
   An object at ~0 m with car-like size would be the agent's own box.

Run in gpu-test-venv.
"""
import torch

from pilot_dataset import PilotDataset, pilot_collate_fn
from end_to_end_model import EndToEndFusion, EMBED_DIM
from coord_transform import warn_legacy_script
warn_legacy_script("verify_v2x_ablation.py")

D_CKPT = "runs/D_seed0/checkpoint_epoch12.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIGS = {"D": dict(use_cpm=True, use_coop=True), "NoCPM": dict(use_cpm=False, use_coop=True),
           "NoCoop": dict(use_cpm=True, use_coop=False), "EgoOnly": dict(use_cpm=False, use_coop=False)}
n_params = lambda m: sum(p.numel() for p in m.parameters())

# ---------- 1. models ----------
d = EndToEndFusion(device=DEVICE, use_map=True)
sd = torch.load(D_CKPT, map_location="cpu")["model_state_dict"]
n_d = sum(v.numel() for k, v in sd.items() if not k.startswith("graph_"))  # graph_* are buffers
models = {name: EndToEndFusion(device=DEVICE, use_map=True, **kw).to(DEVICE) for name, kw in CONFIGS.items()}
models["D"].load_state_dict(sd, strict=True)
assert n_params(d) == n_params(models["D"]) == n_d, (n_params(d), n_params(models["D"]), n_d)
print(f"1. flags at default: {n_d:,} parameters == D's checkpoint; loads {D_CKPT} with strict=True")

n_cpm = n_params(d.v2x_encoder.cpm_enc)
n_coop = n_params(d.coop_encoder)
fuse_v2x_shrink = EMBED_DIM * EMBED_DIM          # V2X fuse loses one 128-wide input block
fuse_top_shrink = EMBED_DIM * 2 * EMBED_DIM      # top fuse's first Linear loses one 128-wide input block
expected = {"NoCPM": n_d - n_cpm - fuse_v2x_shrink,
            "NoCoop": n_d - n_coop - fuse_top_shrink,
            "EgoOnly": n_d - n_cpm - fuse_v2x_shrink - n_coop - fuse_top_shrink}
for name, exp in expected.items():
    got = n_params(models[name])
    assert got == exp, f"{name}: {got} parameters, expected {exp}"
    assert hasattr(models[name].v2x_encoder, "cpm_enc") == CONFIGS[name]["use_cpm"]
    assert hasattr(models[name], "coop_encoder") == CONFIGS[name]["use_coop"]
    print(f"   {name:8s}: {got:,} parameters ({got - n_d:+,} vs D; removed module(s) and fuse inputs only)")

# ---------- 2. dataset fields + forward ----------
ds = PilotDataset(use_image_cache=True, require_prev_frame=True, use_speed=True)  # D's filter; cam_norm irrelevant here
idxs = [round(i * (len(ds) - 1) / 19) for i in range(20)]


def same(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


full_items = {i: ds[i] for i in idxs}
for name, kw in CONFIGS.items():
    ds.use_cpm, ds.use_coop = kw["use_cpm"], kw["use_coop"]  # plain attributes, read in __getitem__
    dropped = set()
    if not kw["use_cpm"]:
        dropped |= {"cpm_objects"}
    if not kw["use_coop"]:
        dropped |= {"coop_points", "coop_images"}
    for i in idxs:
        item = ds[i]
        assert set(item) == set(full_items[i]) - dropped, f"{name} idx {i}: keys {sorted(set(item))}"
        for k in item:
            assert same(item[k], full_items[i][k]), f"{name} idx {i}: field {k} differs from D"
    batch = pilot_collate_fn([ds[idxs[0]], ds[idxs[-1]]])
    with torch.no_grad():
        _, trajs, conf = models[name](batch)
    print(f"2. {name:8s}: {len(idxs)} samples drop exactly {sorted(dropped) or 'nothing'}, all other fields "
          f"identical; forward ok trajs {tuple(trajs.shape)} conf {tuple(conf.shape)}")
ds.use_cpm, ds.use_coop = True, True

# ---------- 3. own box in CPM ----------
print("\n3. Nearest CPM object to the target agent (50 samples):")
n_near = 0
nearest = []
cpm_idxs = [round(i * (len(ds) - 1) / 49) for i in range(50)]
for i in cpm_idxs:
    agent_id, scene, frame, _ = ds.samples[i]
    cpm = ds.loader.get_cpm(agent_id, scene, frame)
    if cpm.shape[0] == 0:
        nearest.append(float("inf"))
        continue
    dist = cpm[:, :2].norm(dim=1)
    j = int(dist.argmin())
    nearest.append(float(dist[j]))
    if dist[j] < 1.0:
        n_near += 1
        if n_near <= 5:
            print(f"   ({agent_id},{scene},{frame}): object at {dist[j]:.3f} m, size {cpm[j, 3:6].tolist()}, "
                  f"{cpm.shape[0]} objects")
near = torch.tensor(nearest)
finite = near[torch.isfinite(near)]
print(f"   samples with an object < 1.0 m: {n_near}/50 (< 0.1 m: {int((near < 0.1).sum())}); "
      f"nearest-object distance median {finite.median():.3f} m, min {finite.min():.3f} m; "
      f"{int((~torch.isfinite(near)).sum())} samples with no objects")
print("\nALL V2X ABLATION CHECKS PASSED")
