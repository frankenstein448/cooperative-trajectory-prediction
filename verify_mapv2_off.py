"""
verify_mapv2_off.py
With --map-version v1 (the default), the ego task must behave exactly as at --ref-commit, the commit C_D and
C_Dnomap were trained at (runs/DECISION_RULES_MAPV2_EGO.md): this also shows that the reused C_Dnomap runs and
their logs come from the same ego-task code. Runs the same probe in a `git archive` of --ref-commit and in the
working tree, each in its own process, and requires torch.equal for
  - every tensor of 50 random ego samples (seed 0) as pilot_collate_fn batches them (C_D's dataset: run config
    flags, cam_norm, image cache, map transform mirror),
  - the outputs (fused, trajs, conf logits) of C_D (--map-run) and C_Dnomap (--nomap-run) on those samples
    (epoch-12 checkpoints, CPU, eval mode, deterministic),
  - the initial weights of EndToEndFusion(use_map=True) and EndToEndFusion(use_map=False) built after
    torch.manual_seed(0) (same parameter names, shapes and values, so the init order is unchanged).
Read-only on runs/; no training. Prints RESULT: PASS / FAIL.

Run in gpu-test-venv:
  python verify_mapv2_off.py --ref-commit b25c711 --map-run runs/C_D_seed0 --nomap-run runs/C_Dnomap_seed0 \
      2>&1 | tee verify_mapv2_off.log
"""
import os
import sys
import argparse
import tempfile
import subprocess

import torch

N = 50
PROBE = r'''
import os, sys, json, random, zlib
import torch
torch.use_deterministic_algorithms(True)
from pilot_dataset import PilotDataset, pilot_collate_fn
from end_to_end_model import EndToEndFusion
map_run, nomap_run, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfgs = {r: json.load(open(os.path.join(r, "config.json"))) for r in (map_run, nomap_run)}
assert not cfgs[map_run]["no_map"] and cfgs[nomap_run]["no_map"], "map / no-map runs swapped"
for c in cfgs.values():
    assert c.get("task", "ego") == "ego" and "map_version" not in c and c["map_transform"] == "mirror", c
res = {}
for use_map in (True, False):
    torch.manual_seed(0)
    m = EndToEndFusion(device="cpu", use_map=use_map)
    for k, t in m.state_dict().items():
        res[f"init_map{int(use_map)}/{k}"] = t.clone()
c = cfgs[map_run]
ds = PilotDataset(use_image_cache=c["use_image_cache"], require_prev_frame=c["require_prev_frame"],
                  use_speed=c["use_speed"], cam_norm=json.load(open(os.path.join(map_run, "cam_norm.json"))),
                  drop_cam_xy=c["drop_cam_xy"], use_cpm=not c["no_cpm"], use_coop=not c["no_coop"],
                  map_transform=c["map_transform"])
models = {}
for name, r in (("map", map_run), ("nomap", nomap_run)):
    m = EndToEndFusion(device="cpu", use_map=not cfgs[r]["no_map"])
    m.load_state_dict(torch.load(os.path.join(r, "checkpoint_epoch12.pt"), map_location="cpu")["model_state_dict"])
    models[name] = m.eval()

def flat(prefix, x):
    if isinstance(x, torch.Tensor):
        res[prefix] = x.clone()
    elif isinstance(x, (list, tuple)):
        res[prefix + "/len"] = torch.tensor(len(x))
        for j, y in enumerate(x):
            flat(f"{prefix}/{j}", y)
    else:
        res[prefix] = torch.tensor([zlib.crc32(repr(x).encode())])  # ints / strings (scene, town, ...)

idxs = random.Random(0).sample(range(len(ds)), n)
with torch.no_grad():
    for i in idxs:
        batch = pilot_collate_fn([ds[i]])
        for k in sorted(batch):
            flat(f"s{i}/batch/{k}", batch[k])
        for name, m in models.items():
            for j, t in enumerate(m(batch)):
                res[f"s{i}/out_{name}/{j}"] = t.clone()
torch.save({"idxs": idxs, "res": res}, out)
print(f"probe: {len(idxs)} samples, {len(res)} tensors -> {out}")
'''


def run_probe(tree, map_run, nomap_run, out):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run([sys.executable, "-c", PROBE, map_run, nomap_run, out, str(N)], cwd=tree, env=env, check=True)
    return torch.load(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-commit", required=True, help="the commit C_D / C_Dnomap were trained at")
    ap.add_argument("--map-run", required=True, help="a C_D run (map v1)")
    ap.add_argument("--nomap-run", required=True, help="a C_Dnomap run")
    args = ap.parse_args()
    map_run, nomap_run = os.path.abspath(args.map_run), os.path.abspath(args.nomap_run)
    here = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.py"], cwd=here, capture_output=True,
                           text=True).stdout.strip()
    ref = subprocess.run(["git", "rev-parse", args.ref_commit], cwd=here, capture_output=True, text=True,
                         check=True).stdout.strip()
    print(f"reference {ref[:7]} vs working tree (HEAD {head[:7]}, *.py dirty={bool(dirty)}); "
          f"runs {map_run}, {nomap_run} (checkpoint_epoch12.pt)")
    with tempfile.TemporaryDirectory() as tmp:
        ref_tree = os.path.join(tmp, "ref")
        os.makedirs(ref_tree)
        arch = subprocess.run(["git", "archive", ref], cwd=here, capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", ref_tree], input=arch, check=True)
        a = run_probe(ref_tree, map_run, nomap_run, os.path.join(tmp, "ref.pt"))
        b = run_probe(here, map_run, nomap_run, os.path.join(tmp, "new.pt"))
    fails = []
    if a["idxs"] != b["idxs"]:
        fails.append("sample indices differ")
    ka, kb = set(a["res"]), set(b["res"])
    if ka != kb:
        fails.append(f"tensor sets differ: only ref {sorted(ka - kb)[:5]}, only new {sorted(kb - ka)[:5]}")
    for k in sorted(ka & kb):
        x, y = a["res"][k], b["res"][k]
        if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
            fails.append(f"{k}: differs")
    n_out = sum(k.endswith("/out_map/1") for k in ka)
    n_init = sum(k.startswith("init_") for k in ka)
    print(f"compared {len(ka & kb)} tensors: {n_out} samples' batches + C_D / C_Dnomap outputs, "
          f"{n_init} initial-weight tensors")
    for f in fails[:20]:
        print(f"  MISMATCH {f}")
    print("RESULT: " + ("PASS" if not fails else f"FAIL ({len(fails)} mismatches)"))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
