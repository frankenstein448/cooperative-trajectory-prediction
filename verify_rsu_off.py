"""
verify_rsu_off.py
With --use-rsu off, the env code must behave exactly as before the RSU extension (runs/DECISION_RULES_RSU.md):
runs the same probe in a `git archive` of --ref-commit (the code before the RSU change) and in the working tree,
each in its own process, and requires torch.equal for
  - every tensor of 50 random env samples (seed 0) as env_collate_fn batches them (E_V2X's dataset: v2x on, map,
    the run's cam_norm / env_norm, image cache),
  - E_V2X's outputs on those samples (trajs, logits; --run-dir's checkpoint, CPU, eval mode, deterministic),
  - the initial weights of EnvFusion(v2x on) and EnvFusion(v2x off) built after torch.manual_seed(0) (same
    parameter names, shapes and values, so the init order is unchanged).
Read-only on runs/; no training. Prints RESULT: PASS / FAIL.

Run in gpu-test-venv:
  python verify_rsu_off.py --ref-commit <commit before the RSU change> --run-dir runs/E_V2X_seed0 \
      2>&1 | tee verify_rsu_off.log
"""
import os
import sys
import json
import argparse
import tempfile
import subprocess

import torch

N = 50
PROBE = r'''
import os, sys, json, random, zlib
import torch
torch.use_deterministic_algorithms(True)
from env_dataset import EnvDataset, env_collate_fn
from env_model import EnvFusion
run_dir, ckpt, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg = json.load(open(os.path.join(run_dir, "config.json")))
assert cfg["task"] == "env" and cfg["v2x"] == "on" and not cfg.get("use_rsu", False), cfg
res = {}
for v2x in (True, False):
    torch.manual_seed(0)
    m = EnvFusion(device="cpu", v2x=v2x, use_map=not cfg["no_map"])
    for k, t in m.state_dict().items():
        res[f"init_v2x{int(v2x)}/{k}"] = t.clone()
ds = EnvDataset(v2x=cfg["v2x"], use_map=not cfg["no_map"], map_transform=cfg["map_transform"],
                env_norm=json.load(open(os.path.join(run_dir, "env_norm.json"))),
                use_image_cache=cfg["use_image_cache"], use_speed=cfg["use_speed"], drop_cam_xy=cfg["drop_cam_xy"])
ds.cam_norm = json.load(open(os.path.join(run_dir, "cam_norm.json")))
model = EnvFusion(device="cpu", v2x=True, use_map=not cfg["no_map"])
model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
model.eval()

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
        batch = env_collate_fn([ds[i]])
        for k in sorted(batch):
            flat(f"s{i}/batch/{k}", batch[k])
        trajs, logits = model(batch)
        res[f"s{i}/out/trajs"], res[f"s{i}/out/logits"] = trajs.clone(), logits.clone()
torch.save({"idxs": idxs, "res": res}, out)
print(f"probe: {len(idxs)} samples, {len(res)} tensors -> {out}")
'''


def run_probe(tree, run_dir, ckpt, out):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run([sys.executable, "-c", PROBE, run_dir, ckpt, out, str(N)], cwd=tree, env=env, check=True)
    return torch.load(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-commit", required=True, help="the commit before the RSU change")
    ap.add_argument("--run-dir", required=True, help="an E_V2X run (config.json, cam_norm.json, env_norm.json)")
    ap.add_argument("--ckpt", default=None, help="default <run-dir>/checkpoint_epoch12.pt")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    ckpt = os.path.abspath(args.ckpt or os.path.join(run_dir, "checkpoint_epoch12.pt"))
    here = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.py"], cwd=here, capture_output=True,
                           text=True).stdout.strip()
    ref = subprocess.run(["git", "rev-parse", args.ref_commit], cwd=here, capture_output=True, text=True,
                         check=True).stdout.strip()
    print(f"reference {ref[:7]} vs working tree (HEAD {head[:7]}, *.py dirty={bool(dirty)}); run {run_dir}, ckpt {ckpt}")
    with tempfile.TemporaryDirectory() as tmp:
        ref_tree = os.path.join(tmp, "ref")
        os.makedirs(ref_tree)
        arch = subprocess.run(["git", "archive", ref], cwd=here, capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", ref_tree], input=arch, check=True)
        a = run_probe(ref_tree, run_dir, ckpt, os.path.join(tmp, "ref.pt"))
        b = run_probe(here, run_dir, ckpt, os.path.join(tmp, "new.pt"))
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
    n_out = sum(k.endswith("/out/trajs") for k in ka)
    n_init = sum(k.startswith("init_") for k in ka)
    print(f"compared {len(ka & kb)} tensors: {n_out} samples' batches + outputs, {n_init} initial-weight tensors")
    for f in fails[:20]:
        print(f"  MISMATCH {f}")
    print("RESULT: " + ("PASS" if not fails else f"FAIL ({len(fails)} mismatches)"))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
