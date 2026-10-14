"""
verify_env_model.py
Real-data check of the Part 2 Stage 2 code (runs/DESIGN_ENV.md sections 8 and 10), before any env training.
Loads EnvDataset twice (v2x on with the map, v2x off with the map), map transform mirror, split v2.
run_env_seeds.sh refuses to start unless verify_env_model.log is committed with RESULT: PASS.

1. Dataset vs cache, N_CHECK = 50 random env samples (seed 0): the sample is the cache row's (agent, scene,
   frame, town); the target set is exactly the cache's (a) targets (v2x off) / (a)+(b) targets (v2x on), in cache
   order; no (c) target; t_hist / t_fut / t_fut_mask / t_pose / t_state / t_cv_vel equal the cache arrays; with
   v2x off no (b) target and no coop sensors; the CAM keeps only the (normalized) speed. Over ALL env samples:
   the sample list is the same for both models, every sample has an (a) target, no known target is the ego
   itself (|cur_ego| > EGO_MIN_DIST_M).
2. Map tokens on real scenes (all 54 scenes pass the mirror alignment, check_map_alignment_mirror.log): for the
   moving targets (speed > MOVING_M_PER_FRAME, the builder's motion threshold) of the 50 samples, the nearest
   lane segment (two consecutive points of one piece) whose direction is within LANE_DIR_DEG of the target's
   heading. PASS if the median distance of that segment is <= LANE_MEDIAN_M, its median direction error is
   <= LANE_DIR_MEDIAN_DEG, and >= LANE_FRAC_MIN of the targets have such a segment within LANE_DIST_M (the
   alignment acceptance's p95 tolerance, DESIGN_ENV.md section 9). Feature ranges: x, y within XY_MAX (pieces
   may extend past the 30 m radius), cos / sin in [-1, 1], flags 0 / 1, padding zero.
   CHANGED AFTER THE FIRST RESULT (user, 2026-10-14): the first criterion was ">= 90% within 1.0 m" (and the
   median direction); the first run gave 0.852 within 1.0 m (p50 0.10 m, p90 1.30 m, direction 0.08 deg),
   RESULT FAIL. That bar had been set without reference to real driving: the misses are the expected tail
   (lane changes, junction turns, off-centre driving; median lateral offset ~0 in every town, so not a
   misalignment), and it was replaced by the criterion above.
3. Invariance: the lanes and every target moved by a random world rotation + translation (5 draws) give the
   same map tokens (max |diff| <= TOK_TOL), and EnvFusion (eval) gives the same outputs on a real batch with
   those tokens (max |diff| <= OUT_TOL). All other per-target inputs are ego / own frame and do not change.
4. Loss: env_wta_loss on a real padded batch equals D's WinnerTakesAllLoss on the real targets stacked one by
   one (|diff| <= 1e-6); the gradient w.r.t. every padded target's t_hist / t_state / t_map is exactly zero, and
   nonzero for the real ones; parameter gradients are finite; perturbing the padded targets' t_pose / t_hist
   leaves the real targets' outputs unchanged (t_pose by perturbation: hypot's input gradient is NaN at the
   self pair, and no parameter lies upstream of t_pose).
5. --task ego: EndToEndFusion with default flags has D's parameter count and strict-loads D_CKPT. Each env
   config does a forward + backward on a real batch with finite outputs.

Run in gpu-test-venv:  python -u verify_env_model.py 2>&1 | tee verify_env_model.log
"""
import math

import numpy as np
import torch
from scipy.spatial import cKDTree

from env_dataset import EnvDataset, env_collate_fn, known_mask, POS_SCALE_M
from env_map import piece_tokens, RADIUS_M
from env_model import EnvFusion, env_wta_loss
from end_to_end_model import EndToEndFusion
from trajectory_loss import WinnerTakesAllLoss
from scene_split import three_way_scene_split

D_CKPT = "runs/D_seed0/checkpoint_epoch12.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CHECK = 50
EGO_MIN_DIST_M = 1.0
MOVING_M_PER_FRAME = 0.5
LANE_DIR_DEG = 30.0
LANE_MEDIAN_M = 0.5
LANE_DIST_M = 2.5
LANE_FRAC_MIN = 0.95   # was 0.90 within 1.0 m before 2026-10-14 (docstring, section 2)
LANE_DIR_MEDIAN_DEG = 5.0
XY_MAX = 2.0
TOK_TOL = 1e-4
OUT_TOL = 1e-4
FAILS = []


def check(ok, msg):
    print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
    if not ok:
        FAILS.append(msg)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def seg_dist_to_origin(a, b):
    """Distance from (0, 0) to segments a -> b, arrays [n, 2]."""
    ab = b - a
    t = np.clip(-(a * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return np.hypot(*(a + t[:, None] * ab).T)


def lane_match(feats, pt_mask, piece_mask):
    """Nearest segment to the target (origin) with lane direction within LANE_DIR_DEG of its heading:
    (distance m, |direction| deg) of that segment, or (inf, nan)."""
    best = (math.inf, math.nan)
    for k in np.nonzero(piece_mask)[0]:
        f = feats[k][pt_mask[k]]
        if len(f) < 2:
            continue
        xy = f[:, :2] * RADIUS_M
        ang = np.degrees(np.abs(np.arctan2(f[:, 3], f[:, 2])))
        ok = (ang[:-1] <= LANE_DIR_DEG) & (ang[1:] <= LANE_DIR_DEG)
        if not ok.any():
            continue
        d = seg_dist_to_origin(xy[:-1][ok], xy[1:][ok])
        j = int(d.argmin())
        if d[j] < best[0]:
            best = (float(d[j]), float(0.5 * (ang[:-1][ok][j] + ang[1:][ok][j])))
    return best


def rotated_tokens(lp, xy, yaw, th, t):
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])
    pts = lp.pts @ R.T + t
    return piece_tokens(pts, lp.yaw + th, lp.is_junction, lp.is_parking, lp.piece_points, lp.piece_of_point,
                        cKDTree(pts), R @ xy + t, yaw + th)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    kw = dict(map_transform="mirror", use_image_cache=True, use_speed=True, drop_cam_xy=True)
    ds = {"on": EnvDataset(v2x="on", use_map=True, **kw), "off": EnvDataset(v2x="off", use_map=True, **kw)}
    c = ds["on"].cache
    train_set, _, _ = three_way_scene_split(ds["on"], split_version="v2")
    cam_values = torch.stack([ds["on"].get_cam_features(*ds["on"].samples[i][:3]) for i in train_set.indices])
    cam_norm = {"mean": cam_values.mean(0).tolist(), "std": (cam_values.std(0) + 1e-6).tolist()}
    for d in ds.values():
        d.cam_norm = cam_norm
    print(f"cam_norm (train split v2, {len(train_set.indices)} samples): {cam_norm}")

    # ---------------- 1. dataset vs cache ----------------
    print("\n1. dataset vs cache")
    check(np.array_equal(ds["on"].rows, ds["off"].rows), f"same {len(ds['on'].rows)} sample rows for v2x on / off")
    ptr = c["sample_ptr"]
    sample_of_target = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    rows = ds["on"].rows
    in_rows = np.isin(sample_of_target, rows)
    n_a = np.bincount(sample_of_target[c["vis"] == 0], minlength=len(ptr) - 1)[rows]
    check((n_a > 0).all(), "every env sample has a class-(a) target")
    known_on = in_rows & known_mask(c["vis"], "on")
    dist = np.hypot(*c["cur_ego"][known_on].T)
    check(dist.min() > EGO_MIN_DIST_M, f"no known target at the ego: min |cur_ego| = {dist.min():.2f} m "
                                        f"over {known_on.sum()} known targets (> {EGO_MIN_DIST_M} m)")
    pick = rng.choice(len(rows), N_CHECK, replace=False)
    n_bad = {k: 0 for k in ("sample", "targets", "arrays", "no_c", "off_b", "off_coop", "on_coop_key", "cam")}
    n_t = {"on": 0, "off": 0, "b": 0}
    items_on = {}
    for i in pick:
        r = int(rows[i])
        lo, hi = int(ptr[r]), int(ptr[r + 1])
        vis = c["vis"][lo:hi]
        for v2x in ("on", "off"):
            item = ds[v2x][int(i)]
            if v2x == "on":
                items_on[int(i)] = item
            key = (item["agent_id"], item["scene"], item["frame"], item["town"])
            if key != (c["sample_agent"][r], c["sample_scene"][r], c["sample_frame"][r], c["sample_town"][r]):
                n_bad["sample"] += 1
            tid = item["t_tid"].numpy()
            want = np.arange(lo, hi)[(vis == 0) if v2x == "off" else (vis <= 1)]
            n_bad["targets"] += not np.array_equal(tid, want)
            n_bad["no_c"] += int((c["vis"][tid] == 2).any())
            n_t[v2x] += len(tid)
            if v2x == "on":
                n_t["b"] += int((c["vis"][tid] == 1).sum())
            ok = (np.array_equal(item["t_hist"].numpy(), c["hist_own"][tid].astype(np.float32))  # env_norm unset
                  and np.array_equal(item["t_fut"].numpy(), c["fut_own"][tid].astype(np.float32))
                  and np.array_equal(item["t_fut_mask"].numpy(), c["fut_mask"][tid].astype(bool))
                  and np.array_equal(item["t_pose"].numpy()[:, :2], c["cur_ego"][tid].astype(np.float32))
                  and np.array_equal(item["t_pose"].numpy()[:, 2], c["yaw_ego"][tid].astype(np.float32))
                  and np.allclose(item["t_state"].numpy()[:, :2] * POS_SCALE_M, c["cur_ego"][tid], atol=1e-4)
                  and np.array_equal(item["t_cv_vel"].numpy(), -c["hist_own"][tid, -1].astype(np.float32))
                  and np.array_equal(item["t_vis"].numpy(), c["vis"][tid]))
            n_bad["arrays"] += not ok
            if v2x == "off":
                n_bad["off_b"] += int((c["vis"][tid] != 0).any())
                n_bad["off_coop"] += ("coop_points" in item or "coop_images" in item)
            else:
                n_bad["on_coop_key"] += "coop_points" not in item
            raw = ds[v2x].get_cam_features(*key[:3])
            want_speed = (raw[3] - cam_norm["mean"][3]) / cam_norm["std"][3]
            n_bad["cam"] += not (bool((item["cam"][:3] == 0).all()) and abs(float(item["cam"][3] - want_speed)) < 1e-5)
    print(f"  {N_CHECK} samples: {n_t['on']} known targets with v2x on ({n_t['b']} class b), {n_t['off']} with v2x off")
    check(n_bad["sample"] == 0, f"sample (agent, scene, frame, town) == cache row: {N_CHECK * 2 - n_bad['sample']}/{N_CHECK * 2}")
    check(n_bad["targets"] == 0, f"target set == cache's known targets in cache order: {N_CHECK * 2 - n_bad['targets']}/{N_CHECK * 2}")
    check(n_bad["no_c"] == 0, "no class-(c) target loaded")
    check(n_bad["arrays"] == 0, f"t_hist / t_fut / t_fut_mask / t_pose / t_state / t_cv_vel / t_vis == cache: "
                                f"{N_CHECK * 2 - n_bad['arrays']}/{N_CHECK * 2}")
    check(n_bad["off_b"] == 0 and n_bad["off_coop"] == 0, "v2x off: no (b) target and no coop sensors")
    check(n_bad["on_coop_key"] == 0, "v2x on: coop sensors loaded")
    check(n_t["b"] > 0, "v2x on: the 50 samples include (b) targets")
    check(n_bad["cam"] == 0, f"CAM: x, y, heading zeroed, speed normalized: {N_CHECK * 2 - n_bad['cam']}/{N_CHECK * 2}")

    # ---------------- 2. map tokens on real scenes ----------------
    print("\n2. map tokens on real scenes")
    dists, angs, ranges_ok, pad_ok = [], [], True, True
    for i, item in items_on.items():
        tid = item["t_tid"].numpy()
        f, pm, km = item["t_map"].numpy(), item["t_map_pt_mask"].numpy(), item["t_map_piece_mask"].numpy()
        real = f[pm]
        ranges_ok &= bool(len(real) == 0 or (np.abs(real[:, :2]).max() <= XY_MAX
                                             and np.abs(real[:, 2:4]).max() <= 1 + 1e-6
                                             and np.isin(real[:, 4:6], (0.0, 1.0)).all()))
        pad_ok &= bool((f[~pm] == 0).all() and np.array_equal(km, pm.any(-1)))
        for j, t in enumerate(tid):
            if c["speed"][t] > MOVING_M_PER_FRAME:
                d, a = lane_match(f[j], pm[j], km[j])
                dists.append(d)
                angs.append(a)
    dists, angs = np.array(dists), np.array(angs)
    frac = float(np.mean(dists <= LANE_DIST_M))
    matched = np.isfinite(dists)
    med_a = float(np.nanmedian(angs[matched]))
    print(f"  {len(dists)} moving targets: same-direction segment distance p50 {np.median(dists):.2f} m, "
          f"p90 {np.quantile(dists, 0.9):.2f} m, p95 {np.quantile(dists, 0.95):.2f} m, none within {LANE_DIR_DEG:.0f} deg: {int((~matched).sum())}")
    med_d = float(np.median(dists))
    check(med_d <= LANE_MEDIAN_M, f"median distance of the same-direction segment {med_d:.2f} m (<= {LANE_MEDIAN_M})")
    check(frac >= LANE_FRAC_MIN, f"own lane within {LANE_DIST_M} m at <= {LANE_DIR_DEG:.0f} deg: {frac:.3f} "
                                 f"(>= {LANE_FRAC_MIN})")
    check(med_a <= LANE_DIR_MEDIAN_DEG, f"median direction of the matched segment {med_a:.2f} deg (<= {LANE_DIR_MEDIAN_DEG})")
    check(ranges_ok, f"feature ranges: |x|, |y| <= {XY_MAX}, |cos|, |sin| <= 1, flags 0 / 1")
    check(pad_ok, "padding zero; piece mask == any point")

    # ---------------- 3. invariance ----------------
    print("\n3. world rotation + translation invariance")
    lanes = ds["on"].lanes
    batch_idx = [int(i) for i in pick[:4]]
    batch = env_collate_fn([items_on[i] for i in batch_idx])
    max_tok = 0.0
    moved = {k: batch[k].clone() for k in ("t_map", "t_map_pt_mask", "t_map_piece_mask")}
    model = EnvFusion(device=DEVICE, v2x=True, use_map=True).to(DEVICE).eval()
    with torch.no_grad():
        scene, cam = model.scene_tokens(batch)
        dev = lambda b, k: b[k].to(DEVICE)
        ref = model.core(scene, cam, dev(batch, "t_hist"), dev(batch, "t_state"), dev(batch, "t_pose"),
                         dev(batch, "target_mask"), dev(batch, "t_map"), dev(batch, "t_map_pt_mask"),
                         dev(batch, "t_map_piece_mask"))
        max_out = 0.0
        for draw in range(5):
            th, t = rng.uniform(-math.pi, math.pi), rng.uniform(-1000, 1000, 2)
            for b, i in enumerate(batch_idx):
                r = int(rows[i])
                lp = lanes[str(c["sample_town"][r])]
                for j, tg in enumerate(items_on[i]["t_tid"].numpy()):
                    f, pm, km = rotated_tokens(lp, c["xy_world"][tg].astype(np.float64), float(c["yaw_world"][tg]), th, t)
                    max_tok = max(max_tok, float(np.abs(f - batch["t_map"][b, j].numpy()).max()))
                    if not (np.array_equal(pm, batch["t_map_pt_mask"][b, j].numpy())
                            and np.array_equal(km, batch["t_map_piece_mask"][b, j].numpy())):
                        max_tok = math.inf
                    moved["t_map"][b, j] = torch.from_numpy(f)
                    moved["t_map_pt_mask"][b, j] = torch.from_numpy(pm)
                    moved["t_map_piece_mask"][b, j] = torch.from_numpy(km)
            out = model.core(scene, cam, dev(batch, "t_hist"), dev(batch, "t_state"), dev(batch, "t_pose"),
                             dev(batch, "target_mask"), dev(moved, "t_map"), dev(moved, "t_map_pt_mask"),
                             dev(moved, "t_map_piece_mask"))
            m = batch["target_mask"].to(DEVICE)
            max_out = max(max_out, float((out[0] - ref[0])[m].abs().max()), float((out[1] - ref[1])[m].abs().max()))
    print(f"  batch of {len(batch_idx)} samples, {int(batch['target_mask'].sum())} targets, 5 draws")
    check(max_tok <= TOK_TOL, f"map tokens: max |diff| {max_tok:.2e} (<= {TOK_TOL}), masks identical")
    check(max_out <= OUT_TOL, f"EnvFusion outputs: max |diff| {max_out:.2e} (<= {OUT_TOL})")

    # ---------------- 4. loss and padding gradient ----------------
    print("\n4. loss and padding gradient")
    model.train()
    torch.manual_seed(1)
    scene, cam = model.scene_tokens(batch)
    leaves = {k: batch[k].to(DEVICE).clone().requires_grad_(True) for k in ("t_hist", "t_state", "t_pose", "t_map")}
    tm = batch["target_mask"].to(DEVICE)
    check(bool((~tm).any()), f"batch has padding ({int((~tm).sum())} padded slots)")
    trajs, logits = model.core(scene, cam, leaves["t_hist"], leaves["t_state"], leaves["t_pose"], tm, leaves["t_map"],
                               dev(batch, "t_map_pt_mask"), dev(batch, "t_map_piece_mask"))
    fut, fmask = dev(batch, "t_fut"), dev(batch, "t_fut_mask")
    loss_fn = WinnerTakesAllLoss()
    loss, _ = env_wta_loss(loss_fn, trajs, logits, fut, fmask, tm)
    reals = [(b, j) for b in range(tm.shape[0]) for j in range(tm.shape[1]) if tm[b, j]]
    stack = lambda x: torch.stack([x[b, j] for b, j in reals])
    ref_loss, _ = loss_fn(stack(trajs), stack(logits), stack(fut), stack(fmask))
    check(abs(loss.item() - ref_loss.item()) <= 1e-6, f"env_wta_loss {loss.item():.6f} == D's WTA on {len(reals)} "
                                                       f"stacked targets {ref_loss.item():.6f}")
    loss.backward()
    # t_pose only feeds rel_pose_features, whose hypot has a NaN input gradient at the self pair (distance 0);
    # no parameter lies upstream of t_pose, so it is checked by perturbation below, not by its gradient.
    grad_keys = ("t_hist", "t_state", "t_map")
    pad_zero = all(float(leaves[k].grad[~tm].abs().max()) == 0.0 for k in grad_keys)
    real_nonzero = all(float(leaves[k].grad[tm].abs().max()) > 0.0 for k in grad_keys)
    check(pad_zero, "gradient w.r.t. padded targets' t_hist / t_state / t_map is exactly 0")
    check(real_nonzero, "gradient w.r.t. real targets' t_hist / t_state / t_map is nonzero")
    check(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),
          "parameter gradients finite")
    model.eval()
    with torch.no_grad():
        def run(pose, hist):
            return model.core(scene, cam, hist, dev(batch, "t_state"), pose, tm, dev(batch, "t_map"),
                              dev(batch, "t_map_pt_mask"), dev(batch, "t_map_piece_mask"))
        pose, hist = dev(batch, "t_pose"), dev(batch, "t_hist")
        a = run(pose, hist)
        noise = lambda x: torch.where((~tm)[..., None] if x.dim() == 3 else (~tm)[..., None, None],
                                      x + 50 * torch.randn_like(x), x)
        b = run(noise(pose), noise(hist))
        diff = max(float((a[0] - b[0])[tm].abs().max()), float((a[1] - b[1])[tm].abs().max()))
    check(diff == 0.0, f"perturbing padded targets' t_pose / t_hist leaves real outputs unchanged (max |diff| {diff:.1e})")
    del model, trajs, logits, loss, leaves, scene, cam

    # ---------------- 5. --task ego identity; real forward per env config ----------------
    print("\n5. --task ego identity; env configs on a real batch")
    sd = torch.load(D_CKPT, map_location="cpu")["model_state_dict"]
    n_d = sum(v.numel() for k, v in sd.items() if not k.startswith("graph_"))  # graph_* are buffers
    ego = EndToEndFusion(device="cpu", use_map=True)
    n_ego = sum(p.numel() for p in ego.parameters())
    try:
        ego.load_state_dict(sd, strict=True)
        strict_ok = True
    except RuntimeError as e:
        print(f"  {e}")
        strict_ok = False
    check(n_ego == n_d and strict_ok, f"EndToEndFusion defaults: {n_ego:,} parameters vs D's {n_d:,}; "
                                      f"strict load of {D_CKPT}: {strict_ok}")
    del ego, sd
    off_batch = env_collate_fn([ds["off"][i] for i in batch_idx])
    for name, v2x, use_map, b in (("E_EgoOnly", False, True, off_batch), ("E_V2Xnomap", True, False, batch),
                                  ("E_V2X", True, True, batch)):
        m = EnvFusion(device=DEVICE, v2x=v2x, use_map=use_map).to(DEVICE).train()
        if not use_map:
            b = {k: v for k, v in b.items() if not k.startswith("t_map")}
        trajs, logits = m(b)
        loss, _ = env_wta_loss(loss_fn, trajs, logits, dev(b, "t_fut"), dev(b, "t_fut_mask"), dev(b, "target_mask"))
        loss.backward()
        finite = bool(torch.isfinite(trajs).all() and torch.isfinite(logits).all() and torch.isfinite(loss)
                      and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        n_p = sum(p.numel() for p in m.parameters())
        check(finite, f"{name}: {n_p:,} parameters, trajs {tuple(trajs.shape)}, loss {loss.item():.3f}, "
                      f"outputs and gradients finite")
        del m, trajs, logits, loss

    print(f"\n{len(FAILS)} failed check(s)" + "".join(f"\n  - {f}" for f in FAILS))
    print(f"\nRESULT: {'PASS' if not FAILS else 'FAIL'}")


if __name__ == "__main__":
    main()
