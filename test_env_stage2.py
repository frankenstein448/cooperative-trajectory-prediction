"""
test_env_stage2.py
Unit tests for Part 2 Stage 2 (env_map.py, env_dataset.py, env_model.py, train.py --task env), on small
SYNTHETIC tensors only: no V2X-Sim metadata, no PilotDataset, no env cache, no lane files, no GPU
(CUDA_VISIBLE_DEVICES is cleared before torch is imported).

Run in gpu-test-venv:  python test_env_stage2.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import math
import sys
import traceback

import numpy as np
import torch
from scipy.spatial import cKDTree

import env_map
from env_map import piece_tokens, LanePolylines, K_PIECES, MAX_PIECE_POINTS, N_POINT_FEATS, RADIUS_M
import env_dataset as ed
from env_model import EnvFusion, rel_pose_features, env_wta_loss, PolylineEncoder
from trajectory_loss import WinnerTakesAllLoss

torch.set_num_threads(2)
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- helpers
def synth_npz(lanes):
    """lanes: list of (points [n, 2] CARLA, yaw_deg [n], lane_type str, is_junction bool) -> export-like dict,
    each lane cut into pieces of <= 10 points (export_lane_polylines.py layout)."""
    pts, yaw, ltype, junc, order, ptr = [], [], [], [], [], [0]
    for p, y, t, j in lanes:
        base = len(pts)
        pts += list(p); yaw += list(y); ltype += [t] * len(p); junc += [j] * len(p)
        idx = list(range(base, base + len(p)))
        for a in range(0, len(idx), 10):
            order += idx[a:a + 10]
            ptr.append(len(order))
    return {"points": np.array(pts, dtype=np.float64), "yaw_deg": np.array(yaw, dtype=np.float64),
            "lane_type": np.array(ltype), "is_junction": np.array(junc), "order": np.array(order),
            "piece_ptr": np.array(ptr)}


def straight_lane(x0, y0, yaw_deg, n=10, step=2.0):
    a = math.radians(yaw_deg)
    return np.array([[x0 + k * step * math.cos(a), y0 + k * step * math.sin(a)] for k in range(n)]), np.full(n, yaw_deg)


def rot2(pts, th):
    c, s = math.cos(th), math.sin(th)
    return pts @ np.array([[c, -s], [s, c]]).T


def synth_cache(n_targets_per_sample, vis_list):
    """Small env cache dict with the CACHE_KEYS layout; vis_list: flat list of class labels."""
    ptr = np.concatenate([[0], np.cumsum(n_targets_per_sample)])
    n = int(ptr[-1])
    assert len(vis_list) == n
    S = len(n_targets_per_sample)
    return {
        "sample_agent": np.ones(S, np.int32), "sample_scene": np.arange(S, dtype=np.int32) + 3,
        "sample_frame": np.arange(S, dtype=np.int32) + 10, "sample_town": np.array(["Town05"] * S),
        "sample_ptr": ptr.astype(np.int64), "vis": np.array(vis_list, np.int32),
        "cat4": (np.arange(n) % 4).astype(np.int32), "speed": RNG.uniform(0, 3, n).astype(np.float32),
        "size": RNG.uniform(1, 5, (n, 3)).astype(np.float32), "cur_ego": RNG.uniform(-40, 40, (n, 2)).astype(np.float32),
        "yaw_ego": RNG.uniform(-3, 3, n).astype(np.float32), "hist_own": RNG.normal(0, 3, (n, 8, 2)).astype(np.float32),
        "fut_own": RNG.normal(0, 10, (n, 30, 2)).astype(np.float32), "fut_mask": np.ones((n, 30), bool),
        "xy_world": RNG.uniform(-30, 30, (n, 2)), "yaw_world": RNG.uniform(-3, 3, n).astype(np.float32),
        "n_removed": np.zeros(S, np.int32), "manifest": {"git_commit": "synthetic"},
    }


def tiny_model_inputs(B=2, T=5, K=4, P=MAX_PIECE_POINTS, n_real=(5, 3), seed=0):
    g = torch.Generator().manual_seed(seed)
    tm = torch.zeros(B, T, dtype=torch.bool)
    for b, n in enumerate(n_real):
        tm[b, :n] = True
    pt_mask = torch.rand(B, T, K, P, generator=g) > 0.3
    piece_mask = pt_mask.any(-1)
    return dict(scene=torch.randn(B, 3, 128, generator=g), cam_embed=torch.randn(B, 128, generator=g),
                t_hist=torch.randn(B, T, 8, 2, generator=g), t_state=torch.randn(B, T, ed.STATE_DIM, generator=g),
                t_pose=torch.randn(B, T, 3, generator=g) * torch.tensor([20.0, 20.0, 1.0]), target_mask=tm,
                t_map=torch.randn(B, T, K, P, N_POINT_FEATS, generator=g), t_map_pt_mask=pt_mask,
                t_map_piece_mask=piece_mask)


# ---------------------------------------------------------------- env_map
def test_mirror_lane_in_target_frame():
    # CARLA lane along +x at y_c = +5 (yaw 0) -> V2X-Sim y = -5, direction +x; a CARLA lane along +y_c
    # (yaw 90) at x = 40 -> V2X-Sim direction -y (yaw -90).
    p1, y1 = straight_lane(0.0, 5.0, 0.0)
    p2, y2 = straight_lane(40.0, -10.0, 90.0)
    lp = LanePolylines("Town05", "mirror", data=synth_npz([(p1, y1, "Driving", False), (p2, y2, "Parking", True)]))
    assert np.allclose(lp.pts[:10], np.c_[p1[:, 0], -p1[:, 1]]) and np.allclose(lp.yaw[10:], -math.pi / 2)
    f, pm, km = lp.tokens(np.array([10.0, -5.0]), 0.0)
    assert km[0] and not km[2:].any()
    on = f[0][pm[0]]
    assert np.allclose(on[:, 1], 0, atol=1e-6) and np.allclose(on[:, 2], 1, atol=1e-6) and np.allclose(on[:, 3], 0, atol=1e-6)
    assert np.allclose(np.sort(on[:, 0]) * RADIUS_M, np.arange(-10, 10, 2.0), atol=1e-5)
    # standing on the Parking lane, heading along it (-y in V2X-Sim): direction 0, parking flag 1, junction 1
    f, pm, km = lp.tokens(np.array([40.0, 18.0]), -math.pi / 2)
    first = f[0][pm[0]]
    assert np.allclose(first[:, 2], 1, atol=1e-6) and (first[:, 5] == 1).all() and (first[:, 4] == 1).all()
    assert np.allclose(first[:, 1], 0, atol=1e-6)


def test_piece_tokens_rigid_invariance():
    lanes = []
    for _ in range(30):
        p, y = straight_lane(*RNG.uniform(-60, 60, 2), RNG.uniform(-180, 180), n=int(RNG.integers(2, 12)))
        lanes.append((p, y, "Driving" if RNG.random() < 0.8 else "Parking", bool(RNG.random() < 0.3)))
    lp = LanePolylines("Town05", "mirror", data=synth_npz(lanes))
    xy, hd = np.array([3.0, -4.0]), 0.7
    ref = lp.tokens(xy, hd)
    assert ref[2].sum() >= 3, "test needs several pieces in range"
    for _ in range(5):
        th, t = RNG.uniform(-math.pi, math.pi), RNG.uniform(-500, 500, 2)
        pts = rot2(lp.pts, th) + t
        out = piece_tokens(pts, lp.yaw + th, lp.is_junction, lp.is_parking, lp.piece_points, lp.piece_of_point,
                           cKDTree(pts), rot2(xy[None], th)[0] + t, hd + th)
        assert np.array_equal(out[1], ref[1]) and np.array_equal(out[2], ref[2])
        assert np.allclose(out[0], ref[0], atol=1e-5), np.abs(out[0] - ref[0]).max()


def test_piece_selection_radius_cap_order():
    lanes = [straight_lane(0.0, float(k) * 4, 0.0, n=3) for k in range(-20, 21)]  # parallel lanes 4 m apart
    lp = LanePolylines("Town05", "mirror", data=synth_npz([(p, y, "Driving", False) for p, y in lanes]))
    f, pm, km = lp.tokens(np.array([2.0, 0.0]), 0.0, radius=30.0, k_pieces=5)
    assert km.all()  # more than 5 pieces in range -> capped at 5
    near = [np.abs(f[s][pm[s]][:, 1]).min() * RADIUS_M for s in range(5)]
    assert near == sorted(near) and near[0] < 1e-6, near
    f, pm, km = lp.tokens(np.array([2.0, 0.0]), 0.0, radius=30.0, k_pieces=K_PIECES)
    n_in = sum(1 for p, _ in lanes if np.hypot(p[:, 0] - 2.0, p[:, 1]).min() <= 30.0)
    assert km.sum() == n_in and not km[n_in:].any()
    assert np.all(f[~pm] == 0), "padding must be zero"
    f, pm, km = lp.tokens(np.array([500.0, 500.0]), 0.0)
    assert not km.any() and not pm.any() and not f.any()


# ---------------------------------------------------------------- env_dataset (pure parts)
def test_known_targets_and_sample_list():
    # sample 0: a, b, c; sample 1: c only; sample 2: b only; sample 3: empty; sample 4: a, a
    c = synth_cache([3, 1, 1, 0, 2], [0, 1, 2, 2, 1, 0, 0])
    assert list(ed.env_sample_rows(c)) == [0, 4]
    assert list(ed.target_rows(c, 0, "off")) == [0] and list(ed.target_rows(c, 0, "on")) == [0, 1]
    assert list(ed.target_rows(c, 2, "on")) == [4] and list(ed.target_rows(c, 2, "off")) == []
    for r in range(5):
        for v in ("off", "on"):
            assert (c["vis"][ed.target_rows(c, r, v)] != 2).all(), "class (c) must never be loaded"
    try:
        ed.known_mask([0], "maybe")
        raise AssertionError("bad v2x accepted")
    except AssertionError as e:
        assert "v2x must be" in str(e)


def test_features_norm_and_padding():
    c = synth_cache([3, 2], [0, 1, 0, 1, 0])
    norm = ed.compute_env_norm(c, [0, 1], "on")
    hh = c["hist_own"].reshape(-1, 2)
    assert np.allclose(norm["mean"], hh.mean(0), atol=1e-5)
    s0 = ed.sample_features(c, 0, "on", env_norm=norm)
    idx = ed.target_rows(c, 0, "on")
    assert s0["t_hist"].shape == (3, 8, 2) and s0["t_state"].shape == (3, ed.STATE_DIM)
    assert torch.allclose(s0["t_hist"], torch.from_numpy((c["hist_own"][idx] - np.array(norm["mean"], np.float32))
                                                         / np.array(norm["std"], np.float32)))
    st = s0["t_state"].numpy()
    assert np.allclose(st[:, :2], c["cur_ego"][idx] / 50) and np.allclose(st[:, 2], np.cos(c["yaw_ego"][idx]))
    assert np.allclose(st[:, 4], c["speed"][idx] / 5) and np.allclose(st[:, 7:].argmax(1), c["cat4"][idx])
    assert torch.equal(s0["t_pose"][:, 2], torch.from_numpy(c["yaw_ego"][idx]))
    assert torch.equal(s0["t_fut"], torch.from_numpy(c["fut_own"][idx]))
    s1 = ed.sample_features(c, 1, "off")
    assert s1["t_hist"].shape[0] == 1
    pad = ed.pad_targets([s0, s1])
    assert pad["t_hist"].shape == (2, 3, 8, 2) and pad["target_mask"].tolist() == [[True] * 3, [True, False, False]]
    assert (pad["t_fut"][1, 1:] == 0).all() and torch.equal(pad["t_fut"][0], s0["t_fut"])
    empty = ed.sample_features(synth_cache([1], [2]), 0, "on")
    pad = ed.pad_targets([empty])
    assert pad["t_hist"].shape == (1, 1, 8, 2) and not pad["target_mask"].any()


def test_sample_features_with_map():
    c = synth_cache([2], [0, 1])
    p, y = straight_lane(-20.0, 0.0, 0.0, n=20)
    lanes = {t: LanePolylines(t, "mirror", data=synth_npz([(p, y, "Driving", False)])) for t in ed.TOWNS}
    c["xy_world"][:] = [[0.0, 0.0], [5.0, 0.0]]
    s = ed.sample_features(c, 0, "on", lanes=lanes)
    assert s["t_map"].shape == (2, K_PIECES, MAX_PIECE_POINTS, N_POINT_FEATS)
    assert s["t_map_piece_mask"].sum(1).tolist() == [2, 2]
    pad = ed.pad_targets([s, ed.sample_features(c, 0, "off", lanes=lanes)])
    assert pad["t_map"].shape[:2] == (2, 2) and not pad["t_map_piece_mask"][1, 1].any()


# ---------------------------------------------------------------- env_model
def test_rel_pose_invariance():
    pose = torch.randn(2, 6, 3, dtype=torch.float64) * torch.tensor([20.0, 20.0, 2.0], dtype=torch.float64)
    ref = rel_pose_features(pose, pose)
    th, t = 0.83, torch.tensor([7.0, -3.0], dtype=torch.float64)
    c, s = math.cos(th), math.sin(th)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    moved = torch.cat([pose[..., :2] @ R.T + t, pose[..., 2:] + th], dim=-1)
    assert torch.allclose(rel_pose_features(moved, moved), ref, atol=1e-9)
    assert torch.allclose(ref.diagonal(dim1=1, dim2=2)[:, :2], torch.zeros(2, 2, 6, dtype=torch.float64))


def test_core_shapes_padding_permutation():
    torch.manual_seed(0)
    model = EnvFusion(device="cpu", v2x=True, use_map=True).eval()
    x = tiny_model_inputs()
    with torch.no_grad():
        trajs, logits = model.core(**x)
    assert trajs.shape == (2, 5, 6, 30, 2) and logits.shape == (2, 5, 6) and torch.isfinite(trajs).all()
    # padding: sample 1 has 3 real targets; extra padded targets (any content) must not change real outputs
    x2 = {k: v.clone() for k, v in x.items()}
    pad = lambda v: torch.cat([v, torch.randn_like(v[:, :2]) if v.is_floating_point() else torch.zeros_like(v[:, :2])], 1)
    for k in ("t_hist", "t_state", "t_pose", "t_map", "t_map_pt_mask", "t_map_piece_mask", "target_mask"):
        x2[k] = pad(x2[k])
    x2["t_hist"][1, 3:] = 99.0
    with torch.no_grad():
        trajs2, logits2 = model.core(**x2)
    assert torch.allclose(trajs2[:, :5][x["target_mask"]], trajs[x["target_mask"]], atol=1e-5)
    assert torch.allclose(logits2[:, :5][x["target_mask"]], logits[x["target_mask"]], atol=1e-5)
    # permuting the targets permutes the outputs (no order dependence)
    perm = torch.tensor([3, 0, 4, 1, 2])
    xp = {k: (v[:, perm] if v.dim() >= 2 and v.shape[1] == 5 and k not in ("scene", "cam_embed") else v) for k, v in x.items()}
    xp["target_mask"] = torch.ones(2, 5, dtype=torch.bool)
    xa = dict(x, target_mask=torch.ones(2, 5, dtype=torch.bool))
    with torch.no_grad():
        ta, _ = model.core(**xa)
        tp, _ = model.core(**xp)
    assert torch.allclose(tp, ta[:, perm], atol=1e-5)


def test_map_removal_and_effect():
    torch.manual_seed(0)
    m_map = EnvFusion(device="cpu", v2x=False, use_map=True).eval()
    torch.manual_seed(0)
    m_nomap = EnvFusion(device="cpu", v2x=False, use_map=False).eval()
    names = {n for n, _ in m_nomap.named_parameters()}
    assert not any("poly_encoder" in n or "map_attn" in n or "no_lane" in n for n in names)
    assert not any("coop" in n for n in names)
    n_map = sum(p.numel() for p in m_map.parameters())
    n_nomap = sum(p.numel() for p in m_nomap.parameters())
    assert n_map > n_nomap
    x = tiny_model_inputs()
    x["scene"] = x["scene"][:, :2]
    with torch.no_grad():
        a, _ = m_map.core(**x)
        x_alt = dict(x, t_map=x["t_map"] + 1.0)
        b, _ = m_map.core(**x_alt)
        y = {k: v for k, v in x.items() if not k.startswith("t_map")}
        c, _ = m_nomap.core(**y)
    assert not torch.allclose(a, b), "map tokens must influence the output"
    assert torch.isfinite(c).all()
    # a target without any lane piece still works (the learned no-lane token)
    x["t_map_pt_mask"][:] = False
    x["t_map_piece_mask"][:] = False
    with torch.no_grad():
        d, _ = m_map.core(**x)
    assert torch.isfinite(d).all()


def test_polyline_encoder_mask():
    enc = PolylineEncoder()
    f = torch.randn(3, MAX_PIECE_POINTS, N_POINT_FEATS)
    m = torch.zeros(3, MAX_PIECE_POINTS, dtype=torch.bool)
    m[0, :4] = True
    m[1, :] = True
    out = enc(f, m)
    f2 = f.clone()
    f2[0, 4:] = 1e3  # masked points must not matter
    assert torch.allclose(enc(f2, m), out) and (out[2] == 0).all()


def test_env_loss_equals_d_loss_and_padding_gradient():
    torch.manual_seed(0)
    B, T = 2, 4
    trajs = torch.randn(B, T, 6, 30, 2, requires_grad=True)
    logits = torch.randn(B, T, 6, requires_grad=True)
    fut, fm = torch.randn(B, T, 30, 2), torch.ones(B, T, 30, dtype=torch.bool)
    tm = torch.tensor([[True, True, True, False], [True, False, False, False]])
    fn = WinnerTakesAllLoss()
    loss, info = env_wta_loss(fn, trajs, logits, fut, fm, tm)
    flat = torch.cat([trajs[0, :3], trajs[1, :1]]), torch.cat([logits[0, :3], logits[1, :1]])
    ref, _ = fn(flat[0], flat[1], torch.cat([fut[0, :3], fut[1, :1]]), torch.cat([fm[0, :3], fm[1, :1]]))
    assert torch.allclose(loss, ref)
    loss.backward()
    assert (trajs.grad[~tm] == 0).all() and (logits.grad[~tm] == 0).all() and trajs.grad[tm].abs().sum() > 0


def test_full_forward_synthetic_sensors():
    torch.manual_seed(0)
    x = tiny_model_inputs(B=2, T=3, n_real=(3, 2))
    batch = {"points": [torch.randn(60, 5), torch.randn(40, 5)],
             "images": [[torch.rand(3, 32, 32) for _ in range(3)] for _ in range(2)],
             "cam": [torch.randn(4), torch.randn(4)],
             "coop_points": [[torch.randn(30, 5)], []],                        # sample 1: no coop CAV in range
             "coop_images": [[[torch.rand(3, 32, 32) for _ in range(3)]], []],
             **{k: v for k, v in x.items() if k not in ("scene", "cam_embed")}}
    for v2x, use_map in ((True, True), (False, True), (True, False)):
        model = EnvFusion(device="cpu", v2x=v2x, use_map=use_map)
        trajs, logits = model(batch)
        loss, info = env_wta_loss(WinnerTakesAllLoss(), trajs, logits, torch.randn(2, 3, 30, 2),
                                  torch.ones(2, 3, 30, dtype=torch.bool), batch["target_mask"])
        loss.backward()
        assert torch.isfinite(loss) and trajs.shape == (2, 3, 6, 30, 2)
        if v2x:
            assert model.no_coop.grad is not None and model.coop_encoder.shared_encoder.fuse.weight.grad is not None


def test_train_args():
    import contextlib
    import io
    import train
    argv = sys.argv
    quiet = contextlib.redirect_stderr(io.StringIO())
    try:
        quiet.__enter__()
        sys.argv = ["train.py", "--out-dir", "x"]
        a = train.parse_args()
        assert a.task == "ego" and a.v2x is None and a.map_transform == "mirror" and a.split_version == "v2"
        ENV = ["--use-speed", "--normalize-cam", "--drop-cam-xy"]
        sys.argv = ["train.py", "--out-dir", "x", "--task", "env", "--v2x", "on", "--no-map"] + ENV
        a = train.parse_args()
        assert a.task == "env" and a.v2x == "on" and a.no_map
        for bad in (["--task", "env"] + ENV, ["--task", "env", "--v2x", "on", "--no-cpm"] + ENV,
                    ["--task", "env", "--v2x", "off", "--history-frames", "8"] + ENV,
                    ["--task", "env", "--v2x", "off", "--loss", "relaxed_wta"] + ENV, ["--v2x", "on"],
                    ["--task", "env", "--v2x", "on", "--use-speed", "--normalize-cam"],
                    ["--task", "env", "--v2x", "on", "--normalize-cam", "--drop-cam-xy"]):
            sys.argv = ["train.py", "--out-dir", "x"] + bad
            try:
                train.parse_args()
                raise AssertionError(f"accepted {bad}")
            except SystemExit:
                pass
    finally:
        quiet.__exit__(None, None, None)
        sys.argv = argv
    assert train.DEVICE.type == "cpu"


def test_env_cam_keeps_only_speed():
    """env_cam zeroes x, y and the world heading; EnvDataset.__getitem__ applies it (stub PilotDataset, no data)."""
    cam = torch.tensor([0.3, -1.2, 2.5, 0.8])
    out = ed.env_cam(cam)
    assert out.tolist() == [0.0, 0.0, 0.0, pytest_approx(0.8)] and cam[2] == 2.5, "input must not be modified"

    class StubPilot:
        cam_norm = {"mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]}

        def __getitem__(self, r):
            return {"cam": torch.tensor([1.0, 2.0, 3.0, 4.0]), "points": None}

    ds = ed.EnvDataset.__new__(ed.EnvDataset)  # bypasses __init__: no PilotDataset, no metadata
    ds.pilot, ds.cache, ds.rows, ds.v2x, ds.env_norm, ds.lanes = StubPilot(), synth_cache([2], [0, 1]), np.array([0]), "on", None, None
    item = ds[0]
    assert item["cam"].tolist() == [0.0, 0.0, 0.0, 4.0] and item["t_hist"].shape == (2, 8, 2)
    ds.pilot.cam_norm = None
    try:
        ds[0]
        raise AssertionError("loaded a sample without cam_norm")
    except AssertionError as e:
        assert "cam_norm" in str(e)


def pytest_approx(x):
    return float(np.float32(x))


def test_ego_run_epoch_unchanged():
    """train.run_epoch (default ego step) == the committed pre-env run_epoch, on a tiny fake model and batches:
    same losses and bit-identical parameters after a train epoch and a val epoch."""
    import subprocess
    import types
    import train
    revs = subprocess.run(["git", "log", "--format=%H", "--", "train.py"], capture_output=True, text=True).stdout.split()
    base = next(r for r in ["HEAD"] + revs if "def ego_step" not in subprocess.run(
        ["git", "show", f"{r}:train.py"], capture_output=True, text=True).stdout)
    old = types.ModuleType("train_old")
    exec(subprocess.run(["git", "show", f"{base}:train.py"], capture_output=True, text=True, check=True).stdout,
         old.__dict__)

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 6 * 30 * 2 + 6)

        def forward(self, batch):
            out = self.lin(batch["x"])
            return None, out[:, :360].view(-1, 6, 30, 2), out[:, 360:]

    g = torch.Generator().manual_seed(1)
    batches = [{"x": torch.randn(3, 4, generator=g), "gt_traj": torch.randn(3, 30, 2, generator=g),
                "gt_mask": torch.rand(3, 30, generator=g) > 0.1} for _ in range(4)]
    results = []
    for mod in (old, train):
        torch.manual_seed(0)
        m = Fake()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        tr = mod.run_epoch(m, batches, WinnerTakesAllLoss(), opt)
        va = mod.run_epoch(m, batches, WinnerTakesAllLoss(), optimizer=None)
        results.append((tr, va, [p.detach().clone() for p in m.parameters()]))
    (tr0, va0, p0), (tr1, va1, p1) = results
    assert tr0 == tr1 and va0 == va1, (tr0, tr1, va0, va1)
    assert all(torch.equal(a, b) for a, b in zip(p0, p1))


# ---------------------------------------------------------------- Stage 3 / 4: env_metrics, eval_env, analyze_env
import env_metrics as em


def line_fut(n, vx=1.0, vy=0.0, T=30):
    return np.tile(np.arange(1, T + 1)[None, :, None] * np.array([vx, vy])[None, None], (n, 1, 1))


def test_displacement_and_baseline_metrics():
    fut = line_fut(2)
    mask = np.ones((2, 30), bool)
    trajs = np.repeat(fut[:, None], 6, axis=1).copy()
    trajs[:, 1:] += np.array([0.0, 1.0])            # modes 1..5 off by 1 m laterally; mode 0 exact
    trajs[1, 3] += np.array([0.0, 2.0])             # sample 1, mode 3: 3 m off
    logits = np.zeros((2, 6))
    logits[0, 2] = 5.0                              # sample 0: top mode 2 (1 m off)
    logits[1, 3] = 5.0                              # sample 1: top mode 3 (3 m off)
    m = em.displacement_metrics(trajs, logits, fut, mask)
    assert np.allclose(m["minade6"], 0) and np.allclose(m["minfde6"], 0)
    assert np.allclose(m["ade1"], [1, 3]) and np.allclose(m["fde1"], [1, 3])
    assert m["miss1"].tolist() == [0.0, 1.0] and m["miss6"].tolist() == [0.0, 0.0]
    mask2 = mask.copy()
    mask2[0, 20:] = False                           # FDE at the last VALID step
    trajs[0, 0, 20:] += 50.0
    assert np.isclose(em.displacement_metrics(trajs, logits, fut, mask2)["minfde6"][0], 0)
    cv = em.cv_prediction(np.array([[1.0, 0.0], [0.0, 0.0]]))
    b = em.baseline_metrics(cv, fut, mask)
    assert np.isclose(b["ade"][0], 0) and np.isclose(b["ade"][1], np.mean(np.arange(1, 31)))
    s = em.baseline_metrics(np.zeros_like(fut), fut, mask)
    assert np.allclose(s["fde"], 30) and s["miss"].tolist() == [1.0, 1.0]


def test_condition_labels_and_subsets():
    straight = line_fut(1)[0]
    ang = np.radians(np.linspace(0, 90, 31)[1:])
    turn = np.stack([20 * np.sin(ang), 20 * (1 - np.cos(ang))], axis=1)  # quarter circle: 90 deg left turn
    fut = np.stack([straight, turn, turn])
    lab = em.condition_labels(fut, np.ones((3, 30), bool), np.array([True, True, False]))
    assert lab["straight"].tolist() == [True, False, False] and lab["turning"].tolist() == [False, True, False]
    assert lab["stationary"].tolist() == [False, False, True]
    t = {"vis": np.array([0, 1, 0]), "cat4": np.array([0, 2, 3]), **lab}
    m = em.subset_masks(t)
    assert set(m) == set(em.SUBSETS)
    assert m["a_moving"].tolist() == [True, False, False] and m["b_moving"].tolist() == [False, True, False]
    assert m["two_wheeler"].tolist() == [False, True, True] and m["car_moving"].tolist() == [True, False, False]


def test_coverage_counts():
    c = synth_cache([3, 1, 2], [0, 1, 2, 2, 0, 1])
    c["n_removed"] = np.array([1, 0, 4], np.int32)
    off, on = em.coverage(c, [0, 1, 2], "off"), em.coverage(c, [0, 1, 2], "on")
    assert (off["a"], off["b"], off["c"], off["all_kept"], off["removed"]) == (2, 2, 2, 6, 5)
    assert off["predicted"] == 2 and on["predicted"] == 4
    assert np.isclose(on["coverage"], 4 / 6) and np.isclose(on["coverage_incl_removed"], 4 / 11)


def test_rule_forms():
    scenes = list(range(8))
    mk = lambda vals, all_v: {**{s: v for s, v in zip(scenes, vals)}, "ALL": all_v}
    better = [mk([1.0] * 8, 1.00 + d) for d in (0.0, 0.01, -0.01)]
    worse = [mk([1.5] * 8, 1.50 + d) for d in (0.0, 0.01, -0.01)]
    r = em.rule(better, worse, scenes)
    assert r["form"] == "full" and r["wins"] == 8 and r["helps"]
    mixed = [mk([1.0] * 6 + [2.0] * 2, 1.0 + d) for d in (0.0, 0.01, -0.01)]
    r = em.rule(mixed, worse, scenes)
    assert r["wins"] == 6 and not r["helps"], "6/8 scenes must not pass"
    noisy = [mk([1.0] * 8, v) for v in (1.0, 1.4, 1.8)]           # big seed std
    assert not em.rule(noisy, worse, scenes)["helps"]
    holes = [{**x, 7: None} for x in better]                # one scene without samples -> pooled form
    r = em.rule(holes, worse, scenes)
    assert r["form"] == "pooled" and r["scenes_with_samples"] == 7 and r["helps"]
    base = [mk([1.2] * 8, 1.2)] * 3                               # deterministic baseline: sd 0
    r = em.rule(better, base, scenes)
    assert r["sd_without"] == 0 and r["helps"]
    assert not em.rule(base, better, scenes)["helps"]


def test_eval_collect_and_tables():
    import eval_env
    c = synth_cache([3, 2], [0, 1, 2, 0, 0])
    c["fut_own"][:] = line_fut(5)
    items = [ed.sample_features(c, 0, "on"), ed.sample_features(c, 1, "on")]
    batch = ed.pad_targets(items)
    batch.update(scene=[16, 41], town=["Town05", "Town05"])
    B, T = batch["target_mask"].shape
    trajs = torch.zeros(B, T, 6, 30, 2)
    trajs[..., 0] = torch.arange(1, 31, dtype=torch.float32)       # every mode exact
    t = eval_env.concat([eval_env.collect(batch, trajs, torch.zeros(B, T, 6))])
    assert t["tid"].tolist() == [0, 1, 3, 4] and t["scene"].tolist() == [16, 16, 41, 41]
    assert np.allclose(t["minade6"], 0) and np.allclose(t["stay_fde"], 30)
    lines = eval_env.table_lines(t, [("Town05", 16), ("Town05", 41)])
    allrow = [l for l in lines if l.split()[1:3] == ["ALL", "all"]][0]
    assert int(allrow.split()[3]) == 4
    assert any(l.split()[1:4] == ["05#41", "b", "0"] for l in lines)


def test_analyze_env_end_to_end():
    """analyze_env.py on fake per-target npz files (tmp dir): V2X clearly better than EgoOnly on (a),
    map with no effect, V2X better than CV on (b)."""
    import json as _json
    import subprocess
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(3)
    scenes = [("Town03", 85), ("Town04", 30), ("Town05", 3), ("Town05", 14), ("Town05", 15), ("Town05", 16),
              ("Town05", 21), ("Town05", 22)]
    n = 400
    base = {"tid": np.arange(n), "scene": np.array([scenes[i % 8][1] for i in range(n)]),
            "town": np.array([scenes[i % 8][0] for i in range(n)]), "vis": (np.arange(n) % 3 == 0).astype(int),
            "cat4": np.arange(n) % 4, "moving": (np.arange(n) // 8) % 2 == 0}   # every scene has moving targets
    base["turning"] = base["moving"] & ((np.arange(n) // 16) % 2 == 0)
    base["straight"] = base["moving"] & ~base["turning"]
    base["stationary"] = ~base["moving"]
    with tempfile.TemporaryDirectory() as tmp:
        for g, v2x, nomap, level in (("EgoOnly", "off", False, 3.0), ("V2X", "on", False, 2.0), ("V2Xnomap", "on", True, 2.0)):
            for s in range(3):
                keep = base["vis"] == 0 if v2x == "off" else np.ones(n, bool)
                t = {k: v[keep] for k, v in base.items()}
                m = len(t["tid"])
                for k in em.METRICS:
                    t[k] = level + rng.normal(0, 0.01, m)
                for k in ("cv_ade", "cv_fde", "cv_miss", "stay_ade", "stay_fde", "stay_miss"):
                    t[k] = np.full(m, 4.0)
                meta = {"split": "tuning", "split_version": "v2", "map_transform": "mirror", "v2x": v2x, "no_map": nomap,
                        "scenes": [list(x) for x in scenes],
                        "coverage": {"predicted": m, "all_kept": n + 10, "a": 1, "b": 1, "c": 10, "removed": 5,
                                     "coverage": m / (n + 10), "coverage_incl_removed": m / (n + 15)}}
                os.makedirs(f"{tmp}/runs/E_{g}_seed{s}")
                np.savez(f"{tmp}/runs/E_{g}_seed{s}/eval_env_tuning_checkpoint_epoch12.npz", **t, meta=_json.dumps(meta))
        out = subprocess.run([sys.executable, os.path.join(here, "analyze_env.py")], cwd=tmp, capture_output=True,
                             text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert out.returncode == 0, out.stderr[-2000:]
    import re as _re
    v = {_re.split(r"\s{2,}", l.strip())[0]: l for l in out.stdout.split("VERDICTS:")[1].split("\nCoverage")[0].splitlines()
         if l.strip()}
    assert "HELPS" in v["(1) V2X on (a)"] and "HELPS" in v["(2) V2X on (b) vs constant velocity"]
    assert "no detectable" in v["(3) map, moving targets"]
    assert out.stdout.count("form=full (scenes with samples 8/8)") == 4 and "Coverage per model" in out.stdout


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  OK    {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed" + ("" if not failed else "  -> FAIL"))
    sys.exit(1 if failed else 0)
