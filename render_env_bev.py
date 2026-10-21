"""
render_env_bev.py
Bird's-eye-view prediction videos of the env task (Part 2; runs/DECISION_RULES_ENV.md, runs/DECISION_RULES_RSU.md)
for the defense presentation. TUNING split v2 only: a scene outside scene_split.TUNING_SCENES_V2 is refused, and
the model inputs are taken from three_way_scene_split(..., split_version="v2")'s tuning subset. No test sample is
read. Seed 0, checkpoint_epoch12.pt for every model.

  select   Scene selection by rule, printed with the reason (seed-0 tuning npz files + the env cache):
             gain     the largest per-scene V2X gain vs EgoOnly (--gain, see GAINS);
             median   the 4th of the 8 tuning scenes ranked by the same gain (lower median; the 5th if the 4th is
                      the gain scene or scene 30);
             failure  Town04 #30 (named by the user).
           Ego per scene: the CAV with the most moving (b) targets over its env samples in that scene (ties: lower id).
  render   (--print: vector PDFs of the --stills / --all-modes-stills frames for the thesis, see render_frame;
           --no-pred: one panel without predictions, e.g. for the introduction) PNG frames, an mp4 at 2.5 fps (half of V2X-Sim's 5 Hz; ffmpeg) and optional 300 dpi thesis stills, for one
           scene / ego and one model (--runs A) or two side by side (--runs A,B). Predictions are computed once per
           (run, scene, ego) and cached in <out>/pred/; their minADE6 / ADE1 must equal the run's
           eval_env_tuning_checkpoint_epoch12.npz values for the same targets (max |diff| <= 1e-3 m).

Per frame (ego frame, ego heading up, fixed VIEW, scale bar): map v2 lanes (env_map.LanePolylines, mirror transform),
the ego, the other CAVs (1-4) and the RSU (agent 0; an edge marker when outside the view), every kept target of the
env cache as its box coloured by visibility class (ego / coop CAV only / RSU only / nobody), its ground-truth 6 s
future (dashed, drawn above the predictions; not drawn when shorter than STILL_M) and, for the model's known targets,
its top-1 mode (thick, arrowhead at the predicted end; a top-1 path shorter than STILL_M, moving or not, is a dot with a
white rim on the target instead) plus, for moving targets only, the other 5 modes (thin, no arrow; --all-modes: for
every known target; --all-modes-stills: extra all-modes stills). Every line has a thin white outline (the convention of
overlay_carla_replay.py's CARLA video).
CAV / RSU labels sit on a white background, placed beside the marker off the target boxes. A title line above each
panel gives town, scene, ego, time, split and model. Targets some CAV or the RSU sees but this
panel's model does not take as input (e.g. (b) in an E_EgoOnly panel) are drawn faded, with their ground truth and no
prediction; "nobody" targets are hollow grey. Lanes: consecutive pieces of a lane, and a lane's end to a lane start
within LANE_JOIN_M and LANE_JOIN_DEG, are joined (drawing only; the model's tokens are unchanged).
Targets removed by the cache's window filter are not in the cache and are not drawn.

Refuses to start while a run script, train.py or an eval script is running (ps).
Run in gpu-test-venv from the data checkout:
  nice -n 19 python render_env_bev.py select
  nice -n 19 python render_env_bev.py render --scene 30 --runs E_V2X_seed0,E_EgoOnly_seed0 --frames 10
"""
import os
import re
import sys
import json
import math
import argparse
import subprocess

import numpy as np

from scene_split import TUNING_SCENES_V2

RUNS_DIR = "runs"
CKPT = "checkpoint_epoch12.pt"
FPS = 2.5                                 # half of V2X-Sim's 5 Hz
FRAME_RATE_HZ = 5.0
VIEW_LAT, VIEW_BACK, VIEW_FWD = 50.0, 40.0, 60.0   # m: right / left half-width, behind, ahead of the ego
SCALE_BAR_M = 20.0
EGO_SIZE = (2.0, 4.6)                     # w, l (m); the ego's own box is not in the env cache
MIN_BOX_W = 0.6                           # zero-width boxes (most bicycles) are drawn this wide
LANE_RANGE_M = 110.0
LANE_JOIN_M, LANE_JOIN_DEG = 3.0, 30.0    # drawing only: join a lane's end to a lane start this close / aligned
FADE = 0.3                                # alpha of targets that are not an input of the panel's model
GT_LW = 2.2                               # ground-truth line width (drawn above the predictions)
OUTLINE_PT = 1.6                          # white outline under the lines: this much wider (points, x k)
ARROW_L_PT, ARROW_W_PT = 7.0, 6.0         # top-1 arrowhead length / base width (points, x k)
STILL_M = 1.0                             # top-1 path shorter than this: dot on the target; ground truth: not drawn
LABEL_CHAR_EM, LABEL_H_EM, LABEL_GAP_PT = 0.66, 1.6, 11.5   # CAV / RSU label: approx. text box (em), marker gap
LABEL_OFFSETS = ((1, 1), (-1, 1), (1, -1), (-1, -1), (0, 1), (0, -1), (1, 0), (-1, 0))   # tried in this order
CLASS_NAMES = ("seen by the ego", "coop CAV only", "RSU only", "nobody")
CLASS_COLORS = ("#0072B2", "#E69F00", "#009E73", "#8C8C8C")   # Okabe-Ito
EGO_COLOR, CAV_COLOR, RSU_COLOR, GT_COLOR, LANE_COLOR = "#000000", "#CC79A7", "#D55E00", "#000000", "#BEBEBE"
COOP_CAVS = (1, 2, 3, 4)
RSU_AGENT = 0
MODEL_LABELS = {"E_V2X": "E_V2X (ego + coop CAVs)", "E_EgoOnly": "E_EgoOnly (ego sensors only)",
                "E_V2XRSU": "E_V2X+RSU (ego + coop CAVs + RSU)", "E_V2Xnomap": "E_V2X without map"}
BUSY = re.compile(r"(run_\w+\.sh|train\.py|eval_\w+\.py)")
TUNING = {s: town for town, scenes in TUNING_SCENES_V2.items() for s in scenes}
FAILURE_SCENE = 30


def busy():
    out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines() if BUSY.search(l) and int(l.split()[0]) != os.getpid()]


def run_cfg(run):
    with open(os.path.join(RUNS_DIR, run, "config.json")) as f:
        cfg = json.load(f)
    assert cfg.get("task") == "env" and cfg.get("split_version") == "v2" and cfg["map_transform"] == "mirror", run
    return cfg


def label(run):
    base, seed = run.rsplit("_seed", 1)
    return f"{MODEL_LABELS.get(base, base)}, seed {seed}, epoch 12"


def tuning_npz(run):
    z = np.load(os.path.join(RUNS_DIR, run, f"eval_env_tuning_{CKPT[:-3]}.npz"))
    meta = json.loads(str(z["meta"]))
    assert meta["split"] == "tuning" and meta["ckpt"].endswith(CKPT), meta
    return z


def vis_class(vis, rsu_vis):
    """0 ego, 1 coop CAV only, 2 RSU only, 3 nobody."""
    vis = np.asarray(vis)
    return np.where(vis <= 1, vis, np.where(np.asarray(rsu_vis, bool), 2, 3))


def scene_rows(cache, scene):
    from env_dataset import env_sample_rows
    rows = env_sample_rows(cache)
    return rows[cache["sample_scene"][rows] == scene]


def ego_rule(cache, scene):
    """-> (agent, {agent: moving (b) target count}) for the scene's env samples."""
    counts = {}
    for r in scene_rows(cache, scene):
        lo, hi = cache["sample_ptr"][r], cache["sample_ptr"][r + 1]
        a = int(cache["sample_agent"][r])
        counts[a] = counts.get(a, 0) + int(((cache["vis"][lo:hi] == 1) & (cache["speed"][lo:hi] > 0)).sum())
    return min(counts, key=lambda a: (-counts[a], a)), counts


# ---------------------------------------------------------------- select
GAINS = {
    "b_cv": "moving (b) targets: constant-velocity ADE - E_V2X minADE6 (EgoOnly has no prediction for (b))",
    "a_matched": "moving (a) targets (both models predict them): EgoOnly minADE6 - E_V2X minADE6",
}


def cmd_select(args):
    from env_dataset import load_env_cache
    v, e = tuning_npz("E_V2X_seed0"), tuning_npz("E_EgoOnly_seed0")
    assert set(v["scene"].tolist()) == set(TUNING), "npz scenes differ from TUNING_SCENES_V2"
    pos = {t: i for i, t in enumerate(v["tid"].tolist())}
    ie = np.array([pos[t] for t in e["tid"].tolist()])
    rows = {}
    print(f"{'scene':>9s} | {'n b_mov':>7s} {'V2X min6':>8s} {'CV ade':>7s} {'b_cv':>7s} | "
          f"{'n a_mov':>7s} {'Ego min6':>8s} {'V2X min6':>8s} {'a_matched':>9s}")
    for s in sorted(TUNING):
        mb = (v["scene"] == s) & (v["vis"] == 1) & v["moving"]
        ma = (e["scene"] == s) & e["moving"]
        b_v, b_cv = v["minade6"][mb].mean(), v["cv_ade"][mb].mean()
        a_e, a_v = e["minade6"][ma].mean(), v["minade6"][ie[ma]].mean()
        rows[s] = {"b_cv": b_cv - b_v, "a_matched": a_e - a_v}
        print(f"{TUNING[s][-2:]}#{s:<6d} | {mb.sum():7d} {b_v:8.3f} {b_cv:7.3f} {b_cv - b_v:7.3f} | "
              f"{ma.sum():7d} {a_e:8.3f} {a_v:8.3f} {a_e - a_v:9.3f}")
    g = args.gain
    print(f"\nGain ({g}): {GAINS[g]}; seed 0, epoch 12, tuning split v2")
    ranked = sorted(TUNING, key=lambda s: (rows[s][g], s))
    gain = ranked[-1]
    median = next(s for s in ranked[3:] if s not in (gain, FAILURE_SCENE))
    cache = load_env_cache()
    picks = [("GAIN", gain, f"largest {g} = {rows[gain][g]:.3f} m"),
             ("MEDIAN", median, f"{g} = {rows[median][g]:.3f} m, rank {ranked.index(median) + 1} of 8 ascending "
                                f"(order: {', '.join(f'{s}={rows[s][g]:.2f}' for s in ranked)})"),
             ("FAILURE", FAILURE_SCENE, f"named by the user; {g} = {rows[FAILURE_SCENE][g]:.3f} m")]
    for tag, s, why in picks:
        a, counts = ego_rule(cache, s)
        print(f"{tag:8s} {TUNING[s]} #{s}, ego CAV {a}: {why}; ego rule: most moving (b) targets "
              f"{ {k: counts[k] for k in sorted(counts)} }")


# ---------------------------------------------------------------- predictions
def predict(run, scene, agent, out_path):
    import torch
    from torch.utils.data import DataLoader, Subset
    from env_dataset import EnvDataset, env_collate_fn
    from env_model import EnvFusion
    from scene_split import three_way_scene_split
    rd = os.path.join(RUNS_DIR, run)
    cfg = run_cfg(run)
    use_rsu = cfg.get("use_rsu", False)
    with open(os.path.join(rd, "env_norm.json")) as f:
        env_norm = json.load(f)
    ds = EnvDataset(v2x=cfg["v2x"], use_map=not cfg["no_map"], map_transform=cfg["map_transform"], env_norm=env_norm,
                    use_image_cache=cfg["use_image_cache"], use_speed=cfg["use_speed"], drop_cam_xy=cfg["drop_cam_xy"],
                    use_rsu=use_rsu)
    with open(os.path.join(rd, "cam_norm.json")) as f:
        ds.cam_norm = json.load(f)
    assert ds.cache["manifest"]["git_commit"] == cfg["env_cache_commit"], "env cache differs from the run's"
    _, tuning, _ = three_way_scene_split(ds, seed=42, split_version="v2")
    idx = [i for i in tuning.indices if ds.samples[i][1] == scene and ds.samples[i][0] == agent]
    assert idx, f"no tuning samples for scene {scene}, ego {agent}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnvFusion(device=device, v2x=cfg["v2x"] == "on", use_map=not cfg["no_map"], use_rsu=use_rsu).to(device)
    model.load_state_dict(torch.load(os.path.join(rd, CKPT), map_location=device)["model_state_dict"])
    model.eval()
    tid, trajs, logits = [], [], []
    with torch.no_grad():
        for batch in DataLoader(Subset(ds, idx), batch_size=8, shuffle=False, num_workers=2, collate_fn=env_collate_fn):
            t, l = model(batch)
            tm = batch["target_mask"]
            tid.append(batch["t_tid"][tm].numpy())
            trajs.append(t.cpu()[tm].numpy())
            logits.append(l.cpu()[tm].numpy())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, tid=np.concatenate(tid), trajs=np.concatenate(trajs).astype(np.float32),
             logits=np.concatenate(logits).astype(np.float32),
             meta=json.dumps({"run": run, "ckpt": CKPT, "scene": scene, "agent": agent, "split": "tuning",
                              "n_samples": len(idx), "device": str(device), "code_commit": commit}))
    print(f"  {run}: {len(idx)} samples on {device} -> {out_path}")


def load_preds(run, scene, agent, out_dir, cache):
    from env_metrics import displacement_metrics
    path = os.path.join(out_dir, "pred", f"{run}_scene{scene}_ego{agent}.npz")
    if not os.path.exists(path):
        predict(run, scene, agent, path)
    z = np.load(path)
    tid = z["tid"]
    m = displacement_metrics(z["trajs"], z["logits"], cache["fut_own"][tid], cache["fut_mask"][tid])
    ref = tuning_npz(run)
    pos = {t: i for i, t in enumerate(ref["tid"].tolist())}
    j = np.array([pos[t] for t in tid.tolist()])
    diff = max(np.abs(m[k] - ref[k][j]).max() for k in ("minade6", "ade1"))
    print(f"  {run}: {len(tid)} predicted targets; max |minADE6 / ADE1 - eval npz| = {diff:.2e} m")
    assert diff <= 1e-3, f"{run}: predictions do not reproduce eval_env_tuning npz"
    return {int(t): (z["trajs"][i], z["logits"][i]) for i, t in enumerate(tid.tolist())}


# ---------------------------------------------------------------- geometry (ego frame: x ahead, y left)
def to_frame(xy, origin, yaw):
    """build_env_target_cache.to_frame: world -> frame at origin with heading yaw."""
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx, dy = xy[..., 0] - origin[0], xy[..., 1] - origin[1]
    return np.stack([dx * c - dy * s, dx * s + dy * c], axis=-1)


def own_to_ego(p, cur, yaw):
    """p [..., 2] in a target's own frame -> ego frame, given its cur_ego [2] and yaw_ego."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([p[..., 0] * c - p[..., 1] * s + cur[0], p[..., 0] * s + p[..., 1] * c + cur[1]], axis=-1)


def scr(p):
    """ego frame -> screen (right, up): heading up."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([-p[..., 1], p[..., 0]], axis=-1)


def box(cur, yaw, w, l):
    w = max(w, MIN_BOX_W)
    corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    return scr(own_to_ego(corners, cur, yaw))


def lane_segments(lanes):
    """[M, 2, 2] world segments for drawing, and [M] is_parking: consecutive points of each piece; the last point of
    a piece to the first of the next piece of the same lane (export_lane_polylines.py cuts a lane into consecutive
    pieces); and a lane's last point to the nearest lane start within LANE_JOIN_M whose direction and the step to it
    agree with the end's direction within LANE_JOIN_DEG (road / section / junction transitions)."""
    from scipy.spatial import cKDTree
    z = np.load(f"lane_polylines_{lanes.town.lower()}.npz")
    key = [tuple(k) + (str(t),) for k, t in zip(z["piece_lane"], z["piece_type"])]
    assert len(key) == len(lanes.piece_points)
    pieces = [idx[idx >= 0] for idx in lanes.piece_points]
    seg, park = [], []
    for idx in pieces:
        if len(idx) >= 2:
            seg.append(np.stack([lanes.pts[idx[:-1]], lanes.pts[idx[1:]]], axis=1))
            park.append(lanes.is_parking[idx[:-1]] > 0)
    starts, ends = [], []
    for p in range(len(pieces)):
        if p + 1 < len(pieces) and key[p + 1] == key[p]:
            seg.append(lanes.pts[[pieces[p][-1], pieces[p + 1][0]]][None])
            park.append(lanes.is_parking[pieces[p][-1:]] > 0)
        else:
            ends.append(pieces[p][-1])
        if p == 0 or key[p - 1] != key[p]:
            starts.append(pieces[p][0])
    starts, ends = np.array(starts), np.array(ends)
    tree = cKDTree(lanes.pts[starts])
    cos_max = math.cos(math.radians(LANE_JOIN_DEG))
    n_join = 0
    for e in ends:
        best = None
        for j in tree.query_ball_point(lanes.pts[e], LANE_JOIN_M):
            s0 = starts[j]
            step = lanes.pts[s0] - lanes.pts[e]
            dist = float(np.hypot(*step))
            if s0 == e or dist < 1e-6:
                continue
            u = np.array([math.cos(lanes.yaw[e]), math.sin(lanes.yaw[e])])
            if math.cos(lanes.yaw[s0] - lanes.yaw[e]) >= cos_max and step @ u / dist >= cos_max \
                    and (best is None or dist < best[0]):
                best = (dist, s0)
        if best is not None:
            seg.append(lanes.pts[[e, best[1]]][None])
            park.append(lanes.is_parking[[e]] > 0)
            n_join += 1
    print(f"  lanes {lanes.town}: {len(pieces)} pieces, {len(ends)} lane ends, {n_join} joined to a lane start")
    return np.concatenate(seg), np.concatenate(park)


def in_view(p):
    return (-VIEW_LAT <= p[..., 0]) & (p[..., 0] <= VIEW_LAT) & (-VIEW_BACK <= p[..., 1]) & (p[..., 1] <= VIEW_FWD)


# ---------------------------------------------------------------- drawing
# k scales line widths / marker sizes; fs is the text size (pt) of labels, scale bar and legend. VIDEO: 7 in panels
# at 110 dpi; PRINT: vector PDF at its final size (16 cm wide side by side, 8 cm single), every text >= 8 pt.
VIDEO = {"k": 1.0, "fs": 8.0, "fs_scale": 9.0, "fs_legend": 8.5, "print": False}
PRINT = {"k": 0.5, "fs": 8.0, "fs_scale": 8.0, "fs_legend": 8.0, "print": True}
PRINT_W_CM = {1: 8.0, 2: 16.0}
PRINT_MARGIN_IN, PRINT_GAP_IN, PRINT_TITLE_IN, PRINT_ROW_IN = 0.06, 0.08, 0.2, 0.165
PANEL_LABELS = {"E_V2X": "E_V2X", "E_EgoOnly": "E_EgoOnly", "E_V2XRSU": "E_V2X+RSU", "E_V2Xnomap": "E_V2X without map"}


def outline(lw, k, alpha=1.0):
    import matplotlib.patheffects as pe
    return [pe.Stroke(linewidth=lw + OUTLINE_PT * k, foreground="white", alpha=alpha), pe.Normal()]


def path_len(p):
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def arrow_line(ax, p, color, lw, k, m_per_pt, zorder):
    """Top-1 line (screen m) ending in an arrowhead at the predicted end, pointing along the path's last ARROW_L_PT;
    the line stops at the arrowhead's base so its round cap does not poke through the tip."""
    from matplotlib.patches import Polygon
    L, W = ARROW_L_PT * k * m_per_pt, ARROW_W_PT * k * m_per_pt
    seg = np.hypot(*np.diff(p, axis=0).T)
    back = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])     # path length from each point to the end
    tip = p[-1]
    if back[0] <= L:                       # shorter than the arrowhead: arrow along the whole path
        base, body = p[0], p[:1]
    else:
        i = int(np.nonzero(back > L)[0][-1])                        # back[i] > L >= back[i + 1]
        base = p[i + 1] + (p[i] - p[i + 1]) * (L - back[i + 1]) / seg[i]
        body = np.vstack([p[:i + 1], base])
    if len(body) >= 2:
        ax.plot(body[:, 0], body[:, 1], color=color, lw=lw, zorder=zorder, solid_capstyle="round",
                path_effects=outline(lw, k))
    u = (tip - base) / np.hypot(*(tip - base))
    n = np.array([-u[1], u[0]]) * W / 2
    ax.add_patch(Polygon(np.array([tip, tip - u * L + n, tip - u * L - n]), closed=True, facecolor=color,
                         edgecolor="white", lw=OUTLINE_PT * k / 2, joinstyle="miter", zorder=zorder + 0.1))


def draw_panel(ax, fr, run, known, preds, all_modes, st):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon
    k = st["k"]
    ax.set_xlim(-VIEW_LAT, VIEW_LAT)
    ax.set_ylim(-VIEW_BACK, VIEW_FWD)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8 * k)
    pos = ax.get_position()
    m_per_pt = max(2 * VIEW_LAT / (pos.width * ax.figure.get_figwidth() * 72),
                   (VIEW_BACK + VIEW_FWD) / (pos.height * ax.figure.get_figheight() * 72))
    seg, park = fr["lanes"], fr["lanes_park"]
    ax.add_collection(LineCollection(seg[~park], colors=LANE_COLOR, linewidths=0.9 * k, zorder=1))
    if park.any():
        ax.add_collection(LineCollection(seg[park], colors=LANE_COLOR, linewidths=0.9 * k, linestyles=":", zorder=1))
    for j, t in enumerate(fr["tid"]):
        cls, cur, yaw = fr["cls"][j], fr["cur"][j], fr["yaw"][j]
        col = CLASS_COLORS[cls]
        is_known = known[j]
        faded = not is_known and cls != 3      # seen by a CAV / the RSU, but not an input of this model
        gt = scr(np.vstack([cur, fr["fut_ego"][j]]))
        if path_len(gt) >= STILL_M:          # shorter ground truth: no line, no outline
            a = FADE if faded else 0.8
            ax.plot(gt[:, 0], gt[:, 1], color=GT_COLOR, lw=GT_LW * k, ls=(0, (3, 2)), alpha=a, zorder=4.8,
                    path_effects=outline(GT_LW * k, k, a))
        if cls == 3:
            ax.add_patch(Polygon(box(cur, yaw, *fr["wl"][j]), closed=True, zorder=5, facecolor="white", edgecolor=col,
                                 lw=1.4 * k))
        else:
            ax.add_patch(Polygon(box(cur, yaw, *fr["wl"][j]), closed=True, zorder=5, facecolor=col, edgecolor="black",
                                 lw=0.4 * k, alpha=FADE if faded else 0.95))
        if is_known and int(t) in preds:
            trajs, logits = preds[int(t)]
            top = int(np.argmax(logits))
            for m in range(trajs.shape[0]):
                p = scr(np.vstack([cur, own_to_ego(trajs[m], cur, yaw)]))
                if m == top:
                    if path_len(p) < STILL_M:
                        ax.plot(*p[0], "o", ms=4.0 * k, color=col, mec="white", mew=1.0 * k, zorder=5.1)
                    else:
                        arrow_line(ax, p, col, 2.4 * k, k, m_per_pt, 4.5)
                elif all_modes or fr["moving"][j]:
                    ax.plot(p[:, 0], p[:, 1], color=col, lw=0.9 * k, alpha=0.55, zorder=4,
                            path_effects=outline(0.9 * k, k, 0.55))
    obstacles = [np.r_[b.min(axis=0), b.max(axis=0)] for b in
                 (box(fr["cur"][j], fr["yaw"][j], *fr["wl"][j]) for j in range(len(fr["tid"])))]
    obstacles.append(np.r_[-EGO_SIZE[0] / 2, -EGO_SIZE[1] / 2, EGO_SIZE[0] / 2, EGO_SIZE[1] / 2])
    for a, (xy, yaw) in fr["cavs"].items():
        p = scr(xy)
        if in_view(p):
            ax.plot(*p, marker="D", ms=10 * k, mfc="none", mec=CAV_COLOR, mew=2.2 * k, zorder=6)
            place_label(ax, p, f"CAV {a}", CAV_COLOR, obstacles, st, m_per_pt)
    if fr["rsu"] is not None:
        p = scr(fr["rsu"])
        inside = bool(in_view(p))
        q = p if inside else np.array([np.clip(p[0], -VIEW_LAT + 2.5, VIEW_LAT - 2.5),
                                       np.clip(p[1], -VIEW_BACK + 2.5, VIEW_FWD - 2.5)])
        ax.plot(*q, marker="^", ms=12 * k, mfc=RSU_COLOR if inside else "none", mec=RSU_COLOR, mew=2 * k, zorder=6)
        place_label(ax, q, "RSU" if inside else f"RSU ({math.hypot(*p):.0f} m)", RSU_COLOR, obstacles, st, m_per_pt)
    ax.add_patch(Polygon(box(np.zeros(2), 0.0, EGO_SIZE[0], EGO_SIZE[1]), closed=True, facecolor=EGO_COLOR,
                         edgecolor=EGO_COLOR, zorder=6))
    x0, y0 = VIEW_LAT - 4 - SCALE_BAR_M, -VIEW_BACK + 4
    ax.plot([x0, x0 + SCALE_BAR_M], [y0, y0], color="black", lw=3 * k, solid_capstyle="butt", zorder=7)
    ax.text(x0 + SCALE_BAR_M / 2, y0 + 1.5, f"{SCALE_BAR_M:g} m", ha="center", va="bottom", fontsize=st["fs_scale"],
            zorder=7)
    if st["print"]:   # model name only; scene, time and split go in the LaTeX caption
        if run is not None:
            ax.set_title(PANEL_LABELS.get(run.rsplit("_seed", 1)[0], run), fontsize=9, fontweight="bold", loc="left",
                         pad=3)
    else:
        ax.set_title(f"{fr['town']} #{fr['scene']}, ego CAV {fr['agent']}  |  t = {fr['frame'] / FRAME_RATE_HZ:.1f} s "
                     f"(frame {fr['frame']})  |  tuning split\n{label(run) if run else 'no predictions'}",
                     fontsize=9.5, loc="left", pad=5)


def place_label(ax, p, text, color, obstacles, st, m_per_pt):
    """Text on a white background beside marker p (screen m): the candidate offset whose approximate text box overlaps
    the least target-box area (bounding boxes) and stays in the view; obstacles gets the chosen text box."""
    fs = st["fs"]
    w, h = (LABEL_CHAR_EM * len(text) + 0.6) * fs * m_per_pt, LABEL_H_EM * fs * m_per_pt
    gap = LABEL_GAP_PT * st["k"] * m_per_pt
    best = None
    for dx, dy in LABEL_OFFSETS:
        x = p[0] + dx * gap
        y = p[1] + dy * gap
        x0 = x - (w if dx < 0 else w / 2 if dx == 0 else 0)
        y0 = y - (h if dy < 0 else h / 2 if dy == 0 else 0)
        r = np.r_[x0, y0, x0 + w, y0 + h]
        over = sum(max(0.0, min(r[2], o[2]) - max(r[0], o[0])) * max(0.0, min(r[3], o[3]) - max(r[1], o[1]))
                   for o in obstacles)
        out = r[0] < -VIEW_LAT or r[2] > VIEW_LAT or r[1] < -VIEW_BACK or r[3] > VIEW_FWD
        if best is None or (out, over) < best[0]:
            best = ((out, over), x, y, dx, dy, r)
    _, x, y, dx, dy, r = best
    obstacles.append(r)
    ax.text(x, y, text, ha="right" if dx < 0 else "center" if dx == 0 else "left",
            va="top" if dy < 0 else "center" if dy == 0 else "bottom", fontsize=fs, color=color, fontweight="bold",
            zorder=7, clip_on=True, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.95))


def legend_handles(all_modes, st, no_pred):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    k, short = st["k"], st["print"]
    names = ("seen by the ego", "coop CAV only", "RSU only", "seen by nobody") if short else \
        tuple(f"target, {n}" for n in CLASS_NAMES)          # print: the caption says boxes are targets
    h = [Patch(facecolor=c, edgecolor="black", lw=0.4 * k, label=n) for c, n in zip(CLASS_COLORS[:3], names[:3])]
    h.append(Patch(facecolor="white", edgecolor=CLASS_COLORS[3], lw=1.4 * k, label=names[3]))
    if not no_pred:
        h.append(Patch(facecolor=CLASS_COLORS[1], edgecolor="black", lw=0.4 * k, alpha=FADE,
                       label="faded: not a model input" if short else
                       "faded: not an input of this model (no prediction)"))
    h.append(Line2D([], [], color=GT_COLOR, lw=GT_LW * k, ls=(0, (3, 2)), label="ground truth, 6 s",
                    path_effects=outline(GT_LW * k, k)))
    if not no_pred:   # entries as in overlay_carla_replay.py's video
        h += [ArrowHandle(label="prediction, top-1 (6 s, arrow = predicted end)", k=k),
              Line2D([], [], ls="", marker="o", ms=4.0 * k, color="#555555", mec="white", mew=1.0 * k,
                     label="dot: predicted to stay (< 1 m in 6 s)"),
              Line2D([], [], color="#555555", lw=0.9 * k, alpha=0.55, path_effects=outline(0.9 * k, k, 0.55),
                     label="other 5 modes" + ("" if all_modes else " (moving)"))]
    h += [Line2D([], [], color=LANE_COLOR, lw=1.2 * k, label="lane centre line" + ("" if short else " (map v2)")),
          Patch(facecolor=EGO_COLOR, label="ego (heading up)" if short else "ego vehicle (heading up)"),
          Line2D([], [], ls="", marker="D", ms=8 * k, mfc="none", mec=CAV_COLOR, mew=2 * k, label="other CAV"),
          Line2D([], [], ls="", marker="^", ms=9 * k, mfc=RSU_COLOR, mec=RSU_COLOR,
                 label="RSU (hollow: off view)" if short else "RSU (hollow: outside view)")]
    return h


class ArrowHandle:
    """Legend entry for the top-1 line: drawn by arrow_handler() as a grey line with an arrowhead at its right end."""
    def __init__(self, label, k):
        self.label, self.k = label, k

    def get_label(self):
        return self.label


def arrow_handler():
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    class ArrowHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            k = orig_handle.k
            y, x0, x1 = height / 2 - ydescent, -xdescent, width - xdescent
            L, W = ARROW_L_PT * k, ARROW_W_PT * k
            ln = Line2D([x0, x1 - L], [y, y], color="#555555", lw=2.4 * k, solid_capstyle="round",
                        path_effects=outline(2.4 * k, k), transform=trans)
            head = Polygon([[x1, y], [x1 - L, y + W / 2], [x1 - L, y - W / 2]], closed=True, facecolor="#555555",
                           edgecolor="white", lw=OUTLINE_PT * k / 2, joinstyle="miter", transform=trans)
            return [ln, head]
    return ArrowHandler()


def render_frame(fr, runs, known, preds, path, dpi, all_modes, st=VIDEO):
    """runs: one or two run names, or [None] (--no-pred: one panel, every target as an input, no predictions)."""
    import matplotlib.pyplot as plt
    n = len(runs)
    no_pred = runs == [None]
    handles = legend_handles(all_modes, st, no_pred)
    if st["print"]:
        ncol = 3 if n == 2 else 2
        rows = -(-len(handles) // ncol)
        W = PRINT_W_CM[n] / 2.54
        panel = (W - 2 * PRINT_MARGIN_IN - (n - 1) * PRINT_GAP_IN) / n
        panel_h = panel * (VIEW_BACK + VIEW_FWD) / (2 * VIEW_LAT)
        top = PRINT_TITLE_IN if not no_pred else PRINT_MARGIN_IN
        leg = rows * PRINT_ROW_IN + 0.08
        H = top + panel_h + leg
        fig = plt.figure(figsize=(W, H))
        axes = [fig.add_axes([(PRINT_MARGIN_IN + i * (panel + PRINT_GAP_IN)) / W, leg / H, panel / W, panel_h / H])
                for i in range(n)]
        lg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=ncol,
                        fontsize=st["fs_legend"], frameon=False, handlelength=1.5, columnspacing=0.8,
                        handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3,
                        handler_map={ArrowHandle: arrow_handler()})
    else:
        fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 8.2 if n == 1 else 7.7), squeeze=False)
        axes = axes[0]
        lg = fig.legend(handles=handles, loc="lower center", ncol=4 if n == 1 else 5, fontsize=st["fs_legend"],
                        frameon=False, handler_map={ArrowHandle: arrow_handler()})
        fig.subplots_adjust(left=0.02, right=0.98, top=0.93 if n == 1 else 0.925, bottom=0.15 if n == 1 else 0.10,
                            wspace=0.03)
    for ax, run in zip(axes, runs):
        draw_panel(ax, fr, run, known[run], preds[run], all_modes, st)
    fig.canvas.draw()
    bb = lg.get_window_extent().transformed(fig.transFigure.inverted())
    assert bb.x0 >= 0 and bb.x1 <= 1 and bb.y0 >= 0 and bb.y1 <= axes[0].get_position().y0 + 1e-3, \
        f"legend does not fit below the panels: {bb}"
    if st["print"]:
        from matplotlib.text import Text
        small = {t.get_text(): t.get_fontsize() for t in fig.findobj(Text)
                 if t.get_visible() and t.get_text() and t.get_fontsize() < 8}
        assert not small, f"text below 8 pt: {small}"
        fig.savefig(path, facecolor="white")
    else:
        fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- render
def cmd_render(args):
    import matplotlib
    matplotlib.use("Agg")
    from env_dataset import load_env_cache, known_mask, ENV_CACHE_DIR
    from env_map import LanePolylines
    from v2x_message_stream import V2XMetadataLoader, quaternion_to_yaw
    s = args.scene
    if s not in TUNING:
        sys.exit(f"scene {s} is not a tuning scene of split v2 {sorted(TUNING)}; refusing")
    town = TUNING[s]
    if args.no_pred:
        assert not args.runs, "--no-pred draws no model: leave out --runs"
        runs = []
    else:
        assert args.runs, "--runs is required (or --no-pred)"
        runs = args.runs.split(",")
        assert 1 <= len(runs) <= 2, "--runs: one run or two (side by side)"
    panels = runs or [None]
    st = PRINT if args.print else VIDEO
    matplotlib.rcParams["pdf.fonttype"] = 42     # TrueType text in the PDFs
    cfgs = {r: run_cfg(r) for r in runs}
    cache = load_env_cache()
    fut_ego_ref = np.load(os.path.join(ENV_CACHE_DIR, "env_targets.npz"))["fut_ego"]
    agent = args.agent if args.agent is not None else ego_rule(cache, s)[0]
    rows = scene_rows(cache, s)
    rows = rows[cache["sample_agent"][rows] == agent]
    rows = rows[np.argsort(cache["sample_frame"][rows])]
    if args.start_frame is not None:
        rows = rows[cache["sample_frame"][rows] >= args.start_frame]
    if args.frames:
        rows = rows[:args.frames]
    assert len(rows), f"no env samples for scene {s}, ego {agent}"
    print(f"{town} scene {s}, ego CAV {agent}: {len(rows)} frames "
          f"({cache['sample_frame'][rows[0]]}..{cache['sample_frame'][rows[-1]]}), runs {runs}")
    preds = {r: load_preds(r, s, agent, args.out, cache) for r in runs}
    preds[None] = {}

    L = V2XMetadataLoader()
    lanes = LanePolylines(town, "mirror")
    seg_w, seg_park = lane_segments(lanes)
    seg_mid = seg_w.mean(axis=1)

    def pose(a, f):
        p = L.ego_pose[L.find_sample_data(a, s, f)["ego_pose_token"]]
        return np.array(p["translation"][:2], dtype=np.float64), quaternion_to_yaw(p["rotation"])

    name = f"scene{s}_{town}_ego{agent}_" + ("_vs_".join(runs) if runs else "nopred")
    out_dir = os.path.join(args.out, "thesis_figures" if args.print else name)
    os.makedirs(os.path.join(out_dir, "" if args.print else "frames"), exist_ok=True)
    if args.stills == "auto":  # the frame with the most moving (b) targets (ties: earliest)
        nb = [int(((cache["vis"][cache["sample_ptr"][r]:cache["sample_ptr"][r + 1]] == 1)
                   & (cache["speed"][cache["sample_ptr"][r]:cache["sample_ptr"][r + 1]] > 0)).sum()) for r in rows]
        stills = {int(cache["sample_frame"][rows[int(np.argmax(nb))]])}
        print(f"still (auto): frame {min(stills)}, {max(nb)} moving (b) targets")
    else:
        stills = {int(f) for f in args.stills.split(",")} if args.stills else set()
    am_stills = {int(f) for f in args.all_modes_stills.split(",")} if args.all_modes_stills else set()
    if args.print:   # only the requested frames, as PDFs
        rows = rows[np.isin(cache["sample_frame"][rows], sorted(stills | am_stills))]
        missing = (stills | am_stills) - set(cache["sample_frame"][rows].tolist())
        assert not missing and len(rows), f"--print: frames {sorted(missing)} have no env sample for ego {agent}"
    max_geo, lane_d = 0.0, []
    for n, r in enumerate(rows):
        f = int(cache["sample_frame"][r])
        exy, eyaw = pose(agent, f)
        lo, hi = int(cache["sample_ptr"][r]), int(cache["sample_ptr"][r + 1])
        tid = np.arange(lo, hi)
        cur, yaw = cache["cur_ego"][tid].astype(np.float64), cache["yaw_ego"][tid].astype(np.float64)
        fut = np.stack([own_to_ego(cache["fut_own"][t].astype(np.float64), cur[j], yaw[j]) for j, t in enumerate(tid)]) \
            if len(tid) else np.zeros((0, 30, 2))
        if len(tid):
            max_geo = max(max_geo, float(np.abs(fut - fut_ego_ref[tid]).max()))
        near = np.hypot(*(seg_mid - exy).T) <= LANE_RANGE_M
        seg = scr(to_frame(seg_w[near], exy, eyaw))
        lane_pts = to_frame(seg_w[near][:, 0], exy, eyaw)
        mov = cache["speed"][tid] > 0
        for c in cur[mov]:
            lane_d.append(np.hypot(*(lane_pts - c).T).min() if len(lane_pts) else np.inf)
        cavs = {}
        for a in COOP_CAVS:
            if a == agent:
                continue
            try:
                xy, y = pose(a, f)
            except ValueError:
                continue
            cavs[a] = (to_frame(xy, exy, eyaw), y - eyaw)
        try:
            rsu = to_frame(pose(RSU_AGENT, f)[0], exy, eyaw)
        except ValueError:
            rsu = None
        fr = {"town": town, "scene": s, "agent": agent, "frame": f, "tid": tid, "cur": cur, "yaw": yaw, "fut_ego": fut,
              "wl": cache["size"][tid, :2].astype(np.float64), "lanes": seg, "lanes_park": seg_park[near],
              "cls": vis_class(cache["vis"][tid], cache["rsu_vis"][tid]), "cavs": cavs, "rsu": rsu, "moving": mov}
        known = {run: known_mask(cache["vis"][tid], cfgs[run]["v2x"],
                                 cache["rsu_vis"][tid] if cfgs[run].get("use_rsu", False) else None) for run in runs}
        known[None] = np.ones(len(tid), bool)
        for run in runs:
            miss = [t for t in tid[known[run]] if int(t) not in preds[run]]
            assert not miss, f"{run}: known targets without a prediction in frame {f}: {miss[:5]}"
        if args.print:
            jobs = {args.all_modes} if f in stills else set()
            if f in am_stills:
                jobs.add(True)
            for am in sorted(jobs):
                path = os.path.join(out_dir, f"bev_{name}_frame{f:03d}{'_allmodes' if am else ''}.pdf")
                render_frame(fr, panels, known, preds, path, None, am, st)
                print(f"  wrote {path}")
            continue
        render_frame(fr, panels, known, preds, os.path.join(out_dir, "frames", f"frame_{n:04d}.png"), args.dpi,
                     args.all_modes)
        if f in stills | am_stills:
            os.makedirs(os.path.join(out_dir, "stills"), exist_ok=True)
        if f in stills:
            render_frame(fr, panels, known, preds, os.path.join(out_dir, "stills", f"{name}_frame{f:03d}.png"), 300,
                         args.all_modes)
        if f in am_stills:
            render_frame(fr, panels, known, preds, os.path.join(out_dir, "stills", f"{name}_frame{f:03d}_allmodes.png"),
                         300, True)
    print(f"checks: max |own->ego future - cache fut_ego| = {max_geo:.2e} m; moving targets' nearest lane point "
          f"median {np.median(lane_d):.2f} m, p95 {np.percentile(lane_d, 95):.2f} m (n={len(lane_d)})")
    assert max_geo <= 1e-3, "own -> ego frame conversion disagrees with the cache's fut_ego"
    if args.print:
        return
    print(f"wrote {len(rows)} frames to {out_dir}/frames" + (f", stills {sorted(stills)}" if stills else "")
          + (f", all-modes stills {sorted(am_stills)}" if am_stills else ""))
    if not args.no_video:
        mp4 = os.path.join(out_dir, f"{name}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", os.path.join(out_dir, "frames", "frame_%04d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-r", "25", mp4], check=True)
        print(f"wrote {mp4} ({len(rows)} frames at {FPS} fps = {len(rows) / FPS:.1f} s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--gain", choices=sorted(GAINS), default="b_cv")
    p = sub.add_parser("render")
    p.add_argument("--scene", type=int, required=True)
    p.add_argument("--agent", type=int, default=None, help="ego CAV (default: the select ego rule)")
    p.add_argument("--runs", default="", help="RUN or RUN_A,RUN_B (side by side), e.g. E_V2X_seed0,E_EgoOnly_seed0")
    p.add_argument("--no-pred", action="store_true", help="one panel without a model: vehicles, visibility colours, "
                                                          "ground truth, lanes, CAVs and RSU; no predictions")
    p.add_argument("--print", action="store_true", help="thesis figures: only the --stills / --all-modes-stills frames, "
                                                        "as vector PDFs in <out>/thesis_figures/, 16 cm wide side by "
                                                        "side (8 cm single), text >= 8 pt, model name as panel label")
    p.add_argument("--frames", type=int, default=0, help="first N frames only (0 = all)")
    p.add_argument("--start-frame", type=int, default=None)
    p.add_argument("--stills", default="", help="comma-separated scene frame indices to save at 300 dpi, or 'auto' (most moving (b) targets)")
    p.add_argument("--all-modes", action="store_true", help="draw the other 5 modes for every predicted target, not "
                                                             "only the moving ones")
    p.add_argument("--all-modes-stills", default="", help="comma-separated scene frame indices to also save at 300 dpi "
                                                          "with all modes (*_allmodes.png, for the appendix)")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--out", default="viz_env")
    args = ap.parse_args()
    running = busy()
    if running:
        sys.exit("refusing: a run / training / evaluation process is running:\n  " + "\n  ".join(running))
    (cmd_select if args.cmd == "select" else cmd_render)(args)


if __name__ == "__main__":
    main()
