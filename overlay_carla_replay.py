"""
overlay_carla_replay.py
Stage 2 of the CARLA replay (visualisation only): draws the env predictions of export_carla_replay.py's per-frame .npz
into carla_replay.py's rendered image with its saved camera matrices (camera_frameNNN.json), two panels side by side
over the same render (e.g. E_V2X | E_EgoOnly), model label on top, caption below. Tuning split, seed 0, epoch 12.

Display rules as render_env_bev.py --print: per target its ground-truth 6 s future (dashed black, above the
predictions), for the panel model's known targets the top-1 mode (thick, end dot) and, with --other-modes, for moving
targets the other 5 modes (thin); targets some CAV / the RSU sees but the panel's model does not take as input are
faded (no
prediction, faded ground truth). CARLA paints the cars by visibility class; the overlay adds what the paint cannot
show: two-wheelers get their projected footprint filled in the class colour (hollow grey: seen by nobody), faded targets
a white veil over their projected box, the other CAVs and the RSU a label.

Projection: V2X-Sim world (x, y) -> CARLA (x, -y, ground z of the target from the camera json) -> UE camera (x fwd,
y right, z up) -> pixel u = fx * y / x + cx, v = -fy * z / x + cy. Points less than NEAR_M in front of the camera are
dropped. Check: the ego's projected centre and the camera json's K reproduce carla_replay.project (printed).

Video (--all): top-1 + ground truth only (caption says so); --other-modes adds the other 5 modes of moving targets.
Two-wheelers are drawn as their flat footprint on the ground.

Run in gpu-test-venv from the data checkout:
  python ~/hd-maps-dev/overlay_carla_replay.py --frame 35
  python ~/hd-maps-dev/overlay_carla_replay.py --all          # frames/, mp4 at 2.5 fps, 300 dpi still of frame 35
"""
import os
import glob
import json
import math
import argparse
import subprocess

import numpy as np

from render_env_bev import (CLASS_COLORS, GT_COLOR, GT_LW, FADE, EGO_COLOR, CAV_COLOR, RSU_COLOR, PANEL_LABELS,
                            MIN_BOX_W, FRAME_RATE_HZ)

CAPTION = "Replay of recorded V2X-Sim 2.0 data rendered in CARLA 0.9.15; tuning split"
CAPTION_TOP1 = "; top-1 prediction shown"
FPS = 2.5                      # render_env_bev.FPS
FOOT_UP_M = 0.05               # two-wheeler footprints this far above the ground
NEAR_M = 1.0
K_LW = 1.3                     # line-width scale vs render_env_bev's VIDEO style (panels 8 in wide)
PANEL_W_IN, GAP_IN, MARGIN_IN, TOP_IN, BOTTOM_IN = 8.0, 0.12, 0.08, 0.34, 0.62
VEIL_ALPHA = 0.6               # white over a faded car: its paint shows at about FADE strength
LABEL_UP_M = 2.6               # CAV / RSU labels this far above the ground point


class Camera:
    def __init__(self, cam):
        self.K, self.w2c = np.array(cam["K"]), np.linalg.inv(np.array(cam["cam2world"]))
        self.W, self.H = cam["width"], cam["height"]

    def project(self, xyz):
        """CARLA points [..., 3] -> (pixels [..., 2], depth [...])."""
        p = np.einsum("ij,...j->...i", self.w2c, np.concatenate([xyz, np.ones(xyz.shape[:-1] + (1,))], -1))[..., :3]
        x = np.where(np.abs(p[..., 0]) < 1e-9, 1e-9, p[..., 0])
        uv = np.stack([self.K[0, 0] * p[..., 1] / x + self.K[0, 2], -self.K[1, 1] * p[..., 2] / x + self.K[1, 2]], -1)
        return uv, p[..., 0]


def carla_xyz(xy, z):
    """V2X-Sim world xy [..., 2] -> CARLA [..., 3] (the mirror) at height z."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([xy[..., 0], -xy[..., 1], np.broadcast_to(z, xy.shape[:-1])], -1)


def box_corners(xy, yaw, w, l, h, z):
    """8 corners [8, 3] (CARLA) of a V2X-Sim box resting on the ground at z."""
    w = max(w, MIN_BOX_W)
    own = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    c, s = math.cos(yaw), math.sin(yaw)
    world = np.stack([own[:, 0] * c - own[:, 1] * s + xy[0], own[:, 0] * s + own[:, 1] * c + xy[1]], -1)
    return np.concatenate([carla_xyz(world, z), carla_xyz(world, z + h)])


def hull(cam, corners):
    from scipy.spatial import ConvexHull
    uv, d = cam.project(corners)
    if (d < NEAR_M).any():
        return None
    return uv[ConvexHull(uv).vertices]


def line(ax, cam, xyz, **kw):
    uv, d = cam.project(xyz)
    uv = uv[d >= NEAR_M]
    if len(uv) >= 2:
        ax.plot(uv[:, 0], uv[:, 1], **kw)
    return uv


def draw_panel(ax, img, cam, fr, gz, model, k, other_modes):
    from matplotlib.patches import Polygon
    ax.imshow(img, interpolation="antialiased")
    ax.set_xlim(0, cam.W)
    ax.set_ylim(cam.H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    known, modes, top1 = fr[f"{model}_known"], fr[f"{model}_modes_world"], fr[f"{model}_top1"]
    for j, t in enumerate(fr["tid"]):
        cls, xy, yaw = int(fr["cls"][j]), fr["xy_world"][j], float(fr["yaw_world"][j])
        z, col = gz[int(t)], CLASS_COLORS[int(fr["cls"][j])]
        faded = not known[j] and cls != 3
        line(ax, cam, carla_xyz(np.vstack([xy, fr["gt_world"][j]]), z), color=GT_COLOR, lw=GT_LW * k,
             ls=(0, (3, 2)), alpha=FADE if faded else 0.8, zorder=4.8)
        if known[j]:
            for m in range(modes.shape[1]):
                pts = carla_xyz(np.vstack([xy, modes[j, m]]), z)
                if m == top1[j]:
                    uv = line(ax, cam, pts, color=col, lw=2.4 * k, zorder=4.5, solid_capstyle="round")
                    if len(uv):
                        ax.plot(uv[-1, 0], uv[-1, 1], "o", ms=4.5 * k, color=col, mec="black", mew=0.5 * k, zorder=4.6)
                elif other_modes and fr["moving"][j]:
                    line(ax, cam, pts, color=col, lw=0.9 * k, alpha=0.55, zorder=4)
        w, l, h = fr["size"][j]
        if fr["two_wheeler"][j]:   # flat footprint on the ground (the bike blueprint cannot be recoloured)
            uv, d = cam.project(box_corners(xy, yaw, w, l, 0.0, z + FOOT_UP_M)[:4])
            poly = uv if (d >= NEAR_M).all() else None
        else:
            poly = hull(cam, box_corners(xy, yaw, w, l, h, z))
        if poly is None:
            continue
        if fr["two_wheeler"][j]:
            if cls == 3:
                ax.add_patch(Polygon(poly, closed=True, zorder=5, facecolor="none", edgecolor=col, lw=1.4 * k))
            else:
                ax.add_patch(Polygon(poly, closed=True, zorder=5, facecolor=col, edgecolor="black", lw=0.4 * k,
                                     alpha=FADE if faded else 0.85))
        elif faded:
            ax.add_patch(Polygon(poly, closed=True, zorder=5, facecolor="white", edgecolor="none", alpha=VEIL_ALPHA))
    for a, pose in zip(fr["cav_ids"], fr["cav_poses"]):
        if np.isnan(pose[0]):
            continue
        j = int(np.argmin(np.hypot(*(fr["xy_world"] - pose[:2]).T)))   # the CAV is a target (export check)
        label(ax, cam, carla_xyz(pose[:2], gz[int(fr["tid"][j])] + LABEL_UP_M), f"CAV {a}", CAV_COLOR, k)
    if not np.isnan(fr["rsu_pose"][0]):
        p = carla_xyz(fr["rsu_pose"][:2], gz[-1])
        uv, d = cam.project(p)
        if d >= NEAR_M and 0 <= uv[0] < cam.W and 0 <= uv[1] < cam.H:
            ax.plot(*uv, marker="^", ms=12 * k, mfc=RSU_COLOR, mec="white", mew=1.0 * k, zorder=6)
            label(ax, cam, p + np.array([0, 0, LABEL_UP_M]), "RSU", RSU_COLOR, k)
    label(ax, cam, carla_xyz(fr["ego_pose"][:2], gz[-1] + LABEL_UP_M), "ego", EGO_COLOR, k)


def label(ax, cam, xyz, text, color, k):
    uv, d = cam.project(xyz)
    if d >= NEAR_M and 0 <= uv[0] < cam.W and 0 <= uv[1] < cam.H:
        ax.text(uv[0], uv[1], text, ha="center", va="bottom", fontsize=9, color=color, fontweight="bold", zorder=7,
                clip_on=True, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9))


def legend_handles(k, other_modes):
    """render_env_bev.legend_handles (print names) for the 3D view: cars are painted (no lane / heading entries)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    names = ("seen by the ego", "coop CAV only", "RSU only", "seen by nobody")
    h = [Patch(facecolor=c, edgecolor="black", lw=0.4 * k, label=n) for c, n in zip(CLASS_COLORS, names)]
    h += [Patch(facecolor=CLASS_COLORS[1], edgecolor="black", lw=0.4 * k, alpha=FADE, label="faded: not a model input"),
          Line2D([], [], color=GT_COLOR, lw=GT_LW * k, ls=(0, (3, 2)), label="ground truth, 6 s"),
          Line2D([], [], color="#555555", lw=2.4 * k, marker="o", ms=4.5 * k, markevery=[1],
                 label="prediction, top-1 mode"),
          Patch(facecolor=EGO_COLOR, label="ego vehicle"),
          Line2D([], [], ls="", marker="^", ms=9 * k, mfc=RSU_COLOR, mec=RSU_COLOR, label="RSU")]
    if other_modes:
        h.insert(8, Line2D([], [], color="#555555", lw=0.9 * k, alpha=0.55, label="other 5 modes (moving)"))
    return h


def render(fr, meta, img, cam_json, models, path, dpi, other_modes):
    import matplotlib.pyplot as plt
    cam = Camera(cam_json)
    gz = {int(t): v for t, v in cam_json["ground_z"].items()}
    k = K_LW
    n = len(models)
    panel_h = PANEL_W_IN * cam.H / cam.W
    W = 2 * MARGIN_IN + n * PANEL_W_IN + (n - 1) * GAP_IN
    H = TOP_IN + panel_h + BOTTOM_IN
    fig = plt.figure(figsize=(W, H))
    for i, model in enumerate(models):
        ax = fig.add_axes([(MARGIN_IN + i * (PANEL_W_IN + GAP_IN)) / W, BOTTOM_IN / H, PANEL_W_IN / W, panel_h / H])
        draw_panel(ax, img, cam, fr, gz, model, k, other_modes)
        ax.set_title(PANEL_LABELS.get(model, model), fontsize=12, fontweight="bold", loc="left", pad=4)
        if i == n - 1:
            ax.set_title(f"{meta['town']} #{meta['scene']}, ego CAV {meta['agent']}, t = "
                         f"{meta['frame'] / FRAME_RATE_HZ:.1f} s (frame {meta['frame']})", fontsize=10, loc="right",
                         pad=4)
    fig.legend(handles=legend_handles(k, other_modes), loc="lower center", bbox_to_anchor=(0.5, 0.22 / H), ncol=10,
               fontsize=9,
               frameon=False, handlelength=1.6, columnspacing=1.0, handletextpad=0.5)
    fig.text(MARGIN_IN / W, 0.06 / H, CAPTION + ("" if other_modes else CAPTION_TOP1), fontsize=8.5, ha="left",
             va="bottom", color="#333333")
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def load(args, f):
    from PIL import Image
    z = np.load(os.path.join(args.data, f"frame_{f:03d}.npz"))
    fr, meta = {key: z[key] for key in z.files}, json.loads(str(z["meta"]))
    assert meta["split"] == "tuning", meta
    with open(os.path.join(args.render, f"camera_frame{f:03d}.json")) as fh:
        cam_json = json.load(fh)
    assert (cam_json["scene"], cam_json["agent"], cam_json["frame"]) == (meta["scene"], meta["agent"], meta["frame"])
    img = np.asarray(Image.open(os.path.join(args.render, f"frame{f:03d}.png")).convert("RGB"))
    return fr, meta, img, cam_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=None, help="one frame (default: --all)")
    ap.add_argument("--all", action="store_true", help="every exported frame -> frames/, an mp4 at FPS and --stills")
    ap.add_argument("--stills", default="35", help="--all: comma-separated frames also saved at 300 dpi")
    ap.add_argument("--models", default="E_V2X,E_EgoOnly", help="one or two models of the export (left | right)")
    ap.add_argument("--other-modes", action="store_true", help="also the other 5 modes of moving targets")
    ap.add_argument("--data", default="viz_env/carla_replay/scene16_ego4", help="export_carla_replay.py output")
    ap.add_argument("--render", default="viz_env/carla_replay/scene16_ego4_render", help="carla_replay.py output")
    ap.add_argument("--out", default="viz_env/carla_replay/scene16_ego4_overlay")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    assert args.all != (args.frame is not None), "--frame N or --all"
    import matplotlib
    matplotlib.use("Agg")
    models = args.models.split(",")
    if args.all:
        frames = sorted(int(os.path.basename(p)[6:9]) for p in glob.glob(os.path.join(args.data, "frame_*.npz")))
    else:
        frames = [args.frame]
    stills = {int(f) for f in args.stills.split(",")} if args.all and args.stills else set()
    os.makedirs(os.path.join(args.out, "frames") if args.all else args.out, exist_ok=True)
    for n, f in enumerate(frames):
        fr, meta, img, cam_json = load(args, f)
        assert all(m in meta["models"] for m in models), f"{models} not in the export {meta['models']}"
        cam = Camera(cam_json)
        uv, d = cam.project(carla_xyz(fr["ego_pose"][:2], cam_json["ground_z"]["-1"]))
        if not args.all or n % 10 == 0:
            print(f"frame {f}: ego centre projects to ({uv[0]:.0f}, {uv[1]:.0f}), depth {d:.1f} m; "
                  + "; ".join(f"{m} {int(fr[m + '_known'].sum())} known, "
                              f"{int((~fr[m + '_known'] & (fr['cls'] != 3)).sum())} faded" for m in models), flush=True)
        if args.all:
            render(fr, meta, img, cam_json, models, os.path.join(args.out, "frames", f"frame_{n:04d}.png"), args.dpi,
                   args.other_modes)
            if f in stills:
                path = os.path.join(args.out, f"overlay_frame{f:03d}_300dpi.png")
                render(fr, meta, img, cam_json, models, path, 300, args.other_modes)
                print(f"wrote {path}")
        else:
            path = os.path.join(args.out, f"overlay_frame{f:03d}.png")
            render(fr, meta, img, cam_json, models, path, args.dpi, args.other_modes)
            print(f"wrote {path}")
    if args.all:
        name = f"scene{meta['scene']}_{meta['town']}_ego{meta['agent']}_carla_" + "_vs_".join(models)
        mp4 = os.path.join(args.out, f"{name}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", os.path.join(args.out, "frames", "frame_%04d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-r", "25", mp4], check=True)
        print(f"wrote {mp4} ({len(frames)} frames at {FPS} fps = {len(frames) / FPS:.1f} s)")


if __name__ == "__main__":
    main()
