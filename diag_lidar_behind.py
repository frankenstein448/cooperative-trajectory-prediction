"""
diag_lidar_behind.py
Read-only diagnostic: why a CAV 7 m behind the ego can have class "nobody" in the env cache (scene 30, frame 19,
ego CAV 3, target CAV 2). Draws the ego's LIDAR_TOP and SEMLIDAR_TOP (model input) points top-down in the ego frame
(+-25 m, heading up) with the annotation boxes, CAV 2's in red, and prints:
  - the LiDAR mount (calibrated_sensor translation / rotation); the vertical field of view is not in the metadata,
    so the elevation range of the returned points is printed instead (sensor frame);
  - a frame check: points counted in every 3D annotation box vs the dataset's num_lidar_pts[ego];
  - per 15-degree azimuth bin (vehicle frame, 0 = ahead, +-180 = behind; points on the ego's own body excluded):
    point count, lowest returning beam elevation, nearest range, ground points. A crop gives an empty sector with
    straight edges; an occlusion by the ego's own body gives a lowest-beam elevation rising smoothly towards the rear.
Result (2026-10-15): 250/250 LIDAR_TOP box counts reproduce num_lidar_pts[3]; the lowest returning beam rises from
-30 deg at the side to -0.3 deg straight behind: a widening shadow of the ego's body, not a crop.

--print: the thesis figure (print_figure) from one sample's records only (print_main): the LIDAR_TOP .pcd.bin, the
ego pose and calibrated sensor of that sample_data record (found by an mmap search of the one-record-per-line JSON,
never json.load of the 0.2-2.5 GB files), the boxes of every kept target from the env cache (CAV 2 = is_cav 2). The
full-metadata path (default) loads ~10 GB; a --print run of it once hung at 0% CPU for ~50 min, cause not recovered
(no plt.show / subprocess / stdin / lock on that path, no cgroup memory pressure). --print arms a faulthandler
watchdog that dumps every thread's stack each WATCHDOG_S seconds, and prints the time per step.

Run in gpu-test-venv from the data checkout:
  nice -n 19 python diag_lidar_behind.py --out diag_lidar_scene30_frame19_ego3.png
  nice -n 19 python diag_lidar_behind.py --print --out viz_env/thesis_figures/lidar_blindspot.pdf
"""
import os
os.environ["MPLBACKEND"] = "Agg"
import sys
import json
import math
import mmap
import time
import argparse
import faulthandler

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from v2x_message_stream import V2X_ROOT
from pilot_dataset import SWEEPS_ROOT

SCENE, FRAME, EGO, CAV = 30, 19, 3, 2
SELF_BOX = (2.6, 1.1)     # |x|, |y| (m, vehicle frame): points on the ego's own body
GROUND_Z = -1.3           # sensor-frame z below this = ground (sensor 1.547 m above the pose origin)
PRINT_W_CM, PRINT_VIEW_M, PRINT_SCALE_BAR_M = 16.0, 25.0, 10.0   # --print: width, +- view, scale bar
WATCHDOG_S = 60


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def json_record(path, key, value):
    """The one record of a nuScenes-style JSON list (one field per line, flat records) whose field `key` is `value`,
    by an mmap search: the file is never parsed as a whole."""
    needle = f'"{key}": "{value}"'.encode()
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        i = mm.find(needle)
        assert i >= 0, f"{path}: no record with {key} = {value}"
        assert mm.find(needle, i + 1) < 0, f"{path}: several records with {key} = {value}"
        lo, hi = mm.rfind(b"{", 0, i), mm.find(b"}", i)
        return json.loads(mm[lo:hi + 1])


def step(t0, what):
    print(f"  [{time.perf_counter() - t0:6.2f} s] {what}", flush=True)


def corners(cur, yaw, w, l):
    """Box footprint in screen coordinates (right, ahead) from ego-frame centre / yaw (render_env_bev.box)."""
    c, sn = math.cos(yaw), math.sin(yaw)
    p = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    e = np.stack([p[:, 0] * c - p[:, 1] * sn + cur[0], p[:, 0] * sn + p[:, 1] * c + cur[1]], 1)
    return np.stack([-e[:, 1], e[:, 0]], 1)


def print_main(out):
    from env_dataset import ENV_CACHE_DIR
    t0 = time.perf_counter()
    faulthandler.dump_traceback_later(WATCHDOG_S, repeat=True, file=sys.stderr)
    print(f"--print: scene {SCENE}, frame {FRAME}, ego CAV {EGO}, LIDAR_TOP; watchdog {WATCHDOG_S} s", flush=True)
    sdp = f"{V2X_ROOT}/sample_data.json"
    fn = lambda a: f"sweeps/LIDAR_TOP_id_{a}/scene_{SCENE}_{FRAME:06d}.pcd.bin"
    r = json_record(sdp, "filename", fn(EGO))
    r2 = json_record(sdp, "filename", fn(CAV))
    step(t0, f"sample_data records of {fn(EGO)} and {fn(CAV)} (mmap search)")
    e = json_record(f"{V2X_ROOT}/ego_pose.json", "token", r["ego_pose_token"])
    e2 = json_record(f"{V2X_ROOT}/ego_pose.json", "token", r2["ego_pose_token"])
    c = json_record(f"{V2X_ROOT}/calibrated_sensor.json", "token", r["calibrated_sensor_token"])
    step(t0, "ego poses (CAV 3, CAV 2) and calibrated sensor")
    p = np.fromfile(f"{SWEEPS_ROOT}/{r['filename'].split('/', 1)[1]}", dtype=np.float32).reshape(-1, 5).astype(np.float64)
    w = (qmat(e["rotation"]) @ (qmat(c["rotation"]) @ p[:, :3].T + np.array(c["translation"])[:, None])).T \
        + e["translation"]
    yaw = math.atan2(*qmat(e["rotation"])[[1, 0], 0])

    def to_ego(xy):
        cs, sn = math.cos(-yaw), math.sin(-yaw)
        d = xy[:, :2] - np.array(e["translation"][:2])
        return np.stack([d[:, 0] * cs - d[:, 1] * sn, d[:, 0] * sn + d[:, 1] * cs], 1)

    ego, z = to_ego(w), w[:, 2] - e["translation"][2]
    step(t0, f"{len(p)} LIDAR_TOP points to the ego frame")
    keys = ("sample_scene", "sample_agent", "sample_frame", "sample_ptr", "is_cav", "category", "vis", "nlp",
            "xy_world", "cur_ego", "yaw_ego", "size")
    with np.load(os.path.join(ENV_CACHE_DIR, "env_targets.npz")) as zf:
        z_ = {k: zf[k] for k in keys}
    row = np.flatnonzero((z_["sample_scene"] == SCENE) & (z_["sample_agent"] == EGO) & (z_["sample_frame"] == FRAME))
    assert len(row) == 1, f"env cache: {len(row)} samples for scene {SCENE}, ego {EGO}, frame {FRAME}"
    t = np.arange(z_["sample_ptr"][row[0]], z_["sample_ptr"][row[0] + 1])
    T = {k: z_[k][t] for k in keys[4:]}
    step(t0, f"env cache: {len(t)} kept targets of the sample")
    geo = float(np.abs(to_ego(T["xy_world"]) - T["cur_ego"]).max())
    assert geo < 1e-3, f"ego frame of the pose disagrees with the cache's cur_ego by {geo:.4f} m"
    k = np.flatnonzero(T["is_cav"] == CAV)
    assert len(k) == 1, f"env cache: {len(k)} targets with is_cav = {CAV}"
    k = int(k[0])
    d2 = float(np.hypot(*(to_ego(np.array([e2["translation"][:2]]))[0] - T["cur_ego"][k])))
    assert d2 < 1.0, f"CAV {CAV} pose is {d2:.2f} m from its cache target"
    sq = corners(T["cur_ego"][k], float(T["yaw_ego"][k]), float(T["size"][k][0]), float(T["size"][k][1]))
    from matplotlib.path import Path
    in_fp = int(Path(sq).contains_points(np.stack([-ego[:, 1], ego[:, 0]], 1)).sum())
    print(f"  checks: pose vs cache cur_ego max {geo:.1e} m; CAV {CAV} pose vs its target {d2:.2f} m; CAV {CAV} at "
          f"{np.round(T['cur_ego'][k], 2).tolist()} m, class {int(T['vis'][k])}, num_lidar_pts {T['nlp'][k].tolist()}, "
          f"LIDAR_TOP points in its footprint {in_fp}", flush=True)
    print_figure(out, ego, z, T, k)
    step(t0, f"wrote {out}")
    faulthandler.cancel_dump_traceback_later()


def print_figure(out, ego, z, T, k):
    """Thesis PDF in render_env_bev.py --print style: LIDAR_TOP top-down in the ego frame (+-PRINT_VIEW_M, heading
    up), PRINT_W_CM wide, every text >= 8 pt, the points rasterized (300 dpi) inside the vector PDF, no ticks but a
    scale bar, no title (it goes in the LaTeX caption). Ego box filled black (render_env_bev.EGO_SIZE, the ego is not
    in the env cache), CAV k's box and label in the CAV colour, the other kept targets thin grey."""
    from matplotlib.text import Text
    import matplotlib.patheffects as pe
    from render_env_bev import CAV_COLOR, EGO_COLOR, EGO_SIZE, MIN_BOX_W
    matplotlib.rcParams["pdf.fonttype"] = 42
    W = PRINT_W_CM / 2.54
    margin, cbar_w, cbar_gap, cbar_lab = 0.06, 0.12, 0.08, 0.62     # in
    panel = W - 2 * margin - cbar_gap - cbar_w - cbar_lab
    H = panel + 2 * margin
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([margin / W, margin / H, panel / W, panel / H])
    cax = fig.add_axes([(margin + panel + cbar_gap) / W, (margin + 0.15 * panel) / H, cbar_w / W, 0.7 * panel / H])
    m = (np.abs(ego[:, 0]) <= PRINT_VIEW_M) & (np.abs(ego[:, 1]) <= PRINT_VIEW_M)
    sc = ax.scatter(-ego[m, 1], ego[m, 0], s=0.25, lw=0, c=z[m], cmap="viridis", vmin=-2, vmax=3, rasterized=True,
                    zorder=1)
    white = [pe.withStroke(linewidth=2.5, foreground="white")]
    for j in range(len(T["cur_ego"])):
        sq = corners(T["cur_ego"][j], float(T["yaw_ego"][j]), max(float(T["size"][j][0]), MIN_BOX_W),
                     float(T["size"][j][1]))
        if np.abs(sq).min(axis=0).max() > PRINT_VIEW_M:
            continue
        if j == k:
            ax.add_patch(Polygon(sq, closed=True, fill=False, ec=CAV_COLOR, lw=1.6, zorder=4))
            ax.text(sq[:, 0].min() - 0.6, sq[:, 1].mean(), f"CAV {CAV}", ha="right", va="center", fontsize=8,
                    fontweight="bold", color=CAV_COLOR, path_effects=white, zorder=5)
        else:
            ax.add_patch(Polygon(sq, closed=True, fill=False, ec="#555555", lw=0.6, zorder=3))
    sq = corners((0.0, 0.0), 0.0, EGO_SIZE[0], EGO_SIZE[1])
    ax.add_patch(Polygon(sq, closed=True, fc=EGO_COLOR, ec=EGO_COLOR, lw=0.8, zorder=4))
    ax.text(sq[:, 0].max() + 0.6, 0.0, f"ego (CAV {EGO})", ha="left", va="center", fontsize=8, fontweight="bold",
            color=EGO_COLOR, path_effects=white, zorder=5)
    x0, y0 = PRINT_VIEW_M - 2 - PRINT_SCALE_BAR_M, PRINT_VIEW_M - 3.5
    ax.plot([x0, x0 + PRINT_SCALE_BAR_M], [y0, y0], color="black", lw=1.5, solid_capstyle="butt", zorder=6,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.text(x0 + PRINT_SCALE_BAR_M / 2, y0 + 0.7, f"{PRINT_SCALE_BAR_M:g} m", ha="center", va="bottom", fontsize=8,
            path_effects=white, zorder=6)
    ax.set_xlim(-PRINT_VIEW_M, PRINT_VIEW_M)
    ax.set_ylim(-PRINT_VIEW_M, PRINT_VIEW_M)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("height relative to the ego pose (m)", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    cb.solids.set_rasterized(True)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    small = {t.get_text(): t.get_fontsize() for t in texts if t.get_fontsize() < 8}
    assert not small, f"text below 8 pt: {small}"
    for t in texts:
        bb = t.get_window_extent(r).transformed(fig.transFigure.inverted())
        assert bb.x0 >= 0 and bb.x1 <= 1 and bb.y0 >= 0 and bb.y1 <= 1, f"text outside the figure: {t.get_text()}"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diag_lidar_scene30_frame19_ego3.png")
    ap.add_argument("--print", action="store_true", help="thesis PDF from this sample's records only (print_main)")
    args = ap.parse_args()
    if args.print:
        return print_main(args.out)
    load = lambda n: json.load(open(f"{V2X_ROOT}/{n}"))
    sd = load("sample_data.json")
    poses = {p["token"]: p for p in load("ego_pose.json")}
    cal = {c["token"]: c for c in load("calibrated_sensor.json")}
    anns = load("sample_annotation.json")

    def rec(ch):
        fn = f"sweeps/{ch}_id_{EGO}/scene_{SCENE}_{FRAME:06d}.pcd.bin"
        return next(r for r in sd if r["filename"] == fn)

    frame_anns = [a for a in anns if a["sample_token"] == rec("LIDAR_TOP")["sample_token"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    for ax, ch in zip(axes, ("LIDAR_TOP", "SEMLIDAR_TOP")):
        r = rec(ch)
        p = np.fromfile(f"{SWEEPS_ROOT}/{r['filename'].split('/', 1)[1]}", dtype=np.float32).reshape(-1, 5).astype(np.float64)
        c, e = cal[r["calibrated_sensor_token"]], poses[r["ego_pose_token"]]
        w = (qmat(e["rotation"]) @ (qmat(c["rotation"]) @ p[:, :3].T + np.array(c["translation"])[:, None])).T \
            + e["translation"]
        yaw = math.atan2(*qmat(e["rotation"])[[1, 0], 0])

        def to_ego(xyz):
            cs, sn = math.cos(-yaw), math.sin(-yaw)
            d = xyz[:, :2] - np.array(e["translation"][:2])
            return np.stack([d[:, 0] * cs - d[:, 1] * sn, d[:, 0] * sn + d[:, 1] * cs], 1)

        def in_box(a):
            loc = (w - np.array(a["translation"])) @ qmat(a["rotation"])
            wd, ln, ht = a["size"]
            return (np.abs(loc[:, 0]) <= ln / 2) & (np.abs(loc[:, 1]) <= wd / 2) & (np.abs(loc[:, 2]) <= ht / 2)

        rng = np.hypot(p[:, 0], p[:, 1])
        elev = np.degrees(np.arctan2(p[:, 2], rng))
        print(f"\n{ch}: {r['filename']}, {len(p)} points")
        print(f"  mount (calibrated_sensor): translation {np.round(c['translation'], 3).tolist()} m, "
              f"rotation {np.round(c['rotation'], 4).tolist()}; ego_pose z {e['translation'][2]:.2f}")
        print(f"  vertical FOV: not in the metadata; returned points span elevation {elev.min():.2f} .. "
              f"{elev.max():.2f} deg in {len(np.unique(np.round(elev, 1)))} beams, max range "
              f"{np.hypot(rng, p[:, 2]).max():.1f} m")
        ok = n = 0
        for a in frame_anns:
            nl = a.get("num_lidar_pts") or []
            if len(nl) > EGO and a["size"][0] > 0:
                n += 1
                ok += int(in_box(a).sum()) == nl[EGO]
        print(f"  frame check: points in 3D box == num_lidar_pts[{EGO}] for {ok}/{n} annotations (width > 0)")
        x, y = p[:, 0] + c["translation"][0], p[:, 1] + c["translation"][1]       # vehicle frame
        az = np.degrees(np.arctan2(y, x))
        body = (np.abs(x) < SELF_BOX[0]) & (np.abs(y) < SELF_BOX[1])
        print(f"  points on the ego's own body: {body.sum()}")
        print(f"  {'azimuth':>12s} {'n':>6s} {'lowest beam':>11s} {'nearest m':>9s} {'ground':>6s}")
        for lo in range(-180, 180, 15):
            m = ~body & (az >= lo) & (az < lo + 15)
            print(f"  {lo:+5d}..{lo + 15:+4d} {m.sum():6d} {elev[m].min():10.1f}d {rng[m].min():9.1f} "
                  f"{(m & (p[:, 2] < GROUND_Z)).sum():6d}")

        ego = to_ego(w)
        m = (np.abs(ego[:, 0]) <= 25) & (np.abs(ego[:, 1]) <= 25)
        sc = ax.scatter(-ego[m, 1], ego[m, 0], s=0.3, c=w[m, 2] - e["translation"][2], cmap="viridis", vmin=-2, vmax=3)
        cr2 = next(q for q in sd if q["filename"] == f"sweeps/{ch}_id_{CAV}/scene_{SCENE}_{FRAME:06d}.pcd.bin")
        cav_xy = np.array(poses[cr2["ego_pose_token"]]["translation"][:2])
        for a in frame_anns:
            wd, ln, _ = a["size"]
            cr = np.array([[ln / 2, wd / 2, 0], [ln / 2, -wd / 2, 0], [-ln / 2, -wd / 2, 0], [-ln / 2, wd / 2, 0]]) \
                @ qmat(a["rotation"]).T + a["translation"]
            ce = to_ego(cr)
            if np.abs(ce).max() > 30:
                continue
            is_cav = math.hypot(*(np.array(a["translation"][:2]) - cav_xy)) < 1.0
            ax.add_patch(Polygon(np.stack([-ce[:, 1], ce[:, 0]], 1), fill=False, ec="red" if is_cav else "#555555",
                                 lw=2 if is_cav else 0.7))
            if is_cav:
                nl = a.get("num_lidar_pts") or []
                ax.annotate(f"CAV {CAV}: num_lidar_pts[{EGO}] = {nl[EGO]}, in box {int(in_box(a).sum())}\n"
                            f"num_lidar_pts (0 = RSU .. 5) = {nl}", (-ce[:, 1].max(), ce[:, 0].mean()),
                            xytext=(8, 0), textcoords="offset points", color="red", fontsize=8)
        ax.plot(0, 0, "k^", ms=8)
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.set_aspect("equal")
        ax.set_title(f"{ch}_id_{EGO}, scene {SCENE} frame {FRAME}\nego frame, heading up", fontsize=10)
        ax.set_xlabel("right (m)")
        ax.set_ylabel("ahead (m)")
        fig.colorbar(sc, ax=ax, shrink=0.7, label="z - ego pose z (m)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
