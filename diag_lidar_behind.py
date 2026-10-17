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

Run in gpu-test-venv from the data checkout:
  nice -n 19 python diag_lidar_behind.py --out diag_lidar_scene30_frame19_ego3.png
"""
import json
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from v2x_message_stream import V2X_ROOT
from pilot_dataset import SWEEPS_ROOT

SCENE, FRAME, EGO, CAV = 30, 19, 3, 2
SELF_BOX = (2.6, 1.1)     # |x|, |y| (m, vehicle frame): points on the ego's own body
GROUND_Z = -1.3           # sensor-frame z below this = ground (sensor 1.547 m above the pose origin)


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diag_lidar_scene30_frame19_ego3.png")
    args = ap.parse_args()
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
