import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import cKDTree
from pilot_dataset import PilotDataset
from v2x_message_stream import V2XMetadataLoader
from coord_transform import TOWN_TRANSFORMS

MIN_PATH_LEN = 10.0   # meters; skips stationary agents
MAX_POINTS = 2000
CAP = 10.0            # meters; caps the cost of any single point


def pose_xy(loader, agent_id, scene, frame):
    rec = loader.find_sample_data(agent_id, scene, frame)
    tr = loader.ego_pose[rec["ego_pose_token"]]["translation"]
    return np.array([tr[0], tr[1]])


def lane_points(topo, step=1.0):
    pts = []
    for e in topo:
        if e["relation"] != 0:
            continue
        a, b = np.array(e["start_pos"], float), np.array(e["end_pos"], float)
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.vstack(pts)


def rot(deg):
    th = np.radians(deg)
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def transform(raw, theta, tx, ty):
    return raw @ rot(theta).T + np.array([tx, ty])


def search(tree, p_cur, c, dths, dxs, dys):
    best = (1e9, 0.0, np.zeros(2))
    for dth in dths:
        rotated = (p_cur - c) @ rot(dth).T + c
        for dx in dxs:
            for dy in dys:
                d, _ = tree.query(rotated + np.array([dx, dy]))
                cst = np.minimum(d, CAP).mean()
                if cst < best[0]:
                    best = (cst, dth, np.array([dx, dy]))
    return best


def stats(tree, pts):
    d, _ = tree.query(pts)
    return np.median(d), (d < 2.0).mean() * 100


dataset = PilotDataset()
loader = V2XMetadataLoader()
towns = sys.argv[1:] or ["Town05", "Town04", "Town03"]

for town in towns:
    cur = TOWN_TRANSFORMS[town]
    topo = pickle.load(open(f"topology/{town}_topology.pkl", "rb"))
    tree = cKDTree(lane_points(topo))

    frames = defaultdict(set)
    for agent_id, scene, frame, tn in dataset.samples:
        if tn == town:
            frames[(agent_id, scene)].add(frame)
    paths = []
    for (agent_id, scene), fs in frames.items():
        pts = []
        for f in sorted(fs):
            try:
                pts.append(pose_xy(loader, agent_id, scene, f))
            except Exception:
                pass
        pts = np.array(pts)
        if len(pts) >= 2 and np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() >= MIN_PATH_LEN:
            paths.append(pts)
    raw_all = np.vstack(paths)
    rng = np.random.default_rng(0)
    raw = raw_all[rng.choice(len(raw_all), min(MAX_POINTS, len(raw_all)), replace=False)]
    print(f"\n{town}: {len(paths)} moving paths, {len(raw)} points used")

    p_cur = transform(raw, cur["theta"], cur["tx"], cur["ty"])
    c = p_cur.mean(0)
    med0, pct0 = stats(tree, p_cur)
    print(f"  current  theta={cur['theta']:.2f}: median dist {med0:.2f} m, within 2 m: {pct0:.0f}%")

    _, dth1, d1 = search(tree, p_cur, c, np.arange(-35, 35.01, 0.5),
                         np.arange(-30, 30.01, 3.0), np.arange(-30, 30.01, 3.0))
    _, dth2, d2 = search(tree, p_cur, c, np.arange(dth1 - 1, dth1 + 1.01, 0.1),
                         np.arange(d1[0] - 3, d1[0] + 3.01, 0.5),
                         np.arange(d1[1] - 3, d1[1] + 3.01, 0.5))

    theta_new = cur["theta"] + dth2
    t_new = rot(dth2) @ (np.array([cur["tx"], cur["ty"]]) - c) + c + d2
    p_new = transform(raw, theta_new, t_new[0], t_new[1])
    med1, pct1 = stats(tree, p_new)
    print(f"  refit    theta={theta_new:.2f}: median dist {med1:.2f} m, within 2 m: {pct1:.0f}%")
    print(f'  suggested: "theta": {theta_new:.2f}, "tx": {t_new[0]:.2f}, "ty": {t_new[1]:.2f}')

    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    allp = transform(raw_all, theta_new, t_new[0], t_new[1])
    lo, hi = allp.min(0) - 40, allp.max(0) + 40
    for ax, (name, th, tx, ty) in zip(axes, [("current", cur["theta"], cur["tx"], cur["ty"]),
                                             ("refit", theta_new, t_new[0], t_new[1])]):
        for e in topo:
            if e["relation"] == 0:
                a, b = e["start_pos"], e["end_pos"]
                ax.plot([a[0], b[0]], [a[1], b[1]], color="0.65", lw=1)
        for p in paths:
            q = transform(p, th, tx, ty)
            ax.plot(q[:, 0], q[:, 1], color="tab:red", lw=1.3, alpha=0.8)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_title(f"{town} {name}")
    fig.savefig(f"viz_output/refit_{town}.png", dpi=110, bbox_inches="tight")
    print(f"  saved viz_output/refit_{town}.png")