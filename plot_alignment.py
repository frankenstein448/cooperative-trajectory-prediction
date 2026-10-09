import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pilot_dataset import PilotDataset
from v2x_message_stream import V2XMetadataLoader
from coord_transform import TOWN_TRANSFORMS


def pose_xy(loader, agent_id, scene, frame):
    rec = loader.find_sample_data(agent_id, scene, frame)
    tr = loader.ego_pose[rec["ego_pose_token"]]["translation"]
    return np.array([tr[0], tr[1]])


dataset = PilotDataset()
loader = V2XMetadataLoader()

for town in ("Town03", "Town04", "Town05"):
    t = TOWN_TRANSFORMS[town]
    th = np.radians(t["theta"])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    off = np.array([t["tx"], t["ty"]])
    topo = pickle.load(open(f"topology/{town}_topology.pkl", "rb"))

    frames = defaultdict(set)
    for agent_id, scene, frame, tn in dataset.samples:
        if tn == town:
            frames[(agent_id, scene)].add(frame)

    paths = []
    for (agent_id, scene), fs in frames.items():
        pts = []
        for f in sorted(fs):
            try:
                pts.append(R @ pose_xy(loader, agent_id, scene, f) + off)
            except Exception:
                pass
        if len(pts) >= 2:
            paths.append(np.array(pts))
    if not paths:
        continue

    allp = np.vstack(paths)
    lo, hi = allp.min(0) - 40, allp.max(0) + 40

    fig, ax = plt.subplots(figsize=(14, 14))
    for e in topo:
        if e["relation"] != 0:
            continue
        a, b = e["start_pos"], e["end_pos"]
        if max(a[0], b[0]) < lo[0] or min(a[0], b[0]) > hi[0] or max(a[1], b[1]) < lo[1] or min(a[1], b[1]) > hi[1]:
            continue
        ax.plot([a[0], b[0]], [a[1], b[1]], color="0.6", lw=1)
    for p in paths:
        ax.plot(p[:, 0], p[:, 1], lw=1.5, color="tab:red", alpha=0.8)
        ax.plot(p[0, 0], p[0, 1], "k.", ms=4)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_title(f"{town}: agent paths (red) over lane graph (gray), {len(paths)} paths")
    fig.savefig(f"viz_output/align_{town}.png", dpi=130, bbox_inches="tight")
    print("saved", f"viz_output/align_{town}.png", len(paths), "paths")