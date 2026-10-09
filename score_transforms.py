import pickle
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from pilot_dataset import PilotDataset
from v2x_message_stream import V2XMetadataLoader
from coord_transform import TOWN_TRANSFORMS

CANDIDATES = {
    "Town05": {"refit": (85.60, -52.50, -30.71)},
    "Town04": {"old_candidate_A": (260.50, 37.75, -41.25), "refit": (262.50, 114.05, 211.86)},
    "Town03": {"refit": (168.75, 28.63, -11.03)},
}


def rot(deg):
    th = np.radians(deg)
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


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


dataset = PilotDataset()
loader = V2XMetadataLoader()

for town in ("Town05", "Town04", "Town03"):
    topo = pickle.load(open(f"topology/{town}_topology.pkl", "rb"))
    tree = cKDTree(lane_points(topo))

    frames = defaultdict(set)
    for agent_id, scene, frame, tn in dataset.samples:
        if tn == town:
            frames[(agent_id, scene)].add(frame)

    moving, still = [], []
    for (agent_id, scene), fs in frames.items():
        pts = []
        for f in sorted(fs):
            try:
                pts.append(pose_xy(loader, agent_id, scene, f))
            except Exception:
                pass
        pts = np.array(pts)
        if len(pts) < 2:
            continue
        length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        if length >= 10.0:
            moving.append(pts)
        else:
            still.append(pts.mean(0))
    moving_pts = np.vstack(moving)
    still = np.array(still)

    cur = TOWN_TRANSFORMS[town]
    cands = {"current": (cur["theta"], cur["tx"], cur["ty"]), **CANDIDATES[town]}
    print(f"\n{town}: {len(moving)} moving paths, {len(still)} stationary agents")
    for name, (th, tx, ty) in cands.items():
        row = []
        for label, raw in (("moving", moving_pts), ("stationary", still)):
            if len(raw) == 0:
                continue
            d, _ = tree.query(raw @ rot(th).T + np.array([tx, ty]))
            row.append(f"{label}: median {np.median(d):5.2f} m, within 2 m {np.mean(d < 2) * 100:3.0f}%")
        print(f"  {name:16s} theta={th:7.2f}  " + " | ".join(row))