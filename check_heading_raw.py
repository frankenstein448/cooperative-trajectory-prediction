import pickle
import numpy as np
from pilot_dataset import PilotDataset
from v2x_message_stream import V2XMetadataLoader
from coord_transform import TOWN_TRANSFORMS

MAX_PER_TOWN = 600
FRAME_STEP = 5
MIN_MOVE = 2.0     # meters between the two poses
MAX_DIST = 3.0     # meters from start to nearest lane segment


def nearest_segment_heading(pos, segs):
    a, b = segs[:, 0], segs[:, 1]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", pos - a, ab) / (np.einsum("ij,ij->i", ab, ab) + 1e-9), 0, 1)
    d = np.linalg.norm(a + t[:, None] * ab - pos, axis=1)
    i = d.argmin()
    return d[i], np.degrees(np.arctan2(ab[i, 1], ab[i, 0]))


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
    segs = np.array([[e["start_pos"], e["end_pos"]] for e in topo if e["relation"] == 0], dtype=float)

    idxs = [i for i, s in enumerate(dataset.samples) if s[3] == town]
    idxs = idxs[::max(1, len(idxs) // MAX_PER_TOWN)]
    diffs = []
    for i in idxs:
        agent_id, scene, frame, _ = dataset.samples[i]
        try:
            p0 = R @ pose_xy(loader, agent_id, scene, frame) + off
            p1 = R @ pose_xy(loader, agent_id, scene, frame + FRAME_STEP) + off
        except Exception:
            continue
        v = p1 - p0
        if np.linalg.norm(v) < MIN_MOVE:
            continue
        dist, lane_yaw = nearest_segment_heading(p0, segs)
        if dist > MAX_DIST:
            continue
        gt_yaw = np.degrees(np.arctan2(v[1], v[0]))
        diffs.append((gt_yaw - lane_yaw + 90) % 180 - 90)
    diffs = np.array(diffs)
    print(f"\n{town}: {len(diffs)} usable samples")
    if len(diffs):
        within = np.mean(np.abs(diffs) <= 20) * 100
        print(f"  median offset {np.median(diffs):.1f} deg, median |offset| {np.median(np.abs(diffs)):.1f} deg, within 20 deg: {within:.0f}%")
        hist, edges = np.histogram(diffs, bins=np.arange(-90, 91, 20))
        for h, lo in zip(hist, edges[:-1]):
            print(f"  {lo:+4.0f} to {lo+20:+4.0f}: {h}")