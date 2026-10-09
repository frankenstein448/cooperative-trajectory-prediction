import pickle
import numpy as np
from pilot_dataset import PilotDataset
from v2x_message_stream import V2XMetadataLoader
from ground_truth_extractor import get_future_trajectory, HORIZON
from visualize_predictions import reconstruct_world_traj

MAX_PER_TOWN = 400
MAX_DIST = 3.0  # meters from start position to nearest lane segment


def nearest_segment_heading(pos, segs):
    a, b = segs[:, 0], segs[:, 1]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", pos - a, ab) / (np.einsum("ij,ij->i", ab, ab) + 1e-9), 0, 1)
    d = np.linalg.norm(a + t[:, None] * ab - pos, axis=1)
    i = d.argmin()
    return d[i], np.degrees(np.arctan2(ab[i, 1], ab[i, 0]))


dataset = PilotDataset()
loader = V2XMetadataLoader()

for town in ("Town03", "Town04", "Town05"):
    topo = pickle.load(open(f"topology/{town}_topology.pkl", "rb"))
    segs = np.array([[e["start_pos"], e["end_pos"]] for e in topo if e["relation"] == 0], dtype=float)
    idxs = [i for i, s in enumerate(dataset.samples) if s[3] == town]
    idxs = idxs[::max(1, len(idxs) // MAX_PER_TOWN)]
    diffs = []
    for i in idxs:
        agent_id, scene, frame, _ = dataset.samples[i]
        gt, mask = get_future_trajectory(loader, agent_id, scene, frame, HORIZON)
        if mask.sum() < HORIZON or (gt[-1] - gt[0]).norm().item() < 8.0:
            continue
        w = reconstruct_world_traj(loader, agent_id, scene, frame, gt, mask, town)
        w = w[~np.isnan(w[:, 0])]
        d0 = np.linalg.norm(w[:, :2] - w[0, :2], axis=1)
        far = np.nonzero(d0 >= 2.0)[0]
        if len(far) == 0:
            continue
        v = w[far[0], :2] - w[0, :2]
        dist, lane_yaw = nearest_segment_heading(w[0, :2], segs)
        if dist > MAX_DIST:
            continue
        gt_yaw = np.degrees(np.arctan2(v[1], v[0]))
        diffs.append((gt_yaw - lane_yaw + 90) % 180 - 90)  # lane direction mod 180
    diffs = np.array(diffs)
    print(f"\n{town}: {len(diffs)} usable samples")
    if len(diffs):
        print(f"  median offset {np.median(diffs):.1f} deg, median |offset| {np.median(np.abs(diffs)):.1f} deg")
        hist, edges = np.histogram(diffs, bins=np.arange(-90, 91, 20))
        for h, lo in zip(hist, edges[:-1]):
            print(f"  {lo:+4.0f} to {lo+20:+4.0f}: {h}")