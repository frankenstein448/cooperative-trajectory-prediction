import pickle
import torch
import matplotlib.pyplot as plt
from rgcn_map_encoder import load_graph, get_nearby_lane_nodes

TOWN = "Town05"
AGENT_ID = 1
RAW_AGENT_POS = torch.tensor([87.72, 105.93])

with open(f"topology/{TOWN}_topology.pkl", "rb") as f:
    edges = pickle.load(f)

node_pos = {}
for e in edges:
    node_pos[e["start_id"]] = e["start_pos"]
    node_pos[e["end_id"]] = e["end_pos"]

node_ids = sorted(node_pos.keys())
raw_xs = torch.tensor([node_pos[nid] for nid in node_ids], dtype=torch.float)
mean = raw_xs.mean(0)
std = raw_xs.std(0) + 1e-6

data, id_to_idx, mean, std = load_graph(f"topology/{TOWN}_topology.pkl")  # data.x is normalized

nearby_idx = get_nearby_lane_nodes(RAW_AGENT_POS, raw_xs, k=10)
print(f"{TOWN} agent {AGENT_ID}: nearest lane node indices = {nearby_idx.tolist()}")

nearby_raw = raw_xs[nearby_idx]
print("nearest lane node raw positions:")
print(nearby_raw)

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(raw_xs[:, 0], raw_xs[:, 1], s=4, c="lightgray", label="lane nodes")
for e in edges:
    x0, y0 = e["start_pos"]
    x1, y1 = e["end_pos"]
    ax.plot([x0, x1], [y0, y1], c="lightgray", linewidth=0.5, zorder=1)
ax.scatter(nearby_raw[:, 0], nearby_raw[:, 1], s=60, c="orange", label="k nearest lane nodes", zorder=3)
ax.scatter([RAW_AGENT_POS[0]], [RAW_AGENT_POS[1]], s=120, c="red", marker="*", label=f"agent {AGENT_ID}", zorder=4)
ax.set_title(f"{TOWN} — agent {AGENT_ID} at ({RAW_AGENT_POS[0]:.2f}, {RAW_AGENT_POS[1]:.2f})")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.set_aspect("equal")
plt.savefig(f"{TOWN}_agent{AGENT_ID}_nearby_lanes.png", dpi=150)
print(f"Saved to {TOWN}_agent{AGENT_ID}_nearby_lanes.png")
