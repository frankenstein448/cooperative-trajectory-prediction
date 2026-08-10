import torch
from rgcn_map_encoder import load_graph, get_nearby_lane_nodes

TOWN = "Town05"
carla_x, carla_y = -130.78, 114.93  # idx=0, dist=0.08m confirmed on-lane

data, id_to_idx, mean, std = load_graph(f"topology/{TOWN}_topology.pkl")

agent_pos = torch.tensor([carla_x, carla_y], dtype=torch.float)
agent_pos_norm = (agent_pos - mean) / std

idx = get_nearby_lane_nodes(agent_pos_norm, data.x, k=5)
idx_to_id = {v: k for k, v in id_to_idx.items()}

node_pos_real = data.x * std + mean
for i in idx.tolist():
    node_xy = node_pos_real[i]
    dist = ((node_xy[0] - carla_x)**2 + (node_xy[1] - carla_y)**2) ** 0.5
    print(f"  node_id={idx_to_id[i]} dist={dist:.2f}m")