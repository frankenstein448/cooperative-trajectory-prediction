import torch
from rgcn_map_encoder import load_graph

TOWN = "Town05"
carla_x, carla_y = -130.78, 114.93

data, id_to_idx, mean, std = load_graph(f"topology/{TOWN}_topology.pkl")
node_pos_real = data.x * std + mean

target = torch.tensor([carla_x, carla_y])
dists = torch.sqrt(((node_pos_real - target) ** 2).sum(dim=1))
sorted_dists, sorted_idx = torch.sort(dists)

idx_to_id = {v: k for k, v in id_to_idx.items()}
for i in range(5):
    idx = sorted_idx[i].item()
    print(f"  node_id={idx_to_id[idx]} dist={sorted_dists[i].item():.2f}m pos={node_pos_real[idx].tolist()}")
