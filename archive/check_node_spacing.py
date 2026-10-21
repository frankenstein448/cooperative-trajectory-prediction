import torch
from rgcn_map_encoder import load_graph

TOWN = "Town05"
data, id_to_idx, mean, std = load_graph(f"topology/{TOWN}_topology.pkl")
node_pos_real = data.x * std + mean

print(f"Total nodes: {node_pos_real.shape[0]}")

# nearest-neighbor distance for every node, as a proxy for typical spacing
from scipy.spatial import cKDTree
tree = cKDTree(node_pos_real.numpy())
dists, _ = tree.query(node_pos_real.numpy(), k=2)  # k=2: self + nearest other
nn_dists = dists[:, 1]
print(f"median nearest-node spacing: {nn_dists.mean():.2f}m (mean), {sorted(nn_dists)[len(nn_dists)//2]:.2f}m (median)")
print(f"max spacing: {nn_dists.max():.2f}m")
