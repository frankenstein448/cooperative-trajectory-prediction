import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv


def load_graph(path):
    with open(path, "rb") as f:
        edges = pickle.load(f)

    node_pos = {}
    for e in edges:
        node_pos[e["start_id"]] = e["start_pos"]
        node_pos[e["end_id"]] = e["end_pos"]

    node_ids = sorted(node_pos.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    xs = torch.tensor([node_pos[nid] for nid in node_ids], dtype=torch.float)
    xs = (xs - xs.mean(0)) / (xs.std(0) + 1e-6)

    src = [id_to_idx[e["start_id"]] for e in edges]
    dst = [id_to_idx[e["end_id"]] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor([e["relation"] for e in edges], dtype=torch.long)

    return Data(x=xs, edge_index=edge_index, edge_type=edge_type), id_to_idx


class RGCNMapEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, out_dim=128, num_relations=3, num_bases=None):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        h = self.conv1(x, edge_index, edge_type)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_type)
        return h  # [num_nodes, out_dim] lane embeddings


if __name__ == "__main__":
    model = RGCNMapEncoder(in_dim=2, hidden_dim=64, out_dim=128, num_relations=3)

    for town in ["Town03", "Town04", "Town05"]:
        path = f"topology/{town}_topology.pkl"
        data, id_to_idx = load_graph(path)
        print(f"{town}: {data}")

        embeddings = model(data.x, data.edge_index, data.edge_type)
        print(f"{town}: output embeddings shape = {embeddings.shape}")
        assert embeddings.shape == (data.x.shape[0], 128), f"{town}: shape mismatch!"
        print(f"{town}: OK\n")