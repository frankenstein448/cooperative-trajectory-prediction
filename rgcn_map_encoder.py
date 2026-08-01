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


def get_nearby_lane_nodes(agent_pos, node_positions, k=10):
    """
    agent_pos: tensor [2] or [B, 2] - agent (x, y) position(s), same normalization as node_positions
    node_positions: tensor [N, 2] - all lane node positions (data.x from load_graph)
    k: number of nearest lane nodes to retrieve

    Returns: indices [k] or [B, k] into node_positions for the k nearest nodes
    """
    if agent_pos.dim() == 1:
        agent_pos = agent_pos.unsqueeze(0)  # [1, 2]

    dists = torch.cdist(agent_pos, node_positions)
    _, indices = torch.topk(dists, k=min(k, node_positions.shape[0]), dim=1, largest=False)

    return indices.squeeze(0) if indices.shape[0] == 1 else indices  # [k] or [B, k]


if __name__ == "__main__":
    model = RGCNMapEncoder(in_dim=2, hidden_dim=64, out_dim=128, num_relations=3)

    for town in ["Town03", "Town04", "Town05"]:
        path = f"topology/{town}_topology.pkl"
        data, id_to_idx = load_graph(path)
        print(f"{town}: {data}")

        embeddings = model(data.x, data.edge_index, data.edge_type)
        print(f"{town}: output embeddings shape = {embeddings.shape}")
        assert embeddings.shape == (data.x.shape[0], 128), f"{town}: shape mismatch!"

        dummy_agent_pos = data.x[0]
        nearby_idx = get_nearby_lane_nodes(dummy_agent_pos, data.x, k=10)
        nearby_embeddings = embeddings[nearby_idx]
        assert nearby_embeddings.shape == (10, 128), f"{town}: nearby embedding shape mismatch!"
        print(f"{town}: nearest-neighbor lookup OK, retrieved {nearby_embeddings.shape}")
        print(f"{town}: OK\n")
