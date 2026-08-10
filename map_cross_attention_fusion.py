import torch
import torch.nn as nn


class MapCrossAttentionFusion(nn.Module):
    """
    Fuses an agent's embedding with its k nearest HD-map lane node embeddings
    via cross-attention: agent embedding is the query, lane node embeddings
    (from RGCNMapEncoder output) are keys/values.

    Each agent queries only its own locally relevant lane nodes -- no pooling
    of the full map into a single vector.
    """

    def __init__(self, agent_dim, lane_dim=128, d_model=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Linear(agent_dim, d_model)
        self.kv_proj = nn.Linear(lane_dim, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, agent_embed, nearby_lane_embeds, key_padding_mask=None):
        """
        agent_embed: [B, agent_dim]              - one embedding per agent
        nearby_lane_embeds: [B, k, lane_dim]      - gathered via gather_nearby_lane_embeddings()
        key_padding_mask: [B, k] bool, True = ignore (for agents with fewer than k valid nodes)

        Returns: [B, d_model] fused agent-map feature
        """
        q = self.query_proj(agent_embed).unsqueeze(1)   # [B, 1, d_model]
        kv = self.kv_proj(nearby_lane_embeds)             # [B, k, d_model]

        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)  # [B, 1, d_model]
        fused = self.norm(attn_out.squeeze(1) + q.squeeze(1))  # residual + norm
        return self.out_proj(fused)  # [B, d_model]


def gather_nearby_lane_embeddings(lane_embeds, nearby_idx):
    """
    lane_embeds: [num_nodes, lane_dim]  - full RGCNMapEncoder output (h)
    nearby_idx: [B, k] or [k]           - output of get_nearby_lane_nodes()

    Returns: [B, k, lane_dim] (or [k, lane_dim] if unbatched -- caller should
    unsqueeze(0) before passing to MapCrossAttentionFusion).
    """
    if nearby_idx.dim() == 1:
        return lane_embeds[nearby_idx]  # [k, lane_dim]
    return lane_embeds[nearby_idx]  # advanced indexing broadcasts to [B, k, lane_dim]


if __name__ == "__main__":
    from rgcn_map_encoder import load_graph, RGCNMapEncoder, get_nearby_lane_nodes

    TOWN = "Town05"
    data, id_to_idx, mean, std = load_graph(f"topology/{TOWN}_topology.pkl")

    map_encoder = RGCNMapEncoder(in_dim=2, hidden_dim=64, out_dim=128, num_relations=3)
    lane_embeds = map_encoder(data.x, data.edge_index, data.edge_type)  # [num_nodes, 128]

    # dummy batch of 4 agents, agent embedding dim 64 (placeholder until
    # ego/RSU sensor stream encoder exists)
    B, agent_dim, k = 4, 64, 5
    dummy_agent_embed = torch.randn(B, agent_dim)
    dummy_agent_pos_norm = data.x[torch.randint(0, data.x.shape[0], (B,))]  # random valid positions

    nearby_idx = get_nearby_lane_nodes(dummy_agent_pos_norm, data.x, k=k)  # [B, k]
    nearby_embeds = gather_nearby_lane_embeddings(lane_embeds, nearby_idx)  # [B, k, 128]

    fusion = MapCrossAttentionFusion(agent_dim=agent_dim, lane_dim=128, d_model=128)
    fused = fusion(dummy_agent_embed, nearby_embeds)
    print(f"Fused output shape: {fused.shape}")  # expect [B, 128]