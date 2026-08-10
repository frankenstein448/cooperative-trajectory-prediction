"""
ego_sensor_encoder.py
Real PointNet-style ego LiDAR encoder, replacing the EgoSensorEncoder stub.
Input: semantic LiDAR point cloud, (x, y, z, cos_angle, semantic_tag) per point.
Run in gpu-test-venv (or coperception-venv if loading .pcd.bin from there).
"""
import numpy as np
import torch
import torch.nn as nn

EMBED_DIM = 128
POINT_FEAT_DIM = 5  # x, y, z, cos_angle, semantic_tag


def load_semantic_lidar(path: str) -> torch.Tensor:
    """Load one .pcd.bin file -> [N, 5] tensor."""
    raw = np.fromfile(path, dtype=np.float32)
    assert raw.size % POINT_FEAT_DIM == 0, f"unexpected float count: {raw.size}"
    points = raw.reshape(-1, POINT_FEAT_DIM)
    return torch.from_numpy(points)


class EgoSensorEncoder(nn.Module):
    """PointNet-style: per-point MLP -> max-pool -> MLP, per agent."""
    def __init__(self, in_dim=POINT_FEAT_DIM, hidden_dim=64, out_dim=EMBED_DIM):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out_mlp = nn.Linear(hidden_dim, out_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: [N, 5] for ONE agent -> [128] embedding."""
        feats = self.point_mlp(points)          # [N, 64]
        pooled, _ = feats.max(dim=0)              # [64]
        return self.out_mlp(pooled)               # [128]

    def encode_batch(self, point_clouds: list[torch.Tensor]) -> torch.Tensor:
        """point_clouds: list of [N_i, 5] tensors (variable N per agent) -> [n_agents, 128]."""
        embeddings = [self.forward(pc) for pc in point_clouds]
        return torch.stack(embeddings, dim=0)


if __name__ == "__main__":
    path = "/home/karra/V2X-Sim-2.0/sweeps/SEMLIDAR_TOP_id_4/scene_89_000100.pcd.bin"
    points = load_semantic_lidar(path)
    print(f"Loaded points shape: {points.shape}")

    encoder = EgoSensorEncoder()
    embedding = encoder(points)
    print(f"Single agent embedding shape: {embedding.shape}")

    # simulate a scene with n agents, each with their own (variable-size) point cloud
    fake_batch = [points, points[:30000], points[:10000]]
    batch_embed = encoder.encode_batch(fake_batch)
    print(f"Batch embedding shape: {batch_embed.shape}")