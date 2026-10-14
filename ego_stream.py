"""
ego_stream.py
Unified ego sensor stream: LiDAR (PointNet) + camera (small CNN, 3 views) -> [128].
Replaces the EgoSensorEncoder stub in four_stream_fusion.py.
Run in gpu-test-venv.
"""
import torch
import torch.nn as nn

from ego_sensor_encoder import EgoSensorEncoder, load_semantic_lidar
from ego_camera_encoder import EgoCameraEncoder, load_image

EMBED_DIM = 128


class EgoStream(nn.Module):
    """Fuses LiDAR + camera embeddings -> single [128] ego embedding, per agent.
    n_views: camera views per agent (3 for the CAVs; 4 for the RSU stream, env_model use_rsu)."""
    def __init__(self, out_dim=EMBED_DIM, n_views=3):
        super().__init__()
        self.lidar_enc = EgoSensorEncoder(out_dim=out_dim)
        self.camera_enc = EgoCameraEncoder(n_views=n_views, out_dim=out_dim)
        self.fuse = nn.Linear(out_dim * 2, out_dim)

    def forward(self, points: torch.Tensor, images: list[torch.Tensor]) -> torch.Tensor:
        """points: [N,5] lidar for ONE agent. images: n_views camera views for ONE agent -> [128]."""
        lidar_embed = self.lidar_enc(points)      # [128]
        cam_embed = self.camera_enc(images)        # [128]
        concat = torch.cat([lidar_embed, cam_embed], dim=0)  # [256]
        return self.fuse(concat)                    # [128]

    def encode_batch(self, agents_points: list[torch.Tensor],
                      agents_images: list[list[torch.Tensor]]) -> torch.Tensor:
        """n_agents worth of (points, images) -> [n_agents, 128]."""
        embeddings = [self.forward(p, imgs) for p, imgs in zip(agents_points, agents_images)]
        return torch.stack(embeddings, dim=0)


if __name__ == "__main__":
    lidar_path = "/home/karra/V2X-Sim-2.0/sweeps/SEMLIDAR_TOP_id_4/scene_89_000100.pcd.bin"
    points = load_semantic_lidar(lidar_path)

    cam_base = "/home/karra/V2X-Sim-2.0/sweeps/{view}_id_4/scene_100_000006.jpg"
    views = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    images = [load_image(cam_base.format(view=v)) for v in views]

    ego = EgoStream()
    embedding = ego(points, images)
    print(f"Single agent ego embedding shape: {embedding.shape}")

    fake_batch_points = [points, points]
    fake_batch_images = [images, images]
    batch_embed = ego.encode_batch(fake_batch_points, fake_batch_images)
    print(f"Batch ego embedding shape: {batch_embed.shape}")