"""
coop_sensor_stream.py
Cooperative sensor stream: an encoder with EgoStream's architecture but its own
weights (not shared with the ego stream) runs on OTHER agents' LiDAR+camera data
and mean-pools across them -> one [128] embedding per ego agent. In the pipeline
the cooperators are the other camera-equipped CAVs (pilot_dataset passes agents
1-4 minus the ego to find_coop_neighbors). The env task's RSU stream (env_model use_rsu) is a separate
instance with n_views=4 and its own weights, run on the RSU (agent 0) alone.
Run in gpu-test-venv.
"""
import torch
import torch.nn as nn

from ego_stream import EgoStream
from ego_sensor_encoder import load_semantic_lidar
from ego_camera_encoder import load_image

EMBED_DIM = 128


class CoopSensorStream(nn.Module):
    """Encodes each cooperating agent with a shared EgoStream encoder, then mean-pools."""
    def __init__(self, out_dim=EMBED_DIM, n_views=3):
        super().__init__()
        # Its own EgoStream instance: shared across all cooperating agents, but NOT
        # shared with EndToEndFusion's ego_encoder -- separate weights, same architecture.
        self.shared_encoder = EgoStream(out_dim=out_dim, n_views=n_views)

    def forward(self, coop_points: list[torch.Tensor],
                coop_images: list[list[torch.Tensor]]) -> torch.Tensor:
        """
        coop_points: list of [N_i,5] lidar, one per cooperating agent (excludes ego itself)
        coop_images: list of (3-view image lists), one per cooperating agent
        -> [128] pooled cooperative embedding for the ego agent asking.
        """
        if len(coop_points) == 0:
            return torch.zeros(EMBED_DIM)  # no cooperators in range this frame

        embeds = [self.shared_encoder(pts, imgs)
                  for pts, imgs in zip(coop_points, coop_images)]
        stacked = torch.stack(embeds, dim=0)   # [n_coop_agents, 128]
        return stacked.mean(dim=0)              # [128]

    def encode_batch(self, agents_coop_points: list[list[torch.Tensor]],
                      agents_coop_images: list[list[list[torch.Tensor]]]) -> torch.Tensor:
        """Per-ego-agent lists of cooperators -> [n_ego_agents, 128]."""
        embeddings = [self.forward(cp, ci)
                      for cp, ci in zip(agents_coop_points, agents_coop_images)]
        return torch.stack(embeddings, dim=0)


if __name__ == "__main__":
    lidar_path = "/home/karra/V2X-Sim-2.0/sweeps/SEMLIDAR_TOP_id_4/scene_89_000100.pcd.bin"
    points = load_semantic_lidar(lidar_path)

    cam_base = "/home/karra/V2X-Sim-2.0/sweeps/{view}_id_4/scene_100_000006.jpg"
    views = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    images = [load_image(cam_base.format(view=v)) for v in views]

    coop = CoopSensorStream()

    # simulate ego agent 1 having 2 cooperators nearby (reusing same files as stand-ins)
    coop_points_for_agent1 = [points, points]
    coop_images_for_agent1 = [images, images]
    single_embed = coop(coop_points_for_agent1, coop_images_for_agent1)
    print(f"Single ego agent's coop embedding shape: {single_embed.shape}")

    # simulate a batch of 2 ego agents, each with their own cooperator lists
    batch_points = [coop_points_for_agent1, [points]]       # agent1: 2 coops, agent2: 1 coop
    batch_images = [coop_images_for_agent1, [images]]
    batch_embed = coop.encode_batch(batch_points, batch_images)
    print(f"Batch coop embedding shape: {batch_embed.shape}")