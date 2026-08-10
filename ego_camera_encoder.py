"""
ego_camera_encoder.py
Small custom CNN camera encoder for the ego stream (3 front views: CAM_FRONT,
CAM_FRONT_LEFT, CAM_FRONT_RIGHT). Shared backbone across views, concat-then-project.
Run in gpu-test-venv.
"""
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

EMBED_DIM = 128
IMG_SIZE = 224  # resize target, keeps compute manageable

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),  # [3, H, W], values in [0,1]
])


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transform(img)  # [3, 224, 224]


class SmallCNN(nn.Module):
    """Per-view feature extractor: raw image -> [128] embedding."""
    def __init__(self, out_dim=EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),   # 224 -> 112
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 112 -> 56
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> [64, 1, 1]
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [3, H, W] single image -> [out_dim]."""
        feat = self.conv(x.unsqueeze(0))   # [1, 64, 1, 1]
        feat = feat.flatten(1)             # [1, 64]
        return self.proj(feat).squeeze(0)  # [out_dim]


class EgoCameraEncoder(nn.Module):
    """3 front views -> shared CNN -> concat -> project to [128]."""
    def __init__(self, n_views=3, out_dim=EMBED_DIM):
        super().__init__()
        self.backbone = SmallCNN(out_dim=out_dim)  # shared weights across views
        self.view_fuse = nn.Linear(out_dim * n_views, out_dim)

    def forward(self, images: list[torch.Tensor]) -> torch.Tensor:
        """images: list of n_views tensors [3,H,W] for ONE agent -> [128]."""
        view_embeds = [self.backbone(img) for img in images]
        concat = torch.cat(view_embeds, dim=0)   # [128*n_views]
        return self.view_fuse(concat)             # [128]

    def encode_batch(self, agents_images: list[list[torch.Tensor]]) -> torch.Tensor:
        """agents_images: n_agents lists of n_views tensors -> [n_agents, 128]."""
        embeddings = [self.forward(imgs) for imgs in agents_images]
        return torch.stack(embeddings, dim=0)


if __name__ == "__main__":
    base = "/home/karra/V2X-Sim-2.0/sweeps/{view}_id_4/scene_100_000006.jpg"
    views = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    images = [load_image(base.format(view=v)) for v in views]

    encoder = EgoCameraEncoder()
    embedding = encoder(images)
    print(f"Single agent camera embedding shape: {embedding.shape}")

    fake_batch = [images, images]  # simulate 2 agents reusing same images
    batch_embed = encoder.encode_batch(fake_batch)
    print(f"Batch camera embedding shape: {batch_embed.shape}")