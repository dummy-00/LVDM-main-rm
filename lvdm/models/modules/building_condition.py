import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BuildingConditionEncoder(nn.Module):
    def __init__(self, context_dim=768, resolution=256, patch_size=16, video_length=20):
        super().__init__()
        self.context_dim = context_dim
        self.resolution = resolution
        self.patch_size = patch_size
        self.video_length = video_length
        self.proj = nn.Linear(3 * patch_size * patch_size, context_dim)
        num_patches = video_length * (resolution // patch_size) * (resolution // patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, context_dim) * 0.02)
        self.norm = nn.LayerNorm(context_dim)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"Expected building condition with shape [B,C,T,H,W], got {x.shape}")
        if x.shape[-2:] != (self.resolution, self.resolution):
            x = F.interpolate(
                x,
                size=(x.shape[2], self.resolution, self.resolution),
                mode="trilinear",
                align_corners=False,
            )
        tokens = rearrange(
            x,
            "b c t (h p1) (w p2) -> b (t h w) (c p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        tokens = self.proj(tokens)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1]]
        return self.norm(tokens)
