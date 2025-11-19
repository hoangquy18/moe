"""
Patch Embedding for Vision-Language models
Similar to CLIPVisionEmbeddings but compatible with ImageTextModel architecture
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Patch embedding for images, compatible with ImageTextModel architecture"""

    def __init__(self, image_size, patch_size, emb_dim, num_channels=3):
        super().__init__()
        self.image_size = (
            image_size
            if isinstance(image_size, (list, tuple))
            else [image_size, image_size]
        )
        self.patch_size = (
            patch_size
            if isinstance(patch_size, (list, tuple))
            else [patch_size, patch_size]
        )
        self.emb_dim = emb_dim
        self.num_channels = num_channels

        # Calculate number of patches
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (
            self.image_size[1] // self.patch_size[1]
        )

        # Patch embedding: Conv2d to convert image patches to embeddings
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # Position embedding for patches
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, emb_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Image tensor of shape [batch_size, num_channels, height, width]

        Returns:
            embeddings: Patch embeddings of shape [batch_size, num_patches, emb_dim]
        """
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype

        # Convert image to patch embeddings
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        # Flatten spatial dimensions: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # Add position embeddings
        embeddings = patch_embeds + self.position_embedding(self.position_ids)

        return embeddings
