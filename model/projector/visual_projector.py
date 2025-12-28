# coding=utf-8
"""
Visual Projector for compressing CLIP visual tokens using Perceiver Resampler.

Reference: Flamingo (DeepMind), BLIP-2 (Salesforce)

Tensor Flow:
============
CLIP Output: [B, 257, 1024] (256 patches + 1 CLS, hidden=1024)
       ↓
Resampler: Cross-attention with learnable queries
       ↓
Output: [B, 64, 768] (compressed tokens, matched to text hidden size)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class VisualProjectorConfig:
    """Configuration for Visual Projector (Resampler)"""

    # Input configuration (from CLIP)
    vision_hidden_size: int = 1024  # CLIP ViT-L/14 hidden size
    vision_num_tokens: int = 257  # 256 patches + 1 CLS token

    # Output configuration (to match text encoder)
    output_hidden_size: int = 768  # RoBERTa/XLM-R hidden size
    num_output_tokens: int = 64  # Compressed number of visual tokens

    # Resampler configuration
    num_heads: int = 12
    num_layers: int = 2  # Number of cross-attention layers

    # Regularization
    dropout: float = 0.1


class Resampler(nn.Module):
    """
    Perceiver Resampler from Flamingo.

    Uses learnable query tokens to compress visual tokens via cross-attention.
    This is the recommended projector for high-quality visual representations.

    Architecture:
    - Learnable query tokens: [1, num_output_tokens, output_hidden_size]
    - Cross-attention layers: Query attends to visual features
    - FFN after each cross-attention

    Why Resampler is best:
    1. Learnable queries can specialize for different visual patterns
    2. Cross-attention captures global relationships between patches
    3. Used in SOTA models: Flamingo, BLIP-2, LLaVA-1.5

    Tensor Flow:
    - Input: [B, 257, 1024] (CLIP output)
    - Query: [1, 64, 768] → [B, 64, 768]
    - Cross-Attention: Query attends to projected visual features
    - Output: [B, 64, 768]
    """

    def __init__(self, config: VisualProjectorConfig):
        super().__init__()
        self.config = config

        # Project CLIP features to output dimension
        self.input_proj = nn.Linear(
            config.vision_hidden_size, config.output_hidden_size
        )

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_output_tokens, config.output_hidden_size) * 0.02
        )

        # Cross-attention layers with FFN
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=config.output_hidden_size,
                            num_heads=config.num_heads,
                            dropout=config.dropout,
                            batch_first=True,
                        ),
                        "cross_attn_norm": nn.LayerNorm(config.output_hidden_size),
                        "ffn": nn.Sequential(
                            nn.Linear(
                                config.output_hidden_size,
                                config.output_hidden_size * 4,
                            ),
                            nn.GELU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(
                                config.output_hidden_size * 4,
                                config.output_hidden_size,
                            ),
                            nn.Dropout(config.dropout),
                        ),
                        "ffn_norm": nn.LayerNorm(config.output_hidden_size),
                    }
                )
                for _ in range(config.num_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(config.output_hidden_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability"""
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Compress visual features using cross-attention.

        Args:
            visual_features: [batch_size, 257, 1024] - CLIP output

        Returns:
            compressed_features: [batch_size, 64, 768] - Compressed visual tokens
        """
        batch_size = visual_features.shape[0]

        # Project input to output dimension
        visual_features = self.input_proj(visual_features)  # [B, 257, 768]

        # Expand query tokens for batch
        queries = self.query_tokens.expand(batch_size, -1, -1)  # [B, 64, 768]

        # Apply cross-attention layers
        for layer in self.layers:
            # Pre-norm cross-attention
            residual = queries
            queries_normed = layer["cross_attn_norm"](queries)
            queries_attn, _ = layer["cross_attn"](
                query=queries_normed,
                key=visual_features,
                value=visual_features,
            )
            queries = residual + queries_attn

            # Pre-norm FFN
            residual = queries
            queries_normed = layer["ffn_norm"](queries)
            queries = residual + layer["ffn"](queries_normed)

        return self.output_norm(queries)


class VisualProjector(nn.Module):
    """
    Visual Projector using Perceiver Resampler.

    Compresses CLIP visual tokens (257 tokens) to a smaller number (64 tokens)
    while projecting to the text encoder's hidden dimension.

    This compression is CRUCIAL for:
    1. Balancing visual vs text tokens in MoE (64 visual ≈ 32-64 text)
    2. Reducing computational cost in MoE layers
    3. Matching hidden dimensions between CLIP (1024) and RoBERTa (768)

    Example:
        config = VisualProjectorConfig(
            vision_hidden_size=1024,  # CLIP ViT-L/14
            output_hidden_size=768,   # RoBERTa
            num_output_tokens=64,     # Compressed tokens
        )
        projector = VisualProjector(config)

        # CLIP output
        visual_features = clip_model(images)  # [B, 257, 1024]

        # Compress
        compressed = projector(visual_features)  # [B, 64, 768]
    """

    def __init__(self, config: VisualProjectorConfig):
        super().__init__()
        self.config = config
        self.projector = Resampler(config)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Compress and project visual features.

        Args:
            visual_features: [batch_size, vision_num_tokens, vision_hidden_size]
                           e.g., [B, 257, 1024] from CLIP ViT-L/14

        Returns:
            compressed_features: [batch_size, num_output_tokens, output_hidden_size]
                               e.g., [B, 64, 768] to match RoBERTa
        """
        return self.projector(visual_features)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Visual Projector (Resampler)")
    print("=" * 60)

    # Configuration
    config = VisualProjectorConfig(
        vision_hidden_size=1024,  # CLIP ViT-L/14
        vision_num_tokens=257,  # 256 patches + 1 CLS
        output_hidden_size=768,  # RoBERTa
        num_output_tokens=64,  # Compressed tokens
        num_heads=12,
        num_layers=2,
    )

    print(f"\nConfiguration:")
    print(f"  Input:  [B, {config.vision_num_tokens}, {config.vision_hidden_size}]")
    print(f"  Output: [B, {config.num_output_tokens}, {config.output_hidden_size}]")
    print(
        f"  Compression ratio: {config.vision_num_tokens} → {config.num_output_tokens} "
        f"({config.vision_num_tokens / config.num_output_tokens:.1f}x)"
    )

    # Create projector
    projector = VisualProjector(config)

    # Count parameters
    num_params = sum(p.numel() for p in projector.parameters())
    print(f"  Parameters: {num_params:,}")

    # Test forward pass
    print(f"\nTesting forward pass:")
    batch_size = 2
    visual_features = torch.randn(batch_size, 257, 1024)
    print(f"  Input shape:  {visual_features.shape}")

    compressed = projector(visual_features)
    print(f"  Output shape: {compressed.shape}")

    # Verify dimensions
    assert compressed.shape == (batch_size, 64, 768), "Shape mismatch!"

    print(f"\n✓ Resampler projector test passed!")
    print("=" * 60)
