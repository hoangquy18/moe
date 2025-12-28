# coding=utf-8
"""
Vision-Language Mixture of Experts (VL-MoE) Model

This model extends the trained MultiModalEncoder with MoE layers.
It preserves the trained encoders and adds MoE on top.

Architecture:
=============
From mm_encoder.py (pretrained):
- text_encoder: Full XLM-RoBERTa
- vision_text_model: CLIP (frozen)
- xlmr_text_projection: 768 → 512
- text_projection_output: 512 → 768
- vision_projection_output: 768 → 768

New (this file):
- Visual Projector (Resampler): Compress visual tokens
- MoE Layers: Shared + Routed experts
- Output Head: Task-specific

Tensor Flow:
============
1. Vision: Image → CLIP → [B, 257, 768] → Projector → [B, 64, 768]
2. Text: Tokens → XLM-RoBERTa → [B, T, 768] → CLS → [B, 768]
3. Fusion: Concat → [B, 64+1, 768] or use separately
4. MoE Layers: Process fused tokens
5. Output: Contrastive/MLM/Causal LM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Literal
import math

from transformers import CLIPModel, AutoModel, AutoConfig

# Import our custom modules
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from model.moe.deepseek_moe import DeepSeekMoELayer, DeepSeekMoEConfig
from model.projector.visual_projector import VisualProjector, VisualProjectorConfig
from model.text.text_encoder import TextEncoder
from model.config import TextConfig


@dataclass
class VLMoEConfig:
    """Configuration for Vision-Language MoE Model"""

    # Vision Encoder (same as mm_encoder.py)
    vision_model_name: str = "openai/clip-vit-base-patch32"
    freeze_vision_encoder: bool = True

    # Visual Projector (Resampler) - NEW for MoE
    num_visual_tokens: int = 64  # Compressed number of visual tokens
    projector_num_layers: int = 2  # Number of cross-attention layers in Resampler

    # Text Encoder (same as mm_encoder.py)
    text_model_name: str = "FacebookAI/xlm-roberta-base"
    freeze_text_encoder: bool = False
    max_text_length: int = 32

    # Hidden dimensions (will be auto-filled from models)
    clip_text_dim: int = 512  # CLIP text hidden size
    clip_vision_dim: int = 768  # CLIP vision hidden size
    xlmr_text_dim: int = 768  # XLM-R hidden size
    vision_num_tokens: int = 50  # 7x7 + 1 CLS for ViT-B/32

    # MoE Configuration
    num_shared_experts: int = 2
    num_routed_experts: int = 64
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 3072
    num_moe_layers: int = 6

    # Loss weights
    load_balance_weight: float = 0.01
    z_loss_weight: float = 0.001
    mi_loss_weight: float = 0.0

    # Training
    dropout: float = 0.1
    hidden_act: str = "silu"

    # Task configuration
    task_type: Literal["mlm", "causal_lm", "contrastive"] = "contrastive"
    vocab_size: int = 250002  # XLM-R vocab size


class VLMoETransformerLayer(nn.Module):
    """
    Transformer layer with DeepSeek-style MoE replacing FFN.
    """

    def __init__(
        self, hidden_size: int, moe_config: DeepSeekMoEConfig, dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=12,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout)

        # MoE layer (replaces FFN)
        self.moe = DeepSeekMoELayer(moe_config)
        self.moe_norm = nn.LayerNorm(hidden_size)
        self.moe_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        modality_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        if attention_mask is not None and attention_mask.dim() == 2:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        hidden_states, _ = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
        )
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        # MoE with residual
        residual = hidden_states
        hidden_states = self.moe_norm(hidden_states)
        hidden_states = self.moe(hidden_states, modality_indices)
        hidden_states = self.moe_dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def get_aux_losses(self) -> Dict[str, torch.Tensor]:
        return self.moe.get_aux_losses()


class VLMoEModel(nn.Module):
    """
    Vision-Language MoE Model that extends trained MultiModalEncoder.

    This model:
    1. Uses the same encoder structure as mm_encoder.py
    2. Adds Visual Projector for token compression
    3. Adds MoE layers for fusion and processing
    4. Supports loading pretrained weights via upcycling
    """

    def __init__(self, config: VLMoEConfig):
        super().__init__()
        self.config = config

        # ==================== Text Encoder (same as mm_encoder.py) ====================
        # Full XLM-RoBERTa text encoder
        text_config = TextConfig(text_model_name=config.text_model_name)
        self.text_encoder = TextEncoder(text_config)
        config.xlmr_text_dim = self.text_encoder.text_config.hidden_size
        config.vocab_size = self.text_encoder.text_config.vocab_size

        if config.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # ==================== Vision Encoder (same as mm_encoder.py) ====================
        # CLIP model (frozen, serves as vision backbone)
        try:
            self.vision_text_model = CLIPModel.from_pretrained(
                config.vision_model_name, trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Could not load CLIP weights: {e}")
            from transformers import CLIPConfig

            clip_config = CLIPConfig.from_pretrained(config.vision_model_name)
            self.vision_text_model = CLIPModel(clip_config)

        # Freeze CLIP model (same as mm_encoder.py)
        for param in self.vision_text_model.parameters():
            param.requires_grad = False

        # Get dimensions from CLIP
        config.clip_text_dim = self.vision_text_model.config.text_config.hidden_size
        config.clip_vision_dim = self.vision_text_model.config.vision_config.hidden_size

        # ==================== Projection Layers (same as mm_encoder.py) ====================
        # These are the trained layers we want to preserve
        self.xlmr_text_projection = nn.Linear(
            config.xlmr_text_dim, config.clip_text_dim
        )  # 768 -> 512

        self.text_projection_output = nn.Linear(
            config.clip_text_dim, config.xlmr_text_dim
        )  # 512 -> 768

        self.vision_projection_output = nn.Linear(
            config.clip_vision_dim, config.xlmr_text_dim
        )  # 768 -> 768

        # Logit scale and bias (same as mm_encoder.py)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_bias = nn.Parameter(torch.ones([]) * -10)

        # ==================== Visual Projector (NEW for MoE) ====================
        # Compress visual tokens for MoE
        projector_config = VisualProjectorConfig(
            vision_hidden_size=config.clip_vision_dim,
            vision_num_tokens=config.vision_num_tokens,
            output_hidden_size=config.xlmr_text_dim,
            num_output_tokens=config.num_visual_tokens,
            num_layers=config.projector_num_layers,
            dropout=config.dropout,
        )
        self.visual_projector = VisualProjector(projector_config)

        # ==================== Modality Embeddings ====================
        self.modality_embedding = nn.Embedding(2, config.xlmr_text_dim)

        # ==================== MoE Transformer Layers ====================
        moe_config = DeepSeekMoEConfig(
            hidden_size=config.xlmr_text_dim,
            intermediate_size=config.moe_intermediate_size,
            num_shared_experts=config.num_shared_experts,
            num_routed_experts=config.num_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            load_balance_weight=config.load_balance_weight,
            z_loss_weight=config.z_loss_weight,
            mi_loss_weight=config.mi_loss_weight,
            hidden_act=config.hidden_act,
            expert_dropout=config.dropout,
        )

        self.moe_layers = nn.ModuleList(
            [
                VLMoETransformerLayer(config.xlmr_text_dim, moe_config, config.dropout)
                for _ in range(config.num_moe_layers)
            ]
        )

        # ==================== Output Head ====================
        self.output_norm = nn.LayerNorm(config.xlmr_text_dim)

        if config.task_type == "contrastive":
            self.visual_head = nn.Linear(config.xlmr_text_dim, config.xlmr_text_dim)
            self.text_head = nn.Linear(config.xlmr_text_dim, config.xlmr_text_dim)

        self._init_new_weights()

    def _init_new_weights(self):
        """Initialize only the NEW weights (not pretrained ones)"""
        # Only initialize projector, modality embedding, MoE layers, output heads
        for module in [self.visual_projector, self.modality_embedding, self.moe_layers]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.02)

    def encode_vision(
        self, pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images using CLIP vision encoder.

        Returns:
            visual_hidden: [B, num_tokens, hidden_size] - All hidden states
            visual_cls: [B, hidden_size] - CLS token features (projected)
        """
        with torch.no_grad():
            vision_outputs = self.vision_text_model.vision_model(pixel_values)
            visual_hidden = vision_outputs.last_hidden_state  # [B, 50, 768]

        # Extract CLS token and project (same as mm_encoder.py)
        visual_cls = visual_hidden[:, 0]  # [B, 768]
        visual_cls = self.vision_projection_output(visual_cls)  # [B, 768]

        return visual_hidden, visual_cls

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text using XLM-RoBERTa.

        Returns:
            text_hidden: [B, T, hidden_size] - All hidden states
            text_cls: [B, hidden_size] - CLS token features (projected)
        """
        # Full text encoding (same as mm_encoder.py)
        text_hidden = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            extract_type="cls_patch",  # Get all tokens
        )  # [B, T, 768]

        # Extract CLS token and project (same as mm_encoder.py)
        text_cls = text_hidden[:, 0]  # [B, 768] - CLS token
        text_cls = self.xlmr_text_projection(text_cls)  # [B, 512]
        text_cls = self.text_projection_output(text_cls)  # [B, 768]

        return text_hidden, text_cls

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        use_moe: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pixel_values: [B, 3, 224, 224]
            input_ids: [B, T]
            attention_mask: [B, T]
            labels: [B, T] for LM tasks
            use_moe: If True, use MoE layers. If False, use original mm_encoder style.
        """
        batch_size = (
            pixel_values.shape[0] if pixel_values is not None else input_ids.shape[0]
        )
        device = pixel_values.device if pixel_values is not None else input_ids.device

        # ==================== Encode ====================
        if pixel_values is not None:
            visual_hidden, visual_cls = self.encode_vision(pixel_values)
        else:
            visual_hidden = None
            visual_cls = torch.zeros(
                batch_size, self.config.xlmr_text_dim, device=device
            )

        if input_ids is not None:
            text_hidden, text_cls = self.encode_text(input_ids, attention_mask)
        else:
            text_hidden = None
            text_cls = torch.zeros(batch_size, self.config.xlmr_text_dim, device=device)

        # ==================== MoE Processing ====================
        if use_moe and visual_hidden is not None:
            # Compress visual tokens
            visual_tokens = self.visual_projector(visual_hidden)  # [B, 64, 768]
            num_visual = visual_tokens.shape[1]

            # Use text CLS token expanded or all text tokens
            if text_hidden is not None:
                text_tokens = text_hidden  # [B, T, 768]
                num_text = text_tokens.shape[1]
            else:
                text_tokens = text_cls.unsqueeze(1)  # [B, 1, 768]
                num_text = 1

            # Add modality embeddings
            visual_modality = torch.zeros(
                batch_size, num_visual, dtype=torch.long, device=device
            )
            text_modality = torch.ones(
                batch_size, num_text, dtype=torch.long, device=device
            )

            visual_tokens = visual_tokens + self.modality_embedding(visual_modality)
            text_tokens = text_tokens + self.modality_embedding(text_modality)

            # Concatenate
            fused_tokens = torch.cat([visual_tokens, text_tokens], dim=1)
            modality_indices = torch.cat([visual_modality, text_modality], dim=1)

            # Create attention mask
            if attention_mask is not None:
                visual_attention = torch.ones(batch_size, num_visual, device=device)
                full_attention_mask = torch.cat(
                    [visual_attention, attention_mask.float()], dim=1
                )
            else:
                full_attention_mask = None

            # Apply MoE layers
            hidden_states = fused_tokens
            all_aux_losses = []

            for moe_layer in self.moe_layers:
                hidden_states = moe_layer(
                    hidden_states,
                    attention_mask=full_attention_mask,
                    modality_indices=modality_indices,
                )
                all_aux_losses.append(moe_layer.get_aux_losses())

            hidden_states = self.output_norm(hidden_states)

            # Extract features after MoE
            visual_features = hidden_states[:, :num_visual, :].mean(dim=1)
            text_features = hidden_states[:, num_visual:, :].mean(dim=1)

            # Project for contrastive
            if self.config.task_type == "contrastive":
                visual_features = self.visual_head(visual_features)
                text_features = self.text_head(text_features)

            # Aggregate aux losses
            load_balance_loss = sum(
                aux["load_balance_loss"] for aux in all_aux_losses
            ) / len(all_aux_losses)
            z_loss = sum(aux["z_loss"] for aux in all_aux_losses) / len(all_aux_losses)
            mi_loss = sum(aux["mi_loss"] for aux in all_aux_losses) / len(
                all_aux_losses
            )

        else:
            # Original mm_encoder.py style (no MoE)
            visual_features = visual_cls
            text_features = text_cls
            hidden_states = None
            load_balance_loss = torch.tensor(0.0, device=device)
            z_loss = torch.tensor(0.0, device=device)
            mi_loss = torch.tensor(0.0, device=device)

        # ==================== Normalize and Compute Loss ====================
        visual_features = F.normalize(visual_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Contrastive loss
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * visual_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        contrastive_labels = torch.arange(batch_size, device=device)
        task_loss = (
            F.cross_entropy(logits_per_image, contrastive_labels)
            + F.cross_entropy(logits_per_text, contrastive_labels)
        ) / 2

        total_loss = task_loss + load_balance_loss + z_loss

        if return_dict:
            return {
                "total_loss": total_loss,
                "task_loss": task_loss,
                "load_balance_loss": load_balance_loss,
                "z_loss": z_loss,
                "mi_loss": mi_loss,
                "logits": logits_per_image,
                "visual_features": visual_features,
                "text_features": text_features,
                "hidden_states": hidden_states,
                "logit_scale": logit_scale,
            }
        else:
            return total_loss, logits_per_image

    def forward_original(
        self,
        text_input_ids: torch.Tensor,
        image_features: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Original mm_encoder.py style forward (for compatibility).
        No MoE, just the trained encoders + projections.
        """
        return self.forward(
            pixel_values=image_features,
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            use_moe=False,
        )


def print_architecture():
    """Print architecture diagram"""
    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        VL-MoE ARCHITECTURE                                   ║
║                  (Extends trained mm_encoder.py)                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FROM PRETRAINED (mm_encoder.py):                                            ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │ text_encoder (XLM-RoBERTa)      - Full encoder, trained                 │ ║
║  │ vision_text_model (CLIP)        - Frozen, serves as backbone            │ ║
║  │ xlmr_text_projection (768→512)  - Trained projection                    │ ║
║  │ text_projection_output (512→768) - Trained projection                   │ ║
║  │ vision_projection_output (768→768) - Trained projection                 │ ║
║  │ logit_scale, logit_bias         - Trained temperature                   │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  NEW FOR MOE:                                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │ visual_projector (Resampler)    - Compress 50→64 tokens                 │ ║
║  │ modality_embedding              - Distinguish visual/text               │ ║
║  │ moe_layers (×N)                 - DeepSeek MoE layers                   │ ║
║  │ output heads                    - Task-specific                         │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Testing VL-MoE Model")
    print("=" * 70)

    print_architecture()

    config = VLMoEConfig(
        vision_model_name="openai/clip-vit-base-patch32",
        text_model_name="FacebookAI/xlm-roberta-base",
        num_visual_tokens=64,
        num_shared_experts=2,
        num_routed_experts=8,
        num_moe_layers=2,
    )

    print("\nConfiguration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")

    print("\nCreating model...")

    try:
        model = VLMoEModel(config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        print("\nTesting forward pass...")
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 32))
        attention_mask = torch.ones(batch_size, 32)

        model.eval()
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        print("\nOutputs:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")

        print("\n✓ VL-MoE Model test passed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
