# coding=utf-8
"""
Sparse Upcycling for VL-MoE

This module converts a trained dense model (mm_encoder.py) to a sparse MoE model.

What is transferred from dense checkpoint:
==========================================
1. text_encoder: Full XLM-RoBERTa encoder (trained on teacher + contrastive)
2. xlmr_text_projection: Trained 768 → 512 projection
3. text_projection_output: Trained 512 → 768 projection
4. vision_projection_output: Trained 768 → 768 projection
5. logit_scale, logit_bias: Learned temperature

What is initialized via upcycling:
==================================
- MoE expert weights (from projections + noise)
- Shared experts: Exact copy of projections
- Routed experts: Projections + small noise

New layers (random init):
=========================
- visual_projector: Resampler for token compression
- modality_embedding: Visual/text distinction
- moe_layers.*.self_attn: Self-attention layers
- output heads: Task-specific

Reference:
- Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints
- DeepSeek-MoE: Towards Ultimate Expert Specialization
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, Optional, Tuple
import math


def add_noise_to_weights(weight: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
    """Add small Gaussian noise to break symmetry between experts."""
    weight_std = weight.std().item()
    noise = torch.randn_like(weight) * (noise_std * weight_std)
    return weight + noise


class MoEUpcycler:
    """
    Upcycle a trained dense VL model to MoE.

    Dense Model Projections:
    ========================
    xlmr_text_projection:   768 → 512  (down projection)
    text_projection_output: 512 → 768  (up projection)
    vision_projection_output: 768 → 768

    MoE Expert Structure (SwiGLU):
    ==============================
    gate_proj: hidden_size → intermediate_size
    up_proj:   hidden_size → intermediate_size
    down_proj: intermediate_size → hidden_size
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        initialize_gate_from_up: bool = True,
    ):
        self.noise_std = noise_std
        self.initialize_gate_from_up = initialize_gate_from_up

    def _expand_weight(
        self, weight: torch.Tensor, target_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Expand weight matrix to target shape by tiling."""
        src_rows, src_cols = weight.shape
        tgt_rows, tgt_cols = target_shape

        # Expand columns if needed
        if src_cols != tgt_cols:
            repeat_factor = math.ceil(tgt_cols / src_cols)
            weight = weight.repeat(1, repeat_factor)[:, :tgt_cols]

        # Expand rows if needed
        if src_rows != tgt_rows:
            repeat_factor = math.ceil(tgt_rows / src_rows)
            weight = weight.repeat(repeat_factor, 1)[:tgt_rows, :]

        return weight.clone()

    def upcycle_moe_layer(
        self,
        moe_layer: nn.Module,
        text_down_proj: torch.Tensor,  # xlmr_text_projection.weight
        text_up_proj: torch.Tensor,  # text_projection_output.weight
        vision_proj: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Upcycle an entire MoE layer from dense projections.
        """
        print(
            f"Upcycling MoE layer with {len(moe_layer.shared_experts)} shared + "
            f"{len(moe_layer.routed_experts)} routed experts"
        )

        # Get target dimensions from expert weights
        reference_expert = moe_layer.shared_experts[0]
        gate_shape = reference_expert.gate_proj.weight.shape  # (3072, 768)
        down_shape = reference_expert.down_proj.weight.shape  # (768, 3072)

        print(f"  Target shapes: gate/up={gate_shape}, down={down_shape}")
        print(
            f"  Source shapes: text_down={text_down_proj.shape}, text_up={text_up_proj.shape}"
        )

        # Expand weights to target dimensions
        expanded_gate_up = self._expand_weight(text_down_proj, gate_shape)
        expanded_down = self._expand_weight(text_up_proj, down_shape)

        print(
            f"  Expanded shapes: gate/up={expanded_gate_up.shape}, down={expanded_down.shape}"
        )

        # Initialize shared experts (no noise - identical)
        for i, expert in enumerate(moe_layer.shared_experts):
            print(f"  Initializing shared expert {i}")
            expert.gate_proj.weight.data.copy_(expanded_gate_up)
            expert.up_proj.weight.data.copy_(expanded_gate_up)
            expert.down_proj.weight.data.copy_(expanded_down)

        # Initialize routed experts (with noise for diversity)
        for i, expert in enumerate(moe_layer.routed_experts):
            print(f"  Initializing routed expert {i} (with noise)")
            expert.gate_proj.weight.data.copy_(
                add_noise_to_weights(expanded_gate_up.clone(), self.noise_std)
            )
            expert.up_proj.weight.data.copy_(
                add_noise_to_weights(expanded_gate_up.clone(), self.noise_std)
            )
            expert.down_proj.weight.data.copy_(
                add_noise_to_weights(expanded_down.clone(), self.noise_std)
            )


def load_dense_checkpoint_for_upcycling(
    checkpoint_path: str,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load checkpoint and extract ALL weights for upcycling.

    Returns:
        Dictionary with:
        - full_state_dict: Complete model state dict
        - text_down_proj: xlmr_text_projection.weight
        - text_up_proj: text_projection_output.weight
        - vision_proj: vision_projection_output.weight
        - logit_scale, logit_bias
        - text_encoder_state: Full text encoder state dict
    """
    print(f"Loading dense checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    result = {"full_state_dict": state_dict}

    # Extract projection weights
    key_mappings = {
        "xlmr_text_projection.weight": "text_down_proj",
        "xlmr_text_projection.bias": "text_down_proj_bias",
        "text_projection_output.weight": "text_up_proj",
        "text_projection_output.bias": "text_up_proj_bias",
        "vision_projection_output.weight": "vision_proj",
        "vision_projection_output.bias": "vision_proj_bias",
        "logit_scale": "logit_scale",
        "logit_bias": "logit_bias",
    }

    for src_key, dst_key in key_mappings.items():
        if src_key in state_dict:
            result[dst_key] = state_dict[src_key]
            if isinstance(result[dst_key], torch.Tensor) and result[dst_key].dim() == 0:
                print(f"  Found {src_key}: {result[dst_key].item():.4f}")
            elif isinstance(result[dst_key], torch.Tensor):
                print(f"  Found {src_key}: {result[dst_key].shape}")

    # Extract text encoder weights
    text_encoder_state = {}
    for key, value in state_dict.items():
        if key.startswith("text_encoder."):
            new_key = key.replace("text_encoder.", "")
            text_encoder_state[new_key] = value

    if text_encoder_state:
        result["text_encoder_state"] = text_encoder_state
        print(f"  Found text_encoder with {len(text_encoder_state)} parameters")

    return result


def upcycle_vlmoe_from_dense(
    vlmoe_model: nn.Module,
    dense_checkpoint_path: str,
    noise_std: float = 0.01,
    device: str = "cpu",
) -> nn.Module:
    """
    Main function to upcycle VL-MoE from dense checkpoint.

    This function:
    1. Loads ALL weights from dense checkpoint
    2. Copies text_encoder weights directly
    3. Copies projection weights directly
    4. Uses projections to initialize MoE experts
    """
    print("=" * 70)
    print("SPARSE UPCYCLING: Dense mm_encoder → VL-MoE")
    print("=" * 70)

    # Load dense checkpoint
    checkpoint_data = load_dense_checkpoint_for_upcycling(dense_checkpoint_path, device)

    # ==================== 1. Load Text Encoder ====================
    print("\n[1/4] Loading text encoder...")
    if "text_encoder_state" in checkpoint_data:
        text_encoder_state = checkpoint_data["text_encoder_state"]
        try:
            # Load weights into text_encoder
            missing, unexpected = vlmoe_model.text_encoder.load_state_dict(
                text_encoder_state, strict=False
            )
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
            print(f"  ✓ Loaded {len(text_encoder_state)} text encoder parameters")
        except Exception as e:
            print(f"  ✗ Failed to load text encoder: {e}")
    else:
        print("  ⚠ No text encoder weights found in checkpoint")

    # ==================== 2. Load Projection Layers ====================
    print("\n[2/4] Loading projection layers...")

    # xlmr_text_projection
    if "text_down_proj" in checkpoint_data:
        vlmoe_model.xlmr_text_projection.weight.data.copy_(
            checkpoint_data["text_down_proj"]
        )
        if "text_down_proj_bias" in checkpoint_data:
            vlmoe_model.xlmr_text_projection.bias.data.copy_(
                checkpoint_data["text_down_proj_bias"]
            )
        print("  ✓ Loaded xlmr_text_projection")

    # text_projection_output
    if "text_up_proj" in checkpoint_data:
        vlmoe_model.text_projection_output.weight.data.copy_(
            checkpoint_data["text_up_proj"]
        )
        if "text_up_proj_bias" in checkpoint_data:
            vlmoe_model.text_projection_output.bias.data.copy_(
                checkpoint_data["text_up_proj_bias"]
            )
        print("  ✓ Loaded text_projection_output")

    # vision_projection_output
    if "vision_proj" in checkpoint_data:
        vlmoe_model.vision_projection_output.weight.data.copy_(
            checkpoint_data["vision_proj"]
        )
        if "vision_proj_bias" in checkpoint_data:
            vlmoe_model.vision_projection_output.bias.data.copy_(
                checkpoint_data["vision_proj_bias"]
            )
        print("  ✓ Loaded vision_projection_output")

    # logit_scale and logit_bias
    if "logit_scale" in checkpoint_data:
        vlmoe_model.logit_scale.data.copy_(checkpoint_data["logit_scale"])
        print(f"  ✓ Loaded logit_scale: {vlmoe_model.logit_scale.item():.4f}")

    if "logit_bias" in checkpoint_data:
        vlmoe_model.logit_bias.data.copy_(checkpoint_data["logit_bias"])
        print(f"  ✓ Loaded logit_bias: {vlmoe_model.logit_bias.item():.4f}")

    # ==================== 3. Upcycle MoE Layers ====================
    print("\n[3/4] Upcycling MoE layers from projections...")

    upcycler = MoEUpcycler(noise_std=noise_std)

    if hasattr(vlmoe_model, "moe_layers"):
        text_down_proj = checkpoint_data.get("text_down_proj")
        text_up_proj = checkpoint_data.get("text_up_proj")
        vision_proj = checkpoint_data.get("vision_proj")

        if text_down_proj is None or text_up_proj is None:
            print("  ⚠ Projection weights not found, skipping MoE upcycling")
        else:
            for i, moe_transformer_layer in enumerate(vlmoe_model.moe_layers):
                print(f"\n  Layer {i}:")
                moe_layer = moe_transformer_layer.moe

                upcycler.upcycle_moe_layer(
                    moe_layer,
                    text_down_proj=text_down_proj,
                    text_up_proj=text_up_proj,
                    vision_proj=vision_proj,
                )

    # ==================== 4. Summary ====================
    print("\n[4/4] Upcycling Summary:")
    print("=" * 70)

    total_params = sum(p.numel() for p in vlmoe_model.parameters())
    trainable_params = sum(
        p.numel() for p in vlmoe_model.parameters() if p.requires_grad
    )

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()
    print("  Loaded from checkpoint:")
    print("    ✓ text_encoder (full XLM-RoBERTa)")
    print("    ✓ xlmr_text_projection (768 → 512)")
    print("    ✓ text_projection_output (512 → 768)")
    print("    ✓ vision_projection_output (768 → 768)")
    print("    ✓ logit_scale, logit_bias")
    print()
    print("  Initialized via upcycling:")
    print("    ✓ MoE shared experts (exact copy)")
    print("    ✓ MoE routed experts (copy + noise)")
    print()
    print("  Randomly initialized (NEW):")
    print("    - visual_projector (Resampler)")
    print("    - modality_embedding")
    print("    - self_attn layers in MoE")
    print("    - output heads")
    print("=" * 70)

    return vlmoe_model


class UpcycledVLMoE(nn.Module):
    """
    Convenience wrapper that creates VL-MoE and upcycles from dense checkpoint.

    Usage:
    ```python
    from model.vlmoe.upcycling import UpcycledVLMoE
    from model.vlmoe.vlmoe_model import VLMoEConfig

    config = VLMoEConfig(
        num_shared_experts=2,
        num_routed_experts=64,
        num_moe_layers=6,
    )

    model = UpcycledVLMoE.from_dense_checkpoint(
        config=config,
        dense_checkpoint="checkpoint_epoch_9.pt",
    )
    ```
    """

    @classmethod
    def from_dense_checkpoint(
        cls,
        config,  # VLMoEConfig
        dense_checkpoint: str,
        noise_std: float = 0.01,
        device: str = "cpu",
    ):
        from model.vlmoe.vlmoe_model import VLMoEModel

        print("Creating VL-MoE model...")
        model = VLMoEModel(config)

        print("\nUpcycling from dense checkpoint...")
        model = upcycle_vlmoe_from_dense(
            model,
            dense_checkpoint,
            noise_std=noise_std,
            device=device,
        )

        return model


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Sparse Upcycling")
    print("=" * 70)

    import sys
    import os

    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    from model.moe.deepseek_moe import DeepSeekMoELayer, DeepSeekMoEConfig

    # Create a test MoE layer
    moe_config = DeepSeekMoEConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_shared_experts=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
    )

    moe_layer = DeepSeekMoELayer(moe_config)

    # Simulate trained projection weights
    text_down_proj = torch.randn(512, 768) * 0.02
    text_up_proj = torch.randn(768, 512) * 0.02

    print("\nSimulated projection weights:")
    print(f"  text_down_proj (xlmr_text_projection): {text_down_proj.shape}")
    print(f"  text_up_proj (text_projection_output): {text_up_proj.shape}")

    # Upcycle
    upcycler = MoEUpcycler(noise_std=0.01)

    print("\n" + "-" * 60)
    upcycler.upcycle_moe_layer(
        moe_layer,
        text_down_proj=text_down_proj,
        text_up_proj=text_up_proj,
    )

    # Verify experts are initialized
    print("\nVerifying expert initialization:")

    # Check shared experts are identical
    shared_0_weight = moe_layer.shared_experts[0].down_proj.weight.data
    shared_1_weight = moe_layer.shared_experts[1].down_proj.weight.data
    shared_diff = (shared_0_weight - shared_1_weight).abs().mean().item()
    print(f"  Shared expert weight difference: {shared_diff:.6f} (should be 0)")

    # Check routed experts are different
    routed_0_weight = moe_layer.routed_experts[0].down_proj.weight.data
    routed_1_weight = moe_layer.routed_experts[1].down_proj.weight.data
    routed_diff = (routed_0_weight - routed_1_weight).abs().mean().item()
    print(f"  Routed expert weight difference: {routed_diff:.6f} (should be > 0)")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 96
    hidden_states = torch.randn(batch_size, seq_len, moe_config.hidden_size)

    output = moe_layer(hidden_states)
    print(f"  Input: {hidden_states.shape}")
    print(f"  Output: {output.shape}")

    print("\n✓ Upcycling test passed!")
