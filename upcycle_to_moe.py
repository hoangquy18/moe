#!/usr/bin/env python
# coding=utf-8
"""
Upcycle Dense Model to VL-MoE

This script converts your trained mm_encoder model to VL-MoE by:
1. Loading ALL trained weights (text_encoder, projections, logit_scale)
2. Creating VL-MoE model with same encoder structure
3. Initializing MoE experts from projection weights (sparse upcycling)
4. Saving the upcycled model

Usage:
======
python upcycle_to_moe.py \
    --dense_checkpoint checkpoint_epoch_9.pt \
    --output_dir outputs/vlmoe \
    --num_routed_experts 64 \
    --num_shared_experts 2 \
    --num_moe_layers 6

What gets transferred:
======================
FROM CHECKPOINT (trained):
- text_encoder: Full XLM-RoBERTa (all layers, trained on teacher + contrastive)
- xlmr_text_projection: 768 → 512 (trained)
- text_projection_output: 512 → 768 (trained)
- vision_projection_output: 768 → 768 (trained)
- logit_scale, logit_bias: Temperature (trained)

INITIALIZED VIA UPCYCLING:
- MoE shared experts: Exact copy of projection weights
- MoE routed experts: Projection weights + noise

NEW (random init):
- visual_projector: Resampler for token compression
- modality_embedding: Visual/text distinction
- self_attn layers in MoE transformer
- output heads
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.vlmoe.vlmoe_model import VLMoEModel, VLMoEConfig
from model.vlmoe.upcycling import upcycle_vlmoe_from_dense


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upcycle trained dense model to VL-MoE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required
    parser.add_argument(
        "--dense_checkpoint",
        type=str,
        required=True,
        help="Path to trained dense model checkpoint (e.g., checkpoint_epoch_9.pt)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/vlmoe",
        help="Directory to save upcycled model",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="vlmoe_upcycled.pt",
        help="Name of output checkpoint file",
    )

    # Model architecture (should match dense model)
    parser.add_argument(
        "--vision_model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP vision model (same as dense training)",
    )
    parser.add_argument(
        "--text_model_name",
        type=str,
        default="FacebookAI/xlm-roberta-base",
        help="Text model (same as dense training)",
    )

    # MoE configuration
    parser.add_argument(
        "--num_routed_experts",
        type=int,
        default=64,
        help="Number of routed experts per MoE layer",
    )
    parser.add_argument(
        "--num_shared_experts",
        type=int,
        default=2,
        help="Number of shared experts per MoE layer",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=2,
        help="Top-k experts per token",
    )
    parser.add_argument(
        "--num_moe_layers",
        type=int,
        default=6,
        help="Number of MoE transformer layers",
    )
    parser.add_argument(
        "--moe_intermediate_size",
        type=int,
        default=3072,
        help="Intermediate size for MoE FFN",
    )

    # Visual projector
    parser.add_argument(
        "--num_visual_tokens",
        type=int,
        default=64,
        help="Number of compressed visual tokens",
    )
    parser.add_argument(
        "--projector_num_layers",
        type=int,
        default=2,
        help="Number of cross-attention layers in Resampler",
    )

    # Upcycling parameters
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.01,
        help="Noise std for routed experts (default: 1% of weight std)",
    )

    # Training settings
    parser.add_argument(
        "--freeze_vision",
        action="store_true",
        help="Freeze vision encoder",
    )
    parser.add_argument(
        "--freeze_text",
        action="store_true",
        help="Freeze text encoder",
    )

    # Other
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for upcycling",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("DENSE → VL-MoE UPCYCLING")
    print("=" * 70)

    # Check checkpoint exists
    if not os.path.exists(args.dense_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.dense_checkpoint}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create VL-MoE config
    print("\n[1/4] Creating VL-MoE configuration...")
    config = VLMoEConfig(
        # Vision (same as dense)
        vision_model_name=args.vision_model_name,
        freeze_vision_encoder=args.freeze_vision,
        # Text (same as dense)
        text_model_name=args.text_model_name,
        freeze_text_encoder=args.freeze_text,
        # Visual projector
        num_visual_tokens=args.num_visual_tokens,
        projector_num_layers=args.projector_num_layers,
        # MoE
        num_shared_experts=args.num_shared_experts,
        num_routed_experts=args.num_routed_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        num_moe_layers=args.num_moe_layers,
        moe_intermediate_size=args.moe_intermediate_size,
    )

    print("\nConfiguration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")

    # Create VL-MoE model
    print("\n[2/4] Creating VL-MoE model...")
    model = VLMoEModel(config)

    # Print model info before upcycling
    total_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters before upcycling: {total_before:,}")

    # Upcycle from dense checkpoint
    print("\n[3/4] Upcycling from dense checkpoint...")
    model = upcycle_vlmoe_from_dense(
        model,
        args.dense_checkpoint,
        noise_std=args.noise_std,
        device=args.device,
    )

    # Save upcycled model
    print("\n[4/4] Saving upcycled model...")
    output_path = os.path.join(args.output_dir, args.output_name)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "upcycling_info": {
            "dense_checkpoint": args.dense_checkpoint,
            "noise_std": args.noise_std,
        },
    }

    torch.save(checkpoint, output_path)
    print(f"  Saved to: {output_path}")

    # Print final summary
    total_after = sum(p.numel() for p in model.parameters())
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 70)
    print("UPCYCLING COMPLETE!")
    print("=" * 70)
    print(f"  Dense model → VL-MoE")
    print(f"  Parameters: {total_before:,} → {total_after:,}")
    print(f"  Trainable: {trainable_before:,} → {trainable_after:,}")
    print(f"  Output: {output_path}")
    print()
    print("What was loaded from checkpoint:")
    print("  ✓ text_encoder (XLM-RoBERTa)")
    print("  ✓ xlmr_text_projection (768 → 512)")
    print("  ✓ text_projection_output (512 → 768)")
    print("  ✓ vision_projection_output (768 → 768)")
    print("  ✓ logit_scale, logit_bias")
    print()
    print("MoE experts initialized from projections:")
    print(f"  ✓ {args.num_shared_experts} shared experts (exact copy)")
    print(f"  ✓ {args.num_routed_experts} routed experts (copy + noise)")
    print()
    print("New layers (random init, need training):")
    print("  - visual_projector (Resampler)")
    print("  - modality_embedding")
    print("  - self_attn layers")
    print("  - output heads")
    print("=" * 70)

    # Quick test
    print("\nQuick forward pass test...")
    model.eval()
    model.to(args.device)

    with torch.no_grad():
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224).to(args.device)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 32)).to(args.device)
        attention_mask = torch.ones(batch_size, 32).to(args.device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        print(f"  ✓ Forward pass successful!")
        print(f"  Total loss: {outputs['total_loss'].item():.4f}")
        print(f"  Task loss: {outputs['task_loss'].item():.4f}")


if __name__ == "__main__":
    main()
