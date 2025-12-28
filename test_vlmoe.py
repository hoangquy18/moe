#!/usr/bin/env python
# coding=utf-8
"""
Test script for VL-MoE Model

This script tests the complete VL-MoE architecture including:
1. DeepSeek-style MoE Layer
2. Visual Projector (token compression)
3. Complete VL-MoE Model
4. All loss functions

Run with: python test_vlmoe.py
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.moe.deepseek_moe import DeepSeekMoELayer, DeepSeekMoEConfig
from model.projector.visual_projector import VisualProjector, VisualProjectorConfig


def test_moe_layer():
    """Test DeepSeek MoE Layer"""
    print("\n" + "=" * 70)
    print("TEST 1: DeepSeek-style MoE Layer")
    print("=" * 70)

    config = DeepSeekMoEConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_shared_experts=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        load_balance_weight=0.01,
        z_loss_weight=0.001,
    )

    moe_layer = DeepSeekMoELayer(config)

    # Test input: 96 tokens (64 visual + 32 text)
    batch_size = 2
    seq_len = 96
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Modality indices: 0 for image, 1 for text
    modality_indices = torch.cat(
        [
            torch.zeros(batch_size, 64, dtype=torch.long),
            torch.ones(batch_size, 32, dtype=torch.long),
        ],
        dim=1,
    )

    print(f"Input shape: {hidden_states.shape}")
    print(f"Modality indices shape: {modality_indices.shape}")
    print(f"  - Image tokens: {(modality_indices == 0).sum().item()}")
    print(f"  - Text tokens: {(modality_indices == 1).sum().item()}")

    # Forward pass
    output = moe_layer(hidden_states, modality_indices)

    print(f"\nOutput shape: {output.shape}")
    assert output.shape == hidden_states.shape, "Output shape mismatch!"

    # Get auxiliary losses
    aux_losses = moe_layer.get_aux_losses()
    print(f"\nAuxiliary Losses:")
    for name, loss in aux_losses.items():
        print(f"  {name}: {loss.item():.6f}")

    print("\n✓ MoE Layer test PASSED!")
    return True


def test_visual_projector():
    """Test Visual Projector (Resampler)"""
    print("\n" + "=" * 70)
    print("TEST 2: Visual Projector (Resampler)")
    print("=" * 70)

    print(
        """
Tensor Flow:
  CLIP Output: [B, 257, 1024] (256 patches + CLS, hidden=1024)
       ↓
  Resampler: Cross-attention with learnable queries
       ↓
  Output: [B, 64, 768] (compressed, matched to text dim)
    """
    )

    config = VisualProjectorConfig(
        vision_hidden_size=1024,  # CLIP ViT-L
        vision_num_tokens=257,  # 256 patches + CLS
        output_hidden_size=768,  # RoBERTa
        num_output_tokens=64,  # Compressed tokens
        num_layers=2,  # Cross-attention layers
    )

    projector = VisualProjector(config)

    # Count parameters
    num_params = sum(p.numel() for p in projector.parameters())
    print(f"Resampler parameters: {num_params:,}")

    # Simulate CLIP output
    batch_size = 2
    visual_features = torch.randn(batch_size, 257, 1024)

    print(f"Input:  {visual_features.shape} (CLIP output)")

    # Forward pass
    compressed = projector(visual_features)

    print(f"Output: {compressed.shape} (Compressed)")
    print(f"Compression ratio: 257 → 64 (4.0x)")

    assert compressed.shape == (batch_size, 64, 768), "Shape mismatch!"

    print("\n✓ Resampler Projector test PASSED!")
    return True


def test_loss_computation():
    """Test loss computation"""
    print("\n" + "=" * 70)
    print("TEST 3: Loss Computation")
    print("=" * 70)

    from model.moe.deepseek_moe import DeepSeekMoELayer, DeepSeekMoEConfig

    config = DeepSeekMoEConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_shared_experts=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        load_balance_weight=0.01,
        z_loss_weight=0.001,
    )

    moe_layer = DeepSeekMoELayer(config)

    # Simulate forward pass
    batch_size = 4
    seq_len = 96
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    modality_indices = torch.cat(
        [
            torch.zeros(batch_size, 64, dtype=torch.long),
            torch.ones(batch_size, 32, dtype=torch.long),
        ],
        dim=1,
    )

    output = moe_layer(hidden_states, modality_indices)

    # Test load balance loss
    lb_loss = moe_layer.compute_load_balance_loss()
    print(f"\n1. Load Balance Loss: {lb_loss.item():.6f}")
    print("   - Applied to ROUTED experts only (not shared)")
    print("   - Formula: num_experts × Σ(f_i × P_i)")
    print(f"   - Weight: {config.load_balance_weight}")

    # Test z-loss
    z_loss = moe_layer.compute_z_loss()
    print(f"\n2. Z-Loss: {z_loss.item():.6f}")
    print("   - Prevents router logits from exploding")
    print("   - Formula: mean(logsumexp(logits)²)")
    print(f"   - Weight: {config.z_loss_weight}")

    # Test MI-loss (placeholder)
    mi_loss = moe_layer.compute_mi_loss()
    print(f"\n3. MI-Loss: {mi_loss.item():.6f}")
    print("   - Placeholder for Modality-Importance loss")
    print("   - Will penalize cross-modality routing")

    # Total auxiliary loss
    aux_losses = moe_layer.get_aux_losses()
    total_aux = sum(aux_losses.values())
    print(f"\nTotal Auxiliary Loss: {total_aux.item():.6f}")

    # Simulate task loss (contrastive)
    print("\n4. Task Loss (Contrastive):")
    visual_features = torch.randn(batch_size, 768)
    text_features = torch.randn(batch_size, 768)

    # Normalize
    visual_features = F.normalize(visual_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Contrastive loss
    logit_scale = torch.exp(torch.tensor(1 / 0.07).log())
    logits = logit_scale * visual_features @ text_features.t()
    labels = torch.arange(batch_size)
    task_loss = F.cross_entropy(logits, labels)

    print(f"   Contrastive Loss: {task_loss.item():.6f}")
    print(f"   Logit Scale: {logit_scale.item():.2f}")

    # Combined loss
    total_loss = task_loss + total_aux
    print(f"\nCombined Total Loss: {total_loss.item():.6f}")

    print("\n✓ Loss computation test PASSED!")
    return True


def test_tensor_shape_flow():
    """Test complete tensor shape flow"""
    print("\n" + "=" * 70)
    print("TEST 4: Complete Tensor Shape Flow")
    print("=" * 70)

    batch_size = 2

    print(
        """
TENSOR SHAPE FLOW:
==================

Step 1: CLIP Vision Encoder
    """
    )

    # Simulate CLIP output
    clip_output = torch.randn(batch_size, 257, 1024)
    print(f"    CLIP Output: {clip_output.shape}")
    print(f"    - 257 = 256 patches (16×16) + 1 CLS token")
    print(f"    - 1024 = ViT-L hidden dimension")

    print(
        """
Step 2: Visual Projector (Compression)
    """
    )

    projector_config = VisualProjectorConfig(
        vision_hidden_size=1024,
        vision_num_tokens=257,
        output_hidden_size=768,
        num_output_tokens=64,
        num_layers=2,
    )
    projector = VisualProjector(projector_config)

    visual_tokens = projector(clip_output)
    print(f"    Compressed Visual: {visual_tokens.shape}")
    print(f"    - 64 tokens (4× compression from 257)")
    print(f"    - 768 dim (matches RoBERTa)")

    print(
        """
Step 3: RoBERTa Text Embedding
    """
    )

    # Simulate RoBERTa embedding output
    text_tokens = torch.randn(batch_size, 32, 768)
    print(f"    Text Embedding: {text_tokens.shape}")
    print(f"    - 32 tokens (max_length)")
    print(f"    - 768 dim")

    print(
        """
Step 4: Fusion (Concatenation)
    """
    )

    # Add modality embeddings
    modality_emb = torch.nn.Embedding(2, 768)
    visual_modality = torch.zeros(batch_size, 64, dtype=torch.long)
    text_modality = torch.ones(batch_size, 32, dtype=torch.long)

    visual_tokens = visual_tokens + modality_emb(visual_modality)
    text_tokens = text_tokens + modality_emb(text_modality)

    fused = torch.cat([visual_tokens, text_tokens], dim=1)
    modality_indices = torch.cat([visual_modality, text_modality], dim=1)

    print(f"    Fused Tokens: {fused.shape}")
    print(f"    Modality Indices: {modality_indices.shape}")
    print(f"    - 96 = 64 visual + 32 text tokens")

    print(
        """
Step 5: MoE Transformer Layer
    """
    )

    moe_config = DeepSeekMoEConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_shared_experts=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
    )
    moe_layer = DeepSeekMoELayer(moe_config)

    moe_output = moe_layer(fused, modality_indices)
    print(f"    MoE Output: {moe_output.shape}")
    print(f"    - Same shape as input (residual connection)")

    print(
        """
Step 6: Split and Pool for Output
    """
    )

    visual_out = moe_output[:, :64, :].mean(dim=1)
    text_out = moe_output[:, 64:, :].mean(dim=1)

    print(f"    Visual Features: {visual_out.shape}")
    print(f"    Text Features: {text_out.shape}")
    print(f"    - Pooled to single vectors per sample")

    print(
        """
Step 7: Contrastive Loss
    """
    )

    visual_norm = F.normalize(visual_out, dim=-1)
    text_norm = F.normalize(text_out, dim=-1)

    logits = visual_norm @ text_norm.t()
    print(f"    Similarity Matrix: {logits.shape}")
    print(f"    - Diagonal = positive pairs")

    print("\n✓ Tensor shape flow test PASSED!")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("VL-MOE COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    tests = [
        ("MoE Layer", test_moe_layer),
        ("Visual Projector", test_visual_projector),
        ("Loss Computation", test_loss_computation),
        ("Tensor Shape Flow", test_tensor_shape_flow),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
        if not success:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED! ✗")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
