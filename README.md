# 🇻🇳 Vietnamese Vision-Language MoE

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.40+-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

A **Vietnamese Vision-Language model** using Mixture of Experts (MoE) architecture, featuring:

- 🎯 **DeepSeek-style MoE**: Shared + Routed Experts with Top-k routing
- 🔄 **Full Encoder Reuse**: Load trained text_encoder & projections from checkpoint
- 🖼️ **Token Compression**: Resampler to balance visual (50) vs text (32) tokens
- ⚡ **Sparse Upcycling**: Initialize MoE experts from trained projection weights
- 🇻🇳 **Vietnamese Support**: XLM-RoBERTa for Vietnamese text

## 🚀 Quick Start

```bash
# Install dependencies
conda create -n moe python=3.11
conda activate moe
pip install torch torchvision transformers datasets tqdm

# Train Stage 1: Text Alignment (Teacher Learning)
python train_contrastive.py --training_stage teacher --num_epochs 15

# Train Stage 2: Vision-Language (Contrastive Learning)
python train_contrastive.py --training_stage contrastive --stage1_checkpoint checkpoint.pt

# Stage 3: Upcycle to MoE (loads ALL trained weights + init MoE from projections)
python upcycle_to_moe.py \
    --dense_checkpoint checkpoint.pt \
    --output_dir outputs/vlmoe \
    --num_routed_experts 64 \
    --num_shared_experts 2

# Stage 4: Train VL-MoE (contrastive + MoE losses)
python train_vlmoe.py \
    --upcycled_checkpoint outputs/vlmoe/vlmoe_upcycled.pt \
    --num_epochs 10 \
    --batch_size 512 \
    --learning_rate 2e-5
```

## 📁 Project Structure

```
moe/
├── model/
│   ├── vlmoe/           # VL-MoE model
│   │   ├── vlmoe_model.py   # Main model (extends mm_encoder)
│   │   └── upcycling.py     # Sparse upcycling utilities
│   ├── moe/             # MoE layers
│   │   └── deepseek_moe.py  # DeepSeek-style MoE
│   ├── projector/       # Visual token compression
│   │   └── visual_projector.py  # Resampler
│   ├── mm/              # Dense multimodal encoder
│   │   └── mm_encoder.py    # Original trained model
│   ├── text/            # Text encoder
│   │   └── text_encoder.py  # XLM-RoBERTa
│   └── vision/          # Vision encoder
├── losses/              # Loss functions
├── data/                # Data loaders
├── train_contrastive.py # Training script (Stage 1 & 2)
├── upcycle_to_moe.py    # Upcycling script (Stage 3)
├── train_vlmoe.py       # VL-MoE training script (Stage 4)
└── README_VLMOE.md      # Detailed documentation
```

## 📖 Documentation

See [README_VLMOE.md](README_VLMOE.md) for detailed documentation including:

- Architecture diagrams
- Training pipeline
- Configuration options
- Tensor shape flow
- API reference
- Sparse upcycling details

## 🏗️ Architecture Overview

```
╔══════════════════════════════════════════════════════════════════╗
║  FROM CHECKPOINT (trained):                                      ║
║  ├── text_encoder (XLM-RoBERTa)      ✓ Full encoder             ║
║  ├── vision_text_model (CLIP)        ✓ Frozen backbone          ║
║  ├── xlmr_text_projection (768→512)  ✓ Trained                  ║
║  ├── text_projection_output (512→768) ✓ Trained                 ║
║  ├── vision_projection_output        ✓ Trained                  ║
║  └── logit_scale, logit_bias         ✓ Trained                  ║
╠══════════════════════════════════════════════════════════════════╣
║  NEW FOR MOE:                                                    ║
║  ├── visual_projector (Resampler)    - Token compression        ║
║  ├── modality_embedding              - Visual/text distinction  ║
║  ├── moe_layers (×N)                 - DeepSeek MoE             ║
║  │   ├── shared_experts (×2)         - From projections         ║
║  │   └── routed_experts (×64)        - From projections + noise ║
║  └── output heads                    - Task-specific            ║
╚══════════════════════════════════════════════════════════════════╝
```

**Data Flow:**
```
Image [B,3,224,224] → CLIP → [B,50,768] → Resampler → [B,64,768] ─┐
                                                                  ├→ MoE Layers → Output
Text  [B,32]        → XLM-RoBERTa       → [B,32,768] ────────────┘
```

**MoE Layer Structure:**
- **Shared Experts (×2)**: Always active, capture common patterns
- **Routed Experts (×64)**: Top-2 selected per token, specialized

## 🔄 Sparse Upcycling

Upcycling loads **ALL** trained weights and uses projections to initialize MoE:

| Component | From Checkpoint | Notes |
|-----------|-----------------|-------|
| `text_encoder` | ✅ Full | All XLM-RoBERTa layers |
| `vision_text_model` | ✅ Full | CLIP (frozen) |
| `xlmr_text_projection` | ✅ | 768 → 512 |
| `text_projection_output` | ✅ | 512 → 768 |
| `vision_projection_output` | ✅ | 768 → 768 |
| `logit_scale/bias` | ✅ | Temperature |
| **MoE shared experts** | ✅ via upcycling | Exact copy |
| **MoE routed experts** | ✅ via upcycling | Copy + noise |
| `visual_projector` | ❌ Random | Need training |
| `modality_embedding` | ❌ Random | Need training |

## 📊 Training Pipeline

| Stage | Script | Epochs | Description |
|-------|--------|--------|-------------|
| 1. Teacher | `train_contrastive.py` | 15 | XLM-R ↔ CLIP text alignment |
| 2. Contrastive | `train_contrastive.py` | 10 | Vision-language alignment |
| 3. Upcycle | `upcycle_to_moe.py` | - | Convert to MoE |
| 4. Train MoE | `train_vlmoe.py` | 5-10 | Fine-tune MoE |

### Recommended MoE Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_routed_experts` | 64 | Specialized experts |
| `num_shared_experts` | 2 | Always active |
| `num_experts_per_tok` | 2 | Top-k routing |
| `learning_rate` | 2e-5 | Lower than dense |
| `batch_size` | 512 | + gradient accumulation |

## 🔗 References

- [DeepSeek-MoE](https://arxiv.org/abs/2401.06066)
- [LIMoE](https://arxiv.org/abs/2206.02770)
- [Sparse Upcycling](https://arxiv.org/abs/2212.05055)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [XLM-RoBERTa](https://arxiv.org/abs/1911.02116)

## 📄 License

MIT License
