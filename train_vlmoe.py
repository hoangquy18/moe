#!/usr/bin/env python
# coding=utf-8
"""
Stage 3: Train VL-MoE Model

After upcycling from dense checkpoint, this script trains the VL-MoE model
with contrastive learning + MoE auxiliary losses.

Training Strategy:
==================
- Freeze: vision_text_model (CLIP), optionally text_encoder
- Train: visual_projector, modality_embedding, moe_layers, output_heads
- Losses: Contrastive + Load Balance + Z-Loss + (optional) MI-Loss

Recommended Settings:

- Learning rate: 1e-5 to 5e-5 (lower than dense training)
- Batch s=====================ize: 512-1024 (smaller due to MoE memory)
- Epochs: 5-10 (MoE converges faster with upcycling)
- Gradient accumulation: 4-8 if memory limited

Usage:
======
python train_vlmoe.py \
    --upcycled_checkpoint outputs/vlmoe/vlmoe_upcycled.pt \
    --output_dir outputs/vlmoe_trained \
    --num_epochs 10 \
    --batch_size 512 \
    --learning_rate 2e-5
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.vlmoe.vlmoe_model import VLMoEModel, VLMoEConfig
from data.contrastive_dataloader import ContrastiveJsonDataset
from utils.logger_config import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VL-MoE model (Stage 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Checkpoint
    parser.add_argument(
        "--upcycled_checkpoint",
        type=str,
        required=True,
        help="Path to upcycled VL-MoE checkpoint",
    )

    # Dataset
    parser.add_argument(
        "--json_path",
        type=str,
        default="moe_dataset/moe_dataset.json",
        help="Path to dataset JSON",
    )
    parser.add_argument(
        "--base_image_path",
        type=str,
        default="moe_dataset",
        help="Base path for images",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="Maximum text length",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/vlmoe_trained",
        help="Output directory",
    )

    # Training hyperparameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size (recommend 512-1024)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (recommend 1e-5 to 5e-5)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (0.1 = 10% of training)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16"],
        help="Training precision",
    )

    # MoE specific
    parser.add_argument(
        "--freeze_text_encoder",
        action="store_true",
        help="Freeze text encoder (recommended for stability)",
    )
    parser.add_argument(
        "--mi_loss_weight",
        type=float,
        default=0.0,
        help="MI-Loss weight (0 = disabled, try 0.01 if routing imbalanced)",
    )

    # Logging
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Log every N steps",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


class VLMoETrainer:
    """Trainer for VL-MoE model (Stage 3)"""

    def __init__(
        self,
        model: VLMoEModel,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler,
        config: argparse.Namespace,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Mixed precision
        self.use_amp = config.precision == "fp16" and device != "cpu"
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking
        self.global_step = 0
        self.best_loss = float("inf")

        os.makedirs(config.output_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        self.model.train()

        epoch_stats = {
            "total_loss": 0.0,
            "task_loss": 0.0,
            "load_balance_loss": 0.0,
            "z_loss": 0.0,
            "mi_loss": 0.0,
            "budget_loss": 0.0,
        }
        num_batches = len(self.train_loader)
        start_time = time.time()

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.num_epochs}",
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            pixel_values = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Forward pass
            if self.use_amp:
                with autocast(device_type=self.device, dtype=torch.float16):
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    loss = (
                        outputs["total_loss"] / self.config.gradient_accumulation_steps
                    )

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = outputs["total_loss"] / self.config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

            # Accumulate stats
            batch_loss = loss.item() * self.config.gradient_accumulation_steps
            epoch_stats["total_loss"] += batch_loss
            epoch_stats["task_loss"] += outputs["task_loss"].item()
            epoch_stats["load_balance_loss"] += outputs["load_balance_loss"].item()
            epoch_stats["z_loss"] += outputs["z_loss"].item()
            epoch_stats["mi_loss"] += outputs["mi_loss"].item()
            epoch_stats["budget_loss"] += outputs["budget_loss"].item()

            # Log
            if batch_idx % self.config.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                log_msg = (
                    f"Step {batch_idx}/{num_batches}: "
                    f"Loss={batch_loss:.4f}, "
                    f"Task={outputs['task_loss'].item():.4f}, "
                    f"LB={outputs['load_balance_loss'].item():.4f}, "
                    f"Z={outputs['z_loss'].item():.4f}, "
                )
                if outputs["budget_loss"].item() > 0:
                    log_msg += f"Budget={outputs['budget_loss'].item():.4f}, "
                log_msg += f"LR={lr:.2e}"
                logger.info(log_msg)

            progress_bar.set_postfix(
                {
                    "loss": f"{batch_loss:.4f}",
                    "task": f"{outputs['task_loss'].item():.4f}",
                    "lb": f"{outputs['load_balance_loss'].item():.4f}",
                }
            )

        # Average stats
        for key in epoch_stats:
            epoch_stats[key] /= num_batches

        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch} completed in {epoch_time:.1f}s - "
            f"Avg Loss: {epoch_stats['total_loss']:.4f}"
        )

        return epoch_stats

    def save_checkpoint(self, epoch: int, stats: dict):
        """Save checkpoint"""
        checkpoint_path = os.path.join(
            self.config.output_dir, f"vlmoe_epoch_{epoch}.pt"
        )

        # Get config dict from model
        model_config = (
            self.model.config.__dict__ if hasattr(self.model.config, "__dict__") else {}
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "stats": stats,
            "config": model_config,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best
        if stats["total_loss"] < self.best_loss:
            self.best_loss = stats["total_loss"]
            best_path = os.path.join(self.config.output_dir, "vlmoe_best.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved (loss={self.best_loss:.4f})")

    def train(self):
        """Full training loop"""
        logger.info("Starting VL-MoE training...")
        logger.info(f"Total epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(
            f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}"
        )

        for epoch in range(1, self.config.num_epochs + 1):
            stats = self.train_epoch(epoch)

            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch, stats)

        logger.info("Training completed!")
        logger.info(f"Best loss: {self.best_loss:.4f}")


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load upcycled checkpoint
    logger.info(f"Loading upcycled checkpoint from {args.upcycled_checkpoint}")
    checkpoint = torch.load(args.upcycled_checkpoint, map_location="cpu")

    # Create config from checkpoint
    config_dict = checkpoint.get("config", {})
    config = VLMoEConfig(**config_dict)

    # Override MI loss weight if specified
    if args.mi_loss_weight > 0:
        config.mi_loss_weight = args.mi_loss_weight
        logger.info(f"MI-Loss enabled with weight={args.mi_loss_weight}")

    # Create model
    logger.info("Creating VL-MoE model...")
    model = VLMoEModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded model weights from checkpoint")

    # Free memory
    del checkpoint
    torch.cuda.empty_cache()

    # Freeze components
    # Always freeze CLIP vision
    for param in model.vision_text_model.parameters():
        param.requires_grad = False
    logger.info("Frozen: vision_text_model (CLIP)")

    # Optionally freeze text encoder
    if args.freeze_text_encoder:
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        logger.info("Frozen: text_encoder (XLM-RoBERTa)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {trainable_params/total_params*100:.1f}%")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)

    # Create dataset
    logger.info("Loading dataset...")
    dataset = ContrastiveJsonDataset(
        json_path=args.json_path,
        tokenizer=tokenizer,
        base_image_path=args.base_image_path,
        max_length=args.max_length,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create optimizer
    # Different learning rates for different components
    param_groups = [
        # New components (higher lr)
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and ("visual_projector" in n or "modality_embedding" in n)
            ],
            "lr": args.learning_rate * 2,
            "name": "new_components",
        },
        # MoE layers (standard lr)
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and "moe_layers" in n
            ],
            "lr": args.learning_rate,
            "name": "moe_layers",
        },
        # Output heads and projections (lower lr since pretrained)
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and ("projection" in n or "head" in n or "output_norm" in n)
            ],
            "lr": args.learning_rate * 0.5,
            "name": "projections",
        },
        # Everything else
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and not any(
                    x in n
                    for x in [
                        "visual_projector",
                        "modality_embedding",
                        "moe_layers",
                        "projection",
                        "head",
                        "output_norm",
                    ]
                )
            ],
            "lr": args.learning_rate,
            "name": "other",
        },
    ]

    # Filter empty groups
    param_groups = [g for g in param_groups if len(list(g["params"])) > 0]

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )

    # Log param groups
    for group in param_groups:
        count = len([p for p in group["params"]])
        logger.info(
            f"Param group '{group['name']}': {count} tensors, lr={group['lr']:.2e}"
        )

    # Create scheduler
    total_steps = (
        len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    )
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    # Create trainer
    trainer = VLMoETrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=args,
        device=device,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
