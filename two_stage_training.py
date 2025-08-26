#!/usr/bin/env python3
"""
Two-Stage Training Script for AltCLIP-like Model

This script implements the two-stage training approach described in the AltCLIP paper:
1. Teacher Learning Stage: Align CLIP text encoder with XLM-R text encoder using parallel text
2. Contrastive Learning Stage: Train on text-image pairs for multimodal understanding

Usage:
    # Stage 1: Teacher Learning
    python two_stage_training.py --training_stage teacher --num_epochs 5 --parallel_text_path sample_parallel_text.json

    # Stage 2: Contrastive Learning
    python two_stage_training.py --training_stage contrastive --num_epochs 10 --resume_from outputs/checkpoint_epoch_4.pt
"""

import argparse
import os
import torch
from transformers import AutoTokenizer
from model.builder import build_model
from model.config import VisionConfig
from trainer import ContrastiveTrainer
from data.contrastive_dataloader import ContrastiveJsonDataset, PhoMTParallelDataset
from utils.logger_config import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage training for multimodal model"
    )

    # Basic training arguments
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Resume from checkpoint"
    )

    # Two-stage training
    parser.add_argument(
        "--training_stage",
        type=str,
        default="teacher",
        choices=["teacher", "contrastive"],
        help="Training stage: teacher (align text encoders) or contrastive (text-image pairs)",
    )

    # Teacher stage specific
    parser.add_argument(
        "--en_file_path",
        type=str,
        default="PhoMT/tokenization/dev/dev.en",
        help="Path to English text file (.en) for teacher learning stage",
    )
    parser.add_argument(
        "--vi_file_path",
        type=str,
        default="PhoMT/tokenization/dev/dev.vi",
        help="Path to Vietnamese text file (.vi) for teacher learning stage",
    )
    parser.add_argument(
        "--max_parallel_samples",
        type=int,
        default=1000,
        help="Maximum number of parallel text samples to use for demo (None for all)",
    )

    # Contrastive stage specific
    parser.add_argument(
        "--image_text_json",
        type=str,
        default="moe_dataset/moe_dataset.json",
        help="Path to image-text pairs JSON for contrastive learning",
    )
    parser.add_argument(
        "--image_base_path",
        type=str,
        default="moe_dataset",
        help="Base path for images",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Starting {args.training_stage} stage training")
    logger.info(f"Arguments: {args}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model
    vision_config = VisionConfig()
    model = build_model(vision_config=vision_config)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.text_encoder.config.text_model_name)

    # Create datasets based on training stage
    if args.training_stage == "teacher":
        logger.info("=== Teacher Learning Stage ===")
        logger.info("Training to align CLIP text encoder with XLM-R text encoder")
        logger.info(f"Using PhoMT dataset: {args.en_file_path} & {args.vi_file_path}")

        train_dataset = PhoMTParallelDataset(
            en_file_path=args.en_file_path,
            vi_file_path=args.vi_file_path,
            tokenizer=tokenizer,
            max_length=32,
            max_samples=args.max_parallel_samples,
        )
        val_dataset = train_dataset

        logger.info(f"Parallel text dataset size: {len(train_dataset)}")

        # Show sample
        sample = train_dataset[0]
        logger.info(f"Sample parallel text:")
        logger.info(f"  English: {sample['raw_text_1']}")
        logger.info(f"  Vietnamese: {sample['raw_text_2']}")

    else:
        logger.info("=== Contrastive Learning Stage ===")
        logger.info("Training on text-image pairs for multimodal understanding")

        train_dataset = ContrastiveJsonDataset(
            json_path=args.image_text_json,
            tokenizer=tokenizer,
            base_image_path=args.image_base_path,
            max_length=32,
            image_key="image_id",
            caption_key="caption",
        )
        val_dataset = train_dataset

        logger.info(f"Image-text dataset size: {len(train_dataset)}")
        if hasattr(train_dataset, "unique_image_ids"):
            logger.info(f"Unique images: {len(train_dataset.unique_image_ids)}")

    # Initialize trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn="clip",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir=args.output_dir,
        training_stage=args.training_stage,
    )

    # Start training
    trainer.train(resume_from=args.resume_from)

    logger.info(f"{args.training_stage} stage training completed!")

    if args.training_stage == "teacher":
        logger.info("\nNext steps:")
        logger.info("1. Run contrastive learning stage with:")
        logger.info(
            f"   python two_stage_training.py --training_stage contrastive --resume_from {args.output_dir}/best_model.pt"
        )
        logger.info("\nFor larger training, use the full train set:")
        logger.info(
            "   python two_stage_training.py --training_stage teacher --en_file_path PhoMT/tokenization/train/train.en --vi_file_path PhoMT/tokenization/train/train.vi --max_parallel_samples None"
        )


if __name__ == "__main__":
    main()
