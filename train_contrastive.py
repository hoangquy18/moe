import os
import argparse
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer
from model.builder import build_model
from utils.logger_config import logger
from trainer import ContrastiveTrainer
from datasets import load_dataset
import json
from PIL import Image
from torchvision import transforms
from model.config import VisionConfig
from data.contrastive_dataloader import ContrastiveJsonDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a multimodal model using contrastive learning"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nhq188/moe-dataset-2",
        help="Name of the Hugging Face dataset",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="Column name for images in the dataset",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="Column name for captions in the dataset",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="linear",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--save_every", type=int, default=1, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Resume training from checkpoint"
    )

    # Model arguments
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="clip",
        choices=["clip", "siglip"],
        help="Loss function to use",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "fp8"],
        help="Precision for training (fp32, fp16, or fp8)",
    )

    # Self-distillation and masking arguments
    parser.add_argument(
        "--use_masking",
        action="store_true",
        help="Use masked vision encoder with self-distillation",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.4,
        help="Ratio of image tokens to mask",
    )
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default=None,
        help="Name of the Hugging Face model to use as teacher (larger model)",
    )
    parser.add_argument(
        "--distillation_alpha",
        type=float,
        default=1.0,  # α = 1
        help="Weight of distillation loss in the total loss",
    )
    parser.add_argument(
        "--masking_beta",
        type=float,
        default=2.0,  # β = 2
        help="Weight of masking loss in the total loss",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=1.4,
        help="Number of epochs for warmup",
    )
    parser.add_argument(
        "--teacher_momentum_base",
        type=float,
        default=0.994,
        help="Base momentum for teacher EMA",
    )

    # Distributed training arguments
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )

    # Contrastive learning arguments
    parser.add_argument(
        "--use_controlled_negatives",
        action="store_true",
        help="Use controlled negative pairs in contrastive training",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # Setup distributed training if enabled
    rank = 0
    world_size = 1
    if args.distributed:
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(args.local_rank)

    logger.info(f"Starting training with arguments: {args}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build model with enhanced configurations
    vision_config = VisionConfig()
    vision_config.use_self_distillation = args.use_masking
    vision_config.mask_ratio = args.mask_ratio
    vision_config.teacher_momentum_base = args.teacher_momentum_base
    vision_config.teacher_momentum_final = 1.0  # Final value is always 1.0
    vision_config.distillation_alpha = args.distillation_alpha
    vision_config.masking_beta = args.masking_beta
    
    # Add teacher model configuration
    if args.teacher_model_name:
        vision_config.teacher_model_name = args.teacher_model_name
        logger.info(f"Using different teacher model: {args.teacher_model_name}")
    
    model = build_model(use_masking=args.use_masking, vision_config=vision_config)

    # Initialize tokenizer from text encoder config
    tokenizer = AutoTokenizer.from_pretrained(model.text_encoder.config.text_model_name)

    # Create datasets - using our contrastive dataset
    train_dataset = ContrastiveJsonDataset(
        json_path="moe_dataset/moe_dataset.json",
        tokenizer=tokenizer,
        base_image_path="moe_dataset",
        max_length=32,  # Set appropriate max length based on model requirements
        image_key="image_id",  # Ensure this matches your JSON structure
        caption_key="caption",  # Ensure this matches your JSON structure
    )
    print(train_dataset[0])
    val_dataset = (
        train_dataset  # For demonstration - ideally use a separate validation set
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(
        f"Unique images in training dataset: {len(train_dataset.unique_image_ids)}"
    )

    # Initialize trainer with new parameters
    trainer = ContrastiveTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=args.loss_fn,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        scheduler_type="custom_linear",  # Use our custom scheduler
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir=args.output_dir,
        save_every=args.save_every,
        use_distributed=args.distributed,
        rank=rank,
        world_size=world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        precision=args.precision,
        num_workers=args.num_workers,
        use_controlled_negatives=args.use_controlled_negatives,
        seed=args.seed,
        use_masking=args.use_masking,
        mask_ratio=args.mask_ratio,
        distillation_alpha=args.distillation_alpha,
        masking_beta=args.masking_beta,
    )

    # Start training
    trainer.train(resume_from=args.resume_from)


if __name__ == "__main__":
    main()
