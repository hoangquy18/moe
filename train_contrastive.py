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
        "--grad_norm",
        type=float,
        default=1.0,
        help="Gradient norm to clip",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "fp8"],
        help="Precision for training (fp32, fp16, or fp8)",
    )

    # Two-stage training arguments
    parser.add_argument(
        "--training_stage",
        type=str,
        default="contrastive",
        choices=["teacher", "contrastive"],
        help="Training stage: teacher (align text encoders) or contrastive (text-image pairs)",
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default=None,
        help="Path to stage 1 (teacher) checkpoint to load for stage 2 (contrastive) training",
    )
    parser.add_argument(
        "--en_file_path",
        type=str,
        default="PhoMT/tokenization/train/train.en",
        help="Path to English text file (.en) for teacher learning stage",
    )
    parser.add_argument(
        "--vi_file_path",
        type=str,
        default="PhoMT/tokenization/train/train.vi",
        help="Path to Vietnamese text file (.vi) for teacher learning stage",
    )
    parser.add_argument(
        "--max_parallel_samples",
        type=int,
        default=None,
        help="Maximum number of parallel text samples to use (None for all)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="Maximum length of tokenized text",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=1.4,
        help="Number of epochs for warmup",
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

    # Build model with simplified configuration
    vision_config = VisionConfig()
    model = build_model(vision_config=vision_config)

    # Load stage 1 checkpoint if specified and we're in stage 2
    if args.training_stage == "contrastive" and args.stage1_checkpoint:
        logger.info(f"Loading stage 1 checkpoint from {args.stage1_checkpoint}")

        if not os.path.exists(args.stage1_checkpoint):
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found: {args.stage1_checkpoint}"
            )

        # Load the checkpoint
        checkpoint = torch.load(args.stage1_checkpoint, map_location="cpu")

        # Load only the model state dict (not optimizer, scheduler, etc.)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info("Successfully loaded stage 1 model weights for stage 2 training")

        # Log some info about the loaded checkpoint
        if "epoch" in checkpoint:
            logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if "val_loss" in checkpoint and checkpoint["val_loss"] is not None:
            logger.info(f"Stage 1 final validation loss: {checkpoint['val_loss']:.4f}")

    # Initialize tokenizer from text encoder config
    tokenizer = AutoTokenizer.from_pretrained(model.text_encoder.config.text_model_name)

    # Create datasets based on training stage
    if args.training_stage == "teacher":
        # Teacher Learning Stage: Use PhoMT parallel text files
        from data.contrastive_dataloader import PhoMTParallelDataset

        train_dataset = PhoMTParallelDataset(
            en_file_path=args.en_file_path,
            vi_file_path=args.vi_file_path,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_samples=args.max_parallel_samples,
        )

        logger.info(f"Teacher stage - Parallel text dataset size: {len(train_dataset)}")
    else:
        # Contrastive Learning Stage: Use image-text pairs
        train_dataset = ContrastiveJsonDataset(
            json_path="moe_dataset/moe_dataset.json",
            tokenizer=tokenizer,
            base_image_path="moe_dataset",
            max_length=args.max_length,  # Set appropriate max length based on model requirements
            image_key="image_id",  # Ensure this matches your JSON structure
            caption_key="caption",  # Ensure this matches your JSON structure
        )

        logger.info(
            f"Contrastive stage - Image-text dataset size: {len(train_dataset)}"
        )
        logger.info(
            f"Unique images in training dataset: {len(train_dataset.unique_image_ids)}"
        )

    logger.info(f"Train dataset size: {len(train_dataset)}")

    # Initialize trainer with new parameters
    trainer = ContrastiveTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=None,
        loss_fn=args.loss_fn,
        grad_norm=args.grad_norm,
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
        training_stage=args.training_stage,
    )

    # Start training
    trainer.train(resume_from=args.resume_from)


if __name__ == "__main__":
    main()
