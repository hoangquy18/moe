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
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
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


class MultimodalDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        image_column="image",
        caption_column="caption",
        max_length=77,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_column = image_column
        self.caption_column = caption_column
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column]
        caption = item[self.caption_column]

        # Tokenize text
        encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove batch dimension added by tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # Add image to encoding
        encoding["image"] = image

        return encoding


class JsonMultimodalDataset(Dataset):
    def __init__(
        self, json_path, tokenizer, base_image_path="", max_length=77, transform=None
    ):
        """
        Dataset class for loading image-caption pairs from a JSON file

        Args:
            json_path: Path to the JSON file containing image paths and captions
            tokenizer: Tokenizer for processing captions
            base_image_path: Base path to prepend to the image paths in the JSON
            max_length: Maximum length of tokenized captions
            transform: Optional transform to apply to images
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.base_image_path = base_image_path

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (224, 224), antialias=True
                    ),  # args.crop_size, by default it is set to be 224
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = transform

        # Load the JSON data
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Convert dictionary to list for easier indexing
        self.samples = []
        for index in self.data.keys():
            self.samples.append(self.data[index])
        # self.samples = [self.data[str(i)] for i in range(len(self.data))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = os.path.join(self.base_image_path, item["image_id"])
        caption = item["caption"]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image or raise exception based on your error handling strategy
            raise RuntimeError(f"Failed to load image at {image_path}")

        # Tokenize caption
        encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove batch dimension added by tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # Add image to encoding
        encoding["image"] = image

        return encoding


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

    # Build model
    model = build_model()

    # Initialize tokenizer from text encoder config
    tokenizer = AutoTokenizer.from_pretrained(model.text_encoder.config.text_model_name)

    # Create datasets - using our contrastive dataset
    train_dataset = ContrastiveJsonDataset(
        json_path="moe_dataset/moe_dataset.json",
        tokenizer=tokenizer,
        base_image_path="moe_dataset",
        max_length=77,  # Set appropriate max length based on model requirements
        image_key="image_id",  # Ensure this matches your JSON structure
        caption_key="caption",  # Ensure this matches your JSON structure
    )

    val_dataset = (
        train_dataset  # For demonstration - ideally use a separate validation set
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(
        f"Unique images in training dataset: {len(train_dataset.unique_image_ids)}"
    )

    # Initialize trainer with new contrastive options
    trainer = ContrastiveTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=args.loss_fn,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        scheduler_type=args.scheduler_type,
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
    )

    # Start training
    trainer.train(resume_from=args.resume_from)


if __name__ == "__main__":
    main()
