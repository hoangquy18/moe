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
                    transforms.ToTensor(),
                    # transforms.RandomHorizontalFlip(),
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
        self.samples = [self.data[str(i)] for i in range(len(self.data))]

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
        print(f"Image shape: {image.shape}")

        return encoding


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    train_dataset = JsonMultimodalDataset(
        json_path="moe_dataset/moe_dataset.json",
        tokenizer=tokenizer,
        base_image_path="moe_dataset",
    )
    print(f"Dataset length: {len(train_dataset)}")
    print(f"First sample: {train_dataset[0]}")
    print(f"First sample image shape: {train_dataset[0]['image'].shape}")
