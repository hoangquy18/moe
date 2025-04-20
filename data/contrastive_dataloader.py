import torch
import random
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from PIL import Image
import os
import json
from torchvision import transforms
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict


class ContrastiveDataset(Dataset):
    """Base class for contrastive learning datasets that ensures proper image-text pairing"""

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement this method")

    def __len__(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_image_text_pair(self, idx):
        """Returns image and text that form a positive pair"""
        raise NotImplementedError("Subclasses must implement this method")


class ContrastiveJsonDataset(ContrastiveDataset):
    """Dataset for loading image-caption pairs from a JSON file with correct pairing"""

    def __init__(
        self,
        json_path,
        tokenizer,
        base_image_path="",
        max_length=77,
        transform=None,
        image_key="image_id",
        caption_key="caption",
    ):
        """
        Args:
            json_path: Path to the JSON file containing image paths and captions
            tokenizer: Tokenizer for processing captions
            base_image_path: Base path to prepend to the image paths in the JSON
            max_length: Maximum length of tokenized captions
            transform: Optional transform to apply to images
            image_key: Key in JSON for image path/identifier
            caption_key: Key in JSON for caption text
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.base_image_path = base_image_path
        self.image_key = image_key
        self.caption_key = caption_key

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224), antialias=True),
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

        # Create image_id to indices mapping
        self.image_to_indices = defaultdict(list)
        for idx, item in enumerate(self.samples):
            self.image_to_indices[item[self.image_key]].append(idx)

        # Get unique image_ids
        self.unique_image_ids = list(self.image_to_indices.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.get_image_text_pair(idx)

    def get_image_text_pair(self, idx):
        item = self.samples[idx]
        image_path = os.path.join(self.base_image_path, item[self.image_key])
        caption = item[self.caption_key]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {image_path}: {e}")

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

        # Add image and original caption to encoding
        encoding["image"] = image
        encoding["raw_caption"] = caption
        encoding["image_id"] = item[self.image_key]

        return encoding


class UniqueImageBatchSampler(Sampler):
    """
    Sampler that ensures each image appears at most once in each batch,
    even when there are multiple captions for the same image.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            dataset: Dataset with image_to_indices mapping and unique_image_ids list
            batch_size: Batch size
            shuffle: Whether to shuffle the images
            drop_last: Whether to drop the last incomplete batch
            seed: Random seed for shuffling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rng = random.Random(seed)

    def __iter__(self):
        # Get all unique image IDs
        unique_image_ids = list(self.dataset.unique_image_ids)

        # Shuffle the unique image IDs if required
        if self.shuffle:
            self.rng.shuffle(unique_image_ids)

        # For each unique image ID, select one caption randomly
        selected_indices = []
        for image_id in unique_image_ids:
            # Get all indices for this image
            indices_for_image = self.dataset.image_to_indices[image_id]
            # Randomly select one caption for this image
            selected_idx = self.rng.choice(indices_for_image)
            selected_indices.append(selected_idx)

        # Yield batches of selected indices
        for i in range(0, len(selected_indices), self.batch_size):
            if i + self.batch_size <= len(selected_indices) or not self.drop_last:
                yield selected_indices[i : i + self.batch_size]

    def __len__(self):
        num_unique_images = len(self.dataset.unique_image_ids)
        if self.drop_last:
            return num_unique_images // self.batch_size
        else:
            return (num_unique_images + self.batch_size - 1) // self.batch_size


class ContrastiveCollator:
    """
    Collates individual samples into batches while preserving image-text pairing
    Can also generate controlled negative pairs if needed
    """

    def __init__(self, tokenizer=None, use_controlled_negatives: bool = False):
        self.tokenizer = tokenizer
        self.use_controlled_negatives = use_controlled_negatives

    def __call__(self, batch):
        """
        Collate function that ensures proper matching between images and texts

        Args:
            batch: List of dictionaries, each containing image and text data

        Returns:
            Dictionary with batched data, ensuring correct positive pairs
        """
        # Extract images and text features
        images = torch.stack([item["image"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])

        # Handle token_type_ids if present
        if "token_type_ids" in batch[0]:
            token_type_ids = torch.stack([item["token_type_ids"] for item in batch])
        else:
            token_type_ids = None

        # Collect raw captions and image IDs for debugging
        raw_captions = [item.get("raw_caption", "") for item in batch]
        image_ids = [item.get("image_id", "") for item in batch]

        # Create output batch
        collated_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": images,
            "raw_captions": raw_captions,
            "image_ids": image_ids,
            "positive_indices": list(
                range(len(batch))
            ),  # Each item is its own positive match
        }

        if token_type_ids is not None:
            collated_batch["token_type_ids"] = token_type_ids

        # If using controlled negatives, add indices of deliberate negative pairs
        if self.use_controlled_negatives:
            # Create deliberate hard negative pairs (implementation depends on specific requirements)
            # This is just a placeholder - you would implement your own negative sampling strategy
            negative_indices = []
            batch_size = len(batch)
            for i in range(batch_size):
                # Simple strategy: each image is paired with the next text in the batch (wrapped around)
                negative_indices.append((i + 1) % batch_size)
            collated_batch["negative_indices"] = negative_indices

        return collated_batch


def create_contrastive_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    tokenizer=None,
    use_controlled_negatives: bool = False,
    seed: int = 42,
):
    """
    Create a DataLoader for contrastive learning with proper image-text pairing,
    ensuring each image appears at most once per batch.

    Args:
        dataset: Dataset instance that implements get_image_text_pair and has image_to_indices mapping
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        tokenizer: Optional tokenizer for the collator
        use_controlled_negatives: Whether to generate controlled negative pairs
        seed: Random seed for reproducibility

    Returns:
        DataLoader configured for contrastive learning
    """
    # Use the new sampler that ensures unique images per batch
    sampler = UniqueImageBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )

    collator = ContrastiveCollator(
        tokenizer=tokenizer, use_controlled_negatives=use_controlled_negatives
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )

    return dataloader
