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
from datasets import Dataset as HFDataset
from utils.logger_config import logger

try:
    import clip

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print(
        "Warning: CLIP not available. Install with 'pip install git+https://github.com/openai/CLIP.git'"
    )

from text_preprocess import TextNormalize


class PhoMTParallelDataset(Dataset):
    """Dataset for PhoMT parallel text files (.en and .vi) used in Teacher Learning Stage"""

    def __init__(
        self,
        en_file_path,
        vi_file_path,
        tokenizer,
        max_length=77,
        preprocess_text=True,
        max_samples=None,
        use_dual_tokenization=True,
    ):
        """
        Args:
            en_file_path: Path to English text file (.en)
            vi_file_path: Path to Vietnamese text file (.vi)
            tokenizer: XLM-R tokenizer for processing text
            max_length: Maximum length of tokenized text
            preprocess_text: Whether to apply Vietnamese text preprocessing
            max_samples: Maximum number of samples to load (None for all)
            use_dual_tokenization: Whether to use both CLIP and XLM-R tokenizers
        """
        self.tokenizer = tokenizer  # XLM-R tokenizer
        self.max_length = max_length
        self.preprocess_text = preprocess_text
        self.use_dual_tokenization = use_dual_tokenization

        # Initialize CLIP tokenizer for dual tokenization
        if self.use_dual_tokenization:
            from transformers import CLIPTokenizer

            self.clip_tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

        # Initialize text normalizer for Vietnamese
        if self.preprocess_text:
            self.text_normalizer = TextNormalize()
            self.text_normalizer.createVowelsTable()

        # Load parallel text files
        with open(en_file_path, "r", encoding="utf-8") as f:
            self.en_texts = [line.strip() for line in f.readlines()]

        with open(vi_file_path, "r", encoding="utf-8") as f:
            self.vi_texts = [line.strip() for line in f.readlines()]

        # Ensure both files have the same number of lines
        assert len(self.en_texts) == len(
            self.vi_texts
        ), f"English file has {len(self.en_texts)} lines, Vietnamese file has {len(self.vi_texts)} lines"

        # Limit samples if specified
        if max_samples is not None:
            self.en_texts = self.en_texts[:max_samples]
            self.vi_texts = self.vi_texts[:max_samples]

        # Filter out empty lines
        valid_pairs = [
            (en, vi)
            for en, vi in zip(self.en_texts, self.vi_texts)
            if en.strip() and vi.strip()
        ]
        self.en_texts, self.vi_texts = zip(*valid_pairs) if valid_pairs else ([], [])

        logger.info(
            f"Loaded {len(self.en_texts)} parallel text pairs from PhoMT dataset"
        )

    def __len__(self):
        return len(self.en_texts)

    def __getitem__(self, idx):
        en_text = self.en_texts[idx]
        vi_text = self.vi_texts[idx]

        # Preprocess Vietnamese text if enabled
        if self.preprocess_text:
            vi_text = self.text_normalizer.normalize(vi_text)

        if self.use_dual_tokenization:
            # Tokenize English with CLIP tokenizer (for CLIP model)
            en_clip_encoding = self.clip_tokenizer(
                en_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,  # CLIP max length
                return_tensors="pt",
            )

            # Tokenize Vietnamese with XLM-R tokenizer (for XLM-R model)
            vi_xlmr_encoding = self.tokenizer(
                vi_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Remove batch dimension
            en_clip_encoding = {k: v.squeeze(0) for k, v in en_clip_encoding.items()}
            vi_xlmr_encoding = {k: v.squeeze(0) for k, v in vi_xlmr_encoding.items()}

            return {
                "text_1": en_clip_encoding,  # English text for CLIP
                "text_2": vi_xlmr_encoding,  # Vietnamese text for XLM-R
                "raw_text_1": en_text,
                "raw_text_2": vi_text,
            }

        else:
            # Original single tokenizer approach
            en_encoding = self.tokenizer(
                en_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            vi_encoding = self.tokenizer(
                vi_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Remove batch dimension
            en_encoding = {k: v.squeeze(0) for k, v in en_encoding.items()}
            vi_encoding = {k: v.squeeze(0) for k, v in vi_encoding.items()}

            return {
                "text_1": en_encoding,  # English text
                "text_2": vi_encoding,  # Vietnamese text
            }


class ContrastiveDataset(Dataset):
    """Base class for contrastive learning datasets that ensures proper image-text pairing"""

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement this method")

    def __len__(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_image_text_pair(self, idx):
        """Returns image and text that form a positive pair"""
        raise NotImplementedError("Subclasses must implement this method")

    def _create_empty_image(self, size=(224, 224)):
        """Create an empty RGB image as fallback"""
        return Image.new("RGB", size, color=(128, 128, 128))


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
        use_clip_processor=True,
        preprocess_text=True,
        create_local_crops=False,
        num_local_crops=4,
        local_crop_size=98,
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
            use_clip_processor: Whether to use CLIP's processor
            preprocess_text: Whether to apply Vietnamese text preprocessing
            create_local_crops: Whether to create local crops for self-distillation
            num_local_crops: Number of local crops to generate per image
            local_crop_size: Size of each local crop
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.base_image_path = base_image_path
        self.image_key = image_key
        self.caption_key = caption_key
        self.use_clip_processor = use_clip_processor
        self.preprocess_text = preprocess_text
        self.create_local_crops = create_local_crops
        self.num_local_crops = num_local_crops
        self.local_crop_size = local_crop_size

        # Initialize text normalizer for Vietnamese
        if self.preprocess_text:
            self.text_normalizer = TextNormalize()
            self.text_normalizer.createVowelsTable()

        # Initialize CLIP processor if available and requested
        if self.use_clip_processor and CLIP_AVAILABLE:
            _, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
            self.transform = self.clip_preprocess
            logger.info("Use CLIP processor")
        elif transform is None:
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

    def create_local_crops_from_image(self, image, num_crops=4, crop_size=98):
        """
        Create random local crops from an input PIL image using vectorization

        Args:
            image: PIL Image to crop
            num_crops: Number of local crops to generate
            crop_size: Size of each local crop (crop_size x crop_size)

        Returns:
            List of transformed local crops
        """
        width, height = image.size

        if height < crop_size or width < crop_size:
            # If the image is smaller than the crop size, resize it first
            scale_factor = max(crop_size / height, crop_size / width) * 1.2
            new_height, new_width = int(height * scale_factor), int(
                width * scale_factor
            )
            image = image.resize((new_width, new_height), Image.LANCZOS)
            height, width = new_height, new_width

        # Generate all random crop coordinates at once using numpy
        top_coords = np.random.randint(0, height - crop_size, size=num_crops)
        left_coords = np.random.randint(0, width - crop_size, size=num_crops)

        # Create crops using list comprehension (more efficient than explicit loops)
        crops = [
            image.crop((left, top, left + crop_size, top + crop_size))
            for top, left in zip(top_coords, left_coords)
        ]

        # Apply transforms using list comprehension
        if self.transform:
            transformed_crops = [self.transform(crop) for crop in crops]
        else:
            transformed_crops = crops

        return transformed_crops

    def get_image_text_pair(self, idx):
        item = self.samples[idx]
        image_path = os.path.join(self.base_image_path, item[self.image_key])
        caption = item[self.caption_key]

        # Preprocess caption if enabled
        if self.preprocess_text:
            caption = self.text_normalizer.normalize(caption)

        # Load image with fallback to empty image
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")

                # Create local crops if enabled
                local_crops = None
                if self.create_local_crops:
                    # Create crops from the original image before applying transforms
                    local_crops = self.create_local_crops_from_image(
                        image,
                        num_crops=self.num_local_crops,
                        crop_size=self.local_crop_size,
                    )

                # Apply transform to main image
                if self.transform:
                    image = self.transform(image)
            else:
                logger.info(
                    f"Warning: Image not found at {image_path}, using empty image"
                )
                image = self._create_empty_image()
                local_crops = None
                if self.transform:
                    image = self.transform(image)
        except Exception as e:
            logger.info(
                f"Warning: Failed to load image at {image_path}: {e}, using empty image"
            )
            image = self._create_empty_image()
            local_crops = None
            if self.transform:
                image = self.transform(image)

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

        # Add image, local crops, and original caption to encoding
        encoding["image"] = image
        if local_crops:
            encoding["local_crops"] = local_crops
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

        # Handle local crops if present
        has_local_crops = (
            "local_crops" in batch[0] and batch[0]["local_crops"] is not None
        )
        if has_local_crops:
            # Collect all local crops into a single batch
            all_local_crops = []
            for item in batch:
                all_local_crops.extend(item["local_crops"])
            local_crops = torch.stack(all_local_crops) if all_local_crops else None
        else:
            local_crops = None

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

        if local_crops is not None:
            collated_batch["local_crops"] = local_crops

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
    create_local_crops: bool = False,
    num_local_crops: int = 4,
    local_crop_size: int = 98,
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
        create_local_crops: Whether to create local crops for self-distillation
        num_local_crops: Number of local crops to generate per image
        local_crop_size: Size of each local crop
    """

    # # Use the new sampler that ensures unique images per batch
    # sampler = UniqueImageBatchSampler(
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     shuffle=shuffle,
    #     drop_last=drop_last,
    #     seed=seed,
    # )

    # collator = ContrastiveCollator(
    #     tokenizer=tokenizer, use_controlled_negatives=use_controlled_negatives
    # )

    dataloader = DataLoader(
        dataset,
        # batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # collate_fn=collator,
        batch_size = batch_size
    )

    return dataloader
