"""
Script to convert weights from old MultiModalEncoder (contrastive learning)
to new ImageTextMultiModalEncoder (vision-language architecture)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def convert_mm_encoder_weights(
    old_state_dict: Dict[str, torch.Tensor],
    new_model: nn.Module,
    text_encoder_model_name: str = "FacebookAI/xlm-roberta-base",
    vision_model_name: str = "openai/clip-vit-base-patch32",
) -> Dict[str, torch.Tensor]:
    """
    Convert weights from old MultiModalEncoder to new ImageTextMultiModalEncoder

    Args:
        old_state_dict: State dict from old MultiModalEncoder
        new_model: New ImageTextMultiModalEncoder model
        text_encoder_model_name: Name of text encoder model (for config)
        vision_model_name: Name of vision model (for config)

    Returns:
        New state dict compatible with new model
    """
    new_state_dict = {}
    unmapped_keys = []
    missing_keys = []

    # Get new model's state dict to know what keys we need
    new_model_state_dict = new_model.state_dict()

    logger.info("Starting weight conversion...")
    logger.info(f"Old model has {len(old_state_dict)} parameters")
    logger.info(f"New model has {len(new_model_state_dict)} parameters")

    # 0. Map text_encoder.* weights to _text_encoder.* (preserve trained text encoder weights)
    # This includes text_encoder.text_model.*, text_encoder.map_head.*, etc.
    text_encoder_ref_mappings = {}
    for old_key in old_state_dict.keys():
        if old_key.startswith("text_encoder."):
            # Map to _text_encoder.* in new model
            new_key = old_key.replace("text_encoder.", "_text_encoder.")
            if new_key in new_model_state_dict:
                old_shape = old_state_dict[old_key].shape
                new_shape = new_model_state_dict[new_key].shape
                if old_shape == new_shape:
                    new_state_dict[new_key] = old_state_dict[old_key]
                    text_encoder_ref_mappings[old_key] = new_key
                else:
                    logger.warning(
                        f"Shape mismatch for {old_key}: {old_shape} vs {new_shape}"
                    )
                    # Don't add to unmapped yet, will check in section 1

    logger.info(
        f"Mapped {len(text_encoder_ref_mappings)} text_encoder reference parameters"
    )

    # 0.5. Map vision_encoder.* weights to _vision_encoder.* (preserve trained vision encoder weights)
    # This includes vision_encoder.vision_model.*, vision_encoder.map_head.*, etc.
    vision_encoder_ref_mappings = {}
    for old_key in old_state_dict.keys():
        if old_key.startswith("vision_encoder.") and not old_key.startswith(
            "vision_text_model."
        ):
            # Map to _vision_encoder.* in new model
            new_key = old_key.replace("vision_encoder.", "_vision_encoder.")
            if new_key in new_model_state_dict:
                old_shape = old_state_dict[old_key].shape
                new_shape = new_model_state_dict[new_key].shape
                if old_shape == new_shape:
                    new_state_dict[new_key] = old_state_dict[old_key]
                    vision_encoder_ref_mappings[old_key] = new_key
                else:
                    logger.warning(
                        f"Shape mismatch for {old_key}: {old_shape} vs {new_shape}"
                    )
                    # Don't add to unmapped yet, will check in section 2

    logger.info(
        f"Mapped {len(vision_encoder_ref_mappings)} vision_encoder reference parameters"
    )

    # 1. Convert text encoder (RoBERTa) weights - MAP ALL COMPONENTS
    # Old: text_encoder.text_model.* -> New: text_embedding.* or encoder.*
    text_mappings = {}
    for old_key in old_state_dict.keys():
        if old_key.startswith("text_encoder.text_model."):
            # Handle embeddings separately (text_embedding.*)
            if old_key.startswith("text_encoder.text_model.embeddings."):
                new_key = old_key.replace(
                    "text_encoder.text_model.embeddings.", "text_embedding."
                )
            # Handle encoder separately (encoder.*) - already handled in section 2.5
            elif old_key.startswith("text_encoder.text_model.encoder."):
                # Skip here, will be handled in encoder_mappings
                continue
            # Handle pooler separately (pooler.*) - will be handled later
            elif old_key.startswith("text_encoder.text_model.pooler."):
                # Skip here, will be handled in pooler_mappings
                continue
            # Handle other components (LayerNorm, etc.) - try to map to text_embedding
            else:
                # Try to map to text_embedding first
                new_key = old_key.replace("text_encoder.text_model.", "text_embedding.")
                # If not found, might be in encoder
                if new_key not in new_model_state_dict:
                    # Try encoder
                    new_key = old_key.replace("text_encoder.text_model.", "encoder.")

            if new_key in new_model_state_dict:
                # Check if shapes match
                old_shape = old_state_dict[old_key].shape
                new_shape = new_model_state_dict[new_key].shape
                if old_shape == new_shape:
                    new_state_dict[new_key] = old_state_dict[old_key]
                    text_mappings[old_key] = new_key
                else:
                    logger.warning(
                        f"Shape mismatch for {old_key}: {old_shape} vs {new_shape}"
                    )
                    unmapped_keys.append(old_key)
            else:
                unmapped_keys.append(old_key)

    logger.info(f"Mapped {len(text_mappings)} text encoder (RoBERTa) parameters")

    # 2. Convert CLIP vision model weights - MAP ALL COMPONENTS
    # Note: vision_text_model is the CLIP model in old MultiModalEncoder
    # Old: vision_text_model.vision_model.* (CLIP vision encoder) -> New: image_embedding.*
    # This preserves trained CLIP vision weights by mapping them to PatchEmbedding
    vision_mappings = {}
    for old_key in old_state_dict.keys():
        if old_key.startswith("vision_text_model.vision_model."):
            # Handle embeddings
            if old_key.startswith("vision_text_model.vision_model.embeddings."):
                # Map CLIP vision embeddings to PatchEmbedding
                if "patch_embedding" in old_key:
                    new_key = old_key.replace(
                        "vision_text_model.vision_model.embeddings.patch_embedding",
                        "image_embedding.patch_embedding",
                    )
                elif "position_embedding" in old_key:
                    new_key = old_key.replace(
                        "vision_text_model.vision_model.embeddings.position_embedding",
                        "image_embedding.position_embedding",
                    )
                    # CLIP has num_positions = num_patches + 1 (with class token)
                    # PatchEmbedding has num_positions = num_patches (no class token)
                    # So we need to skip the first position (class token) and take the rest
                    if new_key in new_model_state_dict:
                        old_shape = old_state_dict[old_key].shape
                        new_shape = new_model_state_dict[new_key].shape
                        if old_shape[0] == new_shape[0] + 1:
                            # Skip first position (class token) and map the rest
                            old_weight = old_state_dict[old_key]
                            new_state_dict[new_key] = old_weight[
                                1:
                            ]  # Skip position 0 (class token)
                            vision_mappings[old_key] = new_key
                            logger.info(
                                f"Mapped position_embedding (skipped class token): {old_key} -> {new_key}"
                            )
                            continue
                        elif old_shape == new_shape:
                            # Same shape, map directly
                            new_state_dict[new_key] = old_state_dict[old_key]
                            vision_mappings[old_key] = new_key
                            continue
                        else:
                            logger.warning(
                                f"Shape mismatch for {old_key}: {old_shape} vs {new_shape}"
                            )
                            unmapped_keys.append(old_key)
                            continue
                    else:
                        unmapped_keys.append(old_key)
                        continue
                elif "class_embedding" in old_key:
                    # Skip class_embedding (not in PatchEmbedding, but log for info)
                    logger.info(
                        f"Skipping class_embedding (not in PatchEmbedding): {old_key}"
                    )
                    unmapped_keys.append(old_key)
                    continue
                else:
                    # Other embedding components - try to map
                    new_key = old_key.replace(
                        "vision_text_model.vision_model.embeddings.", "image_embedding."
                    )
            # Handle pre_layernorm (if exists in new model)
            elif old_key.startswith("vision_text_model.vision_model.pre_layrnorm"):
                # Try to map to image_embedding or skip if not in new model
                new_key = old_key.replace(
                    "vision_text_model.vision_model.pre_layrnorm",
                    "image_embedding.pre_layernorm",  # Note: might be different name
                )
                # If not found, try without pre_layernorm
                if new_key not in new_model_state_dict:
                    unmapped_keys.append(old_key)
                    continue
            # Handle encoder layers - CLIP encoder has different architecture
            # We can't directly map CLIP encoder to RobertaEncoder because:
            # 1. Model mới dùng RobertaEncoder chung cho cả text và image
            # 2. CLIP encoder có kiến trúc khác (q_proj/k_proj/v_proj vs query/key/value)
            # 3. CLIP embeddings đã được map sang image_embedding.*
            # Note: Đây là expected behavior, không phải lỗi
            elif old_key.startswith("vision_text_model.vision_model.encoder."):
                # Only log first few to avoid spam
                clip_encoder_unmapped_count = sum(
                    1
                    for k in unmapped_keys
                    if "vision_text_model.vision_model.encoder" in k
                )
                if clip_encoder_unmapped_count < 3:
                    logger.info(
                        f"CLIP encoder layer (cannot map - model mới dùng RobertaEncoder): {old_key}"
                    )
                elif clip_encoder_unmapped_count == 3:
                    logger.info(
                        f"... (và {len([k for k in old_state_dict.keys() if k.startswith('vision_text_model.vision_model.encoder.')]) - 3} CLIP encoder layers khác - expected, không phải lỗi)"
                    )
                unmapped_keys.append(old_key)
                continue
            # Handle other CLIP vision model components
            else:
                # Try to map other components if they exist
                new_key = old_key.replace(
                    "vision_text_model.vision_model.", "image_embedding."
                )

            if new_key in new_model_state_dict:
                old_shape = old_state_dict[old_key].shape
                new_shape = new_model_state_dict[new_key].shape
                if old_shape == new_shape:
                    new_state_dict[new_key] = old_state_dict[old_key]
                    vision_mappings[old_key] = new_key
                else:
                    logger.warning(
                        f"Shape mismatch for {old_key}: {old_shape} vs {new_shape}"
                    )
                    unmapped_keys.append(old_key)
            else:
                # Only add to unmapped if it's not a known skip (like encoder layers)
                if "encoder" not in old_key:
                    unmapped_keys.append(old_key)

    logger.info(f"Mapped {len(vision_mappings)} vision encoder (CLIP) parameters")

    # 2.5. Try to map text encoder's encoder layers to new encoder
    # Old: text_encoder.text_model.encoder.* -> New: encoder.*
    encoder_mappings = {}
    for old_key in old_state_dict.keys():
        if old_key.startswith("text_encoder.text_model.encoder."):
            # Map RobertaEncoder layers from text encoder to new shared encoder
            new_key = old_key.replace("text_encoder.text_model.encoder.", "encoder.")
            if new_key in new_model_state_dict:
                old_shape = old_state_dict[old_key].shape
                new_shape = new_model_state_dict[new_key].shape
                if old_shape == new_shape:
                    new_state_dict[new_key] = old_state_dict[old_key]
                    encoder_mappings[old_key] = new_key
                else:
                    logger.warning(
                        f"Shape mismatch for {old_key}: {old_shape} vs {new_shape}"
                    )
                    unmapped_keys.append(old_key)
            else:
                unmapped_keys.append(old_key)

    logger.info(f"Mapped {len(encoder_mappings)} encoder layer parameters")

    # 2.6. Map text encoder's pooler to new pooler
    # Old: text_encoder.text_model.pooler.* -> New: pooler.*
    pooler_mappings = {}
    for old_key in old_state_dict.keys():
        if old_key.startswith("text_encoder.text_model.pooler."):
            # Map RobertaPooler from text encoder to new pooler
            new_key = old_key.replace("text_encoder.text_model.pooler.", "pooler.")
            if new_key in new_model_state_dict:
                old_shape = old_state_dict[old_key].shape
                new_shape = new_model_state_dict[new_key].shape
                if old_shape == new_shape:
                    new_state_dict[new_key] = old_state_dict[old_key]
                    pooler_mappings[old_key] = new_key
                else:
                    logger.warning(
                        f"Shape mismatch for {old_key}: {old_shape} vs {new_shape}"
                    )
                    unmapped_keys.append(old_key)
            else:
                unmapped_keys.append(old_key)

    logger.info(f"Mapped {len(pooler_mappings)} pooler parameters")

    # 3. Handle projection layers - MAP THEM to new model
    # These are pretrained and should be preserved!
    # Old model has: xlmr_text_projection, vision_projection_output, text_projection_output
    projection_mappings = {}
    projection_keys = [
        "xlmr_text_projection",
        "vision_projection_output",
        "text_projection_output",
    ]

    for proj_key in projection_keys:
        weight_key = f"{proj_key}.weight"
        bias_key = f"{proj_key}.bias"

        # Map to new model (they have the same names)
        if weight_key in old_state_dict:
            if weight_key in new_model_state_dict:
                old_shape = old_state_dict[weight_key].shape
                new_shape = new_model_state_dict[weight_key].shape
                if old_shape == new_shape:
                    new_state_dict[weight_key] = old_state_dict[weight_key]
                    projection_mappings[weight_key] = weight_key
                    logger.info(f"Mapped projection layer: {proj_key}.weight")
                else:
                    logger.warning(
                        f"Shape mismatch for {weight_key}: {old_shape} vs {new_shape}"
                    )
                    unmapped_keys.append(weight_key)
            else:
                unmapped_keys.append(weight_key)

        if bias_key in old_state_dict:
            if bias_key in new_model_state_dict:
                old_shape = old_state_dict[bias_key].shape
                new_shape = new_model_state_dict[bias_key].shape
                if old_shape == new_shape:
                    new_state_dict[bias_key] = old_state_dict[bias_key]
                    projection_mappings[bias_key] = bias_key
                    logger.info(f"Mapped projection layer: {proj_key}.bias")
                else:
                    logger.warning(
                        f"Shape mismatch for {bias_key}: {old_shape} vs {new_shape}"
                    )
                    unmapped_keys.append(bias_key)
            else:
                unmapped_keys.append(bias_key)

    # 4. Handle logit_scale and logit_bias - MAP THEM to new model
    # These are pretrained and should be preserved!
    logit_mappings = {}
    if "logit_scale" in old_state_dict:
        if "logit_scale" in new_model_state_dict:
            old_shape = old_state_dict["logit_scale"].shape
            new_shape = new_model_state_dict["logit_scale"].shape
            if old_shape == new_shape:
                new_state_dict["logit_scale"] = old_state_dict["logit_scale"]
                logit_mappings["logit_scale"] = "logit_scale"
                logger.info("Mapped logit_scale")
            else:
                unmapped_keys.append("logit_scale")
        else:
            unmapped_keys.append("logit_scale")

    if "logit_bias" in old_state_dict:
        if "logit_bias" in new_model_state_dict:
            old_shape = old_state_dict["logit_bias"].shape
            new_shape = new_model_state_dict["logit_bias"].shape
            if old_shape == new_shape:
                new_state_dict["logit_bias"] = old_state_dict["logit_bias"]
                logit_mappings["logit_bias"] = "logit_bias"
                logger.info("Mapped logit_bias")
            else:
                unmapped_keys.append("logit_bias")
        else:
            unmapped_keys.append("logit_bias")

    # 5. Handle map_head if exists
    map_head_mappings = {}
    for old_key in old_state_dict.keys():
        if "map_head" in old_key:
            # Try to map map_head if it exists in new model
            if old_key in new_model_state_dict:
                old_shape = old_state_dict[old_key].shape
                new_shape = new_model_state_dict[old_key].shape
                if old_shape == new_shape:
                    new_state_dict[old_key] = old_state_dict[old_key]
                    map_head_mappings[old_key] = old_key
                    logger.info(f"Mapped map_head parameter: {old_key}")
                else:
                    unmapped_keys.append(old_key)
            else:
                unmapped_keys.append(old_key)

    # 6. Handle other unmapped keys
    for old_key in old_state_dict.keys():
        if (
            old_key not in text_mappings
            and old_key not in vision_mappings
            and old_key not in encoder_mappings
            and old_key not in pooler_mappings
            and old_key not in projection_mappings
            and old_key not in logit_mappings
            and old_key not in map_head_mappings
            and old_key not in text_encoder_ref_mappings
            and old_key not in vision_encoder_ref_mappings
        ):
            if any(proj_key in old_key for proj_key in projection_keys):
                continue  # Already handled in projection_mappings
            if old_key not in unmapped_keys:
                unmapped_keys.append(old_key)
    logger.info(f"Mapped {len(projection_mappings)} projection layer parameters")
    logger.info(f"Mapped {len(logit_mappings)} logit parameters")
    logger.info(f"Mapped {len(map_head_mappings)} map_head parameters")

    # 7. Check for missing keys in new model
    # Note: _text_encoder.* and _vision_encoder.* keys that are NOT in new_state_dict
    # are keys that were not in the old checkpoint. These will be initialized from
    # pretrained models when creating TextEncoder and VisionEncoder, so we don't need to load them.
    for new_key in new_model_state_dict.keys():
        if new_key not in new_state_dict:
            # If it's a _text_encoder or _vision_encoder key that we didn't map,
            # it means it wasn't in the old checkpoint (will be initialized from pretrained)
            if new_key.startswith("_text_encoder.") or new_key.startswith(
                "_vision_encoder."
            ):
                # Only skip if we didn't map it (meaning it's not in old checkpoint)
                # If we mapped it, it should already be in new_state_dict
                continue
            missing_keys.append(new_key)

    # 8. Check if there are any keys from old model that we might have missed
    # Specifically check for vision_encoder.* (not vision_text_model.*) that might need mapping
    vision_encoder_keys = [
        k
        for k in old_state_dict.keys()
        if k.startswith("vision_encoder.") and not k.startswith("vision_text_model.")
    ]
    if vision_encoder_keys:
        logger.info(
            f"Found {len(vision_encoder_keys)} keys from vision_encoder.* (not vision_text_model.*)"
        )
        logger.info(
            f"These might need special mapping. First 5: {vision_encoder_keys[:5]}"
        )
        # Note: vision_encoder.* keys are from the VisionEncoder module in old model
        # They might contain pretrained weights that should be preserved
        # However, in the new model, we use PatchEmbedding which is initialized from CLIP config
        # So we might not need these weights, but we should log them for review

    logger.info(f"Conversion complete!")
    logger.info(f"Successfully mapped: {len(new_state_dict)} parameters")
    logger.info(
        f"  - Text encoder references: {len(text_encoder_ref_mappings)} (TRAINED - PRESERVED!)"
    )
    logger.info(
        f"  - Vision encoder references: {len(vision_encoder_ref_mappings)} (TRAINED - PRESERVED!)"
    )
    logger.info(f"  - Text embeddings: {len(text_mappings)}")
    logger.info(f"  - Vision embeddings: {len(vision_mappings)}")
    logger.info(f"  - Encoder layers: {len(encoder_mappings)}")
    logger.info(f"  - Pooler: {len(pooler_mappings)}")
    logger.info(
        f"  - Projection layers: {len(projection_mappings)} (PRETRAINED - PRESERVED!)"
    )
    logger.info(
        f"  - Logit scale/bias: {len(logit_mappings)} (PRETRAINED - PRESERVED!)"
    )
    logger.info(f"  - Map head: {len(map_head_mappings)}")
    logger.info(f"Unmapped keys: {len(unmapped_keys)}")
    logger.info(f"Missing keys: {len(missing_keys)}")

    if unmapped_keys:
        logger.warning(f"Some keys could not be mapped: {unmapped_keys[:10]}...")
    if missing_keys:
        logger.warning(f"Some new model keys are missing: {missing_keys[:10]}...")

    return new_state_dict, unmapped_keys, missing_keys


def load_and_convert_weights(
    old_checkpoint_path: str,
    new_model: nn.Module,
    output_path: Optional[str] = None,
    text_encoder_model_name: str = "FacebookAI/xlm-roberta-base",
    vision_model_name: str = "openai/clip-vit-base-patch32",
) -> Dict[str, torch.Tensor]:
    """
    Load old checkpoint and convert to new format

    Args:
        old_checkpoint_path: Path to old model checkpoint
        new_model: New model instance
        output_path: Optional path to save converted weights
        text_encoder_model_name: Text encoder model name
        vision_model_name: Vision model name

    Returns:
        Converted state dict
    """
    logger.info(f"Loading old checkpoint from {old_checkpoint_path}")

    # Load old checkpoint
    checkpoint = torch.load(old_checkpoint_path, map_location="cpu")

    # Extract model state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        old_state_dict = checkpoint["model_state_dict"]
    else:
        old_state_dict = checkpoint

    # Convert weights
    new_state_dict, unmapped_keys, missing_keys = convert_mm_encoder_weights(
        old_state_dict,
        new_model,
        text_encoder_model_name,
        vision_model_name,
    )

    # Save if output path provided
    if output_path:
        logger.info(f"Saving converted weights to {output_path}")
        torch.save(new_state_dict, output_path)
        logger.info("Conversion saved successfully!")

    return new_state_dict, unmapped_keys, missing_keys
