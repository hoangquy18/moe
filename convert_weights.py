"""
Script to convert weights from old MultiModalEncoder to new ImageTextMultiModalEncoder
"""
import argparse
import torch
import logging
from pathlib import Path

from model.builder import build_model
from model.config import VisionConfig, TextConfig, MultiModalConfig
from model.mm.mm_encoder_vl import ImageTextMultiModalEncoder
from model.mm.weight_converter import load_and_convert_weights
from model.text.text_encoder import TextEncoder
from model.vision.simple_vision_encoder import VisionEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Convert MultiModalEncoder weights to ImageTextMultiModalEncoder")
    parser.add_argument(
        "--old_checkpoint",
        type=str,
        required=True,
        help="Path to old MultiModalEncoder checkpoint"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save converted weights"
    )
    parser.add_argument(
        "--text_model_name",
        type=str,
        default="FacebookAI/xlm-roberta-base",
        help="Text encoder model name"
    )
    parser.add_argument(
        "--vision_model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Vision encoder model name"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image size [height, width]"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=2,
        default=[32, 32],
        help="Patch size [height, width]"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Weight Conversion Script")
    logger.info("=" * 60)
    logger.info(f"Old checkpoint: {args.old_checkpoint}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Text model: {args.text_model_name}")
    logger.info(f"Vision model: {args.vision_model_name}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Patch size: {args.patch_size}")
    
    # Check if old checkpoint exists
    if not Path(args.old_checkpoint).exists():
        raise FileNotFoundError(f"Old checkpoint not found: {args.old_checkpoint}")
    
    # Build old model structure to understand config
    logger.info("\nBuilding old model structure...")
    vision_config = VisionConfig()
    text_config = TextConfig()
    multimodal_config = MultiModalConfig()
    
    # Build encoders
    vision_encoder = VisionEncoder(vision_config)
    text_encoder = TextEncoder(text_config)
    
    # Create new model
    logger.info("Creating new ImageTextMultiModalEncoder...")
    
    # Set image size and patch size in config
    multimodal_config.image_size = args.image_size
    multimodal_config.patch_size = args.patch_size
    
    new_model = ImageTextMultiModalEncoder(
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        config=multimodal_config
    )
    
    logger.info(f"New model created with {sum(p.numel() for p in new_model.parameters())} parameters")
    
    # Convert weights
    logger.info("\nConverting weights...")
    new_state_dict, unmapped_keys, missing_keys = load_and_convert_weights(
        old_checkpoint_path=args.old_checkpoint,
        new_model=new_model,
        output_path=None,  # We'll save manually
        text_encoder_model_name=args.text_model_name,
        vision_model_name=args.vision_model_name,
    )
    
    # Load converted weights into new model
    logger.info("\nLoading converted weights into new model...")
    missing, unexpected = new_model.load_state_dict(new_state_dict, strict=False)
    
    if missing:
        logger.warning(f"Missing keys in new model: {len(missing)}")
        logger.warning(f"First 10 missing keys: {missing[:10]}")
    
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")
        logger.warning(f"First 10 unexpected keys: {unexpected[:10]}")
    
    # Save converted model
    logger.info(f"\nSaving converted model to {args.output_path}...")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save full model state dict
    torch.save({
        'model_state_dict': new_model.state_dict(),
        'config': {
            'text_model_name': args.text_model_name,
            'vision_model_name': args.vision_model_name,
            'image_size': args.image_size,
            'patch_size': args.patch_size,
            'multimodal_config': multimodal_config.__dict__,
        },
        'conversion_info': {
            'unmapped_keys_count': len(unmapped_keys),
            'missing_keys_count': len(missing_keys),
            'unmapped_keys_sample': unmapped_keys[:20],
            'missing_keys_sample': missing_keys[:20],
        }
    }, args.output_path)
    
    logger.info("=" * 60)
    logger.info("Conversion completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Converted model saved to: {args.output_path}")
    logger.info(f"Unmapped keys: {len(unmapped_keys)}")
    logger.info(f"Missing keys: {len(missing_keys)}")
    
    # Print summary
    if unmapped_keys:
        logger.info("\nSample unmapped keys (from old model):")
        for key in unmapped_keys[:10]:
            logger.info(f"  - {key}")
    
    if missing_keys:
        logger.info("\nSample missing keys (in new model):")
        for key in missing_keys[:10]:
            logger.info(f"  - {key}")


if __name__ == "__main__":
    main()

