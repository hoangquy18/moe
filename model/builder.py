import os
import torch
from model.vision.vision_encoder import VisionEncoder, MaskedVisionEncoder
from model.text.text_encoder import TextEncoder
from model.mm.mm_encoder import MultiModalEncoder
from model.config import VisionConfig, TextConfig, MultiModalConfig


def build_vision_encoder(use_masking=False, vision_config=None):
    if vision_config is None:
        vision_config = VisionConfig()
    
    if use_masking:
        # Use MaskedVisionEncoder with specified configurations
        vision_encoder = MaskedVisionEncoder(vision_config)
    else:
        vision_encoder = VisionEncoder(vision_config)
        
    if vision_config.load_vision_pretrained:
        if os.path.exists(vision_config.vision_model_weights):
            # For MaskedVisionEncoder, only load weights for base encoder
            if use_masking:
                base_state_dict = torch.load(vision_config.vision_model_weights, map_location="cpu")
                # Filter state dict to only include keys for base vision_encoder
                filtered_state_dict = {
                    f"vision_encoder.{k}": v for k, v in base_state_dict.items()
                }
                vision_encoder.load_state_dict(filtered_state_dict, strict=False)
            else:
                vision_encoder.load_state_dict(
                    torch.load(vision_config.vision_model_weights, map_location="cpu"),
                    strict=False,
                )
    
    return vision_encoder


def build_text_encoder():
    text_config = TextConfig()
    text_encoder = TextEncoder(text_config)
    if text_config.load_text_pretrained:
        if os.path.exists(text_config.text_model_weights):
            text_encoder.load_state_dict(
                torch.load(text_config.text_model_weights, map_location="cpu"),
                strict=False,
            )

    return text_encoder


def build_model(use_masking=False, vision_config=None):
    vision_encoder = build_vision_encoder(use_masking=use_masking, vision_config=vision_config)
    text_encoder = build_text_encoder()

    multimodal_config = MultiModalConfig()
    multimodal_encoder = MultiModalEncoder(
        text_encoder, vision_encoder, multimodal_config
    )

    return multimodal_encoder
