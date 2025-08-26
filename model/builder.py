import os
import torch
from model.vision.simple_vision_encoder import VisionEncoder
from model.text.text_encoder import TextEncoder
from model.mm.mm_encoder import MultiModalEncoder
from model.config import VisionConfig, TextConfig, MultiModalConfig


def build_vision_encoder(vision_config=None):
    """
    Build a simplified vision encoder for two-stage training
    """
    if vision_config is None:
        vision_config = VisionConfig()

    # Always use the simplified VisionEncoder
    vision_encoder = VisionEncoder(vision_config)

    # Load pretrained weights if specified
    if vision_config.load_vision_pretrained and vision_config.vision_model_weights:
        if os.path.exists(vision_config.vision_model_weights):
            try:
                state_dict = torch.load(
                    vision_config.vision_model_weights, map_location="cpu"
                )
                vision_encoder.load_state_dict(state_dict, strict=False)
                print(
                    f"Loaded vision pretrained weights from {vision_config.vision_model_weights}"
                )
            except Exception as e:
                print(f"Warning: Could not load vision weights: {e}")

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


def build_model(vision_config=None):
    """
    Build a simplified multimodal model for two-stage training
    """
    vision_encoder = build_vision_encoder(vision_config=vision_config)
    text_encoder = build_text_encoder()

    multimodal_config = MultiModalConfig()
    multimodal_encoder = MultiModalEncoder(
        text_encoder, vision_encoder, multimodal_config
    )

    return multimodal_encoder
