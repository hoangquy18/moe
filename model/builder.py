import os
import torch
from model.vision.vision_encoder import VisionEncoder, MaskedVisionEncoder
from model.text.text_encoder import TextEncoder
from model.mm.mm_encoder import MultiModalEncoder
from model.config import VisionConfig, TextConfig, MultiModalConfig


def build_vision_encoder(use_masking=False):
    vision_config = VisionConfig()

    if use_masking:
        # Add masking-specific configuration
        vision_config.mask_ratio = 0.4
        vision_config.mask_strategy = "random"
        vision_config.use_self_distillation = True
        vision_config.teacher_momentum = 0.999
        vision_config.distillation_alpha = 0.5

        vison_encoder = MaskedVisionEncoder(vision_config)
    else:
        vison_encoder = VisionEncoder(vision_config)

    if vision_config.load_vision_pretrained:
        if os.path.exists(vision_config.vision_model_weights):
            vison_encoder.load_state_dict(
                torch.load(vision_config.vision_model_weights, map_location="cpu"),
                strict=False,
            )
    return vison_encoder


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


def build_model(use_masking=False):
    vision_encoder = build_vision_encoder(use_masking=use_masking)
    text_encoder = build_text_encoder()

    multimodal_config = MultiModalConfig()
    multimodal_encoder = MultiModalEncoder(
        text_encoder, vision_encoder, multimodal_config
    )

    return multimodal_encoder
