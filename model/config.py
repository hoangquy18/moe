from dataclasses import dataclass
from typing import Literal


@dataclass
class VisionConfig:
    """Simplified vision configuration for two-stage training"""

    vision_model_name: str = "openai/clip-vit-base-patch32"
    proj_type: Literal["patch", "cls_patch", "cls", "map", "gap"] = "cls"
    vision_model_weights: str = ""  # Path to pretrained weights (optional)
    load_vision_pretrained: bool = True  # Load from HuggingFace by default


@dataclass
class TextConfig:
    text_model_name: str = "FacebookAI/xlm-roberta-base"
    proj_type: Literal["patch", "cls_patch", "cls", "map", "gap"] = "patch"
    text_model_weights: str = ""
    load_text_pretrained: bool = True


@dataclass
class MultiModalConfig:
    proj_type: Literal["patch", "cls_patch", "cls", "map", "gap"] = "map"
    num_hidden_layers: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_encoder_layers: int = 12
    hidden_act: str = "gelu"
    max_position_embeddings: int = 32
    layer_norm_eps: float = 1e-5
    hidden_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    text_frozen: bool = False
    vision_frozen: bool = False
