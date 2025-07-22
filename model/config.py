from dataclasses import dataclass
from typing import Literal


@dataclass
class VisionConfig:
    vision_model_name: str = "openai/clip-vit-base-patch32"
    proj_type: Literal["patch", "cls_patch", "cls", "map", "gap"] = "patch"
    vision_model_weights: str = "weights/vision/clip/clip-vit-base-patch32.bin"
    load_vision_pretrained: bool = True
    # Masking configuration
    mask_ratio: float = 0.4
    mask_strategy: Literal["random", "block"] = "random"
    # Self-distillation configuration
    use_self_distillation: bool = False
    teacher_model_name: str = None  # Name of the teacher model (if different from student)
    teacher_momentum_base: float = 0.994  # Initial momentum value
    teacher_momentum_final: float = 1.0   # Final momentum value
    distillation_alpha: float = 1.0  # α = 1, weight for distillation loss
    masking_beta: float = 2.0        # β = 2, weight for masking loss
    # Local crop parameters for self-distillation
    use_local_crops: bool = True     # Whether to use local crops for self-distillation
    num_local_crops: int = 4         # Number of local crops to use (M)
    local_crop_size: int = 98        # Size of local crops (98x98 pixels)


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
