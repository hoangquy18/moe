from dataclasses import dataclass
from typing import Literal

@dataclass
class VisionConfig:
    vision_model_name: str = "openai/clip-vit-base-patch32"
    proj_type: Literal["patch", "cls_patch", "cls", 'map','gap'] = "patch"
    vision_model_weights: str = "weights/vision/clip/clip-vit-base-patch32.bin"
    load_vision_pretrained: bool = True
    

@dataclass
class TextConfig:
    text_model_name: str = "FacebookAI/xlm-roberta-base"
    proj_type: Literal["patch", "cls_patch", "cls", 'map','gap'] = "patch"
    text_model_weights: str = ""
    load_text_pretrained: bool = True
    
@dataclass
class MultiModalConfig:
    pass