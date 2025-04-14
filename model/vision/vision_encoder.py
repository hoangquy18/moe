from typing import Literal
import torch
import torch.nn as nn
from model.activations import ACT2FN
from model.modeling import MLP, MultiheadAttentionPoolingHead
from model.vision.clip import CLIPVisionTransformer
from model.vision.sigclip import SiglipVisionTransformer
from transformers import CLIPImageProcessor, CLIPVisionConfig

MODEL_CONFIG_CLASSES = {
    "clip": CLIPVisionTransformer,
    "sigclip": SiglipVisionTransformer
}

class VisionEncoder(nn.Module):
    """Vision Encoder."""

    def __init__(self, config):
        super().__init__()        

        self.config = config 
        if "sigclip" in config.vision_model_name:
            vision_type = "sigclip"
        elif "clip" in config.vision_model_name:
            vision_type = "clip"
        else:
            raise ValueError(f"Unknown vision model name: {config.vision_model_name}")
        
        self.vision_config = CLIPVisionConfig.from_pretrained(config.vision_model_name)
        self.vision_model = MODEL_CONFIG_CLASSES[vision_type](self.vision_config)
        self.image_processor = CLIPImageProcessor.from_pretrained(config.vision_model_name)
        
        if config.proj_type:
            self.map_head = MultiheadAttentionPoolingHead(self.vision_config)
    
    def feature_extraction(self, 
                           hidden_states: torch.Tensor,
                           extract_type: Literal["patch", "cls_patch", "cls", 'map','gap']) -> torch.Tensor:
        """
        Extract features from the image.
        Args:
            pixel_values: The input image tensor.
        Returns:
            The extracted features.
        """
        
        if extract_type == 'patch':
            hidden_states = hidden_states[:, 1:]
        elif extract_type == 'cls':
            hidden_states = hidden_states[:, 0]
        elif extract_type == 'cls_patch':
            hidden_states = hidden_states
        elif extract_type == 'map':
            hidden_states = self.map_head(hidden_states)
        elif extract_type == 'gap':
            hidden_states = torch.mean(hidden_states, dim=1)
        else:
            raise ValueError(f"Unknown extract_type: {extract_type}")
        
        return hidden_states
    
    def forward(self, image_features, extract_type):
        
        images_features = self.vision_model(image_features)
        images_features = self.feature_extraction(images_features, extract_type)
    
        return images_features
    