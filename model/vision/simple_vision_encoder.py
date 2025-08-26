from typing import Literal
import torch
import torch.nn as nn
from model.activations import ACT2FN
from model.modeling import MLP, MultiheadAttentionPoolingHead
from model.vision.clip import CLIPVisionTransformer
from model.vision.sigclip import SiglipVisionTransformer
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPModel

MODEL_CONFIG_CLASSES = {
    "clip": CLIPVisionTransformer,
    "sigclip": SiglipVisionTransformer,
}


class VisionEncoder(nn.Module):
    """Simplified Vision Encoder for two-stage training without masking/distillation"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Determine vision model type from config
        if "sigclip" in config.vision_model_name.lower():
            self.vision_model_type = "sigclip"
        else:
            self.vision_model_type = "clip"

        # Initialize vision model
        if self.vision_model_type == "clip":
            self.vision_config = CLIPVisionConfig.from_pretrained(
                config.vision_model_name
            )
            self.vision_model = CLIPVisionTransformer(self.vision_config)
        else:
            # For SigLIP or other models
            self.vision_model = MODEL_CONFIG_CLASSES[self.vision_model_type](
                config.vision_model_name
            )

        # Add projection head if needed
        if config.proj_type == "map":
            self.map_head = MultiheadAttentionPoolingHead(self.vision_config)

    def feature_extraction(
        self,
        hidden_states: torch.Tensor,
        extract_type: Literal["patch", "cls_patch", "cls", "map", "gap"],
    ) -> torch.Tensor:
        """Extract features based on the specified type"""

        if extract_type == "patch":
            # Remove CLS token, keep only patch tokens
            hidden_states = hidden_states[:, 1:]
        elif extract_type == "cls":
            # Keep only CLS token
            hidden_states = hidden_states[:, 0]
        elif extract_type == "cls_patch":
            # Keep both CLS and patch tokens
            hidden_states = hidden_states
        elif extract_type == "map":
            # Use multihead attention pooling
            hidden_states = self.map_head(hidden_states)
        elif extract_type == "gap":
            # Global average pooling
            hidden_states = torch.mean(hidden_states, dim=1)
        else:
            raise ValueError(f"Unknown extract_type: {extract_type}")

        return hidden_states

    def forward(self, image_features, extract_type="cls"):
        """
        Forward pass through vision encoder

        Args:
            image_features: Input image tensor
            extract_type: Type of feature extraction to use

        Returns:
            Extracted image features
        """
        # Get hidden states from vision model
        if self.vision_model_type == "clip":
            # CLIP vision model returns tensor directly, not object with .last_hidden_state
            hidden_states = self.vision_model(pixel_values=image_features)
        else:
            # For other vision models
            hidden_states = self.vision_model(image_features)

        # Extract features based on type
        features = self.feature_extraction(hidden_states, extract_type)

        return features
