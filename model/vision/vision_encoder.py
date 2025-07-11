from typing import Literal, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from model.activations import ACT2FN
from model.modeling import MLP, MultiheadAttentionPoolingHead
from model.vision.clip import CLIPVisionTransformer
from model.vision.sigclip import SiglipVisionTransformer
from transformers import CLIPImageProcessor, CLIPVisionConfig

MODEL_CONFIG_CLASSES = {
    "clip": CLIPVisionTransformer,
    "sigclip": SiglipVisionTransformer,
}


class MaskedVisionEncoder(nn.Module):
    """Vision Encoder with masking capabilities for self-supervised learning."""

    def __init__(self, config):
        super().__init__()
        self.vision_config = CLIPVisionConfig.from_pretrained(config.vision_model_name)
        self.config = config
        self.mask_ratio = getattr(config, "mask_ratio", 0.4)
        self.mask_strategy = getattr(config, "mask_strategy", "random")
        self.teacher_momentum = getattr(config, "teacher_momentum", 0.999)
        self.use_self_distillation = getattr(config, "use_self_distillation", False)
        self.distillation_alpha = getattr(config, "distillation_alpha", 0.5)
        
        # Initialize the base vision encoder
        self.vision_encoder = VisionEncoder(config)
        
        # Create a momentum teacher model for self-distillation if enabled
        if self.use_self_distillation:
            self.teacher_model = VisionEncoder(config)
            # Initialize teacher with the same parameters as student
            self.teacher_model.load_state_dict(self.vision_encoder.state_dict())
            # Freeze the teacher model
            for param in self.teacher_model.parameters():
                param.requires_grad = False
    
    def _get_mask(self, batch_size, seq_len):
        """Generate a binary mask based on the selected strategy."""
        if self.mask_strategy == "random":
            # Random masking
            mask = torch.rand(batch_size, seq_len) < self.mask_ratio
        elif self.mask_strategy == "block":
            # Block masking (masking contiguous regions)
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
            block_size = int(math.sqrt(seq_len * self.mask_ratio))
            for b in range(batch_size):
                # Random starting position for the block
                start_idx = random.randint(0, seq_len - block_size)
                mask[b, start_idx:start_idx+block_size] = True
        else:
            raise ValueError(f"Unsupported masking strategy: {self.mask_strategy}")
        
        return mask
    
    def apply_mask(self, hidden_states, mask=None):
        """Apply mask to hidden states."""
        batch_size, seq_len, dim = hidden_states.shape
        
        # Generate mask if not provided
        if mask is None:
            mask = self._get_mask(batch_size, seq_len)
            
        # Convert mask to device of hidden_states
        mask = mask.to(hidden_states.device)
        
        # Create a mask token (learnable or fixed)
        mask_token = torch.zeros_like(hidden_states[:, 0, :]).unsqueeze(1)
        
        # Apply masking: replace masked tokens with mask token
        masked_indices = mask.unsqueeze(-1).expand_as(hidden_states)
        masked_hidden_states = hidden_states.clone()
        masked_hidden_states = torch.where(masked_indices, mask_token, masked_hidden_states)
        
        return masked_hidden_states, mask
    
    def update_teacher(self):
        """Update teacher model using momentum update."""
        if not self.use_self_distillation:
            return
            
        with torch.no_grad():
            for student_param, teacher_param in zip(
                self.vision_encoder.parameters(), self.teacher_model.parameters()
            ):
                teacher_param.data.mul_(self.teacher_momentum).add_(
                    student_param.data * (1 - self.teacher_momentum)
                )
    
    def distillation_loss(self, student_features, teacher_features):
        """Calculate self-distillation loss between student and teacher outputs."""
        # Normalize features
        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)
        
        # Cosine similarity loss
        cos_sim = torch.sum(student_features * teacher_features, dim=-1).mean()
        loss = 1.0 - cos_sim
        
        return loss
    
    def forward(self, image_features, extract_type="patch", apply_masking=False):
        """
        Forward pass with optional masking and self-distillation.
        Args:
            image_features: The input image tensor.
            extract_type: Type of feature extraction.
            apply_masking: Whether to apply masking to the input.
        """
        if apply_masking:
            # Process with masking for self-supervised learning
            # Get hidden states from the base encoder first
            with torch.no_grad():
                hidden_states = self.vision_encoder.vision_model(image_features)
                
            # Apply masking to the hidden states
            masked_hidden_states, mask = self.apply_mask(hidden_states)
            
            # Process masked hidden states
            masked_features = self.vision_encoder.feature_extraction(
                masked_hidden_states, extract_type
            )
            
            # If using self-distillation, get teacher predictions
            if self.use_self_distillation:
                with torch.no_grad():
                    # Teacher processes original unmasked input
                    teacher_hidden_states = self.teacher_model.vision_model(image_features)
                    teacher_features = self.teacher_model.feature_extraction(
                        teacher_hidden_states, extract_type
                    )
                
                # Calculate distillation loss
                dist_loss = self.distillation_loss(masked_features, teacher_features)
                
                return masked_features, {
                    "mask": mask,
                    "distillation_loss": dist_loss
                }
            
            return masked_features, {"mask": mask}
        
        else:
            # Regular forward pass without masking
            features = self.vision_encoder(image_features, extract_type)
            return features


class VisionEncoder(nn.Module):
    """Vision Encoder."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        if "siglip" in config.vision_model_name or "sigclip" in config.vision_model_name:
            vision_type = "sigclip"
        elif "clip" in config.vision_model_name:
            vision_type = "clip"
        else:
            raise ValueError(f"Unknown vision model name: {config.vision_model_name}")

        self.vision_config = CLIPVisionConfig.from_pretrained(config.vision_model_name)
        self.vision_model = MODEL_CONFIG_CLASSES[vision_type](self.vision_config)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            config.vision_model_name
        )

        if config.proj_type == "map":
            self.map_head = MultiheadAttentionPoolingHead(self.vision_config)

    def feature_extraction(
        self,
        hidden_states: torch.Tensor,
        extract_type: Literal["patch", "cls_patch", "cls", "map", "gap"],
    ) -> torch.Tensor:
        """
        Extract features from the image.
        Args:
            pixel_values: The input image tensor.
        Returns:
            The extracted features.
        """

        if extract_type == "patch":
            hidden_states = hidden_states[:, 1:]
        elif extract_type == "cls":
            hidden_states = hidden_states[:, 0]
        elif extract_type == "cls_patch":
            hidden_states = hidden_states
        elif extract_type == "map":
            hidden_states = self.map_head(hidden_states)
        elif extract_type == "gap":
            hidden_states = torch.mean(hidden_states, dim=1)
        else:
            raise ValueError(f"Unknown extract_type: {extract_type}")

        return hidden_states

    def forward(self, image_features, extract_type="patch"):

        images_features = self.vision_model(image_features)
        images_features = self.feature_extraction(images_features, extract_type)

        return images_features
