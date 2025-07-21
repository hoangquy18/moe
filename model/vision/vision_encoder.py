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
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPModel

MODEL_CONFIG_CLASSES = {
    "clip": CLIPVisionTransformer,
    "sigclip": SiglipVisionTransformer,
}


class ProjectionHead(nn.Module):
    """
    Projection head for self-distillation and masking as specified in Sec 3.2:
    3-layer MLP, l2 normalization, and weight-normalized projection
    """
    def __init__(self, input_dim, hidden_dim, output_dim=32000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Weight-normalized projection layer
        self.projection = nn.utils.weight_norm(
            nn.Linear(hidden_dim, output_dim, bias=False)
        )
        
        # EMA of means for centering
        self.register_buffer("center", torch.zeros(output_dim))
        self.momentum = 0.9  # Constant momentum for EMA of means
        
        # Temperature parameters
        self.student_temp = 0.1  # τs
        self.student_temp_prime = 0.1  # τ's
        self.teacher_temp = 0.07  # τt
        self.register_buffer("teacher_temp_prime", torch.tensor(0.04))  # τ't starting value
        
    def forward(self, x, is_student=True, step=0, total_steps=1):
        """
        Apply projection head with centering and sharpening
        
        Args:
            x: Input features
            is_student: Whether processing student or teacher features
            step: Current training step (for temperature scheduling)
            total_steps: Total training steps (for temperature scheduling)
        """
        # Handle potential 3D input (batch_size, seq_len, hidden_dim)
        orig_shape = x.shape
        if len(orig_shape) == 3:
            # Flatten sequence dimension for processing
            batch_size, seq_len, hidden_dim = orig_shape
            x = x.reshape(-1, hidden_dim)
        
        # Apply MLP layers
        x = self.mlp(x)
        
        # L2 normalization
        x = F.normalize(x, dim=-1)
        
        # Apply weight-normalized projection
        logits = self.projection(x)
        
        # Update teacher_temp_prime with linear warmup if needed
        if not is_student and self.training:
            warmup_factor = min(1.0, step / total_steps)
            self.teacher_temp_prime = torch.tensor(0.04 + warmup_factor * (0.07 - 0.04))
        
        # Apply centering for student
        if is_student and self.training:
            # Update center EMA
            batch_center = logits.mean(dim=0)
            self.center = self.momentum * self.center + (1 - self.momentum) * batch_center.detach()
            
            # Apply centering
            logits = logits - self.center
        
        # Apply sharpening (temperature scaling)
        if is_student:
            # For student features
            logits = logits / self.student_temp if self.training else logits / self.student_temp_prime
        else:
            # For teacher features
            logits = logits / self.teacher_temp if self.training else logits / self.teacher_temp_prime
            
        return logits


class MaskedVisionEncoder(nn.Module):
    """Vision Encoder with masking capabilities for self-supervised learning."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mask_ratio = getattr(config, "mask_ratio", 0.4)
        self.mask_strategy = getattr(config, "mask_strategy", "random")
        self.teacher_momentum_base = getattr(config, "teacher_momentum_base", 0.994)
        self.teacher_momentum_final = getattr(config, "teacher_momentum_final", 1.0)
        self.use_self_distillation = getattr(config, "use_self_distillation", False)
        self.distillation_alpha = getattr(config, "distillation_alpha", 1.0)  # α = 1
        self.masking_beta = getattr(config, "masking_beta", 2.0)  # β = 2
        
        # Initialize the base vision encoder
        self.vision_encoder = VisionEncoder(config)
        self.vision_config = CLIPVisionConfig.from_pretrained(config.vision_model_name)
        
        # Initialize projection heads for self-distillation and masking
        if self.use_self_distillation:
            student_hidden_size = self.vision_encoder.vision_config.hidden_size
            self.student_projection = ProjectionHead(student_hidden_size, student_hidden_size)
            self.masking_projection = ProjectionHead(student_hidden_size, student_hidden_size)
            
            # Create a teacher model that can be different from student
            if hasattr(config, "teacher_model_name") and config.teacher_model_name:
                # Use a different model for the teacher
                teacher_config = config
                teacher_config.vision_model_name = config.teacher_model_name
                self.teacher_model = VisionEncoder(teacher_config)
                self.teacher_vision_config = CLIPVisionConfig.from_pretrained(config.teacher_model_name)
                teacher_hidden_size = self.teacher_vision_config.hidden_size
            else:
                # Use the same model architecture for teacher and student
                self.teacher_model = VisionEncoder(config)
                teacher_hidden_size = student_hidden_size
            
            # Create teacher projection with appropriate hidden size
            self.teacher_projection = ProjectionHead(teacher_hidden_size, student_hidden_size)
            
            # If using the same model, copy weights from student to teacher
            if not hasattr(config, "teacher_model_name") or not config.teacher_model_name:
                self.teacher_model.load_state_dict(self.vision_encoder.state_dict())
                self.teacher_projection.load_state_dict(self.student_projection.state_dict())
            
            # Freeze the teacher model
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            for param in self.teacher_projection.parameters():
                param.requires_grad = False
                
            # Register buffer for tracking current momentum value
            self.register_buffer("current_momentum", torch.tensor(self.teacher_momentum_base))
    
    def _get_cosine_momentum(self, step, max_steps):
        """Calculate cosine schedule momentum from 0.994 to 1.0"""
        min_momentum = self.teacher_momentum_base
        max_momentum = self.teacher_momentum_final
        
        # Cosine schedule
        cosine_factor = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        momentum = max_momentum - (max_momentum - min_momentum) * cosine_factor
        return momentum
    
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
    
    def update_teacher(self, step=0, max_steps=1):
        """Update teacher model using momentum update with cosine schedule."""
        if not self.use_self_distillation:
            return
        
        # Calculate current momentum value based on cosine schedule
        self.current_momentum = torch.tensor(self._get_cosine_momentum(step, max_steps))
            
        with torch.no_grad():
            # Update vision encoder
            for student_param, teacher_param in zip(
                self.vision_encoder.parameters(), self.teacher_model.parameters()
            ):
                teacher_param.data.mul_(self.current_momentum).add_(
                    student_param.data * (1 - self.current_momentum)
                )
                
            # Update projection head
            for student_param, teacher_param in zip(
                self.student_projection.parameters(), self.teacher_projection.parameters()
            ):
                teacher_param.data.mul_(self.current_momentum).add_(
                    student_param.data * (1 - self.current_momentum)
                )
    
    def distillation_loss(self, student_features, teacher_features, step=0, total_steps=1):
        """Calculate self-distillation loss between student and teacher outputs."""
        # Ensure features are 2D tensors by pooling if necessary
        if len(student_features.shape) == 3:
            # Pool over sequence dimension (simple mean pooling)
            student_features = torch.mean(student_features, dim=1)  # [batch_size, hidden_dim]
            
        if len(teacher_features.shape) == 3:
            # Pool over sequence dimension (simple mean pooling)
            teacher_features = torch.mean(teacher_features, dim=1)  # [batch_size, hidden_dim]
        
        # Process through projection heads - these handle the different hidden sizes
        student_projections = self.student_projection(student_features, is_student=True, step=step, total_steps=total_steps)
        with torch.no_grad():
            teacher_projections = self.teacher_projection(teacher_features, is_student=False, step=step, total_steps=total_steps)
        
        # Self-distillation loss as cross-entropy
        # Compute similarity matrix
        sim = student_projections @ teacher_projections.t()  # Now both are 2D tensors
        
        # Cross-entropy loss
        labels = torch.arange(sim.size(0), device=sim.device)
        loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)
        loss = loss / 2  # Average of the two directions
        
        return loss
        
    def masking_loss(self, masked_features, original_features, mask, step=0, total_steps=1):
        """Calculate masking reconstruction loss."""
        # Ensure features are 2D tensors by pooling if necessary
        if len(masked_features.shape) == 3:
            # Pool over sequence dimension (simple mean pooling)
            masked_features = torch.mean(masked_features, dim=1)  # [batch_size, hidden_dim]
            
        if len(original_features.shape) == 3:
            # Pool over sequence dimension (simple mean pooling)
            original_features = torch.mean(original_features, dim=1)  # [batch_size, hidden_dim]
        
        # Process through projection heads
        masked_projections = self.masking_projection(masked_features, is_student=True, step=step, total_steps=total_steps)
        original_projections = self.student_projection(original_features, is_student=True, step=step, total_steps=total_steps).detach()
        
        # Compute similarity matrix
        sim = masked_projections @ original_projections.t()  # Now both are 2D tensors
        
        # Cross-entropy loss
        labels = torch.arange(sim.size(0), device=sim.device)
        loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)
        loss = loss / 2  # Average of the two directions
        
        return loss
    
    def forward(self, image_features, extract_type="patch", apply_masking=False, step=0, total_steps=1):
        """
        Forward pass with optional masking and self-distillation.
        Args:
            image_features: The input image tensor.
            extract_type: Type of feature extraction.
            apply_masking: Whether to apply masking to the input.
            step: Current training step.
            total_steps: Total training steps.
        """
        if apply_masking and self.use_self_distillation:
            # Get original features from student model
            original_hidden_states = self.vision_encoder.vision_model(pixel_values=image_features).last_hidden_state
            
            # Apply masking to hidden states
            masked_hidden_states, mask = self.apply_mask(original_hidden_states.clone())
            
            # Process masked hidden states
            masked_features = self.vision_encoder.feature_extraction(
                masked_hidden_states, extract_type
            )
            
            # Process original hidden states 
            original_features = self.vision_encoder.feature_extraction(
                original_hidden_states, extract_type
            )
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_hidden_states = self.teacher_model.vision_model(image_features)
                teacher_features = self.teacher_model.feature_extraction(
                    teacher_hidden_states, extract_type
                )
            
            # Calculate distillation loss
            dist_loss = self.distillation_loss(original_features, teacher_features, step, total_steps)
            
            # Calculate masking loss
            mask_loss = self.masking_loss(masked_features, original_features, mask, step, total_steps)
            
            # Return masked features for downstream tasks
            return masked_features, {
                "mask": mask,
                "distillation_loss": dist_loss,
                "masking_loss": mask_loss,
                "original_features": original_features
            }
        elif apply_masking:
            # Simpler path when not using self-distillation
            hidden_states = self.vision_encoder.vision_model(image_features)
            masked_hidden_states, mask = self.apply_mask(hidden_states)
            masked_features = self.vision_encoder.feature_extraction(
                masked_hidden_states, extract_type
            )
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
        self.vision_model = CLIPModel.from_pretrained(
            config.vision_model_name 
        )
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

        images_features = self.vision_model(pixel_values=image_features).last_hidden_state
        images_features = self.feature_extraction(images_features, extract_type)

        return images_features
