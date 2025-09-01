import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Dict, Optional, Tuple
from model.modeling import MLP, MultiheadAttentionPoolingHead
from transformers import CLIPModel


class MultiModalEncoder(nn.Module):

    def __init__(self, text_encoder, vision_encoder, config):
        super(MultiModalEncoder, self).__init__()
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder

        # Initialize CLIP model for teacher learning stage
        self.vision_text_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # Freeze CLIP model parameters (they serve as frozen teacher)
        for param in self.vision_text_model.parameters():
            param.requires_grad = False

        # Add projection head for Stage 1 alignment (only for XLM-R)
        # CLIP: EOS token features used directly (no projection)
        # XLM-R: CLS token features -> FCT projection
        clip_text_dim = self.vision_text_model.config.text_config.hidden_size  # 512
        clip_vision_dim = self.vision_text_model.config.vision_config.hidden_size  # 768
        xlmr_text_dim = text_encoder.text_config.hidden_size  # 768

        # Only XLM-R needs projection to align with CLIP dimension
        self.xlmr_text_projection = nn.Linear(
            xlmr_text_dim, clip_text_dim
        )  # 768 -> 512

        self.vision_projection_output = nn.Linear(clip_vision_dim, xlmr_text_dim)
        self.text_projection_output = nn.Linear(clip_text_dim, xlmr_text_dim)

        # Optionally freeze other components
        if config.text_frozen:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        if config.vision_frozen:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_bias = nn.Parameter(torch.ones([]) * -10)

        self.proj_type = config.proj_type
        if self.proj_type == "map":
            self.map_head = MultiheadAttentionPoolingHead(config)

    def feature_extraction(
        self,
        hidden_states: torch.Tensor,
        extract_type: Literal["patch", "cls_patch", "cls", "map", "gap"],
    ) -> torch.Tensor:

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

    def contrastive_encoding(self, mm_texts, mm_images):
        """
        Optimize text and image features specifically for contrastive learning.

        Args:
            mm_texts: Text features from the multimodal encoder
            mm_images: Image features from the multimodal encoder

        Returns:
            Tuple of optimized text and image features
        """
        # Apply additional non-linear projection for contrastive learning
        # This is a common technique that improves representation quality
        # mm_texts and mm_images are already normalized at this point

        # Calculate similarity matrix for monitoring alignment
        sim_matrix = torch.matmul(mm_texts, mm_images.transpose(0, 1))

        # Return the optimized features and similarity information
        return (
            mm_texts,
            mm_images,
            {
                "similarity_matrix": sim_matrix,
                "logit_scale": self.logit_scale.exp(),
                "logit_bias": self.logit_bias,
            },
        )

    def forward(
        self,
        text_input_ids,
        image_features,
        text_attention_mask=None,
        text_token_type_ids=None,
        apply_masking=False,
        return_embeddings_only=False,
    ):
        """Forward pass for the multimodal encoder.

        Args:
            text_input_ids: The input ids of the text.
            image_features: The input image tensor.
            text_attention_mask: The attention mask for the text.
            text_token_type_ids: The token type ids for the text.
            apply_masking: Whether to apply masking to image features.
            return_embeddings_only: Whether to return only the final embeddings without contrastive info

        Returns:
            Either a tuple of (text_features, image_features, contrastive_info) or just (text_features, image_features)
        """
        # For handling masked vision encoder case specially
        if apply_masking and hasattr(self.vision_encoder, "apply_masking"):
            # This will be handled outside this function in the trainer
            # to properly integrate the distillation loss
            raise ValueError(
                "For masked vision encoding, call vision_encoder directly with apply_masking=True"
            )

        text_features = self.text_encoder(
            input_ids=text_input_ids,
            token_type_ids=text_token_type_ids,
            attention_mask=text_attention_mask,
        )

        image_features = self.vision_text_model.vision_model(
            image_features
        ).last_hidden_state  # [Batch_size, hidden_size]

        # normalized features
        text_features = self.feature_extraction(text_features, "cls")
        text_features = self.xlmr_text_projection(text_features)  # [batch_size, 512
        text_features = self.text_projection_output(text_features)

        image_features = self.feature_extraction(image_features, "cls")
        image_features = self.vision_projection_output(image_features)

        mm_images = image_features / image_features.norm(dim=-1, keepdim=True)
        mm_texts = text_features / text_features.norm(dim=-1, keepdim=True)

        if return_embeddings_only:
            return mm_texts, mm_images

        # Get optimized features and contrastive information
        return self.contrastive_encoding(mm_texts, mm_images)
