"""
MultiModalEncoder with Vision-Language Architecture
Similar to ImageTextModel but compatible with old MultiModalEncoder weights
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Dict, Optional, Tuple

logger = logging.getLogger(__name__)
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaEncoder,
    RobertaPooler,
)
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers import CLIPVisionConfig, CLIPModel
from model.mm.patch_embedding import PatchEmbedding
from model.modeling import MLP, MultiheadAttentionPoolingHead


class ImageTextMultiModalEncoder(nn.Module):
    """
    Vision-Language MultiModal Encoder
    Similar to ImageTextModel but designed to be compatible with old MultiModalEncoder weights
    """

    def __init__(self, text_encoder, vision_encoder, config):
        super(ImageTextMultiModalEncoder, self).__init__()

        # Store original encoders for weight extraction
        self._text_encoder = text_encoder
        self._vision_encoder = vision_encoder
        self.config = config

        # Get text config from text encoder
        text_config = (
            text_encoder.text_config if hasattr(text_encoder, "text_config") else None
        )

        # Create RobertaConfig from text encoder config
        if text_config is None:
            # Fallback: create default config
            roberta_config = RobertaConfig(
                vocab_size=250002,
                hidden_size=(
                    config.hidden_size if hasattr(config, "hidden_size") else 768
                ),
                num_hidden_layers=(
                    config.num_hidden_layers
                    if hasattr(config, "num_hidden_layers")
                    else 12
                ),
                num_attention_heads=(
                    config.num_attention_heads
                    if hasattr(config, "num_attention_heads")
                    else 12
                ),
                intermediate_size=(
                    config.intermediate_size
                    if hasattr(config, "intermediate_size")
                    else 3072
                ),
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=514,
                type_vocab_size=1,
                initializer_range=0.02,
                layer_norm_eps=1e-5,
            )
        else:
            # Convert text config to RobertaConfig
            roberta_config = RobertaConfig(
                vocab_size=text_config.vocab_size,
                hidden_size=text_config.hidden_size,
                num_hidden_layers=text_config.num_hidden_layers,
                num_attention_heads=text_config.num_attention_heads,
                intermediate_size=text_config.intermediate_size,
                hidden_act=text_config.hidden_act,
                hidden_dropout_prob=text_config.hidden_dropout_prob,
                attention_probs_dropout_prob=text_config.attention_probs_dropout_prob,
                max_position_embeddings=text_config.max_position_embeddings,
                type_vocab_size=text_config.type_vocab_size,
                initializer_range=text_config.initializer_range,
                layer_norm_eps=text_config.layer_norm_eps,
            )

        # Set _attn_implementation for newer transformers versions
        # This is required to avoid KeyError when creating RobertaEncoder
        # Try to get from text_config if available, otherwise use default "eager"
        if (
            text_config is not None
            and hasattr(text_config, "_attn_implementation")
            and text_config._attn_implementation is not None
        ):
            roberta_config._attn_implementation = text_config._attn_implementation
        elif (
            not hasattr(roberta_config, "_attn_implementation")
            or roberta_config._attn_implementation is None
        ):
            # Default to "eager" (standard attention) for compatibility
            roberta_config._attn_implementation = "eager"

        self.roberta_config = roberta_config

        # Text embedding (RobertaEmbeddings)
        self.text_embedding = RobertaEmbeddings(roberta_config)

        # Image embedding (PatchEmbedding)
        # Get image size and patch size from CLIP config if available, otherwise from config or use defaults
        # Try to get from vision_encoder first
        clip_vision_config = None
        if hasattr(vision_encoder, "vision_config"):
            clip_vision_config = vision_encoder.vision_config
        elif hasattr(vision_encoder, "config") and hasattr(
            vision_encoder.config, "vision_config"
        ):
            clip_vision_config = vision_encoder.config.vision_config

        # If not found, try to load from CLIP model name in config
        if clip_vision_config is None:
            vision_model_name = getattr(
                config, "vision_model_name", "openai/clip-vit-base-patch32"
            )
            try:
                clip_vision_config = CLIPVisionConfig.from_pretrained(vision_model_name)
            except Exception as e:
                logger.warning(
                    f"Could not load CLIP config from {vision_model_name}: {e}"
                )

        # Get image_size and patch_size from CLIP config
        if clip_vision_config is not None:
            # CLIP config uses single int for image_size and patch_size
            clip_image_size = clip_vision_config.image_size
            clip_patch_size = clip_vision_config.patch_size
            # Convert to list format [height, width] for PatchEmbedding
            image_size = (
                [clip_image_size, clip_image_size]
                if isinstance(clip_image_size, int)
                else clip_image_size
            )
            patch_size = (
                [clip_patch_size, clip_patch_size]
                if isinstance(clip_patch_size, int)
                else clip_patch_size
            )
            logger.info(
                f"Using CLIP config: image_size={image_size}, patch_size={patch_size}"
            )
        else:
            # Fallback to config or defaults
            image_size = getattr(config, "image_size", [224, 224])
            patch_size = getattr(config, "patch_size", [32, 32])
            logger.info(
                f"Using config/default: image_size={image_size}, patch_size={patch_size}"
            )

        emb_dim = roberta_config.hidden_size

        self.image_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
        )

        # Shared encoder (RobertaEncoder)
        self.encoder = RobertaEncoder(roberta_config)

        # Pooler (optional)
        self.pooler = RobertaPooler(roberta_config)

        # Keep projection layers for backward compatibility and weight loading
        # These can be used to initialize from old model
        if hasattr(text_encoder, "text_config"):
            xlmr_text_dim = text_encoder.text_config.hidden_size
        else:
            xlmr_text_dim = roberta_config.hidden_size

        # Projection layers (for compatibility with old model)
        # These might be used during weight transfer
        self.xlmr_text_projection = nn.Linear(
            xlmr_text_dim, 512
        )  # 768 -> 512 (for CLIP alignment)
        self.vision_projection_output = nn.Linear(
            768, xlmr_text_dim
        )  # CLIP vision -> text dim
        self.text_projection_output = nn.Linear(
            512, xlmr_text_dim
        )  # CLIP text -> text dim

        # Logit scale and bias (for contrastive learning compatibility)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_bias = nn.Parameter(torch.ones([]) * -10)

        # Feature extraction type
        self.proj_type = getattr(config, "proj_type", "cls")
        if self.proj_type == "map":
            self.map_head = MultiheadAttentionPoolingHead(config)

    def get_num_patches(self):
        """Get number of image patches"""
        return self.image_embedding.num_patches

    def feature_extraction(
        self,
        hidden_states: torch.Tensor,
        extract_type: Literal["patch", "cls_patch", "cls", "map", "gap"],
    ) -> torch.Tensor:
        """Extract features based on type"""
        if extract_type == "patch":
            hidden_states = hidden_states[:, 1:]  # Remove CLS token
        elif extract_type == "cls":
            hidden_states = hidden_states[:, 0]  # Keep only CLS token
        elif extract_type == "cls_patch":
            hidden_states = hidden_states  # Keep all
        elif extract_type == "map":
            hidden_states = self.map_head(hidden_states)
        elif extract_type == "gap":
            hidden_states = torch.mean(hidden_states, dim=1)
        else:
            raise ValueError(f"Unknown extract_type: {extract_type}")
        return hidden_states

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # Legacy parameters for backward compatibility
        text_input_ids: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        text_token_type_ids: Optional[torch.Tensor] = None,
        return_embeddings_only: bool = False,
    ):
        """
        Forward pass for vision-language multimodal encoder

        Args:
            input_ids: Text input ids (new API)
            attention_mask: Attention mask (new API)
            token_type_ids: Token type ids (new API)
            image_input: Image input tensor (new API)
            text_input_ids: Text input ids (legacy API)
            image_features: Image features (legacy API)
            text_attention_mask: Text attention mask (legacy API)
            text_token_type_ids: Text token type ids (legacy API)
            return_embeddings_only: Whether to return only embeddings

        Returns:
            Model outputs
        """
        # Handle legacy API
        if text_input_ids is not None:
            input_ids = text_input_ids
        if image_features is not None:
            image_input = image_features
        if text_attention_mask is not None:
            attention_mask = text_attention_mask
        if text_token_type_ids is not None:
            token_type_ids = text_token_type_ids

        if input_ids is None:
            raise ValueError("input_ids or text_input_ids must be provided")
        if image_input is None:
            raise ValueError("image_input or image_features must be provided")

        return_dict = return_dict if return_dict is not None else True
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        # Create token type ids if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_ids.shape, dtype=torch.long, device=device
            )

        # Get text embeddings
        text_embeddings = self.text_embedding(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        # Get image embeddings
        image_embeddings = self.image_embedding(image_input)

        # Concatenate text and image embeddings
        # Text first, then image patches
        concat_embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)

        # Extend attention mask to cover image patches
        num_patches = image_embeddings.shape[1]
        image_attention_mask = torch.ones(
            (batch_size, num_patches), dtype=attention_mask.dtype, device=device
        )
        extended_attention_mask = torch.cat(
            [attention_mask, image_attention_mask], dim=1
        )

        # Prepare extended attention mask for transformer
        extended_attention_mask = extended_attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(
            dtype=concat_embeddings.dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            concat_embeddings.dtype
        ).min

        # Pass through encoder
        encoder_outputs = self.encoder(
            concat_embeddings,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = (
            encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state
        )

        # Pooled output
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if return_embeddings_only:
            # Return text and image features separately (for backward compatibility)
            # Extract text and image parts from sequence output
            text_output = sequence_output[:, :seq_length, :]
            image_output = sequence_output[:, seq_length:, :]

            # Apply feature extraction if needed
            text_features = self.feature_extraction(text_output, self.proj_type)
            image_features = self.feature_extraction(image_output, self.proj_type)

            return text_features, image_features

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=(
                encoder_outputs.hidden_states
                if hasattr(encoder_outputs, "hidden_states")
                else None
            ),
            attentions=(
                encoder_outputs.attentions
                if hasattr(encoder_outputs, "attentions")
                else None
            ),
        )

    def get_text_features(
        self, text_input_ids, text_attention_mask=None, text_token_type_ids=None
    ):
        """Get text features (for backward compatibility)"""
        text_embeddings = self.text_embedding(
            input_ids=text_input_ids,
            token_type_ids=text_token_type_ids,
            attention_mask=text_attention_mask,
        )

        # Pass through encoder (text only)
        batch_size, seq_length = text_input_ids.shape
        device = text_input_ids.device

        if text_attention_mask is None:
            text_attention_mask = torch.ones((batch_size, seq_length), device=device)

        extended_attention_mask = text_attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(
            dtype=text_embeddings.dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            text_embeddings.dtype
        ).min

        encoder_outputs = self.encoder(
            text_embeddings,
            attention_mask=extended_attention_mask,
            return_dict=True,
        )

        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.feature_extraction(sequence_output, self.proj_type)

        return sequence_output, pooled_output

    def get_image_features(self, image_features):
        """Get image features (for backward compatibility)"""
        image_embeddings = self.image_embedding(image_features)

        # Pass through encoder (image only)
        batch_size = image_features.shape[0]
        device = image_features.device
        num_patches = image_embeddings.shape[1]

        attention_mask = torch.ones((batch_size, num_patches), device=device)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(
            dtype=image_embeddings.dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            image_embeddings.dtype
        ).min

        encoder_outputs = self.encoder(
            image_embeddings,
            attention_mask=extended_attention_mask,
            return_dict=True,
        )

        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.feature_extraction(sequence_output, self.proj_type)

        return sequence_output, pooled_output
