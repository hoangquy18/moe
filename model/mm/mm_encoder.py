import numpy as np
import torch
import torch.nn as nn
from typing import Literal
from model.modeling import MLP, MultiheadAttentionPoolingHead


class MultiModalEncoder(nn.Module):

    def __init__(self, text_encoder, vision_encoder, config):
        super(MultiModalEncoder, self).__init__()
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.text_projection = nn.Linear(
            self.text_encoder.text_config.hidden_size, config.hidden_size
        )
        self.vision_projection = nn.Linear(
            self.vision_encoder.vision_config.hidden_size, config.hidden_size
        )

        self.mm_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, nhead=config.num_attention_heads
        )
        self.mm_encoder = nn.TransformerEncoder(
            self.mm_encoder_layer, num_layers=config.num_encoder_layers
        )

        self.text_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.vision_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_bias = nn.Parameter(torch.ones([]) * -10)

        self.proj_type = config.proj_type
        if self.proj_type == "map":
            self.map_head = MultiheadAttentionPoolingHead(config)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.xavier_uniform_(self.vision_projection.weight)
        nn.init.constant_(self.text_projection.bias, 0)
        nn.init.constant_(self.vision_projection.bias, 0)

        nn.init.normal_(self.text_layernorm.weight, std=0.02)
        nn.init.normal_(self.text_layernorm.bias, std=0.02)
        nn.init.normal_(self.vision_layernorm.weight, std=0.02)
        nn.init.normal_(self.vision_layernorm.bias, std=0.02)

        self.map_head.initialize_parameters()

    def project_text_features(self, text_features: torch.Tensor) -> torch.Tensor:
        residual = text_features
        text_features = self.text_layernorm(text_features)
        text_features = self.text_projection(text_features)
        text_features = residual + text_features
        return text_features

    def project_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        residual = image_features
        image_features = self.vision_layernorm(image_features)
        image_features = self.vision_projection(image_features)
        image_features = residual + image_features
        return image_features

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

    def forward(
        self,
        text_input_ids,
        image_features,
        text_token_type_ids=None,
        text_attention_mask=None,
    ):
        text_features = self.text_encoder(
            input_ids=text_input_ids,
            token_type_ids=text_token_type_ids,
            attention_mask=text_attention_mask,
        )

        image_features = self.vision_encoder(image_features)

        B, text_len, _ = text_features.size()

        text_projection = self.project_text_features(text_features)
        image_projection = self.project_image_features(image_features)

        text_image_features = torch.cat((text_projection, image_projection), dim=1)
        mm_features = self.mm_encoder(text_image_features)

        mm_images = mm_features[:, text_len:, :]
        mm_texts = mm_features[:, :text_len, :]

        mm_images = self.feature_extraction(mm_images, self.proj_type)
        mm_texts = self.feature_extraction(mm_texts, self.proj_type)

        # normalized features
        mm_images = mm_images / mm_images.norm(dim=1, keepdim=True)
        mm_texts = mm_texts / mm_texts.norm(dim=1, keepdim=True)

        return mm_texts, mm_images
