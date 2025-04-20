from typing import Literal
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from model.activations import ACT2FN
from model.modeling import MLP, MultiheadAttentionPoolingHead

class TextEncoder(nn.Module):
    """Text Encoder."""

    def __init__(self, config):
        super().__init__()        

        self.config = config 
        
        self.text_config = AutoConfig.from_pretrained(config.text_model_name)
        self.text_model = AutoModel.from_pretrained(config.text_model_name, config=self.text_config)
        
        if config.proj_type == "map":
            self.map_head = MultiheadAttentionPoolingHead(self.text_config)
    
    def feature_extraction(self, 
                           hidden_states: torch.Tensor,
                           extract_type: Literal["patch", "cls_patch", "cls", 'map','gap']) -> torch.Tensor:
        
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
    
    def forward(self, input_ids, token_type_ids = None, attention_mask = None, extract_type = 'patch'):
        
        text_features = self.text_model(input_ids=input_ids,
                                         token_type_ids=token_type_ids,
                                         attention_mask=attention_mask)
        text_features = text_features.last_hidden_state
        text_features = self.feature_extraction(text_features, extract_type)
    
        return text_features
    