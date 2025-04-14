import torch
import torch.nn as nn

class MultiModal(nn.Module):
    
    def __init__(self, text_encoder, vision_encoder):
        super(MultiModal, self).__init__()
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        
    def forward(self, text_features, image_features):
        text_features = self.text_encoder(text_features)
        image_features = self.vision_encoder(image_features)
        
        mm_features = torch.cat((text_features, image_features), dim=1)
        
        