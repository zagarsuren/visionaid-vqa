# ./modules/resnet_backbone.py

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Use ResNet50 pretrained on ImageNet.
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        # Remove the final fully-connected layer.
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.output_dim = 2048

    def forward(self, x):
        """
        x: [B, 3, H, W]
        Returns a tensor of shape [B, 1, output_dim] (treat the entire image as one token).
        """
        x = self.features(x)  # shape: [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # shape: [B, 2048]
        return x.unsqueeze(1)  # shape: [B, 1, 2048]
