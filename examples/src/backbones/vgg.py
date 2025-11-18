import torch
import torch.nn as nn
from src.backbones.base import BaseBackbone
from torchvision import models


class VGG16Backbone(BaseBackbone):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.feature_dim = 512

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = models.vgg16_bn(weights=weights)

        self.features = nn.Sequential(backbone.features, nn.Flatten(start_dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
