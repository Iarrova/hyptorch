import torch
import torch.nn as nn
from src.backbones.base import BaseBackbone
from torchvision import models


class ResNet50Backbone(BaseBackbone):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.feature_dim = 2048

        weights = "IMAGENET1K_V2" if pretrained else None
        backbone = models.resnet50(weights=weights)

        backbone.fc = nn.Identity()

        self.features = nn.Sequential(backbone, nn.Flatten(start_dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
