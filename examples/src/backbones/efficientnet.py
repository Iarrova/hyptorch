import torch
import torch.nn as nn
from src.backbones.base import BaseBackbone
from torchvision import models


class EfficientNetV2Backbone(BaseBackbone):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.feature_dim = 1280

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = models.efficientnet_v2_s(weights=weights)

        backbone.classifier = nn.Identity()

        self.features = nn.Sequential(backbone, nn.Flatten(start_dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
