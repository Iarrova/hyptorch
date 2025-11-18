import torch.nn as nn
from src.config import Config
from src.enums import Backbone
from src.models.base import BaseModel

from hyptorch import HyperbolicMLR, HypLinear, PoincareBall, ToPoincare


class HyperbolicModel(BaseModel):
    _NAME = "HyperbolicModel"

    def __init__(self, config: Config, backbone: Backbone, num_classes: int):
        super().__init__(config, backbone, num_classes)
        self.config = config

        self.manifold = PoincareBall(
            curvature=self.config.hyperbolic.CURVATURE, trainable_curvature=self.config.hyperbolic.TRAINABLE_CURVATURE
        )
        self._build_hyperbolic_layers()

    def _build_hyperbolic_layers(self):
        self.feature_layer = nn.Sequential(
            ToPoincare(self.manifold),
            HypLinear(self.backbone_feature_dim, self.config.model.EMBEDDING_DIMENSION, self.manifold),
        )

        self.classifier = HyperbolicMLR(
            ball_dim=self.config.model.EMBEDDING_DIMENSION,
            n_classes=self.num_classes,
            manifold=self.manifold,
        )

    def forward(self, x):
        x = self._forward_backbone(x)
        features = self.feature_layer(x)
        logits = self.classifier(features)

        return logits, features
