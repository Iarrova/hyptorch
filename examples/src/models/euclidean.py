import torch
import torch.nn as nn
from src.config import Config
from src.enums import Backbone
from src.models.base import BaseModel


class EuclideanModel(BaseModel):
    _NAME = "EuclideanModel"

    def __init__(self, config: Config, backbone: Backbone, num_classes: int):
        super().__init__(config, backbone, num_classes)
        self.config = config

        self._build_euclidean_layers()

    def _build_euclidean_layers(self):
        self.feature_layer = nn.Linear(self.backbone_feature_dim, self.config.model.EMBEDDING_DIMENSION)
        self.classifier = nn.Linear(self.config.model.EMBEDDING_DIMENSION, self.num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._forward_backbone(x)
        features = self.feature_layer(x)
        logits = self.classifier(features)

        return logits, features
