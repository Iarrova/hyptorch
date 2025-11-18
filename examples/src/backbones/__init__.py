from src.backbones.base import BaseBackbone
from src.backbones.efficientnet import EfficientNetV2Backbone
from src.backbones.resnet import ResNet50Backbone
from src.backbones.vgg import VGG16Backbone
from src.enums import Backbone

__all__ = ["VGG16Backbone", "ResNet50Backbone", "EfficientNetV2Backbone", "create_backbone"]


BACKBONE_MAPPING = {
    Backbone.VGG16: VGG16Backbone,
    Backbone.ResNet50: ResNet50Backbone,
    Backbone.EfficientNetV2: EfficientNetV2Backbone,
}


def create_backbone(name: Backbone, pretrained: bool = True) -> BaseBackbone:
    if name not in BACKBONE_MAPPING:
        raise ValueError(f"Backbone {name} is not supported. Available: {list(BACKBONE_MAPPING.keys())}")

    backbone_class = BACKBONE_MAPPING[name]
    return backbone_class(pretrained=pretrained)
