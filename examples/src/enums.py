from enum import StrEnum


class Dataset(StrEnum):
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"


class Backbone(StrEnum):
    VGG16 = "VGG16"
    ResNet50 = "ResNet50"
    EfficientNetV2 = "EfficientNetV2"
