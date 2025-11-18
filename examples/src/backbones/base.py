from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.feature_dim = 0

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
