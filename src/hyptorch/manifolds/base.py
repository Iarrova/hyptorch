from abc import ABC, abstractmethod

import torch


class HyperbolicManifold(ABC):
    def __init__(self, curvature: float = 1.0):
        self._curvature = torch.tensor(curvature, dtype=torch.float32)

    @property
    def curvature(self) -> torch.Tensor:
        return self._curvature

    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def exponential_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def exponential_map_at_zero(self, v: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def logarithmic_map_at_zero(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def matrix_vector_multiplication(self, matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        pass
