from abc import ABC, abstractmethod

import torch

from hyptorch.exceptions import ManifoldError


class HyperbolicManifold(ABC):
    def __init__(self, curvature: float = 1.0) -> None:
        if curvature <= 0:
            raise ManifoldError(f"Curvature must be positive, got {curvature}")
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
    def exponential_map_at_origin(self, v: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def logarithmic_map_at_origin(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def mobius_matvec(self, matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        pass
