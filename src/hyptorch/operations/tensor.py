import torch

from hyptorch.config import NumericalConstants


def norm(tensor: torch.Tensor, safe: bool = False) -> torch.Tensor:
    if safe:
        return torch.clamp_min(
            torch.linalg.norm(tensor, dim=-1, keepdim=True), NumericalConstants.MIN_NORM_THRESHOLD
        )
    return torch.linalg.norm(tensor, dim=-1, keepdim=True)


def squared_norm(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sum(tensor**2, dim=-1, keepdim=True)


def dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, dim=-1, keepdim=True)


