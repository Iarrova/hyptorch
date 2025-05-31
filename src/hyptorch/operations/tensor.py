import torch

from hyptorch.config import NumericalConstants


def norm(tensor: torch.Tensor, *, safe: bool = False) -> torch.Tensor:
    norm = torch.linalg.norm(tensor, dim=-1, keepdim=True)
    if safe:
        return torch.clamp_min(norm, NumericalConstants.MIN_NORM_THRESHOLD)
    return norm


def squared_norm(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sum(tensor.pow(2), dim=-1, keepdim=True)


def dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, dim=-1, keepdim=True)
