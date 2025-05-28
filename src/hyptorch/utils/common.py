import torch
from torch.linalg import norm as linalg_norm

from hyptorch.config import MIN_NORM_THRESHOLD


def norm(tensor: torch.Tensor, safe: bool = False) -> torch.Tensor:
    """
    Compute the Euclidean (L2) norm of a tensor along the last dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.
    safe : bool, optional
        If True, clamps the norm to a minimum threshold to avoid division by zero or numerical instability.
        Default is False.

    Returns
    -------
    torch.Tensor
        The L2 norm of the input tensor.
    """
    if safe:
        return torch.clamp_min(torch.linalg.norm(tensor, dim=-1, keepdim=True), MIN_NORM_THRESHOLD)
    return linalg_norm(tensor, dim=-1, keepdim=True)


def squared_norm(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the squared L2 norm of a tensor along the specified dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.
    dim : int, optional
        The dimension along which to compute the squared norm. Default is -1 (last dimension).

    Returns
    -------
    torch.Tensor
        The squared L2 norm of the input tensor.
    """
    return torch.sum(tensor**2, dim=dim, keepdim=True)


def dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the dot product of two tensors along the last dimension.

    Parameters
    ----------
    x : torch.Tensor
        First input tensor.
    y : torch.Tensor
        Second input tensor.

    Returns
    -------
    torch.Tensor
        The dot product of the two tensors.
    """
    return (x * y).sum(dim=-1, keepdim=True)
