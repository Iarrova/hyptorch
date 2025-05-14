"""
Numerical stability utilities for hyperbolic operations.
"""

import torch
from torch.linalg import norm as linalg_norm

from hyptorch.config import TANH_CLAMP


def safe_tanh(x: torch.Tensor, clamp: float = TANH_CLAMP) -> torch.Tensor:
    """
    Numerically stable implementation of tanh.

    Parameters
    ----------
    x : tensor
        Input tensor.
    clamp : float
        Clamping value to ensure numerical stability.

    Returns
    -------
    tensor
        Tanh of the input tensor.
    """
    return x.clamp(-clamp, clamp).tanh()


def norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the Euclidean (L2) norm of a tensor along the last dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The L2 norm of the input tensor.
    """
    return linalg_norm(tensor, dim=-1, keepdim=True)
