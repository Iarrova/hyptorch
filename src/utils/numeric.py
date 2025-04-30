"""
Numerical stability utilities for hyperbolic operations.
"""

import torch

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
