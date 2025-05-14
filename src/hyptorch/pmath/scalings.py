"""
Scales for Poincare and Klein models.
This module contains functions to compute the conformal factor and Lorentz factor
for points in the Poincare and Klein models of hyperbolic geometry.
"""

from typing import Union

import torch


def compute_conformal_factor(
    x: torch.Tensor, curvature: Union[float, torch.Tensor], *, keepdim: bool = False
) -> torch.Tensor:
    """
    Compute the conformal factor for a point on the ball.

    Parameters
    ----------
    x : tensor
        Point on the Poincare ball.
    curvature : float or tensor
        Ball negative curvature.
    keepdim : bool
        Retain the last dim? (default: false)

    Returns
    -------
    tensor
        Conformal factor.
    """
    c = torch.as_tensor(curvature).type_as(x)
    return 2 / (1 - c * x.pow(2).sum(-1, keepdim=keepdim))


def lorenz_factor(
    x: torch.Tensor,
    curvature: Union[float, torch.Tensor],
    *,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    Compute Lorenz factor.

    Parameters
    ----------
    x : tensor
        Point on Klein disk.
    curvature : float
        Negative curvature.
    dim : int
        Dimension to calculate Lorenz factor.
    keepdim : bool
        Retain the last dim? (default: false)

    Returns
    -------
    tensor
        Lorenz factor.
    """
    return 1 / torch.sqrt(1 - curvature * x.pow(2).sum(dim=dim, keepdim=keepdim))
