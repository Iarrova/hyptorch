"""
Distance computations in hyperbolic space.
"""

from typing import Union

import numpy as np
import torch
from scipy.special import gamma

from hyptorch.pmath.autograd import artanh
from hyptorch.pmath.poincare import mobius_addition_batch


def distance_matrix(x: torch.Tensor, y: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Compute distance matrix between two sets of points.

    Parameters
    ----------
    x : tensor
        First set of points.
    y : tensor
        Second set of points.
    curvature : float or tensor
        Negative ball curvature.

    Returns
    -------
    tensor
        Distance matrix.
    """
    c = torch.as_tensor(curvature).type_as(x)
    sqrt_curvature = c**0.5

    mobius_addition_result = mobius_addition_batch(-x, y, curvature=c)
    norm = torch.norm(mobius_addition_result, dim=-1)

    return 2 / sqrt_curvature * artanh(sqrt_curvature * norm)


def auto_select_c(dimension: int) -> float:
    """
    Calculate the radius of the Poincare ball such that the d-dimensional ball has constant volume equal to pi.

    Parameters
    ----------
    d : int
        Dimension of the ball.

    Returns
    -------
    float
        Computed curvature.
    """
    dim2 = dimension / 2.0
    R = gamma(dim2 + 1) / (np.pi ** (dim2 - 1))
    R = R ** (1 / float(dimension))
    curvature = 1 / (R**2)
    return curvature
