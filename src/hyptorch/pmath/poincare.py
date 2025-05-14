"""
Core operations in the Poincaré ball model of hyperbolic space.
"""

from typing import Union

import numpy as np
import torch
from scipy.special import gamma

from hyptorch.pmath.autograd import arsinh
from hyptorch.pmath.operations import mobius_addition_batch
from hyptorch.pmath.scalings import lorenz_factor
from hyptorch.pmath.transformations import klein_to_poincare, poincare_to_klein


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


def poincare_mean(
    x: torch.Tensor,
    curvature: Union[float, torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    """
    Compute mean in Poincare ball model.

    Parameters
    ----------
    x : tensor
        Points on Poincare ball.
    curvature : float or tensor
        Negative ball curvature.
    dim : int
        Dimension along which to compute mean.

    Returns
    -------
    tensor
        Mean in Poincare ball model.
    """
    # Convert to Klein model
    x_klein = poincare_to_klein(x, curvature)

    # Compute Lorenz factor
    lamb = lorenz_factor(x_klein, curvature=curvature, keepdim=True)

    # Compute weighted sum
    lamb_sum = torch.sum(lamb, dim=dim, keepdim=True)
    weighted_sum = torch.sum(lamb * x_klein, dim=dim, keepdim=True) / lamb_sum

    # Convert back to Poincare ball
    mean_poincare = klein_to_poincare(weighted_sum, curvature)

    return mean_poincare.squeeze(dim)


def hyperbolic_softmax(
    X: torch.Tensor,
    A: torch.Tensor,
    P: torch.Tensor,
    curvature: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Compute hyperbolic softmax.

    Parameters
    ----------
    X : tensor
        Input tensor.
    A : tensor
        Weights tensor.
    P : tensor
        Points tensor.
    curvature : float or tensor
        Negative ball curvature.

    Returns
    -------
    tensor
        Hyperbolic softmax result.
    """
    # Pre-compute common values
    lambda_pkc = 2 / (1 - curvature * P.pow(2).sum(dim=1))
    k = lambda_pkc * torch.norm(A, dim=1) / torch.sqrt(curvature)

    # Calculate mobius addition and other values
    mob_add = mobius_addition_batch(-P, X, curvature)

    num = 2 * torch.sqrt(curvature) * torch.sum(mob_add * A.unsqueeze(1), dim=-1)

    denom = torch.norm(A, dim=1, keepdim=True) * (1 - curvature * mob_add.pow(2).sum(dim=2))

    logit = k.unsqueeze(1) * arsinh(num / denom)

    return logit.permute(1, 0)
