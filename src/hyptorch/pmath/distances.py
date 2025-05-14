"""
Distance computations in hyperbolic space.
"""

from typing import Union

import torch

from hyptorch.pmath.autograd import artanh
from hyptorch.pmath.operations import mobius_addition, mobius_addition_batch
from hyptorch.utils.numeric import norm


def distance(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Distance between two points on the Poincaré ball.
    The distance is computed using the formula:
    .. math::
        d_c(\\mathbf{x}, \\mathbf{y}) = \\frac{2}{\\sqrt{c}} * arctanh(\\sqrt{c} \\|\\mathbf{-x} \\oplus_{c} \\mathbf{y}\\|)
    where :math:`c` is the curvature of the ball, and :math:`\\oplus_{c}` is the Mobius addition operation.


    Parameters
    ----------
    x : tensor
        Point on Poincaré ball.
    y : tensor
        Point on Poincaré ball.
    curvature : float or tensor
        Ball negative curvature.

    Returns
    -------
    tensor
        Geodesic distance between x and y.
    """
    c = torch.as_tensor(curvature).type_as(x)
    sqrt_c = c**0.5

    dist_c = artanh(sqrt_c * norm(mobius_addition(-x, y, c)))

    return dist_c * 2 / sqrt_c


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
