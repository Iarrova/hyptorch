from typing import Union

import torch

from hyptorch.config import EPS
from hyptorch.pmath.autograd import artanh
from hyptorch.pmath.operations import mobius_addition
from hyptorch.pmath.scales import compute_conformal_factor
from hyptorch.utils.numeric import safe_tanh


def project(x: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Safe projection on the manifold for numerical stability.

    Parameters
    ----------
    x : tensor
        Point on the Poincare ball.
    curvature : float or tensor
        Ball negative curvature.

    Returns
    -------
    tensor
        Projected vector on the manifold.
    """
    c = torch.as_tensor(curvature).type_as(x)
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), EPS)
    # TODO: Used to be EPS=1e-3, just in case of numerical instability
    maxnorm = (1 - EPS) / (c**0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def exponential_map_at_zero(
    tangent_vector: torch.Tensor, curvature: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Exponential map for Poincare ball model from 0.

    Parameters
    ----------
    tangent_vector : tensor
        Speed vector on poincare ball.
    curvature : float or tensor
        Ball negative curvature.

    Returns
    -------
    tensor
        End point gamma_{0,tangent_vector}(1).
    """
    c = torch.as_tensor(curvature).type_as(tangent_vector)
    sqrt_c = c**0.5

    tangent_norm = torch.clamp_min(tangent_vector.norm(dim=-1, keepdim=True, p=2), EPS)

    return safe_tanh(sqrt_c * tangent_norm) * tangent_vector / (sqrt_c * tangent_norm)


def exponential_map(
    x: torch.Tensor, tangent_vector: torch.Tensor, curvature: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Exponential map for Poincare ball model.

    Parameters
    ----------
    x : tensor
        Starting point on poincare ball.
    tangent_vector : tensor
        Speed vector on poincare ball.
    curvature : float or tensor
        Ball negative curvature.

    Returns
    -------
    tensor
        End point gamma_{x,tangent_vector}(1).
    """
    c = torch.as_tensor(curvature).type_as(x)
    sqrt_c = c**0.5

    tangent_norm = torch.clamp_min(tangent_vector.norm(dim=-1, keepdim=True, p=2), EPS)

    # Calculate lambda_x(x, curvature)
    conformal_factor = compute_conformal_factor(x, curvature=c, keepdim=True)

    # Calculate second term
    second_term = (
        safe_tanh(sqrt_c / 2 * conformal_factor * tangent_norm) * tangent_vector / (sqrt_c * tangent_norm)
    )

    # Calculate result using mobius addition
    return mobius_addition(x, second_term, curvature=c)


def logarithmic_map_at_zero(y: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Logarithmic map for y from 0 on the manifold.

    Parameters
    ----------
    y : tensor
        Target point on poincare ball.
    curvature : float or tensor
        Ball negative curvature.

    Returns
    -------
    tensor
        Tangent vector that transports 0 to y.
    """
    c = torch.as_tensor(curvature).type_as(y)
    sqrt_c = c**0.5

    y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), EPS)

    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def logarithmic_map(x: torch.Tensor, y: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Logarithmic map for two points x and y on the manifold.

    Parameters
    ----------
    x : tensor
        Starting point on poincare ball.
    y : tensor
        Target point on poincare ball.
    curvature : float or tensor
        Ball negative curvature.

    Returns
    -------
    tensor
        Tangent vector that transports x to y.
    """
    c = torch.as_tensor(curvature).type_as(x)

    # Calculate -x ⊕_c y
    difference_vector = mobius_addition(-x, y, curvature=c)

    difference_norm = difference_vector.norm(dim=-1, p=2, keepdim=True)

    # Calculate lambda_x(x, curvature)
    conformal_factor = compute_conformal_factor(x, curvature=c, keepdim=True)

    sqrt_c = c**0.5

    return (
        2 / sqrt_c / conformal_factor * artanh(sqrt_c * difference_norm) * difference_vector / difference_norm
    )
