"""
Core operations in the Poincaré ball model of hyperbolic space.
"""

from typing import Union

import torch

from hyptorch.config import EPS
from hyptorch.pmath.autograd import arsinh, artanh
from hyptorch.pmath.mappings import klein_to_poincare, poincare_to_klein
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


def mobius_addition(x: torch.Tensor, y: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Mobius addition in hyperbolic space.
    In general, this operation is not commutative.

    Parameters
    ----------
    x : tensor
        Point on the Poincare ball.
    y : tensor
        Point on the Poincare ball.
    curvature : float or tensor
        Ball negative curvature.

    Returns
    -------
    tensor
        The result of mobius addition.
    """
    c = torch.as_tensor(curvature).type_as(x)

    # x2 and y2 are the squared norms of x and y
    # xy is the dot product of x and y
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c**2 * x2 * y2

    return numerator / (denominator + EPS)


def mobius_matrix_vector_multiplication(
    matrix: torch.Tensor, x: torch.Tensor, curvature: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Generalization for matrix-vector multiplication to hyperbolic space.

    Parameters
    ----------
    m : tensor
        Matrix for multiplication.
    x : tensor
        Point on poincare ball.
    curvature : float or tensor
        Negative ball curvature.

    Returns
    -------
    tensor
        Mobius matvec result.
    """
    c = torch.as_tensor(curvature).type_as(x)

    x_norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), EPS)
    sqrt_c = c**0.5

    mx = x @ matrix.transpose(-1, -2)
    mx_norm = mx.norm(dim=-1, keepdim=True, p=2)

    res_c = safe_tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)

    cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)

    return project(res, curvature=c)


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


def distance(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: Union[float, torch.Tensor],
    *,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    Distance on the Poincare ball.

    Parameters
    ----------
    x : tensor
        Point on poincare ball.
    y : tensor
        Point on poincare ball.
    curvature : float or tensor
        Ball negative curvature.
    keepdim : bool
        Retain the last dim? (default: false)

    Returns
    -------
    tensor
        Geodesic distance between x and y.
    """
    c = torch.as_tensor(curvature).type_as(x)
    sqrt_c = c**0.5

    dist_c = artanh(sqrt_c * mobius_addition(-x, y, c).norm(dim=-1, p=2, keepdim=keepdim))

    return dist_c * 2 / sqrt_c


def distance_from_center(
    x: torch.Tensor,
    curvature: Union[float, torch.Tensor],
    *,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    Distance on the Poincare ball to zero.

    Parameters
    ----------
    x : tensor
        Point on poincare ball.
    curvature : float or tensor
        Ball negative curvature.
    keepdim : bool
        Retain the last dim? (default: false)

    Returns
    -------
    tensor
        Geodesic distance between x and 0.
    """
    c = torch.as_tensor(curvature).type_as(x)
    sqrt_c = c**0.5
    dist_c = artanh(sqrt_c * x.norm(dim=-1, p=2, keepdim=keepdim))

    return dist_c * 2 / sqrt_c


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


def mobius_addition_batch(
    x: torch.Tensor, y: torch.Tensor, curvature: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Compute mobius addition in batch mode.

    Parameters
    ----------
    x : tensor
        First tensor (batch of points).
    y : tensor
        Second tensor (batch of points).
    curvature : float or tensor
        Negative ball curvature.

    Returns
    -------
    tensor
        Batch mobius addition result.
    """
    xy = torch.einsum("ij,kj->ik", (x, y))  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1

    num = 1 + 2 * curvature * xy + curvature * y2.permute(1, 0)  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - curvature * x2).unsqueeze(2) * y  # B x C x D

    denom_part1 = 1 + 2 * curvature * xy  # B x C
    denom_part2 = curvature**2 * x2 * y2.permute(1, 0)
    denom = denom_part1 + denom_part2

    return num / (denom.unsqueeze(2) + EPS)


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
