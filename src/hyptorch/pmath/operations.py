from typing import Union

import torch

from hyptorch.config import EPS
from hyptorch.pmath.autograd import artanh
from hyptorch.pmath.mappings import project
from hyptorch.utils.numeric import safe_tanh


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
