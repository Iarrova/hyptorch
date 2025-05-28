from typing import Union

import torch

from hyptorch.config import EPS
from hyptorch.pmath.autograd import artanh, tanh
from hyptorch.pmath.mappings import project
from hyptorch.pmath.scalings import lorenz_factor
from hyptorch.pmath.transformations import klein_to_poincare, poincare_to_klein
from hyptorch.utils.common import dot_product, norm, squared_norm
from hyptorch.utils.validation import validate_curvature


def mobius_addition(x: torch.Tensor, y: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Mobius addition in hyperbolic space.

    This operation is defined as:

    .. math::
        \\mathbf{x} \\oplus_{c} \\mathbf{y} = \\frac{(1 + 2c \\langle \\mathbf{x}, \\mathbf{y} \\rangle + c \\|\\mathbf{y}\\|^2) \\mathbf{x} + (1 - c \\|\\mathbf{x}\\|^2) \\mathbf{y}}{1 + 2c \\langle \\mathbf{x}, \\mathbf{y} \\rangle + c^2 \\|\\mathbf{x}\\|^2 \\|\\mathbf{y}\\|^2}

    where :math:`\\langle ., .\\rangle` is the inner product, and :math:`\\|.\\|` is the norm.

    Parameters
    ----------
    x, y : torch.Tensor
        Points on the Poincaré ball.
    curvature : float or torch.Tensor
        Ball negative curvature.

    Returns
    -------
    torch.Tensor
        Result of Möbius addition.
    """
    c = validate_curvature(curvature)

    x2 = squared_norm(x)
    y2 = squared_norm(y)
    xy = dot_product(x, y)

    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c**2 * x2 * y2

    return numerator / (denominator + EPS)


def batch_mobius_addition(
    x: torch.Tensor, y: torch.Tensor, curvature: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Mobius addition for batches of points in hyperbolic space.

    Parameters
    ----------
    x : torch.Tensor
        First tensor (batch of points).
    y : torch.Tensor
        Second tensor (batch of points).
    curvature : float or torch.Tensor
        Negative ball curvature.

    Returns
    -------
    torch.Tensor
        Batch mobius addition result.
    """
    xy = torch.einsum("ij,kj->ik", (x, y))  # B x C
    x2 = squared_norm(x)  # B x 1
    y2 = squared_norm(y)  # C x 1

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
    Generalized matrix-vector multiplication in hyperbolic space.

    This operation extends standard matrix-vector multiplication to the Poincaré ball model of hyperbolic geometry.

    Given a matrix :math:`M` and a point :math:`\\mathbf{x} \\in \\mathbb{D}_c^n`, the Möbius matrix-vector multiplication is defined as:

    .. math::
        M \\otimes_c \\mathbf{x} = \\frac{1}{\\sqrt{c}}\\tanh\\left(\\frac{\\|M\\mathbf{x}\\|}{\\|\\mathbf{x}\\|}\\tanh^{-1}{\\sqrt{c}\\|\\mathbf{x}\\|}\\right)\\frac{M \\mathbf{x}}{\\|M \\mathbf{x}\\|}



    Parameters
    ----------
    m : torch.Tensor
        Matrix used for the Möbius multiplication.
    x : torch.Tensor
        Point on the Poincaré ball.
    curvature : float or torch.Tensor
        Ball negative curvature.

    Returns
    -------
    torch.Tensor
        Result of the Möbius matrix-vector multiplication.
    """
    c = validate_curvature(curvature)
    sqrt_c = torch.sqrt(c)

    x_norm = norm(x, safe=True)

    mx = x @ matrix.transpose(-1, -2)
    mx_norm = norm(mx)

    res_c = (1 / sqrt_c) * tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * (mx / mx_norm)

    # Handle the case where mx is zero
    # This is done to avoid division by zero in the case of zero vectors
    cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)

    return project(res, curvature=c)


def poincare_mean(x: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Compute the weighted mean of points in the Poincaré ball model of hyperbolic space.

    The mean is calculated by first mapping points from the Poincaré model to the Klein model,
    where averaging is simpler due to the model's affine structure. A Lorentz (conformal) factor
    is used to appropriately weight each point in accordance with the underlying hyperbolic geometry.
    The resulting average is then mapped back to the Poincaré ball.

    The weighted mean in the Klein model is given by:

    .. math::
        \\text{HypAve}(\\mathbf{x}_1, \\dots, \\mathbf{x}_n) = \\frac{\\sum_{i=1}^{n} \\lambda_i \\mathbf{x}_i}{\\sum_{i=1}^{n} \\lambda_i}

    where :math:`\\lambda_i` is the Lorentz factor for point :math:`\\mathbf{x}_i`.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of points on the Poincaré ball.
    curvature : float or torch.Tensor
        Ball negative curvature.

    Returns
    -------
    torch.Tensor
        Mean point in the Poincaré ball model.
    """
    x_klein = poincare_to_klein(x, curvature)
    lamb = lorenz_factor(x_klein, curvature=curvature)

    mean = torch.sum(lamb * x_klein, dim=0, keepdim=True) / torch.sum(lamb, dim=0, keepdim=True)
    mean_poincare = klein_to_poincare(mean, curvature)

    return mean_poincare.squeeze(0)
