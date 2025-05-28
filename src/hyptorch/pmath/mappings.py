from typing import Union

import torch

from hyptorch.config import MAX_NORM_SCALE
from hyptorch.pmath.autograd import artanh, tanh
from hyptorch.pmath.scalings import compute_conformal_factor
from hyptorch.utils.common import norm
from hyptorch.utils.validation import validate_curvature


def project(x: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Project a point onto the Poincaré ball manifold to maintain numerical stability.

    During optimization or computation in hyperbolic space, numerical errors can cause
    points to drift slightly outside the valid manifold. This function safely projects such
    points back inside the Poincaré ball, ensuring that all points lie within the allowable
    radius defined by the curvature.

    In the Poincaré ball model with curvature :math:`-c`, the manifold is the open ball of radius
    :math:`\\frac{1}{\\sqrt{c}}`. Any point :math:`\\mathbf{x} \\in \\mathbb{R}^n` with norm greater than this radius lies outside the
    manifold. This function scales such points to lie just inside the boundary.

    Projection is done using the formula:

    .. math::

        \\text{proj}(\\mathbf{x}) =
        \\begin{cases}
            \\frac{x}{\\|x\\|} \\cdot r_{\\text{max}} & \\text{if } \\|x\\| > r_{\\text{max}} \\
            x & \\text{otherwise}
        \\end{cases}
        \\quad \\text{where} \\quad r_{\\text{max}} = \\frac{1 - \\epsilon}{\\sqrt{c}}
        
    where :math:`\\epsilon` is a small constant to ensure the point lies strictly within the ball.

    Parameters
    ----------
    x : torch.Tensor
        Point on the Poincaré ball.
    curvature : float or torch.Tensor
        Ball negative curvature.

    Returns
    -------
    torch.Tensor
        A projected point lying strictly within the Poincaré ball.
    """
    c = validate_curvature(curvature)

    # TODO: Check the impact of changing the value of 1e-3
    r_max = (MAX_NORM_SCALE) / torch.sqrt(c)
    x_norm = norm(x, safe=True)

    return torch.where(x_norm > r_max, x / x_norm * r_max, x)


def exponential_map(x: torch.Tensor, v: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Compute the exponential map on the Poincaré ball manifold.

    The exponential map takes a vector :math:`\\mathbf{v} \\in T_x \\mathbb{D}^n_c` (the tangent space at point :math:`\\mathbf{x}`)
    and maps it to a point :math:`\\textbf{y} \\in \\mathbb{D}^n_c`, where :math:`\\mathbb{D}^n_c` is the Poincaré ball model of hyperbolic space (i.e., the manifold).

    The exponential map is used to move along geodesics starting at :math:`\\mathbf{x}` in the direction of a given tangent vector.

    The exponential map from point :math:`\\mathbf{x}` with tangent vector :math:`\\mathbf{v}` is given by:

    .. math::

        \\exp_{\\mathbf{x}}^c(\\mathbf{v}) =
        \\mathbf{x} \\oplus_c \\left( \\tanh\\left(\\sqrt{c} \\frac{\\lambda_{\\mathbf{x}}^c \\|\\mathbf{v}\\|}{2}\\right)
        \\frac{\\mathbf{v}}{\\sqrt{c}\\|\\mathbf{v}\\|} \\right)

    where :math:`\\lambda_{\\mathbf{x}}^c = \\frac{2}{1 - c \\|\\mathbf{x}\\|^2}` is the conformal factor and
    :math:`\\oplus_c` denotes Möbius addition under curvature :math:`-c`.

    Parameters
    ----------
    x : torch.Tensor
        Base point on the Poincaré ball manifold.
    v : torch.Tensor
        Tangent vector at `x` indicating the direction and magnitude of movement.
    curvature : float or torch.Tensor
        Ball negative curvature.

    Returns
    -------
    torch.Tensor
        The resulting point on the Poincaré ball after applying the exponential map.
    """
    from hyptorch.pmath.operations import mobius_addition

    c = validate_curvature(curvature)
    sqrt_c = torch.sqrt(c)

    v_norm = norm(v, safe=True)
    lambda_x = compute_conformal_factor(x, curvature=c)

    return mobius_addition(x, tanh(sqrt_c * lambda_x * v_norm / 2) * v / (sqrt_c * v_norm), curvature=c)


def exponential_map_at_zero(v: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Compute the exponential map on the Poincaré ball manifold at 0.

    Parameters
    ----------
    v : torch.Tensor
        Tangent vector at `x` indicating the direction and magnitude of movement.
    curvature : float or torch.Tensor
        Ball negative curvature.

    Returns
    -------
    torch.Tensor
        The resulting point on the Poincaré ball after applying the exponential map.
    """
    c = validate_curvature(curvature)
    sqrt_c = torch.sqrt(c)

    v_norm = norm(v, safe=True)

    return tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)


def logarithmic_map(x: torch.Tensor, y: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Compute the logarithmic map on the Poincaré ball manifold.

    The logarithmic map is the inverse of the exponential map. It maps a point
    :math:`\\mathbf{y} \\in \\mathbb{D}^n_c` on the Poincaré ball (the manifold)
    back to a vector :math:`\\mathbf{v} \\in T_x \\mathbb{D}^n_c` in the tangent space
    at a base point :math:`\\mathbf{x} \\in \\mathbb{D}^n_c`.

    This tangent vector describes the initial velocity of the geodesic starting at
    :math:`\\mathbf{x}` that reaches :math:`\\mathbf{y}`.

    The logarithmic map from point :math:`\\mathbf{x}` to :math:`\\mathbf{y}` is given by:

    .. math::

        \\log_{\\mathbf{x}}^c(\\mathbf{y}) =
        \\frac{2}{\\sqrt{c} \\lambda_{\\mathbf{x}}^c}
        \\text{arctanh}\\left( \\sqrt{c} \\| -\\mathbf{x} \\oplus_c \\mathbf{y} \\| \\right)
        \\frac{-\\mathbf{x} \\oplus_c \\mathbf{y}}{\\| -\\mathbf{x} \\oplus_c \\mathbf{y} \\|}

    where :math:`\\lambda_{\\mathbf{x}}^c = \\frac{2}{1 - c \\|\\mathbf{x}\\|^2}` is the conformal factor and
    :math:`\\oplus_c` denotes Möbius addition under curvature :math:`-c`.

    Parameters
    ----------
    x : torch.Tensor
        Base point on the Poincaré ball manifold (starting point of the geodesic).
    y : torch.Tensor
        Target point on the Poincaré ball manifold (endpoint of the geodesic).
    curvature : float or torch.Tensor
        Ball negative curvature.

    Returns
    -------
    torch.Tensor
        Tangent vector at `x` pointing toward `y`, representing the geodesic direction and magnitude.
    """
    from hyptorch.pmath.operations import mobius_addition

    c = validate_curvature(curvature)
    sqrt_c = torch.sqrt(c)

    xy = mobius_addition(-x, y, curvature=c)
    xy_norm = norm(xy)
    lambda_x = compute_conformal_factor(x, curvature=c)

    return 2 / (sqrt_c * lambda_x) * artanh(sqrt_c * xy_norm) * xy / xy_norm


def logarithmic_map_at_zero(y: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Compute the logarithmic map on the Poincaré ball manifold at 0.

    Parameters
    ----------
    y : torch.Tensor
        Target point on Poincaré ball.
    curvature : float or torch.Tensor
        Ball negative curvature.

    Returns
    -------
    torch.Tensor
        Tangent vector that transports 0 to y.
    """
    c = validate_curvature(curvature)
    sqrt_c = torch.sqrt(c)

    y_norm = norm(y, safe=True)

    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)
