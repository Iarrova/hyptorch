from typing import Union

import torch

from hyptorch.operations.tensor import squared_norm
from hyptorch.utils.validation import validate_curvature


def poincare_to_klein(x: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Map a point from the Poincaré ball model to the Klein model of hyperbolic space.

    This transformation preserves the hyperbolic geometry while converting the representation of a point from the conformal Poincaré model
    to the projective Klein model. The mapping is defined as:

    .. math::
        f(\\mathbf{x}) = \\frac{\\mathbf{x}}{\\sqrt{1 + c \\|\\mathbf{x}\\|^2}}

    where :math:`\\mathbf{x}` is a point on the Poincaré ball, :math:`c` is the (negative) curvature, and :math:`\\|\\mathbf{x}\\|` is the Euclidean norm.

    Parameters
    ----------
    x : torch.Tensor
        Point on the Poincaré ball.
    curvature : float or torch.Tensor
        Ball negative curvature.

    Returns
    -------
    torch.Tensor
        Corresponding point in the Klein model.
    """
    c = validate_curvature(curvature)
    return (2 * x) / (1 + c * squared_norm(x))


def klein_to_poincare(x: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Map a point from the Klein model to the Poincaré ball model of hyperbolic space.

    This transformation converts a point represented in the projective Klein model to its equivalent in the conformal Poincaré model.
    The mapping ensures that the hyperbolic structure is preserved across representations.

    The transformation is defined as:

    .. math::
        f(\\mathbf{x}) = \\frac{\\mathbf{x}}{1 + \\sqrt{1 - c \\|\\mathbf{x}\\|^2}}

    where :math:`\\mathbf{x}` is a point in the Klein model, :math:`c` is the (negative) curvature, and :math:`\\|\\mathbf{x}\\|` is the Euclidean norm.

    Parameters
    ----------
    x : torch.Tensor
        Point in the Klein model.
    curvature : float or torch.Tensor
        Ball negative curvature.

    Returns
    -------
    torch.Tensor
        Corresponding point on the Poincaré ball.
    """
    c = validate_curvature(curvature)
    return x / (1 + torch.sqrt(1 - c * squared_norm(x)))


def lorenz_factor(x: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Compute the Lorentz (gamma) factor for a point on the Klein disk model of hyperbolic space.

    The Lorentz factor arises in the hyperboloid and Klein models of hyperbolic geometry and is used to account for the distortion introduced
    by the curvature when transforming between Euclidean and hyperbolic representations.

    The Lorentz factor for a point :math:`\\mathbf{x}` with curvature :math:`c` is given by:

    .. math::
        \\gamma_{\\mathbf{x}}^c = \\frac{1}{\\sqrt{1 - c \\|\\mathbf{x}\\|^2}}

    where :math:`\\|\\mathbf{x}\\|` is the Euclidean norm of the point in the Klein disk model.

    Parameters
    ----------
    x : torch.Tensor
        Point on the Klein disk.
    curvature : float
        Negative curvature of the space.

    Returns
    -------
    torch.Tensor
        Lorentz factor at the input point.
    """
    return 1 / torch.sqrt(1 - curvature * squared_norm(x))
