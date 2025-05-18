from typing import Union

import torch


def compute_conformal_factor(x: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Compute the conformal factor for a point on the Poincaré ball.

    The conformal factor is used to scale the Euclidean metric into the hyperbolic metric, preserving angles between vectors.

    The conformal factor at point :math:`\\mathbf{x} \\in \\mathbb{D}_c^n` is given by:

    .. math::
        \\lambda_{\\mathbf{x}}^c = \\frac{2}{1 - c \\|\\mathbf{x}\\|^2}

    where :math:`c` is the (negative) curvature of the ball, and :math:`\\|\\mathbf{x}\\|` is the Euclidean norm.

    Parameters
    ----------
    x : torch.Tensor
        Point on the Poincaré ball.
    curvature : float or torch.Tensor
        Ball negative curvature.

    Returns
    -------
    torch.Tensor
        The conformal factor at the input point.
    """
    c = torch.as_tensor(curvature).type_as(x)
    return 2 / (1 - c * x.pow(2).sum(-1, keepdim=True))


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
    return 1 / torch.sqrt(1 - curvature * x.pow(2).sum(dim=-1, keepdim=True))
