from typing import Union

import torch

from hyptorch.pmath.autograd import artanh
from hyptorch.pmath.operations import mobius_addition
from hyptorch.utils.numeric import norm


def distance(x: torch.Tensor, y: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Distance between two points on the Poincaré ball.

    The distance is computed using the formula:

    .. math::
        d_c(\\mathbf{x}, \\mathbf{y}) = \\frac{2}{\\sqrt{c}} \\text{arctanh}(\\sqrt{c} \\|\\mathbf{-x} \\oplus_{c} \\mathbf{y}\\|)

    where :math:`c` is the curvature of the ball, and :math:`\\oplus_c` denotes Möbius addition under curvature :math:`-c`.

    Parameters
    ----------
    x : torch.Tensor
        Point on the Poincaré ball.
    y : torch.Tensor
        Point on the Poincaré ball.
    curvature : float or torch.Tensor
        Ball negative curvature.

    Returns
    -------
    torch.Tensor
        Geodesic distance between x and y.
    """
    c = torch.as_tensor(curvature).type_as(x)
    sqrt_c = torch.sqrt(c)

    return 2 / sqrt_c * artanh(sqrt_c * norm(mobius_addition(-x, y, c)))
