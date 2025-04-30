"""
Operations for conversions between models.
"""

from typing import Union

import torch


def poincare_to_klein(x: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Map from Poincare to Klein model.

    Parameters
    ----------
    x : tensor
        Point on Poincare ball.
    curvature : float or tensor
        Negative ball curvature.

    Returns
    -------
    tensor
        Point in Klein model.
    """
    c = torch.as_tensor(curvature).type_as(x)
    denominator = 1 + c * x.pow(2).sum(-1, keepdim=True)
    return 2 * x / denominator


def klein_to_poincare(x: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Map from Klein to Poincare model.

    Parameters
    ----------
    x : tensor
        Point in Klein model.
    curvature : float or tensor
        Negative ball curvature.

    Returns
    -------
    tensor
        Point on Poincare ball.
    """
    c = torch.as_tensor(curvature).type_as(x)
    denominator = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
    return x / denominator
