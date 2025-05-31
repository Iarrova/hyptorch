from typing import Union

import torch

from hyptorch.manifolds.transformations import klein_to_poincare, lorenz_factor, poincare_to_klein


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
