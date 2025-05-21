from typing import Union

import torch

from hyptorch.pmath.autograd import arsinh
from hyptorch.pmath.operations import batch_mobius_addition


def hyperbolic_softmax_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    points: torch.Tensor,
    curvature: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute the softmax cross-entropy loss in hyperbolic space (Poincaré ball model).

    This function applies a softmax operation adapted for hyperbolic geometry, using
    Möbius operations and curvature-aware logits. The log-probabilities are computed from
    the hyperbolic softmax, and the standard cross-entropy loss is applied to the target labels.

    Parameters
    ----------
    logits : torch.Tensor
        Raw class scores (logits) for each input.
    targets : torch.Tensor
        Ground truth class indices.
    weights : torch.Tensor
        Weight vectors associated with each class.
    points : torch.Tensor
        Embedding points in the Poincaré ball for each input.
    curvature : float
        Negative curvature of the hyperbolic space.
    reduction : str, optional
        Specifies the reduction to apply to the output:
        'none' returns the loss per example, 'sum' returns the sum,
        and 'mean' (default) returns the average loss.

    Returns
    -------
    torch.Tensor
        The computed softmax loss in hyperbolic space.
    """
    probs = hyperbolic_softmax(logits, weights, points, curvature)
    log_probs = torch.log(probs + 1e-8)

    if reduction == "none":
        return -log_probs.gather(1, targets.unsqueeze(-1))
    elif reduction == "sum":
        return -log_probs.gather(1, targets.unsqueeze(-1)).sum()
    else:  # mean
        return -log_probs.gather(1, targets.unsqueeze(-1)).mean()


def hyperbolic_softmax(
    X: torch.Tensor,
    A: torch.Tensor,
    P: torch.Tensor,
    curvature: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Compute class scores in hyperbolic space using a geometry-aware softmax formulation.

    This function calculates a softmax-like output by projecting data into hyperbolic space
    using Möbius addition and curvature-aware transformations. It uses Lorentz scaling and
    distance-aware terms to produce logits that respect hyperbolic geometry.

    The logits are computed as:

    .. math::
        \\text{logit}_{ij} = k_j \\cdot \\sinh^{-1} \\left(
        \\frac{2 \\sqrt{c} \\langle \\mathbf{a}_j, -\\mathbf{p}_j \\oplus \\mathbf{x}_i \\rangle}
             {\\|\\mathbf{a}_j\\| (1 - c \\| -\\mathbf{p}_j \\oplus \\mathbf{x}_i \\|^2)}
        \\right)

    where :math:`\\oplus` denotes Möbius addition, :math:`c` is the negative curvature,
    :math:`\\mathbf{a}_j` and :math:`\\mathbf{p}_j` are the weight and point for class :math:`j`.
    TODO: Change to the formula in the paper.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape `(batch_size, dim)`.
    A : torch.Tensor
        Weight tensor of shape `(num_classes, dim)`.
    P : torch.Tensor
        Point tensor of shape `(num_classes, dim)`, representing class anchors in the Poincaré ball.
    curvature : float or torch.Tensor
        Negative curvature of the hyperbolic space.

    Returns
    -------
    torch.Tensor
        Logits tensor of shape `(batch_size, num_classes)` computed in hyperbolic space.
    """
    # Pre-compute common values
    lambda_pkc = 2 / (1 - curvature * P.pow(2).sum(dim=1))
    k = lambda_pkc * torch.norm(A, dim=1) / torch.sqrt(curvature)

    # Calculate mobius addition and other values
    mob_add = batch_mobius_addition(-P, X, curvature)

    num = 2 * torch.sqrt(curvature) * torch.sum(mob_add * A.unsqueeze(1), dim=-1)

    denom = torch.norm(A, dim=1, keepdim=True) * (1 - curvature * mob_add.pow(2).sum(dim=2))

    logit = k.unsqueeze(1) * arsinh(num / denom)

    return logit.permute(1, 0)
