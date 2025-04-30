"""
Functional operations for hyperbolic neural networks.
"""

import torch

from hyptorch.pmath.poincare import hyperbolic_softmax


def hyperbolic_softmax_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    points: torch.Tensor,
    curvature: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Softmax loss in hyperbolic space.

    Parameters
    ----------
    logits : tensor
        Predicted logits from the model.
    targets : tensor
        Target labels.
    points : tensor
        Points in Poincaré ball for hyperbolic softmax.
    curvature : float
        Negative ball curvature.
    reduction : str
        Reduction method ('mean', 'sum', or 'none').

    Returns
    -------
    tensor
        Loss value.
    """
    probs = hyperbolic_softmax(logits, weights, points, curvature)
    log_probs = torch.log(probs + 1e-8)

    if reduction == "none":
        return -log_probs.gather(1, targets.unsqueeze(-1))
    elif reduction == "sum":
        return -log_probs.gather(1, targets.unsqueeze(-1)).sum()
    else:  # mean
        return -log_probs.gather(1, targets.unsqueeze(-1)).mean()
