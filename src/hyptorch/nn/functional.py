import torch

from typing import Union

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


def hyperbolic_softmax(
    X: torch.Tensor,
    A: torch.Tensor,
    P: torch.Tensor,
    curvature: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Compute hyperbolic softmax.

    Parameters
    ----------
    X : tensor
        Input tensor.
    A : tensor
        Weights tensor.
    P : tensor
        Points tensor.
    curvature : float or tensor
        Negative ball curvature.

    Returns
    -------
    tensor
        Hyperbolic softmax result.
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
