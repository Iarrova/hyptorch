import torch

from hyptorch.config import NumericalConstants
from hyptorch.manifolds.base import HyperbolicManifold
from hyptorch.manifolds.poincare import PoincareBall
from hyptorch.operations.tensor import squared_norm


def _batch_mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    xy = torch.einsum("ij,kj->ik", (x, y))
    x2 = squared_norm(x)
    y2 = squared_norm(y)

    num = 1 + 2 * c * xy + c * y2.permute(1, 0)
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y

    denom = 1 + 2 * c * xy + c**2 * x2 * y2.permute(1, 0)

    return num / (denom.unsqueeze(2) + NumericalConstants.EPS)


def compute_hyperbolic_mlr_logits(
    x: torch.Tensor, weights: torch.Tensor, points: torch.Tensor, manifold: HyperbolicManifold
) -> torch.Tensor:
    if not isinstance(manifold, PoincareBall):
        raise NotImplementedError("Hyperbolic softmax only implemented for Poincaré ball")

    c = manifold.curvature
    sqrt_c = torch.sqrt(c)

    lambda_pkc = 2 / (1 - c * points.pow(2).sum(dim=1))
    k = lambda_pkc * torch.norm(weights, dim=1) / sqrt_c

    mob_add = _batch_mobius_add(-points, x, c)

    num = 2 * sqrt_c * torch.sum(mob_add * weights.unsqueeze(1), dim=-1)
    denom = torch.norm(weights, dim=1, keepdim=True) * (1 - c * mob_add.pow(2).sum(dim=2))

    logits = k.unsqueeze(1) * torch.asinh(num / denom)

    return logits.permute(1, 0)
