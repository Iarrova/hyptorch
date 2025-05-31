import torch

from hyptorch.config import NumericalConstants, ReductionType
from hyptorch.manifolds.base import HyperbolicManifold
from hyptorch.manifolds.poincare import PoincareBall
from hyptorch.operations.tensor import squared_norm


class HyperbolicFunctional:
    def __init__(self, manifold: HyperbolicManifold):
        self.manifold = manifold

    def hyperbolic_softmax(
        self, x: torch.Tensor, weights: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(self.manifold, PoincareBall):
            raise NotImplementedError("Hyperbolic softmax only implemented for Poincaré ball")

        c = self.manifold.curvature
        sqrt_c = torch.sqrt(c)

        lambda_pkc = 2 / (1 - c * points.pow(2).sum(dim=1))
        k = lambda_pkc * torch.norm(weights, dim=1) / sqrt_c

        mob_add = self._batch_mobius_addition(-points, x)

        num = 2 * sqrt_c * torch.sum(mob_add * weights.unsqueeze(1), dim=-1)
        denom = torch.norm(weights, dim=1, keepdim=True) * (1 - c * mob_add.pow(2).sum(dim=2))

        logits = k.unsqueeze(1) * torch.asinh(num / denom)

        return logits.permute(1, 0)

    def _batch_mobius_addition(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.manifold.curvature

        xy = torch.einsum("ij,kj->ik", (x, y))
        x2 = squared_norm(x)
        y2 = squared_norm(y)

        num = 1 + 2 * c * xy + c * y2.permute(1, 0)
        num = num.unsqueeze(2) * x.unsqueeze(1)
        num = num + (1 - c * x2).unsqueeze(2) * y

        denom = 1 + 2 * c * xy + c**2 * x2 * y2.permute(1, 0)

        return num / (denom.unsqueeze(2) + NumericalConstants.EPS)


def hyperbolic_softmax_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    points: torch.Tensor,
    manifold: HyperbolicManifold,
    reduction: ReductionType = ReductionType.MEAN,
) -> torch.Tensor:
    functional = HyperbolicFunctional(manifold)
    probs = functional.hyperbolic_softmax(logits, weights, points)

    log_probs = torch.log(probs + 1e-8)

    losses = -log_probs.gather(1, targets.unsqueeze(-1))

    if reduction == ReductionType.NONE:
        return losses
    elif reduction == ReductionType.SUM:
        return losses.sum()
    else:
        return losses.mean()
