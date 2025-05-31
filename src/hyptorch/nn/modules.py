from typing import Optional

import torch
import torch.nn as nn

from hyptorch.manifolds.base import HyperbolicManifold
from hyptorch.manifolds.poincare import PoincareBall
from hyptorch.nn.functional import HyperbolicFunctional
from hyptorch.nn.layers import HyperbolicLayer
from hyptorch.nn.mixins import ParameterInitializationMixin
from hyptorch.operations.autograd import riemannian_gradient


class HyperbolicMLR(HyperbolicLayer, ParameterInitializationMixin):
    def __init__(self, ball_dim: int, n_classes: int, manifold: Optional[HyperbolicManifold] = None):
        if manifold is None:
            manifold = PoincareBall()

        super().__init__(manifold)

        self.ball_dim = ball_dim
        self.n_classes = n_classes

        self.a_vals = nn.Parameter(torch.empty(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.empty(n_classes, ball_dim))
        self._functional = HyperbolicFunctional(manifold)

        self._init_parameters()

    def _init_parameters(self) -> None:
        self._init_kaiming_uniform(self.a_vals)
        self._init_kaiming_uniform(self.p_vals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.manifold.project(x)

        # Map points to Poincare ball
        p_vals_on_manifold = self.manifold.exponential_map_at_zero(self.p_vals)

        # Calculate conformal factor
        conformal_factor = 1 - self.manifold.curvature * p_vals_on_manifold.pow(2).sum(dim=1, keepdim=True)

        # Apply conformal factor to weights
        a_vals_scaled = self.a_vals * conformal_factor

        # Compute hyperbolic softmax (logits)
        return self._functional.hyperbolic_softmax(x, a_vals_scaled, p_vals_on_manifold)

    def extra_repr(self) -> str:
        return f"ball_dim={self.ball_dim}, n_classes={self.n_classes}"


class ToPoincare(HyperbolicLayer):
    def __init__(self, manifold: Optional[HyperbolicManifold] = None):
        if manifold is None:
            manifold = PoincareBall()

        super().__init__(manifold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map input directly to Poincare ball
        mapped = self.manifold.exponential_map_at_zero(x)
        projected = self.manifold.project(mapped)

        # Apply Riemannian gradient fix
        return riemannian_gradient(projected, self.manifold.curvature)


class FromPoincare(HyperbolicLayer):
    def __init__(self, manifold: Optional[HyperbolicManifold] = None):
        if manifold is None:
            manifold = PoincareBall()

        super().__init__(manifold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.manifold.project(x)
        return self.manifold.logarithmic_map_at_zero(x)
