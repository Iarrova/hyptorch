from typing import Optional

import torch
import torch.nn as nn

from hyptorch.manifolds.base import HyperbolicManifold
from hyptorch.manifolds.poincare import PoincareBall
from hyptorch.nn.mixins import ParameterInitializationMixin


class HyperbolicLayer(nn.Module):
    def __init__(self, manifold: HyperbolicManifold):
        super().__init__()
        self.manifold = manifold

    @property
    def curvature(self) -> torch.Tensor:
        return self.manifold.curvature


class HypLinear(HyperbolicLayer, ParameterInitializationMixin):
    def __init__(
        self, in_features: int, out_features: int, manifold: Optional[PoincareBall] = None, bias: bool = True
    ):
        if manifold is None:
            manifold = PoincareBall()

        super().__init__(manifold)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self._init_parameters()

    def _init_parameters(self) -> None:
        self._init_kaiming_uniform(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            self._init_bias_uniform(self.bias, fan_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.manifold.project(x)
        output = self.manifold.mobius_matvec(self.weight, x)

        if self.bias is not None:
            bias_on_manifold = self.manifold.exponential_map_at_origin(self.bias)
            output = self.manifold.mobius_add(output, bias_on_manifold)

        return self.manifold.project(output)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"


class ConcatPoincareLayer(HyperbolicLayer):
    def __init__(self, d1: int, d2: int, d_out: int, manifold: Optional[PoincareBall] = None):
        if manifold is None:
            manifold = PoincareBall()

        super().__init__(manifold)

        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out

        self.layer1 = HypLinear(d1, d_out, manifold=manifold, bias=False)
        self.layer2 = HypLinear(d2, d_out, manifold=manifold, bias=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        out1 = self.layer1(x1)
        out2 = self.layer2(x2)

        return self.manifold.mobius_add(out1, out2)

    def extra_repr(self) -> str:
        return f"d1={self.d1}, d2={self.d2}, d_out={self.d_out}"


class HyperbolicDistanceLayer(HyperbolicLayer):
    def __init__(self, manifold: Optional[PoincareBall] = None):
        if manifold is None:
            manifold = PoincareBall()

        super().__init__(manifold)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.manifold.distance(x1, x2)
