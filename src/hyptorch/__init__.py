"""
Hyperbolic Neural Networks package (hyptorch).

This package provides tools for working with neural networks in hyperbolic space,
specifically using the Poincaré ball model of hyperbolic geometry.
"""

from hyptorch.nn.layers import ConcatPoincareLayer, HyperbolicDistanceLayer, HypLinear
from hyptorch.nn.modules import FromPoincare, HyperbolicMLR, ToPoincare
from hyptorch.pmath.distances import distance

__all__ = [
    "HypLinear",
    "ConcatPoincareLayer",
    "HyperbolicDistanceLayer",
    "HyperbolicMLR",
    "ToPoincare",
    "FromPoincare",
    "distance",
]
