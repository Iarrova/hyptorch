"""
Hyperbolic Neural Networks package (hyptorch).

This package provides tools for working with neural networks in hyperbolic space,
specifically using the Poincaré ball model of hyperbolic geometry.
"""

__version__ = "0.1.0"

# Import commonly used modules to make them available at package level
from nn.layers import HypLinear, ConcatPoincareLayer, HyperbolicDistanceLayer
from nn.modules import HyperbolicMLR, ToPoincare, FromPoincare
from math.poincare import mobius_add, dist, project