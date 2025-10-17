"""
Hyperbolic multinomial logistic regression layer.

This module implements multi-class classification in hyperbolic space,
extending logistic regression to operate on the Poincaré ball with
geodesic decision boundaries.

Classes
-------
HyperbolicMLR
    Hyperbolic MLR layer for classification tasks.
"""

import torch
import torch.nn as nn

from hyptorch.manifolds import PoincareBall
from hyptorch.manifolds.base import MobiusManifold
from hyptorch.nn._base import HyperbolicLayer
from hyptorch.nn._mixins import ParameterInitializationMixin
from hyptorch.nn.functional import compute_hyperbolic_mlr_logits


class HyperbolicMLR(HyperbolicLayer, ParameterInitializationMixin):
    """
    Hyperbolic Multinomial Logistic Regression (MLR) layer.

    This module implements multi-class classification in hyperbolic space,
    generalizing softmax regression to the Poincaré ball. Each class is
    represented by a point (p-value) and weight vector (a-value) in
    hyperbolic space.

    Parameters
    ----------
    ball_dim : int
        Dimension of the Poincaré ball (input feature dimension).
    n_classes : int
        Number of classes for classification.
    manifold : MobiusManifold
        The hyperbolic manifold representing hyperbolic space.
        Currently only PoincareBall is supported.

    Attributes
    ----------
    ball_dim : int
        Dimension of the input space.
    n_classes : int
        Number of output classes.
    weight : nn.Parameter
        Weight vectors for each class. These are used for the hyperplane definitions. Shape (n_classes, ball_dim).
    class_points : nn.Parameter
        Class representatives in tangent space at origin. Shape (n_classes, ball_dim).
        These are mapped to the manifold during forward pass.

    Notes
    -----
    The hyperbolic MLR extends logistic regression to hyperbolic space by:

    1. Each class :math:`k` has a representative point :math:`p_k` on the Poincaré ball
    2. Decision boundaries are geodesic hyperplanes
    3. The logit for class :math:`k` given input :math:`x` is based on the hyperbolic distance
       and angle between :math:`x` and :math:`p_k`

    Examples
    --------
    >>> # Multi-class classification in hyperbolic space
    >>> manifold = PoincareBall(curvature=1.0)
    >>> mlr = HyperbolicMLR(ball_dim=10, n_classes=5, manifold=manifold)
    >>> x = torch.randn(32, 10) * 0.3  # Batch of inputs
    >>> logits = mlr(x)  # Shape: (32, 5)
    >>> probs = torch.softmax(logits, dim=1)  # Class probabilities

    See Also
    --------
    compute_hyperbolic_mlr_logits : Function that computes the hyperbolic logits
    """

    def __init__(self, ball_dim: int, n_classes: int, manifold: MobiusManifold) -> None:
        if not isinstance(manifold, PoincareBall):
            raise NotImplementedError("Currently only PoincareBall manifold is supported.")

        if ball_dim <= 0:
            raise ValueError(f"ball_dim must be positive, got {ball_dim}")
        if n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {n_classes}")

        super().__init__(manifold)

        self.ball_dim = ball_dim
        self.n_classes = n_classes

        self.weight = nn.Parameter(torch.empty(n_classes, ball_dim))
        self.class_points = nn.Parameter(torch.empty(n_classes, ball_dim))

        self._init_parameters()

    def _init_parameters(self) -> None:
        """
        Initialize parameters using Kaiming uniform initialization.

        Both a_vals (weights) and p_vals (class representatives in tangent space)
        are initialized with the same scheme.
        """
        self._init_kaiming_uniform(self.weight)
        self._init_kaiming_uniform(self.class_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic MLR logits for multi-class classification.

        Parameters
        ----------
        x : torch.Tensor
            Input points on the Poincaré ball. Shape (batch_size, ball_dim).

        Returns
        -------
        torch.Tensor
            Logits for each class. Shape (batch_size, n_classes).
            These can be passed to softmax for class probabilities.

        Notes
        -----
        The forward pass:
        1. Projects input to ensure it's on the manifold
        2. Maps class_points from tangent space at origin to the manifold
        3. Scales weight by the conformal factor at each class representative
        4. Computes hyperbolic logits using the functional interface
        """
        x = self.manifold.project(x)

        class_points_on_manifold = self.manifold.exponential_map_at_origin(self.class_points)

        conformal_factor = 1 - self.manifold.curvature * class_points_on_manifold.pow(2).sum(
            dim=1, keepdim=True
        )
        weight_scaled = self.weight * conformal_factor

        return compute_hyperbolic_mlr_logits(x, weight_scaled, class_points_on_manifold, self.manifold)

    def extra_repr(self) -> str:
        return f"ball_dim={self.ball_dim}, n_classes={self.n_classes}"
