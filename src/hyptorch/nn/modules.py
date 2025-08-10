from typing import Optional

import torch
import torch.nn as nn

from hyptorch.models.base import HyperbolicMobiusModel
from hyptorch.models.poincare_ball import PoincareBall
from hyptorch.nn.functional import compute_hyperbolic_mlr_logits
from hyptorch.nn.layers import HyperbolicLayer
from hyptorch.nn.mixins import ParameterInitializationMixin
from hyptorch.operations.autograd import apply_riemannian_gradient


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
    manifold : MobiusManifold, optional
        The hyperbolic manifold to use. If None, creates a new PoincareBall
        with default curvature. Default is None.

    Attributes
    ----------
    ball_dim : int
        Dimension of the input space.
    n_classes : int
        Number of output classes.
    a_vals : nn.Parameter
        Weight vectors for each class. Shape (n_classes, ball_dim).
    p_vals : nn.Parameter
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

    def __init__(self, ball_dim: int, n_classes: int, manifold: Optional[PoincareBall] = None):
        if manifold is None:
            manifold = PoincareBall()

        super().__init__(manifold)

        self.ball_dim = ball_dim
        self.n_classes = n_classes

        self.a_vals = nn.Parameter(torch.empty(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.empty(n_classes, ball_dim))

        self._init_parameters()

    def _init_parameters(self) -> None:
        """
        Initialize parameters using Kaiming uniform initialization.

        Both a_vals (weights) and p_vals (class representatives in tangent space)
        are initialized with the same scheme.
        """
        self._init_kaiming_uniform(self.a_vals)
        self._init_kaiming_uniform(self.p_vals)

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
        2. Maps p_vals from tangent space at origin to the manifold
        3. Scales a_vals by the conformal factor at each class representative
        4. Computes hyperbolic logits using the functional interface
        """
        x = self.manifold.project(x)

        p_vals_on_manifold = self.manifold.exponential_map_at_origin(self.p_vals)

        conformal_factor = 1 - self.manifold.curvature * p_vals_on_manifold.pow(2).sum(dim=1, keepdim=True)
        a_vals_scaled = self.a_vals * conformal_factor

        return compute_hyperbolic_mlr_logits(x, a_vals_scaled, p_vals_on_manifold, self.manifold)

    def extra_repr(self) -> str:
        return f"ball_dim={self.ball_dim}, n_classes={self.n_classes}"


class ToPoincare(HyperbolicLayer):
    """
    Layer that maps Euclidean points to the Poincaré ball.

    This module provides a differentiable mapping from Euclidean space to
    hyperbolic space, using the exponential map at the origin. It also
    applies a Riemannian gradient correction to ensure proper gradient
    flow in the hyperbolic space.

    Parameters
    ----------
    manifold : MobiusManifold, optional
        The hyperbolic manifold to use. If None, creates a new PoincareBall
        with default curvature. Default is None.

    Notes
    -----
    The mapping is performed via:

    1. Exponential map at origin: Maps Euclidean vectors to the Poincaré ball
    2. Projection: Ensures numerical stability by keeping points within bounds
    3. Riemannian gradient: Applies gradient scaling for proper optimization

    The Riemannian gradient correction is crucial for optimization as it
    accounts for the distortion of the hyperbolic metric, scaling gradients
    by :math:`\\frac{(1 - c\\|x\\|^2)^2}{4}`.

    Examples
    --------
    >>> # Map Euclidean embeddings to hyperbolic space
    >>> to_poincare = ToPoincare()
    >>> euclidean_features = torch.randn(32, 10)  # Euclidean vectors
    >>> hyperbolic_features = to_poincare(euclidean_features)
    >>> # hyperbolic_features are now on the Poincaré ball

    >>> # Use in a neural network
    >>> model = nn.Sequential(
    ...     nn.Linear(20, 10),
    ...     nn.ReLU(),
    ...     ToPoincare(),  # Map to hyperbolic space
    ...     HypLinear(10, 5)  # Process in hyperbolic space
    ... )

    See Also
    --------
    FromPoincare : Inverse operation mapping from Poincaré ball to Euclidean
    PoincareBall.exponential_map_at_origin : The underlying mapping function
    """

    # TODO: Maybe add apply_riemannian_gradient to See Also section

    def __init__(self, manifold: Optional[HyperbolicMobiusModel] = None):
        if manifold is None:
            manifold = PoincareBall()

        super().__init__(manifold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map Euclidean points to the Poincaré ball.

        Parameters
        ----------
        x : torch.Tensor
            Input points in Euclidean space. Shape (..., dim).

        Returns
        -------
        torch.Tensor
            Points on the Poincaré ball with Riemannian gradient correction.
            Shape (..., dim).
        """
        mapped = self.manifold.exponential_map_at_origin(x)
        projected = self.manifold.project(mapped)

        return apply_riemannian_gradient(projected, self.manifold.curvature)


class FromPoincare(HyperbolicLayer):
    """
    Layer that maps points from the Poincaré ball to Euclidean space.

    This module provides a differentiable mapping from hyperbolic space back
    to Euclidean space using the logarithmic map at the origin. This is useful
    for extracting features from hyperbolic representations for use in
    Euclidean layers.

    Parameters
    ----------
    manifold : MobiusManifold, optional
        The hyperbolic manifold to use. If None, creates a new PoincareBall
        with default curvature. Default is None.

    Notes
    -----
    The mapping uses the logarithmic map at the origin, which is the inverse
    of the exponential map. For a point :math:`x` on the Poincaré ball, it computes
    the tangent vector at the origin that would map to :math:`x` under the exponential
    map.

    This layer is useful when:
    - Transitioning from hyperbolic to Euclidean processing
    - Extracting hyperbolic features for Euclidean classifiers
    - Creating hybrid architectures with both geometries

    Examples
    --------
    >>> # Extract Euclidean features from hyperbolic embeddings
    >>> from_poincare = FromPoincare()
    >>> hyperbolic_points = torch.randn(32, 10) * 0.3
    >>> hyperbolic_points = PoincareBall().project(hyperbolic_points)
    >>> euclidean_features = from_poincare(hyperbolic_points)

    >>> # Hybrid architecture
    >>> model = nn.Sequential(
    ...     HypLinear(10, 8),  # Process in hyperbolic space
    ...     FromPoincare(),    # Convert to Euclidean
    ...     nn.Linear(8, 5),   # Process in Euclidean space
    ...     nn.Softmax(dim=1)
    ... )

    See Also
    --------
    ToPoincare : Inverse operation mapping from Euclidean to Poincaré ball
    PoincareBall.logarithmic_map_at_origin : The underlying mapping function
    """

    def __init__(self, manifold: Optional[HyperbolicMobiusModel] = None):
        if manifold is None:
            manifold = PoincareBall()

        super().__init__(manifold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map points from the Poincaré ball to Euclidean space.

        Parameters
        ----------
        x : torch.Tensor
            Input points on the Poincaré ball. Shape (..., dim).

        Returns
        -------
        torch.Tensor
            Points in Euclidean space (tangent space at origin). Shape (..., dim).
        """
        x = self.manifold.project(x)
        return self.manifold.logarithmic_map_at_origin(x)
