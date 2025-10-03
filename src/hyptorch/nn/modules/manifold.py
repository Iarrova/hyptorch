import torch

from hyptorch.exceptions import NoHyperbolicModelProvidedError
from hyptorch.models.base import HyperbolicMobiusModel
from hyptorch.nn._base import HyperbolicLayer
from hyptorch.tensor import apply_riemannian_gradient


class ToPoincare(HyperbolicLayer):
    """
    Layer that maps Euclidean points to the Poincaré ball.

    This module provides a differentiable mapping from Euclidean space to
    hyperbolic space, using the exponential map at the origin. It also
    applies a Riemannian gradient correction to ensure proper gradient
    flow in the hyperbolic space.

    Parameters
    ----------
    model : HyperbolicMobiusModel
        The model that represents hyperbolic space to use.

    Notes
    -----
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

    def __init__(self, model: HyperbolicMobiusModel):
        if model is None:
            raise NoHyperbolicModelProvidedError

        super().__init__(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map Euclidean points to the Hyperbolic model.

        Parameters
        ----------
        x : torch.Tensor
            Input points in Euclidean space. Shape (..., dim).

        Returns
        -------
        torch.Tensor
            Points on the Hyperbolic model with Riemannian gradient correction.
            Shape (..., dim).
        """
        mapped = self.model.exponential_map_at_origin(x)
        projected = self.model.project(mapped)

        return apply_riemannian_gradient(projected, self.model.curvature)


class FromPoincare(HyperbolicLayer):
    """
    Layer that maps points from the Hyperbolic model to Euclidean space.

    This module provides a differentiable mapping from hyperbolic space back
    to Euclidean space using the logarithmic map at the origin. This is useful
    for extracting features from hyperbolic representations for use in
    Euclidean layers.

    Parameters
    ----------
    model : HyperbolicMobiusModel
        The model that represents hyperbolic space to use.

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

    def __init__(self, model: HyperbolicMobiusModel):
        if model is None:
            raise NoHyperbolicModelProvidedError

        super().__init__(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map points from the Hyperbolic model to Euclidean space.

        Parameters
        ----------
        x : torch.Tensor
            Input points on the Hyperbolic model. Shape (..., dim).

        Returns
        -------
        torch.Tensor
            Points in Euclidean space (tangent space at origin). Shape (..., dim).
        """
        projected = self.model.project(x)

        return self.model.logarithmic_map_at_origin(projected)
