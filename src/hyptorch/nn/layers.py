import torch
import torch.nn as nn

from hyptorch.models.base import HyperbolicMobiusModel
from hyptorch.nn._mixins import ParameterInitializationMixin
from hyptorch.nn.base import HyperbolicLayer
from hyptorch.tensor import apply_riemannian_gradient


class HypLinear(HyperbolicLayer, ParameterInitializationMixin):
    """
    Hyperbolic linear transformation layer.

    Implements a linear transformation in hyperbolic space using Möbius
    matrix-vector multiplication and Möbius addition for bias. This is the
    hyperbolic analog of nn.Linear.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    model : HyperbolicMobiusModel
        The model that represents hyperbolic space to use.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default is True.

    Attributes
    ----------
    in_features : int
        Size of input features.
    out_features : int
        Size of output features.
    use_bias : bool
        Whether bias is used.
    weight : nn.Parameter
        The learnable weight matrix of shape (out_features, in_features).
    bias : nn.Parameter or None
        The learnable bias of shape (out_features) if bias=True, else None.

    Notes
    -----
    The hyperbolic linear transformation is computed as:

    1. Apply Möbius matrix-vector multiplication: :math:`\\mathbf{h} = \\mathbf{M} \\otimes_c \\mathbf{x}`
    2. If bias is used, apply Möbius addition: :math:`y = \\mathbf{h} \\oplus_c \\mathbf{b}`
    3. Project result back to manifold for numerical stability

    The weight matrix is initialized in Euclidean space but the transformation
    respects the hyperbolic geometry through Möbius operations.

    Examples
    --------
    >>> model = PoincareBall(curvature=1.0)
    >>> layer = HypLinear(10, 5, model=model)
    >>> x = torch.randn(32, 10) * 0.1  # Batch of 32 samples
    >>> y = layer(x)  # Output shape: (32, 5)

    See Also
    --------
    PoincareBall.mobius_matvec : Möbius matrix-vector multiplication
    PoincareBall.mobius_add : Möbius addition
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        model: HyperbolicMobiusModel,
        bias: bool = True,
    ):
        if model is None:
            raise ValueError("A hyperbolic model must be provided.")

        super().__init__(model)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self._init_parameters()

    def _init_parameters(self) -> None:
        """
        Initialize layer parameters.

        Uses Kaiming uniform initialization for weights and uniform
        initialization for bias based on fan-in.
        """
        self._init_kaiming_uniform(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            self._init_bias_uniform(self.bias, fan_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hyperbolic linear transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of points on the model. Shape (..., in_features).

        Returns
        -------
        torch.Tensor
            Transformed points on the model. Shape (..., out_features).
        """
        projected = self.model.project(x)

        output = self.model.mobius_matvec(self.weight, projected)
        if self.bias is not None:
            bias_on_model = self.model.exponential_map_at_origin(self.bias)
            output = self.model.mobius_add(output, bias_on_model)

        return self.model.project(output)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"


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
            raise ValueError("A hyperbolic model must be provided.")

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
            raise ValueError("A hyperbolic model must be provided.")

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
