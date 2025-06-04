from typing import Optional

import torch
import torch.nn as nn

from hyptorch.manifolds.base import MobiusManifold
from hyptorch.manifolds.poincare import PoincareBall
from hyptorch.nn.mixins import ParameterInitializationMixin


class HyperbolicLayer(nn.Module):
    """
    Base class for hyperbolic neural network layers.

    This abstract class provides a foundation for all hyperbolic layers,
    maintaining a reference to the underlying hyperbolic manifold and
    providing convenient access to its curvature.

    Parameters
    ----------
    manifold : HyperbolicManifold
        The hyperbolic manifold on which the layer operates.

    Attributes
    ----------
    manifold : HyperbolicManifold
        The hyperbolic manifold instance.
    curvature : torch.Tensor
        The curvature of the manifold (accessible via property).

    Notes
    -----
    All hyperbolic layers should inherit from this base class to ensure
    consistent handling of the manifold and its properties.
    """

    def __init__(self, manifold: MobiusManifold):
        super().__init__()
        self.manifold = manifold

    @property
    def curvature(self) -> torch.Tensor:
        """
        Get the curvature of the layer's manifold.

        Returns
        -------
        torch.Tensor
            The curvature parameter of the manifold.
        """
        return self.manifold.curvature


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
    manifold : MobiusManifold, optional
        The Poincaré ball manifold to use. If None, creates a new PoincareBall
        with default curvature. Default is None.
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
    >>> manifold = PoincareBall(curvature=1.0)
    >>> layer = HypLinear(10, 5, manifold=manifold)
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
        manifold: Optional[MobiusManifold] = None,
        bias: bool = True,
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
            Input tensor of points on the manifold. Shape (..., in_features).

        Returns
        -------
        torch.Tensor
            Transformed points on the manifold. Shape (..., out_features).
        """
        x = self.manifold.project(x)
        output = self.manifold.mobius_matvec(self.weight, x)

        if self.bias is not None:
            bias_on_manifold = self.manifold.exponential_map_at_origin(self.bias)
            output = self.manifold.mobius_add(output, bias_on_manifold)

        return self.manifold.project(output)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"


class ConcatPoincareLayer(HyperbolicLayer):
    """
    Hyperbolic concatenation layer for the Poincaré ball.

    This layer concatenates two hyperbolic embeddings by applying separate
    linear transformations and combining them with Möbius addition. This is
    useful for combining features from different sources in hyperbolic space.

    Parameters
    ----------
    d1 : int
        Dimension of the first input.
    d2 : int
        Dimension of the second input.
    d_out : int
        Dimension of the output.
    manifold : MobiusManifold, optional
        The Poincaré ball manifold to use. If None, creates a new PoincareBall
        with default curvature. Default is None.

    Attributes
    ----------
    d1 : int
        First input dimension.
    d2 : int
        Second input dimension.
    d_out : int
        Output dimension.
    layer1 : HypLinear
        Linear transformation for first input.
    layer2 : HypLinear
        Linear transformation for second input.

    Notes
    -----
    The concatenation operation in hyperbolic space is performed as:

    .. math::
        \\text{concat}(x_1, x_2) = (W_1 \\otimes_c x_1) \\oplus_c (W_2 \\otimes_c x_2)

    where :math:`\\otimes_c` is Möbius matrix multiplication and :math:`\\oplus_c`
    is Möbius addition. This preserves the hyperbolic structure while combining
    information from both inputs.

    Examples
    --------
    >>> manifold = PoincareBall()
    >>> concat_layer = ConcatPoincareLayer(5, 3, 8, manifold)
    >>> x1 = torch.randn(32, 5) * 0.1
    >>> x2 = torch.randn(32, 3) * 0.1
    >>> output = concat_layer(x1, x2)  # Shape: (32, 8)

    See Also
    --------
    HypLinear : Hyperbolic linear layer used for transformations
    """

    def __init__(self, d1: int, d2: int, d_out: int, manifold: Optional[MobiusManifold] = None):
        if manifold is None:
            manifold = PoincareBall()

        super().__init__(manifold)

        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out

        self.layer1 = HypLinear(d1, d_out, manifold=manifold, bias=False)
        self.layer2 = HypLinear(d2, d_out, manifold=manifold, bias=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Concatenate two inputs in hyperbolic space.

        Parameters
        ----------
        x1 : torch.Tensor
            First input on the manifold. Shape (..., d1).
        x2 : torch.Tensor
            Second input on the manifold. Shape (..., d2).

        Returns
        -------
        torch.Tensor
            Concatenated output on the manifold. Shape (..., d_out).
        """
        out1 = self.layer1(x1)
        out2 = self.layer2(x2)

        return self.manifold.mobius_add(out1, out2)

    def extra_repr(self) -> str:
        return f"d1={self.d1}, d2={self.d2}, d_out={self.d_out}"


class HyperbolicDistanceLayer(HyperbolicLayer):
    """
    Layer for computing pairwise hyperbolic distances.

    This layer computes the geodesic distance between pairs of points
    on a hyperbolic manifold. It can be used as a similarity measure
    or as part of distance-based models.

    Parameters
    ----------
    manifold : MobiusManifold, optional
        The Poincaré ball manifold to use. If None, creates a new PoincareBall
        with default curvature. Default is None.

    Notes
    -----
    The layer is useful for tasks that require measuring distances such as:
    - Similarity computation in hyperbolic embeddings
    - Distance-based classification or clustering
    - Metric learning in hyperbolic space

    Examples
    --------
    >>> manifold = PoincareBall()
    >>> dist_layer = HyperbolicDistanceLayer(manifold)
    >>> x1 = torch.randn(32, 10) * 0.3
    >>> x2 = torch.randn(32, 10) * 0.3
    >>> distances = dist_layer(x1, x2)  # Shape: (32,)

    See Also
    --------
    PoincareBall.distance : The underlying distance computation
    """

    def __init__(self, manifold: Optional[MobiusManifold] = None):
        if manifold is None:
            manifold = PoincareBall()

        super().__init__(manifold)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic distance between pairs of points.

        Parameters
        ----------
        x1 : torch.Tensor
            First set of points on the manifold. Shape (..., dim).
        x2 : torch.Tensor
            Second set of points on the manifold. Shape (..., dim).

        Returns
        -------
        torch.Tensor
            Pairwise distances. Shape (...,), where the output shape
            is the broadcast-compatible shape of the input batch dimensions.
        """
        return self.manifold.distance(x1, x2)
