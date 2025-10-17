"""
Hyperbolic linear transformation layer.

This module provides the hyperbolic analog of PyTorch's nn.Linear,
implementing linear transformations in hyperbolic space using Möbius operations.

Classes
-------
HypLinear
    Hyperbolic linear layer with Möbius matrix-vector multiplication.
"""

import torch
import torch.nn as nn

from hyptorch.exceptions import NoHyperbolicManifoldProvidedError
from hyptorch.manifolds.base import MobiusManifold
from hyptorch.nn._base import HyperbolicLayer
from hyptorch.nn._mixins import ParameterInitializationMixin


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
    manifold : MobiusManifold
        The manifold that represents hyperbolic space to use.
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
        manifold: MobiusManifold,
        bias: bool = True,
    ) -> None:
        if manifold is None:
            raise NoHyperbolicManifoldProvidedError("HypLinear")

        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")

        super().__init__(manifold)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

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
        projected = self.manifold.project(x)

        output = self.manifold.mobius_matvec(self.weight, projected)
        if self.bias is not None:
            bias_on_manifold = self.manifold.exponential_map_at_origin(self.bias)
            output = self.manifold.mobius_add(output, bias_on_manifold)

        return self.manifold.project(output)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"
