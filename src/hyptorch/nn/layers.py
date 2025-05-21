import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.init as init

from hyptorch.pmath.distances import distance
from hyptorch.pmath.mappings import exponential_map_at_zero, project
from hyptorch.pmath.operations import mobius_addition, mobius_matrix_vector_multiplication


class HypLinear(nn.Module):
    """
    Hyperbolic linear transformation layer using Möbius operations.

    Applies a Möbius matrix-vector multiplication followed by an optional bias addition
    in the Poincaré ball model of hyperbolic space.

    Parameters
    ----------
    in_features : int
        Dimensionality of input features.
    out_features : int
        Dimensionality of output features.
    curvature : float
        Negative curvature value of the Poincaré ball.
    bias : bool
        If True, includes a learnable bias term.
    """

    def __init__(self, in_features: int, out_features: int, curvature: float, bias: bool = True):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = curvature
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights and optional bias using Kaiming uniform distribution.

        Weights are initialized with Kaiming uniform initializer.
        Bias is uniformly sampled within a bound based on fan-in.
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(
        self, x: torch.Tensor, curvature: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Perform forward pass through the hyperbolic linear layer.

        Projects input onto the Poincaré ball using Möbius matrix-vector multiplication
        followed by optional Möbius bias addition, and returns the result reprojected to the manifold.

        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch_size, in_features).
        curvature : float or tensor, optional
            Negative curvature. If None, uses the layer's stored curvature.

        Returns
        -------
        tensor
            Output tensor of shape (batch_size, out_features) in the Poincaré ball.
        """
        # Use provided curvature or fall back to class attribute
        if curvature is None:
            curvature = self.curvature

        # Apply mobius matrix-vector multiplication
        mv = mobius_matrix_vector_multiplication(self.weight, x, curvature=curvature)

        # Apply bias if provided
        if self.bias is None:
            return project(mv, curvature=curvature)
        else:
            # Map bias to Poincare ball
            bias = exponential_map_at_zero(self.bias, curvature=curvature)

            # Add bias and project back to manifold
            return project(mobius_addition(mv, bias, curvature=curvature), curvature=curvature)

    def extra_repr(self) -> str:
        """
        Return a string representation of the layer configuration.

        Returns
        -------
        str
            Human-readable summary of the module's parameters.
        """
        return "in_features={}, out_features={}, bias={}, curvature={}".format(
            self.in_features, self.out_features, self.bias is not None, self.curvature
        )


class ConcatPoincareLayer(nn.Module):
    """
    Concatenation layer for combining two hyperbolic vectors in the Poincaré ball.

    Each input is independently transformed into a shared hyperbolic space using
    separate `HypLinear` layers, and then combined using Möbius addition.

    Parameters
    ----------
    d1 : int
        Dimensionality of the first input.
    d2 : int
        Dimensionality of the second input.
    d_out : int
        Dimensionality of the shared output space.
    curvature : float
        Negative curvature of the hyperbolic space.
    """

    def __init__(self, d1: int, d2: int, d_out: int, curvature: float):
        super(ConcatPoincareLayer, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out

        # Create hyperbolic linear layers for each input
        self.l1 = HypLinear(d1, d_out, bias=False, curvature=curvature)
        self.l2 = HypLinear(d2, d_out, bias=False, curvature=curvature)
        self.curvature = curvature

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        curvature: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Perform forward pass through the concatenation layer.

        Transforms and merges two input tensors into a single tensor in the
        hyperbolic space using Möbius operations.

        Parameters
        ----------
        x1 : tensor
            First input tensor of shape (batch_size, d1).
        x2 : tensor
            Second input tensor of shape (batch_size, d2).
        curvature : float or tensor, optional
            Negative curvature of the Poincaré ball. If None, uses the stored value.

        Returns
        -------
        tensor
            Combined tensor of shape (batch_size, d_out) in the Poincaré ball.
        """
        # Use provided curvature or fall back to class attribute
        if curvature is None:
            curvature = self.curvature

        # Transform inputs using hyperbolic linear layers
        l1_out = self.l1(x1, curvature=curvature)
        l2_out = self.l2(x2, curvature=curvature)

        # Combine using mobius addition
        return mobius_addition(l1_out, l2_out, curvature=curvature)

    def extra_repr(self) -> str:
        """
        Return a string representation of the layer configuration.

        Returns
        -------
        str
            Human-readable summary of input and output dimensions.
        """
        return "dims {} and {} ---> dim {}".format(self.d1, self.d2, self.d_out)


class HyperbolicDistanceLayer(nn.Module):
    """
    Layer to compute pairwise hyperbolic distance between input vectors.

    Operates in the Poincaré ball using the distance function for hyperbolic geometry.

    Parameters
    ----------
    curvature : float
        Negative curvature of the hyperbolic space.
    """

    def __init__(self, curvature: float):
        super(HyperbolicDistanceLayer, self).__init__()
        self.curvature = curvature

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        curvature: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute hyperbolic distance between two input tensors.

        Parameters
        ----------
        x1 : tensor
            First input tensor of shape (batch_size, dim).
        x2 : tensor
            Second input tensor of shape (batch_size, dim).
        curvature : float or tensor, optional
            Negative curvature of the Poincaré ball. If None, uses the stored value.

        Returns
        -------
        tensor
            Tensor of distances of shape (batch_size, 1).
        """
        # Use provided curvature or fall back to class attribute
        if curvature is None:
            curvature = self.curvature

        # Compute hyperbolic distance
        return distance(x1, x2, curvature=curvature)

    def extra_repr(self) -> str:
        """
        Return a string representation of the layer configuration.

        Returns
        -------
        str
            Human-readable summary of the curvature.
        """
        return "curvature={}".format(self.curvature)
