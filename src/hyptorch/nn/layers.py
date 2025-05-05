"""
Basic neural network layers for hyperbolic space.
"""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.init as init

from hyptorch.pmath.poincare import (
    distance,
    exponential_map_at_zero,
    mobius_addition,
    mobius_matrix_vector_multiplication,
    project,
)


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    curvature : float
        Negative ball curvature.
    bias : bool
        Whether to use bias.
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
        Reset the parameters of the module.
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
        Forward pass for hyperbolic linear layer.

        Parameters
        ----------
        x : tensor
            Input tensor.
        curvature : float or tensor, optional
            Negative ball curvature. If None, uses the class attribute.

        Returns
        -------
        tensor
            Output tensor.
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
        Return a string representation of module parameters.

        Returns
        -------
        str
            String representation.
        """
        return "in_features={}, out_features={}, bias={}, curvature={}".format(
            self.in_features, self.out_features, self.bias is not None, self.curvature
        )


class ConcatPoincareLayer(nn.Module):
    """
    Layer for concatenating two points in Poincare ball.

    Parameters
    ----------
    d1 : int
        Dimension of first input.
    d2 : int
        Dimension of second input.
    d_out : int
        Dimension of output.
    curvature : float
        Negative ball curvature.
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
        Forward pass for concatenation layer.

        Parameters
        ----------
        x1 : tensor
            First input tensor.
        x2 : tensor
            Second input tensor.
        curvature : float or tensor, optional
            Negative ball curvature. If None, uses the class attribute.

        Returns
        -------
        tensor
            Concatenated tensor in Poincare ball.
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
        Return a string representation of module parameters.

        Returns
        -------
        str
            String representation.
        """
        return "dims {} and {} ---> dim {}".format(self.d1, self.d2, self.d_out)


class HyperbolicDistanceLayer(nn.Module):
    """
    Layer for computing hyperbolic distance between points.

    Parameters
    ----------
    curvature : float
        Negative ball curvature.
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
        Forward pass for distance layer.

        Parameters
        ----------
        x1 : tensor
            First input tensor.
        x2 : tensor
            Second input tensor.
        curvature : float or tensor, optional
            Negative ball curvature. If None, uses the class attribute.

        Returns
        -------
        tensor
            Hyperbolic distance between inputs.
        """
        # Use provided curvature or fall back to class attribute
        if curvature is None:
            curvature = self.curvature

        # Compute hyperbolic distance
        return distance(x1, x2, curvature=curvature, keepdim=True)

    def extra_repr(self) -> str:
        """
        Return a string representation of module parameters.

        Returns
        -------
        str
            String representation.
        """
        return "curvature={}".format(self.curvature)
