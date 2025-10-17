"""
Mixin classes for neural network parameter initialization.

This module provides reusable initialization methods for hyperbolic neural
network layers, following PyTorch's initialization conventions.

Classes
-------
ParameterInitializationMixin
    Mixin providing Kaiming uniform and bias initialization methods.
"""

import math

import torch


class ParameterInitializationMixin:
    """
    Mixin class providing parameter initialization methods for neural network layers.

    This mixin provides standardized initialization methods that can be used
    across different hyperbolic neural network layers. It implements common
    initialization schemes adapted for hyperbolic geometry.

    Notes
    -----
    This class uses the mixin pattern to provide reusable initialization
    functionality without requiring inheritance from a specific base class.
    Classes that use this mixin should inherit from both their primary base
    class and this mixin.
    """

    @staticmethod
    def _init_kaiming_uniform(parameter: torch.nn.Parameter, a: float = math.sqrt(5)) -> None:
        """
        Initialize a parameter using Kaiming uniform initialization.

        This method implements He initialization with uniform distribution,
        which is particularly effective for layers with ReLU-like activations.
        The initialization helps maintain gradient magnitudes during training.

        Parameters
        ----------
        parameter : torch.nn.Parameter
            The parameter tensor to initialize. Can be of any shape, though
            typically used for weight matrices of shape (out_features, in_features).
        a : float, optional
            The negative slope of the rectifier used after this layer (only used
            for 'leaky_relu' mode). Default is sqrt(5), which is the PyTorch
            default for linear layers.
        """
        torch.nn.init.kaiming_uniform_(parameter, a=a)

    @staticmethod
    def _init_bias_uniform(parameter: torch.nn.Parameter, fan_in: int) -> None:
        """
        Initialize a bias parameter using uniform distribution based on fan-in.

        This method initializes bias parameters using a uniform distribution
        where the range is determined by the fan-in (number of input features).
        This is the standard PyTorch approach for linear layer biases.

        Parameters
        ----------
        parameter : torch.nn.Parameter
            The bias parameter to initialize. Typically a 1D tensor of shape
            (out_features,).
        fan_in : int
            The number of input features (connections) to the layer. Used to
            calculate the initialization range.
        """
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(parameter, -bound, bound)
