from typing import Union

import torch
from torch.autograd.function import FunctionCtx

from hyptorch.config import CLAMP_MAX, CLAMP_MIN, EPS, TANH_CLAMP


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the inverse hyperbolic tangent function.

        The input tensor is clamped to the range [-1 + EPS, 1 - EPS] to avoid numerical instability.
        This function computes the inverse hyperbolic tangent of the input tensor using the formula:

        .. math::
            \\text{artanh}(x) = \\frac{1}{2} \\log\\left(\\frac{1+x}{1-x}\\right)

        where :math:`x` is the input tensor.
        """
        x = x.clamp(CLAMP_MIN, CLAMP_MAX)
        ctx.save_for_backward(x)
        return 0.5 * (torch.log(1 + x) - torch.log(1 - x))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass for the inverse hyperbolic tangent function.

        This function computes the gradient of the inverse hyperbolic tangent with respect to the input tensor.
        The gradient is computed using the formula:

        .. math::
            \\frac{d}{dx} \\text{artanh}(x) = \\frac{1}{1 - x^2}

        where :math:`x` is the input tensor.
        """
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input**2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the inverse hyperbolic sine function.

        The input tensor is clamped to the range [EPS, +inf) to avoid numerical instability.
        This function computes the inverse hyperbolic sine of the input tensor using the formula:

        .. math::
            \\text{arsinh}(x) = \\log\\left(x + \\sqrt{1+x^2}\\right)

        where :math:`x` is the input tensor.
        """
        ctx.save_for_backward(x)
        return (x + torch.sqrt_(1 + x.pow(2))).clamp_min_(EPS).log_()

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass for the inverse hyperbolic sine function.

        This function computes the gradient of the inverse hyperbolic sine with respect to the input tensor.
        The gradient is computed using the formula:

        .. math::
            \\frac{d}{dx} \\text{arsinh}(x) = \\frac{1}{\\sqrt{1+x^2}}

        where :math:`x` is the input tensor.
        """
        (input,) = ctx.saved_tensors
        return grad_output / (1 + input**2) ** 0.5


class RiemannianGradient(torch.autograd.Function):
    """
    Custom autograd function that applies a Riemannian gradient correction
    for optimization on the Poincaré ball model of hyperbolic space.

    This function modifies the gradient during the backward pass to account for
    the geometry of the space, specifically adjusting the Euclidean gradient
    to the corresponding Riemannian gradient under the Poincaré ball metric.

    The forward pass is the identity function and simply saves the input and
    curvature for use in the backward computation.
    """

    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Identity function that stores input and curvature for backward pass.
        """
        c = torch.as_tensor(curvature).type_as(x)
        ctx.save_for_backward(x, c)
        return x

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Applies Riemannian correction to the gradient.
        """
        input, c = ctx.saved_tensors
        # Compute Riemannian gradient scale factor
        scale = (1 - c * input.pow(2).sum(-1, keepdim=True)).pow(2) / 4
        return grad_output * scale, None


def artanh(x: torch.Tensor) -> torch.Tensor:
    return Artanh.apply(x)


def arsinh(x: torch.Tensor) -> torch.Tensor:
    return Arsinh.apply(x)


def tanh(x: torch.Tensor, clamp: float = TANH_CLAMP) -> torch.Tensor:
    """
    Numerically stable implementation of tanh.

    Parameters
    ----------
    x : tensor
        Input tensor.
    clamp : float
        Clamping value to ensure numerical stability.

    Returns
    -------
    tensor
        Tanh of the input tensor.
    """
    return x.clamp(-clamp, clamp).tanh()
