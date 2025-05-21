from typing import Union

import torch
from torch.autograd.function import FunctionCtx

from hyptorch.config import CLAMP_MAX, CLAMP_MIN, EPS


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(CLAMP_MIN, CLAMP_MAX)
        ctx.save_for_backward(x)
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input**2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x + torch.sqrt_(1 + x.pow(2))).clamp_min_(EPS).log_()

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        return grad_output / (1 + input**2) ** 0.5


class RiemannianGradient(torch.autograd.Function):
    # TODO: Add curvature as a parameter
    # curvature = 1.0

    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor, curvature: Union[float, torch.Tensor]) -> torch.Tensor:
        c = torch.as_tensor(curvature).type_as(x)
        ctx.save_for_backward(x, c)
        return x

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        input, c = ctx.saved_tensors
        # Compute Riemannian gradient scale factor
        scale = (1 - c * input.pow(2).sum(-1, keepdim=True)).pow(2) / 4
        return grad_output * scale


def artanh(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse hyperbolic tangent function.

    Parameters
    ----------
    x : tensor
        Input tensor.

    Returns
    -------
    tensor
        Inverse hyperbolic tangent of the input tensor.
    """
    return Artanh.apply(x)


def arsinh(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse hyperbolic sine function.

    Parameters
    ----------
    x : tensor
        Input tensor.

    Returns
    -------
    tensor
        Inverse hyperbolic sine of the input tensor.
    """
    return Arsinh.apply(x)


def arcosh(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Inverse hyperbolic cosine function.

    Parameters
    ----------
    x : tensor
        Input tensor.
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    tensor
        Inverse hyperbolic cosine of the input tensor.
    """
    x = x.clamp(-1 + eps, 1 - eps)
    return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))
