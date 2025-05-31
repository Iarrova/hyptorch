import torch
from torch.autograd.function import FunctionCtx


class RiemannianGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, curvature)
        return x

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        x, c = ctx.saved_tensors
        scale = (1 - c * x.pow(2).sum(-1, keepdim=True)).pow(2) / 4
        return grad_output * scale, None


def riemannian_gradient(x: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    return RiemannianGradient.apply(x, curvature)
