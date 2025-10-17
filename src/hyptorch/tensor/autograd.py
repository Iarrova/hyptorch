"""
Automatic differentiation for hyperbolic geometry.

This module provides custom autograd functions for Riemannian gradient
computation in hyperbolic space, ensuring correct gradient flow during
backpropagation on manifolds.

Classes
-------
RiemannianGradient
    Custom autograd function for Riemannian gradient scaling.

Functions
---------
apply_riemannian_gradient
    Apply Riemannian gradient correction to tensors on the Poincaré ball.
"""

import torch
from torch.autograd.function import FunctionCtx


class RiemannianGradient(torch.autograd.Function):
    """
    Custom autograd function for Riemannian gradient computation in hyperbolic space.

    This class implements a custom backward pass that scales Euclidean gradients
    to Riemannian gradients appropriate for optimization on the Poincaré ball model.
    It's essential for correct gradient-based optimization in hyperbolic neural networks.

    Notes
    -----
    In hyperbolic geometry, the metric tensor is different from Euclidean space,
    requiring gradient scaling to account for the curvature of the model.
    The Poincaré ball has a conformal metric, meaning the metric tensor is a
    scalar multiple of the identity matrix.

    The scaling factor at point x is:

    .. math::
        \\text{scale} = \\frac{(1 - c\\|x\\|^2)^2}{4}

    This ensures gradients respect the hyperbolic geometry during backpropagation.

    This is implemented as a custom autograd function to efficiently handle
    the gradient transformation during the backward pass without affecting
    the forward computation.
    """

    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns input unchanged but saves context for backward.

        Parameters
        ----------
        ctx : FunctionCtx
            Context object for storing information needed in backward pass.
        x : torch.Tensor
            Point on the Poincaré ball.
        curvature : torch.Tensor
            Curvature of the hyperbolic space (positive scalar).

        Returns
        -------
        torch.Tensor
            The input x unchanged.

        Notes
        -----
        The forward pass is an identity operation. The gradient scaling
        only affects the backward pass, allowing this function to be
        transparently inserted into computational graphs.
        """
        ctx.save_for_backward(x, curvature)
        return x

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Backward pass - scales Euclidean gradient to Riemannian gradient.

        Parameters
        ----------
        ctx : FunctionCtx
            Context with saved tensors from forward pass.
        grad_output : torch.Tensor
            Euclidean gradient flowing backward.

        Returns
        -------
        tuple[torch.Tensor, None]
            - Scaled Riemannian gradient with respect to x.
            - None for the curvature gradient (no gradient needed).

        Notes
        -----
        The Riemannian gradient is computed as:

        .. math::
            \\nabla_R f(x) = \\frac{(1 - c\\|x\\|^2)^2}{4} \\nabla_E f(x)

        where :math:`\\nabla_R` is the Riemannian gradient and :math:`\\nabla_E`
        is the Euclidean gradient.

        This scaling ensures that gradient descent steps respect the hyperbolic
        geometry, moving along geodesics rather than straight lines.
        """
        x, curvature = ctx.saved_tensors
        scale = (1 - curvature * x.pow(2).sum(-1, keepdim=True)).pow(2) / 4
        return grad_output * scale, None


def apply_riemannian_gradient(x: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    """
    Apply Riemannian gradient transformation for hyperbolic optimization.

    This function wraps the RiemannianGradient autograd function, providing
    a convenient interface for applying gradient scaling in hyperbolic neural
    networks. It ensures that gradient-based optimization respects the
    geometry of the Poincaré ball.

    Parameters
    ----------
    x : torch.Tensor
        Point on the Poincaré ball where gradient scaling should be applied.
    curvature : torch.Tensor
        Positive curvature parameter of the hyperbolic space. Scalar tensor.

    Returns
    -------
    torch.Tensor
        The input x with Riemannian gradient computation attached.
        Forward pass returns x unchanged, but backward pass will scale gradients appropriately.

    Notes
    -----
    This function should be applied to points on the Poincaré ball when
    they are created or after projecting to the manifold. It's particularly
    important for:

    1. Points created by mapping from Euclidean space (e.g., in ToPoincare)
    2. Learned parameters that live on the manifold
    3. After manifold projections that may affect gradient flow

    The gradient scaling factor :math:`\\frac{(1 - c\\|x\\|^2)^2}{4}` approaches
    zero as x approaches the boundary of the Poincaré ball, reflecting the
    infinite distance to the boundary in hyperbolic geometry.
    """
    return RiemannianGradient.apply(x, curvature)
