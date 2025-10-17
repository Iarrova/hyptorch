"""
Functional interface for hyperbolic neural network operations.

This module provides stateless functions for hyperbolic computations,
following PyTorch's functional API pattern (similar to torch.nn.functional).

Functions
---------
compute_hyperbolic_mlr_logits
    Compute logits for hyperbolic multinomial logistic regression.
"""

import torch

from hyptorch._config import NumericalConstants
from hyptorch.manifolds import PoincareBall
from hyptorch.manifolds.base import MobiusManifold
from hyptorch.tensor import squared_norm


def _batch_mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Perform batch Möbius addition between tensors with different batch dimensions.

    This function computes Möbius addition between all pairs of points from two
    batches, useful for operations like hyperbolic MLR where we need to compute
    additions between data points and multiple class representatives.

    Parameters
    ----------
    x, y : torch.Tensor
        Batch of points on the Poincaré ball.
    c : torch.Tensor
        Curvature parameter (positive). Scalar tensor.

    Returns
    -------
    torch.Tensor
        Result of batch Möbius addition. Shape (batch_x, batch_y, dim).
        Element [i, j, :] contains the Möbius sum of x[i] and y[j].

    Notes
    -----
    This function implements a batched version of Möbius addition where the
    operation is performed between all pairs from the two input batches:

    .. math::
        \\text{result}[i, j] = x[i] \\oplus_c y[j]

    See Also
    --------
    PoincareBall.mobius_add : Single pair Möbius addition

    Examples
    --------
    >>> x = torch.randn(10, 5) * 0.3  # 10 points in 5D
    >>> y = torch.randn(3, 5) * 0.3   # 3 points in 5D
    >>> c = torch.tensor(1.0)
    >>> result = _batch_mobius_add(x, y, c)  # Shape (10, 3, 5)
    """
    # Calculate all pairwise dot products
    xy = torch.einsum("ij,kj->ik", (x, y))

    # Compute squared norms (||x_i||^2 for each x_i) and the same for y
    x2 = squared_norm(x)
    y2 = squared_norm(y)

    num = 1 + 2 * c * xy + c * y2.permute(1, 0)
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y

    denom = 1 + 2 * c * xy + c**2 * x2 * y2.permute(1, 0)

    return num / (denom.unsqueeze(2) + NumericalConstants.EPS)


def compute_hyperbolic_mlr_logits(
    x: torch.Tensor, weights: torch.Tensor, class_points: torch.Tensor, manifold: MobiusManifold
) -> torch.Tensor:
    """
    Compute logits for hyperbolic multinomial logistic regression (MLR).

    This function implements the hyperbolic generalization of softmax/MLR,
    computing class logits for input points based on their hyperbolic distances
    to learned class representatives in the Poincaré ball.

    Parameters
    ----------
    x : torch.Tensor
        Input points on the Poincaré ball. Shape (batch_size, dim).
    weights : torch.Tensor
        Weight vectors (a-values) for each class, scaled by conformal factor.
        Shape (n_classes, dim).
    class_points : torch.Tensor
        Class representatives (p-values) on the Poincaré ball.
        Shape (n_classes, dim).
    manifold : MobiusManifold
        The hyperbolic manifold instance.
        Currently only PoincareBall is supported.

    Returns
    -------
    torch.Tensor
        Logits for each input point and class. Shape (batch_size, n_classes).
        Can be passed to standard softmax for classification probabilities.

    Raises
    ------
    NotImplementedError
        If manifold is not an instance of PoincareBall.

    Notes
    -----
    The hyperbolic MLR generalizes logistic regression to hyperbolic space.
    For each class k with representative :math:`p_k` and weights :math:`a_k`,
    the logit for an input point :math:`x` is:

    .. math::
        \\text{logit}_k(x) = \\frac{\\lambda_{p_k}^c \\|a_k\\|}{\\sqrt{c}}
        \\sinh^{-1}\\left(\\frac{2\\sqrt{c} \\langle a_k, -p_k \\oplus_c x \\rangle}
        {(1 - c\\|-p_k \\oplus_c x\\|^2)\\|a_k\\|}\\right)

    where:
    - :math:`\\lambda_{p_k}^c = \\frac{2}{1 - c\\|p_k\\|^2}` is the conformal factor
    - :math:`\\oplus_c` denotes Möbius addition
    - :math:`\\sinh^{-1}` is the inverse hyperbolic sine (arcsinh)

    The formulation ensures that decision boundaries are geodesic hyperplanes
    in hyperbolic space.

    Examples
    --------
    >>> manifold = PoincareBall(curvature=1.0)
    >>> batch_size, dim, n_classes = 32, 10, 5
    >>> x = torch.randn(batch_size, dim) * 0.3
    >>> x = manifold.project(x)
    >>> weights = torch.randn(n_classes, dim)
    >>> points = torch.randn(n_classes, dim) * 0.3
    >>> points = manifold.project(points)
    >>> logits = compute_hyperbolic_mlr_logits(x, weights, points, manifold)
    >>> probs = torch.softmax(logits, dim=1)  # Classification probabilities
    """
    if not isinstance(manifold, PoincareBall):
        raise NotImplementedError("Hyperbolic softmax only implemented for Poincaré ball")

    c = manifold.curvature
    sqrt_c = torch.sqrt(c)

    # Step 1: Compute conformal factors at each class point
    # λ^c_pk = 2 / (1 - c||p_k||²)
    conformal_factors = 2 / (1 - c * class_points.pow(2).sum(dim=1))

    # Step 2: Compute overall scale factors for each class
    # scale_k = λ^c_pk * ||a_k|| / √c
    class_weight_norms = torch.norm(weights, dim=1)
    scale_factors = conformal_factors * class_weight_norms / sqrt_c

    # Step 3: Compute hyperbolic differences between input points and class points
    # hyperbolic_diff[i,k,:] = -p_k ⊕_c x_i
    hyperbolic_differences = _batch_mobius_add(-class_points, x, c)

    # Step 4: Compute logits using the hyperbolic MLR formula
    # logits[i,k] = scale_k * asinh( (2√c <a_k, hyperbolic_diff[i,k,:]> / (||a_k||(1 - c||hyperbolic_diff[i,k,:]||²)) )
    num = 2 * sqrt_c * torch.sum(hyperbolic_differences * weights.unsqueeze(1), dim=-1)
    denom = torch.norm(weights, dim=1, keepdim=True) * (1 - c * hyperbolic_differences.pow(2).sum(dim=2))

    logits = scale_factors.unsqueeze(1) * torch.asinh(num / denom)

    return logits.permute(1, 0)
