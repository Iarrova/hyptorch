import torch

from hyptorch.config import NumericalConstants
from hyptorch.manifolds.base import MobiusManifold
from hyptorch.manifolds.poincare import PoincareBall
from hyptorch.operations.tensor import squared_norm


def _batch_mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Perform batch Möbius addition between tensors with different batch dimensions.

    This function computes Möbius addition between all pairs of points from two
    batches, useful for operations like hyperbolic MLR where we need to compute
    additions between data points and multiple class representatives.

    Parameters
    ----------
    x : torch.Tensor
        Batch of points on the Poincaré ball.
    y : torch.Tensor
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
    xy = torch.einsum("ij,kj->ik", (x, y))
    x2 = squared_norm(x)
    y2 = squared_norm(y)

    num = 1 + 2 * c * xy + c * y2.permute(1, 0)
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y

    denom = 1 + 2 * c * xy + c**2 * x2 * y2.permute(1, 0)

    return num / (denom.unsqueeze(2) + NumericalConstants.EPS)


def compute_hyperbolic_mlr_logits(
    x: torch.Tensor, weights: torch.Tensor, points: torch.Tensor, manifold: MobiusManifold
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
        Weights (a-values) for each class, scaled by conformal factor.
        Shape (n_classes, dim).
    points : torch.Tensor
        Class representatives (p-values) on the Poincaré ball.
        Shape (n_classes, dim).
    manifold : MobiusManifold
        The hyperbolic manifold (must be PoincareBall).

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
    # TODO: Check correctness of the formula described in Notes
    if not isinstance(manifold, PoincareBall):
        raise NotImplementedError("Hyperbolic softmax only implemented for Poincaré ball")

    c = manifold.curvature
    sqrt_c = torch.sqrt(c)

    lambda_pkc = 2 / (1 - c * points.pow(2).sum(dim=1))
    k = lambda_pkc * torch.norm(weights, dim=1) / sqrt_c

    mob_add = _batch_mobius_add(-points, x, c)

    num = 2 * sqrt_c * torch.sum(mob_add * weights.unsqueeze(1), dim=-1)
    denom = torch.norm(weights, dim=1, keepdim=True) * (1 - c * mob_add.pow(2).sum(dim=2))

    logits = k.unsqueeze(1) * torch.asinh(num / denom)

    return logits.permute(1, 0)
