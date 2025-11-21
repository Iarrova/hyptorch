"""
Poincaré ball model of hyperbolic space.

This module implements the Poincaré ball model, one of the most widely used
representations of hyperbolic geometry in machine learning. The model represents
hyperbolic space as the interior of a unit ball with a Riemannian metric.

Classes
-------
PoincareBall
    Complete implementation of hyperbolic operations in the Poincaré ball model.
"""

import torch

from hyptorch._config import NumericalConstants
from hyptorch.manifolds.base import MobiusManifold
from hyptorch.tensor import atanh, dot_product, norm, squared_norm, tanh


class PoincareBall(MobiusManifold):
    """
    Poincaré ball model of hyperbolic space.

    The Poincaré ball model represents n-dimensional hyperbolic geometry as the interior
    of a unit ball in Euclidean space. It provides a conformal representation, preserving angles
    but distorting distances.

    The model domain is defined as:

    .. math::
        \\mathbb{D}^n_c = \\{\\mathbf{x} \\in \\mathbb{R}^n : c\\|\\mathbf{x}\\|^2 < 1\\}

    where the boundary of the ball (:math:`c\\|\\mathbf{x}\\|^2 = 1`) represents points at infinity.

    Parameters
    ----------
    curvature : float, optional
        The curvature parameter. Default is 1.0 for the unit Poincaré ball.
    trainable_curvature : bool, optional
        If True, the curvature parameter will be a learnable parameter of the model.
        Default is False.
    device : torch.device or str, optional
        Device on which to create tensors. Default is None (CPU).
    dtype : torch.dtype, optional
        Data type for tensors. Default is torch.float32.

    Notes
    -----
    This implementation is optimized for neural networks and provides numerically stable
    versions of all hyperbolic operations. All returned points are automatically projected
    to ensure they remain within the valid domain.
    """

    def __init__(
        self,
        curvature: float = 1.0,
        trainable_curvature: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(curvature, trainable_curvature, device, dtype)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic distance using the Poincaré ball metric.

        .. math::
            d_c(\\mathbf{x}, \\mathbf{y}) = \\left\\frac{2}{\\sqrt{c}}\\right \\text{arctanh}(\\sqrt{c} \\|-\\mathbf{x} \\oplus_{c} \\mathbf{y}\\|)
        """
        sqrt_c = torch.sqrt(self.curvature)
        return (2 / sqrt_c) * atanh(sqrt_c * norm(self.mobius_add(-x, y)))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a point to lie strictly within the Poincaré ball.

        Points with norm :math:`\\geq \\frac{1}{\\sqrt{c}}` are scaled to lie within the valid domain,
        with a small margin for numerical stability.

        .. math::

            \\text{proj}(\\mathbf{x}) =
            \\begin{cases}
                \\frac{\\mathbf{x}}{\\|\\mathbf{x}\\|} \\cdot r_{\\text{max}} & \\text{if } \\|x\\| > r_{\\text{max}} \\
                \\mathbf{x} & \\text{otherwise}
            \\end{cases}
            \\quad \\text{where} \\quad r_{\\text{max}} = \\frac{1 - \\epsilon}{\\sqrt{c}}
            
        where :math:`\\epsilon` is a small constant to ensure the point lies strictly within the ball.
        """
        max_radius = NumericalConstants.MAX_NORM / torch.sqrt(self.curvature)
        x_norm = norm(x, safe=True)
        return torch.where(x_norm > max_radius, x / x_norm * max_radius, x)

    def exponential_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map using the Poincaré ball formulation.

        .. math::
            \\exp_{\\mathbf{x}}^c(\\mathbf{v}) =
            \\mathbf{x} \\oplus_c \\left( \\tanh\\left(\\sqrt{c} \\frac{\\lambda_{\\mathbf{x}}^c \\|\\mathbf{v}\\|}{2}\\right)
            \\frac{\\mathbf{v}}{\\sqrt{c}\\|\\mathbf{v}\\|} \\right)

        where :math:`\\lambda_{\\mathbf{x}}^c = \\frac{2}{1 - c \\|\\mathbf{x}\\|^2}` is the conformal factor at :math:`\\mathbf{x}`.
        """
        sqrt_c = torch.sqrt(self.curvature)
        v_norm = norm(v, safe=True)
        lambda_x = self.conformal_factor(x)
        return self.mobius_add(x, tanh(sqrt_c * lambda_x * v_norm / 2) * v / (sqrt_c * v_norm))

    def exponential_map_at_origin(self, v: torch.Tensor) -> torch.Tensor:
        """
        Simplified exponential map from the origin.

        .. math::
            \\exp_{\\mathbf{0}}^c(\\mathbf{v}) = \\tanh(\\sqrt{c}\\|\\mathbf{v}\\|) \\frac{\\mathbf{v}}{\\sqrt{c}\\|\\mathbf{v}\\|}
        """
        sqrt_c = torch.sqrt(self.curvature)
        v_norm = norm(v, safe=True)
        return tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)

    def logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map using the Poincaré ball formulation.

        .. math::
            \\log_{\\mathbf{x}}^c(\\mathbf{y}) =
            \\frac{2}{\\sqrt{c} \\lambda_{\\mathbf{x}}^c}
            \\text{arctanh}\\left( \\sqrt{c} \\| -\\mathbf{x} \\oplus_c \\mathbf{y} \\| \\right)
            \\frac{-\\mathbf{x} \\oplus_c \\mathbf{y}}{\\| -\\mathbf{x} \\oplus_c \\mathbf{y} \\|}

        where :math:`\\lambda_{\\mathbf{x}}^c = \\frac{2}{1 - c \\|\\mathbf{x}\\|^2}` is the conformal factor at :math:`\\mathbf{x}`
        """
        sqrt_c = torch.sqrt(self.curvature)
        xy = self.mobius_add(-x, y)
        xy_norm = norm(xy, safe=True)
        lambda_x = self.conformal_factor(x)
        return (2 / (sqrt_c * lambda_x)) * atanh(sqrt_c * xy_norm) * xy / xy_norm

    def logarithmic_map_at_origin(self, y: torch.Tensor) -> torch.Tensor:
        """
        Simplified logarithmic map at the origin.

        .. math::
            \\log_{\\mathbf{0}}^c(\\mathbf{y}) = \\frac{1}{\\sqrt{c}} \\text{arctanh}(\\sqrt{c}\\|\\mathbf{y}\\|) \\frac{\\mathbf{y}}{\\|\\mathbf{y}\\|}
        """
        sqrt_c = torch.sqrt(self.curvature)
        y_norm = norm(y, safe=True)
        return (1 / sqrt_c) * atanh(sqrt_c * y_norm) * (y / y_norm)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition in the Poincaré ball.

        .. math::
            \\mathbf{x} \\oplus_{c} \\mathbf{y} = \\frac{(1 + 2c \\langle \\mathbf{x}, \\mathbf{y} \\rangle + c \\|\\mathbf{y}\\|^2) \\mathbf{x} + (1 - c \\|\\mathbf{x}\\|^2) \\mathbf{y}}{1 + 2c \\langle \\mathbf{x}, \\mathbf{y} \\rangle + c^2 \\|\\mathbf{x}\\|^2 \\|\\mathbf{y}\\|^2}
        """
        c = self.curvature
        x2 = squared_norm(x)
        y2 = squared_norm(y)
        xy = dot_product(x, y)

        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c**2 * x2 * y2

        return num / (denom + NumericalConstants.EPS)

    def mobius_matvec(self, m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Mobius matrix-vector multiplication for the Poincaré ball.

        .. math::
            M \\otimes_c \\mathbf{x} = \\frac{1}{\\sqrt{c}}\\tanh\\left(\\frac{\\|M\\mathbf{x}\\|}{\\|\\mathbf{x}\\|}\\arctanh^{-1}{\\sqrt{c}\\|\\mathbf{x}\\|}\\right)\\frac{M \\mathbf{x}}{\\|M \\mathbf{x}\\|}

        The result is automatically projected to ensure it remains in the valid domain.
        """
        sqrt_c = torch.sqrt(self.curvature)

        v_norm = norm(v, safe=True)
        mx = v @ m.transpose(-1, -2)
        mx_norm = norm(mx, safe=True)

        res_c = (1 / sqrt_c) * tanh((mx_norm / v_norm) * atanh(sqrt_c * v_norm)) * (mx / mx_norm)

        cond = torch.linalg.norm(mx, dim=-1, keepdim=True) < 1e-15
        res_0 = torch.zeros_like(res_c)
        res = torch.where(cond, res_0, res_c)

        return self.project(res)

    def conformal_factor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the conformal factor :math:`\\lambda_{\\mathbf{x}}^c = \\frac{2}{1 - c \\|\\mathbf{x}\\|^2}` at point :math:`\\mathbf{x}`.

        The conformal factor relates the Euclidean metric to the hyperbolic
        metric on the Poincaré ball, and approaches infinity as the point
        approaches the boundary.

        Parameters
        ----------
        x : torch.Tensor
            Point in the Poincaré ball.

        Returns
        -------
        torch.Tensor
            Conformal factor at x.
        """
        return 2 / (1 - self.curvature * squared_norm(x))
