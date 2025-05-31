import torch

from hyptorch.config import NumericalConstants
from hyptorch.manifolds.base import HyperbolicManifold
from hyptorch.operations.tensor import dot_product, norm, squared_norm


class PoincareBall(HyperbolicManifold):
    def __init__(self, curvature: float = 1.0):
        super().__init__(curvature)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a point onto the Poincaré ball manifold to maintain numerical stability.

        During optimization or computation in hyperbolic space, numerical errors can cause
        points to drift slightly outside the valid manifold. This function safely projects such
        points back inside the Poincaré ball, ensuring that all points lie within the allowable
        radius defined by the curvature.

        In the Poincaré ball model with curvature :math:`-c`, the manifold is the open ball of radius
        :math:`\\frac{1}{\\sqrt{c}}`. Any point :math:`\\mathbf{x} \\in \\mathbb{R}^n` with norm greater than this radius lies outside the
        manifold. This function scales such points to lie just inside the boundary.

        Projection is done using the formula:

        .. math::

            \\text{proj}(\\mathbf{x}) =
            \\begin{cases}
                \\frac{x}{\\|x\\|} \\cdot r_{\\text{max}} & \\text{if } \\|x\\| > r_{\\text{max}} \\
                x & \\text{otherwise}
            \\end{cases}
            \\quad \\text{where} \\quad r_{\\text{max}} = \\frac{1 - \\epsilon}{\\sqrt{c}}
            
        where :math:`\\epsilon` is a small constant to ensure the point lies strictly within the ball.

        Parameters
        ----------
        x : torch.Tensor
            Point on the Poincaré ball.

        Returns
        -------
        torch.Tensor
            A projected point lying strictly within the Poincaré ball.
        """
        max_radius = NumericalConstants.MAX_NORM_SCALE / torch.sqrt(self.curvature)
        x_norm = norm(x, safe=True)
        return torch.where(x_norm > max_radius, x / x_norm * max_radius, x)

    def exponential_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the exponential map on the Poincaré ball manifold.

        The exponential map takes a vector :math:`\\mathbf{v} \\in T_x \\mathbb{D}^n_c` (the tangent space at point :math:`\\mathbf{x}`)
        and maps it to a point :math:`\\textbf{y} \\in \\mathbb{D}^n_c`, where :math:`\\mathbb{D}^n_c` is the Poincaré ball model of hyperbolic space (i.e., the manifold).

        The exponential map is used to move along geodesics starting at :math:`\\mathbf{x}` in the direction of a given tangent vector.

        The exponential map from point :math:`\\mathbf{x}` with tangent vector :math:`\\mathbf{v}` is given by:

        .. math::

            \\exp_{\\mathbf{x}}^c(\\mathbf{v}) =
            \\mathbf{x} \\oplus_c \\left( \\tanh\\left(\\sqrt{c} \\frac{\\lambda_{\\mathbf{x}}^c \\|\\mathbf{v}\\|}{2}\\right)
            \\frac{\\mathbf{v}}{\\sqrt{c}\\|\\mathbf{v}\\|} \\right)

        where :math:`\\lambda_{\\mathbf{x}}^c = \\frac{2}{1 - c \\|\\mathbf{x}\\|^2}` is the conformal factor and
        :math:`\\oplus_c` denotes Möbius addition under curvature :math:`-c`.

        Parameters
        ----------
        x : torch.Tensor
            Base point on the Poincaré ball manifold.
        v : torch.Tensor
            Tangent vector at `x` indicating the direction and magnitude of movement.

        Returns
        -------
        torch.Tensor
            The resulting point on the Poincaré ball after applying the exponential map.
        """
        sqrt_c = torch.sqrt(self.curvature)
        v_norm = norm(v, safe=True)
        lambda_x = self.conformal_factor(x)
        scaled_v = torch.tanh(sqrt_c * lambda_x * v_norm / 2) * v / (sqrt_c * v_norm)
        return self.mobius_add(x, scaled_v)

    def exponential_map_at_origin(self, v: torch.Tensor) -> torch.Tensor:
        sqrt_c = torch.sqrt(self.curvature)
        v_norm = norm(v, safe=True)
        return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)

    def logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the logarithmic map on the Poincaré ball manifold.

        The logarithmic map is the inverse of the exponential map. It maps a point
        :math:`\\mathbf{y} \\in \\mathbb{D}^n_c` on the Poincaré ball (the manifold)
        back to a vector :math:`\\mathbf{v} \\in T_x \\mathbb{D}^n_c` in the tangent space
        at a base point :math:`\\mathbf{x} \\in \\mathbb{D}^n_c`.

        This tangent vector describes the initial velocity of the geodesic starting at
        :math:`\\mathbf{x}` that reaches :math:`\\mathbf{y}`.

        The logarithmic map from point :math:`\\mathbf{x}` to :math:`\\mathbf{y}` is given by:

        .. math::

            \\log_{\\mathbf{x}}^c(\\mathbf{y}) =
            \\frac{2}{\\sqrt{c} \\lambda_{\\mathbf{x}}^c}
            \\text{arctanh}\\left( \\sqrt{c} \\| -\\mathbf{x} \\oplus_c \\mathbf{y} \\| \\right)
            \\frac{-\\mathbf{x} \\oplus_c \\mathbf{y}}{\\| -\\mathbf{x} \\oplus_c \\mathbf{y} \\|}

        where :math:`\\lambda_{\\mathbf{x}}^c = \\frac{2}{1 - c \\|\\mathbf{x}\\|^2}` is the conformal factor and
        :math:`\\oplus_c` denotes Möbius addition under curvature :math:`-c`.

        Parameters
        ----------
        x : torch.Tensor
            Base point on the Poincaré ball manifold (starting point of the geodesic).
        y : torch.Tensor
            Target point on the Poincaré ball manifold (endpoint of the geodesic).

        Returns
        -------
        torch.Tensor
            Tangent vector at `x` pointing toward `y`, representing the geodesic direction and magnitude.
        """
        sqrt_c = torch.sqrt(self.curvature)
        xy = self.mobius_add(-x, y)
        xy_norm = norm(xy)
        lambda_x = self.conformal_factor(x)

        return 2 / (sqrt_c * lambda_x) * torch.atanh(sqrt_c * xy_norm) * xy / xy_norm

    def logarithmic_map_at_origin(self, x: torch.Tensor) -> torch.Tensor:
        sqrt_c = torch.sqrt(self.curvature)
        x_norm = norm(x, safe=True)
        return x / x_norm / sqrt_c * torch.atanh(sqrt_c * x_norm)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Distance between two points on the Poincaré ball.

        The distance is computed using the formula:

        .. math::
            d_c(\\mathbf{x}, \\mathbf{y}) = \\frac{2}{\\sqrt{c}} \\text{arctanh}(\\sqrt{c} \\|\\mathbf{-x} \\oplus_{c} \\mathbf{y}\\|)

        where :math:`c` is the curvature of the ball, and :math:`\\oplus_c` denotes Möbius addition under curvature :math:`-c`.

        Parameters
        ----------
        x : torch.Tensor
            Point on the Poincaré ball.
        y : torch.Tensor
            Point on the Poincaré ball.

        Returns
        -------
        torch.Tensor
            Geodesic distance between x and y.
        """
        sqrt_c = torch.sqrt(self.curvature)
        return 2 / sqrt_c * torch.atanh(sqrt_c * norm(self.mobius_add(-x, y)))

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Mobius addition in hyperbolic space.

        This operation is defined as:

        .. math::
            \\mathbf{x} \\oplus_{c} \\mathbf{y} = \\frac{(1 + 2c \\langle \\mathbf{x}, \\mathbf{y} \\rangle + c \\|\\mathbf{y}\\|^2) \\mathbf{x} + (1 - c \\|\\mathbf{x}\\|^2) \\mathbf{y}}{1 + 2c \\langle \\mathbf{x}, \\mathbf{y} \\rangle + c^2 \\|\\mathbf{x}\\|^2 \\|\\mathbf{y}\\|^2}

        where :math:`\\langle ., .\\rangle` is the inner product, and :math:`\\|.\\|` is the norm.

        Parameters
        ----------
        x, y : torch.Tensor
            Points on the Poincaré ball.
        curvature : float or torch.Tensor
            Ball negative curvature.

        Returns
        -------
        torch.Tensor
            Result of Möbius addition.
        """
        c = self.curvature
        x2 = squared_norm(x)
        y2 = squared_norm(y)
        xy = dot_product(x, y)

        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c**2 * x2 * y2

        return num / (denom + NumericalConstants.EPS)

    def mobius_matvec(self, matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """
        Generalized matrix-vector multiplication in hyperbolic space.

        This operation extends standard matrix-vector multiplication to the Poincaré ball model of hyperbolic geometry.

        Given a matrix :math:`M` and a point :math:`\\mathbf{x} \\in \\mathbb{D}_c^n`, the Möbius matrix-vector multiplication is defined as:

        .. math::
            M \\otimes_c \\mathbf{x} = \\frac{1}{\\sqrt{c}}\\tanh\\left(\\frac{\\|M\\mathbf{x}\\|}{\\|\\mathbf{x}\\|}\\tanh^{-1}{\\sqrt{c}\\|\\mathbf{x}\\|}\\right)\\frac{M \\mathbf{x}}{\\|M \\mathbf{x}\\|}

        Parameters
        ----------
        m : torch.Tensor
            Matrix used for the Möbius multiplication.
        x : torch.Tensor
            Point on the Poincaré ball.
        curvature : float or torch.Tensor
            Ball negative curvature.

        Returns
        -------
        torch.Tensor
            Result of the Möbius matrix-vector multiplication.
        """
        sqrt_c = torch.sqrt(self.curvature)

        vector_norm = norm(vector, safe=True)
        mx = vector @ matrix.transpose(-1, -2)
        mx_norm = norm(mx)

        res_c = (
            (1 / sqrt_c)
            * torch.tanh(mx_norm / vector_norm * torch.atanh(sqrt_c * vector_norm))
            * (mx / mx_norm)
        )

        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)

        return self.project(res)

    def conformal_factor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the conformal factor for a point on the Poincaré ball.

        The conformal factor is used to scale the Euclidean metric into the hyperbolic metric, preserving angles between vectors.

        The conformal factor at point :math:`\\mathbf{x} \\in \\mathbb{D}_c^n` is given by:

        .. math::
            \\lambda_{\\mathbf{x}}^c = \\frac{2}{1 - c \\|\\mathbf{x}\\|^2}

        where :math:`c` is the (negative) curvature of the ball, and :math:`\\|\\mathbf{x}\\|` is the Euclidean norm.

        Parameters
        ----------
        x : torch.Tensor
            Point on the Poincaré ball.
        curvature : float or torch.Tensor
            Ball negative curvature.

        Returns
        -------
        torch.Tensor
            The conformal factor at the input point.
        """
        return 2 / (1 - self.curvature * squared_norm(x))
