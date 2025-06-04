import torch

from hyptorch.config import NumericalConstants
from hyptorch.manifolds.base import MobiusManifold
from hyptorch.operations.tensor import dot_product, norm, squared_norm


class PoincareBall(MobiusManifold):
    """
    Poincaré ball model of hyperbolic space.

    The Poincaré ball is a model of n-dimensional hyperbolic geometry where
    the hyperbolic space is represented as the interior of a unit ball in
    Euclidean space. This model is conformal, meaning it preserves angles
    but not distances.

    Parameters
    ----------
    curvature : float, optional
        The (absolute) curvature parameter :math:`c` of the hyperbolic space.
        The actual curvature of the space is :math:`-c`, so this value must be strictly positive.

    Attributes
    ----------
    curvature : torch.Tensor
        The curvature parameter as a tensor.

    Notes
    -----
    The Poincaré ball model represents hyperbolic space :math:`\\mathbb{H}^n_c` as:

    .. math::
        \\mathbb{D}^n_c = \\{\\mathbf{x} \\in \\mathbb{R}^n : c\\|\\mathbf{x}\\|^2 < 1\\}

    where :math:`c \\geq 0` is the negative curvature. The boundary of the ball
    (where :math:`c\\|\\mathbf{x}\\|^2 = 1`) represents points at infinity.

    This implementation provides all standard operations on the Poincaré
    ball including geodesic distances, exponential and logarithmic maps,
    and Möbius operations for neural network layers.
    """

    def __init__(self, curvature: float = 1.0):
        """
        Initialize the Poincaré ball manifold.

        Parameters
        ----------
        curvature : float, optional
            The (absolute) curvature parameter :math:`c` of the hyperbolic space.
            The actual curvature of the space is :math:`-c`, so this value must be strictly positive.
            Default is 1.0, representing a unit Poincaré ball.
        """
        super().__init__(curvature)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the geodesic distance between two points.

        The distance is measured along the shortest path (geodesic) connecting
        the two points on the Poincaré ball.

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

        Notes
        -----
        The distance is computed using the formula:

        .. math::
            d_c(\\mathbf{x}, \\mathbf{y}) = \\frac{2}{\\sqrt{c}} \\text{arctanh}(\\sqrt{c} \\|\\mathbf{-x} \\oplus_{c} \\mathbf{y}\\|)

        where:
        - :math:`c` is the curvature of the ball
        - :math:`\\oplus_c` denotes Möbius addition under curvature :math:`-c`.

        This metric makes the Poincaré ball a model of hyperbolic space with
        constant negative curvature -c.
        """
        sqrt_c = torch.sqrt(self.curvature)
        return 2 / sqrt_c * torch.atanh(sqrt_c * norm(self.mobius_add(-x, y)))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a point onto the Poincaré ball manifold.

        During optimization or computation, numerical errors can cause points to
        drift outside the valid manifold. This function projects such points back
        to lie strictly within the Poincaré ball boundary.

        Parameters
        ----------
        x : torch.Tensor
            Point to project.

        Returns
        -------
        torch.Tensor
            Projected point guaranteed to lie within the Poincaré ball.

        Notes
        -----
        The projection formula is defined as:

        .. math::

            \\text{proj}(\\mathbf{x}) =
            \\begin{cases}
                \\frac{\\mathbf{x}}{\\|\\mathbf{x}\\|} \\cdot r_{\\text{max}} & \\text{if } \\|x\\| > r_{\\text{max}} \\
                \\mathbf{x} & \\text{otherwise}
            \\end{cases}
            \\quad \\text{where} \\quad r_{\\text{max}} = \\frac{1 - \\epsilon}{\\sqrt{c}}
            
        where:
        - :math:`\\epsilon` is a small constant to ensure the point lies strictly within the ball.
        
        """
        max_radius = NumericalConstants.MAX_NORM_SCALE / torch.sqrt(self.curvature)
        x_norm = norm(x, safe=True)
        return torch.where(x_norm > max_radius, x / x_norm * max_radius, x)

    def exponential_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the exponential map on the Poincaré ball.

        The exponential map takes a tangent vector v at point x and follows
        the geodesic in that direction, returning the endpoint after unit time.

        Parameters
        ----------
        x : torch.Tensor
            Base point on the Poincaré ball.
        v : torch.Tensor
            Tangent vector at `x`.

        Returns
        -------
        torch.Tensor
            Point reached by following the geodesic from x in direction v.

        Notes
        -----
        The exponential map at point :math:`\\mathbf{x} \\in \\mathbb{D}^n_c` in direction :math:`\\mathbf{v}` is given by:

        .. math::

            \\exp_{\\mathbf{x}}^c(\\mathbf{v}) =
            \\mathbf{x} \\oplus_c \\left( \\tanh\\left(\\sqrt{c} \\frac{\\lambda_{\\mathbf{x}}^c \\|\\mathbf{v}\\|}{2}\\right)
            \\frac{\\mathbf{v}}{\\sqrt{c}\\|\\mathbf{v}\\|} \\right)

        where:
        - :math:`\\lambda_{\\mathbf{x}}^c = \\frac{2}{1 - c \\|\\mathbf{x}\\|^2}` is the conformal factor
        - :math:`\\oplus_c` denotes Möbius addition under curvature :math:`-c`.

        See Also
        --------
        logarithmic_map : Inverse operation
        exponential_map_at_origin : Special case when x is the origin
        """
        sqrt_c = torch.sqrt(self.curvature)
        v_norm = norm(v, safe=True)
        lambda_x = self.conformal_factor(x)
        scaled_v = torch.tanh(sqrt_c * lambda_x * v_norm / 2) * v / (sqrt_c * v_norm)
        return self.mobius_add(x, scaled_v)

    def exponential_map_at_origin(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the exponential map from the origin.

        Special case of the exponential map when the base point is the origin,
        which has a simpler formula and is more numerically stable.

        Parameters
        ----------
        v : torch.Tensor
            Tangent vector at the origin.

        Returns
        -------
        torch.Tensor
            Point on the Poincaré ball reached from the origin.

        Notes
        -----
        The exponential map from the origin simplifies to:

        .. math::

            \\exp_{\\mathbf{0}}^c(\\mathbf{v}) = \\tanh(\\sqrt{c}\\|\\mathbf{v}\\|) \\frac{\\mathbf{v}}{\\sqrt{c}\\|\\mathbf{v}\\|}

        This is particularly useful for parameterizing points on the manifold
        using Euclidean vectors.

        See Also
        --------
        exponential_map : General exponential map
        logarithmic_map_at_origin : Inverse operation
        """
        sqrt_c = torch.sqrt(self.curvature)
        v_norm = norm(v, safe=True)
        return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)

    def logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the logarithmic map between two points.

        The logarithmic map finds the initial tangent vector of the unique
        geodesic connecting x to y.

        Parameters
        ----------
        x : torch.Tensor
            Base point on the Poincaré ball.
        y : torch.Tensor
            Target point on the Poincaré ball.

        Returns
        -------
        torch.Tensor
            Tangent vector at x pointing toward y.

        Notes
        -----
        The logarithmic map at point :math:`\\mathbf{x} \\in \\mathbb{D}^n_c` to point :math:`\\mathbf{y}` is given by:

        .. math::

            \\log_{\\mathbf{x}}^c(\\mathbf{y}) =
            \\frac{2}{\\sqrt{c} \\lambda_{\\mathbf{x}}^c}
            \\text{arctanh}\\left( \\sqrt{c} \\| -\\mathbf{x} \\oplus_c \\mathbf{y} \\| \\right)
            \\frac{-\\mathbf{x} \\oplus_c \\mathbf{y}}{\\| -\\mathbf{x} \\oplus_c \\mathbf{y} \\|}

        where:
        - :math:`\\lambda_{\\mathbf{x}}^c = \\frac{2}{1 - c \\|\\mathbf{x}\\|^2}` is the conformal factor
        - :math:`\\oplus_c` denotes Möbius addition under curvature :math:`-c`.

        See Also
        --------
        exponential_map : Inverse operation
        logarithmic_map_at_origin : Special case when x is the origin
        """
        sqrt_c = torch.sqrt(self.curvature)
        xy = self.mobius_add(-x, y)
        xy_norm = norm(xy)
        lambda_x = self.conformal_factor(x)

        return 2 / (sqrt_c * lambda_x) * torch.atanh(sqrt_c * xy_norm) * xy / xy_norm

    def logarithmic_map_at_origin(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the logarithmic map from a point to the origin.

        Special case of the logarithmic map when the base point is the origin,
        mapping points on the manifold back to tangent vectors.

        Parameters
        ----------
        x : torch.Tensor
            Point on the Poincaré ball.

        Returns
        -------
        torch.Tensor
            Tangent vector at origin that exponentially maps to x.

        Notes
        -----
        The logarithmic map to the origin simplifies to:

        .. math::

            \\log_{\\mathbf{0}}^c(\\mathbf{x}) = \\frac{1}{\\sqrt{c}} \\text{arctanh}(\\sqrt{c}\\|\\mathbf{x}\\|) \\frac{\\mathbf{x}}{\\|\\mathbf{x}\\|}

        This is particularly useful for transforming points on the manifold
        into tangent vectors at the origin.

        See Also
        --------
        logarithmic_map : General logarithmic map
        exponential_map_at_origin : Inverse operation
        """
        sqrt_c = torch.sqrt(self.curvature)
        x_norm = norm(x, safe=True)
        return x / x_norm / sqrt_c * torch.atanh(sqrt_c * x_norm)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Perform Möbius addition in the Poincaré ball.

        Möbius addition is the extension of vector addition to hyperbolic space,
        providing a non-commutative group operation on the Poincaré ball.

        Parameters
        ----------
        x : torch.Tensor
            Point on the Poincaré ball.
        y : torch.Tensor
            Point on the Poincaré ball.

        Returns
        -------
        torch.Tensor
            Result of Möbius addition.

        Notes
        -----
        The Möbius addition of two points :math:`\\mathbf{x}, \\mathbf{y} \\in \\mathbb{D}^n_c` is defined as:

        .. math::
            \\mathbf{x} \\oplus_{c} \\mathbf{y} = \\frac{(1 + 2c \\langle \\mathbf{x}, \\mathbf{y} \\rangle + c \\|\\mathbf{y}\\|^2) \\mathbf{x} + (1 - c \\|\\mathbf{x}\\|^2) \\mathbf{y}}{1 + 2c \\langle \\mathbf{x}, \\mathbf{y} \\rangle + c^2 \\|\\mathbf{x}\\|^2 \\|\\mathbf{y}\\|^2}

        where:
        - :math:`\\langle ., .\\rangle` is the inner product

        Properties:
        - Identity: :math:`x \\oplus 0 = x`
        - Inverse: :math:`x \\oplus (-x) = 0`
        - Left cancellation: :math:`(-x) \\oplus_c (x \\oplus_c y) = y`
        - Non-commutative: :math:`x \\oplus y \\neq y \\oplus x` (in general)
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
        Perform Möbius matrix-vector multiplication.

        Generalizes matrix-vector multiplication to the Poincaré ball, preserving
        the hyperbolic structure. This is essential for linear layers in
        hyperbolic neural networks.

        Parameters
        ----------
        matrix : torch.Tensor
            Weight matrix.
        vector : torch.Tensor
            Point on the Poincaré ball.

        Returns
        -------
        torch.Tensor
            Result of the Möbius matrix-vector multiplication.

        Notes
        -----
        The Möbius matrix-vector multiplication is defined as:

        .. math::
            M \\otimes_c \\mathbf{x} = \\frac{1}{\\sqrt{c}}\\tanh\\left(\\frac{\\|M\\mathbf{x}\\|}{\\|\\mathbf{x}\\|}\\tanh^{-1}{\\sqrt{c}\\|\\mathbf{x}\\|}\\right)\\frac{M \\mathbf{x}}{\\|M \\mathbf{x}\\|}

        This operation ensures that linear transformations respect the hyperbolic
        geometry of the Poincaré ball.
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
        Compute the conformal factor at a point.

        The conformal factor relates the Euclidean metric to the hyperbolic
        metric on the Poincaré ball, preserving angles but not distances.

        Parameters
        ----------
        x : torch.Tensor
            Point on the Poincaré ball.

        Returns
        -------
        torch.Tensor
            Conformal factor at x.

        Notes
        -----
        The conformal factor at point :math:`\\mathbf{x} \\in \\mathbb{D}_c^n` is given by:

        .. math::
            \\lambda_{\\mathbf{x}}^c = \\frac{2}{1 - c \\|\\mathbf{x}\\|^2}

        This factor approaches infinity as x approaches the boundary of the ball,
        reflecting the infinite distance to the boundary in hyperbolic geometry.
        """
        return 2 / (1 - self.curvature * squared_norm(x))
