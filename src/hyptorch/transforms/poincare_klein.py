import torch

from hyptorch.tensor import squared_norm
from hyptorch.transforms.base import GeometricTransform


class PoincareToKlein(GeometricTransform):
    """
    Transform points from the Poincaré ball model to the Klein disk model.

    This transformation converts points between two models of hyperbolic geometry:
    from the conformal Poincaré ball model to the projective Klein disk model.

    Parameters
    ----------
    manifold : HyperbolicManifold
        The Poincaré ball manifold providing the curvature parameter.

    Notes
    -----
    The Poincaré model is conformal (preserves angles) while the Klein model
    has straight lines as geodesics. This transformation is useful for operations
    that are simpler in the Klein model, such as computing hyperbolic means.

    The transformation is defined as:

    .. math::

        \\mathbf{x}_\\mathbb{K} = \\frac{2\\mathbf{x}_\\mathbb{D}}{1 + c \\|\\mathbf{x}_\\mathbb{D}\\|^2}

    where:
    - :math:`\\mathbf{x}_\\mathbb{D}` is a point on the Poincaré ball
    - :math:`\\mathbf{x}_\\mathbb{K}` is the corresponding point in the Klein disk
    - :math:`c` is the curvature parameter

    See Also
    --------
    KleinToPoincare : The inverse transformation.

    Examples
    --------
    >>> from hyptorch.manifolds import PoincareBall
    >>> manifold = PoincareBall(curvature=1.0)
    >>> transform = PoincareToKlein(manifold)
    >>> poincare_point = torch.tensor([[0.3, 0.4]])
    >>> klein_point = transform(poincare_point)
    """

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a point from the Poincaré ball to the Klein disk.

        Parameters
        ----------
        x : torch.Tensor
            Point on the Poincaré ball. Shape (..., dim).

        Returns
        -------
        torch.Tensor
            Corresponding point in the Klein disk. Shape (..., dim).
        """
        return (2 * x) / (1 + self.curvature * squared_norm(x))


class KleinToPoincare(GeometricTransform):
    """
    Transform points from the Klein disk model to the Poincaré ball model.

    This transformation converts points from the projective Klein disk model
    back to the conformal Poincaré ball model of hyperbolic space.

    Parameters
    ----------
    manifold : HyperbolicManifold
        The Poincaré ball manifold providing the curvature parameter.

    Notes
    -----
    This transformation is useful when computations are performed in the Klein
    model (e.g., for hyperbolic averaging) but results need to be expressed in
    the Poincaré model for use with neural network layers.

    The transformation is defined as:

    .. math::

        \\mathbf{x}_\\mathbb{D} = \\frac{\\mathbf{x}_\\mathbb{K}}{1 + \\sqrt{1 - c \\|\\mathbf{x}_\\mathbb{K}\\|^2}}

    where:
    - :math:`\\mathbf{x}_\\mathbb{K}` is a point in the Klein disk
    - :math:`\\mathbf{x}_\\mathbb{D}` is the corresponding point on the Poincaré ball
    - :math:`c` is the curvature parameter

    See Also
    --------
    PoincareToKlein : The inverse transformation.

    Examples
    --------
    >>> from hyptorch.manifolds import PoincareBall
    >>> manifold = PoincareBall(curvature=1.0)
    >>> transform = KleinToPoincare(manifold)
    >>> klein_point = torch.tensor([[0.5, 0.3]])
    >>> poincare_point = transform(klein_point)
    """

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a point from the Klein disk to the Poincaré ball.

        Parameters
        ----------
        x : torch.Tensor
            Point in the Klein disk. Shape (..., dim).

        Returns
        -------
        torch.Tensor
            Corresponding point on the Poincaré ball. Shape (..., dim).
        """
        return x / (1 + torch.sqrt(1 - self.curvature * squared_norm(x)))
