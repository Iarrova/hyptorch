from abc import ABC, abstractmethod

import torch

from hyptorch.exceptions import ModelError
from hyptorch.operations.tensor import squared_norm


class GeometricTransform(ABC):
    """
    Abstract base class for geometric transformations with fixed curvature.

    This class defines the interface for implementing transformations
    between different models of hyperbolic geometry.
    All transformations maintain a fixed curvature parameter.

    Parameters
    ----------
    curvature : float, optional
        The (absolute) curvature parameter :math:`c` of the hyperbolic space.
        The actual curvature of the space is :math:`-c`, so this value must be strictly positive.

    Attributes
    ----------
    curvature : torch.Tensor
        The curvature parameter as a tensor.

    Raises
    ------
    ModelError
        If curvature is not positive.

    Notes
    -----
    Subclasses must implement the `transform` method to define the specific
    geometric transformation. The class provides a `__call__` method for
    convenient function-like usage.
    """

    def __init__(self, curvature: float = 1.0) -> None:
        if curvature <= 0:
            raise ModelError(f"Curvature must be positive, got {curvature}")
        self._curvature = torch.tensor(curvature, dtype=torch.float32)

    @property
    def curvature(self) -> torch.Tensor:
        """
        Get the curvature parameter of the transformation.

        Returns
        -------
        torch.Tensor
            The curvature parameter as a tensor.
        """
        return self._curvature

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the geometric transformation to a point.

        Parameters
        ----------
        x : torch.Tensor
            Input point to transform.

        Returns
        -------
        torch.Tensor
            Transformed point.
        """
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the transformation using function call syntax.

        This method allows the transform object to be used as a callable,
        providing a convenient interface for applying transformations.

        Parameters
        ----------
        x : torch.Tensor
            Input point to transform.

        Returns
        -------
        torch.Tensor
            Transformed point.

        See Also
        --------
        transform : The underlying transformation method.
        """
        return self.transform(x)


class PoincareToKleinTransform(GeometricTransform):
    """
    Transform points from the Poincaré ball model to the Klein model.

    This transformation converts points between two models of hyperbolic geometry:
    from the conformal (preserves angles) Poincaré ball model to the projective
    (preserves straight lines as geodesics) Klein model.

    See Also
    --------
    KleinToPoincareTransform : The inverse transformation.
    """

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a point from the Poincaré ball model to the Klein model.

        .. math::
            f(\\mathbf{x}_\\mathbb{K}) = \\frac{2\\mathbf{x}_\\mathbb{D}}{1 + c \\|\\mathbf{x}\\|^2}

        Examples
        --------
        >>> transform = PoincareToKleinTransform(curvature=1.0)
        >>> poincare_point = torch.tensor([0.5, 0.3])
        >>> klein_point = transform(poincare_point)
        """
        return (2 * x) / (1 + self.curvature * squared_norm(x))


class KleinToPoincareTransform(GeometricTransform):
    """
    Transform points from the Klein model to the Poincaré ball model.

    This transformation converts points from the projective Klein model back to
    the conformal Poincaré ball model of hyperbolic space, performing the inverse
    of the PoincareToKleinTransform.

    See Also
    --------
    PoincareToKleinTransform : The inverse transformation.
    """

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a point from the Klein model to the Poincaré ball model.

        .. math::
            f(\\mathbf{x}_\\mathbb{D}) = \\frac{\\mathbf{x}_\\mathbb{K}}{1 + \\sqrt{1 - c \\|\\mathbf{x}_\\mathbb{K}\\|^2}}

        Examples
        --------
        >>> transform = KleinToPoincareTransform(curvature=1.0)
        >>> klein_point = torch.tensor([0.5, 0.3])
        >>> poincare_point = transform(klein_point)
        """
        return x / (1 + torch.sqrt(1 - self.curvature * squared_norm(x)))
