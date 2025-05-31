from abc import ABC, abstractmethod

import torch

from hyptorch.exceptions import ManifoldError
from hyptorch.operations.tensor import squared_norm


class GeometricTransform(ABC):
    """Base class for geometric transformations with fixed curvature."""

    def __init__(self, curvature: float = 1.0) -> None:
        if curvature <= 0:
            raise ManifoldError(f"Curvature must be positive, got {curvature}")
        self._curvature = torch.tensor(curvature, dtype=torch.float32)

    @property
    def curvature(self) -> torch.Tensor:
        return self._curvature

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class PoincareToKleinTransform(GeometricTransform):
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a point from the Poincaré ball model to the Klein model of hyperbolic space.

        This transformation preserves the hyperbolic geometry while converting the representation of a point from the conformal Poincaré model
        to the projective Klein model. The mapping is defined as:

        .. math::
            f(\\mathbf{x}) = \\frac{\\mathbf{x}}{\\sqrt{1 + c \\|\\mathbf{x}\\|^2}}

        where :math:`\\mathbf{x}` is a point on the Poincaré ball, :math:`c` is the (negative) curvature, and :math:`\\|\\mathbf{x}\\|` is the Euclidean norm.

        Parameters
        ----------
        x : torch.Tensor
            Point on the Poincaré ball.
        curvature : float or torch.Tensor
            Ball negative curvature.

        Returns
        -------
        torch.Tensor
            Corresponding point in the Klein model.
        """
        return (2 * x) / (1 + self.curvature * squared_norm(x))


class KleinToPoincareTransform(GeometricTransform):
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a point from the Klein model to the Poincaré ball model of hyperbolic space.

        This transformation converts a point represented in the projective Klein model to its equivalent in the conformal Poincaré model.
        The mapping ensures that the hyperbolic structure is preserved across representations.

        The transformation is defined as:

        .. math::
            f(\\mathbf{x}) = \\frac{\\mathbf{x}}{1 + \\sqrt{1 - c \\|\\mathbf{x}\\|^2}}

        where :math:`\\mathbf{x}` is a point in the Klein model, :math:`c` is the (negative) curvature, and :math:`\\|\\mathbf{x}\\|` is the Euclidean norm.

        Parameters
        ----------
        x : torch.Tensor
            Point in the Klein model.
        curvature : float or torch.Tensor
            Ball negative curvature.

        Returns
        -------
        torch.Tensor
            Corresponding point on the Poincaré ball.
        """
        return x / (1 + torch.sqrt(1 - self.curvature * squared_norm(x)))
