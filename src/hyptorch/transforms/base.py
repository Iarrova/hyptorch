from abc import ABC, abstractmethod

import torch

from hyptorch.manifolds.base import HyperbolicManifold


class GeometricTransform(ABC):
    """
    Base class for geometric transformations between hyperbolic models.

    Parameters
    ----------
    manifold : HyperbolicManifold
        The source hyperbolic manifold providing the curvature parameter.

    Attributes
    ----------
    curvature : torch.Tensor
        The curvature parameter from the linked manifold.
    """

    def __init__(self, manifold: HyperbolicManifold) -> None:
        self._manifold = manifold

    @property
    def manifold(self) -> HyperbolicManifold:
        return self._manifold

    @property
    def curvature(self) -> torch.Tensor:
        return self._manifold.curvature

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
