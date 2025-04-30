"""
Base manifold implementation.
"""

from abc import ABC, abstractmethod

import torch


class Manifold(ABC):
    """
    Abstract base class for manifolds.

    Parameters
    ----------
    name : str
        Name of the manifold.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def dist(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute distance between points on the manifold.

        Parameters
        ----------
        x : tensor
            First point.
        y : tensor
            Second point.

        Returns
        -------
        tensor
            Distance between points.
        """
        pass

    @abstractmethod
    def expmap(self, x: torch.Tensor, tangent_vector: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Exponential map from tangent space to manifold.

        Parameters
        ----------
        x : tensor
            Point on the manifold.
        tangent_vector : tensor
            Tangent vector at x.

        Returns
        -------
        tensor
            Point on the manifold.
        """
        pass

    @abstractmethod
    def logmap(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Logarithmic map from manifold to tangent space.

        Parameters
        ----------
        x : tensor
            Point on the manifold.
        y : tensor
            Another point on the manifold.

        Returns
        -------
        tensor
            Tangent vector at x that points toward y.
        """
        pass

    @abstractmethod
    def ptransp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Parallel transport of tangent vector.

        Parameters
        ----------
        x : tensor
            Starting point on the manifold.
        y : tensor
            End point on the manifold.
        v : tensor
            Tangent vector at x.

        Returns
        -------
        tensor
            Tangent vector at y.
        """
        pass

    @abstractmethod
    def projection(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Project point onto the manifold.

        Parameters
        ----------
        x : tensor
            Point to project.

        Returns
        -------
        tensor
            Projected point on the manifold.
        """
        pass
