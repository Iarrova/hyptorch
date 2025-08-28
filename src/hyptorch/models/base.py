from abc import ABC, abstractmethod

import torch

from hyptorch.exceptions import ModelError


class HyperbolicModel(ABC):
    """
    Abstract base class for hyperbolic manifold models.

    This class defines the interface for computational models of hyperbolic geometry,
    providing a common set of operations that must be implemented by specific models.

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
    In hyperbolic geometry, the curvature parameter `c` corresponds to a space
    with constant negative curvature :math:`-c`. The convention used here is that the
    stored value is positive, representing the absolute value of the curvature.
    """

    def __init__(self, curvature: float = 1.0) -> None:
        if curvature <= 0:
            raise ModelError(f"Curvature must be positive, got {curvature}")
        self._curvature = torch.tensor(curvature, dtype=torch.float32)

    @property
    def curvature(self) -> torch.Tensor:
        """
        Get the curvature parameter of the hyperbolic model.

        Returns
        -------
        torch.Tensor
            The curvature parameter as a tensor.
        """
        return self._curvature

    @abstractmethod
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the geodesic distance between two points.

        The distance is measured along the shortest path (geodesic) in the
        hyperbolic space connecting the two points.

        Parameters
        ----------
        x, y : torch.Tensor
            Points in the model's coordinate system.

        Returns
        -------
        torch.Tensor
            Geodesic distance between x and y.
        """
        pass

    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a point onto the valid domain of the model.

        This method ensures that a point lies within the valid coordinate domain
        of the hyperbolic model, correcting for numerical errors that may have
        caused the point to drift outside during computation.

        Parameters
        ----------
        x : torch.Tensor
            Point to project onto the model domain.

        Returns
        -------
        torch.Tensor
            Projected point guaranteed to lie within the valid model domain.
            Same shape as input.

        Notes
        -----
        Projection is essential for maintaining numerical stability during
        iterative optimization or when chaining multiple operations.

        Each model has its own valid coordinate domain.
        """
        pass

    @abstractmethod
    def exponential_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the exponential map from a point in a given direction.

        The exponential map serves as a bridge between the tangent space (Euclidean space)
        and the manifold (Hyperbolic space). Essentially, it converts Euclidean features to
        Hyperbolic embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Base point in the model's coordinate system.
        v : torch.Tensor
            Tangent vector at x (the Euclidean embedding).

        Returns
        -------
        torch.Tensor
            Point in the model reached by the exponential map.

        See Also
        --------
        logarithmic_map : Inverse operation.
        exponential_map_at_origin : Specialized version for origin.
        """
        pass

    @abstractmethod
    def exponential_map_at_origin(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the exponential map from the model's origin.

        Specialized and often more efficient version of the exponential map
        when the base point is the origin model's coordinate system.

        Parameters
        ----------
        v : torch.Tensor
            Tangent vector at the origin (the Euclidean embedding).

        Returns
        -------
        torch.Tensor
            Point in the model reached from the origin.

        See Also
        --------
        exponential_map : General exponential map.
        logarithmic_map_at_origin : Inverse operation.
        """
        pass

    @abstractmethod
    def logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the logarithmic map between two points.

        The logarithmic map serves as a bridge between the manifold (Hyperbolic space)
        and the tangent space (Euclidean space). Essentially, it extracts Hyperbolic features to
        Euclidean embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Base point in the model's coordinate system.
        y : torch.Tensor
            Target point in the model's coordinate system.

        Returns
        -------
        torch.Tensor
            Tangent vector at x pointing toward y.

        See Also
        --------
        exponential_map : Inverse operation.
        logarithmic_map_at_origin : Specialized version for origin.
        """
        pass

    @abstractmethod
    def logarithmic_map_at_origin(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the logarithmic map from a point to the origin.

        Specialized version of the logarithmic map when the base point
        is the origin, mapping a point in the model to a tangent vector
        at the origin.

        Parameters
        ----------
        y : torch.Tensor
            Point in the model's coordinate system.

        Returns
        -------
        torch.Tensor
            Tangent vector at origin that maps to y.

        See Also
        --------
        logarithmic_map : General logarithmic map.
        exponential_map_at_origin : Inverse operation.
        """
        pass


class HyperbolicMobiusModel(HyperbolicModel):
    """
    Extension of HyperbolicModel that supports Möbius operations.

    This interface adds operations specific to hyperbolic models that support
    Möbius operations, such as Möbius addition and Möbius matrix-vector
    multiplication. These operations are essential for implementing hyperbolic
    neural networks and other geometric transformations in hyperbolic space.
    """

    @abstractmethod
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Perform Möbius addition of two points.

        Möbius addition is the hyperbolic analog of vector addition in
        Euclidean space, providing a group operation for points in the
        hyperbolic model.

        Parameters
        ----------
        x, y : torch.Tensor
            Points in the model's coordinate system.

        Returns
        -------
        torch.Tensor
            Result of Möbius addition.
        """
        pass

    @abstractmethod
    def mobius_matvec(self, m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Perform Möbius matrix-vector multiplication.

        Generalizes matrix-vector multiplication to hyperbolic space, essential
        for implementing linear transformations in hyperbolic neural networks.

        Parameters
        ----------
        m : torch.Tensor
            Weight matrix for the transformation.
        v : torch.Tensor
            Point in the model's coordinate system.

        Returns
        -------
        torch.Tensor
            Result of Möbius matrix-vector multiplication in the model's coordinate system.

        See Also
        --------
        mobius_add : Möbius addition operation.
        """
        pass
