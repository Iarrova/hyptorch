from abc import ABC, abstractmethod

import torch

from hyptorch.exceptions import ManifoldError


class HyperbolicManifold(ABC):
    """
    Abstract base class for hyperbolic manifolds.

    This class defines the interface for hyperbolic manifolds, providing a common
    set of operations that must be implemented by specific manifold models such as
    the Poincaré ball or the hyperboloid model.

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
    ManifoldError
        If curvature is not positive.

    Notes
    -----
    In hyperbolic geometry, the curvature parameter `c` corresponds to a space
    with constant negative curvature -c. The convention used here is that the
    stored value is positive, representing the absolute value of the curvature.
    """

    def __init__(self, curvature: float = 1.0) -> None:
        if curvature <= 0:
            raise ManifoldError(f"Curvature must be positive, got {curvature}")
        self._curvature = torch.tensor(curvature, dtype=torch.float32)

    @property
    def curvature(self) -> torch.Tensor:
        """
        Get the curvature of the manifold.

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

        The distance is measured along the shortest path (geodesic) on the
        manifold connecting the two points.

        Parameters
        ----------
        x : torch.Tensor
            First point on the manifold.
        y : torch.Tensor
            Second point on the manifold.

        Returns
        -------
        torch.Tensor
            Geodesic distance between x and y.

        Notes
        -----
        The geodesic distance is the hyperbolic metric and is invariant under
        isometries of the hyperbolic space.
        """
        pass

    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a point onto the manifold.

        This method ensures that a point lies within the valid domain of the
        manifold, correcting for numerical errors that may have caused the
        point to drift outside the manifold during computation.

        Parameters
        ----------
        x : torch.Tensor
            Point to project onto the manifold.

        Returns
        -------
        torch.Tensor
            Projected point guaranteed to lie on the manifold. Same shape as input.

        Notes
        -----
        Projection is essential for maintaining numerical stability during
        iterative optimization or when chaining multiple operations.
        """
        pass

    @abstractmethod
    def exponential_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the exponential map from a point in a given direction.

        The exponential map takes a point on the manifold and a tangent vector
        at that point, and returns the point reached by following the geodesic
        in that direction for unit time.

        Parameters
        ----------
        x : torch.Tensor
            Base point on the manifold.
        v : torch.Tensor
            Tangent vector at x.

        Returns
        -------
        torch.Tensor
            Point on the manifold reached by the exponential map.

        See Also
        --------
        logarithmic_map : Inverse operation.
        exponential_map_at_origin : Specialized version for origin.

        Notes
        -----
        The exponential map is fundamental for optimization on manifolds,
        as it allows moving from a point in the direction of a gradient
        while remaining on the manifold.
        """
        pass

    @abstractmethod
    def exponential_map_at_origin(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the exponential map from the origin.

        Specialized and often more efficient version of the exponential map
        when the base point is the origin of the manifold.

        Parameters
        ----------
        v : torch.Tensor
            Tangent vector at the origin.

        Returns
        -------
        torch.Tensor
            Point on the manifold reached from the origin.

        See Also
        --------
        exponential_map : General exponential map.
        logarithmic_map_at_origin : Inverse operation.

        Notes
        -----
        This operation is particularly useful for parameterizing points on
        the manifold using vectors in Euclidean space.
        """
        pass

    @abstractmethod
    def logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the logarithmic map between two points.

        The logarithmic map finds the initial velocity of the geodesic
        starting at x that reaches y in unit time.

        Parameters
        ----------
        x : torch.Tensor
            Base point on the manifold.
        y : torch.Tensor
            Target point on the manifold.

        Returns
        -------
        torch.Tensor
            Tangent vector at x pointing toward y.

        See Also
        --------
        exponential_map : Inverse operation.
        logarithmic_map_at_origin : Specialized version for origin.

        Notes
        -----
        The logarithmic map is the inverse of the exponential map and is
        essential for computing geodesics and parallel transport.
        """
        pass

    @abstractmethod
    def logarithmic_map_at_origin(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the logarithmic map from a point to the origin.

        Specialized version of the logarithmic map when the base point
        is the origin, mapping a point on the manifold to a tangent vector
        at the origin.

        Parameters
        ----------
        x : torch.Tensor
            Point on the manifold.

        Returns
        -------
        torch.Tensor
            Tangent vector at origin that maps to x.

        See Also
        --------
        logarithmic_map : General logarithmic map.
        exponential_map_at_origin : Inverse operation.
        """
        pass


class MobiusManifold(HyperbolicManifold):
    """
    Extension of HyperbolicManifold that supports Möbius operations.

    This interface adds operations specific to the Poincaré ball model and
    other Möbius-compatible manifolds, such as Möbius addition and Möbius
    matrix-vector multiplication. These operations are useful for defining
    hyperbolic neural networks and other geometric transformations.
    """

    @abstractmethod
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Perform Möbius addition of two points.

        Möbius addition is the hyperbolic analog of vector addition in
        Euclidean space, providing a group operation on the manifold.

        Parameters
        ----------
        x : torch.Tensor
            First point on the manifold.
        y : torch.Tensor
            Second point on the manifold.

        Returns
        -------
        torch.Tensor
            Result of Möbius addition.

        Notes
        -----
        Möbius addition satisfies the properties:
        - Identity: :math:`x \\oplus 0 = x`
        - Inverse: :math:`x \\oplus (-x) = 0`
        - Non-commutativity: :math:`x \\oplus y \\neq y \\oplus x` (in general)
        - Non-associativity: :math:`(x \\oplus y) \\oplus z \\neq x \\oplus (y \\oplus z)` (in general)
        """
        pass

    @abstractmethod
    def mobius_matvec(self, matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """
        Perform Möbius matrix-vector multiplication.

        Generalizes matrix-vector multiplication to hyperbolic space, essential
        for implementing linear layers in hyperbolic neural networks.

        Parameters
        ----------
        matrix : torch.Tensor
            Weight matrix.
        vector : torch.Tensor
            Point on the manifold.

        Returns
        -------
        torch.Tensor
            Result of Möbius matrix-vector multiplication on the manifold.

        See Also
        --------
        mobius_add : Möbius addition operation.

        Notes
        -----
        This operation is crucial for defining linear transformations that
        respect the hyperbolic geometry, used in layers like HypLinear.
        """
        pass
