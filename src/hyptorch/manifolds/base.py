"""
Abstract base classes for hyperbolic manifolds.

This module defines the core interfaces for hyperbolic geometry operations.
All manifold implementations inherit from these base classes to ensure
consistent API across different models of hyperbolic space.

Classes
-------
HyperbolicManifold
    Abstract base class providing fundamental hyperbolic operations.
MobiusManifold
    Extension supporting Möbius arithmetic operations.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from hyptorch._config import NumericalConstants
from hyptorch.exceptions import ManifoldError


class HyperbolicManifold(ABC, nn.Module):
    """
    Abstract base class for hyperbolic manifold models.

    This class defines the interface for computational models of hyperbolic geometry,
    providing a common set of operations that must be implemented by specific models.

    Parameters
    ----------
    curvature : float, optional
        The (absolute) curvature parameter :math:`c` of the hyperbolic space.
        The actual curvature of the space is :math:`-c`, so this value must be strictly positive.
    trainable_curvature : bool, optional
        If True, the curvature parameter will be a learnable parameter of the model.
        Default is False.
    device : torch.device or str, optional
        Device on which to create tensors. Default is None (CPU).
    dtype : torch.dtype, optional
        Data type for tensors. Default is torch.float32.

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
    with constant negative curvature :math:`-c`. The convention used here is that the
    stored value is positive, representing the absolute value of the curvature.
    """

    def __init__(
        self,
        curvature: float = 1.0,
        trainable_curvature: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if curvature <= 0:
            raise ManifoldError(f"Curvature must be positive, got {curvature}")

        if device is None:
            device = torch.device("cpu")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.trainable_curvature = trainable_curvature

        if trainable_curvature:
            target = curvature - NumericalConstants.MIN_CURVATURE
            if target <= 0:
                raise ValueError(
                    f"Curvature must be greater than {NumericalConstants.MIN_CURVATURE} to be trainable, got {curvature}"
                )

            self._curvature = nn.Parameter(
                inverse_softplus(target, dtype=dtype).detach().clone().to(**factory_kwargs), requires_grad=True
            )

        else:
            self.register_buffer("_curvature", torch.tensor(curvature, **factory_kwargs))

    @property
    def curvature(self) -> torch.Tensor:
        """
        Get the curvature parameter of the hyperbolic model.

        Returns
        -------
        torch.Tensor
            The curvature parameter as a tensor.
        """
        if self.trainable_curvature:
            curvature = torch.nn.functional.softplus(self._curvature) + NumericalConstants.MIN_CURVATURE
            return torch.clamp(curvature, min=NumericalConstants.MIN_CURVATURE, max=NumericalConstants.MAX_CURVATURE)

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


class MobiusManifold(HyperbolicManifold):
    """
    Extension of HyperbolicManifold that supports Möbius operations.

    This interface adds operations specific to hyperbolic manifolds that support
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

    def extra_repr(self) -> str:
        """
        Return extra representation string showing curvature configuration.

        Returns
        -------
        str
            String representation of curvature parameters.
        """
        return f"curvature={self.curvature.item():.6f}, trainable_curvature={self.trainable_curvature}"


def inverse_softplus(y: float | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Compute inverse of softplus: x such that softplus(x) = y.

    Uses the numerically stable form: log(exp(y) - 1) = log(expm1(y))
    """
    y_tensor = torch.as_tensor(y, dtype=dtype)
    if (y_tensor <= 0).any():
        raise ValueError("Input to inverse_softplus must be positive")
    return torch.log(torch.expm1(y_tensor))
