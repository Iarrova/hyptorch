"""
Poincare ball manifold implementation.
"""

from typing import Optional, Union

import torch
from hyptorch.manifolds.base import Manifold
from hyptorch.pmath.poincare import distance, exponential_map, compute_conformal_factor, logarithmic_map, mobius_addition, project


class PoincareManifold(Manifold):
    """
    Poincare ball manifold.

    Parameters
    ----------
    curvature : float
        Negative curvature.
    """

    def __init__(self, curvature: float = 1.0):
        super(PoincareManifold, self).__init__("Poincare")
        self.curvature = curvature

    def dist(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        curvature: Optional[Union[float, torch.Tensor]] = None,
        keepdim: bool = False,
    ) -> torch.Tensor:
        """
        Compute distance between points on the Poincare ball.

        Parameters
        ----------
        x : tensor
            First point.
        y : tensor
            Second point.
        curvature : float or tensor, optional
            Negative curvature. If None, uses the class attribute.
        keepdim : bool
            Whether to keep the dimension.

        Returns
        -------
        tensor
            Distance between points.
        """
        if curvature is None:
            curvature = self.curvature
        return distance(x, y, curvature=curvature, keepdim=keepdim)

    def expmap(
        self,
        x: torch.Tensor,
        tangent_vector: torch.Tensor,
        curvature: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Exponential map from tangent space to Poincare ball.

        Parameters
        ----------
        x : tensor
            Point on the Poincare ball.
        tangent_vector : tensor
            Tangent vector at x.
        curvature : float or tensor, optional
            Negative curvature. If None, uses the class attribute.

        Returns
        -------
        tensor
            Point on the Poincare ball.
        """
        if curvature is None:
            curvature = self.curvature
        return exponential_map(x, tangent_vector, curvature=curvature)

    def logmap(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        curvature: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Logarithmic map from Poincare ball to tangent space.

        Parameters
        ----------
        x : tensor
            Point on the Poincare ball.
        y : tensor
            Another point on the Poincare ball.
        curvature : float or tensor, optional
            Negative curvature. If None, uses the class attribute.

        Returns
        -------
        tensor
            Tangent vector at x that points toward y.
        """
        if curvature is None:
            curvature = self.curvature
        return logarithmic_map(x, y, curvature=curvature)

    def ptransp(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
        curvature: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Parallel transport of tangent vector in Poincare ball.

        Parameters
        ----------
        x : tensor
            Starting point on the Poincare ball.
        y : tensor
            End point on the Poincare ball.
        v : tensor
            Tangent vector at x.
        curvature : float or tensor, optional
            Negative curvature. If None, uses the class attribute.

        Returns
        -------
        tensor
            Tangent vector at y.
        """
        if curvature is None:
            curvature = self.curvature

        c = torch.as_tensor(curvature).type_as(x)

        # Get conformal factors
        lambda_x_val = compute_conformal_factor(x, curvature=c, keepdim=True)
        lambda_y_val = compute_conformal_factor(y, curvature=c, keepdim=True)

        # Compute parallel transport
        return v * (lambda_y_val / lambda_x_val)

    def projection(
        self, x: torch.Tensor, curvature: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Project point onto the Poincare ball.

        Parameters
        ----------
        x : tensor
            Point to project.
        curvature : float or tensor, optional
            Negative curvature. If None, uses the class attribute.

        Returns
        -------
        tensor
            Projected point on the Poincare ball.
        """
        if curvature is None:
            curvature = self.curvature
        return project(x, curvature=curvature)

    def mobius_addition(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        curvature: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Mobius addition in the Poincare ball.

        Parameters
        ----------
        x : tensor
            First point on the Poincare ball.
        y : tensor
            Second point on the Poincare ball.
        curvature : float or tensor, optional
            Negative curvature. If None, uses the class attribute.

        Returns
        -------
        tensor
            Result of Mobius addition.
        """
        if curvature is None:
            curvature = self.curvature
        return mobius_addition(x, y, curvature=curvature)
