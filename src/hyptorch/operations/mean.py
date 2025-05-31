from typing import Optional

import torch

from hyptorch.manifolds.base import HyperbolicManifold
from hyptorch.manifolds.poincare import PoincareBall
from hyptorch.manifolds.transformations import KleinToPoincareTransform, PoincareToKleinTransform
from hyptorch.operations.tensor import squared_norm


class HyperbolicMean:
    def __init__(self, manifold: Optional[HyperbolicManifold] = None):
        if manifold is None:
            manifold = PoincareBall()

        if not isinstance(manifold, PoincareBall):
            raise NotImplementedError("Hyperbolic mean currently only supports Poincaré ball manifold")

        self.manifold = manifold
        self.curvature = manifold.curvature

        # Initialize transformation objects
        self._poincare_to_klein = PoincareToKleinTransform(curvature=self.curvature.item())
        self._klein_to_poincare = KleinToPoincareTransform(curvature=self.curvature.item())

    def lorenz_factor(self, x_klein: torch.Tensor) -> torch.Tensor:
        """
        Compute the Lorentz (gamma) factor for a point on the Klein disk model of hyperbolic space.

        The Lorentz factor arises in the hyperboloid and Klein models of hyperbolic geometry and is used to account for the distortion introduced
        by the curvature when transforming between Euclidean and hyperbolic representations.

        The Lorentz factor for a point :math:`\\mathbf{x}` with curvature :math:`c` is given by:

        .. math::
            \\gamma_{\\mathbf{x}}^c = \\frac{1}{\\sqrt{1 - c \\|\\mathbf{x}\\|^2}}

        where :math:`\\|\\mathbf{x}\\|` is the Euclidean norm of the point in the Klein disk model.

        Parameters
        ----------
        x : torch.Tensor
            Point on the Klein disk.
        curvature : float
            Negative curvature of the space.

        Returns
        -------
        torch.Tensor
            Lorentz factor at the input point.
        """
        return 1 / torch.sqrt(1 - self.curvature * squared_norm(x_klein))

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the hyperbolic mean of points in the Poincaré ball model.

        This method first transforms the points to the Klein model, computes the weighted mean using Lorentz factors,
        and then transforms the result back to the Poincaré ball.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of points on the Poincaré ball.

        Returns
        -------
        torch.Tensor
            Mean point in the Poincaré ball model.
        """
        x_klein = self._poincare_to_klein(x)
        lamb = self.lorenz_factor(x_klein)

        mean = torch.sum(lamb * x_klein, dim=0, keepdim=True) / torch.sum(lamb, dim=0, keepdim=True)
        mean_poincare = self._klein_to_poincare(mean)

        return mean_poincare.squeeze(0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the hyperbolic mean of a batch of points.

        This method allows the HyperbolicMean instance to be called like a function,
        computing the mean of the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of points on the Poincaré ball.

        Returns
        -------
        torch.Tensor
            Mean point in the Poincaré ball model.
        """
        return self.mean(x)
