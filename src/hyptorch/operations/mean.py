from typing import Optional

import torch

from hyptorch.models.base import HyperbolicMobiusModel
from hyptorch.models.poincare_ball import PoincareBall
from hyptorch.models.transformations import KleinToPoincareTransform, PoincareToKleinTransform
from hyptorch.operations.tensor import squared_norm


class HyperbolicMean:
    """
    Compute the mean of points in hyperbolic space.

    Parameters
    ----------
    manifold : PoincareBall, optional
        The hyperbolic manifold to use. Currently only supports PoincareBall.
        If None, creates a new PoincareBall with default curvature. Default is None.

    Attributes
    ----------
    manifold : MobiusManifold
        The hyperbolic manifold (must be PoincareBall).
    curvature : torch.Tensor
        The curvature parameter of the manifold.

    Raises
    ------
    NotImplementedError
        If manifold is not an instance of PoincareBall.

    Notes
    -----
    The hyperbolic mean is computed using the following algorithm:

    1. Transform points from Poincaré to Klein model
    2. Compute Lorentz factors for each point in Klein model
    3. Calculate weighted average using Lorentz factors as weights
    4. Transform result back to Poincaré model

    This approach provides a closed-form solution that avoids iterative
    optimization. The Lorentz factors account for the hyperbolic metric
    distortion, giving more weight to points closer to the origin.
    """

    def __init__(self, manifold: Optional[HyperbolicMobiusModel] = None):
        if manifold is None:
            manifold = PoincareBall()

        if not isinstance(manifold, PoincareBall):
            raise NotImplementedError("Hyperbolic mean currently only supports Poincaré ball manifold")

        self.manifold = manifold
        self.curvature = manifold.curvature

        # Initialize transformation objects
        self._poincare_to_klein = PoincareToKleinTransform(curvature=self.curvature.item())
        self._klein_to_poincare = KleinToPoincareTransform(curvature=self.curvature.item())

    def lorentz_factor(self, x_klein: torch.Tensor) -> torch.Tensor:
        """
        Compute the Lorentz factor for points in the Klein disk model.

        The Lorentz factor (also called gamma factor) quantifies the metric
        distortion at each point in hyperbolic space. It's used as a weight
        in the mean computation to account for the non-uniform geometry.

        Parameters
        ----------
        x_klein : torch.Tensor
            Points in the Klein disk model.

        Returns
        -------
        torch.Tensor
            Lorentz factors for each point.
            Values are >= 1, increasing toward infinity at the boundary.

        Notes
        -----
        The Lorentz factor for a point :math:`\\mathbf{x}` in the Klein model
        with curvature :math:`c` is:

        .. math::
            \\gamma_{\\mathbf{x}}^c = \\frac{1}{\\sqrt{1 - c \\|\\mathbf{x}\\|^2}}

        This factor:
        - Equals 1 at the origin (no distortion)
        - Approaches infinity at the boundary (infinite distortion)
        - Weights the contribution of each point in the mean calculation
        """
        return 1 / torch.sqrt(1 - self.curvature * squared_norm(x_klein))

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the hyperbolic mean of points in the Poincaré ball.

        This method computes the Fréchet mean by transforming to the Klein
        model where a closed-form weighted average can be computed.

        Parameters
        ----------
        x : torch.Tensor
            Points on the Poincaré ball.
            All points should be properly projected to the manifold.

        Returns
        -------
        torch.Tensor
            Mean point on the Poincaré ball.
        """
        x_klein = self._poincare_to_klein(x)
        lamb = self.lorentz_factor(x_klein)

        mean = torch.sum(lamb * x_klein, dim=0, keepdim=True) / torch.sum(lamb, dim=0, keepdim=True)
        mean_poincare = self._klein_to_poincare(mean)

        return mean_poincare.squeeze(0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the hyperbolic mean of points (callable interface).

        This method allows the HyperbolicMean instance to be used as a
        function, providing a convenient interface consistent with PyTorch
        operations.

        Parameters
        ----------
        x : torch.Tensor
            Points on the Poincaré ball.

        Returns
        -------
        torch.Tensor
            Mean point on the Poincaré ball.

        See Also
        --------
        mean : The underlying mean computation method.
        """
        return self.mean(x)
