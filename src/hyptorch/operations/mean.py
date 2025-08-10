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
    model : HyperbolicMobiusModel, optional
        The hyperbolic model to use. Currently only supports PoincareBall.
        If None, creates a new PoincareBall with default curvature. Default is None.

    Attributes
    ----------
    model : HyperbolicMobiusModel
        The hyperbolic model (currently only PoincareBall is supported).
    curvature : torch.Tensor
        The curvature parameter of the model.

    Raises
    ------
    NotImplementedError
        If model is not an instance of PoincareBall.

    Notes
    -----
    The hyperbolic mean is computed using the following algorithm:

    1. Transform points from Poincaré to Klein model
    2. Compute Lorentz factors for each point in Klein model
    3. Calculate weighted average using Lorentz factors as weights
    4. Transform result back to Poincaré model
    """

    def __init__(self, model: Optional[HyperbolicMobiusModel] = None):
        if model is None:
            model = PoincareBall()

        if not isinstance(model, PoincareBall):
            raise NotImplementedError("Hyperbolic mean currently only supports Poincaré ball model")

        self.model = model
        self.curvature = model.curvature

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
