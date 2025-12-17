import torch

from hyptorch.manifolds import KleinDisk, PoincareBall
from hyptorch.nn._base import HyperbolicLayer
from hyptorch.transforms import KleinToPoincare, PoincareToKlein


class HyperbolicMean(HyperbolicLayer):
    """
    Hyperbolic mean aggregation layer.

    This layer computes the hyperbolic mean (Einstein midpoint) of a set of
    points on the Poincaré ball by transforming to the Klein model, computing
    the Lorentz-weighted average, and transforming back.

    Parameters
    ----------
    manifold : PoincareBall
        The Poincaré ball manifold on which points lie.

    Attributes
    ----------
    manifold : PoincareBall
        The Poincaré ball manifold.

    Notes
    -----
    The hyperbolic mean is computed via the "Klein trick":

    1. Transform points from Poincaré ball to Klein disk
    2. Compute Lorentz-weighted average in Klein model
    3. Transform result back to Poincaré ball

    This approach leverages the fact that the Einstein midpoint formula
    in the Klein model provides a natural definition of hyperbolic centroid.

    Examples
    --------
    >>> from hyptorch.manifolds import PoincareBall
    >>> manifold = PoincareBall(curvature=1.0)
    >>> mean_layer = HyperbolicMean(manifold)
    >>> # Batch of 10 points in 5D
    >>> points = torch.randn(10, 5) * 0.3
    >>> points = manifold.project(points)
    >>> mean = mean_layer(points)  # Shape: (1, 5)

    See Also
    --------
    KleinDisk.mean : The underlying mean computation in Klein model.
    PoincareToKlein : Transformation used internally.
    KleinToPoincare : Transformation used internally.
    """

    def __init__(self, manifold: PoincareBall) -> None:
        if not isinstance(manifold, PoincareBall):
            raise TypeError(f"Expected PoincareBall manifold, got {type(manifold).__name__}")

        super().__init__(manifold)

        self.to_klein = PoincareToKlein(manifold)
        self.from_klein = KleinToPoincare(manifold)
        self.klein = KleinDisk(manifold.curvature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the hyperbolic mean of input points.

        Parameters
        ----------
        x : torch.Tensor
            Points on the Poincaré ball. Shape (n_points, dim).

        Returns
        -------
        torch.Tensor
            Hyperbolic mean on the Poincaré ball. Shape (1, dim).
        """
        x = self.manifold.project(x)

        klein_points = self.to_klein(x)
        mean_klein = self.klein.mean(klein_points)
        mean_poincare = self.from_klein(mean_klein)

        return self.manifold.project(mean_poincare)

    def extra_repr(self) -> str:
        return f"curvature={self.manifold.curvature.item():.6f}"
