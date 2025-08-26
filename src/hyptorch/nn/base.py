import torch
import torch.nn as nn

from hyptorch.models.base import HyperbolicMobiusModel


class HyperbolicLayer(nn.Module):
    """
    Base class for hyperbolic neural network layers.

    This abstract class provides a foundation for all hyperbolic layers,
    maintaining a reference to the underlying hyperbolic manifold and
    providing convenient access to its curvature.

    Parameters
    ----------
    manifold : HyperbolicManifold
        The hyperbolic manifold on which the layer operates.

    Attributes
    ----------
    manifold : HyperbolicManifold
        The hyperbolic manifold instance.
    curvature : torch.Tensor
        The curvature of the manifold (accessible via property).

    Notes
    -----
    All hyperbolic layers should inherit from this base class to ensure
    consistent handling of the manifold and its properties.
    """

    def __init__(self, manifold: HyperbolicMobiusModel):
        super().__init__()
        self.manifold = manifold

    @property
    def curvature(self) -> torch.Tensor:
        """
        Get the curvature of the layer's manifold.

        Returns
        -------
        torch.Tensor
            The curvature parameter of the manifold.
        """
        return self.manifold.curvature
