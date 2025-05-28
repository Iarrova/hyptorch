import torch
import torch.nn as nn

from hyptorch.nn.functional import hyperbolic_softmax
from hyptorch.nn.mixins import ParameterInitializationMixin
from hyptorch.pmath.autograd import RiemannianGradient
from hyptorch.pmath.mappings import (
    exponential_map_at_zero,
    logarithmic_map_at_zero,
    project,
)
from hyptorch.utils.validation import validate_curvature


class HyperbolicMLR(nn.Module, ParameterInitializationMixin):
    """
    Multinomial logistic regression (MLR) classifier operating in hyperbolic space.

    This module performs classification by projecting learned parameters onto the Poincaré
    ball and computing hyperbolic softmax logits. The formulation is designed to respect the
    geometry of hyperbolic space.

    Parameters
    ----------
    ball_dim : int
        Dimensionality of the Poincaré ball (input space).
    n_classes : int
        Number of target classes.
    curvature : float
        Negative curvature of the Poincaré ball.
    """

    def __init__(self, ball_dim: int, n_classes: int, curvature: float):
        super(HyperbolicMLR, self).__init__()
        self.curvature = curvature
        self.n_classes = n_classes
        self.ball_dim = ball_dim

        self.a_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the module.
        """
        self._init_kaiming_uniform(self.a_vals)
        self._init_kaiming_uniform(self.p_vals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for hyperbolic MLR.

        Parameters
        ----------
        x : tensor
            Input tensor.

        Returns
        -------
        tensor
            Logits for classification.
        """
        curvature = validate_curvature(self.curvature)

        # Map points to Poincare ball
        p_vals_poincare = exponential_map_at_zero(self.p_vals, curvature=curvature)

        # Calculate conformal factor
        conformal_factor = 1 - curvature * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)

        # Apply conformal factor to weights
        a_vals_poincare = self.a_vals * conformal_factor

        # Compute hyperbolic softmax (logits)
        return hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, curvature)

    def extra_repr(self) -> str:
        """String representation."""
        return f"ball_dim={self.ball_dim}, n_classes={self.n_classes}, curvature={self.curvature}"


class ToPoincare(nn.Module):
    """
    Maps Euclidean points to the Poincaré ball model of hyperbolic space.

    This module supports optional training of the curvature and a learned reference point
    for exponential mapping.

    Parameters
    ----------
    curvature : float
        Negative curvature of the Poincaré ball.
    """

    def __init__(self, curvature: float):
        super(ToPoincare, self).__init__()

        self.curvature = curvature

        self.riemannian = RiemannianGradient
        self.grad_fix = lambda x: self.riemannian.apply(x, self.curvature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for mapping to Poincare ball.

        Parameters
        ----------
        x : tensor
            Input tensor in Euclidean space.

        Returns
        -------
        tensor
            Mapped tensor on Poincare ball.
        """
        # Map input directly to Poincare ball
        mapped = project(exponential_map_at_zero(x, curvature=self.curvature), curvature=self.curvature)

        # Apply Riemannian gradient fix
        return self.grad_fix(mapped)

    def extra_repr(self) -> str:
        """String representation."""
        return f"curvature={self.curvature}"


class FromPoincare(nn.Module):
    """
    Maps points from the Poincaré ball model back to Euclidean space.

    This module supports optional training of the curvature and a learned reference point
    for logarithmic mapping.

    Parameters
    ----------
    curvature : float
        Negative curvature of the Poincaré ball.
    """

    def __init__(self, curvature: float):
        super(FromPoincare, self).__init__()

        self.curvature = curvature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for mapping from Poincare ball.

        Parameters
        ----------
        x : tensor
            Input tensor on Poincare ball.

        Returns
        -------
        tensor
            Mapped tensor in Euclidean space.
        """
        # Map input directly from Poincare ball
        return logarithmic_map_at_zero(x, curvature=self.curvature)

    def extra_repr(self) -> str:
        """String representation."""
        return f"curvature={self.curvature}"
