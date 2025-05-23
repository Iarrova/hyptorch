import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.init as init

from hyptorch.nn.functional import hyperbolic_softmax
from hyptorch.pmath.autograd import RiemannianGradient
from hyptorch.pmath.mappings import (
    exponential_map,
    exponential_map_at_zero,
    logarithmic_map,
    logarithmic_map_at_zero,
    project,
)


class HyperbolicMLR(nn.Module):
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
        self.a_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.curvature = curvature
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.reset_parameters()

    def forward(
        self, x: torch.Tensor, curvature: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass for hyperbolic MLR.

        Parameters
        ----------
        x : tensor
            Input tensor.
        curvature : float or tensor, optional
            Negative ball curvature. If None, uses the class attribute.

        Returns
        -------
        tensor
            Logits for classification.
        """
        # Use provided curvature or fall back to class attribute
        if curvature is None:
            curvature = torch.as_tensor(self.curvature).type_as(x)
        else:
            curvature = torch.as_tensor(curvature).type_as(x)

        # Map points to Poincare ball
        p_vals_poincare = exponential_map_at_zero(self.p_vals, curvature=curvature)

        # Calculate conformal factor
        conformal_factor = 1 - curvature * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)

        # Apply conformal factor to weights
        a_vals_poincare = self.a_vals * conformal_factor

        # Compute hyperbolic softmax
        logits = hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, curvature)

        return logits

    def extra_repr(self) -> str:
        """
        Return a string representation of module parameters.

        Returns
        -------
        str
            String representation.
        """
        return "Poincare ball dim={}, n_classes={}, curvature={}".format(
            self.ball_dim, self.n_classes, self.curvature
        )

    def reset_parameters(self):
        """
        Reset the parameters of the module.
        """
        init.kaiming_uniform_(self.a_vals, a=math.sqrt(5))
        init.kaiming_uniform_(self.p_vals, a=math.sqrt(5))


class ToPoincare(nn.Module):
    """
    Maps Euclidean points to the Poincaré ball model of hyperbolic space.

    This module supports optional training of the curvature and a learned reference point
    for exponential mapping.

    Parameters
    ----------
    curvature : float
        Negative curvature of the Poincaré ball.
    train_x : bool, optional
        If True, the exponential map will use a trainable reference point.
        In this case, `ball_dim` must be specified.
    ball_dim : int, optional
        Dimensionality of the Poincaré ball. Required if `train_x=True`.
    riemannian : bool, optional
        If True, uses Riemannian gradient correction (default: True).
    """

    def __init__(
        self,
        curvature: float,
        train_x: bool = False,
        ball_dim: Optional[int] = None,
    ):
        super(ToPoincare, self).__init__()

        # Initialize trainable reference point if requested
        if train_x:
            if ball_dim is None:
                raise ValueError("if train_x=True, ball_dim has to be integer, got {}".format(ball_dim))
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        self.curvature = curvature

        self.train_x = train_x

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
        if self.train_x:
            # Map reference point to Poincare ball
            xp = project(exponential_map_at_zero(self.xp, curvature=self.curvature), curvature=self.curvature)

            # Map input using reference point
            mapped = project(exponential_map(xp, x, curvature=self.curvature), curvature=self.curvature)

            # Apply Riemannian gradient fix
            return self.grad_fix(mapped)

        # Map input directly to Poincare ball
        mapped = project(exponential_map_at_zero(x, curvature=self.curvature), curvature=self.curvature)

        # Apply Riemannian gradient fix
        return self.grad_fix(mapped)

    def extra_repr(self) -> str:
        """
        Return a string representation of module parameters.

        Returns
        -------
        str
            String representation.
        """
        return "curvature={}, train_x={}".format(self.curvature, self.train_x)


class FromPoincare(nn.Module):
    """
    Maps points from the Poincaré ball model back to Euclidean space.

    This module supports optional training of the curvature and a learned reference point
    for logarithmic mapping.

    Parameters
    ----------
    curvature : float
        Negative curvature of the Poincaré ball.
    train_x : bool, optional
        If True, the logarithmic map will use a trainable reference point.
        In this case, `ball_dim` must be specified.
    ball_dim : int, optional
        Dimensionality of the Poincaré ball. Required if `train_x=True`.
    """

    def __init__(
        self,
        curvature: float,
        train_x: bool = False,
        ball_dim: Optional[int] = None,
    ):
        super(FromPoincare, self).__init__()

        # Initialize trainable reference point if requested
        if train_x:
            if ball_dim is None:
                raise ValueError("if train_x=True, ball_dim has to be integer, got {}".format(ball_dim))
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        self.curvature = curvature
        self.train_x = train_x

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
        if self.train_x:
            # Map reference point to Poincare ball
            xp = project(exponential_map_at_zero(self.xp, curvature=self.curvature), curvature=self.curvature)

            # Map input using reference point
            return logarithmic_map(xp, x, curvature=self.curvature)

        # Map input directly from Poincare ball
        return logarithmic_map_at_zero(x, curvature=self.curvature)

    def extra_repr(self) -> str:
        """
        Return a string representation of module parameters.

        Returns
        -------
        str
            String representation.
        """
        return "train_x={}".format(self.train_x)
