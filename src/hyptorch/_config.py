from dataclasses import dataclass
from typing import ClassVar, Final


@dataclass(frozen=True)
class NumericalConstants:
    """
    Numerical constants for stable hyperbolic geometry operations.

    These constants ensure numerical stability when working with hyperbolic
    geometry, particularly near the boundary of the Poincaré ball and in
    operations involving hyperbolic trigonometric functions.
    """

    EPS: ClassVar[Final[float]] = 1e-5

    # Mathematical operation thresholds
    MIN_NORM_THRESHOLD: ClassVar[Final[float]] = EPS
    TANH_CLAMP_MIN: ClassVar[Final[float]] = -15.0
    TANH_CLAMP_MAX: ClassVar[Final[float]] = 15.0
    ATANH_CLAMP_MIN: ClassVar[Final[float]] = -1 + EPS
    ATANH_CLAMP_MAX: ClassVar[Final[float]] = 1 - EPS

    # Projection to Poincaré ball
    PROJECTION_EPS: ClassVar[Final[float]] = EPS
    MAX_NORM: ClassVar[Final[float]] = 1 - PROJECTION_EPS

    # Curvature bounds
    MIN_CURVATURE: ClassVar[Final[float]] = 1e-5
    MAX_CURVATURE: ClassVar[Final[float]] = 10.0
