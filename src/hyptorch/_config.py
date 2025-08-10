from dataclasses import dataclass
from typing import ClassVar, Final


@dataclass(frozen=True)
class NumericalConstants:
    EPS: ClassVar[Final[float]] = 1e-5

    MIN_NORM_THRESHOLD: ClassVar[Final[float]] = EPS

    PROJECTION_EPS: ClassVar[Final[float]] = 1e-3
    MAX_NORM: ClassVar[Final[float]] = 1 - PROJECTION_EPS

    TANH_CLAMP_MIN: ClassVar[Final[float]] = -15.0
    TANH_CLAMP_MAX: ClassVar[Final[float]] = 15.0
    ATANH_CLAMP_MIN: ClassVar[Final[float]] = -1 + EPS
    ATANH_CLAMP_MAX: ClassVar[Final[float]] = 1 - EPS
