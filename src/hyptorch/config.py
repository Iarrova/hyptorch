from dataclasses import dataclass
from typing import ClassVar, Final


@dataclass(frozen=True)
class NumericalConstants:
    EPS: ClassVar[Final[float]] = 1e-5

    MIN_NORM_THRESHOLD: ClassVar[Final[float]] = EPS

    PROJECTION_EPS: ClassVar[Final[float]] = 1e-3
    MAX_NORM_SCALE: ClassVar[Final[float]] = 1 - PROJECTION_EPS
