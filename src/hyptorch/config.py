from typing import Final

# Numerical stability constants
EPS: Final[float] = 1e-5
CLAMP_MIN: Final[float] = -1.0 + EPS
CLAMP_MAX: Final[float] = 1.0 - EPS
TANH_CLAMP: Final[float] = 15.0

# Projection constants
PROJECTION_EPS: Final[float] = 1e-3
MANIFOLD_TOLERANCE: Final[float] = 1e-3

# Default values
DEFAULT_CURVATURE: Final[float] = 1.0
DEFAULT_REDUCTION: Final[str] = "mean"

# Numerical thresholds
MIN_NORM_THRESHOLD: Final[float] = EPS
MAX_NORM_SCALE: Final[float] = 1 - PROJECTION_EPS
