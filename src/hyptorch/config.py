"""
Configuration and constants for hyperbolic neural networks.
"""

# Numerical stability constants
EPS = 1e-5
CLAMP_MIN = -1.0 + EPS
CLAMP_MAX = 1.0 - EPS
TANH_CLAMP = 15.0

# Default values
DEFAULT_CURVATURE = 1.0
