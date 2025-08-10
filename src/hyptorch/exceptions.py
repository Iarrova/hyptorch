class HyperbolicError(Exception):
    """Base exception for hyperbolic operations."""

    pass


class ModelError(HyperbolicError):
    """Raised for model-specific errors."""

    pass
