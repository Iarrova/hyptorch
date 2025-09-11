class HyperbolicError(Exception):
    """Base exception for hyperbolic operations."""

    pass


class ModelError(HyperbolicError):
    """Raised for model-specific errors."""

    pass


class HyperbolicLayerError(HyperbolicError):
    """Raised for errors in hyperbolic layers."""

    pass


class NoHyperbolicModelProvidedError(HyperbolicLayerError):
    """Raised when no hyperbolic model is provided."""

    def __init__(self):
        message = "No hyperbolic model provided."
        super().__init__(message)
