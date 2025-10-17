class HyperbolicError(Exception):
    """Base exception for hyperbolic operations."""

    pass


class ManifoldError(HyperbolicError):
    """Raised for manifold-specific errors."""

    pass


class HyperbolicLayerError(HyperbolicError):
    """Raised for errors in hyperbolic layers."""

    pass


class NoHyperbolicManifoldProvidedError(HyperbolicLayerError):
    """Raised when no hyperbolic manifold is provided."""

    def __init__(self, layer_name: str = "Unknown layer") -> None:
        message = f"No hyperbolic manifold provided to {layer_name}."
        super().__init__(message)
