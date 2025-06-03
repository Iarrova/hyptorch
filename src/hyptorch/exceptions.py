class HyperbolicError(Exception):
    """Base exception for hyperbolic operations."""

    pass


class ManifoldError(HyperbolicError):
    """Raised for manifold-specific errors."""

    pass
