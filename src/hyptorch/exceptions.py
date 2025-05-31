class HyperbolicError(Exception):
    """Base exception for hyperbolic operations."""

    pass


class ValidationError(HyperbolicError):
    """Raised when validation fails."""

    pass


class ManifoldError(HyperbolicError):
    """Raised for manifold-specific errors."""

    pass
