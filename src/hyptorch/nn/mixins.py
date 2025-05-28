import torch


class ParameterInitializationMixin:
    """Mixin for consistent parameter initialization."""

    @staticmethod
    def _init_kaiming_uniform(parameter: torch.nn.Parameter, a: float = 5**0.5) -> None:
        """Initialize parameter with Kaiming uniform distribution."""
        torch.nn.init.kaiming_uniform_(parameter, a=a)

    @staticmethod
    def _init_bias_uniform(parameter: torch.nn.Parameter, fan_in: int) -> None:
        """Initialize bias parameter."""
        bound = 1 / (fan_in**0.5)
        torch.nn.init.uniform_(parameter, -bound, bound)
