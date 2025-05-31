import math

import torch


class ParameterInitializationMixin:
    @staticmethod
    def _init_kaiming_uniform(parameter: torch.nn.Parameter, a: float = math.sqrt(5)) -> None:
        torch.nn.init.kaiming_uniform_(parameter, a=a)

    @staticmethod
    def _init_bias_uniform(parameter: torch.nn.Parameter, fan_in: int) -> None:
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(parameter, -bound, bound)
