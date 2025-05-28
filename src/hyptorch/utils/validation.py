from typing import Union

import torch


class HyperbolicValidationError(ValueError):
    pass


def validate_curvature(curvature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Validate and convert curvature parameter.

    Parameters
    ----------
    curvature : float or torch.Tensor
        The curvature value to validate.

    Returns
    -------
    torch.Tensor
        Validated curvature as tensor.

    Raises
    ------
    HyperbolicValidationError
        If curvature is not positive.
    """
    c = torch.as_tensor(curvature, dtype=torch.float32)
    if torch.any(c <= 0):
        raise HyperbolicValidationError("Curvature must be positive")
    return c
