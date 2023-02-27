# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

__all__ = ["mappings", "inv_softplus", "biased_softplus"]


def inv_softplus(bias: float | torch.Tensor) -> float | torch.Tensor:
    """Inverse softplus function.

    Args:
        bias (float or tensor): the value to be softplus-inverted.
    """
    is_tensor = True
    if not isinstance(bias, torch.Tensor):
        is_tensor = False
        bias = torch.tensor(bias)
    out = bias.expm1().clamp_min(1e-6).log()
    if not is_tensor and out.numel() == 1:
        return out.item()
    return out


class biased_softplus(nn.Module):
    """A biased softplus module.

    The bias indicates the value that is to be returned when a zero-tensor is
    passed through the transform.

    Args:
        bias (scalar): 'bias' of the softplus transform. If bias=1.0, then a _bias shift will be computed such that
            softplus(0.0 + _bias) = bias.
        min_val (scalar): minimum value of the transform.
            default: 0.1
    """

    def __init__(self, bias: float, min_val: float = 0.01) -> None:
        super().__init__()
        self.bias = inv_softplus(bias - min_val)
        self.min_val = min_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x + self.bias) + self.min_val


def mappings(key: str) -> Callable:
    """Given an input string, returns a surjective function f(x): R -> R^+.

    Args:
        key (str): one of "softplus", "exp", "relu", "expln",
            or "biased_softplus". If the key beggins with "biased_softplus",
            then it needs to take the following form:
            ```"biased_softplus_{bias}"``` where ```bias``` can be converted to a floating point number that will be used to bias the softplus function.
            Alternatively, the ```"biased_softplus_{bias}_{min_val}"``` syntax can be used. In that case, the additional ```min_val``` term is a floating point
            number that will be used to encode the minimum value of the softplus transform.
            In practice, the equation used is softplus(x + bias) + min_val, where bias and min_val are values computed such that the conditions above are met.

    Returns:
         a Callable

    """
    _mappings: dict[str, Callable] = {
        "softplus": torch.nn.functional.softplus,
        "exp": torch.exp,
        "relu": torch.relu,
        "biased_softplus": biased_softplus(1.0),
    }
    if key in _mappings:
        return _mappings[key]
    elif key.startswith("biased_softplus"):
        stripped_key = key.split("_")
        if len(stripped_key) == 3:
            return biased_softplus(float(stripped_key[-1]))
        elif len(stripped_key) == 4:
            return biased_softplus(
                float(stripped_key[-2]), min_val=float(stripped_key[-1])
            )
        else:
            raise ValueError(f"Invalid number of args in  {key}")

    else:
        raise NotImplementedError(f"Unknown mapping {key}")
