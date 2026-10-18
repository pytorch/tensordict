# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from numbers import Number
from typing import Sequence

import numpy as np
import torch
from tensordict.nn.utils import mappings
from torch import distributions as D, nn

# We need this to build the distribution maps
__all__ = [
    "NormalParamExtractor",
    "AddStateIndependentNormalScale",
    "Delta",
]

# speeds up distribution construction
# D.Distribution.set_default_validate_args(False)


class NormalParamExtractor(nn.Module):
    """A module without parameters that splits a tensor into the location and scale of a normal distribution.

    The module splits the last dimension of its first input into two halves:
    the first half is the location, the second half goes through
    ``scale_mapping`` and is clamped to at least ``scale_lb`` to give the
    scale. With an odd size, the location gets one element more than the
    scale. Calling the module with
    more inputs returns them unchanged after the scale:
    ``module(x, *others)`` returns ``(loc, scale, *others)``.

    Args:
        scale_mapping (str, optional): the function that maps the second half
            to positive values, by name: for example ``"softplus"``,
            ``"exp"``, ``"relu"``, ``"biased_softplus_<bias>"`` or ``"none"``
            (no mapping). See :func:`~tensordict.nn.mappings`. Defaults to
            ``"biased_softplus_1.0"``, a softplus shifted so that an input of
            ``0`` gives a scale of ``1``.
        scale_lb (Number, optional): the minimum value of the scale (the
            standard deviation). Defaults to ``1e-4``.

    Examples:
        >>> import torch
        >>> from tensordict.nn.distributions import NormalParamExtractor
        >>> from torch import nn
        >>> module = nn.Linear(3, 4)
        >>> normal_params = NormalParamExtractor()
        >>> tensor = torch.randn(3)
        >>> loc, scale = normal_params(module(tensor))
        >>> print(loc.shape, scale.shape)
        torch.Size([2]) torch.Size([2])
        >>> assert (scale > 0).all()
        >>> # with modules that return more than one tensor
        >>> module = nn.LSTM(3, 4)
        >>> tensor = torch.randn(4, 2, 3)
        >>> loc, scale, others = normal_params(*module(tensor))
        >>> print(loc.shape, scale.shape)
        torch.Size([4, 2, 2]) torch.Size([4, 2, 2])
        >>> assert (scale > 0).all()

    """

    def __init__(
        self,
        scale_mapping: str = "biased_softplus_1.0",
        scale_lb: Number = 1e-4,
    ) -> None:
        super().__init__()
        self.scale_mapping = mappings(scale_mapping)
        self.scale_lb = scale_lb

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tensor, *others = tensors
        loc, scale = tensor.chunk(2, -1)
        scale = self.scale_mapping(scale).clamp_min(self.scale_lb)
        return (loc, scale, *others)


class AddStateIndependentNormalScale(torch.nn.Module):
    """A module that returns its input as a location, with a learned scale that does not depend on the input.

    The module holds one tensor, ``state_independent_scale``, of shape
    ``scale_shape``. ``module(loc, *others)`` returns
    ``(loc, scale, *others)``, where ``scale`` is that tensor expanded to the
    shape of ``loc``, mapped by ``scale_mapping`` and clamped to at least
    ``scale_lb``. Unlike :class:`NormalParamExtractor`, the input is not split:
    all of it is the location.

    Args:
        scale_shape (torch.Size, int or tuple of int, optional): the shape of
            the scale tensor. It must match the trailing dimensions of
            ``loc``, or the call raises a ``RuntimeError``. Defaults to
            ``torch.Size(())``: one scale for all the elements.

    Keyword Args:
        scale_mapping (str, optional): the function that maps the scale tensor
            to positive values, by name, for example ``"softplus"``,
            ``"exp"`` or ``"biased_softplus_1.0"``. See
            :func:`~tensordict.nn.mappings`. Defaults to ``"exp"``.
        scale_lb (Number, optional): the minimum value of the scale (the
            standard deviation). Defaults to ``1e-4``.
        device (torch.device, optional): the device of the scale tensor.
            Defaults to ``None``, the default device.
        make_param (bool, optional): if ``True``, the scale tensor is a
            trainable :class:`~torch.nn.Parameter`. If ``False``, it is a
            buffer. Defaults to ``True``.
        init_value (float, optional): the initial value of the scale tensor,
            before ``scale_mapping``. With the default ``"exp"`` mapping,
            ``0.0`` gives a scale of ``1``. Defaults to ``0.0``.

    Examples:
        >>> from torch import nn
        >>> import torch
        >>> from tensordict.nn.distributions import AddStateIndependentNormalScale
        >>> num_outputs = 4
        >>> module = nn.Linear(3, num_outputs)
        >>> module_normal = AddStateIndependentNormalScale(num_outputs)
        >>> tensor = torch.randn(3)
        >>> loc, scale = module_normal(module(tensor))
        >>> print(loc.shape, scale.shape)
        torch.Size([4]) torch.Size([4])
        >>> assert (scale > 0).all()
        >>> # with modules that return more than one tensor
        >>> module = nn.LSTM(3, num_outputs)
        >>> module_normal = AddStateIndependentNormalScale(num_outputs)
        >>> tensor = torch.randn(4, 2, 3)
        >>> loc, scale, others = module_normal(*module(tensor))
        >>> print(loc.shape, scale.shape)
        torch.Size([4, 2, 4]) torch.Size([4, 2, 4])
        >>> assert (scale > 0).all()
    """

    def __init__(
        self,
        scale_shape: torch.Size | int | tuple = None,
        *,
        scale_mapping: str = "exp",
        scale_lb: Number = 1e-4,
        device: torch.device | None = None,
        make_param: bool = True,
        init_value: float = 0.0,
    ) -> None:

        super().__init__()
        if scale_shape is None:
            scale_shape = torch.Size(())
        self.scale_lb = scale_lb
        if isinstance(scale_shape, int):
            scale_shape = (scale_shape,)
        self.scale_shape = torch.Size(scale_shape)
        self.scale_mapping = scale_mapping
        if make_param:
            self.state_independent_scale = torch.nn.Parameter(
                torch.full(
                    scale_shape,
                    init_value,
                    dtype=torch.get_default_dtype(),
                    device=device,
                )
            )
        else:
            self.state_independent_scale = torch.nn.Buffer(
                torch.full(
                    scale_shape,
                    init_value,
                    dtype=torch.get_default_dtype(),
                    device=device,
                )
            )

    def forward(
        self, loc: torch.Tensor, *others: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Returns ``loc``, the scale expanded to the shape of ``loc``, and ``others``.

        Args:
            loc (torch.Tensor): the location. Its trailing dimensions must
                match ``scale_shape``.
            *others: other values, returned unchanged.

        Returns:
            The tuple ``(loc, scale, *others)``.

        Raises:
            RuntimeError: if the trailing dimensions of ``loc`` do not match
                ``scale_shape``.
        """
        # Slice from the start: with a 0-dim scale, loc.shape[-0:] would be
        # the whole shape of loc instead of an empty one.
        loc_trailing_shape = loc.shape[max(loc.ndim - len(self.scale_shape), 0) :]
        if self.scale_shape != loc_trailing_shape:
            raise RuntimeError(
                f"Last dimensions of loc ({loc_trailing_shape}) do not match the number of dimensions "
                f"in scale ({self.state_independent_scale.shape})"
            )

        scale = self.state_independent_scale.expand_as(loc)
        scale = mappings(self.scale_mapping)(scale).clamp_min(self.scale_lb)

        return (loc, scale, *others)


class Delta(D.Distribution):
    """A distribution with all its mass on one value, ``param``.

    :meth:`sample` and :meth:`rsample` return ``param`` (expanded to the
    sample shape), and :attr:`mean`, :attr:`mode` and
    :attr:`deterministic_sample` return it as it is. :meth:`log_prob` returns
    ``inf`` for a value equal to ``param`` (within ``atol`` and ``rtol``, on
    every element of the event) and ``-inf`` otherwise. It is the default
    ``distribution_class`` of
    :class:`~tensordict.nn.ProbabilisticTensorDictModule`, which then copies
    its input to its output.

    Args:
        param (torch.Tensor): the value that holds all the mass.
        atol (float, optional): the absolute tolerance with which
            :meth:`log_prob` compares a value with ``param``. Defaults to
            ``1e-6``.
        rtol (float, optional): the relative tolerance with which
            :meth:`log_prob` compares a value with ``param``. Defaults to
            ``1e-6``.
        batch_shape (torch.Size, optional): the batch shape. Defaults to
            ``None``: see ``event_shape``.
        event_shape (torch.Size, optional): the shape of one value. If
            neither ``batch_shape`` nor ``event_shape`` is given (or both
            are empty), the last dimension of ``param`` is the event shape and
            the others are the batch shape. Defaults to ``None``.

    Examples:
        >>> import torch
        >>> from tensordict.nn.distributions import Delta
        >>> dist = Delta(torch.zeros(4, 2))
        >>> dist.batch_shape, dist.event_shape
        (torch.Size([4]), torch.Size([2]))
        >>> dist = Delta(torch.tensor([1.0, 2.0]))
        >>> dist.sample()
        tensor([1., 2.])
        >>> dist.log_prob(torch.tensor([1.0, 2.0])), dist.log_prob(torch.tensor([1.0, 3.0]))
        (tensor(inf), tensor(-inf))

    """

    arg_constraints: dict = {}

    def __init__(
        self,
        param: torch.Tensor,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        batch_shape: torch.Size | Sequence[int] | None = None,
        event_shape: torch.Size | Sequence[int] | None = None,
    ) -> None:
        if batch_shape is None:
            batch_shape = torch.Size([])
        if event_shape is None:
            event_shape = torch.Size([])
        self.update(param)
        self.atol = atol
        self.rtol = rtol
        if not len(batch_shape) and not len(event_shape):
            batch_shape = param.shape[:-1]
            event_shape = param.shape[-1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def update(self, param: torch.Tensor) -> None:
        """Replaces ``param``. The batch and event shapes do not change."""
        self.param = param

    def _is_equal(self, value: torch.Tensor) -> torch.Tensor:
        param = self.param.expand_as(value)
        is_equal = abs(value - param) < self.atol + self.rtol * abs(param)
        for i in range(-1, -len(self.event_shape) - 1, -1):
            is_equal = is_equal.all(i)
        return is_equal

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        is_equal = self._is_equal(value)
        out = torch.zeros_like(is_equal, dtype=value.dtype)
        out.masked_fill_(is_equal, np.inf)
        out.masked_fill_(~is_equal, -np.inf)
        return out

    @torch.no_grad()
    def sample(
        self,
        sample_shape: torch.Size | Sequence[int] | None = None,
    ) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([])
        return self.param.expand((*sample_shape, *self.param.shape))

    def rsample(
        self,
        sample_shape: torch.Size | Sequence[int] | None = None,
    ) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([])
        return self.param.expand((*sample_shape, *self.param.shape))

    @property
    def mode(self) -> torch.Tensor:
        return self.param

    @property
    def mean(self) -> torch.Tensor:
        return self.param

    @property
    def deterministic_sample(self) -> torch.Tensor:
        return self.param


@property
def _logistic_deterministic_sample(self):
    s = self.loc
    for t in self.transforms:
        s = t(s)
    return s


D.LogisticNormal.deterministic_sample = _logistic_deterministic_sample
