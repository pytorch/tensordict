# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from https://github.com/toshas/torch_truncnorm

from __future__ import annotations

import math
from numbers import Number
from typing import Sequence

import torch
from tensordict._deprecation import deprecated_attributes
from torch.distributions import constraints, Distribution
from torch.distributions.utils import broadcast_all

__all__ = ["TruncatedNormal", "TruncatedStandardNormal"]

_CONST_SQRT_2 = math.sqrt(2)
_CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
_CONST_INV_SQRT_2 = 1 / math.sqrt(2)
_CONST_LOG_INV_SQRT_2PI = math.log(_CONST_INV_SQRT_2PI)
_CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """A standard normal distribution restricted to the interval ``[a, b]``.

    The distribution before truncation has mean 0 and standard deviation 1.
    Its samples are reparameterized (``rsample``). For other means and
    standard deviations, use :class:`~tensordict.nn.distributions.TruncatedNormal`.

    Source: https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    Args:
        a (Number or torch.Tensor): the lower bound of the support.
        b (Number or torch.Tensor): the upper bound of the support. ``a`` and
            ``b`` are broadcast together, and give the batch shape.
        validate_args (bool, optional): passed to
            :class:`~torch.distributions.Distribution`. Defaults to ``None``.

    Raises:
        ValueError: if ``a`` and ``b`` have different dtypes, or if ``a`` is
            not smaller than ``b`` everywhere.

    Examples:
        >>> import torch
        >>> from tensordict.nn.distributions.truncated_normal import TruncatedStandardNormal
        >>> dist = TruncatedStandardNormal(a=torch.zeros(2), b=torch.tensor([1.0, 2.0]))
        >>> dist.mean
        tensor([0.4599, 0.7228])
        >>> dist.auc
        tensor([0.3413, 0.4772])
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True
    eps = 1e-6

    def __init__(
        self,
        a: Number | torch.Tensor,
        b: Number | torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super().__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any((self.a >= self.b).view(-1).tolist()):
            raise ValueError("Incorrect truncation range")
        # eps = torch.finfo(self.a.dtype).eps * 10
        eps = self.eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp(eps, 1 - eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b
            - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = (
            1
            - self._lpbb_m_lpaa_d_Z
            - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        )
        self._entropy = (
            _CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z
        )

    @constraints.dependent_property
    def support(self) -> constraints.Constraints:
        return constraints.interval(self.a, self.b)

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        return self._variance

    def entropy(self) -> torch.Tensor:
        return self._entropy

    @property
    def auc(self) -> torch.Tensor:
        """The mass of the standard normal distribution between ``a`` and ``b``, before truncation."""
        return self._Z

    @staticmethod
    def _little_phi(x: torch.Tensor) -> torch.Tensor:
        return (-(x**2) * 0.5).exp() * _CONST_INV_SQRT_2PI

    def _big_phi(self, x: torch.Tensor) -> torch.Tensor:
        phi = 0.5 * (1 + (x * _CONST_INV_SQRT_2).erf())
        return phi.clamp(self.eps, 1 - self.eps)

    @staticmethod
    def _inv_big_phi(x: torch.Tensor) -> torch.Tensor:
        return _CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        y = self._big_phi_a + value * self._Z
        y = y.clamp(self.eps, 1 - self.eps)
        return self._inv_big_phi(y)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        return _CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5

    def rsample(
        self,
        sample_shape: torch.Size | Sequence[int] | None = None,
    ) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([])
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(
            self._dtype_min_gt_0, self._dtype_max_lt_1
        )
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """A normal distribution of mean ``loc`` and standard deviation ``scale``, restricted to ``[a, b]``.

    Its samples are reparameterized (``rsample``).

    Source: https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    Args:
        loc (Number or torch.Tensor): the mean of the normal distribution
            before truncation.
        scale (Number or torch.Tensor): the standard deviation of the normal
            distribution before truncation. Values under ``1e-6`` are set to
            ``1e-6``.
        a (Number or torch.Tensor): the lower bound of the support, in the
            same units as ``loc``.
        b (Number or torch.Tensor): the upper bound of the support, in the
            same units as ``loc``.
        validate_args (bool, optional): passed to
            :class:`~torch.distributions.Distribution`. Defaults to ``None``.

    Raises:
        ValueError: if ``a`` is not smaller than ``b`` everywhere.

    Examples:
        >>> import torch
        >>> from tensordict.nn.distributions import TruncatedNormal
        >>> _ = torch.manual_seed(0)
        >>> dist = TruncatedNormal(loc=torch.zeros(3), scale=torch.ones(3), a=-1.0, b=1.0)
        >>> dist.sample()
        tensor([-0.0064,  0.4764, -0.7754])
        >>> dist.mean
        tensor([0., 0., 0.])
        >>> TruncatedNormal(loc=5.0, scale=1.0, a=4.0, b=6.0).mean
        tensor(5.)
    """

    has_rsample = True

    def __init__(
        self,
        loc: Number | torch.Tensor,
        scale: Number | torch.Tensor,
        a: Number | torch.Tensor,
        b: Number | torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        self.scale = self.scale.clamp_min(self.eps)
        self._non_std_a = a
        self._non_std_b = b
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super().__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale

    @constraints.dependent_property
    def support(self) -> constraints.Constraint:
        return constraints.interval(self._non_std_a, self._non_std_b)

    def _to_std_rv(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale + self.loc

    def _validated_std_rv(self, value: torch.Tensor) -> torch.Tensor:
        std_value = self._to_std_rv(value)
        if self._validate_args:
            # The support is [a, b], so check the value itself, not the
            # standardised value that TruncatedStandardNormal checks. A Python
            # number is checked in the dtype that it is computed in.
            self._validate_sample(
                torch.as_tensor(value, dtype=std_value.dtype, device=std_value.device)
            )
        return std_value

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        value = self._validated_std_rv(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        sample = self._from_std_rv(super().icdf(value))

        # clamp data but keep gradients
        sample_clip = torch.stack(
            [sample.detach(), self._non_std_a.detach().expand_as(sample)], 0
        ).max(0)[0]
        sample_clip = torch.stack(
            [sample_clip, self._non_std_b.detach().expand_as(sample)], 0
        ).min(0)[0]
        sample.data.copy_(sample_clip)
        return sample

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = self._validated_std_rv(value)
        return (
            _CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5 - self._log_scale
        )


__getattr__ = deprecated_attributes(
    __name__,
    {
        "CONST_SQRT_2": (_CONST_SQRT_2, None),
        "CONST_INV_SQRT_2PI": (_CONST_INV_SQRT_2PI, None),
        "CONST_INV_SQRT_2": (_CONST_INV_SQRT_2, None),
        "CONST_LOG_INV_SQRT_2PI": (_CONST_LOG_INV_SQRT_2PI, None),
        "CONST_LOG_SQRT_2PI_E": (_CONST_LOG_SQRT_2PI_E, None),
    },
    removal="0.17",
)
