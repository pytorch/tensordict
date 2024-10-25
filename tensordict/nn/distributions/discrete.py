# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Sequence

import torch
from torch import distributions as D
from torch.nn import functional as F

__all__ = [
    "OneHotCategorical",
]


def _treat_categorical_params(
    params: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if params is None:
        return None
    if params.shape[-1] == 1:
        params = params[..., 0]
    return params


def rand_one_hot(values: torch.Tensor, do_softmax: bool = True) -> torch.Tensor:
    if do_softmax:
        values = values.softmax(-1)
    out = values.cumsum(-1) > torch.rand_like(values[..., :1])
    out = (out.cumsum(-1) == 1).to(torch.long)
    return out


class OneHotCategorical(D.Categorical):
    """One-hot categorical distribution.

    This class behaves excacly as torch.distributions.Categorical except that it reads and produces one-hot encodings
    of the discrete tensors.

    """

    num_params: int = 1

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        logits = _treat_categorical_params(logits)
        probs = _treat_categorical_params(probs)
        super().__init__(probs=probs, logits=logits, **kwargs)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value.argmax(dim=-1))

    @property
    def mode(self) -> torch.Tensor:
        if hasattr(self, "logits"):
            return (self.logits == self.logits.max(-1, True)[0]).to(torch.long)
        else:
            return (self.probs == self.probs.max(-1, True)[0]).to(torch.long)

    determnistic_sample = mode

    def sample(
        self,
        sample_shape: torch.Size | Sequence[int] | None = None,
    ) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([])
        out = super().sample(sample_shape=sample_shape)
        out = torch.nn.functional.one_hot(out, self.logits.shape[-1]).to(torch.long)
        return out

    def rsample(
        self,
        sample_shape: torch.Size | Sequence[int] | None = None,
    ) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([])
        d = D.relaxed_categorical.RelaxedOneHotCategorical(
            1.0, probs=self.probs, logits=self.logits
        )
        out = d.rsample(sample_shape)
        out.data.copy_((out == out.max(-1)[0].unsqueeze(-1)).to(out.dtype))
        return out


class Ordinal(D.Categorical):
    """
    A discrete distribution for learning to sample from finite ordered sets.
    It is defined in contrast with the `Categorical` distribution, which does
    not impose any notion of proximity or ordering over its support's atoms.
    The `Ordinal` distribution explicitly encodes those concepts, which is
    useful for learning discrete sampling from continuous sets. See §5 of
    [Tang & Agrawal, 2020](https://arxiv.org/pdf/1901.10500.pdf) for details.

    Notes:
        This class is mostly useful when you want to learn a distribution over
        a finite set which is obtained by discretising a continuous set.
    """
    def __init__(self, scores: torch.Tensor):
        """
        Args:
            scores: a tensor of shape [..., N] where N is the size of the set which supports the distributions.
            Typically, the output of a neural network parametrising the distribution.
        """
        logits = _generate_ordinal_logits(scores)
        super().__init__(logits=logits)


class OneHotOrdinal(OneHotCategorical):
    """The one-hot version of the :class:`~.Ordinal` distribution."""
    def __init__(self, scores: torch.Tensor):
        """
        Args:
            scores: a tensor of shape [..., N] where N is the size of the set which supports the distributions.
            Typically, the output of a neural network parametrising the distribution.
        """
        logits = _generate_ordinal_logits(scores)
        super().__init__(logits=logits)


def _generate_ordinal_logits(scores: torch.Tensor) -> torch.Tensor:
    """Implements Eq. 4 of [Tang & Agrawal, 2020](https://arxiv.org/pdf/1901.10500.pdf)."""
    # Assigns Bernoulli-like probabilities for each class in the set
    log_probs = F.logsigmoid(scores)
    complementary_log_probs = F.logsigmoid(-scores)

    # Total log-probability for being "larger than k"
    larger_than_log_probs = log_probs.cumsum(dim=-1)

    # Total log-probability for being "smaller than k"
    smaller_than_log_probs = (
        complementary_log_probs.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1]) - complementary_log_probs
    )

    return larger_than_log_probs + smaller_than_log_probs
