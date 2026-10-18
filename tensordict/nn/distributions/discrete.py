# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Sequence

import torch
from torch import distributions as D

# We need this to build the distribution maps
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
    """Draws a one-hot sample along the last dimension of ``values``.

    Args:
        values (torch.Tensor): the logits, or the probabilities if
            ``do_softmax=False``, along the last dimension.
        do_softmax (bool, optional): if ``True``, applies a softmax to
            ``values`` first. Defaults to ``True``.

    Returns:
        A ``torch.long`` tensor with the shape of ``values``, with one ``1``
        per row of the last dimension.

    Examples:
        >>> import torch
        >>> from tensordict.nn import rand_one_hot
        >>> _ = torch.manual_seed(0)
        >>> rand_one_hot(torch.tensor([0.0, 2.0, 1.0]))
        tensor([0, 1, 0])
        >>> rand_one_hot(torch.tensor([[0.1, 0.9], [0.8, 0.2]]), do_softmax=False)
        tensor([[0, 1],
                [1, 0]])
    """
    if do_softmax:
        values = values.softmax(-1)
    out = values.cumsum(-1) > torch.rand_like(values[..., :1])
    out = (out.cumsum(-1) == 1).to(torch.long)
    return out


class OneHotCategorical(D.Categorical):
    """A categorical distribution whose values are one-hot tensors.

    It behaves as :class:`torch.distributions.Categorical`, except that it
    reads and produces one-hot encodings. :meth:`sample`, :attr:`mode` and
    :attr:`deterministic_sample` return ``torch.long`` one-hot tensors, and
    :meth:`log_prob` takes one. When several logits tie for the maximum,
    :attr:`mode` marks the first, as :class:`torch.distributions.OneHotCategorical`
    does.

    :meth:`rsample` returns a straight-through sample: its values are one-hot,
    and its gradient is the gradient of a
    :class:`~torch.distributions.RelaxedOneHotCategorical` sample with
    temperature 1.

    Args:
        logits (torch.Tensor, optional): the unnormalized log-probabilities,
            with the categories along the last dimension. A last dimension of
            size 1 is removed first.
        probs (torch.Tensor, optional): the probabilities, as for ``logits``.
            Pass ``logits`` or ``probs``, not both.
        **kwargs: passed to :class:`~torch.distributions.Categorical`, for
            example ``validate_args``.

    Examples:
        >>> import torch
        >>> from tensordict.nn.distributions import OneHotCategorical
        >>> _ = torch.manual_seed(0)
        >>> dist = OneHotCategorical(logits=torch.tensor([0.0, 2.0, 1.0]))
        >>> dist.mode
        tensor([0, 1, 0])
        >>> dist.sample((2,))
        tensor([[0, 0, 1],
                [0, 1, 0]])
        >>> dist.log_prob(torch.tensor([0, 1, 0]))
        tensor(-0.4076)

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
        """Returns the log-probability of the category of ``value``, a one-hot tensor.

        The category is the index of the maximum along the last dimension.
        """
        return super().log_prob(value.argmax(dim=-1))

    @property
    def mode(self) -> torch.Tensor:
        """The one-hot encoding of the most likely category, the first one if several tie."""
        # One-hot of the first maximum: comparing with the max marks every tie.
        index = self.logits.max(-1).indices
        return torch.nn.functional.one_hot(index, self.logits.shape[-1])

    deterministic_sample = mode

    def sample(
        self,
        sample_shape: torch.Size | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Draws one-hot ``torch.long`` samples of shape ``sample_shape + batch_shape + (num_categories,)``."""
        if sample_shape is None:
            sample_shape = torch.Size([])
        out = super().sample(sample_shape=sample_shape)
        out = torch.nn.functional.one_hot(out, self.logits.shape[-1]).to(torch.long)
        return out

    def rsample(
        self,
        sample_shape: torch.Size | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Draws straight-through one-hot samples, which carry the gradient of a relaxed sample.

        See the class description.
        """
        if sample_shape is None:
            sample_shape = torch.Size([])
        # A Categorical always has both probs and logits; pass only one.
        d = D.relaxed_categorical.RelaxedOneHotCategorical(1.0, logits=self.logits)
        out = d.rsample(sample_shape)
        out.data.copy_((out == out.max(-1)[0].unsqueeze(-1)).to(out.dtype))
        return out


D.RelaxedBernoulli.deterministic_sample = D.Bernoulli.mode


@property
def _relaxed_onehot_mode(self) -> torch.Tensor:
    probs = self.base_dist.probs
    mode = probs.argmax(dim=-1)
    return torch.nn.functional.one_hot(mode, num_classes=probs.shape[-1]).to(probs)


D.RelaxedOneHotCategorical.deterministic_sample = _relaxed_onehot_mode
