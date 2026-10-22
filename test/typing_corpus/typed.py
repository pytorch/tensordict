# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Declare and use a TypedTensorDict."""

from __future__ import annotations

import torch
from tensordict import TensorDict, TypedTensorDict
from torch import Tensor


class State(TypedTensorDict):
    eta: Tensor
    X: Tensor


class RichState(State):
    reward: Tensor


state = State(eta=torch.zeros(5, 3), X=torch.zeros(5, 4), batch_size=[5])
eta: Tensor = state.eta
state.X = torch.ones(5, 4)
row = state[0]
rich = RichState(
    eta=torch.zeros(5, 3), X=torch.zeros(5, 4), reward=torch.zeros(5), batch_size=[5]
)
wrapped = State.from_tensordict(
    TensorDict(eta=torch.zeros(5, 3), X=torch.zeros(5, 4), batch_size=[5])
)
stacked = State.stack([state, state])
for key, value in state.items():
    assert value.shape[0] == 5, key
assert eta.shape == torch.Size([5, 3])
assert row.batch_size == torch.Size([])
assert rich.reward.shape == torch.Size([5])
assert wrapped.eta.shape == state["eta"].shape
assert stacked.batch_size == torch.Size([2, 5])
