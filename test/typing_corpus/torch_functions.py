# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Stack and concatenate with torch.stack and torch.cat, as the docs show."""

from __future__ import annotations

import torch
from tensordict import TensorClass, TensorDict, TypedTensorDict
from torch import Tensor


class Observation(TensorClass):
    pixels: torch.Tensor


class State(TypedTensorDict):
    eta: Tensor


td = TensorDict({"a": torch.zeros(4, 3)}, batch_size=[4, 3])
stacked = torch.stack([td, td], dim=0)
catted = torch.cat([td, td], dim=0)
assert stacked.batch_size == torch.Size([2, 4, 3])
assert catted.batch_size == torch.Size([8, 3])

obs = Observation(pixels=torch.zeros(4, 3), batch_size=[4])
obs_stacked = torch.stack([obs, obs])
obs_catted = torch.cat([obs, obs])
assert obs_stacked.batch_size == torch.Size([2, 4])
assert obs_catted.pixels.shape == torch.Size([8, 3])

state = State(eta=torch.zeros(5, 3), batch_size=[5])
state_stacked = torch.stack([state, state])
assert state_stacked.batch_size == torch.Size([2, 5])
