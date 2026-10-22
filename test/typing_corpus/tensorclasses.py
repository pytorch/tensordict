# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Define and use tensorclasses."""

from __future__ import annotations

import torch
from tensordict import TensorClass, tensorclass, TensorDict


class Observation(TensorClass):
    pixels: torch.Tensor
    reward: torch.Tensor
    label: str = "train"


@tensorclass
class Step:
    obs: torch.Tensor
    action: torch.Tensor


obs = Observation(pixels=torch.zeros(4, 3), reward=torch.zeros(4), batch_size=[4])
obs = Observation(torch.zeros(4, 3), torch.zeros(4), "eval", batch_size=[4])
first = obs[0]
pixels: torch.Tensor = obs.pixels
obs.reward = torch.ones(4)
stacked = Observation.stack([obs, obs])
both = Observation.cat([obs, obs])
moved = obs.to("cpu")
as_td = obs.to_tensordict()
back = Observation.from_tensordict(as_td)
for item in obs.unbind(0):
    assert item.label == "eval"
assert first.batch_size == torch.Size([])
assert stacked.batch_size == torch.Size([2, 4])
assert both.batch_size == torch.Size([8])
assert isinstance(as_td, TensorDict)
assert back.pixels.shape == pixels.shape == moved.pixels.shape

# Type checkers know the fields of a @tensorclass class, but not its
# tensordict methods or its batch_size argument (docs/source/tensorclass.rst).
step = Step(obs=torch.zeros(4, 3), action=torch.zeros(4, 2))
step.action = step.action + 1
assert step.obs.shape == torch.Size([4, 3])
