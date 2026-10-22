# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Build TensorDictModule and TensorDictSequential pipelines."""

from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import (
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
)
from torch import nn

encoder = TensorDictModule(nn.Linear(3, 4), in_keys=["obs"], out_keys=["hidden"])
head = TensorDictModule(nn.Linear(4, 2), in_keys=["hidden"], out_keys=["action"])
policy = TensorDictSequential(encoder, head)
policy_from_list = TensorDictSequential([encoder, head])
nested = TensorDictModule(
    nn.Linear(3, 4), in_keys=[("agent", "obs")], out_keys=[("agent", "hidden")]
)


def add_one(td: TensorDictBase) -> TensorDictBase:
    td["count"] = td["action"] + 1
    return td


pipeline = TensorDictSequential(encoder, head, add_one)
partial = TensorDictSequential(encoder, head, selected_out_keys=["action"])


class Scale(TensorDictModuleBase):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor
        self.in_keys = ["action"]
        self.out_keys = ["scaled"]

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict["scaled"] = tensordict["action"] * self.factor
        return tensordict


scaled_policy = TensorDictSequential(encoder, head, Scale(2.0))
encoder.out_keys = ["features"]
encoder.out_keys = ["hidden"]

td = TensorDict({"obs": torch.zeros(5, 3)}, batch_size=[5])
out = policy(td)
out = pipeline(td.clone())
out = scaled_policy(td.clone())
selected = policy.select_out_keys("action")
assert out["scaled"].shape == torch.Size([5, 2])
assert partial(td.clone()).keys() is not None
assert policy_from_list.in_keys == policy.in_keys
assert nested.in_keys == [("agent", "obs")]
assert selected.out_keys == ["action"]
