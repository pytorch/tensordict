# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Build TensorDictModule and TensorDictSequential pipelines."""

from __future__ import annotations

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.nn import (
    as_tensordict_module,
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
    WrapModule,
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

in_key_list: list[str] = ["obs"]
nested_out_keys: list[tuple[str, str]] = [("agent", "hidden")]
kwarg_map: dict[str, str] = {"x": "obs"}
from_typed_lists = TensorDictModule(
    nn.Linear(3, 4), in_keys=in_key_list, out_keys=nested_out_keys
)
from_kwarg_map = TensorDictModule(
    lambda x: x + 1, in_keys=kwarg_map, out_keys="shifted"
)
wrapped = WrapModule(lambda td: td, in_keys=in_key_list, out_keys=nested_out_keys)


@as_tensordict_module(in_keys=in_key_list, out_keys=nested_out_keys)
def copy_obs(obs: torch.Tensor) -> torch.Tensor:
    return obs


class ClassAttrModule(TensorDictModuleBase):
    in_keys = ["obs", ("agent", "obs")]
    out_keys = [("agent", "out")]

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict["agent", "out"] = tensordict["obs"]
        return tensordict


class_attr_mod = ClassAttrModule()
encoder.in_keys = in_key_list
encoder.out_keys = nested_out_keys
encoder.out_keys = ["hidden"]
# The getters return a Sequence: build a new list to extend it.
encoder.in_keys = [*encoder.in_keys, "extra"]
encoder.in_keys = list(encoder.in_keys)[:-1]
encoder_out_keys: list[NestedKey] = list(encoder.out_keys)
nested_kwarg_map: dict[tuple[str, str], str] = {("agent", "obs"): "x"}
from_nested_kwarg_map = TensorDictModule(
    lambda x: x + 1,
    in_keys=nested_kwarg_map,
    out_keys="shifted",
    out_to_in_map=False,
)

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
assert from_typed_lists(td.clone())["agent", "hidden"].shape == torch.Size([5, 4])
assert from_kwarg_map(td.clone())["shifted"].shape == torch.Size([5, 3])
assert wrapped(td.clone())["obs"].shape == torch.Size([5, 3])
assert copy_obs(td.clone())["agent", "hidden"].shape == torch.Size([5, 3])
assert class_attr_mod(td.clone())["agent", "out"].shape == torch.Size([5, 3])
nested_td = TensorDict({("agent", "obs"): torch.zeros(5, 3)}, batch_size=[5])
assert from_nested_kwarg_map(nested_td)["shifted"].shape == torch.Size([5, 3])
