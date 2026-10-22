# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Loop over keys(), values() and items()."""

from __future__ import annotations

import torch
from tensordict import NestedKey, TensorDict

td = TensorDict({"a": torch.zeros(3), "b": {"c": torch.ones(3)}}, batch_size=[3])

top_level: list[NestedKey] = []
for top_key in td.keys():
    top_level.append(top_key)
assert top_level == ["a", "b"]
for nested_key in td.keys(include_nested=True, leaves_only=True):
    assert nested_key in td.keys(True)

names: list[NestedKey] = list(td.keys(True, True))
assert ("b", "c") in names

total = 0
for key, value in td.items():
    assert value.shape[0] == 3, key
    if not isinstance(value, torch.Tensor):
        assert value.batch_size == torch.Size([3])
    total += value.numel()
assert total == 6

for value in td.values():
    assert value.to("cpu").shape == value.shape

sums: dict[NestedKey, float] = {}
for leaf_key, leaf in td.items(include_nested=True, leaves_only=True):
    if isinstance(leaf, torch.Tensor):
        sums[leaf_key] = leaf.sum().item()
assert sums[("b", "c")] == 3

leaves = [value.shape for value in td.values(True, True)]
assert len(leaves) == 2
sizes = {key: value.shape for key, value in td.items()}
assert sizes["a"] == torch.Size([3])
