# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Index, apply, stack, cat, unbind and split."""

from __future__ import annotations

import torch
from tensordict import LazyStackedTensorDict, TensorDict, TensorDictBase

td = TensorDict({"a": torch.zeros(4, 3), "b": {"c": torch.ones(4, 3, 2)}}, [4, 3])

row = td[0]
col = td[:, 1]
masked = td[torch.tensor([True, False, True, False])]
picked = td[torch.tensor([0, 2])]
a = td["a"]
c = td["b", "c"]
c = td.get(("b", "c"))
sub = td.get("b")
default = td.get("missing", None)
assert row.batch_size == torch.Size([3])
assert col.batch_size == torch.Size([4])
assert masked.shape == picked.shape
assert a.shape == (4, 3) and c.shape == (4, 3, 2)

doubled = td.apply(lambda x: x * 2)
summed = td.apply(lambda x, y: x + y, doubled)
floats = td.apply(lambda x: x.float(), batch_size=[4])
td.apply_(lambda x: x.zero_())
normed = td.named_apply(lambda name, x: x if name == "a" else x * 0)
assert doubled is not None and summed is not None and floats is not None
assert normed is not None

stacked = TensorDict.stack([td, td], 0)
catted = TensorDict.cat([td, td], dim=1)
lazy = LazyStackedTensorDict.lazy_stack([td, td], dim=0)
assert stacked.batch_size == torch.Size([2, 4, 3])
assert catted.batch_size == torch.Size([4, 6])
assert lazy.batch_size == stacked.batch_size

parts = td.unbind(0)
first: TensorDictBase = parts[0]
chunks = td.split(2, dim=0)
uneven = td.split([1, 3], dim=0)
for chunk in td.chunk(2, dim=0):
    assert chunk.batch_size == torch.Size([2, 3])
assert len(parts) == 4 and first.batch_size == torch.Size([3])
assert len(chunks) == 2 and len(uneven) == 2

reshaped = td.reshape(-1)
viewed = td.view(12)
flat = td.flatten(0, 1)
moved = td.to("cpu")
cloned = td.clone()
selected = td.select("a")
excluded = td.exclude("a")
assert reshaped.shape == viewed.shape == flat.shape
assert moved.device == torch.device("cpu") and cloned.batch_size == td.batch_size
assert list(selected.keys()) == ["a"] and list(excluded.keys()) == ["b"]
