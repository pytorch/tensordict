# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Update and set entries, with plain and nested keys."""

from __future__ import annotations

import torch
from tensordict import TensorDict

td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])

td.update({"b": torch.ones(3)})
td.update({("next", "obs"): torch.ones(3)})
td.update({"c": torch.ones(3), ("next", "reward"): torch.ones(3)})
new_values: dict[str, torch.Tensor] = {"a": torch.full((3,), 2.0)}
td.update(new_values)
nested_values: dict[tuple[str, str], torch.Tensor] = {("next", "done"): torch.ones(3)}
td.update(nested_values)
td.update(TensorDict({"d": torch.ones(3)}, batch_size=[3]))
td.update_({"a": torch.ones(3)})
td.update({"a": torch.zeros(3)}, inplace=True)

td["e"] = torch.zeros(3)
td["next", "e"] = torch.zeros(3)
td.set(("next", "f"), torch.zeros(3))
td.set_(("next", "f"), torch.ones(3))
key = ("next", "obs")
td[key] = torch.zeros(3)
del td["e"]
popped = td.pop("c")
td = td.rename_key_("b", "b2")
assert ("next", "done") in td.keys(include_nested=True)
assert popped.shape == torch.Size([3])
