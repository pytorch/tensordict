# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Build tensordicts from dicts, nested dicts and nested-key dicts."""

from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase

data: dict[str, torch.Tensor] = {"obs": torch.zeros(4, 3), "reward": torch.zeros(4)}
td = TensorDict(data, batch_size=[4])
td = TensorDict({"obs": torch.zeros(4, 3), "next": {"obs": torch.zeros(4, 3)}}, [4])
td = TensorDict({("next", "obs"): torch.zeros(4, 3)}, batch_size=[4])
td = TensorDict({"obs": torch.zeros(4, 3), ("next", "done"): torch.zeros(4, 1)}, [4])
td = TensorDict(obs=torch.zeros(4, 3), batch_size=[4], device="cpu")
from_dict = TensorDict.from_dict(
    {"obs": torch.zeros(4, 3), "next": {"obs": torch.zeros(4)}}
)
empty = TensorDict(batch_size=[4])
empty["obs"] = torch.zeros(4, 3)
empty["next", "obs"] = torch.zeros(4, 3)


def make(batch: int) -> TensorDictBase:
    return TensorDict({"x": torch.zeros(batch)}, batch_size=[batch])


assert make(2).batch_size == torch.Size([2])
assert from_dict["next", "obs"].shape == torch.Size([4])
assert td.batch_size == torch.Size([4])
assert td.to_dict()["obs"].shape == (4, 3)
