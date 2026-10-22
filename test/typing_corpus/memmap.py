# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Write a tensordict to disk with memmap and load it back."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from tensordict import TensorDict, TensorDictBase

td = TensorDict({"a": torch.zeros(4, 3), ("b", "c"): torch.ones(4)}, batch_size=[4])

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "td"
    on_disk = td.memmap(path)
    loaded = TensorDict.load_memmap(path)
    loaded_any: TensorDictBase = TensorDictBase.load_memmap(path)
    copy = td.clone().memmap_(Path(tmpdir) / "td2", num_threads=2)
    like = td.memmap_like(str(Path(tmpdir) / "td3"))
    td.save(Path(tmpdir) / "td4")
    reloaded = TensorDict.load(Path(tmpdir) / "td4")
    assert on_disk.is_memmap() and copy.is_memmap()
    assert (loaded == on_disk).all()
    assert loaded_any.batch_size == like.batch_size == reloaded.batch_size
