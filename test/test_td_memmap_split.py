# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import torch
from tensordict import TensorDict


def test_td_memmap_helper_import_paths_are_preserved():
    dense_module = importlib.import_module("tensordict._td")
    memmap_module = importlib.import_module("tensordict._td_memmap")

    for name in memmap_module.__all__:
        assert getattr(dense_module, name) is getattr(memmap_module, name)


def test_td_memmap_roundtrip_after_helper_split(tmp_path):
    td = TensorDict(
        {
            "a": torch.arange(3),
            "nested": {"b": torch.ones(3, 2)},
        },
        batch_size=[3],
    )
    td.memmap_(tmp_path)

    loaded = TensorDict.load_memmap(tmp_path)

    assert (loaded["a"] == td["a"]).all()
    assert (loaded["nested", "b"] == td["nested", "b"]).all()
