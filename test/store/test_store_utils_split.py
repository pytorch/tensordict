# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import torch


def test_store_helper_import_paths_are_preserved():
    store_module = importlib.import_module("tensordict.store._store")
    helper_module = importlib.import_module("tensordict.store._utils")

    for name in helper_module.__all__:
        assert getattr(store_module, name) is getattr(helper_module, name)

    assert store_module._tensor_to_bytes.__module__ == "tensordict.store._store"


def test_store_byte_range_helpers():
    helper_module = importlib.import_module("tensordict.store._utils")

    assert helper_module._compute_byte_ranges([5, 2], torch.float32, 1) == [(8, 8)]
    assert helper_module._compute_byte_ranges([5, 2], torch.float32, slice(1, 4)) == [
        (8, 24)
    ]
    assert helper_module._compute_covering_range(
        [5, 2], torch.float32, slice(1, 5, 2)
    ) == (8, 24)
    assert helper_module._get_local_idx(slice(1, 5, 2), 5) == slice(None, None, 2)
    assert helper_module._is_scattered_index(torch.tensor([1, 3]))
    assert helper_module._getitem_result_shape([5, 2], torch.tensor([1, 3])) == [2, 2]


def test_store_tensor_byte_roundtrip():
    helper_module = importlib.import_module("tensordict.store._utils")
    tensor = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    data = helper_module._tensor_to_bytes(tensor)
    restored = helper_module._bytes_to_tensor(data, [3, 2], torch.float32)

    assert torch.equal(restored, tensor)
