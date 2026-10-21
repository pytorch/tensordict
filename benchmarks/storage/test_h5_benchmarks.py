# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark reads from HDF5-backed PersistentTensorDict.

Skipped automatically when ``h5py`` is missing.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch
from tensordict import lazy_stack, PersistentTensorDict, TensorDict

_has_h5py = importlib.util.find_spec("h5py", None) is not None

pytestmark = pytest.mark.skipif(not _has_h5py, reason="h5py not found.")


def _make_h5td(tmp_path_factory, n_keys, n_rows, feat):
    td = TensorDict(
        {
            **{f"key_{i}": torch.randn(n_rows, feat) for i in range(n_keys)},
            "nested": {
                f"key_{i}": torch.randn(n_rows, feat) for i in range(n_keys // 10)
            },
        },
        batch_size=[n_rows],
    )
    filename = tmp_path_factory.mktemp("h5") / "data.h5"
    return PersistentTensorDict.from_dict(td, filename=filename)


@pytest.fixture(scope="module")
def many_keys(tmp_path_factory):
    return _make_h5td(tmp_path_factory, n_keys=300, n_rows=10, feat=4)


@pytest.fixture(scope="module")
def large_arrays(tmp_path_factory):
    return _make_h5td(tmp_path_factory, n_keys=10, n_rows=100_000, feat=64)


@pytest.mark.parametrize(
    "idx",
    [slice(0, 5), torch.tensor([1, 3, 5]), torch.tensor([5, 1, 3])],
    ids=["slice", "sorted_fancy", "unsorted_fancy"],
)
def test_h5_index_many_keys(benchmark, many_keys, idx):
    benchmark(lambda: many_keys[idx].to_tensordict())


@pytest.mark.parametrize(
    "idx",
    [
        slice(100, 356),
        torch.arange(0, 100_000, 400),
        # many selected rows: h5py point selection is quadratic in their number
        torch.arange(0, 100_000, 2),
        torch.arange(100_000) % 2 == 0,
    ],
    ids=["slice", "sorted_fancy", "dense_fancy", "dense_mask"],
)
def test_h5_index_large_arrays(benchmark, large_arrays, idx):
    benchmark(lambda: large_arrays[idx].to_tensordict())


def test_h5_to_tensordict_many_keys(benchmark, many_keys):
    benchmark(many_keys.to_tensordict)


@pytest.fixture(scope="module")
def lazy_stack_h5(tmp_path_factory):
    # two files of about 2.8 MB each: big enough that reading the datasets shows
    return lazy_stack(
        [
            _make_h5td(tmp_path_factory, n_keys=10, n_rows=1000, feat=64)
            for _ in range(2)
        ]
    )


@pytest.mark.parametrize("operation", ["keys", "entry_class"])
def test_h5_lazy_stack_key_metadata(benchmark, lazy_stack_h5, operation):
    if operation == "keys":
        benchmark(lambda: list(lazy_stack_h5.keys(True, True)))
    else:
        benchmark(lambda: lazy_stack_h5.entry_class("key_0"))


@pytest.fixture(scope="module")
def string_column(tmp_path_factory):
    import h5py

    # a string column that another tool wrote, one string per row
    n_rows = 100_000
    filename = tmp_path_factory.mktemp("h5") / "strings.h5"
    with h5py.File(filename, "w") as f:
        f["pos"] = np.zeros((n_rows, 3), dtype=np.float32)
        f.create_dataset(
            "mission",
            data=[f"mission {i}" for i in range(n_rows)],
            dtype=h5py.string_dtype(),
        )
    return PersistentTensorDict.from_h5(filename, non_tensor_as_stack=True)


@pytest.mark.parametrize(
    "idx",
    [3, slice(100, 356), torch.arange(0, 100_000, 400)],
    ids=["int", "slice", "sorted_fancy"],
)
def test_h5_index_string_column(benchmark, string_column, idx):
    benchmark(lambda: string_column[idx]["mission"])
