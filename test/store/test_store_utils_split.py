# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
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


@pytest.mark.parametrize("size", [0, 3])
@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("helper", ["_compute_byte_ranges", "_compute_covering_range"])
def test_store_scalar_index_bounds(size, side, helper):
    helper_module = importlib.import_module("tensordict.store._utils")
    idx = -size - 1 if side == "lower" else size
    with pytest.raises(IndexError):
        getattr(helper_module, helper)([size, 2], torch.float32, idx)


@pytest.mark.parametrize("value_shape", [(), (1,)])
def test_store_masked_scalar_gradient(value_shape):
    helper_module = importlib.import_module("tensordict.store._utils")
    value = torch.full(value_shape, 7.0, requires_grad=True)
    mask = torch.tensor([True, False, True])
    expected = torch.zeros(3, 2, dtype=torch.float64)
    expected[mask] = value
    actual = helper_module._prepare_indexed_value(value, [3, 2], torch.float64, mask)
    torch.testing.assert_close(actual, expected[mask])
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), value)[0],
        torch.autograd.grad(expected.sum(), value)[0],
    )


def _assign_like_torch(shape, dtype, idx, value):
    """Return what ``tensor[idx] = value`` writes, or the error it raises."""
    target = torch.zeros(shape, dtype=dtype)
    try:
        target[idx] = value
    except (IndexError, RuntimeError) as error:
        return type(error)
    return target[idx]


_MASK = torch.zeros(10, dtype=torch.bool)
_MASK[[0, 2]] = True


@pytest.mark.parametrize(
    "idx",
    [
        0,
        torch.tensor(0),
        (0,),
        (0, slice(None)),
        slice(0, 2),
        slice(0, 4, 2),
        range(0, 4, 2),
        range(4, 0, -2),
        [0, 2],
        torch.tensor([0, 2]),
        torch.tensor([[0, 2], [1, 3]]),
        _MASK,
        _MASK.tolist(),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
@pytest.mark.parametrize(
    "value_shape", [(), (1,), (3,), (1, 3), (2, 3), (2, 2, 3), (4,)]
)
def test_store_prepare_indexed_value_matches_torch(idx, dtype, value_shape):
    helper_module = importlib.import_module("tensordict.store._utils")
    value = (
        torch.arange(torch.Size(value_shape).numel(), dtype=dtype)
        .reshape(value_shape)
        .add(7)
    )
    expected = _assign_like_torch([10, 3], torch.float32, idx, value)
    if isinstance(expected, type):
        with pytest.raises(expected):
            helper_module._prepare_indexed_value(value, [10, 3], torch.float32, idx)
    else:
        actual = helper_module._prepare_indexed_value(
            value, [10, 3], torch.float32, idx
        )
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("value_shape", [(), (1,)])
@pytest.mark.parametrize(
    "dtype,number,boundary",
    [
        (torch.uint8, 300, 255),
        (torch.int8, 128, 127),
        (torch.int8, -129, -128),
        (torch.float16, 100000.0, 65504.0),
    ],
)
def test_store_masked_scalar_overflow(value_shape, dtype, number, boundary):
    helper_module = importlib.import_module("tensordict.store._utils")
    mask = torch.tensor([True, False, True])
    for fill, error in [(number, RuntimeError), (boundary, None)]:
        value = torch.full(value_shape, fill)
        expected = _assign_like_torch([3, 2], dtype, mask, value)
        assert expected is error if error else not isinstance(expected, type)
        if error:
            with pytest.raises(error, match="overflow"):
                helper_module._prepare_indexed_value(value, [3, 2], dtype, mask)
        else:
            actual = helper_module._prepare_indexed_value(value, [3, 2], dtype, mask)
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "idx",
    [
        10,
        -11,
        [0, 10],
        [-11, 0],
        range(9, 11),
        range(-11, -9),
        torch.tensor([0, 10]),
        torch.tensor([-11, 0]),
        torch.tensor([True] + [False] * 8),
        torch.tensor([True] + [False] * 10),
        [True] + [False] * 8,
        [True] + [False] * 10,
    ],
)
def test_store_byte_ranges_reject_out_of_bounds(idx):
    helper_module = importlib.import_module("tensordict.store._utils")
    with pytest.raises(IndexError):
        torch.zeros(10, 3)[idx]
    with pytest.raises(IndexError):
        helper_module._compute_byte_ranges([10, 3], torch.float32, idx)


@pytest.mark.parametrize(
    "idx",
    [
        -1,
        [0, -1, -10],
        range(-2, 2),
        torch.tensor([0, -1, -10]),
        torch.tensor([[0, -1], [-10, 2]]),
    ],
)
def test_store_byte_ranges_normalize_negative_indices(idx):
    helper_module = importlib.import_module("tensordict.store._utils")
    rows = torch.arange(10)[idx].reshape(-1).tolist()
    assert helper_module._compute_byte_ranges([10, 3], torch.float32, idx) == [
        (row * 12, 12) for row in rows
    ]


@pytest.mark.parametrize("idx", [_MASK.tolist(), (_MASK.tolist(),)])
def test_store_bool_list_is_a_mask(idx):
    helper_module = importlib.import_module("tensordict.store._utils")
    rows = torch.arange(10)[idx].tolist()
    assert helper_module._compute_byte_ranges([10, 3], torch.float32, idx) == [
        (row * 12, 12) for row in rows
    ]
    assert helper_module._getitem_result_shape([10, 3], idx) == [len(rows), 3]
