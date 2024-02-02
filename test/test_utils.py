# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import pytest
import torch
from tensordict import TensorDict, unravel_key, unravel_key_list
from tensordict._tensordict import _unravel_key_to_tuple
from tensordict.utils import (
    _getitem_batch_size,
    _make_cache_key,
    convert_ellipsis_to_idx,
    isin,
)


@pytest.mark.parametrize("tensor", [torch.rand(2, 3, 4, 5), torch.rand(2, 3, 4, 5, 6)])
@pytest.mark.parametrize(
    "index1",
    [
        slice(None),
        slice(0, 1),
        0,
        [0],
        [0, 1],
        np.arange(2),
        torch.arange(2),
        [True, True],
        Ellipsis,
    ],
)
@pytest.mark.parametrize(
    "index2",
    [
        slice(None),
        slice(1, 3, 1),
        slice(-3, -1),
        0,
        [0],
        [0, 1],
        np.arange(0, 1),
        torch.arange(2),
        range(2),
        torch.tensor([[0, 1], [0, 1]]),
        # [True, False, True],
        Ellipsis,
    ],
)
@pytest.mark.parametrize(
    "index3",
    [
        slice(None),
        slice(1, 3, 1),
        slice(-3, -1),
        0,
        [0],
        [0, 1],
        np.arange(1, 3),
        torch.arange(2),
        range(2),
        torch.tensor([[0, 1], [0, 1]]),
        # [True, False, True, False],
        Ellipsis,
    ],
)
@pytest.mark.parametrize(
    "index4",
    [
        slice(None),
        slice(0, 4, 2),
        slice(-4, -2),
        0,
        [0],
        [0, 1],
        np.arange(0, 4, 2),
        torch.arange(2),
        range(2),
        torch.tensor([[0, 1], [0, 1]]),
        # [True, False, False, False, True],
        Ellipsis,
    ],
)
def test_getitem_batch_size(tensor, index1, index2, index3, index4):
    # cannot have 2 ellipsis
    if (index1 is Ellipsis) + (index2 is Ellipsis) + (index3 is Ellipsis) + (
        index4 is Ellipsis
    ) > 1:
        pytest.skip("cannot have more than one ellipsis in an index.")
    if (index1 is Ellipsis) + (index2 is Ellipsis) + (index3 is Ellipsis) + (
        index4 is Ellipsis
    ) == 1 and tensor.ndim == 5:
        pytest.skip("index possibly incompatible with tensor shape.")
    index = (index1, index2, index3, index4)
    index = convert_ellipsis_to_idx(index, tensor.shape)
    assert tensor[index].shape == _getitem_batch_size(tensor.shape, index), index


@pytest.mark.parametrize("tensor", [torch.rand(2, 3, 4, 5), torch.rand(2, 3, 4, 5, 6)])
@pytest.mark.parametrize("idx", range(3))
@pytest.mark.parametrize("ndim", range(1, 4))
@pytest.mark.parametrize("slice_leading_dims", [True, False])
def test_getitem_batch_size_mask(tensor, idx, ndim, slice_leading_dims):
    # test n-dimensional boolean masks are handled correctly
    if idx + ndim > 4:
        pytest.skip(
            "Not enough dimensions in test tensor for this combination of parameters"
        )
    mask_shape = (2, 3, 4, 5)[idx : idx + ndim]
    mask = torch.randint(2, mask_shape, dtype=torch.bool)
    if slice_leading_dims:
        index = (slice(None),) * idx + (mask,)
    else:
        index = (0,) * idx + (mask,)
    index = convert_ellipsis_to_idx(index, tensor.shape)
    assert tensor[index].shape == _getitem_batch_size(tensor.shape, index), index


def test_make_cache_key():
    Q = torch.rand(3)
    V = torch.zeros(2)
    args = (1, (2, 3), Q)
    kwargs = {"a": V, "b": "c", "d": ("e", "f")}
    assert _make_cache_key(args, kwargs) == (
        (1, (2, 3), id(Q)),
        (("a", id(V)), ("b", "c"), ("d", ("e", "f"))),
    )


@pytest.mark.parametrize("listtype", (list, tuple))
def test_unravel_key_list(listtype):
    keys_in = listtype(["a0", ("b0",), ("c0", ("d",))])
    keys_out = unravel_key_list(keys_in)
    assert keys_out == ["a0", "b0", ("c0", "d")]


def test_unravel_key():
    keys_in = ["a0", ("b0",), ("c0", ("d",))]
    keys_out = [unravel_key(key_in) for key_in in keys_in]
    assert keys_out == ["a0", "b0", ("c0", "d")]


def test_unravel_key_to_tuple():
    keys_in = ["a", ("b",), ("c", ("d",))]
    keys_out = [_unravel_key_to_tuple(key_in) for key_in in keys_in]
    assert keys_out == [("a",), ("b",), ("c", "d")]
    assert not _unravel_key_to_tuple(("a", (1,), ("b",)))
    assert not _unravel_key_to_tuple(("a", (slice(None),), ("b",)))


@pytest.mark.parametrize("key", ("tensor1", "tensor3"))
@pytest.mark.parametrize("dim", (0, 1))
@pytest.mark.parametrize("invert", (True, False))
def test_isin(key, dim, invert):
    td = TensorDict(
        {
            "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]]),
            "tensor2": torch.tensor([[10, 20], [30, 40], [40, 50], [50, 60]]),
        },
        batch_size=[4],
    )
    td_ref = TensorDict(
        {
            "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [10, 11, 12]]),
            "tensor2": torch.tensor([[10, 20], [30, 40], [50, 60]]),
        },
        batch_size=[3],
    )

    if key == "tensor3":
        with pytest.raises(KeyError, match=f"Key '{key}' not found in tensordict."):
            isin(td, td_ref, key, dim, invert)
    elif dim == 1:
        with pytest.raises(
            ValueError,
            match="The specified dimension '1' is invalid for a TensorDict with batch size .*.",
        ):
            isin(td, td_ref, key, dim, invert)
    else:
        in_ref = isin(td, td_ref, key, dim, invert)

        expected_in_ref = torch.tensor([True, True, True, False])
        if invert:
            expected_in_ref = ~expected_in_ref

        assert torch.equal(in_ref, expected_in_ref)

    with pytest.raises(
        KeyError,
        match="The number of dimensions in the batch size of the tensordict and reference_tensordict must be the same.",
    ):
        td.batch_size = []
        isin(td, td_ref, key, dim, invert)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
