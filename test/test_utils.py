# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import pytest
import torch

from tensordict.utils import _getitem_batch_size, _make_cache_key


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
        [True, False, True],
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
        [True, False, True, False],
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
        [True, False, False, False, True],
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
    assert tensor[index].shape == _getitem_batch_size(tensor.shape, index)


def test_make_cache_key():
    Q = torch.rand(3)
    V = torch.zeros(2)
    args = (1, (2, 3), Q)
    kwargs = {"a": V, "b": "c", "d": ("e", "f")}
    print(_make_cache_key(args, kwargs))
    assert _make_cache_key(args, kwargs) == (
        (
            1,
            (
                2,
                3,
            ),
            id(Q),
        ),
        (("a", id(V)), ("b", "c"), ("d", ("e", "f"))),
    )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
