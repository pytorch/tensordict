# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import permutations

import pytest
import torch
from tensordict import MetaTensor, TensorDict
from tensordict.metatensor import _MetaTensorWithDims

try:
    from functorch.dim import Tensor, dims
except ImportError:
    pytest.skip(reason="functorch.dim not found")


def _get_all_dims(tensor: Tensor):
    tensor, levels, ndim = tensor._tensor, tensor._levels, tensor.ndim
    return tuple(l + ndim if isinstance(l, int) else l for l in levels)


def test_batch_size():
    td = TensorDict({}, [3, 4, 5])
    d = dims(1)
    td_dim = td[:, d]

    assert td_dim.batch_size == torch.Size([3, 5])
    assert td_dim.dim() == 2


def test_items():
    td = TensorDict({"a": torch.rand(3, 4, 5, 6)}, [3, 4, 5])
    d1, d2 = dims(2)
    td_dim = td[d1, :, d2]

    assert td_dim.batch_size == torch.Size([4])

    assert td_dim["a"].shape == torch.Size([4, 6])
    assert td_dim["a"].ndim == 2
    assert td_dim["a"].dims == (d1, d2)


def test_levels():
    # test repeated slicing results in same levels behaviour as functorch.dim.Tensor
    td = TensorDict(
        {"a": torch.rand(3, 4, 5, 6), "b": torch.rand(3, 4, 5, 6, 7)}, [3, 4, 5]
    )

    d1, d2, d3 = dims(3)

    td_dim = td[:, d1]
    td_dim2 = td_dim[:, d2]
    td_dim3 = td_dim2[d3]

    assert (
        td_dim.all_dims
        == _get_all_dims(td_dim["a"])[:3]
        == _get_all_dims(td_dim["b"])[:3]
    )
    assert (
        td_dim2.all_dims
        == _get_all_dims(td_dim2["a"])[:3]
        == _get_all_dims(td_dim2["b"])[:3]
    )
    assert (
        td_dim3.all_dims
        == _get_all_dims(td_dim3["a"])[:3]
        == _get_all_dims(td_dim3["b"])[:3]
    )


def test_view():
    td = TensorDict({"a": torch.rand(3, 4, 5, 6)}, [3, 4, 5])

    d = dims(1)
    td_dim = td[d]

    # FIXME: equality comparison of functorch.dim.Tensor with torch.Tensor causes
    # an exception, so compare to underlying tensor for now.
    assert (td_dim.view(20)["a"]._tensor == td["a"].view(3, 20, 6)).all()


def test_reshape():
    td = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])

    d = dims(1)
    td_dim = td[d]

    # FIXME: equality comparison of functorch.dim.Tensor with torch.Tensor causes
    # an exception, so compare to underlying tensor for now.
    assert (td_dim.reshape(3, 1)["a"]._tensor == td["a"].reshape(2, 3, 1, -1)).all()


def test_metatensor_indexing():
    t = torch.empty(2, 3, 4, 5)
    mt = MetaTensor(2, 3, 4, 5)

    d1, d2, d3 = dims(3)

    t_dim = t[d1]
    mt_dim = mt[d1]

    assert t_dim.shape == mt_dim.shape
    assert t_dim._levels == mt_dim._levels


def test_metatensor_order():
    t = torch.empty(2, 3, 4, 5, 6, 7)
    mt = MetaTensor(2, 3, 4, 5, 6, 7)

    d1, d2, d3 = dims(3)

    t_dim = t[d1, :, d2, ..., d3]
    mt_dim = mt[d1, :, d2, ..., d3]

    for r in range(4):
        for args in permutations((d1, d2, d3), r=r):
            mt_ordered = mt_dim.order(*args)
            t_ordered = t_dim.order(*args)

            assert mt_ordered.shape == t_ordered.shape
            if r == 3:
                assert not isinstance(mt_ordered, _MetaTensorWithDims)
            else:
                assert mt_ordered.underlying_shape == t_ordered._tensor.shape
                try:
                    mt_ordered._levels == t_ordered._levels
                except RuntimeError:
                    # FIXME: If the levels are not equal, we get a confusing
                    # RuntimeError rather than simply False. Catch this error and raise
                    # a saner AssertionError to help debugging
                    raise AssertionError(
                        "Levels do not match: "
                        f"{mt_ordered._levels} != {t_ordered._levels}"
                    )


def test_checks_at_set():
    td = TensorDict({}, batch_size=[2, 3])

    d1, d2 = dims(2)
    td_dim = td[d1]

    with pytest.raises(ValueError, match="First-class and positional dimensions"):
        td_dim["a"] = torch.empty(2, 3, 4)[:, d2]

    with pytest.raises(ValueError, match="First-class and positional dimensions"):
        td_dim["b"] = torch.empty(2, 3, 4)[d1, d2]

    t = torch.empty(2, 3, 4)
    td_dim["c"] = t[d1]
    assert td_dim["c"].dims == (d1,)
    assert (td_dim["c"]._tensor == t).all()
