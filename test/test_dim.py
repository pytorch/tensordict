# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import permutations

import pytest
import torch
from _utils_internal import TestTensorDictsBase, get_available_devices
from tensordict import MetaTensor, TensorDict
from tensordict.metatensor import _MetaTensorWithDims
from tensordict.tensordict import _TensorDictWithDims

try:
    from functorch.dim import Tensor, dims
except ImportError:
    pytest.skip(reason="functorch.dim not found")


def _get_all_dims(tensor: Tensor):
    tensor, levels, ndim = tensor._tensor, tensor._levels, tensor.ndim
    return tuple(lvl + ndim if isinstance(lvl, int) else lvl for lvl in levels)


@pytest.mark.parametrize(
    "td_name",
    [
        "td",
        "stacked_td",
        "sub_td",
        "sub_td2",
        "idx_td",
        "unsqueezed_td",
        "squeezed_td",
        "td_reset_bs",
        "nested_td",
        "permute_td",
    ],
)
@pytest.mark.parametrize("device", get_available_devices())
class TestTensorDicts(TestTensorDictsBase):
    def test_batch_size(self, td_name, device):
        td = getattr(self, td_name)(device)
        d = dims(1)
        td_dim = td[:, d]

        assert td_dim.batch_size == torch.Size([4, 2, 1])
        assert td_dim.dim() == 3

    def test_items(self, td_name, device):
        td = getattr(self, td_name)(device)
        d1, d2 = dims(2)
        td_dim = td[d1, :, d2]

        assert td_dim.batch_size == torch.Size([3, 1])

        assert td_dim["a"].shape == torch.Size([3, 1, 5])
        assert td_dim["a"].ndim == 3
        assert td_dim["a"].dims == (d1, d2)

    def test_levels(self, td_name, device):
        # test repeated slicing results in same levels behaviour as functorch.dim.Tensor
        td = getattr(self, td_name)(device)

        d1, d2, d3 = dims(3)

        td_dim = td[:, d1]
        td_dim2 = td_dim[:, d2]
        td_dim3 = td_dim2[d3]

        assert (
            td_dim.all_dims
            == _get_all_dims(td_dim["a"])[:4]
            == _get_all_dims(td_dim["b"])[:4]
        )
        assert (
            td_dim2.all_dims
            == _get_all_dims(td_dim2["a"])[:4]
            == _get_all_dims(td_dim2["b"])[:4]
        )
        assert (
            td_dim3.all_dims
            == _get_all_dims(td_dim3["a"])[:4]
            == _get_all_dims(td_dim3["b"])[:4]
        )

    def test_view(self, td_name, device):
        if td_name == "permute_td":
            pytest.skip(
                reason="Test is not compatible with permuted tensor's size and stride."
            )
        td = getattr(self, td_name)(device)

        d = dims(1)
        td_dim = td[d]

        # FIXME: equality comparison of functorch.dim.Tensor with torch.Tensor causes
        # an exception, so compare to underlying tensor for now.
        torch.testing.assert_close(td_dim.view(6)["a"]._tensor, td["a"].view(4, 6, 5))

    def test_reshape(self, td_name, device):
        td = getattr(self, td_name)(device)
        # td = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])

        d = dims(1)
        td_dim = td[d]

        # FIXME: equality comparison of functorch.dim.Tensor with torch.Tensor causes
        # an exception, so compare to underlying tensor for now.
        torch.testing.assert_close(
            td_dim.reshape(6)["a"]._tensor, td["a"].reshape(4, 6, -1)
        )

    def test_checks_at_set(self, td_name, device):
        td = getattr(self, td_name)(device)
        # td = TensorDict({}, batch_size=[2, 3])

        d1, d2, d3, d4 = dims(4)
        td_dim = td[d1]

        with pytest.raises(ValueError, match="First-class and positional dimensions"):
            td_dim["a"] = torch.rand(4, 3, 2, 1, 5)[:, d2]

        with pytest.raises(ValueError, match="First-class and positional dimensions"):
            td_dim["b"] = torch.rand(4, 3, 2, 1, 5)[d1, d3]

        t = torch.rand(4, 3, 2, 1, 5)
        td_dim["c"] = t[d1]
        assert td_dim["c"].dims == (d1,)
        torch.testing.assert_close(td_dim["c"]._tensor, t)

    def test_tensordict_order(self, td_name, device):
        t = torch.rand(4, 3, 2, 1)
        td = getattr(self, td_name)(device)
        a = td["a"]

        d1, d2, d3 = dims(3)

        t_dim = t[d1, :, d2, d3]
        a_dim = a[d1, :, d2, d3]
        td_dim = td[d1, :, d2, d3]

        for r in range(4):
            for args in permutations((d1, d2, d3), r=r):
                t_ordered = t_dim.order(*args)
                a_ordered = a_dim.order(*args)
                td_ordered = td_dim.order(*args)

                assert td_ordered.shape == t_ordered.shape[: td_ordered.batch_dims]
                if r == 3:
                    assert not isinstance(td_ordered, _TensorDictWithDims)
                    assert not isinstance(td_ordered["a"], Tensor)
                    torch.testing.assert_close(td_ordered["a"], a_ordered)
                else:
                    torch.testing.assert_close(
                        td_ordered["a"]._tensor, a_ordered._tensor
                    )
                    td_ordered._source_batch_size == t_ordered._tensor.shape


def test_error_on_reuse():
    td = TensorDict({}, [2, 2])
    mt = MetaTensor(2, 2)
    d = dims(1)

    with pytest.raises(
        ValueError, match="Indexing a TensorDict or MetaTensor more than once"
    ):
        td[d, d]

    with pytest.raises(
        ValueError, match="Indexing a TensorDict or MetaTensor more than once"
    ):
        mt[d, d]


def test_metatensor_indexing():
    t = torch.empty(2, 3, 4, 5)
    mt = MetaTensor(2, 3, 4, 5)

    d1, d2, d3 = dims(3)

    t_dim = t[d1]
    mt_dim = mt[d1]

    t_dim2 = t_dim[:, d2]
    mt_dim2 = mt_dim[:, d2]

    t_dim3 = t_dim2[d3]
    mt_dim3 = mt_dim2[d3]

    assert t_dim.shape == mt_dim.shape
    assert t_dim._levels == mt_dim._levels

    assert t_dim2.shape == mt_dim2.shape
    assert t_dim2._levels == mt_dim2._levels

    assert t_dim3.shape == mt_dim3.shape
    assert t_dim3._levels == mt_dim3._levels


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
                assert mt_ordered._tensor.shape == t_ordered._tensor.shape
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
