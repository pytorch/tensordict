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

# try:
#     from functorch.dim import Dim, DimensionBindError, Tensor, dims
# except ImportError:
#     pytest.skip(reason="functorch.dim not found")
# for linting purposes
dims = None

def _is_in(d, args):
    return any(d is item for item in args)


def _contains_all(tensordict_dims, tensor_dims):
    return all(_is_in(d, tensor_dims) for d in tensordict_dims)


@pytest.mark.skip(reason="functorch.dim disabled")
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

        assert _contains_all(td_dim.dims, td_dim["a"].dims)
        assert td_dim.batch_size == td_dim["a"].shape[: td_dim.batch_dims]
        assert _contains_all(td_dim.dims, td_dim["b"].dims)
        assert td_dim.batch_size == td_dim["b"].shape[: td_dim.batch_dims]

        assert _contains_all(td_dim2.dims, td_dim2["a"].dims)
        assert td_dim2.batch_size == td_dim2["a"].shape[: td_dim2.batch_dims]
        assert _contains_all(td_dim2.dims, td_dim2["b"].dims)
        assert td_dim2.batch_size == td_dim2["a"].shape[: td_dim2.batch_dims]

        assert _contains_all(td_dim3.dims, td_dim3["a"].dims)
        assert td_dim3.batch_size == td_dim3["a"].shape[: td_dim3.batch_dims]
        assert _contains_all(td_dim3.dims, td_dim3["b"].dims)
        assert td_dim3.batch_size == td_dim3["a"].shape[: td_dim3.batch_dims]

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

        d = dims(1)
        td_dim = td[d]

        # FIXME: equality comparison of functorch.dim.Tensor with torch.Tensor causes
        # an exception, so compare to underlying tensor for now.
        torch.testing.assert_close(
            td_dim.reshape(6)["a"]._tensor, td["a"].reshape(4, 6, -1)
        )

    def test_checks_at_set(self, td_name, device):
        td = getattr(self, td_name)(device)

        d1, d2, d3 = dims(3)
        td_dim = td[d1]

        with pytest.raises(
            ValueError,
            match="First-class dimensions of tensordict and value are not compatible.",
        ):
            td_dim["a"] = torch.rand(4, 3, 2, 1, 5, device=device)[:, d2]

        with pytest.raises(RuntimeError, match="batch dimension mismatch"):
            td_dim["b"] = torch.rand(4, 3, 2, 1, 5, device=device)[d1, d3]

        t = torch.rand(4, 3, 2, 1, 5, device=device)
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

        for r in range(5):
            for args in permutations((0, d1, d2, d3), r=r):
                t_ordered = t_dim.order(*args)
                a_ordered = a_dim.order(*args)
                td_ordered = td_dim.order(*args)

                assert td_ordered.shape == t_ordered.shape[: td_ordered.batch_dims]
                if sum(isinstance(arg, Dim) for arg in args) == 3:
                    assert not isinstance(td_ordered, _TensorDictWithDims)
                    assert not isinstance(td_ordered["a"], Tensor)
                    torch.testing.assert_close(td_ordered["a"], a_ordered)
                else:
                    torch.testing.assert_close(
                        td_ordered["a"]._tensor, a_ordered._tensor
                    )
                    assert td_ordered.dims == t_ordered.dims

    @pytest.mark.parametrize("dim", range(4))
    def test_stack(self, td_name, device, dim):
        td = getattr(self, td_name)(device)

        d = dims(1)
        td_dim = td[d]

        stacked = torch.stack([td_dim, td_dim, td_dim], dim=dim)

        shape = list(td_dim.batch_size)
        shape.insert(dim, 3)

        assert stacked.batch_size == torch.Size(shape)
        torch.testing.assert_close(
            stacked["a"].order(d), torch.stack([td["a"]] * 3, dim=dim + 1)
        )

    @pytest.mark.parametrize("dim", range(3))
    def test_cat(self, td_name, device, dim):
        td = getattr(self, td_name)(device)

        d = dims(1)
        td_dim = td[d]

        catted = torch.cat([td_dim, td_dim, td_dim], dim=dim)

        shape = list(td_dim.batch_size)
        shape[dim] *= 3

        assert catted.batch_size == torch.Size(shape)
        torch.testing.assert_close(
            catted["a"].order(d), torch.cat([td["a"]] * 3, dim=dim + 1)
        )

    def test_splitting(self, td_name, device):
        td = getattr(self, td_name)(device)

        d1, d2, d3 = dims(3)
        d1.size = 2
        td_dim = td[(d1, d2), d3]

        assert td_dim.batch_size == torch.Size([2, 1])

        td2 = td_dim.order(d1, d2, d3)

        assert td2.batch_size == torch.Size([2, 2, 3, 2, 1])
        torch.testing.assert_close(td2["a"], td_dim["a"].order(d1, d2, d3))
        torch.testing.assert_close(td2["a"], td["a"].reshape(2, 2, 3, 2, 1, -1))

        d1, d2, d3 = dims(3)
        with pytest.raises(
            DimensionBindError,
            match=r"cannot infer the sizes of 2 dimensions at once \(d1, d2\)",
        ):
            td[(d1, d2), d3]

        d1, d2, d3 = dims(3)
        d1.size = 3
        with pytest.raises(
            DimensionBindError,
            match="inferred dimension does not evenly fit into larger dimension",
        ):
            td[(d1, d2), d3]

        d1, d2, d3 = dims(3)
        d1.size = d2.size = 3
        with pytest.raises(
            DimensionBindError, match=r"Dimension sizes do not match \(4 != 9\)"
        ):
            td[(d1, d2), d3]

    def test_flatten(self, td_name, device):
        td = getattr(self, td_name)(device)

        d1, d2, d3 = dims(3)
        td_dim = td[d1, d2, d3]

        td2 = td_dim.order(d1, (d2, d3))

        assert td2.batch_size == torch.Size([4, 6, 1])

        torch.testing.assert_close(td2["a"], td_dim["a"].order(d1, (d2, d3)))
        torch.testing.assert_close(td2["a"], td["a"].reshape(4, 6, 1, -1))

        # test nested in in order
        td3 = td_dim.order((d1, d2, d3, 0))

        assert td3.batch_size == torch.Size([24])

        torch.testing.assert_close(td3["a"], td_dim["a"].order((d1, d2, d3, 0)))
        torch.testing.assert_close(td3["a"], td["a"].reshape(24, -1))


@pytest.mark.skip(reason="functorch.dim disabled")
def test_dim_reuse():
    t = torch.rand(2, 3, 2)
    td = TensorDict({"a": torch.rand(2, 3, 2, 4)}, [2, 3, 2])
    mt = MetaTensor(2, 3, 2)

    d = dims(1)

    t_dim = t[d, :, d]
    td_dim = td[d, :, d]
    mt_dim = mt[d, :, d]

    # repeated dim should only appear once
    assert td_dim.dims == t_dim.dims
    assert mt_dim.dims == t_dim.dims

    assert td_dim.batch_size == t_dim.shape
    assert td_dim["a"].shape == torch.Size([3, 4])
    assert mt_dim.shape == t_dim.shape

    torch.testing.assert_close(
        td_dim["a"].order(0, 1, d), torch.diagonal(td["a"], dim1=0, dim2=2)
    )


@pytest.mark.skip(reason="functorch.dim disabled")
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
    assert _contains_all(t_dim.dims, mt_dim.dims)
    assert _contains_all(mt_dim.dims, t_dim.dims)

    assert t_dim2.shape == mt_dim2.shape
    assert _contains_all(t_dim2.dims, mt_dim2.dims)
    assert _contains_all(mt_dim2.dims, t_dim2.dims)

    assert t_dim3.shape == mt_dim3.shape
    assert _contains_all(t_dim3.dims, mt_dim3.dims)
    assert _contains_all(mt_dim3.dims, t_dim3.dims)


@pytest.mark.skip(reason="functorch.dim disabled")
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
                try:
                    mt_ordered.dims == t_ordered.dims
                except RuntimeError:
                    # FIXME: If the dims are not equal, we get a confusing
                    # RuntimeError rather than simply False. Catch this error and raise
                    # a saner AssertionError to help debugging
                    raise AssertionError(
                        f"Dims do not match: {mt_ordered.dims} != {t_ordered.dims}"
                    )


@pytest.mark.skip(reason="functorch.dim disabled")
def test_metatensor_splitting():
    mt = MetaTensor(4, 3, 2, 1)

    d1, d2, d3 = dims(3)
    d1.size = 2
    mt_dim = mt[(d1, d2), d3]

    assert mt_dim.shape == torch.Size([2, 1])

    mt2 = mt_dim.order(d1, d2, d3)

    assert mt2.shape == torch.Size([2, 2, 3, 2, 1])

    d1, d2, d3 = dims(3)
    with pytest.raises(
        DimensionBindError,
        match=r"cannot infer the sizes of 2 dimensions at once \(d1, d2\)",
    ):
        mt[(d1, d2), d3]

    d1, d2, d3 = dims(3)
    d1.size = 3
    with pytest.raises(
        DimensionBindError,
        match="inferred dimension does not evenly fit into larger dimension",
    ):
        mt[(d1, d2), d3]

    d1, d2, d3 = dims(3)
    d1.size = d2.size = 3
    with pytest.raises(
        DimensionBindError, match=r"Dimension sizes do not match \(4 != 9\)"
    ):
        mt[(d1, d2), d3]


@pytest.mark.skip(reason="functorch.dim disabled")
def test_metatensor_flatten():
    mt = MetaTensor(4, 3, 2, 1)

    d1, d2 = dims(2)
    mt_dim = mt[d1, d2]

    mt2 = mt_dim.order((d1, d2))

    assert mt2.shape == torch.Size([12, 2, 1])

    mt3 = mt_dim.order((d1, d2, 0, 1))
    assert mt3.shape == torch.Size([24])


@pytest.mark.skip(reason="functorch.dim disabled")
@pytest.mark.parametrize("dim", range(4))
def test_metatensor_stack(dim):
    mt = MetaTensor(2, 3, 4, 5)
    d = dims(1)
    mt_dim = mt[d]

    stacked = torch.stack([mt_dim] * 3, dim=dim)

    shape = list(mt_dim.shape)
    shape.insert(dim, 3)

    assert stacked.shape == torch.Size(shape)
    assert stacked.dims == (d,)
    assert isinstance(stacked, _MetaTensorWithDims)
