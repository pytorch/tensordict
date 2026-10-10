# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import pickle
import platform
import sys
from pathlib import Path

import pytest
import torch
from tensordict import (
    lazy_stack,
    LazyStackedTensorDict,
    NonTensorStack,
    TensorClass,
    TensorDict,
    UnbatchedTensor,
)
from tensordict._indexing import _getitem_batch_size
from tensordict._reductions import _reduce_td
from tensordict._torch_func import _stack as stack_td
from tensordict.base import _NESTED_TENSORS_AS_LISTS, TensorDictBase
from tensordict.utils import (
    assert_allclose_td,
    logger as tdlogger,
    set_lazy_legacy,
    set_list_to_stack,
)

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.tensordict.test._utils_internal import (
        decompose,
        get_available_devices,
        is_npu_available,
    )
else:
    IS_FB = False
    from _utils_internal import decompose, get_available_devices, is_npu_available


_has_streaming = importlib.util.find_spec("streaming", None) is not None

try:
    import h5py  # noqa

    _has_h5py = True
except ImportError:
    _has_h5py = False

_has_onnx = importlib.util.find_spec("onnxruntime", None) is not None

PYTORCH_TEST_FBCODE = os.getenv("PYTORCH_TEST_FBCODE")

_IS_OSX = platform.system() == "Darwin"
_IS_WINDOWS = sys.platform == "win32"

TD_BATCH_SIZE = 4

# Capture all warnings
pytestmark = [
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings(
        "ignore:There is a performance drop because we have not yet implemented the batching rule"
    ),
    pytest.mark.filterwarnings(
        "ignore:A destination should be provided when cloning a PersistentTensorDict"
    ),
    pytest.mark.filterwarnings(
        "ignore:Replacing an array with another one is inefficient"
    ),
    pytest.mark.filterwarnings(
        "ignore:Indexing an h5py.Dataset object with a boolean mask that needs broadcasting does not work directly"
    ),
    pytest.mark.filterwarnings(
        "ignore:The PyTorch API of nested tensors is in prototype"
    ),
    pytest.mark.filterwarnings(
        "ignore:Lazy modules are a new feature under heavy development so changes to the API or functionality"
    ),
    pytest.mark.filterwarnings(
        "ignore:The content of the stacked NonTensorData objects matched in value but not identity"
    ),
    pytest.mark.filterwarnings(
        "ignore:No PYTORCH_KERNEL_CACHE_PATH or HOME environment variable set"
    ),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]

mp_ctx = "spawn"
cur_device = "cpu"
npu_device_count = 0
if torch.cuda.is_available():
    cur_device = "cuda"
elif is_npu_available():
    cur_device = "npu"
    npu_device_count = torch.npu.device_count()


class TestLazyStackedTensorDict:
    @property
    def _idx_list(self):
        return {
            0: 1,
            1: slice(None),
            2: slice(1, 2),
            3: self._tensor_index,
            4: range(1, 2),
            5: None,
            6: [0, 1],
            7: self._tensor_index.numpy(),
        }

    @property
    def _tensor_index(self):
        torch.manual_seed(0)
        return torch.randint(2, (5, 2))

    def dense_stack_tds_v1(self, td_list, stack_dim: int) -> TensorDictBase:
        shape = list(td_list[0].shape)
        shape.insert(stack_dim, len(td_list))

        out = td_list[0].unsqueeze(stack_dim).expand(shape).clone()
        for i in range(1, len(td_list)):
            index = (slice(None),) * stack_dim + (i,)  # this is index_select
            out[index] = td_list[i]

        return out

    def dense_stack_tds_v2(self, td_list, stack_dim: int) -> TensorDictBase:
        shape = list(td_list[0].shape)
        shape.insert(stack_dim, len(td_list))
        out = td_list[0].unsqueeze(stack_dim).expand(shape).clone()

        data_ptr_set_before = {val.data_ptr() for val in decompose(out)}
        res = LazyStackedTensorDict.lazy_stack(td_list, dim=stack_dim, out=out)
        data_ptr_set_after = {val.data_ptr() for val in decompose(out)}
        assert data_ptr_set_before == data_ptr_set_after

        return res

    @staticmethod
    def nested_lazy_het_td(batch_size):
        shared = torch.zeros(4, 4, 2)
        hetero_3d = torch.zeros(3)
        hetero_2d = torch.zeros(2)

        individual_0_tensor = torch.zeros(1)
        individual_1_tensor = torch.zeros(1, 2)
        individual_2_tensor = torch.zeros(1, 2, 3)

        td_list = [
            TensorDict(
                {
                    "shared": shared,
                    "hetero": hetero_3d,
                    "individual_0_tensor": individual_0_tensor,
                },
                [],
            ),
            TensorDict(
                {
                    "shared": shared,
                    "hetero": hetero_3d,
                    "individual_1_tensor": individual_1_tensor,
                },
                [],
            ),
            TensorDict(
                {
                    "shared": shared,
                    "hetero": hetero_2d,
                    "individual_2_tensor": individual_2_tensor,
                },
                [],
            ),
        ]
        for i, td in enumerate(td_list):
            td[f"individual_{i}_td"] = td.clone()
            td["shared_td"] = td.clone()

        td_stack = LazyStackedTensorDict.lazy_stack(td_list, dim=0)
        obs = TensorDict(
            {"lazy": td_stack, "dense": torch.zeros(3, 3, 2)},
            [],
        )
        obs = obs.expand(batch_size).clone()
        return obs

    def recursively_check_key(self, td, value: int):
        if isinstance(td, LazyStackedTensorDict):
            for t in td.tensordicts:
                if not self.recursively_check_key(t, value):
                    return False
        elif isinstance(td, TensorDict):
            for i in td.values():
                if not self.recursively_check_key(i, value):
                    return False
        elif isinstance(td, torch.Tensor):
            return (td == value).all()
        else:
            return False

        return True

    def test_add_batch_dim_cache(self):
        td = TensorDict(
            {"a": torch.rand(3, 4, 5), ("b", "c"): torch.rand(3, 4, 5)}, [3, 4, 5]
        )
        td = LazyStackedTensorDict.lazy_stack([td, td.clone()], 0)
        from tensordict.nn import TensorDictModule  # noqa
        from torch import vmap

        tdlogger.info("first call to vmap")
        fun = vmap(lambda x: x)
        fun(td)
        td.zero_()
        # this value should be cached
        tdlogger.info("second call to vmap")
        std = fun(td)
        for value in std.values(True, True):
            assert (value == 0).all()

    def test_add_batch_dim_cache_nested(self):
        td = TensorDict(
            {"a": torch.rand(3, 4, 5), ("b", "c"): torch.rand(3, 4, 5)}, [3, 4, 5]
        )
        td = TensorDict(
            {"parent": LazyStackedTensorDict.lazy_stack([td, td.clone()], 0)},
            [2, 3, 4, 5],
        )
        from tensordict.nn import TensorDictModule  # noqa
        from torch import vmap

        fun = vmap(lambda x: x)
        tdlogger.info("first call to vmap")
        fun(td)
        td.zero_()
        # this value should be cached
        tdlogger.info("second call to vmap")
        std = fun(td)
        for value in std.values(True, True):
            assert (value == 0).all()

    def test_hooks_are_deprecated(self):
        td = LazyStackedTensorDict(
            TensorDict({"a": torch.zeros(3)}, [3]),
            TensorDict({"a": torch.ones(3)}, [3]),
        )
        assert td._hook_out is None
        assert td._hook_in is None

        def hook(x):
            return x + 1

        for name in ("hook_out", "hook_in"):
            match = (
                rf"^LazyStackedTensorDict\.{name} is deprecated and will be removed "
                r"in TensorDict 0\.17\.$"
            )
            with pytest.warns(DeprecationWarning, match=match) as record:
                assert getattr(td, name) is None
            assert record[0].filename == __file__
            with pytest.warns(DeprecationWarning, match=match) as record:
                setattr(td, name, hook)
            assert record[0].filename == __file__
            assert getattr(td, f"_{name}") is hook
        for name in ("hook_out", "hook_in"):
            with pytest.warns(
                DeprecationWarning,
                match=rf"^LazyStackedTensorDict\({name}=\.\.\.\) is deprecated and "
                r"will be removed in TensorDict 0\.17\.$",
            ) as record:
                td2 = LazyStackedTensorDict(*td.tensordicts, **{name: hook})
            assert record[0].filename == __file__
            assert getattr(td2, f"_{name}") is hook
            if name == "hook_out":
                # hook_out still applies to the values that get returns
                assert (td2.get("a") == torch.tensor([[1.0], [2.0]])).all()

    def test_setstate_hooks_of_old_pickles(self):
        # TensorDict 0.14 pickled the hooks under their public names
        td = lazy_stack(
            [
                TensorDict({"a": torch.zeros(3)}, [3]),
                TensorDict({"a": torch.ones(3)}, [3]),
            ]
        )
        state = td.__getstate__()
        state["hook_out"] = state.pop("_hook_out")
        state["hook_in"] = state.pop("_hook_in")
        loaded = LazyStackedTensorDict.__new__(LazyStackedTensorDict)
        loaded.__setstate__(state)
        assert "hook_out" not in vars(loaded)
        assert "hook_in" not in vars(loaded)
        assert loaded._hook_out is None
        assert loaded._hook_in is None
        assert (loaded == td).all()
        loaded = pickle.loads(pickle.dumps(td))
        assert loaded._hook_out is None
        assert (loaded == td).all()

    def test_valid_keys_is_deprecated(self):
        td = lazy_stack(
            [
                TensorDict({"a": torch.zeros(3), "b": {"c": torch.zeros(3)}}, [3]),
                TensorDict({"a": torch.ones(3), "b": {"c": torch.ones(3)}}, [3]),
            ]
        )
        with pytest.warns(
            DeprecationWarning,
            match=r"^LazyStackedTensorDict\.valid_keys\(\) is deprecated and will be "
            r"removed in TensorDict 0\.17\. Use keys\(\) instead\.$",
        ) as record:
            keys = td.valid_keys(True, True)
        assert record[0].filename == __file__
        assert set(keys) == set(td.keys(True, True)) == {"a", ("b", "c")}

    def test_get_nestedtensor_is_deprecated(self):
        td = lazy_stack(
            [
                TensorDict({"a": torch.zeros(4), "n": {"c": torch.zeros(2)}}, []),
                TensorDict({"a": torch.ones(5), "n": {"c": torch.ones(3)}}, []),
            ]
        )
        for key in ("a", ("n", "c")):
            with pytest.warns(
                DeprecationWarning,
                match=r"^LazyStackedTensorDict\.get_nestedtensor\(\) is deprecated "
                r"and will be removed in TensorDict 0\.17\. Use "
                r"get\(key, as_nested_tensor=True\) instead\.$",
            ) as record:
                nested = td.get_nestedtensor(key)
            # one warning, also for nested keys
            record = [w for w in record if w.category is DeprecationWarning]
            assert len(record) == 1
            assert record[0].filename == __file__
            assert nested.layout == torch.strided
            expected = td.get(key, as_nested_tensor=True, layout=torch.strided)
            for t1, t2 in zip(nested.unbind(), expected.unbind()):
                assert (t1 == t2).all()

    @pytest.mark.skipif(not _has_streaming, reason="streaming is not installed")
    def test_to_mds(self, tmpdir):
        td = LazyStackedTensorDict(
            TensorDict(a=0, b=1, c=torch.randn(2), d="a string"),
            TensorDict(a=0, b=1, c=torch.randn(3), d="another string"),
            TensorDict(a=0, b=1, c=torch.randn(3), d="yet another string"),
        )

        tmpdir = str(tmpdir)
        td.to_mds(out=tmpdir)

        # Create a dataloader
        from streaming import StreamingDataset

        # Load the dataset
        dataset = StreamingDataset(local=tmpdir, remote=None, batch_size=2)
        dl = torch.utils.data.DataLoader(  # noqa: TOR401
            dataset=dataset, batch_size=2, collate_fn=LazyStackedTensorDict.from_list
        )
        batches = list(dl)
        batches = [_batch for batch in batches for _batch in batch.unbind(0)]
        test_td = LazyStackedTensorDict(*batches)
        assert_allclose_td(td, test_td)

    def test_batched_indexing_ragged_lazy_stack(self):
        tensordict = lazy_stack([TensorDict(x=torch.zeros(i)) for i in range(5)])
        indices = [3, 0, 4]
        batch = tensordict.__getitems__(indices)
        assert isinstance(batch, LazyStackedTensorDict)
        assert [len(td["x"]) for td in batch] == indices

    def test_all_keys(self):
        td = TensorDict({"a": torch.zeros(1)}, [])
        td2 = TensorDict({"a": torch.zeros(2)}, [])
        stack = LazyStackedTensorDict.lazy_stack([td, td2])
        assert set(stack.keys(True, True)) == {"a"}

    @pytest.mark.parametrize("ragged", [False, True])
    def test_arithmetic_ops(self, ragged):
        td0 = LazyStackedTensorDict(
            *[
                LazyStackedTensorDict(
                    *[
                        TensorDict(
                            {
                                "a": torch.zeros(
                                    (
                                        2,
                                        (_i + 2 if ragged else 4),
                                        3,
                                    )
                                ),
                                ("b", "c"): torch.zeros(2),
                            },
                            [2],
                        )
                        for _i in range(3)
                    ],
                    stack_dim=1,
                )
            ],
            stack_dim=0,
        )
        td1 = td0.clone()
        assert (td1 + 1 == td1.apply(lambda x: x + 1)).all()
        td1 += 1
        assert (td1 == td0.apply(lambda x: x + 1)).all()
        assert ((td0 + td1) == td0.apply(lambda x: x + 1)).all()
        assert (td0 * td1 == 0).all()
        assert ((td1 * 0) == 0).all()
        if ragged:
            # This doesn't work because tensors can't be reduced to a single value
            # as they're not contiguous
            with pytest.raises(
                RuntimeError, match="Failed to stack tensors within a tensordict"
            ):
                td1.norm()
        else:
            td1.norm()

    @set_list_to_stack(True)
    def test_best_intention_stack(self):
        td0 = TensorDict({"a": 1, "b": TensorDict({"c": 2}, [])}, [])
        td1 = TensorDict({"a": 1, "b": TensorDict({"d": 2}, [])}, [])
        with set_lazy_legacy(False):
            td = LazyStackedTensorDict.maybe_dense_stack([td0, td1])
        assert isinstance(td, TensorDict)
        assert isinstance(td.get("b"), LazyStackedTensorDict)
        td1 = TensorDict({"a": 1, "b": TensorDict({"c": [2]}, [])}, [])
        with set_lazy_legacy(False):
            td = LazyStackedTensorDict.maybe_dense_stack([td0, td1])
        assert isinstance(td, TensorDict)
        assert isinstance(td.get("b"), LazyStackedTensorDict)

    @pytest.mark.parametrize("batch_size", [(), (2,), (1, 2)])
    @pytest.mark.parametrize("cat_dim", [0, 1, 2])
    def test_cat_lazy_stack(self, batch_size, cat_dim):
        if cat_dim > len(batch_size):
            return
        td_lazy = self.nested_lazy_het_td(batch_size)["lazy"]
        assert isinstance(td_lazy, LazyStackedTensorDict)
        res = torch.cat([td_lazy], dim=cat_dim)
        assert assert_allclose_td(res, td_lazy)
        assert res is not td_lazy
        td_lazy_clone = td_lazy.clone()
        data_ptr_set_before = {val.data_ptr() for val in decompose(td_lazy)}
        res = torch.cat([td_lazy_clone], dim=cat_dim, out=td_lazy)
        data_ptr_set_after = {val.data_ptr() for val in decompose(td_lazy)}
        assert data_ptr_set_after == data_ptr_set_before
        assert res is td_lazy
        assert assert_allclose_td(res, td_lazy_clone)

        td_lazy_2 = td_lazy.clone()
        td_lazy_2.apply_(lambda x: x + 1)

        res = torch.cat([td_lazy, td_lazy_2], dim=cat_dim)
        assert res.stack_dim == len(batch_size)
        assert res.shape[cat_dim] == td_lazy.shape[cat_dim] + td_lazy_2.shape[cat_dim]
        index = (slice(None),) * cat_dim + (slice(0, td_lazy.shape[cat_dim]),)
        assert assert_allclose_td(res[index], td_lazy)
        index = (slice(None),) * cat_dim + (slice(td_lazy.shape[cat_dim], None),)
        assert assert_allclose_td(res[index], td_lazy_2)

        res = torch.cat([td_lazy, td_lazy_2], dim=cat_dim)
        assert res.stack_dim == len(batch_size)
        assert res.shape[cat_dim] == td_lazy.shape[cat_dim] + td_lazy_2.shape[cat_dim]
        index = (slice(None),) * cat_dim + (slice(0, td_lazy.shape[cat_dim]),)
        assert assert_allclose_td(res[index], td_lazy)
        index = (slice(None),) * cat_dim + (slice(td_lazy.shape[cat_dim], None),)
        assert assert_allclose_td(res[index], td_lazy_2)

        if cat_dim != len(batch_size):  # cat dim is not stack dim
            batch_size = list(batch_size)
            batch_size[cat_dim] *= 2
            td_lazy_dest = self.nested_lazy_het_td(batch_size)["lazy"]
            data_ptr_set_before = {val.data_ptr() for val in decompose(td_lazy_dest)}
            res = torch.cat([td_lazy, td_lazy_2], dim=cat_dim, out=td_lazy_dest)
            data_ptr_set_after = {val.data_ptr() for val in decompose(td_lazy_dest)}
            assert data_ptr_set_after == data_ptr_set_before
            assert res is td_lazy_dest
            index = (slice(None),) * cat_dim + (slice(0, td_lazy.shape[cat_dim]),)
            assert assert_allclose_td(res[index], td_lazy)
            index = (slice(None),) * cat_dim + (slice(td_lazy.shape[cat_dim], None),)
            assert assert_allclose_td(res[index], td_lazy_2)

    def test_cat_mixed_lazy_dense_stack_dim(self):
        td_lazy = LazyStackedTensorDict(
            *[
                TensorDict({"a": torch.full((2, 3), float(i))}, [2, 3])
                for i in range(2)
            ],
            stack_dim=0,
        )
        td_dense = TensorDict(
            {"a": torch.arange(12, dtype=torch.float).reshape(2, 2, 3)},
            [2, 2, 3],
        )

        res = torch.cat([td_lazy, td_dense], dim=0)
        assert isinstance(res, LazyStackedTensorDict)
        assert res.stack_dim == 0
        assert res.batch_size == torch.Size([4, 2, 3])
        assert len(res.tensordicts) == 4
        assert assert_allclose_td(res[:2], td_lazy)
        assert assert_allclose_td(res[2:], td_dense)

        res = torch.cat([td_dense, td_lazy], dim=0)
        assert isinstance(res, LazyStackedTensorDict)
        assert res.stack_dim == 0
        assert res.batch_size == torch.Size([4, 2, 3])
        assert len(res.tensordicts) == 4
        assert assert_allclose_td(res[:2], td_dense)
        assert assert_allclose_td(res[2:], td_lazy)

    @pytest.mark.parametrize("cat_dim", [1, -2])
    def test_cat_mixed_lazy_dense_non_stack_dim(self, cat_dim):
        td_lazy = LazyStackedTensorDict(
            *[
                TensorDict({"a": torch.full((2, 3), float(i))}, [2, 3])
                for i in range(2)
            ],
            stack_dim=0,
        )
        td_dense = TensorDict(
            {"a": torch.arange(18, dtype=torch.float).reshape(2, 3, 3)},
            [2, 3, 3],
        )

        res = torch.cat([td_lazy, td_dense], dim=cat_dim)
        assert isinstance(res, LazyStackedTensorDict)
        assert res.stack_dim == 0
        assert res.batch_size == torch.Size([2, 5, 3])
        dense_lazy = td_dense.to_lazystack(0)
        for i, sub_td in enumerate(res.tensordicts):
            expected = torch.cat([td_lazy.tensordicts[i], dense_lazy.tensordicts[i]], 0)
            assert assert_allclose_td(sub_td, expected)

    def test_cat_mixed_lazy_dense_preserves_non_densifiable_lazy_stack(self):
        td_lazy = LazyStackedTensorDict(
            TensorDict({"a": torch.ones(1), "only0": torch.zeros(1)}, []),
            TensorDict({"a": torch.ones(2), "only1": torch.zeros(1)}, []),
            stack_dim=0,
        )
        td_dense = TensorDict(
            {
                "a": torch.full((1, 3), 3.0),
                "dense_only": torch.ones(1, 4),
            },
            [1],
        )

        res = torch.cat([td_lazy, td_dense], dim=0)
        assert isinstance(res, LazyStackedTensorDict)
        assert res.stack_dim == 0
        assert res.batch_size == torch.Size([3])
        assert len(res.tensordicts) == 3
        assert [item.shape for item in res.get("a", as_list=True)] == [
            torch.Size([1]),
            torch.Size([2]),
            torch.Size([3]),
        ]
        assert "dense_only" in res.tensordicts[2].keys()

    def test_cat_mixed_lazy_zero_length_dense_stack_dim(self):
        td_lazy = LazyStackedTensorDict(
            *[TensorDict({"a": torch.full((2,), float(i))}, [2]) for i in range(2)],
            stack_dim=0,
        )
        td_dense_empty = TensorDict({"a": torch.empty(0, 2)}, [0, 2])

        res = torch.cat([td_lazy, td_dense_empty], dim=0)
        assert isinstance(res, LazyStackedTensorDict)
        assert res.batch_size == torch.Size([2, 2])
        assert len(res.tensordicts) == 2
        assert assert_allclose_td(res, td_lazy)

        res = torch.cat([td_dense_empty, td_lazy], dim=0)
        assert isinstance(res, LazyStackedTensorDict)
        assert res.batch_size == torch.Size([2, 2])
        assert len(res.tensordicts) == 2
        assert assert_allclose_td(res, td_lazy)

        td_lazy_empty = LazyStackedTensorDict(stack_dim=0, batch_size=[2])
        res = torch.cat([td_dense_empty, td_lazy_empty], dim=0)
        assert isinstance(res, LazyStackedTensorDict)
        assert res.batch_size == torch.Size([0, 2])
        assert len(res.tensordicts) == 0

    def test_cat_mixed_lazy_dense_non_stack_dim_mismatched_slices(self):
        td_lazy = LazyStackedTensorDict(
            *[TensorDict({"a": torch.full((2,), float(i))}, [2]) for i in range(2)],
            stack_dim=0,
        )
        td_dense = TensorDict({"a": torch.zeros(3, 2)}, [3, 2])

        with pytest.raises(
            RuntimeError,
            match="same number of stacked tensordicts",
        ):
            torch.cat([td_lazy, td_dense], dim=1)

    def test_cat_mixed_lazy_dense_out_dense(self):
        td_lazy = LazyStackedTensorDict(
            *[
                TensorDict({"a": torch.full((2, 3), float(i))}, [2, 3])
                for i in range(2)
            ],
            stack_dim=0,
        )
        td_dense = TensorDict(
            {"a": torch.arange(12, dtype=torch.float).reshape(2, 2, 3)},
            [2, 2, 3],
        )
        out = TensorDict({"a": torch.empty(4, 2, 3)}, [4, 2, 3])

        res = torch.cat([td_lazy, td_dense], dim=0, out=out)
        assert res is out
        assert assert_allclose_td(
            out,
            TensorDict(
                {"a": torch.cat([td_lazy.get("a"), td_dense.get("a")], 0)},
                [4, 2, 3],
            ),
        )

    @pytest.mark.parametrize("device", [None, *get_available_devices()])
    @pytest.mark.parametrize("use_file", [False, True])
    def test_consolidate(self, device, use_file, tmpdir):
        td = TensorDict(
            {
                "a": torch.arange(3).expand(1, 3).clone(),
                "b": {"c": torch.arange(3, dtype=torch.double).expand(1, 3).clone()},
                "d": "a string!",
            },
            device=device,
            batch_size=[1, 3],
        )
        td = LazyStackedTensorDict(*td.unbind(1), stack_dim=1)

        if not use_file:
            td_c = td.consolidate()
            assert td_c.device == device
        else:
            filename = Path(tmpdir) / "file.mmap"
            td_c = td.consolidate(filename=filename)
            assert td_c.device == torch.device("cpu")
            assert (TensorDict.from_consolidated(filename) == td_c).all()
        assert hasattr(td_c, "_consolidated")
        assert type(td_c) == type(td)  # noqa
        assert (td.to(td_c.device) == td_c).all()
        assert td["d"] == [["a string!"] * 3]
        assert td_c["d"] == [["a string!"] * 3]

        storage = td_c._consolidated["storage"]
        storage *= 0
        assert (td.to(td_c.device) != td_c).any()

        filename = Path(tmpdir) / "file.pkl"
        torch.save(td, filename)
        tdload = torch.load(filename, weights_only=False)
        assert (td == tdload).all()

        td_c = td.consolidate()
        torch.save(td_c, filename)
        tdload = torch.load(filename, weights_only=False)
        assert (td == tdload).all()

        def check_id(a, b):
            if isinstance(a, (torch.Size, str)):
                assert a == b
            if isinstance(a, torch.Tensor):
                assert (a == b).all()

        torch.utils._pytree.tree_map(check_id, td_c._consolidated, tdload._consolidated)
        assert tdload.is_consolidated()

    @pytest.mark.parametrize("device", [None, *get_available_devices()])
    @pytest.mark.parametrize("use_file", [False, True])
    @pytest.mark.parametrize("num_threads", [0, 1, 4])
    def test_consolidate_njt(self, device, use_file, tmpdir, num_threads):
        td = TensorDict(
            {
                "a": torch.arange(3).expand(4, 3).clone(),
                "b": {"c": torch.arange(3, dtype=torch.double).expand(4, 3).clone()},
                "d": "a string!",
                "njt": torch.nested.nested_tensor_from_jagged(
                    torch.arange(10, device=device),
                    offsets=torch.tensor([0, 2, 5, 8, 10], device=device),
                ),
                "njt_lengths": torch.nested.nested_tensor_from_jagged(
                    torch.arange(10, device=device),
                    offsets=torch.tensor([0, 2, 5, 8, 10], device=device),
                    lengths=torch.tensor([2, 3, 3, 2], device=device),
                ),
            },
            device=device,
            batch_size=[4],
        )

        if not use_file:
            td_c = td.consolidate(num_threads=num_threads)
            assert td_c.device == device
        else:
            filename = Path(tmpdir) / "file.mmap"
            td_c = td.consolidate(filename=filename, num_threads=num_threads)
            assert td_c.device == torch.device("cpu")
            assert assert_allclose_td(TensorDict.from_consolidated(filename), td_c)
        assert hasattr(td_c, "_consolidated")
        assert type(td_c) == type(td)  # noqa
        assert td_c["d"] == "a string!"

        assert_allclose_td(td.to(td_c.device), td_c)

        tdload_make, tdload_data = _reduce_td(td)
        tdload = tdload_make(*tdload_data)
        assert (td == tdload).all()

        td_c = td.consolidate(num_threads=num_threads)
        tdload_make, tdload_data = _reduce_td(td_c)
        tdload = tdload_make(*tdload_data)
        assert assert_allclose_td(td, tdload)

        def check_id(a, b):
            if isinstance(a, (torch.Size, str)):
                assert a == b
            if isinstance(a, torch.Tensor):
                assert (a == b).all()

        torch.utils._pytree.tree_map(check_id, td_c._consolidated, tdload._consolidated)
        assert tdload.is_consolidated()
        assert tdload["njt_lengths"]._lengths is not None

    @pytest.mark.parametrize("device", [None, *get_available_devices()])
    @pytest.mark.parametrize("num_threads", [0, 1, 4])
    def test_consolidate_njt_ragged_idx(self, device, num_threads):
        ragged_tensors = [torch.randn(20, 10, i, device=device) for i in range(1, 5)]
        ragged = torch.nested.as_nested_tensor(ragged_tensors, layout=torch.jagged)
        ragged_td = TensorDict({"ragged": ragged}, batch_size=[len(ragged_tensors)])
        ragged_td_c = ragged_td.consolidate(num_threads=num_threads)
        buffer = io.BytesIO()
        pickle.dump(ragged_td_c, buffer)
        buffer.seek(0)
        ragged_td_load = pickle.load(buffer)
        assert ragged_td_load["ragged"]._ragged_idx == ragged._ragged_idx

    @pytest.mark.skipif(
        not torch.cuda.is_available() and not is_npu_available(),
        reason="no cuda or npu device detected",
    )
    def test_consolidate_to_device(self):
        td = TensorDict(
            {
                "a": torch.arange(3).expand(1, 3).clone(),
                "b": {"c": torch.arange(3, dtype=torch.double).expand(1, 3).clone()},
                "d": "a string!",
            },
            device="cpu",
            batch_size=[1, 3],
        )
        td = LazyStackedTensorDict(*td.unbind(1), stack_dim=1)
        td_c = td.consolidate()
        assert td_c.device == torch.device("cpu")

        td_c_device = td_c.to(f"{cur_device}:0")
        assert td_c_device.device == torch.device(f"{cur_device}:0")
        assert td_c_device.is_consolidated()
        dataptrs = set()
        for tensor in td_c_device.values(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
            assert tensor.device == torch.device(f"{cur_device}:0")
            dataptrs.add(tensor.untyped_storage().data_ptr())
        assert (td_c_device.cpu() == td).all()
        assert td_c_device["d"] == [["a string!"] * 3]
        assert len(dataptrs) == 1

    @pytest.mark.skipif(
        not torch.cuda.is_available() and not is_npu_available(),
        reason="no cuda or npu device detected",
    )
    def test_consolidate_to_device_njt(self):
        td = TensorDict(
            {
                "a": torch.arange(3).expand(4, 3).clone(),
                "d": "a string!",
                "njt": torch.nested.nested_tensor_from_jagged(
                    torch.arange(10), offsets=torch.tensor([0, 2, 5, 8, 10])
                ),
                "njt_lengths": torch.nested.nested_tensor_from_jagged(
                    torch.arange(10),
                    offsets=torch.tensor([0, 2, 5, 8, 10]),
                    lengths=torch.tensor([2, 3, 3, 2]),
                ),
            },
            device="cpu",
            batch_size=[4],
        )
        device = torch.device(f"{cur_device}:0")
        td_c = td.consolidate()
        assert td_c.device == torch.device("cpu")
        td_c_device = td_c.to(device)
        assert td_c_device.device == device
        assert td_c_device.is_consolidated()
        dataptrs = set()
        for tensor in td_c_device.values(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS):
            assert tensor.device == device
            if tensor.is_nested:
                vals = tensor._values
                dataptrs.add(vals.untyped_storage().data_ptr())
                offsets = tensor._offsets
                dataptrs.add(offsets.untyped_storage().data_ptr())
                lengths = tensor._lengths
                if lengths is not None:
                    dataptrs.add(lengths.untyped_storage().data_ptr())
            else:
                dataptrs.add(tensor.untyped_storage().data_ptr())
        assert len(dataptrs) == 1
        assert assert_allclose_td(td_c_device.cpu(), td)
        assert td_c_device["njt_lengths"]._lengths is not None

    def test_create_empty(self):
        td = LazyStackedTensorDict(stack_dim=0)
        assert td.device is None
        assert td.shape == torch.Size([0])
        td = LazyStackedTensorDict(stack_dim=1, batch_size=[1, 2], device="cpu")
        assert td.device == torch.device("cpu")
        assert td.shape == torch.Size([1, 0, 2])
        for clone in (td.clone(), td.clone(recurse=False)):
            assert clone.shape == torch.Size([1, 0, 2])
            assert clone.device == torch.device("cpu")

    def test_densify(self):
        td0 = TensorDict(
            a=torch.zeros((1,)),
            b=torch.zeros((2,)),
            d=TensorDict(e=torch.zeros(())),
        )
        td1 = TensorDict(
            b=torch.ones((1,)), c=torch.ones((2,)), d=TensorDict(e=torch.ones(()))
        )
        td = LazyStackedTensorDict(td0, td1, stack_dim=0)
        td_jagged = td.densify(layout=torch.jagged)
        assert (td_jagged.exclude("c").unbind(0)[0] == 0).all()
        assert (td_jagged.exclude("a").unbind(0)[1] == 1).all()
        assert not td_jagged["d", "e"].is_nested
        td_strided = td.densify(layout=torch.strided)
        assert (td_strided.exclude("c")[0] == 0).all()
        assert (td_strided.exclude("a")[1] == 1).all()
        assert not td_strided["d", "e"].is_nested
        td_nest = TensorDict(td=td, batch_size=[2])
        td_nest_jagged = td_nest.densify(layout=torch.jagged)
        assert (td_nest_jagged.exclude(("td", "c")).unbind(0)[0] == 0).all()
        assert (td_nest_jagged.exclude(("td", "a")).unbind(0)[1] == 1).all()
        assert not td_nest_jagged["td", "d", "e"].is_nested
        td_nest_strided = td_nest.densify(layout=torch.strided)
        assert (td_nest_strided.exclude(("td", "c"))[0] == 0).all()
        assert (td_nest_strided.exclude(("td", "a"))[1] == 1).all()
        assert not td_nest_strided["td", "d", "e"].is_nested

    def test_densify_names(self):
        named = TensorDict(x=torch.zeros(3), batch_size=[3], names=["n"])
        unnamed = TensorDict(x=torch.zeros(3), batch_size=[3])
        # When only some of the stacked tensordicts have names, the dims are
        # left unnamed, as in contiguous().
        td = lazy_stack([named, unnamed])
        assert td.densify().names == td.contiguous().names == [None, None]
        # The name of the stack dim is kept.
        td = LazyStackedTensorDict.lazy_stack([named, unnamed], 1, stack_dim_name="s")
        assert td.densify().names == [None, "s"]
        td = LazyStackedTensorDict.lazy_stack(
            [unnamed, unnamed.clone()], stack_dim_name="s"
        )
        assert td.densify().names == ["s", None]
        td = LazyStackedTensorDict.lazy_stack(
            [named, named.clone()], stack_dim_name="s"
        )
        assert td.densify().names == ["s", "n"]

    def test_lazy_get(self):
        inner_td = lazy_stack(
            [
                TensorDict({"x": torch.ones(1)}),
                TensorDict({"x": torch.ones(2) * 2}),
            ]
        )
        td = TensorDict(inner=inner_td, batch_size=[2])
        with pytest.raises(
            RuntimeError, match="Failed to stack tensors within a tensordict"
        ):
            td.get(("inner", "x"))
        x = td.get(("inner", "x"), as_nested_tensor=True)
        assert x.is_nested
        x = td.get(("inner", "x"), as_list=True)
        assert isinstance(x, list)
        x = td.get(("inner", "x"), as_padded_tensor=True)
        assert isinstance(x, torch.Tensor)
        assert x[0, 1] == 0
        x = td.get(
            ("inner", "x"),
            as_padded_tensor=True,
            padding_side="left",
            padding_value=100,
        )
        assert isinstance(x, torch.Tensor)
        assert x[0, 0] == 100

    def test_lazy_stack_nested_get(self):
        td0 = TensorDict(a=torch.randn(3))
        td1 = TensorDict(a=torch.randn(4))

        td = lazy_stack([td0, td1])
        td = lazy_stack([td, td.copy()])
        td = lazy_stack([td, td.copy()])

        assert isinstance(td.get("a", as_list=True), list)

        assert isinstance(td.get("a", as_padded_tensor=True), torch.Tensor)

    def test_lazy_stack_nested_tensordict_as_list(self):
        """Regression test: as_list should return list for leaf tensors,
        but properly stack nested TensorDicts"""
        # Create nested structure with TensorDicts
        inner_td0 = TensorDict({"full": torch.randn(3)})
        inner_td1 = TensorDict({"full": torch.randn(4)})

        td0 = TensorDict({"log_probs": inner_td0})
        td1 = TensorDict({"log_probs": inner_td1})

        stacked = lazy_stack([td0, td1])

        # This should return a list of tensors (the leaf values), not a list of TensorDicts
        result = stacked.get(("log_probs", "full"), as_list=True)
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)
        assert result[0].shape == torch.Size([3])
        assert result[1].shape == torch.Size([4])

        # Verify intermediate TensorDict is properly stacked
        log_probs = stacked.get("log_probs")
        assert isinstance(log_probs, LazyStackedTensorDict)

    @pytest.mark.parametrize("pos1", range(8))
    @pytest.mark.parametrize("pos2", range(8))
    @pytest.mark.parametrize("pos3", range(8))
    def test_lazy_indexing(self, pos1, pos2, pos3):
        torch.manual_seed(0)
        td_leaf_1 = TensorDict({"a": torch.ones(2, 3)}, [])
        inner = LazyStackedTensorDict.lazy_stack([td_leaf_1] * 4, 0)
        middle = LazyStackedTensorDict.lazy_stack([inner] * 3, 0)
        outer = LazyStackedTensorDict.lazy_stack([middle] * 2, 0)
        outer_dense = outer.to_tensordict(retain_none=True)
        pos1 = self._idx_list[pos1]
        pos2 = self._idx_list[pos2]
        pos3 = self._idx_list[pos3]
        index = (pos1, pos2, pos3)
        result = outer[index]
        ref_tensor = torch.zeros(outer.shape)
        assert result.batch_size == ref_tensor[index].shape, (
            result.batch_size,
            ref_tensor[index].shape,
            index,
        )
        assert result.batch_size == outer_dense[index].shape, (
            result.batch_size,
            outer_dense[index].shape,
            index,
        )

    def test_lazy_mask_hetero(self):
        td = LazyStackedTensorDict(
            TensorDict({"a": torch.zeros(6, 3), "b": "a string!"}, batch_size=[6]),
            TensorDict({"a": torch.ones(6, 4), "b": "another string!"}, batch_size=[6]),
            TensorDict(
                {"a": torch.ones(6, 5) * 2, "b": "a third string!"}, batch_size=[6]
            ),
            stack_dim=1,
        )
        mask0a = torch.zeros(6, dtype=torch.bool)
        mask0a[:3] = 1
        mask0b = ~mask0a
        split0 = td[mask0a], td[mask0b]
        assert (torch.cat(split0, dim=0) == td).all()

        mask0a = torch.zeros(6, dtype=torch.bool)
        mask0c = mask0a.clone()
        mask0a[:3] = 1
        mask0b = ~mask0a
        split0 = td[mask0a], td[mask0b], td[mask0c]
        assert (torch.cat(split0, dim=0) == td).all()

        mask1a = torch.zeros(3, dtype=torch.bool)
        mask1a[:2] = 1
        mask1b = ~mask1a
        split1 = td[:, mask1a], td[:, mask1b]
        assert (torch.cat(split1, dim=1) == td).all()

        mask1a = torch.zeros(3, dtype=torch.bool)
        mask1c = mask1a.clone()
        mask1a[:2] = 1
        mask1b = ~mask1a
        split1b = td[:, mask1a], td[:, mask1b], td[:, mask1c]
        assert (torch.cat(split1b, dim=1) == td).all()

    @pytest.mark.parametrize("stack_dim", [0, 1, 2])
    @pytest.mark.parametrize("mask_dim", [0, 1, 2])
    @pytest.mark.parametrize("single_mask_dim", [True, False])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_lazy_mask_indexing(self, stack_dim, mask_dim, single_mask_dim, device):
        torch.manual_seed(0)
        td = TensorDict({"a": torch.zeros(9, 10, 11)}, [9, 10, 11], device=device)
        td = LazyStackedTensorDict.lazy_stack(
            [
                td,
                td.apply(lambda x: x + 1),
                td.apply(lambda x: x + 2),
                td.apply(lambda x: x + 3),
            ],
            stack_dim,
        )
        mask = torch.zeros(())
        while not mask.any():
            if single_mask_dim:
                mask = torch.zeros(td.shape[mask_dim], dtype=torch.bool).bernoulli_()
            else:
                mask = torch.zeros(
                    td.shape[mask_dim : mask_dim + 2], dtype=torch.bool
                ).bernoulli_()
        index = (slice(None),) * mask_dim + (mask,)
        tdmask = td[index]
        assert tdmask["a"].shape == td["a"][index].shape
        assert (tdmask["a"] == td["a"][index]).all()
        index = (0,) * mask_dim + (mask,)
        tdmask = td[index]
        assert tdmask["a"].shape == td["a"][index].shape
        assert (tdmask["a"] == td["a"][index]).all()
        index = (slice(1),) * mask_dim + (mask,)
        tdmask = td[index]
        assert tdmask["a"].shape == td["a"][index].shape
        assert (tdmask["a"] == td["a"][index]).all()

    @pytest.mark.parametrize("stack_dim", [0, 1, 2, 3])
    @pytest.mark.parametrize(
        "pattern",
        [
            "ts t",
            "t st",
            "ts ts",
            "st st",
            "ts s t",
            "t s s t",
            "t is t",
            "t s i t",
            "s t s t",
        ],
    )
    def test_lazy_separated_tensor_indices(self, stack_dim, pattern):
        # tensor indices separated by a slice put the broadcast dim first,
        # as in the dense tensordict (#1838)
        shape = (3, 4, 2)
        tds = [TensorDict({"x": torch.randn(*shape)}, list(shape)) for _ in range(5)]
        lazy = lazy_stack(tds, stack_dim)
        dense = lazy.to_tensordict()
        index = []
        for code, size in zip(pattern.replace(" ", ""), lazy.batch_size):
            if code == "t":
                index.append(torch.randint(0, size, (2,)))
            elif code == "s":
                index.append(slice(None))
            else:
                index.append(0)
        index = tuple(index)
        expected = dense[index]
        result = lazy[index]
        assert result.batch_size == expected.batch_size
        assert (result["x"] == expected["x"]).all()

    def test_lazy_separated_tensor_indices_slice_bounds(self):
        tds = [TensorDict({"x": torch.randn(3, 4)}, [3, 4]) for _ in range(5)]
        lazy = lazy_stack(tds, 2)
        dense = lazy.to_tensordict()
        index = (torch.tensor([0, 2]), slice(1, 3), torch.tensor([4, 1]))
        result = lazy[index]
        expected = dense[index]
        assert result.batch_size == expected.batch_size
        assert (result["x"] == expected["x"]).all()

    @pytest.mark.parametrize("stack_dim", [0, 1, 2])
    @pytest.mark.parametrize(
        "index",
        [
            (slice(None), 0),
            # scalar bools inside a tuple are advanced indices that use no dim
            (True, slice(None), [0, 1]),
            (slice(None), torch.tensor(True), [1, 0]),
            (0, False),
            (False, slice(None)),
            # None between advanced indices puts their block first
            ([0, 1], None, [1, 0]),
            # N-D index tensors that broadcast together
            (torch.tensor([[0], [2]]), torch.tensor([1, 3])),
            # the same position twice
            (torch.tensor([1, 0, 1]), torch.tensor([1, 0, 1])),
            # a 2-D mask
            (
                slice(None),
                torch.tensor(
                    [[True, False], [False, True], [True, True], [False, False]]
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "op", ["getitem", "setitem", "setitem_scalar", "set_at_", "update_at_"]
    )
    def test_lazy_index_matches_dense(self, stack_dim, index, op):
        dense = TensorDict(
            {
                "a": torch.arange(24.0).view(3, 4, 2),
                # torch casts a written scalar to the dtype of the entry
                "c": torch.arange(24).view(3, 4, 2),
                "nested": {"b": torch.arange(48.0).view(3, 4, 2, 2)},
            },
            [3, 4, 2],
        )
        lazy = LazyStackedTensorDict.lazy_stack(
            dense.clone().unbind(stack_dim), stack_dim
        )
        if op == "getitem":
            result, expected = lazy[index], dense[index]
            assert result.batch_size == expected.batch_size
            assert (result.to_tensordict() == expected).all()
            return
        if op == "setitem":
            value = dense[index].apply(lambda x: -1 - x)
            lazy[index] = value
            dense[index] = value
        elif op == "setitem_scalar":
            lazy[index] = -1.0
            dense[index] = -1.0
        elif op == "set_at_":
            value = -1 - dense["nested", "b"][index]
            lazy.set_at_(("nested", "b"), value, index)
            dense.set_at_(("nested", "b"), value, index)
        else:
            value = dense[index].apply(lambda x: -1 - x)
            lazy.update_at_(value, index)
            dense.update_at_(value, index)
        assert (lazy.to_tensordict() == dense).all()

    @pytest.mark.parametrize("stack_dim", [0, 1, 2])
    @pytest.mark.parametrize(
        "index", [(..., 0), (0, ..., 1), (..., [1, 0]), (..., None)]
    )
    @pytest.mark.parametrize("op", ["set_at_", "set_at_ scalar", "update_at_"])
    def test_lazy_set_at_ellipsis(self, stack_dim, index, op):
        # set_at_ reads the index on the entry, as torch does: the Ellipsis
        # also covers the dims of the entry after the batch dims
        dense = TensorDict(
            {
                "b": torch.arange(48.0).view(3, 4, 2, 2),
                "nested": {"c": torch.arange(48).view(3, 4, 2, 2)},
            },
            [3, 4, 2],
        )
        lazy = LazyStackedTensorDict.lazy_stack(
            dense.clone().unbind(stack_dim), stack_dim
        )
        if op == "set_at_":
            value = -1 - dense["b"][index]
            lazy.set_at_("b", value, index)
            dense.set_at_("b", value, index)
        elif op == "set_at_ scalar":
            lazy.set_at_(("nested", "c"), -7.5, index)
            dense.set_at_(("nested", "c"), -7.5, index)
        else:
            value = TensorDict({"b": -1 - dense["b"][index]})
            lazy.update_at_(value, index)
            dense.update_at_(value, index)
        assert (lazy.to_tensordict() == dense).all()

    def test_lazy_setitem_scalar_unbatched(self):
        # each member writes a scalar as it is, also into an entry that the
        # batch dims don't index
        tds = [
            TensorDict({"a": torch.zeros(2), "u": UnbatchedTensor(torch.zeros(5))}, [2])
            for _ in range(3)
        ]
        lazy = lazy_stack(tds)
        lazy[1] = 5.0
        lazy[[0, 2]] = torch.tensor(4.0)
        for td, value in zip(tds, (4.0, 5.0, 4.0)):
            assert (td["a"] == value).all()
            assert (td["u"] == value).all()

    def test_lazy_mask_indexing_single(self):
        td = LazyStackedTensorDict(
            TensorDict({"a": torch.ones(())}),
            TensorDict({"a": torch.zeros(())}),
        )
        tdi = td[torch.tensor([True, False])]
        assert tdi.shape == (1,)

        td[torch.tensor([True, False])] = TensorDict({"a": 0.0})
        assert (td == 0).all()
        td[torch.tensor([True, False])] = TensorDict({"b": 0.0})
        td[torch.tensor([False, True])] = TensorDict({"b": 0.0})
        assert (td["b"] == 0).all()
        assert (td == 0).all()
        td[torch.tensor([True, False])] = TensorDict({("c", "d"): 0.0})
        td[torch.tensor([False, True])] = TensorDict({("c", "d"): 0.0})
        assert (td["c", "d"] == 0).all()
        assert (td == 0).all()

    @pytest.mark.parametrize("stack_dim", [0, 1, 2])
    @pytest.mark.parametrize("mask_dim", [0, 1, 2])
    @pytest.mark.parametrize("single_mask_dim", [True, False])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_lazy_mask_setitem(self, stack_dim, mask_dim, single_mask_dim, device):
        torch.manual_seed(0)
        td = TensorDict({"a": torch.zeros(9, 10, 11)}, [9, 10, 11], device=device)
        td = LazyStackedTensorDict.lazy_stack(
            [
                td,
                td.apply(lambda x: x + 1),
                td.apply(lambda x: x + 2),
                td.apply(lambda x: x + 3),
            ],
            stack_dim,
        )
        mask = torch.zeros(())
        while not mask.any():
            if single_mask_dim:
                mask = torch.zeros(td.shape[mask_dim], dtype=torch.bool).bernoulli_()
            else:
                mask = torch.zeros(
                    td.shape[mask_dim : mask_dim + 2], dtype=torch.bool
                ).bernoulli_()
        index = (slice(None),) * mask_dim + (mask,)
        tdset = TensorDict({"a": td["a"][index] * 0 - 1}, [])
        # we know that the batch size is accurate from test_lazy_mask_indexing
        tdset.batch_size = td[index].batch_size
        td[index] = tdset
        assert (td["a"][index] == tdset["a"]).all()
        assert (td["a"][index] == tdset["a"]).all()
        index = (slice(1),) * mask_dim + (mask,)
        tdset = TensorDict({"a": td["a"][index] * 0 - 1}, [])
        tdset.batch_size = td[index].batch_size
        td[index] = tdset
        assert (td["a"][index] == tdset["a"]).all()
        assert (td["a"][index] == tdset["a"]).all()

    @pytest.mark.parametrize(
        "batch_size,index",
        [
            # the examples of issue #1834
            ((3, 4), (torch.tensor([True, False, True]), torch.tensor([0, 3]))),
            ((3, 4), (torch.tensor([0, 2]), torch.tensor([True, False, True, False]))),
            ((2, 2), (torch.tensor([True, True]), torch.tensor([1, 0]))),
            # int list and second mask as the other advanced index
            ((3, 4, 2), (torch.tensor([True, False, True]), [3, 1])),
            (
                (3, 4, 2),
                (
                    torch.tensor([True, False, True]),
                    torch.tensor([False, True, False, True]),
                ),
            ),
            # mask on a middle dim, after a slice, an int or an ellipsis
            (
                (3, 4, 2),
                (
                    slice(None),
                    torch.tensor([True, False, True, True]),
                    torch.tensor([1, 0, 1]),
                ),
            ),
            (
                (3, 4, 2),
                (0, torch.tensor([True, False, True, True]), torch.tensor([1, 0, 1])),
            ),
            (
                (3, 4, 2),
                (
                    Ellipsis,
                    torch.tensor([True, False, True, True]),
                    torch.tensor([1, 0, 1]),
                ),
            ),
            # an int between the mask and the other advanced index
            ((3, 4, 2), (torch.tensor([True, False, True]), 0, torch.tensor([1, 0]))),
            # three advanced indices, two of them masks
            (
                (3, 4, 2),
                (
                    torch.tensor([True, False, True]),
                    torch.tensor([1]),
                    torch.tensor([False, True]),
                ),
            ),
            # 2-dim masks before and after the other advanced index
            (
                (3, 4, 2),
                (
                    torch.tensor(
                        [
                            [True, False, False, True],
                            [False, False, False, False],
                            [False, True, False, False],
                        ]
                    ),
                    torch.tensor([0, 1, 1]),
                ),
            ),
            (
                (3, 4, 2),
                (
                    torch.tensor([2, 0]),
                    torch.tensor(
                        [[False, True], [False, False], [True, False], [False, False]]
                    ),
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("stack_dim", [0, 1, -1])
    @pytest.mark.parametrize("op", ["getitem", "setitem", "set_at_"])
    def test_lazy_mask_with_advanced_index(self, batch_size, index, stack_dim, op):
        # a mask combined with other advanced indices must behave like its
        # nonzero() indices, as in torch and in the dense TensorDict
        numel = torch.Size(batch_size).numel()
        dense = TensorDict(
            {
                "a": torch.arange(numel).view(batch_size),
                "b": torch.arange(numel * 2).view(*batch_size, 2),
            },
            batch_size,
        )
        lazy = LazyStackedTensorDict.lazy_stack(
            dense.clone().unbind(stack_dim), stack_dim
        )
        if op == "getitem":
            result = lazy[index]
            expected = dense[index]
            assert result.batch_size == expected.batch_size
            assert (result == expected).all()
            return
        value = dense[index].apply(lambda x: -x - 1)
        if op == "setitem":
            lazy[index] = value
            dense[index] = value
        else:
            lazy.set_at_("a", value["a"], index)
            dense.set_at_("a", value["a"], index)
        assert (lazy.to_tensordict() == dense).all()

    @staticmethod
    def _lazy_mask_op(td, index, op):
        value = TensorDict(a=-1.0, nested=TensorDict(b=-1.0)).expand(
            _getitem_batch_size(td.batch_size, index)
        )
        if op == "getitem":
            return td[index]
        if op == "get_at":
            return td.get_at("nested", index)
        if op == "setitem":
            td[index] = -1.0
        elif op == "setitem_td":
            td[index] = value
        elif op == "set_at_":
            td.set_at_("a", value["a"], index)
        else:
            td.update_at_(value, index)

    @staticmethod
    def _lazy_mask_tds(stack_dim):
        batch_size = (4, 3)
        dense = TensorDict(
            a=torch.arange(12.0).view(batch_size),
            nested=TensorDict(
                b=torch.arange(12.0).view(batch_size), batch_size=batch_size
            ),
            batch_size=batch_size,
        )
        lazy = LazyStackedTensorDict.lazy_stack(
            dense.clone().unbind(stack_dim), stack_dim
        )
        return dense, lazy

    _LAZY_MASK_OPS = [
        "getitem",
        "get_at",
        "setitem",
        "setitem_td",
        "set_at_",
        "update_at_",
    ]

    @pytest.mark.parametrize(
        "index",
        [
            torch.tensor([True, False, True]),
            torch.tensor([True, False, True, False, True]),
            (slice(None), torch.tensor([True, False])),
            (slice(None), torch.tensor([True, False, True, True])),
            torch.ones(4, 2, dtype=torch.bool),
            torch.ones(3, 3, dtype=torch.bool),
            (torch.tensor([0, 2]), torch.tensor([True, False])),
            (torch.tensor([True, False, True]), torch.tensor([0, 2])),
            (Ellipsis, torch.tensor([True, True])),
        ],
        ids=[
            "dim0-short",
            "dim0-long",
            "dim1-short",
            "dim1-long",
            "2d-dim1-short",
            "2d-dim0-short",
            "index-and-dim1-short",
            "dim0-short-and-index",
            "ellipsis-dim1-short",
        ],
    )
    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("op", _LAZY_MASK_OPS)
    def test_lazy_mask_wrong_shape(self, index, stack_dim, op):
        # a mask that does not have the shape of the dims it indexes raises as
        # in torch, along the stack dim and along the dims of the members
        dense, lazy = self._lazy_mask_tds(stack_dim)
        with pytest.raises(IndexError) as torch_error:
            torch.zeros(dense.batch_size)[index]
        with pytest.raises(IndexError) as error:
            self._lazy_mask_op(lazy, index, op)
        assert str(error.value) == str(torch_error.value)
        # nothing was written
        assert (lazy.to_tensordict() == dense).all()

    @pytest.mark.parametrize(
        "index",
        [
            torch.tensor([True, False, True, False]),
            (slice(None), torch.tensor([True, False, True])),
            torch.tensor([[True, False, True]] * 2 + [[False, True, False]] * 2),
            (torch.tensor([0, 2]), torch.tensor([True, False, True])),
            (torch.tensor([True, False, True, False]), torch.tensor([0, 2])),
            (Ellipsis, torch.tensor([True, True, False])),
        ],
        ids=["dim0", "dim1", "2d", "index-and-dim1", "dim0-and-index", "ellipsis-dim1"],
    )
    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("op", _LAZY_MASK_OPS)
    def test_lazy_mask_right_shape(self, index, stack_dim, op):
        # a mask that has the shape of the dims it indexes behaves as in the
        # dense TensorDict
        dense, lazy = self._lazy_mask_tds(stack_dim)
        result = self._lazy_mask_op(lazy, index, op)
        expected = self._lazy_mask_op(dense, index, op)
        if result is not None:
            assert result.batch_size == expected.batch_size
            assert (result == expected).all()
        assert (lazy.to_tensordict() == dense).all()

    @pytest.mark.parametrize("batch_size", [(), (32,), (32, 4)])
    def test_lazy_stack_stack(self, batch_size):
        obs = self.nested_lazy_het_td(batch_size)

        assert isinstance(obs, TensorDict)
        assert isinstance(obs["lazy"], LazyStackedTensorDict)
        assert obs["lazy"].stack_dim == len(obs["lazy"].shape) - 1  # succeeds
        assert obs["lazy"].shape == (*batch_size, 3)
        assert isinstance(obs["lazy"][..., 0], TensorDict)  # succeeds

        obs_stack = LazyStackedTensorDict.lazy_stack([obs])

        assert (
            isinstance(obs_stack, LazyStackedTensorDict) and obs_stack.stack_dim == 0
        )  # succeeds
        assert obs_stack.batch_size == (1, *batch_size)  # succeeds
        assert obs_stack[0] is obs  # succeeds
        assert isinstance(obs_stack["lazy"], LazyStackedTensorDict)
        assert obs_stack["lazy"].shape == (1, *batch_size, 3)
        assert obs_stack["lazy"].stack_dim == 0  # succeeds
        assert obs_stack["lazy"][0] is obs["lazy"]

        obs2 = obs.clone()
        obs_stack = LazyStackedTensorDict.lazy_stack([obs, obs2])

        assert (
            isinstance(obs_stack, LazyStackedTensorDict) and obs_stack.stack_dim == 0
        )  # succeeds
        assert obs_stack.batch_size == (2, *batch_size)  # succeeds
        assert obs_stack[0] is obs  # succeeds
        assert isinstance(obs_stack["lazy"], LazyStackedTensorDict)
        assert obs_stack["lazy"].shape == (2, *batch_size, 3)
        assert obs_stack["lazy"].stack_dim == 0  # succeeds
        assert obs_stack["lazy"][0] is obs["lazy"]

    @pytest.mark.parametrize("dim", range(2))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_lazy_stacked_append(self, dim, device):
        td = TensorDict({"a": torch.zeros(4)}, [4], device=device)
        lstd = LazyStackedTensorDict.lazy_stack([td] * 2, dim=dim)

        lstd.append(
            TensorDict(
                {"a": torch.ones(4), "invalid": torch.rand(4)}, [4], device=device
            )
        )

        bs = [4]
        bs.insert(dim, 3)

        assert lstd.batch_size == torch.Size(bs)
        assert set(lstd.keys()) == {"a"}

        t = torch.zeros(*bs, device=device)

        if dim == 0:
            t[-1] = 1
        else:
            t[:, -1] = 1

        torch.testing.assert_close(lstd["a"], t)

        with pytest.raises(
            TypeError, match="Expected new value to be TensorDictBase instance"
        ):
            lstd.append(torch.rand(10))

        if device != torch.device("cpu"):
            with pytest.raises(ValueError, match="Devices differ"):
                lstd.append(TensorDict({"a": torch.ones(4)}, [4], device="cpu"))

        with pytest.raises(ValueError, match="Batch sizes in tensordicts differs"):
            lstd.append(TensorDict({"a": torch.ones(17)}, [17], device=device))

    def test_lazy_stacked_contains(self):
        td = TensorDict(
            {"a": TensorDict({"b": torch.rand(1, 2)}, [1, 2]), "c": torch.rand(1)}, [1]
        )
        lstd = LazyStackedTensorDict.lazy_stack([td, td, td])

        assert td in lstd
        assert td.clone() not in lstd

        assert "random_string" not in lstd
        assert "a" in lstd

    def test_lazy_stack_keys_tc(self):
        class MC(TensorClass):
            a: torch.Tensor

        x = TensorDict(
            b=MC.from_tensordict(
                lazy_stack([TensorDict(a=torch.randn(3)), TensorDict(a=torch.randn(5))])
            )
        )
        assert list(x.keys()) == ["b"]
        assert list(x["b"].keys()) == ["a"]
        assert set(x.keys(include_nested=True)) == {"b", ("b", "a")}
        assert set(x.keys(include_nested=True, leaves_only=True)) == {("b", "a")}

    @pytest.mark.parametrize("dim", range(2))
    @pytest.mark.parametrize("index", range(2))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_lazy_stacked_insert(self, dim, index, device):
        td = TensorDict({"a": torch.zeros(4)}, [4], device=device)
        lstd = LazyStackedTensorDict.lazy_stack([td] * 2, dim=dim)

        lstd.insert(
            index,
            TensorDict(
                {"a": torch.ones(4), "invalid": torch.rand(4)}, [4], device=device
            ),
        )

        bs = [4]
        bs.insert(dim, 3)

        assert lstd.batch_size == torch.Size(bs)
        assert set(lstd.keys()) == {"a"}

        t = torch.zeros(*bs, device=device)

        if dim == 0:
            t[index] = 1
        else:
            t[:, index] = 1

        torch.testing.assert_close(lstd["a"], t)

        with pytest.raises(
            TypeError, match="Expected new value to be TensorDictBase instance"
        ):
            lstd.insert(index, torch.rand(10))

        if device != torch.device("cpu"):
            with pytest.raises(ValueError, match="Devices differ"):
                lstd.insert(index, TensorDict({"a": torch.ones(4)}, [4], device="cpu"))

        with pytest.raises(ValueError, match="Batch sizes in tensordicts differs"):
            lstd.insert(index, TensorDict({"a": torch.ones(17)}, [17], device=device))

    def test_lazy_stack_view_full_size(self):
        tds = LazyStackedTensorDict(*[TensorDict(a=i) for i in range(60)], stack_dim=0)
        tdview = tds.view(3, 4, 5)
        assert isinstance(tdview, LazyStackedTensorDict)
        assert isinstance(tdview[0], LazyStackedTensorDict)
        assert isinstance(tdview[0, 0], LazyStackedTensorDict)
        assert (tdview["a"].view(60) == tds["a"]).all()
        assert (tdview.view(tds.shape) == tds).all()
        assert (tdview == tds.unflatten(0, (3, 4, 5))).all()
        assert (tds == tdview.flatten()).all()

    def test_lazy_stack_view_part_size(self):
        tds = LazyStackedTensorDict(
            *[TensorDict(a=a, batch_size=(2,)) for a in torch.arange(120).chunk(60)],
            stack_dim=1,
        )
        assert tds.shape == (2, 60)
        tdview = tds.view(2, 3, 4, 5)
        assert isinstance(tdview[0], LazyStackedTensorDict)
        assert isinstance(tdview[0, 0], LazyStackedTensorDict)
        assert isinstance(tdview[0, 0, 0], LazyStackedTensorDict)
        assert (tdview["a"].view(120) == tds["a"].view(120)).all()
        assert (tdview.view(tds.shape) == tds).all()
        assert (tdview == tds.unflatten(1, (3, 4, 5))).all()
        assert (tds == tdview.flatten(1, -1)).all()

    def test_neg_dim_lazystack(self):
        td0 = TensorDict(batch_size=(3, 5))
        td1 = TensorDict(batch_size=(4, 5))
        assert lazy_stack([td0, td1], -1).shape == (-1, 5, 2)
        assert lazy_stack([td0, td1], -2).shape == (-1, 2, 5)
        assert lazy_stack([td0, td1], -3).shape == (2, -1, 5)
        with pytest.raises(RuntimeError):
            assert lazy_stack([td0, td1], -4)
        with pytest.raises(RuntimeError):
            assert lazy_stack([td0, td1, TensorDict()])
        with pytest.raises(RuntimeError):
            assert lazy_stack([TensorDict(), td0, td1])

    def test_new_methods(self):
        td = TensorDict(
            dense=torch.randn(1, 10),
            sparse=LazyStackedTensorDict(
                *[TensorDict(smth=torch.ones((1,)), batch_size=1) for _ in range(10)],
                stack_dim=1,
            ),
            batch_size=(1, 10),
        )
        assert isinstance(
            td.new_empty((10, 1), empty_lazy=True)["sparse"], LazyStackedTensorDict
        )
        assert isinstance(
            td.new_empty((1, 10, 1), empty_lazy=True)["sparse"], LazyStackedTensorDict
        )
        assert isinstance(
            td.new_empty((1, 100, 1), empty_lazy=True)["sparse"], LazyStackedTensorDict
        )
        assert td.new_empty((1, 100, 1), empty_lazy=True)["sparse"].is_empty()
        assert not isinstance(td.new_empty((10, 1)), LazyStackedTensorDict)

    def test_lazy_swap_stack_dim(self):
        ltd = LazyStackedTensorDict(
            TensorDict(a=torch.randn(3, 4), batch_size=[3, 4]),
            TensorDict(a=torch.randn(3, 4), batch_size=[3, 4]),
            stack_dim=1,
        )
        assert len(ltd.tensordicts) == 2
        ltd2 = ltd.to_lazystack(0)
        assert_allclose_td(ltd2, ltd)
        assert len(ltd2.tensordicts) == 3
        ltd4 = ltd.to_lazystack(-1)
        assert len(ltd4.tensordicts) == 4
        assert_allclose_td(ltd4, ltd)

    @pytest.mark.parametrize(
        "reduction", ["sum", "nansum", "mean", "nanmean", "std", "var", "prod"]
    )
    def test_reduction_feature_full(self, reduction):
        td = TensorDict.lazy_stack(
            [
                TensorDict.lazy_stack(
                    [
                        TensorDict(
                            a=torch.ones(3, 4),
                            b=torch.zeros(3, 4, 5),
                            batch_size=[3, 4],
                        )
                        for _ in range(2)
                    ],
                    1,
                )
                for _ in range(5)
            ],
            -1,
        )
        assert td.shape == (3, 2, 4, 5)
        tensor = getattr(td, reduction)(dim="feature", reduce=True)
        assert tensor.shape == td.shape

    @set_list_to_stack(True)
    def test_set_list_stack(self):
        td = LazyStackedTensorDict(TensorDict(), TensorDict())
        td["a"] = ["0", "1"]
        assert td[0]["a"] == "0"
        assert td[1]["a"] == "1"
        td["b"] = [torch.ones((2,)) * 2, torch.ones((1,))]
        assert (
            td.get("b", as_padded_tensor=True) == torch.tensor([[2, 2], [1, 0]])
        ).all()
        # note that we start by the outer td then nest the inner one (ie, batch-dim 1 is more out than batch-dim 0)
        td = lazy_stack(
            [lazy_stack([TensorDict() for _ in range(2)]) for _ in range(3)], 1
        )
        shapes = [[1, 2, 3], [4, 5, 6]]
        # note that the order matches the shape [2, 3]
        elt = [[torch.ones(shapes[i][j]) for j in range(3)] for i in range(2)]
        td["a"] = elt
        assert td[0, 0]["a"].shape == (1,)
        assert td[0, 1]["a"].shape == (2,)
        assert td[0, 2]["a"].shape == (3,)
        assert td[1, 0]["a"].shape == (4,)
        assert td[1, 1]["a"].shape == (5,)
        assert td[1, 2]["a"].shape == (6,)

    @pytest.mark.parametrize("batch_size", [(), (2,), (1, 2)])
    @pytest.mark.parametrize("stack_dim", [0, 1, 2])
    def test_setitem_hetero(self, batch_size, stack_dim):
        obs = self.nested_lazy_het_td(batch_size)
        obs1 = obs.clone()
        obs1.apply_(lambda x: x + 1)

        if stack_dim > len(batch_size):
            return

        res1 = self.dense_stack_tds_v1([obs, obs1], stack_dim=stack_dim)
        res2 = self.dense_stack_tds_v2([obs, obs1], stack_dim=stack_dim)

        index = (slice(None),) * stack_dim + (0,)  # get the first in the stack
        assert self.recursively_check_key(res1[index], 0)  # check all 0
        assert self.recursively_check_key(res2[index], 0)  # check all 0
        index = (slice(None),) * stack_dim + (1,)  # get the second in the stack
        assert self.recursively_check_key(res1[index], 1)  # check all 1
        assert self.recursively_check_key(res2[index], 1)  # check all 1

    def test_split_lazy(self):
        td = LazyStackedTensorDict(
            TensorDict({"a": torch.zeros(2, 3), "b": "a string!"}, batch_size=[2]),
            TensorDict(
                {"a": torch.zeros(2, 4), "b": "another string!"}, batch_size=[2]
            ),
            TensorDict(
                {"a": torch.zeros(2, 5), "b": "a third string!"}, batch_size=[2]
            ),
            stack_dim=1,
        )
        split0 = td.split([1, 1], 0)
        split1 = td.split([2, 1], 1)
        split1b = td.split([2, 0, 1], 1)
        assert (torch.cat(split0, dim=0) == td).all()
        assert (torch.cat(split1, dim=1) == td).all()
        assert (torch.cat(split1b, dim=1) == td).all()

    @pytest.mark.parametrize("device", get_available_devices())
    def test_stack(self, device):
        torch.manual_seed(1)
        tds_list = [TensorDict(source={}, batch_size=(4, 5)) for _ in range(3)]
        tds = LazyStackedTensorDict.lazy_stack(tds_list, 0)
        assert tds[0] is tds_list[0]

        td = TensorDict(
            source={"a": torch.randn(4, 5, 3, device=device)}, batch_size=(4, 5)
        )
        td_list = list(td)
        td_reconstruct = stack_td(td_list, 0)
        assert td_reconstruct.batch_size == td.batch_size
        assert (td_reconstruct == td).all()

    def test_stack_apply(self):
        td0 = TensorDict(
            {
                ("a", "b", "c"): torch.ones(3, 4),
                ("a", "b", "d"): torch.ones(3, 4),
                "common": torch.ones(3),
            },
            [3],
        )
        td1 = TensorDict(
            {
                ("a", "b", "c"): torch.ones(3, 5) * 2,
                "common": torch.ones(3) * 2,
            },
            [3],
        )
        td = TensorDict(
            {"parent": LazyStackedTensorDict.lazy_stack([td0, td1], 0)}, [2]
        )
        td2 = td.clone()
        tdapply = td.apply(lambda x, y: x + y, td2)
        assert isinstance(tdapply["parent", "a", "b"], LazyStackedTensorDict)
        assert (tdapply["parent", "a", "b"][0]["c"] == 2).all()
        assert (tdapply["parent", "a", "b"][1]["c"] == 4).all()
        assert (tdapply["parent", "a", "b"][0]["d"] == 2).all()

    @pytest.mark.parametrize("batch_size", [(), (32,), (32, 4)])
    def test_stack_hetero(self, batch_size):
        obs = self.nested_lazy_het_td(batch_size)

        obs2 = obs.clone()
        obs2.apply_(lambda x: x + 1)

        obs_stack = LazyStackedTensorDict.lazy_stack([obs, obs2])
        obs_stack_resolved = self.dense_stack_tds_v2([obs, obs2], stack_dim=0)

        assert isinstance(obs_stack, LazyStackedTensorDict) and obs_stack.stack_dim == 0
        assert isinstance(obs_stack_resolved, TensorDict)

        assert obs_stack.batch_size == (2, *batch_size)
        assert obs_stack_resolved.batch_size == obs_stack.batch_size

        assert obs_stack["lazy"].shape == (2, *batch_size, 3)
        assert obs_stack_resolved["lazy"].batch_size == obs_stack["lazy"].batch_size

        assert obs_stack["lazy"].stack_dim == 0
        assert (
            obs_stack_resolved["lazy"].stack_dim
            == len(obs_stack_resolved["lazy"].batch_size) - 1
        )
        for stack in [obs_stack_resolved, obs_stack]:
            for index in range(2):
                assert (stack[index]["dense"] == index).all()
                assert (stack["dense"][index] == index).all()
                assert (stack["lazy"][index]["shared"] == index).all()
                assert (stack[index]["lazy"]["shared"] == index).all()
                assert (stack["lazy"]["shared"][index] == index).all()
                assert (
                    stack["lazy"][index][..., 0]["individual_0_tensor"] == index
                ).all()
                assert (
                    stack[index]["lazy"][..., 0]["individual_0_tensor"] == index
                ).all()
                assert (
                    stack["lazy"][..., 0]["individual_0_tensor"][index] == index
                ).all()
                assert (
                    stack["lazy"][..., 0][index]["individual_0_tensor"] == index
                ).all()

    def test_stack_keys(self):
        td1 = TensorDict(source={"a": torch.randn(3)}, batch_size=[])
        td2 = TensorDict(
            source={
                "a": torch.randn(3),
                "b": torch.randn(3),
                "c": torch.randn(4),
                "d": torch.randn(5),
            },
            batch_size=[],
        )
        td = LazyStackedTensorDict.maybe_dense_stack([td1, td2], 0)
        assert "a" in td.keys()
        assert "b" not in td.keys()
        assert "b" in td[1].keys()
        td.set("b", torch.randn(2, 10), inplace=False)  # overwrites
        with pytest.raises(KeyError):
            td.set_("c", torch.randn(2, 10))  # overwrites
        td.set_("b", torch.randn(2, 10))  # b has been set before

        td1.set("c", torch.randn(4))
        td[
            "c"
        ]  # we must first query that key for the stacked tensordict to update the list
        assert "c" in td.keys(), list(td.keys())  # now all tds have the key c
        td.get("c")

        td1.set("d", torch.randn(6))
        with pytest.raises(RuntimeError):
            td.get("d")

        td["e"] = torch.randn(2, 4)
        assert "e" in td.keys()  # now all tds have the key c
        td.get("e")

    def test_stack_keys_order(self):
        # The keys follow the insertion order of the first tensordict, as for a
        # dense tensordict, whatever the order in the other tensordicts.
        td0 = TensorDict(x=0.0, R=1.0, n=TensorDict(b=2.0, a=3.0))
        td1 = TensorDict(R=1.0, x=0.0, n=TensorDict(a=3.0, b=2.0))
        td = lazy_stack([td0, td1])
        for args in ((), (False, True), (True,), (True, True)):
            assert list(td.keys(*args)) == list(td0.keys(*args))
            assert [key for key, _ in td.items(*args)] == [
                key for key, _ in td0.items(*args)
            ]
        assert list(td.flatten_keys().keys()) == ["x", "R", "n.b", "n.a"]
        # values() follows the same order, so it can be zipped with keys()
        for key, value in zip(td.keys(True, True), td.values(True, True)):
            assert (value == td.get(key)).all()

    @pytest.mark.parametrize("unsqueeze_dim", [0, 1, -1, -2])
    def test_stack_unsqueeze(self, unsqueeze_dim):
        td = TensorDict({("a", "b"): torch.ones(3, 4, 5)}, [3, 4])
        td_stack = LazyStackedTensorDict.lazy_stack(td.unbind(1), 1)
        td_unsqueeze = td.unsqueeze(unsqueeze_dim)
        td_stack_unsqueeze = td_stack.unsqueeze(unsqueeze_dim)
        assert isinstance(td_stack_unsqueeze, LazyStackedTensorDict)
        for key in td_unsqueeze.keys(True, True):
            assert td_unsqueeze.get(key).shape == td_stack_unsqueeze.get(key).shape

    @pytest.mark.parametrize("stack_dim", [0, 1, -1])
    def test_stack_update_heter_stacked_td(self, stack_dim):
        td1 = TensorDict({"a": torch.randn(3, 4)}, [3])
        td2 = TensorDict({"a": torch.randn(3, 5)}, [3])
        td_a = LazyStackedTensorDict.lazy_stack([td1, td2], stack_dim)
        td_b = td_a.clone()
        td_a.update(td_b)
        with pytest.raises(
            RuntimeError,
            match="Failed to stack tensors within a tensordict",
        ):
            td_a.update(td_b.to_tensordict(retain_none=True))
        td_a.update_(td_b)
        with pytest.raises(
            RuntimeError,
            match="Failed to stack tensors within a tensordict",
        ):
            td_a.update_(td_b.to_tensordict(retain_none=True))

    def test_stack_with_heterogeneous_stacks(self):
        # tests that we can stack several stacks in a dense manner
        def make_tds():
            td0 = TensorDict(
                {
                    "a": torch.zeros(3),
                    "b": torch.zeros(4),
                    "c": {"d": 0, "e": "a string!"},
                    "f": "another string",
                },
                [],
            )
            # we intentionally swap the order of the keys to make sure that the comparison is robust to that
            td1 = TensorDict(
                {
                    "b": torch.zeros(5),
                    "c": {"d": 0, "e": "a string!"},
                    "f": "another string",
                    "a": torch.zeros(3),
                },
                [],
            )
            td_a = TensorDict.maybe_dense_stack([td0, td1])
            td_b = TensorDict.maybe_dense_stack([td0, td1]).clone()
            td = TensorDict.maybe_dense_stack([td_a, td_b])
            return (td, td_a, td_b, td0, td1)

        td, td_a, td_b, td0, td1 = make_tds()
        assert isinstance(td, LazyStackedTensorDict)
        assert isinstance(td.tensordicts[0], TensorDict)
        assert isinstance(td.tensordicts[1], TensorDict)
        # If we remove the "a" key from one of the tds, the resulting element of the stack cannot be dense
        td, td_a, td_b, td0, td1 = make_tds()
        del td_a["a"]
        td = TensorDict.maybe_dense_stack([td_a, td_b])
        assert isinstance(td, LazyStackedTensorDict)
        assert isinstance(td.tensordicts[0], LazyStackedTensorDict)
        assert isinstance(td.tensordicts[1], LazyStackedTensorDict)

        td, td_a, td_b, td0, td1 = make_tds()
        del td0["a"]
        td = TensorDict.maybe_dense_stack([td_a, td_b])
        assert isinstance(td, LazyStackedTensorDict)
        assert isinstance(td.tensordicts[0], LazyStackedTensorDict)
        assert isinstance(td.tensordicts[1], LazyStackedTensorDict)

        # this isn't true if we remove ("c", "d") on one td
        td, td_a, td_b, td0, td1 = make_tds()
        del td0["c", "d"]
        assert isinstance(td, LazyStackedTensorDict)
        assert isinstance(td.tensordicts[0], TensorDict)
        assert isinstance(td.tensordicts[1], TensorDict)

    @pytest.mark.parametrize(
        "stack_order", [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2], [2, 0, 1]]
    )
    def test_stack_with_homogeneous_stack(self, stack_order):
        # tests the condition where all(
        #                 isinstance(_tensordict, LazyStackedTensorDict)
        #                 for _tensordict in list_of_tensordicts
        #             ):
        data = TensorDict(
            {
                ("level0", "level1", "level2", "entry"): torch.arange(60).reshape(
                    3, 4, 5
                )
            },
            [3, 4, 5],
        )
        stacked_data = data.clone()
        stacked_data["level0", "level1"] = LazyStackedTensorDict(
            *stacked_data["level0", "level1"].unbind(stack_order[0]),
            stack_dim=stack_order[0],
        )
        stacked_data["level0"] = LazyStackedTensorDict(
            *stacked_data["level0"].unbind(stack_order[1]), stack_dim=stack_order[1]
        )
        stacked_data = stacked_data.unbind(stack_order[2])
        stacked_data = torch.stack(stacked_data, stack_order[2])
        assert stacked_data.batch_size == data.batch_size
        assert (stacked_data == data).all()

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("stack_dim", [0, 1, 2])
    def test_stacked_indexing(self, device, stack_dim):
        tensordict = TensorDict(
            {"a": torch.randn(3, 4, 5), "b": torch.randn(3, 4, 5)},
            batch_size=[3, 4, 5],
            device=device,
        )

        tds = LazyStackedTensorDict.lazy_stack(
            list(tensordict.unbind(stack_dim)), stack_dim
        )

        for item, expected_shape in (
            ((2, 2), torch.Size([5])),
            ((slice(1, 2), 2), torch.Size([1, 5])),
            ((..., 2), torch.Size([3, 4])),
        ):
            assert tds[item].batch_size == expected_shape
            assert (tds[item].get("a") == tds.get("a")[item]).all()
            assert (tds[item].get("a") == tensordict[item].get("a")).all()

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_stacked_td(self, stack_dim, device):
        tensordicts = [
            TensorDict(
                batch_size=[11, 12],
                source={
                    "key1": torch.randn(11, 12, 5, device=device),
                    "key2": torch.zeros(
                        11, 12, 50, device=device, dtype=torch.bool
                    ).bernoulli_(),
                },
            )
            for _ in range(10)
        ]

        tensordicts0 = tensordicts[0]
        tensordicts1 = tensordicts[1]
        tensordicts2 = tensordicts[2]
        tensordicts3 = tensordicts[3]
        sub_td = LazyStackedTensorDict(*tensordicts, stack_dim=stack_dim)

        std_bis = stack_td(tensordicts, dim=stack_dim, contiguous=False)
        assert (sub_td == std_bis).all()

        item = (*[slice(None) for _ in range(stack_dim)], 0)
        tensordicts0.zero_()
        assert (sub_td[item].get("key1") == sub_td.get("key1")[item]).all()
        assert (
            sub_td.contiguous()[item].get("key1")
            == sub_td.contiguous().get("key1")[item]
        ).all()
        assert (sub_td.contiguous().get("key1")[item] == 0).all()

        item = (*[slice(None) for _ in range(stack_dim)], 1)
        std2 = sub_td[:5]
        tensordicts1.zero_()
        assert (std2[item].get("key1") == std2.get("key1")[item]).all()
        assert (
            std2.contiguous()[item].get("key1") == std2.contiguous().get("key1")[item]
        ).all()
        assert (std2.contiguous().get("key1")[item] == 0).all()

        std3 = sub_td[:5, :, :5]
        tensordicts2.zero_()
        item = (*[slice(None) for _ in range(stack_dim)], 2)
        assert (std3[item].get("key1") == std3.get("key1")[item]).all()
        assert (
            std3.contiguous()[item].get("key1") == std3.contiguous().get("key1")[item]
        ).all()
        assert (std3.contiguous().get("key1")[item] == 0).all()

        std4 = sub_td.select("key1")
        tensordicts3.zero_()
        item = (*[slice(None) for _ in range(stack_dim)], 3)
        assert (std4[item].get("key1") == std4.get("key1")[item]).all()
        assert (
            std4.contiguous()[item].get("key1") == std4.contiguous().get("key1")[item]
        ).all()
        assert (std4.contiguous().get("key1")[item] == 0).all()

        std5 = sub_td.unbind(1)[0]
        assert (std5.contiguous() == sub_td.contiguous().unbind(1)[0]).all()

    @set_list_to_stack(True)
    def test_stacked_td_nested_keys(self):
        td = LazyStackedTensorDict.lazy_stack(
            [
                TensorDict({"a": {"b": {"d": [1]}, "c": [2]}}, []),
                TensorDict({"a": {"b": {"d": [1]}, "d": [2]}}, []),
            ],
            0,
        )
        assert ("a", "b") in td.keys(True)
        assert ("a", "c") not in td.keys(True)
        assert ("a", "b", "d") in td.keys(True)
        td["a", "c"] = [[2], [3]]
        assert ("a", "c") in td.keys(True)

        keys, items = zip(*td.items(True))
        assert ("a", "b") in keys
        assert ("a", "c") in keys
        assert ("a", "d") not in keys

        td["a", "c"] = td["a", "c"] + 1
        assert (td["a", "c"] == torch.tensor([[3], [4]], device=td.device)).all()

    def test_strict_shape(self):
        td0 = TensorDict(batch_size=[3, 4])
        td1 = TensorDict(batch_size=[3, 5])
        with pytest.raises(RuntimeError, match="stacking tensordicts requires"):
            TensorDict.lazy_stack([td0, td1], strict_shape=True)
        with pytest.raises(RuntimeError, match="batch sizes in tensordicts differs"):
            LazyStackedTensorDict(*[td0, td1], strict_shape=True)
        td = LazyStackedTensorDict(td0, td1, stack_dim=1)
        assert td.shape == torch.Size([3, 2, -1])
        assert td[0].shape == torch.Size([2, -1])
        assert td[:, :, 0].shape == torch.Size([3, 2])
        assert td[:, :, :2].shape == torch.Size([3, 2, 2])
        assert td[:, :, -2:].shape == torch.Size([3, 2, 2])
        assert td[:, 0, :].shape == torch.Size([3, 4])

    def test_unbind_lazystack(self):
        td0 = TensorDict(
            {
                "a": {"b": torch.randn(3, 4), "d": torch.randn(3, 4)},
                "c": torch.randn(3, 4),
            },
            [3, 4],
        )
        td = LazyStackedTensorDict.lazy_stack([td0, td0, td0], 1)

        assert all(_td is td0 for _td in td.unbind(1))

    def test_update_with_lazy(self):
        td0 = TensorDict(
            {
                ("a", "b", "c"): torch.ones(3, 4),
                ("a", "b", "d"): torch.ones(3, 4),
                "common": torch.ones(3),
            },
            [3],
        )
        td1 = TensorDict(
            {
                ("a", "b", "c"): torch.ones(3, 5) * 2,
                "common": torch.ones(3) * 2,
            },
            [3],
        )
        td = TensorDict(
            {"parent": LazyStackedTensorDict.lazy_stack([td0, td1], 0)}, [2]
        )

        td_void = TensorDict(
            {
                ("parent", "a", "b", "c"): torch.zeros(2, 3, 4),
                ("parent", "a", "b", "e"): torch.zeros(2, 3, 4),
                ("parent", "a", "b", "d"): torch.zeros(2, 3, 5),
            },
            [2],
        )
        td_void.update(td)
        assert type(td_void.get("parent")) is LazyStackedTensorDict
        assert type(td_void.get(("parent", "a"))) is LazyStackedTensorDict
        assert type(td_void.get(("parent", "a", "b"))) is LazyStackedTensorDict
        assert (td_void.get(("parent", "a", "b"))[0].get("c") == 1).all()
        assert (td_void.get(("parent", "a", "b"))[1].get("c") == 2).all()
        assert (td_void.get(("parent", "a", "b"))[0].get("d") == 1).all()
        assert (td_void.get(("parent", "a", "b"))[1].get("d") == 0).all()  # unaffected
        assert (td_void.get(("parent", "a", "b")).get("e") == 0).all()  # unaffected

    @pytest.mark.parametrize("source_is_lazy", [True, False])
    def test_update_batch_size(self, source_is_lazy):
        td = TensorDict(
            a=torch.zeros(3, 4), b=torch.randn(3, 4), batch_size=[3, 4]
        ).to_lazystack(0)
        td2 = TensorDict(a=torch.ones(2, 4), b=torch.randn(2, 4), batch_size=[2, 4])
        if source_is_lazy:
            td2 = td2.to_lazystack(0)
        td.update(td2, update_batch_size=True)
        assert td.batch_size == td2.batch_size
        assert td.batch_size == (2, 4)
        assert td.batch_size == td2.batch_size

    def test_lazy_mask_nested_stack(self):
        # Boolean-masking a lazy stack whose constituents are themselves lazy
        # stacks: the per-constituent scalar masks must behave like new-axis
        # indices, not integer indices.
        outer = LazyStackedTensorDict(
            *[
                LazyStackedTensorDict(
                    *[
                        TensorDict({"a": torch.full((), float(i * 10 + j))}, [])
                        for j in range(2)
                    ],
                    stack_dim=0,
                )
                for i in range(3)
            ],
            stack_dim=0,
        )
        mask = torch.tensor([True, False, True])
        masked = outer[mask]
        assert masked.batch_size == torch.Size([2, 2])
        assert masked["a"].tolist() == [[0.0, 1.0], [20.0, 21.0]]

    def test_lazy_scalar_bool_index(self):
        lst = LazyStackedTensorDict(
            *[TensorDict({"a": torch.full((), float(i))}, []) for i in range(2)],
            stack_dim=0,
        )
        # bare scalar masks: True adds a size-1 leading dim, False a size-0 one
        assert lst[torch.tensor(True)].batch_size == torch.Size([1, 2])
        assert lst[torch.tensor(False)].batch_size == torch.Size([0, 2])
        assert lst[True].batch_size == torch.Size([1, 2])
        assert lst[False].batch_size == torch.Size([0, 2])
        # scalar True inside a tuple index (matches dense semantics)
        indexed = lst[(torch.tensor(True),)]
        assert indexed.batch_size == torch.Size([1, 2])
        assert indexed["a"].tolist() == [[0.0, 1.0]]
        # scalar False inside a tuple index adds a size-0 dim
        assert lst[(torch.tensor(False),)].batch_size == torch.Size([0, 2])
        assert lst[(0, False)].batch_size == torch.Size([0])
        # None and scalar bools next to each other
        assert lst[(None, True)].batch_size == torch.Size([1, 1, 2])
        assert lst[(True, None, 1)].batch_size == torch.Size([1, 1])
        # scalar False assignment is a no-op
        before = lst["a"].tolist()
        lst[torch.tensor(False)] = TensorDict({"a": torch.full((1, 2), 5.0)}, [1, 2])
        assert lst["a"].tolist() == before

    def test_lazy_empty_selection_batch_size(self):
        lst = LazyStackedTensorDict(
            *[TensorDict({"a": torch.ones(2)}, [2]) for _ in range(3)],
            stack_dim=0,
        )
        # empty slice along the stack dim used to raise "items cannot be empty"
        assert lst[0:0].batch_size == torch.Size([0, 2])
        # all-False mask used to return batch_size [0, 0, 2]
        assert lst[torch.zeros(3, dtype=torch.bool)].batch_size == torch.Size([0, 2])
        # the empty result is dense, and cat with siblings still gives a lazy stack
        recat = torch.cat([lst[0:1], lst[0:0], lst[1:]], 0)
        assert isinstance(recat, LazyStackedTensorDict)
        assert recat.batch_size == torch.Size([3, 2])
        assert (recat == lst).all()
        # zero-size in a non-leading position
        lst_sd1 = LazyStackedTensorDict(
            *[TensorDict({"a": torch.ones(3)}, [3]) for _ in range(2)],
            stack_dim=1,
        )
        empty = lst_sd1[:, torch.zeros(2, dtype=torch.bool)]
        assert empty.batch_size == torch.Size([3, 0])
        # with an Ellipsis before the empty selection
        assert lst_sd1[..., 1:1].batch_size == torch.Size([3, 0])

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize(
        "selection",
        ["slice", "mask", "index tensor", "list", "slice and int", "2-D mask", "masks"],
    )
    def test_lazy_empty_selection_matches_dense(self, stack_dim, selection):
        # An index that keeps no element along the stack dim gives what the
        # dense copy gives, keys and names included
        lazy = LazyStackedTensorDict(
            *[
                TensorDict(
                    {"a": torch.zeros(4, 2), "nested": {"b": torch.zeros(4)}},
                    [4],
                    names=["r"],
                )
                for _ in range(3)
            ],
            stack_dim=stack_dim,
            stack_dim_name="s",
        )
        none3 = torch.zeros(3, dtype=torch.bool)
        none4 = torch.zeros(4, dtype=torch.bool)
        # the index for a stack along dim 0 and along dim 1
        index = {
            "slice": (slice(1, 1), (slice(None), slice(1, 1))),
            "mask": (none3, (slice(None), none3)),
            "index tensor": (
                torch.zeros(0, dtype=torch.long),
                (slice(None), torch.zeros(0, dtype=torch.long)),
            ),
            "list": ([], (slice(None), [])),
            "slice and int": ((slice(1, 1), 0), (0, slice(1, 1))),
            "2-D mask": (
                torch.zeros(3, 4, dtype=torch.bool),
                torch.zeros(4, 3, dtype=torch.bool),
            ),
            "masks": ((none3, none4), (none4, none3)),
        }[selection][stack_dim]
        expected = lazy.to_tensordict()[index]
        result = lazy[index]
        assert type(result) is TensorDict
        assert result.batch_size == expected.batch_size
        assert result.names == expected.names
        assert set(result.keys(True)) == set(expected.keys(True))
        assert (result == expected).all()

    def test_lazy_empty_selection_heterogeneous(self):
        # Members that differ: the result has the keys that all members share,
        # as the dense copy does, with the shapes of the first member
        lazy = LazyStackedTensorDict(
            TensorDict({"a": torch.zeros(4, 1), "b": torch.zeros(4)}, [4]),
            TensorDict({"a": torch.zeros(4, 2), "c": torch.zeros(4)}, [4]),
            stack_dim=0,
        )
        empty = lazy[torch.zeros(2, dtype=torch.bool)]
        assert set(empty.keys()) == {"a"}
        assert empty["a"].shape == torch.Size([0, 4, 1])
        recat = torch.cat([lazy[:1], empty, lazy[1:]], 0)
        assert isinstance(recat, LazyStackedTensorDict)
        assert len(recat.tensordicts) == 2
        assert all(a is b for a, b in zip(recat.tensordicts, lazy.tensordicts))

    def test_lazy_empty_selection_non_tensor(self):
        # A non-tensor stack keeps returning a non-tensor stack without members
        stack = NonTensorStack("a", "b", "c")
        empty = stack[torch.zeros(3, dtype=torch.bool)]
        assert isinstance(empty, NonTensorStack)
        assert empty.batch_size == torch.Size([0])
        assert empty.tolist() == []
        # a mask copies the selection, and the copy keeps the batch size
        stack = NonTensorStack(*[NonTensorStack("a", "b") for _ in range(3)])
        empty = stack[torch.zeros(3, dtype=torch.bool)]
        assert empty.batch_size == torch.Size([0, 2])
        # an N-D index that selects nothing keeps the other dims of the block
        index = torch.zeros(2, 0, dtype=torch.long)
        for idx in ((index,), (index, index), (slice(None), index)):
            expected = torch.zeros(stack.batch_size)[idx].shape
            assert stack[idx].batch_size == expected

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_lazy_empty_selection_non_tensor_entries(self, stack_dim):
        # Members with a non-tensor entry: the entry follows the batch size
        lazy = LazyStackedTensorDict(
            *[
                TensorDict({"a": torch.zeros(4), "b": f"string {i}"}, [4])
                for i in range(3)
            ],
            stack_dim=stack_dim,
        )
        index = (slice(None),) * stack_dim + (torch.zeros(3, dtype=torch.bool),)
        empty = lazy[index]
        assert empty.get("b").batch_size == empty.batch_size
        assert empty.clone().get("b").batch_size == empty.batch_size
        assert empty["b"] == []


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
