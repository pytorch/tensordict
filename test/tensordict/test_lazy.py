# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import functools
import gc
import importlib.util
import io
import math
import os
import pickle
import platform
import re
import sys
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from tensordict import (
    is_leaf_nontensor,
    lazy_stack,
    LazyStackedTensorDict,
    NonTensorData,
    NonTensorStack,
    PersistentTensorDict,
    set_printoptions,
    TensorClass,
    TensorDict,
    TypedTensorDict,
    UnbatchedTensor,
)
from tensordict._indexing import _getitem_batch_size
from tensordict._lazy import _LazyStackedTensorDictKeysView
from tensordict._reductions import _reduce_td
from tensordict._td import _SubTensorDict, _TensorDictKeysView
from tensordict._torch_func import _stack as stack_td
from tensordict.base import (
    _default_is_leaf,
    _NESTED_TENSORS_AS_LISTS,
    TensorDictBase,
)
from tensordict.nn import TensorDictParams
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
        streaming_dataset,
    )
else:
    IS_FB = False
    from _utils_internal import (
        decompose,
        get_available_devices,
        is_npu_available,
        streaming_dataset,
    )


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

        # Load the dataset
        dataset = streaming_dataset(local=tmpdir, remote=None, batch_size=2)
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

    def test_to_dict_state_dict_exclusive_keys_raise(self):
        # to_dict and state_dict would silently drop the keys that are not in
        # every member, so they raise
        match = "cannot complete when there are exclusive keys"
        td = lazy_stack(
            [
                TensorDict(a=torch.zeros(()), b=torch.ones(())),
                TensorDict(a=torch.zeros(()), c=torch.ones(())),
            ]
        )
        nested = lazy_stack(
            [
                TensorDict(a=torch.zeros(()), n=TensorDict(x=torch.zeros(()))),
                TensorDict(a=torch.zeros(()), n=TensorDict(y=torch.ones(()))),
            ]
        )
        for data in (td, nested, TensorDict(s=nested)):
            for call in (data.to_dict, data.state_dict):
                with pytest.raises(RuntimeError, match=match):
                    call()
        # with tolist_first, the members of a nested stack are listed one by
        # one, so only exclusive keys at the top level are lost
        with pytest.raises(RuntimeError, match=match):
            td.to_dict(tolist_first=True)
        result = nested.to_dict(tolist_first=True)
        assert list(result) == ["a", "n"]
        assert (result["a"] == torch.zeros(2)).all()
        assert result["n"] == [{"x": torch.zeros(())}, {"y": torch.ones(())}]
        assert TensorDict(s=nested).to_dict(tolist_first=True) == {
            "s": [
                {"a": torch.zeros(()), "n": {"x": torch.zeros(())}},
                {"a": torch.zeros(()), "n": {"y": torch.ones(())}},
            ]
        }
        same_keys = lazy_stack(
            [TensorDict(a=torch.zeros(())), TensorDict(a=torch.ones(()))]
        )
        assert (same_keys.to_dict()["a"] == torch.tensor([0.0, 1.0])).all()
        assert list(same_keys.state_dict()) == ["a"]

    @set_printoptions(compact=False)
    def test_repr_exact(self):
        td = lazy_stack(
            [
                TensorDict(a=torch.zeros(3), b=torch.zeros(3, 2), batch_size=[3]),
                TensorDict(a=torch.ones(3), batch_size=[3]),
            ],
            1,
        )
        assert repr(td) == (
            "LazyStackedTensorDict(\n"
            "    fields={\n"
            "        a: Tensor(shape=torch.Size([3, 2]), device=cpu, "
            "dtype=torch.float32, is_shared=False)},\n"
            "    exclusive_fields={\n"
            "        0 ->\n"
            "            b: Tensor(shape=torch.Size([3, 2]), device=cpu, "
            "dtype=torch.float32, is_shared=False)},\n"
            "    batch_size=torch.Size([3, 2]),\n"
            "    device=None,\n"
            "    is_shared=False,\n"
            "    stack_dim=1)"
        )

    @set_printoptions(compact=False)
    def test_repr_does_not_stack(self):
        # Printing reads the entries' shapes and dtypes from the members.
        td = lazy_stack(
            [
                lazy_stack(
                    [
                        TensorDict(
                            a=torch.zeros(3, 2),
                            b=torch.zeros(3, i + 1),
                            n=TensorDict(c=torch.zeros(3), batch_size=[3]),
                            batch_size=[3],
                        )
                        for i in range(2)
                    ],
                    1,
                )
                for _ in range(2)
            ]
        )
        stack = torch.stack
        calls = []

        def counting_stack(tensors, *args, **kwargs):
            calls.append(len(tensors))
            return stack(tensors, *args, **kwargs)

        with mock.patch.object(torch, "stack", counting_stack):
            text = repr(td)
        assert calls == []
        assert (
            "a: Tensor(shape=torch.Size([2, 3, 2, 2]), device=cpu, "
            "dtype=torch.float32, is_shared=False)" in text
        )
        assert "b: Tensor(shape=torch.Size([2, 3, 2, -1])" in text
        assert "c: Tensor(shape=torch.Size([2, 3, 2])," in text

    @set_printoptions(compact=False)
    def test_repr_promoted_dtype(self):
        # The dtype is the one that td[key] has: torch.stack promotes it.
        td = lazy_stack(
            [
                TensorDict(a=torch.zeros(3)),
                TensorDict(a=torch.zeros(3, dtype=torch.float64)),
            ]
        )
        assert td["a"].dtype == torch.float64
        assert (
            "a: Tensor(shape=torch.Size([2, 3]), device=cpu, "
            "dtype=torch.float64, is_shared=False)" in repr(td)
        )

    def test_to_keeps_nested_tensorclass(self):
        class MC(TensorClass):
            x: torch.Tensor

        td = lazy_stack(
            [
                TensorDict(tc=MC(x=torch.zeros(3), batch_size=[3]), batch_size=[3])
                for _ in range(2)
            ]
        )
        out = td.to(torch.float64)
        assert type(out) is LazyStackedTensorDict
        assert type(out.tensordicts[0]["tc"]) is MC
        assert type(out["tc"]) is MC
        assert out["tc"].x.dtype == torch.float64
        # to_tensordict converts every tensorclass to a TensorDict
        dense = td.to_tensordict()
        assert type(dense) is TensorDict
        assert type(dense["tc"]) is TensorDict

    def test_all_keys(self):
        td = TensorDict({"a": torch.zeros(1)}, [])
        td2 = TensorDict({"a": torch.zeros(2)}, [])
        stack = LazyStackedTensorDict.lazy_stack([td, td2])
        assert set(stack.keys(True, True)) == {"a"}

    @pytest.mark.parametrize("include_nested", [False, True])
    @pytest.mark.parametrize("leaves_only", [False, True])
    def test_keys_len(self, include_nested, leaves_only):
        td = TensorDict(
            a=torch.zeros(3),
            nested=TensorDict(b=torch.zeros(3), c=torch.zeros(3), batch_size=[3]),
            batch_size=[3],
        )
        stack = lazy_stack([td[0], td[1]])
        keys = stack.keys(include_nested, leaves_only)
        assert len(keys) == len(list(keys))
        assert len(keys) == len(td.keys(include_nested, leaves_only))

    @pytest.mark.parametrize("include_nested", [False, True])
    @pytest.mark.parametrize("leaves_only", [False, True])
    def test_keys_heterogeneous_nested(self, include_nested, leaves_only):
        # A sub-tensordict that only one member has is not a key of the stack
        td0 = TensorDict(
            a=torch.zeros(3),
            nested0=TensorDict(b=torch.zeros(3, 1), batch_size=[3]),
            batch_size=[3],
        )
        td1 = TensorDict(
            a=torch.zeros(3),
            nested1=TensorDict(c=torch.zeros(3, 2), batch_size=[3]),
            batch_size=[3],
        )
        stack = lazy_stack([td0, td1])
        keys = stack.keys(include_nested, leaves_only)
        assert list(keys) == ["a"]
        assert len(keys) == 1
        assert len(list(stack.values(include_nested, leaves_only))) == 1
        assert [key for key, _ in stack.items(include_nested, leaves_only)] == ["a"]

    @staticmethod
    def _keys_nested_members(case):
        def member(i):
            return TensorDict(
                a=torch.zeros(3, 1 + i),
                nested=TensorDict(
                    b=torch.zeros(3),
                    sub=TensorDict(c=torch.zeros(3, 2 + i), batch_size=[3]),
                    batch_size=[3],
                ),
                batch_size=[3],
            )

        class MC(TensorClass):
            x: torch.Tensor

        tds = [member(0), member(1), member(2)]
        if case == "exclusive":
            tds[0]["nested", "sub", "d"] = torch.zeros(3)
            tds[1]["nested", "other"] = TensorDict(e=torch.zeros(3), batch_size=[3])
        elif case == "non_tensor":
            for i, td in enumerate(tds):
                td["nested", "s"] = "a string" if i else "another string"
        elif case == "tensorclass":
            for td in tds:
                td["nested", "tc"] = MC(x=torch.zeros(3), batch_size=[3])
        elif case == "tensor_and_tensordict":
            tds[1]["nested", "sub"] = torch.zeros(3)
        elif case == "batch_dims_differ":
            tds[1]["nested", "sub"] = TensorDict(c=torch.zeros(3, 2), batch_size=[3, 2])
        elif case == "devices_differ":
            tds[1]["nested", "sub"] = TensorDict(
                c=torch.zeros(3), batch_size=[3], device="cpu"
            )
        elif case == "lazy_stack_entry":
            for td in tds:
                td["nested", "ls"] = lazy_stack([member(0), member(1)], 1)
        elif case == "locked_entries":
            for td in tds:
                td["nested"].lock_()
        elif case == "locked":
            for td in tds:
                td.lock_()
        elif case == "list_entries":
            # Not stored by the public API
            for td in tds:
                td["nested"]._tensordict["lst"] = [torch.zeros(3)]
        return tds

    @pytest.mark.parametrize(
        "case",
        [
            "tensordicts",
            "exclusive",
            "non_tensor",
            "tensorclass",
            "tensor_and_tensordict",
            "batch_dims_differ",
            "devices_differ",
            "lazy_stack_entry",
            "locked_entries",
            "locked",
            "list_entries",
        ],
    )
    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("leaves_only", [False, True])
    @pytest.mark.parametrize(
        "is_leaf",
        [None, is_leaf_nontensor, lambda cls: True],
        ids=["default", "nontensor", "all"],
    )
    def test_keys_nested_members(self, case, stack_dim, leaves_only, is_leaf):
        # The nested keys view of a lazy stack of TensorDict instances reads the
        # entries of the members. It must list what _TensorDictKeysView lists
        # by building the lazy stack of every sub-tensordict, raise the same
        # errors and call is_leaf with the same classes.
        stack = lazy_stack(self._keys_nested_members(case), stack_dim)

        def listed(view_type):
            calls = []

            def recorded_is_leaf(cls):
                calls.append(cls)
                return (is_leaf or _default_is_leaf)(cls)

            view = view_type(stack, True, leaves_only, recorded_is_leaf)
            try:
                return list(view), len(view), calls
            except Exception as err:
                return type(err), str(err), calls

        assert listed(_LazyStackedTensorDictKeysView) == listed(_TensorDictKeysView)

    @pytest.mark.parametrize(
        "change", ["delete_first", "delete_second", "replace", "add"]
    )
    def test_keys_nested_members_changed_while_listed(self, change):
        # The keys view lists the entries that the members have when it gets
        # to them, as _TensorDictKeysView does.
        def listed(view_type):
            tds = self._keys_nested_members("non_tensor")
            stack = lazy_stack(tds)
            keys = []
            try:
                for key in view_type(stack, True, False):
                    keys.append(key)
                    if key != ("nested", "b"):
                        continue
                    if change == "delete_first":
                        del tds[0]["nested", "s"]
                    elif change == "delete_second":
                        del tds[1]["nested", "sub"]
                    for td in tds:
                        if change == "replace":
                            td["nested", "sub"] = TensorDict(
                                z=torch.zeros(3), batch_size=[3]
                            )
                        elif change == "add":
                            td["nested", "new"] = torch.zeros(3)
            except Exception as err:
                return keys, type(err), str(err)
            return keys

        assert listed(_LazyStackedTensorDictKeysView) == listed(_TensorDictKeysView)

    def test_keys_nested_members_relocked(self):
        # A lazy stack whose members are locked caches its sub-tensordicts
        # when _TensorDictKeysView lists them. After the members are changed
        # while unlocked, the keys view lists the cached ones, as
        # _TensorDictKeysView does.
        tds = self._keys_nested_members("tensordicts")
        stack = lazy_stack([td.lock_() for td in tds])
        list(_TensorDictKeysView(stack, True, False))
        for td in tds:
            td.unlock_()
            td["nested", "new"] = torch.zeros(3)
            td.lock_()
        expected = list(_TensorDictKeysView(stack, True, False))
        assert list(stack.keys(True)) == expected

    @pytest.mark.parametrize("include_nested", [False, True])
    def test_non_tensor_items_listed_keys(self, include_nested):
        # non_tensor_items() gives the non-tensor entries of the keys that the
        # stack lists, which none of these stacks has (#2705)
        stacks = [
            NonTensorStack("a", "b"),
            lazy_stack(
                [TensorDict(s=NonTensorData("x")), TensorDict(t=NonTensorData("y"))]
            ),
            lazy_stack(
                [
                    TensorDict(s=NonTensorData("x"), a=torch.zeros(())),
                    TensorDict(a=torch.ones(())),
                ]
            ),
            lazy_stack(
                [
                    TensorDict(n=TensorDict(s=NonTensorData("x"))),
                    TensorDict(n=TensorDict(t=NonTensorData("y"))),
                ]
            ),
        ]
        for stack in stacks:
            assert stack.non_tensor_items(include_nested=include_nested) == ()
        # A key that every member holds is stacked
        stack = lazy_stack(
            [
                TensorDict(s=NonTensorData("x"), n=TensorDict(u=NonTensorData("u"))),
                TensorDict(s=NonTensorData("y"), n=TensorDict(u=NonTensorData("v"))),
            ]
        )
        items = stack.non_tensor_items(include_nested=include_nested)
        expected = {"s": ["x", "y"]}
        if include_nested:
            expected["n", "u"] = ["u", "v"]
        assert [key for key, _ in items] == list(expected)
        for (_, val), data in zip(items, expected.values()):
            assert isinstance(val, NonTensorStack)
            assert val.tolist() == data

    @pytest.mark.parametrize("filter_empty", [True, None])
    @pytest.mark.parametrize("num_threads", [0, 2])
    def test_apply_member_without_result(self, filter_empty, num_threads):
        # A member for which fn gives no result becomes an empty member, as
        # with filter_empty=False, when other members give a result (#2136).
        a0 = torch.zeros(3, requires_grad=True)
        a1 = torch.zeros(3, requires_grad=True)
        a0.sum().backward()
        stack = lazy_stack(
            [
                TensorDict(w=a0, batch_size=[3], names=["n"]),
                TensorDict(w=a1, batch_size=[3], names=["n"]),
            ]
        )

        def check(out):
            assert isinstance(out, LazyStackedTensorDict)
            assert out.batch_size == stack.batch_size
            assert out.names == [None, "n"]
            assert (out.tensordicts[0]["w"] == 1).all()
            assert out.tensordicts[1].is_empty()
            assert out.tensordicts[1].batch_size == torch.Size([3])

        # apply() takes no num_threads: _fast_apply runs the threaded path
        if num_threads:
            apply = functools.partial(stack._fast_apply, num_threads=num_threads)
        else:
            apply = stack.apply
        check(apply(lambda x: x.grad, filter_empty=filter_empty))
        check(stack.grad)
        check(TensorDict(s=stack).grad["s"])
        # All members without a result still give None
        assert apply(lambda x: None, filter_empty=True) is None

    def test_apply_returns_self_and_kwargs(self):
        tds = [
            TensorDict(
                a=torch.arange(3.0) + i, n={"b": torch.zeros(3, 2)}, batch_size=[3]
            )
            for i in range(4)
        ]
        stack = lazy_stack(tds)
        dense = stack.to_tensordict()
        # fn returning None: apply_ returns self
        for td in (stack, dense):
            assert td.apply_(lambda x: None) is td
        # batch_size, names and device are accepted and ignored, as by TensorDict
        for kwargs in (
            {"batch_size": [4]},
            {"names": ["x", "y"]},
            {"device": "cpu"},
        ):
            for td in (stack, dense):
                assert td.apply_(lambda x: x + 1, **kwargs) is td
        assert stack.names == dense.names == [None, None]
        torch.testing.assert_close(stack.to_tensordict(), dense)
        for td in (stack, dense):
            assert td.apply_(lambda x: x + 1, propagate_lock=True) is td
        torch.testing.assert_close(stack.to_tensordict(), dense)
        # a fn that changes the shape of the leaves leaves the members unchanged
        before = stack.to_tensordict()
        assert stack.apply_(lambda x: x[..., :1]) is stack
        torch.testing.assert_close(stack.to_tensordict(), before)
        with pytest.raises(RuntimeError, match="batch dimension mismatch"):
            dense.apply_(lambda x: x[..., :1])

    @pytest.mark.parametrize("call", ["apply", "named_apply", "threads"])
    @pytest.mark.parametrize("shape", [(2, 2), (2, 3)])
    def test_apply_out_other_stack_dim(self, shape, call):
        # #2982: out stacked along another dim than self gets the results in
        # the matching entries, and is returned
        dense = TensorDict(
            x=torch.arange(math.prod(shape)).reshape(shape),
            n=TensorDict(y=torch.ones(*shape, 2), batch_size=shape),
            batch_size=shape,
        )
        expected = dense.apply(lambda v: v + 1)

        def run(source, out, fn=lambda v: v + 1, *others, **kwargs):
            if call == "apply":
                return source.apply(fn, *others, out=out, **kwargs)
            if call == "named_apply":
                return source.named_apply(
                    lambda key, *values: fn(*values), *others, out=out, **kwargs
                )
            return source._fast_apply(fn, *others, out=out, num_threads=2, **kwargs)

        source = lazy_stack(dense.unbind(0), 0)
        for out_dim in (0, 1):
            out = lazy_stack(torch.zeros_like(dense).unbind(out_dim), out_dim)
            members = list(out.tensordicts)
            result = run(source, out)
            assert out.stack_dim == out_dim
            assert out.tensordicts == members
            torch.testing.assert_close(out.to_tensordict(), expected)
            torch.testing.assert_close(result.to_tensordict(), expected)
            if out_dim == 0:
                # the same stack dim: a new stack of the members of out
                assert result is not out
                assert result.tensordicts == members
            else:
                assert result is out
        # out=self
        source = lazy_stack(dense.clone().unbind(0), 0)
        assert run(source, source).tensordicts == source.tensordicts
        torch.testing.assert_close(source.to_tensordict(), expected)
        if call == "threads":
            return
        source = lazy_stack(dense.unbind(0), 0)
        out = lazy_stack(torch.zeros_like(dense).unbind(1), 1)
        # others, names and batch_size
        assert run(source, out, lambda v, w: v + w, source) is out
        torch.testing.assert_close(out.to_tensordict(), dense * 2)
        assert run(source, out, names=["a", "b"]) is out
        assert out.names == ["a", "b"]
        torch.testing.assert_close(out.to_tensordict(), expected.refine_names("a", "b"))
        out = lazy_stack(torch.zeros_like(dense).unbind(1), 1)
        assert run(source, out, batch_size=shape) is out
        torch.testing.assert_close(out.to_tensordict(), expected)
        # fn returns None everywhere: out is returned unchanged
        out = lazy_stack(torch.zeros_like(dense).unbind(1), 1)
        for filter_empty in (None, True):
            assert run(source, out, lambda v: None, filter_empty=filter_empty) is out
            assert (out.to_tensordict() == 0).all()
        if call == "apply":
            assert source.apply(lambda v: v + 1, out=out, call_on_nested=True) is out
            torch.testing.assert_close(out.to_tensordict(), expected)
        # inplace=True writes into self, as without out
        out = lazy_stack(torch.zeros_like(dense).unbind(1), 1)
        source = lazy_stack(dense.clone().unbind(0), 0)
        assert source.apply(lambda v: v.add_(1), out=out, inplace=True) is source
        torch.testing.assert_close(source.to_tensordict(), expected)
        assert (out.to_tensordict() == 0).all()
        # a dense out still raises
        with pytest.raises(ValueError, match="out must be a LazyStackedTensorDict"):
            run(source, dense.clone())

    @pytest.mark.parametrize(
        "kind", ["nested_source", "nested_out", "tensorclass", "typed", "h5"]
    )
    def test_apply_out_other_stack_dim_members(self, kind, tmp_path):
        # #2982: with members of other types, nested lazy stacks included
        if kind == "h5" and not _has_h5py:
            pytest.skip("h5py not found.")

        class MC(TensorClass):
            x: torch.Tensor

        class Typed(TypedTensorDict):
            x: torch.Tensor

        dense = TensorDict(x=torch.arange(24).reshape(2, 3, 4), batch_size=[2, 3, 4])
        if kind == "tensorclass":
            dense = TensorDict(c=MC.from_tensordict(dense), batch_size=[2, 3, 4])
        expected = dense.apply(lambda v: v + 1)

        def members(td, dim):
            members = td.unbind(dim)
            if kind == "typed":
                return [Typed.from_tensordict(member) for member in members]
            if kind == "h5":
                return [
                    PersistentTensorDict.from_dict(
                        member, filename=tmp_path / f"{dim}-{i}.h5"
                    )
                    for i, member in enumerate(members)
                ]
            return members

        def nested(td, dim, inner_dim):
            return lazy_stack(
                [lazy_stack(m.unbind(inner_dim), inner_dim) for m in td.unbind(dim)],
                dim,
            )

        source = lazy_stack(members(dense, 0), 0)
        out = lazy_stack(members(torch.zeros_like(dense), 2), 2)
        if kind == "nested_source":
            source = nested(dense, 0, 0)
        elif kind == "nested_out":
            out = nested(torch.zeros_like(dense), 2, 0)
        types = [type(member) for member in out.tensordicts]
        assert source.apply(lambda v: v + 1, out=out) is out
        assert [type(member) for member in out.tensordicts] == types
        key = ("c", "x") if kind == "tensorclass" else "x"
        torch.testing.assert_close(out[key], expected[key])
        if kind == "tensorclass":
            assert isinstance(out["c"], MC)

    @pytest.mark.parametrize("stack_dim,vmap_dim", [(1, 0), (2, 0), (0, 1), (2, 1)])
    def test_apply_out_other_stack_dim_vmap(self, stack_dim, vmap_dim):
        # #2982: under vmap, with out stacked along another dim than self
        dense = TensorDict(x=torch.arange(12).reshape(2, 2, 3), batch_size=[2, 2, 3])
        source = lazy_stack(dense.unbind(stack_dim), stack_dim)
        for out_dim in {0, 1, 2} - {stack_dim, vmap_dim}:
            out = lazy_stack(torch.zeros_like(dense).unbind(out_dim), out_dim)
            result = torch.vmap(
                lambda td, out: td.apply(lambda v: v + 1, out=out),
                in_dims=vmap_dim,
                out_dims=vmap_dim,
            )(source, out)
            torch.testing.assert_close(result.to_tensordict(), dense + 1)

    @pytest.mark.parametrize("heterogeneous", [False, True])
    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_detach_members(self, heterogeneous, stack_dim):
        w = torch.ones(3, requires_grad=True)
        tds = [
            TensorDict(
                a=w * 2,
                n={"b": w[:, None].expand(3, 2 + heterogeneous * i) * 3},
                batch_size=[3],
            )
            for i in range(4)
        ]
        stack = lazy_stack(tds, stack_dim)
        assert stack.detach_() is stack
        for td in tds:
            for value in td.values(True, True):
                assert not value.requires_grad
        assert not stack["a"].requires_grad

    @pytest.mark.parametrize("stack_dim", range(4))
    def test_transpose_matches_dense(self, stack_dim):
        dense = TensorDict(a=torch.randn(2, 3, 4, 5, 6), batch_size=[2, 3, 4, 5])
        stack = lazy_stack(list(dense.unbind(stack_dim)), stack_dim)
        for dim0 in range(4):
            for dim1 in range(dim0 + 1, 4):
                torch.testing.assert_close(
                    stack.transpose(dim0, dim1).to_tensordict(),
                    dense.transpose(dim0, dim1),
                )

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("mask_dims", [2, 1, 0])
    def test_masked_fill_mask_shapes(self, stack_dim, mask_dims):
        # a mask aligned with the leading batch dims, as in TensorDict
        td = TensorDict(
            a=torch.zeros(4, 3, 5), n={"b": torch.zeros(4, 3)}, batch_size=[4, 3]
        )
        stack = lazy_stack(list(td.unbind(stack_dim)), stack_dim)
        mask = (torch.arange(12).view(4, 3) % 3 == 0)[
            (Ellipsis,) + (0,) * (2 - mask_dims)
        ]
        assert stack.masked_fill_(mask, 1.0) is stack
        torch.testing.assert_close(stack.to_tensordict(), td.masked_fill_(mask, 1.0))

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("index", [(), slice(1, None)])
    def test_masked_fill_params(self, stack_dim, index):
        # Out-of-place masked_fill leaves the parameters untouched, as on a
        # single TensorDictParams (#2779)
        params = [
            TensorDictParams(
                TensorDict(a=torch.ones(3, 2), n={"b": torch.ones(3)}, batch_size=[3])
            )
            for _ in range(3)
        ]
        stack = lazy_stack(params, stack_dim)[index]
        mask = torch.arange(stack.numel()).view(stack.shape) % 2 == 0
        result = stack.masked_fill(mask, 0.0)
        assert isinstance(result, LazyStackedTensorDict)
        for p in params:
            assert all((value == 1).all() for value in p.values(True, True))
        expected = lazy_stack(
            [
                td.masked_fill(m, 0.0)
                for td, m in zip(stack.tensordicts, mask.unbind(stack_dim))
            ],
            stack_dim,
        )
        assert_allclose_td(result, expected)
        assert result["a"].requires_grad

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

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_arithmetic_ops_broadcast_tensor(self, stack_dim):
        # A tensor with the batch dims is split along the stack dim, as the
        # members' leaves lack it (#2365)
        dense = TensorDict(
            a=torch.arange(24.0).view(2, 3, 4),
            b=torch.arange(6.0).view(2, 3),
            batch_size=[2, 3],
        )
        for other in (torch.arange(6.0).view(2, 3), torch.arange(3.0)):
            lazy = lazy_stack(list(dense.clone().unbind(stack_dim)), stack_dim)
            for op in ("add", "mul", "sub", "rsub"):
                result = getattr(lazy, op)(other)
                assert isinstance(result, LazyStackedTensorDict)
                assert result.stack_dim == stack_dim
                assert (result == getattr(dense, op)(other)).all(), op
            weight = torch.full((3,), 0.5)
            assert (lazy.lerp(other, weight) == dense.lerp(other, weight)).all()
            assert lazy.add_(other) is lazy
            assert (lazy == dense + other).all()

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("depth", [1, 2])
    def test_arithmetic_ops_broadcast_tensor_nested(self, stack_dim, depth):
        # A lazy stack nested in a tensordict also gives each member its slice
        # of a batch-shaped tensor. Here the leaves of a member have the shape
        # of the batch, so a misaligned tensor would give wrong values (#3005)
        def make(lazy):
            n = TensorDict(a=torch.arange(8.0).view(2, 2, 2), batch_size=[2, 2])
            if lazy:
                n = lazy_stack(list(n.unbind(stack_dim)), stack_dim)
            td = TensorDict(n=n, b=torch.arange(4.0).view(2, 2), batch_size=[2, 2])
            if depth == 2:
                td = TensorDict(o=td, batch_size=[2, 2])
            return td

        def check(result, expected):
            assert set(result.keys(True, True)) == set(expected.keys(True, True))
            for key, value in expected.items(True, True):
                torch.testing.assert_close(result[key], value)

        lazy_key = ("n",) if depth == 1 else ("o", "n")
        for other in (torch.arange(4.0).view(2, 2) * 10, torch.tensor([1.0, 2.0])):
            for op in ("add", "sub", "mul", "maximum", "clamp_min"):
                result = getattr(make(True), op)(other)
                assert isinstance(result.get(lazy_key), LazyStackedTensorDict)
                check(result, getattr(make(False), op)(other))
            check(make(True) + other, make(False) + other)
            check(
                make(True).lerp(make(True) + 1, other),
                make(False).lerp(make(False) + 1, other),
            )
            check(
                make(True).addcmul(other, make(True) + 1),
                make(False).addcmul(other, make(False) + 1),
            )
            td = make(True)
            assert td.mul_(other) is td
            check(td, make(False) * other)

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_pointwise_keyword_tensor(self, stack_dim):
        # A tensor passed by keyword broadcasts with each leaf from the right,
        # as on the dense tensordict: each member leaf gets its part of it,
        # which depends on the rank of the leaf (#3042)
        torch.manual_seed(0)
        members = [
            TensorDict(x=torch.randn(4, 3), y=torch.randn(4), batch_size=[4])
            for _ in range(3)
        ]
        lazy = lazy_stack(members, stack_dim)  # batch [3, 4] or [4, 3]
        dense = lazy.to_tensordict()

        def check(result, expected):
            assert set(result.keys(True, True)) == set(expected.keys(True, True))
            for key, value in expected.items(True, True):
                torch.testing.assert_close(result[key], value)

        # per-feature bounds for x, and bounds of the batch shape for y
        for lo in (torch.full((3,), -0.5), -torch.rand(lazy.batch_size)):
            if lo.shape != (3,):
                lazy_y = lazy.exclude("x")
                dense_y = dense.exclude("x")
            else:
                lazy_y = lazy.exclude("y")
                dense_y = dense.exclude("y")
            check(
                lazy_y.clamp(min=lo, max=lo + 0.5),
                dense_y.clamp(min=lo, max=lo + 0.5),
            )
            minus_one = torch.tensor(-1.0)
            check(lazy_y.clamp(minus_one, max=lo), dense_y.clamp(minus_one, max=lo))
        # the leaves of a member differ in rank: x gets the whole bound and y
        # its slice along the stack dim
        if stack_dim == 1:
            lo = torch.rand(3) - 0.5
            check(lazy.clamp(min=lo, max=lo + 0.5), dense.clamp(min=lo, max=lo + 0.5))
            check(lazy.mod(other=lo + 2), dense.mod(other=lo + 2))
            # a tensor bound next to a number raises, as on the dense
            # tensordict, though the slice of y's bound is 0-d
            with pytest.raises(TypeError):
                dense.exclude("x").clamp(min=lo, max=0.75)
            with pytest.raises(TypeError):
                lazy.exclude("x").clamp(min=lo, max=0.75)

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("kind", ["tensorclass", "typed"])
    def test_arithmetic_ops_broadcast_tensor_nested_wrapped(self, stack_dim, kind):
        # A tensorclass or a TypedTensorDict that wraps a lazy stack, nested in
        # a tensordict, also gives each member its slice of a batch-shaped
        # tensor (#3005)
        class MC(TensorClass):
            a: torch.Tensor

        class Typed(TypedTensorDict):
            a: torch.Tensor

        def make(lazy):
            n = TensorDict(a=torch.arange(8.0).view(2, 2, 2), batch_size=[2, 2])
            if lazy:
                n = lazy_stack(list(n.unbind(stack_dim)), stack_dim)
            n = (MC if kind == "tensorclass" else Typed).from_tensordict(n)
            return TensorDict(n=n, batch_size=[2, 2])

        def check(result, expected):
            assert type(result["n"]) is type(expected["n"])
            torch.testing.assert_close(result["n", "a"], expected["n", "a"])

        other = torch.arange(4.0).view(2, 2) * 10
        for op in ("add", "mul", "maximum"):
            check(getattr(make(True), op)(other), getattr(make(False), op)(other))
        check(
            make(True).lerp(make(True) + 1, other),
            make(False).lerp(make(False) + 1, other),
        )
        check(
            make(True).addcmul(other, make(True) + 1),
            make(False).addcmul(other, make(False) + 1),
        )
        td = make(True)
        assert td.add_(other) is td
        check(td, make(False) + other)

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize(
        "op,args",
        [
            ("add", ()),
            ("sub", ()),
            ("rsub", ()),
            ("mul", ()),
            ("div", ()),
            ("pow", ()),
            ("mod", ()),
            ("maximum", ()),
            ("minimum", ()),
            ("clamp_max", ()),
            ("clamp_min", ()),
            ("lerp", (0.3,)),
            ("addcmul", ("other",)),
            ("addcdiv", ("other",)),
            ("add_", ()),
            ("sub_", ()),
            ("mul_", ()),
            ("div_", ()),
            ("pow_", ()),
            ("maximum_", ()),
            ("minimum_", ()),
            ("clamp_max_", ()),
            ("clamp_min_", ()),
            ("lerp_", (0.3,)),
            ("addcmul_", ("other",)),
            ("addcdiv_", ("other",)),
        ],
    )
    @pytest.mark.parametrize("lazy_first", [True, False])
    def test_arithmetic_with_dense(self, stack_dim, op, args, lazy_first):
        # The pointwise ops pair the leaves of a lazy stack and of a dense
        # tensordict by key, in both orders (#1932).
        def make_dense():
            return TensorDict(
                a=torch.rand(2, 3) + 1,
                nested=TensorDict(b=torch.rand(2, 3, 4) + 1, batch_size=[2, 3]),
                batch_size=[2, 3],
            )

        stack = lazy_stack(list(make_dense().unbind(stack_dim)), stack_dim)
        dense = make_dense()
        self_td, other = (stack, dense) if lazy_first else (dense, stack)
        expected = getattr(self_td.to_tensordict(), op)(
            other.to_tensordict(),
            *[other.to_tensordict() if arg == "other" else arg for arg in args],
        )
        args = [other if arg == "other" else arg for arg in args]
        members = stack.tensordicts if lazy_first else [dense]
        data_ptrs = [td["nested", "b"].data_ptr() for td in members]

        out = getattr(self_td, op)(other, *args)

        assert isinstance(out, type(self_td))
        # pow rounds differently on the strided members of a lazy stack
        for key, value in expected.items(True, True):
            torch.testing.assert_close(out[key], value)
        if op.endswith("_"):
            assert out is self_td
            assert [td["nested", "b"].data_ptr() for td in members] == data_ptrs

    def test_arithmetic_with_dense_mismatched_keys(self):
        # As with two dense tensordicts, a key that only one operand has raises
        dense = TensorDict(a=torch.ones(2, 3), z=torch.ones(2, 3), batch_size=[2, 3])
        stack = lazy_stack(list(dense.exclude("z").unbind(0)))
        with pytest.raises(KeyError, match="keys in tensordicts mismatch"):
            stack + dense
        with pytest.raises(KeyError, match="keys in tensordicts mismatch"):
            dense + stack

    @pytest.mark.parametrize("lazy_first", [True, False])
    def test_arithmetic_with_dense_keeps_lock(self, lazy_first):
        # As with two dense tensordicts, a locked left operand gives a locked result
        dense = TensorDict(a=torch.ones(2, 3), batch_size=[2, 3])
        stack = lazy_stack(list(dense.clone().unbind(0)))
        self_td, other = (stack, dense) if lazy_first else (dense, stack)
        self_td.lock_()
        assert (self_td + other).is_locked
        assert self_td.lerp(other, 0.5).is_locked
        assert not (other + self_td).is_locked

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("lazy_first", [True, False])
    def test_and_with_dense(self, stack_dim, lazy_first):
        # & pairs the leaves by key, as | and ^ do (#1932)
        def make_dense():
            return TensorDict(
                a=torch.rand(2, 3) > 0.5,
                nested=TensorDict(b=torch.rand(2, 3, 4) > 0.5, batch_size=[2, 3]),
                batch_size=[2, 3],
            )

        stack = lazy_stack(list(make_dense().unbind(stack_dim)), stack_dim)
        dense = make_dense()
        self_td, other = (stack, dense) if lazy_first else (dense, stack)
        expected = self_td.to_tensordict() & other.to_tensordict()

        out = self_td & other

        assert isinstance(out, type(self_td))
        assert_allclose_td(out.to_tensordict(), expected)

    @pytest.mark.parametrize("op", ["add", "add_", "lerp_", "__and__"])
    @pytest.mark.parametrize("lazy_first", [True, False])
    @pytest.mark.parametrize("wrap", ["both", "other"])
    def test_arithmetic_with_dense_tensorclass(self, op, lazy_first, wrap):
        # A tensorclass operand counts as the tensordict it wraps (#1932)
        class MC(TensorClass):
            a: torch.Tensor

        dtype = torch.bool if op == "__and__" else torch.float
        dense = TensorDict(a=torch.rand(2, 3) + 1, batch_size=[2, 3]).to(dtype)
        stack = lazy_stack(
            list(TensorDict(a=torch.rand(2, 3) + 1, batch_size=[2, 3]).unbind(0))
        ).to(dtype)
        self_td, other = (stack, dense) if lazy_first else (dense, stack)
        args = (0.3,) if op == "lerp_" else ()
        expected = getattr(self_td.to_tensordict(), op)(other.to_tensordict(), *args)
        if wrap == "both":
            self_td = MC._from_tensordict(self_td)
        other = MC._from_tensordict(other)

        out = getattr(self_td, op)(other, *args)

        if op.endswith("_") and not op.endswith("__"):
            assert out is self_td
        if wrap == "both" or op == "__and__":
            # as with |, & with a tensorclass operand gives a tensorclass
            assert isinstance(out, MC)
            out = out._tensordict
        torch.testing.assert_close(out["a"], expected["a"])

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize(
        "op,args", [("add", ()), ("mul", ()), ("lerp", (0.3,)), ("add_", ())]
    )
    @pytest.mark.parametrize("lazy_first", [True, False])
    def test_arithmetic_with_nested_dense(self, stack_dim, op, args, lazy_first):
        # A lazy stack nested in a dense tensordict pairs with the dense
        # tensordict under the same key (#1932)
        def make_dense():
            return TensorDict(
                a=torch.rand(2, 3) + 1,
                nested=TensorDict(b=torch.rand(2, 3, 4) + 1, batch_size=[2, 3]),
                batch_size=[2, 3],
            )

        stack = lazy_stack(list(make_dense().unbind(stack_dim)), stack_dim)
        dense = make_dense()
        first, second = (stack, dense) if lazy_first else (dense, stack)
        self_td = TensorDict(x=torch.ones(2, 3), n=first, batch_size=[2, 3])
        other = TensorDict(x=torch.ones(2, 3), n=second, batch_size=[2, 3])
        expected = getattr(self_td.to_tensordict(), op)(other.to_tensordict(), *args)

        out = getattr(self_td, op)(other, *args)

        if op.endswith("_"):
            assert out is self_td
        assert_allclose_td(out.to_tensordict(), expected)

    def test_arithmetic_with_nested_dense_mismatched_keys(self):
        # A key that only one operand has still raises, before any write
        dense = TensorDict(a=torch.ones(2, 3), z=torch.ones(2, 3), batch_size=[2, 3])
        stack = lazy_stack(list(dense.exclude("z").unbind(0)))
        self_td = TensorDict(n=stack, batch_size=[2, 3])
        other = TensorDict(n=dense, batch_size=[2, 3])
        with pytest.raises(KeyError):
            self_td + other
        with pytest.raises(KeyError):
            other + self_td
        with pytest.raises(KeyError):
            self_td.add_(other)
        assert (self_td["n", "a"] == 1).all()

    @pytest.mark.parametrize("op", ["add", "add_", "mul_"])
    def test_arithmetic_with_nested_dense_member_key(self, op):
        # A key that only one member of the lazy stack has raises with that
        # key, and an in-place op writes nothing first
        dense = TensorDict(a=torch.ones(2, 3), batch_size=[2, 3])
        members = list(dense.clone().unbind(0))
        members[1].set("z", torch.ones(3))
        self_td = TensorDict(n=lazy_stack(members), batch_size=[2, 3])
        other = TensorDict(n=dense * 2, batch_size=[2, 3])
        with pytest.raises(KeyError, match="z"):
            getattr(self_td, op)(other)
        assert (self_td["n", "a"] == 1).all()

    def test_auto_device(self):
        # The device of a lazy stack is read from its members, so auto_device_
        # sets it on them, also when the lazy stack is nested in a tensordict.
        cpu = torch.device("cpu")
        stack = lazy_stack([TensorDict(a=torch.zeros(2)), TensorDict(a=torch.zeros(2))])
        assert stack.auto_device_() is stack
        assert stack.device == cpu
        assert [td.device for td in stack.tensordicts] == [cpu, cpu]

        td = TensorDict(
            x=torch.zeros(2),
            stack=lazy_stack(
                [TensorDict(a=torch.zeros(2)), TensorDict(a=torch.zeros(2))]
            ),
            batch_size=[2],
        )
        assert td.auto_device_().device == cpu
        assert td["stack"].device == cpu

        # Members whose leaves have different shapes, which cannot be stacked
        stack = lazy_stack([TensorDict(a=torch.zeros(2)), TensorDict(a=torch.zeros(3))])
        assert stack.auto_device_().device == cpu

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

    @pytest.mark.parametrize("num_inputs", [2, 3])
    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("out_stack_dim", [0, 1])
    @pytest.mark.parametrize("cat_dim", [0, 1])
    @pytest.mark.parametrize("batch_dims", [2, 3])
    def test_cat_lazy_stack_out(
        self, num_inputs, stack_dim, out_stack_dim, cat_dim, batch_dims
    ):
        # The members of out have the same shapes, so they could be stacked densely
        inputs = []
        for i in range(num_inputs):
            batch_size = [2, 3, 2][:batch_dims]
            batch_size[cat_dim] = i + 1
            td = TensorDict(
                {
                    "a": torch.randn(*batch_size, 4),
                    ("nested", "b"): torch.randn(batch_size),
                },
                batch_size,
            )
            inputs.append(td.to_lazystack(stack_dim))
        expected = torch.cat([td.to_tensordict() for td in inputs], cat_dim)
        out = lazy_stack(
            [td.clone() for td in torch.zeros_like(expected).unbind(out_stack_dim)],
            out_stack_dim,
        )
        data_ptrs = [td["a"].data_ptr() for td in out.tensordicts]

        assert torch.cat(inputs, cat_dim, out=out) is out
        assert [td["a"].data_ptr() for td in out.tensordicts] == data_ptrs
        assert assert_allclose_td(out, expected)

    @pytest.mark.parametrize("empty_lazy", [False, True])
    @pytest.mark.parametrize("empty_first", [False, True])
    def test_cat_lazy_stack_out_zero_size(self, empty_lazy, empty_first):
        # An input with size 0 along the stack dim fills no member of out
        td = TensorDict(a=torch.arange(6.0).view(2, 3), batch_size=[2, 3])
        empty = TensorDict(a=torch.zeros(0, 3), batch_size=[0, 3])
        if empty_lazy:
            empty = LazyStackedTensorDict(stack_dim=0, batch_size=[3])
        inputs = [td.to_lazystack(0), empty]
        if empty_first:
            inputs = inputs[::-1]
        out = lazy_stack(
            [TensorDict(a=torch.zeros(3), batch_size=[3]) for _ in range(2)]
        )
        assert torch.cat(inputs, 0, out=out) is out
        assert (out["a"] == td["a"]).all()

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

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize(
        "batch_size,other_batch_size",
        [
            ((2, 3), (4, 2, 3)),
            ((2, 3), (1, 3)),
            ((2, 3), (2, 1)),
            ((2, 1), (2, 4)),
            ((1, 3), (4, 1, 3)),
        ],
    )
    def test_comparison_broadcast(self, stack_dim, batch_size, other_batch_size):
        # Comparisons broadcast the two batch sizes, as for dense tensordicts
        torch.manual_seed(0)
        dense = TensorDict(a=torch.randint(3, (*batch_size, 2)), batch_size=batch_size)
        lazy = lazy_stack(list(dense.unbind(stack_dim)), stack_dim)
        other = TensorDict(
            a=torch.randint(3, (*other_batch_size, 2)), batch_size=other_batch_size
        )
        for op in ("__eq__", "__ne__", "__lt__", "__ge__"):
            result = getattr(lazy, op)(other)
            expected = getattr(dense, op)(other)
            assert result.batch_size == expected.batch_size
            assert torch.equal(result["a"], expected["a"])

    def test_comparison_broadcast_ragged(self):
        lazy = lazy_stack(
            [
                TensorDict(a=torch.zeros(3, 1), batch_size=[3]),
                TensorDict(a=torch.ones(3, 2), batch_size=[3]),
            ]
        )
        others = [
            lazy_stack(
                [
                    TensorDict(a=torch.zeros(4, 3, 1), batch_size=[4, 3]),
                    TensorDict(a=torch.zeros(4, 3, 2), batch_size=[4, 3]),
                ],
                1,
            ),
            lazy_stack(
                [
                    TensorDict(a=torch.zeros(1, 1), batch_size=[1]),
                    TensorDict(a=torch.zeros(1, 2), batch_size=[1]),
                ]
            ),
        ]
        for other in others:
            result = lazy == other
            assert isinstance(result, LazyStackedTensorDict)
            assert result.batch_size == torch.broadcast_shapes(
                lazy.batch_size, other.batch_size
            )
            # the first member holds zeros, the second ones
            assert result[..., 0, :]["a"].all()
            assert not result[..., 1, :]["a"].any()

    @staticmethod
    def _compare_stack(stack_dim, heterogeneous):
        torch.manual_seed(0)
        return lazy_stack(
            [
                TensorDict(
                    a=torch.randint(3, (4, 2)),
                    n=TensorDict(
                        b=torch.randint(3, (4, i + 1) if heterogeneous else (4,)),
                        batch_size=[4],
                    ),
                    batch_size=[4],
                )
                for i in range(3)
            ],
            stack_dim,
        )

    @pytest.mark.parametrize("heterogeneous", [False, True])
    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize(
        "op",
        [
            "__eq__",
            "__ne__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
            "__or__",
            "__xor__",
        ],
    )
    def test_comparison_ops_lazy_result(self, op, stack_dim, heterogeneous):
        # The comparisons run on each member and stack the results lazily
        lazy = self._compare_stack(stack_dim, heterogeneous)
        lazy.lock_()
        for other in (1, lazy.clone()):
            result = getattr(lazy, op)(other)
            assert isinstance(result, LazyStackedTensorDict)
            assert result.stack_dim == stack_dim
            assert result.batch_size == lazy.batch_size
            assert not result.is_locked
            for member, td in zip(result.tensordicts, lazy.tensordicts):
                expected = getattr(td, op)(
                    other if isinstance(other, int) else td.clone()
                )
                assert (member == expected).all()
            if not heterogeneous:
                dense_other = other if isinstance(other, int) else other.to_tensordict()
                expected = getattr(lazy.to_tensordict(), op)(dense_other)
                assert (result.to_tensordict() == expected).all()

    @pytest.mark.parametrize("operand", [None, "abc"])
    def test_comparison_unsupported_operand(self, operand):
        # == and != return a bool, as on TensorDict; the other ops raise
        lazy = self._compare_stack(0, False)
        assert (lazy == operand) is False
        assert (lazy != operand) is True
        for op in ("__lt__", "__le__", "__gt__", "__ge__", "__or__", "__xor__"):
            with pytest.raises(ValueError, match=f"Incompatible value .* for op {op}"):
                getattr(lazy, op)(operand)

    def test_comparison_prefix_tensor_stack_dim1(self):
        # A tensor of the batch shape up to the stack dim goes to each member
        lazy = self._compare_stack(1, False)
        assert lazy.batch_size == (4, 3)
        other = torch.ones(4, dtype=torch.long)
        result = lazy == other
        assert isinstance(result, LazyStackedTensorDict)
        assert result.batch_size == (4, 3)
        for member, td in zip(result.tensordicts, lazy.tensordicts):
            assert (member == (td == other)).all()

    @pytest.mark.parametrize("op", ["__eq__", "__lt__", "__or__"])
    @pytest.mark.parametrize("shape", ["batch", "ends in 1", "starts with 1"])
    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_comparison_full_batch_tensor(self, stack_dim, shape, op):
        # A tensor that broadcasts to the batch shape is split along the stack
        # dim, as TensorDict pairs it with its rows
        lazy = self._compare_stack(stack_dim, False)
        other_shape = {
            "batch": lazy.shape,
            "ends in 1": (*lazy.shape[:-1], 1),
            "starts with 1": (1, *lazy.shape[1:]),
        }[shape]
        other = torch.arange(math.prod(other_shape)).view(other_shape) % 3
        result = getattr(lazy, op)(other)
        assert isinstance(result, LazyStackedTensorDict)
        assert result.stack_dim == stack_dim
        expected = getattr(lazy.to_tensordict(), op)(other)
        assert result.batch_size == expected.batch_size == lazy.batch_size
        assert (result.to_tensordict() == expected).all()

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_comparison_dict_operand(self, stack_dim):
        # A dict is read with the batch size of the stack
        lazy = self._compare_stack(stack_dim, False)
        other = lazy.to_tensordict().apply(torch.ones_like).to_dict()
        result = lazy == other
        assert isinstance(result, LazyStackedTensorDict)
        assert result.stack_dim == stack_dim
        expected = lazy.to_tensordict() == other
        assert (result.to_tensordict() == expected).all()

    def test_comparison_broadcast_error(self):
        lazy = self._compare_stack(1, False)
        other = TensorDict(a=torch.ones(5, 2, dtype=torch.long), batch_size=[5])
        with pytest.raises(
            RuntimeError,
            match=r"Could not compare tensordicts with shapes torch.Size\(\[4, 3\]\) "
            r"and torch.Size\(\[5\]\)",
        ):
            _ = lazy == other

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

    def test_batch_size_equal_list(self):
        lazy = lazy_stack(
            [TensorDict(a=torch.zeros(3), batch_size=[3]) for _ in range(2)]
        )
        sub = _SubTensorDict(
            TensorDict(a=torch.zeros(4, 3), batch_size=[4, 3]), (slice(0, 2),)
        )
        nontensor = lazy_stack(
            [
                lazy_stack([NonTensorData("a"), NonTensorData("b"), NonTensorData("c")])
                for _ in range(2)
            ]
        )
        for td in (lazy, sub, nontensor):
            td.batch_size = torch.Size([2, 3])
            td.batch_size = [2, 3]
            td.batch_size = (2, 3)
            for new_batch_size in ([2], 3, None, [2.0]):
                with pytest.raises(RuntimeError, match="Modifying the batch size"):
                    td.batch_size = new_batch_size
            assert td.batch_size == torch.Size([2, 3])

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

    def test_create_empty_state(self):
        # as on an empty TensorDict
        empty = LazyStackedTensorDict(stack_dim=0, batch_size=[3], device="cpu")
        assert empty._is_shared is False
        assert empty._is_memmap is False
        assert empty.is_shared() is False
        assert empty.is_memmap() is False
        assert empty.clear_device_() is empty
        assert empty.device is None

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("names", [None, ["a"]])
    def test_create_empty_names(self, stack_dim, names):
        # A stack without members keeps the names it was built with, and the
        # stack dim has no name.
        td = LazyStackedTensorDict(stack_dim=stack_dim, batch_size=[3], names=names)
        expected = [None] if names is None else list(names)
        expected.insert(stack_dim, None)
        batch_size = [3]
        batch_size.insert(stack_dim, 0)
        assert td.names == expected
        for out in (td.to_tensordict(), td.contiguous(), td.clone()):
            assert out.batch_size == torch.Size(batch_size)
            assert out.names == expected
        refined = ["s", "a"] if stack_dim == 0 else ["a", "s"]
        assert td.refine_names(*refined) is td
        assert td.names == refined
        assert td.to_tensordict().names == refined
        td.names = None
        assert td.names == [None, None]

    @pytest.mark.parametrize("recurse", [False, True])
    def test_lazy_empty_keeps_member_types(self, recurse):
        # empty() gives a lazy stack of the empty members, each of its own type
        inner = lazy_stack(
            [TensorDict(a=torch.zeros(()), sub=TensorDict()) for _ in range(3)]
        )
        dense = TensorDict(
            a=torch.zeros(3), sub=TensorDict(batch_size=[3]), batch_size=[3]
        )
        lazy = lazy_stack([dense, inner], 1, stack_dim_name="s")
        empty = lazy.empty(recurse)
        assert type(empty) is LazyStackedTensorDict
        assert empty.stack_dim == 1
        assert empty.batch_size == torch.Size([3, 2])
        assert empty.names == [None, "s"]
        assert [type(td) for td in empty.tensordicts] == [
            TensorDict,
            LazyStackedTensorDict,
        ]
        assert empty.tensordicts[1].stack_dim == 0
        assert all(
            td is not member for td, member in zip(empty.tensordicts, lazy.tensordicts)
        )
        assert list(empty.keys(True)) == (["sub"] if recurse else [])

    def test_empty_recurse_leaves_no_reference_cycle(self):
        # empty(recurse=True) frees what it builds by reference counting: a
        # cycle per call would leave garbage for the collector, and the
        # collections it adds can run at any point of the caller's code
        tds = [
            TensorDict(
                a=torch.zeros(2),
                n=TensorDict(b=torch.zeros(2), batch_size=[2]),
                batch_size=[2],
            ),
            TensorDict(
                n=lazy_stack(
                    [
                        TensorDict(a=torch.zeros(2), batch_size=[2]),
                        TensorDict(a=torch.zeros(2), e=torch.ones(2), batch_size=[2]),
                    ]
                ),
                batch_size=[2],
            ),
        ]
        for td in tds:
            td.empty(recurse=True)  # warm up caches
        gc.collect()
        gc.disable()
        try:
            for td in tds:
                for _ in range(10):
                    td.empty(recurse=True)
            assert gc.collect() == 0
        finally:
            gc.enable()

    def test_lazy_empty_recurse_drops_member_leaves(self):
        # empty(recurse=True) drops the tensors that only some members of a
        # nested lazy stack hold, as on a lazy stack at the root, and keeps
        # the non-tensor entries, as a dense tensordict does (#2840)
        class MC(TensorClass):
            n: TensorDict

        class Typed(TypedTensorDict):
            n: TensorDictBase

        lazy = lazy_stack(
            [
                TensorDict(a=torch.zeros(2), batch_size=[2]),
                TensorDict(
                    a=torch.zeros(2),
                    e=torch.ones(2),
                    s=TensorDict(x=torch.ones(2, 3), batch_size=[2, 3]),
                    f=NonTensorData("f", batch_size=[2]),
                    batch_size=[2],
                ),
            ],
            1,
            stack_dim_name="m",
        )
        expected = [[], ["f", "s"]]
        assert [sorted(td.keys()) for td in lazy.empty(True).tensordicts] == expected
        stacks = lazy_stack([lazy, lazy.clone()])
        for td, path in (
            (TensorDict(n=lazy, batch_size=[2]), ("n",)),
            (TensorDict(t=TensorDict(n=lazy, batch_size=[2])), ("t", "n")),
            (MC(n=lazy, batch_size=[2]), ("n",)),
            (Typed(n=lazy, batch_size=[2]), ("n",)),
            (TensorDict(n=stacks, batch_size=[2]), ("n",)),
        ):
            for empty in (
                td.empty(recurse=True),
                td.empty(recurse=True, batch_size=td.batch_size),
            ):
                assert type(empty) is type(td)
                nested = empty.get(path)
                assert type(nested) is LazyStackedTensorDict
                assert nested.batch_size == td.get(path).batch_size
                members = nested.tensordicts
                if type(members[0]) is LazyStackedTensorDict:
                    # the stack of stacks
                    assert nested.stack_dim == 0
                    members = [td for stack in members for td in stack.tensordicts]
                else:
                    assert nested.stack_dim == 1
                    assert nested.names == [None, "m"]
                assert [sorted(td.keys()) for td in members] == expected * (
                    len(members) // 2
                )
                assert members[1]["s"].batch_size == torch.Size([2, 3])
                assert members[1]["s"].is_empty()
                assert members[1]["f"] == "f"
                assert not list(empty.keys(True, True))

    def test_lazy_empty_batch_size_list_vs_size(self):
        # the same batch size as a list, a tuple or a torch.Size gives a lazy stack
        lazy = lazy_stack([TensorDict(a=torch.zeros(3), batch_size=[3])] * 2, 1)
        for recurse in (False, True):
            for batch_size in ([3, 2], torch.Size([3, 2]), (3, 2)):
                empty = lazy.empty(recurse, batch_size=batch_size)
                assert type(empty) is LazyStackedTensorDict
                assert empty.batch_size == torch.Size([3, 2])

    def test_lazy_empty_new_stack_dim_size(self):
        # a new size of the stack dim, or fewer dims, gives a TensorDict of the
        # requested size, as a new size of the leading dims does
        lazy = lazy_stack([TensorDict(a=torch.zeros(3), batch_size=[3])] * 2)
        for recurse in (False, True):
            for batch_size in ([5, 3], torch.Size([5, 3]), [], [3, 3]):
                empty = lazy.empty(recurse, batch_size=batch_size)
                assert type(empty) is TensorDict
                assert empty.batch_size == torch.Size(batch_size)
        nested = lazy_stack(
            [
                TensorDict(
                    b=TensorDict(a=torch.zeros(3, 2), batch_size=[3, 2]), batch_size=[3]
                )
            ]
            * 2,
            1,
        )
        # as on the dense copy, a sub-tensordict must fit the new batch size
        for td in (nested, nested.to_tensordict()):
            with pytest.raises(RuntimeError, match="cannot give the batch size"):
                td.empty(True, batch_size=[3, 4])
        # the leading dims and the stack dim kept: a lazy stack
        empty = nested.empty(True, batch_size=[3, 2])
        assert type(empty) is LazyStackedTensorDict
        assert empty["b"].batch_size == torch.Size([3, 2, 2])
        # along dim 0, the members still take the batch size without the stack
        # dim when a sub-tensordict does not fit it, with a FutureWarning
        member = TensorDict(
            a=torch.zeros(3),
            s=TensorDict(x=torch.zeros(3, 4), batch_size=[3, 4]),
            batch_size=[3],
        )
        heterogeneous = lazy_stack(
            [TensorDict(s=TensorDict(x=torch.zeros(n), batch_size=[n])) for n in (3, 4)]
        )
        for td, batch_size, expected in (
            (lazy_stack([member] * 2), [5, 3], [2, 3]),
            (lazy_stack([member] * 2), [5], [2]),
            (heterogeneous, [5], [2]),
            # along dim 1 too, for a batch size that is not a list and keeps
            # the dims before the stack dim
            (lazy_stack([member] * 2, 1), torch.Size([3, 5]), [3, 2]),
            (lazy_stack([member] * 2, 1), (3, 5), [3, 2]),
        ):
            with pytest.warns(
                FutureWarning,
                match=r"cannot give that batch size to the entry 's'.*"
                r"Starting with TensorDict 0\.17, the default will change to "
                r"raising a RuntimeError",
            ) as record:
                empty = td.empty(True, batch_size=batch_size)
            assert record[0].filename == __file__
            assert type(empty) is LazyStackedTensorDict
            assert empty.batch_size == torch.Size(expected)
            assert list(empty.keys()) == ["s"]
            assert empty["s"].batch_size[: len(expected)] == torch.Size(expected)
        # a list, or new dims before the stack dim, raised a TypeError before
        for batch_size in ([3, 5], torch.Size([4, 2])):
            with pytest.raises(RuntimeError, match="cannot give the batch size"):
                lazy_stack([member] * 2, 1).empty(True, batch_size=batch_size)
        # where the members raise, as before 0.15, the call does not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            with pytest.raises(RuntimeError, match="cannot give the batch size"):
                lazy_stack([member] * 2).empty(True, batch_size=[5, 7])

    def test_lazy_empty_names_other_length(self):
        # where the members took the batch size without the stack dim before
        # 0.15, names of another length than the new batch size keep that
        # result, with the FutureWarning of empty() (#2711), instead of a
        # TensorDict with invalid names
        member = TensorDict(a=torch.zeros(3), batch_size=[3])
        wide = TensorDict(a=torch.zeros(2, 3), batch_size=[2, 3])
        for td, recurse, batch_size, names, expected, expected_names in (
            (lazy_stack([member] * 2), False, [], ["a"], [2], ["a"]),
            (lazy_stack([member] * 2), True, [], ["a"], [2], ["a"]),
            (
                lazy_stack([member] * 2, 1),
                False,
                torch.Size([3, 5]),
                ["a"],
                [3, 2],
                ["a", None],
            ),
            # the members drop names with as many None entries as their batch
            # dims, with the warning of the names setter (#2600)
            (lazy_stack([wide] * 2), True, [5], [None, "x"], [2], [None]),
            (
                lazy_stack([wide] * 2),
                True,
                [5, 3],
                [None, None, "x"],
                [2, 3],
                [None, None],
            ),
        ):
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                empty = td.empty(recurse, batch_size=batch_size, names=names)
            record = [w for w in record if str(w.message).startswith("empty(")]
            assert len(record) == 1
            assert record[0].category is FutureWarning
            assert re.search(
                r"returns a tensordict with the batch size \[.*\] and the names "
                r".*Starting with TensorDict 0\.17, the default will change to "
                r"returning the batch size \[.*\] and applying the names as the "
                r"names setter does",
                str(record[0].message),
            )
            assert record[0].filename == __file__
            assert type(empty) is LazyStackedTensorDict
            assert empty.batch_size == torch.Size(expected)
            assert empty.names == expected_names
        # names that fit, and names=None, give the TensorDict
        lazy = lazy_stack([member] * 2)
        lazy.names = ["p", "q"]
        for recurse in (False, True):
            empty = lazy.empty(recurse, batch_size=[5, 3], names=["a", "b"])
            assert type(empty) is TensorDict and empty.names == ["a", "b"]
            empty = lazy.empty(recurse, batch_size=[5, 3])
            assert type(empty) is TensorDict and empty.names == ["p", "q"]
            empty = lazy.empty(recurse, batch_size=[])
            assert type(empty) is TensorDict and empty.batch_size == torch.Size([])

    def test_lazy_stack_onto_non_stack_dim(self):
        # torch.stack(..., out=lazy) along another dim than the stack dim
        # writes into the members, which keep their storage
        tds = [TensorDict(a=torch.zeros(3), batch_size=[3]) for _ in range(2)]
        ptrs = [td["a"].data_ptr() for td in tds]
        lazy = lazy_stack(tds)
        items = [
            TensorDict(a=torch.full((2,), float(i)), batch_size=[2]) for i in range(3)
        ]
        assert torch.stack(items, 1, out=lazy) is lazy
        for td, ptr in zip(tds, ptrs):
            assert td["a"].data_ptr() == ptr
            assert (td["a"] == torch.arange(3.0)).all()

    def test_create_names_with_members_warns(self):
        # With members, the constructor ignores names until 0.17 (#2559).
        tds = [TensorDict(batch_size=[3]), TensorDict(batch_size=[3])]
        with pytest.warns(
            FutureWarning,
            match=r"ignores the names when tensordicts are given.*"
            r"Starting with TensorDict 0\.17, the default will change",
        ) as record:
            td = LazyStackedTensorDict(*tds, names=["s", None])
        assert record[0].filename == __file__
        assert td.names == [None, None]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            td = LazyStackedTensorDict(*tds, stack_dim_name="s")
            assert td.names == ["s", None]
            td = LazyStackedTensorDict(batch_size=[3], names=["a"])
            assert td.names == [None, "a"]

    @pytest.mark.parametrize("key", ["a", ("n", "x"), "n"])
    @pytest.mark.parametrize("op", ["del", "pop"])
    def test_del_repeated_member(self, key, op):
        # A member that appears twice in the stack loses the entry once.
        def make():
            return TensorDict(
                a=torch.zeros(3),
                b=torch.zeros(3),
                n=TensorDict(x=torch.zeros(3), y=torch.zeros(3), batch_size=[3]),
                batch_size=[3],
            )

        def remove(td):
            if op == "del":
                del td[key]
            else:
                td.pop(key)

        t, u = make(), make()
        td = lazy_stack([t, t, u])
        remove(td)
        for member in (td, t, u):
            assert key not in member.keys(True)
        assert "b" in td.keys()

        inner = lazy_stack([make(), make()])
        td = lazy_stack([inner, inner, lazy_stack([make(), make()])])
        remove(td)
        for member in (td, inner, *inner.tensordicts):
            assert key not in member.keys(True)

        with pytest.raises(KeyError):
            del td[key]

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
        # The names of nested lazy stacks are kept.
        inner = LazyStackedTensorDict.lazy_stack(
            [unnamed, unnamed.clone()], stack_dim_name="s"
        )
        td = lazy_stack([inner, inner.clone()])
        assert td.densify().names == [None, "s", None]

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_expand_minus_one(self, stack_dim):
        # A -1 keeps the size of its dim, as in torch.Tensor.expand
        td = lazy_stack(
            [TensorDict(a=torch.arange(3), batch_size=[3]) for _ in range(2)], stack_dim
        )
        batch = torch.zeros(td.batch_size)
        for shape in [(4, -1, -1), (4, *td.batch_size[:1], -1), (-1, -1)]:
            expected = batch.expand(shape).shape
            assert td.expand(*shape).batch_size == expected
            assert td.broadcast_to(shape).batch_size == expected
            assert td.expand(*shape)["a"].shape == expected
        single = lazy_stack([TensorDict(a=torch.arange(3), batch_size=[3])])
        assert single.expand(4, -1, 3).batch_size == (4, 1, 3)
        with pytest.raises(RuntimeError, match="leading, non-existing dimension"):
            td.expand(-1, *td.batch_size)
        # the -1 of a ragged dim stays
        ragged = lazy_stack(
            [TensorDict(a=torch.zeros(n, 4), batch_size=[n]) for n in (2, 3)]
        )
        assert ragged.expand(3, -1, -1).batch_size == (3, 2, -1)
        ragged.update_at_(TensorDict({}, []), 0)
        assert ragged.batch_size == (2, -1)

    @pytest.mark.parametrize("auto_batch_size", [False, True])
    @pytest.mark.parametrize("stack_dim", [0, 1, -1])
    @pytest.mark.parametrize("as_list", [False, True])
    def test_from_dict_names(self, auto_batch_size, stack_dim, as_list):
        # names includes the stack dimension: names[stack_dim] names the stack
        # and the other entries name the member dimensions
        members = [{"a": torch.zeros(3)}, {"a": torch.ones(3)}]
        if not as_list:
            members = {str(i): member for i, member in enumerate(members)}
        names = ["x", "s"] if stack_dim else ["s", "x"]
        td = LazyStackedTensorDict.from_dict(
            members,
            auto_batch_size=auto_batch_size,
            batch_size=[3],
            stack_dim=stack_dim,
            names=names,
        )
        assert td.batch_size == torch.Size([3, 2] if stack_dim else [2, 3])
        assert td.names == names
        assert all(member.names == ["x"] for member in td.tensordicts)

    @pytest.mark.parametrize("batch_size", [None, [3]])
    @pytest.mark.parametrize("stack_dim", [-1, -2])
    @pytest.mark.parametrize("names", [False, True])
    @pytest.mark.parametrize("as_list", [False, True])
    def test_from_dict_negative_stack_dim(self, batch_size, stack_dim, names, as_list):
        # a negative stack_dim counts from the end of the stack's dims, as in
        # LazyStackedTensorDict(*members, stack_dim=stack_dim)
        members = [{"a": torch.zeros(3, 4)}, {"a": torch.ones(3, 4)}]
        if not as_list:
            members = {str(i): member for i, member in enumerate(members)}
        kwargs = {"batch_size": batch_size}
        if batch_size is None:
            kwargs["auto_batch_size"] = True
            kwargs["batch_dims"] = 1
        if names:
            kwargs["names"] = ["x", "s"] if stack_dim == -1 else ["s", "x"]
        with warnings.catch_warnings():
            # without a batch_size, names warns about the 0.17 change
            warnings.simplefilter("ignore", FutureWarning)
            td = LazyStackedTensorDict.from_dict(members, stack_dim=stack_dim, **kwargs)
        expected = LazyStackedTensorDict(
            TensorDict(a=torch.zeros(3, 4), batch_size=[3]),
            TensorDict(a=torch.ones(3, 4), batch_size=[3]),
            stack_dim=stack_dim,
        )
        assert td.stack_dim == expected.stack_dim
        assert td.batch_size == expected.batch_size
        assert td["a"].shape == expected["a"].shape
        assert td.to_tensordict()["a"].shape == expected["a"].shape
        if names:
            assert td.names[td.stack_dim] == "s"

    @pytest.mark.parametrize("auto_batch_size", [False, True])
    def test_from_dict_member_names(self, auto_batch_size):
        # names with one entry per member dimension keeps its current meaning:
        # the members get all of it and names[stack_dim] names the stack
        members = [{"a": torch.zeros(3, 4)}, {"a": torch.ones(3, 4)}]
        td = LazyStackedTensorDict.from_dict(
            members,
            auto_batch_size=auto_batch_size,
            batch_size=[3, 4],
            names=[None, None],
        )
        assert td.names == [None, None, None]
        td = LazyStackedTensorDict.from_dict(
            members,
            auto_batch_size=auto_batch_size,
            batch_size=[3, 4],
            stack_dim=1,
            names=["s", None],
        )
        assert td.batch_size == torch.Size([3, 2, 4])
        assert td.names == ["s", None, None]

    @pytest.mark.parametrize(
        "batch_size,stack_dim,names,stack_names",
        [
            ([3], 0, ["x"], ["x", "x"]),
            ([3], 0, ["x", "x"], ["x", "x"]),
            ([3, 4], 0, ["s", None], ["s", "s", None]),
            ([3, 4], 1, [None, "s"], [None, "s", "s"]),
        ],
    )
    def test_from_dict_names_non_unique(
        self, batch_size, stack_dim, names, stack_names
    ):
        # names that would name two dimensions of the stack alike are rejected
        # as TensorDict rejects them, instead of building an invalid stack
        members = [{"a": torch.zeros(3, 4)}, {"a": torch.ones(3, 4)}]
        with pytest.raises(ValueError) as excinfo:
            LazyStackedTensorDict.from_dict(
                members, batch_size=batch_size, stack_dim=stack_dim, names=names
            )
        assert str(excinfo.value) == (
            f"Some dimension names are non-unique: {stack_names}."
        )

    def test_from_dict_names_tensordict_members(self):
        # TensorDict members keep their own names and batch size: the check
        # looks at the names they have, not at the names that were asked for
        td = TensorDict(a=torch.zeros(3, 4), batch_size=[3], names=["t"])
        stack = LazyStackedTensorDict.from_list(
            [td, td.clone()], batch_size=[3], names=["x"]
        )
        assert stack.names == ["x", "t"]
        assert stack.to_tensordict().names == ["x", "t"]
        stack = LazyStackedTensorDict.from_list([], batch_size=[3], names=["x"])
        assert stack.names == ["x"]
        td = TensorDict(a=torch.zeros(3, 4), batch_size=[3], names=["x"])
        with pytest.raises(ValueError, match="names are non-unique: \\['x', 'x'\\]"):
            LazyStackedTensorDict.from_list([td, td.clone()], names=["x"])

    @pytest.mark.parametrize("from_list", [False, True])
    def test_from_dict_names_without_batch_size(self, from_list):
        # without a batch_size the members get no names today; the calls that
        # 0.17 changes warn and keep their result: a FutureWarning where the
        # names will be given to the members, a DeprecationWarning where they
        # do not match the member batch dimensions and the call will raise
        def build(members, **kwargs):
            if from_list:
                return LazyStackedTensorDict.from_list(members, **kwargs)
            return LazyStackedTensorDict.from_dict(members, **kwargs)

        members = [{"a": torch.zeros(3)}, {"a": torch.ones(3)}]
        scalars = [{"a": torch.zeros(())}, {"a": torch.ones(())}]
        mixed = [{"a": torch.zeros(3)}, {"a": torch.ones(3, 4)}]
        future = (FutureWarning, "Starting with TensorDict 0.17")
        removal = (DeprecationWarning, "will be removed in TensorDict 0.17")
        for data, kwargs, batch_size, names, (category, match) in [
            (members, {"auto_batch_size": True}, [2, 3], ["s", None], future),
            (members, {"auto_batch_size": True, "batch_dims": 1}, [2, 3], None, future),
            (
                members,
                {"auto_batch_size": True, "names": ["s"]},
                [2, 3],
                None,
                removal,
            ),
            (members, {}, [2], ["s"], removal),
            (scalars, {"auto_batch_size": True}, [2], ["s"], removal),
        ]:
            kwargs.setdefault("names", ["s", "x"])
            with pytest.warns(category, match=match) as record:
                td = build(data, **kwargs)
            assert len(record) == 1
            assert record[0].filename == __file__
            assert td.batch_size == torch.Size(batch_size)
            assert td.names == (names or ["s", None])
            assert all(
                name is None for member in td.tensordicts for name in member.names
            )
        # one member matches and the other does not: the call will raise
        with pytest.warns(DeprecationWarning, match="removed in TensorDict 0.17"):
            td = build(mixed, auto_batch_size=True, names=["s", "x"])
        assert td.batch_size == torch.Size([2, 3])
        assert td._td_dim_name == "s"
        assert all(name is None for member in td.tensordicts for name in member.names)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            td = build(scalars, auto_batch_size=True, names=["s"])
            assert td.names == ["s"]
            td = build(scalars, names=["s"])
            assert td.names == ["s"]
            td = build(members, batch_size=[3], names=["s", "x"])
            assert td.names == ["s", "x"]
            td = build(members, auto_batch_size=True)
            assert td.names == [None, None]

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

    @pytest.mark.parametrize("build", ["torch.stack", "contiguous", "sub"])
    def test_dense_get_as_list_warns(self, build):
        # a dense tensordict ignores as_list and as_nested_tensor, where a
        # lazy stack does not, and warns (#2551)
        rows = [TensorDict(x=torch.arange(4.0) + i) for i in range(2)]
        lazy = lazy_stack(rows)
        if build == "torch.stack":
            td = torch.stack(rows)
        elif build == "contiguous":
            td = lazy.contiguous()
        else:
            td = torch.stack([torch.stack(rows)] * 3)._get_sub_tensordict(1)
        x = torch.stack([row["x"] for row in rows])
        for name in ("as_list", "as_nested_tensor"):
            with pytest.warns(
                FutureWarning,
                match=rf"get\(key, {name}=True\) returns the stored tensor today",
            ):
                out = td.get("x", **{name: True})
            assert type(out) is torch.Tensor and not out.is_nested
            assert (out == x).all()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (td.get("x", as_padded_tensor=True) == x).all()
            assert td.get("missing", 0, as_list=True) == 0
            assert isinstance(lazy.get("x", as_list=True), list)
            assert lazy.get("x", as_nested_tensor=True).is_nested
            outer = TensorDict(inner=lazy, batch_size=[2])
            assert isinstance(outer.get(("inner", "x"), as_list=True), list)
        with pytest.raises(TypeError, match="as_lst"):
            lazy.get("x", as_lst=True)

    def test_lazy_stack_nested_get(self):
        td0 = TensorDict(a=torch.randn(3))
        td1 = TensorDict(a=torch.randn(4))

        td = lazy_stack([td0, td1])
        td = lazy_stack([td, td.copy()])
        td = lazy_stack([td, td.copy()])

        assert isinstance(td.get("a", as_list=True), list)

        assert isinstance(td.get("a", as_padded_tensor=True), torch.Tensor)

    @pytest.mark.parametrize("padding_side", ["right", "left"])
    def test_lazy_stack_nested_get_padded_ragged(self, padding_side):
        # a [prompts, samples] stack with ragged lengths: every ragged dim is
        # padded, not only dim 0 of each member
        lengths = [[1, 2], [2, 3]]
        grouped = lazy_stack(
            [
                lazy_stack([TensorDict(tokens=torch.full((n,), n + 1.0)) for n in row])
                for row in lengths
            ]
        )
        padded = grouped.get("tokens", as_padded_tensor=True, padding_side=padding_side)
        assert padded.shape == (2, 2, 3)
        flat = grouped.view(-1).get(
            "tokens", as_padded_tensor=True, padding_side=padding_side
        )
        assert (padded == flat.view(2, 2, 3)).all()
        # members ragged after dim 0, such as [1, L] token rows
        rows = [torch.ones(1, 3), torch.ones(1, 5)]
        td = lazy_stack([TensorDict(tokens=row) for row in rows])
        expected = torch.zeros(2, 1, 5)
        if padding_side == "right":
            expected[0, :, :3] = 1
        else:
            expected[0, :, 2:] = 1
        expected[1] = 1
        for out in (
            td.get("tokens", as_padded_tensor=True, padding_side=padding_side),
            LazyStackedTensorDict.lazy_stack(
                rows, as_padded_tensor=True, padding_side=padding_side
            ),
        ):
            assert (out == expected).all()
        # the jagged layout has a single ragged level
        with pytest.raises(ValueError, match=r"view\(-1\)"):
            grouped.get("tokens", as_nested_tensor=True)

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

    @pytest.mark.parametrize(
        "stack_dim,member_shape,index,source_index",
        [
            (0, (), torch.tensor([1, 0]), torch.tensor([0, 1])),
            (0, (), [1, 0], [0, 1]),
            (0, (), np.array([1, 0]), np.array([0, 1])),
            (0, (), [3, 2, 1, 0], [0, 1, 2, 3]),
            (0, (), [1, 2, 3], [0, 1, 2]),
            (0, (), torch.tensor([[1], [0]]), torch.tensor([[0], [1]])),
            (
                0,
                (),
                torch.tensor([False, True, True, True]),
                torch.tensor([True, True, True, False]),
            ),
            (0, (), slice(None), [3, 2, 1, 0]),
            (0, (3,), torch.tensor([1, 0]), torch.tensor([0, 1])),
            (
                0,
                (3,),
                torch.tensor([False, True, True, True]),
                torch.tensor([True, True, True, False]),
            ),
            (0, (3,), ([1, 0], slice(0, 2)), ([0, 1], slice(0, 2))),
            (0, (3,), ([1, 0], slice(None)), ([0, 1], slice(None))),
            (1, (3,), (slice(None), [1, 0]), (slice(None), [0, 1])),
            (
                1,
                (3,),
                (slice(None), torch.tensor([False, True, True, True])),
                (slice(None), torch.tensor([True, True, True, False])),
            ),
        ],
    )
    @pytest.mark.parametrize("op", ["setitem", "setitem_copy", "update_at_"])
    def test_lazy_setitem_overlapping_value(
        self, stack_dim, member_shape, index, source_index, op
    ):
        # the value is a read of the same stack, so its members share memory
        # with the members that the write changes
        lazy = LazyStackedTensorDict.lazy_stack(
            [
                TensorDict(
                    {
                        "a": torch.arange(2.0 * torch.Size(member_shape).numel())
                        .view(*member_shape, 2)
                        .add(100 * i),
                        "nested": {"b": torch.full(member_shape, float(i))},
                    },
                    member_shape,
                )
                for i in range(4)
            ],
            stack_dim,
        )
        dense = lazy.to_tensordict()
        for td in (lazy, dense):
            value = td[source_index]
            if op == "setitem":
                td[index] = value
            elif op == "setitem_copy":
                # new tensordicts that share the tensors of the members
                td[index] = value.copy()
            else:
                td.update_at_(value, index)
        assert (lazy.to_tensordict() == dense).all()

    def test_lazy_setitem_overlapping_value_non_tensor(self):
        stack = NonTensorStack("a", "b", "c", "d")
        stack[1:] = stack[:3]
        assert stack.tolist() == ["a", "a", "b", "c"]

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

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_set_unbatched(self, stack_dim):
        # #2864: the stack dim does not apply to an unbatched value, so each
        # member holds the whole of it, as unbind of a tensordict gives it
        members = [
            TensorDict(a=torch.zeros(2), n={"b": torch.zeros(2)}, batch_size=[2])
            for _ in range(3)
        ]
        lazy = lazy_stack(members, stack_dim)
        u = UnbatchedTensor(torch.zeros(5))
        lazy.set("u", u)
        lazy["v"] = u
        lazy["n", "u"] = u
        lazy.set(("m", "u"), u)
        for key in ("u", "v", ("n", "u"), ("m", "u")):
            for member in members:
                value = member.get(key)
                assert isinstance(value, UnbatchedTensor)
                assert value.data_ptr() == u.data_ptr()
            with warnings.catch_warnings():
                # the members share the storage
                warnings.simplefilter("error")
                value = lazy.get(key)
            assert isinstance(value, UnbatchedTensor)
            assert value.batch_size == lazy.batch_size
        lazy.set_("u", UnbatchedTensor(torch.ones(5)))
        assert (u == 1).all()

        def check_flatten(td):
            flat = td.flatten_keys()
            assert isinstance(flat.get("n.u"), UnbatchedTensor)
            assert set(flat.keys()) == {"a", "n.b", "n.u", "u", "v", "m.u"}
            out = flat.unflatten_keys()
            assert isinstance(out.get(("n", "u")), UnbatchedTensor)
            assert (out == td).all()

        check_flatten(lazy)
        check_flatten(lazy_stack([lazy, lazy.copy()], 0))
        nested = TensorDict(s=lazy, batch_size=lazy.batch_size).flatten_keys()
        assert isinstance(nested.get("s.n.u"), UnbatchedTensor)
        assert (nested.unflatten_keys()["s"] == lazy).all()

    @pytest.mark.parametrize("index", [0, (0,), (0, Ellipsis), slice(0, 1)])
    def test_lazy_setitem_new_key_is_exclusive(self, index):
        # a new key written at an index goes only into the members that the
        # index touches; the members keep their storage
        tds = [TensorDict(a=torch.zeros(3), batch_size=[3]) for _ in range(2)]
        a_ptrs = [td["a"].data_ptr() for td in tds]
        lazy = lazy_stack(tds)
        value = TensorDict(a=torch.ones(3), zz=torch.ones(3), batch_size=[3])
        if isinstance(index, slice):
            value = value.unsqueeze(0)
        lazy[index] = value
        assert [list(td.keys()) for td in lazy.tensordicts] == [["a", "zz"], ["a"]]
        assert [td["a"].data_ptr() for td in tds] == a_ptrs
        assert (tds[0]["a"] == 1).all() and (tds[1]["a"] == 0).all()
        assert (tds[0]["zz"] == 1).all()

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

    @pytest.mark.parametrize("include_nested", [False, True])
    @pytest.mark.parametrize(
        "is_leaf", [None, lambda cls: True], ids=["default", "all_leaves"]
    )
    def test_lazy_stacked_contains_leaves_only(self, include_nested, is_leaf):
        class MC(TensorClass):
            x: torch.Tensor

        td = TensorDict(
            a=torch.zeros(3),
            nested=TensorDict(
                b=torch.zeros(3),
                sub=TensorDict(c=torch.zeros(3), batch_size=[3]),
                s="a string",
                batch_size=[3],
            ),
            tc=MC(x=torch.zeros(3), batch_size=[3]),
            batch_size=[3],
        )
        lstd = lazy_stack([td, td.clone()])
        keys = lstd.keys(include_nested, leaves_only=True, is_leaf=is_leaf)
        listed = list(keys)
        for key in [
            *lstd.keys(include_nested=True),
            ("a", "x"),
            ("nested", "x"),
            ("nested", "s", "data"),
        ]:
            assert (key in keys) == (key in listed), key
        if include_nested and is_leaf is None:
            assert ("nested", "b") in keys
            assert ("nested", "sub", "c") in keys
            assert ("nested", "sub") not in keys

    @pytest.mark.parametrize(
        "is_leaf", [None, is_leaf_nontensor], ids=["default", "nontensor"]
    )
    def test_lazy_stacked_contains_nested(self, is_leaf):
        # A nested key that goes through a tensor is absent, as in TensorDict
        td = TensorDict(
            a=torch.zeros(3),
            nested=TensorDict(b=torch.zeros(3), s="a string", batch_size=[3]),
            batch_size=[3],
        )
        lstd = lazy_stack([td, td.clone()])
        keys = lstd.keys(include_nested=True, is_leaf=is_leaf)
        listed = list(keys)
        for key in [
            *listed,
            ("a", "x"),
            ("a", "x", "y"),
            ("nested", "x"),
            ("nested", "b", "x"),
            ("nested", "s", "data"),
        ]:
            assert (key in keys) == (key in listed), key
            assert (key in keys) == (key in td.keys(True, is_leaf=is_leaf)), key

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

    @pytest.mark.parametrize(
        "entries",
        ["nt", "nt_stack", "nt_td", "tc", "tc_other", "tc_td", "td_tc"],
    )
    @pytest.mark.parametrize("nested", [False, True])
    def test_lazy_stack_entry_class(self, entries, nested):
        # entry_class() gives the class of get(), which keeps non-tensor data
        # and tensorclasses only if every member holds one
        class MC(TensorClass):
            x: torch.Tensor

        class Other(TensorClass):
            x: torch.Tensor

        values = {
            "nt": lambda i: NonTensorData(str(i), batch_size=[3]),
            "nt_stack": lambda i: NonTensorStack(
                *[NonTensorData(str(i)) for _ in range(3)]
            ),
            "td": lambda i: TensorDict(x=torch.zeros(3), batch_size=[3]),
            "tc": lambda i: MC(x=torch.zeros(3), batch_size=[3]),
            "other": lambda i: Other(x=torch.zeros(3), batch_size=[3]),
        }
        first, second = {
            "nt": ("nt", "nt"),
            "nt_stack": ("nt", "nt_stack"),
            "nt_td": ("nt", "td"),
            "tc": ("tc", "tc"),
            "tc_other": ("tc", "other"),
            "tc_td": ("tc", "td"),
            "td_tc": ("td", "tc"),
        }[entries]
        tds = [
            TensorDict(a=torch.zeros(3), v=values[name](i), batch_size=[3])
            for i, name in enumerate((first, second))
        ]
        key = "v"
        if nested:
            tds = [TensorDict(n=td, batch_size=[3]) for td in tds]
            key = ("n", "v")
        lstd = lazy_stack(tds)
        assert lstd.entry_class(key) is type(lstd.get(key))
        for is_leaf in (_default_is_leaf, is_leaf_nontensor, lambda cls: cls is MC):
            keys = lstd.keys(True, True, is_leaf=is_leaf)
            assert (key in keys) == (key in list(keys))
            if not nested:
                # the keys view of leaves tests the class of each entry
                keys = lstd.keys(False, True, is_leaf=is_leaf)
                expected = [k for k in lstd.keys() if is_leaf(type(lstd.get(k)))]
                assert list(keys) == expected
                assert (key in keys) == (key in expected)

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

    @pytest.mark.parametrize("dim", range(2))
    def test_lazy_stacked_extend(self, dim):
        td = TensorDict({"a": torch.zeros(4)}, [4])
        lstd = LazyStackedTensorDict(stack_dim=dim)
        lstd.extend([])
        # an empty stack takes the batch size of the new members, as append does
        lstd.extend([td, td.clone()])
        # a generator is not used up by the type check
        lstd.extend(td.clone() for _ in range(2))
        lstd.extend([])
        lstd.extend(torch.stack([TensorDict({"a": torch.ones(4)}, [4])] * 2, dim))

        bs = [4]
        bs.insert(dim, 6)
        assert lstd.batch_size == torch.Size(bs)
        expected = torch.stack([torch.zeros(4)] * 4 + [torch.ones(4)] * 2, dim)
        torch.testing.assert_close(lstd["a"], expected)

        with pytest.raises(ValueError, match="Batch sizes in tensordicts differs"):
            LazyStackedTensorDict(stack_dim=dim).extend(
                [td, TensorDict({"a": torch.ones(17)}, [17])]
            )

    @pytest.mark.parametrize("dim", range(3))
    @pytest.mark.parametrize("method", ["append", "insert", "extend"])
    def test_lazy_stacked_add_heterogeneous(self, dim, method):
        # the dims where the members differ stay -1, as in lazy_stack
        td0 = TensorDict({"a": torch.zeros(4, 2)}, [4, 2])
        td1 = TensorDict({"a": torch.zeros(4, 3)}, [4, 3])
        lstd = lazy_stack([td0, td1], dim)
        if method == "append":
            lstd.append(td0.clone())
        elif method == "insert":
            lstd.insert(0, td0.clone())
        else:
            lstd.extend([td0.clone(), td0.clone()])

        bs = [4, -1]
        bs.insert(dim, len(lstd.tensordicts))
        assert lstd.batch_size == torch.Size(bs)

        # a member that differs from the first one only in the -1 dim fits,
        # whatever the order of the members
        td5 = TensorDict({"a": torch.ones(4, 5)}, [4, 5])
        if method == "append":
            lstd.append(td5)
            idx = -1
        elif method == "insert":
            lstd.insert(1, td5)
            idx = 1
        else:
            lstd.extend([td5, td5.clone()])
            idx = -2
        bs = [4, -1]
        bs.insert(dim, len(lstd.tensordicts))
        assert lstd.batch_size == torch.Size(bs)
        assert lstd.tensordicts[idx].batch_size == torch.Size([4, 5])
        assert lstd.tensordicts[idx] is td5

        # a member that differs in another dim, or has another number of
        # dims, is still rejected
        for bad in (
            TensorDict({"a": torch.zeros(3, 2)}, [3, 2]),
            TensorDict({"a": torch.zeros(4)}, [4]),
            TensorDict({"a": torch.zeros(4, 2, 1)}, [4, 2, 1]),
        ):
            with pytest.raises(ValueError, match="Batch sizes in tensordicts differs"):
                if method == "append":
                    lstd.append(bad)
                elif method == "insert":
                    lstd.insert(0, bad)
                else:
                    lstd.extend([td0.clone(), bad])
        assert lstd.batch_size == torch.Size(bs)

    def test_lazy_stacked_add_heterogeneous_nested(self):
        # a stack of lazy stacks has -1 in the dim where the inner members
        # differ
        inner = lazy_stack([TensorDict(batch_size=[2]), TensorDict(batch_size=[3])])
        outer = lazy_stack([inner, inner.clone()])
        assert outer.batch_size == torch.Size([2, 2, -1])
        outer.append(TensorDict(batch_size=[2, 5]))
        assert outer.batch_size == torch.Size([3, 2, -1])
        with pytest.raises(ValueError, match="Batch sizes in tensordicts differs"):
            outer.append(TensorDict(batch_size=[3, 5]))
        assert outer.batch_size == torch.Size([3, 2, -1])

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

    @pytest.mark.parametrize(
        "old_shape,new_shape",
        [((2, 1, 3), (2, 3)), ((3, 1, 2), (3, 2)), ((2, 1, 1, 1, 3), (2, 1, 3))],
    )
    @pytest.mark.parametrize("op", ["view", "reshape"])
    def test_lazy_stack_view_drop_size_one_dims(self, old_shape, new_shape, op):
        dense = TensorDict(
            {
                "a": torch.arange(6).view(old_shape),
                "nested": {"b": torch.arange(24.0).view(*old_shape, 4)},
            },
            old_shape,
        )
        expected = dense.reshape(new_shape)
        for stack_dim in range(len(old_shape)):
            tds = lazy_stack(list(dense.unbind(stack_dim)), stack_dim)
            out = getattr(tds, op)(new_shape)
            assert out.batch_size == new_shape
            assert out["nested", "b"].shape == (*new_shape, 4)
            assert (out == expected).all()
        # members that cannot be squeezed, such as sub-tensordicts
        subs = [dense._get_sub_tensordict(k) for k in range(old_shape[0])]
        out = getattr(lazy_stack(subs, 0), op)(new_shape)
        assert out.batch_size == new_shape
        assert (out == expected).all()

    def test_neg_dim_lazystack(self):
        td0 = TensorDict(batch_size=(3, 5))
        td1 = TensorDict(batch_size=(4, 5))
        assert lazy_stack([td0, td1], -1).shape == (-1, 5, 2)
        assert lazy_stack([td0, td1], -2).shape == (-1, 2, 5)
        assert lazy_stack([td0, td1], -3).shape == (2, -1, 5)
        # a dim below the batch dims wraps around again, and warns (#2148)
        with pytest.warns(FutureWarning, match="lazy_stack got a dim outside"):
            assert lazy_stack([td0, td1], -4).shape == (-1, 5, 2)
        with pytest.raises(RuntimeError):
            assert lazy_stack([td0, td1, TensorDict()])
        with pytest.raises(RuntimeError):
            assert lazy_stack([TensorDict(), td0, td1])

    def test_lazy_stack_stack_dim_too_big(self):
        # The valid stack dims of two tensordicts of batch size [3] are 0 and 1
        tds = [
            TensorDict({"a": torch.zeros(3, 4)}, [3]),
            TensorDict({"a": torch.ones(3, 4)}, [3]),
        ]
        with pytest.raises(RuntimeError, match="too big for batch size"):
            LazyStackedTensorDict(*tds, stack_dim=2)
        with pytest.raises(RuntimeError, match="too big for batch size"):
            lazy_stack(tds, 2)

    def test_lazy_stack_empty_stack_dim(self):
        # An empty stack built without a batch size does not know the batch dims
        # of its members until they are added: it accepts stack dims 0 and 1
        td = TensorDict({"a": torch.zeros(3, 4)}, [3])
        for stack_dim, batch_size in ((0, [2, 3]), (1, [3, 2])):
            lstd = LazyStackedTensorDict(stack_dim=stack_dim)
            lstd.append(td)
            lstd.append(td.clone())
            assert lstd.batch_size == torch.Size(batch_size)
            assert lstd["a"].shape == torch.Size([*batch_size, 4])
        with pytest.raises(RuntimeError, match="too big for batch size"):
            LazyStackedTensorDict(stack_dim=2)
        # append takes the batch size of the new members, even if the empty stack
        # was built with a batch size: with [3], it accepts stack dims 0 to 2
        lstd = LazyStackedTensorDict(stack_dim=1, batch_size=[])
        lstd.append(td)
        assert lstd.batch_size == torch.Size([3, 1])
        assert LazyStackedTensorDict(stack_dim=2, batch_size=[3]).batch_size == (3, 0)
        with pytest.raises(RuntimeError, match="too big for batch size"):
            LazyStackedTensorDict(stack_dim=3, batch_size=[3])

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

    def test_popitem(self):
        tds = [
            TensorDict(a=torch.zeros(2), b=TensorDict(c=torch.zeros(1))),
            TensorDict(a=torch.ones(2), b=TensorDict(c=torch.ones(1))),
        ]
        lazy = lazy_stack(tds)
        key, val = lazy.popitem()
        assert key == "b"
        # popitem pops as pop does: a sub-tensordict is a lazy stack
        assert type(val) is LazyStackedTensorDict
        assert (val["c"] == torch.tensor([[0.0], [1.0]])).all()
        key, val = lazy.popitem()
        assert key == "a"
        assert (val == torch.tensor([[0.0, 0.0], [1.0, 1.0]])).all()
        assert all(td.is_empty() for td in tds)

    def test_pop_default_under_tensor(self):
        td = TensorDict(a=torch.zeros(2), n=TensorDict(x=torch.ones(2)))
        lazy = lazy_stack([td.clone(), td.clone()])
        assert lazy.pop(("n", "x", "deeper"), "D") == "D"
        assert lazy.pop(("n", "x", "deeper", "z"), None) is None
        assert lazy.pop(("n", "y"), 0) == 0
        # two parts: the default too, as for three or more (#2678)
        assert lazy.pop(("a", "b"), "D") == "D"
        assert lazy.get(("a", "b"), "D") == "D"
        assert lazy.get(("n", "x", "deeper"), None) is None
        with pytest.raises(ValueError, match="'a' is a Tensor, not a tensordict"):
            lazy.pop(("a", "b"))
        with pytest.raises(ValueError, match="'x' is a Tensor, not a tensordict"):
            lazy.pop(("n", "x", "deeper"))
        assert (lazy.pop(("n", "x"), None) == 1).all()
        assert list(lazy.keys(True)) == ["a", "n"]

    def test_popitem_error_keeps_entries(self):
        # popitem pops the last key of keys(), "a"; the error of the ragged
        # stack must leave every member with the same entries, in the same
        # order.
        tds = [
            TensorDict(b=torch.ones(1), a=torch.zeros(2)),
            TensorDict(a=torch.zeros(2), b=torch.ones(1)),
            TensorDict(b=torch.ones(1), a=torch.zeros(3)),
        ]
        entries = [list(td.items()) for td in tds]
        with pytest.raises(RuntimeError, match="Failed to stack tensors"):
            lazy_stack(tds).popitem()
        for td, td_entries in zip(tds, entries):
            assert list(td.keys()) == [key for key, _ in td_entries]
            assert all(td[key] is value for key, value in td_entries)

    def test_popitem_key_missing_from_a_member(self):
        # popitem pops the last key that every member has
        tds = [
            TensorDict(b=torch.ones(1), a=torch.zeros(2)),
            TensorDict(a=torch.zeros(2), b=torch.ones(1)),
            TensorDict(b=torch.ones(1)),
        ]
        key, val = lazy_stack(tds).popitem()
        assert key == "b"
        assert (val == 1).all()
        assert [list(td.keys()) for td in tds] == [["a"], ["a"], []]

    def test_popitem_repeated_member(self):
        # A member that is stacked twice is popped once
        td0 = TensorDict(b=torch.ones(1), a=torch.zeros(2))
        td1 = TensorDict(b=torch.ones(1), a=torch.ones(2))
        key, val = lazy_stack([td0, td1, td1]).popitem()
        assert key == "a"
        assert val.tolist() == [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
        assert list(td0.keys()) == ["b"]
        assert list(td1.keys()) == ["b"]

    def test_popitem_shared_source(self):
        # Two rows of one tensordict share its entries: deleting "a" from the
        # first row deletes it from the second
        td0 = TensorDict(b=torch.ones(1), a=torch.zeros(2))
        src = TensorDict(b=torch.ones(2, 1), a=torch.ones(2, 2), batch_size=[2])
        lazy = LazyStackedTensorDict(
            td0, src._get_sub_tensordict(0), src._get_sub_tensordict(1)
        )
        key, val = lazy.popitem()
        assert key == "a"
        assert val.tolist() == [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
        assert list(td0.keys()) == ["b"]
        assert list(src.keys()) == ["b"]

    def test_items_values_as_lists_member_prefix(self):
        # with _NESTED_TENSORS_AS_LISTS, items() and values() walk each member
        # (items() prefixes the member index), and keys() lists the keys once
        td = TensorDict(
            a=torch.zeros(2, 3),
            n=TensorDict(x=torch.ones(2, 3), batch_size=[2, 3]),
            batch_size=[2, 3],
        )
        lazy = lazy_stack(list(td.unbind(0)))
        items = list(lazy.items(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS))
        assert [key for key, _ in items] == [
            ("0", "a"),
            ("0", "n", "x"),
            ("1", "a"),
            ("1", "n", "x"),
        ]
        assert all(value is lazy.tensordicts[0]["a"] for _, value in items[:1])
        values = list(lazy.values(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS))
        assert all(v is w for v, (_, w) in zip(values, items, strict=True))
        assert list(lazy.keys(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)) == [
            "a",
            ("n", "x"),
        ]
        # sort sorts each member, in the order of the members ("10" after "9")
        lazy = lazy_stack([TensorDict(b=1, a=0) for _ in range(11)])
        keys = [
            key
            for key, _ in lazy.items(
                True, True, is_leaf=_NESTED_TENSORS_AS_LISTS, sort=True
            )
        ]
        assert keys[:2] == [("0", "a"), ("0", "b")]
        assert keys[-2:] == [("10", "a"), ("10", "b")]
        values = list(
            lazy.values(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS, sort=True)
        )
        assert [v.item() for v in values[:2]] == [0, 1]

    def test_contains_member_identity(self):
        td = TensorDict(a=torch.zeros(2, 3), batch_size=[2, 3])
        lazy = lazy_stack(list(td.unbind(0)))
        assert lazy.tensordicts[0] in lazy
        assert lazy.tensordicts[0].clone() not in lazy
        assert "a" in lazy
        assert ("a",) in lazy
        with pytest.raises(RuntimeError, match="NestedKey"):
            td in td  # noqa: B015
        with pytest.raises(RuntimeError, match="NestedKey"):
            0 in lazy  # noqa: B015

    def test_get_lazy_kwargs_nested(self):
        lazy = lazy_stack(
            [
                TensorDict(n=TensorDict(x=torch.zeros(2))),
                TensorDict(n=TensorDict(x=torch.ones(3))),
            ]
        )
        as_list = lazy.get(("n", "x"), as_list=True)
        assert [t.tolist() for t in as_list] == [[0.0, 0.0], [1.0, 1.0, 1.0]]
        padded = lazy.get(("n", "x"), as_padded_tensor=True)
        assert padded.tolist() == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        padded = lazy.get(("n", "x"), as_padded_tensor=True, padding_value=-1)
        assert padded.tolist() == [[0.0, 0.0, -1.0], [1.0, 1.0, 1.0]]
        nested = lazy.get(("n", "x"), as_nested_tensor=True)
        assert nested.is_nested
        with pytest.raises(TypeError):
            lazy.get(("n", "x"), not_a_keyword=True)

    def test_get_missing_tensor_default(self):
        td = TensorDict(a=torch.zeros(2, 3), batch_size=[2, 3])
        lazy = lazy_stack(list(td.unbind(0)))
        default = torch.zeros(1)
        assert lazy.get("zz", default) is default
        assert lazy.get(("zz", "y"), default) is default
        assert lazy.get("zz") is None

    @pytest.mark.parametrize("op", ["flatten_keys", "unflatten_keys"])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_flatten_keys_exclusive_keys_keeps_members(self, op, inplace):
        if op == "flatten_keys":
            tds = [
                TensorDict(a=0, n=TensorDict(x=1)),
                TensorDict(a=0, n=TensorDict(x=1, y=2)),
            ]
        else:
            tds = [
                TensorDict({"a": 0, "n.x": 1}),
                TensorDict({"a": 0, "n.x": 1, "n.y": 2}),
            ]
        keys = [list(td.keys(True)) for td in tds]
        with pytest.raises(RuntimeError, match=f"the method {op} cannot complete"):
            getattr(lazy_stack(tds), op)(inplace=inplace)
        assert [list(td.keys(True)) for td in tds] == keys

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_flatten_keys_per_member(self, stack_dim):
        # each member is flattened: it keeps its own values, an UnbatchedTensor
        # included, and the result shares the leaves of the members (#2869)
        tds = [
            TensorDict(
                n=TensorDict(
                    x=torch.full((2,), float(i)),
                    s=NonTensorData(f"s{i}", batch_size=[2]),
                ),
                u=UnbatchedTensor(torch.full((4,), float(i))),
                batch_size=[2],
                names=["b"],
            )
            for i in range(3)
        ]
        lazy = lazy_stack(tds, stack_dim, stack_dim_name="s")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            flat = lazy.flatten_keys()
            out = flat.unflatten_keys()
        assert isinstance(flat, LazyStackedTensorDict)
        assert flat.stack_dim == stack_dim
        assert flat.names == lazy.names
        assert set(flat.keys()) == {"n.x", "n.s", "u"}
        for i, (td, member, back) in enumerate(
            zip(tds, flat.tensordicts, out.tensordicts)
        ):
            assert member.get("n.x") is td.get(("n", "x"))
            assert member.get("u").data_ptr() == td.get("u").data_ptr()
            assert member["n.s"] == f"s{i}"
            assert back.get(("n", "x")) is td.get(("n", "x"))
            assert back.get("u").data_ptr() == td.get("u").data_ptr()
        assert (out == lazy).all()
        # a stack of stacks flattens each inner stack in the same way
        outer = lazy_stack([lazy, lazy_stack([td.clone() for td in tds], stack_dim)])
        flat = outer.flatten_keys()
        assert isinstance(flat.tensordicts[0], LazyStackedTensorDict)
        assert flat.tensordicts[0].tensordicts[2].get("u").data_ptr() == (
            tds[2].get("u").data_ptr()
        )
        assert (flat.unflatten_keys() == outer).all()

    def test_non_tensor_items_ragged_tensors(self):
        lazy = lazy_stack(
            [
                TensorDict(s=NonTensorData("x"), a=torch.zeros(2)),
                TensorDict(s=NonTensorData("y"), a=torch.zeros(3)),
            ]
        )
        ((key, value),) = lazy.non_tensor_items()
        assert key == "s"
        assert value.tolist() == ["x", "y"]
        nested = lazy_stack(
            [
                TensorDict(n=TensorDict(s=NonTensorData("x"))),
                TensorDict(n=TensorDict(s=NonTensorData("y"))),
            ]
        )
        ((key, value),) = nested.non_tensor_items(include_nested=True)
        assert key == ("n", "s")
        assert value.tolist() == ["x", "y"]
        assert nested.non_tensor_items() == ()

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

    @staticmethod
    def _reduce_tds(stack_dim):
        dense = TensorDict(
            a=torch.rand(2, 3, 4),
            nested=TensorDict(b=torch.rand(2, 3, 4), batch_size=[2, 3]),
            batch_size=[2, 3],
        )
        return dense, lazy_stack(dense.unbind(stack_dim), stack_dim)

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"dim": 0}, {"dim": 1}, {"dim": 1, "keepdim": True}, {"dim": "feature"}],
        ids=["no-dim", "dim0", "dim1", "dim1-keepdim", "feature"],
    )
    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize(
        "reduction",
        [
            "sum",
            "nansum",
            "mean",
            "nanmean",
            "prod",
            "std",
            "var",
            "amax",
            "amin",
            "quantile",
        ],
    )
    def test_reduction_reduce_true(self, reduction, stack_dim, kwargs):
        # with reduce=True, a lazy stack gives the tensor of its dense copy
        dense, lazy = self._reduce_tds(stack_dim)
        args = (0.5,) if reduction == "quantile" else ()
        torch.testing.assert_close(
            getattr(lazy, reduction)(*args, reduce=True, **kwargs),
            getattr(dense, reduction)(*args, reduce=True, **kwargs),
        )

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_reduction_reduce_true_max_min(self, stack_dim):
        # max and min without a dim, cummax and cummin along a batch dim
        dense, lazy = self._reduce_tds(stack_dim)
        for reduction in ("max", "min"):
            torch.testing.assert_close(
                getattr(lazy, reduction)(reduce=True),
                getattr(dense, reduction)(reduce=True),
            )
        # the (values, indices) named tuples are compared item by item
        for reduction in ("cummax", "cummin"):
            for dim in (0, 1):
                torch.testing.assert_close(
                    getattr(lazy, reduction)(dim, reduce=True),
                    getattr(dense, reduction)(dim, reduce=True),
                )

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize(
        "reduction",
        ["sum", "nansum", "mean", "nanmean", "std", "var", "amax", "amin", "quantile"],
    )
    def test_reduction_reduce_true_dim_none(self, reduction, stack_dim):
        # dim=None reduces all the dims, as a call without a dim does
        dense, lazy = self._reduce_tds(stack_dim)
        args = (0.5,) if reduction == "quantile" else ()
        expected = getattr(dense, reduction)(*args, reduce=True)
        for kwargs in ({}, {"keepdim": False}):
            torch.testing.assert_close(
                getattr(lazy, reduction)(*args, dim=None, reduce=True, **kwargs),
                expected,
            )

    @pytest.mark.parametrize("heterogeneous", [False, True])
    def test_reduction_reduce_true_dim_none_mixed_ndims(self, heterogeneous):
        # Leaves with different numbers of dims, or members with different
        # shapes: dim=None reduces all the elements
        lazy = lazy_stack(
            [
                TensorDict(
                    a=torch.ones(4, 2 + i * heterogeneous),
                    b=torch.ones(4),
                    n=TensorDict(c=torch.ones(4, 5), batch_size=[4]),
                    batch_size=[4],
                )
                for i in range(3)
            ]
        )
        expected = sum(td.sum(reduce=True) for td in lazy.tensordicts)
        for kwargs in ({}, {"keepdim": False}):
            torch.testing.assert_close(
                lazy.sum(dim=None, reduce=True, **kwargs), expected
            )
        if not heterogeneous:
            torch.testing.assert_close(
                lazy.mean(dim=None, reduce=True),
                lazy.to_tensordict().mean(dim=None, reduce=True),
            )

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_all_any_dim_none_heterogeneous(self, stack_dim):
        # Without a dim, all() and any() combine the members, which need not
        # stack into dense tensors
        lazy = lazy_stack(
            [
                TensorDict(a=torch.ones(4, 2), batch_size=[4]),
                TensorDict(a=torch.ones(4, 3), batch_size=[4]),
            ],
            stack_dim,
        )
        assert lazy.all() is True
        assert lazy.any() is True
        lazy.tensordicts[1]["a"][0, 0] = 0
        assert lazy.all() is False
        assert lazy.any() is True
        lazy.apply_(torch.zeros_like)
        assert lazy.all() is False
        assert lazy.any() is False

    @pytest.mark.parametrize("op,value", [("all", 0), ("any", 1)])
    def test_all_any_dim_none_stops_early(self, op, value):
        # Without a dim, all() stops at the first member that is not all True
        # and any() at the first member that holds a True
        lazy = lazy_stack(
            [TensorDict(a=torch.full((4,), value), batch_size=[4]) for _ in range(3)]
        )
        with mock.patch.object(
            TensorDict, op, autospec=True, side_effect=getattr(TensorDict, op)
        ) as reduce:
            assert getattr(lazy, op)() is bool(value)
        assert reduce.call_count == 1

    @pytest.mark.parametrize("op", ["all", "any"])
    def test_all_any_dim_mismatched_member_names(self, op):
        # The result has no names when the members' names differ, or when the
        # names left are not unique
        lazy = lazy_stack(
            [
                TensorDict(a=torch.ones(3, 4, 2), batch_size=[3, 4], names=["x", "y"]),
                TensorDict(a=torch.ones(3, 4, 2), batch_size=[3, 4], names=["x", "z"]),
            ]
        )
        for dim in (0, 1, 2, -1):
            result = getattr(lazy, op)(dim)
            assert result.names == [None, None]
            assert (result["a"] == getattr(lazy["a"], op)(dim % lazy.ndim)).all()
        member = TensorDict(a=torch.ones(3, 4, 2), batch_size=[3, 4], names=["x", "y"])
        # non-unique names warn until 0.17 (#2690)
        with pytest.warns(FutureWarning, match="stack dim name 'y'"):
            lazy = lazy_stack([member, member.clone()], stack_dim_name="y")
        assert lazy.names == ["y", "x", "y"]
        assert getattr(lazy, op)(1).names == [None, None]
        assert getattr(lazy, op)(0).names == ["x", "y"]
        assert getattr(lazy, op)(2).names == ["y", "x"]

    @pytest.mark.parametrize("op", ["all", "any"])
    def test_all_any_dim_out_of_range(self, op):
        lazy = self._compare_stack(0, False)
        for dim in (2, -3):
            with pytest.raises(RuntimeError, match="dim must be greater than"):
                getattr(lazy, op)(dim)

    @pytest.mark.parametrize("method", ["rename_", "refine_names"])
    def test_rename_locked_names(self, method):
        tds = [TensorDict(a=torch.zeros(3), batch_size=[3]) for _ in range(2)]
        lazy = lazy_stack(tds).lock_()
        assert lazy.names == [None, None]
        getattr(lazy, method)("s", "x")
        assert tds[0].names == ["x"]
        assert lazy.names == ["s", "x"]
        lazy.unlock_()
        assert lazy.names == ["s", "x"]

    @pytest.mark.parametrize("method", ["rename", "rename_"])
    def test_rename_none(self, method):
        tds = [TensorDict(batch_size=[3], names=["x"]) for _ in range(2)]
        lazy = lazy_stack(tds, stack_dim_name="s")
        out = getattr(lazy, method)(None)
        assert out.names == [None, None]
        assert lazy.names == ([None, None] if method == "rename_" else ["s", "x"])
        assert tds[0].names == ([None] if method == "rename_" else ["x"])

    @pytest.mark.parametrize("method", ["rename", "rename_"])
    def test_rename_short_names_raises(self, method):
        # one None left for the members is a list too short, not a clear
        tds = [TensorDict(batch_size=[3, 4], names=["x", "y"]) for _ in range(2)]
        lazy = lazy_stack(tds, stack_dim_name="s")
        with pytest.raises(ValueError, match="the length of the dimension names"):
            getattr(lazy, method)("s", None)
        assert lazy.names == ["s", "x", "y"]

    def test_views_is_shared_mixed_members(self):
        tds = [TensorDict(a=torch.zeros(3), batch_size=[3]) for _ in range(2)]
        tds[0].share_memory_()
        lazy = lazy_stack(tds)
        with pytest.raises(RuntimeError, match="shared status mismatch"):
            lazy.is_shared()
        assert not lazy.is_memmap()
        # a view reads the flags of its source, which hold for every member
        sub = _SubTensorDict(lazy, (slice(0, 1),))
        assert sub.is_shared() is False
        assert sub.is_memmap() is False

    @pytest.mark.parametrize("safe", [False, True])
    @pytest.mark.parametrize(
        "key,new_key",
        [("a", "c"), (("n", "x"), ("n", "y")), ("n", ("n", "m")), ("a", ("a", "b"))],
    )
    def test_rename_key_repeated_member(self, key, new_key, safe):
        # A tensordict that is in the stack more than once is renamed once,
        # also when the old key is still there after the rename.
        def make():
            return TensorDict(
                a=torch.zeros(3),
                n=TensorDict(x=torch.zeros(3), batch_size=[3]),
                batch_size=[3],
            )

        expected = make()
        expected.set(new_key, expected.pop(key))
        keys = set(expected.keys(True, True))
        t, u = make(), make()
        td = lazy_stack([t, t, u])
        assert td.rename_key_(key, new_key, safe=safe) is td
        for member in (td, t, u):
            assert set(member.keys(True, True)) == keys

        # also when a nested lazy stack repeats it, or is repeated
        t = make()
        inner = lazy_stack([t, make()])
        others = [t, make()]
        td = lazy_stack([inner, inner, lazy_stack(others)])
        td.rename_key_(key, new_key, safe=safe)
        for member in (td, inner, *inner.tensordicts, *others):
            assert set(member.keys(True, True)) == keys

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("error", ["safe", "missing", "locked"])
    def test_rename_key_error_leaves_stack_unchanged(self, error, nested):
        # A rename that raises renames the key in no member.
        tds = [TensorDict(a=torch.zeros(3), batch_size=[3]) for _ in range(4)]
        if error == "safe":
            tds[-1]["c"] = torch.ones(3)
            raises = pytest.raises(KeyError, match="key c already present")
        elif error == "missing":
            tds[-1] = TensorDict(b=torch.zeros(3), batch_size=[3])
            raises = pytest.raises(KeyError, match='key "a" not found')
        else:
            tds[-1].lock_()
            raises = pytest.raises(RuntimeError, match="Cannot modify locked")
        if nested:
            td = lazy_stack([lazy_stack(tds[:2]), lazy_stack(tds[2:])])
        else:
            td = lazy_stack(tds)
        assert not td.is_locked
        with raises:
            td.rename_key_("a", "c", safe=error == "safe")
        for member in tds[:-1]:
            assert list(member.keys()) == ["a"]

    @pytest.mark.parametrize("safe", [False, True])
    def test_rename_key_inside_old_key_leaves_stack_unchanged(self, safe):
        # A new key inside the old one is not taken, even with safe=True, and
        # a member without the old key raises before any member moves it.
        tds = [
            TensorDict(a=TensorDict(b=torch.zeros(3), batch_size=[3]), batch_size=[3])
            for _ in range(3)
        ]
        tds[-1] = TensorDict(c=torch.zeros(3), batch_size=[3])
        td = lazy_stack(tds)
        with pytest.raises(KeyError, match='key "a" not found'):
            td.rename_key_("a", ("a", "b"), safe=safe)
        for member in tds[:-1]:
            assert set(member.keys(True, True)) == {("a", "b")}

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize(
        "repeats",
        [2, torch.tensor(2), torch.tensor([2]), torch.tensor([1, 0, 2])],
        ids=["int", "0d", "one-element", "per-element"],
    )
    def test_repeat_interleave_matches_torch(self, stack_dim, dim, repeats):
        # a 0-d or one-element repeats is broadcast to the size of dim, also
        # along the stack dim
        a = torch.arange(9.0).view(3, 3)
        lazy = LazyStackedTensorDict.lazy_stack(
            TensorDict(a=a, batch_size=[3, 3]).unbind(stack_dim), stack_dim
        )
        result = lazy.repeat_interleave(repeats, dim=dim)
        expected = a.repeat_interleave(repeats, dim=dim)
        assert result.batch_size == expected.shape
        assert (result["a"] == expected).all()

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

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_split_stack_dim_invalid_sizes(self, stack_dim):
        # Along the stack dim, sizes that are negative or do not add up to the
        # stack size warn: they will raise in 0.17, as in TensorDict.split.
        # The pieces stay as they are until then.
        td = lazy_stack(
            [TensorDict({"a": torch.full((2,), float(i))}, [2]) for i in range(3)],
            stack_dim,
        )
        for split_size, sizes in (
            ([1, 1], [1, 1]),
            ([2, 2], [2, 1]),
            ([4], [3]),
            ([], []),
            ([-1, 4], [2, 1]),
            ([1, -1, 3], [1, None, 3]),
            ((s for s in (1, 1)), [1, 1]),
        ):
            with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
                pieces = td.split(split_size, stack_dim)
            # None marks the malformed piece of batch size [0]
            assert [piece.batch_size for piece in pieces] == [
                torch.Size([0])
                if size is None
                else torch.Size([size, 2] if stack_dim == 0 else [2, size])
                for size in sizes
            ]
        assert (pieces[0] == td[(slice(None),) * stack_dim + (slice(0, 1),)]).all()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for split_size in ([1, 0, 2], [3], (s for s in (1, 2)), 1, 2, 4):
                pieces = td.split(split_size, stack_dim)
                assert (torch.cat(pieces, stack_dim) == td).all()
            td.split([1, 1], 1 - stack_dim)
            td.chunk(2, stack_dim)
            with pytest.raises(ValueError, match="split_size must be positive"):
                td.split(0, stack_dim)

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

    def test_split_lazy_zero_size_dim(self):
        td = LazyStackedTensorDict(
            *[TensorDict({"a": torch.zeros(0, 4)}, [0]) for _ in range(3)],
            stack_dim=0,
        )
        (piece,) = td.split(2, 1)
        assert piece.batch_size == torch.Size([3, 0])
        assert piece["a"].shape == torch.Size([3, 0, 4])
        # split(0) gives one empty piece and split([]) none, as in torch
        (piece,) = td.split(0, 1)
        assert piece.batch_size == torch.Size([3, 0])
        assert piece["a"].shape == torch.Size([3, 0, 4])
        assert td.split([], 1) == ()
        # along the stack dim of a stack with no members
        empty = LazyStackedTensorDict(batch_size=(3,), stack_dim=0)
        for split_size in (0, 2):
            (piece,) = empty.split(split_size, 0)
            assert piece.batch_size == torch.Size([0, 3])
        assert empty.split([], 0) == ()

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize(
        "nesting", ["stack", "stack of stacks", "in a td", "typed", "typed in a td"]
    )
    @pytest.mark.parametrize("inplace", [False, True])
    @pytest.mark.parametrize("reproduce_struct", [False, True])
    def test_split_keys_exclusive_entries(
        self, stack_dim, nesting, inplace, reproduce_struct
    ):
        # An entry that only some members hold goes to the part that holds it
        # in those members, the last one, and is not copied into the other
        # parts with reproduce_struct=True (#2803)
        class Typed(TypedTensorDict):
            a: torch.Tensor
            b: torch.Tensor

        prefix = ("s",) if nesting.endswith("in a td") else ()

        def make():
            members = [
                TensorDict(
                    a=torch.zeros(2),
                    b=torch.ones(2),
                    n=TensorDict(c=torch.zeros(2), batch_size=[2]),
                    batch_size=[2],
                )
                for _ in range(2)
            ]
            members[1]["extra"] = NonTensorData("q", batch_size=[2])
            members[1]["t"] = torch.ones(2)
            members[1]["n", "x"] = NonTensorData("r", batch_size=[2])
            td = lazy_stack(members, stack_dim)
            if nesting == "stack of stacks":
                td = lazy_stack([td, td.clone()], stack_dim)
            if nesting.startswith("typed"):
                td = Typed.from_tensordict(td)
            if prefix:
                td = TensorDict(s=td, batch_size=td.batch_size)
            return td

        def leaves(td, depth=2 if nesting == "stack of stacks" else 1):
            # the leaves of each member: a part may be dense
            if depth:
                return [leaves(member, depth - 1) for member in td.unbind(stack_dim)]
            return {
                key if isinstance(key, tuple) else (key,)
                for key in td.keys(True, True, is_leaf=is_leaf_nontensor)
            }

        def split(*key_sets, **kwargs):
            td = make()
            key_sets = [[(*prefix, *key) for key in keys] for keys in key_sets]
            parts = td.split_keys(
                *key_sets,
                inplace=inplace,
                reproduce_struct=reproduce_struct,
                **kwargs,
            )
            if inplace:
                assert parts[-1] is td
            if nesting == "typed":
                assert all(type(part) is Typed for part in parts)
            elif nesting == "typed in a td":
                assert type(parts[-1]["s"]) is Typed
            if prefix:
                # a part without leaves may have no "s"
                parts = [part.get("s", part) for part in parts]
            return [leaves(part) for part in parts]

        def expected(*parts):
            if nesting == "stack of stacks":
                return [[part, part] for part in parts]
            return list(parts)

        common = {("b",), ("n", "c")}
        exclusive = {("extra",), ("t",), ("n", "x")}
        assert split([("a",)]) == expected(
            [{("a",)}, {("a",)}], [common, common | exclusive]
        )
        assert split([("a",)], [("b",)]) == expected(
            [{("a",)}, {("a",)}],
            [{("b",)}, {("b",)}],
            [{("n", "c")}, {("n", "c"), ("extra",), ("t",), ("n", "x")}],
        )
        # A key set that names an exclusive entry gets it from the members
        # that hold it; a key that no member holds raises (#2827)
        assert split([("a",)], [("extra",)]) == expected(
            [{("a",)}, {("a",)}],
            [set(), {("extra",)}],
            [common, common | {("t",), ("n", "x")}],
        )
        assert split([("b",), ("t",), ("n", "x")]) == expected(
            [{("b",)}, {("b",), ("t",), ("n", "x")}],
            [{("a",), ("n", "c")}, {("a",), ("n", "c"), ("extra",)}],
        )
        with pytest.raises(KeyError):
            split([("a",)], [("extra",), ("nope",)])
        # with strict=False the entry stays in the last part, as on main, with
        # one FutureWarning per call until 0.17
        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17") as w:
            parts = split([("a",)], [("extra",), ("t",)], strict=False)
        assert len(w) == 1
        assert parts == expected(
            [{("a",)}, {("a",)}], [set(), set()], [common, common | exclusive]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            split([("a",)], [("nope",)], strict=False)
        # a renamed exclusive entry stays in its lazy stack
        parts = make().split_keys({(*prefix, "extra"): (*prefix, "e")})
        parts = [part.get("s", part) if prefix else part for part in parts]
        assert leaves(parts[0]) == expected([set(), {("e",)}])[0]
        # a key set may be an iterator
        parts = make().split_keys(
            (key for key in [(*prefix, "a")]), [(*prefix, "extra")], inplace=inplace
        )
        parts = [part.get("s", part) if prefix else part for part in parts]
        assert [leaves(part) for part in parts[:2]] == expected(
            [{("a",)}, {("a",)}], [set(), {("extra",)}]
        )
        if prefix:
            # main's error for a key that no member holds
            for keys, missed in (
                ([("s", "nope")], "nope"),
                ([("s", "extra"), ("s", "nope")], "extra"),
            ):
                with pytest.raises(KeyError) as err:
                    make().split_keys(keys)
                assert err.value.args[0].startswith(f"key \"('s', '{missed}')\"")
            with pytest.raises(KeyError, match="outside that stack"):
                make().split_keys({("s", "extra"): "e"})
            # an entry moved out of the stack is in no other part
            parts = make().split_keys(
                {("s", "a"): "a2"},
                [("s", "extra")],
                inplace=inplace,
                reproduce_struct=reproduce_struct,
            )
            assert set(parts[0].keys(True, True)) == {"a2"}
            assert leaves(parts[1]["s"]) == [set(), {("extra",)}]
            assert leaves(parts[2]["s"]) == [common, common | {("t",), ("n", "x")}]

        if nesting == "in a td":
            # a lazy stack in a tensorclass is split in the same way
            class MC(TensorClass):
                a: torch.Tensor
                extra: torch.Tensor | None = None

            members = [TensorDict(a=torch.zeros(2), batch_size=[2]) for _ in range(2)]
            members[1]["extra"] = torch.ones(2)
            td = TensorDict(
                s=MC._from_tensordict(lazy_stack(members, stack_dim)),
                batch_size=[2, 2],
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                parts = td.split_keys(
                    [("s", "extra")],
                    inplace=inplace,
                    reproduce_struct=reproduce_struct,
                )
            assert all(isinstance(part["s"], MC) for part in parts)
            assert [leaves(part["s"]._tensordict) for part in parts] == [
                [set(), {("extra",)}],
                [{("a",)}, {("a",)}],
            ]

    def test_chunk_lazy_zero_size_dim(self):
        # torch.chunk gives `chunks` empty pieces along a zero-size dim
        for stack_dim, dim in ((0, 1), (0, -1), (1, 0)):
            td = LazyStackedTensorDict(
                *[TensorDict({"a": torch.zeros(0, 4)}, [0]) for _ in range(3)],
                stack_dim=stack_dim,
            )
            for chunks in (2, 3):
                pieces = td.chunk(chunks, dim)
                ref_pieces = torch.zeros(td.batch_size).chunk(chunks, dim)
                assert [p.batch_size for p in pieces] == [r.shape for r in ref_pieces]
                for piece in pieces:
                    assert isinstance(piece, LazyStackedTensorDict)
                    assert piece["a"].shape == (*piece.batch_size, 4)
        # unflattening the zero-size stack dim goes through chunk
        out = td.reshape(2, 0, 3)
        assert out.batch_size == torch.Size([2, 0, 3])
        assert out["a"].shape == torch.Size([2, 0, 3, 4])

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_reshape_view_lazy_zero_size_dim(self, stack_dim):
        # Unbinding a zero-size dim leaves no members to stack, so these
        # calls build a dense tensordict with the shared keys
        td = LazyStackedTensorDict(
            *[
                TensorDict({"a": torch.zeros(0, 4), "n": {"b": torch.zeros(0)}}, [0])
                for _ in range(3)
            ],
            stack_dim=stack_dim,
        )
        ref = torch.zeros(td.batch_size)
        calls = {
            "reshape(-1)": (lambda x: x.reshape(-1)),
            "view(-1)": (lambda x: x.view(-1)),
            "flatten()": (lambda x: x.flatten()),
            "reshape(3, -1)": (lambda x: x.reshape(3, -1)),
            "reshape(-1, 2)": (lambda x: x.reshape(-1, 2)),
        }
        for name, call in calls.items():
            out = call(td)
            assert out.batch_size == call(ref).shape, name
            assert out["a"].shape == (*out.batch_size, 4), name
            assert out["n", "b"].shape == out.batch_size, name
        # A shape that matches the batch size returns the stack itself
        assert td.reshape([size or -1 for size in td.batch_size]) is td

    def test_reshape_same_shape_returns_self(self):
        # a 1-D stack returns itself, as a multi-dim stack (which raised
        # before) and a TensorDict do
        td = LazyStackedTensorDict(
            *[TensorDict(a=torch.zeros(2) + i) for i in range(3)]
        )
        for call in (
            lambda x: x.reshape(3),
            lambda x: x.view(3),
            lambda x: x.flatten(),
            lambda x: x.flatten(0, 0),
        ):
            assert call(td) is td

    @pytest.mark.parametrize("keyed", [True, False])
    @pytest.mark.parametrize("n", [3, 0])
    def test_unflatten_same_shape_returns_self(self, keyed, n):
        # an unflatten to the batch size returns the stack, as reshape and
        # view do. Before, it returned the stack for some sizes with a -1 on
        # an empty batch and raised "Cannot call view" otherwise.
        members = [
            TensorDict(a=torch.zeros(n, 2), batch_size=[n])
            if keyed
            else TensorDict(batch_size=[n])
            for _ in range(2)
        ]
        non_tensor = [NonTensorData("x", batch_size=[n]) for _ in range(2)]
        for td in (
            LazyStackedTensorDict(*members, stack_dim=0),
            LazyStackedTensorDict(*members, stack_dim=1),
            NonTensorStack(*non_tensor, stack_dim=1),
        ):
            for dim in (0, 1, -1):
                for sizes in ((td.batch_size[dim],), (-1,)):
                    assert td.unflatten(dim, sizes) is td

    @pytest.mark.parametrize("sizes", [(3, 3), (2, 2)])
    def test_unflatten_lazy_zero_size_dim_bad_sizes(self, sizes):
        # Sizes that don't multiply up to the dim raise, as on TensorDict.
        # They returned batch sizes [3, 1, 0] and [2, -1, 0] before.
        td = LazyStackedTensorDict(
            *[TensorDict(a=torch.zeros(0, 4), batch_size=[0]) for _ in range(3)],
            stack_dim=0,
        )
        with pytest.raises(RuntimeError, match="don't multiply up"):
            td.unflatten(0, sizes)
        with pytest.raises(RuntimeError, match="don't multiply up"):
            td.to_tensordict().unflatten(0, sizes)

    def test_unflatten_lazy_no_keys_bad_sizes(self):
        # A stack with no keys raises, as a TensorDict with no keys does: it
        # can't regroup its members into these sizes.
        td = LazyStackedTensorDict(*[TensorDict(batch_size=[3])] * 2, stack_dim=1)
        with pytest.raises(RuntimeError, match="don't multiply up"):
            td.unflatten(1, (3, 1))
        empty = LazyStackedTensorDict(*[TensorDict(batch_size=[0])] * 2, stack_dim=1)
        with pytest.raises(RuntimeError, match="don't multiply up"):
            empty.unflatten(1, (3, 1))
        # with no keys to lose, the members are regrouped, as on a non-empty
        # stack
        out = empty.unflatten(-1, (1, -1))
        assert type(out) is LazyStackedTensorDict
        assert out.batch_size == (0, 1, 2)
        # regrouping into a stack of no members would lose the device and the
        # lock: these calls return a dense copy, as before
        empty = LazyStackedTensorDict(
            *[TensorDict(batch_size=[0], device="cpu")] * 2, stack_dim=1
        ).lock_()
        for out in (empty.flatten(), empty.reshape(-1), empty.repeat_interleave(2)):
            assert type(out) is TensorDict
            assert out.batch_size == (0,)
            assert out.device == torch.device("cpu")
            assert out.is_locked
        with pytest.raises(RuntimeError, match="don't multiply up"):
            TensorDict(batch_size=[3, 2]).unflatten(1, (3, 1))

    @staticmethod
    def _shape_stack(stack_dim=1, n=2, hetero=False, **kwargs):
        # members of batch [3] with a nested tensordict; with hetero=True the
        # feature shape of "a" and the keys differ between members
        tds = []
        for i in range(n):
            td = TensorDict(
                a=torch.arange(6.0 * (1 + hetero * i)).view(3, -1) + 10 * i,
                nested=TensorDict(c=torch.full((3,), float(i)), batch_size=[3]),
                batch_size=[3],
            )
            if hetero and i:
                td["extra"] = torch.zeros(3)
            tds.append(td)
        return LazyStackedTensorDict(*tds, stack_dim=stack_dim, **kwargs)

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_split_stack_dim_shares_members(self, stack_dim):
        td = self._shape_stack(stack_dim, n=3)
        pieces = td.split([1, 2], stack_dim)
        assert [type(p) for p in pieces] == [LazyStackedTensorDict] * 2
        assert [p.batch_size[stack_dim] for p in pieces] == [1, 2]
        members = [m for p in pieces for m in p.tensordicts]
        assert all(m is orig for m, orig in zip(members, td.tensordicts))
        # sizes given as a tensor, a numpy array or integer scalars, as
        # TorchRL's TopKRewardSelector passes them
        for sizes in (
            torch.tensor([1, 2]),
            np.array([1, 2]),
            [torch.tensor(1), torch.tensor(2)],
            [np.int64(1), np.int64(2)],
        ):
            pieces = td.split(sizes, stack_dim)
            assert [p.batch_size[stack_dim] for p in pieces] == [1, 2]
            assert pieces[1].tensordicts[0] is td.tensordicts[1]
        # along another dim, the pieces stay lazy and share the leaves
        other = 1 - stack_dim
        pieces = td.split(2, other)
        assert [type(p) for p in pieces] == [LazyStackedTensorDict] * 2
        assert [p.batch_size[other] for p in pieces] == [2, 1]
        for p in pieces:
            for member, orig in zip(p.tensordicts, td.tensordicts):
                assert member["a"].untyped_storage().data_ptr() == (
                    orig["a"].untyped_storage().data_ptr()
                )
        assert (torch.cat(pieces, other) == td).all()

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_split_stack_dim_bad_sizes_warn(self, stack_dim):
        # #2361: sizes that do not add up to the stack size warn until 0.17
        td = self._shape_stack(stack_dim, n=3)
        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
            assert td.split([], stack_dim) == ()
        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
            pieces = td.split([1, 5], stack_dim)
        assert [p.batch_size[stack_dim] for p in pieces] == [1, 2]
        # an invalid int raises as in TensorDict.split (#2361), also on a
        # NonTensorStack and along the other dim
        stack = NonTensorStack.from_list([["a", "b"], ["c", "d"], ["e", "f"]])
        for split_size in (0, -1):
            msg = f"TensorDict.split: split_size must be positive, got {split_size}."
            for x, dim in ((td, stack_dim), (td, 1 - stack_dim), (stack, 0)):
                with pytest.raises(ValueError, match=re.escape(msg)):
                    x.split(split_size, dim)
        with pytest.raises(TypeError, match="must be int or list of ints"):
            td.split(2.0, stack_dim)

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_expand_dense_or_lazy_result(self, stack_dim):
        td = self._shape_stack(stack_dim, stack_dim_name="s")
        result = td.expand(4, *td.batch_size)
        assert type(result) is TensorDict
        assert result.batch_size == (4, *td.batch_size)
        assert result.names[stack_dim + 1] == "s"
        assert (result[0] == td).all()
        hetero = self._shape_stack(stack_dim, hetero=True, stack_dim_name="s")
        result = hetero.expand(4, *hetero.batch_size)
        assert type(result) is LazyStackedTensorDict
        assert result.batch_size == (4, *hetero.batch_size)
        assert result._td_dim_name == "s"
        assert (result[1] == hetero).all()
        # a stack dim of size 1 grows by copying its member
        single = self._shape_stack(stack_dim, n=1)
        shape = list(single.batch_size)
        shape[stack_dim] = 3
        result = single.expand(shape)
        assert result.batch_size == torch.Size(shape)
        index = (slice(None),) * stack_dim
        for i in range(3):
            assert (result[index + (i,)] == single[index + (0,)]).all()
        # an incompatible size, too few sizes, and the size= keyword, as in
        # TensorDict.expand
        for size in (3, 5):
            shape = list(td.batch_size)
            shape[stack_dim] = size
            with pytest.raises(RuntimeError, match="Incompatible expanded shape"):
                td.expand(shape)
        with pytest.raises(RuntimeError, match="the number of sizes provided"):
            td.expand(())
        result = td.expand(size=(4, *td.batch_size))
        assert result.batch_size == (4, *td.batch_size)

    def test_expand_inplace_raises(self):
        # LazyStackedTensorDict.expand had an undocumented inplace=True that
        # changed the members but not the batch size of the stack: it raises
        # as in TensorDict.expand
        td = self._shape_stack(1)
        with pytest.raises(TypeError):
            td.expand(2, 3, 2, inplace=True)
        assert td.batch_size == (3, 2)
        assert td.tensordicts[0].batch_size == (3,)

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_heterogeneous_shape_ops_stay_lazy(self, stack_dim):
        td = self._shape_stack(stack_dim, hetero=True)
        calls = {
            "reshape(6)": lambda x: x.reshape(6),
            "reshape(-1)": lambda x: x.reshape(-1),
            "flatten(0, 1)": lambda x: x.flatten(0, 1),
            "unflatten": lambda x: x.unflatten(1 - stack_dim, (3, 1)),
            "expand": lambda x: x.expand(4, *x.batch_size),
            "repeat_interleave": lambda x: x.repeat_interleave(2, stack_dim),
            "split": lambda x: x.split(1, stack_dim)[1],
        }

        def feature_shapes(x):
            if isinstance(x, LazyStackedTensorDict):
                return {s for m in x.tensordicts for s in feature_shapes(m)}
            return {x["a"].shape[x.ndim :]}

        for name, call in calls.items():
            out = call(td)
            assert type(out) is LazyStackedTensorDict, name
            assert feature_shapes(out) == {(2,), (4,)} or name == "split", name
        assert feature_shapes(td.split(1, stack_dim)[1]) == {(4,)}

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_repeat_interleave_stack_dim_tensor_repeats(self, stack_dim):
        td = self._shape_stack(stack_dim)
        dense = td.to_tensordict()
        for repeats in (torch.tensor(3), torch.tensor([3]), torch.tensor([1, 2])):
            out = td.repeat_interleave(repeats, stack_dim)
            assert type(out) is LazyStackedTensorDict
            assert (out == dense.repeat_interleave(repeats, stack_dim)).all()
        # the members are repeated, not copied
        out = td.repeat_interleave(torch.tensor([2, 1]), stack_dim)
        assert [id(m) for m in out.tensordicts] == [
            id(m) for m in td.tensordicts[:1]
        ] * 2 + [id(td.tensordicts[1])]
        # dim=None on a 1-d stack, and a negative dim out of range
        flat = self._shape_stack(0).reshape(-1)
        assert (
            flat.repeat_interleave(2) == flat.to_tensordict().repeat_interleave(2)
        ).all()
        with pytest.raises(ValueError, match="out of range"):
            td.repeat_interleave(2, -3)
        # the repeats are checked along the stack dim, as torch does
        with pytest.raises(RuntimeError, match="repeats can not be negative"):
            td.repeat_interleave(torch.tensor(-1), stack_dim)
        with pytest.raises(RuntimeError, match="repeats must have the same size"):
            td.repeat_interleave(torch.tensor([1, 2, 3]), stack_dim)

    def test_repeat_interleave_past_batch_dims_warns(self):
        # a stack of 0-d members warns for a dim past its batch dims, as a
        # TensorDict does (#2148), and keeps the result of its members, which
        # unsqueeze to batch size [1] and repeat along it
        td = LazyStackedTensorDict(
            *[TensorDict(a=torch.arange(4.0) + i) for i in range(3)]
        )
        with pytest.warns(FutureWarning, match="repeat_interleave got a dim outside"):
            out = td.repeat_interleave(2, 1)
        assert type(out) is LazyStackedTensorDict
        assert out.batch_size == (3, 2)
        expected = td.to_tensordict()["a"].unsqueeze(1).expand(3, 2, 4)
        assert (out["a"] == expected).all()

    def test_densify_paths(self):
        # an exclusive key is stacked with empty entries for the members that
        # lack it
        td = LazyStackedTensorDict(
            TensorDict(a=torch.ones(3), b=torch.ones(2, 3), batch_size=[]),
            TensorDict(a=torch.ones(3), batch_size=[]),
            stack_dim=0,
        )
        out = td.densify()
        assert type(out) is TensorDict
        assert [t.shape for t in out.get("b").unbind(0)] == [(2, 3), (2, 0)]
        # a key of empty shape that is exclusive cannot be padded
        td = LazyStackedTensorDict(
            TensorDict(a=torch.ones(3), b=torch.ones(())),
            TensorDict(a=torch.ones(3)),
        )
        with pytest.raises(RuntimeError, match="empty shape and exclusive keys"):
            td.densify()
        # different shapes give a nested tensor along stack dim 0
        td = LazyStackedTensorDict(
            TensorDict(a=torch.ones(2), batch_size=[]),
            TensorDict(a=torch.ones(3), batch_size=[]),
            stack_dim=0,
        )
        assert td.densify().get("a").is_nested
        # ... and raise along another stack dim
        td = LazyStackedTensorDict(
            TensorDict(a=torch.ones(2, 2), batch_size=[2]),
            TensorDict(a=torch.ones(2, 3), batch_size=[2]),
            stack_dim=1,
        )
        with pytest.raises(NotImplementedError, match="stack_dim is 1"):
            td.densify()
        # a nested non-tensor entry and a missing nested collection
        td = LazyStackedTensorDict(
            TensorDict(
                s=NonTensorData("x"), n=TensorDict(c=torch.ones(())), batch_size=[]
            ),
            TensorDict(s=NonTensorData("y"), batch_size=[]),
        )
        out = td.densify()
        assert out["s"] == ["x", "y"]
        # the layout can be positional, as in TensorDict.densify
        assert td.densify(torch.strided)["s"] == ["x", "y"]
        # a NonTensorStack is its own dense form
        stack = NonTensorStack("a", "b")
        assert stack.densify() is stack

    def test_get_item_shape_heterogeneous(self):
        td = LazyStackedTensorDict(
            TensorDict(
                a=torch.ones(2, 3, 4), n=TensorDict(b=torch.ones(2, 5)), batch_size=[2]
            ),
            TensorDict(
                a=torch.ones(2, 3, 5), n=TensorDict(b=torch.ones(2, 5)), batch_size=[2]
            ),
            stack_dim=1,
        )
        assert td.get_item_shape("a") == torch.Size([2, 2, 3, -1])
        assert td.get_item_shape(("n", "b")) == torch.Size([2, 2, 5])
        td = LazyStackedTensorDict(
            TensorDict(a=torch.ones(2, 3), batch_size=[2]),
            TensorDict(a=torch.ones(2, 3, 1), batch_size=[2]),
        )
        assert td.get_item_shape("a") == torch.Size([-1])
        with pytest.raises(KeyError, match="missing"):
            td.get_item_shape("missing")

    @pytest.mark.parametrize("stack_dim", [0, 1, 2])
    def test_view_flatten_regroup_shares(self, stack_dim):
        tds = [
            TensorDict(a=torch.arange(8.0).view(2, 2, 2) + 10 * i, batch_size=[2, 2])
            for i in range(2)
        ]
        td = LazyStackedTensorDict(*tds, stack_dim=stack_dim)
        dense = td.to_tensordict()

        def leaf_members(x):
            if isinstance(x, LazyStackedTensorDict):
                return [m for member in x.tensordicts for m in leaf_members(member)]
            return [x]

        for call in (
            lambda x: x.view(-1),
            lambda x: x.flatten(0, 1),
            lambda x: x.flatten(1, 2),
            lambda x: x.view(4, 2) if stack_dim < 2 else x.view(2, 4),
            lambda x: x.unflatten(0, (2, 1)),
        ):
            out = call(td)
            assert type(out) is LazyStackedTensorDict
            assert (out == call(dense)).all()
            ptrs = {m["a"].untyped_storage().data_ptr() for m in tds}
            assert {
                m["a"].untyped_storage().data_ptr() for m in leaf_members(out)
            } <= ptrs

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_shape_ops_names(self, stack_dim):
        # The lazy shape ops give the names that TensorDict gives
        td = self._shape_stack(stack_dim, stack_dim_name="s")
        td.names = ["s", "r"] if stack_dim == 0 else ["r", "s"]
        dense = td.to_tensordict()
        calls = {
            "split(1, 0)": lambda x: [p.names for p in x.split(1, 0)],
            "split(1, 1)": lambda x: [p.names for p in x.split(1, 1)],
            "chunk(2, 1)": lambda x: [p.names for p in x.chunk(2, 1)],
            "reshape": lambda x: x.reshape(*reversed(x.batch_size)).names,
            "reshape(6)": lambda x: x.reshape(6).names,
            "flatten()": lambda x: x.flatten().names,
            "unflatten(0)": lambda x: x.unflatten(0, (1, -1)).names,
            "unflatten(1)": lambda x: x.unflatten(1, (-1, 1)).names,
            "expand": lambda x: x.expand(2, *x.batch_size).names,
            "broadcast_to": lambda x: x.broadcast_to((2, *x.batch_size)).names,
        }
        for name, call in calls.items():
            assert call(td) == call(dense), name
        # the input keeps its names
        assert td.names == dense.names
        assert td.tensordicts[0].names == ["r"]

    def test_shape_ops_arguments(self):
        td = self._shape_stack(1)
        # the identity returns the stack in view and flatten, as torch does
        assert td.view(3, 2) is td
        assert td.flatten(1, 1) is td
        # start_dim > end_dim adds a dim, and repeated dims including the
        # stack dim keep the input, with a warning until 0.17
        with pytest.warns(FutureWarning, match="TensorDict.flatten rejects"):
            assert td.flatten(1, 0).batch_size == (3, 1, 2)
        with pytest.warns(FutureWarning, match="TensorDict.permute rejects"):
            out = self._shape_stack(0).permute(0, 0)
        assert out.batch_size == (2, 3)
        # squeeze and unsqueeze out of range, as in TensorDict
        with pytest.raises(IndexError, match="Incompatible dim"):
            td.squeeze(5)
        with pytest.raises(RuntimeError, match="unsqueezing is allowed"):
            td.unsqueeze(5)
        # unflatten with sizes that do not fit, as TensorDict.unflatten
        with pytest.raises(RuntimeError, match="don't multiply up to the size"):
            td.unflatten(0, (2, 2))
        # flatten past the batch dims raises, as before: the members have no
        # such dims
        with pytest.raises(IndexError, match="Incompatible dim"):
            td.flatten(0, 3)
        # reshape(inplace=True)
        with pytest.raises(
            NotImplementedError,
            match="reshape.inplace=True. is not supported on LazyStackedTensorDict",
        ):
            td.reshape(6, inplace=True)
        with pytest.raises(
            NotImplementedError,
            match="repeat_interleave.inplace=True. is not supported on LazyStackedTensorDict",
        ):
            td.repeat_interleave(2, 0, inplace=True)
        with pytest.raises(
            NotImplementedError,
            match="repeat_interleave.inplace=True. is not supported on LazyStackedTensorDict",
        ):
            td.repeat_interleave(2, inplace=True)

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("method", ["reshape", "view"])
    @pytest.mark.parametrize(
        "shape", [(6, 0), (1, 0), (0,), (3, 0, 1), (0, 3), (0, 1, 3), (2, 0, 2)]
    )
    def test_reshape_view_lazy_zero_size_dim_explicit(self, stack_dim, method, shape):
        # Explicit sizes give torch's shape and keep the keys, as -1 does
        td = LazyStackedTensorDict(
            *[
                TensorDict({"a": torch.zeros(0, 4), "n": {"b": torch.zeros(0)}}, [0])
                for _ in range(3)
            ],
            stack_dim=stack_dim,
        )
        expected = torch.zeros(td.batch_size).reshape(shape).shape
        out = getattr(td, method)(shape)
        assert out.batch_size == expected
        assert out["a"].shape == (*expected, 4)
        assert out["n", "b"].shape == expected
        assert not out.is_locked
        assert not td.is_locked
        assert getattr(td.lock_(), method)(shape).is_locked

    @pytest.mark.parametrize("method", ["reshape", "view"])
    @pytest.mark.parametrize(
        "stack_dim,shape", [(0, (1, 3, 0)), (0, (3, 1, 0)), (1, (0, 1, 3))]
    )
    def test_reshape_view_lazy_zero_size_dim_regroup(self, method, stack_dim, shape):
        # Shapes the regrouping gets right keep a lazy stack of the members,
        # with their ragged leaves and exclusive keys
        td = LazyStackedTensorDict(
            *[
                TensorDict({"a": torch.zeros(0, i + 1), f"e{i}": torch.zeros(0)}, [0])
                for i in range(3)
            ],
            stack_dim=stack_dim,
        )
        out = getattr(td, method)(shape)
        assert isinstance(out, LazyStackedTensorDict)
        assert out.batch_size == shape

        def member_keys(td):
            if isinstance(td, LazyStackedTensorDict):
                return set().union(*map(member_keys, td.tensordicts))
            return set(td.keys())

        assert member_keys(out) == {"a", "e0", "e1", "e2"}
        if method == "view":
            out.set("new", torch.zeros(shape))
            assert "new" in td.keys()

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

    @pytest.mark.parametrize("non_tensor", [False, True])
    def test_update__heterogeneous_nested_lazy(self, non_tensor):
        # update_ must not stack the leaves of a nested lazy stack: here they
        # have different shapes and cannot be stacked.
        members = [
            TensorDict(obs=torch.zeros(2, 8), batch_size=[2]),
            TensorDict(obs=torch.zeros(2, 0), batch_size=[2]),
        ]
        if non_tensor:
            for i, member in enumerate(members):
                member.set_non_tensor("s", f"old{i}")
        td = TensorDict(nested=lazy_stack(members), a=torch.zeros(2), batch_size=[2])
        source = td.clone()
        source.apply_(lambda x: x.fill_(1))
        if non_tensor:
            for member in source.get("nested").tensordicts:
                member.set_non_tensor("s", "new")
        td.update_(source)
        assert (td.get("a") == 1).all()
        for member in td.get("nested").tensordicts:
            assert (member.get("obs") == 1).all()
            if non_tensor:
                assert member.get_non_tensor("s") == "new"

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_update__keys_to_update_nested_lazy(self, stack_dim):
        # #2780: update_ with keys_to_update writes the members of a lazy stack
        # that self holds, as without keys_to_update, also for a key that names
        # a tensordict, and raises before it writes for a key that self lacks
        class MC(TensorClass):
            x: torch.Tensor

        def holder(cls=TensorDict):
            # a lazy stack of batch size [2, 3]
            size = (2, 3)[1 - stack_dim]
            members = [
                cls(x=torch.zeros(size, 4), batch_size=[size])
                for _ in range((2, 3)[stack_dim])
            ]
            return TensorDict(
                a=torch.zeros(2, 3), n=lazy_stack(members, stack_dim), batch_size=[2, 3]
            )

        def deep():
            return TensorDict(b=torch.zeros(2, 3), m=holder(), batch_size=[2, 3])

        for make, key in (
            (holder, ("n", "x")),
            (holder, "n"),
            (lambda: holder(MC), ("n", "x")),
            (lambda: lazy_stack([holder(), holder()]), ("n", "x")),
            (deep, ("m", "n", "x")),
            (deep, "m"),
        ):
            td = make()
            expected = td.to_tensordict()
            source = expected + 1
            assert td.update_(source, keys_to_update=[key]) is td
            expected.update_(source, keys_to_update=[key])
            assert (expected.get(key) == 1).all()
            torch.testing.assert_close(td.to_tensordict(), expected)
            source.get(key).zero_()
            prefix = (key,) if isinstance(key, str) else key[:-1]
            source.set(prefix + ("y",), torch.zeros(source.batch_size))
            with pytest.raises(KeyError, match="not found in the dest"):
                td.update_(source, keys_to_update=[prefix])
            torch.testing.assert_close(td.to_tensordict(), expected)

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_update__dense_source_nested_lazy(self, stack_dim):
        # #2775: update_ and copy_ from a dense tensordict write the members of
        # a lazy stack that self holds, also in a lazy stack of such members,
        # and a lazy stack of tensorclasses
        class MC(TensorClass):
            x: torch.Tensor

        def holder(cls=TensorDict):
            # a lazy stack of batch size [2, 3]
            size = (2, 3)[1 - stack_dim]
            members = [
                cls(x=torch.zeros(size, 4), batch_size=[size])
                for _ in range((2, 3)[stack_dim])
            ]
            return TensorDict(
                a=torch.zeros(2, 3), n=lazy_stack(members, stack_dim), batch_size=[2, 3]
            )

        for td in (holder(), lazy_stack([holder(), holder()]), holder(MC)):
            source = td.to_tensordict() + 1
            assert td.update_(source) is td
            assert (td.to_tensordict() == 1).all()
            source["n", "x"] += 1
            assert td.copy_(source) is td
            assert (td["a"] == 1).all()
            assert (td["n", "x"] == 2).all()
            # an input without the other entries
            assert td.update_(source.select(("n", "x")) + 1) is td
            assert (td["a"] == 1).all()
            assert (td["n", "x"] == 3).all()

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("kind", ["dense", "h5", "holds_h5", "lazy"])
    @pytest.mark.parametrize("kind_rejects", [False, True])
    def test_update__rejected_writes_nothing(
        self, stack_dim, kind, kind_rejects, tmp_path
    ):
        # #2776: a part of the source that a member rejects raises before the
        # members before it are written
        if kind in ("h5", "holds_h5") and not _has_h5py:
            pytest.skip("h5py not found.")
        from tensordict import PersistentTensorDict

        def make(kind, rejects):
            # a member of batch size [3], whose "a" an entry of shape [3, 2] fits
            shape = (3, 3) if rejects else (3, 2)
            if kind == "lazy":
                return lazy_stack(
                    [TensorDict(a=torch.ones(shape[1:])) for _ in range(3)]
                )
            if kind == "dense":
                return TensorDict(a=torch.ones(shape), batch_size=[3])
            h5 = PersistentTensorDict.from_dict(
                TensorDict(a=torch.ones(shape), batch_size=[3]),
                filename=tmp_path / f"{rejects}.h5",
            )
            if kind == "h5":
                return h5
            return TensorDict(a=torch.ones(shape), n=h5, batch_size=[3])

        if kind_rejects:
            members = [make("dense", False), make(kind, True)]
        else:
            members = [make(kind, False), make("dense", True)]
        td = lazy_stack(members, stack_dim)
        source = TensorDict(a=torch.zeros(*td.batch_size, 2), batch_size=td.batch_size)
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            td.update_(source)
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            td.copy_(source)
        for member in td.tensordicts:
            assert (member["a"] == 1).all()
        # a member before it that lacks the key raises first
        td_key = lazy_stack(
            [TensorDict(b=torch.ones(3, 2), batch_size=[3]), members[1]], stack_dim
        )
        with pytest.raises(KeyError, match="The key a was not found"):
            td_key.update_(source)
        if kind == "holds_h5" and not kind_rejects:
            # update_ writes such a member before it reaches the next one
            return
        # a key that a member lacks
        td = lazy_stack(
            [members[0], TensorDict(b=torch.ones(3, 2), batch_size=[3])], stack_dim
        )
        with pytest.raises(KeyError, match="The key a was not found"):
            td.update_(source)
        assert (members[0]["a"] == 1).all()

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("kind", ["dense", "tensorclass", "typed", "h5"])
    def test_update__broadcast_nested_lazy(self, stack_dim, kind, tmp_path):
        # #2834: update_ and copy_ broadcast an entry that lacks the stack dim
        # of a lazy stack that self holds, or has it with size 1, into each
        # member, as into a dense tensordict, and raise before they write
        # where it cannot broadcast
        if kind == "h5" and not _has_h5py:
            pytest.skip("h5py not found.")

        class MC(TensorClass):
            x: torch.Tensor

        class Typed(TypedTensorDict):
            x: torch.Tensor

        files = []

        def holder():
            # a lazy stack of batch size [2, 3]
            size = (2, 3)[1 - stack_dim]
            members = []
            for _ in range((2, 3)[stack_dim]):
                member = TensorDict(x=torch.ones(size, 4), batch_size=[size])
                if kind == "tensorclass":
                    member = MC.from_tensordict(member)
                elif kind == "h5":
                    files.append(tmp_path / f"{len(files)}.h5")
                    member = PersistentTensorDict.from_dict(member, filename=files[-1])
                members.append(member)
            n = lazy_stack(members, stack_dim)
            if kind == "typed":
                n = Typed.from_tensordict(n)
            return TensorDict(a=torch.ones(2, 3), n=n, batch_size=[2, 3])

        calls = (
            lambda td, src: td.update_(src),
            lambda td, src: td.update_(src, keys_to_update=[("n", "x")]),
            lambda td, src: td.update_(src, keys_to_update=["n"]),
            lambda td, src: td.copy_(src),
        )
        # n["x"] has shape [2, 3, 4]: the last shape has the stack dim, of size 1
        shapes = [(), (4,), (3, 1), ((1, 3, 4), (2, 1, 4))[stack_dim]]
        for call in calls:
            for shape in shapes:
                td = holder()
                expected = td.to_tensordict()
                source = TensorDict(
                    a=torch.zeros(2, 3),
                    n={"x": torch.arange(2.0, 2 + math.prod(shape)).reshape(shape)},
                )
                assert call(td, source) is td
                call(expected, source)
                assert (expected["n", "x"] != 1).all()
                torch.testing.assert_close(td.to_tensordict(), expected)
            td = holder()
            source = TensorDict(a=torch.zeros(2, 3), n={"x": torch.zeros(5)})
            with pytest.raises(RuntimeError, match="must match the size of tensor"):
                call(td.to_tensordict(), source)
            with pytest.raises(RuntimeError, match="must match the size of tensor"):
                call(td, source)
            assert (td.to_tensordict() == 1).all()

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_update__broadcast_nested_lazy_unbatched(self, stack_dim):
        # #2834: each member holds the whole of an unbatched leaf, which has
        # no stack dim: update_ and copy_ broadcast the source into it
        batch_size = [(2, 3)[stack_dim], (2, 3)[1 - stack_dim]]

        def holder():
            members = [
                TensorDict(
                    x=torch.ones(3, 4),
                    u=UnbatchedTensor(torch.ones(7)),
                    batch_size=[3],
                )
                for _ in range(2)
            ]
            n = lazy_stack(members, stack_dim)
            return TensorDict(n=n, batch_size=batch_size), members

        calls = (
            lambda td, src: td.update_(src),
            lambda td, src: td.update_(src, keys_to_update=[("n", "x"), ("n", "u")]),
            lambda td, src: td.copy_(src),
        )
        for call in calls:
            for u in (torch.zeros(7), torch.zeros(1), UnbatchedTensor(torch.zeros(7))):
                td, members = holder()
                # copy_ needs a source of the same batch size, which only
                # holds u unbatched
                copy = call is calls[2]
                source = TensorDict(
                    n={
                        "x": torch.zeros(*batch_size, 4),
                        "u": UnbatchedTensor(u) if copy else u,
                    },
                    batch_size=batch_size if copy else [],
                )
                assert call(td, source) is td
                for member in members:
                    assert (member["x"] == 0).all()
                    assert (member["u"] == 0).all()
            td, members = holder()
            source = TensorDict(
                n=TensorDict(
                    x=torch.zeros(*batch_size, 4),
                    u=UnbatchedTensor(torch.zeros(5)),
                    batch_size=batch_size,
                ),
                batch_size=batch_size,
            )
            with pytest.raises(RuntimeError, match="must match the size of tensor"):
                call(td, source)
            for member in members:
                assert (member["x"] == 1).all()
                assert (member["u"] == 1).all()

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_update__unbatched(self, stack_dim):
        # #2868: the stack dim does not index an unbatched leaf of the
        # members, along any stack dim: update_ writes the source into each
        # member, in place, as along dim 0
        def u(value):
            return UnbatchedTensor(torch.full((4,), float(value)))

        members = [
            TensorDict(a=torch.zeros(3), u=u(0), batch_size=[3]) for _ in range(2)
        ]
        lazy = lazy_stack(members, stack_dim)
        ptrs = [member["u"].data_ptr() for member in members]
        calls = (
            lambda: lazy.update_({"u": u(1)}),
            lambda: lazy.update_(TensorDict(u=u(2))),
            lambda: lazy.update_(TensorDict(u=u(3), batch_size=lazy.batch_size)),
            lambda: lazy.update_({"u": torch.full((4,), 4.0)}),
            lambda: lazy.copy_(TensorDict(a=torch.tensor(5.0), u=u(5))),
        )
        for value, call in enumerate(calls, 1):
            assert call() is lazy
            for member, ptr in zip(members, ptrs):
                assert isinstance(member["u"], UnbatchedTensor)
                assert member["u"].data_ptr() == ptr
                assert (member["u"] == value).all()
        # a source with a leading dim of the size of the stack gives each
        # member its row
        lazy.update_({"u": torch.arange(8.0).view(2, 4)})
        for member, row in zip(members, torch.arange(8.0).view(2, 4)):
            assert (member["u"] == row).all()
        # a leaf whose first dim is the size of the stack: a dict source is
        # split along that dim, along any stack dim, as on main
        members = [
            TensorDict(
                a=torch.zeros(3), u=UnbatchedTensor(torch.zeros(2)), batch_size=[3]
            )
            for _ in range(2)
        ]
        lazy = lazy_stack(members, stack_dim)
        lazy.update_({"u": torch.arange(2.0)})
        for i, member in enumerate(members):
            assert (member["u"] == i).all()

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

    @staticmethod
    def _members(n, value=0.0, **kwargs):
        return [
            TensorDict(a=torch.full((3,), value + i), batch_size=[3], **kwargs)
            for i in range(n)
        ]

    def test_lazy_update_per_member_shares_storage(self):
        # update writes each member of a lazy source into the member of self
        # by reference, and keeps the members of self
        members = self._members(2)
        lazy = lazy_stack(members)
        source = lazy_stack(self._members(2, 10.0))
        assert lazy.update(source) is lazy
        for member, dest, src in zip(lazy.tensordicts, members, source.tensordicts):
            assert member is dest
            assert member["a"] is src["a"]
        # a dense source is split along the stack dim
        dense = TensorDict(
            a=torch.full((2, 3), 20.0), b=torch.ones(2, 3), batch_size=[2, 3]
        )
        assert lazy.update(dense) is lazy
        assert (lazy["a"] == 20).all() and (lazy["b"] == 1).all()
        assert lazy.tensordicts[0]["a"].data_ptr() == dense["a"].data_ptr()
        # dicts, kwargs, None, self and empty keys_to_update
        assert lazy.update({"c": torch.zeros(2, 3)}, d=torch.ones(2, 3)) is lazy
        assert (lazy["c"] == 0).all() and (lazy["d"] == 1).all()
        assert (
            lazy.update(
                TensorDict(c=torch.ones(2, 3), batch_size=[2, 3]), e=torch.ones(2, 3)
            )
            is lazy
        )
        assert (lazy["c"] == 1).all() and (lazy["e"] == 1).all()
        assert lazy.update(e=torch.zeros(2, 3)) is lazy and (lazy["e"] == 0).all()
        assert lazy.update(None) is lazy and lazy.update(lazy) is lazy
        assert lazy.update({"e": torch.ones(2, 3)}, keys_to_update=[]) is lazy
        assert (lazy["e"] == 0).all()
        # a source whose batch size can be reset to that of self
        assert lazy.update(TensorDict(a=torch.zeros(2, 3), batch_size=[])) is lazy
        assert (lazy["a"] == 0).all()
        with pytest.raises(ValueError, match="different shapes"):
            lazy.update(TensorDict(a=torch.zeros(3, 3), batch_size=[]))
        with pytest.raises(ValueError, match="update_batch_size=False"):
            lazy.update(lazy_stack(self._members(1)))
        with pytest.raises(RuntimeError, match="Some keys of the dest"):
            lazy.update(lazy_stack(self._members(1)), update_batch_size=True)
        lazy.lock_()
        with pytest.raises(RuntimeError, match="locked"):
            lazy.update({"a": torch.ones(2, 3)})
        assert lazy.update({"a": torch.ones(2, 3)}, inplace=True) is lazy
        # inplace given by position, as TensorDict accepts it
        assert lazy.update({"a": torch.zeros(2, 3)}, False, True) is lazy
        assert (lazy["a"] == 0).all()

    def test_lazy_update_vmap(self):
        # Under vmap over the stack dim, update writes each member
        lazy = lazy_stack(self._members(2))

        def add_b(td):
            return td.update({"b": td["a"] + 1})

        def add_c(td):
            return td.update(TensorDict(c=td["a"] + 2, batch_size=[]))

        out = torch.vmap(add_b)(lazy)
        assert (out["b"] == lazy["a"] + 1).all()
        out = torch.vmap(add_c)(lazy)
        assert (out["c"] == lazy["a"] + 2).all()

    @pytest.mark.parametrize(
        "stack_dim,source",
        [
            (0, {}),
            (0, {"a": torch.full((), 2.0)}),
            (0, {"a": torch.arange(3.0)}),
            (0, TensorDict(a=torch.arange(3.0), batch_size=[])),
            (0, TensorDict(a=torch.arange(3.0).expand(1, 3), batch_size=[1, 3])),
            (1, TensorDict(a=torch.arange(6.0).view(2, 3), batch_size=[2])),
            (1, TensorDict(a=torch.arange(6.0).view(2, 3), batch_size=[])),
            (1, {"a": torch.arange(6.0).view(2, 3)}),
            (1, {"a": torch.arange(3.0)}),
        ],
    )
    def test_lazy_update__broadcasts(self, stack_dim, source):
        # update_ broadcasts the leaves of a source that does not have the
        # stack dim to the leaves of self, as TensorDict.update_ does
        lazy = lazy_stack(
            [TensorDict(a=torch.zeros(2), batch_size=[2]) for _ in range(3)], 1
        )
        if stack_dim == 0:
            lazy = lazy_stack(self._members(2))
        expected = lazy.to_tensordict().update_(source)
        assert lazy.update_(source) is lazy
        assert (lazy == expected).all()

    def test_lazy_update__arguments(self):
        lazy = lazy_stack(self._members(2))
        # kwargs, None, self, and a source of another length
        assert lazy.update_({"a": torch.ones(2, 3)}, a=torch.full((2, 3), 2.0)) is lazy
        assert (lazy["a"] == 2).all()
        lazy.update_(
            TensorDict(a=torch.ones(2, 3), batch_size=[2, 3]), a=torch.zeros(2, 3)
        )
        assert (lazy["a"] == 0).all()
        assert lazy.update_(a=torch.ones(2, 3)) is lazy and (lazy["a"] == 1).all()
        assert lazy.update_(None) is lazy and lazy.update_(lazy) is lazy
        with pytest.raises(ValueError, match="different shapes"):
            lazy.update_(TensorDict(a=torch.ones(3, 3), batch_size=[3, 3]))

    def test_lazy_update_at_signature(self):
        # LazyStackedTensorDict.update_at_ takes the arguments of
        # TensorDictBase.update_at_, and still the name `index` for idx
        lazy = lazy_stack(self._members(2))
        with pytest.warns(
            DeprecationWarning,
            match=r"update_at_\(index=.*0\.17.*update_at_\(idx=",
        ) as record:
            assert lazy.update_at_({"a": torch.full((3,), 5.0)}, index=0) is lazy
        assert record[0].filename == __file__
        assert (lazy[0]["a"] == 5).all() and (lazy[1]["a"] == 1).all()
        assert lazy.update_at_({"a": torch.full((3,), 7.0)}, idx=0) is lazy
        assert (lazy[0]["a"] == 7).all()
        lazy.update_at_(
            {"a": torch.full((3,), 8.0), "b": torch.ones(3)}, 0, keys_to_update=["a"]
        )
        assert (lazy[0]["a"] == 8).all() and "b" not in lazy.keys()
        assert lazy.update_at_({"a": torch.full((3,), 6.0)}, (1, Ellipsis)) is lazy
        assert (lazy[1]["a"] == 6).all()

    def test_lazy_update_longer_source(self):
        # A source with more members along the stack dim: with
        # update_batch_size=True, the members of self are updated and copies
        # of the extra members are appended (#2206)
        for source in (
            lazy_stack(self._members(3, 10.0)),
            TensorDict(a=torch.full((3, 3), 10.0), batch_size=[3, 3]),
        ):
            lazy = lazy_stack(self._members(2))
            members = list(lazy.tensordicts)
            lazy.update(source, update_batch_size=True)
            assert lazy.batch_size == (3, 3)
            assert (lazy["a"] == source["a"]).all()
            assert lazy.tensordicts[:2] == members
            if isinstance(source, LazyStackedTensorDict):
                assert lazy.tensordicts[2] is not source.tensordicts[2]
                assert lazy.tensordicts[2]["a"] is source.tensordicts[2]["a"]
        # without it, the extra members are left out, as before
        lazy = lazy_stack(self._members(2))
        lazy.update(lazy_stack(self._members(3, 10.0)))
        assert lazy.batch_size == (2, 3)
        # it raises where a TensorDict does, before writing anything
        lazy = lazy_stack(self._members(2))
        lazy.set("b", torch.zeros(2, 3))
        before = lazy["a"].clone()
        for kwargs, match in (
            ({}, "Some keys of the dest"),
            ({"inplace": True}, "inplace=True"),
            ({"keys_to_update": ["a"]}, "keys_to_update"),
        ):
            with pytest.raises(RuntimeError, match=match):
                lazy.to_tensordict().update(
                    lazy_stack(self._members(3, 10.0)).to_tensordict(),
                    update_batch_size=True,
                    **kwargs,
                )
            with pytest.raises(RuntimeError, match=match):
                lazy.update(
                    lazy_stack(self._members(3, 10.0)),
                    update_batch_size=True,
                    **kwargs,
                )
            assert lazy.batch_size == (2, 3) and (lazy["a"] == before).all()

    @pytest.mark.parametrize("members", ["lazy0", "lazy1", "dense"])
    def test_lazy_update_batch_size_members_grow(self, members):
        # update(..., update_batch_size=True) can grow the members of a lazy
        # stack along a dim other than the stack dim: the stack then takes
        # the batch size of its members, as the dense update does (#2206)
        if members == "dense":
            lazy = lazy_stack(
                [TensorDict(a=torch.zeros(2), batch_size=[2]) for _ in range(2)]
            )
            source = TensorDict(a=torch.ones(2, 3), batch_size=[2, 3])
        else:
            stack_dim = int(members[-1])
            lazy = lazy_stack(
                [lazy_stack(self._members(2)) for _ in range(2)], stack_dim
            )
            source = lazy_stack(
                [lazy_stack(self._members(3, 10.0)) for _ in range(2)], stack_dim
            )
        expected = lazy.to_tensordict().update(
            source.to_tensordict(), update_batch_size=True
        )
        lazy.update(source, update_batch_size=True)
        assert lazy.batch_size == source.batch_size == expected.batch_size
        assert (lazy.to_tensordict() == expected).all()
        # members that grow to different sizes give -1, as lazy_stack does
        lazy = lazy_stack(
            [TensorDict(a=torch.zeros(2), batch_size=[2]) for _ in range(2)]
        )
        source = lazy_stack(
            [
                TensorDict(a=torch.ones(3), batch_size=[3]),
                TensorDict(a=torch.ones(4), batch_size=[4]),
            ]
        )
        lazy.update(source, update_batch_size=True)
        assert lazy.batch_size == source.batch_size == (2, -1)
        # under vmap, the batch size of the stack leaves out the stack dim
        sizes = []

        def fn(x, s):
            sizes.append(x.update(s, update_batch_size=True).batch_size)
            return x["a"]

        lazy = lazy_stack(self._members(2))
        torch.vmap(fn)(lazy, lazy_stack(self._members(2)))
        source = TensorDict(a=torch.ones(2, 4), batch_size=[2, 4])
        torch.vmap(fn)(lazy, source.to_lazystack(0))
        assert sizes == [(3,), (4,)]

    def test_lazy_update_tuple_key_over_tensor(self):
        # A tuple key whose prefix is a tensor of the members raises, as on a
        # TensorDict, instead of replacing the tensor with a sub-tensordict
        lazy = lazy_stack([TensorDict(x=torch.zeros(3), batch_size=[3])] * 2)
        with pytest.raises(KeyError, match="already present"):
            lazy.to_tensordict().update({("x", "y"): torch.ones(2, 3)})
        with pytest.raises(KeyError, match="already present"):
            lazy.update({("x", "y"): torch.ones(2, 3)})
        assert (lazy.get("x") == 0).all()
        nested = TensorDict(lazy=lazy, batch_size=[2])
        with pytest.raises(KeyError, match="already present"):
            nested.update({("lazy", "x", "y"): torch.ones(2, 3)})
        lazy.update({("z", "y"): torch.ones(2, 3)})
        assert (lazy.get(("z", "y")) == 1).all()

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_where_scalar_tensor(self, stack_dim):
        # A 0-dim tensor goes whole to each member, as a scalar does
        lazy = lazy_stack(
            [
                TensorDict(a=torch.arange(3.0) + 10 * i, batch_size=[3])
                for i in range(2)
            ],
            stack_dim,
        )
        cond = torch.arange(6).reshape(lazy.shape) % 2 == 0
        other = torch.tensor(-1.0)
        expected = lazy.to_tensordict().where(cond, other)
        assert (lazy.where(cond, other) == expected).all()
        assert (torch.where(cond, lazy, other) == expected).all()

    @pytest.mark.parametrize(
        "stack_dim,other_shape,broadcasts",
        [
            (0, (3,), True),
            (0, (1, 3), True),
            (1, (3, 1), True),
            (0, (1, 2), False),
            (1, (3,), False),
        ],
    )
    def test_where_tensor_not_batch_shaped(self, stack_dim, other_shape, broadcasts):
        # A tensor `other` is split along the stack dim, so it must start with
        # the batch shape up to the stack dim
        lazy = lazy_stack(
            [
                TensorDict(a=torch.arange(3.0) + 10 * i, batch_size=[3])
                for i in range(2)
            ],
            stack_dim,
        )
        cond = torch.arange(6).reshape(lazy.shape) % 2 == 0
        other = torch.full(other_shape, -1.0)
        if not broadcasts:
            # A tensor that does not broadcast against the batch fails as before
            with pytest.raises((RuntimeError, IndexError)):
                lazy.where(cond, other)
            return
        with pytest.raises(ValueError, match="Expand it to the batch shape"):
            lazy.where(cond, other)
        # Expanded to the batch shape, it gives what TensorDict.where gives
        other = other.expand(lazy.shape)
        expected = lazy.to_tensordict().where(cond, other)
        assert (lazy.where(cond, other) == expected).all()

    @pytest.mark.parametrize("other_type", ["lazy", "dense", "tensor", "scalar"])
    def test_where_whole_members(self, other_type):
        # With 0-dim members, the condition selects whole members: the result
        # matches TensorDict.where, in tensors that are not those of the inputs
        lazy = lazy_stack([TensorDict(a=torch.tensor(float(i))) for i in range(3)])
        dense_other = TensorDict(a=-torch.arange(1.0, 4.0), batch_size=[3])
        other = {
            "lazy": lazy_stack(list(dense_other.unbind(0))),
            "dense": dense_other,
            "tensor": dense_other["a"],
            "scalar": torch.tensor(-1.0),
        }[other_type]
        cond = torch.tensor([True, False, True])
        expected = lazy.to_tensordict().where(
            cond, other.to_tensordict() if other_type == "lazy" else other
        )
        result = lazy.where(cond, other)
        assert (result == expected).all()
        result["a"] = torch.full((3,), 100.0)
        assert (lazy["a"] == torch.arange(3.0)).all()
        assert (dense_other["a"] == -torch.arange(1.0, 4.0)).all()

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_where_tensor_other_leaf_dims(self, stack_dim):
        # A tensor `other` with the dims of the leaves gives what
        # TensorDict.where gives, without a warning
        lazy = lazy_stack(
            [
                TensorDict(a=torch.full((3,), 11.0 * i), batch_size=[3])
                for i in range(3)
            ],
            stack_dim,
        )
        cond = torch.eye(3, dtype=torch.bool)
        other = -torch.arange(1.0, 10.0).view(3, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = lazy.where(cond, other)
        assert (result == lazy.to_tensordict().where(cond, other)).all()

    def test_where_tensor_other_fewer_dims(self):
        # A tensor `other` with fewer dims than a leaf is split along the stack
        # dim as before, which differs from TensorDict.where (#2056)
        lazy = lazy_stack(
            [TensorDict(a=torch.full((3,), 11.0 * i), batch_size=[3]) for i in range(3)]
        )
        cond = torch.eye(3, dtype=torch.bool)
        other = -torch.arange(1.0, 4.0)
        match = "the default will change to broadcasting like TensorDict.where"
        with pytest.warns(FutureWarning, match=match):
            result = lazy.where(cond, other)
        assert (result["a"] == torch.where(cond, lazy["a"], other[:, None])).all()
        assert not (result == lazy.to_tensordict().where(cond, other)).all()
        # Members with feature dims, where TensorDict.where raises
        lazy = lazy_stack([TensorDict(a=torch.full((4,), float(i))) for i in range(3)])
        cond = torch.tensor([True, False, True])
        with pytest.warns(FutureWarning, match=match):
            result = lazy.where(cond, other)
        assert (
            result["a"] == torch.where(cond[:, None], lazy["a"], other[:, None])
        ).all()
        with pytest.raises(RuntimeError):
            lazy.to_tensordict().where(cond, other)

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_where_condition_fewer_dims(self, stack_dim):
        # A condition with fewer dims than the batch is expanded on the right
        lazy = lazy_stack(
            [
                TensorDict(a=torch.arange(8.0).view(4, 2) + 10 * i, batch_size=[4])
                for i in range(3)
            ],
            stack_dim,
        )
        cond = torch.arange(lazy.shape[0]) % 2 == 0
        expected = lazy.to_tensordict().where(cond, -1.0)
        result = lazy.where(cond, -1.0)
        assert isinstance(result, LazyStackedTensorDict)
        assert result.stack_dim == stack_dim
        assert (result == expected).all()

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

    def test_select_exclude_nested_through_leaf(self):
        # A nested key through a leaf is missing in each member (#2617)
        lst = LazyStackedTensorDict(
            *[
                TensorDict(x=torch.zeros(3), n=TensorDict(y=torch.zeros(3)))
                for _ in range(2)
            ],
            stack_dim=0,
        )
        with pytest.raises(KeyError, match="'x' is a Tensor, not a tensordict"):
            lst.select(("x", "q"))
        out = lst.select(("x", "q"), "n", strict=False)
        assert isinstance(out, LazyStackedTensorDict)
        assert set(out.keys(True)) == {"n", ("n", "y")}
        out = lst.exclude(("x", "q"), ("n", "y", "q"))
        assert isinstance(out, LazyStackedTensorDict)
        assert set(out.keys(True)) == {"x", "n", ("n", "y")}

    @staticmethod
    def _member_with_empty_groups(value):
        return TensorDict(
            x=torch.full((2,), value),
            e=TensorDict(batch_size=[2]),
            n=TensorDict(
                e=TensorDict(f=TensorDict(batch_size=[2]), batch_size=[2]),
                y=torch.zeros(2),
                batch_size=[2],
            ),
            batch_size=[2],
        )

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("nesting", ["stack", "stack of stacks", "in a td"])
    def test_filter_empty_matches_dense(self, stack_dim, nesting):
        # filter_empty_ and split_keys remove the empty sub-tensordicts that
        # the dense stack loses, instead of raising a KeyError (#2796)
        def make(stack):
            members = [self._member_with_empty_groups(float(i)) for i in range(4)]
            if nesting == "stack of stacks":
                return stack(
                    [stack(members[:2], stack_dim), stack(members[2:], stack_dim)],
                    stack_dim,
                )
            td = stack(members[:2], stack_dim)
            if nesting == "in a td":
                td = TensorDict(a=td, b=td.clone(), batch_size=td.batch_size)
            return td

        def check(result, expected):
            assert set(result.keys(True)) == set(expected.keys(True))
            assert (result == expected).all()

        lazy = make(lazy_stack)
        dense = make(torch.stack)
        result = lazy.copy().filter_empty_()
        assert type(result) is type(lazy)
        check(result, dense.copy().filter_empty_())
        keys = ["x"] if nesting != "in a td" else [("a", "x")]
        for reproduce_struct in (False, True):
            parts = lazy.split_keys(keys, reproduce_struct=reproduce_struct)
            expected = dense.split_keys(keys, reproduce_struct=reproduce_struct)
            for part, part_expected in zip(parts, expected, strict=True):
                check(part, part_expected)

    def test_filter_empty_member_types(self, tmp_path):
        # An empty group in only some members is removed from them, and the
        # other members keep their entries (#2796)
        member = self._member_with_empty_groups
        other = TensorDict(
            x=torch.zeros(2),
            e=TensorDict(z=torch.zeros(2), batch_size=[2]),
            n=TensorDict(y=torch.zeros(2), batch_size=[2]),
            batch_size=[2],
        )
        lazy = lazy_stack([member(0.0), other]).filter_empty_()
        assert set(lazy[0].keys(True)) == {"x", "n", ("n", "y")}
        assert set(lazy[1].keys(True)) == set(other.keys(True))

        # a tensorclass over a lazy stack
        class MC(TensorClass):
            x: torch.Tensor
            e: TensorDict
            n: TensorDict

        tc = MC.from_tensordict(lazy_stack([member(0.0), member(1.0)]))
        assert type(tc.filter_empty_()) is MC
        assert set(tc.keys(True)) == {"x", "n", ("n", "y")}
        if not _has_h5py:
            return
        lazy = lazy_stack(
            [
                PersistentTensorDict.from_dict(
                    member(float(i)), filename=tmp_path / f"{i}.h5"
                )
                for i in range(2)
            ]
        )
        assert set(lazy.filter_empty_().keys(True)) == {"x", "n", ("n", "y")}
        assert set(lazy[1].keys(True)) == {"x", "n", ("n", "y")}

    def test_filter_empty_keeps_member_leaves(self):
        # A lazy stack in a tensordict is kept if one of its members keeps a
        # leaf, even when the members share no leaf (#2796)
        td = TensorDict(
            l=lazy_stack(
                [TensorDict(e=TensorDict()), TensorDict(e=TensorDict(x=torch.ones(())))]
            ),
            y=torch.zeros(2),
            batch_size=[2],
        )
        td.filter_empty_()
        assert set(td.keys()) == {"l", "y"}
        assert set(td["l"].tensordicts[0].keys()) == set()
        assert set(td["l"].tensordicts[1].keys(True)) == {"e", ("e", "x")}
        assert td["l"].tensordicts[1]["e", "x"] == 1
        assert (td["y"] == 0).all()
        td = TensorDict(
            p=TensorDict(
                l=lazy_stack(
                    [TensorDict(a=torch.ones(())), TensorDict(b=torch.full((), 2.0))]
                ),
                batch_size=[2],
            ),
            x=torch.zeros(2),
            batch_size=[2],
        )
        td.filter_empty_()
        assert set(td.keys(True)) == {"p", ("p", "l"), "x"}
        assert td["p", "l"].tensordicts[0]["a"] == 1
        assert td["p", "l"].tensordicts[1]["b"] == 2
        assert (td["x"] == 0).all()

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

    @pytest.mark.parametrize("member_batch_size", [(), (2,)])
    @pytest.mark.parametrize("index", [1, slice(0, 2), torch.tensor([0, 2])])
    def test_lazy_set_at_non_tensor(self, member_batch_size, index):
        # An index along the stack dim writes whole members, with the index ()
        lazy = LazyStackedTensorDict.lazy_stack(
            [TensorDict(a="s0", batch_size=member_batch_size) for _ in range(3)]
        )
        dense = TensorDict(a="s0", batch_size=(3, *member_batch_size))
        lazy.set_at_("a", "s1", index)
        dense.set_at_("a", "s1", index)
        assert lazy.get("a").tolist() == dense.get("a").tolist()

    @pytest.mark.parametrize("member_batch_size", [(), (2,)])
    @pytest.mark.parametrize("value", [[1, 2, 3], torch.tensor([1, 2])], ids=type)
    @pytest.mark.parametrize(
        "index",
        [(), slice(0, 2), 1, [0, 2], torch.tensor([True, False, True]), Ellipsis],
    )
    def test_lazy_set_at_non_tensor_list(self, member_batch_size, value, index):
        # A value with more dims than the non-tensor entry is written whole
        # into each element, as in a dense tensordict, not split over the
        # members, which also raised when its size did not match (#2522)
        dense = TensorDict(a="s0", batch_size=(3, *member_batch_size))
        lazy = lazy_stack(list(dense.clone().unbind(0)))
        for td in (dense, lazy):
            if isinstance(value, list):
                # a list is cast to a tensor until 0.17
                with pytest.warns(FutureWarning, match="storing it as non-tensor data"):
                    td.set_at_("a", value, index)
            else:
                td.set_at_("a", value, index)
        written = torch.arange(3)[index].reshape(-1).tolist()
        for td in (dense, lazy):
            for i in range(3):
                elements = td[i].get("a").tolist()
                if not member_batch_size:
                    elements = [elements]
                for element in elements:
                    if i in written:
                        assert torch.equal(element, torch.as_tensor(value))
                    else:
                        assert element == "s0"

    @pytest.mark.parametrize(
        "index",
        [1, slice(0, 2), torch.tensor([0, 2]), (slice(None), 0), (Ellipsis, 1)],
    )
    @pytest.mark.parametrize("entry_dims", [(), (5,)])
    def test_set_at_non_tensor_expand(self, index, entry_dims):
        # The string is converted to a NonTensorData with the batch size of
        # the tensordict, which must not be expanded to the indexed batch
        # size, a shape that torch.Tensor.expand rejects (#2175)
        batch_size = (3, 4, 2)
        tds = [
            TensorDict(
                a=NonTensorData("s0", batch_size=(*batch_size, *entry_dims)),
                batch_size=batch_size,
            )
        ]
        if not entry_dims:
            tds.append(
                LazyStackedTensorDict.lazy_stack(
                    [TensorDict(a="s0", batch_size=batch_size[1:]) for _ in range(3)]
                )
            )
        expected = np.full((*batch_size, *entry_dims), "s0", dtype=object)
        expected[
            tuple(
                idx.numpy() if isinstance(idx, torch.Tensor) else idx
                for idx in (index if isinstance(index, tuple) else (index,))
            )
        ] = "s1"
        for td in tds:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                td.set_at_("a", "s1", index)
            assert td.get("a").tolist() == expected.tolist()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
