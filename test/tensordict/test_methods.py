# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import importlib.util
import io
import json
import os
import pickle
import platform
import re
import stat
import struct
import sys
import sysconfig
import warnings
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
import tensordict._archive as tensordict_archive
import tensordict.base as tensordict_base
import torch
from tensordict import (
    is_memmap_archive,
    is_tensorclass,
    lazy_legacy,
    lazy_stack,
    LazyStackedTensorDict,
    load_memmap,
    MemoryMappedTensor,
    pack_memmap,
    PersistentTensorDict,
    refresh_archive_checksums,
    TensorClass,
    TensorDict,
    TypedTensorDict,
    unpack_memmap,
)
from tensordict._archive import _ArchiveReader
from tensordict._indexing import convert_ellipsis_to_idx
from tensordict._lazy import _CustomOpTensorDict
from tensordict._td import _str_to_index, _SubTensorDict, is_tensor_collection
from tensordict._torch_func import _stack as stack_td
from tensordict.base import _is_leaf_nontensor, TensorDictBase
from tensordict.functional import pad
from tensordict.nn import TensorDictParams
from tensordict.tensorclass import NonTensorData, NonTensorDataBase, NonTensorStack
from tensordict.utils import (
    _check_recursive_properties,
    _getitem_batch_size,
    _LOCK_ERROR,
    assert_allclose_td,
    expand_as_right,
    is_non_tensor,
    set_lazy_legacy,
)
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.autograd import forward_ad as fwAD
from torch.overrides import TorchFunctionMode

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.tensordict.test._utils_internal import (
        decompose,
        DummyPicklableClass,
        get_available_devices,
        is_npu_available,
        legacy_lazy_mode,
        prod,
        TestTensorDictsBase,
    )
else:
    IS_FB = False
    from _utils_internal import (
        decompose,
        DummyPicklableClass,
        get_available_devices,
        is_npu_available,
        legacy_lazy_mode,
        prod,
        TestTensorDictsBase,
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


def _compare_tensors_identity(td0, td1):
    if isinstance(td0, LazyStackedTensorDict):
        if not isinstance(td1, LazyStackedTensorDict):
            return False
        for _td0, _td1 in zip(td0.tensordicts, td1.tensordicts):
            if not _compare_tensors_identity(_td0, _td1):
                return False
        return True
    if td0 is td1:
        return True
    for key, val in td0.items():
        if is_tensor_collection(val):
            if not _compare_tensors_identity(val, td1.get(key)):
                return False
        else:
            if val is not td1.get(key):
                return False
    else:
        return True


@pytest.mark.parametrize(
    "td_name,device",
    TestTensorDictsBase.TYPES_DEVICES,
)
class TestTensorDicts(TestTensorDictsBase):
    def test_recursive_properties_common_ops(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)

        def check(result):
            if isinstance(result, (tuple, list)):
                for item in result:
                    check(item)
            elif isinstance(result, TensorDictBase):
                _check_recursive_properties(result)
            return result

        check(td.clone())
        dense_td = check(td.to_tensordict(retain_none=True))
        op_td = dense_td if td_name == "td_h5" else td
        check(op_td.select(*list(op_td.keys()), strict=False))
        check(op_td.exclude())
        check(op_td.apply(lambda x: x))
        check(op_td[:])
        check(op_td[0])
        check(op_td.reshape(-1))
        check(torch.stack([op_td, op_td.contiguous()], 0))
        check(torch.cat([op_td, op_td.clone()], 0))
        check(op_td.unbind(0))
        check(op_td.split(1, 0))
        check(op_td.chunk(2, 0))

        if td_name not in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
        ):
            check(op_td.unsqueeze(0))
            check(op_td.permute(*range(op_td.batch_dims - 1, -1, -1)))

        td_updated = op_td.clone()
        with td_updated.unlock_():
            td_updated.update(
                {"recursive_check": torch.zeros(td_updated.shape, device=device)}
            )
        check(td_updated)

    @pytest.mark.skipif(PYTORCH_TEST_FBCODE, reason="vmap now working in fbcode")
    @pytest.mark.parametrize("nested", [False, True])
    def test_add_batch_dim_cache(self, td_name, device, nested):
        td = getattr(self, td_name)(device)
        if nested:
            td = TensorDict({"parent": td}, td.batch_size)
        from tensordict.nn import TensorDictModule  # noqa
        from torch import vmap

        fun = vmap(lambda x: x)
        if td_name == "td_h5":
            with pytest.raises(
                RuntimeError, match="PersistentTensorDict cannot be used with vmap"
            ):
                fun(td)
            return
        if td_name == "memmap_td" and device.type != "cpu":
            with pytest.raises(
                RuntimeError,
                match="MemoryMappedTensor with non-cpu device are not supported in vmap ops",
            ):
                fun(td)
            return
        fun(td)

        td.zero_()
        # this value should be cached
        std = fun(td)
        for value in std.values(True, True):
            assert (value == 0).all()

    @pytest.mark.parametrize("inplace", [False, True])
    def test_apply(self, td_name, device, inplace):
        td = getattr(self, td_name)(device)
        td_c = td.to_tensordict(retain_none=True)
        if inplace and td_name == "td_params":
            with pytest.raises(ValueError, match="Failed to update"):
                td.apply(lambda x: x + 1, inplace=inplace)
            return
        td_1 = td.apply(lambda x: x + 1, inplace=inplace)
        if inplace:
            for key in td.keys(True, True):
                assert (td_c[key] + 1 == td[key]).all()
                assert (td_1[key] == td[key]).all()
        else:
            for key in td.keys(True, True):
                assert (td_c[key] + 1 != td[key]).any()
                assert (td_1[key] == td[key] + 1).all()

    @pytest.mark.parametrize("inplace", [False, True])
    def test_apply_default(self, td_name, device, inplace):
        if td_name in ("td_h5",):
            pytest.skip("Cannot test assignment in persistent tensordict.")
        td = getattr(self, td_name)(device)
        td_c = td.to_tensordict(retain_none=True)
        if td_name in ("td_params",):
            td.data.zero_()
        else:
            td.zero_()
        with td.unlock_():
            td["nested", "newkey"] = torch.zeros(td.shape)

        def get_old_val(newval, oldval):
            if oldval is not None:
                return oldval
            return newval

        if inplace and td_name == "td_params":
            with pytest.raises(ValueError, match="Failed to update"):
                td.apply(get_old_val, td_c, inplace=inplace, default=None)
            return
        with td.unlock_() if inplace else contextlib.nullcontext():
            td_1 = td.apply(get_old_val, td_c, inplace=inplace, default=None)
        if inplace:
            for key in td.keys(True, True):
                td_c_val = td_c.get(key)
                if td_c_val is not None:
                    assert (td_c[key] == td[key]).all()
                else:
                    assert key == ("nested", "newkey")
                    assert (td_1[key] == 0).all()
                assert (td_1[key] == td[key]).all()
        else:
            for key in td.keys(True, True):
                td_c_val = td_c.get(key)
                if td_c_val is not None:
                    assert (td_c[key] == td_1[key]).all()
                else:
                    assert key == ("nested", "newkey")
                    assert (td_1[key] == 0).all()

    @pytest.mark.parametrize("inplace", [False, True])
    def test_apply_filter(self, td_name, device, inplace):
        td = getattr(self, td_name)(device)
        assert td.apply(lambda x: None, filter_empty=False) is not None
        assert (
            td.apply(lambda x: None, filter_empty=True, is_leaf=_is_leaf_nontensor)
            is None
        )

    @pytest.mark.parametrize("inplace", [False, True])
    @pytest.mark.parametrize(
        "named,nested_keys", [[False, False], [True, False], [True, True]]
    )
    def test_apply_multithread(self, td_name, device, inplace, named, nested_keys):
        td = getattr(self, td_name)(device)
        if not named:
            func = lambda x: x + 1
        else:

            def func(name, x):
                return x + 1 + isinstance(name, tuple)

        td0 = td._fast_apply(func, named=named, nested_keys=nested_keys, checked=False)
        if isinstance(td, TensorDictParams):
            td = td.data
        td1 = td._fast_apply(
            func,
            inplace=inplace,
            named=named,
            nested_keys=nested_keys,
            num_threads=2,
            checked=False,
        )
        if inplace:
            assert td1 is td
        assert (td0 == td1).all()

    @pytest.mark.parametrize("checked", [False, True])
    def test_apply_multithread_exception(self, td_name, device, checked):
        td = getattr(self, td_name)(device)

        def func(*args):
            raise RuntimeError("Test")

        with pytest.raises(RuntimeError, match="Test"):
            td._fast_apply(func, checked=checked)

    @pytest.mark.parametrize("inplace", [False, True])
    def test_apply_multithread_filter_empty(self, td_name, device, inplace):
        td = getattr(self, td_name)(device)

        def func(x):
            return None

        result0 = td._fast_apply(
            func, filter_empty=True, inplace=inplace, num_threads=0, checked=False
        )
        result1 = td._fast_apply(
            func, filter_empty=True, inplace=inplace, num_threads=2, checked=False
        )
        if not td._has_non_tensor:
            # outplace, no non-tensor
            assert result0 is None
            assert result1 is None, (td, result1)
        elif inplace:
            # inplace, has non tensor
            assert result0 is td
            assert result1 is td, result1
        else:
            # not inplace, has non tensor
            assert result0 is not None
            assert result1 is not None, result0

    @pytest.mark.parametrize("inplace", [False, True])
    def test_apply_other(self, td_name, device, inplace):
        td = getattr(self, td_name)(device)
        td_c = td.to_tensordict(retain_none=True)
        if inplace and td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_1 = td_set.apply(lambda x, y: x + y, td_c, inplace=inplace)
        if inplace:
            for key in td.keys(True, True):
                assert (td_c[key] * 2 == td[key]).all()
                assert (td_1[key] == td[key]).all()
        else:
            for key in td.keys(True, True):
                assert (td_c[key] * 2 != td[key]).any()
                assert (td_1[key] == td[key] * 2).all()

    def test_apply_out(self, td_name, device):
        td = getattr(self, td_name)(device)
        if not isinstance(td, LazyStackedTensorDict):
            td_c = td.to_tensordict(retain_none=True)
        else:
            td_c = td.clone()
        td.apply(lambda x: x + 1, out=td_c)
        assert_allclose_td(
            td.filter_non_tensor_data().data + 1,
            td_c.filter_non_tensor_data().filter_empty_(),
        )

    def test_as_tensor(self, td_name, device):
        td = getattr(self, td_name)(device)
        # MemoryMappedTensor leaves are kept with a FutureWarning, since they
        # become plain tensors in 0.17 (#2979); other tensordicts stay silent.
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            tdt = td.as_tensor()
        record = [w for w in record if issubclass(w.category, FutureWarning)]
        if "memmap" in td_name:
            assert len(record) == 1
            assert "Starting with TensorDict 0.17" in str(record[0].message)
            assert record[0].filename == __file__
            assert type(tdt["a"]) is MemoryMappedTensor
            assert tdt["a"].data_ptr() == td["a"].data_ptr()
            assert (tdt == td).all()
        else:
            assert record == []

    def test_assert(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        with pytest.raises(
            RuntimeError,
            match=r"Converting a tensordict to boolean value is not permitted. "
            r"Use len\(td.keys\(\)\) to test for keys, td.is_empty\(\) to test "
            r"for leaves, or \(a == b\).all\(\) / .any\(\) to reduce a comparison.",
        ):
            assert td

    def test_auto_batch_size_(self, td_name, device):
        td = getattr(self, td_name)(device)
        batch_size = td.batch_size
        error = None
        try:
            td.batch_size = []
        except Exception as err:
            error = err
        if error is not None:
            # a lazy tensordict cannot change its batch size: auto_batch_size_
            # finds the batch size that it has and returns, and raises for a
            # smaller one
            assert isinstance(
                td, (LazyStackedTensorDict, _SubTensorDict, _CustomOpTensorDict)
            )
            td.auto_batch_size_()
            assert td.batch_size == batch_size
            with pytest.raises(type(error)):
                td.auto_batch_size_(1)
            return
        td.auto_batch_size_()
        assert td.batch_size[: len(batch_size)] == batch_size
        td.auto_batch_size_(1)
        assert len(td.batch_size) == 1

    def test_broadcast(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        sub_td = td[:, :2].to_tensordict(retain_none=True)
        sub_td.zero_()
        sub_dict = sub_td.to_dict()
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:, :2] = sub_dict
        assert (td[:, :2] == 0).all()

    @pytest.mark.parametrize("op", ["flatten", "unflatten"])
    def test_cache(self, td_name, device, op):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        try:
            td.lock_()
        except Exception:
            return
        if op == "keys_root":
            a = list(td.keys())
            b = list(td.keys())
            assert a == b
        elif op == "keys_nested":
            a = list(td.keys(True))
            b = list(td.keys(True))
            assert a == b
        elif op == "values":
            a = list(td.values(True))
            b = list(td.values(True))
            assert all((_a == _b).all() for _a, _b in zip(a, b))
        elif op == "items":
            keys_a, values_a = zip(*td.items(True))
            keys_b, values_b = zip(*td.items(True))
            assert all((_a == _b).all() for _a, _b in zip(values_a, values_b))
            assert keys_a == keys_b
        elif op == "flatten":
            a = td.flatten_keys()
            b = td.flatten_keys()
            # The result is a copy of the cached value: a write into it must
            # not show up in later calls (gh-2375).
            assert a is not b
            assert set(a.keys(True)) == set(b.keys(True))
            a["_new_key"] = torch.zeros(a.batch_size, device=a.device)
            assert "_new_key" not in td.flatten_keys().keys()
        elif op == "unflatten":
            a = td.unflatten_keys()
            b = td.unflatten_keys()
            # The result is a copy of the cached value: a write into it must
            # not show up in later calls (gh-2375).
            assert a is not b
            assert set(a.keys(True)) == set(b.keys(True))
            a["_new_key"] = torch.zeros(a.batch_size, device=a.device)
            assert "_new_key" not in td.unflatten_keys().keys()

        if td_name not in ("td_params", "td_h5"):
            assert len(td._cache)
        td.unlock_()
        assert td._cache is None
        for val in td.values(True):
            if is_tensor_collection(val):
                assert td._cache is None

    @pytest.mark.skipif(
        torch.cuda.device_count() == 0 and npu_device_count == 0,
        reason="no cuda or npu device detected",
    )
    @pytest.mark.parametrize("device_cast", get_available_devices())
    @pytest.mark.parametrize(
        "non_blocking_pin",
        (
            [False]
            if not torch.cuda.is_available() and not is_npu_available()
            else [False, True]
        ),
    )
    @pytest.mark.parametrize("num_threads", [0, 1, 4, None])
    def test_cast_device(
        self, td_name, device, device_cast, non_blocking_pin, num_threads
    ):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if non_blocking_pin and td_name == "td_h5":
            with pytest.raises(
                RuntimeError,
                match="Cannot use non_blocking_pin=True PersistentTensorDict.to()",
            ):
                td_device = td.to(
                    device_cast,
                    non_blocking_pin=non_blocking_pin,
                    num_threads=num_threads,
                )
            return

        if device.type == "cuda" and device_cast.type == "cpu" and non_blocking_pin:
            with pytest.raises(
                RuntimeError, match="only dense CPU tensors can be pinned"
            ):
                td_device = td.to(
                    device_cast,
                    non_blocking_pin=non_blocking_pin,
                    num_threads=num_threads,
                )
            return

        if device.type == "npu" and device_cast != td.device and non_blocking_pin:
            with pytest.raises(RuntimeError, match="cannot pin"):
                _ = td.to(
                    device_cast,
                    non_blocking_pin=non_blocking_pin,
                    num_threads=num_threads,
                )
            return
        td_device = td.to(
            device_cast, non_blocking_pin=non_blocking_pin, num_threads=num_threads
        )

        for item in td_device.values():
            assert item.device == device_cast
        for item in td_device.clone().values():
            assert item.device == device_cast

        assert td_device.device == device_cast, (
            f"td_device first tensor device is {next(td_device.items())[1].device}"
        )
        assert td_device.clone().device == device_cast
        if device_cast != td.device:
            assert td_device is not td
        assert td_device.to(device_cast) is td_device
        assert td.to(device) is td
        assert_allclose_td(td, td_device.to(device))

    def test_cast_to(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            # UnbatchedTensor values() iteration includes UnbatchedTensor which may have issues
            pytest.skip("UnbatchedTensor cast_to needs investigation")
            return
        td_device = td.to("cpu:1")
        assert td_device.device == torch.device("cpu:1")
        td_dtype = td.to(torch.int)
        assert all(t.dtype == torch.int for t in td_dtype.values(True, True))
        del td_dtype
        # device (str), dtype
        td_dtype_device = td.to("cpu:1", torch.int)
        assert all(t.dtype == torch.int for t in td_dtype_device.values(True, True))
        assert td_dtype_device.device == torch.device("cpu:1")
        del td_dtype_device
        # device, dtype
        td_dtype_device = td.to(torch.device("cpu:1"), torch.int)
        assert all(t.dtype == torch.int for t in td_dtype_device.values(True, True))
        assert td_dtype_device.device == torch.device("cpu:1")
        del td_dtype_device
        # example tensor
        td_dtype_device = td.to(torch.randn(3, dtype=torch.half, device="cpu:1"))
        assert all(t.dtype == torch.half for t in td_dtype_device.values(True, True))
        # tensor on cpu:1 is actually on cpu. This is still meaningful for tensordicts on cuda.
        assert td_dtype_device.device == torch.device("cpu")
        del td_dtype_device
        # example td
        td_dtype_device = td.to(
            other=TensorDict(
                {"a": torch.randn(3, dtype=torch.half, device="cpu:1")},
                [],
                device="cpu:1",
            )
        )
        assert all(t.dtype == torch.half for t in td_dtype_device.values(True, True))
        assert td_dtype_device.device == torch.device("cpu:1")
        del td_dtype_device
        # example td, many dtypes
        td_nodtype_device = td.to(
            other=TensorDict(
                {
                    "a": torch.randn(3, dtype=torch.half, device="cpu:1"),
                    "b": torch.randint(10, ()),
                },
                [],
                device="cpu:1",
            )
        )
        assert all(t.dtype != torch.half for t in td_nodtype_device.values(True, True))
        assert td_nodtype_device.device == torch.device("cpu:1")
        del td_nodtype_device
        # batch-size: check errors (or not)
        if td_name in (
            "stacked_td",
            "unsqueezed_td",
            "squeezed_td",
            "permute_td",
            "nested_stacked_td",
        ):
            with pytest.raises(TypeError, match="Cannot pass batch-size to "):
                td_dtype_device = td.to(
                    torch.device("cpu:1"), torch.int, batch_size=torch.Size([])
                )
        else:
            # These have batched nested entries, which keep their batch size
            # in 0.17 instead of getting batch_size=[]
            with (
                pytest.warns(FutureWarning, match="TensorDict 0.17")
                if td_name
                in (
                    "nested_td",
                    "nested_tensorclass",
                    "td_h5",
                    "td_with_non_tensor",
                    "td_with_non_tensor_and_metadata",
                    "nested_typed_td",
                )
                else contextlib.nullcontext()
            ):
                td_dtype_device = td.to(
                    torch.device("cpu:1"), torch.int, batch_size=torch.Size([])
                )
            assert all(t.dtype == torch.int for t in td_dtype_device.values(True, True))
            assert td_dtype_device.device == torch.device("cpu:1")
            assert td_dtype_device.batch_size == torch.Size([])
            del td_dtype_device
        if td_name in (
            "stacked_td",
            "unsqueezed_td",
            "squeezed_td",
            "permute_td",
            "nested_stacked_td",
        ):
            with pytest.raises(TypeError, match="Cannot pass batch-size to "):
                td.to(batch_size=torch.Size([]))
        else:
            td_batchsize = td.to(batch_size=torch.Size([]))
            assert td_batchsize.batch_size == torch.Size([])
            del td_batchsize

    def test_to_batch_size_per_class(self, td_name, device):
        # to(batch_size=...) without a device or dtype: a lazy stack raises, a
        # sub-tensordict returns a TensorDict copy (also for the same batch
        # size), and the other types set the batch size and return themselves
        td = getattr(self, td_name)(device)
        batch_size = td.batch_size[:-1]
        if td_name in (
            "stacked_td",
            "unsqueezed_td",
            "squeezed_td",
            "permute_td",
            "nested_stacked_td",
        ):
            with pytest.raises(TypeError, match="Cannot pass batch-size to "):
                td.to(batch_size=batch_size)
            with pytest.raises(TypeError, match="Cannot pass batch-size to "):
                td.to("meta", batch_size=batch_size)
            return
        is_sub = td_name in ("sub_td", "sub_td2")
        old_batch_size = td.batch_size
        same = td.to(batch_size=old_batch_size)
        assert (same is not td) if is_sub else (same is td)
        out = td.to(batch_size=batch_size)
        assert out.batch_size == batch_size
        if is_sub:
            assert type(same) is TensorDict and type(out) is TensorDict
            assert td.batch_size == old_batch_size
        else:
            assert out is td
        with warnings.catch_warnings():
            # nested tensordicts get the batch size (the 0.17 FutureWarning)
            warnings.simplefilter("ignore", FutureWarning)
            meta = getattr(self, td_name)(device).to("meta", batch_size=batch_size)
        assert type(meta) is (TensorDict if is_sub else type(td))
        assert meta.device == torch.device("meta")
        assert meta.batch_size == batch_size

    @pytest.mark.skipif(
        is_npu_available(), reason="torch.double is not adapted on NPU currently"
    )
    def test_casts(self, td_name, device):
        td = getattr(self, td_name)(device)
        # exclude non-tensor data
        is_leaf = lambda cls: issubclass(cls, torch.Tensor)
        tdfloat = td.float()
        assert all(
            value.dtype is torch.float
            for value in tdfloat.values(True, True, is_leaf=is_leaf)
        )
        tddouble = td.double()
        assert all(
            value.dtype is torch.double
            for value in tddouble.values(True, True, is_leaf=is_leaf)
        )
        tdbfloat16 = td.bfloat16()
        assert all(
            value.dtype is torch.bfloat16
            for value in tdbfloat16.values(True, True, is_leaf=is_leaf)
        )
        tdhalf = td.half()
        assert all(
            value.dtype is torch.half
            for value in tdhalf.values(True, True, is_leaf=is_leaf)
        )
        tdint = td.int()
        assert all(
            value.dtype is torch.int
            for value in tdint.values(True, True, is_leaf=is_leaf)
        )
        with (
            pytest.warns(FutureWarning, match="type\\(\\) returns an unlocked")
            if td.is_locked
            else contextlib.nullcontext()
        ):
            tdint = td.type(torch.int)
        assert all(
            value.dtype is torch.int
            for value in tdint.values(True, True, is_leaf=is_leaf)
        )

    @pytest.mark.parametrize("cast_filter", [None, lambda key, value: True])
    def test_type_locked(self, td_name, device, cast_filter):
        # type() of a locked tensordict is unlocked, unlike int(): a
        # FutureWarning says it stays locked from 0.17 (#2959)
        td = getattr(self, td_name)(device)
        if td.is_locked:
            td.unlock_()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            td.type(torch.int, cast_filter=cast_filter)
        assert not [x for x in w if "type() returns an unlocked" in str(x.message)]
        try:
            td.lock_()
        except RuntimeError as err:
            if "Cannot lock" not in str(err):
                raise
            pytest.skip(str(err))
        with pytest.warns(FutureWarning, match="type\\(\\) returns an unlocked"):
            tdint = td.type(torch.int, cast_filter=cast_filter)
        assert not tdint.is_locked
        assert td.int().is_locked
        is_leaf = lambda cls: issubclass(cls, torch.Tensor)
        assert all(
            value.dtype is torch.int
            for value in tdint.values(True, True, is_leaf=is_leaf)
        )

    @pytest.mark.parametrize("keep_entries", [False, True, None])
    def test_cat_tensors(self, td_name, device, keep_entries):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        with td.unlock_():
            a = td.pop("a")
            td["a"] = a.unsqueeze(-1)
            td["a_bis"] = td["a"] + 1
            kwargs = {}
            if keep_entries is not None:
                kwargs["keep_entries"] = keep_entries
            pred_stack = torch.cat([td["a"], td["a_bis"]], -1)
            td.cat_tensors("a", "a_bis", out_key="cat", dim=-1, **kwargs)
            assert (td["cat"] == pred_stack).all()
            if keep_entries:
                assert "a" in td
            else:
                assert "a" not in td

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("chunks", [1, 2])
    def test_chunk(self, td_name, device, dim, chunks):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if len(td.shape) - 1 < dim:
            pytest.skip(f"no dim {dim} in td")

        chunks = min(td.shape[dim], chunks)
        td_chunks = td.chunk(chunks, dim)
        assert len(td_chunks) == chunks
        assert sum([_td.shape[dim] for _td in td_chunks]) == td.shape[dim]
        assert (torch.cat(td_chunks, dim) == td).all()

    def test_clamp(self, td_name, device):
        td = getattr(self, td_name)(device)
        tdc = td.clamp(-1, 1)
        assert (tdc <= 1).all()
        assert (tdc >= -1).all()
        if td.requires_grad:
            td = td.detach()
        tdc = td.clamp(None, 1)
        assert (tdc <= 1).all()
        tdc = td.clamp(-1)
        assert (tdc >= -1).all()

    def test_clear(self, td_name, device):
        td = getattr(self, td_name)(device)
        with td.unlock_():
            tdc = td.clear()
            assert tdc.is_empty()
            assert tdc is td

    def test_clear_device_(self, td_name, device):
        td = getattr(self, td_name)(device)
        leaf_devices = [leaf.device for leaf in td.values(True, True)]
        assert td.clear_device_() is td
        if not isinstance(td, PersistentTensorDict):
            # the leaves are not moved (a file reads its leaves onto the device
            # of the tensordict, so without one they come back as stored)
            assert [leaf.device for leaf in td.values(True, True)] == leaf_devices
        if isinstance(td, (_SubTensorDict, _CustomOpTensorDict, TypedTensorDict)):
            # a view reports the device of the tensordict it reads, which
            # clear_device_ leaves alone
            assert td.device == torch.device(device)
            return
        assert td.device is None
        for value in td.values(True):
            if is_tensor_collection(value):
                assert value.device is None

    def test_clone_td(self, td_name, device, tmp_path):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_h5":
            # need a new file
            newfile = tmp_path / "file.h5"
            clone = td.clone(newfile=newfile)
        else:
            clone = torch.clone(td)
        assert (clone == td).all()
        assert td.batch_size == clone.batch_size
        assert type(td.clone(recurse=False)) is type(td)
        if td_name in (
            "stacked_td",
            "nested_stacked_td",
            "saved_td",
            "squeezed_td",
            "unsqueezed_td",
            "sub_td",
            "sub_td2",
            "permute_td",
            "td_h5",
        ):
            assert td.clone(recurse=False).get("a") is not td.get("a")
        else:
            assert td.clone(recurse=False).get("a") is td.get("a")

    @pytest.mark.parametrize("locked", [False, True])
    @pytest.mark.parametrize("named", [False, True])
    def test_contiguous_result_state(self, td_name, device, locked, named):
        td = getattr(self, td_name)(device)
        if (
            named
            and td_name != "td_with_unbatched"
            and not isinstance(td, _CustomOpTensorDict)
        ):
            # an UnbatchedTensor has no batch dims to name, and the names of a
            # legacy lazy view cannot be set
            if isinstance(td, _SubTensorDict):
                td._source.names = [f"n{i}" for i in range(td._source.ndim)]
            else:
                td.names = ["n0", "n1", "n2", "n3"]
        if locked:
            (td._source if isinstance(td, _SubTensorDict) else td).lock_()
        result = td.contiguous()
        if isinstance(td, TypedTensorDict):
            assert type(result) is type(td)
        else:
            assert type(result) is TensorDict
        assert result.batch_size == td.batch_size
        assert result.names == td.names
        assert (result == td).all()
        for key, value in td.items():
            if is_tensorclass(value):
                assert type(result.get(key)) is type(value)
        # only the legacy lazy views keep the lock
        assert result.is_locked is (locked and isinstance(td, _CustomOpTensorDict))

    @pytest.mark.skipif(
        torch.cuda.device_count() == 0 and npu_device_count == 0,
        reason="no cuda or npu device detected",
    )
    def test_cpu_cuda(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if torch.cuda.is_available():
            td_device = td.cuda()
        elif is_npu_available():
            td_device = td.to("npu:0")
        td_back = td_device.cpu()
        assert td_device.device == torch.device(f"{cur_device}:0")
        assert td_back.device == torch.device("cpu")

    # getting values from lazy tensordicts in non-lazy contexts messes things up
    # so we set it to True. When we'll deprecate lazy tensordicts, we will just
    # remove this decorator
    @legacy_lazy_mode()
    def test_create_nested(self, td_name, device):
        td = getattr(self, td_name)(device)
        with td.unlock_():
            td.create_nested("root")
            assert td.get("root").shape == td.shape
            assert is_tensor_collection(td.get("root"))
            td.create_nested(("some", "nested", "key"))

            some = td.get("some")
            nested = some.get("nested")
            _ = nested.get("key")
            assert td.get(("some", "nested", "key")).shape == td.shape
            assert is_tensor_collection(td.get(("some", "nested", "key")))
            # The plain TensorDicts already on the path keep their entries.
            # Views, typed tensordicts and files replace each level.
            if td_name not in (
                "sub_td",
                "sub_td2",
                "permute_td",
                "unsqueezed_td",
                "squeezed_td",
                "td_h5",
                "typed_td",
                "nested_typed_td",
            ):
                leaf = torch.ones(td.shape)
                td.set(("some", "leaf"), leaf)
                td.create_nested(("some", "other"))
                td.create_nested(("some", "nested", "key2"))
                keys = set(td.keys(include_nested=True))
                assert {
                    ("some", "nested", "key"),
                    ("some", "nested", "key2"),
                    ("some", "other"),
                } <= keys
                assert (td.get(("some", "leaf")) == 1).all()
            del td["root"]
        if td_name in ("sub_td", "sub_td2"):
            return

        with td.lock_(), pytest.raises(RuntimeError):
            td.create_nested("root")

    def test_data_ptr(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        assert td.data_ptr().batch_size == torch.Size(())
        assert td.data_ptr(storage=True).batch_size == torch.Size(())

    def test_default_nested(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        default_val = torch.randn(())
        timbers = td.get(("shiver", "my", "timbers"), default_val)
        assert timbers == default_val

    def test_delitem(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name in ("memmap_td",):
            with pytest.raises(RuntimeError, match="Cannot modify"):
                del td["a"]
            return
        del td["a"]
        assert "a" not in td.keys()

    @pytest.mark.parametrize("recurse", [False, True])
    def test_empty(self, td_name, device, recurse):
        td = getattr(self, td_name)(device)
        td_empty = td.empty(recurse=recurse)
        assert not td_empty.is_locked
        assert td_empty.shape == td.shape
        assert td_empty.names == td.names
        assert td_empty.device == td.device
        td_empty.set("a", torch.zeros(()).expand(td.shape))
        batch_size = torch.Size([*td.batch_size, 1])
        names = [f"dim{i}" for i in range(len(batch_size))]
        td_empty = td.empty(
            recurse=recurse, batch_size=batch_size, names=names, device="meta"
        )
        assert td_empty.batch_size == batch_size
        assert td_empty.names == names
        assert td_empty.device == torch.device("meta")

    @pytest.mark.parametrize("recurse", [False, True])
    def test_empty_names_none(self, td_name, device, recurse):
        # names=None drops the names of a TensorDict, and keeps them on the
        # other classes (and with recurse=True)
        td = getattr(self, td_name)(device)
        if isinstance(td, (_SubTensorDict, _CustomOpTensorDict)):
            # their names cannot be set
            return
        names = ["w", "x", "y", "z"]
        td.names = names
        assert td.empty(recurse).names == names
        dropped = type(td) is TensorDict and not recurse
        assert td.empty(recurse, names=None).names == ([None] * 4 if dropped else names)

    def test_setitem_self_nesting(self, td_name, device):
        td = getattr(self, td_name)(device)
        if isinstance(td, (TypedTensorDict, TensorDictParams)):
            # they write into their source, which is not the value
            return
        with pytest.raises(
            ValueError,
            match=re.escape("Cannot set a tensordict inside itself (key 'me')"),
        ):
            td["me"] = td
        assert "me" not in td.keys()

    def test_update_self_nesting(self, td_name, device):
        # #2741: update() stored the tensordict in itself, and the next
        # repr or clone raised a RecursionError
        td = getattr(self, td_name)(device)
        td.unlock_()
        td["nested"] = TensorDict(
            z=torch.zeros(td.batch_size, device=device), batch_size=td.batch_size
        )
        keys = set(td.keys(True))
        # an existing sub-tensordict is left unchanged too
        for source in ({"me": td}, {("nested", "me"): td}, {"nested": td}):
            with pytest.raises(
                ValueError, match=re.escape("Cannot set a tensordict inside itself")
            ):
                td.update(source)
        assert set(td.keys(True)) == keys
        # a copy is not the tensordict itself
        td.update({"me": td}, clone=True)
        assert "me" in td.keys()

    def test_setitem_key_existing_inplace(self, td_name, device):
        # td[key] = value replaces the entry; a sub-tensordict writes it into
        # its source in place, and an h5 file into the file
        td = getattr(self, td_name)(device)
        td.unlock_()
        before = td.get("a")
        value = torch.full_like(before, 3.0)
        td["a"] = value
        assert (td.get("a") == 3).all()
        if isinstance(td, _SubTensorDict):
            assert td.get("a").data_ptr() == before.data_ptr()
            assert (before == 3).all()
        elif isinstance(td, (LazyStackedTensorDict, PersistentTensorDict)):
            assert not (before == 3).all()
        else:
            assert td.get("a").data_ptr() == value.data_ptr()
            assert not (before == 3).all()
        td["new"] = value
        assert td.get("new").shape == value.shape

    def test_empty_like(self, td_name, device):
        if "sub_td" in td_name:
            # we do not call skip to avoid systematic skips in internal code base
            return
        td = getattr(self, td_name)(device)
        if isinstance(td, _CustomOpTensorDict):
            # we do not call skip to avoid systematic skips in internal code base
            return
        td_empty = torch.empty_like(td)

        td.apply_(lambda x: x + 1.0)
        # assert type(td) is type(td_empty)
        # exclude non tensor data
        comp = td.filter_non_tensor_data() != td_empty.filter_non_tensor_data()
        assert all(val.any() for val in comp.values(True, True))

        td_empty = torch.empty_like(td, device="meta")
        assert td_empty.device == torch.device("meta")

        def assert_meta(x):
            assert x.device == torch.device("meta")

        td_empty.apply(assert_meta, filter_empty=True)

    def test_enter_exit(self, td_name, device):
        torch.manual_seed(1)
        if td_name in ("sub_td", "sub_td2"):
            return
        td = getattr(self, td_name)(device)
        is_locked = td.is_locked
        with td.lock_() as other:
            assert other is td
            assert td.is_locked
            with td.unlock_() as other:
                assert other is td
                assert not td.is_locked
            assert td.is_locked
        assert td.is_locked is is_locked

    def test_entry_type(self, td_name, device):
        td = getattr(self, td_name)(device)
        for key in td.keys(include_nested=True):
            assert type(td.get(key)) is td.entry_class(key)

    def test_equal(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        assert (td == td.to_tensordict(retain_none=True)).all()
        td0 = td.to_tensordict(retain_none=True).zero_()
        assert (td != td0).any()

    def test_equal_dict(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        assert (td == td.to_dict()).all()
        td0 = td.to_tensordict(retain_none=True).zero_().to_dict()
        assert (td != td0).any()

    def test_equal_float(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set.zero_()
        assert (td == 0.0).all()
        td0 = td.clone()
        if td_name == "td_params":
            td_set = td0.data
        else:
            td_set = td0
        td_set.zero_()
        assert (td0 != 1.0).all()

    def test_equal_int(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set.zero_()
        assert (td == 0).all()
        td0 = td.to_tensordict(retain_none=True).zero_()
        assert (td0 != 1).all()

    def test_equal_other(self, td_name, device):
        td = getattr(self, td_name)(device)
        assert not td == "z"
        assert td != "z"

    def test_equal_tensor(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set.zero_()
        assert (td == torch.zeros([], dtype=torch.int, device=device)).all()
        td0 = td.to_tensordict(retain_none=True).zero_()
        assert (td0 != torch.ones([], dtype=torch.int, device=device)).all()

    @pytest.mark.skipif(
        is_npu_available(),
        reason="ForeachAddScalar is not fully adapted on NPU currently",
    )
    def test_exclude(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_h5":
            with pytest.raises(NotImplementedError, match="Cannot call exclude"):
                _ = td.exclude("a")
            return
        td2 = td.exclude("a")
        assert td2 is not td
        assert (
            len(list(td2.keys())) == len(list(td.keys())) - 1 and "a" not in td2.keys()
        )
        assert (
            len(list(td2.clone().keys())) == len(list(td.keys())) - 1
            and "a" not in td2.clone().keys()
        )

        if td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "squeezed_td",
            "unsqueezed_td",
        ):
            with pytest.raises(RuntimeError, match="Cannot call exclude"):
                td.exclude("a", inplace=True)
            return

        with td.unlock_():
            td2 = td.exclude("a", inplace=True)
        assert td2 is td

    def test_exclude_inplace_locked(self, td_name, device):
        if td_name in (
            "td_h5",
            "sub_td",
            "sub_td2",
            "permute_td",
            "squeezed_td",
            "unsqueezed_td",
        ):
            pytest.skip(f"{td_name} does not support exclude(inplace=True)")
        td = getattr(self, td_name)(device)
        td.lock_()
        with pytest.raises(RuntimeError, match=re.escape(_LOCK_ERROR)):
            td.exclude("a", inplace=True)
        assert "a" in td.keys()
        assert td.is_locked
        if td_name != "td_params":
            # Excluding a missing key changes nothing, so it does not raise.
            # TensorDictParams raises on any in-place exclude while locked.
            assert td.exclude("this key is missing", inplace=True) is td
        td.unlock_()

    @pytest.mark.parametrize("nested", [True, False])
    def test_exclude_missing(self, td_name, device, nested):
        if td_name == "td_h5":
            raise pytest.skip("exclude not implemented for PersitentTensorDict")
        td = getattr(self, td_name)(device)
        if nested:
            td2 = td.exclude("this key is missing", ("this one too",))
        else:
            td2 = td.exclude(
                "this key is missing",
            )
        assert (td == td2).all()

    @pytest.mark.parametrize("nested", [True, False])
    def test_exclude_nested(self, td_name, device, nested):
        if td_name == "td_h5":
            raise pytest.skip("exclude not implemented for PersitentTensorDict")
        td = getattr(self, td_name)(device)
        td.unlock_()  # make sure that the td is not locked
        if td_name == "stacked_td":
            for _td in td.tensordicts:
                _td["newnested", "first"] = torch.randn(_td.shape)
        else:
            td["newnested", "first"] = torch.randn(td.shape)
        if nested:
            td2 = td.exclude("a", ("newnested", "first"))
            assert "a" in td.keys(), list(td.keys())
            assert "a" not in td2.keys()
            assert ("newnested", "first") in td.keys(True), list(td.keys(True))
            assert ("newnested", "first") not in td2.keys(True)
        else:
            td2 = td.exclude(
                "a",
            )
            assert "a" in td.keys()
            assert "a" not in td2.keys()
        if td_name not in (
            "sub_td",
            "sub_td2",
            "unsqueezed_td",
            "squeezed_td",
            "permute_td",
            "td_params",
        ):
            # TODO: document this as an edge-case: with a sub-tensordict, exclude acts on the parent tensordict
            # perhaps exclude should return an error in these cases?
            assert type(td2) is type(td)

    def test_expand(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        batch_size = td.batch_size
        expected_size = torch.Size([3, *batch_size])

        new_td = td.expand(3, *batch_size)
        assert new_td.batch_size == expected_size
        assert all((_new_td == td).all() for _new_td in new_td)

        new_td_torch_size = td.expand(expected_size)
        assert new_td_torch_size.batch_size == expected_size
        assert all((_new_td == td).all() for _new_td in new_td_torch_size)

        new_td_iterable = td.expand([3, *batch_size])
        assert new_td_iterable.batch_size == expected_size
        assert all((_new_td == td).all() for _new_td in new_td_iterable)

    def test_fill_(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        new_td = td_set.fill_("a", 0.1)
        assert (td.get("a") == 0.1).all()
        assert new_td is td_set

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("separator", [",", "-"])
    def test_flatten_keys(self, td_name, device, inplace, separator):
        td = getattr(self, td_name)(device)
        locked = td.is_locked
        td.unlock_()
        nested_nested_tensordict = TensorDict(
            {
                "a": torch.zeros(*td.shape, 2, 3),
            },
            [*td.shape, 2],
        )
        nested_tensordict = TensorDict(
            {
                "a": torch.zeros(*td.shape, 2),
                "nested_nested_tensordict": nested_nested_tensordict,
            },
            td.shape,
        )
        td["nested_tensordict"] = nested_tensordict
        if locked:
            td.lock_()

        if inplace and locked:
            with pytest.raises(RuntimeError, match="Cannot modify locked TensorDict"):
                td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
            return
        elif td_name in ("td_h5",) and inplace:
            with pytest.raises(
                ValueError,
                match="Cannot call flatten_keys in_place with a PersistentTensorDict",
            ):
                td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
            return
        else:
            if inplace and td_name in (
                "sub_td",
                "sub_td2",
                "squeezed_td",
                "unsqueezed_td",
                "permute_td",
            ):
                with pytest.raises(RuntimeError, match="Cannot call exclude"):
                    td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
                return
            expected_keys = set(td.flatten_keys(separator=separator).keys())
            td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
        # the inplace version keeps the leaves at the root too
        assert set(td_flatten.keys()) == expected_keys
        for value in td_flatten.values():
            assert not isinstance(value, TensorDictBase)
        assert (
            separator.join(["nested_tensordict", "nested_nested_tensordict", "a"])
            in td_flatten.keys()
        )
        if inplace:
            assert td_flatten is td
        else:
            assert td_flatten is not td

    def test_flatten_keys_decorator(self, td_name, device):
        td = getattr(self, td_name)(device)
        with td.flatten_keys(",") as tdflat:
            assert set(tdflat.keys(True, True, is_leaf=_is_leaf_nontensor)) == set(
                tdflat.keys(is_leaf=_is_leaf_nontensor)
            )
            with tdflat.unflatten_keys(",") as td_orig:
                assert (td_orig == td).all()
                if not td.is_locked:
                    td_orig["new", "data"] = torch.zeros(td_orig.shape)
            if not td.is_locked:
                assert (tdflat["new,data"] == 0).all()
        if not td.is_locked:
            assert (td["new", "data"] == 0).all()

    def test_flatten_keys_decorator_inplace(self, td_name, device):
        # with inplace=True, the nested keys are restored on exit too (gh-2700)
        td = getattr(self, td_name)(device)
        td.unlock_()
        keys = set(td.keys(True, True, is_leaf=_is_leaf_nontensor))
        if td_name in (
            "sub_td",
            "sub_td2",
            "squeezed_td",
            "unsqueezed_td",
            "permute_td",
            "td_h5",
        ):
            with pytest.raises((RuntimeError, ValueError)):
                td.flatten_keys(",", inplace=True)
            return
        with td.flatten_keys(",", inplace=True) as tdflat:
            assert tdflat is td
            assert set(tdflat.keys(True, True, is_leaf=_is_leaf_nontensor)) == set(
                tdflat.keys(is_leaf=_is_leaf_nontensor)
            )
        assert set(td.keys(True, True, is_leaf=_is_leaf_nontensor)) == keys

    def test_flatten_unflatten(self, td_name, device):
        td = getattr(self, td_name)(device)
        shape = td.shape[:3]
        td_flat = td.flatten(0, 2)
        td_unflat = td_flat.unflatten(0, shape)
        assert (td.to_tensordict(retain_none=True) == td_unflat).all()
        assert td.batch_size == td_unflat.batch_size

    @pytest.mark.parametrize("start_dim", [0, 1, -2, -3])
    def test_flatten_unflatten_decorator(self, td_name, device, start_dim):
        td = getattr(self, td_name)(device)
        with td.unlock_(), td.flatten(start_dim=start_dim, end_dim=3) as td_flat:
            assert (td_flat == td.flatten(start_dim, 3)).all()
            new_start_dim = -1 if start_dim in (-2, -3) else start_dim
            with td_flat.unflatten(
                dim=new_start_dim, unflattened_size=td.shape[start_dim:]
            ) as td_unflat:
                assert (td_unflat == td).all()

        with td.unlock_(), td.flatten(start_dim, end_dim=3) as td_flat:
            assert (td_flat == td.flatten(start_dim, 3)).all()
            new_start_dim = (
                -1 if start_dim == -2 else -1 if start_dim == -3 else start_dim
            )
            with td_flat.unflatten(
                new_start_dim, unflattened_size=td.shape[start_dim:]
            ) as td_unflat:
                assert (td_unflat == td).all()

        with td.unlock_(), td.flatten(start_dim, -1) as td_flat:
            assert (td_flat == td.flatten(start_dim, -1)).all()
            new_start_dim = (
                -1 if start_dim == -2 else -1 if start_dim == -3 else start_dim
            )
            with td_flat.unflatten(new_start_dim, td.shape[start_dim:]) as td_unflat:
                assert (td_unflat == td).all()

    @pytest.mark.parametrize(
        "args,kwargs",
        [
            ((0, None), {}),
            ((None,), {}),
            ((), {"start_dim": None}),
            ((), {"end_dim": None}),
        ],
    )
    def test_flatten_decorator_none_dims(self, td_name, device, args, kwargs):
        # None means the first or the last dim, also when the block exits
        td = getattr(self, td_name)(device)
        batch_size = td.batch_size
        with td.unlock_(), td.flatten(*args, **kwargs) as td_flat:
            assert (td_flat == td.flatten(0, -1)).all()
            if not td_flat.requires_grad:
                td_flat.apply_(lambda x: x * 0 + 1)
            else:
                td_flat.apply(lambda x: x.data.mul_(0).add_(1))
        assert td.batch_size == batch_size
        assert (td == 1).all()

    @pytest.mark.parametrize(
        "op,args", [("flatten", (0, 1)), ("unflatten", (0, (2, 2)))]
    )
    def test_shape_decorator_inplace(self, td_name, device, op, args):
        # with inplace=True, the shape is restored on exit too (gh-2700)
        td = getattr(self, td_name)(device)
        td.unlock_()
        batch_size = td.batch_size
        shapes = {key: val.shape for key, val in td.items(True, True)}
        if td_name in (
            "nested_stacked_td",
            "permute_td",
            "squeezed_td",
            "stacked_td",
            "sub_td",
            "sub_td2",
            "td_h5",
            "unsqueezed_td",
        ):
            with pytest.raises(NotImplementedError):
                getattr(td, op)(*args, inplace=True)
            return
        with getattr(td, op)(*args, inplace=True) as tdt:
            assert tdt is td
            assert tdt.batch_size != batch_size
            if not tdt.requires_grad:
                tdt.apply_(lambda x: x * 0 + 1)
            else:
                tdt.apply(lambda x: x.data.mul_(0).add_(1))
        assert td.batch_size == batch_size
        assert {key: val.shape for key, val in td.items(True, True)} == shapes
        assert (td == 1).all()

    def test_flatten_decorator_inplace_to_last_dim(self, td_name, device):
        # this block kept the flat shape before gh-2700: it still does, and warns
        td = getattr(self, td_name)(device)
        td.unlock_()
        if td_name in (
            "nested_stacked_td",
            "permute_td",
            "squeezed_td",
            "stacked_td",
            "sub_td",
            "sub_td2",
            "td_h5",
            "unsqueezed_td",
        ):
            return
        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
            with td.flatten(0, -1, inplace=True):
                pass
        assert td.batch_size == (24,)

    def test_flatten_unflatten_bis(self, td_name, device):
        td = getattr(self, td_name)(device)
        shape = td.shape[1:4]
        td_flat = td.flatten(1, 3)
        td_unflat = td_flat.unflatten(1, shape)
        assert (td.to_tensordict(retain_none=True) == td_unflat).all()
        assert td.batch_size == td_unflat.batch_size

    def test_from_empty(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        new_td = TensorDict(batch_size=td.batch_size, device=device)
        for key, item in td.items():
            new_td.set(key, item)
        assert_allclose_td(td, new_td)
        assert td.device == new_td.device
        assert td.shape == new_td.shape

    @pytest.mark.skipif(
        is_npu_available(),
        reason="ForeachAddScalar is not fully adapted on NPU currently",
    )
    @pytest.mark.parametrize("dim", [0, 1, 2, 3, -1, -2, -3])
    def test_gather(self, td_name, device, dim):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            original_unbatched = td.get("unbatched")
            index = torch.ones(td.shape, device=td.device, dtype=torch.long)
            other_dim = dim + index.ndim if dim < 0 else dim
            idx = (*[slice(None) for _ in range(other_dim)], slice(2))
            index = index[idx]
            index = index.cumsum(dim=other_dim) - 1
            td_gather = torch.gather(td, dim=dim, index=index)
            assert (
                td_gather.get("unbatched").data_ptr() == original_unbatched.data_ptr()
            )
            assert td_gather.get("unbatched").batch_size == td_gather.batch_size
            return
        index = torch.ones(td.shape, device=td.device, dtype=torch.long)
        other_dim = dim + index.ndim if dim < 0 else dim
        idx = (*[slice(None) for _ in range(other_dim)], slice(2))
        index = index[idx]
        index = index.cumsum(dim=other_dim) - 1
        # gather
        td_gather = torch.gather(td, dim=dim, index=index)
        assert td_gather.device == td.device
        assert td_gather.names == td.names
        # gather with out
        td_gather.zero_()
        out = td_gather.clone()
        if td_name == "td_params":
            with pytest.raises(
                RuntimeError, match="don't support automatic differentiation"
            ):
                torch.gather(td, dim=dim, index=index, out=out)
            return
        td_gather2 = torch.gather(td, dim=dim, index=index, out=out)
        assert (td_gather2 != 0).any()

    @pytest.mark.parametrize(
        "actual_index,expected_index",
        [
            (..., (slice(None),) * TD_BATCH_SIZE),
            ((..., 0), (slice(None),) * (TD_BATCH_SIZE - 1) + (0,)),
            ((0, ...), (0,) + (slice(None),) * (TD_BATCH_SIZE - 1)),
            ((0, ..., 0), (0,) + (slice(None),) * (TD_BATCH_SIZE - 2) + (0,)),
        ],
    )
    def test_getitem_ellipsis(self, td_name, device, actual_index, expected_index):
        torch.manual_seed(1)

        td = getattr(self, td_name)(device)

        actual_td = td[actual_index]
        expected_td = td[expected_index]
        other_expected_td = td.to_tensordict(retain_none=False)[expected_index]
        assert expected_td.shape == _getitem_batch_size(
            td.batch_size, convert_ellipsis_to_idx(actual_index, td.batch_size)
        )
        assert other_expected_td.shape == actual_td.shape
        assert_allclose_td(actual_td, other_expected_td)
        assert_allclose_td(actual_td, expected_td)

    def test_getitem_nestedtuple(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        assert isinstance(td[("a",)], torch.Tensor)
        assert isinstance(td.get((("a",))), torch.Tensor)

    def test_getitem_range(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        assert_allclose_td(td[range(2)], td[[0, 1]])
        if td_name not in ("td_h5",):
            # for h5, we can't use a double list index
            assert td[range(1), range(1)].shape == td[[0], [0]].shape
            assert_allclose_td(td[range(1), range(1)], td[[0], [0]])
        assert_allclose_td(td[:, range(2)], td[:, [0, 1]])
        assert_allclose_td(td[..., range(1)], td[..., [0]])

        if td_name in ("stacked_td", "nested_stacked_td"):
            # this is a bit contrived, but want to check that if we pass something
            # weird as the index to the stacking dimension we'll get the error
            idx = (slice(None),) * td.stack_dim + ({1, 2, 3},)
            with pytest.raises(TypeError, match="Invalid index"):
                td[idx]

    def test_getitem_bool_list(self, td_name, device):
        # a list of bools is a boolean mask, as in torch
        td = getattr(self, td_name)(device)
        mask0 = [True, False, True, False]
        mask1 = [False, True, True]
        for index, mask_index in (
            (mask0, torch.tensor(mask0)),
            ((slice(None), mask1), (slice(None), torch.tensor(mask1))),
        ):
            result, expected = td[index], td[mask_index]
            # assert_allclose_td does not compare batch sizes
            assert result.batch_size == expected.batch_size
            assert_allclose_td(result, expected)

    @pytest.mark.parametrize(
        "index", [[[0], [1]], [0, 2], (slice(None), [[0], [1]]), [True, False] * 2]
    )
    def test_get_at_set_at_index_like_getitem(self, td_name, device, index):
        # get_at and set_at_ read the index as getitem does: a bare nested list
        # is one index, where torch reads it as a tuple of indices
        td = getattr(self, td_name)(device)
        expected = td[index]["a"]
        assert (td.get_at("a", index) == expected).all()
        if td_name == "td_h5":
            # h5py does not take every selection that torch takes
            return
        with torch.no_grad():
            td.set_at_("a", torch.zeros_like(expected), index)
        assert (td[index]["a"] == 0).all()

    def test_getitem_string(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        assert isinstance(td["a"], torch.Tensor)

    @pytest.mark.parametrize(
        "idx",
        [
            (..., None),
            (None, ...),
            (None,),
            None,
            (slice(None), None),
            (0, None),
            (None, slice(None), slice(None)),
            (None, ..., None),
            (None, 1, ..., None),
            (1, ..., None),
            (..., None, 0),
            ([1], ..., None),
        ],
    )
    def test_index_none(self, td_name, device, idx):
        td = getattr(self, td_name)(device)
        tdnone = td[idx]
        tensor = torch.zeros(td.shape)
        assert tdnone.shape == tensor[idx].shape, idx
        # Fixed by 451
        # if td_name == "td_h5":
        #     with pytest.raises(TypeError, match="can't process None"):
        #         assert (tdnone.to_tensordict(retain_none=True) == td.to_tensordict(retain_none=True)[idx]).all()
        #     return
        assert (
            tdnone.to_tensordict(retain_none=True)
            == td.to_tensordict(retain_none=True)[idx]
        ).all()

    def test_indexed_properties(self, td_name, device):
        td = getattr(self, td_name)(device)
        td_index = td[0]
        assert td_index.is_memmap() is td.is_memmap()
        assert td_index.is_shared() is td.is_shared()
        assert td_index.device == td.device

    @pytest.mark.parametrize("npy", [False, True])
    def test_index_tensor_nd_names(self, td_name, device, npy):
        td = getattr(self, td_name)(device)
        names = ("a", "b", "c", "d")
        try:
            td.refine_names(*names)
        except RuntimeError:
            names = td.names
        tensor_example = torch.zeros(()).expand(td.shape)
        assert td.names == list(names)
        index = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        if npy:
            index = index.numpy()
        # the index varies along both dims of the block, which can't both
        # take the name of the indexed dim: names are unique
        td_idx = td[:, index]
        assert tensor_example[:, index].shape == td_idx.shape
        assert td_idx.names == [names[0], None, None, *names[2:]]
        td_idx = td[0, index]
        assert tensor_example[0, index].shape == td_idx.shape
        assert td_idx.names == [None, None, *names[2:]]
        td_idx = td[..., index, :, :]
        assert tensor_example[..., index, :, :].shape == td_idx.shape
        assert td_idx.names == [names[0], None, None, *names[2:]]
        # along a dim of size 1, the index does not vary
        td_idx = td[:, index[:1]]
        assert td_idx.names == [names[0], None, names[1], *names[2:]]

    def test_inferred_view_size(self, td_name, device):
        if td_name in ("permute_td", "sub_td2"):
            pytest.skip("view incompatible with stride / permutation")
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        for i in range(len(td.shape)):
            # replacing every index one at a time
            # with -1, to test that td.view(..., -1, ...)
            # always returns the original tensordict
            new_shape = [
                dim_size if dim_idx != i else -1
                for dim_idx, dim_size in enumerate(td.shape)
            ]
            if lazy_legacy():
                if td_name in ("td_params",):
                    assert td.view(-1).view(*new_shape)._param_td is td._param_td
                    assert td.view(*new_shape)._param_td is td._param_td
                else:
                    assert td.view(*new_shape) is td
                    assert td.view(-1).view(*new_shape) is td

    def test_is_contiguous_per_type(self, td_name, device, monkeypatch):
        td = getattr(self, td_name)(device)
        if isinstance(td, (LazyStackedTensorDict, PersistentTensorDict)):
            # each read builds a new tensor: False, without reading the leaves
            def no_read(*args, **kwargs):
                raise AssertionError("is_contiguous() read a leaf")

            monkeypatch.setattr(type(td), "_get_str", no_read)
            monkeypatch.setattr(type(td), "_get_tuple", no_read)
            assert td.is_contiguous() is False
            return
        assert td.is_contiguous() is all(value.is_contiguous() for value in td.values())

    def test_isfinite(self, td_name, device):
        td = getattr(self, td_name)(device)
        assert td.isfinite().all()

    def test_isnan(self, td_name, device):
        td = getattr(self, td_name)(device)
        assert not td.isnan().any()

    def test_isreal(self, td_name, device):
        td = getattr(self, td_name)(device)
        assert td.isreal().all()

    def test_isposinf(self, td_name, device):
        td = getattr(self, td_name)(device)
        assert not td.isposinf().any()

    def test_isneginf(self, td_name, device):
        td = getattr(self, td_name)(device)
        assert not td.isneginf().any()

    def test_keys_contains_nested_key_without_include_nested(self, td_name, device):
        # A nested key is not among the keys of a view without nesting. A lazy
        # stack still finds it, with a FutureWarning, until 0.17 (#2060).
        td = getattr(self, td_name)(device)
        td.unlock_()
        td["nested", "leaf"] = torch.zeros(td.shape, device=device)
        td["leaf"] = torch.zeros(td.shape, device=device)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            for key in [("nested", "leaf", "x"), ("leaf", "x")]:
                assert key not in td.keys()
            assert ("nested", "leaf") not in td.keys(leaves_only=True)
            assert ("nested", "leaf") in td.keys(include_nested=True)
            if not isinstance(td, LazyStackedTensorDict):
                assert ("nested", "leaf") not in td.keys()
                return
        with pytest.warns(FutureWarning, match="returns False"):
            assert ("nested", "leaf") in td.keys()

    def test_items_values_keys(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td.unlock_()
        keys = list(td.keys())
        values = list(td.values())
        items = list(td.items())

        # Test td.items()
        constructed_td1 = TensorDict(batch_size=td.shape)
        for key, value in items:
            constructed_td1.set(key, value)

        assert (td == constructed_td1).all()

        # Test td.keys() and td.values()
        # items = [key, value] should be verified
        assert len(values) == len(items)
        assert len(keys) == len(items)
        constructed_td2 = TensorDict(batch_size=td.shape)
        for key, value in list(zip(td.keys(), td.values())):
            constructed_td2.set(key, value)

        assert (td == constructed_td2).all()

        # Test that keys is sorted
        assert all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))

        # Add new element to tensor
        a = td.get("a")
        td.set("x", torch.randn_like(a))
        keys = list(td.keys())
        values = list(td.values())
        items = list(td.items())

        # Test that keys is still sorted after adding the element
        assert all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))

        # Test td.items()
        # after adding the new element
        constructed_td1 = TensorDict(batch_size=td.shape)
        for key, value in items:
            constructed_td1.set(key, value)

        assert (td == constructed_td1).all()

        # Test td.keys() and td.values()
        # items = [key, value] should be verified
        # even after adding the new element
        assert len(values) == len(items)
        assert len(keys) == len(items)

        constructed_td2 = TensorDict(batch_size=td.shape)
        for key, value in list(zip(td.keys(), td.values())):
            constructed_td2.set(key, value)

        assert (td == constructed_td2).all()

    def test_iter(self, td_name, device):
        td = getattr(self, td_name)(device)
        items = list(td)
        assert len(items) == td.shape[0]
        for i, item in enumerate(items):
            assert (item == td[i]).all()
        # A batch dimension of length 0 gives an empty iteration, not an error
        assert list(td[:0]) == []

    def test_lock(self, td_name, device):
        td = getattr(self, td_name)(device)
        is_locked = td.is_locked
        for item in td.values():
            if isinstance(item, TensorDictBase):
                assert item.is_locked == is_locked
        if isinstance(td, _SubTensorDict):
            with pytest.raises(RuntimeError, match="the parent tensordict instead"):
                td.is_locked = not is_locked
            return
        td.is_locked = not is_locked
        assert td.is_locked != is_locked
        for _, item in td.items():
            if isinstance(item, TensorDictBase):
                assert item.is_locked != is_locked
        td.lock_()
        assert td.is_locked
        for _, item in td.items():
            if isinstance(item, TensorDictBase):
                assert item.is_locked
        td.unlock_()
        assert not td.is_locked
        for _, item in td.items():
            if isinstance(item, TensorDictBase):
                assert not item.is_locked

    def test_lock_change_names(self, td_name, device):
        if td_name == "td_with_unbatched":
            # UnbatchedTensor doesn't have names because its shape doesn't follow batch dims
            pytest.skip(
                "UnbatchedTensor has no names attribute (shape independent of batch)"
            )
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        try:
            td.names = [str(i) for i in range(td.ndim)]
            td.lock_()
        except Exception:
            return
        # cache values
        list(td.values(True))
        td.names = [str(-i) for i in range(td.ndim)]
        for val in td.values(True):
            if not is_tensor_collection(val):
                continue
            assert val.names[: td.ndim] == [str(-i) for i in range(td.ndim)]

    # getting values from lazy tensordicts in non-lazy contexts messes things up
    # so we set it to True. When we'll deprecate lazy tensordicts, we will just
    # remove this decorator
    @legacy_lazy_mode()
    def test_lock_nested(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name in ("sub_td", "sub_td2") and td.is_locked:
            with pytest.raises(RuntimeError, match="Cannot unlock"):
                td.unlock_()
        else:
            td.unlock_()
        td.set(("some", "nested"), torch.zeros(td.shape))
        if td_name in ("sub_td", "sub_td2") and not td.is_locked:
            with pytest.raises(RuntimeError, match="Cannot lock"):
                td.lock_()
            return
        assert not td.is_locked
        td.lock_()
        some = td.get("some")
        assert some.is_locked
        with pytest.raises(RuntimeError):
            some.unlock_()
        # this assumes that td is out of scope after the call to del.
        # an error in unlock_() is likely due to td leaving a trace somewhere.
        del td
        gc.collect()
        some.unlock_()

    def test_lock_write(self, td_name, device):
        td = getattr(self, td_name)(device)
        if isinstance(td, _SubTensorDict):
            with pytest.raises(RuntimeError, match="the parent tensordict instead"):
                td.lock_()
            return

        td.lock_()
        td_clone = td.clone()
        assert not td_clone.is_locked
        td_clone = td.to_tensordict(retain_none=True)
        assert not td_clone.is_locked
        assert td.is_locked
        if td_name == "td_h5":
            td.unlock_()
            for key in list(td.keys()):
                del td[key]
            td.lock_()
        else:
            if td_name in (
                "sub_td",
                "sub_td2",
                "permute_td",
                "squeezed_td",
                "unsqueezed_td",
            ):
                # we can't call select inplace on these guys so we exit here
                return
            with td.unlock_() if td.is_locked else contextlib.nullcontext():
                td = td.select(inplace=True)
        for key, item in td_clone.items(True):
            with pytest.raises(RuntimeError, match="Cannot modify locked TensorDict"):
                td.set(key, item)
        td.unlock_()
        for key, item in td_clone.items(True):
            td.set(key, item)
        td.lock_()
        for key, item in td_clone.items(True):
            with pytest.raises(RuntimeError, match="Cannot modify locked TensorDict"):
                td.set(key, item)
            if td_name == "td_params":
                td_set = td.data
            else:
                td_set = td
            td_set.set_(key, item)

    def test_lock_context_manager_restores_on_error(self, td_name, device):
        # lock_() and unlock_() undo their change also when the body raises
        td = getattr(self, td_name)(device)
        if isinstance(td, _SubTensorDict):
            pytest.skip("a _SubTensorDict cannot be locked or unlocked")
        is_locked = td.is_locked
        with pytest.raises(ValueError, match="error in the body"):
            with td.unlock_():
                raise ValueError("error in the body")
        assert td.is_locked is is_locked
        td.unlock_()
        with pytest.raises(ValueError, match="error in the body"):
            with td.lock_():
                raise ValueError("error in the body")
        assert not td.is_locked
        assert not td._last_op_queue
        td.lock_()
        with pytest.raises(ValueError, match="error in the body"):
            with td.unlock_():
                raise ValueError("error in the body")
        assert td.is_locked
        assert not td._last_op_queue

    @pytest.mark.parametrize("has_out", [False, "complete", "empty"])
    @pytest.mark.parametrize("keepdim", [True, False])
    @pytest.mark.parametrize("dim", [1, (1,), (1, -1)])
    def test_logsumexp(self, td_name, device, has_out, keepdim, dim):
        if td_name == "td_with_unbatched":
            pytest.skip(
                "UnbatchedTensor incompatible with reduction ops that check shape"
            )
        td = getattr(self, td_name)(device)
        if not has_out:
            out = None
        elif has_out == "complete":
            out = (
                td.to_tensordict(retain_none=False)
                .detach()
                .logsumexp(dim=dim, keepdim=keepdim)
            )
            if td.requires_grad:
                td = td.detach()
        else:
            out = (
                td.to_tensordict(retain_none=False)
                .detach()
                .logsumexp(dim=dim, keepdim=keepdim)
                .empty()
            )
            if td.requires_grad:
                td = td.detach()
        if out is not None:
            out_c = out.copy()
        tdlse = td.logsumexp(dim=dim, out=out, keepdim=keepdim)
        assert tdlse.batch_size != td.batch_size
        if out is not None:

            def check(x, y):
                if y is not None:
                    assert x is y

            assert tdlse is out
            tdlse.apply(check, out_c, default=None)
        tdlse._check_batch_size()

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_losses(self, td_name, device, reduction):
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            # UnbatchedTensor doesn't implement __torch_function__ for loss functions
            pytest.skip("UnbatchedTensor loss functions not yet implemented")
            return
        assert is_tensor_collection(
            torch.nn.functional.l1_loss(td.float(), -td.float(), reduction=reduction)
        )
        assert is_tensor_collection(
            torch.nn.functional.mse_loss(td.float(), -td.float(), reduction=reduction)
        )
        assert is_tensor_collection(
            torch.nn.functional.smooth_l1_loss(
                td.float(), -td.float(), reduction=reduction
            )
        )

    @pytest.mark.skipif(
        is_npu_available(),
        reason="ForeachAddScalar is not fully adapted on NPU currently",
    )
    def test_masked_fill(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            # UnbatchedTensor: masked_fill has expand_as_right dimension mismatch
            pytest.skip("UnbatchedTensor masked_fill has dimension mismatch")
            return
        mask = torch.zeros(td.shape, dtype=torch.bool, device=device).bernoulli_()
        new_td = td.masked_fill(mask, -10.0)
        assert new_td is not td
        for item in new_td.values():
            assert (item[mask] == -10).all()

    def test_masked_fill_(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            # UnbatchedTensor: masked_fill_ has expand_as_right dimension mismatch
            pytest.skip("UnbatchedTensor masked_fill_ has dimension mismatch")
            return
        mask = torch.zeros(td.shape, dtype=torch.bool, device=device).bernoulli_()
        td_before = td.to_tensordict()
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        new_td = td_set.masked_fill_(mask, -10.0)
        assert new_td is td_set
        for item in td.values():
            assert (item[mask] == -10).all(), item[mask]
        # the entries outside the mask keep their values
        for key in td.keys(True, True):
            assert (td.get(key)[~mask] == td_before.get(key)[~mask]).all(), key

    def test_masking(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        while True:
            mask = torch.zeros(
                td.batch_size, dtype=torch.bool, device=device
            ).bernoulli_(0.8)
            if not mask.all() and mask.any():
                break
        td_masked = td[mask]
        td_masked2 = torch.masked_select(td, mask)
        assert_allclose_td(td_masked, td_masked2)
        assert td_masked.batch_size[0] == mask.sum()
        assert td_masked.batch_dims == 1

    def test_masking_set(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            # UnbatchedTensor: masking set is a no-op for UnbatchedTensor data
            original_unbatched = td.get("unbatched")
            mask = torch.zeros(
                td.batch_size, dtype=torch.bool, device=device
            ).bernoulli_(0.8)
            n = mask.sum()
            d = td.ndimension()
            pseudo_td = td.exclude("unbatched").apply(
                lambda item: torch.zeros(
                    (n, *item.shape[d:]), dtype=item.dtype, device=device
                ),
                batch_size=[n, *td.batch_size[d:]],
            )
            td[mask] = pseudo_td
            # UnbatchedTensor data should be unchanged
            assert td["unbatched"] is original_unbatched
            return
        mask = torch.zeros(td.batch_size, dtype=torch.bool, device=device).bernoulli_(
            0.8
        )
        n = mask.sum()
        d = td.ndimension()
        pseudo_td = td.apply(
            lambda item: torch.zeros(
                (n, *item.shape[d:]), dtype=item.dtype, device=device
            ),
            batch_size=[n, *td.batch_size[d:]],
        )

        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td

        td_set[mask] = pseudo_td
        for item in td.values():
            assert (item[mask] == 0).all()

    @pytest.mark.parametrize("dim", [None, 0])
    def test_mean(self, td_name, device, dim):
        td = getattr(self, td_name)(device)
        assert is_tensor_collection(torch.mean(td.float(), dim=dim))

    @pytest.mark.parametrize("dim", [None, 0])
    def test_var(self, td_name, device, dim):
        td = getattr(self, td_name)(device)
        assert is_tensor_collection(torch.var(td.float(), dim=dim))

    @pytest.mark.parametrize("dim", [None, 0])
    def test_std(self, td_name, device, dim):
        td = getattr(self, td_name)(device)
        assert is_tensor_collection(torch.std(td.float(), dim=dim))

    def test_maximum(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            pytest.skip("Non differentiable output.")
        assert is_tensor_collection(torch.maximum(td, td))

    @pytest.mark.parametrize("use_dir", [True, False])
    @pytest.mark.parametrize("num_threads", [0, 2])
    def test_memmap_(self, td_name, device, use_dir, tmpdir, num_threads):
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor memmap support not yet implemented")
        td = getattr(self, td_name)(device)
        if td_name in ("sub_td", "sub_td2"):
            with pytest.raises(
                RuntimeError,
                match="Converting a sub-tensordict values to memmap cannot be done",
            ):
                td.memmap_(
                    prefix=tmpdir if use_dir else None,
                    num_threads=num_threads,
                    copy_existing=True,
                )
            return
        elif td_name in ("td_h5", "td_params"):
            with pytest.raises(
                RuntimeError,
                match="Cannot build a memmap TensorDict in-place",
            ):
                td.memmap_(
                    prefix=tmpdir if use_dir else None,
                    num_threads=num_threads,
                    copy_existing=True,
                )
            return
        else:
            td.memmap_(
                prefix=tmpdir if use_dir else None,
                num_threads=num_threads,
                copy_existing=True,
            )
            assert td.is_memmap(), (td, td._is_memmap)
        if use_dir:
            # This would fail if we were not filtering out unregistered sub-folders
            os.mkdir(Path(tmpdir) / "some_other_path")
            assert_allclose_td(TensorDict.load_memmap(tmpdir), td)

    @pytest.mark.parametrize("copy_existing", [False, True])
    def test_memmap_existing(self, td_name, device, copy_existing, tmp_path):
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor memmap support not yet implemented")
        if td_name == "memmap_td":
            pytest.skip(
                "Memmap case is redundant, functionality checked by other cases"
            )
        elif td_name in ("sub_td", "sub_td2", "td_h5", "td_params"):
            pytest.skip(
                "_SubTensorDict/H5 and memmap_ incompatibility is checked elsewhere"
            )

        td = getattr(self, td_name)(device).memmap_(prefix=tmp_path / "tensordict")
        td2 = getattr(self, td_name)(device).memmap_()

        if copy_existing:
            td3 = td.memmap_(prefix=tmp_path / "tensordict2", copy_existing=True)
            assert (td == td3).all()
        else:
            with pytest.raises(
                RuntimeError,
                match="A filename was provided but the tensor already has a file associated",
            ):
                # calling memmap_ with prefix that is different to contents gives error
                td.memmap_(prefix=tmp_path / "tensordict2")

            # calling memmap_ without prefix means no-op, regardless of whether contents
            # were saved in temporary or designated location (td vs. td2 resp.)
            td3 = td.memmap_()
            td4 = td2.memmap_()

            if td_name in ("stacked_td", "nested_stacked_td"):
                assert all(
                    all(
                        td3_[key] is value
                        for key, value in td_.items(
                            include_nested=True, leaves_only=True
                        )
                    )
                    for td_, td3_ in zip(td.tensordicts, td3.tensordicts)
                )
                assert all(
                    all(
                        td4_[key] is value
                        for key, value in td2_.items(
                            include_nested=True, leaves_only=True
                        )
                    )
                    for td2_, td4_ in zip(td2.tensordicts, td4.tensordicts)
                )
            elif td_name in ("permute_td", "squeezed_td", "unsqueezed_td"):
                assert all(
                    td3._source[key] is value
                    for key, value in td._source.items(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    td4._source[key] is value
                    for key, value in td2._source.items(
                        include_nested=True, leaves_only=True
                    )
                )
            else:
                assert all(
                    td3[key] is value
                    for key, value in td.items(include_nested=True, leaves_only=True)
                )
                assert all(
                    td4[key] is value
                    for key, value in td2.items(include_nested=True, leaves_only=True)
                )

    @pytest.mark.parametrize("use_dir", [True, False])
    @pytest.mark.parametrize("num_threads", [0, 2])
    def test_memmap_like(self, td_name, device, use_dir, tmpdir, num_threads):
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor memmap support not yet implemented")
        td = getattr(self, td_name)(device)
        tdmemmap = td.memmap_like(
            prefix=tmpdir if use_dir else None,
            num_threads=num_threads,
            copy_existing=True,
        )
        assert tdmemmap is not td
        for key in td.keys(True):
            v1 = td[key]
            v2 = tdmemmap[key]
            if isinstance(v1, str):
                # non-tensor data storing strings share the same id in python
                assert v1 == v2
            else:
                assert v1 is not v2
                if isinstance(v1, torch.Tensor):
                    assert (
                        v1.untyped_storage().data_ptr()
                        != v2.untyped_storage().data_ptr()
                    )
        # assert (td != tdmemmap).any()
        assert tdmemmap.is_memmap()

    def test_memmap_prefix(self, td_name, device, tmp_path):
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor memmap support not yet implemented")
        if td_name == "memmap_td":
            pytest.skip(
                "Memmap case is redundant, functionality checked by other cases"
            )

        td = getattr(self, td_name)(device)
        if td_name in ("sub_td", "sub_td2"):
            with pytest.raises(
                RuntimeError,
                match="Converting a sub-tensordict values to memmap cannot be done",
            ):
                td.memmap_(tmp_path / "tensordict")
            return
        elif td_name in ("td_h5", "td_params"):
            with pytest.raises(
                RuntimeError,
                match="Cannot build a memmap TensorDict in-place",
            ):
                td.memmap_(tmp_path / "tensordict")
            return
        else:
            td.memmap_(tmp_path / "tensordict")
        if td_name not in ("stacked_td", "nested_stacked_td"):
            jsonpath = tmp_path / "tensordict" / "meta.json"
        else:
            jsonpath = tmp_path / "tensordict" / "0" / "meta.json"
        assert jsonpath.exists(), td
        with open(jsonpath) as file:
            metadata = json.load(file)
        if td_name in ("stacked_td", "nested_stacked_td"):
            assert metadata["shape"] == list(td.tensordicts[0].batch_size)
        else:
            assert metadata["shape"] == list(td.batch_size)

        td2 = td.load_memmap(tmp_path / "tensordict", device=device)
        assert (td.cpu() == td2.cpu()).all()

    def test_memmap_archive(self, td_name, device, tmp_path):
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor memmap support not yet implemented")
        td = getattr(self, td_name)(device)
        archive = tmp_path / "saved.tdz"
        td.save(archive, copy_existing=True)
        assert archive.is_file()
        assert is_memmap_archive(archive)
        loaded = TensorDict.load_memmap(archive, device=device)
        assert (td.cpu() == loaded.cpu()).all()
        # the archive is a plain zip: extracting it yields a loadable
        # memmap directory
        unpack_memmap(archive, tmp_path / "unpacked")
        loaded_dir = TensorDict.load_memmap(tmp_path / "unpacked", device=device)
        assert (td.cpu() == loaded_dir.cpu()).all()

    @pytest.mark.parametrize("use_dir", [True, False])
    @pytest.mark.parametrize("num_threads", [2])
    def test_memmap_threads(self, td_name, device, use_dir, tmpdir, num_threads):
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor memmap support not yet implemented")
        td = getattr(self, td_name)(device)
        tdmmap = td.memmap(
            prefix=tmpdir if use_dir else None,
            num_threads=num_threads,
            copy_existing=True,
        )
        assert_allclose_td(td.cpu().detach(), tdmmap)

        tdfuture = td.memmap(
            prefix=tmpdir if use_dir else None,
            num_threads=num_threads,
            copy_existing=True,
            return_early=True,
        )
        assert_allclose_td(td.cpu().detach(), tdfuture.result())

    @pytest.mark.parametrize(
        "dim, keepdim, return_indices",
        [
            [None, False, False],
            [0, False, False],
            [0, True, False],
            [0, False, True],
            [0, True, True],
            [1, False, False],
            [1, True, False],
            [1, False, True],
            [1, True, True],
            [-1, False, False],
            [-1, True, False],
            [-1, False, True],
            [-1, True, True],
        ],
    )
    def test_min_max_cummin_cummax(self, td_name, device, dim, keepdim, return_indices):
        def _get_td(v):
            if not is_tensor_collection(v):
                return v.values
            return v

        td = getattr(self, td_name)(device)
        # min
        if dim is not None:
            kwargs = {"dim": dim, "keepdim": keepdim, "return_indices": return_indices}
        else:
            kwargs = {}
        r = td.min(**kwargs)
        if not return_indices and dim is not None:
            assert_allclose_td(r, td.amin(dim=dim, keepdim=keepdim))
        if return_indices:
            # assert is_tensorclass(r)
            assert isinstance(r, torch.return_types.min)
            assert not r.values.is_empty()
            assert not r.indices.is_empty()
        if dim is None:
            assert _get_td(r).batch_size == ()
        elif keepdim:
            s = list(td.batch_size)
            s[dim] = 1
            assert _get_td(r).batch_size == tuple(s)
        else:
            s = list(td.batch_size)
            s.pop(dim)
            assert _get_td(r).batch_size == tuple(s)

        r = td.max(**kwargs)
        if not return_indices and dim is not None:
            assert_allclose_td(r, td.amax(dim=dim, keepdim=keepdim))
        if return_indices:
            # assert is_tensorclass(r)
            assert isinstance(r, torch.return_types.max)
            assert not r.values.is_empty()
            assert not r.indices.is_empty()
        if dim is None:
            assert _get_td(r).batch_size == ()
        elif keepdim:
            s = list(td.batch_size)
            s[dim] = 1
            assert _get_td(r).batch_size == tuple(s)
        else:
            s = list(td.batch_size)
            s.pop(dim)
            assert _get_td(r).batch_size == tuple(s)
        if dim is None:
            return
        kwargs.pop("keepdim")
        r = td.cummin(**kwargs)
        if return_indices:
            # assert is_tensorclass(r)
            assert isinstance(r, torch.return_types.cummin)
            assert not r.values.is_empty()
            assert not r.indices.is_empty()
        if dim is None:
            assert _get_td(r).batch_size == ()
        else:
            assert _get_td(r).batch_size == td.batch_size

        r = td.cummax(**kwargs)
        if return_indices:
            # assert is_tensorclass(r)
            assert isinstance(r, torch.return_types.cummax)
            assert not r.values.is_empty()
            assert not r.indices.is_empty()
        if dim is None:
            assert _get_td(r).batch_size == ()
        else:
            assert _get_td(r).batch_size == td.batch_size

    @pytest.mark.parametrize("inplace", [False, True])
    def test_named_apply(self, td_name, device, inplace):
        td = getattr(self, td_name)(device)
        td_c = td.to_tensordict(retain_none=True)

        def named_plus(name, x):
            if "a" in name:
                return x + 1

        if inplace and td_name == "td_params":
            with pytest.raises(ValueError, match="Failed to update"):
                td.named_apply(named_plus, inplace=inplace)
            return
        td_1 = td.named_apply(named_plus, inplace=inplace, filter_empty=True)
        if inplace:
            assert td_1 is td
            for key in td_1.keys(True, True):
                if isinstance(key, tuple):
                    subkey = key[-1]
                else:
                    subkey = key
                if "a" in subkey:
                    assert (td_c[key] + 1 == td_1[key]).all()
                else:
                    assert (td_c[key] == td_1[key]).all()
                assert (td_1[key] == td[key]).all()
        else:
            for key in td_1.keys(True, True):
                assert "a" in key
                assert (td_c[key] + 1 != td[key]).any()
                assert (td_1[key] == td[key] + 1).all()

    def test_named_apply_out(self, td_name, device):
        td = getattr(self, td_name)(device)
        if not isinstance(td, LazyStackedTensorDict):
            td_c = td.to_tensordict(retain_none=True)
        else:
            td_c = td.clone()
        td.named_apply(lambda name, x: x + 1, out=td_c)
        assert_allclose_td(
            td.filter_non_tensor_data().data + 1,
            td_c.filter_non_tensor_data().filter_empty_(),
        )

    def test_named_apply_complete(self, td_name, device):
        td = getattr(self, td_name)(device)
        td.unlock_()
        # "a" conflicts with root key with the same name
        td.set(("some", "a"), td.get(list(td.keys())[0]))
        keys_complete = set()
        keys_not_complete = set()

        def count(name, value, keys):
            keys.add(name)

        td.named_apply(
            functools.partial(count, keys=keys_complete),
            nested_keys=True,
            filter_empty=True,
        )
        td.named_apply(
            functools.partial(count, keys=keys_not_complete),
            nested_keys=False,
            filter_empty=True,
        )
        assert len(keys_complete) == len(list(td.keys(True, True)))
        assert len(keys_complete) > len(keys_not_complete)

    def test_nested_dict_init(self, td_name, device):
        if td_name == "td_with_unbatched":
            # UnbatchedTensor: to_dict() returns nested dict structure which can't be re-initialized directly
            pytest.skip(
                "UnbatchedTensor to_dict() returns nested structure incompatible with TensorDict init"
            )
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td.unlock_()

        # Create TensorDict and dict equivalent values, and populate each with according nested value
        td_clone = td.clone(recurse=True)
        td_dict = td.to_dict()
        nested_dict_value = {"e": torch.randn(4, 3, 2, 1, 10)}
        nested_tensordict_value = TensorDict(
            nested_dict_value, batch_size=td.batch_size, device=device
        )
        td_dict["d"] = nested_dict_value
        td_clone["d"] = nested_tensordict_value

        # Re-init new TensorDict from dict, and check if they're equal
        td_dict_init = TensorDict(td_dict, batch_size=td.batch_size, device=device)

        assert (td_clone == td_dict_init).all()

    def test_nested_td(self, td_name, device):
        td = getattr(self, td_name)(device)
        td.unlock_()
        tdin = TensorDict({"inner": torch.randn(td.shape)}, td.shape, device=device)
        td.set("inner_td", tdin)
        assert (td["inner_td"] == tdin).all()

    def test_nested_td_emptyshape(self, td_name, device):
        td = getattr(self, td_name)(device)
        td.unlock_()
        tdin = TensorDict({"inner": torch.randn(*td.shape, 1)}, [], device=device)
        td["inner_td"] = tdin
        tdin.batch_size = td.batch_size
        assert (td["inner_td"] == tdin).all()

    def test_nested_td_index(self, td_name, device):
        td = getattr(self, td_name)(device)
        td.unlock_()

        sub_td = TensorDict({}, [*td.shape, 2], device=device)
        a = torch.zeros([*td.shape, 2, 2], device=device)
        sub_sub_td = TensorDict({"a": a}, [*td.shape, 2, 2], device=device)
        sub_td.set("sub_sub_td", sub_sub_td)
        td.set("sub_td", sub_td)
        assert (td["sub_td", "sub_sub_td", "a"] == 0).all()
        assert (
            td["sub_td"]["sub_sub_td"]["a"] == td["sub_td", "sub_sub_td", "a"]
        ).all()

        a = torch.ones_like(a)
        other_sub_sub_td = TensorDict({"a": a}, [*td.shape, 2, 2])
        td["sub_td", "sub_sub_td"] = other_sub_sub_td
        assert (td["sub_td", "sub_sub_td", "a"] == 1).all()
        assert (
            td["sub_td"]["sub_sub_td"]["a"] == td["sub_td", "sub_sub_td", "a"]
        ).all()

        b = torch.ones_like(a)
        other_sub_sub_td = TensorDict({"b": b}, [*td.shape, 2, 2])

        if td_name in ("sub_td", "sub_td2"):
            td["sub_td", "sub_sub_td"] = other_sub_sub_td
        else:
            td["sub_td", "sub_sub_td"] = other_sub_sub_td
            assert (td["sub_td", "sub_sub_td", "b"] == 1).all()
            assert (
                td["sub_td"]["sub_sub_td"]["b"] == td["sub_td", "sub_sub_td", "b"]
            ).all()

    @pytest.mark.skipif(
        is_npu_available(), reason="torch.nested_tensor is not adapted on NPU currently"
    )
    @pytest.mark.parametrize("dim", [0, 1, -1, -5])
    @pytest.mark.parametrize(
        "key", ["heterogeneous-entry", ("sub", "heterogeneous-entry")]
    )
    def test_nestedtensor_stack(self, td_name, device, dim, key):
        torch.manual_seed(1)
        td1 = getattr(self, td_name)(device).unlock_()
        td2 = getattr(self, td_name)(device).unlock_()
        if td_name == "td_with_unbatched":
            # Stacking two different UnbatchedTensors warns: share it so that
            # only the heterogeneous entry fails.
            td2.set("unbatched", td1.get("unbatched"))

        td1[key] = torch.randn(*td1.shape, 2)
        td2[key] = torch.randn(*td1.shape, 3)
        td_stack = LazyStackedTensorDict.lazy_stack([td1, td2], dim)
        # get will fail
        with pytest.raises(
            RuntimeError, match="Failed to stack tensors within a tensordict"
        ):
            td_stack.get(key)
        with pytest.raises(
            RuntimeError, match="Failed to stack tensors within a tensordict"
        ):
            td_stack[key]
        if dim in (0, -5):
            # this will work if stack_dim is 0 (or equivalently -self.batch_dims)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                td_stack._get_nestedtensor(key)
        else:
            # if the stack_dim is not zero, then calling _get_nestedtensor is disallowed
            with pytest.raises(
                RuntimeError,
                match="LazyStackedTensorDict.get_nestedtensor can only be called "
                "when the stack_dim is 0.",
            ):
                td_stack._get_nestedtensor(key)
        with pytest.raises(
            RuntimeError, match="Failed to stack tensors within a tensordict"
        ):
            td_stack.contiguous()
        with pytest.raises(
            RuntimeError, match="Failed to stack tensors within a tensordict"
        ):
            td_stack.to_tensordict(retain_none=True)
        # cloning is type-preserving: we can do that operation
        td_stack.clone()

    def test_new_empty(self, td_name, device):
        td = getattr(self, td_name)(device)
        tdn = td.new_empty([0])
        assert tdn.shape == (0,)
        tdn = td.new_empty(0)
        assert tdn.shape == (0,)
        tdn = td.new_empty(2, 3)
        assert tdn.shape == (2, 3)
        # assert (tdn != 0).any()
        if td._has_non_tensor:
            assert tdn._has_non_tensor

    def test_new_full(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor new_full comparison needs investigation")
            return
        tdn = td.new_full([0], 2)
        assert tdn.shape == (0,)
        tdn = td.new_full((2, 3), 2)
        assert tdn.shape == (2, 3)
        assert (tdn == 2).all()
        if td._has_non_tensor:
            assert tdn._has_non_tensor

    def test_new_ones(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor new_ones comparison needs investigation")
            return
        tdn = td.new_ones([0])
        assert tdn.shape == (0,)
        tdn = td.new_ones(0)
        assert tdn.shape == (0,)
        tdn = td.new_ones(2, 3)
        assert tdn.shape == (2, 3)
        assert (tdn == 1).all()
        if td._has_non_tensor:
            assert tdn._has_non_tensor

    def test_new_tensor(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor new_tensor comparison needs investigation")
            return
        if td_name in ("td_params",):
            td = td.data
        with pytest.warns(UserWarning, match="To copy construct from a tensor"):
            tdn = td.new_tensor(torch.zeros(0, device="cpu"))
            assert tdn.device == torch.device("cpu")
            assert tdn.shape == (0,)
            tdn = td.new_tensor(torch.zeros(2, device="cpu"))
            assert tdn.device == torch.device("cpu")
            assert tdn.shape == (2,)
            tdn = td.new_tensor(td[0] * 0)
            assert tdn.device == td.device
            assert (tdn == 0).all()
            assert tdn.shape == td.shape[1:]
            if td._has_non_tensor:
                assert tdn._has_non_tensor
        # Array-likes, as torch.Tensor.new_tensor takes them: no warning about
        # copying a tensor, the default device, and the dtype of each entry.
        for data in ([0, 1], (0, 1), np.array([0, 1])):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                tdn = td.new_tensor(data)
            assert not any("To copy construct" in str(w.message) for w in caught)
            assert tdn.device == torch.device("cpu")
            assert tdn.shape == (2,)
            for key in tdn.keys(True, True):
                value = tdn.get(key)
                if not isinstance(value, torch.Tensor):
                    continue
                assert value.dtype == td.get(key).dtype
                assert value.tolist() == torch.tensor([0, 1]).to(value.dtype).tolist()
        tdn = td.new_tensor(1)
        assert tdn.shape == ()
        assert (tdn == 1).all()

    def test_new_zeros(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor new_zeros comparison needs investigation")
            return
        tdn = td.new_zeros([0])
        assert tdn.shape == (0,)
        tdn = td.new_zeros(0)
        assert tdn.shape == (0,)
        tdn = td.new_zeros(2, 3)
        assert tdn.shape == (2, 3)
        assert (tdn == 0).all()
        if td._has_non_tensor:
            assert tdn._has_non_tensor

    # This test fails on lazy tensordicts when lazy-legacy is False
    # Deprecating lazy modules will make this decorator useless (the test should
    # still run ok).
    @legacy_lazy_mode()
    def test_non_tensor_data(self, td_name, device):
        td = getattr(self, td_name)(device)
        # check lock
        if td_name not in ("sub_td", "sub_td2"):
            with td.lock_(), pytest.raises(RuntimeError, match=re.escape(_LOCK_ERROR)):
                td.set_non_tensor(("this", "will"), "fail")
        # check set
        with td.unlock_():
            td.set(("this", "tensor"), torch.zeros(td.shape))
            reached = False
            with (
                pytest.raises(
                    RuntimeError,
                    match="set_non_tensor is not compatible with the tensordict type",
                )
                if td_name in ("td_h5",)
                else contextlib.nullcontext()
            ):
                td.set_non_tensor(("this", "will"), "succeed")
                reached = True
            if not reached:
                return
        # check get (for tensor)
        assert (td.get_non_tensor(("this", "tensor")) == 0).all()

        # check get (for non-tensor)

        def check(x):
            assert x == "succeed"

        torch.utils._pytree.tree_map(check, td.get_non_tensor(("this", "will")))

        assert is_non_tensor(td.get(("this", "will")))

        with td.unlock_():
            td["this", "other", "tensor"] = "success"

            def check(x):
                assert x == "success"

            assert not is_non_tensor(td["this", "other", "tensor"]), td
            torch.utils._pytree.tree_map(check, td["this", "other", "tensor"])
            assert is_non_tensor(td.get(("this", "other", "tensor")))
            torch.utils._pytree.tree_map(
                check, td.get_non_tensor(("this", "other", "tensor"))
            )

    # This test fails on lazy tensordicts when lazy-legacy is False
    # Deprecating lazy modules will make this decorator useless (the test should
    # still run ok).
    @legacy_lazy_mode()
    def test_non_tensor_data_flatten_keys(self, td_name, device):
        td = getattr(self, td_name)(device)
        with td.unlock_():
            td.set(("this", "tensor"), torch.zeros(td.shape))
            reached = False
            with (
                pytest.raises(
                    RuntimeError,
                    match="set_non_tensor is not compatible with the tensordict type",
                )
                if td_name in ("td_h5",)
                else contextlib.nullcontext()
            ):
                td.set_non_tensor(("this", "will"), "succeed")
                reached = True
            if not reached:
                return
        td_flat = td.flatten_keys()
        assert (td_flat.get("this.tensor") == 0).all()

        def check(x):
            assert x == "succeed"

        torch.utils._pytree.tree_map(check, td_flat.get_non_tensor("this.will"))

    # This test fails on lazy tensordicts when lazy-legacy is False
    # Deprecating lazy modules will make this decorator useless (the test should
    # still run ok).
    @legacy_lazy_mode()
    def test_non_tensor_data_pickle(self, td_name, device, tmpdir):
        if td_name == "td_with_unbatched":
            # UnbatchedTensor memmap/pickle requires special metadata handling
            pytest.skip("UnbatchedTensor memmap/pickle requires special handling")
        td = getattr(self, td_name)(device)
        with td.unlock_():
            td.set(("this", "tensor"), torch.zeros(td.shape))
            reached = False
            with (
                pytest.raises(
                    RuntimeError,
                    match="set_non_tensor is not compatible with the tensordict type",
                )
                if td_name in ("td_h5",)
                else contextlib.nullcontext()
            ):
                td.set_non_tensor(("this", "will"), "succeed")
                reached = True
            if not reached:
                return
            td.set_non_tensor(("non", "json", "serializable"), DummyPicklableClass(10))
        td.memmap(prefix=tmpdir, copy_existing=True)
        with pytest.raises(RuntimeError, match="allow_pickle=True"):
            TensorDict.load_memmap(tmpdir, allow_pickle=False)
        with pytest.warns(FutureWarning, match="TensorDict 0.16, the default"):
            TensorDict.load_memmap(tmpdir)
        loaded = TensorDict.load_memmap(tmpdir, allow_pickle=True)
        assert is_non_tensor(loaded.get(("non", "json", "serializable")))

        def check(x, val):
            assert x == val

        torch.utils._pytree.tree_map(
            lambda x: check(x, val=10),
            loaded.get_non_tensor(("non", "json", "serializable")),
        )
        torch.utils._pytree.tree_map(
            lambda x: check(x, val="succeed"), loaded.get_non_tensor(("this", "will"))
        )

    def test_numpy(self, td_name, device):
        td = getattr(self, td_name)(device)
        td_numpy = td.data.cpu().numpy()

        def assert_leaves(leaf):
            assert not isinstance(leaf, torch.Tensor)

        torch.utils._pytree.tree_map(assert_leaves, td_numpy)
        assert_allclose_td(TensorDict(td_numpy), td.data.cpu())

    def test_pad(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            original_unbatched = td.get("unbatched")
            pad_size = [0, 1, 0, 2]
            padded_td = pad(td, pad_size)
            assert padded_td.get("unbatched") is original_unbatched
            return
        paddings = [
            [0, 1, 0, 2],
            [1, 0, 0, 2],
            [1, 0, 2, 1],
        ]

        for pad_size in paddings:
            padded_td = pad(td, pad_size)
            padded_td._check_batch_size()
            amount_expanded = [0] * (len(pad_size) // 2)
            for i in range(0, len(pad_size), 2):
                amount_expanded[i // 2] = pad_size[i] + pad_size[i + 1]

            for key in padded_td.keys():
                expected_dims = tuple(
                    sum(p)
                    for p in zip(
                        td[key].shape,
                        amount_expanded
                        + [0] * (len(td[key].shape) - len(amount_expanded)),
                    )
                )
                assert padded_td[key].shape == expected_dims

        with pytest.raises(RuntimeError):
            pad(td, [0] * 100)

        with pytest.raises(RuntimeError):
            pad(td, [0])

    @legacy_lazy_mode()
    def test_permute_applied_twice(self, td_name, device):
        torch.manual_seed(0)
        tensordict = getattr(self, td_name)(device)
        for _ in range(10):
            p = torch.randperm(4)
            inv_p = p.argsort()
            other_p = inv_p
            while (other_p == inv_p).all():
                other_p = torch.randperm(4)
            other_p = tuple(other_p.tolist())
            p = tuple(p.tolist())
            inv_p = tuple(inv_p.tolist())
            if td_name in ("td_params",):
                # TODO: Should we break this?
                assert (
                    tensordict.permute(*p).permute(*inv_p)._param_td
                    is tensordict._param_td
                )
                assert (
                    tensordict.permute(*p).permute(*other_p)._param_td
                    is not tensordict._param_td
                )
                assert (
                    torch.permute(tensordict, p).permute(inv_p)._param_td
                    is tensordict._param_td
                )
                assert (
                    torch.permute(tensordict, p).permute(other_p)._param_td
                    is not tensordict._param_td
                )
            else:
                assert assert_allclose_td(
                    tensordict.permute(*p).permute(*inv_p), tensordict
                )
                assert tensordict.permute(*p).permute(*inv_p) is tensordict
                assert tensordict.permute(*p).permute(*other_p) is not tensordict
                assert assert_allclose_td(
                    torch.permute(tensordict, p).permute(inv_p), tensordict
                )
                assert torch.permute(tensordict, p).permute(inv_p) is tensordict
                assert torch.permute(tensordict, p).permute(other_p) is not tensordict

    @set_lazy_legacy(False)
    def test_permute_decorator(self, td_name, device):
        td = getattr(self, td_name)(device)
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="to_tensordict")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec, td.unlock_().permute(1, 0, 3, 2) as tdt:
            if not tdt.requires_grad:
                tdt.apply_(lambda x: x * 0 + 1)
            else:
                tdt.apply(lambda x: x.data.mul_(0).add_(1))
        if is_lazy:
            return
        assert (td == 1).all()

    @pytest.mark.skipif(
        torch.cuda.device_count() == 0 and npu_device_count == 0,
        reason="no cuda or npu device detected",
    )
    @pytest.mark.parametrize(
        "device_cast", [0, f"{cur_device}:0", torch.device(f"{cur_device}:0")]
    )
    @pytest.mark.parametrize("inplace", [False, True])
    def test_pin_memory(self, td_name, device_cast, device, inplace):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td.unlock_()
        if isinstance(td, (_SubTensorDict, PersistentTensorDict)):
            with pytest.raises(RuntimeError, match="Cannot pin memory"):
                td.pin_memory()
            return
        if device.type == "cuda" or device.type == "npu":
            with pytest.raises(RuntimeError, match="cannot pin"):
                td.pin_memory()
            return
        if isinstance(td, TensorDictParams) and inplace:
            with pytest.raises(
                RuntimeError, match="Cannot pin_memory in-place with TensorDictParams."
            ):
                td_pin = td.pin_memory(inplace=inplace)
            return
        td_pin = td.pin_memory(inplace=inplace)
        assert all(leaf.is_pinned for leaf in td_pin.values(True, True))
        if inplace:
            assert td_pin is td

        td_device = td_pin.to(device_cast)
        _device_cast = torch.device(device_cast)
        assert td_device.device == _device_cast

    def test_pin_memory_unsupported(self, td_name, device):
        td = getattr(self, td_name)(device)
        if not isinstance(td, (_SubTensorDict, PersistentTensorDict)):
            pytest.skip("pin_memory pins the leaves")
        with pytest.raises(
            RuntimeError,
            match=rf"^Cannot pin memory of a {type(td).__name__}\. Call to_tensordict\(\) "
            r"before making this call\.$",
        ):
            td.pin_memory()

    def test_pop(self, td_name, device):
        td = getattr(self, td_name)(device)
        assert "a" in td.keys()
        a = td["a"].clone()
        with td.unlock_():
            out = td.pop("a")
            assert (out == a).all()
            assert "a" not in td.keys()

            assert "b" in td.keys()
            b = td["b"].clone()
            default = torch.zeros_like(b).to(device)
            assert (default != b).all()
            out = td.pop("b", default)

            assert torch.ne(out, default).all()
            assert (out == b).all()

            assert "z" not in td.keys()
            out = td.pop("z", default)
            assert (out == default).all()

            with pytest.raises(KeyError, match='key "z" not found in'):
                td.pop("z")

    def test_popitem(self, td_name, device):
        td = getattr(self, td_name)(device)
        with td.unlock_():
            keys = list(td.keys())
            key, val = td.popitem()
            assert key == keys[-1]
            assert list(td.keys()) == keys[:-1]

    @pytest.mark.parametrize(
        "red", ("mean", "nanmean", "sum", "nansum", "prod", "std", "var", "quantile")
    )
    def test_reduction(self, td_name, device, red, tmpdir):
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor incompatible with reduction ops")
        td = getattr(self, td_name)(device)
        td = _to_float(td, td_name, tmpdir)
        if red == "quantile":
            assert getattr(td, red)(0.5).batch_size == torch.Size(())
            assert getattr(td, red)(0.5, dim=None).batch_size == torch.Size(())
            assert getattr(td, red)(0.5, 1).shape == torch.Size(
                [s for i, s in enumerate(td.shape) if i != 1]
            )
            assert getattr(td, red)(0.5, 2, keepdim=True).shape == torch.Size(
                [s if i != 2 else 1 for i, s in enumerate(td.shape)]
            )
            assert isinstance(td.quantile(0.5, reduce=True), torch.Tensor)
        else:
            assert getattr(td, red)().batch_size == torch.Size(())
            assert getattr(td, red)(dim=None).batch_size == torch.Size(())
            assert getattr(td, red)(1).shape == torch.Size(
                [s for i, s in enumerate(td.shape) if i != 1]
            )
            assert getattr(td, red)(2, keepdim=True).shape == torch.Size(
                [s if i != 2 else 1 for i, s in enumerate(td.shape)]
            )
            assert isinstance(getattr(td, red)(reduce=True), torch.Tensor)

    @pytest.mark.parametrize(
        "red",
        ("mean", "nanmean", "sum", "nansum", "prod", "std", "var", "amax", "amin"),
    )
    def test_reduction_empty_dim(self, td_name, device, red, tmpdir):
        # dim=() reduces over all the elements, as torch.sum(x, dim=()) does
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor incompatible with reduction ops")
        td = getattr(self, td_name)(device)
        td = _to_float(td, td_name, tmpdir)
        tol = {}
        if red == "prod":
            # Keep the product of all the elements of a leaf finite: 60 values
            # up to 9 overflow float32, and torch.prod then gives nan or 0.0
            # depending on the order in which it multiplies.
            with torch.no_grad():
                td.apply_(lambda x: x.mul_(0.01).add_(1))
            # Products over hundreds of float32 values in another order differ
            # by up to about n * eps, above assert_close's default rtol.
            tol = {"rtol": 1e-4, "atol": 1e-5}
        leaves = dict(td.items(True, True))
        # torch.prod does not take dim=(); prod(dim=()) gives prod()
        kwargs = {} if red == "prod" else {"dim": ()}
        tdr = getattr(td, red)(dim=())
        assert tdr.batch_size == torch.Size(())
        for key, val in leaves.items():
            torch.testing.assert_close(
                tdr[key], getattr(torch, red)(val, **kwargs), equal_nan=True, **tol
            )
        torch.testing.assert_close(
            getattr(td, red)(dim=(), reduce=True),
            getattr(td, red)(reduce=True),
            equal_nan=True,
            **tol,
        )
        # a NumPy integer dim is not compared with ()
        assert (
            getattr(td, red)(dim=np.int64(0)).batch_size
            == getattr(td, red)(dim=0).batch_size
        )
        if red == "prod":
            return
        tdr = getattr(td, red)(dim=(), keepdim=True)
        assert tdr.batch_size == torch.Size([1] * td.ndim)
        for key, val in leaves.items():
            torch.testing.assert_close(
                tdr[key],
                getattr(torch, red)(val, dim=(), keepdim=True),
                equal_nan=True,
            )

    @pytest.mark.parametrize(
        "red", ("mean", "nanmean", "sum", "nansum", "prod", "std", "var", "quantile")
    )
    def test_reduction_feature(self, td_name, device, red, tmpdir):
        if td_name == "td_with_unbatched":
            pytest.skip(
                "UnbatchedTensor incompatible with reduction ops that check shape"
            )
        td = getattr(self, td_name)(device)
        td = _to_float(td, td_name, tmpdir)
        if td_name in ("nested_tensorclass", "td_h5"):
            td = td.apply(
                lambda x: torch.stack(
                    [
                        x,
                    ]
                    * 3,
                    -1,
                )
            )
        if red == "quantile":
            tdr = getattr(td, red)(0.5, dim="feature")
            assert tdr.batch_size == td.batch_size
            assert is_tensor_collection(tdr)
            tensor = getattr(td, red)(0.5, dim="feature", reduce=True)
        else:
            tdr = getattr(td, red)(dim="feature")
            assert tdr.batch_size == td.batch_size
            assert is_tensor_collection(tdr)
            tensor = getattr(td, red)(dim="feature", reduce=True)
        assert tensor.shape == td.batch_size
        assert isinstance(tensor, torch.Tensor)

    @pytest.mark.parametrize("call_del", [True, False])
    def test_remove(self, td_name, device, call_del):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        with td.unlock_():
            if call_del:
                del td["a"]
            else:
                td = td.del_("a")
        assert td is not None
        assert "a" not in td.keys()
        if td_name in ("sub_td", "sub_td2"):
            return
        td.lock_()
        with pytest.raises(RuntimeError, match="locked"):
            del td["b"]

    def test_rename_key(self, td_name, device) -> None:
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td.is_locked:
            with pytest.raises(RuntimeError, match=re.escape(_LOCK_ERROR)):
                td.rename_key_("a", "b", safe=True)
        else:
            with pytest.raises(KeyError, match="already present in TensorDict"):
                td.rename_key_("a", "b", safe=True)
        a = td.get("a")
        if td.is_locked:
            with pytest.raises(RuntimeError, match="Cannot modify"):
                td.rename_key_("a", "z")
            return
        else:
            td.rename_key_("a", "z")
        with pytest.raises(KeyError):
            td["a"]
        assert "a" not in td.keys()

        z = td.get("z")
        torch.testing.assert_close(a, z)

        new_z = torch.randn_like(z)
        if td_name in ("sub_td", "sub_td2"):
            td.set_("z", new_z)
        else:
            td.set("z", new_z)

        torch.testing.assert_close(new_z, td.get("z"))

        new_z = torch.randn_like(z)
        if td_name == "td_params":
            td.data.set_("z", new_z)
        else:
            td.set_("z", new_z)
        torch.testing.assert_close(new_z, td.get("z"))

    def test_rename_key_nested(self, td_name, device) -> None:
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td.unlock_()
        td["nested", "conflict"] = torch.zeros(td.shape)
        with pytest.raises(KeyError, match="already present in TensorDict"):
            td.rename_key_(("nested", "conflict"), "b", safe=True)
        td["nested", "first"] = torch.zeros(td.shape)
        td.rename_key_(("nested", "first"), "second")
        assert (td["second"] == 0).all()
        assert ("nested", "first") not in td.keys(True)
        td.rename_key_("second", ("nested", "back"))
        assert (td[("nested", "back")] == 0).all()
        assert "second" not in td.keys()

    def test_rename_key_overwrite(self, td_name, device) -> None:
        # with safe=False (the default), an existing entry is replaced, as in
        # TensorDict: PersistentTensorDict raised a KeyError
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td.unlock_()
        a = td.get("a").clone()
        assert td.rename_key_("a", "b") is td
        assert "a" not in td.keys()
        torch.testing.assert_close(td.get("b"), a)

    @pytest.mark.parametrize(
        "old_key,new_key",
        [("s", ("s", "s")), ("s", ("s", "x", "y")), ("a", ("a", "b"))],
    )
    @pytest.mark.parametrize("safe", [False, True])
    def test_rename_key_inside_old_key(self, td_name, device, old_key, new_key, safe):
        # a new key inside the old one moves the entry there, as pop and set
        # do: a sub-tensordict was deleted with everything below it, and a
        # leaf raised
        td = getattr(self, td_name)(device)
        td.unlock_()
        td["s", "t"] = torch.zeros(td.shape)
        td["s", "s", "u"] = torch.ones(td.shape)
        expected = td.to_tensordict()
        expected.set(new_key, expected.pop(old_key))
        assert td.rename_key_(old_key, new_key, safe=safe) is td
        assert set(td.keys(True, True)) == set(expected.keys(True, True))
        for key in expected.keys(True, True):
            torch.testing.assert_close(td[key], expected[key])

    def test_repeat(self, td_name, device):
        td = getattr(self, td_name)(device)
        assert (td.repeat(1, 1, 1, 1) == td).all()
        assert (td.repeat(2, 1, 1, 1) == torch.cat([td] * 2, 0)).all()
        assert (td.repeat(1, 2, 1, 1) == torch.cat([td] * 2, 1)).all()
        assert (td.repeat(1, 1, 2, 1) == torch.cat([td] * 2, 2)).all()
        assert (td.repeat(1, 1, 1, 2) == torch.cat([td] * 2, 3)).all()

    def test_repeat_interleave(self, td_name, device):
        td = getattr(self, td_name)(device)
        for d in [0, 1, 2, 3, -1, -2, -3, -4]:
            t = torch.empty(td.shape)
            t_shape = t.repeat_interleave(3, dim=d).shape
            td_repeat = td.repeat_interleave(3, dim=d)
            assert td_repeat.shape == t_shape
            assert td_repeat.device == td.device
            if d < 0:
                d = td.ndim + d
            a = td["a"]
            a_repeat = td_repeat["a"]
            torch.testing.assert_close(a.repeat_interleave(3, dim=d), a_repeat)

        t = torch.empty(td.shape)
        t_shape = t.repeat_interleave(3).shape
        assert t_shape == td.repeat_interleave(3).shape

    def test_repeat_interleave_tensor(self, td_name, device):
        td = getattr(self, td_name)(device)
        d = 1
        t = torch.empty(td.shape, device=td.device)
        t_shape = t.repeat_interleave(
            torch.tensor([3, 4, 5], device=td.device), dim=d
        ).shape
        td_repeat = td.repeat_interleave(
            torch.tensor([3, 4, 5], device=td.device), dim=d
        )
        assert td_repeat.shape == t_shape
        assert td_repeat.device == td.device
        if d < 0:
            d = td.ndim + d
        a = td["a"]
        a_repeat = td_repeat["a"]
        torch.testing.assert_close(
            a.repeat_interleave(torch.tensor([3, 4, 5], device=a.device), dim=d),
            a_repeat,
        )

        t = torch.empty(td.shape, device=td.device)
        t_shape = t.repeat_interleave(
            torch.tensor([3, 4, 5], device=td.device), 1
        ).shape
        assert (
            t_shape
            == td.repeat_interleave(torch.tensor([3, 4, 5], device=td.device), 1).shape
        )

    def test_replace(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            # For UnbatchedTensor, test replace with only regular tensors
            td_replace = td.replace(a=torch.zeros_like(td["a"]))
            assert td_replace is not td
            assert (td_replace["a"] == 0).all()
            # Original unbatched should be preserved
            assert (td_replace["unbatched"] == td["unbatched"]).all()
            return
        td_dict = td.to_dict()
        td_dict = torch.utils._pytree.tree_map(
            lambda x: torch.zeros_like(x) if isinstance(x, torch.Tensor) else x, td_dict
        )
        td_replace = td.replace(td_dict)
        assert td_replace is not td
        assert (td_replace == 0).all()

        td_dict = td.clone().zero_()
        td_replace = td.replace(**td_dict)
        assert td_replace is not td
        assert (td_replace == 0).all()

    def test_replace_does_not_modify_arg(self, td_name, device):
        # https://github.com/pytorch/tensordict/issues/2949
        td = getattr(self, td_name)(device)
        new_a = torch.zeros_like(td.get("a"))
        new_b = torch.ones_like(td.get("a"))
        arg = {"a": new_a}
        td_replace = td.replace(arg, new_key=new_b)
        assert list(arg) == ["a"]
        assert (td_replace.get("a") == 0).all()
        assert (td_replace.get("new_key") == 1).all()

        arg = TensorDict(
            a=new_a,
            nested=TensorDict(x=new_a, batch_size=td.batch_size),
            batch_size=td.batch_size,
            device=td.device,
        ).lock_()
        td_replace = td.replace(arg, new_key=new_b, nested={"y": new_b})
        assert set(arg.keys(True)) == {"a", "nested", ("nested", "x")}
        assert (td_replace.get("new_key") == 1).all()
        assert (td_replace.get(("nested", "x")) == 0).all()
        assert (td_replace.get(("nested", "y")) == 1).all()

    def test_repr(self, td_name, device):
        td = getattr(self, td_name)(device)
        _ = str(td)

    @pytest.mark.parametrize("lock", [False, True])
    def test_reshape(self, td_name, device, lock):
        td = getattr(self, td_name)(device)
        if lock:
            if td_name in ("sub_td", "sub_td2"):
                pytest.skip()
            td.lock_()
        td_reshape = td.reshape(td.shape)
        # assert isinstance(td_reshape, TensorDict)
        assert td_reshape.shape.numel() == td.shape.numel()
        assert td_reshape.shape == td.shape
        td_reshape = td.reshape(*td.shape)
        # assert isinstance(td_reshape, TensorDict)
        assert td_reshape.shape.numel() == td.shape.numel()
        assert td_reshape.shape == td.shape
        td_reshape = td.reshape(size=td.shape)
        # assert isinstance(td_reshape, TensorDict)
        assert td_reshape.shape.numel() == td.shape.numel()
        assert td_reshape.shape == td.shape
        td_reshape = td.reshape(-1)
        exp_instance = (
            LazyStackedTensorDict
            if isinstance(td, LazyStackedTensorDict)
            else TensorDictBase
        )
        assert isinstance(td_reshape, exp_instance)
        assert td_reshape.shape.numel() == td.shape.numel()
        assert td_reshape.shape == torch.Size([td.shape.numel()])
        td_reshape = td.reshape((-1,))
        assert isinstance(td_reshape, exp_instance)
        assert td_reshape.shape.numel() == td.shape.numel()
        assert td_reshape.shape == torch.Size([td.shape.numel()])
        td_reshape = td.reshape(size=(-1,))
        assert isinstance(td_reshape, exp_instance)
        assert td_reshape.shape.numel() == td.shape.numel()
        assert td_reshape.shape == torch.Size([td.shape.numel()])
        if td.is_locked:
            assert td_reshape.is_locked

    def test_save_load_memmap(self, td_name, device, tmpdir):
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor memmap support not yet implemented")
        if td_name in ("sub_td2",):
            pytest.skip("sub_td2 is not supported")
        td = getattr(self, td_name)(device)
        td.save(tmpdir, copy_existing=True)
        td_load = TensorDict.load_memmap(tmpdir)

        # check the shape of the leaves
        def check_shape(v0, v1):
            assert v0.shape == v1.shape

        td.apply(check_shape, td_load)
        check_shape(td, td_load)
        assert (td.cpu() == td_load.cpu()).all()
        # get a list of all the metadata and non-tensor data
        if "non_tensor" in td_name or "metadata" in td_name:
            td_non_tensor = [
                v for v in td.values(True) if isinstance(v, NonTensorDataBase)
            ]
            assert len(td_non_tensor) > 0
            td_load_non_tensor = [
                v for v in td_load.values(True) if isinstance(v, NonTensorDataBase)
            ]
            assert len(td_load_non_tensor) > 0
            for v0, v1 in zip(td_non_tensor, td_load_non_tensor):
                assert v0.data == v1.data
                assert v0.batch_size == v1.batch_size
                assert v0.shape == v1.shape

    @pytest.mark.parametrize("strict", [True, False])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_select(self, td_name, device, strict, inplace):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        keys = ["a"]
        if td_name == "td_h5":
            with pytest.raises(NotImplementedError, match="Cannot call select"):
                td.select(*keys, strict=strict, inplace=inplace)
            return

        if td_name in ("nested_stacked_td", "nested_td"):
            keys += [("my_nested_td", "inner")]

        if inplace and td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "squeezed_td",
            "unsqueezed_td",
        ):
            with pytest.raises(RuntimeError, match="Cannot call select"):
                td.select(*keys, strict=strict, inplace=inplace)
            return
        with td.unlock_() if td.is_locked else contextlib.nullcontext():
            td2 = td.select(*keys, strict=strict, inplace=inplace)
        if inplace:
            assert td2 is td
        else:
            assert td2 is not td
        if td_name == "saved_td":
            assert (len(list(td2.keys())) == len(keys)) and ("a" in td2.keys())
            assert (len(list(td2.clone().keys())) == len(keys)) and (
                "a" in td2.clone().keys()
            )
        else:
            assert (len(list(td2.keys(True, True))) == len(keys)) and (
                "a" in td2.keys()
            )
            assert (len(list(td2.clone().keys(True, True))) == len(keys)) and (
                "a" in td2.clone().keys()
            )

    @pytest.mark.parametrize("strict", [True, False])
    def test_select_exception(self, td_name, device, strict):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_h5":
            with pytest.raises(NotImplementedError, match="Cannot call select"):
                _ = td.select("tada", strict=strict)
            return

        if strict:
            with pytest.raises(KeyError):
                _ = td.select("tada", strict=strict)
        else:
            td2 = td.select("tada", strict=strict)
            assert td2 is not td
            assert len(list(td2.keys())) == 0

    @legacy_lazy_mode()
    def test_set_lazy_legacy(self, td_name, device):
        if td_name in (
            "sub_td",
            "sub_td2",
            "td_h5",
            "squeezed_td",
            "unsqueezed_td",
            "permute_td",
            "transpose_td",
        ):
            raiser = pytest.raises(RuntimeError)
            raiser_view = raiser
        else:
            raiser = contextlib.nullcontext()
            raiser_view = raiser

        def test_not_id(td, td_name=td_name, raiser=raiser, raiser_view=raiser_view):
            # view
            with raiser_view:
                td_view = td.view(-1).view(td.shape)
                if td_name in ("td_params",):
                    assert isinstance(td_view, TensorDict)
                else:
                    assert td_view is not td
            # permute
            with raiser:
                td_perm = td.permute(3, 2, 1, 0).permute(3, 2, 1, 0)
                if td_perm in ("td_params",):
                    assert isinstance(td_view, TensorDict)
                else:
                    assert td_perm is not td
            # transpose
            with raiser if "stack" not in td_name else contextlib.nullcontext():
                td_trsp = td.transpose(-1, -2).transpose(-2, -1)
                if td_name in ("td_params",):
                    assert isinstance(td_view, TensorDict)
                else:
                    assert td_trsp is not td
            # squeeze
            with raiser:
                td_squeeze = td.squeeze(-1).unsqueeze(-1)
                if td_name in ("td_params",):
                    assert isinstance(td_view, TensorDict)
                else:
                    assert td_squeeze is not td
            # unsqueeze
            with raiser:
                td_unsqueeze = td.unsqueeze(-1).squeeze(-1)
                if td_name in ("td_params",):
                    assert isinstance(td_view, TensorDict)
                else:
                    assert td_unsqueeze is not td

        def test_id(td, td_name=td_name):
            # view
            td_view = td.view(-1).view(td.shape)
            if td_name in ("td_params",):
                assert td_view._param_td is td._param_td
            else:
                assert td_view is td
            # permute
            td_perm = td.permute(3, 2, 1, 0).permute(3, 2, 1, 0)
            if td_name in ("td_params",):
                assert td_perm._param_td is td._param_td
            else:
                assert td_perm is td
            # transpose
            td_trsp = td.transpose(-1, -2).transpose(-2, -1)
            if td_name in ("td_params",):
                assert td_trsp._param_td is td._param_td
            else:
                assert td_trsp is td
            if "stacked_td" not in td_name and td_name not in (
                "unsqueezed_td",
                "squeezed_td",
            ):
                # squeeze
                td_squeeze = td.squeeze(-1).unsqueeze(-1)
                if td_name in ("td_params",):
                    assert td_squeeze._param_td is td._param_td
                else:
                    assert td_squeeze.shape == td.shape
                    assert td_squeeze is td
                # unsqueeze
                td_unsqueeze = td.unsqueeze(-1).squeeze(-1)
                if td_name in ("td_params",):
                    assert td_unsqueeze._param_td is td._param_td
                else:
                    assert td_unsqueeze.shape == td.shape
                    assert td_unsqueeze is td

        td = getattr(self, td_name)(device)
        with legacy_lazy_mode():
            assert lazy_legacy()
            test_id(td)
            with set_lazy_legacy(False):
                assert not lazy_legacy()
                test_not_id(td)
            assert lazy_legacy()
            test_id(td)

    def test_set_nested_batch_size(self, td_name, device):
        td = getattr(self, td_name)(device)
        td.unlock_()
        batch_size = torch.Size([*td.batch_size, 3])
        td.set("some_other_td", TensorDict({}, batch_size))
        assert td["some_other_td"].batch_size == batch_size

    def test_set_nontensor(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td.unlock_()
        r = torch.randn_like(td.get("a"))
        td.set("numpy", r.cpu().numpy())
        torch.testing.assert_close(td.get("numpy"), r)

    def test_set_requires_grad(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name in ("td_params",):
            td.apply(lambda x: x.requires_grad_(False))
        td.unlock_()
        assert not td.get("a").requires_grad
        if td_name in ("td_h5",):
            with pytest.raises(
                RuntimeError, match="Cannot set a tensor that has requires_grad=True"
            ):
                td.set("a", torch.randn_like(td.get("a")).requires_grad_())
            return
        if td_name in ("sub_td", "sub_td2"):
            td.set_("a", torch.randn_like(td.get("a")).requires_grad_())
        else:
            td.set("a", torch.randn_like(td.get("a")).requires_grad_())

        assert td.get("a").requires_grad

    def test_set_unexisting(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td.is_locked:
            with pytest.raises(
                RuntimeError,
                match="Cannot modify locked TensorDict. For in-place modification",
            ):
                td.set("z", torch.ones_like(td.get("a")))
        else:
            td.set("z", torch.ones_like(td.get("a")), non_blocking=False)
            assert (td.get("z") == 1).all()

    def test_setdefault_existing_key(self, td_name, device):
        td = getattr(self, td_name)(device)
        td.unlock_()
        expected = td.get("a")
        inserted = td.setdefault("a", torch.ones_like(td.get("b")))
        assert (inserted == expected).all()

    def test_setdefault_missing_key(self, td_name, device):
        td = getattr(self, td_name)(device)
        td.unlock_()
        expected = torch.ones_like(td.get("a"))
        inserted = td.setdefault("z", expected)
        assert (inserted == expected).all()

    def test_setdefault_nested(self, td_name, device):
        td = getattr(self, td_name)(device)
        td.unlock_()

        tensor = torch.randn(4, 3, 2, 1, 5, device=device)
        tensor2 = torch.ones(4, 3, 2, 1, 5, device=device)
        sub_sub_tensordict = TensorDict({"c": tensor}, [4, 3, 2, 1], device=device)
        sub_tensordict = TensorDict(
            {"b": sub_sub_tensordict}, [4, 3, 2, 1], device=device
        )
        if td_name == "td_h5":
            del td["a"]
        if td_name == "sub_td":
            td = td._source.set(
                "a", sub_tensordict.expand(2, *sub_tensordict.shape)
            )._get_sub_tensordict(1)
        elif td_name == "sub_td2":
            td = td._source.set(
                "a",
                sub_tensordict.expand(2, *sub_tensordict.shape).permute(1, 0, 2, 3, 4),
            )._get_sub_tensordict((slice(None), 1))
        else:
            td.set("a", sub_tensordict)

        # if key exists we return the existing value
        torch.testing.assert_close(td.setdefault(("a", "b", "c"), tensor2), tensor)

        if not td_name == "stacked_td":
            torch.testing.assert_close(td.setdefault(("a", "b", "d"), tensor2), tensor2)
            torch.testing.assert_close(td.get(("a", "b", "d")), tensor2)

    @pytest.mark.parametrize(
        "idx", [slice(1), torch.tensor([0]), torch.tensor([0, 1]), range(1), range(2)]
    )
    def test_setitem(self, td_name, device, idx):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if isinstance(idx, torch.Tensor) and idx.numel() > 1 and td.shape[0] == 1:
            pytest.skip("cannot index tensor with desired index")

        td_clone = td[idx].to_tensordict(retain_none=True).zero_()
        if td_name == "td_params":
            td.data[idx] = td_clone
        else:
            td[idx] = td_clone
        assert (td[idx].get("a") == 0).all()

        td_clone = torch.cat([td_clone, td_clone], 0)
        with pytest.raises(
            RuntimeError,
            match=r"differs from the source batch size|batch dimension mismatch|Cannot broadcast the tensordict",
        ):
            td[idx] = td_clone

    @pytest.mark.skipif(
        is_npu_available(),
        reason="ForeachAddScalar is not fully adapted on NPU currently",
    )
    @pytest.mark.parametrize("actual_index", [..., (..., 0), (0, ...), (0, ..., 0)])
    def test_setitem_ellipsis(self, td_name, device, actual_index):
        if td_name == "td_with_unbatched":
            pytest.skip(
                "UnbatchedTensor indexed assignment not yet implemented "
                "(internal _tensordict batch_size mismatch)"
            )
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)

        idx = actual_index
        td_clone = td.clone()
        actual_td = td_clone[idx].clone()
        if td_name in ("td_params",):
            td_set = actual_td.apply(lambda x: x.data)
        else:
            td_set = actual_td
        td_set.zero_()

        for key in actual_td.keys():
            assert (actual_td.get(key) == 0).all()

        if td_name in ("td_params",):
            td_set = td_clone.data
        else:
            td_set = td_clone

        td_set[idx] = actual_td
        for key in td_clone.keys():
            assert (td_clone[idx].get(key) == 0).all()

    def test_setitem_nested_dict_value(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)

        # Create equivalent TensorDict and dict nested values for setitem
        nested_dict_value = {"e": torch.randn(4, 3, 2, 1, 10)}
        nested_tensordict_value = TensorDict(
            nested_dict_value, batch_size=td.batch_size, device=device
        )
        td_clone1 = td.clone(recurse=True)
        td_clone2 = td.clone(recurse=True)

        td_clone1["d"] = nested_dict_value
        td_clone2["d"] = nested_tensordict_value
        assert (td_clone1 == td_clone2).all()

    def test_setitem_nestedtuple(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td.is_locked:
            td.unlock_()
        td[" a ", (("little", "story")), "about", ("myself",)] = torch.zeros(td.shape)
        assert (td[" a ", "little", "story", "about", "myself"] == 0).all()

    @pytest.mark.skipif(
        is_npu_available(),
        reason="ForeachAddScalar is not fully adapted on NPU currently",
    )
    def test_setitem_slice(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            # UnbatchedTensor indexed/slice assignment requires special handling
            pytest.skip(
                "UnbatchedTensor slice assignment not yet implemented "
                "(internal _tensordict batch_size mismatch)"
            )
            return
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:] = td.clone()
        td_set[:1] = td[:1].clone().zero_()
        assert (td[:1] == 0).all()
        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1] = td[:1].to_tensordict(retain_none=True).zero_()
        assert (td[:1] == 0).all()

        # with broadcast
        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1] = td[0].clone().zero_()
        assert (td[:1] == 0).all()
        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1] = td[0].to_tensordict(retain_none=True).zero_()
        assert (td[:1] == 0).all()

        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1, 0] = td[0, 0].clone().zero_()
        assert (td[:1, 0] == 0).all()
        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1, 0] = td[0, 0].to_tensordict(retain_none=True).zero_()
        assert (td[:1, 0] == 0).all()

        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1, :, 0] = td[0, :, 0].clone().zero_()
        assert (td[:1, :, 0] == 0).all()
        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            td_set = td.data
        else:
            td_set = td
        td_set[:1, :, 0] = td[0, :, 0].to_tensordict(retain_none=True).zero_()
        assert (td[:1, :, 0] == 0).all()

    def test_setitem_string(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td.unlock_()
        td["d"] = torch.randn(4, 3, 2, 1, 5)
        assert "d" in td.keys()

    def test_shape(self, td_name, device):
        td = getattr(self, td_name)(device)
        assert td.shape == td.batch_size

    def test_share_memory_per_type(self, td_name, device):
        td = getattr(self, td_name)(device)
        if isinstance(td, _SubTensorDict):
            with pytest.raises(
                RuntimeError,
                match="Casting a sub-tensordict values to shared memory cannot be done",
            ):
                td.share_memory_()
            return
        if isinstance(td, PersistentTensorDict):
            with pytest.raises(
                NotImplementedError,
                match="Cannot call share_memory_ on a PersistentTensorDict",
            ):
                td.share_memory_()
            return
        if td.is_memmap():
            with pytest.raises(RuntimeError, match="mutually exclusive"):
                td.share_memory_()
            return
        assert td.share_memory_() is td
        assert td.is_shared()
        if (
            (type(td) is TensorDict or isinstance(td, TypedTensorDict))
            and td.device is not None
            and td.device.type == "cuda"
        ):
            # CUDA tensors are shared already: TensorDict returns without
            # locking, and so does a TypedTensorDict, through its TensorDict
            assert not td.is_locked
        else:
            assert td.is_locked is not isinstance(td, TensorDictParams)

    @pytest.mark.parametrize("dim", [0, -1, 3])
    def test_softmax(self, td_name, device, dim):
        if td_name == "td_with_unbatched":
            pytest.skip("UnbatchedTensor incompatible with torch.softmax")
        td = getattr(self, td_name)(device)
        if td_name in ("sub_td", "sub_td2"):
            return
        with td.unlock_():
            td.apply(lambda x: x.float(), out=td)
        tds = td.softmax(dim=dim)
        assert tds.shape == td.shape
        tds._check_batch_size()
        tds._check_device()

    def test_sorted_keys(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        sorted_keys = td.sorted_keys
        i = -1
        for i, (key1, key2) in enumerate(zip(sorted_keys, td.keys())):  # noqa: B007
            assert key1 == key2
        assert i == len(td.keys()) - 1
        if td.is_locked:
            target = td
        elif td_name not in ("sub_td", "sub_td2"):  # we cannot lock sub tensordicts
            if isinstance(td, _CustomOpTensorDict):
                target = td._source
            else:
                target = td
            assert target._cache is None
            td.lock_()
        else:
            return
        sorted_keys = td.sorted_keys
        assert target._cache.get("_sorted_keys") is not None
        # Changing the returned list must not change the lock cache (#1946).
        expected = list(sorted_keys)
        sorted_keys.append("not_a_key")
        new_sorted_keys = td.sorted_keys
        assert isinstance(new_sorted_keys, list)
        assert new_sorted_keys is not sorted_keys
        assert new_sorted_keys == expected
        td.unlock_()
        assert target._cache is None

    @pytest.mark.parametrize("performer", ["torch", "tensordict"])
    @pytest.mark.parametrize("dim", range(4))
    def test_split(self, td_name, device, performer, dim):
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            original_unbatched = td.get("unbatched")
            if performer == "torch":
                tds = torch.split(td, 2, dim)
            else:
                tds = td.split(2, dim)
            for split_td in tds:
                assert (
                    split_td.get("unbatched").data_ptr()
                    == original_unbatched.data_ptr()
                )
                assert split_td.get("unbatched").batch_size == split_td.batch_size
            return
        t = torch.zeros(()).expand(td.shape)
        for dim in range(td.batch_dims):
            rep, remainder = divmod(td.shape[dim], 2)
            split_sizes = [2] * rep + [1] * remainder
            for test_split_size in (2, split_sizes):
                tensorsplit = t.split(test_split_size, dim=dim)
                length = len(tensorsplit)
                if performer == "torch":
                    tds = torch.split(td, test_split_size, dim)
                elif performer == "tensordict":
                    tds = td.split(test_split_size, dim)
                assert len(tds) == length

                for idx, split_td in enumerate(tds):
                    expected_split_dim_size = 1 if idx == rep else 2
                    expected_batch_size = tensorsplit[idx].shape
                    # Test each split_td has the expected batch_size
                    assert split_td.batch_size == torch.Size(expected_batch_size)

                    if td_name == "nested_td":
                        assert isinstance(split_td["my_nested_td"], TensorDict)
                        assert isinstance(
                            split_td["my_nested_td", "inner"], torch.Tensor
                        )

                    # Test each tensor (or nested_td) in split_td has the expected shape
                    for key, item in split_td.items():
                        expected_shape = [
                            expected_split_dim_size if dim_idx == dim else dim_size
                            for (dim_idx, dim_size) in enumerate(td[key].shape)
                        ]
                        assert item.shape == torch.Size(expected_shape)

                        if key == "my_nested_td":
                            expected_inner_tensor_size = [
                                expected_split_dim_size if dim_idx == dim else dim_size
                                for (dim_idx, dim_size) in enumerate(
                                    td[key]["inner"].shape
                                )
                            ]
                            assert item["inner"].shape == torch.Size(
                                expected_inner_tensor_size
                            )

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("strict", [True, False])
    def test_split_keys(self, td_name, device, strict, inplace):
        td = getattr(self, td_name)(device)
        keys = list(td.keys(True, True))[:3]
        keys.append("something that does not exist")
        if strict:
            with pytest.raises(KeyError):
                td.split_keys(keys, strict=strict)
            keys = keys[:-1]
        if td.is_locked and inplace:
            with pytest.raises(RuntimeError, match="Cannot modify"):
                td0, td1 = td.split_keys(keys, inplace=inplace, strict=strict)
            return
        td0, td1 = td.split_keys(keys, inplace=inplace, strict=strict)
        if inplace:
            assert td1 is td
        else:
            assert td is not td1
        for key in td0.keys(True, True):
            assert key not in td1
        for key in td1.keys(True, True):
            assert key not in td0

    @pytest.mark.parametrize("inplace", [True, False])
    def test_split_keys_callable(self, td_name, device, inplace):
        td = getattr(self, td_name)(device)
        keys = list(td.keys(True, True))[:3]
        tuple_keys = {(k,) if isinstance(k, str) else k for k in keys}
        if td.is_locked and inplace:
            with pytest.raises(RuntimeError, match="Cannot modify"):
                td.split_keys(lambda k, v: k in tuple_keys, inplace=inplace)
            return
        td0, td1 = td.split_keys(lambda k, v: k in tuple_keys, inplace=inplace)
        if inplace:
            assert td1 is td
        else:
            assert td is not td1
        assert set(td0.keys(True, True)) == set(keys)
        for key in td1.keys(True, True):
            assert key not in td0

    @legacy_lazy_mode()
    def test_squeeze_legacy(self, td_name, device, squeeze_dim=-1):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        with td.unlock_():  # make sure that the td is not locked
            td_squeeze = torch.squeeze(td, dim=-1)
            tensor_squeeze_dim = td.batch_dims + squeeze_dim
            tensor = torch.ones_like(td.get("a").squeeze(tensor_squeeze_dim))
            if td_name in ("sub_td", "sub_td2"):
                td_squeeze.set_("a", tensor)
            else:
                td_squeeze.set("a", tensor)
            assert td.batch_size[squeeze_dim] == 1
            assert (td_squeeze.get("a") == tensor).all()
            assert (td.get("a") == tensor.unsqueeze(tensor_squeeze_dim)).all()
            if td_name != "unsqueezed_td":
                assert _compare_tensors_identity(td_squeeze.unsqueeze(squeeze_dim), td)
            else:
                assert td_squeeze is td._source
            assert (td_squeeze.get("a") == 1).all()
            assert (td.get("a") == 1).all()

    @set_lazy_legacy(False)
    def test_squeeze(self, td_name, device, squeeze_dim=-1):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)

        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="to_tensordict")
            if is_lazy
            else contextlib.nullcontext()
        )

        with td.unlock_():  # make sure that the td is not locked
            with error_dec:
                td_squeeze = torch.squeeze(td, dim=-1)
            if is_lazy:
                return
            tensor_squeeze_dim = td.batch_dims + squeeze_dim
            tensor = torch.ones_like(td.get("a").squeeze(tensor_squeeze_dim))
            if td_name in ("sub_td", "sub_td2"):
                td_squeeze.set_("a", tensor)
            else:
                td_squeeze.set("a", tensor)
            assert td.batch_size[squeeze_dim] == 1
            assert (td_squeeze.get("a") == tensor).all()
            assert (td_squeeze.get("a") == 1).all()

    @set_lazy_legacy(False)
    def test_squeeze_decorator(self, td_name, device):
        td = getattr(self, td_name)(device)
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="to_tensordict")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec, td.unlock_().squeeze(-1) as tdt:
            if not tdt.requires_grad:
                tdt.apply_(lambda x: x * 0 + 1)
            else:
                tdt.apply(lambda x: x.data.mul_(0).add_(1))
        if is_lazy:
            return
        assert (td == 1).all()

    @pytest.mark.parametrize("dim", [0, -2])
    def test_squeeze_decorator_noop_dim(self, td_name, device, dim):
        # squeezing a dim whose size is not 1 changes nothing, on exit too
        td = getattr(self, td_name)(device)
        batch_size = td.batch_size
        shapes = {key: val.shape for key, val in td.items(True, True)}
        with td.unlock_().squeeze(dim) as tdt:
            assert tdt.batch_size == batch_size
            if not tdt.requires_grad:
                tdt.apply_(lambda x: x * 0 + 1)
            else:
                tdt.apply(lambda x: x.data.mul_(0).add_(1))
        assert td.batch_size == batch_size
        assert {key: val.shape for key, val in td.items(True, True)} == shapes
        assert (td == 1).all()
        # A plain call leaves the reversal on td for a later bare `with td:`
        td.squeeze(dim)
        with td:
            pass
        assert td.batch_size == batch_size

    @legacy_lazy_mode()
    def test_squeeze_with_none_legacy(self, td_name, device, squeeze_dim=None):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        # torch.squeeze no longer accepts dim=None; omitting dim is equivalent
        td_squeeze = torch.squeeze(td)
        tensor = torch.ones_like(td.get("a").squeeze())
        td_squeeze.set_("a", tensor)
        assert (td_squeeze.get("a") == tensor).all()
        if td_name == "unsqueezed_td":
            assert td_squeeze._source is td
        assert (td_squeeze.get("a") == 1).all()
        assert (td.get("a") == 1).all()

    @set_lazy_legacy(False)
    def test_squeeze_with_none(self, td_name, device, squeeze_dim=None):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="to_tensordict")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec:
            td_squeeze = torch.squeeze(td)
        if is_lazy:
            return
        assert all(d > 1 for d in td_squeeze.batch_size), td_squeeze.batch_size
        if td_name not in ("td_params",):
            tensor = torch.ones_like(td.get("a").squeeze())
            td_squeeze.set_("a", tensor)
            assert (td_squeeze.get("a") == tensor).all()
            assert (td_squeeze.get("a") == 1).all()
            assert (td.get("a") == 1).all()

    @pytest.mark.filterwarnings("error")
    @legacy_lazy_mode()
    def test_stack_onto(self, td_name, device, tmpdir):
        if td_name == "td_with_unbatched":
            # UnbatchedTensor: stack_onto has issues with UnbatchedTensor validation
            pytest.skip("UnbatchedTensor incompatible with stack_onto validation")
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_h5":
            td0 = td.clone(newfile=tmpdir / "file0.h5").apply_(lambda x: x.zero_())
            td1 = td.clone(newfile=tmpdir / "file1.h5").apply_(lambda x: x.zero_() + 1)
        else:
            td0 = td.clone()
            if td_name in ("td_params",):
                td0.data.apply_(lambda x: x.zero_())
            else:
                td0.apply_(lambda x: x.zero_())
            td1 = td.clone()
            if td_name in ("td_params",):
                td1.data.apply_(lambda x: x.zero_() + 1)
            else:
                td1.apply_(lambda x: x.zero_() + 1)

        is_lazy = (
            td_name
            in (
                "sub_td",
                "sub_td2",
                "permute_td",
                "unsqueezed_td",
                "squeezed_td",
                "td_h5",
            )
            and not lazy_legacy()
        )
        error_dec = (
            pytest.raises(RuntimeError, match="to_tensordict")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec:
            td_out = td.unsqueeze(1)
        if is_lazy:
            return
        td_out = td_out.expand(td.shape[0], 2, *td.shape[1:]).clone()
        if td_name == "td_params":
            with pytest.raises(RuntimeError, match="out.batch_size and stacked"):
                LazyStackedTensorDict.lazy_stack([td0, td1], 0, out=td_out)
            return
        td_stack = LazyStackedTensorDict.lazy_stack(
            [td0.detach(), td1.detach()], 1, out=td_out
        )
        data_ptr_set_before = {val.data_ptr() for val in decompose(td_out)}
        data_ptr_set_after = {val.data_ptr() for val in decompose(td_out)}
        assert data_ptr_set_before == data_ptr_set_after
        assert (td_stack == td_out).all()

    @pytest.mark.filterwarnings("error")
    @legacy_lazy_mode()
    def test_stack_subclasses_on_td(self, td_name, device):
        if td_name == "td_with_unbatched":
            # UnbatchedTensor: stack subclasses has validation issues with UnbatchedTensor
            pytest.skip("UnbatchedTensor incompatible with stack subclasses validation")
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td = td.expand(3, *td.batch_size).clone().zero_()
        tds_list = [getattr(self, td_name)(device) for _ in range(3)]
        if td_name == "td_params":
            with pytest.raises(RuntimeError, match="arguments don't support automatic"):
                LazyStackedTensorDict.lazy_stack(tds_list, 0, out=td)
            return
        data_ptr_set_before = {val.data_ptr() for val in decompose(td)}
        stacked_td = stack_td(tds_list, 0, out=td)
        data_ptr_set_after = {val.data_ptr() for val in decompose(td)}
        assert data_ptr_set_before == data_ptr_set_after
        assert stacked_td.batch_size == td.batch_size
        for key in ("a", "b", "c"):
            assert (stacked_td[key] == td[key]).all()

    @pytest.mark.parametrize("keep_entries", [False, True, None])
    def test_stack_tensors(self, td_name, device, keep_entries):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        with td.unlock_():
            td["a_bis"] = td["a"] + 1
            kwargs = {}
            if keep_entries is not None:
                kwargs["keep_entries"] = keep_entries
            pred_stack = torch.stack([td["a"], td["a_bis"]], -1)
            td.stack_tensors("a", "a_bis", out_key="stack", dim=-1, **kwargs)
            assert (td["stack"] == pred_stack).all()
            if keep_entries:
                assert "a" in td
            else:
                assert "a" not in td

    @pytest.mark.filterwarnings("error")
    def test_stack_tds_on_subclass(self, td_name, device):
        # Skip td_h5 on free-threaded Python - h5py may have thread-safety issues
        # that cause data_ptr changes during the operation
        if td_name == "td_h5" and sysconfig.get_config_var("Py_GIL_DISABLED"):
            pytest.skip("h5py data_ptr behavior is unreliable on free-threaded Python")
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        tds_count = td.batch_size[0]
        tds_batch_size = td.batch_size[1:]
        tds_list = [
            TensorDict(
                source={
                    "a": torch.ones(*tds_batch_size, 5),
                    "b": torch.ones(*tds_batch_size, 10),
                    "c": torch.ones(*tds_batch_size, 3, dtype=torch.long),
                },
                batch_size=tds_batch_size,
                device=device,
            )
            for _ in range(tds_count)
        ]
        data_ptr_set_before = {val.data_ptr() for val in decompose(td)}

        stacked_td = LazyStackedTensorDict.lazy_stack(tds_list, 0, out=td)
        data_ptr_set_after = {val.data_ptr() for val in decompose(td)}
        assert data_ptr_set_before == data_ptr_set_after
        assert stacked_td.batch_size == td.batch_size
        assert stacked_td is td
        for key in ("a", "b", "c"):
            assert (stacked_td[key] == 1).all()

    def test_state_dict(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        sd = td.state_dict()
        td_zero = td.clone().detach().zero_()
        td_zero.load_state_dict(sd)
        assert_allclose_td(td, td_zero)

    def test_state_dict_assign(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        sd = td.state_dict()
        td_zero = td.clone().detach().zero_()
        shallow_copy = td_zero.clone(False)
        td_zero.load_state_dict(sd, assign=True)
        assert (shallow_copy == 0).all()
        assert_allclose_td(td, td_zero)

    def test_state_dict_strict(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        sd = td.state_dict()
        td_zero = td.clone().detach().zero_()
        del sd["a"]
        td_zero.load_state_dict(sd, strict=False)
        with pytest.raises(RuntimeError):
            td_zero.load_state_dict(sd, strict=True)

    def test_tensor_split(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        assert td.batch_size == (4, 3, 2, 1)
        tensor_compare = torch.zeros(td.shape)
        tensor_split = torch.tensor_split(tensor_compare, (1, 3))
        assert len(tensor_split) == 3, tensor_split
        assert tensor_split[0].shape == (1, 3, 2, 1)
        assert tensor_split[1].shape == (2, 3, 2, 1)
        assert tensor_split[2].shape == (1, 3, 2, 1)
        tensor_split = torch.tensor_split(tensor_compare, (1, 3), 1)
        assert len(tensor_split) == 3, [t.shape for t in tensor_split]
        assert tensor_split[0].shape == (4, 1, 2, 1)
        assert tensor_split[1].shape == (4, 2, 2, 1)
        assert tensor_split[2].shape == (4, 0, 2, 1)
        tensor_split = torch.tensor_split(tensor_compare, 2, 2)
        assert len(tensor_split) == 2, [t.shape for t in tensor_split]
        assert tensor_split[0].shape == (4, 3, 1, 1)
        assert tensor_split[1].shape == (4, 3, 1, 1)

        td_split = td.tensor_split((1, 3))
        assert len(td_split) == 3, td_split
        assert td_split[0].batch_size == torch.Size([1, *td.shape[1:]])
        assert td_split[1].batch_size == torch.Size([2, *td.shape[1:]])
        assert td_split[2].batch_size == torch.Size([1, *td.shape[1:]])
        td_split = td.tensor_split((1, 3), 1)
        assert len(td_split) == 3, td_split
        assert td_split[0].batch_size == torch.Size([4, 1, *td.shape[2:]])
        assert td_split[1].batch_size == torch.Size([4, 2, *td.shape[2:]])
        assert td_split[2].batch_size == torch.Size([4, 0, *td.shape[2:]])
        td_split = td.tensor_split(2, 2)
        assert len(td_split) == 2, td_split
        assert td_split[0].batch_size == torch.Size([4, 3, 1, *td.shape[3:]])
        assert td_split[1].batch_size == torch.Size([4, 3, 1, *td.shape[3:]])

    @pytest.mark.parametrize(
        "indices_or_sections",
        [
            torch.tensor(3),
            torch.tensor([1, 3]),
            [-1],
            [-6],
            [6],
            [3, 1],
            [2, 2, 1, 5],
        ],
        ids=["0d", "1d", "neg", "neg-out-of-range", "out-of-range", "desc", "mixed"],
    )
    @pytest.mark.parametrize("dim", [0, -3])
    def test_tensor_split_indices(self, td_name, device, indices_or_sections, dim):
        # Same sections as torch.tensor_split on a tensor of the batch shape
        td = getattr(self, td_name)(device)
        expected = torch.tensor_split(torch.zeros(td.shape), indices_or_sections, dim)
        td_split = td.tensor_split(indices_or_sections, dim)
        assert [t.batch_size for t in td_split] == [e.shape for e in expected]
        expected_a = torch.tensor_split(
            td.get("a"), indices_or_sections, dim % td.batch_dims
        )
        for t, e in zip(td_split, expected_a):
            if e.numel():
                assert (t.get("a") == e).all()

    def test_tensordict_set(self, td_name, device):
        torch.manual_seed(1)
        np.random.seed(1)
        td = getattr(self, td_name)(device)
        td.unlock_()

        # test set
        val1 = np.ones(shape=(4, 3, 2, 1, 10))
        td.set("key1", val1)
        assert (td.get("key1") == 1).all()
        with pytest.raises(RuntimeError):
            td.set("key1", np.ones(shape=(5, 10)))

        # test set_
        val2 = np.zeros(shape=(4, 3, 2, 1, 10))
        td.set_("key1", val2)
        assert (td.get("key1") == 0).all()
        if td_name not in ("stacked_td", "nested_stacked_td"):
            err_msg = r"key.*smartypants.*not found in "
        elif td_name in ("td_h5",):
            err_msg = "Unable to open object"
        else:
            err_msg = "setting a value in-place on a stack of TensorDict"

        with pytest.raises(KeyError, match=err_msg):
            td.set_("smartypants", np.ones(shape=(4, 3, 2, 1, 5)))

        # test set_at_
        td.set("key2", np.random.randn(4, 3, 2, 1, 5))
        x = np.ones(shape=(2, 1, 5)) * 42
        td.set_at_("key2", x, (2, 2))
        assert (td.get("key2")[2, 2] == 42).all()

    def test_tensordict_set_dict_value(self, td_name, device):
        torch.manual_seed(1)
        np.random.seed(1)
        td = getattr(self, td_name)(device)
        td.unlock_()

        # test set
        val1 = {"subkey1": torch.ones(4, 3, 2, 1, 10)}
        td.set("key1", val1)
        assert (td.get("key1").get("subkey1") == 1).all()
        with pytest.raises(RuntimeError):
            td.set("key1", torch.ones(5, 10))

        # test set_
        val2 = {"subkey1": torch.zeros(4, 3, 2, 1, 10)}
        if td_name in ("td_params",):
            td.data.set_("key1", val2)
        else:
            td.set_("key1", val2)
        assert (td.get("key1").get("subkey1") == 0).all()

        if td_name not in ("stacked_td", "nested_stacked_td"):
            err_msg = r"key.*smartypants.*not found in "
        elif td_name in ("td_h5",):
            err_msg = "Unable to open object"
        else:
            err_msg = "setting a value in-place on a stack of TensorDict"

        with pytest.raises(KeyError, match=err_msg):
            td.set_("smartypants", np.ones(shape=(4, 3, 2, 1, 5)))

    def test_to_device_dtype_inplace(self, td_name, device):
        td = getattr(self, td_name)(device)
        if torch.cuda.is_available():
            dest = torch.device("cuda:0")
        elif is_npu_available() and npu_device_count:
            dest = torch.device("npu:0")
        # elif torch.mps.is_available():
        #     dest = torch.device("mps:0")
        else:
            dest = torch.device("cpu")

        if td_name in ("sub_td", "sub_td2"):
            cm_device = cm_dtype = pytest.raises(
                TypeError,
                match="Cannot send a _SubTensorDict instance to device/dtype inplace",
            )
        elif td_name in ("permute_td", "unsqueezed_td", "squeezed_td", "td_h5"):
            cm_device = cm_dtype = pytest.raises(
                TypeError, match="Cannot use inplace=True with"
            )
        elif td_name in ("memmap_td",) and dest.type == "cpu":
            cm_device = contextlib.nullcontext()
            cm_dtype = pytest.raises(
                RuntimeError, match="Cannot modify locked TensorDict."
            )
        elif td.is_locked:
            cm_device = cm_dtype = pytest.raises(
                RuntimeError, match="Cannot modify locked TensorDict."
            )
        else:
            cm_device = cm_dtype = contextlib.nullcontext()
        with cm_dtype:
            td.to(torch.float32, inplace=True)
            assert td.dtype == torch.float32, td

        with cm_device:
            td.to(dest, inplace=True)
            assert td.device == dest
            for v in td.values(
                True, True, is_leaf=tensordict_base._is_tensor_collection
            ):
                assert v.device == dest

    def test_to_dict_nested(self, td_name, device):
        def recursive_checker(cur_dict):
            for _, value in cur_dict.items():
                if is_tensor_collection(value):
                    return False
                elif isinstance(value, dict) and not recursive_checker(value):
                    return False
            return True

        td = getattr(self, td_name)(device)
        td.unlock_()

        # Create nested TensorDict
        nested_tensordict_value = TensorDict(
            {"e": torch.randn(4, 3, 2, 1, 10)}, batch_size=td.batch_size, device=device
        )
        td["d"] = nested_tensordict_value

        # Convert into dictionary and recursively check if the values are TensorDicts
        td_dict = td.to_dict()
        assert recursive_checker(td_dict)
        if td_name == "td_with_non_tensor":
            assert td_dict["data"]["non_tensor"] == "some text data"
        assert (TensorDict.from_dict(td_dict, auto_batch_size=False) == td).all()

    def test_to_lazystack(self, td_name, device):
        td = getattr(self, td_name)(device)
        td2 = td.to_lazystack()
        assert isinstance(td2, LazyStackedTensorDict)
        assert td2.batch_size == td.batch_size
        assert td2.names == td.names

    def test_to_namedtuple(self, td_name, device):
        def is_namedtuple(obj):
            """Check if obj is a namedtuple."""
            return isinstance(obj, tuple) and hasattr(obj, "_fields")

        td = getattr(self, td_name)(device)
        td_namedtuple = td.to_namedtuple()
        assert is_namedtuple(td_namedtuple)
        assert_allclose_td(TensorDict.from_namedtuple(td_namedtuple), td)

    def test_to_tensordict(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td2 = td.to_tensordict(retain_none=True)
        assert (td2 == td).all()

    @pytest.mark.parametrize(
        "torch_call,method_call",
        [
            (
                lambda td: torch.chunk(td, 2, 1),
                lambda td: td.chunk(2, 1),
            ),
            (
                lambda td: torch.chunk(td, chunks=2, dim=1),
                lambda td: td.chunk(2, 1),
            ),
            (
                lambda td: torch.repeat_interleave(td, 2, 1),
                lambda td: td.repeat_interleave(2, 1),
            ),
            (
                lambda td: torch.repeat_interleave(
                    td,
                    repeats=torch.tensor([1, 2, 3], device=td.device),
                    dim=1,
                    output_size=6,
                ),
                lambda td: td.repeat_interleave(
                    torch.tensor([1, 2, 3], device=td.device), 1, output_size=6
                ),
            ),
            (
                lambda td: torch.reshape(td, (-1,)),
                lambda td: td.reshape(-1),
            ),
            (
                lambda td: torch.reshape(td, shape=(-1,)),
                lambda td: td.reshape(-1),
            ),
            (
                lambda td: torch.tensor_split(td, (1, 3)),
                lambda td: td.tensor_split((1, 3)),
            ),
            (
                lambda td: torch.tensor_split(td, sections=3),
                lambda td: td.tensor_split(3),
            ),
            (
                lambda td: torch.tensor_split(td, indices=[1, 2], dim=1),
                lambda td: td.tensor_split([1, 2], 1),
            ),
            (
                lambda td: torch.tensor_split(td, torch.tensor([1, 2]), 1),
                lambda td: td.tensor_split(torch.tensor([1, 2]), 1),
            ),
            (
                lambda td: torch.tensor_split(
                    td, tensor_indices_or_sections=torch.tensor(3)
                ),
                lambda td: td.tensor_split(torch.tensor(3)),
            ),
        ],
        ids=[
            "chunk",
            "chunk-kwargs",
            "repeat_interleave",
            "repeat_interleave-tensor-kwargs",
            "reshape",
            "reshape-kwargs",
            "tensor_split",
            "tensor_split-sections",
            "tensor_split-indices",
            "tensor_split-tensor",
            "tensor_split-tensor-kwargs",
        ],
    )
    def test_torch_shape_functions(self, td_name, device, torch_call, method_call):
        td = getattr(self, td_name)(device)
        result = torch_call(td)
        expected = method_call(td)
        if isinstance(expected, tuple):
            assert isinstance(result, tuple)
            assert len(result) == len(expected)
        else:
            result, expected = (result,), (expected,)
        for r, e in zip(result, expected):
            assert type(r) is type(e)
            assert r.batch_size == e.batch_size
            assert (r == e).all()

    @legacy_lazy_mode()
    def test_transpose_legacy(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            original_unbatched = td.get("unbatched")
            tdt = td.transpose(0, 1)
            assert tdt.get("unbatched").data_ptr() == original_unbatched.data_ptr()
            assert tdt.get("unbatched").batch_size == tdt.batch_size
            return
        tdt = td.transpose(0, 1)
        assert tdt.shape == torch.Size([td.shape[1], td.shape[0], *td.shape[2:]])
        for key, value in tdt.items(True):
            assert value.shape == torch.Size(
                [td.get(key).shape[1], td.get(key).shape[0], *td.get(key).shape[2:]]
            )
        tdt = td.transpose(-1, -2)
        for key, value in tdt.items(True):
            assert value.shape == td.get(key).transpose(2, 3).shape
        if td_name in ("td_params",):
            assert tdt.transpose(-1, -2)._param_td is td._param_td
        else:
            assert tdt.transpose(-1, -2) is td
        with td.unlock_():
            tdt.set(("some", "transposed", "tensor"), torch.zeros(tdt.shape))
        assert td.get(("some", "transposed", "tensor")).shape == td.shape
        if td_name in ("td_params",):
            assert td.transpose(0, 0)._param_td is td._param_td
        else:
            assert td.transpose(0, 0) is td
        with pytest.raises(
            ValueError, match="The provided dimensions are incompatible"
        ):
            td.transpose(-5, -6)
        with pytest.raises(
            ValueError, match="The provided dimensions are incompatible"
        ):
            tdt.transpose(-5, -6)

    @set_lazy_legacy(False)
    def test_transpose(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            original_unbatched = td.get("unbatched")
            tdt = td.transpose(0, 1)
            assert tdt.get("unbatched").data_ptr() == original_unbatched.data_ptr()
            assert tdt.get("unbatched").batch_size == tdt.batch_size
            return
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="to_tensordict")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec:
            tdt = td.transpose(0, 1)
        if is_lazy:
            return
        assert tdt.shape == torch.Size([td.shape[1], td.shape[0], *td.shape[2:]])
        for key, value in tdt.items(True):
            assert value.shape == torch.Size(
                [td.get(key).shape[1], td.get(key).shape[0], *td.get(key).shape[2:]]
            )
        tdt = td.transpose(-1, -2)
        for key, value in tdt.items(True):
            assert value.shape == td.get(key).transpose(2, 3).shape
        with tdt.unlock_():
            tdt.set(("some", "transposed", "tensor"), torch.zeros(tdt.shape))
        with pytest.raises(
            ValueError,
            match="dim0 and dim1 must be within the range of the number of dimensions",
        ):
            td.transpose(-5, -6)
        with pytest.raises(
            ValueError,
            match="dim0 and dim1 must be within the range of the number of dimensions",
        ):
            tdt.transpose(-5, -6)

    @set_lazy_legacy(False)
    def test_transpose_decorator(self, td_name, device):
        td = getattr(self, td_name)(device)
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="to_tensordict")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec, td.unlock_().transpose(0, 1) as tdt:
            if not tdt.requires_grad:
                tdt.apply_(lambda x: x * 0 + 1)
            else:
                tdt.apply(lambda x: x.data.mul_(0).add_(1))
        if is_lazy:
            return
        assert (td == 1).all()

    @set_lazy_legacy(False)
    @pytest.mark.parametrize(
        "op,args,kwargs",
        [
            ("transpose", (), {"dim0": 0, "dim1": 1}),
            ("transpose", (0,), {"dim1": 1}),
            ("swapaxes", (), {"axis0": 0, "axis1": 1}),
            ("swapdims", (0,), {"axis1": 1}),
        ],
    )
    def test_transpose_decorator_keywords(self, td_name, device, op, args, kwargs):
        # the reversal reads the dims passed as keywords
        td = getattr(self, td_name)(device)
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="to_tensordict")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec, getattr(td.unlock_(), op)(*args, **kwargs) as tdt:
            if not tdt.requires_grad:
                tdt.apply_(lambda x: x * 0 + 1)
            else:
                tdt.apply(lambda x: x.data.mul_(0).add_(1))
        if is_lazy:
            return
        assert (td == 1).all()

    @set_lazy_legacy(False)
    def test_movedim(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            original_unbatched = td.get("unbatched")
            td_moved = td.movedim(0, -1)
            assert td_moved.get("unbatched").data_ptr() == original_unbatched.data_ptr()
            assert td_moved.get("unbatched").batch_size == td_moved.batch_size
            return
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="to_tensordict")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec:
            td_moved = td.movedim(0, -1)
        if is_lazy:
            return
        expected_shape = torch.Size([*td.shape[1:], td.shape[0]])
        assert td_moved.shape == expected_shape
        for key, value in td_moved.items(True):
            original_value = td.get(key)
            assert value.shape == torch.Size(
                [
                    *original_value.shape[1 : td.ndim],
                    original_value.shape[0],
                    *original_value.shape[td.ndim :],
                ]
            )

        # Test with tuple dims
        with error_dec:
            td_moved2 = td.movedim((0, 1), (1, 0))
        if is_lazy:
            return
        expected_shape2 = torch.Size([td.shape[1], td.shape[0], *td.shape[2:]])
        assert td_moved2.shape == expected_shape2

        # Test out of range error
        with pytest.raises(IndexError):
            td.movedim(0, td.ndim + 5)
        with pytest.raises(IndexError):
            td.movedim(td.ndim + 5, 0)

    @set_lazy_legacy(False)
    def test_movedim_decorator(self, td_name, device):
        td = getattr(self, td_name)(device)
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="to_tensordict")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec, td.unlock_().movedim(0, -1) as td_moved:
            if not td_moved.requires_grad:
                td_moved.apply_(lambda x: x * 0 + 1)
            else:
                td_moved.apply(lambda x: x.data.mul_(0).add_(1))
        if is_lazy:
            return
        assert (td == 1).all()

    @set_lazy_legacy(False)
    def test_swapaxes(self, td_name, device):
        td = getattr(self, td_name)(device)
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="to_tensordict")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec:
            td_swapped = td.swapaxes(0, 1)
        if is_lazy:
            return
        expected_shape = torch.Size([td.shape[1], td.shape[0], *td.shape[2:]])
        assert td_swapped.shape == expected_shape

        # Test swapdims alias
        with error_dec:
            td_swapped2 = td.swapdims(0, 1)
        assert td_swapped2.shape == expected_shape

    @set_lazy_legacy(False)
    def test_flip(self, td_name, device):
        td = getattr(self, td_name)(device)
        td_flipped = td.flip(0)
        # Shape should be unchanged
        assert td_flipped.shape == td.shape

        # Test flip multiple dims
        td_flipped2 = td.flip((0, 1))
        assert td_flipped2.shape == td.shape

    @set_lazy_legacy(False)
    def test_fliplr_flipud(self, td_name, device):
        td = getattr(self, td_name)(device)
        # All test TDs have at least 2 batch dims
        td_lr = td.fliplr()
        assert td_lr.shape == td.shape

        td_ud = td.flipud()
        assert td_ud.shape == td.shape

    @set_lazy_legacy(False)
    def test_roll(self, td_name, device):
        td = getattr(self, td_name)(device)
        td_rolled = td.roll(1, 0)
        assert td_rolled.shape == td.shape

        # Test roll multiple dims
        td_rolled2 = td.roll((1, 2), (0, 1))
        assert td_rolled2.shape == td.shape

    @set_lazy_legacy(False)
    def test_rot90(self, td_name, device):
        td = getattr(self, td_name)(device)
        original_shape = td.shape
        td_rotated = td.rot90()
        # Shape should swap first two dims
        assert td_rotated.shape == torch.Size(
            [original_shape[1], original_shape[0], *original_shape[2:]]
        )

        # Test rot90 twice should preserve shape
        td_rotated2 = td.rot90(2)
        assert td_rotated2.shape == original_shape

    @set_lazy_legacy(False)
    def test_narrow(self, td_name, device):
        td = getattr(self, td_name)(device)
        # Narrow first dim
        td_narrow = td.narrow(0, 0, 1)
        assert td_narrow.shape[0] == 1
        assert td_narrow.shape[1:] == td.shape[1:]

        # Narrow second dim
        td_narrow2 = td.narrow(1, 0, 2)
        assert td_narrow2.shape[0] == td.shape[0]
        assert td_narrow2.shape[1] == 2

    @set_lazy_legacy(False)
    def test_tile(self, td_name, device):
        if td_name in ("sub_td", "sub_td2"):
            pytest.skip("sub_td cannot be tiled due to shape constraints")
        td = getattr(self, td_name)(device)
        original_shape = td.shape
        # Tile all dims by 2
        tile_dims = (2,) * td.ndim
        td_tiled = td.tile(tile_dims)
        expected_shape = torch.Size([s * 2 for s in original_shape])
        assert td_tiled.shape == expected_shape

    @set_lazy_legacy(False)
    def test_broadcast_to(self, td_name, device):
        if td_name in ("sub_td", "sub_td2"):
            pytest.skip("sub_td cannot be broadcast due to shape constraints")
        td = getattr(self, td_name)(device)
        original_shape = td.shape
        # Broadcast to same shape
        td_broadcast = td.broadcast_to(original_shape)
        assert td_broadcast.shape == original_shape

        # Broadcast with extra dim
        new_shape = (2,) + tuple(original_shape)
        td_broadcast2 = td.broadcast_to(new_shape)
        assert td_broadcast2.shape == torch.Size(new_shape)

    @set_lazy_legacy(False)
    def test_atleast_nd(self, td_name, device):
        td = getattr(self, td_name)(device)
        original_ndim = td.ndim

        # atleast_1d
        td1 = td.atleast_1d()
        assert td1.ndim >= 1
        if original_ndim >= 1:
            assert td1 is td

        # atleast_2d
        td2 = td.atleast_2d()
        assert td2.ndim >= 2
        if original_ndim >= 2:
            assert td2 is td

        # atleast_3d
        td3 = td.atleast_3d()
        assert td3.ndim >= 3
        if original_ndim >= 3:
            assert td3 is td

    @pytest.mark.parametrize("dim", range(4))
    def test_unbind(self, td_name, device, dim):
        if td_name not in ["sub_td", "idx_td", "td_reset_bs"]:
            torch.manual_seed(1)
            td = getattr(self, td_name)(device)
            td_unbind = torch.unbind(td, dim=dim)
            assert (td == stack_td(td_unbind, dim).contiguous()).all()
            idx = (slice(None),) * dim + (0,)
            assert (td[idx] == td_unbind[0]).all()

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("separator", [",", "-"])
    def test_unflatten_keys(self, td_name, device, inplace, separator):
        td = getattr(self, td_name)(device)
        locked = td.is_locked
        td.unlock_()
        nested_nested_tensordict = TensorDict(
            {
                "a": torch.zeros(*td.shape, 2, 3),
            },
            [*td.shape, 2],
        )
        nested_tensordict = TensorDict(
            {
                "a": torch.zeros(*td.shape, 2),
                "nested_nested_tensordict": nested_nested_tensordict,
            },
            td.shape,
        )
        td["nested_tensordict"] = nested_tensordict

        if inplace and locked:
            td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
            td_flatten.lock_()
            with pytest.raises(RuntimeError, match="Cannot modify locked TensorDict"):
                td_unflatten = td_flatten.unflatten_keys(
                    inplace=inplace, separator=separator
                )
            return
        else:
            if locked:
                td.lock_()
            if td_name in ("td_h5",) and inplace:
                with pytest.raises(
                    ValueError,
                    match="Cannot call flatten_keys in_place with a PersistentTensorDict",
                ):
                    td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
                return
            if inplace and td_name in (
                "sub_td",
                "sub_td2",
                "permute_td",
                "squeezed_td",
                "unsqueezed_td",
            ):
                with pytest.raises(RuntimeError, match="Cannot call exclude"):
                    td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
                return

            td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
            td_unflatten = td_flatten.unflatten_keys(
                inplace=inplace, separator=separator
            )
        assert (td == td.empty(recurse=True).update(td_unflatten)).all()
        if inplace:
            assert td is td_unflatten

    def test_unlock(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td.unlock_()
        assert not td.is_locked
        if td.device is not None:
            assert td.device.type == "cuda" or not td.is_shared()
        else:
            assert not td.is_shared()
        assert not td.is_memmap()

    @pytest.mark.parametrize("squeeze_dim", [0, 1])
    @legacy_lazy_mode()
    def test_unsqueeze_legacy(self, td_name, device, squeeze_dim):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        with td.unlock_():  # make sure that the td is not locked
            td_unsqueeze = torch.unsqueeze(td, dim=squeeze_dim)
            tensor = torch.ones_like(td.get("a").unsqueeze(squeeze_dim))
            if td_name in ("sub_td", "sub_td2"):
                td_unsqueeze.set_("a", tensor)
            else:
                td_unsqueeze.set("a", tensor)
        assert (td_unsqueeze.get("a") == tensor).all()
        assert (td.get("a") == tensor.squeeze(squeeze_dim)).all()
        # the tensors should match
        assert _compare_tensors_identity(td_unsqueeze.squeeze(squeeze_dim), td)
        assert (td_unsqueeze.get("a") == 1).all()
        assert (td.get("a") == 1).all()

    @pytest.mark.parametrize("squeeze_dim", [0, 1])
    @set_lazy_legacy(False)
    def test_unsqueeze(self, td_name, device, squeeze_dim):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="to_tensordict")
            if is_lazy
            else contextlib.nullcontext()
        )
        with td.unlock_():  # make sure that the td is not locked
            with error_dec:
                td_unsqueeze = torch.unsqueeze(td, dim=squeeze_dim)
            if is_lazy:
                return
            tensor = torch.ones_like(td.get("a").unsqueeze(squeeze_dim))
            if td_name in ("sub_td", "sub_td2"):
                td_unsqueeze.set_("a", tensor)
            else:
                td_unsqueeze.set("a", tensor)
        assert (td_unsqueeze.get("a") == tensor).all()
        assert (td_unsqueeze.get("a") == 1).all()

    @set_lazy_legacy(False)
    def test_unsqueeze_decorator(self, td_name, device):
        td = getattr(self, td_name)(device)
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="to_tensordict")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec, td.unlock_().unsqueeze(2) as tdt:
            if not tdt.requires_grad:
                tdt.apply_(lambda x: x * 0 + 1)
            else:
                tdt.apply(lambda x: x.data.mul_(0).add_(1))
        if is_lazy:
            return
        assert (td == 1).all()

    @pytest.mark.parametrize("clone", [True, False])
    # This is needed because update in lazy permute/view etc does not behave correctly when
    # legacy is False. When these classes will be deprecated, we can just remove the decorator
    @legacy_lazy_mode()
    def test_update(self, td_name, device, clone):
        td = getattr(self, td_name)(device)
        td.unlock_()  # make sure that the td is not locked
        keys = set(td.keys())
        td.update({"x": torch.zeros(td.shape)}, clone=clone)
        assert set(td.keys()) == keys.union({"x"})
        # now with nested: using tuples for keys
        td.update({("somenested", "z"): torch.zeros(td.shape)})
        assert td["somenested"].shape == td.shape
        assert td["somenested", "z"].shape == td.shape
        td.update({("somenested", "zz"): torch.zeros(td.shape)})
        assert td["somenested"].shape == td.shape
        assert td["somenested", "zz"].shape == td.shape
        # now with nested: using nested dicts
        td["newnested"] = {"z": torch.zeros(td.shape)}
        keys = set(td.keys(True))
        assert ("newnested", "z") in keys
        td.update({"newnested": {"y": torch.zeros(td.shape)}}, clone=clone)
        keys = keys.union({("newnested", "y")})
        assert keys == set(td.keys(True))
        td.update(
            {
                ("newnested", "x"): torch.zeros(td.shape),
                ("newnested", "w"): torch.zeros(td.shape),
            },
            clone=clone,
        )
        keys = keys.union({("newnested", "x"), ("newnested", "w")})
        assert keys == set(td.keys(True))
        td.update({("newnested",): {"v": torch.zeros(td.shape)}}, clone=clone)
        keys = keys.union(
            {
                ("newnested", "v"),
            }
        )
        assert keys == set(td.keys(True))

        if td_name in ("sub_td", "sub_td2"):
            with pytest.raises(ValueError, match="Tried to replace a tensordict with"):
                td.update({"newnested": torch.zeros(td.shape)}, clone=clone)
        else:
            td.update({"newnested": torch.zeros(td.shape)}, clone=clone)
            assert isinstance(td["newnested"], torch.Tensor)

    @pytest.mark.skipif(
        is_npu_available(),
        reason="ForeachAddScalar is not fully adapted on NPU currently",
    )
    def test_update_at_(self, td_name, device):
        if td_name == "td_with_unbatched":
            pytest.skip(
                "UnbatchedTensor indexed assignment not yet implemented "
                "(internal _tensordict batch_size mismatch)"
            )
        td = getattr(self, td_name)(device)
        td0 = td[1].clone().zero_()
        td.update_at_(td0, 0)
        assert (td[0] == 0).all()

    def test_update_at_nested_dict(self, td_name, device):
        td = getattr(self, td_name)(device)
        td.unlock_()
        td.set(("n", "x"), torch.zeros(td.shape, device=device))
        td.set(("n", "y"), torch.zeros(td.shape, device=device))
        td.update_at_({"n": {"x": torch.ones(td.shape[1:], device=device)}}, 0)
        assert (td["n", "x"][0] == 1).all()
        assert (td["n", "x"][1:] == 0).all()
        assert (td["n", "y"] == 0).all()

    def test_update_at_nested_keys_to_update(self, td_name, device):
        if td_name in ("stacked_td", "nested_stacked_td"):
            pytest.skip("LazyStackedTensorDict.update_at_ has no keys_to_update")
        td = getattr(self, td_name)(device)
        td.unlock_()
        shape = td.shape[1:]
        td.set(("n", "x"), torch.zeros(td.shape, device=device))
        td.set(("n", "y"), torch.zeros(td.shape, device=device))
        x = torch.ones(shape, device=device)
        sources = [{("n", "x"): x, ("n", "y"): x * 2}]
        if td_name not in ("permute_td", "unsqueezed_td", "squeezed_td"):
            # set_at_ cannot write a nested tensordict into these
            sources.append(
                TensorDict({"n": {"x": x, "y": x * 3}}, batch_size=shape, device=device)
            )
        for source in sources:
            td.update_at_(source, 0, keys_to_update=[("n", "y")])
            assert (td["n", "y"][0] == source["n", "y"]).all()
            assert (td["n", "x"] == 0).all()

    def test_update_at_nested_time_slice(self, td_name, device):
        td = TensorDict(
            {
                "a": torch.zeros(4, 3, 5, device=device),
                "b": TensorDict(
                    {"c": torch.zeros(4, 3, 2, device=device)},
                    batch_size=[4, 3],
                    device=device,
                ),
            },
            batch_size=[4, 3],
            device=device,
        )
        td0 = TensorDict(
            {
                "a": torch.ones(4, 5, device=device),
                "b": TensorDict(
                    {"c": torch.ones(4, 2, device=device)},
                    batch_size=[4],
                    device=device,
                ),
            },
            batch_size=[4],
            device=device,
        )
        td.update_at_(td0, (slice(None), 1))
        assert (td[:, 1] == td0).all()
        assert (td[:, 0] == 0).all()
        assert (td[:, 2] == 0).all()

    def test_update_at_nested_time_slice_locked(self, td_name, device):
        td = TensorDict(
            {
                "a": torch.zeros(4, 3, 5, device=device),
                "b": TensorDict(
                    {"c": torch.zeros(4, 3, 2, device=device)},
                    batch_size=[4, 3],
                    device=device,
                ),
            },
            batch_size=[4, 3],
            device=device,
        ).lock_()
        td0 = TensorDict(
            {
                "a": torch.ones(4, 5, device=device),
                "b": TensorDict(
                    {"c": torch.ones(4, 2, device=device)},
                    batch_size=[4],
                    device=device,
                ),
            },
            batch_size=[4],
            device=device,
        )
        td.update_at_(td0, (slice(None), 2))
        assert (td[:, 2] == td0).all()

    @pytest.mark.parametrize("method", ["copy_at_", "update_at_"])
    def test_update_at_nontensor_data(self, td_name, device, method):
        td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
        newdata = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])

        if method == "copy_at_":
            getattr(td, method)(newdata, slice(1, None, 2), fast=False)
        else:
            getattr(td, method)(newdata, slice(1, None, 2))

        assert td.get("val").tolist() == [0, 1] * 5

    @legacy_lazy_mode()
    def test_update__nontensor_and_tensor(self, td_name, device):
        # update_ writes the non-tensor leaves of the source also when the source
        # shares tensor leaves with the dest.
        if td_name == "td_h5":
            pytest.skip("set_non_tensor is not compatible with td_h5")
        if td_name == "td_params":
            pytest.skip("TensorDictParams.update_ does not write non-tensor leaves")
        td = getattr(self, td_name)(device)
        source = td.clone()
        source.apply_(lambda x: x.zero_())
        source.set_non_tensor(("this", "will"), "world")
        with td.unlock_():
            td.set_non_tensor(("this", "will"), "hello")
            td.update_(source)

        def check(x):
            assert x == "world"

        torch.utils._pytree.tree_map(check, td.get_non_tensor(("this", "will")))
        assert (td.filter_non_tensor_data() == 0).all()

    def test_copy_at_fast_transition(self, td_name, device):
        td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
        newdata = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])

        with pytest.raises(RuntimeError, match="fast=True"):
            td.copy_at_(newdata, slice(1, None, 2))
        assert td.get("val").tolist() == [0] * 10

        with pytest.raises(TypeError, match="fast=None"):
            td.copy_at_(newdata, slice(1, None, 2), fast=None)
        assert td.get("val").tolist() == [0] * 10

    def test_copy_at_fast_non_tensor_warns(self, td_name, device):
        # #2951: fast=True skips the non-tensor entries of a source that also
        # holds tensors, and warns that this raises from TensorDict 0.17.
        td = TensorDict(
            x=torch.zeros(3), a=TensorDict(y=torch.zeros(3)), batch_size=[3]
        )
        td.set_non_tensor("z", "a")
        source = TensorDict(x=torch.ones(()), a=TensorDict(y=torch.ones(())))
        nested = source.clone()
        source.set_non_tensor("z", "b")
        nested.set_non_tensor(("a", "z"), "b")
        for src in (source, nested):
            with pytest.warns(FutureWarning, match="TensorDict 0.17"):
                td.copy_at_(src, 0)
            assert td["x"].tolist() == [1, 0, 0]
            assert td["a", "y"].tolist() == [1, 0, 0]
            assert td.get_non_tensor("z") == "a"
            assert ("a", "z") not in td.keys(True)
            td.zero_()

        # the calls whose result stays the same in 0.17 do not warn
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            td.copy_at_(source.filter_non_tensor_data(), 1)
            with pytest.raises(RuntimeError, match="fast=True"):
                td.copy_at_(source.select("z"), 1)
            td.copy_at_(source, 2, fast=False)
        assert not [w for w in record if issubclass(w.category, FutureWarning)]
        assert td["x"].tolist() == [0, 1, 1]
        assert td.get("z").tolist() == ["a", "a", "b"]

    # This is needed because update in lazy permute/view etc does not behave correctly when
    # legacy is False. When these classes will be deprecated, we can just remove the decorator
    @legacy_lazy_mode()
    def test_update_select(self, td_name, device):
        if td_name in ("memmap_td",):
            pytest.skip(reason="update not possible with memory-mapped td")
        td = getattr(self, td_name)(device)
        t = lambda: torch.zeros(()).expand((4, 3, 2, 1))
        other_td = TensorDict(
            {
                "My": {"father": {"was": t(), "a": t()}, "relentlessly": t()},
                "self-improving": t(),
            },
            batch_size=(4, 3, 2, 1),
        )
        td.update(
            other_td,
            keys_to_update=(("My", ("father",), "was"), ("My", "relentlessly")),
        )
        assert ("My", "father", "was") in td.keys(True)
        assert ("My", ("father",), "was") in td.keys(True)
        assert ("My", "relentlessly") in td.keys(True)
        assert ("My", "father", "a") in td.keys(True)
        assert ("self-improving",) not in td.keys(True)
        t = lambda: torch.ones(()).expand((4, 3, 2, 1))
        other_td = TensorDict(
            {
                "My": {"father": {"was": t(), "a": t()}, "relentlessly": t()},
                "self-improving": t(),
            },
            batch_size=(4, 3, 2, 1),
        )
        td.update(other_td, keys_to_update=(("My", "relentlessly"),))
        assert (td["My", "relentlessly"] == 1).all()
        assert (td["My", "father", "was"] == 0).all()
        td.update(other_td, keys_to_update=(("My", ("father",), "was"),))
        assert (td["My", "father", "was"] == 1).all()

    @pytest.mark.parametrize(
        "index", ["tensor1", "mask", "int", "range", "tensor2", "slice_tensor"]
    )
    def test_update_subtensordict(self, td_name, device, index):
        if td_name == "td_with_unbatched":
            # UnbatchedTensor: subtensordict update has validation issues
            pytest.skip("UnbatchedTensor incompatible with subtensordict update")
        td = getattr(self, td_name)(device)
        if index == "mask":
            index = torch.zeros(td.shape[0], dtype=torch.bool, device=device)
            index[-1] = 1
        elif index == "int":
            index = td.shape[0] - 1
        elif index == "range":
            index = range(td.shape[0] - 1, td.shape[0])
        elif index == "tensor1":
            index = torch.tensor(td.shape[0] - 1, device=device)
        elif index == "tensor2":
            index = torch.tensor([td.shape[0] - 2, td.shape[0] - 1], device=device)
        elif index == "slice_tensor":
            index = (
                slice(None),
                torch.tensor([td.shape[1] - 2, td.shape[1] - 1], device=device),
            )

        sub_td = td._get_sub_tensordict(index)
        assert sub_td.shape == td.to_tensordict(retain_none=True)[index].shape
        assert sub_td.shape == td[index].shape, (td, index)
        td0 = td[index]
        td0 = td0.to_tensordict(retain_none=True)
        td0 = td0.apply(lambda x: x * 0 + 2)
        assert sub_td.shape == td0.shape
        if td_name == "td_params":
            with pytest.raises(RuntimeError, match="a leaf Variable"):
                sub_td.update(td0)
            return
        with td.unlock_():
            sub_td.update(td0)
        assert (sub_td == 2).all()
        assert (td[index] == 2).all()

    @legacy_lazy_mode()
    def test_view_legacy(self, td_name, device):
        if td_name in ("permute_td", "sub_td2"):
            pytest.skip("view incompatible with stride / permutation")
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        with td.unlock_():  # make sure that the td is not locked
            td_view = td.view(-1)
            tensor = td.get("a")
            tensor = tensor.view(-1, tensor.numel() // prod(td.batch_size))
            tensor = torch.ones_like(tensor)
            if td_name == "sub_td":
                td_view.set_("a", tensor)
            else:
                td_view.set("a", tensor)
            assert (td_view.get("a") == tensor).all()
            assert (td.get("a") == tensor.view(td.get("a").shape)).all()
            if td_name in ("td_params",):
                assert td_view.view(td.shape)._param_td is td._param_td
                assert td_view.view(*td.shape)._param_td is td._param_td
            else:
                assert td_view.view(td.shape) is td
                assert td_view.view(*td.shape) is td
            assert (td_view.get("a") == 1).all()
            assert (td.get("a") == 1).all()

    @set_lazy_legacy(False)
    def test_view(self, td_name, device):
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="Cannot call `view`")
            if is_lazy
            else contextlib.nullcontext()
        )
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        with td.unlock_():  # make sure that the td is not locked
            with error_dec:
                td_view = td.view(-1)
            if is_lazy:
                return
            tensor = td.get("a")
            tensor = tensor.view(-1, tensor.numel() // prod(td.batch_size))
            tensor = torch.ones_like(tensor)
            if td_name == "sub_td":
                td_view.set_("a", tensor)
            else:
                td_view.set("a", tensor)
            assert (td_view.get("a") == tensor).all()

            assert (td_view.get("a") == 1).all()

    @set_lazy_legacy(False)
    def test_view_dtype(self, td_name, device):
        if td_name == "td_with_unbatched":
            # UnbatchedTensor: view_dtype operates on underlying data, needs special handling
            pytest.skip("UnbatchedTensor view_dtype requires special handling")
        td = getattr(self, td_name)(device)
        tview = td.view(torch.uint8, batch_size=[])
        assert all(p.dtype == torch.uint8 for p in tview.values(True, True))

    @set_lazy_legacy(False)
    def test_view_decorator(self, td_name, device):
        td = getattr(self, td_name)(device)
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="Cannot call `view`")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec, td.unlock_().view(-1) as tdt:
            if not tdt.requires_grad:
                tdt.apply_(lambda x: x * 0 + 1)
            else:
                tdt.apply(lambda x: x.data.mul_(0).add_(1))
        if is_lazy or "stack" in td_name:
            return
        assert (td == 1).all()

    def test_where(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            # UnbatchedTensor: where has expand_as_right issues with mismatched dimensions
            pytest.skip(
                "UnbatchedTensor where() has expand_as_right dimension mismatch"
            )
            return
        mask = torch.zeros(td.shape, dtype=torch.bool, device=device).bernoulli_()
        td_where = torch.where(mask, td, 0)
        for k in td.keys(True, True):
            assert (td_where.get(k)[~mask] == 0).all()
        td_where = torch.where(mask, td, torch.ones_like(td))
        for k in td.keys(True, True):
            assert (td_where.get(k)[~mask] == 1).all()
        td_where = td.clone()

        if td_name == "td_h5":
            with pytest.raises(
                RuntimeError,
                match="Cannot use a persistent tensordict as output of torch.where",
            ):
                torch.where(mask, td, torch.ones_like(td), out=td_where)
            return
        torch.where(mask, td, torch.ones_like(td), out=td_where)
        for k in td.keys(True, True):
            assert (td_where.get(k)[~mask] == 1).all()

    def test_where_td_other(self, td_name, device):
        # torch.where(mask, value, td), with the tensordict as other (#2743)
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            pytest.skip(
                "UnbatchedTensor where() has expand_as_right dimension mismatch"
            )
        # the mask covers the leading batch dim and broadcasts over the others
        mask = torch.zeros(td.shape[:1], dtype=torch.bool, device=device).bernoulli_()
        for value in (0, torch.tensor(2, device=device)):
            td_where = torch.where(mask, value, td)
            assert td_where.batch_size == td.batch_size
            for k in td.keys(True, True):
                leaf = td.get(k)
                expected = torch.where(expand_as_right(mask, leaf), value, leaf)
                assert (td_where.get(k) == expected).all()
        # a uint8 mask (deprecated by torch, still accepted) selects the same values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            td_where = torch.where(mask.to(torch.uint8), 0, td)
        for k in td.keys(True, True):
            leaf = td.get(k)
            expected = torch.where(expand_as_right(mask, leaf), 0, leaf)
            assert (td_where.get(k) == expected).all()
        if td_name == "td_h5":
            return
        td_where = td.clone()
        value = torch.tensor(2, device=device)
        assert torch.where(mask, value, td, out=td_where) is td_where
        for k in td.keys(True, True):
            leaf = td.get(k)
            expected = torch.where(expand_as_right(mask, leaf), value, leaf)
            assert (td_where.get(k) == expected).all()

    def test_where_pad(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            # UnbatchedTensor: where_pad has expand_as_right issues with mismatched dimensions
            pytest.skip(
                "UnbatchedTensor where_pad() has expand_as_right dimension mismatch"
            )
            return
        # test with other empty td
        mask = torch.zeros(td.shape, dtype=torch.bool, device=td.device).bernoulli_()
        if td_name in ("td_h5",):
            td_full = td.to_tensordict(retain_none=True)
        else:
            td_full = td
        td_empty = td_full.empty()
        result = td.where(mask, td_empty, pad=1)
        for v in result.values(True, True):
            assert (v[~mask] == 1).all()
        td_empty = td_full.empty()
        result = td_empty.where(~mask, td, pad=1)
        for v in result.values(True, True):
            assert (v[~mask] == 1).all()
        # with output
        td_out = td_full.empty()
        result = td.where(mask, td_empty, pad=1, out=td_out)
        for v in result.values(True, True):
            assert (v[~mask] == 1).all()
        if td_name not in ("td_params",):
            assert result is td_out
        # TODO: decide if we want where to return a TensorDictParams.
        # probably not, given
        # else:
        #     assert isinstance(result, TensorDictParams)
        td_out = td_full.empty()
        td_empty = td_full.empty()
        result = td_empty.where(~mask, td, pad=1, out=td_out)
        for v in result.values(True, True):
            assert (v[~mask] == 1).all()
        assert result is td_out

        with pytest.raises(KeyError, match="not found and no pad value provided"):
            td.where(mask, td_full.empty())
        with pytest.raises(KeyError, match="not found and no pad value provided"):
            td_full.empty().where(mask, td)

    def test_where_out_scalar(self, td_name, device):
        # A scalar or tensor ``other`` with ``out`` used to raise a TypeError
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            pytest.skip(
                "UnbatchedTensor where() has expand_as_right dimension mismatch"
            )
        mask = torch.zeros(td.shape, dtype=torch.bool, device=device).bernoulli_()
        expected = td.where(mask, -1)
        for other in (-1, torch.tensor(-1, device=device)):
            td_out = td.to_tensordict()
            result = td.where(mask, other, out=td_out)
            assert result is td_out
            for key, value in expected.items(True, True):
                if isinstance(value, torch.Tensor):
                    assert (result.get(key) == value).all()

    def test_write_on_subtd(self, td_name, device):
        td = getattr(self, td_name)(device)
        sub_td = td._get_sub_tensordict(0)
        # should not work with td_params
        if td_name == "td_params":
            with pytest.raises(RuntimeError, match="a view of a leaf"):
                sub_td["a"] = torch.full((3, 2, 1, 5), 1.0, device=device)
            return
        sub_td["a"] = torch.full((3, 2, 1, 5), 1.0, device=device)
        assert (td["a"][0] == 1).all()

    def test_zero_(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        new_td = td.zero_()
        assert new_td is td
        for k in td.keys():
            assert (td.get(k) == 0).all()

    def test_detach_(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name in ("sub_td", "sub_td2"):
            with pytest.raises(RuntimeError, match="sub-tensordict in-place"):
                td.detach_()
            return
        if td_name in ("idx_td", "memmap_td"):
            # the leaves are views, which torch cannot detach in place
            with pytest.raises(RuntimeError, match="Can't detach views in-place"):
                td.detach_()
            return
        assert td.detach_() is td
        for value in td.values(True, True):
            assert not value.requires_grad

    @pytest.mark.parametrize("set_to_none", [True, False])
    def test_zero_grad(self, td_name, device, set_to_none):
        td = getattr(self, td_name)(device)
        if td_name == "td_with_unbatched":
            # UnbatchedTensor: backward() fails because UnbatchedTensor isn't scalar
            pytest.skip("UnbatchedTensor zero_grad: backward() requires scalar output")
            return
        tdr = td.float().requires_grad_()
        td1 = tdr + 1
        # "a" is left out of the loss, so its grad stays None
        sum(td1.exclude("a").sum().values(True, True)).backward()
        assert (tdr.grad == 1).all(), tdr.grad.to_dict()
        assert "a" not in tdr.grad.keys()
        tdr.zero_grad(set_to_none=set_to_none)
        if set_to_none:
            assert tdr.filter_non_tensor_data().grad is None, (td, tdr, tdr.grad)
        else:
            assert (tdr.grad == 0).all()
            assert "a" not in tdr.grad.keys()

    def test_autograd_grad(self, td_name, device):
        td = getattr(self, td_name)(device)
        inputs = td.float().requires_grad_()
        outputs = inputs + 1
        grads = torch.autograd.grad(outputs, inputs, torch.ones_like(outputs))
        assert (grads == 1).all()

    def test_backward(self, td_name, device):
        td = getattr(self, td_name)(device)
        inputs = td.float().requires_grad_()
        outputs = inputs + 1
        outputs.backward(torch.ones_like(outputs))
        assert (inputs.grad == 1).all()


# The unary pointwise methods of tensordict/_base/pointwise.py, each with an
# in-place form named "<op>_".
_POINTWISE_UNARY_OPS = (
    "abs",
    "acos",
    "asin",
    "atan",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "floor",
    "frac",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "neg",
    "reciprocal",
    "round",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
)
# Input ranges: the default keeps acos and asin in their domain, the positive
# range keeps log*, sqrt and reciprocal in theirs, and the wide range keeps
# ceil, floor, frac, round and trunc from being the identity or a constant.
_POINTWISE_DEFAULT_RANGE = (-0.95, 0.95)
_POINTWISE_POSITIVE_RANGE = (0.05, 0.95)
_POINTWISE_WIDE_RANGE = (-2.9, 2.9)
_POINTWISE_RANGES = {
    **dict.fromkeys(
        ("log", "log10", "log2", "sqrt", "reciprocal"), _POINTWISE_POSITIVE_RANGE
    ),
    **dict.fromkeys(("ceil", "floor", "frac", "round", "trunc"), _POINTWISE_WIDE_RANGE),
}
# name: (number of tensordict operands, other positional args, keyword args).
# The same arguments are passed to the method and to the torch function.
_POINTWISE_BINARY_OPS = {
    "add": (1, (), {"alpha": 2.0}),
    "sub": (1, (), {"alpha": 2.0}),
    "maximum": (1, (), {}),
    "minimum": (1, (), {}),
    "lerp": (1, (0.3,), {}),
    "addcdiv": (2, (), {"value": 0.5}),
    "addcmul": (2, (), {"value": 0.5}),
}
_POINTWISE_TD_NAMES = (
    "td",
    "nested_td",
    "nested_stacked_td",
    "sub_td",
    "memmap_td",
    "td_h5",
)


@pytest.mark.parametrize(
    "td_name,device",
    [
        [td_name, device]
        for td_name, device in TestTensorDictsBase.TYPES_DEVICES
        if td_name in _POINTWISE_TD_NAMES
    ],
)
class TestPointwise(TestTensorDictsBase):
    """Compares the pointwise methods with the torch function applied to each leaf."""

    def _float_td(self, td_name, device, value_range):
        td = getattr(self, td_name)(device)
        td.unlock_()
        # "c" holds integers, which the in-place forms cannot write a float into
        del td["c"]
        td.apply_(lambda x: x.uniform_(*value_range))
        return td

    def _operand(self, td, value_range, operand_type):
        # A tensordict of the same structure as td, with other values: of the
        # same type as td (a lazy stack for a lazy stack), or dense (#1932)
        if operand_type == "dense" or isinstance(td, PersistentTensorDict):
            other = td.to_tensordict()
        else:
            other = td.clone()
        return other.apply_(lambda x: x.uniform_(*value_range))

    @pytest.mark.parametrize("op", _POINTWISE_UNARY_OPS)
    def test_unary(self, td_name, device, op):
        value_range = _POINTWISE_RANGES.get(op, _POINTWISE_DEFAULT_RANGE)
        td = self._float_td(td_name, device, value_range)
        expected = {key: getattr(torch, op)(val) for key, val in td.items(True, True)}
        out = getattr(td, op)()
        assert out is not td
        assert set(out.keys(True, True)) == set(expected)
        for key, val in expected.items():
            torch.testing.assert_close(out.get(key), val)

    @pytest.mark.parametrize("op", _POINTWISE_UNARY_OPS)
    def test_unary_(self, td_name, device, op):
        value_range = _POINTWISE_RANGES.get(op, _POINTWISE_DEFAULT_RANGE)
        td = self._float_td(td_name, device, value_range)
        expected = {key: getattr(torch, op)(val) for key, val in td.items(True, True)}
        assert getattr(td, op + "_")() is td
        for key, val in expected.items():
            torch.testing.assert_close(td.get(key), val)

    @pytest.mark.parametrize("operand_type", ["same", "dense"])
    @pytest.mark.parametrize("inplace", [False, True])
    @pytest.mark.parametrize("op", sorted(_POINTWISE_BINARY_OPS))
    def test_binary(self, td_name, device, op, inplace, operand_type):
        num_operands, args, kwargs = _POINTWISE_BINARY_OPS[op]
        # addcdiv divides by its second operand, which must not be close to 0
        td = self._float_td(td_name, device, _POINTWISE_POSITIVE_RANGE)
        operands = [
            self._operand(td, _POINTWISE_POSITIVE_RANGE, operand_type)
            for _ in range(num_operands)
        ]
        expected = {
            key: getattr(torch, op)(
                val, *[operand.get(key) for operand in operands], *args, **kwargs
            )
            for key, val in td.items(True, True)
        }
        if inplace:
            assert getattr(td, op + "_")(*operands, *args, **kwargs) is td
            out = td
        else:
            out = getattr(td, op)(*operands, *args, **kwargs)
            assert out is not td
        assert set(out.keys(True, True)) == set(expected)
        for key, val in expected.items():
            torch.testing.assert_close(out.get(key), val)


@pytest.mark.skipif(PYTORCH_TEST_FBCODE, reason="vmap now working in fbcode")
@pytest.mark.parametrize("op", _POINTWISE_UNARY_OPS)
def test_pointwise_unary_vmap(op):
    # Under vmap, _foreach calls torch.Tensor.<op> instead of torch._foreach_<op>
    value_range = _POINTWISE_RANGES.get(op, _POINTWISE_DEFAULT_RANGE)
    td = TensorDict(
        a=torch.empty(4, 3).uniform_(*value_range),
        nested=TensorDict(b=torch.empty(4, 2).uniform_(*value_range), batch_size=[4]),
        batch_size=[4],
    )
    expected = td.apply(getattr(torch, op))
    assert_allclose_td(torch.vmap(lambda t: getattr(t, op)())(td), expected)
    assert_allclose_td(
        torch.vmap(lambda t: getattr(t.clone(), op + "_")())(td), expected
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
@pytest.mark.parametrize(
    "op",
    [
        lambda x, t: x + t,
        lambda x, t: t + x,
        lambda x, t: x - t,
        lambda x, t: t - x,
        lambda x, t: x.add(t, alpha=2),
        lambda x, t: x.sub(t, alpha=2),
        lambda x, t: x.clone().add_(t),
        lambda x, t: x.clone().sub_(t),
        lambda x, t: x.clone().sub_(t, alpha=2),
        lambda x, t: x.add(t, alpha=np.float32(2)),
        lambda x, t: x.clone().add_(t, alpha=torch.tensor(2.0)),
    ],
    ids=[
        "add",
        "radd",
        "sub",
        "rsub",
        "add-alpha",
        "sub-alpha",
        "add_",
        "sub_",
        "sub_-alpha",
        "add-alpha-numpy",
        "add_-alpha-tensor",
    ],
)
def test_add_sub_cuda_scalar_tensor_no_sync(op):
    # Python passes a 0-dim tensor to the overload of _foreach_add or
    # _foreach_sub that takes a number, which synchronizes the device (#2616)
    td = TensorDict(
        a=torch.rand(3, 2, device="cuda"),
        nested=TensorDict(b=torch.rand(3, device="cuda", dtype=torch.float64)),
        batch_size=[3],
    )
    t = torch.tensor(0.75, device="cuda")
    expected = td.apply(lambda x: op(x, t))
    op(td, t)  # warm up outside the check
    torch.cuda.synchronize()
    try:
        with warnings.catch_warnings():
            # "Synchronization debug mode is a prototype feature"
            warnings.simplefilter("ignore", UserWarning)
            torch.cuda.set_sync_debug_mode("error")
        out = op(td, t)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    for key, val in expected.items(True, True):
        assert out.get(key).dtype == val.dtype
        torch.testing.assert_close(out.get(key), val, rtol=0, atol=0)


@pytest.mark.parametrize("mode", [None, "cpu"])
@pytest.mark.parametrize(
    "op",
    [
        lambda x, t: x.sub(t, alpha=2),
        lambda x, t: x.clone().sub_(t, alpha=2),
        lambda x, t: x.sub(t, alpha=-0.5),
    ],
    ids=["sub-alpha", "sub_-alpha", "sub-float-alpha"],
)
def test_sub_tensor_alpha(op, mode):
    # torch._foreach_sub has no overload for one tensor with alpha (#2709)
    td = TensorDict(
        a=torch.rand(3, 2),
        nested=TensorDict(b=torch.rand(3, 2, dtype=torch.float64), batch_size=[3]),
        batch_size=[3],
    )
    t = torch.tensor(0.75)
    expected = td.apply(lambda x: op(x, t))
    with torch.device(mode) if mode else contextlib.nullcontext():
        out = op(td, t)
    for key, val in expected.items(True, True):
        assert out.get(key).dtype == val.dtype
        torch.testing.assert_close(out.get(key), val, rtol=0, atol=0)


def test_sub_tensor_alpha_int_leaves():
    # An integer leaf gets the dtype that Tensor.sub gives it (#2709)
    td = TensorDict(a=torch.arange(3), b=torch.ones(3))
    t = torch.tensor(0.5)
    out = td.sub(t, alpha=2)
    assert out["a"].dtype == torch.arange(3).sub(t, alpha=2).dtype
    torch.testing.assert_close(out["a"], torch.arange(3).sub(t, alpha=2))
    torch.testing.assert_close(out["b"], torch.zeros(3))
    # A bool alpha keeps raising, as torch._foreach_sub did
    with pytest.raises(TypeError):
        td.exclude("a").sub(t, alpha=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
def test_add_cuda_scalar_tensor_other_leaves():
    # Integer and CPU leaves keep the result of torch._foreach_add(vals, t)
    t = torch.tensor(0.5, device="cuda", dtype=torch.float64)
    td = TensorDict(a=torch.arange(3, device="cuda"), b=torch.ones(3, device="cuda"))
    assert (td + t)["a"].dtype == torch.float32
    assert (td.cpu() + t)["a"].device == torch.device("cpu")
    with pytest.raises((NotImplementedError, RuntimeError), match="bool tensor"):
        td.exclude("a") - torch.tensor(True, device="cuda")
    with pytest.raises((NotImplementedError, RuntimeError), match="bool tensor"):
        torch.tensor(True, device="cuda") - td.exclude("a")
    # A bool alpha keeps the path of main, where sub takes no alpha
    with pytest.raises(TypeError):
        td.exclude("a").sub(torch.tensor(0.5, device="cuda"), alpha=True)

    class SubTensor(torch.Tensor):
        pass

    # A subclass of Tensor does not turn the leaves into that subclass
    out = td.exclude("a") + t.float().as_subclass(SubTensor)
    assert type(out["b"]) is torch.Tensor

    class LoggedTensor(torch.Tensor):
        funcs = []

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            cls.funcs.append(func)
            return super().__torch_function__(func, types, args, kwargs or {})

    # Leaves of such a subclass still see the _foreach op of the method
    leaves = TensorDict(b=torch.ones(3, device="cuda").as_subclass(LoggedTensor))
    leaves.sub(t.float())
    assert LoggedTensor.funcs == [torch._foreach_sub]

    class LoggingMode(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.funcs = set()

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.funcs.add(getattr(func, "__name__", str(func)))
            return func(*args, **(kwargs or {}))

    # A torch function mode sees the _foreach op of the method, and nothing else
    for op, name in [("sub", "_foreach_sub"), ("add_", "_foreach_add_")]:
        leaves = TensorDict(b=torch.ones(3, device="cuda"))
        with torch.device("cuda"), LoggingMode() as mode:
            out = getattr(leaves, op)(t.float())
        torch.testing.assert_close(out["b"], getattr(torch.ones(3), op)(0.5).cuda())
        assert name in mode.funcs
        assert not mode.funcs & {"get_device", "clone", "_foreach_add"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
@pytest.mark.parametrize(
    "dtype,other_dtype,value",
    [
        (torch.bfloat16, torch.float32, 0.1234567),
        (torch.bfloat16, torch.float16, 0.1234567),
        (torch.float16, torch.float64, 0.1234567),
        (torch.bfloat16, torch.int64, 257),
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        lambda x, t: x + t,
        lambda x, t: x - t,
        lambda x, t: t - x,
        lambda x, t: x.clone().add_(t),
        lambda x, t: x.clone().sub_(t),
    ],
    ids=["add", "sub", "rsub", "add_", "sub_"],
)
def test_add_sub_cuda_scalar_tensor_half_leaves(op, dtype, other_dtype, value):
    # Half-precision leaves with a wider 0-dim tensor keep the values they get
    # with the number, which differ from those of the 0-dim tensor overload
    td = TensorDict(a=torch.linspace(0, 3, 1000, device="cuda", dtype=dtype))
    t = torch.tensor(value, device="cuda", dtype=other_dtype)
    expected = op(td, t.item())
    torch.testing.assert_close(op(td, t)["a"], expected["a"], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
@pytest.mark.parametrize("op", ["add_", "sub_"])
@pytest.mark.parametrize("view", [True, False], ids=["element", "leaf"])
def test_add_sub_cuda_scalar_tensor_inplace_alias(op, view):
    # A 0-dim tensor held by a leaf is read before any leaf is written, as with
    # the number. Only CUDA leaves take the 0-dim tensor overload; with this
    # many elements its kernels read the 0-dim tensor after writing ``a``.
    a = torch.full((1,) if view else (), 0.5, device="cuda")
    td = TensorDict(
        a=a, **{f"b{i}": torch.ones(200_000, device="cuda") for i in range(20)}
    )
    t = a[0] if view else a
    expected = getattr(td.clone(), op)(t.item())
    getattr(td, op)(t)
    for key, val in expected.items():
        torch.testing.assert_close(td[key], val, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
@pytest.mark.parametrize("inference", [False, True], ids=["changed", "inference"])
@pytest.mark.parametrize(
    "op",
    [lambda x, t: x + t, lambda x, t: x - t, lambda x, t: t - x],
    ids=["add", "sub", "rsub"],
)
def test_add_sub_cuda_scalar_tensor_backward(op, inference):
    # As with the number, the 0-dim tensor is not saved for backward: it can be
    # changed in place before backward, or come from inference mode
    if inference:
        with torch.inference_mode():
            t = torch.tensor(0.25, device="cuda")
    else:
        t = torch.tensor(0.25, device="cuda")
    x = torch.ones(3, device="cuda", requires_grad=True)
    out = op(TensorDict(a=x * 2), t)
    if not inference:
        t.add_(1)
    out["a"].sum().backward()
    y = torch.ones(3, device="cuda", requires_grad=True)
    op(y * 2, 0.25).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
@pytest.mark.parametrize(
    "op,number",
    [
        (lambda x, t: x + t, True),
        (lambda x, t: x.sub(t), True),
        (lambda x, t: t - x, True),
        (lambda x, t: x.clone().add_(t), True),
        (lambda x, t: x.clone().sub_(t), True),
        (lambda x, t: x.add(t, alpha=2), False),
    ],
    ids=["add", "sub", "rsub", "add_", "sub_", "add-alpha"],
)
def test_add_sub_cuda_scalar_tensor_forward_ad(op, number):
    # Without alpha, the number drops the tangent of a dual 0-dim tensor; with
    # alpha, the 0-dim tensor overload keeps it, as Tensor.add does
    with fwAD.dual_level():
        with warnings.catch_warnings():
            # make_dual loads its decompositions with torch.jit.script once,
            # which warns that it is deprecated
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            t = fwAD.make_dual(
                torch.tensor(0.25, device="cuda"), torch.tensor(1.0, device="cuda")
            )
        td = TensorDict(
            a=fwAD.make_dual(
                torch.ones(3, device="cuda"), torch.full((3,), 2.0, device="cuda")
            ),
            b=torch.ones(3, device="cuda"),
        )
        out = op(td, t)
        expected = op(td, t.item()) if number else td.apply(lambda x: op(x, t))
        for key in ("a", "b"):
            primal, tangent = fwAD.unpack_dual(out[key])
            expected_primal, expected_tangent = fwAD.unpack_dual(expected[key])
            torch.testing.assert_close(primal, expected_primal, rtol=0, atol=0)
            if expected_tangent is None:
                assert tangent is None
            else:
                torch.testing.assert_close(tangent, expected_tangent, rtol=0, atol=0)


def test_flatten_keys_context_manager_inplace_separator_in_key():
    # A key that holds the separator before the block cannot be told apart
    # from a flattened nested key on exit, so the with-block keeps every key
    # that main keeps (gh-2700).
    td = TensorDict({"a.b": torch.zeros(2), "c": {"d": torch.ones(2)}}, [2])
    with td.flatten_keys(inplace=True):
        pass
    keys = set(td.keys(True, True))
    assert "a.b" in keys
    assert ("c", "d") in keys
    assert (td["a.b"] == 0).all()
    # without the separator in a key, the nested keys come back exactly
    td = TensorDict({"a": torch.zeros(2), "c": {"d": torch.ones(2)}}, [2])
    with td.flatten_keys(inplace=True):
        pass
    assert set(td.keys(True, True)) == {"a", ("c", "d")}


@pytest.mark.parametrize("device", get_available_devices())
def test_cat_tensors_shape_mismatch_preserves_entries(device):
    # Last dims differ, and dim=1 is not that dim, so cat raises.
    a = torch.zeros(2, 3, 4, device=device)
    b = torch.ones(2, 3, 5, device=device)
    td = TensorDict({"a": a, "b": b}, batch_size=[2], device=device)
    with pytest.raises(RuntimeError):
        td.cat_tensors("a", "b", out_key="c", dim=1)
    assert "a" in td
    assert "b" in td
    assert "c" not in td
    assert td.get("a") is a
    assert td.get("b") is b
    assert torch.equal(td.get("a"), torch.zeros(2, 3, 4, device=device))
    assert torch.equal(td.get("b"), torch.ones(2, 3, 5, device=device))

    a = torch.zeros(2, 3, 4, device=device)
    b = torch.ones(2, 3, 5, device=device)
    td = TensorDict({"a": a, "b": b}, batch_size=[2], device=device)
    with pytest.raises(RuntimeError):
        td.cat_tensors("a", "b", out_key="a", dim=1)
    assert "a" in td
    assert "b" in td
    assert td.get("a") is a
    assert td.get("b") is b


@pytest.mark.parametrize("device", get_available_devices())
def test_cat_tensors_set_failure_preserves_entries(device):
    a = torch.zeros(2, 3, device=device)
    b = torch.ones(2, 3, device=device)
    c = torch.full((2, 1), 4.0, device=device)
    td = TensorDict({"a": a, "b": b, "c": c}, batch_size=[2], device=device)
    with pytest.raises(RuntimeError):
        td.cat_tensors("a", "b", out_key="d", dim=0)
    assert "a" in td
    assert "b" in td
    assert "d" not in td
    assert td.get("a") is a
    assert td.get("b") is b
    assert td.get("c") is c

    a = torch.zeros(2, 3, device=device)
    b = torch.ones(2, 3, device=device)
    c = torch.full((2, 1), 4.0, device=device)
    td = TensorDict({"a": a, "b": b, "c": c}, batch_size=[2], device=device)
    with pytest.raises(RuntimeError):
        td.cat_tensors("a", "b", out_key="c", dim=0)
    assert td.get("a") is a
    assert td.get("b") is b
    assert td.get("c") is c

    a = torch.zeros(2, 3, device=device)
    b = torch.ones(2, 3, device=device)
    td = TensorDict({"a": a, "b": b}, batch_size=[2], device=device)
    with pytest.raises(RuntimeError):
        td.cat_tensors("a", "b", out_key="a", dim=0)
    assert td.get("a") is a
    assert td.get("b") is b


@pytest.mark.parametrize("device", get_available_devices())
def test_cat_tensors_out_key_is_input(device):
    a = torch.zeros(2, 3, device=device)
    b = torch.ones(2, 3, device=device)
    z = torch.full((2, 1), 2.0, device=device)
    td = TensorDict({"z": z, "a": a, "b": b}, batch_size=[2], device=device)
    td.cat_tensors("a", "b", out_key="a", dim=-1)
    assert list(td.keys()) == ["z", "a"]
    assert torch.equal(td.get("a"), torch.cat([a, b], -1))
    assert td.get("z") is z

    a = torch.zeros(2, 3, device=device)
    b = torch.ones(2, 3, device=device)
    z = torch.full((2, 1), 2.0, device=device)
    td = TensorDict({"a": a, "b": b, "z": z}, batch_size=[2], device=device)
    td.cat_tensors("a", "b", out_key="a", dim=-1, keep_entries=True)
    assert list(td.keys()) == ["a", "b", "z"]
    assert torch.equal(td.get("a"), torch.cat([a, b], -1))
    assert td.get("b") is b
    assert td.get("z") is z


@pytest.mark.parametrize("device", get_available_devices())
def test_stack_tensors_shape_mismatch_preserves_entries(device):
    a = torch.zeros(2, 3, device=device)
    b = torch.ones(2, 4, device=device)
    td = TensorDict({"a": a, "b": b}, batch_size=[2], device=device)
    with pytest.raises(RuntimeError):
        td.stack_tensors("a", "b", out_key="c")
    assert "a" in td
    assert "b" in td
    assert "c" not in td
    assert td.get("a") is a
    assert td.get("b") is b
    assert torch.equal(td.get("a"), torch.zeros(2, 3, device=device))
    assert torch.equal(td.get("b"), torch.ones(2, 4, device=device))

    a = torch.zeros(2, 3, device=device)
    b = torch.ones(2, 4, device=device)
    td = TensorDict({"a": a, "b": b}, batch_size=[2], device=device)
    with pytest.raises(RuntimeError):
        td.stack_tensors("a", "b", out_key="a")
    assert "a" in td
    assert "b" in td
    assert td.get("a") is a
    assert td.get("b") is b


@pytest.mark.parametrize("device", get_available_devices())
def test_stack_tensors_set_failure_preserves_entries(device):
    a = torch.zeros(2, 3, device=device)
    b = torch.ones(2, 3, device=device)
    td = TensorDict({"a": a, "b": b}, batch_size=[2, 3], device=device)
    with pytest.raises(RuntimeError):
        td.stack_tensors("a", "b", out_key="c", dim=0)
    assert "a" in td
    assert "b" in td
    assert "c" not in td
    assert td.get("a") is a
    assert td.get("b") is b

    a = torch.zeros(2, 3, device=device)
    b = torch.ones(2, 3, device=device)
    c = torch.full((2, 3), 4.0, device=device)
    td = TensorDict({"a": a, "b": b, "c": c}, batch_size=[2, 3], device=device)
    with pytest.raises(RuntimeError):
        td.stack_tensors("a", "b", out_key="c", dim=0)
    assert td.get("a") is a
    assert td.get("b") is b
    assert td.get("c") is c


@pytest.mark.parametrize("device", get_available_devices())
def test_stack_tensors_out_key_is_input(device):
    a = torch.zeros(2, 3, device=device)
    b = torch.ones(2, 3, device=device)
    z = torch.full((2, 1), 2.0, device=device)
    td = TensorDict({"z": z, "a": a, "b": b}, batch_size=[2], device=device)
    td.stack_tensors("a", "b", out_key="b", dim=0)
    assert list(td.keys()) == ["z", "b"]
    assert torch.equal(td.get("b"), torch.stack([a, b], 0))
    assert td.get("z") is z

    a = torch.zeros(2, 3, device=device)
    b = torch.ones(2, 3, device=device)
    z = torch.full((2, 1), 2.0, device=device)
    td = TensorDict({"a": a, "b": b, "z": z}, batch_size=[2], device=device)
    td.stack_tensors("a", "b", out_key="b", dim=0, keep_entries=True)
    assert list(td.keys()) == ["a", "b", "z"]
    assert torch.equal(td.get("b"), torch.stack([a, b], 0))
    assert td.get("a") is a
    assert td.get("z") is z


@pytest.mark.parametrize("method", ["cat_tensors", "stack_tensors"])
@pytest.mark.parametrize("keep_entries", [False, True])
def test_cat_stack_tensors_missing_key_raises_keyerror(method, keep_entries):
    a = torch.zeros(2, 3)
    td = TensorDict({"a": a, "n": {"a": torch.ones(2, 3)}}, batch_size=[2])
    with pytest.raises(KeyError, match='key "missing" not found'):
        getattr(td, method)("a", "missing", out_key="c", keep_entries=keep_entries)
    with pytest.raises(KeyError, match='key "missing" not found'):
        getattr(td, method)(
            "a", ("n", "missing"), out_key="c", keep_entries=keep_entries
        )
    assert td.get("a") is a
    assert set(td.keys(True, True)) == {"a", ("n", "a")}

    # "x" is only in the first tensordict of the stack.
    td = LazyStackedTensorDict(
        TensorDict({"a": torch.zeros(3), "x": torch.ones(1)}, []),
        TensorDict({"a": torch.ones(3)}, []),
        stack_dim=0,
    )
    with pytest.raises(KeyError, match='key "x" not found'):
        getattr(td, method)("a", "x", out_key="c", keep_entries=keep_entries)
    assert set(td.keys()) == {"a"}


class TestEmptyTensorMemmapRoundtrip:
    @pytest.mark.parametrize("archive", [False, True])
    def test_empty_tensor_leaves(self, tmp_path, archive):
        td = TensorDict(
            {
                "empty": torch.empty(0, dtype=torch.int64),
                "nested": {
                    "matrix": torch.empty(0, 3, dtype=torch.float64),
                    "robust/key": torch.empty(2, 0, dtype=torch.int16),
                },
            },
            batch_size=[],
        )
        prefix = tmp_path / "data.tdz" if archive else tmp_path / "data"

        td.memmap(prefix, robust_key=True)

        assert not list(tmp_path.rglob("*.memmap"))
        loaded = TensorDict.load_memmap(prefix, robust_key=True)
        assert loaded["empty"].shape == (0,)
        assert loaded["empty"].dtype is torch.int64
        assert loaded["nested", "matrix"].shape == (0, 3)
        assert loaded["nested", "matrix"].dtype is torch.float64
        assert loaded["nested", "robust/key"].shape == (2, 0)
        assert loaded["nested", "robust/key"].dtype is torch.int16

    def test_missing_nonempty_leaf_is_not_reconstructed(self, tmp_path):
        td = TensorDict(
            {"empty": torch.empty(0), "missing": torch.ones(1)}, batch_size=[]
        )
        td.memmap_(tmp_path)
        (tmp_path / "missing.memmap").unlink()

        loaded = TensorDict.load_memmap(tmp_path)

        assert "empty" in loaded
        assert "missing" not in loaded


class TestSubTensorDictStack:
    @pytest.mark.parametrize("idx", [0, 1, -1, slice(1, 3), (slice(None), 1)])
    def test_stack_out_sub_td(self, idx):
        # torch.stack(..., out=sub) writes into the source at the sub's index,
        # also when the index does not start at offset 0 and for nested entries.
        td = TensorDict(
            a=torch.zeros(3, 2, 4),
            nested=TensorDict(b=torch.zeros(3, 2, 5), batch_size=[3, 2]),
            batch_size=[3, 2],
        )
        sub = td._get_sub_tensordict(idx)
        expected = td.clone()
        items = [t.to_tensordict() + i + 1 for i, t in enumerate(sub.unbind(0))]
        expected[idx] = torch.stack(items, 0)
        ptrs = {v.data_ptr() for v in td.values(True, True)}
        assert torch.stack(items, 0, out=sub) is sub
        assert {v.data_ptr() for v in td.values(True, True)} == ptrs
        assert (td == expected).all()

    def test_stack_out_sub_td_missing_key(self):
        # A key of the destination that no stacked tensordict has is kept,
        # as with a dense destination.
        td = TensorDict(a=torch.zeros(3, 2), c=torch.zeros(3, 2), batch_size=[3, 2])
        sub = td._get_sub_tensordict(1)
        torch.stack([TensorDict(a=torch.ones(()))] * 2, 0, out=sub)
        assert (td["a"][1] == 1).all()
        assert (td["a"][0] == 0).all() and (td["a"][2] == 0).all()
        assert (td["c"] == 0).all()

    @pytest.mark.parametrize(
        "shape,device", [((3, 0), "cpu"), ((3, 2), "meta")], ids=["empty", "meta"]
    )
    def test_stack_out_sub_td_copy_index_no_data(self, shape, device):
        # Leaves without data have nothing to write: an index that copies is accepted.
        td = TensorDict(a=torch.zeros(shape, device=device), batch_size=[3])
        sub = td._get_sub_tensordict(torch.tensor([0, 2]))
        item = TensorDict(a=torch.zeros(shape[1:], device=device), batch_size=[])
        assert torch.stack([item] * 2, 0, out=sub) is sub

    def test_stack_out_sub_td_nested_mismatch(self):
        td = TensorDict(
            nested=TensorDict(b=torch.zeros(3, 2), batch_size=[3, 2]), batch_size=[3, 2]
        )
        sub = td._get_sub_tensordict(0)
        with pytest.raises(RuntimeError, match="non-tensordict values"):
            torch.stack([TensorDict(nested=torch.ones(()))] * 2, 0, out=sub)

    def test_stack_out_sub_td_nested_non_tensor(self):
        td = TensorDict(
            nested=TensorDict(b=torch.zeros(3, 2), batch_size=[3, 2]), batch_size=[3, 2]
        )
        sub = td._get_sub_tensordict(1)
        with pytest.raises(RuntimeError, match="non-tensordict values"):
            torch.stack([TensorDict(nested=NonTensorData("x"))] * 2, 0, out=sub)

    @pytest.mark.parametrize("idx", [0, 1])
    def test_stack_out_sub_td_shape_mismatch(self, idx):
        # torch.stack must not resize the indexed view and write over other rows.
        td = TensorDict(a=torch.zeros(3, 2, 1), batch_size=[3, 2])
        sub = td._get_sub_tensordict(idx)
        with pytest.raises(RuntimeError, match="Cannot stack values of shape"):
            torch.stack([TensorDict(a=torch.ones(3))] * 2, 0, out=sub)
        assert td["a"].shape == (3, 2, 1)
        assert (td["a"] == 0).all()

    def test_stack_out_sub_td_negative_dim(self):
        td = TensorDict(a=torch.zeros(3, 2, 4, 5), batch_size=[3, 2, 4])
        items = [
            TensorDict(a=torch.full((2, 5), float(i)), batch_size=[2]) for i in range(4)
        ]
        torch.stack(items, -1, out=td._get_sub_tensordict(1))
        assert (td["a"][1] == torch.stack([t["a"] for t in items], -2)).all()
        assert (td["a"][0] == 0).all() and (td["a"][2] == 0).all()

    def test_stack_out_sub_td_copy_index(self):
        td = TensorDict(a=torch.zeros(3, 2), batch_size=[3, 2])
        sub = td._get_sub_tensordict(torch.tensor([0, 2]))
        with pytest.raises(IndexError, match="storages of the indexed tensors"):
            torch.stack([TensorDict(a=torch.ones(2), batch_size=[2])] * 2, 0, out=sub)


class TestSubTensorDictMemmapRoundtrip:
    @pytest.mark.parametrize(
        "idx",
        [
            (slice(None), slice(0, 2)),
            (slice(None), slice(0, 2, None)),
            (torch.tensor([0, 2]),),
            ([0, 2],),
            1,
        ],
    )
    def test_sub_td_memmap_roundtrip(self, idx, tmp_path):
        td = TensorDict(
            {"a": torch.randn(4, 3, 2, 1, 5), "b": {"c": torch.randn(4, 3, 2, 1)}},
            batch_size=[4, 3, 2, 1],
        )
        sub = td._get_sub_tensordict(idx)
        sub.memmap(tmp_path / "sub", copy_existing=True)
        loaded = TensorDict.load_memmap(tmp_path / "sub")
        assert isinstance(loaded, _SubTensorDict)
        assert loaded.batch_size == sub.batch_size
        assert (loaded["a"] == sub["a"]).all()
        assert (loaded["b", "c"] == sub["b", "c"]).all()

    def test_sub_td_memmap_device_dispatch(self, tmp_path):
        # device must be forwarded when load_memmap dispatches to another class
        td = TensorDict({"a": torch.randn(4, 3)}, batch_size=[4, 3])
        sub = td._get_sub_tensordict((slice(0, 2),))
        sub.memmap(tmp_path / "sub", copy_existing=True)
        loaded = TensorDict.load_memmap(tmp_path / "sub", device="meta")
        assert loaded.device == torch.device("meta")
        assert loaded["a"].device == torch.device("meta")

    def test_sub_td_load_memmap_(self, tmp_path):
        td = TensorDict({"a": torch.randn(4, 3)}, batch_size=[4, 3])
        sub = td._get_sub_tensordict((slice(None), slice(0, 2)))
        sub.memmap(tmp_path / "sub", copy_existing=True)
        td_dest = TensorDict({"a": torch.zeros(4, 3)}, batch_size=[4, 3])
        sub_dest = td_dest._get_sub_tensordict((slice(None), slice(0, 2)))
        sub_dest.load_memmap_(tmp_path / "sub")
        assert (td_dest["a"] == td["a"]).all()

    def test_str_to_index_legacy_format(self):
        # metadata saved before the tagged index encoding: tuples arrive as
        # untagged JSON lists
        legacy = [
            ["slice", {"start": None, "stop": None, "step": None}],
            ["slice", {"start": 0, "stop": 2, "step": None}],
        ]
        assert _str_to_index(legacy) == (slice(None), slice(0, 2))


class TestMemmapLoadMode:
    """``load_memmap(..., mode=...)`` on memmap directories."""

    @staticmethod
    def _save(path):
        td = TensorDict(
            {
                "a": torch.zeros(3, 4),
                "nested": {"b": torch.zeros(3)},
                "stack": lazy_stack([TensorDict(c=torch.zeros(2)) for _ in range(3)]),
            },
            batch_size=[3],
        )
        td.memmap(path)
        return td

    @pytest.mark.parametrize("mode", [None, "r", "r+"])
    def test_in_place_writes(self, tmp_path, mode):
        self._save(tmp_path)
        loaded = TensorDict.load_memmap(tmp_path, mode=mode)
        loaded.add_(1)
        # a pickled copy maps the files the same way
        pickle.loads(pickle.dumps(loaded)).add_(1)
        # "r" keeps in-place writes in memory, the other modes write them
        # to the files
        in_memory, on_disk = (1, 0) if mode == "r" else (2, 2)
        assert (loaded == in_memory).all()
        assert (TensorDict.load_memmap(tmp_path) == on_disk).all()

    def test_mode_r_maps_copy_on_write(self, tmp_path, monkeypatch):
        # Shared writable mappings make page faults take write locks on some
        # network file systems (e.g. Lustre), which stalls concurrent readers.
        td = self._save(tmp_path / "td")
        td._get_sub_tensordict((slice(0, 2),)).memmap(tmp_path / "sub")
        nt = torch.nested.nested_tensor([torch.zeros(2), torch.zeros(3)])
        TensorDict({"nt": nt}, batch_size=[]).memmap(tmp_path / "nt")
        shared = []
        from_file = torch.from_file

        def spy(*args, **kwargs):
            shared.append(kwargs["shared"])
            return from_file(*args, **kwargs)

        monkeypatch.setattr(torch, "from_file", spy)
        for name in ("td", "sub", "nt"):
            loaded = TensorDict.load_memmap(tmp_path / name, mode="r")
            pickle.loads(pickle.dumps(loaded))
        assert shared and not any(shared)

    @pytest.mark.skipif(
        _IS_WINDOWS or os.getuid() == 0, reason="root can write to read-only files"
    )
    def test_mode_rplus_requires_writable_files(self, tmp_path):
        self._save(tmp_path)
        (tmp_path / "a.memmap").chmod(stat.S_IREAD)
        with pytest.raises(PermissionError, match="not writable"):
            TensorDict.load_memmap(tmp_path, mode="r+")

    def test_mode_r_memmap_(self, tmp_path):
        self._save(tmp_path)
        loaded = TensorDict.load_memmap(tmp_path, mode="r")
        loaded.add_(1)
        # the loaded tensordict is not bound to the directory
        loaded.memmap_()
        assert (loaded == 1).all()
        assert (TensorDict.load_memmap(tmp_path) == 0).all()

    def test_mode_r_save(self, tmp_path):
        self._save(tmp_path)
        loaded = TensorDict.load_memmap(tmp_path, mode="r")
        loaded.add_(1)
        # saving to the directory writes the values held in memory
        saved = loaded.save(tmp_path)
        assert (saved == 1).all()
        assert (TensorDict.load_memmap(tmp_path) == 1).all()

    def test_load_memmap_(self, tmp_path):
        td = self._save(tmp_path)
        dest = td.clone()
        dest.load_memmap_(tmp_path, mode="r")
        dest.add_(1)
        # the free function forwards the mode too
        load_memmap(tmp_path, mode="r").add_(1)
        assert (TensorDict.load_memmap(tmp_path) == 0).all()

    @pytest.mark.parametrize("mode", [None, "r+"])
    def test_load_memmap_into_memmap(self, tmp_path, mode):
        dest = self._save(tmp_path / "src").memmap(tmp_path / "dest")
        dest.load_memmap_(tmp_path / "src", mode=mode)
        assert dest.is_memmap() and dest.saved_path == tmp_path / "src"
        dest.add_(1)
        assert (TensorDict.load_memmap(tmp_path / "src") == 1).all()

    def test_mode_r_load_memmap_into_memmap(self, tmp_path):
        dest = self._save(tmp_path / "src").memmap(tmp_path / "dest")
        dest.load_memmap_(tmp_path / "src", mode="r")
        dest.add_(1)
        # the leaves keep their copy-on-write mapping of the files
        assert dest["a"].filename == str(tmp_path / "src" / "a.memmap")
        assert (TensorDict.load_memmap(tmp_path / "src") == 0).all()
        # like a load_memmap() result
        assert not dest.is_memmap() and not dest.is_locked

    def test_mode_rplus_load_memmap_archive_into_memmap(self, tmp_path):
        dest = self._save(tmp_path / "td.tdz").memmap(tmp_path / "dest")
        dest.load_memmap_(tmp_path / "td.tdz", mode="r+")
        dest.add_(1)
        # in-place writes reach the archive
        assert (TensorDict.load_memmap(tmp_path / "td.tdz") == 1).all()

    def test_load_memmap_into_memmap_lazy_stack(self, tmp_path):
        td = lazy_stack([TensorDict(c=torch.zeros(2)) for _ in range(3)])
        td.memmap(tmp_path / "src")
        dest = td.memmap(tmp_path / "dest")
        dest.load_memmap_(tmp_path / "src")
        dest.add_(1)
        # in-place writes reach the files
        assert (TensorDict.load_memmap(tmp_path / "src") == 1).all()


class TestLoadMemmapOut:
    """``load_memmap(..., out=...)`` and ``load_memmap_``."""

    @staticmethod
    def _save(path):
        td = TensorDict(
            {"a": torch.ones(3), "nested": {"b": torch.ones(3)}}, batch_size=[3]
        )
        td.memmap(path)
        return td

    def test_nested_device(self, tmp_path):
        self._save(tmp_path)
        # out has no device, so only the requested device can place the
        # leaves of its existing nested containers
        out = TensorDict(
            {
                "a": torch.zeros(3, device="meta"),
                "nested": {"b": torch.zeros(3, device="meta")},
            },
            batch_size=[3],
        )
        loaded = TensorDict.load_memmap(tmp_path, device="meta", out=out)
        assert loaded is out
        assert loaded["a"].device == torch.device("meta")
        assert loaded["nested", "b"].device == torch.device("meta")

    def test_device_from_out(self, tmp_path):
        self._save(tmp_path)
        out = TensorDict({"nested": {}}, batch_size=[3], device="meta")
        loaded = TensorDict.load_memmap(tmp_path, out=out)
        assert loaded["a"].device == torch.device("meta")
        assert loaded["nested", "b"].device == torch.device("meta")

    def test_device_mismatch(self, tmp_path):
        self._save(tmp_path)
        out = TensorDict(batch_size=[3], device="meta")
        with pytest.raises(ValueError, match="Cannot load a tensordict on device"):
            TensorDict.load_memmap(tmp_path, device="cpu", out=out)

    def test_stale_keys(self, tmp_path):
        self._save(tmp_path)
        out = TensorDict(
            {
                "a": torch.zeros(3),
                "stale": torch.zeros(3),
                "nested": {"b": torch.zeros(3), "stale": torch.zeros(3)},
                "stale_nested": {"c": torch.zeros(3)},
            },
            batch_size=[3],
        )
        nested = out["nested"]
        loaded = TensorDict.load_memmap(tmp_path, out=out)
        assert loaded is out
        assert loaded["nested"] is nested
        assert set(loaded.keys(True, True)) == {"a", ("nested", "b")}
        assert (loaded == 1).all()

    def test_load_memmap_stale_keys_into_memmap(self, tmp_path):
        self._save(tmp_path / "src")
        dest = TensorDict(
            {
                "a": torch.zeros(3),
                "stale": torch.zeros(3),
                "nested": {"b": torch.zeros(3), "stale": torch.zeros(3)},
            },
            batch_size=[3],
        ).memmap(tmp_path / "dest")
        dest.load_memmap_(tmp_path / "src")
        assert set(dest.keys(True, True)) == {"a", ("nested", "b")}
        assert (dest == 1).all()
        assert dest.is_memmap() and dest.saved_path == tmp_path / "src"
        assert dest["nested"].is_memmap()
        assert dest["nested"].saved_path == tmp_path / "src" / "nested"
        assert dest["nested", "b"].filename == str(
            tmp_path / "src" / "nested" / "b.memmap"
        )

    def test_lazy_stack_length_mismatch(self, tmp_path):
        lazy_stack([TensorDict(c=torch.zeros(2)) for _ in range(3)]).memmap(tmp_path)
        out = lazy_stack([TensorDict(c=torch.zeros(2)) for _ in range(2)])
        with pytest.raises(ValueError, match="Cannot load 3 stacked tensordicts"):
            TensorDict.load_memmap(tmp_path, out=out)


class TestBackward:
    def test_scalar_implicit_gradient(self):
        x = torch.randn(3, requires_grad=True)
        loss_td = TensorDict(actor_loss=x.sum(), critic_loss=x.pow(2).sum())
        loss_td.backward()
        assert torch.allclose(x.grad, 1 + 2 * x.detach())

    def test_scalar_implicit_matches_sum_reduce(self):
        x = torch.randn(3, requires_grad=True)
        y = x.detach().clone().requires_grad_()

        def make_loss(t):
            return TensorDict(actor_loss=t.sum(), critic_loss=t.pow(2).sum())

        make_loss(x).backward()
        make_loss(y).sum(reduce=True).backward()
        assert torch.allclose(x.grad, y.grad)

    def test_sum_semantics_unchanged(self):
        # plain sum still reduces each leaf independently and returns a TensorDict
        td = TensorDict(a=torch.ones(3), b=torch.ones(2, 2), batch_size=[])
        td_sum = td.sum()
        assert isinstance(td_sum, TensorDict)
        assert td_sum["a"] == 3
        assert td_sum["b"] == 4
        assert isinstance(td.sum(reduce=True), torch.Tensor)

    def test_implicit_nonscalar_raises(self):
        x = torch.randn(3, requires_grad=True)
        out = TensorDict(a=x * 2)
        with pytest.raises(
            RuntimeError, match="grad can be implicitly created only for scalar outputs"
        ):
            out.backward()

    def test_explicit_gradient(self):
        x = torch.randn(3, requires_grad=True)
        out = TensorDict(a=x * 2, b=(x + 1).sum())
        out.backward(torch.ones_like(out))
        assert torch.allclose(x.grad, torch.full((3,), 3.0))

    def test_weighted_gradient(self):
        x = torch.randn(3, requires_grad=True)
        loss_td = TensorDict(actor_loss=x.sum(), critic_loss=x.pow(2).sum())
        weights = TensorDict(
            actor_loss=torch.tensor(0.5), critic_loss=torch.tensor(1.0)
        )
        loss_td.backward(weights)
        assert torch.allclose(x.grad, 0.5 + 2 * x.detach())

    def test_weighted_matches_weighted_sum(self):
        x = torch.randn(3, requires_grad=True)
        y = x.detach().clone().requires_grad_()

        def make_loss(t):
            return TensorDict(actor_loss=t.sum(), critic_loss=t.pow(2).sum())

        weights = TensorDict(
            actor_loss=torch.tensor(0.5), critic_loss=torch.tensor(1.0)
        )
        make_loss(x).backward(weights)
        (make_loss(y) * weights).sum(reduce=True).backward()
        assert torch.allclose(x.grad, y.grad)

    def test_nested_keys(self):
        x = torch.randn(3, requires_grad=True)
        out = TensorDict({"nested": {"a": x * 2}, "b": x.sum()}, batch_size=[])
        gradient = TensorDict(
            {"nested": {"a": torch.ones(3)}, "b": torch.tensor(1.0)}, batch_size=[]
        )
        out.backward(gradient)
        assert torch.allclose(x.grad, torch.full((3,), 3.0))

    def test_heterogeneous_shapes_and_dtypes(self):
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        out = TensorDict(a=x * 2, b=y * 3)
        gradient = TensorDict(a=torch.ones(3), b=torch.ones(2, 2, dtype=torch.float64))
        out.backward(gradient)
        assert torch.allclose(x.grad, torch.full((3,), 2.0))
        assert torch.allclose(y.grad, torch.full((2, 2), 3.0, dtype=torch.float64))

    def test_mixed_differentiable_leaves(self):
        x = torch.randn(3, requires_grad=True)
        out = TensorDict(a=(x * 2).sum(), b=torch.ones(4), c="a non-tensor entry")
        # non-differentiable leaves are ignored, scalar check only applies to `a`
        out.backward()
        assert torch.allclose(x.grad, torch.full((3,), 2.0))

    def test_mixed_differentiable_leaves_gradient(self):
        x = torch.randn(3, requires_grad=True)
        out = TensorDict(a=x * 2, b=torch.ones(4))
        # no gradient needed for the non-differentiable leaf
        out.backward(TensorDict(a=torch.ones(3)))
        assert torch.allclose(x.grad, torch.full((3,), 2.0))

    def test_missing_gradient_key(self):
        x = torch.randn(3, requires_grad=True)
        out = TensorDict(a=x * 2)
        with pytest.raises(KeyError, match="Missing gradient entry for key 'a'"):
            out.backward(TensorDict(b=torch.ones(3)))

    def test_gradient_shape_mismatch(self):
        x = torch.randn(3, requires_grad=True)
        out = TensorDict(a=x * 2)
        with pytest.raises(RuntimeError, match="Mismatch in shape"):
            out.backward(TensorDict(a=torch.ones(4)))

    def test_gradient_wrong_type(self):
        x = torch.randn(3, requires_grad=True)
        out = TensorDict(a=x * 2)
        with pytest.raises(TypeError, match="gradient must be a TensorDictBase"):
            out.backward(torch.ones(3))

    def test_no_differentiable_leaves(self):
        td = TensorDict(a=torch.randn(3), b=torch.ones(2))
        with pytest.raises(RuntimeError, match="no leaf tensor requiring gradients"):
            td.backward()

    def test_retain_graph(self):
        x = torch.randn(3, requires_grad=True)
        out = TensorDict(loss=(x**2).sum())
        out.backward(retain_graph=True)
        out.backward()
        assert torch.allclose(x.grad, 4 * x.detach())

    def test_create_graph(self):
        x = torch.randn(3, requires_grad=True)
        with warnings.catch_warnings():
            # torch warns about the reference cycle created by create_graph=True
            warnings.simplefilter("ignore", UserWarning)
            TensorDict(loss=(x**2).sum()).backward(create_graph=True)
        assert x.grad.requires_grad
        x.grad = None

    def test_inputs(self):
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        out = TensorDict(loss=(x * y).sum())
        out.backward(inputs=[x])
        assert torch.allclose(x.grad, y.detach())
        assert y.grad is None

    def test_inputs_tensordict(self):
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        out = TensorDict(loss=(x * y).sum())
        out.backward(inputs=TensorDict(x=x))
        assert torch.allclose(x.grad, y.detach())
        assert y.grad is None

    def test_parity_with_torch(self):
        x = torch.randn(3, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        gradient = torch.randn(3)
        TensorDict(a=x * 2).backward(TensorDict(a=gradient))
        (y * 2).backward(gradient)
        assert torch.allclose(x.grad, y.grad)


def _to_float(td, td_name, tmpdir):
    if hasattr(td, "_source"):
        td._source = td._source.float()
    elif td_name in ("td_h5",):
        td = PersistentTensorDict.from_dict(
            td.float().to_dict(), filename=tmpdir + "/file.t", auto_batch_size=True
        )
    elif td_name in ("td_params",):
        td = TensorDictParams(td.data.float())
    else:
        with td.unlock_():
            td_typed = td.apply(lambda x: x.float())
        assert isinstance(td_typed, type(td))
        td = td_typed
    return td


class TestMemmapArchive:
    """Format-level tests for single-file memmap archives (.tdz)."""

    @staticmethod
    def _nested_td():
        return TensorDict(
            {
                "a": torch.randn(3, 4),
                "b": {
                    "c": torch.ones(3, 2, dtype=torch.bool),
                    "d": torch.arange(3, dtype=torch.int16),
                },
                "e": NonTensorData("hello", batch_size=[3]),
            },
            batch_size=[3],
        )

    @staticmethod
    def _payload_offsets(archive):
        offsets = {}
        with zipfile.ZipFile(archive) as zf, open(archive, "rb") as f:
            for info in zf.infolist():
                f.seek(info.header_offset + 26)
                name_len, extra_len = struct.unpack("<2H", f.read(4))
                offsets[info.filename] = info.header_offset + 30 + name_len + extra_len
        return offsets

    def test_archive_tensor_payloads_are_aligned(self, tmp_path):
        td = self._nested_td()
        archive = tmp_path / "data.tdz"
        td.save(archive)
        offsets = self._payload_offsets(archive)
        memmap_offsets = {
            name: off for name, off in offsets.items() if name.endswith(".memmap")
        }
        assert memmap_offsets
        for name, off in memmap_offsets.items():
            assert off % 64 == 0, (name, off)

    def test_archive_zero_copy(self, tmp_path):
        # every leaf is a view into a single mapping of the archive
        td = self._nested_td()
        archive = tmp_path / "data.tdz"
        td.save(archive)
        loaded = TensorDict.load_memmap(archive)
        ptrs = {
            v.untyped_storage().data_ptr()
            for v in loaded.values(True, True)
            if isinstance(v, torch.Tensor)
        }
        assert len(ptrs) == 1
        # in-place writes are copy-on-write: the file is not modified
        before = archive.read_bytes()
        loaded["a"] += 1
        assert archive.read_bytes() == before

    @pytest.mark.parametrize("view_kind", ["conjugate", "negative"])
    def test_archive_tensor_view_bits(self, tmp_path, view_kind):
        if view_kind == "conjugate":
            tensor = torch.tensor([1 + 2j, 3 + 4j]).conj()
        else:
            tensor = torch.tensor([1.0, 2.0])._neg_view()
        expected = tensor.resolve_conj().resolve_neg()
        archive = tmp_path / f"{view_kind}.tdz"
        loaded = TensorDict({"x": tensor}, batch_size=[]).save(archive)
        torch.testing.assert_close(loaded["x"], expected)

    def test_archive_explicit_flag(self, tmp_path):
        # archive mode can target any file name when requested explicitly
        td = self._nested_td()
        archive = tmp_path / "weights.bin"
        td.save(archive, archive=True)
        assert is_memmap_archive(archive)
        loaded = TensorDict.load_memmap(archive)
        assert (loaded == td).all()

    def test_archive_foreign_zip(self, tmp_path):
        # `zip -0 -r` over a memmap dir (no alignment) is loadable: unaligned
        # leaves fall back to a copying read
        td = self._nested_td()
        td.memmap(tmp_path / "plain")
        foreign = tmp_path / "foreign.zip"
        with zipfile.ZipFile(foreign, "w") as zf:
            for path in sorted((tmp_path / "plain").rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(tmp_path / "plain").as_posix())
        loaded = TensorDict.load_memmap(foreign)
        assert (loaded == td).all()

    def test_pack_unpack_roundtrip(self, tmp_path):
        td = self._nested_td()
        td.memmap(tmp_path / "plain")
        archive = pack_memmap(tmp_path / "plain", tmp_path / "packed.tdz")
        assert is_memmap_archive(archive)
        assert (TensorDict.load_memmap(archive) == td).all()
        unpack_memmap(archive, tmp_path / "unpacked")
        assert (TensorDict.load_memmap(tmp_path / "unpacked") == td).all()

    @pytest.mark.parametrize("method", ["pack", "save"])
    @pytest.mark.parametrize("location", ["root", "nested", "symlink"])
    @pytest.mark.parametrize("existing", [False, True])
    def test_archive_inside_source(
        self, tmp_path, monkeypatch, method, location, existing
    ):
        prefix = tmp_path / "plain"
        td = self._nested_td().memmap(prefix)
        parent = prefix
        if location == "nested":
            parent = prefix / "archives"
            parent.mkdir()
        elif location == "symlink":
            parent = tmp_path / "alias"
            parent.symlink_to(prefix, target_is_directory=True)
        archive = parent / "packed.tdz"
        if existing:
            archive.write_bytes(b"existing contents")

        def fail_open(*args, **kwargs):
            # Reject the path before opening the archive.
            # Reading the output while writing it can make it grow without limit.
            pytest.fail("opened an archive inside its source directory")

        monkeypatch.setattr(zipfile, "ZipFile", fail_open)
        with pytest.raises(ValueError, match="outside the source directory"):
            if method == "pack":
                pack_memmap(prefix, archive)
            else:
                td.save(archive)
        if existing:
            assert archive.read_bytes() == b"existing contents"
        else:
            assert not archive.exists()
        assert (TensorDict.load_memmap(prefix) == td).all()

    @pytest.mark.parametrize("link", ["symlink", "hardlink"])
    def test_archive_source_file_alias(self, tmp_path, monkeypatch, link):
        prefix = tmp_path / "plain"
        expected = self._nested_td()
        td = expected.memmap(prefix)
        archive = tmp_path / "packed.tdz"
        if link == "symlink":
            archive.symlink_to(prefix / "a.memmap")
        else:
            archive.hardlink_to(prefix / "a.memmap")
        before = archive.read_bytes()

        def fail_open(*args, **kwargs):
            pytest.fail("opened an archive aliasing a source file")

        monkeypatch.setattr(zipfile, "ZipFile", fail_open)
        with pytest.raises(ValueError, match="outside the source directory"):
            pack_memmap(prefix, archive)
        assert archive.read_bytes() == before
        assert (td == expected).all()

    def test_archive_output_reachable_through_symlink(self, tmp_path, monkeypatch):
        prefix = tmp_path / "plain"
        td = self._nested_td().memmap(prefix)
        output = tmp_path / "output"
        output.mkdir()
        (prefix / "linked").symlink_to(output, target_is_directory=True)
        archive = output / "packed.tdz"
        file_chunks = tensordict_archive._file_chunks

        def checked_chunks(path):
            # Stop the writer before it reads its own output.
            assert not path.resolve().is_relative_to(output.resolve())
            yield from file_chunks(path)

        monkeypatch.setattr(tensordict_archive, "_file_chunks", checked_chunks)
        pack_memmap(prefix, archive)
        assert (TensorDict.load_memmap(archive) == td).all()
        with zipfile.ZipFile(archive) as zf:
            assert all(not name.endswith(".tdz") for name in zf.namelist())

    @pytest.mark.parametrize("num_threads", [1, 2, 4])
    @pytest.mark.parametrize("compression", ["deflate", "bzip2", "lzma"])
    def test_archive_compression(self, tmp_path, compression, num_threads):
        td = TensorDict(
            {"a": torch.zeros(1000, dtype=torch.uint8), "b": {"c": torch.zeros(100)}},
            batch_size=[],
        )
        stored = tmp_path / "stored.tdz"
        compressed = tmp_path / "compressed.tdz"
        td.save(stored)
        td.save(compressed, compression=compression)
        assert compressed.stat().st_size < stored.stat().st_size
        assert (TensorDict.load_memmap(compressed, num_threads=num_threads) == td).all()

    @pytest.mark.parametrize("num_threads", [1, 2, 4])
    @pytest.mark.parametrize("subpath", [None, "b"])
    def test_archive_compressed_parallel_load(self, tmp_path, subpath, num_threads):
        td = TensorDict(
            {
                "a": torch.randn(100, 10),
                "b": {f"t{i}": torch.randn(100) for i in range(8)},
            },
            batch_size=[100],
        )
        archive = tmp_path / "comp.tdz"
        td.save(archive, compression="deflate")
        loaded = TensorDict.load_memmap(
            archive, subpath=subpath, num_threads=num_threads
        )
        expected = td if subpath is None else td.get(subpath)
        assert (loaded == expected).all()
        # num_threads is a no-op on uncompressed archives and directories
        td.save(tmp_path / "plain.tdz")
        assert (
            TensorDict.load_memmap(tmp_path / "plain.tdz", num_threads=num_threads)
            == td
        ).all()
        td.memmap(tmp_path / "plaindir")
        assert (
            TensorDict.load_memmap(tmp_path / "plaindir", num_threads=num_threads) == td
        ).all()

    @pytest.mark.parametrize("num_threads", [1, 2, 4])
    def test_archive_local_filename_mismatch(self, tmp_path, num_threads):
        archive = tmp_path / "mismatch.tdz"
        td = TensorDict({"a": torch.arange(4), "b": torch.ones(4)})
        td.save(archive, compression="deflate")
        with zipfile.ZipFile(archive) as zf:
            offset = zf.getinfo("a.memmap").header_offset
        raw = bytearray(archive.read_bytes())
        raw[offset + 30] = ord("z")
        archive.write_bytes(raw)
        with zipfile.ZipFile(archive) as zf:
            with pytest.raises(zipfile.BadZipFile, match="File name.*differ"):
                zf.read("a.memmap")
        with pytest.raises(zipfile.BadZipFile, match="File name.*differ"):
            TensorDict.load_memmap(archive, num_threads=num_threads)

    @pytest.mark.parametrize("num_threads", [1, 2, 4])
    @pytest.mark.parametrize(
        "layout", ["zip64", "descriptor", "prefixed", "utf8", "cp437"]
    )
    def test_archive_local_filename_layouts(self, tmp_path, num_threads, layout):
        class UnseekableBuffer(io.BytesIO):
            def seek(self, *args):
                raise OSError("not seekable")

        buffer = UnseekableBuffer() if layout == "descriptor" else io.BytesIO()
        name = "é.memmap" if layout == "utf8" else "a.memmap"
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for entry_name in (name, "b.memmap"):
                info = zipfile.ZipInfo(entry_name)
                info.compress_type = zipfile.ZIP_DEFLATED
                with zf.open(info, "w", force_zip64=layout == "zip64") as entry:
                    entry.write(b"1234")
        raw = buffer.getvalue()
        if layout == "prefixed":
            raw = b"archive prefix" + raw
        elif layout == "cp437":
            raw = raw.replace(b"a.memmap", b"\x82.memmap")
            name = "é.memmap"
        archive = tmp_path / "layout.tdz"
        archive.write_bytes(raw)
        with zipfile.ZipFile(archive) as zf:
            assert zf.read(name) == b"1234"
            if layout == "descriptor":
                assert zf.getinfo(name).flag_bits & 0x8
        reader = _ArchiveReader(archive)
        reader.schedule_compressed_prefetch("", num_threads)
        for entry_name in (name, "b.memmap"):
            assert reader.flat_view(entry_name, torch.uint8).tolist() == list(b"1234")

    @pytest.mark.parametrize("num_threads", [1, 2, 4])
    @pytest.mark.parametrize(
        "corruption",
        [
            "zero_size",
            "smaller_size",
            "larger_size",
            "truncated",
            "missing",
            "empty_missing",
            "invalid",
            "trailing",
            "crc",
        ],
    )
    def test_archive_deflate_payload_validation(
        self, tmp_path, num_threads, corruption
    ):
        archive = tmp_path / "invalid.tdz"
        payload = b"" if corruption == "empty_missing" else b"a" * 32
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("a.memmap", payload)
            zf.writestr("b.memmap", b"b" * 32)
        with zipfile.ZipFile(archive) as zf:
            info = zf.getinfo("a.memmap")
            central = zf.start_dir
        raw = bytearray(archive.read_bytes())
        if corruption in ("zero_size", "smaller_size", "larger_size"):
            size = {"zero_size": 0, "smaller_size": 31, "larger_size": 128}[corruption]
            struct.pack_into("<I", raw, central + 24, size)
            # Keep the CRC consistent with the declared prefix, not the stream.
            struct.pack_into("<I", raw, central + 16, zlib.crc32(payload[:size]))
        elif corruption in ("truncated", "missing", "empty_missing", "trailing"):
            size = {
                "truncated": info.compress_size - 1,
                "missing": 0,
                "empty_missing": 0,
                "trailing": info.compress_size + 1,
            }[corruption]
            struct.pack_into("<I", raw, central + 20, size)
        elif corruption == "invalid":
            raw[self._payload_offsets(archive)["a.memmap"]] = 0x07
        else:
            struct.pack_into("<I", raw, central + 16, info.CRC ^ 1)
        archive.write_bytes(raw)

        reader = _ArchiveReader(archive)
        reader.schedule_compressed_prefetch("", num_threads)
        with pytest.raises(zipfile.BadZipFile, match="a.memmap"):
            reader.flat_view("a.memmap", torch.uint8)

    @pytest.mark.parametrize("num_threads", [1, 2, 4])
    @pytest.mark.parametrize("size", [0, 32])
    def test_archive_deflate_bounded(self, tmp_path, monkeypatch, num_threads, size):
        archive = tmp_path / "bounded.tdz"
        td = TensorDict(
            {key: torch.zeros(size, dtype=torch.uint8) for key in ("a", "b")}
        )
        td.save(archive, compression="deflate")
        # TensorDict omits empty payload files; add valid empty streams to
        # exercise the reader's zero-size limit independently of that shortcut.
        if not size:
            with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_DEFLATED) as zf:
                for key in ("a", "b"):
                    zf.writestr(f"{key}.memmap", b"")
        decompressobj = zlib.decompressobj
        limits = []

        class CheckedDecompressor:
            def __init__(self, *args, **kwargs):
                self.inner = decompressobj(*args, **kwargs)

            def decompress(self, data, max_length=0):
                assert 0 < max_length <= size + 1
                limits.append(max_length)
                return self.inner.decompress(data, max_length)

            def __getattr__(self, name):
                return getattr(self.inner, name)

        def fail_unbounded(*args, **kwargs):
            pytest.fail("unbounded decompression")

        monkeypatch.setattr(zlib, "decompressobj", CheckedDecompressor)
        monkeypatch.setattr(zlib, "decompress", fail_unbounded)
        loaded = TensorDict.load_memmap(archive, num_threads=num_threads)
        assert (loaded == td).all()
        # TensorDict can skip reading empty leaves; exercise their streams too.
        limits.clear()
        reader = _ArchiveReader(archive)
        reader.schedule_compressed_prefetch("", num_threads)
        for key in ("a", "b"):
            torch.testing.assert_close(
                reader.flat_view(f"{key}.memmap", torch.uint8), td[key]
            )
        assert len(limits) == 2
        # Loaded compressed leaves remain writable without changing the archive.
        before = archive.read_bytes()
        loaded["a"].fill_(1)
        assert archive.read_bytes() == before

    @pytest.mark.parametrize("num_threads", [1, 2, 4])
    def test_archive_compressed_metadata_only_load(
        self, tmp_path, monkeypatch, num_threads
    ):
        td = TensorDict({"a": torch.randn(10), "b": torch.randn(10)}, batch_size=[])
        archive = tmp_path / "compressed.tdz"
        td.save(archive, compression="deflate")

        def fail_prefetch(*args, **kwargs):
            pytest.fail("compressed payloads were prefetched")

        monkeypatch.setattr(_ArchiveReader, "prefetch_compressed", fail_prefetch)
        monkeypatch.setattr(_ArchiveReader, "_inflate", fail_prefetch)
        meta = TensorDict.load_memmap(archive, device="meta", num_threads=num_threads)
        assert meta["a"].device.type == "meta"
        with FakeTensorMode():
            fake = TensorDict.load_memmap(archive, num_threads=num_threads)
        assert isinstance(fake["a"], FakeTensor)

    def test_archive_compression_requires_archive(self, tmp_path):
        td = self._nested_td()
        with pytest.raises(ValueError, match="compression is only supported"):
            td.save(tmp_path / "plain", compression="deflate")

    @pytest.mark.parametrize("archive", [True, False])
    def test_subpath(self, tmp_path, archive):
        td = lazy_stack(
            [
                TensorDict(
                    {"x": torch.full((2,), float(i)), "sub": {"y": torch.ones(2) * i}},
                    batch_size=[2],
                )
                for i in range(3)
            ]
        )
        prefix = tmp_path / ("saved.tdz" if archive else "saved")
        td.save(prefix)
        loaded = TensorDict.load_memmap(prefix)
        assert isinstance(loaded, LazyStackedTensorDict)
        assert (loaded == td).all()
        part = TensorDict.load_memmap(prefix, subpath="1/sub")
        assert (part["y"] == 1.0).all()
        part = TensorDict.load_memmap(prefix, subpath=("1", "sub"))
        assert (part["y"] == 1.0).all()
        # nested keys with arbitrary nesting are accepted
        part = TensorDict.load_memmap(prefix, subpath=("1", ("sub",)))
        assert (part["y"] == 1.0).all()
        with pytest.raises(ValueError, match="No tensordict found under subpath"):
            TensorDict.load_memmap(prefix, subpath="does/not/exist")
        with pytest.raises(ValueError, match="No tensordict found under subpath"):
            TensorDict.load_memmap(prefix, subpath=("does", ("not", "exist")))

    @pytest.mark.parametrize("archive", [True, False])
    def test_subpath_tensorclass(self, tmp_path, archive):
        class Part(TensorClass):
            x: torch.Tensor

        class Assembly(TensorClass):
            parts: TensorDict

        class Outer(TensorClass):
            inner: Assembly

        obj = Outer(
            inner=Assembly(parts=TensorDict(a=Part(x=torch.arange(3.0)))),
            batch_size=[],
        )
        prefix = tmp_path / ("obj.tdz" if archive else "obj")
        obj.memmap(prefix, archive=archive)
        # a nested key loads the entry that obj.get(key) returns
        for subpath in (
            ("inner", "parts", "a"),
            "inner/parts/a",
            "_tensordict/inner/_tensordict/parts/a",
        ):
            part = Part.load_memmap(prefix, subpath=subpath)
            assert isinstance(part, Part)
            assert (part.x == torch.arange(3.0)).all()
        parts = TensorDict.load_memmap(prefix, subpath=("inner", "parts"))
        assert (parts["a"].x == torch.arange(3.0)).all()
        with pytest.raises(ValueError, match="No tensordict found under subpath"):
            Part.load_memmap(prefix, subpath=("inner", "missing"))

        # a tensordict with a key named "_tensordict" is not a tensorclass
        td = TensorDict({"_tensordict": {"b": {"y": torch.zeros(2)}}})
        prefix = tmp_path / ("td.tdz" if archive else "td")
        td.memmap(prefix, archive=archive)
        loaded = TensorDict.load_memmap(prefix, subpath="_tensordict")
        assert (loaded["b", "y"] == 0).all()
        with pytest.raises(ValueError, match="No tensordict found under subpath"):
            TensorDict.load_memmap(prefix, subpath="b")

    def test_archive_meta_device(self, tmp_path):
        td = self._nested_td()
        archive = tmp_path / "data.tdz"
        td.save(archive)
        meta = TensorDict.load_memmap(archive, device="meta")
        assert meta["a"].device.type == "meta"
        assert meta["a"].shape == td["a"].shape

    def test_archive_nested_tensor(self, tmp_path):
        nt = torch.nested.nested_tensor([torch.randn(2, 3), torch.randn(4, 3)])
        td = TensorDict({"nt": nt}, batch_size=[])
        archive = tmp_path / "nt.tdz"
        td.save(archive)
        loaded = TensorDict.load_memmap(archive)
        assert (loaded["nt"][0] == nt[0]).all()
        assert (loaded["nt"][1] == nt[1]).all()

    def test_archive_load_memmap_(self, tmp_path):
        td = self._nested_td()
        archive = tmp_path / "data.tdz"
        td.save(archive)
        dest = td.clone().zero_()
        dest.load_memmap_(archive)
        assert (dest == td).all()

    def test_archive_pickle(self, tmp_path):
        td = self._nested_td()
        archive = tmp_path / "data.tdz"
        td.save(archive)
        loaded = TensorDict.load_memmap(archive)
        assert (pickle.loads(pickle.dumps(loaded)) == td).all()

    def test_archive_pickled_non_tensor_stack(self, tmp_path):
        values = [1 + 2j, 3 + 4j]
        stack = NonTensorStack(
            *(NonTensorData(value, batch_size=[]) for value in values), stack_dim=0
        )
        td = TensorDict({"values": stack}, batch_size=[2])
        archive = tmp_path / "non_tensor.tdz"
        loaded = td.save(archive)
        assert loaded.to_dict()["values"] == values
        with pytest.raises(RuntimeError, match="allow_pickle=True"):
            TensorDict.load_memmap(archive, allow_pickle=False)
        trusted = TensorDict.load_memmap(archive, allow_pickle=True)
        assert trusted.to_dict()["values"] == values

    def test_archive_existsok(self, tmp_path):
        td = self._nested_td()
        archive = tmp_path / "data.tdz"
        td.save(archive)
        with pytest.raises(RuntimeError, match="already exists"):
            td.memmap(archive, existsok=False)
        # overwriting is fine by default
        td.save(archive)

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX filename limits")
    @pytest.mark.parametrize("method", ["save", "pack_memmap"])
    @pytest.mark.parametrize("length", [246, 255])
    @pytest.mark.parametrize("existing", [False, True])
    def test_archive_long_filename(self, tmp_path, method, length, existing):
        if os.pathconf(tmp_path, "PC_NAME_MAX") < length:
            pytest.skip("filesystem filename limit is smaller than this case")
        td = self._nested_td()
        prefix = tmp_path / "source"
        mapped = td.memmap(prefix)
        archive = tmp_path / ("a" * (length - 4) + ".tdz")
        if existing:
            archive.write_bytes(b"existing contents")
        if method == "save":
            mapped.save(archive)
        else:
            pack_memmap(prefix, archive)
        assert (TensorDict.load_memmap(archive) == td).all()

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX file permissions")
    @pytest.mark.parametrize("method", ["save", "pack_memmap"])
    @pytest.mark.parametrize("mode", [None, 0o600, 0o640, 0o664])
    def test_archive_destination_permissions(self, tmp_path, method, mode):
        td = self._nested_td()
        prefix = tmp_path / "source"
        td.memmap(prefix)
        archive = tmp_path / "data.tdz"
        previous_umask = os.umask(0o022)
        try:
            if mode is not None:
                td.save(archive)
                archive.chmod(mode)
            if method == "save":
                td.save(archive)
            else:
                pack_memmap(prefix, archive)
            expected_mode = 0o644 if mode is None else mode
            assert stat.S_IMODE(archive.stat().st_mode) == expected_mode
            assert (TensorDict.load_memmap(archive) == td).all()
        finally:
            os.umask(previous_umask)

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX file permissions")
    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root can write read-only files",
    )
    @pytest.mark.parametrize("method", ["save", "pack_memmap"])
    def test_archive_read_only_destination(self, tmp_path, method):
        td = self._nested_td()
        prefix = tmp_path / "source"
        td.memmap(prefix)
        archive = tmp_path / "data.tdz"
        td.save(archive)
        archive.chmod(0o444)
        before = archive.read_bytes()
        entries_before = set(tmp_path.iterdir())
        with pytest.raises(PermissionError, match="not writable"):
            if method == "save":
                td.save(archive)
            else:
                pack_memmap(prefix, archive)
        assert archive.read_bytes() == before
        assert stat.S_IMODE(archive.stat().st_mode) == 0o444
        assert set(tmp_path.iterdir()) == entries_before

    @pytest.mark.parametrize("mode", ["r", "r+"])
    @pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
    def test_archive_overwrite_mapped_source(self, tmp_path, monkeypatch, mode, alias):
        archive = tmp_path / "data.tdz"
        td = TensorDict(x=torch.arange(16384, dtype=torch.float32), batch_size=[16384])
        td.save(archive)
        loaded = TensorDict.load_memmap(archive, mode=mode)
        loaded["x"].add_(1)
        expected = loaded.clone()
        target = archive
        if alias != "same":
            target = tmp_path / "alias.tdz"
            if alias == "symlink":
                target.symlink_to(archive)
            else:
                target.hardlink_to(archive)

        before = archive.read_bytes()
        tensor_chunks = tensordict_archive._tensor_chunks

        def checked_chunks(tensor):
            # Fail before reading a truncated mapping, which can cause SIGBUS.
            assert archive.read_bytes() == before
            yield from tensor_chunks(tensor)

        monkeypatch.setattr(tensordict_archive, "_tensor_chunks", checked_chunks)
        saved = loaded.save(target)
        assert (saved == expected).all()
        assert (loaded == expected).all()
        assert (TensorDict.load_memmap(target) == expected).all()
        with zipfile.ZipFile(target) as zf:
            assert zf.testzip() is None
        if alias == "symlink":
            # The link is kept and the file it points to is updated.
            assert target.is_symlink()
            assert (TensorDict.load_memmap(archive) == expected).all()
        elif alias == "hardlink":
            # Replacing the file detaches the other hard link.
            assert archive.read_bytes() == before

    def test_archive_failed_overwrite_preserves_target(self, tmp_path, monkeypatch):
        archive = tmp_path / "data.tdz"
        td = self._nested_td()
        td.save(archive)
        before = archive.read_bytes()

        def fail_chunks(tensor):
            raise RuntimeError("failed to read tensor")

        monkeypatch.setattr(tensordict_archive, "_tensor_chunks", fail_chunks)
        with pytest.raises(RuntimeError, match="failed to read tensor"):
            td.save(archive)
        assert archive.read_bytes() == before
        assert list(tmp_path.iterdir()) == [archive]

    def test_archive_bad_file(self, tmp_path):
        bad = tmp_path / "bad.tdz"
        bad.write_bytes(b"not a zip file")
        with pytest.raises(ValueError, match="not a memory-mapped tensordict"):
            TensorDict.load_memmap(bad)

    def test_archive_mode_rplus(self, tmp_path):
        td = self._nested_td()
        archive = tmp_path / "data.tdz"
        td.save(archive)
        loaded = TensorDict.load_memmap(archive, mode="r+")
        loaded["a"].add_(1)
        loaded["b", "d"].add_(2)
        # writes reached the file
        reloaded = TensorDict.load_memmap(archive)
        assert (reloaded["a"] == td["a"] + 1).all()
        assert (reloaded["b", "d"] == td["b", "d"] + 2).all()
        # the CRCs are stale: zip-level reads fail until refreshed
        with pytest.raises(zipfile.BadZipFile), zipfile.ZipFile(archive) as zf:
            zf.read("a.memmap")
        refresh_archive_checksums(archive)
        with zipfile.ZipFile(archive) as zf:
            zf.read("a.memmap")  # does not raise anymore
        unpack_memmap(archive, tmp_path / "unpacked")
        assert (TensorDict.load_memmap(tmp_path / "unpacked")["a"] == td["a"] + 1).all()

    def test_archive_mode_rplus_rejects_non_writable_layouts(self, tmp_path):
        td = self._nested_td()
        compressed = tmp_path / "compressed.tdz"
        td.save(compressed, compression="deflate")
        with pytest.raises(RuntimeError, match="is compressed"):
            TensorDict.load_memmap(compressed, mode="r+")
        # misaligned foreign zip
        td.memmap(tmp_path / "plain")
        foreign = tmp_path / "foreign.zip"
        with zipfile.ZipFile(foreign, "w") as zf:
            for path in sorted((tmp_path / "plain").rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(tmp_path / "plain").as_posix())
        with pytest.raises(RuntimeError, match="is not aligned"):
            TensorDict.load_memmap(foreign, mode="r+")
        # nested tensors cannot write through
        nt = torch.nested.nested_tensor([torch.randn(2, 3), torch.randn(4, 3)])
        ntpath = tmp_path / "nt.tdz"
        TensorDict({"nt": nt}, batch_size=[]).save(ntpath)
        with pytest.raises(RuntimeError, match="nested tensor"):
            TensorDict.load_memmap(ntpath, mode="r+")
        # invalid mode
        with pytest.raises(ValueError, match="mode must be"):
            TensorDict.load_memmap(tmp_path / "plain", mode="w")

    def test_archive_threaded_save(self, tmp_path):
        # regression: num_threads>1 must not scramble the pairing between
        # source tensors and archive entries (staging insertion order used
        # to follow thread completion order)
        td = TensorDict(
            {
                f"g{i}": {f"t{j}": torch.randn(4 + (7 * i + j) % 96) for j in range(8)}
                for i in range(16)
            },
            batch_size=[],
        )
        for i in range(3):
            archive = tmp_path / f"data{i}.tdz"
            td.save(archive, num_threads=32)
            assert (TensorDict.load_memmap(archive) == td).all()

    def test_archive_memmapped_source(self, tmp_path):
        # already-memmapped tensordicts are packed straight from their
        # directory, without re-staging
        td = self._nested_td()
        tdm = td.memmap(tmp_path / "plain")
        archive = tmp_path / "data.tdz"
        tdm.save(archive)
        assert (TensorDict.load_memmap(archive) == td).all()

    def test_archive_memmap_like(self, tmp_path):
        td = TensorDict(
            {"a": torch.randn(3, 4), "b": {"c": torch.randn(3, 2)}}, batch_size=[3]
        )
        archive = tmp_path / "buf.tdz"
        buf = td.memmap_like(archive)
        assert is_memmap_archive(archive)
        # contentless: the buffer is zero-filled, not a copy of td
        assert (buf["a"] == 0).all()
        assert buf.batch_size == td.batch_size
        assert buf["a"].shape == td["a"].shape
        # the buffer writes through to the archive
        buf["a"].copy_(td["a"])
        buf["b", "c"].copy_(td["b", "c"])
        reloaded = TensorDict.load_memmap(archive)
        assert (reloaded == td).all()
        # a refreshed archive works with zip-level tooling
        refresh_archive_checksums(archive)
        unpack_memmap(archive, tmp_path / "unpacked")
        assert (TensorDict.load_memmap(tmp_path / "unpacked") == td).all()

    def test_archive_memmap_like_expanded(self, tmp_path):
        # preallocating a buffer from an expanded (stride-0) tensordict
        # does not materialize the data
        datum = TensorDict(
            {"img": torch.zeros(3, 8, 8, dtype=torch.uint8)}, batch_size=[]
        )
        buf = datum.expand(1000).memmap_like(tmp_path / "ds.tdz")
        assert buf["img"].shape == (1000, 3, 8, 8)
        buf[42] = TensorDict(
            {"img": torch.full((3, 8, 8), 7, dtype=torch.uint8)}, batch_size=[]
        )
        reloaded = TensorDict.load_memmap(tmp_path / "ds.tdz")
        assert (reloaded["img"][42] == 7).all()
        assert (reloaded["img"][41] == 0).all()

    def test_archive_memmap_like_errors(self, tmp_path):
        td = TensorDict({"a": torch.randn(3)}, batch_size=[])
        archive = tmp_path / "buf.tdz"
        td.memmap_like(archive)
        with pytest.raises(RuntimeError, match="already exists"):
            td.memmap_like(archive, existsok=False)
        with pytest.raises(NotImplementedError, match="return_early"):
            td.memmap_like(archive, num_threads=2, return_early=True)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
