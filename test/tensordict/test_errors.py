# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import platform
import re
import sys
import warnings

import pytest
import torch
from packaging import version
from tensordict import (
    get_defaults_to_none,
    LazyStackedTensorDict,
    set_get_defaults_to_none,
    TensorDict,
)
from tensordict.nn import TensorDictParams
from tensordict.utils import _LOCK_ERROR
from torch import nn

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.tensordict.test._utils_internal import is_npu_available
else:
    IS_FB = False
    from _utils_internal import is_npu_available


_has_streaming = importlib.util.find_spec("streaming", None) is not None

try:
    import h5py  # noqa

    _has_h5py = True
except ImportError:
    _has_h5py = False
TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)

_has_onnx = importlib.util.find_spec("onnxruntime", None) is not None

_v2_5 = TORCH_VERSION >= version.parse("2.5.0")
PYTORCH_TEST_FBCODE = os.getenv("PYTORCH_TEST_FBCODE")

_IS_OSX = platform.system() == "Darwin"
_IS_WINDOWS = sys.platform == "win32"

TD_BATCH_SIZE = 4
HAS_NESTED_TENSOR = (
    getattr(torch, "_nested_compute_contiguous_strides_offsets", None) is not None
)

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


class TestErrorMessage:
    @staticmethod
    def test_err_msg_missing_nested():
        td = TensorDict({"a": torch.zeros(())}, [])
        with pytest.raises(ValueError, match="Expected a TensorDictBase instance"):
            td["a", "b"]

    @staticmethod
    def test_inplace_error():
        td = TensorDict({"a": torch.rand(())}, [])
        with pytest.raises(ValueError, match="Failed to update 'a'"):
            td.set_("a", torch.randn(2))


class TestErrors:
    def test_error_get(self):
        td = TensorDict({"a": 0, "b": {"c": 1}})

        def run_assertions():
            if get_defaults_to_none():
                assert td.get("c") is None
                assert td.get(("b", "d")) is None
            elif get_defaults_to_none() is None:
                with pytest.raises(KeyError), pytest.warns(DeprecationWarning):
                    td.get("c")
                with pytest.raises(KeyError), pytest.warns(DeprecationWarning):
                    td.get(("b", "d"))
            else:
                with pytest.raises(KeyError), warnings.catch_warnings():
                    td.get("c")
                with pytest.raises(KeyError), warnings.catch_warnings():
                    td.get(("b", "d"))

        set_back = get_defaults_to_none()
        try:
            run_assertions()
            set_get_defaults_to_none(False)
            assert not get_defaults_to_none()
            run_assertions()
            set_get_defaults_to_none(True)
            assert get_defaults_to_none()
            run_assertions()
            set_get_defaults_to_none(None)
            assert get_defaults_to_none() is False
            run_assertions()
        finally:
            set_get_defaults_to_none(set_back)

    def test_getitem(self):
        td = TensorDict({"a": 0, "b": {"c": 1}})

        def run_assertions():
            with pytest.raises(KeyError), warnings.catch_warnings():
                td["c"]
            with pytest.raises(KeyError), warnings.catch_warnings():
                td["b", "d"]

        set_back = get_defaults_to_none()
        try:
            run_assertions()
            set_get_defaults_to_none(False)
            assert not get_defaults_to_none()
            run_assertions()
            set_get_defaults_to_none(True)
            assert get_defaults_to_none()
            run_assertions()
            set_get_defaults_to_none(None)
            assert get_defaults_to_none() is False
            run_assertions()
        finally:
            set_get_defaults_to_none(set_back)

    def test_rename(self):
        td = TensorDict({"a": 0, "b": {"c": 1}})

        def run_assertions():
            tdclone = td.clone()
            with pytest.raises(KeyError), warnings.catch_warnings():
                td.rename_key_("c", "d")
            assert (td == tdclone).all()
            with pytest.raises(KeyError), warnings.catch_warnings():
                td.rename_key_(("b", "d"), "c")
            assert (td == tdclone).all()

        set_back = get_defaults_to_none()
        try:
            run_assertions()
            set_get_defaults_to_none(False)
            assert not get_defaults_to_none()
            run_assertions()
            set_get_defaults_to_none(True)
            assert get_defaults_to_none()
            run_assertions()
            set_get_defaults_to_none(None)
            assert get_defaults_to_none() is False
            run_assertions()
        finally:
            set_get_defaults_to_none(set_back)

    def test_del(self):
        td = TensorDict({"a": 0, "b": {"c": 1}})

        def run_assertions():
            tdclone = td.clone()
            with pytest.raises(KeyError), warnings.catch_warnings():
                td.del_("c")
            assert (td == tdclone).all()
            with pytest.raises(KeyError), warnings.catch_warnings():
                td.del_(("b", "d"))
            assert (td == tdclone).all()

        set_back = get_defaults_to_none()
        try:
            run_assertions()
            set_get_defaults_to_none(False)
            assert not get_defaults_to_none()
            run_assertions()
            set_get_defaults_to_none(True)
            assert get_defaults_to_none()
            run_assertions()
            set_get_defaults_to_none(None)
            assert get_defaults_to_none() is False
            run_assertions()
        finally:
            set_get_defaults_to_none(set_back)

    def test_select(self):
        td = TensorDict({"a": 0, "b": {"c": 1}})

        def run_assertions():
            tdclone = td.clone()
            with pytest.raises(KeyError), warnings.catch_warnings():
                td.select("c")
            assert (td == tdclone).all()
            with pytest.raises(KeyError), warnings.catch_warnings():
                td.select(("b", "d"))
            assert (td == tdclone).all()

        set_back = get_defaults_to_none()
        try:
            run_assertions()
            set_get_defaults_to_none(False)
            assert not get_defaults_to_none()
            run_assertions()
            set_get_defaults_to_none(True)
            assert get_defaults_to_none()
            run_assertions()
            set_get_defaults_to_none(None)
            assert get_defaults_to_none() is False
            run_assertions()
        finally:
            set_get_defaults_to_none(set_back)

    def test_split_keys(self):
        td = TensorDict({"a": 0, "b": {"c": 1}})

        def run_assertions():
            tdclone = td.clone()
            with pytest.raises(KeyError), warnings.catch_warnings():
                td.split_keys(["c"])
            assert (td == tdclone).all()
            with pytest.raises(KeyError), warnings.catch_warnings():
                td.split_keys([("b", "d")])
            assert (td == tdclone).all()

        set_back = get_defaults_to_none()
        try:
            run_assertions()
            set_get_defaults_to_none(False)
            assert not get_defaults_to_none()
            run_assertions()
            set_get_defaults_to_none(True)
            assert get_defaults_to_none()
            run_assertions()
            set_get_defaults_to_none(None)
            assert get_defaults_to_none() is False
            run_assertions()
        finally:
            set_get_defaults_to_none(set_back)


class TestLock:
    @staticmethod
    def check_weakref_count(weakref_list, expected):
        count = 0
        ids = set()
        for wr in weakref_list:
            td = wr()
            count += (td is not None) and (td.is_locked) and (id(td) not in ids)
            if td is not None:
                ids.add(id(td))
        assert count == expected, {id(ref()) for ref in weakref_list}

    @pytest.mark.skipif(
        not torch.cuda.is_available() and not is_npu_available(),
        # and not torch.backends.mps.is_available(),
        reason="a device is required.",
    )
    def test_cached_data_lock_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif is_npu_available():
            device = torch.device("npu:0")
        else:
            device = torch.device("mps:0")
        td = TensorDictParams(
            TensorDict(a=nn.Parameter(torch.ones(1)), device="cpu"), no_convert=True
        )
        dataptr = td.data.data_ptr()
        assert (td.to(device).data.data_ptr() != dataptr).all()
        original_td = TensorDict(a=nn.Parameter(torch.ones(1)), device="cpu")
        td = TensorDictParams(original_td, no_convert=True)
        td.lock_()
        dataptr = td.data.data_ptr()
        tddevice = nn.ModuleList([td]).to(device)[0]
        assert td.device == device
        assert original_td.device == torch.device("cpu")
        assert (td.data.data_ptr() != dataptr).all()
        assert tddevice.device == device
        assert (tddevice.data.data_ptr() != dataptr).all()

    def test_lock_stack(self):
        td0 = TensorDict({("a", "b", "c", "d"): 1.0}, [])
        td1 = td0.clone()
        td = LazyStackedTensorDict.lazy_stack([td0, td1])
        td = td.lock_()
        a = td["a"]
        b = td["a", "b"]
        c = td["a", "b", "c"]
        a0 = td0["a"]
        b0 = td0["a", "b"]
        c0 = td0["a", "b", "c"]
        self.check_weakref_count(a._lock_parents_weakrefs, 3)  # td, td0, td1
        self.check_weakref_count(b._lock_parents_weakrefs, 5)  # td, td0, td1, a0, a1
        self.check_weakref_count(
            c._lock_parents_weakrefs, 7
        )  # td, td0, td1, a0, a1, b0, b1
        self.check_weakref_count(a0._lock_parents_weakrefs, 2)  # td, td0
        self.check_weakref_count(b0._lock_parents_weakrefs, 3)  # td, td0, a0
        self.check_weakref_count(c0._lock_parents_weakrefs, 4)  # td, td0, a0, b0
        td.unlock_()
        td.lock_()
        del td, td0, td1
        gc.collect()
        a.unlock_()
        a.lock_()
        self.check_weakref_count(a._lock_parents_weakrefs, 0)
        self.check_weakref_count(b._lock_parents_weakrefs, 3)  # a, a0, a1
        self.check_weakref_count(c._lock_parents_weakrefs, 5)  # a, a0, a1, b0, b1
        self.check_weakref_count(a0._lock_parents_weakrefs, 1)  # a
        self.check_weakref_count(b0._lock_parents_weakrefs, 2)  # a, a0
        self.check_weakref_count(c0._lock_parents_weakrefs, 3)  # a, a0, b0
        del a, a0
        gc.collect()
        b.unlock_()
        b.lock_()
        del b

    def test_lock_two_roots(self):
        td = TensorDict({("a", "b", "c", "d"): 1.0}, [])
        td = td.lock_()
        a = td["a"]
        b = td["a", "b"]
        c = td["a", "b", "c"]
        other_td = TensorDict({"a": a}, [])
        other_td.lock_()
        # we cannot unlock anything anymore
        with pytest.raises(
            RuntimeError,
            match="Cannot unlock a tensordict that is part of a locked graph.",
        ):
            other_td.unlock_()
        assert td._is_locked
        assert td.is_locked
        with pytest.raises(
            RuntimeError,
            match="Cannot unlock a tensordict that is part of a locked graph.",
        ):
            td.unlock_()
        # if we group them we can't unlock
        supertd = TensorDict({"td": td, "other": other_td}, [])
        supertd = supertd.lock_()
        supertd = supertd.unlock_()
        supertd = supertd.lock_()
        del supertd, other_td
        gc.collect()
        self.check_weakref_count(td._lock_parents_weakrefs, 0)
        # self.check_td_not_in_weakref_list(supertd, a._lock_parents_weakrefs)
        # self.check_td_not_in_weakref_list(other_td, a._lock_parents_weakrefs)
        self.check_weakref_count(a._lock_parents_weakrefs, 1)
        # self.check_td_not_in_weakref_list(supertd, b._lock_parents_weakrefs)
        # self.check_td_not_in_weakref_list(other_td, b._lock_parents_weakrefs)
        self.check_weakref_count(b._lock_parents_weakrefs, 2)
        self.check_weakref_count(c._lock_parents_weakrefs, 3)
        td.unlock_()

    def test_nested_lock(self):
        td = TensorDict({("a", "b", "c", "d"): 1.0}, [])
        td = td.lock_()
        assert not td._lock_parents_weakrefs, id(td)
        a = td["a"]
        b = td["a", "b"]
        c = td["a", "b", "c"]
        self.check_weakref_count(a._lock_parents_weakrefs, 1)
        self.check_weakref_count(b._lock_parents_weakrefs, 2)
        self.check_weakref_count(c._lock_parents_weakrefs, 3)
        td = td.unlock_()
        self.check_weakref_count(a._lock_parents_weakrefs, 0)
        self.check_weakref_count(b._lock_parents_weakrefs, 0)
        self.check_weakref_count(c._lock_parents_weakrefs, 0)
        td = td.lock_()
        del td
        gc.collect()
        self.check_weakref_count(a._lock_parents_weakrefs, 0)
        self.check_weakref_count(b._lock_parents_weakrefs, 1)
        self.check_weakref_count(c._lock_parents_weakrefs, 2)
        a = a.lock_()
        del a
        gc.collect()
        self.check_weakref_count(b._lock_parents_weakrefs, 0)
        self.check_weakref_count(c._lock_parents_weakrefs, 1)
        b = b.lock_()
        del b
        gc.collect()
        self.check_weakref_count(c._lock_parents_weakrefs, 0)

    def test_nested_lock_erros(self):
        td = TensorDict({("a", "b", "c", "d"): 1.0}, [])
        td = td.lock_()
        a = td["a"]
        b = td["a", "b"]
        c = td["a", "b", "c"]
        # we cannot unlock a
        with pytest.raises(
            RuntimeError,
            match="Cannot unlock a tensordict that is part of a locked graph.",
        ):
            a.unlock_()
        with pytest.raises(
            RuntimeError,
            match="Cannot unlock a tensordict that is part of a locked graph.",
        ):
            b.unlock_()
        self.check_weakref_count(a._lock_parents_weakrefs, 1)
        self.check_weakref_count(b._lock_parents_weakrefs, 2)
        self.check_weakref_count(c._lock_parents_weakrefs, 3)
        del td
        gc.collect()
        a.unlock_()
        a.lock_()
        with pytest.raises(
            RuntimeError,
            match="Cannot unlock a tensordict that is part of a locked graph.",
        ):
            b.unlock_()

    def test_stack_cache_lock(self):
        td0 = TensorDict({("a", "b", "c", "d"): 1.0}, [])
        td1 = td0.clone()
        td = LazyStackedTensorDict.lazy_stack([td0, td1])
        assert td._is_locked is None
        td = td.lock_()
        assert td._is_locked
        td.unlock_()
        assert td._is_locked is None
        td0.lock_()
        # all tds must be locked
        assert not td.is_locked
        # lock td1
        td1.lock_()
        # we can unlock td0, even though td is locked
        assert td.is_locked
        assert td._is_locked is None  # lock wasn't called on td
        td0.unlock_()
        td.unlock_()
        assert not td0.is_locked
        assert td._is_locked is None

        td.lock_()
        assert td1.is_locked
        assert td0.is_locked
        with pytest.raises(RuntimeError):
            td1.unlock_()
        assert td1.is_locked

        # create a parent to td
        super_td = TensorDict({"td": td}, [])
        super_td.lock_()
        assert td._is_locked
        super_td.unlock_()
        assert td._is_locked is None

    def test_stacked_append_and_insert(self):
        td0 = TensorDict({("a", "b", "c", "d"): 1.0}, [])
        td1 = td0.clone()
        td = LazyStackedTensorDict.lazy_stack([td0, td1])
        td.lock_()
        with pytest.raises(RuntimeError, match=re.escape(_LOCK_ERROR)):
            td.insert(0, td0)
        with pytest.raises(RuntimeError, match=re.escape(_LOCK_ERROR)):
            td.append(td0)
        td.unlock_()
        td.insert(0, td0)
        td.append(td0)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
