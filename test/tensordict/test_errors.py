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
import subprocess
import sys
import warnings

import pytest
import torch
from tensordict import (
    get_defaults_to_none,
    is_tensorclass,
    LazyStackedTensorDict,
    NonTensorData,
    NonTensorStack,
    set_get_defaults_to_none,
    tensorclass,
    TensorDict,
    TensorDictBase,
)
from tensordict.base import (
    _get_defaults_to_none,
    _is_tensor_collection,
    _set_get_defaults_to_none,
)
from tensordict.nn import TensorDictParams
from tensordict.utils import _GENERIC_NESTED_ERR, _LOCK_ERROR
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


class TestErrorMessage:
    @staticmethod
    def test_err_msg_missing_nested():
        td = TensorDict({"a": torch.zeros(())}, [])
        with pytest.raises(
            ValueError,
            match="'a' is a Tensor, not a tensordict, so it has no entry 'b'",
        ):
            td["a", "b"]

    @staticmethod
    def test_inplace_error():
        td = TensorDict({"a": torch.rand(())}, [])
        with pytest.raises(ValueError, match="Failed to update 'a'"):
            td.set_("a", torch.randn(2))

    @staticmethod
    def test_inplace_error_is_runtime_error():
        # update_ and copy_ raise torch's RuntimeError for the same mismatch:
        # the error of set_ is caught by except ValueError and except RuntimeError
        td = TensorDict(a=torch.zeros(3, 4), nested=TensorDict(b=torch.zeros(3)))
        for write in (
            lambda: td.set_("a", torch.zeros(3, 5)),
            lambda: td.set("a", torch.zeros(3, 5), inplace=True),
            lambda: td.update({"a": torch.zeros(3, 5)}, inplace=True),
            lambda: td.set_(("nested", "b"), torch.zeros(4)),
            lambda: LazyStackedTensorDict.lazy_stack([td, td]).set_(
                "a", torch.zeros(2, 3, 5)
            ),
        ):
            with pytest.raises(RuntimeError, match="Failed to update") as err:
                write()
            assert isinstance(err.value, ValueError)
        with pytest.raises(RuntimeError, match="must match"):
            td.update_({"a": torch.zeros(3, 5)})

    @staticmethod
    @pytest.mark.parametrize("key", [1, None, (1, "a")])
    def test_non_str_key(key):
        err = re.escape(_GENERIC_NESTED_ERR.format(key))
        value = torch.zeros(2)
        td = TensorDict({"a": value}, [2])
        with pytest.raises(KeyError, match=err):
            TensorDict({key: value}, [2])
        with pytest.raises(KeyError, match=err):
            TensorDict.from_dict({key: value})
        with pytest.raises(KeyError, match=err):
            td.set(key, value)
        with pytest.raises(KeyError, match=err):
            td.set_(key, value)
        with pytest.raises(KeyError, match=err):
            td.update({key: value})
        with pytest.raises(KeyError, match=err):
            td.get(key)
        with pytest.raises(KeyError, match=err):
            td.get_at(key, 0)
        with pytest.raises(KeyError, match=err):
            td.pop(key)
        assert list(td.keys()) == ["a"]

    @staticmethod
    @pytest.mark.parametrize("td_type", ["td", "lazy_stack", "sub_td"])
    def test_set_self_error(td_type):
        td = TensorDict({"a": torch.zeros(3, 2)}, [3, 2])
        if td_type == "lazy_stack":
            td = LazyStackedTensorDict(*td.unbind(0), stack_dim=0)
        elif td_type == "sub_td":
            td = td._get_sub_tensordict(0)
        with pytest.raises(ValueError, match="Cannot set a tensordict inside itself"):
            td["self"] = td
        with pytest.raises(ValueError, match="Cannot set a tensordict inside itself"):
            td.set(("nested", "self"), td)
        assert set(td.keys()) == {"a"}

    @staticmethod
    def test_batch_mismatch_names_key():
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "batch dimension mismatch, got self.batch_size=torch.Size([3]) and "
                "value.shape=torch.Size([2]) for key 'b'."
            ),
        ):
            TensorDict(a=torch.zeros(3), b=torch.zeros(2), batch_size=[3])
        td = TensorDict(a=torch.zeros(3), batch_size=[3])
        with pytest.raises(RuntimeError, match="for key 'c'"):
            td["c"] = torch.zeros(4)
        with pytest.raises(RuntimeError, match="for key 'n'.*NonTensorData"):
            td["n"] = 1
        td["x"] = TensorDict(a=torch.zeros(3), batch_size=[3])
        with pytest.raises(RuntimeError, match="for key 'b'"):
            td["x", "b"] = torch.zeros(2)
        sub = TensorDict(a=torch.zeros(2, 3), batch_size=[2, 3])._get_sub_tensordict(0)
        with pytest.raises(RuntimeError, match="for key 'c'"):
            sub["c"] = torch.zeros(4)
        lazy = LazyStackedTensorDict(td, td.clone())
        with pytest.raises(RuntimeError, match="for key 'c'"):
            lazy["c"] = torch.zeros(4)
        with pytest.raises(RuntimeError, match=re.escape("for key ('x', 'b')")):
            lazy["x", "b"] = torch.zeros(4)

    @staticmethod
    def test_cat_keys_mismatch():
        td0 = TensorDict(a=torch.zeros(2), batch_size=[2])
        td1 = TensorDict(a=torch.zeros(2), b=torch.zeros(2), batch_size=[2])
        with pytest.raises(
            KeyError,
            match=re.escape(
                "tensordict 2 has keys ['a', 'b'] but tensordict 0 has keys ['a']"
            ),
        ):
            torch.cat([td0, td0, td1])

    @staticmethod
    @pytest.mark.parametrize("lazy", [False, True])
    def test_pop_missing_key(lazy):
        td = TensorDict(a=torch.zeros(2), batch_size=[2])
        if lazy:
            td = LazyStackedTensorDict(td, td.clone())
        with pytest.raises(
            KeyError, match=f'key "zz" not found in {type(td).__name__} with keys'
        ):
            td.pop("zz")

    @staticmethod
    @pytest.mark.parametrize("lazy", [False, True])
    def test_pop_invalid_key(lazy):
        td = TensorDict(a=torch.zeros(2), batch_size=[2])
        if lazy:
            td = LazyStackedTensorDict(td, td.clone())
        with pytest.raises(KeyError, match=re.escape(_GENERIC_NESTED_ERR.format(()))):
            td.pop(())
        # with a default, a key that runs through a leaf is missing (#2678)
        assert td.pop(("a", "b"), None) is None
        assert td.get(("a", "b"), None) is None
        # without one, get, pop and indexing name the leaf
        for read in (lambda: td.pop(("a", "b")), lambda: td.get(("a", "b"))):
            with pytest.raises(ValueError, match="'a' is a Tensor, not a tensordict"):
                read()
        assert list(td.keys()) == ["a"]

    @staticmethod
    @pytest.mark.parametrize(
        "shape,match",
        [
            ((4, -1), "invalid shape [4, -1] for a batch of 6 elements"),
            ((-1, -1), "only one dimension can be inferred"),
            ((-2, 3), "invalid shape dimensions in [-2, 3]"),
        ],
        ids=["indivisible", "two_minus_one", "minus_two"],
    )
    @pytest.mark.parametrize("method", ["view", "reshape"])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_view_invalid_shape(shape, match, method, lazy):
        # A tensor raises RuntimeError for these shapes too
        td = TensorDict(a=torch.zeros(6), batch_size=[6])
        if lazy:
            td = LazyStackedTensorDict(*td.unbind(0))
        with pytest.raises(RuntimeError, match=re.escape(match)):
            getattr(td, method)(*shape)

    @staticmethod
    @pytest.mark.parametrize("shape", [(5,), (2, 7)])
    @pytest.mark.parametrize(
        "td_type,method",
        [
            ("empty", "view"),
            ("empty", "reshape"),
            ("non_tensor", "view"),
            ("non_tensor", "reshape"),
            ("lazy", "reshape"),
            ("tensor", "view"),
            ("tensor", "reshape"),
        ],
    )
    def test_view_wrong_element_count(shape, td_type, method):
        # Without tensor leaves, nothing used to check the element count
        if td_type == "empty":
            td = TensorDict(batch_size=[4, 3])
        elif td_type == "non_tensor":
            td = TensorDict(s=NonTensorData("x", batch_size=[4, 3]), batch_size=[4, 3])
        elif td_type == "lazy":
            td = LazyStackedTensorDict(*[TensorDict(batch_size=[3]) for _ in range(4)])
        else:
            td = TensorDict(a=torch.zeros(4, 3), batch_size=[4, 3])
        with pytest.raises(
            RuntimeError,
            match=re.escape(f"invalid shape {list(shape)} for a batch of 12 elements"),
        ):
            getattr(td, method)(*shape)

    @staticmethod
    @pytest.mark.parametrize("td_type", ["td", "sub_td", "params"])
    def test_iter_0d(td_type):
        td = TensorDict({"a": torch.zeros(2, 3)}, [2])
        if td_type == "sub_td":
            td = td._get_sub_tensordict(0)
        else:
            td = td[0]
            if td_type == "params":
                td = TensorDictParams(td)
        assert td.batch_dims == 0
        err = "iteration over a 0-d tensordict"
        # iter() raises at once, as it does for a 0-d tensor
        with pytest.raises(TypeError, match=err):
            iter(td)
        with pytest.raises(TypeError, match=err):
            list(td)


class TestErrors:
    def test_error_get(self):
        td = TensorDict({"a": 0, "b": {"c": 1}})

        def run_assertions():
            if _get_defaults_to_none():
                assert td.get("c") is None
                assert td.get(("b", "d")) is None
            elif _get_defaults_to_none() is None:
                with pytest.raises(KeyError), pytest.warns(DeprecationWarning):
                    td.get("c")
                with pytest.raises(KeyError), pytest.warns(DeprecationWarning):
                    td.get(("b", "d"))
            else:
                with pytest.raises(KeyError), warnings.catch_warnings():
                    td.get("c")
                with pytest.raises(KeyError), warnings.catch_warnings():
                    td.get(("b", "d"))

        set_back = _get_defaults_to_none()
        try:
            run_assertions()
            _set_get_defaults_to_none(False)
            assert not _get_defaults_to_none()
            run_assertions()
            _set_get_defaults_to_none(True)
            assert _get_defaults_to_none()
            run_assertions()
            _set_get_defaults_to_none(None)
            assert _get_defaults_to_none() is False
            run_assertions()
        finally:
            _set_get_defaults_to_none(set_back)

    def test_getitem(self):
        td = TensorDict({"a": 0, "b": {"c": 1}})

        def run_assertions():
            with pytest.raises(KeyError), warnings.catch_warnings():
                td["c"]
            with pytest.raises(KeyError), warnings.catch_warnings():
                td["b", "d"]

        set_back = _get_defaults_to_none()
        try:
            run_assertions()
            _set_get_defaults_to_none(False)
            assert not _get_defaults_to_none()
            run_assertions()
            _set_get_defaults_to_none(True)
            assert _get_defaults_to_none()
            run_assertions()
            _set_get_defaults_to_none(None)
            assert _get_defaults_to_none() is False
            run_assertions()
        finally:
            _set_get_defaults_to_none(set_back)

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

        set_back = _get_defaults_to_none()
        try:
            run_assertions()
            _set_get_defaults_to_none(False)
            assert not _get_defaults_to_none()
            run_assertions()
            _set_get_defaults_to_none(True)
            assert _get_defaults_to_none()
            run_assertions()
            _set_get_defaults_to_none(None)
            assert _get_defaults_to_none() is False
            run_assertions()
        finally:
            _set_get_defaults_to_none(set_back)

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

        set_back = _get_defaults_to_none()
        try:
            run_assertions()
            _set_get_defaults_to_none(False)
            assert not _get_defaults_to_none()
            run_assertions()
            _set_get_defaults_to_none(True)
            assert _get_defaults_to_none()
            run_assertions()
            _set_get_defaults_to_none(None)
            assert _get_defaults_to_none() is False
            run_assertions()
        finally:
            _set_get_defaults_to_none(set_back)

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

        set_back = _get_defaults_to_none()
        try:
            run_assertions()
            _set_get_defaults_to_none(False)
            assert not _get_defaults_to_none()
            run_assertions()
            _set_get_defaults_to_none(True)
            assert _get_defaults_to_none()
            run_assertions()
            _set_get_defaults_to_none(None)
            assert _get_defaults_to_none() is False
            run_assertions()
        finally:
            _set_get_defaults_to_none(set_back)

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

        set_back = _get_defaults_to_none()
        try:
            run_assertions()
            _set_get_defaults_to_none(False)
            assert not _get_defaults_to_none()
            run_assertions()
            _set_get_defaults_to_none(True)
            assert _get_defaults_to_none()
            run_assertions()
            _set_get_defaults_to_none(None)
            assert _get_defaults_to_none() is False
            run_assertions()
        finally:
            _set_get_defaults_to_none(set_back)

    def test_set_get_defaults_to_none_is_deprecated(self):
        td = TensorDict({"a": 0})
        message = (
            "set_get_defaults_to_none() is deprecated and will be removed in "
            "TensorDict 0.17. Use td[key] to raise a KeyError for a missing key "
            "instead."
        )
        set_back = _get_defaults_to_none()
        try:
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                set_get_defaults_to_none(False)
            assert len(record) == 1
            assert record[0].category is DeprecationWarning
            assert str(record[0].message) == message
            assert record[0].filename == __file__
            # the function still works
            assert _get_defaults_to_none() is False
            with pytest.raises(KeyError):
                td.get("b")
            with pytest.warns(DeprecationWarning, match=re.escape(message)):
                set_get_defaults_to_none(True)
            assert td.get("b") is None
        finally:
            _set_get_defaults_to_none(set_back)

    def test_get_defaults_to_none_is_deprecated(self):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = get_defaults_to_none()
        assert len(record) == 1
        assert record[0].category is DeprecationWarning
        assert str(record[0].message) == (
            "get_defaults_to_none() is deprecated and will be removed in "
            "TensorDict 0.17. Use td[key] to raise a KeyError for a missing key "
            "instead."
        )
        assert record[0].filename == __file__
        assert result == _get_defaults_to_none()

    @pytest.mark.parametrize("value", ["0", "1"])
    def test_get_defaults_to_none_env_var(self, value):
        # TD_GET_DEFAULTS_TO_NONE=0 warns once at import, the default (1) does not
        code = (
            "import warnings\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            "    import tensordict\n"
            "    from tensordict import TensorDict\n"
            "for w in caught:\n"
            "    if 'TD_GET_DEFAULTS_TO_NONE' in str(w.message):\n"
            "        print(w.category.__name__, w.message)\n"
            "try:\n"
            "    print('get:', TensorDict().get('missing'))\n"
            "except KeyError:\n"
            "    print('get: KeyError')\n"
        )
        env = {**os.environ, "TD_GET_DEFAULTS_TO_NONE": value}
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        if value == "0":
            assert result.stdout.splitlines() == [
                "FutureWarning TD_GET_DEFAULTS_TO_NONE=0 is deprecated and will "
                "be removed in TensorDict 0.17. Use td[key] to raise a KeyError for "
                "a missing key instead.",
                "get: KeyError",
            ]
        else:
            assert result.stdout.splitlines() == ["get: None"]


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

    @staticmethod
    def _lock_case(case):
        """A tree to lock, and the objects that must stay alive with it."""

        @tensorclass
        class TC:
            x: torch.Tensor

        class SubTD(TensorDict):
            pass

        def leaf(*shape):
            return torch.zeros(*shape, requires_grad=True)

        if case == "nested":
            return TensorDict(a=leaf(2), b={"c": leaf(2), "d": {"e": leaf(2)}}), []
        if case == "names":
            td = TensorDict(a=leaf(2, 3), b={"c": leaf(2, 3)}, batch_size=[2, 3])
            return td.refine_names("x", "y"), []
        if case == "non_tensor":
            return (
                TensorDict(
                    a=leaf(2),
                    s=NonTensorData("s", batch_size=[2]),
                    st=NonTensorStack("a", "b"),
                    sub={"z": NonTensorData(3)},
                    batch_size=[2],
                ),
                [],
            )
        if case == "lazy_child":
            stack = LazyStackedTensorDict(
                TensorDict(x=leaf(3), n={"m": leaf(3)}, batch_size=[3]),
                TensorDict(x=leaf(3), n={"m": leaf(3)}, batch_size=[3]),
            )
            return TensorDict(s=stack, y=leaf(2), batch_size=[2]), []
        if case == "lazy_root":
            return (
                LazyStackedTensorDict(
                    TensorDict(x=leaf(3), n={"m": leaf(3)}, batch_size=[3]),
                    TensorDict(x=leaf(3), n={"m": leaf(3)}, batch_size=[3]),
                ),
                [],
            )
        if case == "tensorclass":
            tc = TC(x=leaf(2, 3), batch_size=[2])
            return TensorDict(tc=tc, n={"a": leaf(2)}, batch_size=[2]), []
        if case == "params":
            # The first TensorDictParams adds itself to the parents of the
            # entries that follow it.
            return (
                TensorDict(
                    p1=TensorDictParams(TensorDict(w=leaf(2), n={"b": leaf(2)})),
                    mid={"x": leaf(2)},
                    p2=TensorDictParams(TensorDict(w=leaf(2))),
                ),
                [],
            )
        if case == "subclass_child":
            return TensorDict(a=leaf(2), s=SubTD(c=leaf(2), d={"e": leaf(2)})), []
        if case == "subclass_root":
            return SubTD(a=leaf(2), b={"c": leaf(2), "d": {"e": leaf(2)}}), []
        if case == "shared_child":
            child = TensorDict(z=leaf(2), n={"y": leaf(2)})
            return TensorDict(a=child, b=child, d={"c": child}), []
        if case == "two_roots":
            # The shared entries hold a lock parent before the root locks.
            child = TensorDict(z=leaf(2), n={"y": leaf(2)})
            other = TensorDict(c=child, d={"c": child}).lock_()
            return TensorDict(a=child, b={"c": child}), [other]
        raise NotImplementedError(case)

    @staticmethod
    def _lock_state(td):
        """The lock state of each tensor collection of a tree, by path."""
        nodes = {}

        def walk(node, path):
            if id(node) in nodes:
                return
            nodes[id(node)] = (path, node)
            if isinstance(node, LazyStackedTensorDict):
                children = enumerate(node.tensordicts)
            elif isinstance(node, TensorDictParams):
                children = [("<params>", node._param_td)]
            elif is_tensorclass(node):
                children = [("<td>", node._tensordict)]
            else:
                children = node._tensordict.items()
            for key, child in children:
                if _is_tensor_collection(type(child)):
                    walk(child, path + (str(key),))

        walk(td, ())
        state = []
        for path, node in nodes.values():
            node_dict = node.__dict__
            parents = node_dict.get("__lock_parents_weakrefs")
            if parents is not None:
                parents = [
                    "dead" if ref() is None else nodes.get(id(ref()), ("outside",))[0]
                    for ref in parents
                ]
            state.append(
                (
                    path,
                    type(node).__name__,
                    node_dict.get("_is_locked"),
                    node.is_locked,
                    parents,
                    node_dict.get("_locked_schema"),
                    sorted(set(node_dict) - {"_cache"}),
                )
            )
        return state

    @pytest.mark.parametrize(
        "case",
        [
            "nested",
            "names",
            "non_tensor",
            "lazy_child",
            "lazy_root",
            "tensorclass",
            "params",
            "subclass_child",
            "subclass_root",
            "shared_child",
            "two_roots",
        ],
    )
    def test_lock_state_matches_generic_path(self, case, monkeypatch):
        # TensorDict locks a tree of exact TensorDicts in one recursion. Lock,
        # unlock and .data must leave the state of the generic
        # TensorDictBase._propagate_lock, which locks node by node.
        def run():
            td, keep_alive = self._lock_case(case)
            states = [self._lock_state(td)]
            for step in ("lock_", "lock_", "data", "unlock_", "lock_", "data"):
                if step == "data":
                    data = td.data
                    states.append(self._lock_state(data))
                    states.append(
                        [
                            (
                                key,
                                type(value).__name__,
                                value.requires_grad,
                                value.data_ptr() == td.get(key).data_ptr(),
                            )
                            for key, value in data.items(True, True)
                            if isinstance(value, torch.Tensor)
                        ]
                    )
                    continue
                try:
                    getattr(td, step)()
                except RuntimeError as err:
                    states.append((step, str(err).split(".")[0]))
                states.append(self._lock_state(td))
            del keep_alive
            gc.collect()
            try:
                td.unlock_()
            except RuntimeError as err:
                states.append(("unlock_", str(err).split(".")[0]))
            states.append(self._lock_state(td))
            return states

        dense = run()
        monkeypatch.setattr(
            TensorDict, "_propagate_lock", TensorDictBase._propagate_lock
        )
        generic = run()
        assert dense == generic

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

    def test_failed_unlock_keeps_storage_flags(self):
        # A locked parent blocks the nested unlock_. The shared flag must stay.
        parent = TensorDict({"child": TensorDict({"a": torch.zeros(3)}, [3])}, [3])
        parent.share_memory_()
        child = parent["child"]
        assert child["a"].is_shared()
        with pytest.raises(
            RuntimeError,
            match="Cannot unlock a tensordict that is part of a locked graph.",
        ):
            child.unlock_()
        assert child.is_locked
        assert child._is_shared
        assert child["a"].is_shared()
        assert parent._is_shared

        # memmap_() locks without recording parents. Locking a new parent does,
        # so the nested unlock_ fails and must keep _is_memmap.
        child = TensorDict({"a": torch.zeros(3)}, [3])
        child.memmap_()
        parent = TensorDict({"child": child}, [3])
        parent.lock_()
        with pytest.raises(
            RuntimeError,
            match="Cannot unlock a tensordict that is part of a locked graph.",
        ):
            child.unlock_()
        assert child.is_locked
        assert child._is_memmap

        # parent.memmap_() still records no lock parents, so the child unlock_
        # succeeds today. It must not start raising.
        parent = TensorDict({"child": TensorDict({"a": torch.zeros(3)}, [3])}, [3])
        parent.memmap_()
        child = parent["child"]
        assert child._is_memmap
        child.unlock_()
        assert not child.is_locked
        assert not child._is_memmap
        assert parent._is_memmap
        assert parent.is_locked

        child = TensorDict({"a": torch.zeros(3)}, [3]).consolidate()
        assert hasattr(child, "_consolidated")
        parent = TensorDict({"child": child}, [3])
        parent.lock_()
        with pytest.raises(
            RuntimeError,
            match="Cannot unlock a tensordict that is part of a locked graph.",
        ):
            child.unlock_()
        assert child.is_locked
        assert hasattr(child, "_consolidated")

        # A successful unlock_ still clears the flags, including nested ones.
        parent = TensorDict({"child": TensorDict({"a": torch.zeros(3)}, [3])}, [3])
        parent.share_memory_()
        assert parent["child"]._is_shared
        parent.unlock_()
        assert not parent._is_shared
        assert not parent["child"]._is_shared

        parent = TensorDict({"child": TensorDict({"a": torch.zeros(3)}, [3])}, [3])
        parent.memmap_()
        assert parent["child"]._is_memmap
        parent.unlock_()
        assert not parent._is_memmap
        assert not parent["child"]._is_memmap

        child = TensorDict(
            {"a": torch.zeros(3), "nested": TensorDict({"b": torch.ones(3)}, [3])},
            [3],
        ).consolidate()
        assert hasattr(child, "_consolidated")
        child.unlock_()
        assert not hasattr(child, "_consolidated")
        assert not child.is_locked
        assert not child["nested"].is_locked

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
