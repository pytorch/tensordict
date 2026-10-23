# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import dataclasses
import html
import importlib.util
import os
import platform
import re
import sys
import warnings

import pytest
import torch
from tensordict import (
    get_printoptions,
    LazyStackedTensorDict,
    set_printoptions,
    tensorclass,
    TensorDict,
    TypedTensorDict,
)
from tensordict._torch_func import _stack as stack_td

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.tensordict.test._utils_internal import (
        get_available_devices,
        is_npu_available,
    )
else:
    IS_FB = False
    from _utils_internal import get_available_devices, is_npu_available


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


@pytest.fixture(autouse=True)
def _verbose_repr(request):
    # Most tests here check the format of compact=False; TestCompactRepr checks
    # the default, the compact repr.
    if request.cls is TestCompactRepr:
        yield
        return
    with set_printoptions(compact=False):
        yield


@pytest.mark.parametrize("device", [None, *get_available_devices()])
@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
class TestTensorDictRepr:
    def memmap_td(self, device, dtype):
        if device is not None and device.type != "cpu":
            pytest.skip("MemoryMappedTensors can only be placed on CPU.")
        return self.td(device, dtype).memmap_()

    def nested_td(self, device, dtype):
        if device is not None:
            device_not_none = device
        elif torch.cuda.is_available() and torch.cuda.device_count():
            device_not_none = torch.device("cuda:0")
        elif is_npu_available() and npu_device_count:
            device_not_none = torch.device("npu:0")
        else:
            device_not_none = torch.device("cpu")
        return TensorDict(
            source={
                "my_nested_td": self.td(device, dtype),
                "b": torch.zeros(4, 3, 2, 1, 5, dtype=dtype, device=device_not_none),
            },
            batch_size=[4, 3, 2, 1],
            device=device,
        )

    def nested_tensorclass(self, device, dtype):
        @tensorclass
        class MyClass:
            X: torch.Tensor
            y: "MyClass"
            z: str

        if device is not None:
            device_not_none = device
        elif torch.cuda.is_available() and torch.cuda.device_count():
            device_not_none = torch.device("cuda:0")
        elif is_npu_available() and npu_device_count:
            device_not_none = torch.device("npu:0")
        else:
            device_not_none = torch.device("cpu")
        nested_class = MyClass(
            X=torch.zeros(4, 3, 2, 1, dtype=dtype, device=device_not_none),
            y=MyClass(
                X=torch.zeros(4, 3, 2, 1, dtype=dtype, device=device_not_none),
                y=None,
                z=None,
                batch_size=[4, 3, 2, 1],
            ),
            z="z",
            batch_size=[4, 3, 2, 1],
        )
        return TensorDict(
            source={
                "my_nested_td": nested_class,
                "b": torch.zeros(4, 3, 2, 1, 5, dtype=dtype, device=device_not_none),
            },
            batch_size=[4, 3, 2, 1],
            device=device,
        )

    def share_memory_td(self, device, dtype):
        return self.td(device, dtype).share_memory_()

    def stacked_td(self, device, dtype):
        if device is not None:
            device_not_none = device
        elif torch.cuda.is_available() and torch.cuda.device_count():
            device_not_none = torch.device("cuda:0")
        elif is_npu_available() and npu_device_count:
            device_not_none = torch.device("npu:0")
        else:
            device_not_none = torch.device("cpu")
        td1 = TensorDict(
            source={
                "a": torch.zeros(4, 3, 1, 5, dtype=dtype, device=device_not_none),
                "c": torch.zeros(4, 3, 1, 5, dtype=dtype, device=device_not_none),
            },
            batch_size=[4, 3, 1],
            device=device,
        )
        td2 = TensorDict(
            source={
                "a": torch.zeros(4, 3, 1, 5, dtype=dtype, device=device_not_none),
                "b": torch.zeros(4, 3, 1, 10, dtype=dtype, device=device_not_none),
            },
            batch_size=[4, 3, 1],
            device=device,
        )

        return stack_td([td1, td2], 2, maybe_dense_stack=True)

    def td(self, device, dtype):
        if device is not None:
            device_not_none = device
        elif torch.cuda.is_available() and torch.cuda.device_count():
            device_not_none = torch.device("cuda:0")
        elif is_npu_available() and npu_device_count:
            device_not_none = torch.device("npu:0")
        else:
            device_not_none = torch.device("cpu")

        return TensorDict(
            source={
                "a": torch.zeros(4, 3, 2, 1, 5, dtype=dtype, device=device_not_none)
            },
            batch_size=[4, 3, 2, 1],
            device=device,
        )

    @pytest.mark.skipif(
        not torch.cuda.device_count() and not npu_device_count, reason="no cuda or npu"
    )
    def test_repr_batch_size_update(self, device, dtype):
        td = self.td(device, dtype)
        td.batch_size = torch.Size([4, 3, 2])
        is_shared = False
        tensor_class = "Tensor"
        if device is not None and device.type == "cuda":
            is_shared = True
        tensor_device = device if device else td["a"].device
        if tensor_device.type == "cuda":
            is_shared_tensor = True
        else:
            is_shared_tensor = is_shared
        expected = f"""TensorDict(
    fields={{
        a: {tensor_class}(shape=torch.Size([4, 3, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
    batch_size=torch.Size([4, 3, 2]),
    device={device},
    is_shared={is_shared})"""
        assert repr(td) == expected

    @pytest.mark.skipif(
        not torch.cuda.device_count() and not npu_device_count, reason="no cuda or npu"
    )
    @pytest.mark.parametrize("device_cast", get_available_devices())
    def test_repr_device_to_device(self, device, dtype, device_cast):
        td = self.td(device, dtype)
        if (device_cast is None and (torch.cuda.device_count() > 0)) or (
            device_cast is not None and device_cast.type == "cuda"
        ):
            is_shared = True
        else:
            is_shared = False
        tensor_class = "Tensor"
        td2 = td.to(device_cast)
        tensor_device = device_cast if device_cast else td2["a"].device
        if tensor_device.type == "cuda":
            is_shared_tensor = True
        else:
            is_shared_tensor = is_shared
        expected = f"""TensorDict(
    fields={{
        a: {tensor_class}(shape=torch.Size([4, 3, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
    batch_size=torch.Size([4, 3, 2, 1]),
    device={str(device_cast)},
    is_shared={is_shared})"""
        assert repr(td2) == expected

    @pytest.mark.parametrize("index", [None, (slice(None), 0)])
    def test_repr_indexed_nested_tensordict(self, device, dtype, index):
        nested_tensordict = self.nested_td(device, dtype)[index]
        if device is not None and device.type == "cuda":
            is_shared = True
        else:
            is_shared = False
        tensor_class = "Tensor"
        tensor_device = device if device else nested_tensordict["b"].device
        if tensor_device.type == "cuda":
            is_shared_tensor = True
        else:
            is_shared_tensor = is_shared
        if index is None:
            expected = f"""TensorDict(
    fields={{
        b: {tensor_class}(shape=torch.Size([1, 4, 3, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor}),
        my_nested_td: TensorDict(
            fields={{
                a: {tensor_class}(shape=torch.Size([1, 4, 3, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
            batch_size=torch.Size([1, 4, 3, 2, 1]),
            device={str(device)},
            is_shared={is_shared})}},
    batch_size=torch.Size([1, 4, 3, 2, 1]),
    device={str(device)},
    is_shared={is_shared})"""
        else:
            expected = f"""TensorDict(
    fields={{
        b: {tensor_class}(shape=torch.Size([4, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor}),
        my_nested_td: TensorDict(
            fields={{
                a: {tensor_class}(shape=torch.Size([4, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
            batch_size=torch.Size([4, 2, 1]),
            device={str(device)},
            is_shared={is_shared})}},
    batch_size=torch.Size([4, 2, 1]),
    device={str(device)},
    is_shared={is_shared})"""
        assert repr(nested_tensordict) == expected

    @pytest.mark.parametrize("index", [None, (slice(None), 0)])
    def test_repr_indexed_stacked_tensordict(self, device, dtype, index):
        stacked_tensordict = self.stacked_td(device, dtype)
        if device is not None and device.type == "cuda":
            is_shared = True
        else:
            is_shared = False
        tensor_class = "Tensor"
        tensor_device = device if device else stacked_tensordict["a"].device
        if tensor_device.type == "cuda":
            is_shared_tensor = True
        else:
            is_shared_tensor = is_shared

        expected = f"""LazyStackedTensorDict(
    fields={{
        a: {tensor_class}(shape=torch.Size([4, 3, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
    exclusive_fields={{
        0 ->
            c: {tensor_class}(shape=torch.Size([4, 3, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor}),
        1 ->
            b: {tensor_class}(shape=torch.Size([4, 3, 1, 10]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
    batch_size=torch.Size([4, 3, 2, 1]),
    device={str(device)},
    is_shared={is_shared},
    stack_dim={stacked_tensordict.stack_dim})"""

        assert repr(stacked_tensordict) == expected

    @pytest.mark.parametrize("index", [None, (slice(None), 0)])
    def test_repr_indexed_tensordict(self, device, dtype, index):
        tensordict = self.td(device, dtype)[index]
        if device is not None and device.type == "cuda":
            is_shared = True
        else:
            is_shared = False
        tensor_class = "Tensor"
        tensor_device = device if device else tensordict["a"].device
        if tensor_device.type == "cuda":
            is_shared_tensor = True
        else:
            is_shared_tensor = is_shared
        if index is None:
            expected = f"""TensorDict(
    fields={{
        a: {tensor_class}(shape=torch.Size([1, 4, 3, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
    batch_size=torch.Size([1, 4, 3, 2, 1]),
    device={str(device)},
    is_shared={is_shared})"""
        else:
            expected = f"""TensorDict(
    fields={{
        a: {tensor_class}(shape=torch.Size([4, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
    batch_size=torch.Size([4, 2, 1]),
    device={str(device)},
    is_shared={is_shared})"""

        assert repr(tensordict) == expected

    def test_repr_memmap(self, device, dtype):
        tensordict = self.memmap_td(device, dtype)
        # tensor_device = device if device else tensordict["a"].device  # noqa: F841
        expected = f"""TensorDict(
    fields={{
        a: MemoryMappedTensor(shape=torch.Size([4, 3, 2, 1, 5]), device=cpu, dtype={dtype}, is_shared=False)}},
    batch_size=torch.Size([4, 3, 2, 1]),
    device=cpu,
    is_shared=False)"""
        assert repr(tensordict) == expected

    def test_repr_nested(self, device, dtype):
        nested_td = self.nested_td(device, dtype)
        if device is not None and device.type == "cuda":
            is_shared = True
        else:
            is_shared = False
        tensor_class = "Tensor"
        tensor_device = device if device else nested_td["b"].device
        if tensor_device.type == "cuda":
            is_shared_tensor = True
        else:
            is_shared_tensor = is_shared
        expected = f"""TensorDict(
    fields={{
        b: {tensor_class}(shape=torch.Size([4, 3, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor}),
        my_nested_td: TensorDict(
            fields={{
                a: {tensor_class}(shape=torch.Size([4, 3, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
            batch_size=torch.Size([4, 3, 2, 1]),
            device={str(device)},
            is_shared={is_shared})}},
    batch_size=torch.Size([4, 3, 2, 1]),
    device={str(device)},
    is_shared={is_shared})"""
        assert repr(nested_td) == expected

    def test_repr_nested_lazy(self, device, dtype):
        nested_td0 = self.nested_td(device, dtype)
        nested_td1 = torch.cat([nested_td0, nested_td0], 1)
        nested_td1["my_nested_td", "another"] = nested_td1["my_nested_td", "a"]
        lazy_nested_td = TensorDict.lazy_stack([nested_td0, nested_td1], dim=1)

        if device is not None and device.type == "cuda":
            is_shared = True
        else:
            is_shared = False
        tensor_class = "Tensor"
        tensor_device = device if device else nested_td0[:, 0]["b"].device
        if tensor_device.type == "cuda":
            is_shared_tensor = True
        else:
            is_shared_tensor = is_shared
        expected = f"""LazyStackedTensorDict(
    fields={{
        b: {tensor_class}(shape=torch.Size([4, 2, -1, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor}),
        my_nested_td: LazyStackedTensorDict(
            fields={{
                a: {tensor_class}(shape=torch.Size([4, 2, -1, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
            exclusive_fields={{
                1 ->
                    another: Tensor(shape=torch.Size([4, 6, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
            batch_size=torch.Size([4, 2, -1, 2, 1]),
            device={str(device)},
            is_shared={is_shared},
            stack_dim=1)}},
    exclusive_fields={{
    }},
    batch_size=torch.Size([4, 2, -1, 2, 1]),
    device={str(device)},
    is_shared={is_shared},
    stack_dim=1)"""
        assert repr(lazy_nested_td) == expected

    def test_repr_nested_update(self, device, dtype):
        nested_td = self.nested_td(device, dtype)
        nested_td["my_nested_td"].rename_key_("a", "z")
        if device is not None and device.type == "cuda":
            is_shared = True
        else:
            is_shared = False
        tensor_class = "Tensor"
        tensor_device = device if device else nested_td["b"].device
        if tensor_device.type == "cuda":
            is_shared_tensor = True
        else:
            is_shared_tensor = is_shared
        expected = f"""TensorDict(
    fields={{
        b: {tensor_class}(shape=torch.Size([4, 3, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor}),
        my_nested_td: TensorDict(
            fields={{
                z: {tensor_class}(shape=torch.Size([4, 3, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
            batch_size=torch.Size([4, 3, 2, 1]),
            device={str(device)},
            is_shared={is_shared})}},
    batch_size=torch.Size([4, 3, 2, 1]),
    device={str(device)},
    is_shared={is_shared})"""
        assert repr(nested_td) == expected

    def test_repr_plain(self, device, dtype):
        tensordict = self.td(device, dtype)
        if device is not None and device.type == "cuda":
            is_shared = True
        else:
            is_shared = False
        tensor_device = device if device else tensordict["a"].device
        if tensor_device.type == "cuda":
            is_shared_tensor = True
        else:
            is_shared_tensor = is_shared
        expected = f"""TensorDict(
    fields={{
        a: Tensor(shape=torch.Size([4, 3, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
    batch_size=torch.Size([4, 3, 2, 1]),
    device={str(device)},
    is_shared={is_shared})"""
        assert repr(tensordict) == expected

    def test_repr_share_memory(self, device, dtype):
        tensordict = self.share_memory_td(device, dtype)
        is_shared = True
        tensor_class = "Tensor"
        tensor_device = device if device else tensordict["a"].device
        if tensor_device.type == "cuda":
            is_shared_tensor = True
        elif (
            tensordict["a"].device is not None and tensordict["a"].device.type == "npu"
        ):
            is_shared_tensor = False
        else:
            is_shared_tensor = is_shared
        expected = f"""TensorDict(
    fields={{
        a: {tensor_class}(shape=torch.Size([4, 3, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
    batch_size=torch.Size([4, 3, 2, 1]),
    device={str(device)},
    is_shared={is_shared})"""
        assert repr(tensordict) == expected

    def test_repr_stacked(self, device, dtype):
        stacked_td = self.stacked_td(device, dtype)
        if device is not None and device.type == "cuda":
            is_shared = True
        else:
            is_shared = False
        tensor_class = "Tensor"
        tensor_device = device if device else stacked_td["a"].device
        if tensor_device.type == "cuda":
            is_shared_tensor = True
        else:
            is_shared_tensor = is_shared
        expected = f"""LazyStackedTensorDict(
    fields={{
        a: {tensor_class}(shape=torch.Size([4, 3, 2, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
    exclusive_fields={{
        0 ->
            c: {tensor_class}(shape=torch.Size([4, 3, 1, 5]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor}),
        1 ->
            b: {tensor_class}(shape=torch.Size([4, 3, 1, 10]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
    batch_size=torch.Size([4, 3, 2, 1]),
    device={str(device)},
    is_shared={is_shared},
    stack_dim={stacked_td.stack_dim})"""
        assert repr(stacked_td) == expected

    def test_repr_stacked_het(self, device, dtype):
        stacked_td = LazyStackedTensorDict.lazy_stack(
            [
                TensorDict(
                    {
                        "a": torch.zeros(3, dtype=dtype),
                        "b": torch.zeros(2, 3, dtype=dtype),
                    },
                    [],
                    device=device,
                ),
                TensorDict(
                    {
                        "a": torch.zeros(2, dtype=dtype),
                        "b": torch.zeros((), dtype=dtype),
                    },
                    [],
                    device=device,
                ),
            ]
        )
        if device is not None and device.type == "cuda":
            is_shared = True
        else:
            is_shared = False
        tensor_device = device if device else torch.device("cpu")
        if tensor_device.type == "cuda":
            is_shared_tensor = True
        else:
            is_shared_tensor = is_shared
        expected = f"""LazyStackedTensorDict(
    fields={{
        a: Tensor(shape=torch.Size([2, -1]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor}),
        b: Tensor(shape=torch.Size([-1]), device={tensor_device}, dtype={dtype}, is_shared={is_shared_tensor})}},
    exclusive_fields={{
    }},
    batch_size=torch.Size([2]),
    device={str(device)},
    is_shared={is_shared},
    stack_dim={stacked_td.stack_dim})"""
        assert repr(stacked_td) == expected


class TestSetPrintoptions:
    """Tests for :class:`tensordict.set_printoptions` and :func:`tensordict.get_printoptions`."""

    def test_get_printoptions_returns_copy(self):
        opts = get_printoptions()
        assert isinstance(opts, dict)
        assert "show_device" in opts
        opts["show_device"] = not opts["show_device"]
        assert get_printoptions()["show_device"] != opts["show_device"]

    def test_default_repr_unchanged(self):
        td = TensorDict({"a": torch.randn(3, 4)})
        r = repr(td)
        assert "batch_size=" in r
        assert "device=" in r
        assert "is_shared=" in r
        assert "shape=torch.Size([3, 4])" in r
        assert "dtype=torch.float32" in r

    def test_hide_device(self):
        td = TensorDict({"a": torch.randn(3, 4)})
        with set_printoptions(show_device=False):
            r = repr(td)
        assert "\n    device=" not in r
        assert "batch_size=" in r
        assert "is_shared=" in r

    def test_hide_is_shared(self):
        td = TensorDict({"a": torch.randn(3, 4)})
        with set_printoptions(show_is_shared=False):
            r = repr(td)
        assert "\n    is_shared=" not in r
        assert "batch_size=" in r
        assert "\n    device=" in r

    def test_hide_batch_size(self):
        td = TensorDict({"a": torch.randn(3, 4)})
        with set_printoptions(show_batch_size=False):
            r = repr(td)
        assert "batch_size=" not in r
        assert "\n    device=" in r

    def test_hide_dtype(self):
        td = TensorDict({"a": torch.randn(3, 4)})
        with set_printoptions(show_dtype=False):
            r = repr(td)
        assert "dtype=" not in r
        assert "shape=" in r

    def test_hide_field_device(self):
        td = TensorDict({"a": torch.randn(3, 4)})
        with set_printoptions(show_field_device=False):
            r = repr(td)
        assert "Tensor(shape=" in r
        # td-level device still visible
        assert "\n    device=" in r

    def test_hide_field_is_shared(self):
        td = TensorDict({"a": torch.randn(3, 4)})
        with set_printoptions(show_field_is_shared=False):
            r = repr(td)
        assert "Tensor(" in r
        # Tensor-level is_shared gone, but TD-level is_shared still present
        assert "is_shared=False)" in r  # TD-level
        fields_line = [line for line in r.split("\n") if "Tensor(" in line][0]
        assert "is_shared=" not in fields_line

    def test_hide_multiple(self):
        td = TensorDict({"a": torch.randn(3, 4)})
        with set_printoptions(
            show_device=False,
            show_is_shared=False,
            show_dtype=False,
            show_field_is_shared=False,
        ):
            r = repr(td)
        assert "\n    device=" not in r
        assert "\n    is_shared=" not in r
        assert "dtype=" not in r

    def test_context_manager_restores(self):
        before = get_printoptions()
        with set_printoptions(show_device=False, show_is_shared=False):
            inner = get_printoptions()
            assert inner["show_device"] is False
            assert inner["show_is_shared"] is False
        after = get_printoptions()
        assert after == before

    def test_global_set(self):
        before = get_printoptions()
        ctx = set_printoptions(show_is_shared=False)
        ctx.set()
        try:
            assert get_printoptions()["show_is_shared"] is False
            td = TensorDict({"a": torch.randn(2)})
            assert "\n    is_shared=" not in repr(td)
        finally:
            ctx.__exit__(None, None, None)
        assert get_printoptions() == before

    def test_show_grad(self):
        td = TensorDict({"a": torch.randn(3, requires_grad=True)})
        with set_printoptions(show_grad=True):
            r = repr(td)
        assert "requires_grad=True" in r

    def test_show_is_contiguous(self):
        td = TensorDict({"a": torch.randn(3, 4)})
        with set_printoptions(show_is_contiguous=True):
            r = repr(td)
        assert "is_contiguous=True" in r

    def test_show_is_view(self):
        base = torch.randn(3, 4)
        td = TensorDict({"a": base[::2]})
        with set_printoptions(show_is_view=True):
            r = repr(td)
        assert "is_view=True" in r

    def test_show_storage_size(self):
        td = TensorDict({"a": torch.randn(3, 4)})
        with set_printoptions(show_storage_size=True):
            r = repr(td)
        assert "storage_size=" in r

    def test_plain_mode(self):
        td = TensorDict({"a": torch.ones(10)})
        with set_printoptions(plain=True):
            r = repr(td)
        assert "mean=" in r

    def test_plain_mode_jagged_and_one_element(self):
        # #3259: plain=True raised on a jagged leaf and warned on a
        # one-element leaf
        nt = torch.nested.nested_tensor(
            [torch.arange(6.0).reshape(2, 3), torch.arange(12.0).reshape(4, 3)],
            layout=torch.jagged,
        )
        td = TensorDict(j=nt, a=torch.tensor(1.0), b=torch.tensor([2.0]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with set_printoptions(compact=False, plain=True):
                r = repr(td)
        assert "mean=4.5, std=3.4" in r
        assert "mean=1, std=nan" in r
        assert "mean=2, std=nan" in r

    def test_plain_mode_narrowed_jagged(self):
        # #3259: the summary of a narrowed jagged leaf leaves out the
        # elements the narrow hides
        nt = torch.nested.narrow(
            torch.arange(12).reshape(2, 6),
            1,
            torch.tensor([0, 1]),
            torch.tensor([2, 3]),
            layout=torch.jagged,
        )
        with set_printoptions(compact=False, plain=True):
            r = repr(TensorDict(j=nt, batch_size=[2]))
        assert "min=0, max=9" in r

    def test_plain_mode_complex(self):
        # plain=True dropped the imaginary part of a complex leaf and warned on every print
        td = TensorDict(c=torch.arange(3.0) * (1 + 1j), s=torch.tensor(1.0 + 1j))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with set_printoptions(compact=False, plain=True):
                r = repr(td)
        assert "mean=1+1j, std=1.414" in r
        assert "mean=1+1j, std=nan" in r

    def test_lazy_stacked_respects_options(self):
        td1 = TensorDict({"a": torch.randn(3)})
        td2 = TensorDict({"a": torch.randn(3)})
        stacked = LazyStackedTensorDict.lazy_stack([td1, td2])
        with set_printoptions(show_device=False, show_is_shared=False):
            r = repr(stacked)
        assert "stack_dim=" in r
        assert "\n    device=" not in r
        assert "\n    is_shared=" not in r

    def test_tensorclass_respects_options(self):
        @tensorclass
        class MyClass:
            x: torch.Tensor

        obj = MyClass(x=torch.randn(3, 4), batch_size=[3])
        with set_printoptions(
            show_device=False,
            show_is_shared=False,
            show_field_device=False,
            show_field_is_shared=False,
        ):
            r = repr(obj)
        assert "MyClass(" in r
        assert "device=" not in r
        assert "is_shared=" not in r

    def test_unknown_option_raises(self):
        with pytest.raises(TypeError, match="Unknown printoptions"):
            set_printoptions(nonexistent_option=True)

    def test_decorator_usage(self):
        @set_printoptions(show_device=False, show_is_shared=False)
        def my_func():
            td = TensorDict({"a": torch.randn(2)})
            r = repr(td)
            assert "\n    device=" not in r
            assert "\n    is_shared=" not in r
            return r

        my_func()
        td = TensorDict({"a": torch.randn(2)})
        r = repr(td)
        assert "\n    device=" in r
        assert "\n    is_shared=" in r

    def test_sort_keys_alphabetical_default(self):
        td = TensorDict({"c": torch.randn(2), "a": torch.randn(2), "b": torch.randn(2)})
        r = repr(td)
        keys_in_order = [
            line.strip().split(":")[0] for line in r.split("\n") if "Tensor(" in line
        ]
        assert keys_in_order == ["a", "b", "c"]

    def test_sort_keys_insertion(self):
        td = TensorDict({"c": torch.randn(2), "a": torch.randn(2), "b": torch.randn(2)})
        with set_printoptions(sort_keys="insertion"):
            r = repr(td)
        keys_in_order = [
            line.strip().split(":")[0] for line in r.split("\n") if "Tensor(" in line
        ]
        assert keys_in_order == ["c", "a", "b"]

    def test_sort_keys_callable(self):
        td2 = TensorDict(
            {"xb": torch.randn(2), "ya": torch.randn(2), "za": torch.randn(2)}
        )
        with set_printoptions(sort_keys=lambda s: s[::-1]):
            r2 = repr(td2)
        keys2 = [
            line.strip().split(":")[0] for line in r2.split("\n") if "Tensor(" in line
        ]
        # sorted by reversed key: "xb"→"bx", "ya"→"ay", "za"→"az"  →  "ya","za","xb"
        assert keys2 == ["ya", "za", "xb"]

    def test_sort_keys_restores(self):
        before = get_printoptions()["sort_keys"]
        with set_printoptions(sort_keys="insertion"):
            assert get_printoptions()["sort_keys"] == "insertion"
        assert get_printoptions()["sort_keys"] == before

    def test_verbose_false(self):
        td = TensorDict({"a": torch.randn(3, 4)})
        with set_printoptions(verbose=False):
            r = repr(td)
        assert "shape=" in r
        assert "batch_size=" in r
        assert "dtype=" not in r
        assert "\n    device=" not in r
        assert "\n    is_shared=" not in r
        fields_line = [line for line in r.split("\n") if "Tensor(" in line][0]
        assert "device=" not in fields_line
        assert "is_shared=" not in fields_line

    def test_verbose_false_explicit_override(self):
        td = TensorDict({"a": torch.randn(3, 4)})
        with set_printoptions(verbose=False, show_dtype=True):
            r = repr(td)
        assert "shape=" in r
        assert "dtype=" in r
        assert "\n    device=" not in r
        assert "\n    is_shared=" not in r

    def test_verbose_true_is_noop(self):
        before = get_printoptions()
        with set_printoptions(verbose=True):
            td = TensorDict({"a": torch.randn(3, 4)})
            after = get_printoptions()
            repr(td)  # noqa
        assert before == after

    def test_verbose_false_restores(self):
        before = get_printoptions()
        with set_printoptions(verbose=False):
            opts = get_printoptions()
            assert opts["show_device"] is False
            assert opts["show_dtype"] is False
            assert opts["show_shape"] is True
            assert opts["show_batch_size"] is True
        assert get_printoptions() == before


@tensorclass
class Mesh:
    points: torch.Tensor
    cells: torch.Tensor


class _TypedMesh(TypedTensorDict):
    points: torch.Tensor
    cells: torch.Tensor


class _CountingStore(TensorDict):
    # keeps its own repr, as TensorDictStore does, and counts its leaf reads
    reads = []

    def _repr_parts(self):
        return [f"keys={sorted(self.keys())}"]

    def _get_str(self, key, *args, **kwargs):
        type(self).reads.append(key)
        return super()._get_str(key, *args, **kwargs)


def _nested_store():
    store = _CountingStore(x=torch.zeros(3, 2), y=torch.zeros(3), batch_size=[3])
    td = TensorDict(s=store, z=torch.zeros(3), batch_size=[3])
    assert type(td.get("s")) is _CountingStore
    _CountingStore.reads.clear()
    return td


def _mesh_sample():
    return TensorDict(
        points=torch.zeros(1000, 3),
        cells=torch.zeros(1000, 4, dtype=torch.long),
        data=TensorDict(
            pressure=torch.zeros(1000),
            mask=torch.zeros(1000, dtype=torch.bool),
            batch_size=[1000],
        ),
        batch_size=[1000],
        device="cpu",
    )


def _mesh():
    return Mesh(
        points=torch.zeros(1000, 3),
        cells=torch.zeros(1000, 4, dtype=torch.long),
        batch_size=[1000],
        device="cpu",
    )


class TestCompactRepr:
    """Tests for ``set_printoptions(compact=True)``, the default."""

    def test_compact_default_on(self):
        assert get_printoptions()["compact"] is True
        td = _mesh_sample()
        assert repr(td) == str(td)
        with set_printoptions(compact=True):
            assert repr(td) == str(td) == repr(_mesh_sample())
        assert repr(td).startswith("TensorDict(batch_size=[1000], device=cpu, fields={")
        # compact=False brings back the verbose format, as a context manager
        # and with set()
        with set_printoptions(compact=False):
            verbose = repr(td)
        assert verbose.startswith("TensorDict(\n    fields={")
        assert "is_shared=False" in verbose
        options = set_printoptions(compact=False)
        options.set()
        try:
            assert repr(td) == verbose
        finally:
            options.__exit__(None, None, None)
        assert get_printoptions()["compact"] is True

    def test_compact_default_containers(self):
        # the repr of each container under the default options
        from tensordict import NonTensorData, NonTensorStack, UnbatchedTensor

        mesh = _mesh()
        lazy = LazyStackedTensorDict.lazy_stack(
            [TensorDict(a=torch.zeros(3)), TensorDict(a=torch.zeros(3))]
        )
        td = TensorDict(
            u=UnbatchedTensor(torch.zeros(5)),
            s="name",
            mesh=mesh[:2],
            batch_size=[2],
        )
        assert repr(mesh) == (
            "Mesh(batch_size=[1000], device=cpu, fields={cells: i64[4], points: f32[3]})"
        )
        assert repr(lazy) == (
            "LazyStackedTensorDict(batch_size=[2], stack_dim=0, fields={a: f32[3]})"
        )
        assert repr(td) == (
            "TensorDict(batch_size=[2], fields={\n"
            "    mesh: Mesh{cells: i64[4], points: f32[3]},\n"
            "    s: str,\n"
            "    u: f32[5]})"
        )
        typed = _TypedMesh(
            points=torch.zeros(2, 3),
            cells=torch.zeros(2, 4, dtype=torch.long),
            batch_size=[2],
        )
        assert repr(typed) == (
            "_TypedMesh(batch_size=[2], fields={cells: i64[4], points: f32[3]})"
        )
        memmap = TensorDict(a=torch.zeros(2, 3), batch_size=[2]).memmap_()
        assert (
            repr(memmap) == "TensorDict(batch_size=[2], device=cpu, fields={a: f32[3]})"
        )
        # the non-tensor classes keep their own repr
        with set_printoptions(compact=False):
            expected = repr(NonTensorData("a")), repr(NonTensorStack("a", "b"))
        assert (repr(NonTensorData("a")), repr(NonTensorStack("a", "b"))) == expected

    @pytest.mark.parametrize("value,name", [("text", "str"), (None, "NoneType")])
    @pytest.mark.parametrize("tensor_first", [True, False])
    def test_compact_mixed_lazy_stack(self, value, name, tensor_first):
        # #3233: a key that holds a tensor in one member and a non-tensor in
        # another shows as the non-tensor entry, whatever the order
        members = [TensorDict(a=torch.zeros(3)), TensorDict(a=value)]
        if not tensor_first:
            members = members[::-1]
        lazy = LazyStackedTensorDict(*members)
        assert repr(lazy) == (
            f"LazyStackedTensorDict(batch_size=[2], stack_dim=0, fields={{a: {name}}})"
        )
        out = lazy._repr_html_()
        assert out.startswith("<div")
        assert "a: <span style=" in out and f">{name}</span>" in out

    @pytest.mark.parametrize(
        "options",
        [{}, {"compact": False}, {"compact": False, "plain": True}],
        ids=["compact", "detailed", "plain"],
    )
    @pytest.mark.parametrize("tensor_first", [True, False])
    def test_mixed_lazy_stack_every_display(self, options, tensor_first):
        # #3258: the detailed display raised on a lazy stack whose first
        # member holds a string where a later one holds a tensor; it lists
        # the members' types, in order
        members = [TensorDict(a=torch.zeros(3)), TensorDict(a="text")]
        if not tensor_first:
            members = members[::-1]
        lazy = LazyStackedTensorDict(*members)
        with set_printoptions(**options):
            out = repr(lazy)
        if not options:
            assert out == (
                "LazyStackedTensorDict(batch_size=[2], stack_dim=0, fields={a: str})"
            )
        else:
            types = "Tensor, NonTensorData" if tensor_first else "NonTensorData, Tensor"
            assert out.startswith(
                f"LazyStackedTensorDict(\n    fields={{\n        a: [{types}]}},\n"
            )

    @pytest.mark.parametrize(
        "options",
        [{}, {"compact": False}, {"compact": False, "plain": True}],
        ids=["compact", "detailed", "plain"],
    )
    @pytest.mark.parametrize(
        "case",
        [
            "td_then_tensor",
            "tensor_then_td",
            "non_tensor_stack",
            "nested_key",
            "lazy_in_td",
            "lazy_of_lazy",
        ],
    )
    def test_mixed_lazy_stack_kinds(self, options, case):
        # #3258: an entry whose members' values cannot be stacked prints as
        # their types, wherever the lazy stack sits
        from tensordict import NonTensorStack

        lazy = LazyStackedTensorDict

        def mixed():
            return lazy(TensorDict(a="text"), TensorDict(a=torch.zeros(3)))

        obj, types = {
            "td_then_tensor": (
                lambda: lazy(
                    TensorDict(a=TensorDict(b=1)), TensorDict(a=torch.zeros(3))
                ),
                "TensorDict, Tensor",
            ),
            "tensor_then_td": (
                lambda: lazy(
                    TensorDict(a=torch.zeros(3)), TensorDict(a=TensorDict(b=1))
                ),
                "Tensor, TensorDict",
            ),
            "non_tensor_stack": (
                lambda: lazy(
                    TensorDict(a=torch.zeros(2, 3), batch_size=[2]),
                    TensorDict(a=NonTensorStack("x", "y"), batch_size=[2]),
                ),
                "Tensor, NonTensorStack",
            ),
            "nested_key": (
                lambda: lazy(
                    TensorDict(n=TensorDict(a="text")),
                    TensorDict(n=TensorDict(a=torch.zeros(3))),
                ),
                "NonTensorData, Tensor",
            ),
            "lazy_in_td": (
                lambda: TensorDict(n=mixed(), batch_size=[2]),
                "NonTensorData, Tensor",
            ),
            "lazy_of_lazy": (lambda: lazy(mixed(), mixed()), "NonTensorData, Tensor"),
        }[case]
        with set_printoptions(**options):
            out = repr(obj())
        if options:
            assert f"a: [{types}]}}," in out

    def test_compact_error_falls_back(self, monkeypatch):
        # #3233: an error in the compact form falls back to the detailed one
        import tensordict.base

        def fail(obj):
            raise TypeError("compact")

        monkeypatch.setattr(tensordict.base, "_compact_repr", fail)
        td = TensorDict(a=torch.zeros(3))
        with set_printoptions(compact=False):
            expected = repr(td)
        assert repr(td) == expected

    def test_compact_default_reads_no_nested_store_leaf(self):
        # A nested TensorDictStore has its own _repr_parts and reads its
        # leaves from a server: the default repr must not fetch them.
        calls = []

        class _Store(TensorDict):
            def _repr_parts(self):
                return [f"keys={sorted(self.keys())}"]

            def _get_str(self, key, *args, **kwargs):
                calls.append(key)
                return super()._get_str(key, *args, **kwargs)

        store = _Store(a=torch.zeros(2, 3), b=torch.zeros(2), batch_size=[2])
        td = TensorDict(store=store, batch_size=[2])
        with set_printoptions(compact=False):
            repr(td)
        assert not calls
        repr(td)
        assert not calls

    def test_compact_sample(self):
        with set_printoptions(compact=True):
            r = repr(_mesh_sample())
        assert r == (
            "TensorDict(batch_size=[1000], device=cpu, fields={\n"
            "    cells: i64[4],\n"
            "    data: {mask: bool[], pressure: f32[]},\n"
            "    points: f32[3]})"
        )

    def test_compact_lazy_stack(self):
        td = LazyStackedTensorDict.lazy_stack(
            [
                TensorDict(points=torch.zeros(10, 3), label=torch.tensor(0)),
                TensorDict(points=torch.zeros(12, 3), label=torch.tensor(1)),
            ]
        )
        with set_printoptions(compact=True):
            r = repr(td)
        assert r == (
            "LazyStackedTensorDict(batch_size=[2], stack_dim=0, fields={\n"
            "    label: i64[],\n"
            "    points: f32[-1,3]})"
        )

    def test_compact_lazy_stack_promotes_dtype(self):
        # as the stacked tensor, which the default repr shows
        td = LazyStackedTensorDict.lazy_stack(
            [
                TensorDict(x=torch.zeros(3)),
                TensorDict(x=torch.zeros(3, dtype=torch.float64)),
            ]
        )
        with set_printoptions(compact=True):
            r = repr(td)
        assert (
            r
            == "LazyStackedTensorDict(batch_size=[2], stack_dim=0, fields={x: f64[3]})"
        )

    def test_compact_lazy_stack_exclusive_fields(self):
        td = LazyStackedTensorDict.lazy_stack(
            [
                TensorDict(a=torch.zeros(3), b=torch.zeros(())),
                TensorDict(a=torch.zeros(3), c=TensorDict(d=torch.zeros(2))),
            ]
        )
        with set_printoptions(compact=True):
            r = repr(td)
        assert r == (
            "LazyStackedTensorDict(batch_size=[2], stack_dim=0, fields={\n"
            "    a: f32[3]}, exclusive_fields={0: {b: f32[]}, 1: {c: {d: f32[2]}}})"
        )

    def test_compact_lazy_stack_calls_no_stack(self, monkeypatch):
        td = LazyStackedTensorDict.lazy_stack(
            [
                TensorDict(
                    x=torch.zeros(10, 10),
                    sub=TensorDict(y=torch.zeros(10, 3), batch_size=[10]),
                    mesh=Mesh(
                        points=torch.zeros(10, 3),
                        cells=torch.zeros(10, 4, dtype=torch.long),
                        batch_size=[10],
                    ),
                    name="a",
                    batch_size=[10],
                )
                for _ in range(4)
            ]
        )
        stack = torch.stack
        calls = []

        def counting_stack(*args, **kwargs):
            calls.append(args)
            return stack(*args, **kwargs)

        monkeypatch.setattr(torch, "stack", counting_stack)
        with set_printoptions(compact=True):
            r = repr(td)
        assert not calls
        assert r == (
            "LazyStackedTensorDict(batch_size=[4,10], stack_dim=0, fields={\n"
            "    mesh: Mesh{cells: i64[4], points: f32[3]},\n"
            "    name: str,\n"
            "    sub: {y: f32[3]},\n"
            "    x: f32[10]})"
        )
        # Since #2799 the default repr of a lazy stack does not stack the
        # leaves either.
        repr(td)
        assert not calls

    @pytest.mark.parametrize("ragged", [False, True])
    def test_compact_typed_lazy_stack_calls_no_stack(self, monkeypatch, ragged):
        # a TypedTensorDict over a lazy stack reads the metadata of its members
        td = _TypedMesh.from_tensordict(
            LazyStackedTensorDict.lazy_stack(
                [
                    TensorDict(
                        points=torch.zeros(3 + ragged * i),
                        cells=torch.zeros(4, dtype=torch.long),
                    )
                    for i in range(2)
                ]
            )
        )
        stack = torch.stack
        calls = []

        def counting_stack(*args, **kwargs):
            calls.append(args)
            return stack(*args, **kwargs)

        monkeypatch.setattr(torch, "stack", counting_stack)
        with set_printoptions(compact=True):
            r = repr(td)
        assert not calls
        points = "f32[-1]" if ragged else "f32[3]"
        assert r == (
            f"_TypedMesh(batch_size=[2], fields={{cells: i64[4], points: {points}}})"
        )

    def test_compact_non_tensor(self):
        td = _mesh_sample()
        td["name"] = "mesh-A"
        with set_printoptions(compact=True):
            r = repr(td)
        assert r == (
            "TensorDict(batch_size=[1000], device=cpu, fields={\n"
            "    cells: i64[4],\n"
            "    data: {mask: bool[], pressure: f32[]},\n"
            "    name: str,\n"
            "    points: f32[3]})"
        )

    def test_compact_tensorclass(self):
        with set_printoptions(compact=True):
            r = repr(_mesh())
        assert r == (
            "Mesh(batch_size=[1000], device=cpu, fields={cells: i64[4], points: f32[3]})"
        )

    def test_compact_nested_tensorclass(self):
        td = TensorDict(
            mesh=_mesh(), pressure=torch.zeros(1000), batch_size=[1000], device="cpu"
        )
        with set_printoptions(compact=True):
            r = repr(td)
        assert r == (
            "TensorDict(batch_size=[1000], device=cpu, fields={\n"
            "    mesh: Mesh{cells: i64[4], points: f32[3]},\n"
            "    pressure: f32[]})"
        )

    def test_compact_image_batch(self):
        td = TensorDict(
            image=torch.zeros(32, 3, 64, 64),
            label=torch.zeros(32, dtype=torch.long),
            batch_size=[32],
        )
        with set_printoptions(compact=True):
            r = repr(td)
        assert (
            r
            == "TensorDict(batch_size=[32], fields={image: f32[3,64,64], label: i64[]})"
        )

    def test_compact_2d_batch(self):
        td = TensorDict(
            action=torch.zeros(8, 100, 6),
            done=torch.zeros(8, 100, 1, dtype=torch.bool),
            obs=torch.zeros(8, 100, 17),
            next=TensorDict(obs=torch.zeros(8, 100, 17), reward=torch.zeros(8, 100, 1)),
            batch_size=[8, 100],
        )
        with set_printoptions(compact=True):
            r = repr(td)
        assert r == (
            "TensorDict(batch_size=[8,100], fields={\n"
            "    action: f32[6],\n"
            "    done: bool[1],\n"
            "    next: {obs: f32[17], reward: f32[1]},\n"
            "    obs: f32[17]})"
        )

    def test_compact_nested_extra_batch_dims(self):
        td = TensorDict(
            sub=TensorDict(y=torch.zeros(1000, 5, 3), batch_size=[1000, 5]),
            x=torch.zeros(1000, 2),
            batch_size=[1000],
        )
        with set_printoptions(compact=True):
            r = repr(td)
        assert (
            r
            == "TensorDict(batch_size=[1000], fields={sub: [5]{y: f32[3]}, x: f32[2]})"
        )

    def test_compact_mixed_devices(self):
        # with device=None, a leaf that is not on the CPU shows its device
        td = TensorDict(
            points=torch.zeros(1000, 3, device="meta"),
            cells=torch.zeros(1000, 4, dtype=torch.long),
            batch_size=[1000],
        )
        with set_printoptions(compact=True):
            r = repr(td)
        assert r == (
            "TensorDict(batch_size=[1000], fields={cells: i64[4], points: f32[3]@meta})"
        )

    def test_compact_width(self):
        # one line while it fits in 80 columns, otherwise one entry per line
        td = TensorDict({key: torch.zeros(2, 3) for key in "abc"}, batch_size=[2])
        with set_printoptions(compact=True):
            r = repr(td)
        assert "\n" not in r
        assert len(r) <= 80
        keys = ["a", "b", "c", "d", "e", "f"]
        td = TensorDict({key * 8: torch.zeros(2, 3) for key in keys}, batch_size=[2])
        with set_printoptions(compact=True):
            r = repr(td)
        lines = r.split("\n")
        assert len(lines) == len(keys) + 1
        assert all(len(line) <= 80 for line in lines)
        assert lines[1] == "    aaaaaaaa: f32[3],"
        # a nested entry that does not fit is broken in the same way
        td = TensorDict(
            sub=TensorDict(
                {key * 12: torch.zeros(2, 3) for key in keys}, batch_size=[2]
            ),
            batch_size=[2],
        )
        with set_printoptions(compact=True):
            r = repr(td)
        lines = r.split("\n")
        assert lines[1] == "    sub: {"
        assert lines[2] == "        aaaaaaaaaaaa: f32[3],"
        assert lines[-1] == "        ffffffffffff: f32[3]}})"

    def test_compact_width_exclusive_fields(self):
        # the fields part leaves room for the exclusive_fields that follow it
        for n in (1, 30, 40):
            td = LazyStackedTensorDict.lazy_stack(
                [
                    TensorDict(a=torch.zeros(3), x=torch.zeros(1)),
                    TensorDict(a=torch.zeros(3), **{"y" * n: torch.zeros(1)}),
                ]
            )
            with set_printoptions(compact=True):
                r = repr(td)
            assert all(len(line) <= 80 for line in r.split("\n")), r

    def test_compact_shape_only(self):
        td = TensorDict(
            image=torch.zeros(32, 3, 64, 64),
            label=torch.zeros(32, dtype=torch.long),
            batch_size=[32],
            device="cpu",
        )
        with set_printoptions(compact=True, show_dtype=False):
            r = repr(td)
        assert r == (
            "TensorDict(batch_size=[32], device=cpu, fields={image: [3,64,64], label: []})"
        )
        with set_printoptions(compact=True, verbose=False, show_batch_size=False):
            r = repr(td)
        assert r == "TensorDict(fields={image: [3,64,64], label: []})"

    def test_compact_ignores_entry_options(self):
        # plain would read the values: compact mode ignores it
        td = TensorDict(a=torch.ones(10, 2), batch_size=[10])
        with set_printoptions(
            compact=True, plain=True, show_grad=True, show_storage_size=True
        ):
            r = repr(td)
        assert r == "TensorDict(batch_size=[10], fields={a: f32[2]})"

    @pytest.mark.parametrize(
        "option",
        [
            "show_is_shared",
            "show_shape",
            "show_field_device",
            "show_field_is_shared",
            "show_grad",
            "show_is_contiguous",
            "show_is_view",
            "show_storage_size",
            "plain",
        ],
    )
    def test_compact_detailed_only_option_turns_compact_off(self, option):
        # #3240: an option that only the detailed form uses turns the compact
        # form off when compact is not passed; an explicit compact=True keeps it.
        assert get_printoptions()["compact"] is True
        value = not get_printoptions()[option]
        td = TensorDict(a=torch.zeros(3, requires_grad=True), batch_size=[3])
        with set_printoptions(**{option: value}):
            assert get_printoptions()["compact"] is False
            r = repr(td)
        assert r.startswith("TensorDict(\n    fields={")
        with set_printoptions(compact=False, **{option: value}):
            assert r == repr(td)
        with set_printoptions(compact=True, **{option: value}):
            assert repr(td) == "TensorDict(batch_size=[3], fields={a: f32[]})"
        assert get_printoptions()["compact"] is True

    def test_compact_show_grad_shows_requires_grad(self):
        # #3240: set_printoptions(show_grad=True) alone used to print the
        # compact form, without requires_grad
        td = TensorDict(a=torch.zeros(3, requires_grad=True), batch_size=[3])
        with set_printoptions(show_grad=True):
            assert "requires_grad=True" in repr(td)
        # verbose=False hides options of the detailed form but keeps compact
        with set_printoptions(verbose=False):
            assert repr(td) == "TensorDict(batch_size=[3], fields={a: []})"

        # as a decorator too, which clones the options on each call
        @set_printoptions(verbose=False)
        def decorated():
            return repr(td)

        assert decorated() == "TensorDict(batch_size=[3], fields={a: []})"

    def test_compact_sort_keys(self):
        td = TensorDict(c=torch.zeros(2), a=torch.zeros(2), b=torch.zeros(2))
        with set_printoptions(compact=True, sort_keys="insertion"):
            r = repr(td)
        assert (
            r == "TensorDict(batch_size=[], fields={c: f32[2], a: f32[2], b: f32[2]})"
        )

    def test_compact_nested_own_repr(self):
        # a nested store shows its class name and reads none of its leaves
        td = _nested_store()
        with set_printoptions(compact=True):
            r = repr(td)
        assert not _CountingStore.reads
        assert r == "TensorDict(batch_size=[3], fields={s: _CountingStore, z: f32[]})"

    def test_compact_non_tensor_stack_keeps_repr(self):
        from tensordict import NonTensorStack

        stack = NonTensorStack("a", "b")
        expected = repr(stack)
        with set_printoptions(compact=True):
            assert repr(stack) == expected

    def test_compact_dtype_codes(self):
        td = TensorDict(
            a=torch.zeros(1, dtype=torch.bfloat16),
            b=torch.zeros(1, dtype=torch.uint8),
            c=torch.zeros(1, dtype=torch.complex64),
            d=torch.zeros(1, dtype=torch.float16),
        )
        with set_printoptions(compact=True):
            r = repr(td)
        assert r == (
            "TensorDict(batch_size=[], fields={a: bf16[1], b: u8[1], c: c64[1], d: f16[1]})"
        )


class TestHtmlRepr:
    """Tests for ``_repr_html_``, the display of notebooks."""

    @staticmethod
    def _items(out):
        # the text of each entry, without the tags
        return [
            re.sub("<[^>]+>", "", item.split("</li>")[0].split("<ul")[0])
            for item in out.split("<li>")[1:]
        ]

    def test_html_sample(self):
        td = _mesh_sample()
        td.refine_names("points")
        out = td._repr_html_()
        assert out.startswith("<div ")
        assert "<b>TensorDict</b>" in out
        assert "batch_size=[1000], names=[points], device=cpu" in out
        assert self._items(out) == [
            "cells: i64[4]",
            "data: 2 entries",
            "mask: bool[]",
            "pressure: f32[]",
            "points: f32[3]",
        ]
        assert out.count("<details open>") == 2

    def test_html_escapes(self):
        td = TensorDict({"<b>&": torch.zeros(()), "x": "<script>"})
        td["<i>"] = TensorDict({"a&b": torch.zeros(())})
        out = td._repr_html_()
        assert "<b>&" not in out and "<i>" not in out and "a&b" not in out
        assert "&lt;b&gt;&amp;: " in out and "&lt;i&gt;: " in out
        assert "a&amp;b: " in out

    def test_html_lazy_stack(self, monkeypatch):
        td = LazyStackedTensorDict.lazy_stack(
            [
                TensorDict(points=torch.zeros(10, 3), label=torch.tensor(0)),
                TensorDict(points=torch.zeros(12, 3), extra=torch.zeros(2)),
            ]
        )
        stack = torch.stack
        calls = []

        def counting_stack(*args, **kwargs):
            calls.append(args)
            return stack(*args, **kwargs)

        monkeypatch.setattr(torch, "stack", counting_stack)
        out = td._repr_html_()
        assert not calls
        assert "<b>LazyStackedTensorDict</b>" in out
        assert "batch_size=[2], stack_dim=0" in out
        assert ">exclusive_fields</i></summary>" in out
        assert self._items(out) == [
            "points: f32[-1,3]",
            "0: 1 entry",
            "label: i64[]",
            "1: 1 entry",
            "extra: f32[2]",
        ]

    def test_html_tensorclass(self):
        td = TensorDict(mesh=_mesh(), batch_size=[1000])
        assert self._items(td._repr_html_()) == [
            "mesh: Mesh 2 entries",
            "cells: i64[4]",
            "points: f32[3]",
        ]
        out = _mesh()._repr_html_()
        assert "<b>Mesh</b>" in out
        assert self._items(out) == ["cells: i64[4]", "points: f32[3]"]

    @pytest.mark.parametrize("flag", ["tensor_only", "shadow"])
    def test_html_tensorclass_flags(self, flag):
        @tensorclass(**{flag: True})
        class Points:
            x: torch.Tensor

        p = Points(x=torch.zeros(2, 3), batch_size=[2])
        assert self._items(p._repr_html_()) == ["x: f32[3]"]
        assert [field.name for field in dataclasses.fields(Points)] == ["x"]

    def test_html_tensorclass_own(self):
        @tensorclass
        class Points:
            x: torch.Tensor

            def _repr_html_(self):
                return "own"

        assert Points(x=torch.zeros(2))._repr_html_() == "own"

    def test_html_tensorclass_own_repr(self):
        @tensorclass
        class Points:
            x: torch.Tensor

            def __repr__(self):
                return "<own>"

        @tensorclass
        class SubPoints(Points):
            y: torch.Tensor

        assert Points(x=torch.zeros(2))._repr_html_() == "<pre>&lt;own&gt;</pre>"
        out = SubPoints(x=torch.zeros(2), y=torch.zeros(2))._repr_html_()
        assert out == "<pre>&lt;own&gt;</pre>"

    def test_html_nested_own_repr(self):
        td = _nested_store()
        out = td._repr_html_()
        assert not _CountingStore.reads
        assert self._items(out) == ["s: _CountingStore", "z: f32[]"]
        assert td.get("s")._repr_html_().startswith("<pre>_CountingStore(")

    def test_html_meta_device(self):
        # meta tensors have no data: the display reads none
        td = TensorDict(
            a=torch.empty(1000, 3, device="meta"),
            sub=TensorDict(b=torch.empty(1000, device="meta"), batch_size=[1000]),
            batch_size=[1000],
        )
        assert self._items(td._repr_html_()) == [
            "a: f32[3]@meta",
            "sub: 1 entry",
            "b: f32[]@meta",
        ]

    def test_html_truncates(self, monkeypatch):
        import tensordict.utils

        monkeypatch.setattr(tensordict.utils, "_HTML_MAX_ENTRIES", 3)
        td = TensorDict({f"k{i}": torch.zeros(()) for i in range(5)})
        td["z"] = TensorDict({f"k{i}": torch.zeros(()) for i in range(4)})
        assert self._items(td._repr_html_()) == [
            "k0: f32[]",
            "k1: f32[]",
            "k2: f32[]",
            "... and 3 more",
        ]
        td = TensorDict(a=torch.zeros(()))
        td["b"] = TensorDict({f"k{i}": torch.zeros(()) for i in range(4)})
        assert self._items(td._repr_html_()) == [
            "a: f32[]",
            "b: 4 entries",
            "k0: f32[]",
            "... and 3 more",
        ]

    def test_html_fallback(self, monkeypatch):
        import tensordict.utils

        def fail(obj):
            raise RuntimeError

        td = TensorDict({"<a>": torch.zeros(())})
        expected = f"<pre>{html.escape(repr(td))}</pre>"
        monkeypatch.setattr(tensordict.utils, "_compact_parts", fail)
        assert td._repr_html_() == expected

    def test_html_keeps_own_repr(self):
        from tensordict import NonTensorData, NonTensorStack

        for obj in (NonTensorStack("a", "b"), NonTensorData("<a>")):
            assert obj._repr_html_() == f"<pre>{html.escape(repr(obj))}</pre>"

    def test_html_option(self):
        td = _mesh_sample()
        assert get_printoptions()["html"] is True
        with set_printoptions(html=False):
            assert td._repr_html_() is None
            assert _mesh()._repr_html_() is None
        assert td._repr_html_() is not None

    def test_html_follows_options(self):
        with set_printoptions(show_batch_size=False, show_device=False):
            out = _mesh_sample()._repr_html_()
        assert "batch_size" not in out and "device" not in out


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
