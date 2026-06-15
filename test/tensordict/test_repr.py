# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys

import pytest
import torch
from packaging import version
from tensordict import (
    get_printoptions,
    LazyStackedTensorDict,
    set_printoptions,
    tensorclass,
    TensorDict,
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
