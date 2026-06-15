# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import os
import platform
import re
import sys
import threading
import warnings
import weakref
from collections import UserDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensordict.base as tensordict_base
import torch
from packaging import version
from tensordict import (
    lazy_stack,
    LazyStackedTensorDict,
    set_capture_non_tensor_stack,
    tensorclass,
    TensorClass,
    TensorDict,
)
from tensordict._td import is_tensor_collection
from tensordict._torch_func import _stack as stack_td
from tensordict.base import _NESTED_TENSORS_AS_LISTS, TensorDictBase
from tensordict.functional import dense_stack_tds, merge_tensordicts, pad, pad_sequence
from tensordict.memmap import MemoryMappedTensor
from tensordict.tensorclass import NonTensorData, NonTensorStack
from tensordict.utils import (
    _getitem_batch_size,
    _LOCK_ERROR,
    assert_allclose_td,
    convert_ellipsis_to_idx,
    is_non_tensor,
    is_tensorclass,
    set_lazy_legacy,
    set_list_to_stack,
)
from torch import nn
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.nn.parameter import UninitializedTensorMixin

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.tensordict.test._utils_internal import (
        decompose,
        get_available_devices,
        is_npu_available,
        TestTensorDictsBase,
    )
else:
    IS_FB = False
    from _utils_internal import (
        decompose,
        get_available_devices,
        is_npu_available,
        TestTensorDictsBase,
    )


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


@pytest.fixture
def device_fixture():
    device = torch.get_default_device()
    if torch.cuda.is_available():
        torch.set_default_device(torch.device("cuda:0"))
    elif is_npu_available():
        torch.set_default_device(torch.device("npu:0"))
    # elif torch.backends.mps.is_available():
    #     torch.set_default_device(torch.device("mps:0"))
    yield
    torch.set_default_device(device)


class TestGeneric:
    # Generic, type-insensitive teests

    def test_batchsize_reset(self):
        td = TensorDict(
            {"a": torch.randn(3, 4, 5, 6), "b": torch.randn(3, 4, 5)}, batch_size=[3, 4]
        )
        # smoke-test
        td.batch_size = torch.Size([3])

        # test with list
        td.batch_size = [3]

        # test with tuple
        td.batch_size = (3,)

        # incompatible size
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "the Tensor a has shape torch.Size([3, 4, 5, 6]) which is incompatible with the batch-size torch.Size([3, 5])."
            ),
        ):
            td.batch_size = [3, 5]

        # test set
        td.set("c", torch.randn(3))

        # test index
        td[torch.tensor([1, 2])]
        td[:]
        td[[1, 2]]
        with pytest.raises(
            IndexError,
            match=re.escape("too many indices for tensor of dimension 1"),
        ):
            td[:, 0]

        # test a greater batch_size
        td = TensorDict(
            {"a": torch.randn(3, 4, 5, 6), "b": torch.randn(3, 4, 5)}, batch_size=[3, 4]
        )
        td.batch_size = torch.Size([3, 4, 5])

        td.set("c", torch.randn(3, 4, 5, 6))
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "batch dimension mismatch, got self.batch_size=torch.Size([3, 4, 5]) and value.shape=torch.Size([3, 4, 2])"
            ),
        ):
            td.set("d", torch.randn(3, 4, 2))

        # test that lazy tds return an exception
        td_stack = LazyStackedTensorDict.lazy_stack(
            [TensorDict({"a": torch.randn(3)}, [3]) for _ in range(2)]
        )
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Modifying the batch size of a lazy representation of a tensordict is not permitted. Consider instantiating the tensordict first by calling `td = td.to_tensordict()` before resetting the batch size."
            ),
        ):
            td_stack.batch_size = [2]
        td_stack.to_tensordict(retain_none=True).batch_size = [2]

        td = TensorDict({"a": torch.randn(3, 4)}, [3, 4])
        subtd = td._get_sub_tensordict((slice(None), torch.tensor([1, 2])))
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Modifying the batch size of a lazy representation of a tensordict is not permitted. Consider instantiating the tensordict first by calling `td = td.to_tensordict()` before resetting the batch size."
            ),
        ):
            subtd.batch_size = [3, 2]
        subtd.to_tensordict(retain_none=True).batch_size = [3, 2]

        td = TensorDict({"a": torch.randn(3, 4)}, [3, 4])
        with set_lazy_legacy(True):
            td_u = td.unsqueeze(0)
            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    "Modifying the batch size of a lazy representation of a tensordict is not permitted. Consider instantiating the tensordict first by calling `td = td.to_tensordict()` before resetting the batch size."
                ),
            ):
                td_u.batch_size = [1]
            td_u.to_tensordict(retain_none=True).batch_size = [1]

    @pytest.mark.parametrize("count_duplicates", [False, True])
    def test_bytes(self, count_duplicates, device_fixture):
        tensor = torch.zeros(3)
        tensor_with_grad = torch.ones(3, requires_grad=True)
        (tensor_with_grad + 1).sum().backward()
        v = torch.ones(3) * 2  # 12 bytes
        offsets = torch.tensor([0, 1, 3])  # 24 bytes
        lengths = torch.tensor([1, 2])  # 16 bytes
        njt = torch.nested.nested_tensor_from_jagged(
            v, offsets, lengths=lengths
        )  # 52 bytes
        tricky = torch.nested.nested_tensor_from_jagged(
            tensor, offsets, lengths=lengths
        )  # 52 bytes or 0
        td = TensorDict(
            tensor=tensor,  # 3 * 4 = 12 bytes
            tensor_with_grad=tensor_with_grad,  # 3 * 4 * 2 = 24 bytes
            njt=njt,  # 32
            tricky=tricky,  # 32 or 0
        )
        if count_duplicates:
            assert td.bytes(count_duplicates=count_duplicates) == 12 + 24 + 52 + 52
        else:
            assert td.bytes(count_duplicates=count_duplicates) == 12 + 24 + 52 + 0

    def test_depth(self):
        td = TensorDict({"a": {"b": {"c": {"d": 0}, "e": 0}, "f": 0}, "g": 0}).lock_()
        assert td.depth == 3
        with td.unlock_():
            del td["a", "b", "c", "d"]
        assert td.depth == 2
        with td.unlock_():
            del td["a", "b", "c"]
            del td["a", "b", "e"]
        assert td.depth == 1
        with td.unlock_():
            del td["a", "b"]
            del td["a", "f"]
        assert td.depth == 0

    @pytest.mark.parametrize("device", get_available_devices())
    def test_cat_td(self, device):
        torch.manual_seed(1)
        d = {
            "key1": torch.randn(4, 5, 6, device=device),
            "key2": torch.randn(4, 5, 10, device=device),
            "key3": {"key4": torch.randn(4, 5, 10, device=device)},
        }
        td1 = TensorDict(batch_size=(4, 5), source=d, device=device)
        d = {
            "key1": torch.randn(4, 10, 6, device=device),
            "key2": torch.randn(4, 10, 10, device=device),
            "key3": {"key4": torch.randn(4, 10, 10, device=device)},
        }
        td2 = TensorDict(batch_size=(4, 10), source=d, device=device)

        td_cat = torch.cat([td1, td2], 1)
        assert td_cat.batch_size == torch.Size([4, 15])
        d = {
            "key1": torch.zeros(4, 15, 6, device=device),
            "key2": torch.zeros(4, 15, 10, device=device),
            "key3": {"key4": torch.zeros(4, 15, 10, device=device)},
        }
        td_out = TensorDict(batch_size=(4, 15), source=d, device=device)
        data_ptr_set_before = {val.data_ptr() for val in decompose(td_out)}
        torch.cat([td1, td2], 1, out=td_out)
        data_ptr_set_after = {val.data_ptr() for val in decompose(td_out)}
        assert data_ptr_set_before == data_ptr_set_after
        assert td_out.batch_size == torch.Size([4, 15])
        assert (td_out["key1"] != 0).all()
        assert (td_out["key2"] != 0).all()
        assert (td_out["key3", "key4"] != 0).all()

    def test_cat_from_tensordict(self):
        td = TensorDict(
            {"a": torch.zeros(3, 4), "b": {"c": torch.ones(3, 4)}}, batch_size=[3, 4]
        )
        tensor = td.cat_from_tensordict(dim=1)
        assert tensor.shape == (3, 8)
        assert (tensor[:, :4] == 0).all()
        assert (tensor[:, 4:] == 1).all()

    def test_chunk_nested_tensor(self):
        # Create two sequences of different lengths with known values
        a_seq = torch.tensor([1, 2, 3])  # shorter sequence
        b_seq = torch.tensor([4, 5, 6, 7])  # longer sequence

        # Create a nested tensor from these sequences
        rmpad_seq = torch.nested.as_nested_tensor([a_seq, b_seq], layout=torch.jagged)

        # Verify that chunking the nested tensor directly works
        chunks = rmpad_seq.chunk(2)  # works fine
        assert len(chunks) == 2

        # Create a TensorDict with the nested tensor
        rmpad_batch = TensorDict({"input_ids": rmpad_seq}, batch_size=[2])

        # Verify that chunking the TensorDict works
        td_chunks = rmpad_batch.chunk(2)  # should work now
        assert len(td_chunks) == 2

    @pytest.mark.parametrize("recurse", [True, False])
    def test_clone_empty(self, recurse):
        td = TensorDict()
        assert td.clone(recurse=recurse) is not None
        td = TensorDict(device="cpu")
        assert td.clone(recurse=recurse) is not None
        td = TensorDict(batch_size=[2])
        assert td.clone(recurse=recurse) is not None
        td = TensorDict(device="cpu", batch_size=[2])
        assert td.clone(recurse=recurse) is not None

    @pytest.mark.skipif(
        is_npu_available(), reason="torch.nested_tensor is not adapted on NPU currently"
    )
    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize("device", [None, *get_available_devices()])
    @pytest.mark.parametrize("num_threads", [0, 1, 2])
    @pytest.mark.parametrize("use_file", [False, True])
    @pytest.mark.parametrize(
        "nested,hetdtype",
        (
            [[False, False], [False, True]]
            if torch.__version__ < "2.4"
            else [[False, False], [False, True], ["NJT", True]]
        ),
    )
    def test_consolidate(self, device, use_file, tmpdir, num_threads, nested, hetdtype):
        if not nested:
            a = torch.zeros((2,))
            c = torch.ones((2,), dtype=torch.float16 if hetdtype else torch.float32)
            g = torch.full(
                (2, 3), 2, dtype=torch.float64 if hetdtype else torch.float32
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                layout = torch.jagged
                a = torch.nested.nested_tensor(
                    [
                        torch.zeros((1,), device=device),
                        torch.zeros((2,), device=device),
                    ],
                    layout=layout,
                )
                c = torch.nested.nested_tensor(
                    [
                        torch.ones(
                            (1,),
                            device=device,
                            dtype=torch.float16 if hetdtype else torch.float32,
                        ),
                        torch.ones(
                            (2,),
                            device=device,
                            dtype=torch.float16 if hetdtype else torch.float32,
                        ),
                    ],
                    layout=layout,
                )
                g0 = torch.full(
                    (1, 3), 2, dtype=torch.float64 if hetdtype else torch.float32
                )
                g1 = torch.full(
                    (2, 3), 2, dtype=torch.float64 if hetdtype else torch.float32
                )
                g = torch.nested.nested_tensor([g0, g1], layout=layout)

        td = TensorDict(
            {
                "a": a,
                "b": {"c": c},
                "d": "a string!",
                ("e", "f", "g"): g,
            },
            device=device,
            batch_size=[2],
        )
        if not use_file:
            td_c = td.consolidate(num_threads=num_threads, metadata=bool(nested))
            assert td_c.device == device
        else:
            filename = Path(tmpdir) / "file.mmap"
            td_c = td.consolidate(filename=filename, num_threads=num_threads)
            assert td_c.device == torch.device("cpu")
            if not nested:
                assert (TensorDict.from_consolidated(filename) == td_c).all()
            else:
                assert all(
                    (_td0 == _td1).all()
                    for (_td0, _td1) in zip(
                        TensorDict.from_consolidated(filename).unbind(0), td_c.unbind(0)
                    )
                )

        # TODO: This does NOT work because of the float16 which screws up the offsets.
        #  to replicate this issue, comment out the try/except within consolidate
        # values = td_c.values(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)
        # data_ptr = torch.tensor([t.untyped_storage().data_ptr() for t in values])
        # assert data_ptr.unique().numel() == 1
        assert hasattr(td_c, "_consolidated")
        if not nested:
            assert (td.to(td_c.device) == td_c).all(), td_c.to_dict()
        else:
            assert all(
                (_td == _td_c).all()
                for (_td, _td_c) in zip(td.to(td_c.device).unbind(0), td_c.unbind(0))
            ), td_c.to_dict()

        assert td_c["d"] == "a string!"
        storage = td_c._consolidated["storage"]
        storage *= 0
        if not nested:
            assert (td.to(td_c.device) != td_c).any(), td_c.to_dict()
        elif nested == "NJT":
            assert (td_c["e", "f", "g"].offsets() == 0).all()
            assert (td_c["e", "f", "g"].values() == 0).all()
        else:
            assert (td_c["a"].values() == 0).all()
            assert (td_c["e", "f", "g"].values() == 0).all()

        filename = Path(tmpdir) / "file.pkl"
        if not nested:
            torch.save(td, filename)
            assert (
                td == torch.load(filename, weights_only=False)
            ).all(), td_c.to_dict()
        else:
            pass
            # wait for https://github.com/pytorch/pytorch/issues/129366 to be resolved
            # assert all(
            #     (_td == _td_c).all()
            #     for (_td, _td_c) in zip(td.unbind(0), torch.load(filename).unbind(0))
            # ), td_c.to_dict()

        td_c = td.consolidate()
        torch.save(td_c, filename)
        if not nested:
            assert (
                td == torch.load(filename, weights_only=False)
            ).all(), td_c.to_dict()
        else:
            assert all(
                (_td == _td_c).all()
                for (_td, _td_c) in zip(
                    td.unbind(0), torch.load(filename, weights_only=False).unbind(0)
                )
            ), td_c.to_dict()

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
        assert td_c_device["d"] == "a string!"
        assert len(dataptrs) == 1

    def test_consolidated_locking_behavior(self):
        """Test that consolidated TensorDicts are automatically locked and unlock properly."""
        td = TensorDict(
            {
                "a": torch.zeros(1, 1),
                "b": {"c": torch.zeros(1, 1)},
                "d": "a string!",
            },
            device="cpu",
            batch_size=[1],
        )

        # Test that consolidation automatically locks the TensorDict
        td_consolidated = td.consolidate()
        assert td_consolidated.is_consolidated()
        assert td_consolidated.is_locked

        # Test that modifications are blocked when locked
        with pytest.raises(RuntimeError, match=re.escape(_LOCK_ERROR)):
            td_consolidated["new_key"] = torch.ones(1, 1)

        with pytest.raises(RuntimeError, match=re.escape(_LOCK_ERROR)):
            td_consolidated.set("new_key", torch.ones(1, 1))

        with pytest.raises(RuntimeError, match=re.escape(_LOCK_ERROR)):
            td_consolidated.update({"new_key": torch.ones(1, 1)})

        # Test that in-place modifications are still allowed
        td_consolidated.set("a", torch.ones(1, 1), inplace=True)
        assert (td_consolidated["a"] == 1).all()

        # Test that unlocking removes consolidated metadata
        td_consolidated.unlock_()
        assert not td_consolidated.is_locked
        assert not td_consolidated.is_consolidated()

        # Test that modifications are now allowed
        td_consolidated["new_key"] = torch.ones(1, 1)
        assert "new_key" in td_consolidated

        # Test that device transfer maintains locking
        td_consolidated = td.consolidate()
        td_consolidated_device = td_consolidated.to("cpu")  # Should stay on CPU
        assert td_consolidated_device.is_consolidated()
        assert td_consolidated_device.is_locked

        # Test that unlocking after device transfer removes consolidated metadata
        td_consolidated_device.unlock_()
        assert not td_consolidated_device.is_consolidated()
        assert not td_consolidated_device.is_locked

    def test_consolidated_locking_with_context_manager(self):
        """Test consolidated TensorDict locking behavior with context managers."""
        td = TensorDict(
            {
                "a": torch.zeros(1, 1),
                "b": {"c": torch.zeros(1, 1)},
            },
            device="cpu",
            batch_size=[1],
        )

        td_consolidated = td.consolidate()

        # Test that we can temporarily unlock for modifications
        with td_consolidated.unlock_():
            assert not td_consolidated.is_locked
            assert not td_consolidated.is_consolidated()
            td_consolidated["new_key"] = torch.ones(1, 1)

        # After context manager, should be locked again but not consolidated
        assert td_consolidated.is_locked
        assert not td_consolidated.is_consolidated()

        # Test that we can't modify after context manager
        with pytest.raises(RuntimeError, match=re.escape(_LOCK_ERROR)):
            td_consolidated["another_key"] = torch.ones(1, 1)

    def test_consolidated_locking_nested(self):
        """Test that nested TensorDicts in consolidated structures are properly locked."""
        td = TensorDict(
            {
                "a": torch.zeros(1, 1),
                "b": TensorDict({"c": torch.zeros(1, 1)}, batch_size=[1]),
            },
            device="cpu",
            batch_size=[1],
        )

        td_consolidated = td.consolidate()
        assert td_consolidated.is_consolidated()
        assert td_consolidated.is_locked
        assert td_consolidated["b"].is_locked

        # Test that nested modifications are blocked
        with pytest.raises(RuntimeError, match=re.escape(_LOCK_ERROR)):
            td_consolidated["b"]["new_key"] = torch.ones(1, 1)

        # Test that unlocking propagates to nested structures
        td_consolidated.unlock_()
        assert not td_consolidated.is_consolidated()
        assert not td_consolidated.is_locked
        assert not td_consolidated["b"].is_locked

        # Test that nested modifications are now allowed
        td_consolidated["b"]["new_key"] = torch.ones(1, 1)
        assert "new_key" in td_consolidated["b"]

    def test_consolidated_locking_device_transfer(self):
        """Test that device transfers maintain consolidated locking behavior."""
        td = TensorDict(
            {
                "a": torch.zeros(1, 1),
                "b": {"c": torch.zeros(1, 1)},
            },
            device="cpu",
            batch_size=[1],
        )

        td_consolidated = td.consolidate()

        # Test device transfer maintains consolidated state and locking
        td_transferred = td_consolidated.to("cpu")  # Should stay on CPU
        assert td_transferred.is_consolidated()
        assert td_transferred.is_locked

        # Test that modifications are blocked
        with pytest.raises(RuntimeError, match=re.escape(_LOCK_ERROR)):
            td_transferred["new_key"] = torch.ones(1, 1)

        # Test that unlocking removes consolidated metadata
        td_transferred.unlock_()
        assert not td_transferred.is_consolidated()
        assert not td_transferred.is_locked

        # Test that modifications are now allowed
        td_transferred["new_key"] = torch.ones(1, 1)
        assert "new_key" in td_transferred

    def test_consolidated_locking_issue_1406_reproduction(self):
        """Test reproduction of the issue described in #1406."""
        # This test reproduces the scenario from the GitHub issue
        a = TensorDict({"a": torch.zeros(1, 1)}, device="cpu", batch_size=[1])
        a_consolidated = a.consolidate()

        # Add a new key to the original (non-consolidated) TensorDict
        a["b"] = torch.ones(1, 1)

        # Try to add the same key to the consolidated TensorDict - should fail
        with pytest.raises(RuntimeError, match=re.escape(_LOCK_ERROR)):
            a_consolidated["b"] = torch.ones(1, 1)

        # Unlock the consolidated TensorDict to allow modifications
        a_consolidated.unlock_()
        a_consolidated["b"] = torch.ones(1, 1)

        # Now both should have the same content
        assert (a["a"] == a_consolidated["a"]).all()
        assert (a["b"] == a_consolidated["b"]).all()

        # The consolidated TensorDict should no longer be consolidated
        assert not a_consolidated.is_consolidated()

    def test_construct_from_kwargs(self):
        with pytest.raises(ValueError, match="not both"):
            TensorDict(a=1, source={"b": 2})
        td = TensorDict(a=1, b=1, batch_size=[])
        assert td.batch_size == ()

    @pytest.mark.parametrize(
        "ellipsis_index, expectation",
        [
            ((..., 0, ...), pytest.raises(RuntimeError)),
            ((0, ..., 0, ...), pytest.raises(RuntimeError)),
        ],
    )
    def test_convert_ellipsis_to_idx_invalid(self, ellipsis_index, expectation):
        torch.manual_seed(1)
        batch_size = [3, 4, 5, 6, 7]

        with expectation:
            _ = convert_ellipsis_to_idx(ellipsis_index, batch_size)

    @pytest.mark.parametrize(
        "ellipsis_index, expected_index",
        [
            (..., (slice(None), slice(None), slice(None), slice(None), slice(None))),
            ((0, ..., 0), (0, slice(None), slice(None), slice(None), 0)),
            ((..., 0), (slice(None), slice(None), slice(None), slice(None), 0)),
            ((0, ...), (0, slice(None), slice(None), slice(None), slice(None))),
            (
                (slice(1, 2), ...),
                (slice(1, 2), slice(None), slice(None), slice(None), slice(None)),
            ),
        ],
    )
    def test_convert_ellipsis_to_idx_valid(self, ellipsis_index, expected_index):
        torch.manual_seed(1)
        batch_size = [3, 4, 5, 6, 7]

        assert convert_ellipsis_to_idx(ellipsis_index, batch_size) == expected_index

    @pytest.mark.skipif(
        not torch.cuda.device_count() and not npu_device_count, reason="no cuda or npu"
    )
    def test_create_on_device(self):
        device = torch.device(0)

        # TensorDict
        td = TensorDict({}, [5])
        assert td.device is None

        td.set("a", torch.randn(5, device=device))
        assert td.device is None

        td = TensorDict({}, [5], device=f"{cur_device}:0")
        td.set("a", torch.randn(5, 1))
        assert td.get("a").device == device

        # stacked TensorDict
        td1 = TensorDict({}, [5])
        td2 = TensorDict({}, [5])
        stackedtd = stack_td([td1, td2], 0)
        assert stackedtd.device is None

        stackedtd.set("a", torch.randn(2, 5, device=device))
        assert stackedtd.device is None

        stackedtd = stackedtd.to(device)
        assert stackedtd.device == device

        td1 = TensorDict({}, [5], device=f"{cur_device}:0")
        td2 = TensorDict({}, [5], device=f"{cur_device}:0")
        stackedtd = LazyStackedTensorDict.lazy_stack([td1, td2], 0)
        stackedtd.set("a", torch.randn(2, 5, 1))
        assert stackedtd.get("a").device == device
        assert td1.get("a").device == device
        assert td2.get("a").device == device

        # TensorDict, indexed
        td = TensorDict({}, [5])
        subtd = td[1]
        assert subtd.device is None

        subtd.set("a", torch.randn(1, device=device))
        # setting element of subtensordict doesn't set top-level device
        assert subtd.device is None

        subtd = subtd.to(device)
        assert subtd.device == device
        assert subtd["a"].device == device

        td = TensorDict({}, [5], device=f"{cur_device}:0")
        subtd = td[1]
        subtd.set("a", torch.randn(1))
        assert subtd.get("a").device == device

        td = TensorDict({}, [5], device=f"{cur_device}:0")
        subtd = td[1:3]
        subtd.set("a", torch.randn(2))
        assert subtd.get("a").device == device

        # ViewedTensorDict
        td = TensorDict({}, [6])
        viewedtd = td.view(2, 3)
        assert viewedtd.device is None

        viewedtd = viewedtd.to(device)
        assert viewedtd.device == device

        td = TensorDict({}, [6], device=f"{cur_device}:0")
        viewedtd = td.view(2, 3)
        a = torch.randn(2, 3)
        viewedtd.set("a", a)
        assert viewedtd.get("a").device == device
        assert (a.to(device) == viewedtd.get("a")).all()

    def test_data_grad(self):
        td = TensorDict(
            {
                "a": torch.randn(3, 4, requires_grad=True),
                "b": {"c": torch.randn(3, 4, 5, requires_grad=True)},
            },
            [3, 4],
        )
        td1 = td + 1
        td1.apply(lambda x: x.sum().backward(retain_graph=True), filter_empty=True)
        assert not td.grad.is_locked
        assert td.grad is not td.grad
        assert not td.data.is_locked
        assert td.data is not td.grad
        td.lock_()
        assert td.grad.is_locked
        assert td.data.is_locked

    @pytest.mark.parametrize(
        "stack_dim",
        [0, 1, 2, 3],
    )
    @pytest.mark.parametrize(
        "nested_stack_dim",
        [0, 1, 2],
    )
    def test_dense_stack_tds(self, stack_dim, nested_stack_dim):
        batch_size = (5, 6)
        td0 = TensorDict(
            {"a": torch.zeros(*batch_size, 3)},
            batch_size,
        )
        td1 = TensorDict(
            {"a": torch.zeros(*batch_size, 4), "b": torch.zeros(*batch_size, 2)},
            batch_size,
        )
        td_lazy = LazyStackedTensorDict.lazy_stack([td0, td1], dim=nested_stack_dim)
        td_container = TensorDict({"lazy": td_lazy}, td_lazy.batch_size)
        td_container_clone = td_container.clone()
        td_container_clone.apply_(lambda x: x + 1)

        assert td_lazy.stack_dim == nested_stack_dim
        td_stack = LazyStackedTensorDict.lazy_stack(
            [td_container, td_container_clone], dim=stack_dim
        )
        assert td_stack.stack_dim == stack_dim

        assert isinstance(td_stack, LazyStackedTensorDict)
        dense_td_stack = dense_stack_tds(td_stack)
        assert isinstance(dense_td_stack, TensorDict)  # check outer layer is non-lazy
        assert isinstance(
            dense_td_stack["lazy"], LazyStackedTensorDict
        )  # while inner layer is still lazy
        assert "b" not in dense_td_stack["lazy"].tensordicts[0].keys()
        assert "b" in dense_td_stack["lazy"].tensordicts[1].keys()

        assert assert_allclose_td(
            dense_td_stack,
            dense_stack_tds([td_container, td_container_clone], dim=stack_dim),
        )  # This shows it is the same to pass a list or a LazyStackedTensorDict

        for i in range(2):
            index = (slice(None),) * stack_dim + (i,)
            assert (dense_td_stack[index] == i).all()

        if stack_dim > nested_stack_dim:
            assert dense_td_stack["lazy"].stack_dim == nested_stack_dim
        else:
            assert dense_td_stack["lazy"].stack_dim == nested_stack_dim + 1

    def test_deepcopy(self):
        td = TensorDict(a=TensorDict(b=0))
        tdc = deepcopy(td)
        assert (td == tdc).all()
        assert (td.data_ptr(storage=True) != tdc.data_ptr(storage=True)).all()

    def test_dtype(self):
        td = TensorDict(
            {("an", "integer"): 1, ("a", "string"): "a", ("the", "float"): 1.0}
        )
        assert td.dtype is None
        td = td.float()
        assert td.dtype == torch.float
        td = td.int()
        assert td.dtype == torch.int

    def test_empty(self):
        td = TensorDict(
            {
                "a": torch.zeros(()),
                ("b", "c"): torch.zeros(()),
                ("b", "d", "e"): torch.zeros(()),
            },
            [],
        )
        td_empty = td.empty(recurse=False)
        assert len(list(td_empty.keys())) == 0
        td_empty = td.empty(recurse=True)
        assert len(list(td_empty.keys())) == 1
        assert len(list(td_empty.get("b").keys())) == 1

    @pytest.mark.parametrize("inplace", [True, False])
    def test_exclude_nested(self, inplace):
        tensor_1 = torch.rand(4, 5, 6, 7)
        tensor_2 = torch.rand(4, 5, 6, 7)
        sub_sub_tensordict = TensorDict(
            {"t1": tensor_1, "t2": tensor_2}, batch_size=[4, 5, 6]
        )
        sub_tensordict = TensorDict(
            {"double_nested": sub_sub_tensordict}, batch_size=[4, 5]
        )
        tensordict = TensorDict(
            {
                "a": torch.rand(4, 3),
                "b": torch.rand(4, 2),
                "c": torch.rand(4, 1),
                "nested": sub_tensordict,
            },
            batch_size=[4],
        )
        # making a copy for inplace tests
        tensordict2 = tensordict.clone()

        excluded = tensordict.exclude(
            "b", ("nested", "double_nested", "t2"), inplace=inplace
        )

        assert set(excluded.keys(include_nested=True)) == {
            "a",
            "c",
            "nested",
            ("nested", "double_nested"),
            ("nested", "double_nested", "t1"),
        }

        if inplace:
            assert excluded is tensordict
            assert set(tensordict.keys(include_nested=True)) == {
                "a",
                "c",
                "nested",
                ("nested", "double_nested"),
                ("nested", "double_nested", "t1"),
            }
        else:
            assert excluded is not tensordict
            assert set(tensordict.keys(include_nested=True)) == {
                "a",
                "b",
                "c",
                "nested",
                ("nested", "double_nested"),
                ("nested", "double_nested", "t1"),
                ("nested", "double_nested", "t2"),
            }

        # excluding "nested" should exclude all subkeys also
        excluded2 = tensordict2.exclude("nested", inplace=inplace)
        assert set(excluded2.keys(include_nested=True)) == {"a", "b", "c"}

    @pytest.mark.parametrize("device", get_available_devices())
    def test_expand(self, device):
        torch.manual_seed(1)
        d = {
            "key1": torch.randn(4, 5, 6, device=device),
            "key2": torch.randn(4, 5, 10, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5), source=d)
        td2 = td1.expand(3, 7, 4, 5)
        assert td2.batch_size == torch.Size([3, 7, 4, 5])
        assert td2.get("key1").shape == torch.Size([3, 7, 4, 5, 6])
        assert td2.get("key2").shape == torch.Size([3, 7, 4, 5, 10])

    def test_expand_as(self):
        td0 = TensorDict(
            {"a": torch.ones(3, 1, 4), "b": {"c": torch.ones(3, 2, 1, 4)}},
            batch_size=[3],
        )
        td1 = TensorDict(
            {"a": torch.zeros(2, 3, 5, 4), "b": {"c": torch.zeros(2, 3, 2, 6, 4)}},
            batch_size=[2, 3],
        )
        expanded = td0.expand_as(td1)
        assert (expanded == 1).all()
        assert expanded["b", "c"].shape == torch.Size([2, 3, 2, 6, 4])

    @pytest.mark.parametrize("device", get_available_devices())
    def test_expand_with_singleton(self, device):
        torch.manual_seed(1)
        d = {
            "key1": torch.randn(1, 5, 6, device=device),
            "key2": torch.randn(1, 5, 10, device=device),
        }
        td1 = TensorDict(batch_size=(1, 5), source=d)
        td2 = td1.expand(3, 7, 4, 5)
        assert td2.batch_size == torch.Size([3, 7, 4, 5])
        assert td2.get("key1").shape == torch.Size([3, 7, 4, 5, 6])
        assert td2.get("key2").shape == torch.Size([3, 7, 4, 5, 10])

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "td_type", ["tensordict", "view", "unsqueeze", "squeeze", "stack"]
    )
    @pytest.mark.parametrize("update", [True, False])
    # getting values from lazy tensordicts in non-lazy contexts messes things up
    # so we set it to True. When we'll deprecate lazy tensordicts, we will just
    # remove this decorator
    @set_lazy_legacy(True)
    def test_filling_empty_tensordict(self, device, td_type, update):
        if td_type == "tensordict":
            td = TensorDict(batch_size=[16], device=device)
        elif td_type == "view":
            td = TensorDict(batch_size=[4, 4], device=device).view(-1)
        elif td_type == "unsqueeze":
            td = TensorDict(batch_size=[16], device=device).unsqueeze(-1)
        elif td_type == "squeeze":
            td = TensorDict(batch_size=[16, 1], device=device).squeeze(-1)
        elif td_type == "stack":
            td = LazyStackedTensorDict.lazy_stack(
                [TensorDict({}, [], device=device) for _ in range(16)], 0
            )
        else:
            raise NotImplementedError

        for i in range(16):
            other_td = TensorDict({"a": torch.randn(10), "b": torch.ones(1)}, [])
            if td_type == "unsqueeze":
                other_td = other_td.unsqueeze(-1).to_tensordict(retain_none=True)
            if update:
                subtd = td._get_sub_tensordict(i)
                subtd.update(other_td, inplace=True, non_blocking=False)
            else:
                td[i] = other_td

        assert td.device == device
        assert td.get("a").device == device
        assert (td.get("b") == 1).all()
        if td_type == "view":
            assert td._source["a"].shape == torch.Size([4, 4, 10])
        elif td_type == "unsqueeze":
            assert td._source["a"].shape == torch.Size([16, 10])
        elif td_type == "squeeze":
            assert td._source["a"].shape == torch.Size([16, 1, 10])
        elif td_type == "stack":
            assert (td[-1] == other_td.to(device)).all()

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("separator", [",", "-"])
    def test_flatten_unflatten_key_collision(self, inplace, separator):
        td1 = TensorDict(
            {
                f"a{separator}b{separator}c": torch.zeros(3),
                "a": {"b": {"c": torch.zeros(3)}},
            },
            [],
        )
        td2 = TensorDict(
            {
                f"a{separator}b": torch.zeros(3),
                "a": {"b": torch.zeros(3)},
                "g": {"d": torch.zeros(3)},
            },
            [],
        )
        td3 = TensorDict(
            {
                f"a{separator}b{separator}c": torch.zeros(3),
                "a": {"b": {"c": torch.zeros(3), "d": torch.zeros(3)}},
            },
            [],
        )

        td4 = TensorDict(
            {
                f"a{separator}b{separator}c{separator}d": torch.zeros(3),
                "a": {"b": {"c": torch.zeros(3)}},
            },
            [],
        )

        td5 = TensorDict(
            {f"a{separator}b": torch.zeros(3), "a": {"b": {"c": torch.zeros(3)}}}, []
        )

        with pytest.raises(KeyError, match="Flattening keys in tensordict causes keys"):
            _ = td1.flatten_keys(separator)

        with pytest.raises(KeyError, match="Flattening keys in tensordict causes keys"):
            _ = td2.flatten_keys(separator)

        with pytest.raises(KeyError, match="Flattening keys in tensordict causes keys"):
            _ = td3.flatten_keys(separator)

        with pytest.raises(
            KeyError,
            match=re.escape(
                "Unflattening key(s) in tensordict will override an existing for unflattened key"
            ),
        ):
            _ = td1.unflatten_keys(separator)

        with pytest.raises(
            KeyError,
            match=re.escape(
                "Unflattening key(s) in tensordict will override an existing for unflattened key"
            ),
        ):
            _ = td2.unflatten_keys(separator)

        with pytest.raises(
            KeyError,
            match=re.escape(
                "Unflattening key(s) in tensordict will override an existing for unflattened key"
            ),
        ):
            _ = td3.unflatten_keys(separator)

        with pytest.raises(
            KeyError,
            match=re.escape(
                "Unflattening key(s) in tensordict will override an existing for unflattened key"
            ),
        ):
            _ = td4.unflatten_keys(separator)

        with pytest.raises(
            KeyError,
            match=re.escape(
                "Unflattening key(s) in tensordict will override an existing for unflattened key"
            ),
        ):
            _ = td5.unflatten_keys(separator)

        td4_flat = td4.flatten_keys(separator)
        assert (f"a{separator}b{separator}c{separator}d") in td4_flat.keys()
        assert (f"a{separator}b{separator}c") in td4_flat.keys()

        td5_flat = td5.flatten_keys(separator)
        assert (f"a{separator}b") in td5_flat.keys()
        assert (f"a{separator}b{separator}c") in td5_flat.keys()

    @pytest.mark.parametrize(
        "in_out", [(None, None), (0, -1), (None, 0), (0, None), (0, 0)]
    )
    def test_flatten_empty(self, in_out):
        td = TensorDict(a=torch.zeros((2,)))
        td_flat = td.flatten(in_out[0], in_out[1])
        assert td_flat.shape == (1,)
        assert td_flat["a"].shape == (
            1,
            2,
        )

    def test_flatten_00(self):
        td = TensorDict(a=torch.zeros((2,)), batch_size=(2,))
        td_flat = td.flatten(0, 0)
        assert td_flat.shape == (2,)
        assert td_flat["a"].shape == (2,)

    def test_fromkeys(self):
        td = TensorDict.fromkeys({"a", "b", "c"})
        assert td["a"] == 0
        td = TensorDict.fromkeys({"a", "b", "c"}, 1)
        assert td["a"] == 1

    def test_from_any(self):
        from dataclasses import dataclass

        @dataclass
        class MyClass:
            a: int

        pytree = (
            [[-1, 0, 1], [2, 3, 4]],
            {
                "tensor": torch.randn(
                    2,
                ),
                "td": TensorDict({"one": 1}),
                "tuple": (1, 2, 3),
            },
            {"named_tuple": TensorDict({"two": torch.ones(1) * 2}).to_namedtuple()},
            {"dataclass": MyClass(a=0)},
        )
        if _has_h5py:
            pytree = pytree + ({"h5py": TestTensorDictsBase.td_h5(device="cpu").file},)
        td = TensorDict.from_any(pytree)
        expected = {
            "0",
            ("1", "td", "one"),
            ("1", "tensor"),
            ("1", "tuple", "0"),
            ("1", "tuple", "1"),
            ("1", "tuple", "2"),
            ("2", "named_tuple", "two"),
            ("3", "dataclass", "a"),
        }
        if _has_h5py:
            expected = expected.union(
                {
                    ("4", "h5py", "a"),
                    ("4", "h5py", "b"),
                    ("4", "h5py", "c"),
                    ("4", "h5py", "my_nested_td", "inner"),
                }
            )
        assert set(td.keys(True, True)) == expected, set(
            td.keys(True, True)
        ).symmetric_difference(expected)

    def test_from_any_list(self):
        t = torch.randn(3, 4, 5)
        t = t.tolist()
        assert isinstance(TensorDict.from_any(t), torch.Tensor)
        t[0][1].extend([0, 2])
        assert isinstance(TensorDict.from_any(t), NonTensorStack)

    def test_from_any_userdict(self):
        class D(UserDict): ...

        d = D(a=0)
        assert TensorDict.from_any(d)["a"] == 0
        assert isinstance(TensorDict.from_any(d)["a"], torch.Tensor)

    def test_from_dataclass(self):
        @dataclass
        class MyClass:
            a: int
            b: Any

        obj = MyClass(a=0, b=1)
        obj_td = TensorDict.from_dataclass(obj)
        obj_tc = TensorDict.from_dataclass(obj, as_tensorclass=True)
        assert is_tensorclass(obj_tc)
        assert not is_tensorclass(obj_td)

    @pytest.mark.parametrize("batch_size", [None, [3, 4]])
    @pytest.mark.parametrize("batch_dims", [None, 1, 2])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_from_dict(self, batch_size, batch_dims, device):
        data = {
            "a": torch.zeros(3, 4, 5),
            "b": {"c": torch.zeros(3, 4, 5, 6)},
            ("d", "e"): torch.ones(3, 4, 5),
            ("b", "f"): torch.zeros(3, 4, 5, 5),
            ("d", "g", "h"): torch.ones(3, 4, 5),
        }
        if batch_dims and batch_size:
            with pytest.raises(ValueError, match="both"):
                TensorDict.from_dict(
                    data, batch_size=batch_size, batch_dims=batch_dims, device=device
                )
            return
        data = TensorDict.from_dict(
            data,
            batch_size=batch_size,
            batch_dims=batch_dims,
            device=device,
            auto_batch_size=True,
        )
        assert data.device == device
        assert "a" in data.keys()
        assert ("b", "c") in data.keys(True)
        assert ("b", "f") in data.keys(True)
        assert ("d", "e") in data.keys(True)
        assert data.device == device
        if batch_dims:
            assert data.ndim == batch_dims
            assert data["b"].ndim == batch_dims
            assert data["d"].ndim == batch_dims
            assert data["d", "g"].ndim == batch_dims
        elif batch_size:
            assert data.batch_size == torch.Size(batch_size)
            assert data["b"].batch_size == torch.Size(batch_size)
            assert data["d"].batch_size == torch.Size(batch_size)
            assert data["d", "g"].batch_size == torch.Size(batch_size)

    def test_from_dict_instance(self):
        @tensorclass(autocast=True)
        class MyClass:
            x: torch.Tensor = None  # type: ignore
            y: int = None  # type: ignore
            z: "MyClass" = None  # type: ignore

        td = TensorDict(
            {"a": torch.randn(()), "b": MyClass(x=torch.zeros(()), y=1, z=MyClass(y=2))}
        )
        td_dict = td.to_dict()
        assert isinstance(td_dict["b"], dict)
        assert isinstance(td_dict["b"]["y"], int)
        assert isinstance(td_dict["b"]["z"], dict)
        assert isinstance(td_dict["b"]["z"]["y"], int)
        td_recon = td.from_dict_instance(td_dict, auto_batch_size=True)
        assert isinstance(td_recon["a"], torch.Tensor)
        assert isinstance(td_recon["b"], MyClass)
        assert isinstance(td_recon["b"].x, torch.Tensor)
        assert isinstance(td_recon["b"].y, int)
        assert isinstance(td_recon["b"].z, MyClass)
        assert isinstance(td_recon["b"].z.y, int)

    @pytest.mark.parametrize("memmap", [True, False])
    @pytest.mark.parametrize("params", [False, True])
    def test_from_module(self, memmap, params):
        net = nn.Transformer(
            d_model=16,
            nhead=2,
            num_encoder_layers=3,
            dim_feedforward=12,
            batch_first=True,
        )
        td = TensorDict.from_module(net, as_module=params, filter_empty=False)
        # check that we have empty tensordicts, reflecting modules without params
        for subtd in td.values(True):
            if isinstance(subtd, TensorDictBase) and subtd.is_empty():
                break
        else:
            raise RuntimeError
        if memmap:
            td = td.detach().memmap_()
        net.load_state_dict(td.flatten_keys("."))

        if not memmap and params:
            assert set(td.parameters()) == set(net.parameters())

    def test_from_module_state_dict(self):
        net = nn.Transformer(
            d_model=16,
            nhead=2,
            num_encoder_layers=3,
            dim_feedforward=12,
            batch_first=True,
        )

        def adder(module, *args, **kwargs):
            for p in module.parameters(recurse=False):
                p.data += 1

        def remover(module, *args, **kwargs):
            for p in module.parameters(recurse=False):
                p.data = p.data - 1

        for module in net.modules():
            module.register_state_dict_pre_hook(adder)
            module._register_state_dict_hook(remover)
        params_reg = TensorDict.from_module(net)
        params_reg = params_reg.select(*params_reg.keys(True, True))

        params_sd = TensorDict.from_module(net, use_state_dict=True)
        params_sd = params_sd.select(*params_sd.keys(True, True))
        assert_allclose_td(params_sd, params_reg.apply(lambda x: x + 1))

        sd = net.state_dict()
        assert_allclose_td(params_sd.flatten_keys("."), TensorDict(sd, []))

    @pytest.mark.skipif(PYTORCH_TEST_FBCODE, reason="vmap now working in fbcode")
    @pytest.mark.parametrize("as_module", [False, True])
    @pytest.mark.parametrize("lazy_stack", [False, True])
    def test_from_modules(self, as_module, lazy_stack):
        empty_module = nn.Linear(3, 4, device="meta")
        modules = [nn.Linear(3, 4) for _ in range(3)]
        if as_module and lazy_stack:
            with pytest.raises(RuntimeError, match="within a TensorDictParams"):
                params = TensorDict.from_modules(
                    *modules, as_module=as_module, lazy_stack=lazy_stack
                )
            return

        params = TensorDict.from_modules(
            *modules, as_module=as_module, lazy_stack=lazy_stack
        )

        def exec_module(params, x):
            with params.to_module(empty_module, preserve_module_state=False):
                return empty_module(x)

        x = torch.zeros(3)
        y = torch.vmap(exec_module, (0, None))(params, x)
        y.sum().backward()
        if lazy_stack:
            leaves = []

            def get_leaf(leaf):
                leaves.append(leaf)

            params.apply(get_leaf, filter_empty=True)
            assert all(param.grad is not None for param in leaves)
            with (
                pytest.warns(
                    UserWarning,
                    match="The .grad attribute of a Tensor that is not a leaf Tensor is being accessed.",
                )
                if lazy_stack
                else contextlib.nullcontext()
            ):
                assert all(param.grad is None for param in params.values(True, True))
        else:
            for p in modules[0].parameters():
                assert p.grad is None
            assert all(param.grad is not None for param in params.values(True, True))

    @pytest.mark.skipif(PYTORCH_TEST_FBCODE, reason="vmap now working in fbcode")
    @pytest.mark.parametrize("as_module", [False, True])
    def test_from_modules_expand(self, as_module):
        empty_module = nn.Sequential(
            nn.Linear(3, 3, device="meta"), nn.Linear(3, 4, device="meta")
        )
        module0 = nn.Linear(3, 3)
        modules = [nn.Sequential(module0, nn.Linear(3, 4)) for _ in range(3)]
        params = TensorDict.from_modules(
            *modules, as_module=as_module, expand_identical=True
        )
        assert not isinstance(params["0", "weight"], nn.Parameter)
        assert params["0", "weight"].data.data_ptr() == module0.weight.data.data_ptr()
        assert isinstance(params["1", "weight"], nn.Parameter)
        assert (
            params["1", "weight"].data.data_ptr()
            != modules[0][1].weight.data.data_ptr()
        )

        def exec_module(params, x):
            with params.to_module(empty_module, preserve_module_state=False):
                return empty_module(x)

        x = torch.zeros(3)
        y = torch.vmap(exec_module, (0, None))(params, x)
        y.sum().backward()
        for k, p in modules[0].named_parameters():
            assert p.grad is None if k.startswith("1") else p.grad is not None, k
        assert all(
            param.grad is not None
            for param in params.values(True, True)
            if isinstance(param, nn.Parameter)
        )

    @pytest.mark.skipif(PYTORCH_TEST_FBCODE, reason="vmap now working in fbcode")
    @pytest.mark.parametrize("as_module", [False, True])
    @pytest.mark.parametrize("lazy_stack", [False, True])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_from_modules_lazy(self, as_module, lazy_stack, device):
        empty_module = nn.LazyLinear(4, device="meta")
        modules = [nn.LazyLinear(4, device=device) for _ in range(3)]
        if lazy_stack:
            with pytest.raises(
                RuntimeError,
                match="lasy_stack=True is not compatible with lazy modules.",
            ):
                params = TensorDict.from_modules(
                    *modules, as_module=as_module, lazy_stack=lazy_stack
                )
            return

        params = TensorDict.from_modules(
            *modules, as_module=as_module, lazy_stack=lazy_stack
        )
        optim = torch.optim.Adam(list(params.values(True, True)))

        def exec_module(params, x):
            with params.to_module(empty_module, preserve_module_state=False):
                return empty_module(x)

        x = torch.zeros(3, device=device)
        assert isinstance(params["weight"], UninitializedTensorMixin)
        y = torch.vmap(exec_module, (0, None), randomness="same")(params, x)
        assert params["weight"].shape == torch.Size((3, 4, 3))

        y.sum().backward()
        optim.step()

        if lazy_stack:
            leaves = []

            def get_leaf(leaf):
                leaves.append(leaf)

            params.apply(get_leaf)
            assert all(param.grad is not None for param in leaves)
            assert all(param.grad is None for param in params.values(True, True))
        else:
            for p in modules[0].parameters():
                assert p.grad is None
            assert all(param.grad is not None for param in params.values(True, True))

    def test_from_pytree(self):
        class WeirdLookingClass:
            pass

        weird_key = WeirdLookingClass()

        pytree = (
            [torch.randint(10, (3,)), torch.zeros(2)],
            {
                "tensor": torch.randn(
                    2,
                ),
                "td": TensorDict({"one": 1}),
                weird_key: torch.randint(10, (2,)),
                "list": [1, 2, 3],
            },
            {"named_tuple": TensorDict({"two": torch.ones(1) * 2}).to_namedtuple()},
        )
        td = TensorDict.from_pytree(pytree)
        pytree_recon = td.to_pytree()

        def check(v1, v2):
            assert (v1 == v2).all()

        torch.utils._pytree.tree_map(check, pytree, pytree_recon)
        assert weird_key in pytree_recon[1]

    def test_from_struct_array(self):
        x = np.array(
            [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
            dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
        )
        td = TensorDict.from_struct_array(x)
        x_recon = td.to_struct_array()
        assert (x_recon == x).all()
        assert x_recon.shape == x.shape
        # Try modifying x age field and check effect on td
        x["age"] += 1
        assert (td["age"] == np.array([10, 4])).all()

        # no shape
        x = np.array(
            ("Rex", 9, 81.0), dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")]
        )
        td = TensorDict.from_struct_array(x)
        assert td.shape == ()
        x_recon = td.to_struct_array()
        assert (x_recon == x).all()
        assert x_recon.shape == x.shape
        x["age"] += 1
        assert td["age"] == np.array(10)

        # nested
        dtype = [
            ("Name", "U10"),
            ("Age", "i4"),
            ("Grades", [("Math", "f4"), ("Science", "f4")]),
        ]
        # Create data for the array
        data = [
            ("Alice", 25, (88.5, 92.0)),
            ("Bob", 30, (85.0, 88.0)),
            ("Cathy", 22, (95.0, 97.0)),
        ]
        # Create the structured array
        students = np.array(data, dtype=dtype)
        td = TensorDict.from_struct_array(students)
        assert (td["Grades", "Math"] == torch.Tensor([88.5, 85.0, 95.0])).all()
        assert (td.to_struct_array() == students).all()

    @pytest.mark.parametrize(
        "idx",
        [
            (slice(None),),
            slice(None),
            (3, 4),
            (3, slice(None), slice(2, 2, 2)),
            (torch.tensor([1, 2, 3]),),
            ([1, 2, 3]),
            (
                torch.tensor([1, 2, 3]),
                torch.tensor([2, 3, 4]),
                torch.tensor([0, 10, 2]),
                torch.tensor([2, 4, 1]),
            ),
            torch.zeros(10, 7, 11, 5, dtype=torch.bool).bernoulli_(),
            torch.zeros(10, 7, 11, dtype=torch.bool).bernoulli_(),
            (0, torch.zeros(7, dtype=torch.bool).bernoulli_()),
        ],
    )
    def test_getitem_batch_size(self, idx):
        shape = [10, 7, 11, 5]
        shape = torch.Size(shape)
        mocking_tensor = torch.zeros(*shape)
        expected_shape = mocking_tensor[idx].shape
        resulting_shape = _getitem_batch_size(shape, idx)
        assert expected_shape == resulting_shape, (idx, expected_shape, resulting_shape)

    def test_getitem_nested(self):
        tensor = torch.randn(4, 5, 6, 7)
        sub_sub_tensordict = TensorDict({"c": tensor}, [4, 5, 6])
        sub_tensordict = TensorDict({}, [4, 5])
        tensordict = TensorDict({}, [4])

        sub_tensordict["b"] = sub_sub_tensordict
        tensordict["a"] = sub_tensordict

        # check that content match
        assert (tensordict["a"] == sub_tensordict).all()
        assert (tensordict["a", "b"] == sub_sub_tensordict).all()
        assert (tensordict["a", "b", "c"] == tensor).all()

        # check that get method returns same contents
        assert (tensordict.get("a") == sub_tensordict).all()
        assert (tensordict.get(("a", "b")) == sub_sub_tensordict).all()
        assert (tensordict.get(("a", "b", "c")) == tensor).all()

        # check that shapes are kept
        assert tensordict.shape == torch.Size([4])
        assert sub_tensordict.shape == torch.Size([4, 5])
        assert sub_sub_tensordict.shape == torch.Size([4, 5, 6])

    @set_lazy_legacy(True)
    def test_inferred_view_size(self):
        td = TensorDict({"a": torch.randn(3, 4)}, [3, 4])
        assert td.view(-1).view(-1, 4) is td

        assert td.view(-1, 4) is td
        assert td.view(3, -1) is td
        assert td.view(3, 4) is td
        assert td.view(-1, 12).shape == torch.Size([1, 12])

    def test_is_empty(self):
        assert TensorDict({"a": {"b": {}}}, []).is_empty()
        assert not TensorDict(
            {"a": {"b": {}, "c": NonTensorData("a string!", batch_size=[])}}, []
        ).is_empty()
        assert not TensorDict({"a": {"b": {}, "c": 1}}, []).is_empty()

    def test_is_in(self):
        td = TensorDict.fromkeys({"a", "b", "c", ("d", "e")}, 0)
        assert "a" in td
        assert "d" in td
        assert ("d", "e") in td
        assert "f" not in td
        with pytest.raises(RuntimeError, match="NestedKey"):
            0 in td  # noqa: B015

    def test_keys_view(self):
        tensor = torch.randn(4, 5, 6, 7)
        sub_sub_tensordict = TensorDict({"c": tensor}, [4, 5, 6])
        sub_tensordict = TensorDict({}, [4, 5])
        tensordict = TensorDict({}, [4])

        sub_tensordict["b"] = sub_sub_tensordict
        tensordict["a"] = sub_tensordict

        assert "a" in tensordict.keys()
        assert "random_string" not in tensordict.keys()

        assert ("a",) in tensordict.keys(include_nested=True)
        assert ("a", "b", "c") in tensordict.keys(include_nested=True)
        assert ("a", "c", "b") not in tensordict.keys(include_nested=True)

        with pytest.raises(
            TypeError, match="checks with tuples of strings is only supported"
        ):
            ("a", "b", "c") in tensordict.keys()  # noqa: B015

        with pytest.raises(TypeError, match="TensorDict keys are always strings."):
            42 in tensordict.keys()  # noqa: B015

        with pytest.raises(TypeError, match="TensorDict keys are always strings."):
            ("a", 42) in tensordict.keys()  # noqa: B015

        keys = set(tensordict.keys())
        keys_nested = set(tensordict.keys(include_nested=True))

        assert keys == {"a"}
        assert keys_nested == {"a", ("a", "b"), ("a", "b", "c")}

        leaves = set(tensordict.keys(leaves_only=True))
        leaves_nested = set(tensordict.keys(include_nested=True, leaves_only=True))

        assert leaves == set()
        assert leaves_nested == {("a", "b", "c")}

    def test_load_device(self, tmpdir):
        t = nn.Transformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=3,
            dim_feedforward=128,
            batch_first=True,
        )

        state_dict = TensorDict.from_module(t)
        state_dict.data.zero_()

        state_dict.save(tmpdir)
        meta_state_dict = TensorDict.load(tmpdir, device="meta")

        def check_meta(tensor):
            assert tensor.device == torch.device("meta")

        meta_state_dict.apply(check_meta, filter_empty=True)

        if torch.cuda.is_available():
            device = "cuda:0"
        elif is_npu_available():
            device = "npu:0"
        # elif torch.backends.mps.is_available():
        #     device = "mps:0"
        else:
            pytest.skip("no device to test")
        device_state_dict = TensorDict.load(tmpdir, device=device)
        assert (device_state_dict == 0).all()

        def assert_device(item):
            assert item.device == torch.device(device)
            if is_tensor_collection(item):
                item.apply(assert_device, filter_empty=True, call_on_nested=True)

        device_state_dict.apply(assert_device, filter_empty=True, call_on_nested=True)

        with FakeTensorMode():
            fake_state_dict = TensorDict.load(tmpdir)

            def assert_fake(tensor):
                assert isinstance(tensor, FakeTensor)

            fake_state_dict.apply(assert_fake, filter_empty=True)

    def test_load_state_dict_incomplete(self):
        data = TensorDict({"a": {"b": {"c": {}}}, "d": 1}, [])
        sd = TensorDict({"d": 0}, []).state_dict()
        data.load_state_dict(sd, strict=True)
        assert data["d"] == 0
        sd = TensorDict({"a": {"b": {"c": {}}}, "d": 1}, []).state_dict()
        data = TensorDict({"d": 0}, [])
        data.load_state_dict(sd, strict=True)
        assert data["d"] == 1

    def test_state_dict_flat_roundtrip(self):
        td = TensorDict({"a": 1, "b": {"c": 2, "d": {"e": 3}}}, [])
        sd = td.state_dict()
        assert set(sd.keys()) == {"a", "b.c", "b.d.e"}
        assert hasattr(sd, "_metadata")
        assert "" in sd._metadata
        assert "b" in sd._metadata
        assert "b.d" in sd._metadata
        td_zero = td.clone().zero_()
        td_zero.load_state_dict(sd)
        assert (td_zero == td).all()

    def test_state_dict_nested_roundtrip(self):
        td = TensorDict({"a": 1, "b": {"c": 2, "d": {"e": 3}}}, [])
        sd = td.state_dict(flatten=False)
        assert set(sd.keys()) == {"a", "b"}
        assert isinstance(sd["b"], dict)
        assert set(sd["b"].keys()) == {"c", "d"}
        td_zero = td.clone().zero_()
        td_zero.load_state_dict(sd)
        assert (td_zero == td).all()

    def test_state_dict_cross_format_loading(self):
        td = TensorDict({"a": 1, "b": {"c": 2}}, [])
        # Save flat, load with auto-detection
        sd_flat = td.state_dict(flatten=True)
        td_zero = td.clone().zero_()
        td_zero.load_state_dict(sd_flat)
        assert (td_zero == td).all()
        # Save nested, load with auto-detection
        sd_nested = td.state_dict(flatten=False)
        td_zero = td.clone().zero_()
        td_zero.load_state_dict(sd_nested)
        assert (td_zero == td).all()
        # Save flat, explicitly load as flat
        td_zero = td.clone().zero_()
        td_zero.load_state_dict(sd_flat, from_flatten=True)
        assert (td_zero == td).all()
        # Save nested, explicitly load as nested
        td_zero = td.clone().zero_()
        td_zero.load_state_dict(sd_nested, from_flatten=False)
        assert (td_zero == td).all()

    def test_state_dict_legacy_format(self):
        import collections

        legacy_sd = collections.OrderedDict()
        legacy_sd["a"] = torch.tensor(1)
        legacy_sd["__batch_size"] = torch.Size([])
        legacy_sd["__device"] = None
        td = TensorDict({"a": 0}, [])
        td.load_state_dict(legacy_sd)
        assert td["a"] == 1

    def test_state_dict_metadata_content(self):
        td = TensorDict(
            {"x": torch.randn(3), "sub": TensorDict({"y": torch.randn(3)}, [3])},
            [3],
        )
        sd = td.state_dict()
        assert sd._metadata[""]["batch_size"] == torch.Size([3])
        assert sd._metadata["sub"]["batch_size"] == torch.Size([3])

    def test_state_dict_keep_vars(self):
        t = torch.randn(3, requires_grad=True)
        td = TensorDict({"a": t}, [])
        sd_detached = td.state_dict(keep_vars=False)
        assert not sd_detached["a"].requires_grad
        sd_kept = td.state_dict(keep_vars=True)
        assert sd_kept["a"].requires_grad
        assert sd_kept["a"].data_ptr() == t.data_ptr()

    def test_make_memmap(self, tmpdir):
        td = TensorDict()
        td.memmap_(tmpdir)
        mmap = td.make_memmap("a", shape=[3, 4], dtype=torch.float32)
        mmap.fill_(1)
        assert td.is_memmap()
        mmap = td.make_memmap(("b", "c"), shape=[5, 6], dtype=torch.float32)
        mmap.fill_(1)
        assert td.is_memmap()

        assert td.is_locked
        assert td["b"].is_locked

        td_load = TensorDict.load_memmap(tmpdir).memmap_()
        assert td_load.saved_path is not None
        assert (td == td_load).all()
        assert (td_load == 1).all()

        with pytest.raises(
            RuntimeError, match="already exists within the target tensordict"
        ):
            td.make_memmap(("b", "c"), shape=[5, 6], dtype=torch.float32)

        if HAS_NESTED_TENSOR:
            # test update
            mmap = td.make_memmap(("e", "f"), shape=torch.tensor([[1, 2], [1, 3]]))
            td_load.memmap_refresh_()
            assert td_load["e", "f"].is_nested

    def test_make_memmap_from_storage(self, tmpdir):
        td_base = TensorDict(
            {"a": torch.zeros((3, 4)), ("b", "c"): torch.zeros((5, 6))}
        ).memmap_(tmpdir)

        td = TensorDict()
        td.memmap_(tmpdir)
        td.make_memmap_from_storage(
            "a", td_base["a"].untyped_storage(), shape=[3, 4], dtype=torch.float32
        )
        assert td.is_memmap()
        td.make_memmap_from_storage(
            ("b", "c"),
            td_base["b", "c"].untyped_storage(),
            shape=[5, 6],
            dtype=torch.float32,
        )
        assert td.is_memmap()
        assert (td == 0).all()

        assert td.is_locked
        assert td["b"].is_locked

        td_load = TensorDict.load_memmap(tmpdir).memmap_()
        assert td_load.saved_path is not None
        assert (td == td_load).all()
        assert (td_load == 0).all()

        with pytest.raises(
            RuntimeError, match="Providing a storage with an associated filename"
        ):
            another_d = MemoryMappedTensor.ones(
                [5, 6], filename=Path(tmpdir) / "another_d.memmap"
            )
            td.make_memmap_from_storage(
                ("b", "d"),
                another_d.untyped_storage(),
                shape=[5, 6],
                dtype=torch.float32,
            )

    def test_make_memmap_from_tensor(self, tmpdir):
        td = TensorDict()
        td.memmap_(tmpdir)
        a = torch.ones((3, 4))
        td.make_memmap_from_tensor("a", a)
        assert td.is_memmap()
        c = torch.ones((5, 6))
        td.make_memmap_from_tensor(("b", "c"), c)
        assert td.is_memmap()
        assert (td == 1).all()

        assert td.is_locked
        assert td["b"].is_locked

        td_load = TensorDict.load_memmap(tmpdir).memmap_()
        assert td_load.saved_path is not None
        assert (td == td_load).all()
        assert (td_load == 1).all()

        # Should work too if the tensor is memmap
        d = MemoryMappedTensor.ones(
            1, 2, filename=tmpdir + "/some_random_tensor.memmap"
        )
        d_copy = td.make_memmap_from_tensor("d", d)
        assert d_copy.untyped_storage().data_ptr() != d.untyped_storage().data_ptr()
        assert (d_copy == 1).all()

        if HAS_NESTED_TENSOR:
            # test update
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                td.make_memmap_from_tensor(
                    ("e", "f"),
                    torch.nested.nested_tensor(
                        [torch.zeros((1, 2)), torch.zeros((1, 3))]
                    ),
                )
            td_load.memmap_refresh_()
            assert td_load["e", "f"].is_nested

    @pytest.mark.parametrize("device", get_available_devices())
    def test_mask_td(self, device):
        torch.manual_seed(1)
        d = {
            "key1": torch.randn(4, 5, 6, device=device),
            "key2": torch.randn(4, 5, 10, device=device),
        }
        mask = torch.zeros(4, 5, dtype=torch.bool, device=device).bernoulli_()
        td = TensorDict(batch_size=(4, 5), source=d)

        td_masked = torch.masked_select(td, mask)
        assert len(td_masked.get("key1")) == td_masked.shape[0]

    @pytest.mark.parametrize("device", get_available_devices())
    def test_memmap_as_tensor(self, device):
        td = TensorDict(
            {"a": torch.randn(3, 4), "b": {"c": torch.randn(3, 4)}},
            [3, 4],
            device="cpu",
        )
        td_memmap = td.clone().memmap_()
        assert (td == td_memmap).all()

        assert (td == td_memmap.apply(lambda x: x.clone())).all()
        if device.type == "cuda":
            td = td.pin_memory()
            td_memmap = td.clone().memmap_()
            td_memmap_pm = td_memmap.apply(lambda x: x.clone()).pin_memory()
            assert (td.pin_memory().to(device) == td_memmap_pm.to(device)).all()

    def test_lazy_stack_tc_weird_case(self):

        class MyClass(TensorClass["nocast"]):
            a: torch.Tensor
            b: torch.Tensor

        td = TensorDict(
            x=MyClass(a=torch.randn(2, 3), b=torch.randn(2, 3), batch_size=(2, 3)),
            batch_size=(2, 3),
        )
        td = lazy_stack(list(td.unbind(0)), 0)
        td2 = TensorDict(
            x=MyClass(a=torch.randn(2, 4), b=torch.randn(2, 4), batch_size=(2, 4)),
            batch_size=(2, 4),
        )
        td2 = lazy_stack(list(td2.unbind(1)), 1)

        assert TensorDict.maybe_dense_stack([td, td2], 0).batch_size == (2, 2, -1)
        assert TensorDict.maybe_dense_stack([td, td2], 1).batch_size == (2, 2, -1)
        assert TensorDict.maybe_dense_stack([td, td2], 2).batch_size == (2, -1, 2)

    def test_load_custom_artifact(self):
        # Build the TD using this
        # td = TensorDict(
        #     {
        #         "nested": {
        #             "int64": [[1], [2]],
        #             "string": ["a string!"],
        #             "bfloat16": torch.ones(2, 1, dtype=torch.bfloat16),
        #         }
        #     },
        #     batch_size=[2, 1],
        #     names=["batch", "time"],
        # ).memmap(f"{Path(__file__).parent}/artifacts/mmap_example/")

        td = TensorDict.load(f"{Path(__file__).parent}/artifacts/mmap_example/")
        assert td.shape == (2, 1)
        # assert td.names == ("batch", "time")
        assert (td["nested", "int64"] == torch.tensor([[1], [2]])).all()
        assert td["nested", "int64"].dtype == torch.int64
        assert td["nested", "string"] == ["a string!"]
        assert (
            td["nested", "bfloat16"] == torch.ones(2, 1, dtype=torch.bfloat16)
        ).all()
        assert td["nested", "bfloat16"].dtype == torch.bfloat16
        td_nested = TensorDict.load(
            f"{Path(__file__).parent}/artifacts/mmap_example/nested"
        )
        assert (td_nested == td["nested"]).all()

    @pytest.mark.parametrize("method", ["share_memory", "memmap"])
    def test_memory_lock(self, method):
        torch.manual_seed(1)
        td = TensorDict({"a": torch.randn(4, 5)}, batch_size=(4, 5))

        # lock=True
        if method == "share_memory":
            td.share_memory_()
        elif method == "memmap":
            td.memmap_()
        else:
            raise NotImplementedError

        td.set("a", torch.randn(4, 5), inplace=True, non_blocking=False)
        td.set_("a", torch.randn(4, 5))  # No exception because set_ ignores the lock

        with pytest.raises(RuntimeError, match="Cannot modify locked TensorDict"):
            td.set("a", torch.randn(4, 5))

        with pytest.raises(RuntimeError, match="Cannot modify locked TensorDict"):
            td.set("b", torch.randn(4, 5))

        with pytest.raises(RuntimeError, match="Cannot modify locked TensorDict"):
            td.set("b", torch.randn(4, 5), inplace=True, non_blocking=False)

    @pytest.mark.parametrize("dist_of_callables", [False, True])
    def test_merge_tensordicts(self, dist_of_callables):
        td0 = TensorDict({"a": {"b0": 0}, "c": {"d": {"e": 0}}, "common": 0})
        td1 = TensorDict({"a": {"b1": 1}, "f": {"g": {"h": 1}}, "common": 1})
        td2 = TensorDict({"a": {"b2": 2}, "f": {"g": {"h": 2}}, "common": 2})
        caller = lambda *v: torch.stack(list(v))
        if dist_of_callables:
            caller = {"common": caller}
        td = merge_tensordicts(td0, td1, td2, callback_exist=caller)
        assert td["a", "b0"] == 0
        assert td["a", "b1"] == 1
        assert td["a", "b2"] == 2
        assert (td["common"] == torch.arange(3)).all()
        assert td["c", "d", "e"] == 0
        assert td["f", "g", "h"] == 1

    def test_no_batch_size(self):
        td = TensorDict({"a": torch.zeros(3, 4)})
        assert td.batch_size == torch.Size([])

    def test_non_blocking(self):
        if torch.cuda.is_available():
            device = "cuda"
        # elif torch.backends.mps.is_available():
        #     device = "mps"
        elif is_npu_available():
            device = "npu"
        else:
            pytest.skip("No device found")
        for _ in range(10):
            td = TensorDict(
                {str(i): torch.ones((10,), device=device) for i in range(5)},
                [10],
                non_blocking=False,
                device="cpu",
            )
            assert (td == 1).all()
        for _ in range(10):
            assert not tensordict_base._device_recorder.marked
            assert not tensordict_base._device_recorder.has_transfer()
            td = TensorDict(
                {str(i): torch.ones((10,), device=device) for i in range(5)},
                [10],
                device="cpu",
            )
            assert (td == 1).all()
        # This is too flaky
        # with pytest.raises(AssertionError):
        #     for _ in range(10):
        #         td = TensorDict(
        #             {str(i): torch.ones((10,), device=device) for i in range(5)},
        #             [10],
        #             non_blocking=True,
        #             device="cpu",
        #         )
        #         assert (td == 1).all()
        for _ in range(10):
            td = TensorDict(
                {str(i): torch.ones((10,), device="cpu") for i in range(5)},
                [10],
                non_blocking=False,
                device="cpu",
            )
            assert (td.to(device, non_blocking=False) == 1).all()
        for _ in range(10):
            td = TensorDict(
                {str(i): torch.ones((10,), device=device) for i in range(5)},
                [10],
                non_blocking=False,
                device=device,
            )
            assert (td.to("cpu", non_blocking=False) == 1).all()
        for _ in range(10):
            td = TensorDict(
                {str(i): torch.ones((10,), device="cpu") for i in range(5)},
                [10],
                device="cpu",
            )
            assert (td.to(device, non_blocking=False) == 1).all()
        for _ in range(10):
            td = TensorDict(
                {str(i): torch.ones((10,), device=device) for i in range(5)},
                [10],
                device=device,
            )
            assert (td.to("cpu", non_blocking=False) == 1).all()

    def test_non_blocking_single_sync(self, _path_td_sync):
        """Tests that we sync at most once in TensorDict creation."""
        global _SYNC_COUNTER
        _SYNC_COUNTER = 0
        # If everything is on cpu, there should be no sync
        td_dict = {
            "a": torch.randn((), device="cpu"),
            ("b", "c"): torch.randn((), device="cpu"),
            "d": {"e": {"f": torch.randn((), device="cpu")}},
        }
        TensorDict(td_dict, device="cpu")
        assert _SYNC_COUNTER == 0

        # if torch.backends.mps.is_available():
        #     device = "mps"
        if torch.cuda.is_available():
            device = "cuda"
        elif is_npu_available():
            device = "npu"
        else:
            device = None
        if device is not None:
            _SYNC_COUNTER = 0
            # If everything is on device, there should be no sync
            td_dict = {
                "a": torch.randn((), device=device),
                ("b", "c"): torch.randn((), device=device),
                "d": {"e": {"f": torch.randn((), device=device)}},
            }
            TensorDict(td_dict, device=device)
            assert _SYNC_COUNTER == 0

            # if sending to cuda, there should be no sync
            _SYNC_COUNTER = 0
            td_dict = {
                "a": torch.randn((), device="cpu"),
                ("b", "c"): torch.randn((), device="cpu"),
                "d": {"e": {"f": torch.randn((), device="cpu")}},
            }
            TensorDict(td_dict, device=device)
            if device == "cuda":
                assert _SYNC_COUNTER == 0
            else:
                assert _SYNC_COUNTER == 1

            # if receiving on CPU from device, there must be a sync
            _SYNC_COUNTER = 0
            td_dict = {
                "a": torch.randn((), device=device),
                ("b", "c"): torch.randn((), device=device),
                "d": {"e": {"f": torch.randn((), device=device)}},
            }
            TensorDict(td_dict, device="cpu")
            assert _SYNC_COUNTER == 1

            # if non-blocking is True, there should be no sync
            _SYNC_COUNTER = 0
            td_dict = {
                "a": torch.randn((), device="cpu"),
                ("b", "c"): torch.randn((), device="cpu"),
                "d": {"e": {"f": torch.randn((), device="cpu")}},
            }
            TensorDict(td_dict, device=device, non_blocking=True)
            assert _SYNC_COUNTER == 0
            _SYNC_COUNTER = 0
            td_dict = {
                "a": torch.randn((), device=device),
                ("b", "c"): torch.randn((), device=device),
                "d": {"e": {"f": torch.randn((), device=device)}},
            }
            TensorDict(td_dict, device="cpu", non_blocking=False)
            assert _SYNC_COUNTER == 0

            # if non-blocking is False, there should be no sync (not needed)
            _SYNC_COUNTER = 0
            td_dict = {
                "a": torch.randn((), device="cpu"),
                ("b", "c"): torch.randn((), device="cpu"),
                "d": {"e": {"f": torch.randn((), device="cpu")}},
            }
            TensorDict(td_dict, device=device, non_blocking=False)
            assert _SYNC_COUNTER == 0
            _SYNC_COUNTER = 0
            td_dict = {
                "a": torch.randn((), device=device),
                ("b", "c"): torch.randn((), device=device),
                "d": {"e": {"f": torch.randn((), device=device)}},
            }
            TensorDict(td_dict, device="cpu", non_blocking=False)
            assert _SYNC_COUNTER == 0

    def test_device_recorder_per_thread_state(self):
        """``_device_recorder`` must keep ``marked`` per-thread (see #1700)."""
        recorder = tensordict_base._device_recorder
        assert not recorder.marked
        recorder.mark()
        try:
            assert recorder.marked
            saw_marked = []
            barrier = threading.Barrier(2)

            def check_other_thread():
                barrier.wait()
                saw_marked.append(recorder.marked)

            t = threading.Thread(target=check_other_thread)
            t.start()
            barrier.wait()
            t.join()
            assert saw_marked == [False], (
                "Other thread observed marked=True; recorder state is shared "
                "across threads."
            )
        finally:
            recorder.unmark()
        assert not recorder.marked

    def test_non_blocking_thread_safe(self):
        """Concurrent ``TensorDict(..., device='cpu')`` must not race on
        ``_device_recorder.mark()`` (regression for #1700)."""
        source = {f"k{i}": torch.zeros(64, dtype=torch.float32) for i in range(64)}

        num_threads = 8
        iters_per_thread = 2000
        barrier = threading.Barrier(num_threads)
        errors: list[BaseException] = []
        errors_lock = threading.Lock()
        stop = threading.Event()

        def worker():
            try:
                barrier.wait()
                for _ in range(iters_per_thread):
                    if stop.is_set():
                        return
                    TensorDict(source, device="cpu")
            except BaseException as e:
                with errors_lock:
                    errors.append(e)
                stop.set()

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"thread workers raised: {errors[0]!r}"

    def test_to_cpu_correctness_many_tensors(self):
        """Transfer many tensors from CUDA to CPU with default .to('cpu').

        Verifies that event-based sync (or full sync) completes before we read
        values; a subsample of keys/indices is checked. Multiple iterations
        help catch flaky sync issues.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        num_tensors = 10_000
        tensor_shape = (8,)
        num_iterations = 10
        for _ in range(num_iterations):
            td = TensorDict(
                {
                    str(i): torch.full(
                        tensor_shape, float(i), device="cuda", dtype=torch.float32
                    )
                    for i in range(num_tensors)
                },
                batch_size=[],
            )
            td_cpu = td.to("cpu")
            # Subsample: first, middle, last keys and a few indices
            for key in ("0", str(num_tensors // 2), str(num_tensors - 1)):
                expected = float(key)
                assert (td_cpu[key] == expected).all(), f"key={key}"
            assert td_cpu["0"][0].item() == 0.0
            assert td_cpu[str(num_tensors - 1)][-1].item() == float(num_tensors - 1)

    def test_pad(self):
        dim0_left, dim0_right, dim1_left, dim1_right = [0, 1, 0, 2]
        td = TensorDict(
            {
                "a": torch.ones(3, 4, 1),
                "b": torch.zeros(3, 4, 1, 1),
            },
            batch_size=[3, 4],
        )

        padded_td = pad(td, [dim0_left, dim0_right, dim1_left, dim1_right], value=0.0)

        expected_a = torch.cat([torch.ones(3, 4, 1), torch.zeros(1, 4, 1)], dim=0)
        expected_a = torch.cat([expected_a, torch.zeros(4, 2, 1)], dim=1)

        assert padded_td["a"].shape == (4, 6, 1)
        assert padded_td["b"].shape == (4, 6, 1, 1)
        assert torch.equal(padded_td["a"], expected_a)
        padded_td._check_batch_size()

    def test_pad_inplace_identity_and_shapes(self):
        td = TensorDict(
            {
                "a": torch.ones(3, 4, 1),
                "b": torch.zeros(3, 4, 1, 1),
                "nested": TensorDict({"c": torch.ones(3, 4, 2)}, batch_size=[3, 4]),
            },
            batch_size=[3, 4],
        )
        td_id = id(td)
        nested_id = id(td["nested"])
        out = td.pad([0, 1, 0, 2], value=0.0, inplace=True)
        assert out is td
        assert id(out) == td_id
        # Nested TD identity is preserved when inplace=True.
        assert id(out["nested"]) == nested_id
        assert out.batch_size == torch.Size([4, 6])
        assert out["a"].shape == (4, 6, 1)
        assert out["b"].shape == (4, 6, 1, 1)
        assert out["nested"].batch_size == torch.Size([4, 6])
        assert out["nested", "c"].shape == (4, 6, 2)
        out._check_batch_size()

    def test_pad_inplace_matches_functional(self):
        def _build():
            return TensorDict(
                {
                    "a": torch.arange(3 * 4).view(3, 4).float(),
                    "b": torch.arange(3 * 4 * 5).view(3, 4, 5).float(),
                    "nested": TensorDict(
                        {"c": torch.arange(3 * 4 * 2).view(3, 4, 2).float()},
                        batch_size=[3, 4],
                    ),
                },
                batch_size=[3, 4],
            )

        for pad_size in [[0, 1, 0, 2], [1, 0, 0, 2], [1, 0, 2, 1]]:
            td_f = _build()
            td_i = _build()
            ref = pad(td_f, pad_size, value=7.0)
            out = pad(td_i, pad_size, value=7.0, inplace=True)
            assert out is td_i
            assert out.batch_size == ref.batch_size
            assert set(out.keys()) == set(ref.keys())
            for key in ref.keys(include_nested=True, leaves_only=True):
                assert torch.equal(out[key], ref[key]), key

    def test_pad_inplace_releases_old_storage(self):
        td = TensorDict(
            {"a": torch.zeros(8, 16, 32), "b": torch.ones(8, 16)},
            batch_size=[8, 16],
        )
        old_a = td["a"]
        old_b = td["b"]
        ref_a = weakref.ref(old_a)
        ref_b = weakref.ref(old_b)
        del old_a, old_b
        td.pad([0, 0, 0, 1], inplace=True)
        gc.collect()
        # The original leaf tensors must no longer be referenced anywhere
        # once their padded replacements have been written back; this is
        # what gives inplace=True its 1x memory profile.
        assert ref_a() is None
        assert ref_b() is None

    def test_pad_inplace_lazy_stack_non_stack_dim(self):
        td_a = TensorDict({"x": torch.ones(4)}, batch_size=[4])
        td_b = TensorDict({"x": torch.ones(4) * 2}, batch_size=[4])
        lst = lazy_stack([td_a, td_b], dim=0)
        constituent_ids = [id(t) for t in lst.tensordicts]
        out = lst.pad([0, 0, 0, 1], value=0.0, inplace=True)
        assert out is lst
        assert out.batch_size == torch.Size([2, 5])
        # Constituents are the same objects; they were padded in place.
        assert [id(t) for t in out.tensordicts] == constituent_ids
        assert out.tensordicts[0]["x"].shape == (5,)
        assert out.tensordicts[0]["x"][-1].item() == 0.0
        assert out.tensordicts[1]["x"][-1].item() == 0.0

    def test_pad_inplace_lazy_stack_stack_dim(self):
        td_a = TensorDict({"x": torch.ones(4)}, batch_size=[4])
        td_b = TensorDict({"x": torch.ones(4) * 2}, batch_size=[4])
        lst = lazy_stack([td_a, td_b], dim=0)
        out = lst.pad([0, 1, 0, 0], value=9.0, inplace=True)
        assert out is lst
        assert out.batch_size == torch.Size([3, 4])
        assert len(out.tensordicts) == 3
        assert out.tensordicts[-1]["x"].tolist() == [9.0, 9.0, 9.0, 9.0]
        # Existing constituents preserved.
        assert torch.equal(out.tensordicts[0]["x"], torch.ones(4))
        assert torch.equal(out.tensordicts[1]["x"], torch.ones(4) * 2)

    def test_pad_inplace_lazy_stack_both_dims(self):
        td_a = TensorDict({"x": torch.ones(4)}, batch_size=[4])
        td_b = TensorDict({"x": torch.ones(4) * 2}, batch_size=[4])
        lst_inplace = lazy_stack([td_a.clone(), td_b.clone()], dim=0)
        lst_ref = lazy_stack([td_a.clone(), td_b.clone()], dim=0)
        ref = pad(lst_ref, [1, 0, 0, 2], value=3.0)
        out = pad(lst_inplace, [1, 0, 0, 2], value=3.0, inplace=True)
        assert out is lst_inplace
        assert out.batch_size == ref.batch_size
        for key in ref.keys(include_nested=True, leaves_only=True):
            assert torch.equal(out[key], ref[key]), key

    def test_pad_safe_catches_bad_pad_before_mutation(self):
        td = TensorDict(
            {"a": torch.ones(3, 4), "b": torch.zeros(3, 4, 2)},
            batch_size=[3, 4],
        )
        old_a = td["a"]
        old_b = td["b"]
        # Pad would crop more than the source dim — safe=True raises before
        # any leaf is rebound.
        with pytest.raises(RuntimeError, match="negative output size"):
            td.pad([-10, 0, 0, 0], inplace=True, safe=True)
        assert td["a"] is old_a
        assert td["b"] is old_b
        assert td.batch_size == torch.Size([3, 4])

    def test_pad_safe_default_is_true(self):
        td = TensorDict({"a": torch.ones(3, 4)}, batch_size=[3, 4])
        old_a = td["a"]
        with pytest.raises(RuntimeError, match="negative output size"):
            td.pad([-10, 0, 0, 0], inplace=True)
        assert td["a"] is old_a

    def test_pad_safe_false_skips_check(self):
        td = TensorDict({"a": torch.ones(3, 4)}, batch_size=[3, 4])
        # With safe=False, the pre-flight is skipped; torch's own pad call
        # raises mid-loop. The TD ends up inconsistent here, but the
        # underlying torch error is surfaced rather than ours.
        with pytest.raises(RuntimeError):
            td.pad([-10, 0, 0, 0], inplace=True, safe=False)

    def test_pad_safe_lazy_stack(self):
        td_a = TensorDict({"x": torch.ones(4)}, batch_size=[4])
        td_b = TensorDict({"x": torch.ones(4) * 2}, batch_size=[4])
        lst = lazy_stack([td_a, td_b], dim=0)
        snapshots = [t["x"] for t in lst.tensordicts]
        with pytest.raises(RuntimeError, match="negative output size"):
            lst.pad([0, 0, -10, 0], inplace=True)
        # Constituents untouched.
        for t, snap in zip(lst.tensordicts, snapshots):
            assert t["x"] is snap
        assert lst.batch_size == torch.Size([2, 4])

    def test_pad_inplace_tensorclass(self):
        @tensorclass
        class _Sample:
            a: torch.Tensor
            b: torch.Tensor

        s = _Sample(a=torch.ones(3, 4), b=torch.zeros(3, 4, 2), batch_size=[3, 4])
        inner = s._tensordict
        out = s.pad([0, 1, 0, 2], value=0.0, inplace=True)
        # The tensorclass wrap returns a fresh tensorclass instance bound
        # to the same (mutated) inner tensordict; the 1x-memory guarantee
        # lives on the inner TD which is preserved.
        assert out._tensordict is inner
        assert s._tensordict is inner
        assert s.batch_size == torch.Size([4, 6])
        assert out.batch_size == torch.Size([4, 6])
        assert out.a.shape == (4, 6)
        assert out.b.shape == (4, 6, 2)

    def _build_nested_td(self, batch_size=(3, 4), feat=(5,)):
        return TensorDict(
            {
                "a": torch.arange(
                    int(torch.tensor(batch_size).prod())
                    * int(torch.tensor(feat).prod())
                )
                .view(*batch_size, *feat)
                .float(),
                "nested": TensorDict(
                    {
                        "b": torch.arange(int(torch.tensor(batch_size).prod()) * 2)
                        .view(*batch_size, 2)
                        .float()
                    },
                    batch_size=list(batch_size),
                ),
            },
            batch_size=list(batch_size),
        )

    @pytest.mark.parametrize("inplace", [True, False])
    def test_repeat_inplace(self, inplace):
        td = self._build_nested_td()
        ref = TensorDict(
            {k: v.clone() for k, v in td.items(include_nested=True, leaves_only=True)},
            batch_size=[3, 4],
        ).repeat(2, 3)
        out = td.repeat(2, 3, inplace=inplace)
        if inplace:
            assert out is td
            assert id(out["nested"]) == id(td["nested"])
        else:
            assert out is not td
        assert out.batch_size == torch.Size([6, 12])
        for key in ref.keys(include_nested=True, leaves_only=True):
            assert torch.equal(out[key], ref[key]), key

    def test_repeat_inplace_releases_storage(self):
        td = TensorDict({"a": torch.zeros(4, 5, 8)}, batch_size=[4, 5])
        ref = weakref.ref(td["a"])
        td.repeat(2, 1, inplace=True)
        gc.collect()
        assert ref() is None

    def test_repeat_inplace_lazy_stack_raises(self):
        lst = lazy_stack(
            [TensorDict({"x": torch.zeros(4)}, [4]) for _ in range(2)], dim=0
        )
        with pytest.raises(NotImplementedError, match="repeat"):
            lst.repeat(2, 1, inplace=True)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_repeat_interleave_inplace(self, inplace):
        td = self._build_nested_td()
        ref = self._build_nested_td().repeat_interleave(2, dim=0)
        out = td.repeat_interleave(2, dim=0, inplace=inplace)
        if inplace:
            assert out is td
        assert out.batch_size == torch.Size([6, 4])
        for key in ref.keys(include_nested=True, leaves_only=True):
            assert torch.equal(out[key], ref[key]), key

    def test_repeat_interleave_inplace_lazy_stack_raises(self):
        lst = lazy_stack(
            [TensorDict({"x": torch.zeros(4)}, [4]) for _ in range(2)], dim=0
        )
        with pytest.raises(NotImplementedError, match="repeat_interleave"):
            lst.repeat_interleave(2, dim=1, inplace=True)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_roll_inplace(self, inplace):
        td = self._build_nested_td()
        ref = self._build_nested_td().roll(1, 0)
        out = td.roll(1, 0, inplace=inplace)
        if inplace:
            assert out is td
            assert id(out["nested"]) == id(td["nested"])
        assert out.batch_size == torch.Size([3, 4])
        for key in ref.keys(include_nested=True, leaves_only=True):
            assert torch.equal(out[key], ref[key]), key

    def test_roll_inplace_releases_storage(self):
        td = TensorDict({"a": torch.zeros(4, 5)}, batch_size=[4, 5])
        ref = weakref.ref(td["a"])
        td.roll(1, 0, inplace=True)
        gc.collect()
        assert ref() is None

    @pytest.mark.parametrize("inplace", [True, False])
    def test_gather_inplace(self, inplace):
        td = self._build_nested_td(batch_size=(3, 4))
        index = torch.tensor([[0, 2], [1, 3], [2, 0]])
        ref = self._build_nested_td(batch_size=(3, 4)).gather(dim=1, index=index)
        out = td.gather(dim=1, index=index, inplace=inplace)
        if inplace:
            assert out is td
            assert id(out["nested"]) == id(td["nested"])
        assert out.batch_size == torch.Size([3, 2])
        for key in ref.keys(include_nested=True, leaves_only=True):
            assert torch.equal(out[key], ref[key]), key

    def test_gather_inplace_ndim_mismatch_raises(self):
        td = TensorDict({"a": torch.zeros(3, 4)}, batch_size=[3, 4])
        with pytest.raises(NotImplementedError, match="index.ndim"):
            td.gather(dim=0, index=torch.tensor([0, 1]), inplace=True)

    def test_gather_inplace_out_conflict(self):
        td = TensorDict({"a": torch.zeros(3, 4)}, batch_size=[3, 4])
        with pytest.raises(ValueError, match="mutually exclusive"):
            td.gather(
                dim=0,
                index=torch.zeros(3, 4, dtype=torch.long),
                out=td.clone(),
                inplace=True,
            )

    def test_gather_inplace_lazy_stack_raises(self):
        lst = lazy_stack(
            [TensorDict({"x": torch.zeros(4)}, [4]) for _ in range(3)], dim=0
        )
        with pytest.raises(NotImplementedError, match="gather"):
            lst.gather(dim=0, index=torch.zeros(3, 4, dtype=torch.long), inplace=True)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_reshape_inplace(self, inplace):
        td = self._build_nested_td(batch_size=(3, 4))
        ref = self._build_nested_td(batch_size=(3, 4)).reshape((12,))
        out = td.reshape((12,), inplace=inplace)
        if inplace:
            assert out is td
        assert out.batch_size == torch.Size([12])
        for key in ref.keys(include_nested=True, leaves_only=True):
            assert torch.equal(out[key], ref[key]), key

    @pytest.mark.parametrize("inplace", [True, False])
    def test_flatten_inplace(self, inplace):
        td = self._build_nested_td(batch_size=(3, 4))
        ref = self._build_nested_td(batch_size=(3, 4)).flatten(0, 1)
        out = td.flatten(0, 1, inplace=inplace)
        if inplace:
            assert out is td
        assert out.batch_size == torch.Size([12])
        for key in ref.keys(include_nested=True, leaves_only=True):
            assert torch.equal(out[key], ref[key]), key

    def test_flatten_inplace_lazy_stack_raises(self):
        lst = lazy_stack(
            [TensorDict({"x": torch.zeros(4)}, [4]) for _ in range(2)], dim=0
        )
        with pytest.raises(NotImplementedError, match="flatten"):
            lst.flatten(0, 1, inplace=True)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_unflatten_inplace(self, inplace):
        td = TensorDict({"a": torch.arange(12).float()}, batch_size=[12])
        out = td.unflatten(0, (3, 4), inplace=inplace)
        if inplace:
            assert out is td
        assert out.batch_size == torch.Size([3, 4])
        assert torch.equal(out["a"], torch.arange(12).view(3, 4).float())

    def test_unflatten_inplace_lazy_stack_raises(self):
        lst = lazy_stack(
            [TensorDict({"x": torch.zeros(4)}, [4]) for _ in range(3)], dim=0
        )
        with pytest.raises(NotImplementedError, match="unflatten"):
            lst.unflatten(0, (3, 1), inplace=True)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_contiguous_inplace(self, inplace):
        # Build a TD whose leaf is a non-contiguous view of a larger tensor.
        big = torch.arange(20).view(4, 5).float()
        td = TensorDict({"a": big[:, ::2]}, batch_size=[4, 3])
        assert not td["a"].is_contiguous()
        old_ptr = td["a"].data_ptr()
        out = td.contiguous(inplace=inplace)
        if inplace:
            assert out is td
        assert out["a"].is_contiguous()
        assert out["a"].data_ptr() != old_ptr
        assert torch.equal(out["a"], big[:, ::2].contiguous())

    def test_contiguous_inplace_lazy_stack_raises(self):
        lst = lazy_stack(
            [TensorDict({"x": torch.zeros(4)}, [4]) for _ in range(2)], dim=0
        )
        with pytest.raises(NotImplementedError, match="contiguous"):
            lst.contiguous(inplace=True)

    def test_pad_sequence_nontensor(self):
        d1 = TensorDict({"a": torch.tensor([1, 1]), "b": "asd"})
        d2 = TensorDict({"a": torch.tensor([2]), "b": "efg"})
        d = pad_sequence([d1, d2])
        assert (d["a"] == torch.tensor([[1, 1], [2, 0]])).all()
        assert d["b"] == ["asd", "efg"]

    def test_pad_sequence_single_nontensor(self):
        d1 = TensorDict({"a": torch.tensor([1, 1]), "b": "asd"})
        d = pad_sequence([d1])
        assert (d["a"] == torch.tensor([[1, 1]])).all()
        assert d["b"] == ["asd"]
        assert isinstance(d.get("b"), NonTensorStack)

    def test_pad_sequence_tensorclass_nontensor(self):
        @tensorclass
        class Sample:
            a: torch.Tensor
            b: str

        d1 = Sample(**{"a": torch.tensor([1, 1]), "b": "asd"}, batch_size=[])
        d2 = Sample(**{"a": torch.tensor([2]), "b": "efg"}, batch_size=[])
        d = pad_sequence([d1, d2])
        assert (d.a == torch.tensor([[1, 1], [2, 0]])).all()
        assert d.b == ["asd", "efg"]

    @pytest.mark.parametrize("make_mask", [True, ("bibbidi", "bobbidi", "boo"), False])
    @pytest.mark.parametrize("pad_val", [0, -1])
    def test_pad_sequence_pad_dim0(self, make_mask, pad_val):
        pad_dim = 0
        list_td = [
            TensorDict(
                {"a": torch.ones((2, 8, 8)), ("b", "c"): torch.ones((2, 3))}, [2]
            ),
            TensorDict(
                {"a": torch.full((4, 8, 8), 2), ("b", "c"): torch.full((4, 3), 2)},
                [4],
            ),
        ]
        padded_td = pad_sequence(
            list_td, pad_dim=pad_dim, return_mask=make_mask, padding_value=pad_val
        )
        assert padded_td.shape == torch.Size(
            [2, 4]
        )  # check the shape of the padded tensordict
        assert torch.all(
            padded_td["a"][0, :2, :, :] == 1
        )  # check the values of the first tensor
        assert torch.all(
            padded_td["a"][1, :, :, :] == 2
        )  # check the values of the second tensor
        assert padded_td["a"].shape == torch.Size(
            [2, 4, 8, 8]
        )  # check the shape of the padded tensor
        assert torch.all(padded_td["a"][0, 2:, :, :] == pad_val)  # check the padding
        assert padded_td["b", "c"].shape == torch.Size(
            [2, 4, 3]
        )  # check the shape of the padded tensor
        assert torch.all(padded_td["b", "c"][0, 2:, :] == pad_val)  # check the padding
        if make_mask:
            masks_key = "masks"
            if not isinstance(make_mask, bool):
                masks_key = make_mask
            padded_td_without_masks = pad_sequence(
                list_td, pad_dim=pad_dim, return_mask=False, padding_value=pad_val
            )
            assert masks_key in padded_td.keys(True)
            assert set(
                padded_td_without_masks.keys(include_nested=True, leaves_only=True)
            ) == set(padded_td[masks_key].keys(include_nested=True, leaves_only=True))
            assert not padded_td[masks_key, "a"].all()
            assert padded_td[masks_key, "a"].ndim == pad_dim + 2
            assert (padded_td["a"][padded_td[masks_key, "a"]] != pad_val).all()
            assert (padded_td["a"][~padded_td[masks_key, "a"]] == pad_val).all()
            assert not padded_td[masks_key, "b", "c"].all()
            assert padded_td[masks_key, "b", "c"].ndim == pad_dim + 2
            assert (
                padded_td["b", "c"][padded_td[masks_key, "b", "c"]] != pad_val
            ).all()
            assert (
                padded_td["b", "c"][~padded_td[masks_key, "b", "c"]] == pad_val
            ).all()
        else:
            assert "masks" not in padded_td.keys()

    @pytest.mark.parametrize("make_mask", [True, False])
    def test_pad_sequence_pad_dim1(self, make_mask):
        pad_dim = 1
        list_td = [
            TensorDict(
                {"a": torch.ones((6, 3, 8)), ("b", "c"): torch.ones((6, 3))}, [6]
            ),
            TensorDict(
                {"a": torch.full((6, 5, 8), 2), ("b", "c"): torch.full((6, 7), 2)},
                [6],
            ),
        ]
        padded_td = pad_sequence(list_td, pad_dim=pad_dim, return_mask=make_mask)
        assert padded_td.shape == torch.Size(
            [2, 6]
        )  # check the shape of the padded tensordict
        assert padded_td["a"].shape == torch.Size(
            [2, 6, 5, 8]
        )  # check the shape of the padded tensor
        assert torch.all(
            padded_td["a"][0, :, :3, :] == 1
        )  # check the values of the first tensor
        assert torch.all(padded_td["a"][0, :, 3:, :] == 0)  # check the padding
        assert torch.all(
            padded_td["a"][1, :, :, :] == 2
        )  # check the values of the second tensor
        assert padded_td["b", "c"].shape == torch.Size(
            [2, 6, 7]
        )  # check the shape of the padded tensor
        assert torch.all(padded_td["b", "c"][0, :, 3:] == 0)  # check the padding
        if isinstance(make_mask, str) or make_mask:
            masks_key = "masks"
            if isinstance(make_mask, str):
                masks_key = make_mask
            padded_td_without_masks = pad_sequence(
                list_td, pad_dim=pad_dim, return_mask=False
            )
            assert masks_key in padded_td.keys()
            assert set(
                padded_td_without_masks.keys(include_nested=True, leaves_only=True)
            ) == set(padded_td[masks_key].keys(include_nested=True, leaves_only=True))
            assert not padded_td[masks_key, "a"].all()
            assert padded_td[masks_key, "a"].ndim == pad_dim + 2
            assert (padded_td["a"][padded_td[masks_key, "a"]] != 0).all()
            assert (padded_td["a"][~padded_td[masks_key, "a"]] == 0).all()
            assert not padded_td[masks_key, "b", "c"].all()
            assert padded_td[masks_key, "b", "c"].ndim == pad_dim + 2
            assert (padded_td["b", "c"][padded_td[masks_key, "b", "c"]] != 0).all()
            assert (padded_td["b", "c"][~padded_td[masks_key, "b", "c"]] == 0).all()
        else:
            assert "masks" not in padded_td.keys()

    @pytest.mark.parametrize("count_duplicates", [False, True])
    def test_param_count(self, count_duplicates):
        td = TensorDict(a=torch.randn(3), b=torch.randn(6))
        td["c"] = td["a"]
        assert len(td._values_list(True, True)) == 3
        if count_duplicates:
            assert td.param_count(count_duplicates=count_duplicates) == 12
        else:
            assert td.param_count(count_duplicates=count_duplicates) == 9

    @pytest.mark.parametrize("device", get_available_devices())
    def test_permute(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.randn(4, 5, 6, 9, device=device),
            "b": torch.randn(4, 5, 6, 7, device=device),
            "c": torch.randn(4, 5, 6, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5, 6), source=d)
        td2 = torch.permute(td1, dims=(2, 1, 0))
        assert td2.shape == torch.Size((6, 5, 4))
        assert td2["a"].shape == torch.Size((6, 5, 4, 9))

        td2 = torch.permute(td1, dims=(-1, -3, -2))
        assert td2.shape == torch.Size((6, 4, 5))
        assert td2["c"].shape == torch.Size((6, 4, 5))

        td2 = torch.permute(td1, dims=(0, 1, 2))
        assert td2["a"].shape == torch.Size((4, 5, 6, 9))

    @pytest.mark.parametrize("device", get_available_devices())
    @set_lazy_legacy(True)
    def test_permute_applied_twice(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.randn(4, 5, 6, 9, device=device),
            "b": torch.randn(4, 5, 6, 7, device=device),
            "c": torch.randn(4, 5, 6, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5, 6), source=d)
        td2 = torch.permute(td1, dims=(2, 1, 0))
        td3 = torch.permute(td2, dims=(2, 1, 0))
        assert td3 is td1
        td1 = TensorDict(batch_size=(4, 5, 6), source=d)
        td2 = torch.permute(td1, dims=(2, 1, 0))
        td3 = torch.permute(td2, dims=(0, 1, 2))
        assert td3 is not td1

    @pytest.mark.parametrize("device", get_available_devices())
    @set_lazy_legacy(True)
    def test_permute_exceptions_legacy(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.randn(4, 5, 6, 7, device=device),
            "b": torch.randn(4, 5, 6, 8, 9, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5, 6), source=d)

        with pytest.raises(RuntimeError):
            td2 = td1.permute(1, 1, 0)
            _ = td2.shape

        with pytest.raises(RuntimeError):
            td2 = td1.permute(3, 2, 1, 0)
            _ = td2.shape

        with pytest.raises(RuntimeError):
            td2 = td1.permute(2, -1, 0)
            _ = td2.shape

        with pytest.raises(IndexError):
            td2 = td1.permute(2, 3, 0)
            _ = td2.shape

        with pytest.raises(IndexError):
            td2 = td1.permute(2, -4, 0)
            _ = td2.shape

        with pytest.raises(RuntimeError):
            td2 = td1.permute(2, 1)
            _ = td2.shape

    @pytest.mark.parametrize("device", get_available_devices())
    @set_lazy_legacy(False)
    def test_permute_exceptions(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.randn(4, 5, 6, 7, device=device),
            "b": torch.randn(4, 5, 6, 8, 9, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5, 6), source=d)

        with pytest.raises(ValueError):
            td1.permute(1, 1, 0)

        with pytest.raises(ValueError):
            td1.permute(3, 2, 1, 0)

        with pytest.raises(ValueError):
            td1.permute(2, -1, 0)

        with pytest.raises(ValueError):
            td1.permute(2, 3, 0)

        with pytest.raises(ValueError):
            td1.permute(2, -4, 0)

        with pytest.raises(ValueError):
            td1.permute(2, 1)

    @pytest.mark.parametrize("device", get_available_devices())
    def test_permute_with_tensordict_operations(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.randn(20, 6, 9, device=device),
            "b": torch.randn(20, 6, 7, device=device),
            "c": torch.randn(20, 6, device=device),
        }
        td1 = TensorDict(batch_size=(20, 6), source=d).view(4, 5, 6).permute(2, 1, 0)
        assert td1.shape == torch.Size((6, 5, 4))

        d = {
            "a": torch.randn(4, 5, 6, 7, 9, device=device),
            "b": torch.randn(4, 5, 6, 7, 7, device=device),
            "c": torch.randn(4, 5, 6, 7, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5, 6, 7), source=d)[
            :, :, :, torch.tensor([1, 2])
        ].permute(3, 2, 1, 0)
        assert td1.shape == torch.Size((2, 6, 5, 4))

        d = {
            "a": torch.randn(4, 5, 9, device=device),
            "b": torch.randn(4, 5, 7, device=device),
            "c": torch.randn(4, 5, device=device),
        }
        td1 = stack_td(
            [TensorDict(batch_size=(4, 5), source=d).clone() for _ in range(6)],
            2,
            contiguous=False,
        ).permute(2, 1, 0)
        assert td1.shape == torch.Size((6, 5, 4))

    @pytest.mark.parametrize("device", get_available_devices())
    def test_movedim(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.randn(4, 5, 6, 9, device=device),
            "b": torch.randn(4, 5, 6, 7, device=device),
            "c": torch.randn(4, 5, 6, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5, 6), source=d)

        # Test single dim move
        td2 = td1.movedim(0, 2)
        assert td2.shape == torch.Size((5, 6, 4))
        assert td2["a"].shape == torch.Size((5, 6, 4, 9))

        # Verify matches torch.movedim behavior
        assert td2["a"].shape == torch.movedim(d["a"], 0, 2).shape

        # Test with negative indices
        td3 = td1.movedim(-1, 0)
        assert td3.shape == torch.Size((6, 4, 5))
        assert td3["c"].shape == torch.Size((6, 4, 5))

        # Test torch.movedim
        td4 = torch.movedim(td1, 0, 2)
        assert td4.shape == torch.Size((5, 6, 4))

        # Test tuple dims
        td5 = td1.movedim((0, 1), (2, 0))
        assert td5.shape == torch.Size((5, 6, 4))
        t_ref = torch.movedim(d["a"], (0, 1), (2, 0))
        assert td5["a"].shape == t_ref.shape

        # Test identity
        td6 = td1.movedim(1, 1)
        assert td6 is td1

        # Test moveaxis alias
        td7 = td1.moveaxis(0, 2)
        assert td7.shape == torch.Size((5, 6, 4))

        td8 = torch.moveaxis(td1, 0, 2)
        assert td8.shape == torch.Size((5, 6, 4))

    @pytest.mark.parametrize("device", get_available_devices())
    def test_movedim_exceptions(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.randn(4, 5, 6, 7, device=device),
            "b": torch.randn(4, 5, 6, 8, 9, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5, 6), source=d)

        # Test out of range
        with pytest.raises(IndexError):
            td1.movedim(0, 5)

        with pytest.raises(IndexError):
            td1.movedim(5, 0)

        # Test duplicate source
        with pytest.raises(RuntimeError, match="repeated dim in source"):
            td1.movedim((0, 0), (1, 2))

        # Test duplicate destination
        with pytest.raises(RuntimeError, match="repeated dim in destination"):
            td1.movedim((0, 1), (2, 2))

        # Test mismatched lengths
        with pytest.raises(ValueError, match="same number of elements"):
            td1.movedim((0,), (1, 2))

    @pytest.mark.parametrize("device", get_available_devices())
    def test_movedim_with_tensordict_operations(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.randn(20, 6, 9, device=device),
            "b": torch.randn(20, 6, 7, device=device),
            "c": torch.randn(20, 6, device=device),
        }
        # Test with view
        td1 = TensorDict(batch_size=(20, 6), source=d).view(4, 5, 6).movedim(0, 2)
        assert td1.shape == torch.Size((5, 6, 4))

        # Test with stacked tensordict
        d2 = {
            "a": torch.randn(4, 5, 9, device=device),
            "b": torch.randn(4, 5, 7, device=device),
            "c": torch.randn(4, 5, device=device),
        }
        td2 = stack_td(
            [TensorDict(batch_size=(4, 5), source=d2).clone() for _ in range(6)],
            2,
            contiguous=False,
        ).movedim(0, 2)
        assert td2.shape == torch.Size((5, 6, 4))

    @pytest.mark.parametrize("device", get_available_devices())
    def test_swapaxes(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.randn(4, 5, 6, 9, device=device),
            "b": torch.randn(4, 5, 6, 7, device=device),
            "c": torch.randn(4, 5, 6, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5, 6), source=d)

        # Test swapaxes
        td2 = td1.swapaxes(0, 2)
        assert td2.shape == torch.Size((6, 5, 4))
        assert td2["a"].shape == torch.Size((6, 5, 4, 9))

        # Verify matches torch.swapaxes behavior
        assert td2["a"].shape == torch.swapaxes(d["a"], 0, 2).shape

        # Test torch.swapaxes
        td3 = torch.swapaxes(td1, 0, 2)
        assert td3.shape == torch.Size((6, 5, 4))

        # Test swapdims alias
        td4 = td1.swapdims(0, 2)
        assert td4.shape == torch.Size((6, 5, 4))

        td5 = torch.swapdims(td1, 0, 2)
        assert td5.shape == torch.Size((6, 5, 4))

        # Test negative indices
        td6 = td1.swapaxes(-1, -3)
        assert td6.shape == torch.Size((6, 5, 4))

    @pytest.mark.parametrize("device", get_available_devices())
    def test_flip(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.arange(24, device=device).view(2, 3, 4),
            "b": torch.arange(6, device=device).view(2, 3),
        }
        td1 = TensorDict(batch_size=(2, 3), source=d)

        # Test flip single dim
        td2 = td1.flip(0)
        assert td2.shape == torch.Size((2, 3))
        assert (td2["b"] == torch.flip(d["b"], [0])).all()

        # Test flip multiple dims
        td3 = td1.flip((0, 1))
        assert td3.shape == torch.Size((2, 3))
        assert (td3["b"] == torch.flip(d["b"], [0, 1])).all()

        # Test torch.flip
        td4 = torch.flip(td1, (0,))
        assert td4.shape == torch.Size((2, 3))
        assert (td4["b"] == torch.flip(d["b"], [0])).all()

        # Test negative indices
        td5 = td1.flip(-1)
        assert td5.shape == torch.Size((2, 3))
        assert (td5["b"] == torch.flip(d["b"], [-1])).all()

    @pytest.mark.parametrize("device", get_available_devices())
    def test_fliplr_flipud(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.arange(24, device=device).view(2, 3, 4),
            "b": torch.arange(6, device=device).view(2, 3),
        }
        td1 = TensorDict(batch_size=(2, 3), source=d)

        # Test fliplr
        td2 = td1.fliplr()
        assert td2.shape == torch.Size((2, 3))
        assert (td2["b"] == torch.fliplr(d["b"])).all()

        # Test torch.fliplr
        td3 = torch.fliplr(td1)
        assert td3.shape == torch.Size((2, 3))
        assert (td3["b"] == torch.fliplr(d["b"])).all()

        # Test flipud
        td4 = td1.flipud()
        assert td4.shape == torch.Size((2, 3))
        assert (td4["b"] == torch.flipud(d["b"])).all()

        # Test torch.flipud
        td5 = torch.flipud(td1)
        assert td5.shape == torch.Size((2, 3))
        assert (td5["b"] == torch.flipud(d["b"])).all()

        # Test fliplr requires at least 2 dims
        td_1d = TensorDict({"a": torch.randn(3)}, batch_size=[3])
        with pytest.raises(RuntimeError, match="requires at least 2"):
            td_1d.fliplr()

    @pytest.mark.parametrize("device", get_available_devices())
    def test_roll(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.arange(24, device=device).view(2, 3, 4),
            "b": torch.arange(6, device=device).view(2, 3),
        }
        td1 = TensorDict(batch_size=(2, 3), source=d)

        # Test roll single dim
        td2 = td1.roll(1, 0)
        assert td2.shape == torch.Size((2, 3))
        assert (td2["b"] == torch.roll(d["b"], 1, 0)).all()

        # Test roll multiple dims
        td3 = td1.roll((1, 2), (0, 1))
        assert td3.shape == torch.Size((2, 3))
        assert (td3["b"] == torch.roll(d["b"], (1, 2), (0, 1))).all()

        # Test torch.roll
        td4 = torch.roll(td1, 1, 0)
        assert td4.shape == torch.Size((2, 3))
        assert (td4["b"] == torch.roll(d["b"], 1, 0)).all()

        # Test negative shifts
        td5 = td1.roll(-1, 0)
        assert (td5["b"] == torch.roll(d["b"], -1, 0)).all()

    @pytest.mark.parametrize("device", get_available_devices())
    def test_rot90(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.arange(24, device=device).view(2, 3, 4),
            "b": torch.arange(6, device=device).view(2, 3),
        }
        td1 = TensorDict(batch_size=(2, 3), source=d)

        # Test rot90
        td2 = td1.rot90()
        assert td2.shape == torch.Size((3, 2))
        assert (td2["b"] == torch.rot90(d["b"])).all()

        # Test rot90 k times
        td3 = td1.rot90(2)
        assert td3.shape == torch.Size((2, 3))
        assert (td3["b"] == torch.rot90(d["b"], 2)).all()

        td4 = td1.rot90(3)
        assert td4.shape == torch.Size((3, 2))
        assert (td4["b"] == torch.rot90(d["b"], 3)).all()

        # Test torch.rot90
        td5 = torch.rot90(td1, 1, (0, 1))
        assert td5.shape == torch.Size((3, 2))
        assert (td5["b"] == torch.rot90(d["b"], 1, (0, 1))).all()

        # Test rot90 requires at least 2 dims
        td_1d = TensorDict({"a": torch.randn(3)}, batch_size=[3])
        with pytest.raises(RuntimeError, match="requires at least 2"):
            td_1d.rot90()

    @pytest.mark.parametrize("device", get_available_devices())
    def test_narrow(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.arange(24, device=device).view(2, 3, 4),
            "b": torch.arange(6, device=device).view(2, 3),
        }
        td1 = TensorDict(batch_size=(2, 3), source=d)

        # Test narrow
        td2 = td1.narrow(0, 0, 1)
        assert td2.shape == torch.Size((1, 3))
        assert (td2["b"] == torch.narrow(d["b"], 0, 0, 1)).all()

        td3 = td1.narrow(1, 1, 2)
        assert td3.shape == torch.Size((2, 2))
        assert (td3["b"] == torch.narrow(d["b"], 1, 1, 2)).all()

        # Test torch.narrow
        td4 = torch.narrow(td1, 0, 0, 1)
        assert td4.shape == torch.Size((1, 3))
        assert (td4["b"] == torch.narrow(d["b"], 0, 0, 1)).all()

        # Test negative dim
        td5 = td1.narrow(-1, 0, 2)
        assert td5.shape == torch.Size((2, 2))

    @pytest.mark.parametrize("device", get_available_devices())
    def test_tile(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.arange(24, device=device).view(2, 3, 4),
            "b": torch.arange(6, device=device).view(2, 3),
        }
        td1 = TensorDict(batch_size=(2, 3), source=d)

        # Test tile
        td2 = td1.tile((2, 1))
        assert td2.shape == torch.Size((4, 3))
        assert (td2["b"] == torch.tile(d["b"], (2, 1))).all()

        # Test tile with more dims
        td3 = td1.tile((2, 2))
        assert td3.shape == torch.Size((4, 6))
        assert (td3["b"] == torch.tile(d["b"], (2, 2))).all()

        # Test torch.tile
        td4 = torch.tile(td1, (2, 1))
        assert td4.shape == torch.Size((4, 3))
        assert (td4["b"] == torch.tile(d["b"], (2, 1))).all()

    @pytest.mark.parametrize("device", get_available_devices())
    def test_broadcast_to(self, device):
        torch.manual_seed(1)
        d = {
            "a": torch.arange(6, device=device).view(2, 3),
            "b": torch.arange(6, device=device).view(2, 3),
        }
        td1 = TensorDict(batch_size=(2, 3), source=d)

        # Test broadcast_to with same shape
        td2 = td1.broadcast_to((2, 3))
        assert td2.shape == torch.Size((2, 3))

        # Test broadcast_to with expanded shape
        td3 = td1.broadcast_to((4, 2, 3))
        assert td3.shape == torch.Size((4, 2, 3))
        assert (td3["a"] == torch.broadcast_to(d["a"], (4, 2, 3))).all()

        # Test torch.broadcast_to
        td4 = torch.broadcast_to(td1, (4, 2, 3))
        assert td4.shape == torch.Size((4, 2, 3))

    @pytest.mark.parametrize("device", get_available_devices())
    def test_atleast_nd(self, device):
        torch.manual_seed(1)

        # Test atleast_1d
        td0 = TensorDict({"a": torch.randn(3, device=device)}, batch_size=[])
        td1 = td0.atleast_1d()
        assert td1.ndim == 1
        assert td1.shape == torch.Size([1])

        td_1d = TensorDict({"a": torch.randn(3, device=device)}, batch_size=[3])
        assert td_1d.atleast_1d() is td_1d  # No change needed

        # Test atleast_2d
        td2 = td_1d.atleast_2d()
        assert td2.ndim == 2
        assert td2.shape == torch.Size([1, 3])

        td_2d = TensorDict({"a": torch.randn(2, 3, device=device)}, batch_size=[2, 3])
        assert td_2d.atleast_2d() is td_2d  # No change needed

        # Test atleast_3d
        td3 = td_1d.atleast_3d()
        assert td3.ndim == 3
        assert td3.shape == torch.Size([1, 1, 3])

        td3_from_2d = td_2d.atleast_3d()
        assert td3_from_2d.ndim == 3
        assert td3_from_2d.shape == torch.Size([1, 2, 3])

        td_3d = TensorDict(
            {"a": torch.randn(2, 3, 4, device=device)}, batch_size=[2, 3, 4]
        )
        assert td_3d.atleast_3d() is td_3d  # No change needed

        # Test torch.atleast_*d
        assert torch.atleast_1d(td0).shape == torch.Size([1])
        assert torch.atleast_2d(td_1d).shape == torch.Size([1, 3])
        assert torch.atleast_3d(td_1d).shape == torch.Size([1, 1, 3])

    @pytest.mark.parametrize("device", get_available_devices())
    def test_requires_grad(self, device):
        torch.manual_seed(1)
        # Just one of the tensors have requires_grad
        tensordicts = [
            TensorDict(
                batch_size=[11, 12],
                source={
                    "key1": torch.randn(
                        11,
                        12,
                        5,
                        device=device,
                        requires_grad=True if i == 5 else False,
                    ),
                    "key2": torch.zeros(
                        11, 12, 50, device=device, dtype=torch.bool
                    ).bernoulli_(),
                },
            )
            for i in range(10)
        ]
        stacked_td = LazyStackedTensorDict(*tensordicts, stack_dim=0)
        # First stacked tensor has requires_grad == True
        assert list(stacked_td.values())[0].requires_grad is True

    def test_refine_names_setitem_subtd(self):
        batch_size = 1
        seq_len = 2
        n_agents = 3
        td = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "obs": torch.zeros((batch_size, seq_len, n_agents, 5)),
                        "dones": torch.zeros((batch_size, seq_len, n_agents, 1)),
                    },
                    batch_size=(batch_size, seq_len, n_agents),
                    names=[None, "time", "other"],
                ),
                "dones": torch.zeros((batch_size, seq_len)),
            },
            batch_size=(batch_size, seq_len),
            names=[None, "time"],
        )
        #
        td["agents"] = td["agents"].repeat_interleave(2, dim=-1)
        assert len(td["agents"].names) == 3
        assert td["agents"].names[-1] == "other"
        td["agents"] = td["agents"].repeat(1, 1, 2)
        assert td["agents"].names[-1] == "other"
        td["agents"] = torch.cat((td["agents"], td["agents"]), dim=2)
        assert td["agents"].names[-1] == "other"

    def test_rename_key_nested(self):
        td = TensorDict(a={"b": {"c": 0}})
        td.rename_key_(("a", "b", "c"), ("a", "b"))
        assert td["a", "b"] == 0

    @set_list_to_stack(True)
    @set_capture_non_tensor_stack(False)
    @pytest.mark.parametrize("like", [True, False])
    def test_save_load_memmap_stacked_td(
        self,
        like,
        tmpdir,
    ):
        a = TensorDict({"a": [1]}, [])
        b = TensorDict({"b": [1]}, [])
        c = LazyStackedTensorDict.lazy_stack([a, b])
        c = c.expand(10, 2)
        if like:
            d = c.memmap_like(prefix=tmpdir)
        else:
            d = c.memmap_(prefix=tmpdir)

        d2 = LazyStackedTensorDict.load_memmap(tmpdir)
        assert (d2 == d).all()
        assert (d2[:, 0] == d[:, 0]).all()
        if like:
            assert (d2[:, 0] == a.zero_()).all()
        else:
            assert (d2[:, 0] == a).all()

    @pytest.mark.parametrize("inplace", [True, False])
    def test_select_nested(self, inplace):
        tensor_1 = torch.rand(4, 5, 6, 7)
        tensor_2 = torch.rand(4, 5, 6, 7)
        sub_sub_tensordict = TensorDict(
            {"t1": tensor_1, "t2": tensor_2}, batch_size=[4, 5, 6]
        )
        sub_tensordict = TensorDict(
            {"double_nested": sub_sub_tensordict}, batch_size=[4, 5]
        )
        tensordict = TensorDict(
            {
                "a": torch.rand(4, 3),
                "b": torch.rand(4, 2),
                "c": torch.rand(4, 1),
                "nested": sub_tensordict,
            },
            batch_size=[4],
        )

        selected = tensordict.select(
            "b", ("nested", "double_nested", "t2"), inplace=inplace
        )

        assert set(selected.keys(include_nested=True)) == {
            "b",
            "nested",
            ("nested", "double_nested"),
            ("nested", "double_nested", "t2"),
        }

        if inplace:
            assert selected is tensordict
            assert set(tensordict.keys(include_nested=True)) == {
                "b",
                "nested",
                ("nested", "double_nested"),
                ("nested", "double_nested", "t2"),
            }
        else:
            assert selected is not tensordict
            assert set(tensordict.keys(include_nested=True)) == {
                "a",
                "b",
                "c",
                "nested",
                ("nested", "double_nested"),
                ("nested", "double_nested", "t1"),
                ("nested", "double_nested", "t2"),
            }

    @set_list_to_stack(True)
    @set_capture_non_tensor_stack(False)
    def test_select_nested_missing(self):
        # checks that we keep a nested key even if missing nested keys are present
        td = TensorDict({"a": {"b": [1], "c": [2]}}, [])
        td_select = td.select(("a", "b"), "r", ("a", "z"), strict=False)
        assert ("a", "b") in list(td_select.keys(True, True))
        assert ("a", "b") in td_select.keys(True, True)

    def test_separates(self):
        td = TensorDict(a=0, b=TensorDict(c=0, d=0))
        td_sep = td.separates("a", ("b", ("d",)))
        assert "a" in td_sep
        assert "a" not in td
        assert ("b", "d") in td_sep
        assert ("b", "d") not in td
        with pytest.raises(KeyError):
            td = TensorDict(a=0, b=TensorDict(c=0, d=0))
            td_sep = td.separates("a", ("b", ("d",)), "e")
        td = TensorDict(a=0, b=TensorDict(c=0, d=0))
        td_sep = td.separates("a", ("b", ("d",)), "e", default=None)
        assert td_sep["e"] is None
        td = TensorDict(a=0, b=TensorDict(c=0, d=0), unique=TensorDict(val=0))
        td_sep = td.separates("a", ("b", ("d",)), "e", ("unique", "val"), default=None)
        assert "unique" not in td
        assert "unique" in td_sep
        td = TensorDict(a=0, b=TensorDict(c=0, d=0), unique=TensorDict(val=0))
        td_sep = td.separates(
            "a", ("b", ("d",)), "e", ("unique", "val"), default=None, filter_empty=False
        )
        assert "unique" in td
        assert "unique" in td_sep

    @set_list_to_stack(True)
    def test_set_list_nontensor(self):
        td = TensorDict(a=["a string", "another string"], batch_size=2)
        assert td["a"] == ["a string", "another string"]
        assert td[0]["a"] == "a string"
        assert td[1]["a"] == "another string"

    @set_list_to_stack(True)
    def test_set_list_tensor(self):
        td = TensorDict(a=[torch.zeros(()), torch.ones(())], batch_size=2)
        assert (td["a"] == torch.tensor([0, 1])).all()
        assert td[0]["a"] == 0
        assert td[1]["a"] == 1

    @set_list_to_stack(True)
    def test_set_list_mixed(self):
        td = TensorDict(a=[torch.zeros(()), "another string"], batch_size=2)
        assert td[0]["a"] == 0
        assert td[1]["a"] == "another string"

    @set_list_to_stack(True)
    def test_set_list_empty(self):
        td = TensorDict()
        td["key"] = []
        assert isinstance(td["key"], torch.Tensor)
        assert td["key"].shape == torch.Size([0])

    @pytest.fixture
    def _set_list_none(self):
        import tensordict.utils

        v = tensordict.utils._LIST_TO_STACK
        tensordict.utils._LIST_TO_STACK = None
        yield
        tensordict.utils._LIST_TO_STACK = v

    def test_set_nested_keys(self):
        tensor = torch.randn(4, 5, 6, 7)
        tensor2 = torch.ones(4, 5, 6, 7)
        tensordict = TensorDict({}, [4])
        sub_tensordict = TensorDict({}, [4, 5])
        sub_sub_tensordict = TensorDict({"c": tensor}, [4, 5, 6])
        sub_sub_tensordict2 = TensorDict({"c": tensor2}, [4, 5, 6])
        sub_tensordict.set("b", sub_sub_tensordict)
        tensordict.set("a", sub_tensordict)
        assert tensordict.get(("a", "b")) is sub_sub_tensordict

        tensordict.set(("a", "b"), sub_sub_tensordict2)
        assert tensordict.get(("a", "b")) is sub_sub_tensordict2
        assert (tensordict.get(("a", "b", "c")) == 1).all()

    @pytest.mark.parametrize("index0", [None, slice(None)])
    def test_set_sub_key(self, index0):
        # tests that parent tensordict is affected when subtensordict is set with a new key
        batch_size = [10, 10]
        source = {"a": torch.randn(10, 10, 10), "b": torch.ones(10, 10, 2)}
        td = TensorDict(source, batch_size=batch_size)
        idx0 = (index0, 0) if index0 is not None else 0
        td0 = td._get_sub_tensordict(idx0)
        idx = (index0, slice(2, 4)) if index0 is not None else slice(2, 4)
        sub_td = td._get_sub_tensordict(idx)
        if index0 is None:
            c = torch.randn(2, 10, 10)
        else:
            c = torch.randn(10, 2, 10)
        sub_td.set("c", c)
        assert (td.get("c")[idx] == sub_td.get("c")).all()
        assert (sub_td.get("c") == c).all()
        assert (td.get("c")[idx0] == 0).all()
        assert (td._get_sub_tensordict(idx0).get("c") == 0).all()
        assert (td0.get("c") == 0).all()

    def test_setdefault_nested(self):
        tensor = torch.randn(4, 5, 6, 7)
        tensor2 = torch.ones(4, 5, 6, 7)
        sub_sub_tensordict = TensorDict({"c": tensor}, [4, 5, 6])
        sub_tensordict = TensorDict({"b": sub_sub_tensordict}, [4, 5])
        tensordict = TensorDict({"a": sub_tensordict}, [4])

        # if key exists we return the existing value
        assert tensordict.setdefault(("a", "b", "c"), tensor2) is tensor

        assert tensordict.setdefault(("a", "b", "d"), tensor2) is tensor2
        assert (tensordict["a", "b", "d"] == 1).all()
        assert tensordict.get(("a", "b", "d")) is tensor2

    @pytest.mark.parametrize("lazy_leg", [True, False])
    @pytest.mark.parametrize("shared", [True, False])
    def test_shared_inheritance(self, shared, lazy_leg):
        with set_lazy_legacy(lazy_leg):
            if shared:

                def assert_not_shared(td0):
                    assert not td0.is_shared()
                    assert not td0.is_locked

                def assert_shared(td0):
                    assert td0.is_shared()
                    assert td0.is_locked

            else:

                def assert_not_shared(td0):
                    assert not td0.is_memmap()
                    assert not td0.is_locked

                def assert_shared(td0):
                    assert td0.is_memmap()
                    assert td0.is_locked

            td = TensorDict({"a": torch.randn(3, 4)}, [3, 4])
            if shared:
                td.share_memory_()
            else:
                td.memmap_()

            # Key-based operations propagate the shared status
            td0 = td.exclude("a")
            assert_not_shared(td0)

            td0 = td.select("a")
            assert_not_shared(td0)

            with td.unlock_():
                td0 = td.rename_key_("a", "a.a")
            if not shared:
                assert not td0.is_memmap()
            else:
                assert not td0.is_shared()
            assert td.is_locked
            if not shared:
                assert not td.is_memmap()
            else:
                assert not td.is_shared()
            if shared:
                td.share_memory_()
            else:
                td.memmap_()

            td0 = td.unflatten_keys(".")
            assert_not_shared(td0)

            td0 = td.flatten_keys(".")
            assert_not_shared(td0)

            # Shape operations propagate the shared status
            td0, *_ = td.unbind(1)
            assert_shared(td0)

            td0, *_ = td.split(1, 0)
            assert_shared(td0)

            td0 = td.view(-1)
            assert_shared(td0)

            td0 = td.permute(1, 0)
            assert_shared(td0)

            td0 = td.unsqueeze(0)
            assert_shared(td0)

            td0 = td0.squeeze(0)
            assert_shared(td0)

    def test_sorted_keys(self):
        td = TensorDict(
            {
                "a": {"b": 0, "c": 1},
                "d": 2,
                "e": {"f": 3, "g": {"h": 4, "i": 5}, "j": 6},
            }
        )
        tdflat = td.flatten_keys()
        tdflat["d"] = tdflat.pop("d")
        tdflat["a.b"] = tdflat.pop("a.b")
        for key1, key2 in zip(
            td.keys(True, True, sort=True), tdflat.keys(True, True, sort=True)
        ):
            if isinstance(key1, str):
                assert key1 == key2
            else:
                assert ".".join(key1) == key2
        for v1, v2 in zip(
            td.values(True, True, sort=True), tdflat.values(True, True, sort=True)
        ):
            assert v1 == v2
        for (k1, v1), (k2, v2) in zip(
            td.items(True, True, sort=True), tdflat.items(True, True, sort=True)
        ):
            if isinstance(k1, str):
                assert k1 == k2
            else:
                assert ".".join(k1) == k2
            assert v1 == v2

    def test_split_keys(self):
        td = TensorDict(
            {
                "a": 0,
                "b": {"c": 1},
                "d": {"e": "a string!", "f": 2},
                "g": 3,
            }
        )
        td0, td1, td3 = td.split_keys(["b"], ["d"])
        assert set(td0.keys()) == {"b"}
        assert set(td1.keys()) == {"d"}
        assert "b" not in td3
        assert "d" not in td3
        assert "a" in td3
        assert "g" in td3
        assert td3 is not td
        td0, td1, td3 = td.split_keys(
            [("b", "c")], [("d", "e"), ("d", "f")], inplace=True
        )
        assert ("b", "c") in td0
        assert ("d", "e") in td1
        assert ("d", "f") in td1
        assert td is td3
        assert "d" not in td

    @pytest.mark.parametrize("sign", ["plus", "minus"])
    def test_split_uneven(self, sign):
        a = torch.arange(6).unsqueeze(-1).expand(6, 3)
        b = torch.arange(18).view(6, 3)
        c = torch.arange(36).view(6, 3, 2)
        td = TensorDict({"a": a, "b": b, "c": c}, [6, 3])

        if sign == "plus":
            tds = td.split(5, 0)
        else:
            tds = td.split(5, -2)
        assert tds[0].shape == torch.Size([5, 3])
        assert tds[1].shape == torch.Size([1, 3])
        assert tds[0]["a"].shape == torch.Size([5, 3])
        assert tds[1]["a"].shape == torch.Size([1, 3])
        assert tds[0]["b"].shape == torch.Size([5, 3])
        assert tds[1]["b"].shape == torch.Size([1, 3])
        assert tds[0]["c"].shape == torch.Size([5, 3, 2])
        assert tds[1]["c"].shape == torch.Size([1, 3, 2])
        if sign == "plus":
            tds = td.split(2, 1)
        else:
            tds = td.split(2, -1)
        assert tds[0].shape == torch.Size([6, 2])
        assert tds[1].shape == torch.Size([6, 1])
        assert tds[0]["a"].shape == torch.Size([6, 2])
        assert tds[1]["a"].shape == torch.Size([6, 1])
        assert tds[0]["b"].shape == torch.Size([6, 2])
        assert tds[1]["b"].shape == torch.Size([6, 1])
        assert tds[0]["c"].shape == torch.Size([6, 2, 2])
        assert tds[1]["c"].shape == torch.Size([6, 1, 2])

    def test_setitem_nested(self):
        tensor = torch.randn(4, 5, 6, 7)
        tensor2 = torch.ones(4, 5, 6, 7)
        tensordict = TensorDict({}, [4])
        sub_tensordict = TensorDict({}, [4, 5])
        sub_sub_tensordict = TensorDict({"c": tensor}, [4, 5, 6])
        sub_sub_tensordict2 = TensorDict({"c": tensor2}, [4, 5, 6])
        sub_tensordict["b"] = sub_sub_tensordict
        tensordict["a"] = sub_tensordict
        assert tensordict["a", "b"] is sub_sub_tensordict
        tensordict["a", "b"] = sub_sub_tensordict2
        assert tensordict["a", "b"] is sub_sub_tensordict2
        assert (tensordict["a", "b", "c"] == 1).all()

        # check the same with set method
        sub_tensordict.set("b", sub_sub_tensordict)
        tensordict.set("a", sub_tensordict)
        assert tensordict["a", "b"] is sub_sub_tensordict

        tensordict.set(("a", "b"), sub_sub_tensordict2)
        assert tensordict["a", "b"] is sub_sub_tensordict2
        assert (tensordict["a", "b", "c"] == 1).all()

    def test_split_with_empty_tensordict(self):
        td = TensorDict({}, [10])

        tds = td.split(4, 0)
        assert len(tds) == 3
        assert tds[0].shape == torch.Size([4])
        assert tds[1].shape == torch.Size([4])
        assert tds[2].shape == torch.Size([2])

        tds = td.split([1, 9], 0)

        assert len(tds) == 2
        assert tds[0].shape == torch.Size([1])
        assert tds[1].shape == torch.Size([9])

        td = TensorDict({}, [10, 10, 3])

        tds = td.split(4, 1)
        assert len(tds) == 3
        assert tds[0].shape == torch.Size([10, 4, 3])
        assert tds[1].shape == torch.Size([10, 4, 3])
        assert tds[2].shape == torch.Size([10, 2, 3])

        tds = td.split([1, 9], 1)
        assert len(tds) == 2
        assert tds[0].shape == torch.Size([10, 1, 3])
        assert tds[1].shape == torch.Size([10, 9, 3])

    def test_split_with_invalid_arguments(self):
        td = TensorDict({"a": torch.zeros(2, 1)}, [])
        # Test empty batch size
        with pytest.raises(IndexError, match="Incompatible dim"):
            td.split(1, 0)

        td = TensorDict({}, [3, 2])

        # Test invalid split_size input
        with pytest.raises(TypeError, match="must be int or list of ints"):
            td.split("1", 0)
        with pytest.raises(TypeError, match="must be int or list of ints"):
            td.split(["1", 2], 0)

        # Test invalid split_size sum
        with pytest.raises(
            RuntimeError, match="Insufficient number of elements in split_size"
        ):
            td.split([], 0)

        with pytest.raises(RuntimeError, match="expects split_size to sum exactly"):
            td.split([1, 1], 0)

        # Test invalid dimension input
        with pytest.raises(IndexError, match="Incompatible dim"):
            td.split(1, 2)
        with pytest.raises(IndexError, match="Incompatible dim"):
            td.split(1, -3)
        with pytest.raises(ValueError, match="split_size must be positive, got 0."):
            td.split(0, -1)
        with pytest.raises(ValueError, match="split_size must be positive, got -1."):
            td.split(-1, -1)

    def test_split_with_negative_dim(self):
        td = TensorDict(
            {"a": torch.zeros(5, 4, 2, 1), "b": torch.zeros(5, 4, 1)}, [5, 4]
        )

        tds = td.split([1, 3], -1)
        assert len(tds) == 2
        assert tds[0].shape == torch.Size([5, 1])
        assert tds[0]["a"].shape == torch.Size([5, 1, 2, 1])
        assert tds[0]["b"].shape == torch.Size([5, 1, 1])
        assert tds[1].shape == torch.Size([5, 3])
        assert tds[1]["a"].shape == torch.Size([5, 3, 2, 1])
        assert tds[1]["b"].shape == torch.Size([5, 3, 1])

    @pytest.mark.parametrize("device", get_available_devices())
    def test_squeeze(self, device):
        torch.manual_seed(1)
        d = {
            "key1": torch.randn(4, 5, 6, device=device),
            "key2": torch.randn(4, 5, 10, device=device),
        }
        td1 = TensorDict(batch_size=(4, 5), source=d)
        td2 = torch.unsqueeze(td1, dim=1)
        assert td2.batch_size == torch.Size([4, 1, 5])

        td1b = torch.squeeze(td2, dim=1)
        assert td1b.batch_size == td1.batch_size

    def test_stack_from_tensordict(self):
        td = TensorDict(
            {"a": torch.zeros(3, 4), "b": {"c": torch.ones(3, 4)}}, batch_size=[3, 4]
        )
        tensor = td.stack_from_tensordict(dim=1)
        assert tensor.shape == (3, 2, 4)
        assert (tensor[:, 0] == 0).all()
        assert (tensor[:, 1] == 1).all()

    def test_stack_names(self):
        td = TensorDict(
            {"a": torch.zeros(3, 4)}, batch_size=[3, 4], names=["first", "second"]
        )
        td2 = torch.stack([td, td], dim=0)
        assert td2.names == [None, "first", "second"]
        td2 = torch.stack([td, td], dim=1)
        assert td2.names == ["first", None, "second"]
        td2 = torch.stack([td, td], dim=-1)
        assert td2.names == ["first", "second", None]

        # Mess with the names
        td_copy = td.clone()
        td_copy.names = ["first", "third"]
        td2 = torch.stack([td, td_copy], dim=0)
        assert td2.names == [None, None, None]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_record_stream(self):
        s0 = torch.cuda.Stream(0)
        s1 = torch.cuda.Stream(0)
        with torch.cuda.stream(s1):
            td = TensorDict(
                {
                    "a": torch.randn(3, device=f"{cur_device}:0"),
                    ("b", "c"): torch.randn(3, device=f"{cur_device}:0"),
                }
            )
            td.record_stream(s1)
        with pytest.raises(
            RuntimeError,
            match="A stream is already associated with this TensorDict instance",
        ):
            td.record_stream(s0)

    @pytest.mark.parametrize(
        "reduction", ["sum", "nansum", "mean", "nanmean", "std", "var", "quantile"]
    )
    def test_reduction_feature(self, reduction):
        td = TensorDict(
            a=torch.ones(3, 4, 5, 6),
            b=TensorDict(
                c=torch.ones(3, 4, 5, 6, 7),
                d=torch.ones(3, 4, 5, 6),
                e=torch.ones(3, 4, 5) if reduction not in ("std", "var") else None,
                none=None,
                batch_size=(3, 4, 5),
            ),
            f=torch.ones(3, 4) if reduction not in ("std", "var") else None,
            none=None,
            batch_size=(3, 4),
        )
        if reduction == "quantile":
            tdr = getattr(td, reduction)(0.5, dim="feature")
        else:
            tdr = getattr(td, reduction)(dim="feature")
        for k, v in td.items(True):
            other = tdr[k]
            if isinstance(v, TensorDict):
                assert v.batch_size == other.batch_size
            elif is_non_tensor(v):
                assert v.data is None
                assert other is None
            elif isinstance(k, str):
                assert other.shape == td.shape
            else:
                assert other.shape == td[k[:-1]].shape

    @pytest.mark.parametrize(
        "reduction", ["sum", "nansum", "mean", "nanmean", "std", "var", "quantile"]
    )
    def test_reduction_feature_full(self, reduction):
        td = TensorDict(
            a=torch.ones(3, 4, 5, 6),
            b=TensorDict(
                c=torch.ones(3, 4, 5, 6, 7),
                d=torch.ones(3, 4, 5, 6),
                e=torch.ones(3, 4, 5) if reduction not in ("std", "var") else None,
                none=None,
                batch_size=(3, 4, 5),
            ),
            f=torch.ones(3, 4) if reduction not in ("std", "var") else None,
            none=None,
            batch_size=(3, 4),
        )
        if reduction == "quantile":
            reduced = getattr(td, reduction)(0.5, dim="feature", reduce=True)
        else:
            reduced = getattr(td, reduction)(dim="feature", reduce=True)
        assert reduced.shape == (3, 4)

        td = TensorDict(
            a=torch.ones(3, 4, 5),
            b=TensorDict(
                c=torch.ones(3, 4, 5),
                d=torch.ones(3, 4, 5),
                batch_size=(3, 4, 5),
            ),
            batch_size=(3, 4),
        )
        if reduction == "quantile":
            assert getattr(td, reduction)(0.5, reduce=True, dim="feature").shape == (
                3,
                4,
            )
            assert getattr(td, reduction)(0.5, reduce=True, dim=1).shape == (3, 5)
        else:
            assert getattr(td, reduction)(reduce=True, dim="feature").shape == (3, 4)
            assert getattr(td, reduction)(reduce=True, dim=1).shape == (3, 5)

    def test_quantile(self):
        """Test quantile reduction functionality."""
        td = TensorDict(
            a=torch.randn(3, 4, 5),
            b=TensorDict(
                c=torch.randn(3, 4, 5, 6),
                d=torch.randn(3, 4, 5),
                batch_size=(3, 4, 5),
            ),
            batch_size=(3, 4),
        )

        # Test median (0.5 quantile)
        median_td = td.quantile(0.5)
        assert median_td.batch_size == torch.Size([])
        assert isinstance(median_td["a"], torch.Tensor)
        assert median_td["a"].shape == torch.Size([])

        # Test median with reduce=True
        median_reduced = td.quantile(0.5, reduce=True)
        assert isinstance(median_reduced, torch.Tensor)
        assert median_reduced.shape == torch.Size([])

        # Test quantile along dimension
        quantile_dim = td.quantile(0.5, dim=0)
        assert quantile_dim.batch_size == torch.Size([4])
        assert quantile_dim["a"].shape == torch.Size([4, 5])

        # Test multiple quantiles
        quantiles = torch.tensor([0.25, 0.5, 0.75])
        multi_quantile = td.quantile(quantiles, dim=0)
        assert multi_quantile.batch_size == torch.Size([4])
        assert multi_quantile["a"].shape == torch.Size([3, 4, 5])

        # Test feature dimension
        quantile_feature = td.quantile(0.5, dim="feature")
        assert quantile_feature.batch_size == torch.Size([3, 4])
        assert quantile_feature["a"].shape == torch.Size([3, 4])

        # Test with keepdim
        quantile_keepdim = td.quantile(0.5, dim=0, keepdim=True)
        assert quantile_keepdim.batch_size == torch.Size([1, 4])
        assert quantile_keepdim["a"].shape == torch.Size([1, 4, 5])

    def test_subclassing(self):
        class SubTD(TensorDict): ...

        t = SubTD(a=torch.randn(3))
        assert isinstance(t + t, SubTD)
        assert isinstance(t / 2, SubTD)
        assert isinstance(2 / t, SubTD)
        assert isinstance(t.to(torch.float), SubTD)
        assert isinstance(t.to("cpu"), SubTD)
        assert isinstance(torch.zeros_like(t), SubTD)
        assert isinstance(t.copy(), SubTD)
        assert isinstance(t.clone(), SubTD)
        assert isinstance(t.empty(), SubTD)
        assert isinstance(t.select(), SubTD)
        assert isinstance(t.exclude("a"), SubTD)
        assert isinstance(t.split_keys({"a"})[0], SubTD)
        assert isinstance(t.flatten_keys(), SubTD)
        assert isinstance(t.unflatten_keys(), SubTD)
        stack = torch.stack([t, t])
        assert isinstance(stack, SubTD)
        assert isinstance(stack[0], SubTD)
        assert isinstance(stack.unbind(0)[0], SubTD)
        assert isinstance(stack.split(1)[0], SubTD)
        assert isinstance(stack.gather(0, torch.ones((1,), dtype=torch.long)), SubTD)
        unsqueeze = stack.unsqueeze(0)
        assert isinstance(unsqueeze, SubTD)
        assert isinstance(unsqueeze.transpose(1, 0), SubTD)
        assert isinstance(unsqueeze.permute(1, 0), SubTD)
        assert isinstance(unsqueeze.squeeze(), SubTD)
        assert isinstance(unsqueeze.reshape(-1), SubTD)
        assert isinstance(unsqueeze.view(-1), SubTD)

    @pytest.mark.parametrize("device", get_available_devices())
    def test_subtensordict_construction(self, device):
        torch.manual_seed(1)
        td = TensorDict(batch_size=(4, 5))
        val1 = torch.randn(4, 5, 1, device=device)
        val2 = torch.randn(4, 5, 6, dtype=torch.double, device=device)
        val1_copy = val1.clone()
        val2_copy = val2.clone()
        td.set("key1", val1)
        td.set("key2", val2)
        std1 = td._get_sub_tensordict(2)
        std2 = std1._get_sub_tensordict(2)
        idx = (2, 2)
        std_control = td._get_sub_tensordict(idx)
        assert (std_control.get("key1") == std2.get("key1")).all()
        assert (std_control.get("key2") == std2.get("key2")).all()

        # write values
        with pytest.raises(RuntimeError, match="is prohibited for existing tensors"):
            std_control.set("key1", torch.randn(1, device=device))
        with pytest.raises(RuntimeError, match="is prohibited for existing tensors"):
            std_control.set("key2", torch.randn(6, device=device, dtype=torch.double))

        subval1 = torch.randn(1, device=device)
        subval2 = torch.randn(6, device=device, dtype=torch.double)
        std_control.set_("key1", subval1)
        std_control.set_("key2", subval2)
        assert (val1_copy[idx] != subval1).all()
        assert (td.get("key1")[idx] == subval1).all()
        assert (td.get("key1")[1, 1] == val1_copy[1, 1]).all()

        assert (val2_copy[idx] != subval2).all()
        assert (td.get("key2")[idx] == subval2).all()
        assert (td.get("key2")[1, 1] == val2_copy[1, 1]).all()

        assert (std_control.get("key1") == std2.get("key1")).all()
        assert (std_control.get("key2") == std2.get("key2")).all()

        assert std_control.get_parent_tensordict() is td
        assert (
            std_control.get_parent_tensordict()
            is std2.get_parent_tensordict().get_parent_tensordict()
        )

    @pytest.mark.parametrize("device", get_available_devices())
    def test_tensordict_device(self, device):
        tensordict = TensorDict({"a": torch.randn(3, 4)}, [])
        assert tensordict.device is None

        tensordict = TensorDict({"a": torch.randn(3, 4, device=device)}, [])
        assert tensordict["a"].device == device
        assert tensordict.device is None

        tensordict = TensorDict(
            {
                "a": torch.randn(3, 4, device=device),
                "b": torch.randn(3, 4),
                "c": torch.randn(3, 4, device="cpu"),
            },
            [],
            device=device,
        )
        assert tensordict.device == device
        assert tensordict["a"].device == device
        assert tensordict["b"].device == device
        assert tensordict["c"].device == device

        tensordict = TensorDict({}, [], device=device)
        tensordict["a"] = torch.randn(3, 4)
        tensordict["b"] = torch.randn(3, 4, device="cpu")
        assert tensordict["a"].device == device
        assert tensordict["b"].device == device

        tensordict = TensorDict(
            {"a": torch.randn(3, 4), "b": {"c": torch.randn(3, 4)}}, []
        )
        tensordict = tensordict.to(device)
        assert tensordict.device == device
        assert tensordict["a"].device == device

        tensordict_cpu = tensordict.to("cpu", inplace=True)
        assert tensordict_cpu.device == torch.device("cpu")
        for v in tensordict_cpu.values(True, True):
            assert v.device == torch.device("cpu")
        assert tensordict_cpu is tensordict
        assert tensordict_cpu["b"] is tensordict["b"]
        assert tensordict_cpu["b"].device == torch.device("cpu")

    @pytest.mark.skipif(
        torch.cuda.device_count() == 0 and npu_device_count == 0,
        reason="no cuda or npu device detected",
    )
    @pytest.mark.parametrize("device", get_available_devices()[1:])
    def test_tensordict_error_messages(self, device):
        sub1 = TensorDict({"a": torch.randn(2, 3)}, [2])
        sub2 = TensorDict({"a": torch.randn(2, 3, device=device)}, [2])
        td1 = TensorDict({"sub": sub1}, [2])
        td2 = TensorDict({"sub": sub2}, [2])

        with pytest.raises(
            RuntimeError, match='tensors on different devices at key "sub" / "a"'
        ):
            torch.cat([td1, td2], 0)

    @pytest.mark.parametrize("device", get_available_devices())
    def test_tensordict_indexing(self, device):
        torch.manual_seed(1)
        td = TensorDict(batch_size=(4, 5))
        td.set("key1", torch.randn(4, 5, 1, device=device))
        td.set("key2", torch.randn(4, 5, 6, device=device, dtype=torch.double))

        td_select = td[2, 2]
        td_select._check_batch_size()

        td_select = td[2, :2]
        td_select._check_batch_size()

        td_select = td[None, :2]
        td_select._check_batch_size()

        td_reconstruct = stack_td(list(td), 0, contiguous=False)
        assert (
            td_reconstruct == td
        ).all(), f"td and td_reconstruct differ, got {td} and {td_reconstruct}"

        superlist = [stack_td(list(_td), 0, contiguous=False) for _td in td]
        td_reconstruct = stack_td(superlist, 0, contiguous=False)
        assert (
            td_reconstruct == td
        ).all(), f"td and td_reconstruct differ, got {td == td_reconstruct}"

        x = torch.randn(4, 5, device=device)
        td = TensorDict(
            source={"key1": torch.zeros(3, 4, 5, device=device)},
            batch_size=[3, 4],
        )
        td[0].set_("key1", x)
        torch.testing.assert_close(td.get("key1")[0], x)
        torch.testing.assert_close(td.get("key1")[0], td[0].get("key1"))

        y = torch.randn(3, 5, device=device)
        td[:, 0].set_("key1", y)
        torch.testing.assert_close(td.get("key1")[:, 0], y)
        torch.testing.assert_close(td.get("key1")[:, 0], td[:, 0].get("key1"))

    def test_tensordict_prealloc_nested(self):
        N = 3
        B = 5
        T = 4
        buffer = TensorDict(batch_size=[B, N])

        td_0 = TensorDict(
            {
                "env.time": torch.rand(N, 1),
                "agent.obs": TensorDict(
                    {  # assuming 3 agents in a multi-agent setting
                        "image": torch.rand(N, T, 64),
                        "state": torch.rand(N, T, 3, 32, 32),
                    },
                    batch_size=[N, T],
                ),
            },
            batch_size=[N],
        )

        td_1 = td_0.clone()
        buffer[0] = td_0
        buffer[1] = td_1
        assert (
            repr(buffer)
            == """TensorDict(
    fields={
        agent.obs: TensorDict(
            fields={
                image: Tensor(shape=torch.Size([5, 3, 4, 64]), device=cpu, dtype=torch.float32, is_shared=False),
                state: Tensor(shape=torch.Size([5, 3, 4, 3, 32, 32]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([5, 3, 4]),
            device=None,
            is_shared=False),
        env.time: Tensor(shape=torch.Size([5, 3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([5, 3]),
    device=None,
    is_shared=False)"""
        )
        assert buffer.batch_size == torch.Size([B, N])
        assert buffer["agent.obs"].batch_size == torch.Size([B, N, T])

    @pytest.mark.parametrize("device", get_available_devices())
    def test_tensordict_set(self, device):
        torch.manual_seed(1)
        td = TensorDict(batch_size=(4, 5), device=device)
        td.set("key1", torch.randn(4, 5))
        assert td.device == torch.device(device)
        # by default inplace:
        with pytest.raises(RuntimeError):
            td.set("key1", torch.randn(5, 5, device=device))

        # robust to dtype casting
        td.set_("key1", torch.ones(4, 5, device=device, dtype=torch.double))
        assert (td.get("key1") == 1).all()

        # robust to device casting
        td.set("key_device", torch.ones(4, 5, device="cpu", dtype=torch.double))
        assert td.get("key_device").device == torch.device(device)

        with pytest.raises(KeyError, match="not found in TensorDict with keys"):
            td.set_("smartypants", torch.ones(4, 5, device="cpu", dtype=torch.double))
        # test set_at_
        td.set("key2", torch.randn(4, 5, 6, device=device))
        x = torch.randn(6, device=device)
        td.set_at_("key2", x, (2, 2))
        assert (td.get("key2")[2, 2] == x).all()

        # test set_at_ with dtype casting
        x = torch.randn(6, dtype=torch.double, device=device)
        td.set_at_("key2", x, (2, 2))  # robust to dtype casting
        torch.testing.assert_close(td.get("key2")[2, 2], x.to(torch.float))

        td.set(
            "key1",
            torch.zeros(4, 5, dtype=torch.double, device=device),
            inplace=True,
            non_blocking=False,
        )
        assert (td.get("key1") == 0).all()
        td.set(
            "key1",
            torch.randn(4, 5, 1, 2, dtype=torch.double, device=device),
            inplace=False,
        )
        assert td["key1"].shape == td._tensordict["key1"].shape

    def test_to_memory_leak(self):
        """Test that the original tensordict is properly garbage collected when using to() method."""
        import gc

        # Create a tensordict
        td_make = lambda: TensorDict(
            {"a": torch.randn(3, 4), "b": torch.randn(3, 4)}, batch_size=[3]
        )

        # Create a weak reference to track the original tensordict
        td = td_make()
        td_ref = weakref.ref(td)

        # Verify the tensordict exists
        assert td_ref() is not None

        # Use the to method to create a new tensordict
        td = td_make()
        td_ref = weakref.ref(td)
        td_new = td.to("cpu")

        # The original tensordict should still exist (we have a reference to it)
        assert td_ref() is not None

        # Delete the original reference
        del td

        # Force garbage collection
        gc.collect()

        # The original tensordict should now be garbage collected
        assert td_ref() is None, "Original tensordict was not garbage collected"

        # Verify the new tensordict still works
        assert td_new["a"].device.type == "cpu"
        assert td_new["b"].device.type == "cpu"

    # Not working on python 3.9 and below
    @pytest.mark.skipif(
        sys.version_info < (3, 10), reason="Not working on python 3.9 and below"
    )
    @pytest.mark.skipif(not _has_streaming, reason="streaming is not installed")
    def test_to_mds(self, tmpdir):
        td = TensorDict(
            a=[0, 1, 2],
            b=[0, 1, 0],
            c=torch.tensor([[4, 5], [6, 7], [8, 9]]),
            d=["a string!", "another string!", "again!"],
            batch_size=[3],
        )

        tmpdir = str(tmpdir)
        td.to_mds(out=tmpdir)

        # Create a dataloader
        from streaming import StreamingDataset

        # Load the dataset
        dataset = StreamingDataset(local=tmpdir, remote=None, batch_size=2)
        dl = torch.utils.data.DataLoader(  # noqa: TOR401
            dataset=dataset, batch_size=2, collate_fn=TensorDict.from_list
        )
        batches = list(dl)
        batches = [_batch for batch in batches for _batch in batch.unbind(0)]
        test_td = torch.stack(batches)
        assert_allclose_td(td, test_td)

    def test_to_module_state_dict(self):
        net0 = nn.Transformer(
            d_model=16,
            nhead=2,
            num_encoder_layers=3,
            dim_feedforward=12,
            batch_first=True,
        )
        net1 = nn.Transformer(
            d_model=16,
            nhead=2,
            num_encoder_layers=3,
            dim_feedforward=12,
            batch_first=True,
        )

        def hook(
            module,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            for key, val in list(state_dict.items()):
                state_dict[key] = val * 0

        for module in net0.modules():
            module._register_load_state_dict_pre_hook(hook, with_module=True)
        for module in net1.modules():
            module._register_load_state_dict_pre_hook(hook, with_module=True)

        params_reg = TensorDict.from_module(net0)
        params_reg.to_module(net0, use_state_dict=True)
        params_reg = TensorDict.from_module(net0)

        sd = net1.state_dict()
        net1.load_state_dict(sd)
        sd = net1.state_dict()

        assert (params_reg == 0).all()
        assert set(params_reg.flatten_keys(".").keys()) == set(sd.keys())
        assert_allclose_td(params_reg.flatten_keys("."), TensorDict(sd, []))

    @pytest.mark.parametrize("mask_key", [None, "mask"])
    def test_to_padded_tensor(self, mask_key):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            td = TensorDict(
                {
                    "nested": torch.nested.nested_tensor(
                        [torch.ones(3, 4, 5), torch.ones(3, 6, 5)]
                    )
                },
                batch_size=[2, 3, -1],
            )
        assert td.shape == torch.Size([2, 3, -1])
        td_padded = td.to_padded_tensor(padding=0, mask_key=mask_key)
        assert td_padded.shape == torch.Size([2, 3, 6])
        if mask_key:
            assert (td_padded[td_padded["mask"]] != 0).all()

    @set_list_to_stack(True)
    def test_to_struct_array_string(self):
        class TC(TensorClass["nocast"]):
            foo: str | list[str]
            bar: torch.Tensor | list[torch.Tensor]

        tc = TC(
            foo=[[f"foo{i}-{j}" for i in range(5)] for j in range(4)],
            bar=torch.zeros(4, 5, 6),
        )
        td = TensorDict(
            a=torch.arange(120).view(4, 5, 6),
            b=[[f"a{i}", f"b{i}", f"c{i}", f"d{i}", f"e{i}"] for i in range(4)],
            c=tc,
            batch_size=(4, 5),
        )
        sa = td.to_struct_array()
        assert (TensorDict.from_struct_array(sa) == td).all()

    @pytest.mark.parametrize("convert_nodes", [False, True])
    @pytest.mark.parametrize("convert_tensors", [False, True])
    def test_tolist(self, convert_nodes, convert_tensors):
        td = TensorDict(
            a=torch.arange(120).view(4, 5, 6),
            b=TensorDict(c=torch.arange(40).reshape(4, 5, 2), batch_size=(4, 5, 2)),
            batch_size=(4, 5),
        )
        with (
            pytest.raises(TypeError, match="convert_tensors")
            if convert_tensors and not convert_nodes
            else contextlib.nullcontext()
        ):
            tdlist = td.tolist(
                convert_nodes=convert_nodes, convert_tensors=convert_tensors
            )
            assert isinstance(tdlist, list)
            assert len(tdlist) == 4
            for i in range(4):
                assert len(tdlist[i]) == 5
            if not convert_tensors:
                assert (tdlist[0][0]["a"] == torch.arange(6)).all()
                assert (tdlist[0][0]["b"]["c"] == torch.arange(2)).all()
            else:
                assert tdlist[0][0]["a"] == torch.arange(6).tolist()
                assert tdlist[0][0]["b"]["c"] == torch.arange(2).tolist()
            if convert_nodes:
                assert isinstance(tdlist[0][0]["b"], dict)
            else:
                assert isinstance(tdlist[0][0]["b"], TensorDict)

    def test_tolist_first(self):
        """Tests the behavior of tolist_first parameter in tolist() method."""
        td = TensorDict(
            a=torch.arange(24).view(2, 3, 4),
            b=TensorDict(c=torch.arange(12).reshape(2, 3, 2), batch_size=(2, 3, 2)),
            batch_size=(2, 3),
        )

        # Test with tolist_first=True
        result_true = td.tolist(tolist_first=True)
        # First element should be a list of 3 dictionaries
        assert len(result_true[0]) == 3
        # Each dictionary should have 'a' as tensor and 'b' as a list of dictionaries
        assert isinstance(result_true[0][0]["a"], torch.Tensor)
        assert result_true[0][0]["a"].equal(torch.tensor([0, 1, 2, 3]))
        assert isinstance(result_true[0][0]["b"], list)
        assert len(result_true[0][0]["b"]) == 2
        assert result_true[0][0]["b"][0]["c"].equal(torch.tensor(0))

        # Test with tolist_first=False
        result_false = td.tolist(tolist_first=False)
        # First element should be a list of 3 dictionaries
        assert len(result_false[0]) == 3
        # Each dictionary should have 'a' as tensor and 'b' as a dictionary with 'c' as tensor
        assert isinstance(result_false[0][0]["a"], torch.Tensor)
        assert result_false[0][0]["a"].equal(torch.tensor([0, 1, 2, 3]))
        assert isinstance(result_false[0][0]["b"], dict)
        assert result_false[0][0]["b"]["c"].equal(torch.tensor([0, 1]))

    def test_unbind_batchsize(self):
        td = TensorDict({"a": TensorDict({"b": torch.zeros(2, 3)}, [2, 3])}, [2])
        td["a"].batch_size
        tds = td.unbind(0)
        assert tds[0].batch_size == torch.Size([])
        assert tds[0]["a"].batch_size == torch.Size([3])

    @pytest.mark.parametrize("device", get_available_devices())
    def test_unbind_td(self, device):
        torch.manual_seed(1)
        d = {
            "key1": torch.randn(4, 5, 6, device=device),
            "key2": torch.randn(4, 5, 10, device=device),
        }
        td = TensorDict(batch_size=(4, 5), source=d)
        td_unbind = torch.unbind(td, dim=1)
        assert (
            td_unbind[0].batch_size == td[:, 0].batch_size
        ), f"got {td_unbind[0].batch_size} and {td[:, 0].batch_size}"

    @pytest.mark.parametrize("stack", [True, False])
    @pytest.mark.parametrize("todict", [True, False])
    def test_update_(self, stack, todict):
        def make(val, todict=False, stack=False):
            if todict:
                return make(val, stack=stack).to_dict()
            if stack:
                return LazyStackedTensorDict.lazy_stack([make(val), make(val)])
            return TensorDict({"a": {"b": val, "c": {}}, "d": {"e": val, "f": val}}, [])

        td1 = make(1, stack=stack)
        td2 = make(2, stack=stack, todict=todict)

        # plain update_
        td1.update_(td2)
        assert (td1 == 2).all()

        td1 = make(1, stack=stack)
        for key in (("a",), "a"):
            td1.update_(td2, keys_to_update=[key])
            assert (td1.select("a") == 2).all()
            assert (td1.exclude("a") == 1).all()

        td1 = make(1, stack=stack)
        for key in (("a", "b"), (("a",), ((("b"),),))):
            td1.update_(td2, keys_to_update=[key])
            assert (td1.select(("a", "b")) == 2).all()
            assert (td1.exclude(("a", "b")) == 1).all()

        # Any extra key in dest will raise an exception
        with pytest.raises(KeyError):
            td_dest = TensorDict(a=0)
            td_source = TensorDict(b=1)
            td_dest.update_(td_source)
        with pytest.raises(KeyError):
            td_dest = TensorDict(a=0)
            td_source = {"b": torch.ones(())}
            td_dest.update_(td_source)
        with pytest.raises(KeyError):
            td_dest = TensorDict(a=0)
            td_source = TensorDict(b=1)
            td_dest.update_(td_source, keys_to_update="b")
        with pytest.raises(KeyError):
            td_dest = TensorDict(a=0)
            td_source = {"b": torch.ones(())}
            td_dest.update_(td_source, keys_to_update="b")

        td_dest = TensorDict(a=0, b=1)
        td_source = TensorDict(a=0)
        td_dest.update_(td_source)

    def test_update_kwargs(self):
        # update: kwargs create new keys
        td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
        td.update(b=torch.ones(3))
        assert "b" in td.keys()
        assert (td["b"] == 1).all()

        # update: kwargs overwrite existing keys
        td.update(a=torch.ones(3))
        assert (td["a"] == 1).all()

        # update: positional + kwargs, kwargs win on conflict
        td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
        td.update({"a": torch.ones(3) * 2, "b": torch.ones(3) * 3}, a=torch.ones(3) * 7)
        assert (td["a"] == 7).all()
        assert (td["b"] == 3).all()

        # update: positional TensorDict + kwargs both applied, kwargs win on conflict
        td = TensorDict({"a": torch.zeros(3), "b": torch.zeros(3)}, batch_size=[3])
        other = TensorDict(
            {"a": torch.ones(3) * 2, "b": torch.ones(3) * 5}, batch_size=[3]
        )
        td.update(other, b=torch.ones(3) * 9)
        assert (td["a"] == 2).all()
        assert (td["b"] == 9).all()

        # update: nested dict value via kwarg still recurses
        td = TensorDict({}, batch_size=[3])
        td.update(outer={"inner": torch.ones(3)})
        assert (td["outer", "inner"] == 1).all()

        # update: empty call is a no-op and returns self
        td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
        assert td.update() is td
        assert set(td.keys()) == {"a"}

        # update_: kwargs on existing key updates in place
        td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
        a_ref = td["a"]
        td.update_(a=torch.ones(3))
        assert td["a"] is a_ref
        assert (td["a"] == 1).all()

        # update_: kwargs on missing key raises KeyError
        td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
        with pytest.raises(KeyError, match="was not found"):
            td.update_(b=torch.ones(3))

        # update_: empty call is a no-op
        td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
        assert td.update_() is td

        # update_: positional + kwargs, kwargs applied (must exist)
        td = TensorDict({"a": torch.zeros(3), "b": torch.zeros(3)}, batch_size=[3])
        td.update_({"a": torch.ones(3)}, b=torch.ones(3) * 2)
        assert (td["a"] == 1).all()
        assert (td["b"] == 2).all()

    def test_update_kwargs_lazy_stack(self):
        td = LazyStackedTensorDict.lazy_stack(
            [TensorDict({"a": torch.zeros(())}, batch_size=[]) for _ in range(3)]
        )
        td.update(b=torch.ones(3))
        assert "b" in td.keys()
        assert (td["b"] == 1).all()

        td.update_(a=torch.ones(3))
        assert (td["a"] == 1).all()

        with pytest.raises(KeyError, match="was not found"):
            td.update_(c=torch.ones(3))

    @set_capture_non_tensor_stack(False)
    @pytest.mark.parametrize("flip", [False, True])
    def test_update_batch_size(self, flip):
        @tensorclass
        class TC:
            a: str
            b: torch.Tensor

        tdc_source = torch.stack(
            [
                TC(a="a string", b=torch.zeros(())),
                TC(a="another string", b=torch.zeros(())),
                TC(a="yet another string", b=torch.zeros(())),
            ]
        )
        tdc_dest = torch.stack(
            [
                TC(a="a fourth string", b=torch.ones(())),
                TC(a="and our fifth string", b=torch.ones(())),
            ]
        )
        td_source = TensorDict(
            td=TensorDict(
                foo=torch.ones((2,)),
                nested=TensorDict(bar=torch.ones((2, 4)), batch_size=(2, 4)),
                batch_size=(2,),
            ),
            tc=tdc_source,
            td_lazy_stack=lazy_stack(
                [
                    TensorDict(ragged=torch.ones((3,)), batch_size=(3,)),
                    TensorDict(ragged=torch.ones((4,)), batch_size=(4,)),
                ],
                -1,
            ),
        )
        td_dest = TensorDict(
            td=TensorDict(
                foo=torch.ones((3,)),
                nested=TensorDict(bar=torch.ones((3, 5)), batch_size=(3, 5)),
                batch_size=(3,),
            ),
            tc=tdc_dest,
            td_lazy_stack=lazy_stack(
                [
                    TensorDict(ragged=torch.ones((3,)), batch_size=(3,)),
                    TensorDict(ragged=torch.ones((4,)), batch_size=(4,)),
                    TensorDict(ragged=torch.ones((5,)), batch_size=(5,)),
                ],
                -1,
            ),
        )

        def make_weakrefs(td):
            return {
                "td": weakref.ref(td["td"]),
                ("td", "nested"): weakref.ref(td["td", "nested"]),
                "tc": weakref.ref(td["tc"]),
                "td_lazy_stack": weakref.ref(td["td_lazy_stack"]),
            }

        if not flip:
            ref = td_source.clone()
            wr_dict = make_weakrefs(td_dest)
            td_dest.update(td_source, update_batch_size=True)
            for k, r in wr_dict.items():
                assert r() is td_dest[k]
            # Check that source is unaltered
            assert (ref == td_source).all()
        else:
            ref = td_dest.clone()
            wr_dict = make_weakrefs(td_source)
            td_source.update(td_dest, update_batch_size=True)
            for k, r in wr_dict.items():
                assert r() is td_source[k]
            # Check that source is unaltered
            assert (ref == td_dest).all()
        assert td_dest.batch_size == td_source.batch_size
        assert td_dest["td"].batch_size == td_source["td"].batch_size
        assert td_dest["td", "foo"].shape == td_source["td", "foo"].shape
        assert (
            td_dest["td", "nested"].batch_size == td_source["td", "nested"].batch_size
        )
        assert (
            td_dest["td", "nested", "bar"].shape
            == td_source["td", "nested", "bar"].shape
        )
        assert td_dest["tc"].batch_size == td_source["tc"].batch_size
        assert td_dest["tc"].a == td_source["tc"].a
        assert (td_dest["tc"].b == td_source["tc"].b).all()

    def test_update_batch_size_keepdim(self):
        t = torch.arange(24).view(2, 3, 4)
        td0 = TensorDict(
            a=t,
            b=TensorDict(c=t, batch_size=(2, 3)),
            d=t,
            e=TensorDict(f=t, batch_size=(2, 3)),
            g=t,
            batch_size=(2,),
        )

        t1 = torch.arange(16).view(2, 2, 4)

        td1 = TensorDict(
            a=t,
            b=TensorDict(c=t1, batch_size=(2, 2)),
            d=t,
            e=TensorDict(f=t, batch_size=(2, 3)),
            g=t,
            batch_size=(2,),
        )

        td0.update(td1, update_batch_size=True, is_leaf=lambda cls: True)

        assert td0.batch_size == (2,)
        assert td0["b"].batch_size == (2, 2)
        assert td0["e"].batch_size == (2, 3)

    def test_update_batch_size_errors(self):
        td0 = TensorDict(batch_size=(3,))
        td1 = TensorDict(batch_size=(4,))
        with pytest.raises(RuntimeError, match="update_batch_size"):
            td0.update(td1)
        td0.update(td1, update_batch_size=True)
        assert td0.batch_size == td1.batch_size
        td0 = TensorDict(batch_size=(3,))
        with pytest.raises(RuntimeError, match="inplace"):
            td0.update(td1, update_batch_size=True, inplace=True)

        td0 = TensorDict(batch_size=(3,), lock=True)
        with pytest.raises(RuntimeError, match="lock"):
            td0.update(td1, update_batch_size=True)
        td0_stack = lazy_stack([td0, td0]).lock_()
        td1_stack = lazy_stack([td1, td1]).lock_()
        with pytest.raises(RuntimeError, match="lock"):
            td0_stack.update(td1_stack, update_batch_size=True)

        td0 = TensorDict(a=torch.zeros((3,)), batch_size=(3,))
        td1 = TensorDict(a=torch.zeros((4,)), batch_size=(4,))
        with pytest.raises(RuntimeError, match="keys_to_update"):
            td0.update(td1, update_batch_size=True, keys_to_update=[("a",)])

    @set_list_to_stack(True)
    def test_update_nested_dict(self):
        t = TensorDict({"a": {"d": [[[0]] * 3] * 2}}, [2, 3])
        assert ("a", "d") in t.keys(include_nested=True)
        t.update({"a": {"b": [[[1]] * 3] * 2}})
        assert ("a", "d") in t.keys(include_nested=True)
        assert ("a", "b") in t.keys(include_nested=True)
        assert t["a", "b"].shape == torch.Size([2, 3, 1])
        t.update({"a": {"d": [[[1]] * 3] * 2}})

    def test_zero_grad_module(self):
        x = torch.randn(3, 3)
        linear = nn.Linear(3, 4)
        y = linear(x)
        y.sum().backward()
        p = TensorDict.from_module(linear).lock_()
        assert not p.grad.is_empty()
        linear.zero_grad(set_to_none=True)
        assert p.grad is None
        assert linear.weight.grad is None

    def test_contiguous_canonical(self):
        # Leaf with a size-1 dim and a non-canonical stride: torch reports
        # is_contiguous() == True even though the strides don't match the
        # C-row-major layout for the shape.
        non_canonical = torch.empty_strided((1, 4, 5), (5, 5, 1))
        non_canonical.copy_(torch.arange(20).view(1, 4, 5))
        assert non_canonical.is_contiguous()
        assert tuple(non_canonical.stride()) != (20, 5, 1)
        canonical = torch.randn(2, 3, 4)
        assert tuple(canonical.stride()) == (12, 4, 1)
        nested_leaf = torch.empty_strided((1, 4, 5), (5, 5, 1))
        nested_leaf.copy_(torch.arange(20).view(1, 4, 5))

        td = TensorDict(
            {
                "non_canonical": non_canonical,
                "canonical": canonical,
                "nested": TensorDict({"leaf": nested_leaf}, batch_size=[]),
            },
            batch_size=[],
        )

        # canonical=False (default) keeps the historical no-copy behaviour
        # for tensors that already pass is_contiguous().
        td_default = td.contiguous()
        assert td_default["non_canonical"].data_ptr() == non_canonical.data_ptr()
        assert tuple(td_default["non_canonical"].stride()) == (5, 5, 1)
        assert td_default["canonical"].data_ptr() == canonical.data_ptr()
        assert td_default["nested", "leaf"].data_ptr() == nested_leaf.data_ptr()

        # canonical=True materialises the leaf whose strides do not match
        # the canonical layout and leaves already-canonical tensors alone.
        td_canon = td.contiguous(canonical=True)
        assert tuple(td_canon["non_canonical"].stride()) == (20, 5, 1)
        assert td_canon["non_canonical"].data_ptr() != non_canonical.data_ptr()
        torch.testing.assert_close(td_canon["non_canonical"], non_canonical)
        assert td_canon["canonical"].data_ptr() == canonical.data_ptr()
        # Nested TensorDict leaf is also canonicalised.
        assert tuple(td_canon["nested", "leaf"].stride()) == (20, 5, 1)
        assert td_canon["nested", "leaf"].data_ptr() != nested_leaf.data_ptr()
        torch.testing.assert_close(td_canon["nested", "leaf"], nested_leaf)
        # Batch size / device / structure are preserved.
        assert td_canon.batch_size == td.batch_size
        assert td_canon.device == td.device
        assert set(td_canon.keys()) == set(td.keys())

    def test_pad_nontensor(self):
        td = TensorDict(
            {
                "x": torch.arange(3).unsqueeze(-1),
                "instr": NonTensorStack("a", "b", "c"),
            },
            batch_size=[3],
        )
        padded = pad(td, [0, 2])
        assert isinstance(padded.get("instr"), NonTensorStack)
        assert padded.get("instr").batch_size == torch.Size([5])
        assert padded.get("instr").tolist() == ["a", "b", "c", None, None]
        assert padded["x"].squeeze(-1).tolist() == [0, 1, 2, 0, 0]
        # left + right pad
        assert pad(td, [1, 1]).get("instr").tolist() == [None, "a", "b", "c", None]
        # negative pad crops
        assert pad(td, [-1, 1]).get("instr").tolist() == ["b", "c", None]

    def test_pad_nontensor_2dim(self):
        instr = torch.stack(
            [NonTensorStack(*[f"{i}{j}" for j in range(2)]) for i in range(3)]
        )
        td = TensorDict({"instr": instr, "x": torch.ones(3, 2)}, batch_size=[3, 2])
        padded = pad(td, [0, 1, 1, 0])
        assert padded.batch_size == torch.Size([4, 3])
        assert padded.get("instr").tolist() == [
            [None, "00", "01"],
            [None, "10", "11"],
            [None, "20", "21"],
            [None, None, None],
        ]

    def test_pad_nontensor_broadcast_data(self):
        td = TensorDict(
            {"meta": NonTensorData("m", batch_size=[3]), "x": torch.ones(3)},
            batch_size=[3],
        )
        padded = pad(td, [0, 2])
        assert padded.get("meta").tolist() == ["m", "m", "m", None, None]

    def test_pad_nontensor_inplace(self):
        td = TensorDict(
            {
                "x": torch.arange(3).float(),
                "instr": NonTensorStack("a", "b", "c"),
            },
            batch_size=[3],
        )
        instr = td.get("instr")
        out = pad(td, [0, 2], inplace=True)
        assert out is td
        # NonTensorStack entries are padded in place: identity preserved.
        assert out.get("instr") is instr
        assert out.get("instr").tolist() == ["a", "b", "c", None, None]
        assert out["x"].tolist() == [0.0, 1.0, 2.0, 0.0, 0.0]

    def test_pad_nontensor_top_level(self):
        stack = NonTensorStack("a", "b")
        padded = pad(stack, [1, 1])
        assert isinstance(padded, NonTensorStack)
        assert padded.tolist() == [None, "a", "b", None]
        # inplace preserves the stack's identity
        out = pad(stack, [0, 1], inplace=True)
        assert out is stack
        assert stack.tolist() == ["a", "b", None]
        # NonTensorData cannot be padded in place (it must become a stack)
        with pytest.raises(RuntimeError, match="cannot preserve the identity"):
            pad(NonTensorData("a", batch_size=[2]), [0, 1], inplace=True)

    def test_pad_nontensor_safe(self):
        td = TensorDict({"instr": NonTensorStack("a", "b", "c")}, batch_size=[3])
        with pytest.raises(RuntimeError, match="negative output size"):
            pad(td, [-4, 0])

    def test_pad_nontensor_lazy_stack(self):
        def _make():
            return lazy_stack(
                [
                    TensorDict(
                        {
                            "x": torch.ones(4) * i,
                            "instr": NonTensorStack(*[f"{i}{j}" for j in range(4)]),
                        },
                        batch_size=[4],
                    )
                    for i in range(2)
                ],
                dim=0,
            )

        expected = [
            ["00", "01", "02", "03", None, None],
            ["10", "11", "12", "13", None, None],
            [None] * 6,
        ]
        # Padding along both the stack dim and the constituent dim keeps
        # non-tensor values for the valid positions; new slots hold None.
        padded = pad(_make(), [0, 1, 0, 2], value=9.0)
        assert padded.get("instr").tolist() == expected
        lst = _make()
        out = pad(lst, [0, 1, 0, 2], value=9.0, inplace=True)
        assert out is lst
        assert out.get("instr").tolist() == expected


_SYNC_COUNTER = 0


@pytest.fixture
def _path_td_sync():
    def _sync_td(self):
        global _SYNC_COUNTER
        _SYNC_COUNTER += 1
        super(TensorDict, self)._sync_all()

    _sync_all = TensorDict._sync_all
    TensorDict._sync_all = _sync_td
    yield
    TensorDict._sync_all = _sync_all


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
