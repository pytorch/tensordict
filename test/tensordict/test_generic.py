# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import dis
import enum
import gc
import importlib.util
import inspect
import io
import json
import os
import pickle
import platform
import re
import shutil
import sys
import threading
import time
import warnings
import weakref
from collections import defaultdict, namedtuple, OrderedDict, UserDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, get_type_hints
from unittest import mock

import numpy as np
import pytest
import tensordict
import tensordict.base as tensordict_base
import torch
from tensordict import (
    from_any,
    from_tuple,
    is_leaf_nontensor,
    lazy_stack,
    LazyStackedTensorDict,
    PackLayout,
    PersistentTensorDict,
    set_capture_non_tensor_stack,
    TensorClass,
    tensorclass,
    TensorDict,
    TypedTensorDict,
)
from tensordict._indexing import convert_ellipsis_to_idx
from tensordict._lazy import _CustomOpTensorDict
from tensordict._td import _SubTensorDict, _TensorDictKeysView, is_tensor_collection
from tensordict._torch_func import _stack as stack_td, TD_HANDLED_FUNCTIONS
from tensordict.base import _NESTED_TENSORS_AS_LISTS, NO_DEFAULT, TensorDictBase
from tensordict.functional import (
    _pad_preflight,
    collate,
    dense_stack_tds,
    merge_tensordicts,
    pad,
    pad_sequence,
    unpad_sequence,
)
from tensordict.memmap import MemoryMappedTensor
from tensordict.nn import TensorDictParams
from tensordict.tensorclass import NonTensorData, NonTensorStack
from tensordict.utils import (
    _getitem_batch_size,
    _LOCK_ERROR,
    _unravel_key_to_tuple,
    assert_allclose_td,
    expand_as_right,
    is_non_tensor,
    is_tensorclass,
    set_lazy_legacy,
    set_list_to_stack,
    TensorDictFuture,
)
from torch import nn
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.autograd import forward_ad
from torch.nn.parameter import UninitializedTensorMixin
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.tensordict.test._utils_internal import (
        _TypedTD,
        decompose,
        DummyPicklableClass,
        get_available_devices,
        is_npu_available,
        legacy_lazy_mode,
        streaming_dataset,
        TestTensorDictsBase,
    )
else:
    IS_FB = False
    from _utils_internal import (
        _TypedTD,
        decompose,
        DummyPicklableClass,
        get_available_devices,
        is_npu_available,
        legacy_lazy_mode,
        streaming_dataset,
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
    # With no default device set, get_default_device() returns cpu, and
    # set_default_device(cpu) would leave a device mode on for later tests.
    torch.set_default_device(None if device.type == "cpu" else device)


# a namedtuple of numbers nested in a tuple (test_set_tuple_of_numbers_dtype)
_Pair = namedtuple("_Pair", ["x", "y"])


def _leaf_storage(td, key):
    # The storage behind a leaf. A lazy stack stacks its members' leaves into
    # a new tensor at every read, so the data_ptr() of what it returns says
    # nothing about whether a write kept the storage (it depends on the
    # allocator reusing an address): compare the members' leaves instead.
    if isinstance(td, LazyStackedTensorDict):
        return tuple(_leaf_storage(member, key) for member in td.tensordicts)
    key = (key,) if isinstance(key, str) else key
    value = td.get(key[0])
    if len(key) == 1:
        return value.data_ptr()
    return _leaf_storage(value, key[1:])


class TestGeneric:
    # Generic, type-insensitive teests

    def test_batchsize_reset(self):
        td = TensorDict(
            {"a": torch.randn(3, 4, 5, 6), "b": torch.randn(3, 4, 5)}, batch_size=[3, 4]
        )
        # a generator, read once
        td.batch_size = (size for size in [3])
        assert td.batch_size == torch.Size([3])
        td.batch_size = [3, 4]

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
            subtd.batch_size = [3]
        # an unchanged batch size returns, also as a list
        subtd.batch_size = [3, 2]
        subtd.to_tensordict(retain_none=True).batch_size = [3]

        td = TensorDict({"a": torch.randn(3, 4)}, [3, 4])
        with legacy_lazy_mode():
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

    def test_pack(self):
        td = TensorDict(
            {
                "v": torch.randn(5, 3),
                "p": torch.randn(5),
                "nested": TensorDict(stress=torch.randn(5, 2, 3, 3), batch_size=[5, 2]),
            },
            batch_size=[5],
        )
        flat, layout = td.pack()
        assert flat.shape == (5, 22)
        assert layout.keys == (("nested", "stress"), "p", "v")
        assert layout.keys == tuple(td.keys(True, True, sort=True))
        assert layout.shapes == ((2, 3, 3), (), (3,))
        assert layout.offsets == (0, 18, 19)
        assert layout.numel == 22
        assert (flat[:, :18] == td["nested", "stress"].flatten(1)).all()
        assert (flat[:, 18] == td["p"]).all()
        assert (flat[:, 19:] == td["v"]).all()

        out = layout.unpack(flat)
        assert type(out) is TensorDict
        assert out["nested"].batch_size == (5, 2)
        assert out.device is None
        assert (out == td).all()
        # the leaves are views of flat
        for key in layout.keys:
            assert out[key]._base is flat

        # the batch size comes from flat
        sub = layout.unpack(flat.index_select(0, torch.tensor([0, 3])))
        assert sub.batch_size == (2,)
        assert (sub == td[torch.tensor([0, 3])]).all()

        # writing into a leaf writes into flat
        out["p"].fill_(0)
        assert (flat[:, 18] == 0).all()

        # gradients flow back to flat
        flat = flat.detach().requires_grad_()
        layout.unpack(flat)["v"].sum().backward()
        assert (flat.grad[:, 19:] == 1).all()
        assert (flat.grad[:, :19] == 0).all()

    @pytest.mark.parametrize("dim", [0, 1, 2, 3, -1, -2, -3, -4])
    def test_pack_dim(self, dim):
        td = TensorDict(
            a=torch.randn(2, 4, 4), b=torch.randn(2, 4, 4, 2), batch_size=[2, 4, 4]
        )
        flat, layout = td.pack(dim=dim)
        pos = dim % 4
        assert flat.shape == (2, 4, 4)[:pos] + (3,) + (2, 4, 4)[pos:]
        assert flat.is_contiguous()
        assert (layout.unpack(flat) == td).all()
        # the dim is stored counted from the end, so both spellings agree
        assert layout.dim == pos - 4
        assert layout == td.pack(dim=pos - 4)[1] == td.pack(dim=pos)[1]
        with pytest.raises(IndexError, match="out of range"):
            layout.unpack(torch.zeros(()))

    @pytest.mark.parametrize("dim", [-4, 3])
    def test_pack_dim_out_of_range(self, dim):
        td = TensorDict(a=torch.randn(2, 4), batch_size=[2, 4])
        with pytest.raises(IndexError, match="out of range"):
            td.pack(dim=dim)

    def test_pack_layout(self):
        td = TensorDict(a=torch.zeros(4, 2), b=torch.ones(4), batch_size=[4])
        # same fields, other insertion order
        other = TensorDict(b=torch.ones(4) * 2, a=torch.zeros(4, 2), batch_size=[4])
        flat, layout = td.pack()
        flat_other, layout_other = other.pack()
        assert layout_other == layout
        assert hash(layout_other) == hash(layout)
        assert (flat_other[:, 2] == 2).all()

        # a layout in another column order
        reverse = PackLayout(keys=("b", "a"), shapes=(torch.Size([]), torch.Size([2])))
        flat_reverse, layout_reverse = td.pack(layout=reverse)
        assert layout_reverse == reverse
        assert (flat_reverse[:, 0] == 1).all()
        assert (reverse.unpack(flat_reverse) == td).all()
        # shapes given as lists or tuples are converted to torch.Size
        by_hand = PackLayout(keys=("b", "a"), shapes=([], (2,)))
        assert by_hand == reverse
        assert hash(by_hand) == hash(reverse)
        assert (td.pack(layout=by_hand)[0] == flat_reverse).all()
        nested = TensorDict(n=TensorDict(x=torch.zeros(4, 2), batch_size=[4, 2]))
        nested.batch_size = [4]
        _, layout_nested = nested.pack()
        by_hand = PackLayout(
            keys=(("n", "x"),), shapes=([2],), nested=(("n", [2], None),)
        )
        assert by_hand == layout_nested
        assert hash(by_hand) == hash(layout_nested)
        assert by_hand.unpack(torch.ones(3, 2))["n"].batch_size == (3, 2)

        with pytest.raises(KeyError, match="differ from the keys of the layout"):
            td.exclude("a").pack(layout=layout)
        with pytest.raises(KeyError, match="differ from the keys of the layout"):
            td.set("c", torch.zeros(4)).pack(layout=layout)
        td = td.exclude("c")
        with pytest.raises(ValueError, match="feature shapes"):
            td.set("a", torch.zeros(4, 3)).pack(layout=layout)
        td = TensorDict(a=torch.zeros(4, 2), b=torch.ones(4), batch_size=[4])
        with pytest.raises(ValueError, match="along dim 0"):
            td.pack(dim=0, layout=layout)
        # -1 and 1 are the same position for one batch dim
        assert (td.pack(dim=1, layout=layout)[0] == flat).all()
        with pytest.raises(ValueError, match="Expected 3 entries"):
            layout.unpack(torch.zeros(4, 2))

    def test_pack_empty(self):
        td = TensorDict(batch_size=[3, 2], device="cpu")
        flat, layout = td.pack(dim=1)
        assert flat.shape == (3, 0, 2)
        assert flat.device == torch.device("cpu")
        assert layout.numel == 0
        out = layout.unpack(torch.zeros(5, 0, 2))
        assert out.is_empty()
        assert out.batch_size == (5, 2)
        assert out.device == torch.device("cpu")
        # empty nested tensordicts are kept
        td = TensorDict(
            a=torch.zeros(3), b=TensorDict(batch_size=[3, 4]), batch_size=[3]
        )
        flat, layout = td.pack()
        out = layout.unpack(flat)
        assert out["b"].batch_size == (3, 4)
        assert (out == td).all()

    def test_pack_lazy_stack(self):
        tds = [TensorDict(a=torch.randn(2), n={"b": torch.randn(())}) for _ in range(3)]
        td = lazy_stack(tds)
        flat, layout = td.pack()
        dense_flat, dense_layout = td.to_tensordict().pack()
        assert layout == dense_layout
        assert (flat == dense_flat).all()
        out = layout.unpack(flat)
        assert type(out) is TensorDict
        assert (out == td).all()

        tds[0]["c"] = torch.zeros(())
        with pytest.raises(RuntimeError, match="different keys"):
            lazy_stack(tds).pack()
        # a lazy stack below a tensordict
        with pytest.raises(RuntimeError, match="different keys"):
            TensorDict(s=lazy_stack(tds), batch_size=[3]).pack()
        tds[0] = tds[0].exclude("c")
        tds[0]["a"] = torch.zeros(3)
        with pytest.raises(RuntimeError, match="Failed to stack"):
            lazy_stack(tds).pack()

    def test_pack_raises(self):
        td = TensorDict(a=torch.zeros(3), b="a string", batch_size=[3])
        with pytest.raises(TypeError, match="non-tensor entry 'b'"):
            td.pack()
        flat, layout = td.filter_non_tensor_data().pack()
        assert layout.keys == ("a",)
        td = TensorDict(a=torch.zeros(3), b=torch.zeros(3, dtype=torch.long))
        with pytest.raises(TypeError, match="different dtypes"):
            td.pack()
        td = TensorDict(a=torch.zeros(3), b=torch.zeros(3, device="meta"))
        with pytest.raises(RuntimeError, match="different devices"):
            td.pack()

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
    @pytest.mark.parametrize("num_threads", [0, 1, 2, 8])
    @pytest.mark.parametrize("use_file", [False, True])
    @pytest.mark.parametrize(
        "nested,hetdtype",
        [[False, False], [False, True], ["NJT", True]],
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
            assert (td == torch.load(filename, weights_only=False)).all(), (
                td_c.to_dict()
            )
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
            assert (td == torch.load(filename, weights_only=False)).all(), (
                td_c.to_dict()
            )
        else:
            assert all(
                (_td == _td_c).all()
                for (_td, _td_c) in zip(
                    td.unbind(0), torch.load(filename, weights_only=False).unbind(0)
                )
            ), td_c.to_dict()

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("num_threads", [0, 2])
    def test_consolidate_non_contiguous(self, device, num_threads):
        transposed = (
            torch.arange(24, dtype=torch.float32, device=device)
            .reshape(2, 3, 4)
            .transpose(0, 1)
        )
        strided = torch.arange(48, dtype=torch.int64, device=device).reshape(3, 4, 4)[
            ..., ::2
        ]
        with pytest.raises(RuntimeError):
            transposed.view(-1)
        assert strided.stride(-1) == 2

        td = TensorDict(
            {"transposed": transposed, "strided": strided},
            batch_size=[],
            device=device,
        )
        td_c = td.consolidate(num_threads=num_threads)

        for key, expected in td.items():
            actual = td_c[key]
            assert torch.equal(actual, expected)
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype

    @pytest.mark.parametrize("num_threads", [0, 2])
    def test_consolidate_column_slice(self, num_threads):
        # x[:, :1] has strides (4, 1) and reshape(-1) returns a stride-4
        # view of it, which cannot be viewed as bytes
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        td = TensorDict(
            {
                "column": x[:, :1],
                "strided": x[:, ::2],
                "transposed": x.t()[:3],
                "expanded": torch.ones(()).expand(3, 1),
            },
            batch_size=[3],
        )
        td_c = td.consolidate(num_threads=num_threads)
        td_p = pickle.loads(pickle.dumps(td_c))

        for key, expected in td.items():
            assert torch.equal(td_c[key], expected)
            assert torch.equal(td_p[key], expected)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("num_threads", [0, 2])
    def test_consolidate_one_element_strided(self, device, num_threads):
        # x[:, 0] of a (1, 5) tensor counts as contiguous, but view(-1) keeps
        # its stride of 5, which cannot be viewed as bytes. A CPU source with
        # a CUDA device and threads takes the per-leaf copy path.
        x = torch.arange(5, dtype=torch.float32).reshape(1, 5)
        td = TensorDict(column=x[:, 0], other=torch.randn(2), batch_size=[])
        td_c = td.consolidate(device=device, num_threads=num_threads)
        for key, expected in td.items():
            assert torch.equal(td_c[key].cpu(), expected)

    @pytest.mark.parametrize("device", get_available_devices())
    def test_consolidate_non_contiguous_requires_grad(self, device):
        base = torch.arange(24, dtype=torch.float32, device=device, requires_grad=True)
        transposed = base.reshape(2, 3, 4).transpose(0, 1)

        td = TensorDict({"value": transposed}, batch_size=[], device=device)
        actual = td.consolidate()["value"]

        assert torch.equal(actual, transposed)
        assert not actual.requires_grad
        transposed.sum().backward()
        assert torch.equal(base.grad, torch.ones_like(base))

    @pytest.mark.parametrize("use_file", [False, True])
    def test_consolidate_many_leaves_num_threads(self, use_file, tmpdir):
        # Exercises the chunked multithreaded copy path with enough leaves
        # to get several leaves per chunk, mixed dtypes (padding) and a
        # nested tensordict structure.
        dtypes = [torch.float32, torch.float16, torch.uint8, torch.float64]
        td = TensorDict(
            {
                f"key{i}": torch.full((i % 7 + 1,), i, dtype=dtypes[i % len(dtypes)])
                for i in range(50)
            },
            batch_size=[],
        )
        td["nested"] = TensorDict(
            {"a": torch.randn(3), "b": torch.arange(11, dtype=torch.int32)},
            batch_size=[],
        )
        if use_file:
            td_ref = td.consolidate(filename=Path(tmpdir) / "ref.td", num_threads=0)
            td_c = td.consolidate(filename=Path(tmpdir) / "threaded.td", num_threads=4)
        else:
            td_ref = td.consolidate(num_threads=0)
            td_c = td.consolidate(num_threads=4)
        assert (td_c == td).all()
        assert (td_ref._consolidated["storage"] == td_c._consolidated["storage"]).all()
        for key, val in td.items(True, True):
            assert td_c[key].dtype == val.dtype

    @pytest.mark.parametrize("use_file", [False, True])
    def test_consolidate_return_early(self, use_file, tmpdir):
        td = TensorDict(
            {
                f"key{i}": torch.full((i % 5 + 1,), i, dtype=torch.float32)
                for i in range(20)
            },
            batch_size=[],
        )
        td["nested"] = TensorDict({"a": torch.randn(3)}, batch_size=[])
        td["string"] = "a string!"
        kwargs = {"filename": Path(tmpdir) / "file.td"} if use_file else {}
        future = td.consolidate(num_threads=4, return_early=True, **kwargs)
        assert isinstance(future, TensorDictFuture)
        td_c = future.result()
        assert td_c.is_consolidated()
        assert (td_c == td).all()
        assert td_c["string"] == "a string!"

        with pytest.raises(NotImplementedError, match="use_buffer"):
            td.consolidate(
                filename=Path(tmpdir) / "file2.td",
                num_threads=4,
                return_early=True,
                use_buffer=True,
            )
        with pytest.raises(NotImplementedError, match="non_blocking"):
            td.consolidate(num_threads=4, return_early=True, non_blocking=True)

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

    @pytest.mark.parametrize("non_blocking", [None, False, True])
    @pytest.mark.parametrize("storage_device", [None, "meta"])
    def test_consolidate_to_meta(self, storage_device, non_blocking):
        # Meta tensors hold no data, so a meta target or storage needs no sync.
        td = TensorDict(
            {"a": torch.zeros(3), "b": {"c": torch.ones(3, 2)}}, batch_size=[3]
        )
        td_c = td.consolidate(device=storage_device)
        td_meta = td_c.to("meta", non_blocking=non_blocking)
        assert td_meta.device == torch.device("meta")
        assert td_meta.is_consolidated()
        assert td_meta.is_locked
        assert td_meta["a"].device.type == "meta"
        assert td_meta["b", "c"].device.type == "meta"
        assert td_meta["b", "c"].shape == (3, 2)

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("device", ["meta", f"{cur_device}:0"])
    def test_consolidate_to_inplace(self, device, lazy):
        # to(device, inplace=True) moves a consolidated tensordict, which is
        # locked, and keeps it consolidated and locked (#2325)
        if device.startswith("cpu"):
            pytest.skip("no cuda or npu device detected")
        td = TensorDict(
            {"a": torch.zeros(3), "b": {"c": torch.ones(3, 2)}, "d": "a string!"},
            batch_size=[3],
        )
        if lazy:
            td = LazyStackedTensorDict(*td.unbind(0))
        td = td.consolidate(metadata=True)
        result = td.to(device, inplace=True)
        assert result is td
        assert td.device == torch.device(device)
        assert td.is_consolidated()
        assert td.is_locked
        assert td["b"].is_locked
        storage = td._consolidated["storage"]
        assert storage.device == torch.device(device)
        # A lazy stack's values are stacked copies: check the leaves of the
        # stacked tensordicts, which are views of the storage
        for sub_td in td.tensordicts if lazy else [td]:
            for value in sub_td.values(True, True):
                assert value.device == torch.device(device)
                assert value.untyped_storage().data_ptr() == storage.data_ptr()
        assert td["b", "c"].shape == (3, 2)
        assert td[0]["d"] == "a string!"
        if device != "meta":
            assert (td["b", "c"] == 1).all()

    def test_consolidate_to_inplace_locked_raises(self):
        # A tensordict that the user locked keeps raising
        td = TensorDict(a=torch.zeros(3), batch_size=[3]).lock_()
        with pytest.raises(RuntimeError, match="Cannot modify locked TensorDict"):
            td.to("meta", inplace=True)

    def test_consolidate_to_no_device(self):
        # Without a device or dtype, to() returns the consolidated tensordict
        # itself, as for other tensordicts, and changes its batch size in place
        td = TensorDict(
            {"a": torch.zeros(3, 2), "b": {"c": torch.ones(3, 2, 4)}}, batch_size=[3]
        ).consolidate()
        assert td.to() is td
        assert td.to(non_blocking=True) is td
        with pytest.raises(RuntimeError, match="incompatible with the batch-size"):
            td.to(batch_size=[5])
        assert td.to(batch_size=[3, 2]) is td
        assert td.batch_size == (3, 2)
        assert td["b"].batch_size == (3, 2)
        assert td.is_consolidated()

    @pytest.mark.parametrize("consolidated", [False, True])
    def test_to_batch_size_same_device(self, consolidated):
        td = TensorDict(
            {"a": torch.zeros(2, 3), "b": {"c": torch.ones(2, 3, 4)}},
            batch_size=[2],
            device="cpu",
            names=["rows"],
        )
        if consolidated:
            td = td.consolidate()
        leaves = dict(td.items(True, True))
        with pytest.raises(RuntimeError, match="incompatible with the batch-size"):
            td.to("cpu", batch_size=[5])
        assert td.to("cpu", batch_size=[2, 3]) is td
        assert td.batch_size == (2, 3)
        assert td["b"].batch_size == (2, 3)
        assert td.names == ["rows", None]
        assert td.is_consolidated() == consolidated
        assert td.is_locked == consolidated
        for key, leaf in leaves.items():
            assert td[key] is leaf

    @pytest.mark.parametrize("batch_size", [[2], [2, 3], []])
    def test_consolidate_to_batch_size(self, batch_size):
        td = TensorDict(
            {"a": torch.zeros(2, 3), "b": {"c": torch.ones(2, 3, 4)}},
            batch_size=[2],
            device="cpu",
        ).consolidate()
        result = td.to("meta", batch_size=batch_size)
        assert result.batch_size == torch.Size(batch_size)
        assert result["b"].batch_size == torch.Size(
            [2, 3] if len(batch_size) > 1 else [2]
        )
        assert result.device == torch.device("meta")
        assert result["a"].shape == (2, 3)
        assert result["b", "c"].shape == (2, 3, 4)
        assert result.is_consolidated()
        assert result.is_locked
        assert result is not td
        assert td.batch_size == (2,)
        assert td["b"].batch_size == (2,)
        assert td.device == torch.device("cpu")

    def test_consolidate_to_batch_size_invalid(self):
        td = TensorDict(a=torch.zeros(2, 3), batch_size=[2]).consolidate()
        with pytest.raises(RuntimeError, match="incompatible with the batch-size"):
            td.to("meta", batch_size=[5])
        assert td.batch_size == (2,)

    @pytest.mark.parametrize("how", ["setter", "to", "to_cpu", "to_meta"])
    def test_consolidated_batch_size_names_serialized(self, how):
        # pickle and torch.save rebuild a consolidated tensordict from the
        # metadata of consolidate(), so the setters must update it
        td = TensorDict(
            a=torch.zeros(2, 3),
            b={"c": torch.ones(2, 3, 4)},
            d="a string",
            batch_size=[2],
            device="cpu",
        ).consolidate()
        if how == "setter":
            td.batch_size = [2, 3]
        elif how == "to":
            td = td.to(batch_size=[2, 3])
        elif how == "to_cpu":
            td = td.to("cpu", batch_size=[2, 3])
        else:
            td = td.to("meta", batch_size=[2, 3])
            metadata = td._consolidated["metadata"]
            assert metadata["cls_metadata"]["batch_size"] == [2, 3]
            assert metadata["b"]["cls_metadata"]["batch_size"] == [2, 3]
            assert metadata["non_tensors"]["d"][1] == [2, 3]
            return
        td.names = ["x", "y"]
        assert td.is_consolidated()
        buffer = io.BytesIO()
        torch.save(td, buffer)
        buffer.seek(0)
        for result in (
            pickle.loads(pickle.dumps(td)),
            torch.load(buffer, weights_only=False),
        ):
            assert result.batch_size == (2, 3)
            assert result["b"].batch_size == (2, 3)
            assert result.get("d").batch_size == (2, 3)
            assert result.names == ["x", "y"]
            assert result["b"].names == ["x", "y"]
            assert result.is_consolidated()

    def test_consolidated_nested_batch_size_consolidate_again(self):
        td = TensorDict(
            a=torch.zeros(2, 3), b={"c": torch.ones(2, 3, 4)}, batch_size=[2]
        ).consolidate()
        td["b"].batch_size = [2, 3, 4]
        # consolidate() on a consolidated tensordict refreshes its metadata
        assert td.consolidate() is td
        assert pickle.loads(pickle.dumps(td))["b"].batch_size == (2, 3, 4)

    @pytest.mark.parametrize("use_buffer", [False, True])
    def test_consolidated_file_batch_size_names_consolidate_again(
        self, tmp_path, use_buffer
    ):
        # the file keeps the batch sizes and names of consolidate(filename=...):
        # calling it again with the same file writes the current ones
        filename = tmp_path / "td.bin"
        td = TensorDict(
            a=torch.zeros(2, 3), b={"c": torch.ones(2, 3, 4)}, batch_size=[2]
        ).consolidate(filename=filename, use_buffer=use_buffer)
        td.batch_size = [2, 3]
        td.names = ["x" * 100, "y"]
        td["b"].batch_size = [2, 3, 4]
        assert td.consolidate(filename=filename) is td
        loaded = TensorDict.from_consolidated(filename)
        assert loaded.batch_size == (2, 3)
        assert loaded.names == ["x" * 100, "y"]
        assert loaded["b"].batch_size == (2, 3, 4)
        assert (loaded == td).all()
        # a shorter metadata
        td.names = None
        td.batch_size = [2]
        assert td.consolidate(filename=str(filename)) is td
        loaded = TensorDict.from_consolidated(filename)
        assert loaded.batch_size == (2,)
        assert loaded.names == [None]
        assert (loaded == td).all()
        # another file name writes nothing
        assert td.consolidate(filename=tmp_path / "other.bin") is td
        assert not (tmp_path / "other.bin").exists()
        # nor does a file that was replaced, or deleted
        os.remove(filename)
        td2 = TensorDict(z=torch.ones(2), batch_size=[2]).consolidate(filename=filename)
        td.batch_size = []
        assert td.consolidate(filename=filename) is td
        loaded = TensorDict.from_consolidated(filename)
        assert loaded.batch_size == (2,)
        assert (loaded == td2).all()
        del td2, loaded
        os.remove(filename)
        assert td.consolidate(filename=filename) is td
        assert not filename.exists()

    def test_consolidated_lazy_stack_names_serialized(self):
        td = lazy_stack(
            [
                TensorDict(a=torch.zeros(3), b={"c": torch.zeros(3, 2)}, batch_size=[3])
                for _ in range(2)
            ]
        ).consolidate()
        td.names = ["x", "y"]
        result = pickle.loads(pickle.dumps(td))
        assert result.names == ["x", "y"]
        assert result[0]["b"].names == ["y"]

    def test_to_device_batch_size_nested_only(self):
        # When the device changes and the root holds only nested tensordicts,
        # their leaves are checked against the new batch size too (#2324)
        td = TensorDict(
            nested=TensorDict(a=torch.zeros(2, 3), batch_size=[2]), batch_size=[2]
        )
        with pytest.raises(RuntimeError, match="incompatible with the batch-size"):
            td.to("meta", batch_size=[5])
        result = td.to("meta", batch_size=[2, 3])
        assert result.batch_size == (2, 3)
        assert result["nested"].batch_size == (2, 3)
        assert result["nested", "a"].device == torch.device("meta")
        td = TensorDict(
            nested=lazy_stack(
                [TensorDict(a=torch.zeros(3)), TensorDict(a=torch.zeros(3))]
            ),
            batch_size=[2],
        )
        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
            result = td.to("meta", batch_size=[2, 3])
        assert result.batch_size == (2, 3)
        assert result["nested"].batch_size == (2, 3)

    @pytest.mark.parametrize("dest", ["meta", torch.float64])
    def test_to_device_batch_size_inplace(self, dest):
        # to(device, inplace=True, batch_size=...) moves the tensordict and
        # sets its batch size as the setter does (#2374)
        td = TensorDict(
            a=torch.zeros(2, 3),
            nested=TensorDict(b=torch.zeros(2, 3, 4), batch_size=[2, 3, 4]),
            batch_size=[2],
            names=["r"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert td.to(dest, inplace=True, batch_size=[2, 3]) is td
        assert td.batch_size == (2, 3)
        assert td.names == ["r", None]
        assert td["nested"].batch_size == (2, 3, 4)
        for leaf in (td["a"], td["nested", "b"]):
            if dest == "meta":
                assert leaf.device == torch.device("meta")
            else:
                assert leaf.dtype == torch.float64
        with pytest.raises(RuntimeError, match="incompatible with the batch-size"):
            td.to(dest, inplace=True, batch_size=[5])

    @pytest.mark.parametrize(
        "names,nested_batch_size,batch_size,result_nested_batch_size,warns",
        [
            # nested batch sizes that the setter would keep and names warn
            (None, [2, 3, 4], [2, 3], [2, 3], True),
            (["r"], [2], [2, 3], [2, 3], True),
            (["r"], [2], [2], [2], True),
            (None, [2, 3], [2], [2], True),
            # the result stays the same in 0.17
            (None, [2], [2, 3], [2, 3], False),
            (None, [2, 3], [2, 3], [2, 3], False),
            (None, [2], [2], [2], False),
        ],
    )
    @pytest.mark.parametrize("dest", ["meta", torch.float64])
    def test_to_device_batch_size_future_warning(
        self,
        names,
        nested_batch_size,
        batch_size,
        result_nested_batch_size,
        warns,
        dest,
    ):
        # When the device or dtype changes, to(batch_size=...) will keep the
        # nested batch sizes and the names in 0.17 (#2374)
        td = TensorDict(
            a=torch.zeros(2, 3, 4),
            nested=TensorDict(b=torch.zeros(2, 3, 4), batch_size=nested_batch_size),
            batch_size=[2],
            names=names,
        )
        if warns:
            with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
                result = td.to(dest, batch_size=batch_size)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = td.to(dest, batch_size=batch_size)
        assert result.batch_size == torch.Size(batch_size)
        assert result["nested"].batch_size == torch.Size(result_nested_batch_size)
        assert result.names == [None] * len(batch_size)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            td.to(dest)
            td.to(dest, inplace=True, batch_size=batch_size)

    @pytest.mark.parametrize("consolidated", [False, True])
    def test_to_batch_size_unchanged(self, consolidated, monkeypatch):
        td = TensorDict(a=torch.zeros(2, 3), batch_size=[2], device="cpu")
        if consolidated:
            td = td.consolidate()

        def check_new_batch_size(*args):
            raise AssertionError("unchanged batch sizes should not check every leaf")

        monkeypatch.setattr(type(td), "_check_new_batch_size", check_new_batch_size)
        assert td.to("cpu", batch_size=[2]) is td
        assert td.to("cpu", batch_size=torch.Size([2])) is td

    @pytest.mark.parametrize("locked", [False, True])
    def test_to_device_walks_once(self, monkeypatch, locked):
        # to() walks an unlocked tree once. A locked tree caches its leaves,
        # so it keeps the cheap first walk through _items_list.
        td = TensorDict(
            a=torch.zeros(2, device="meta"),
            b=torch.zeros(2),
            nested=TensorDict(c=torch.zeros(2, 3), batch_size=[2]),
            batch_size=[2],
            names=["x"],
        )
        if locked:
            td.lock_()
        else:

            def items_list(*args, **kwargs):
                raise AssertionError("to() should walk an unlocked tree once")

            monkeypatch.setattr(type(td), "_items_list", items_list)
        result = td.to("meta")
        assert result.device == torch.device("meta")
        assert result["nested"].device == torch.device("meta")
        assert result.is_locked is locked
        assert result["nested"].is_locked is locked
        assert result.names == ["x"]
        assert result["b"].device == torch.device("meta")
        assert result["nested", "c"].device == torch.device("meta")

    def test_to_cpu_skips_leaves_already_there(self, monkeypatch):
        # A device-less tree whose leaves are on the CPU already: to("cpu")
        # returns them as they are, without calling Tensor.to
        td = TensorDict(
            a=torch.zeros(2),
            nested=TensorDict(c=torch.zeros(2, 3), batch_size=[2]),
            batch_size=[2],
        )
        assert td.device is None
        moved = []
        tensor_to = torch.Tensor.to

        def counting_to(tensor, *args, **kwargs):
            moved.append(tensor.device.type)
            return tensor_to(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", counting_to)
        result = td.to("cpu")
        assert moved == []
        assert result.device == torch.device("cpu")
        assert result["nested"].device == torch.device("cpu")
        assert result["a"] is td["a"]
        assert result["nested", "c"] is td["nested", "c"]

    @pytest.mark.parametrize(
        "kind",
        [
            "flat",
            "nested",
            "nested_device",
            "locked",
            "empty",
            "names",
            "nested_names",
            "lazy",
            "non_tensor",
            "unbatched",
            "params",
            "tensorclass",
        ],
    )
    @pytest.mark.parametrize(
        "args,kwargs",
        [
            (("cpu",), {}),
            ((torch.device("cpu"),), {"non_blocking": True}),
            ((torch.float64,), {}),
            (("cpu", torch.float64), {}),
            (("meta",), {}),
        ],
    )
    def test_to_dense_tree_matches_general_path(self, kind, args, kwargs, monkeypatch):
        # to() on a tree of exact TensorDicts without dim names rebuilds it
        # in one pass, with the target device on every node, and hands the
        # other entries to _apply_nest: the result is the one of the generic
        # walk.
        def make():
            leaves = {"a": torch.zeros(2, 3), "b": torch.ones(2)}
            if kind == "flat":
                return TensorDict(leaves, batch_size=[2])
            if kind == "empty":
                return TensorDict(batch_size=[2])
            if kind == "params":
                return TensorDictParams(TensorDict(leaves, batch_size=[2]))
            sub = {"c": torch.zeros(2, 4), "d": {"e": torch.ones(2, 1)}}
            td = TensorDict({**leaves, "nested": sub}, batch_size=[2])
            if kind == "nested_device":
                td["other"] = TensorDict(f=torch.zeros(2), batch_size=[2], device="cpu")
            elif kind == "locked":
                td.lock_()
            elif kind == "names":
                td.names = ["x"]
            elif kind == "nested_names":
                td["nested"].names = ["x"]
            elif kind == "lazy":
                td["lazy"] = lazy_stack(
                    [TensorDict(f=torch.zeros(3)), TensorDict(f=torch.ones(3))]
                )
            elif kind == "non_tensor":
                td["nested", "s"] = NonTensorData("s", batch_size=[2])
            elif kind == "unbatched":
                td["nested", "u"] = tensordict.UnbatchedTensor(torch.zeros(5))
            elif kind == "tensorclass":

                @tensorclass
                class Holder:
                    x: torch.Tensor
                    td: TensorDict

                return Holder(x=torch.zeros(2), td=td, batch_size=[2])
            return td

        td = make()
        with monkeypatch.context() as m:
            m.setattr(TensorDict, "_is_dense_node", lambda self: False)
            expected = td.to(*args, **kwargs)
        walked = []
        apply_nest = TensorDictBase._apply_nest

        def record_apply_nest(self, *a, **kw):
            walked.append(type(self))
            return apply_nest(self, *a, **kw)

        monkeypatch.setattr(TensorDictBase, "_apply_nest", record_apply_nest)
        result = td.to(*args, **kwargs)
        if kind in ("flat", "nested", "nested_device", "locked", "empty"):
            assert walked == []
        assert type(result) is type(expected)
        if is_tensorclass(td):
            td, result, expected = (
                td._tensordict,
                result._tensordict,
                expected._tensordict,
            )
        assert list(result.keys(True)) == list(expected.keys(True))
        nodes = [k for k in result.keys(True) if is_tensor_collection(result[k])]
        for key in [()] + nodes:
            r = result[key] if key else result
            e = expected[key] if key else expected
            assert type(r) is type(e)
            assert r.batch_size == e.batch_size
            assert r.device == e.device
            assert r.names == e.names
            assert r.is_locked == e.is_locked
        for key in result.keys(True, True):
            r, e = result.get(key), expected.get(key)
            assert type(r) is type(e)
            if isinstance(r, torch.Tensor):
                assert (r.dtype, r.device, r.shape) == (e.dtype, e.device, e.shape)
                assert (r is td.get(key)) == (e is td.get(key))

    def test_sub_to_batch_size_same_device(self):
        source = TensorDict(a=torch.zeros(4, 2, 3), batch_size=[4, 2], device="cpu")
        td = source._get_sub_tensordict(0)
        assert td.to("cpu") is td
        assert td.to("cpu", batch_size=[2]) is td
        result = td.to("cpu", batch_size=[2, 3])
        assert type(result) is TensorDict
        assert result.batch_size == (2, 3)
        assert source.batch_size == (4, 2)
        assert td.batch_size == (2,)
        torch.testing.assert_close(result["a"], td["a"])
        with pytest.raises(RuntimeError, match="incompatible with the batch-size"):
            td.to("cpu", batch_size=[5])

    def test_to_batch_size_nested_lazy(self):
        nested = lazy_stack(
            [TensorDict(a=torch.zeros(3), batch_size=[]) for _ in range(2)]
        )
        td = TensorDict(nested=nested, batch_size=[2], device="cpu")
        with pytest.raises(RuntimeError, match="Modifying the batch size of a lazy"):
            td.to("cpu", batch_size=[2, 3])

    @pytest.mark.parametrize("locked", [False, True])
    @pytest.mark.parametrize("batch_size", [[2], [2, 3]])
    @pytest.mark.parametrize("dest", ["meta", torch.float64])
    def test_to_device_batch_size_nested_lazy(self, locked, batch_size, dest):
        # to(device_or_dtype, batch_size=...) moves the leaves of a nested
        # lazy stack, locked or not (#2374)
        td = TensorDict(
            lazy=lazy_stack(
                [
                    TensorDict(
                        x=torch.zeros(3),
                        n=TensorDict(w=torch.zeros(3, 1), batch_size=[3]),
                    )
                    for _ in range(2)
                ]
            ),
            y=torch.zeros(2, 3),
            batch_size=[2],
        )
        if locked:
            td.lock_()
        with pytest.warns(FutureWarning, match="turns nested lazy stacks into"):
            result = td.to(dest, batch_size=batch_size)
        assert result.batch_size == torch.Size(batch_size)
        assert result.is_locked is locked
        leaves = list(result.values(True, True))
        assert len(leaves) == 3
        for leaf in leaves:
            if dest == "meta":
                assert leaf.device == torch.device("meta")
            else:
                assert leaf.dtype == torch.float64

    @pytest.mark.skipif(not _has_h5py, reason="h5py needed for persistent tensordicts")
    def test_persistent_to_batch_size_same_device(self, tmp_path):
        td = PersistentTensorDict.from_dict(
            {"a": torch.zeros(2, 3)},
            filename=tmp_path / "td.h5",
            batch_size=[2],
            device="cpu",
        )
        try:
            assert td.to("cpu", batch_size=[2]) is td
            assert td.to("cpu", batch_size=[2, 3]) is td
            assert td.batch_size == (2, 3)
            torch.testing.assert_close(td["a"], torch.zeros(2, 3))
            with pytest.raises(RuntimeError, match="mismatch"):
                td.to("cpu", batch_size=[5])
            assert td.batch_size == (2, 3)
        finally:
            td.close()

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("in_parent", [False, True])
    @pytest.mark.parametrize("use_prefix", [False, True])
    def test_consolidated_memmap_pickle(self, tmpdir, lazy, in_parent, use_prefix):
        # memmap_ moves the leaves out of the consolidated storage: pickles
        # must see later writes and stay self-contained (#2317)
        if lazy:
            td = LazyStackedTensorDict(
                *[
                    TensorDict(a=torch.zeros(3), b={"c": torch.ones(3)})
                    for _ in range(2)
                ]
            )
        else:
            td = TensorDict(
                a=torch.zeros(2, 3), b={"c": torch.ones(2, 3)}, batch_size=[2]
            )
        td = td.consolidate(metadata=True)
        prefix = Path(tmpdir) / "memmap" if use_prefix else None
        if in_parent:
            parent = TensorDict(td=td, batch_size=[2]).lock_()
            parent.memmap_(prefix)
        else:
            td.memmap_(prefix)
        td.apply_(lambda x: x.fill_(5))
        data = pickle.dumps(parent if in_parent else td)
        if use_prefix:
            shutil.rmtree(prefix)
        td_rt = pickle.loads(data)
        if in_parent:
            td_rt = td_rt["td"]
        assert (td_rt == 5).all()
        assert td.is_locked
        assert not td.is_consolidated()

    def test_consolidated_inplace_write_stays_consolidated(self):
        td = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3)}).consolidate()
        td["a"].add_(1)
        td.set_(("b", "c"), torch.full((3,), 2.0))
        assert td.is_consolidated()
        td_rt = pickle.loads(pickle.dumps(td))
        assert (td_rt["a"] == 1).all()
        assert (td_rt["b", "c"] == 2).all()

    @pytest.mark.parametrize(
        "write",
        [
            "set",
            "set_new",
            "set_non_tensor",
            "set_non_tensor_inplace",
            "set_nested",
            "del",
            "update",
            "select",
            "select_nested",
            "exclude",
            "exclude_nested",
            "popitem",
        ],
    )
    def test_consolidated_unpickled_write_pickle(self, write):
        # https://github.com/pytorch/tensordict/issues/2503
        td = TensorDict(
            x=torch.zeros(3), s="hello", n={"z": torch.zeros(3)}, batch_size=[3]
        ).consolidate()
        td_rt = pickle.loads(pickle.dumps(td))
        assert not td_rt.is_locked
        assert td_rt.is_consolidated()
        expected = td_rt.to_tensordict()
        if write == "set":
            td_rt["x"] = expected["x"] = torch.ones(3)
        elif write == "set_new":
            td_rt["y"] = expected["y"] = torch.ones(3)
        elif write == "set_non_tensor":
            td_rt["s"] = expected["s"] = "bye"
        elif write == "set_non_tensor_inplace":
            td_rt.set_("s", "bye")
            expected["s"] = "bye"
        elif write == "set_nested":
            td_rt["n", "z"] = expected["n", "z"] = torch.ones(3)
        elif write == "del":
            del td_rt["x"]
            del expected["x"]
        elif write == "update":
            td_rt.update({"x": torch.ones(3)})
            expected["x"] = torch.ones(3)
        elif write == "select":
            td_rt.select("x", "s", inplace=True)
            expected = expected.select("x", "s")
        elif write == "select_nested":
            td_rt.select("s", ("n", "z"), inplace=True)
            expected = expected.select("s", ("n", "z"))
        elif write == "exclude":
            td_rt.exclude("x", inplace=True)
            expected = expected.exclude("x")
        elif write == "exclude_nested":
            td_rt.exclude(("n", "z"), inplace=True)
            expected = expected.exclude(("n", "z"))
        elif write == "popitem":
            key, _ = td_rt.popitem()
            del expected[key]
        assert not td_rt.is_consolidated()
        td_rt2 = pickle.loads(pickle.dumps(td_rt))
        assert set(td_rt2.keys(True, True)) == set(expected.keys(True, True))
        assert_allclose_td(td_rt2.exclude("s"), expected.exclude("s"))
        assert td_rt2["s"] == expected["s"]
        # A copy that is not written to stays consolidated
        assert pickle.loads(pickle.dumps(td)).is_consolidated()

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

    def test_construct_from_mapping(self):
        # Any Mapping is read like a dict (#2680).
        x = torch.zeros(3, 2)
        source = MappingProxyType({"x": x, ("child", "y"): torch.ones(3)})
        for batch_size in (None, [3]):
            td = TensorDict(source, batch_size=batch_size)
            assert type(td) is TensorDict
            assert td.batch_size == (() if batch_size is None else (3,))
            assert td["x"] is x
            assert (td["child", "y"] == 1).all()
        with pytest.raises(ValueError, match="Mapping"):
            TensorDict([("x", x)])

    @pytest.mark.parametrize("device", [None, "cpu", "meta"])
    def test_construct_mixed_tensor_validation(self, device):
        x = torch.randn(3, 4, requires_grad=True)
        parameter = nn.Parameter(torch.randn(3, 4))
        source = {"x": x, "parameter": parameter, ("child", "y"): x.detach()}
        td = TensorDict(source, [3], device=device, names=["batch"], lock=True)
        assert td.is_locked and td["child"].names == ["batch"]
        assert ("child", "y") in source and "child" not in source
        if device != "meta":
            assert td["x"] is x and td["parameter"] is parameter
            (td["x"] + td["parameter"]).sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x))
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        else:
            assert all(value.device.type == "meta" for value in td.values(True, True))
            assert x.device.type == "cpu" and parameter.device.type == "cpu"
        with td.unlock_():
            td.set("extra", td["x"])
        assert "extra" not in source
        with pytest.raises(RuntimeError, match="batch dimension mismatch"):
            TensorDict({"x": x, "bad": torch.zeros(2, 4)}, [3], device=device)

    @pytest.mark.parametrize("source", [None, {}])
    @pytest.mark.parametrize("device", [None, "cpu", torch.device("cpu"), "meta"])
    @pytest.mark.parametrize(
        "names, lock", [(None, False), (["b"], False), (None, True)]
    )
    def test_construct_empty_source(self, source, device, names, lock, monkeypatch):
        # An empty source has nothing to validate or transfer (#2623).
        recorder = tensordict_base._device_recorder
        calls = []
        mark = recorder.mark
        monkeypatch.setattr(recorder, "mark", lambda: (calls.append("mark"), mark())[1])
        validate_value = TensorDictBase._validate_value
        monkeypatch.setattr(
            TensorDict,
            "_validate_value",
            property(lambda td: (calls.append("validate"), validate_value.fget(td))[1]),
        )
        td = TensorDict(source, [3], device=device, names=names, lock=lock)
        assert calls == []
        assert type(td) is TensorDict and td.is_empty()
        assert td.batch_size == (3,)
        assert td.device == (None if device is None else torch.device(device))
        assert td.names == (names or [None])
        assert td.is_locked is lock
        td = TensorDict({"a": torch.zeros(3)}, [3], device=device)
        assert calls.count("mark") == (device is not None)
        assert td["a"].device == torch.device(device or "cpu")

    def test_construct_cuda_device_follows_current_device(self, monkeypatch):
        # A device without index is not remembered from one call to the next.
        for index in (1, 0):
            monkeypatch.setattr(torch.cuda, "current_device", lambda index=index: index)
            for device in ("cuda", torch.device("cuda")):
                for names in (None, []):
                    td = TensorDict(device=device, names=names)
                    assert td.device == torch.device("cuda", index)

    def test_new_unsafe_keeps_device(self, monkeypatch):
        # A torch.device is reused, and MPS availability is read once (#2623).
        calls = []
        is_available = torch.backends.mps.is_available
        monkeypatch.setattr(
            torch.backends.mps,
            "is_available",
            lambda: (calls.append(None), is_available())[1],
        )
        td = TensorDict(
            a=torch.zeros(3), nested={"b": torch.zeros(3)}, batch_size=[3], device="cpu"
        )
        for _ in range(3):
            clone = td.clone()
            assert clone.device is td.device
            assert clone["nested"].device is td["nested"].device
        assert len(calls) <= 1

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

    @pytest.mark.parametrize("as_tensorclass", [False, True])
    @pytest.mark.parametrize(
        "device,expected",
        [
            (None, torch.device("cuda")),
            (1, torch.device("cuda", 1)),
            (torch.tensor(1), torch.device("cuda", 1)),
            ("1", torch.device("cuda", 1)),
            ("cuda", torch.device("cuda")),
            ("cuda:1", torch.device("cuda", 1)),
            (torch.device("cuda", 1), torch.device("cuda", 1)),
        ],
    )
    def test_cuda_device_and_kwargs(self, as_tensorclass, device, expected):
        # `to` is mocked so that the test runs without a GPU.
        td = TensorDict(a=torch.zeros(3), batch_size=[3])
        if as_tensorclass:

            @tensorclass
            class MyClass:
                a: torch.Tensor

            td = MyClass.from_tensordict(td)
        with mock.patch.object(TensorDict, "to", autospec=True) as to:
            if device is None:
                td.cuda(non_blocking=True)
            else:
                td.cuda(device, non_blocking=True)
        to.assert_called_once()
        (target,) = to.call_args.args[1:]
        assert torch.device(target) == expected
        assert to.call_args.kwargs == {"non_blocking": True}

    @pytest.mark.parametrize("device", ["cpu", torch.device("cpu"), "meta"])
    def test_cuda_rejects_non_cuda_device(self, device):
        td = TensorDict(a=torch.zeros(3), batch_size=[3])
        with mock.patch.object(TensorDict, "to", autospec=True) as to:
            with pytest.raises(RuntimeError, match="must be a cuda device"):
                td.cuda(device)
        to.assert_not_called()

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

    def test_data_setter(self):
        td = TensorDict(
            {
                "a": torch.tensor([1.0, 2.0]),
                "b": {"c": torch.tensor([[3.0, 4.0], [5.0, 6.0]])},
            },
            batch_size=[2],
        )
        other = TensorDict(
            {
                "a": torch.tensor([7.0, 8.0]),
                "b": {"c": torch.tensor([[9.0, 10.0], [11.0, 12.0]])},
            },
            batch_size=[2],
        )
        leaf_a = td.get("a")
        leaf_c = td.get(("b", "c"))
        batch_size = td.batch_size

        td.data = other

        assert td.get("a").tolist() == [7.0, 8.0]
        assert td.get(("b", "c")).tolist() == [[9.0, 10.0], [11.0, 12.0]]
        assert td.get("a") is leaf_a
        assert td.get(("b", "c")) is leaf_c
        assert other.get("a").tolist() == [7.0, 8.0]
        assert other.get(("b", "c")).tolist() == [[9.0, 10.0], [11.0, 12.0]]
        assert td.batch_size == batch_size

        missing = TensorDict({"a": torch.ones(2)}, batch_size=[2])
        with pytest.raises(KeyError):
            missing.data = TensorDict({"z": torch.zeros(2)}, batch_size=[2])
        assert missing.get("a").tolist() == [1.0, 1.0]

    def test_data_setter_mismatch_writes_nothing(self):
        td = TensorDict({"a": torch.zeros(2), "b": torch.zeros(2)}, batch_size=[2])

        with pytest.raises(RuntimeError, match="batch size"):
            td.data = TensorDict({"a": torch.ones(3)}, batch_size=[3])
        assert td.get("a").tolist() == [0.0, 0.0]
        td._check_batch_size(raise_exception=True)

        with pytest.raises(KeyError, match="'z'"):
            td.data = TensorDict(
                {"a": torch.ones(2), "z": torch.ones(2)}, batch_size=[2]
            )
        assert td.get("a").tolist() == [0.0, 0.0]
        assert td.get("b").tolist() == [0.0, 0.0]

        nested = TensorDict(
            {
                "a": torch.zeros(2),
                "b": TensorDict(c=torch.zeros(2, 3), batch_size=[2, 3]),
            },
            batch_size=[2],
        )
        with pytest.raises(RuntimeError, match="at key 'b'"):
            nested.data = TensorDict(
                {
                    "a": torch.ones(2),
                    "b": TensorDict(c=torch.ones(2, 4), batch_size=[2]),
                },
                batch_size=[2],
            )
        assert nested.get("a").tolist() == [0.0, 0.0]
        assert nested.get(("b", "c")).shape == (2, 3)
        nested._check_batch_size(raise_exception=True)

        def stack(shape, bs):
            return LazyStackedTensorDict(
                *[
                    TensorDict(x=TensorDict(y=torch.zeros(shape), batch_size=bs))
                    for _ in range(2)
                ]
            )

        lazy = stack((3, 1), [3])
        with pytest.raises(RuntimeError, match=r"at key \('0', 'x'\)"):
            lazy.data = stack((3, 2), [])
        assert all(
            member.get(("x", "y")).shape == (3, 1) for member in lazy.tensordicts
        )

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

    @pytest.mark.parametrize(
        "batch_size,names",
        [([3], ["x"]), ([3, 2, 4], ["x", "y", None]), ([5, 6], ["x", "y"])],
    )
    def test_empty_batch_size_names(self, batch_size, names):
        # A new batch size keeps the leading names, as the batch_size setter does
        td = TensorDict(a=torch.zeros(3, 2), batch_size=[3, 2], names=["x", "y"])
        td_empty = td.empty(batch_size=batch_size)
        assert td_empty.batch_size == torch.Size(batch_size)
        assert td_empty.names == names
        assert td_empty.unsqueeze(0).names == [None, *names]
        td = TensorDict(a=torch.zeros(3, 2), batch_size=[3, 2], names=[None, "y"])
        assert not td.empty(batch_size=[3])._has_names()

    def test_empty_names_misfit_warns(self, monkeypatch):
        # empty() keeps names that the names setter would reject or drop, with
        # a FutureWarning: from 0.17 it applies them as the setter does (#2711)
        # Those names do not fit the batch size, which TD_CHECK_INVARIANTS rejects.
        monkeypatch.setattr(tensordict._td, "_CHECK_INVARIANTS", False)

        @tensorclass
        class Data:
            a: torch.Tensor

        class Typed(TypedTensorDict):
            a: torch.Tensor

        td = TensorDict(a=torch.zeros(3), batch_size=[3])
        lazy = lazy_stack([td, td])
        for obj, kwargs, batch_size, names, future in (
            (td, {"batch_size": [], "names": ["a"]}, [], ["a"], "drops"),
            (td, {"batch_size": [5, 3], "names": ["a"]}, [5, 3], ["a"], "raises"),
            (td, {"names": ["a", "b"]}, [3], ["a", "b"], "raises"),
            (
                TensorDict(batch_size=[2, 3]),
                {"names": ["x", "x"]},
                [2, 3],
                ["x", "x"],
                "raises",
            ),
            (
                Data(a=torch.zeros(3), batch_size=[3]),
                {"names": ["a", "b"]},
                [3],
                ["a", "b"],
                "raises",
            ),
            # a stack of two 0-d members, whose members get the names ["b"]
            (
                lazy_stack([TensorDict(), TensorDict()]),
                {"names": ["a", "b"]},
                [2],
                ["a", "b"],
                "raises",
            ),
            # the empty stack of the members, kept for names of another length
            (lazy, {"batch_size": [], "names": ["a"]}, [2], ["a"], "drops"),
            (
                lazy,
                {"recurse": True, "batch_size": [4], "names": ["x", "y"]},
                [2],
                ["x"],
                "raises",
            ),
        ):
            # the members of a stack may also warn of their names, as before
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                empty = obj.empty(**kwargs)
            ours = [w for w in record if str(w.message).startswith("empty(")]
            assert len(ours) == 1
            assert ours[0].category is FutureWarning
            assert re.search(
                r"empty\(batch_size=.*\) returns a tensordict with the "
                r"(batch size .* and the )?names .*Starting with TensorDict 0\.17, "
                r"the default will change to (returning the batch size .* and )?"
                rf"applying the names as the names setter does, which {future}",
                str(ours[0].message),
            )
            assert ours[0].filename == __file__
            assert empty.batch_size == torch.Size(batch_size)
            assert empty.names == names
        # names that fit do not warn
        params = TensorDictParams(TensorDict(a=torch.zeros(3), batch_size=[3]))
        typed = Typed(a=torch.zeros(3), batch_size=[3])
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            for obj in (td, params, typed, lazy):
                for recurse in (False, True):
                    names = [f"d{i}" for i in range(obj.ndim)]
                    assert obj.empty(recurse, names=names).names == names
                    empty = obj.empty(recurse, batch_size=[5, 4], names=["p", None])
                    assert empty.names == ["p", None]
                    assert obj.empty(recurse, batch_size=[], names=[]).names == []
            # where empty() applies the names setter already, it raises as
            # before, without the warning
            for obj, recurse in ((td, True), (params, False), (typed, False)):
                with pytest.raises(ValueError, match="length of the dimension"):
                    obj.empty(recurse, names=["a", "b"])
        # and on a 0-d batch, it drops the names as before, with the warning of
        # the names setter only
        for obj, recurse in ((td, True), (params, False), (typed, False)):
            with pytest.warns(FutureWarning, match="Got the dimension names") as record:
                empty = obj.empty(recurse, batch_size=[], names=["a"])
            assert empty.names == []
            assert not any(str(w.message).startswith("empty(") for w in record)

    @pytest.mark.parametrize("shared", [None, "memory", "memmap"])
    def test_empty_recurse_kwargs(self, shared, tmpdir):
        # empty(recurse=True) keeps the entries without tensors ("b", "c", "d").
        # The new batch size, names and device must not reach those of td.
        td = TensorDict(
            a=torch.zeros(3, 2),
            b=TensorDict(batch_size=[3, 2]),
            c=NonTensorData("c", batch_size=[3, 2]),
            d=TensorDict(e=torch.zeros(3, 2, 5), batch_size=[3, 2]),
            batch_size=[3, 2],
            names=["x", "y"],
            device="cpu",
        )
        if shared == "memory":
            td.share_memory_()
        elif shared == "memmap":
            td.memmap_(tmpdir, share_non_tensor=True)
        td_empty = td.empty(
            recurse=True, batch_size=[3, 2, 4], names=["p", "q", "r"], device=None
        )
        assert td_empty.batch_size == torch.Size([3, 2, 4])
        assert td_empty.names == ["p", "q", "r"]
        assert td_empty.device is None
        assert set(td_empty.keys(True)) == {"b", "c", "d"}
        assert td_empty["c"] == "c"
        for key in ("b", "c", "d"):
            assert td_empty.get(key).batch_size == torch.Size([3, 2, 4])
            assert td_empty.get(key).names == ["p", "q", "r"]
            assert td_empty.get(key).device is None
            assert td.get(key).batch_size == torch.Size([3, 2])
            assert td.get(key).names == ["x", "y"]
            assert td.get(key).device == torch.device("cpu")
        td_empty = td.empty(recurse=True, batch_size=[3])
        assert td_empty.batch_size == torch.Size([3])
        assert td_empty["d"].batch_size == torch.Size([3, 2])

    @pytest.mark.parametrize(
        "batch_size,key",
        [([10, 4], "'sub'"), ([10], "'sub'"), ([4, 5], r"\('sub', 'nested'\)")],
    )
    def test_empty_recurse_batch_size_mismatch(self, batch_size, key):
        # A sub-tensordict whose batch size does not start with the new one raises
        td = TensorDict(
            a=torch.zeros(4),
            sub=TensorDict(
                nested=TensorDict(b=torch.zeros(4, 3), batch_size=[4, 3]),
                batch_size=[4],
            ),
            batch_size=[4],
        )
        with pytest.raises(RuntimeError, match=f"to the entry {key}"):
            td.empty(recurse=True, batch_size=batch_size)
        # the source is unchanged
        assert td["sub"].batch_size == torch.Size([4])

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

    @pytest.mark.parametrize("inplace", [True, False])
    def test_exclude_nested_unravels_keys_once(self, inplace, monkeypatch):
        # exclude unravels the keys once; the sub-tensordicts get sub-keys
        # that are already unravelled (#2558).
        def leaf():
            return torch.zeros(3, 1)

        td = TensorDict(
            a=leaf(),
            data=TensorDict(
                done=leaf(),
                states=leaf(),
                deep=TensorDict(x=leaf(), y=leaf(), batch_size=[3]),
                batch_size=[3],
            ),
            stack=lazy_stack([TensorDict(u=leaf(), v=leaf()) for _ in range(3)]),
            next=TensorDict(b=leaf()),
            batch_size=[3],
        )

        def unravel_key(key):
            raise AssertionError(f"exclude unravelled {key} again")

        monkeypatch.setattr(tensordict._td, "unravel_key", unravel_key)
        monkeypatch.setattr(tensordict_base, "unravel_key", unravel_key)
        excluded = td.exclude(
            ("next",),
            ("data", ("done",)),
            ("data", "deep", "x"),
            ("data", "missing"),
            ("stack", "u"),
            inplace=inplace,
        )
        assert (excluded is td) is inplace
        assert set(excluded.keys(True, True)) == {
            "a",
            ("data", "states"),
            ("data", "deep", "y"),
            ("stack", "v"),
        }

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

    def test_expand_minus_one(self):
        # A -1 keeps the size of its dim, as in torch.Tensor.expand
        td = TensorDict(
            {"a": torch.zeros(1, 3, 4), "b": {"c": torch.zeros(1, 3)}}, [1, 3]
        )
        parent = TensorDict({"a": torch.zeros(2, 1, 3, 4)}, [2, 1, 3])
        sub_td = parent._get_sub_tensordict(0)
        for shape in [(2, -1, 3), (5, -1, -1), (-1, -1)]:
            expected = torch.zeros(1, 3).expand(shape).shape
            for out in (
                td.expand(*shape),
                td.broadcast_to(shape),
                sub_td.expand(*shape),
                sub_td.broadcast_to(shape),
            ):
                assert out.batch_size == expected
                assert out["a"].shape == (*expected, 4)
            assert td.expand(*shape)["b"].batch_size == expected
        with pytest.raises(RuntimeError, match="leading, non-existing dimension"):
            td.expand(-1, 1, 3)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "td_type", ["tensordict", "view", "unsqueeze", "squeeze", "stack"]
    )
    @pytest.mark.parametrize("update", [True, False])
    # getting values from lazy tensordicts in non-lazy contexts messes things up
    # so we set it to True. When we'll deprecate lazy tensordicts, we will just
    # remove this decorator
    @legacy_lazy_mode()
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

    def test_unflatten_keys_builds_groups_once(self, monkeypatch):
        # unflatten_keys builds each new sub-tensordict once, without a
        # rename_key_ per key, and gives the result the rename loop gave (#2614)
        calls = []
        rename_key_ = TensorDict.rename_key_

        def counting_rename_key_(self, *args, **kwargs):
            calls.append(args)
            return rename_key_(self, *args, **kwargs)

        monkeypatch.setattr(TensorDict, "rename_key_", counting_rename_key_)
        sub = TensorDict({"z": torch.ones(3, 4)}, [3, 4])
        td = TensorDict(
            {
                "a.b": torch.zeros(3),
                "c": torch.ones(3),
                "a.d.e": torch.ones(3, 2),
                "x.y": sub,
                "n": {"q.r": torch.ones(3)},
            },
            [3],
            device="cpu",
        )
        out = td.unflatten_keys(".")
        assert calls == []
        assert list(out.keys()) == ["c", "n", "a", "x"]
        assert list(out.keys(True, True)) == [
            "c",
            ("n", "q.r"),
            ("a", "b"),
            ("a", "d", "e"),
            ("x", "y", "z"),
        ]
        for key in ("a", ("a", "d"), "x"):
            assert type(out.get(key)) is TensorDict
            assert out.get(key).batch_size == (3,)
            assert out.get(key).device == torch.device("cpu")
        assert out.get(("x", "y")).batch_size == (3, 4)
        assert out.get(("a", "b")) is td.get("a.b")
        # a sub-tensordict is moved as a shallow copy, as by the rename loop
        assert out.get(("x", "y")) is not sub
        assert out.get(("x", "y", "z")) is sub.get("z")
        out["x", "y", "w"] = torch.zeros(3, 4)
        assert "w" not in sub.keys()
        assert list(td.keys()) == ["a.b", "c", "a.d.e", "x.y", "n"]
        # a key that meets another key still goes through rename_key_ and raises
        with pytest.raises(KeyError, match="will override an existing"):
            TensorDict(
                {"a.b": torch.zeros(3), "a.b.c": torch.zeros(3)}
            ).unflatten_keys()
        assert calls

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

    def test_flatten_0d_context_manager_restores_batch(self):
        # flatten on a 0-dim tensordict unsqueezes to 1-d. Exit must restore
        # batch [] and the values written inside the block.
        td = TensorDict({"a": torch.tensor([1.0, 2.0])}, batch_size=[])
        with td.flatten() as flat:
            assert flat.batch_size == torch.Size([1])
            flat["a"] = torch.tensor([[3.0, 4.0]])
        assert td.batch_size == torch.Size([])
        assert td["a"].shape == torch.Size([2])
        assert torch.equal(td["a"], torch.tensor([3.0, 4.0]))

    def test_flatten_context_manager_writes_back(self):
        td = TensorDict({"a": torch.arange(6).reshape(2, 3)}, batch_size=[2, 3])
        with td.flatten(0, 1) as flat:
            assert flat.batch_size == torch.Size([6])
            flat["a"] = torch.arange(6) + 1
        assert td.batch_size == torch.Size([2, 3])
        assert torch.equal(td["a"], (torch.arange(6) + 1).reshape(2, 3))

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
        assert td["1", "tensor"] is pytree[1]["tensor"]
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

    def test_from_any_structseq_warns(self):
        # from_any converts a torch.return_types result like a tuple, with the
        # keys "0", "1", and warns that 0.17 gives its field names instead
        t = torch.arange(12.0).reshape(3, 4)
        NT = namedtuple("NT", ["a", "b"])
        calls = [
            (lambda: TensorDict.from_any(t.sort(1)), ()),
            (lambda: from_any(t.sort(1)), ()),
            (lambda: TensorDict.from_any(t.sort(1), auto_batch_size=True), ()),
            (lambda: TensorDict.from_namedtuple(NT(a=t, b=t.sort(1))), ("b",)),
            (lambda: TensorDict.from_any({"x": t.sort(1)}), ("x",)),
            (lambda: TensorDict.from_dict({"x": t.sort(1)}), ("x",)),
        ]
        for call, key in calls:
            with pytest.warns(
                FutureWarning, match="Starting with TensorDict 0.17"
            ) as record:
                td = call()
            assert [w.filename for w in record] == [__file__]
            sub = td[key] if key else td
            assert set(sub.keys()) == {"0", "1"}
            assert (sub["0"] == t.sort(1).values).all()
            assert (sub["1"] == t.sort(1).indices).all()
        # other tuples, and the explicit from_namedtuple and from_tuple, do not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert set(TensorDict.from_any((t, t)).keys()) == {"0", "1"}
            assert set(TensorDict.from_any(NT(a=t, b=t)).keys()) == {"a", "b"}
            assert set(TensorDict.from_any(tuple(t.sort(1))).keys()) == {"0", "1"}
            td = TensorDict.from_namedtuple(t.sort(1))
            assert set(td.keys()) == {"values", "indices"}
            assert set(TensorDict.from_tuple(t.sort(1)).keys()) == {"0", "1"}

    def test_from_any_userdict(self):
        class D(UserDict): ...

        d = D(a=0)
        assert TensorDict.from_any(d)["a"] == 0
        assert isinstance(TensorDict.from_any(d)["a"], torch.Tensor)

    def test_module_level_from_any_from_tuple(self):
        # tensordict.from_any and tensordict.from_tuple match the classmethods,
        # also for dicts and for containers that hold a dict
        @dataclass
        class MyClass:
            d: dict

        NT = namedtuple("NT", ["a", "d"])
        objs = [
            {"a": torch.zeros(3), "b": {"c": torch.ones(3, 2)}},
            UserDict(a=torch.zeros(3)),
            (torch.zeros(3), {"x": torch.ones(3)}),
            NT(a=torch.zeros(3), d={"x": torch.ones(3)}),
            MyClass(d={"x": torch.ones(3)}),
        ]
        for obj in objs:
            td = from_any(obj, auto_batch_size=True)
            expected = TensorDict.from_any(obj, auto_batch_size=True)
            assert isinstance(td, TensorDict)
            assert td.batch_size == expected.batch_size == (3,)
            assert (td == expected).all()
        obj = (torch.zeros(3), {"x": torch.ones(3)})
        td = from_tuple(obj, batch_size=[3])
        assert isinstance(td, TensorDict)
        assert td.batch_size == (3,)
        assert (td == TensorDict.from_tuple(obj, batch_size=[3])).all()

    @pytest.mark.parametrize(
        "name,args,kwargs",
        [
            ["from_dict", ({"a": torch.zeros(3), "b": {"c": torch.ones(3, 2)}},), {}],
            [
                "from_dict",
                ({"a": torch.zeros(3, 2)},),
                {"auto_batch_size": True, "batch_dims": 1},
            ],
            # https://github.com/pytorch/tensordict/issues/1954
            ["from_dict", ({"a": torch.zeros(3, 2)},), {"batch_dims": 1}],
            [
                "from_namedtuple",
                (namedtuple("NT", ["a", "b"])(torch.zeros(3), torch.ones(3, 2)),),
                {"auto_batch_size": True},
            ],
            [
                "from_struct_array",
                (
                    np.array(
                        [(9, 81.0), (3, 27.0)], dtype=[("age", "i4"), ("weight", "f4")]
                    ),
                ),
                {},
            ],
            [
                "from_pytree",
                ({"a": torch.zeros(3), "b": (torch.ones(3, 2),)},),
                {"batch_size": [3]},
            ],
            ["fromkeys", (["a", "b"], 1), {}],
            ["from_module", (nn.Linear(2, 3),), {"lock": True}],
            ["from_modules", (nn.Linear(2, 3), nn.Linear(2, 3)), {}],
        ],
    )
    def test_module_level_constructor(self, name, args, kwargs):
        # The functions documented in td.rst match the classmethods they forward
        # to. The default of from_module(lock) differs between the two, so
        # that case passes it explicitly.
        out = getattr(tensordict, name)(*args, **kwargs)
        expected = getattr(TensorDict, name)(*args, **kwargs)
        assert type(out) is type(expected)
        assert out.batch_size == expected.batch_size
        assert out.is_locked == expected.is_locked
        assert set(out.keys(True, True)) == set(expected.keys(True, True))
        assert (out == expected).all()

    def test_module_level_stack(self):
        td0 = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3, 2)}, batch_size=[3])
        td1 = td0 + 1
        out = tensordict.stack([td0, td1], dim=1)
        assert type(out) is TensorDict
        assert out.batch_size == (3, 2)
        assert (out == torch.stack([td0, td1], dim=1)).all()

        out = tensordict.maybe_dense_stack([td0, td1])
        assert type(out) is TensorDict
        assert (out == torch.stack([td0, td1])).all()
        # Different keys cannot be stacked densely, so the result is a lazy stack
        td2 = TensorDict(a=torch.zeros(3), d=torch.ones(3), batch_size=[3])
        out = tensordict.maybe_dense_stack([td0, td2])
        assert isinstance(out, LazyStackedTensorDict)
        assert out.batch_size == (2, 3)
        assert out[0] is td0
        assert out[1] is td2

    def test_module_level_save_load_memmap(self, tmp_path):
        td = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3, 2)}, batch_size=[3])
        tensordict.save(td, tmp_path / "saved")
        assert not td.is_memmap()
        loaded = tensordict.load(tmp_path / "saved")
        assert type(loaded) is TensorDict
        assert loaded.batch_size == td.batch_size
        assert (loaded == td).all()

        out = tensordict.memmap(td, tmp_path / "memmap")
        assert type(out) is TensorDict
        assert out.is_memmap()
        assert (out == td).all()
        assert (TensorDict.load_memmap(tmp_path / "memmap") == td).all()

    def test_module_level_from_consolidated(self, tmp_path):
        td = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3, 2)}, batch_size=[3])
        td.consolidate(filename=tmp_path / "td.consolidated")
        out = tensordict.from_consolidated(tmp_path / "td.consolidated")
        assert type(out) is TensorDict
        assert out.batch_size == td.batch_size
        assert (out == td).all()

    @pytest.mark.skipif(not _has_h5py, reason="h5py not installed")
    def test_module_level_from_h5(self, tmp_path):
        # https://github.com/pytorch/tensordict/issues/1942
        path = tmp_path / "data.h5"
        with h5py.File(path, "w") as file:
            file["a"] = np.arange(3)
        out = tensordict.from_h5(path)
        assert (out["a"] == torch.arange(3)).all()
        out.close()
        out = tensordict.from_h5(path, mode="r+", batch_size=[3])
        assert out.batch_size == (3,)
        out["a"] = torch.zeros(3, dtype=torch.long)
        out.close()
        out = tensordict.from_h5(path)
        assert (out["a"] == 0).all()
        out.close()

    # Module-level function -> (the TensorDict method it forwards to, the
    # parameters whose default may differ). from_module locks its result by
    # default, as documented, and TensorDict.from_module does not.
    _MODULE_FUNCTION_TARGETS = {
        "from_any": ("from_any", ()),
        "from_consolidated": ("from_consolidated", ()),
        "from_csv": ("from_csv", ()),
        "from_dict": ("from_dict", ()),
        "from_h5": ("from_h5", ()),
        "from_json": ("from_json", ()),
        "from_module": ("from_module", ("lock",)),
        "from_modules": ("from_modules", ()),
        "from_namedtuple": ("from_namedtuple", ()),
        "from_pandas": ("from_pandas", ()),
        "from_parquet": ("from_parquet", ()),
        "from_pytree": ("from_pytree", ()),
        "from_struct_array": ("from_struct_array", ()),
        "from_tuple": ("from_tuple", ()),
        "from_zarr": ("from_zarr", ()),
        "fromkeys": ("fromkeys", ()),
        "lazy_stack": ("lazy_stack", ()),
        # TensorDict.load takes (prefix, *args, **kwargs) and calls load_memmap.
        "load": ("load_memmap", ()),
        "load_memmap": ("load_memmap", ()),
        "maybe_dense_stack": ("maybe_dense_stack", ()),
        "memmap": ("memmap", ()),
        "save": ("save", ()),
    }

    @pytest.mark.parametrize("name", sorted(_MODULE_FUNCTION_TARGETS))
    def test_module_level_signature(self, name):
        # https://github.com/pytorch/tensordict/issues/1954
        target_name, defaults_may_differ = self._MODULE_FUNCTION_TARGETS[name]
        # The first parameter is the input (``data`` where the method has
        # ``self``), and some functions name it differently.
        function_params = list(
            inspect.signature(getattr(tensordict, name)).parameters.values()
        )[1:]
        target_params = list(
            inspect.signature(getattr(TensorDict, target_name)).parameters.values()
        )[1:]
        function_params = {p.name: p for p in function_params}
        target_params = {p.name: p for p in target_params}
        function_var_kw = any(p.kind is p.VAR_KEYWORD for p in function_params.values())
        target_var_kw = any(p.kind is p.VAR_KEYWORD for p in target_params.values())
        assert function_var_kw or not target_var_kw
        for param_name, param in target_params.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if param_name not in function_params:
                assert function_var_kw, param_name
                continue
            assert function_params[param_name].kind == param.kind, param_name
            if param_name not in defaults_may_differ:
                assert function_params[param_name].default == param.default, param_name
        for param_name, param in function_params.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            assert param_name in target_params or target_var_kw, param_name

    def test_module_level_forwards_arguments(self, tmp_path):
        # https://github.com/pytorch/tensordict/issues/1954
        td = TensorDict(a=torch.arange(3), nested={"b": torch.ones(3)}, batch_size=[3])
        stacked = tensordict.lazy_stack([td, td], stack_dim_name="time")
        assert stacked.names[0] == "time"

        saved = tensordict.save(td, tmp_path / "saved.tdz", archive=True)
        assert (saved == td).all()
        nested = tensordict.load_memmap(
            tmp_path / "saved.tdz", subpath="nested", num_threads=2
        )
        assert (nested["b"] == 1).all()
        nested = tensordict.load(tmp_path / "saved.tdz", subpath=("nested",))
        assert (nested["b"] == 1).all()

        tensordict.memmap(td, tmp_path / "memmap")
        with pytest.raises(RuntimeError, match="already exists"):
            tensordict.memmap(td, tmp_path / "memmap", existsok=False)

        module = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
        assert "1" not in tensordict.from_module(module).keys()
        assert "1" in tensordict.from_module(module, filter_empty=False).keys()
        out = tensordict.from_dict(
            {"a": torch.zeros(3)}, batch_size=[3], names=["time"]
        )
        assert out.names == ["time"]

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

    @pytest.mark.parametrize("batch_dims", [None, 1, 2])
    def test_from_any_dataclass_batch_dims(self, batch_dims):
        @dataclass
        class MyClass:
            a: torch.Tensor
            b: Any

        obj = MyClass(
            a=torch.zeros(3, 4, 5),
            b=MyClass(a=torch.zeros(3, 4, 5, 6), b=torch.zeros(3, 4, 5)),
        )
        td = TensorDict.from_any(obj, auto_batch_size=True, batch_dims=batch_dims)
        expected = torch.Size([3, 4, 5][:batch_dims])
        assert td.batch_size == expected
        assert td["b"].batch_size == expected
        td_dataclass = TensorDict.from_dataclass(
            obj, auto_batch_size=True, batch_dims=batch_dims
        )
        assert td_dataclass.batch_size == expected

    def test_from_any_dataclass_batch_dims_non_tensor_stack(self):
        @dataclass
        class MyClass:
            a: torch.Tensor
            b: Any

        obj = MyClass(a=torch.zeros(3, 4, 5), b=[["x", "y"], ["z", "w"], ["u", "v"]])
        td = TensorDict.from_any(obj, auto_batch_size=True, batch_dims=1)
        assert td.batch_size == (3,)
        assert td.get("b").batch_size == (3, 2)

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

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"batch_size": [2, 3]},
            {"auto_batch_size": True},
            {"batch_dims": 2},
            {"batch_size": [2, 3], "lazy": True},
            {"auto_batch_size": True, "lazy": True},
            {"batch_size": [2, 3], "lazy": False},
        ],
    )
    def test_from_list_names(self, kwargs):
        # https://github.com/pytorch/tensordict/issues/2370
        data = [{"a": torch.zeros(3)}, {"a": torch.zeros(3)}]
        td = TensorDict.from_list(data, names=["x", "y"], **kwargs)
        assert td.batch_size == (2, 3)
        assert td.names == ["x", "y"]

    def test_from_list_names_mismatch(self):
        data = [{"a": torch.zeros(3)}, {"a": torch.zeros(3)}]
        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
            td = TensorDict.from_list(data, batch_dims=1, names=["x", "y"])
        assert td.names == [None]
        with pytest.raises(ValueError):
            TensorDict.from_list(data, batch_size=[2, 3], names=["x"])

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

    @pytest.mark.parametrize("cls", [LazyStackedTensorDict, PersistentTensorDict])
    def test_from_module_other_classes(self, cls):
        net = nn.Linear(2, 3)
        td = cls.from_module(net)
        assert type(td) is TensorDict
        assert (td == TensorDict.from_module(net)).all()

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
    @pytest.mark.parametrize("preserve_module_state", [False, True])
    def test_from_modules(self, as_module, lazy_stack, preserve_module_state):
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
            with params.to_module(
                empty_module, preserve_module_state=preserve_module_state
            ):
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

    def test_from_pytree_non_str_keys(self, tmpdir):
        # Dict keys that are not NestedKeys come back unchanged, not as tensors
        def key_types(pytree):
            return {repr(key): type(key) for key in pytree}

        keys = [0, 1.5, True, None, (0, 1), ("a", 0), "a"]
        pytree = {key: torch.full((3,), float(i)) for i, key in enumerate(keys)}
        td = TensorDict.from_pytree(pytree)
        pytree_recon = td.to_pytree()
        assert key_types(pytree_recon) == key_types(pytree)
        for key, value in pytree.items():
            assert (pytree_recon[key] == value).all()
        # also when the keys have batch dims, and after memmap and load_memmap
        td.memmap(tmpdir)
        for td_other in (
            torch.stack([td, td.clone()]),
            td.unsqueeze(0),
            TensorDict.load_memmap(tmpdir, allow_pickle=False),
        ):
            assert key_types(td_other.to_pytree()) == key_types(pytree)
        # The keys take no part in the batch size
        assert TensorDict.from_pytree(pytree, batch_size=[3]).batch_size == (3,)
        assert TensorDict.from_pytree(pytree, auto_batch_size=True).batch_size == (3,)

    def test_from_pytree_non_str_keys_nested_values(self):
        # Lists, tuples, dicts and namedtuples under non-str keys are rebuilt
        NT = namedtuple("NT", ["a"])
        t = torch.zeros(3)
        value = [t, (t, {"x": t}), {0: t}, NT(a=t)]
        for key in ("s", 1, (1, 2)):
            pytree = {key: value}
            pytree_recon = TensorDict.from_pytree(pytree).to_pytree()
            assert torch.utils._pytree.tree_structure(
                pytree_recon
            ) == torch.utils._pytree.tree_structure(pytree)

    @pytest.mark.parametrize(
        "method,args,fields",
        [
            ("max", (), {"values", "indices"}),
            ("sort", (), {"values", "indices"}),
            ("topk", (2,), {"values", "indices"}),
            ("aminmax", (), {"min", "max"}),
        ],
        ids=["max", "sort", "topk", "aminmax"],
    )
    def test_from_namedtuple_return_types(self, method, args, fields):
        # torch.return_types are structseqs: named fields, but no _fields
        t = torch.randn(3, 4)
        result = getattr(t, method)(*args, dim=1)
        td = TensorDict.from_namedtuple(result, batch_size=[3])
        assert set(td.keys()) == fields
        for name in fields:
            assert (td[name] == getattr(result, name)).all()

    def test_to_namedtuple_dest_cls(self):
        # https://github.com/pytorch/tensordict/issues/2372
        Outer = namedtuple("Outer", ["a", "nested"])
        td = TensorDict(a=torch.zeros(2), nested=TensorDict(p=torch.ones(2)))
        nt = td.to_namedtuple(dest_cls=Outer)
        assert type(nt) is Outer
        assert type(nt.nested).__name__ == "GenericDict"
        assert nt.nested._fields == ("p",)
        assert (nt.nested.p == 1).all()
        # a nested level that dest_cls takes keeps dest_cls
        Same = namedtuple("Same", ["a", "b"], defaults=[None])
        td = TensorDict(a=torch.zeros(2), b=TensorDict(a=torch.ones(2)))
        nt = td.to_namedtuple(dest_cls=Same)
        assert type(nt) is Same
        assert type(nt.b) is Same
        assert nt.b.b is None

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

    def test_from_pytree_namedtuple_batch_dims(self):
        # batch_dims without batch_size used to raise for namedtuples
        NT = namedtuple("NT", ["a", "b"])
        nt = NT(a=torch.zeros(3, 2), b=torch.ones(3))
        td = TensorDict.from_pytree(nt, auto_batch_size=True, batch_dims=1)
        assert td.batch_size == (3,)
        pytree_recon = td.to_pytree()
        assert type(pytree_recon) is NT
        assert (pytree_recon.a == nt.a).all()
        assert (pytree_recon.b == nt.b).all()
        # without auto_batch_size, batch_dims is ignored, as for lists and dicts
        assert TensorDict.from_pytree(nt, batch_dims=1).batch_size == ()
        # an int batch_size with batch_dims gives the same batch size as for lists
        td = TensorDict.from_pytree(nt, batch_size=3, batch_dims=1)
        assert td.batch_size == (3,)
        assert (
            td.batch_size
            == TensorDict.from_pytree(list(nt), batch_size=3, batch_dims=1).batch_size
        )
        # batch_size applies without batch_dims too, as for lists and dicts
        for batch_size in ([3], 3):
            td = TensorDict.from_pytree(nt, batch_size=batch_size)
            assert td.batch_size == (3,)
            assert type(td.to_pytree()) is NT
        # and a batch size that does not fit the leaves raises, as for lists
        with pytest.raises(RuntimeError):
            TensorDict.from_pytree(list(nt), batch_size=[4])
        with pytest.raises(RuntimeError):
            TensorDict.from_pytree(nt, batch_size=[4])

    def test_from_pytree_namedtuple_nested(self):
        # dict and namedtuple fields come back with their own type
        Outer = namedtuple("Outer", ["a", "nested"])
        Same = namedtuple("Same", ["a", "nested"])
        Inner = namedtuple("Inner", ["p", "name"])
        a = torch.zeros(3)

        def check(v1, v2):
            assert (v1 == v2).all() if torch.is_tensor(v1) else v1 == v2

        for nested in (
            {"x": torch.ones(3), "y": {"z": torch.ones(3)}},
            Inner(p=torch.ones(3), name="n"),
            Same(a=torch.ones(3), nested="n"),
            Outer(a=torch.ones(3), nested=None),
        ):
            pytree = Outer(a=a, nested=nested)
            for kwargs in ({}, {"batch_size": [3]}):
                recon = TensorDict.from_pytree(pytree, **kwargs).to_pytree()
                assert type(recon) is Outer
                assert type(recon.nested) is type(nested)
                torch.utils._pytree.tree_map(check, pytree, recon)
        # a dict field that from_pytree does not take (here, it holds a str)
        # goes through from_any, as before, and comes back as on main: as the
        # outer class if it takes the keys, and as a GenericDict otherwise
        recon = TensorDict.from_pytree(
            Outer(a=a, nested={"a": torch.ones(3), "nested": "n"})
        ).to_pytree()
        assert type(recon) is Outer
        assert type(recon.nested) is Outer
        assert recon.nested.nested == "n"
        for kwargs in ({}, {"batch_size": [3]}):
            recon = TensorDict.from_pytree(
                Outer(a=a, nested={"s": "x", "t": torch.ones(3)}), **kwargs
            ).to_pytree()
            assert type(recon.nested).__name__ == "GenericDict"
            assert recon.nested.s == "x"
            assert (recon.nested.t == 1).all()
        # so does a dict field that holds a list: the list stays a tensor, as
        # on main, rather than a nested TensorDict as from_pytree would make it
        pytree = Outer(a=a, nested={"l": [1, 2, 3]})
        td = TensorDict.from_pytree(pytree)
        assert isinstance(td["nested", "l"], torch.Tensor)
        assert (td["nested", "l"] == torch.tensor([1, 2, 3])).all()
        assert TensorDict.from_pytree(pytree, auto_batch_size=True).batch_size == (3,)
        assert TensorDict.from_pytree(pytree, batch_size=[3]).batch_size == (3,)
        recon = td.to_pytree().nested
        assert type(recon).__name__ == "GenericDict"
        assert (recon.l == torch.tensor([1, 2, 3])).all()
        # so does a field that holds an empty dict or namedtuple: as on main,
        # it does not stop auto_batch_size from finding the batch dims
        Empty = namedtuple("Empty", [])
        for nested in ({}, Empty(), {"x": torch.ones(3), "y": {}}, Inner(a, {})):
            pytree = Outer(a=a, nested=nested)
            td = TensorDict.from_pytree(pytree, auto_batch_size=True)
            assert td.batch_size == (3,)
            assert type(td.to_pytree().nested).__name__ == "GenericDict"

    def test_from_pytree_namedtuple_tuple(self):
        # https://github.com/pytorch/tensordict/issues/2160
        # a tuple field comes back as a tuple
        Outer = namedtuple("Outer", ["a", "nested"])
        Inner = namedtuple("Inner", ["p", "name"])
        a = torch.zeros(3)

        def check(v1, v2):
            assert (v1 == v2).all() if torch.is_tensor(v1) else v1 == v2

        for nested in (
            (torch.ones(3), torch.ones(3)),
            (torch.ones(3), 1, 2.0),
            (torch.ones(3), {"x": torch.ones(3)}, Inner(p=torch.ones(3), name="n")),
        ):
            pytree = Outer(a=a, nested=nested)
            for kwargs in ({}, {"auto_batch_size": True}):
                td = TensorDict.from_pytree(pytree, **kwargs)
                assert td["nested", "_pytree_type"] is tuple
                recon = td.to_pytree()
                assert type(recon) is Outer
                assert type(recon.nested) is tuple
                torch.utils._pytree.tree_map(check, pytree, recon)
        pytree = Outer(a=a, nested=(torch.ones(3), {"x": torch.ones(3)}))
        recon = TensorDict.from_pytree(pytree, batch_size=[3]).to_pytree()
        torch.utils._pytree.tree_map(check, pytree, recon)
        # a tuple that holds a str or a list, and an empty tuple, go through
        # from_any, as on main
        for nested in (("a", "b"), ([1, 2],), ()):
            td = TensorDict.from_pytree(Outer(a=a, nested=nested))
            assert "_pytree_type" not in td["nested"].keys()
        assert isinstance(
            TensorDict.from_pytree(Outer(a=a, nested=([1, 2],)))["nested", "0"],
            torch.Tensor,
        )

    def test_from_pytree_subclasses(self):
        # https://github.com/pytorch/tensordict/issues/2371
        pytree = OrderedDict(
            b=torch.zeros(3, 2).max(1),
            a=defaultdict(list, c=torch.ones(3)),
        )
        pytree_recon = TensorDict.from_pytree(pytree).to_pytree()
        assert type(pytree_recon) is OrderedDict
        assert list(pytree_recon) == ["b", "a"]
        assert type(pytree_recon["b"]) is torch.return_types.max
        assert type(pytree_recon["a"]) is defaultdict
        assert pytree_recon["a"].default_factory is None

        def check(v1, v2):
            assert (v1 == v2).all()

        torch.utils._pytree.tree_map(check, pytree, pytree_recon)

    def test_from_pytree_memmap_without_pickle(self, tmpdir):
        # https://github.com/pytorch/tensordict/issues/2550
        # dict, OrderedDict, list and tuple are pickled for older versions,
        # and saved by name for load_memmap
        pytree = {
            "model": nn.Linear(2, 3).state_dict(),
            "seq": [torch.zeros(2), (torch.ones(1), {"x": torch.ones(2)})],
        }
        td = TensorDict.from_pytree(pytree)
        assert td["model", "_pytree_type"] is OrderedDict
        td.memmap(tmpdir)
        prefix = Path(tmpdir) / "model" / "_pytree_type"
        # meta.json and other.pickle are those that older versions write
        assert set(json.loads((prefix / "meta.json").read_text())) == {
            "_type",
            "_metadata",
            "_is_non_tensor",
        }
        assert pickle.loads((prefix / "other.pickle").read_bytes()) == {
            "data": OrderedDict
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = TensorDict.load_memmap(tmpdir, allow_pickle=False)
        assert loaded["_pytree_type"] is dict
        assert loaded["model", "_pytree_type"] is OrderedDict
        assert loaded["seq", "_pytree_type"] is list
        assert loaded["seq", "1", "_pytree_type"] is tuple
        pytree_recon = loaded.to_pytree()
        assert torch.utils._pytree.tree_structure(
            pytree_recon
        ) == torch.utils._pytree.tree_structure(pytree)
        torch.utils._pytree.tree_map(
            lambda v1, v2: torch.testing.assert_close(v1, v2), pytree, pytree_recon
        )
        # an unknown name raises
        types_file = prefix / "json_types.json"
        saved = json.loads(types_file.read_text())
        types_file.write_text(json.dumps({**saved, "types": {"data": "set"}}))
        with pytest.raises(ValueError, match="Unknown type name 'set'.*json_types"):
            TensorDict.load_memmap(tmpdir, allow_pickle=False)
        # without the names (as older versions write), or with names of
        # another other.pickle, the pickle is loaded
        types_file.write_text(json.dumps({**saved, "pickle_sha256": "0"}))
        with pytest.raises(RuntimeError, match="Refusing to load pickled"):
            TensorDict.load_memmap(tmpdir, allow_pickle=False)
        types_file.unlink()
        with pytest.raises(RuntimeError, match="Refusing to load pickled"):
            TensorDict.load_memmap(tmpdir, allow_pickle=False)
        assert (
            TensorDict.load_memmap(tmpdir, allow_pickle=True)["model"]["_pytree_type"]
            is OrderedDict
        )
        # other classes are still pickled only
        pytree = defaultdict(list, a=torch.zeros(2))
        TensorDict.from_pytree(pytree).memmap(tmpdir / "other")
        with pytest.raises(RuntimeError, match="Refusing to load pickled"):
            TensorDict.load_memmap(tmpdir / "other", allow_pickle=False)
        loaded = TensorDict.load_memmap(tmpdir / "other", allow_pickle=True)
        assert loaded["_pytree_type"] is defaultdict
        assert not (
            Path(tmpdir) / "other" / "_pytree_type" / "json_types.json"
        ).exists()

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

    def test_from_struct_array_packed(self):
        # 12 bytes per record: torch cannot view the "i8" field, which is
        # copied, while the "f4" field still shares the array's memory
        x = np.array([(1, 4.0), (2, 5.0), (3, 6.0)], dtype=[("a", "i8"), ("b", "f4")])
        td = TensorDict.from_struct_array(x)
        assert td.batch_size == (3,)
        assert td["a"].dtype == torch.int64
        assert (td["a"] == torch.tensor([1, 2, 3])).all()
        assert (td["b"] == torch.tensor([4.0, 5.0, 6.0])).all()
        x["b"] += 1
        assert (td["b"] == torch.tensor([5.0, 6.0, 7.0])).all()
        # to_struct_array writes packed records of int64 and float32 leaves
        td = TensorDict(
            a=torch.tensor([1, 2, 3]), b=torch.tensor([4.0, 5.0, 6.0]), batch_size=[3]
        )
        assert (TensorDict.from_struct_array(td.to_struct_array()) == td).all()

    def test_from_struct_array_big_endian(self):
        # torch cannot view big-endian fields, which are copied
        x = np.array([(1.5, 2), (3.5, 4)], dtype=[("x", ">f8"), ("y", ">i4")])
        td = TensorDict.from_struct_array(x)
        assert td.batch_size == (2,)
        assert td["x"].dtype == torch.float64
        assert td["y"].dtype == torch.int32
        assert (td["x"] == torch.tensor([1.5, 3.5], dtype=torch.float64)).all()
        assert (td["y"] == torch.tensor([2, 4], dtype=torch.int32)).all()

    @pytest.mark.parametrize(
        "array",
        [
            np.arange(6)[::-1],
            np.arange(12.0).reshape(6, 2)[::-1, ::-1],
            np.zeros(6, dtype=[("a", "f4"), ("b", "i8")])["b"],
            # NumPy calls these slices contiguous, but torch cannot view them
            np.zeros(3, dtype=[("a", "i8"), ("b", "i4")])["a"][:1],
            np.zeros(3, dtype=[("a", "i8"), ("b", "i4")])["a"][:0],
            np.arange(6, dtype=">i4"),
            np.arange(12.0, dtype=">f8").reshape(6, 2),
            np.array(3, dtype=">i8"),
        ],
        ids=[
            "negative_stride",
            "negative_strides_2d",
            "packed_field",
            "packed_field_size_1",
            "packed_field_size_0",
            "big_endian",
            "big_endian_2d",
            "big_endian_0d",
        ],
    )
    def test_numpy_array_unviewable(self, array):
        # torch cannot share the memory of these arrays, so they are copied
        expected = torch.from_numpy(array.astype(array.dtype.newbyteorder("=")))
        td = TensorDict(a=array, batch_size=array.shape[:1])
        assert td["a"].dtype == expected.dtype
        assert (td["a"] == expected).all()
        td.set("b", array)
        assert (td["b"] == expected).all()
        td["c"] = array
        assert (td["c"] == expected).all()

    @pytest.mark.parametrize(
        "array",
        [
            np.arange(6),
            np.arange(12.0).reshape(6, 2).T,
            np.arange(12, dtype="<i4")[::2],
            np.zeros(6, dtype=[("a", "f4"), ("b", "i4")])["b"],
            np.zeros(6, dtype=bool),
        ],
        ids=["1d", "transposed", "step", "struct_field", "bool"],
    )
    def test_numpy_array_shares_memory(self, array):
        # torch can view these arrays, so the tensordict shares their memory
        td = TensorDict(a=array, batch_size=array.shape[:1])
        td.set("b", array)
        td["c"] = array
        for key in ("a", "b", "c"):
            assert np.shares_memory(td[key].numpy(), array)

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
            (),
            (-1, 0),
            (9, 6, 10, 4),
            (-10, -7, -11, -5),
            (True, 3),
            (3, True),
            (np.int64(3), 4),
            (3, None, 4),
            (3, slice(1, 3)),
            (3, slice(None), 4),
        ],
    )
    def test_getitem_batch_size(self, idx):
        shape = [10, 7, 11, 5]
        shape = torch.Size(shape)
        mocking_tensor = torch.zeros(*shape)
        expected_shape = mocking_tensor[idx].shape
        resulting_shape = _getitem_batch_size(shape, idx)
        assert expected_shape == resulting_shape, (idx, expected_shape, resulting_shape)
        assert type(resulting_shape) is torch.Size

    @pytest.mark.parametrize("shape", [[10], [10, 7], [0], [0, 3]], ids=str)
    @pytest.mark.parametrize(
        "idx",
        [
            slice(None),
            slice(2, 8),
            slice(-3, None),
            slice(None, -20),
            slice(None, None, 3),
            slice(8, 2),
            slice(20, None),
            slice(None, None, -1),
            slice(None, None, 0),
        ],
        ids=str,
    )
    def test_getitem_batch_size_one_slice(self, shape, idx):
        # The index (slice,) of td[a:b] takes a shortcut: the first dim has
        # the length of range(size)[slice], as in the general path, which
        # raises the same ValueError for a step of 0
        shape = torch.Size(shape)

        def outcome(fun):
            try:
                return fun()
            except Exception as err:
                return type(err), str(err)

        expected = outcome(lambda: torch.Size([len(range(shape[0])[idx]), *shape[1:]]))
        result = outcome(lambda: _getitem_batch_size(shape, (idx,)))
        assert result == expected
        assert outcome(lambda: _getitem_batch_size(shape, (idx, Ellipsis))) == expected
        if idx.step is None or idx.step > 0:
            assert result == torch.zeros(shape)[idx].shape
            assert type(result) is torch.Size

    @pytest.mark.parametrize(
        "owner,name",
        [
            (TensorDictBase, "_apply_nest"),
            (TensorDictBase, "_index_tensordict"),
            (TensorDictBase, "__torch_function__"),
            (TensorDictBase, "_convert_to_tensordict"),
            (TensorDictBase, "_compare"),
            (TensorDictBase, "all"),
            (TensorDictBase, "any"),
            (TensorDictBase, "_cast_reduction"),
            (tensordict._td, "_clone_value"),
        ],
        ids=lambda x: getattr(x, "__name__", x),
    )
    def test_hot_path_has_no_function_level_import(self, owner, name):
        # These run on every call, or for every node of a tree: they read the
        # tensordict classes and tables from module globals, since a
        # function-level import costs about 100 ns each time it runs
        imported = [
            instruction.argval
            for instruction in dis.get_instructions(getattr(owner, name))
            if instruction.opname == "IMPORT_NAME"
        ]
        assert not [module for module in imported if module.startswith("tensordict")]

    def test_torch_function_table(self):
        # TensorDictBase.__torch_function__ reads the table that _torch_func
        # fills
        assert tensordict_base._TD_HANDLED_FUNCTIONS is TD_HANDLED_FUNCTIONS
        assert torch.stack in TD_HANDLED_FUNCTIONS

    @pytest.mark.parametrize("idx", [(10, 0), (0, -8), (0, 7), (0, 0, 0, 0, 0)])
    def test_getitem_batch_size_int_tuple_error(self, idx):
        # an int out of range, or too many ints, raise the IndexError of torch
        shape = torch.Size([10, 7, 11, 5])
        with pytest.raises(IndexError) as expected:
            torch.zeros(shape)[idx]
        with pytest.raises(IndexError, match=re.escape(str(expected.value))):
            _getitem_batch_size(shape, idx)

    def test_getitem_bool_list(self):
        # A list of bools is a boolean mask, as in torch
        td = TensorDict(
            {"a": torch.arange(12).view(3, 4), "n": {"b": torch.arange(3)}}, [3]
        )
        sub = td[[True, False, True]]
        assert sub.batch_size == torch.Size([2])
        assert sub["n"].batch_size == torch.Size([2])
        assert (sub["a"] == torch.tensor([[0, 1, 2, 3], [8, 9, 10, 11]])).all()
        assert (sub["n", "b"] == torch.tensor([0, 2])).all()

        td = TensorDict({"a": torch.arange(12).view(3, 4)}, [3, 4])
        sub = td[:, [True, False, True, False]]
        assert sub.batch_size == torch.Size([3, 2])
        assert (sub["a"] == torch.tensor([[0, 2], [4, 6], [8, 10]])).all()

    @pytest.mark.parametrize("index", [[True, False, True], [0, 2]])
    @pytest.mark.parametrize("batch_size", [[3], [3, 4]])
    def test_setitem_list_index_new_key(self, index, batch_size):
        # "new" is missing from td, so it is written through a sub-tensordict
        td = TensorDict({"a": torch.zeros(batch_size)}, batch_size)
        value_shape = [2, *batch_size[1:]]
        td[index] = TensorDict(
            {"a": torch.ones(value_shape), "new": torch.ones(value_shape)},
            value_shape,
        )
        expected = torch.zeros(batch_size)
        expected[[0, 2]] = 1
        assert (td["a"] == expected).all()
        assert td["new"].shape == torch.Size(batch_size)
        assert (td["new"] == expected).all()

    @pytest.mark.parametrize("index", [None, True, torch.tensor(True), (None,)])
    @pytest.mark.parametrize("container", ["sub", "lazy"])
    def test_setitem_new_dim_index(self, index, container):
        # None and True add a dim of size 1, so the value is written to every
        # element, as td.unsqueeze(0)[:] = value would
        if container == "sub":
            td = TensorDict({"a": torch.zeros(2, 3, 4)}, [2, 3, 4])
            td = td._get_sub_tensordict(1)
        else:
            td = lazy_stack([TensorDict({"a": torch.zeros(4)}, [4]) for _ in range(3)])
        td[index] = TensorDict(
            {"a": torch.ones(1, 3, 4), "new": torch.ones(1, 3, 4)}, [1, 3, 4]
        )
        assert (td["a"] == 1).all()
        assert (td["new"] == 1).all()
        td[index] = {"a": torch.full((1, 3, 4), 2.0)}
        assert (td["a"] == 2).all()
        # the value is broadcast to the batch size with the new dim
        td[index] = TensorDict({"a": torch.full((3, 4), 3.0)}, [3, 4])
        assert (td["a"] == 3).all()

    @pytest.mark.parametrize(
        "index",
        [
            [np.True_, np.False_, np.True_],
            np.array([True, False, True]),
            [[0, 1], [1, 2]],
            (slice(None), [[0, 1], [1, 2]]),
            [[True, False, True, False], [False, True, False, True], [True] * 4],
            True,
            False,
            (slice(None), False),
            (True, [0, 2]),
            (torch.ones(3, 4, dtype=torch.bool), ...),
            (..., torch.ones(4, 5, dtype=torch.bool)),
            (slice(None), [2], None, torch.tensor([2])),
        ],
    )
    def test_index_types_follow_torch(self, index):
        # a bare nested list is a single index, as for (index,) in torch
        torch_index = index if isinstance(index, tuple) else (index,)
        tensor = torch.arange(60.0).view(3, 4, 5)
        expected = tensor[torch_index]
        td = TensorDict({"a": tensor, "n": {"b": tensor}}, [3, 4, 5])
        sub = td[index]
        assert sub.batch_size == expected.shape
        assert (sub["a"] == expected).all()
        assert (sub["n", "b"] == expected).all()

        # write a tensordict with an existing and a new key, and a scalar
        written = tensor.clone()
        written[torch_index] = -1
        new = torch.zeros(3, 4, 5)
        new[torch_index] = -1
        td = TensorDict({"a": tensor.clone()}, [3, 4, 5])
        value = -torch.ones(expected.shape)
        td[index] = TensorDict({"a": value, "new": value}, expected.shape)
        assert (td["a"] == written).all()
        assert (td["new"] == new).all()
        td = TensorDict({"a": tensor.clone()}, [3, 4, 5])
        td[index] = -1.0
        assert (td["a"] == written).all()

    def test_index_uint8_mask(self):
        mask = torch.tensor([1, 0, 2], dtype=torch.uint8)
        tensor = torch.arange(60.0).view(3, 4, 5)
        td = TensorDict({"a": tensor, "n": {"b": tensor}}, [3, 4, 5])
        with warnings.catch_warnings():
            # torch reads a uint8 tensor as a mask, and warns that it is deprecated
            warnings.filterwarnings("ignore", "indexing with dtype torch.uint8")
            expected = tensor[mask]
            sub = td[mask]
            value = -torch.ones(expected.shape)
            written = tensor.clone()
            written[mask] = value
            td_written = TensorDict({"a": tensor.clone()}, [3, 4, 5])
            td_written[mask] = TensorDict({"a": value}, expected.shape)
        assert sub.batch_size == expected.shape
        assert (sub["a"] == expected).all()
        assert (sub["n", "b"] == expected).all()
        assert (td_written["a"] == written).all()

    @pytest.mark.parametrize("mask_shape", [(3,), (3, 4)])
    def test_index_mask_reads_mask_once(self, mask_shape):
        # td[mask] reads the mask once, not once per leaf: on CUDA each read
        # is a device sync. A CPU mask is read once when the tensordict is on
        # the CPU.
        class MaskOps(TorchDispatchMode):
            def __init__(self, mask):
                super().__init__()
                self.mask = mask
                self.ops = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if any(arg is self.mask for arg in tree_leaves((args, kwargs))):
                    self.ops.append(func)
                return func(*args, **kwargs)

        tensor = torch.arange(60.0).view(3, 4, 5)
        td = TensorDict(
            {"a": tensor, "b": tensor[..., 0], "n": {"c": tensor}},
            [3, 4],
            names=["x", "y"],
            device="cpu",
        )
        mask = torch.zeros(mask_shape, dtype=torch.bool)
        mask[0] = True
        mask[-1] = True
        with MaskOps(mask) as mode:
            sub = td[mask]
        assert len(mode.ops) == 1, mode.ops
        expected = tensor[mask]
        assert sub.batch_size == expected.shape[: len(sub.batch_size)]
        assert (sub["a"] == expected).all()
        assert (sub["b"] == expected[..., 0]).all()
        assert (sub["n", "c"] == expected).all()
        assert sub.names == (["x", "y"] if len(mask_shape) == 1 else [None])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("mask_device", ["cuda", "cpu"])
    def test_index_mask_cuda_syncs_once(self, mask_device):
        # a CPU mask on a tensordict without a device is kept as it is: CPU
        # indices would make CUDA leaves sync once each
        x = torch.randn(10, device="cuda")
        mask = (x > 0).to(mask_device)
        td = TensorDict({"a": x, "b": x, "n": {"c": x, "d": {"e": x}}}, [10])
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                sub = td[mask]
            finally:
                torch.cuda.set_sync_debug_mode(0)
        syncs = [w for w in caught if "called a synchronizing" in str(w.message)]
        # nonzero syncs on a CUDA mask; CUDA leaves read a CPU mask without one
        assert len(syncs) == (1 if mask_device == "cuda" else 0)
        assert (sub["n", "d", "e"] == x[mask]).all()

    @pytest.mark.parametrize(
        "index",
        [
            (0, 0, 0),
            (slice(None),) * 3,
            5,
            (slice(None), -5),
            torch.ones(5, dtype=torch.bool),
            torch.ones(3, 5, dtype=torch.bool),
            (slice(None), torch.ones(2, dtype=torch.bool)),
            (-1, torch.ones(2, dtype=torch.bool)),
            (None, 5),
            ([0, 1], [0, 1, 2]),
        ],
    )
    def test_getitem_invalid_index_without_entries(self, index):
        # torch finds an invalid index when it indexes the entries, so a
        # tensordict checks the index against its batch size when it has none,
        # and raises the error that torch raises
        with pytest.raises(IndexError) as torch_error:
            torch.zeros(3, 4)[index]
        for td in (
            TensorDict(batch_size=[3, 4]),
            TensorDict({"a": torch.zeros(3, 4, 2)}, [3, 4]),
        ):
            with pytest.raises(IndexError) as error:
                td[index]
            assert str(error.value) == str(torch_error.value)
            with pytest.raises(IndexError):
                td[index] = TensorDict()

    @pytest.mark.parametrize(
        "index",
        [
            [0, 2],
            torch.tensor([0, 2]),
            (slice(None), [0, 1]),
            torch.tensor([True, False, True]),
            1,
        ],
    )
    def test_setitem_python_scalar_cast(self, index):
        # torch writes a Python scalar in the dtype of the entry, with every
        # kind of index
        td = TensorDict(
            {
                "c": torch.zeros(3, 2, dtype=torch.long),
                "n": {"b": torch.zeros(3, 2, dtype=torch.bool)},
            },
            [3, 2],
        )
        expected_c, expected_b = td["c"].clone(), td["n", "b"].clone()
        expected_c[index] = -3.5
        expected_b[index] = -3.5
        td[index] = -3.5
        assert (td["c"] == expected_c).all()
        assert (td["n", "b"] == expected_b).all()
        td.set_at_("c", 7.5, index)
        expected_c[index] = 7.5
        assert (td["c"] == expected_c).all()

    @pytest.mark.parametrize("write", ["setitem", "sub_setitem", "mask"])
    def test_setitem_tensorclass_value(self, write):
        # a tensorclass value is written key by key, as a tensordict value is
        @tensorclass
        class Data:
            x: torch.Tensor
            s: str

        value = Data(x=torch.ones(2, 3), s="new", batch_size=[2])
        td = TensorDict(x=torch.zeros(4, 3), s="old", batch_size=[4])
        if write == "setitem":
            td[1:3] = value
        elif write == "sub_setitem":
            td._get_sub_tensordict(slice(None))[1:3] = value
        else:
            td[torch.tensor([False, True, True, False])] = value
        assert (td["x"] == torch.tensor([0.0, 1.0, 1.0, 0.0])[:, None]).all()
        assert td.get("s").tolist() == ["old", "new", "new", "old"]

    def test_setitem_tensorclass_value_tensorclass_entries(self):
        # the fields of the value are matched to the keys, as for
        # td[idx] = value.to_tensordict(): the value is not written into
        # each tensorclass entry
        @tensorclass
        class Data:
            x: torch.Tensor

        td = TensorDict(a=Data(x=torch.zeros(4, 3), batch_size=[4]), batch_size=[4])
        td[1:3] = Data(x=torch.ones(2, 3), batch_size=[2])
        assert set(td.keys()) == {"a", "x"}
        assert (td["a"].x == 0).all()
        assert (td["x"] == torch.tensor([0.0, 1.0, 1.0, 0.0])[:, None]).all()

    @pytest.mark.parametrize(
        "case",
        [
            "dense",
            "new_key",
            "non_tensor",
            "nested_lazy",
            "into_lazy",
            "nested_tc",
            "subclass",
            "batch_size",
            "locked_new_key",
            "other_device_leaf",
        ],
    )
    @pytest.mark.parametrize("index", ["tensor", "slice"])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            "meta",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_setitem_value_without_device(self, case, index, device):
        # td[idx] = value, for a value without a device, moves the entries of
        # value that are on another device as it writes them, rather than a
        # moved copy of value: the result is the one of value.to(td.device)
        @tensorclass
        class Data:
            x: torch.Tensor
            s: str

        def make_store():
            store = TensorDict(
                a=torch.zeros(6, 3),
                n=TensorDict(b=torch.zeros(6, dtype=torch.long), batch_size=[6]),
                s=NonTensorData("old", batch_size=[6]),
                lz=lazy_stack([TensorDict(c=torch.zeros(2)) for _ in range(6)]),
                tc=Data(x=torch.zeros(6, 2), s="old", batch_size=[6]),
                batch_size=[6],
                device=device,
            )
            return store.lock_() if case == "locked_new_key" else store

        value = TensorDict(
            a=torch.arange(12.0).view(4, 3),
            n=TensorDict(b=torch.arange(4), batch_size=[4]),
            batch_size=[4],
        )
        if case in ("new_key", "locked_new_key"):
            value["new"] = torch.ones(4, 2)
            value["new_n"] = TensorDict(d=torch.ones(4), batch_size=[4])
            value["new_s"] = NonTensorData("new", batch_size=[4])
        elif case == "non_tensor":
            value["s"] = NonTensorData("new", batch_size=[4])
        elif case == "nested_lazy":
            value["lz"] = lazy_stack(
                [TensorDict(c=torch.full((2,), float(i))) for i in range(4)]
            )
        elif case == "into_lazy":
            # a dense value written into the lazy stack of store
            value["lz"] = TensorDict(c=torch.ones(4, 2), batch_size=[4])
        elif case == "nested_tc":
            value["tc"] = Data(x=torch.ones(4, 2), s="new", batch_size=[4])
        elif case == "subclass":
            value["a"] = nn.Parameter(torch.ones(4, 3))
            value["n", "b"] = MemoryMappedTensor.from_tensor(torch.arange(4))
        elif case == "batch_size":
            # the batch size of value is set to the one of the index
            value.batch_size = []
        elif case == "other_device_leaf":
            if device != "cuda" and torch.cuda.is_available():
                other = "cuda"
            else:
                # a leaf on the meta device cannot be copied to another one
                other = "meta" if device != "meta" else "cpu"
            value["n", "b"] = value["n", "b"].to(other)
        idx = torch.arange(1, 5) if index == "tensor" else slice(1, 5)

        def write(store, value):
            try:
                with torch.no_grad():
                    store[idx] = value
            except Exception as err:
                return type(err), str(err)
            return None

        def describe(store):
            out = [(store.device, store.is_locked)]
            for key in store.keys(True, is_leaf=is_leaf_nontensor):
                entry = store.get(key)
                if isinstance(entry, torch.Tensor):
                    out.append(
                        (key, type(entry), entry.dtype, entry.device, entry.shape)
                    )
                    if entry.device.type != "meta":
                        out.append(entry.cpu().tolist())
                elif is_non_tensor(entry):
                    out.append((key, type(entry), entry.device, entry.tolist()))
                else:
                    out.append((key, type(entry), entry.device))
            return out

        store, expected = make_store(), make_store()
        try:
            moved = value.to(expected.device)
        except Exception as err:
            expected_err = type(err), str(err)
        else:
            expected_err = write(expected, moved)
        assert write(store, value) == expected_err
        # a leaf that cannot be moved raises before anything is written
        assert describe(store) == describe(expected)
        if expected_err is not None:
            assert case in ("locked_new_key", "other_device_leaf")
            return

        if case in ("dense", "other_device_leaf") and device == "cpu":
            # the leaves are on the device of store: no copy of value is made.
            # A leaf on another device is moved with a copy of value, which
            # syncs once for all the leaves
            with mock.patch.object(
                TensorDict, "_to", autospec=True, side_effect=TensorDict._to
            ) as to:
                store[idx] = value
            assert to.called == (case == "other_device_leaf")

    @pytest.mark.parametrize("write", ["update_at_", "sub_update_at_", "sub_set_at_"])
    def test_update_at_tensorclass_value(self, write):
        @tensorclass
        class Data:
            x: torch.Tensor
            s: str

        value = Data(x=torch.ones(2, 3), s="new", batch_size=[2])
        td = TensorDict(
            tc=Data(x=torch.zeros(4, 3), s="old", batch_size=[4]), batch_size=[4]
        )
        if write == "update_at_":
            td.update_at_({"tc": value}, slice(1, 3))
        elif write == "sub_update_at_":
            td._get_sub_tensordict(slice(None)).update_at_({"tc": value}, slice(1, 3))
        else:
            td._get_sub_tensordict(slice(None)).set_at_("tc", value, slice(1, 3))
        written = td["tc"]._tensordict
        assert (written["x"] == torch.tensor([0.0, 1.0, 1.0, 0.0])[:, None]).all()
        assert written.get("s").tolist() == ["old", "new", "new", "old"]

    def test_getitem_scalar_bool_0d(self):
        td = TensorDict({"a": torch.tensor(1.0)}, [])
        assert td[True].batch_size == torch.Size([1])
        assert td[False].batch_size == torch.Size([0])
        # None and scalar bools use no dim, as in torch
        for index in ((True, True), (None, True), (torch.tensor(True), None)):
            assert td[index].batch_size == torch.zeros(())[index].shape

    @pytest.mark.parametrize(
        "index", [np.True_, np.False_, (slice(None), np.True_), (np.False_, ...)]
    )
    def test_index_numpy_bool_scalar(self, index):
        # torch rejects a NumPy bool scalar from NumPy 2.3 on (and reads it as
        # an int before); tensordict rejects it with any NumPy
        td = TensorDict({"a": torch.zeros(3, 4)}, [3, 4], names=["x", "y"])
        with pytest.raises(IndexError, match="NumPy bool"):
            td[index]
        with pytest.raises(IndexError, match="NumPy bool"):
            td[index] = 1.0
        with pytest.raises(IndexError, match="NumPy bool"):
            td[index] = TensorDict({"a": torch.ones(4)}, [4])
        with pytest.raises(IndexError, match="NumPy bool"):
            td.get_at("a", index)
        with pytest.raises(IndexError, match="NumPy bool"):
            td.set_at_("a", 1.0, index)
        with pytest.raises(IndexError, match="NumPy bool"):
            td.copy_at_(TensorDict({"a": torch.ones(4)}, [4]), index)
        assert (td["a"] == 0).all()
        # a list of NumPy bools is a mask, as in torch
        assert td[[np.True_, np.False_, np.True_]].batch_size == torch.Size([2, 4])

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

    @staticmethod
    def _getitem_str_key_td(cls=TensorDict):
        @tensorclass
        class TC:
            x: torch.Tensor

        td = cls(
            a=torch.zeros(3),
            sub=TensorDict(b=torch.zeros(3), batch_size=[3]),
            s=NonTensorData("text", batch_size=[3]),
            ns=NonTensorStack(*[NonTensorData(i) for i in range(3)]),
            lazy_ns=lazy_stack(
                [NonTensorData("x"), NonTensorData("y"), NonTensorData("z")]
            ),
            tc=TC(x=torch.zeros(3), batch_size=[3]),
            lazy=lazy_stack([TensorDict(c=torch.zeros(())) for _ in range(3)]),
            p=nn.Parameter(torch.zeros(3)),
            u=tensordict.UnbatchedTensor(torch.zeros(2)),
            batch_size=[3],
        )
        td["a.b"] = torch.ones(3)
        return td

    def test_getitem_str_key(self):
        # td["k"] on an exact TensorDict reads the entry from its dict, without
        # the generic path; subclasses and nested keys keep the generic path
        class SubTD(TensorDict):
            pass

        class Key(str):
            pass

        generic = TensorDictBase._get_tuple_maybe_non_tensor
        read = []

        def spy(self, key, default, **kwargs):
            read.append(key)
            return generic(self, key, default, **kwargs)

        td = self._getitem_str_key_td()
        subtd = self._getitem_str_key_td(SubTD)
        with mock.patch.object(TensorDictBase, "_get_tuple_maybe_non_tensor", spy):
            for key in ("a", "sub", "a.b", "p", "u", Key("a")):
                assert td[key] is td._tensordict[key]
            assert type(td["a"]) is torch.Tensor
            assert type(td["sub"]) is TensorDict
            assert td["s"] == "text"
            assert td["ns"] == [0, 1, 2]
            with pytest.raises(KeyError, match='key "missing" not found in TensorDict'):
                td["missing"]
            assert read == []
            assert td["sub", "b"] is td["sub"]["b"]
            assert read == [("sub", "b")]
            read.clear()
            assert subtd["a"] is subtd._tensordict["a"]
            assert read == [("a",)]

    @pytest.mark.parametrize(
        "key",
        ["a", "sub", "a.b", "s", "ns", "lazy_ns", "tc", "lazy", "p", "u", "missing"],
    )
    def test_getitem_str_key_as_get_tuple(self, key):
        # __getitem__ repeats the tail of _get_tuple_maybe_non_tensor for a str
        # key of a TensorDict: both must return the same
        td = self._getitem_str_key_td()
        if key == "missing":
            with pytest.raises(KeyError) as generic:
                td._get_tuple_maybe_non_tensor((key,), tensordict_base.NO_DEFAULT)
            with pytest.raises(KeyError) as fast:
                td[key]
            assert str(fast.value) == str(generic.value)
            assert type(fast.value.__context__) is type(generic.value.__context__)
            return
        generic = td._get_tuple_maybe_non_tensor((key,), tensordict_base.NO_DEFAULT)
        fast = td[key]
        assert type(fast) is type(generic)
        if isinstance(generic, (torch.Tensor, TensorDictBase)) or is_tensorclass(
            generic
        ):
            assert fast is generic
        else:
            assert fast == generic

    def test_get_nested_reads_dicts(self, monkeypatch):
        # A nested get through TensorDicts reads their dicts, without a
        # _get_str call per level, and keeps the results and errors of the
        # generic walk.
        leaf = torch.zeros(3)
        td = TensorDict({"a": {"b": {"c": leaf}}, "t": leaf}, [3])
        td["l"] = lazy_stack([TensorDict({"q": torch.ones(())})] * 3)

        @tensorclass
        class TC:
            x: torch.Tensor

        td["tc"] = TC(x=leaf, batch_size=[3])
        calls = []
        get_str = TensorDict._get_str

        def counting_get_str(self, key, default, **kwargs):
            calls.append(key)
            return get_str(self, key, default, **kwargs)

        monkeypatch.setattr(TensorDict, "_get_str", counting_get_str)
        assert td.get(("a", "b", "c")) is leaf
        assert td["a", "b", "c"] is leaf
        assert td.get(("a", "b", "z"), "default") == "default"
        assert td.get(("a", "z", "c"), None) is None
        assert td.get("a") is td._tensordict["a"]
        # a tensorclass on the path reads the rest itself
        assert td.get(("tc", "x")) is leaf
        assert td["tc", "x"] is leaf
        assert not calls

        calls.clear()

        # Subclasses keep the generic walk: they make the same _get_str and
        # _default_get calls, with the same results, as with the generic walk
        # for every class (main's behaviour).
        def logged_get_str(self, key, default, **kwargs):
            calls.append(("_get_str", type(self).__name__, key))
            return get_str(self, key, default, **kwargs)

        def logged_default_get(self, key, default=NO_DEFAULT):
            calls.append(("_default_get", type(self).__name__, key))
            if key == "auto":
                return type(self)()
            return TensorDictBase._default_get(self, key, default)

        class Plain(TensorDict):
            pass

        class OwnGetStr(TensorDict):
            _get_str = logged_get_str

        class OwnDefaultGet(TensorDict):
            _default_get = logged_default_get

        class OwnGetTuple(TensorDict):
            _get_str = logged_get_str

            def _get_tuple(self, key, default, **kwargs):
                calls.append(("_get_tuple", type(self).__name__, key))
                return super()._get_tuple(key, default, **kwargs)

        class Late(TensorDict):
            pass

        Late._get_str = logged_get_str
        Late._default_get = logged_default_get

        def read(cls):
            calls.clear()
            sub = cls({"a": cls({"b": leaf}, [3])}, [3])
            results = [
                sub.get(("a", "b")),
                sub["a", "b"],
                sub.get(("a", "z"), None),
                sub.get(("auto", "y"), None),
            ]
            with pytest.raises(KeyError) as err:
                sub["auto", "y"]
            return results, str(err.value), list(calls)

        for cls in (Plain, OwnGetStr, OwnDefaultGet, OwnGetTuple, Late):
            results, error, cls_calls = read(cls)
            with monkeypatch.context() as m:
                m.setattr(TensorDict, "_get_tuple", TensorDictBase._get_tuple)
                expected, expected_error, expected_calls = read(cls)
            assert cls_calls == expected_calls, cls
            assert error == expected_error, cls
            assert len(results) == len(expected)
            for result, value in zip(results, expected):
                assert result is value or (result == value).all(), cls
        calls.clear()
        # a default that is the value on the path is returned, as before
        sub = td["a", "b"]
        assert td.get(("a", "b", "c"), sub) is sub
        # a lazy stack on the path reads the rest itself
        assert (td.get(("l", "q")) == 1).all()
        assert td.get(("l", "z"), None) is None
        with pytest.raises(
            KeyError, match=r'key "z" not found in TensorDict with keys \[\W*c\W*\]'
        ):
            td["a", "b", "z"]
        # with a default, a key that runs through a leaf is missing (#2678)
        assert td.get(("a", "b", "c", "z"), None) is None
        with pytest.raises(
            ValueError,
            match="'c' is a Tensor, not a tensordict, so it has no entry 'z'",
        ):
            td.get(("a", "b", "c", "z"))
        with pytest.raises(
            ValueError,
            match=r"'t' is a Tensor, not a tensordict, so it has no entry \('y', 'z'\)",
        ):
            td.get(("t", "y", "z"))

    @legacy_lazy_mode()
    def test_inferred_view_size(self):
        td = TensorDict({"a": torch.randn(3, 4)}, [3, 4])
        assert td.view(-1).view(-1, 4) is td

        assert td.view(-1, 4) is td
        assert td.view(3, -1) is td
        assert td.view(3, 4) is td
        assert td.view(-1, 12).shape == torch.Size([1, 12])

    @legacy_lazy_mode()
    def test_inferred_view_size_zero_size_dim(self):
        td = TensorDict({"a": torch.zeros(0, 3, 4)}, [0, 3])
        assert td.view(-1).batch_size == torch.Size([0])
        assert td.view(-1)["a"].shape == torch.Size([0, 4])
        assert td.view(-1).view(-1, 3) is td
        assert td.view(0, 3) is td
        assert td.view(3, 0).view(-1).batch_size == torch.Size([0])

    def test_is_empty(self):
        assert TensorDict({"a": {"b": {}}}, []).is_empty()
        assert not TensorDict(
            {"a": {"b": {}, "c": NonTensorData("a string!", batch_size=[])}}, []
        ).is_empty()
        assert not TensorDict({"a": {"b": {}, "c": 1}}, []).is_empty()

    def test_is_empty_non_tensor_only(self):
        # a non-tensor counts as a leaf on a dense tensordict; the lazy stack
        # and the sub-tensordict, which use TensorDictBase.is_empty, do not
        # count it, and the lazy stack warns that it counts it from 0.17
        td = TensorDict(x=NonTensorData("a", batch_size=[2]), batch_size=[2])
        assert not td.is_empty()
        with pytest.warns(FutureWarning, match="returning True only if every"):
            assert lazy_stack([td[0], td[1]]).is_empty()
        assert td._get_sub_tensordict(0).is_empty()

    @pytest.mark.parametrize(
        "make",
        [
            lambda s: s,
            lambda s: TensorDict(x=s, batch_size=[2]),
            lambda s: TensorDict(x=s, y=s.clone(), batch_size=[2]),
            lambda s: lazy_stack([s, s.clone()]),
            lambda s: lazy_stack([TensorDict(sub=s[0]), TensorDict(sub=s[1])]),
        ],
        ids=["stack", "nested", "siblings", "stack_of_stacks", "nested_in_members"],
    )
    def test_is_empty_lazy_members_hold_leaves(self, make):
        # gh-2807: a lazy stack lists only the leaves that its members share.
        # is_empty() keeps returning True when they share none, and warns that
        # it returns False from 0.17.
        s = lazy_stack([TensorDict(a=torch.ones(())), TensorDict(b=torch.ones(()))])
        td = make(s)
        with pytest.warns(FutureWarning, match="returning True only if every") as w:
            assert td.is_empty()
        assert len(w) == 1
        assert w[0].filename == __file__

    def test_is_empty_lazy_wrappers(self):
        @tensorclass
        class TC:
            x: TensorDict = None

        class Typed(TypedTensorDict):
            pass

        s = lazy_stack([TensorDict(a=torch.ones(())), TensorDict(b=torch.ones(()))])
        tc = lazy_stack([TC(x=s[0]), TC(x=s[1])])
        with legacy_lazy_mode():
            view = TensorDict(x=s, batch_size=[2]).unsqueeze(0)
        assert isinstance(view, _CustomOpTensorDict)
        for td in (tc, Typed.from_tensordict(s), view):
            with pytest.warns(FutureWarning, match="returning True only if every"):
                assert td.is_empty()

    @pytest.mark.parametrize(
        "make,expected",
        [
            (lambda: lazy_stack([TensorDict(), TensorDict()]), True),
            (lambda: lazy_stack([TensorDict(x=TensorDict()), TensorDict()]), True),
            (lambda: lazy_stack([TensorDict(a=torch.ones(()))] * 2), False),
            (lambda: TensorDict(x=TensorDict()), True),
            (lambda: TensorDict(a=torch.ones(())), False),
            (lambda: NonTensorStack("a", "b"), False),
            # the nested stack comes first, but the result is False today and
            # from 0.17
            (
                lambda: TensorDict(
                    x=lazy_stack([TensorDict(a=torch.ones(())), TensorDict()]),
                    y=torch.ones(2),
                    batch_size=[2],
                ),
                False,
            ),
        ],
        ids=[
            "empty",
            "empty_members",
            "shared",
            "dense_empty",
            "dense",
            "nts",
            "nested_then_leaf",
        ],
    )
    def test_is_empty_no_lazy_warning(self, make, expected):
        td = make()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert td.is_empty() is expected

    def test_items_values_sort_flat(self):
        td = TensorDict(b=torch.ones(()), a=torch.zeros(()))
        assert [key for key, _ in td.items(sort=True)] == ["a", "b"]
        assert [value.item() for value in td.values(sort=True)] == [0.0, 1.0]
        assert list(TensorDict().items(sort=True)) == []
        assert TensorDict().values(sort=True) == ()

    @pytest.mark.parametrize(
        "make",
        [
            lambda td: td,
            lambda td: lazy_stack(list(td.unbind(0))),
            lambda td: td._get_sub_tensordict(0),
        ],
        ids=["dense", "lazy", "sub"],
    )
    def test_del_empty_key_and_through_tensor(self, make):
        td = make(
            TensorDict(
                a=torch.zeros(2, 3),
                n=TensorDict(x=torch.ones(2, 3), batch_size=[2, 3]),
                batch_size=[2, 3],
            )
        )
        with pytest.raises(KeyError, match="Only NestedKeys are supported"):
            td.del_(())
        with pytest.raises(ValueError, match="'a' is a Tensor, not a tensordict"):
            td.del_(("a", "x"))
        with pytest.raises(ValueError, match="'a' is a Tensor, not a tensordict"):
            td.del_(("a", "x", "y"))
        with pytest.raises(KeyError):
            td.del_(("zz", "y", "z"))
        with pytest.raises(KeyError):
            td.del_("zz")
        with pytest.raises(KeyError):
            td.del_(("n", "zz"))
        assert td.del_(("n", "x")) is td
        assert list(td.keys(True)) == ["a", "n"]

    @pytest.mark.parametrize(
        "make",
        [
            lambda td: td,
            lambda td: lazy_stack(list(td.unbind(0))),
            lambda td: td._get_sub_tensordict(0),
        ],
        ids=["dense", "lazy", "sub"],
    )
    def test_rename_key_empty_string_key(self, make):
        # "" is a valid key: only the empty tuple is not
        for old, new in (("a", ""), ("a", ("",)), ("", "b")):
            td = make(TensorDict({old: torch.zeros(2, 3)}, batch_size=[2, 3]))
            assert td.rename_key_(old, new) is td
            new_key = new if isinstance(new, str) else new[0]
            assert list(td.keys()) == [new_key]
            assert (td.get(new_key) == 0).all()
        td = make(TensorDict({"": torch.zeros(2, 3)}, batch_size=[2, 3]))
        assert td.del_("") is td
        assert list(td.keys()) == []

    @pytest.mark.parametrize(
        "make",
        [
            lambda td: td,
            lambda td: lazy_stack(list(td.unbind(0))),
            lambda td: td._get_sub_tensordict(0),
        ],
        ids=["dense", "lazy", "sub"],
    )
    def test_rename_key_empty_new_key(self, make):
        td = make(TensorDict(a=torch.zeros(2, 3), batch_size=[2, 3]))
        with pytest.raises(KeyError, match="Only NestedKeys are supported"):
            td.rename_key_("a", ())
        assert list(td.keys()) == ["a"]
        with pytest.raises(KeyError, match="Only NestedKeys are supported"):
            td.rename_key_((), "b")

    @pytest.mark.parametrize("device", [None, "cpu", "meta"])
    def test_is_meta(self, device):
        td = TensorDict({"a": torch.zeros(3), ("b", "c"): torch.zeros(3)}, [3])
        if device is not None:
            td = td.to(device)
        assert td.is_meta is (device == "meta")
        assert td["b"].is_meta is (device == "meta")
        assert lazy_stack([td, td]).is_meta is (device == "meta")

    def test_is_in(self):
        td = TensorDict.fromkeys({"a", "b", "c", ("d", "e")}, 0)
        assert "a" in td
        assert "d" in td
        assert ("d", "e") in td
        assert "f" not in td
        with pytest.raises(RuntimeError, match="NestedKey"):
            0 in td  # noqa: B015

    def test_is_in_nested(self):
        # A TensorDict answers `key in td` from its dicts (#2892), as its keys
        # view does
        class SubTD(TensorDict):
            pass

        @tensorclass
        class TC:
            a: torch.Tensor

        member = TensorDict(q=torch.zeros(()), r=TensorDict(z=torch.zeros(())))
        td = TensorDict(
            a=TensorDict(b=TensorDict(c=torch.zeros(2)), t=torch.zeros(2)),
            x=torch.zeros(2),
            ls=lazy_stack([member, member.clone()]),
            tc=TC(a=torch.zeros(2), batch_size=[2]),
            sub=SubTD(b=TensorDict(c=torch.zeros(2)), batch_size=[2]),
            batch_size=[2],
        )
        td.set_non_tensor("nt", "s")
        present = ["x", ("x",), "a", ("a", "b"), ("a", "b", "c"), ("a", ("b", "c"))]
        present += [("a", "t"), "nt", ("nt",), ("ls", "q"), ("ls", "r", "z")]
        present += [("tc", "a"), ("sub", "b"), ("sub", "b", "c")]
        missing = ["y", ("y",), ("y", "b"), ("a", "y"), ("a", "y", "c")]
        # a path through a tensor or a non-tensor is not a key
        missing += [("x", "y"), ("x", "y", "z"), ("a", "t", "u"), ("nt", "data")]
        missing += [("a", "b", "c", "d"), ("ls", "w"), ("tc", "y"), ("sub", "y")]
        keys = td.keys(True, is_leaf=is_leaf_nontensor)
        for key in present:
            assert key in td, key
            assert key in keys, key
        for key in missing:
            assert key not in td, key
            assert key not in keys, key
        with pytest.raises(ValueError, match="'t' is a Tensor, not a tensordict"):
            ("a", "t", "u", "v") in td  # noqa: B015
        for key in [(), ((),), ("a", 1), 0, None, td]:
            with pytest.raises(RuntimeError, match="NestedKey"):
                key in td  # noqa: B015

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

        assert ("a", "b", "c") not in tensordict.keys()
        assert ("a", "b") not in tensordict.keys(leaves_only=True)

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

    @pytest.mark.parametrize("stack", [False, True])
    @pytest.mark.parametrize("sort", [False, True])
    @pytest.mark.parametrize("include_nested", [False, True])
    @pytest.mark.parametrize("leaves_only", [False, True])
    @pytest.mark.parametrize("is_leaf", ["default", "nontensor", "custom"])
    def test_keys_view_iteration(
        self, stack, sort, include_nested, leaves_only, is_leaf
    ):
        @tensorclass
        class MyClass:
            x: torch.Tensor

        td = TensorDict(
            a=torch.zeros(3, 1),
            nested=TensorDict(
                b=torch.zeros(3),
                c=TensorDict(d=torch.zeros(3, 2), batch_size=[3]),
                batch_size=[3],
            ),
            tc=MyClass(x=torch.zeros(3, 4), batch_size=[3]),
            nt=NonTensorData("a string", batch_size=[3]),
            batch_size=[3],
        )
        if stack:
            td = lazy_stack([td[0], td[1]])
        leaves = {"a", ("nested", "b"), ("nested", "c", "d"), ("tc", "x")}
        if not leaves_only:
            expected = leaves | {"nested", ("nested", "c"), "tc", "nt"}
        elif is_leaf == "default":
            expected = leaves
        elif is_leaf == "nontensor":
            expected = leaves | {"nt"}
        else:
            # the custom is_leaf rejects tensors
            expected = {"tc"}
        if not include_nested:
            expected = {key for key in expected if isinstance(key, str)}
        is_leaf = {
            "default": None,
            "nontensor": is_leaf_nontensor,
            "custom": lambda cls: cls is MyClass,
        }[is_leaf]
        keys = td.keys(include_nested, leaves_only, is_leaf=is_leaf, sort=sort)
        # list() and tuple() ask the view for its length first: they must
        # give the same keys as a plain loop.
        listed = list(keys)
        assert listed == [key for key in keys]  # noqa: C416
        assert tuple(listed) == tuple(keys)
        assert len(listed) == len(expected)
        assert set(listed) == expected
        if sort:
            assert listed == sorted(
                expected,
                key=lambda key: ".".join(key) if isinstance(key, tuple) else key,
            )
        # len() of nested and leaves_only lazy-stack views counts only the
        # top-level keys: the lazy-keys-len branch fixes it.
        if not stack:
            assert len(keys) == len(expected)

    @pytest.mark.parametrize("leaves_only", [False, True])
    @pytest.mark.parametrize("is_leaf_name", ["default", "nontensor", "all", "tensor"])
    def test_keys_nested_tensordict_entries(
        self, leaves_only, is_leaf_name, monkeypatch
    ):
        # The nested keys view of a TensorDict does not check its TensorDict
        # entries with _is_tensor_collection (#2618). Its generator, len() and
        # sort=True list the same keys, in the same order, and call is_leaf
        # with the same classes.
        @tensorclass
        class MyClass:
            x: torch.Tensor

        td = TensorDict(
            a=torch.zeros(3),
            nested=TensorDict(
                b=torch.zeros(3),
                c=TensorDict(d=torch.zeros(3, 2), batch_size=[3]),
                empty=TensorDict(batch_size=[3]),
                batch_size=[3],
            ),
            tc=MyClass(x=torch.zeros(3, 4), batch_size=[3]),
            nt=NonTensorData("a string", batch_size=[3]),
            u=tensordict.UnbatchedTensor(torch.zeros(2)),
            stack=lazy_stack(
                [TensorDict(f=torch.zeros(()), g=TensorDict(h=torch.zeros(())))] * 3
            ),
            z=torch.zeros(3),
            batch_size=[3],
        )
        is_leaf = {
            "default": None,
            "nontensor": is_leaf_nontensor,
            "all": lambda cls: True,
            "tensor": lambda cls: cls is torch.Tensor,
        }[is_leaf_name]
        checked = []
        is_tensor_collection = tensordict._td._is_tensor_collection

        def recorded_is_tensor_collection(cls):
            checked.append(cls)
            return is_tensor_collection(cls)

        monkeypatch.setattr(
            tensordict._td, "_is_tensor_collection", recorded_is_tensor_collection
        )

        def listed(list_keys):
            calls = []

            def recorded_is_leaf(cls):
                calls.append(cls)
                return is_leaf(cls)

            return list_keys(recorded_is_leaf if is_leaf else None), calls

        # iter(): list() of the view would also call is_leaf for its len()
        keys, calls = listed(
            lambda is_leaf: list(iter(td.keys(True, leaves_only, is_leaf=is_leaf)))
        )
        # The order on main: a sub-tensordict comes after its entries
        common = ["a", ("nested", "b"), ("nested", "c", "d")]
        stack_leaves = [("stack", "f"), ("stack", "g", "h")]
        if not leaves_only or is_leaf_name == "all":
            expected = [
                *common,
                ("nested", "c"),
                ("nested", "empty"),
                "nested",
                ("tc", "x"),
                "tc",
                "nt",
                "u",
                *stack_leaves,
                ("stack", "g"),
                "stack",
                "z",
            ]
        elif is_leaf_name == "default":
            expected = [*common, ("tc", "x"), "u", *stack_leaves, "z"]
        elif is_leaf_name == "nontensor":
            expected = [*common, ("tc", "x"), "nt", "u", *stack_leaves, "z"]
        else:
            expected = [*common, ("tc", "x"), *stack_leaves, "z"]
        assert keys == expected
        assert listed(
            lambda is_leaf: len(td.keys(True, leaves_only, is_leaf=is_leaf))
        ) == (len(keys), calls)

        def keyfunc(key):
            return ".".join(key) if isinstance(key, tuple) else key

        assert listed(
            lambda is_leaf: list(
                iter(td.keys(True, leaves_only, is_leaf=is_leaf, sort=True))
            )
        ) == (sorted(keys, key=keyfunc), calls)
        assert TensorDict not in checked

    def test_keys_nested_lazy(self, monkeypatch):
        # Iterating the nested keys of a TensorDict reads each level when it
        # gets to it (#2618).
        td = TensorDict(
            a=torch.zeros(3),
            nest=TensorDict(c=torch.zeros(3), batch_size=[3]),
            other=TensorDict(d=torch.zeros(3), batch_size=[3]),
            batch_size=[3],
        )
        keys = []
        for key in td.keys(True, True):
            keys.append(key)
            if key == "a":
                td["nest"] = TensorDict(x=torch.zeros(3), batch_size=[3])
                del td["other", "d"]
                td["other", "e"] = torch.zeros(3)
        assert keys == ["a", ("nest", "x"), ("other", "e")]

        # The first key reads the levels it goes through, not the others.
        wide = TensorDict(
            {f"g{i}": TensorDict(x=torch.zeros(3), batch_size=[3]) for i in range(50)},
            batch_size=[3],
        )
        read = []
        items = _TensorDictKeysView._items

        def recorded_items(self, tensordict=None):
            read.append(tensordict)
            return items(self, tensordict)

        monkeypatch.setattr(_TensorDictKeysView, "_items", recorded_items)
        assert next(iter(wide.keys(True, True))) == ("g0", "x")
        assert len(read) == 2
        assert read[0] is wide
        assert read[1] is wide.get("g0")

    @pytest.mark.parametrize("stack", [False, True])
    def test_items_nested_deep(self, stack):
        # items(True) builds each key from the keys of the level below.
        td = TensorDict(
            {
                ".".join([str(j) for j in range(i)] + ["t"]): torch.zeros(3)
                for i in range(1, 8)
            },
            batch_size=[3],
        ).unflatten_keys(".")
        if stack:
            td = lazy_stack([td[0], td[1]])
        items = list(td.items(True))
        keys = [key for key, _ in items]
        # the keys view walks the levels in another order
        assert len(set(keys)) == len(keys)
        assert set(keys) == set(td.keys(True))
        assert ("0", "1", "2", "3", "4", "5", "t") in keys
        for key, value in items:
            if type(key) is not str:
                assert type(key) is tuple
                assert all(type(part) is str for part in key)
            if not stack:
                assert td.get(key) is value

    def test_items_nested_unravels_child_keys(self):
        # items(True) gives _unravel_key_to_tuple((key, child_key)) for every
        # key that the items() of a child yields, whatever its shape.
        child_keys = ["a", ("b", "c"), ("d", ("e",)), ("f", 1), (), 7, ("g", "h")]

        class ChildTD(TensorDict):
            def items(
                self,
                include_nested=False,
                leaves_only=False,
                is_leaf=None,
                *,
                sort=False,
            ):
                if not include_nested:
                    yield from super().items(
                        include_nested, leaves_only, is_leaf, sort=sort
                    )
                    return
                for i, key in enumerate(child_keys):
                    yield key, torch.full((3,), i)

        td = TensorDict(x=torch.zeros(3), batch_size=[3])
        td._tensordict["child"] = ChildTD(batch_size=[3])
        keys = [key for key, _ in td.items(True)]
        assert keys == ["x", "child"] + [
            _unravel_key_to_tuple(("child", key)) for key in child_keys
        ]
        assert keys[2:] == [
            ("child", "a"),
            ("child", "b", "c"),
            ("child", "d", "e"),
            (),
            (),
            (),
            ("child", "g", "h"),
        ]

    @staticmethod
    def _is_leaf_nested_td():
        class MyClass(TensorClass):
            x: torch.Tensor

        return TensorDict(
            a=torch.zeros(3),
            nested=TensorDict(b=torch.zeros(3), batch_size=[3]),
            tc=MyClass(x=torch.zeros(3), batch_size=[3]),
            batch_size=[3],
        )

    @pytest.mark.parametrize("stack", [False, True])
    def test_items_is_leaf_filters(self, stack):
        # is_leaf filters the leaves; it does not stop the recursion, so
        # items, keys and values list the same entries (#2131).
        td = self._is_leaf_nested_td()
        if stack:
            td = lazy_stack([td, td.clone()])
        for is_leaf in (
            lambda cls: True,
            lambda cls: not is_tensor_collection(cls) or is_tensorclass(cls),
        ):
            keys = list(td.keys(True, True, is_leaf=is_leaf))
            items = list(td.items(True, True, is_leaf=is_leaf))
            assert sorted(map(str, keys)) == sorted(str(key) for key, _ in items)
            assert ("tc", "x") in keys
            assert len(list(td.values(True, True, is_leaf=is_leaf))) == len(keys)

    @pytest.mark.parametrize("stack", [False, True])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_flatten_keys_is_leaf_keeps_value(self, stack, inplace):
        # flatten_keys keeps a value that is_leaf accepts whole, in place and
        # out of place, on dense and lazy tensordicts (#2131).
        td = self._is_leaf_nested_td()
        if stack:
            td = lazy_stack([td, td.clone()])
        out = td.flatten_keys(
            is_leaf=lambda cls: not is_tensor_collection(cls) or is_tensorclass(cls),
            inplace=inplace,
        )
        assert set(out.keys()) == {"a", "nested.b", "tc"}
        assert is_tensorclass(out.get("tc"))
        td = self._is_leaf_nested_td()
        if stack:
            td = lazy_stack([td, td.clone()])
        out = td.flatten_keys(is_leaf=lambda cls: True, inplace=inplace)
        assert set(out.keys()) == {"a", "nested", "tc"}

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

    def test_load_underscore_deprecated(self, tmpdir):
        td = TensorDict({"a": torch.arange(3), "b": {"c": torch.ones(3)}}, [3])
        td.memmap(tmpdir)
        dest = td.clone().zero_()
        with pytest.warns(
            DeprecationWarning,
            match=r"^TensorDictBase\.load_\(\) is deprecated and will be removed in "
            r"TensorDict 0\.17\. Use load_memmap_\(\) instead\.$",
        ) as record:
            out = dest.load_(tmpdir)
        assert record[0].filename == __file__
        assert out is dest
        assert (dest == td).all()

    def test_pin_memory_underscore_deprecated(self, monkeypatch):
        td = TensorDict({"a": torch.arange(3)}, [3])
        calls = []

        def pin_memory(self, num_threads=None, inplace=False):
            calls.append((num_threads, inplace))
            return self

        # pinning needs an accelerator, so check what pin_memory_ forwards
        monkeypatch.setattr(TensorDict, "pin_memory", pin_memory)
        with pytest.warns(
            DeprecationWarning,
            match=r"^TensorDictBase\.pin_memory_\(\) is deprecated and will be "
            r"removed in TensorDict 0\.17\. Use pin_memory\(inplace=True\) instead\.$",
        ) as record:
            assert td.pin_memory_() is td
        assert record[0].filename == __file__
        assert calls == [(0, True)]
        with pytest.warns(DeprecationWarning, match="pin_memory_"):
            td.pin_memory_(num_threads=2)
        assert calls[-1] == (2, True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("num_threads", [None, 0, 2])
    def test_pin_memory_inplace(self, num_threads, monkeypatch):
        # Each leaf is replaced by its pinned copy, and the tensordict, its
        # nested tensordicts, tensorclasses and lazy stacks stay the same
        # objects. Keys that hold one tensor share its pinned copy.
        @tensorclass
        class MeshLike:
            points: torch.Tensor
            cache: TensorDict
            name: str = "mesh"

        mesh = MeshLike(
            points=torch.ones(4, 3),
            cache=TensorDict(normals=torch.ones(4, 3), batch_size=[4]),
            batch_size=[4],
        )
        stack = lazy_stack([TensorDict(c=torch.ones(2)) for _ in range(4)])
        shared = torch.ones(4)
        td = TensorDict(
            a=shared,
            same=shared,
            nested=TensorDict(b=torch.ones(4, 2), same=shared, batch_size=[4]),
            mesh=mesh,
            stack=stack,
            batch_size=[4],
            names=["n"],
        )
        # the tensordict stores its own tensorclass and lazy stack
        mesh, stack = td["mesh"], td["stack"]
        nested, cache, members = td["nested"], mesh.cache, list(stack.tensordicts)
        keys = list(td.keys(True, True))

        threads = set()
        pin_memory = torch.Tensor.pin_memory

        def spy(tensor, *args, **kwargs):
            threads.add(threading.current_thread().name)
            return pin_memory(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "pin_memory", spy)
        assert td.pin_memory(num_threads=num_threads, inplace=True) is td
        if num_threads:
            # one pool for the whole tree
            assert threading.current_thread().name not in threads
            assert len({name.rsplit("_", 1)[0] for name in threads}) == 1
        else:
            assert threads == {threading.current_thread().name}

        leaves = [td["a"], nested["b"], mesh.points, cache["normals"]]
        leaves += [member["c"] for member in members]
        assert all(leaf.is_pinned() for leaf in leaves)
        assert td["same"] is td["a"]
        assert nested["same"] is td["a"]
        assert td["nested"] is nested
        assert td["mesh"] is mesh
        assert mesh.cache is cache
        assert mesh.name == "mesh"
        assert td["stack"] is stack
        assert all(a is b for a, b in zip(stack.tensordicts, members))
        assert list(td.keys(True, True)) == keys
        assert td.batch_size == mesh.batch_size == torch.Size([4])
        assert td.names == ["n"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pin_memory_inplace_tensorclass_override(self):
        # A nested tensorclass is pinned without calling its own pin_memory
        @tensorclass
        class WithPinMemory:
            x: torch.Tensor

            def pin_memory(self):
                raise RuntimeError("WithPinMemory.pin_memory was called")

        td = TensorDict(
            t=WithPinMemory(x=torch.ones(3), batch_size=[3]), batch_size=[3]
        )
        assert td.pin_memory(inplace=True) is td
        assert td["t"].x.is_pinned()

    @staticmethod
    def _check_written_back(td, num_threads):
        # pin_memory(inplace=True) on a tree that is not made only of unlocked
        # TensorDicts, lazy stacks and tensorclasses over plain leaves writes
        # the pinned copies back into every leaf, as before
        leaves = list(td.items(True, True))
        assert td.pin_memory(num_threads=num_threads, inplace=True) is td
        for key, leaf in leaves:
            # a _SubTensorDict returns a new view of the same memory
            assert td.get(key).data_ptr() == leaf.data_ptr()
            assert not td.get(key).is_pinned()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("num_threads", [None, 2])
    def test_pin_memory_inplace_locked(self, num_threads):
        td = TensorDict(a=torch.ones(3), nested=TensorDict(b=torch.ones(3)))
        td["nested"].lock_()
        self._check_written_back(td, num_threads)
        assert td["nested"].is_locked
        self._check_written_back(TensorDict(a=torch.ones(3)).lock_(), num_threads)
        # a lazy stack with a locked member falls back as a whole
        unlocked = TensorDict(a=torch.ones(3))
        locked = TensorDict(a=torch.ones(3)).lock_()
        leaves = [unlocked["a"], locked["a"]]
        stack = lazy_stack([unlocked, locked])
        assert stack.pin_memory(num_threads=num_threads, inplace=True) is stack
        for member, leaf in zip(stack.tensordicts, leaves):
            assert member["a"] is leaf
            assert not leaf.is_pinned()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("num_threads", [None, 2])
    def test_pin_memory_inplace_memmap_and_shared(self, num_threads, tmpdir):
        TensorDict(a=torch.ones(3), nested=TensorDict(b=torch.ones(3))).memmap_(tmpdir)
        td = TensorDict.load_memmap(tmpdir)
        assert not td.is_locked
        td.set("c", torch.ones(3))
        self._check_written_back(td, num_threads)
        td["a"].fill_(5)
        assert (TensorDict.load_memmap(tmpdir)["a"] == 5).all()
        td = TensorDict(a=torch.ones(3).share_memory_(), c=torch.ones(3))
        self._check_written_back(td, num_threads)
        assert td["a"].is_shared()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("num_threads", [None, 2])
    def test_pin_memory_inplace_other_types(self, num_threads, tmpdir):
        source = TensorDict(a=torch.ones(3, 2), batch_size=[3])
        td = TensorDict(c=torch.ones(3), batch_size=[3])
        td._set_str("sub", source._get_sub_tensordict(0), validated=True, inplace=False)
        self._check_written_back(td, num_threads)
        td = TensorDict(
            c=torch.ones(3),
            p=TensorDictParams(TensorDict(w=torch.ones(3)), no_convert=True),
        )
        self._check_written_back(td, num_threads)
        # a leaf that is not strided makes the whole tree fall back
        c = torch.ones(3)
        td = TensorDict(c=c, s=torch.eye(3).to_sparse())
        assert td.pin_memory(num_threads=num_threads, inplace=True) is td
        assert td["c"] is c
        assert not c.is_pinned()
        if _has_h5py:
            persistent = PersistentTensorDict.from_dict(
                source, filename=str(tmpdir / "file.h5")
            )
            td = TensorDict(c=torch.ones(3), batch_size=[3])
            td._set_str("p", persistent, validated=True, inplace=False)
            assert td.pin_memory(num_threads=num_threads, inplace=True) is td
            assert td["p"] is persistent
            assert not td["c"].is_pinned()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pin_memory_underscore_pins(self):
        td = TensorDict(a=torch.ones(3), nested=TensorDict(b=torch.ones(3)))
        nested = td["nested"]
        with pytest.warns(DeprecationWarning, match="pin_memory_"):
            assert td.pin_memory_() is td
        assert td["nested"] is nested
        assert td["a"].is_pinned()
        assert nested["b"].is_pinned()

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_to_non_blocking_pin_slow_leaf(self, device, monkeypatch):
        # A leaf that takes longer than the wait timeout to pin (the first read
        # of a large memory-mapped file, say) must not stop the transfer
        monkeypatch.setattr("tensordict._base.device._PIN_MEM_TIMEOUT", 0.1)
        pin_memory = torch.Tensor.pin_memory

        def slow_pin_memory(tensor, *args, **kwargs):
            time.sleep(0.3)
            if device == "cpu":
                # pinning needs an accelerator
                return tensor
            return pin_memory(tensor, *args, **kwargs)

        td = TensorDict(a=torch.arange(3), b={"c": torch.ones(3)}, batch_size=[3])
        monkeypatch.setattr(torch.Tensor, "pin_memory", slow_pin_memory)
        out = td.to(device, non_blocking_pin=True, num_threads=2)
        assert out.device.type == device
        assert (out.cpu() == td).all()

    def test_to_non_blocking_pin_dead_thread(self, monkeypatch):
        # A pin thread that exits without returning its leaf raises a clear error
        monkeypatch.setattr("tensordict._base.device._PIN_MEM_TIMEOUT", 0.1)
        monkeypatch.setattr(
            "tensordict._base.device._pin_mem", lambda q_in, q_out: None
        )
        td = TensorDict(a=torch.arange(3), b={"c": torch.ones(3)}, batch_size=[3])
        with pytest.raises(
            RuntimeError, match=r"exited without returning 2 of the 2 leaves"
        ):
            td.to("cpu", non_blocking_pin=True, num_threads=2)

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

    @pytest.mark.parametrize(
        "make_td",
        [
            lambda: TensorDict(
                a=torch.randn(3, 4),
                b={"c": torch.zeros(3, dtype=torch.int64)},
                batch_size=[3],
            ),
            lambda: TensorDict(
                a=torch.randn(3), b={"c": torch.zeros(3)}, batch_size=[3]
            ).lock_(),
            lambda: TensorDict(
                a=torch.randn(3, 4), batch_size=[3, 4], names=["x", "y"]
            ),
            lambda: TensorDict(
                a=torch.randn(3), b={"c": "text"}, batch_size=[3]
            ).consolidate(),
            lambda: lazy_stack(
                [TensorDict(a=torch.randn(3)), TensorDict(a=torch.randn(4))]
            ),
            lambda: TensorDict(
                s="text",
                i=NonTensorData(1),
                l=NonTensorData([1, "a"]),
                d=NonTensorData({"k": 0.5}),
            ),
            lambda: TensorDict(
                a=torch.zeros(2), s=NonTensorStack("x", "y"), batch_size=[2]
            ),
        ],
        ids=[
            "nested",
            "locked",
            "named",
            "consolidated",
            "lazy_stack",
            "non_tensor",
            "non_tensor_stack",
        ],
    )
    def test_torch_load_weights_only(self, make_td, tmpdir):
        td = make_td()
        filename = Path(tmpdir) / "td.pt"
        torch.save(td, filename)
        td_load = torch.load(filename, weights_only=True)
        assert type(td_load) is type(td)
        assert td_load.batch_size == td.batch_size
        assert td_load.names == td.names
        assert (td_load == td).all()

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            ("max", {"dim": 0}),
            ("min", {"dim": 0}),
            ("cummax", {"dim": 0}),
            ("cummin", {"dim": 0}),
            ("cummax", {"dim": 0, "reduce": True}),
        ],
    )
    @pytest.mark.parametrize("non_tensor", [False, True])
    def test_torch_load_weights_only_return_types(
        self, method, kwargs, non_tensor, tmpdir
    ):
        # tensordict allowlists the torch.return_types of these reductions
        if non_tensor:
            if kwargs.get("reduce"):
                pytest.skip("reduce=True needs tensor leaves")
            td = NonTensorData("text", batch_size=[3])
        else:
            td = TensorDict(a=torch.randn(3, 4), batch_size=[3])
        result = getattr(td, method)(**kwargs)
        filename = Path(tmpdir) / "result.pt"
        torch.save(result, filename)
        result_load = torch.load(filename, weights_only=True)
        assert type(result_load) is type(result)
        for value, value_load in zip(result, result_load):
            assert type(value_load) is type(value)
            assert (value_load == value).all()

    @pytest.mark.parametrize(
        "payload", [np.zeros(2), DummyPicklableClass(0)], ids=["numpy", "custom_class"]
    )
    def test_torch_load_weights_only_unlisted_payload(self, payload, tmpdir):
        # tensordict only allowlists its own classes: loading stops at the
        # payload's class, which the user has to allowlist.
        td = TensorDict(a=torch.zeros(3), s=NonTensorData(payload), batch_size=[3])
        filename = Path(tmpdir) / "td.pt"
        torch.save(td, filename)
        module = re.escape(type(payload).__module__)
        with pytest.raises(
            pickle.UnpicklingError, match=f"Unsupported global: GLOBAL {module}"
        ):
            torch.load(filename, weights_only=True)

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

        # the entries pickle by the files of td_base
        td_pickle = pickle.loads(pickle.dumps(td))
        assert td_pickle["b", "c"].filename == td_base["b", "c"].filename
        assert (td_pickle == td).all()

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

        # test update
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            td.make_memmap_from_tensor(
                ("e", "f"),
                torch.nested.nested_tensor([torch.zeros((1, 2)), torch.zeros((1, 3))]),
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
        # ).memmap(f"{Path(__file__).parent.parent}/artifacts/mmap_example/")

        td = TensorDict.load(f"{Path(__file__).parent.parent}/artifacts/mmap_example/")
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
            f"{Path(__file__).parent.parent}/artifacts/mmap_example/nested"
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

    def test_mixin_method_annotations(self):
        # The mixins under tensordict/_base/ import TensorDictBase for type
        # checking only. Annotations that name it must still resolve at run
        # time, as they did when these methods were defined in tensordict.base.
        unresolved = {}
        for mixin in tensordict_base._TENSORDICTBASE_MIXINS:
            for name, attr in vars(mixin).items():
                func = getattr(attr, "__func__", attr)
                if name.startswith("_") or not inspect.isfunction(func):
                    continue
                try:
                    get_type_hints(func)
                except NameError as err:
                    if err.name in vars(tensordict_base):
                        unresolved[name] = err.name
                except TypeError:
                    # Annotations such as "str" | None fail for other reasons.
                    pass
        assert not unresolved, unresolved

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

    def test_merge_tensordicts_non_tensor(self):
        # The non-tensor entries of every tensordict are kept, and the first
        # one wins, as for tensors.
        td0 = TensorDict(a=torch.zeros(()), s="s0", sub=TensorDict(u="u0"))
        td1 = TensorDict(
            b=torch.ones(()),
            s="s1",
            t="t1",
            sub=TensorDict(u="u1", v="v1"),
            nested=TensorDict(w="w1"),
        )
        td2 = TensorDict(t="t2", x="x2")
        td = merge_tensordicts(td0, td1, td2)
        assert set(td.keys(True, True, is_leaf=is_leaf_nontensor)) == {
            "a",
            "b",
            "s",
            "t",
            "x",
            ("sub", "u"),
            ("sub", "v"),
            ("nested", "w"),
        }
        assert td["s"] == "s0"
        assert td["t"] == "t1"
        assert td["x"] == "x2"
        assert td["sub", "u"] == "u0"
        assert td["sub", "v"] == "v1"
        assert td["nested", "w"] == "w1"
        # td0["sub"] has no tensor: the entries merged into it stay out of td0
        assert set(td0["sub"].keys()) == {"u"}
        # A sub-tensordict of an earlier tensordict stays
        td = merge_tensordicts(
            TensorDict(k=TensorDict(x=torch.zeros(()))), TensorDict(k="k1")
        )
        assert td["k", "x"] == 0
        # Stacks of non-tensor data, also from a lazy stack
        td0 = TensorDict(a=torch.zeros(2), batch_size=[2])
        td1 = TensorDict(s=NonTensorStack("x", "y"), batch_size=[2])
        td = merge_tensordicts(td0, td1)
        assert td.get("s").tolist() == ["x", "y"]
        td = merge_tensordicts(td0, lazy_stack([TensorDict(s="x"), TensorDict(s="y")]))
        assert td.get("s").tolist() == ["x", "y"]
        # A value that out cannot take is left out, as before
        td = merge_tensordicts(
            TensorDict(a=torch.zeros(2, 3), batch_size=[2, 3]),
            TensorDict(s=NonTensorStack("x", "y"), batch_size=[2]),
        )
        assert "s" not in td.keys()

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

    def test_to_cpu_syncs_only_from_device(self, _path_td_sync):
        """to("cpu") syncs only if a leaf comes from another device (#2607)."""

        def make(nested_device="cpu"):
            return TensorDict(
                a=torch.randn(3),
                b=TensorDict(c=torch.randn(3, device=nested_device), batch_size=[3]),
                batch_size=[3],
            )

        def count_syncs(td, *args, **kwargs):
            global _SYNC_COUNTER
            _SYNC_COUNTER = 0
            td.to(*args, **kwargs)
            return _SYNC_COUNTER

        # Every leaf is on the CPU: nothing to wait for
        for td in (make(), make().lock_()):
            assert count_syncs(td, "cpu") == 0
            assert count_syncs(td, "cpu", torch.float64) == 0
        if torch.cuda.is_available():
            assert count_syncs(make(), "cpu", non_blocking_pin=True) == 0
            # One leaf comes from the GPU: the copies must be awaited, once
            for td in (make("cuda"), make("cuda").lock_()):
                assert count_syncs(td, "cpu") == 1
                assert count_syncs(td, "cpu", torch.float64) == 1
                assert count_syncs(td, "cpu", non_blocking=True) == 0
                assert count_syncs(td, "cpu", non_blocking=False) == 0

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

    @pytest.mark.parametrize(
        "case",
        [
            "dense",
            "nested",
            "names_locked",
            "lazy_dim0",
            "lazy_dim1",
            "tensorclass",
            "params",
            "non_tensor",
            "unbatched",
            "empty_nested",
            "top_lazy",
            "top_tensorclass",
        ],
    )
    @pytest.mark.parametrize(
        "pad_size",
        [[0, 2], [1, -1, 0, 1], [-3, 0], [-4, 0], [0, 0, -3, 0], [-4, 0, 0, -3]],
    )
    def test_pad_safe_matches_preflight(self, case, pad_size):
        # pad(safe=True) checks a dense tensordict while it pads it, not in a
        # walk before: it must give the result or the error of that walk
        # followed by an unchecked pad, including when a nested lazy stack
        # fails in its own pad before a later leaf fails the check. The error
        # must not chain the error that the check repeats.
        @tensorclass
        class _Sample:
            x: torch.Tensor

        def lazy(dim):
            # a lazy stack of batch size [3, 2] along dim
            size = (3, 2)[1 - dim]
            tds = [
                TensorDict(x=torch.ones(size), batch_size=[size])
                for _ in range((3, 2)[dim])
            ]
            return lazy_stack(tds, dim)

        b = torch.zeros(3, 2, dtype=torch.bool)
        inputs = {
            "dense": lambda: TensorDict(a=torch.ones(3, 2, 4), b=b, batch_size=[3, 2]),
            "nested": lambda: TensorDict(
                n=TensorDict(c=torch.ones(3, 2, 1), batch_size=[3, 2]),
                b=b,
                batch_size=[3, 2],
            ),
            "names_locked": lambda: TensorDict(
                {"a": torch.ones(3, 2), "n": {"c": torch.ones(3, 2, 5)}},
                batch_size=[3, 2],
                names=["t", "u"],
            ).lock_(),
            "lazy_dim0": lambda: TensorDict(l=lazy(0), b=b, batch_size=[3, 2]),
            "lazy_dim1": lambda: TensorDict(l=lazy(1), b=b, batch_size=[3, 2]),
            "tensorclass": lambda: TensorDict(
                t=_Sample(x=torch.ones(3, 2), batch_size=[3, 2]), b=b, batch_size=[3, 2]
            ),
            "params": lambda: TensorDict(
                p=TensorDictParams(TensorDict(w=torch.ones(3, 2), batch_size=[3, 2])),
                b=b,
                batch_size=[3, 2],
            ),
            "non_tensor": lambda: TensorDict(b=b, batch_size=[3, 2]).set_non_tensor(
                "s", "x"
            ),
            "unbatched": lambda: TensorDict(
                u=tensordict.UnbatchedTensor(torch.zeros(5)), b=b, batch_size=[3, 2]
            ),
            "empty_nested": lambda: TensorDict(
                e=TensorDict(batch_size=[3, 2]), b=b, batch_size=[3, 2]
            ),
            "top_lazy": lambda: lazy(0),
            "top_tensorclass": lambda: _Sample(x=torch.ones(3, 2), batch_size=[3, 2]),
        }
        td = inputs[case]()

        def run(fn):
            try:
                return fn(), None
            except Exception as err:
                return None, (type(err), str(err), type(err.__context__))

        def reference():
            _pad_preflight(td, pad_size)
            return pad(td, pad_size, safe=False)

        out, err = run(lambda: pad(td, pad_size))
        ref, ref_err = run(reference)
        assert err == ref_err
        if ref_err is not None:
            return
        assert type(out) is type(ref)
        assert out.batch_size == ref.batch_size
        assert out.names == ref.names
        assert out.is_locked == ref.is_locked
        keys = list(ref.keys(True, is_leaf=is_leaf_nontensor))
        assert list(out.keys(True, is_leaf=is_leaf_nontensor)) == keys
        for key in keys:
            value, expected = out.get(key), ref.get(key)
            assert type(value) is type(expected), key
            if is_non_tensor(expected):
                assert value.tolist() == expected.tolist(), key
            elif not is_tensor_collection(expected):
                assert torch.equal(value, expected), key

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

    def test_pad_inplace_tensorclass_no_warning(self):
        # pad(..., inplace=True) sets the batch size of a tensorclass, top
        # level or nested, without the tensorclass fallback warning
        @tensorclass
        class _Sample:
            a: torch.Tensor

        s = _Sample(a=torch.ones(3, 4), batch_size=[3])
        td = TensorDict(s=_Sample(a=torch.ones(3), batch_size=[3]), batch_size=[3])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert pad(s, [0, 1], inplace=True) is s
            pad(td, [0, 1], inplace=True)
        assert s.batch_size == (4,)
        assert td["s"].batch_size == (4,)
        assert td["s"].a.shape == (4,)

    def test_pad_tensorclass(self):
        # pad without inplace=True returns a tensorclass for a tensorclass,
        # at the top level and nested, like the tensorclass method
        @tensorclass
        class _Inner:
            y: torch.Tensor

        @tensorclass
        class _Sample:
            a: torch.Tensor
            inner: _Inner
            s: str = "x"

        s = _Sample(
            a=torch.ones(3, 4),
            inner=_Inner(y=torch.ones(3), batch_size=[3]),
            batch_size=[3],
        )
        out = pad(s, [0, 1])
        assert type(out) is _Sample
        assert type(out.inner) is _Inner
        assert out.batch_size == (4,)
        assert out.a.shape == (4, 4)
        assert out.inner.y.shape == (4,)
        assert out.s == ["x", "x", "x", None]
        assert (out.a == s.pad([0, 1]).a).all()
        assert type(s.pad([0, 1]).inner) is _Inner
        td = TensorDict(s=s, batch_size=[3])
        assert type(pad(td, [0, 1])["s"]) is _Sample

        # a tensorclass may override pad with another signature
        @tensorclass
        class _CustomPad:
            x: torch.Tensor

            def pad(self, n):
                raise AssertionError

        c = _CustomPad(x=torch.ones(3), batch_size=[3])
        out = pad(c, [0, 1])
        assert type(out) is _CustomPad
        assert out.x.shape == (4,)
        out = pad(TensorDict(c=c, batch_size=[3]), [0, 1])
        assert type(out["c"]) is _CustomPad
        assert out["c"].x.shape == (4,)

    def test_pad_keeps_names(self):
        td = TensorDict(
            a=torch.ones(3, 4),
            nested=TensorDict(b=torch.ones(3, 4, 2), batch_size=[3, 4, 2]),
            batch_size=[3, 4],
            names=["t", "f"],
        )
        out = pad(td, [0, 1, 1, 0])
        assert out.names == ["t", "f"]
        assert out["nested"].names == ["t", "f", None]
        assert out.names == pad(td.clone(), [0, 1, 1, 0], inplace=True).names
        # a lazy stack of named tensordicts keeps the name of its stack dim too
        lazy = lazy_stack([td[0], td[1]]).rename_("s", "f")
        out = pad(lazy, [0, 1, 1, 0])
        assert out.names == ["s", "f"]
        assert out["nested"].names == ["s", "f", None]

        @tensorclass
        class _Sample:
            a: torch.Tensor

        s = _Sample(a=torch.ones(3, 4), batch_size=[3, 4], names=["t", "f"])
        assert pad(s, [0, 1]).names == ["t", "f"]
        # non-tensor values keep their names too, so the result can be used
        # like the input
        td = TensorDict(a=torch.ones(3, 4), batch_size=[3, 4], names=["t", "f"])
        td.set_non_tensor("s", "x")
        out = pad(td, [0, 1])
        assert out.get("s").names == ["t", "f"]
        out.apply(lambda x: x)
        out.update(out.clone())
        out.clone().set("s2", out.get("s"))
        assert pad(td.clone(), [0, 1], inplace=True).get("s").names == ["t", "f"]
        # a stack of non-tensor values with different names still pads
        x, y = (NonTensorData("v", batch_size=[2], names=[name]) for name in "xy")
        pad(NonTensorStack(x, y), [0, 1])
        # a lazy stack whose members have different names, or that has no
        # members, has no names to keep: the result is unnamed, as before
        a, b = (TensorDict(batch_size=[2], names=[name]) for name in "ab")
        assert pad(lazy_stack([a, b]), [0, 0, 0, 1]).names == [None, None]
        empty = LazyStackedTensorDict(stack_dim=0, batch_size=[2])
        assert pad(empty, [0, 0, 0, 1]).names == [None, None]

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
    @pytest.mark.parametrize(
        "repeats", [torch.Size([2, 3]), [2, 3], (2, 3)], ids=["size", "list", "tuple"]
    )
    def test_repeat_size_list_tuple(self, repeats, inplace):
        # repeats can be one torch.Size, list or tuple, as in torch, with or
        # without inplace
        td = self._build_nested_td()
        ref = self._build_nested_td().repeat(2, 3)
        out = td.repeat(repeats, inplace=inplace)
        assert (out is td) is inplace
        assert out.batch_size == torch.Size([6, 12])
        assert (out == ref).all()

    @pytest.mark.parametrize("inplace", [True, False])
    def test_repeat_tensor(self, inplace):
        # a tensor of counts is a sequence of integers, and inplace is kept
        td = self._build_nested_td()
        ref = self._build_nested_td().repeat(2, 1)
        leaf = td["a"]
        out = td.repeat(torch.tensor([2, 1]), inplace=inplace)
        assert (out is td) is inplace
        if inplace:
            assert td.batch_size == torch.Size([6, 4])
        else:
            assert td.batch_size == torch.Size([3, 4])
            assert td["a"] is leaf
        assert out.batch_size == torch.Size([6, 4])
        assert (out == ref).all()

    def test_repeat_zero_dim_tensor(self):
        # a 0-dim tensor is one repeat count, the same as passing that int
        td = self._build_nested_td(batch_size=(4,))
        ref = self._build_nested_td(batch_size=(4,)).repeat(2)
        out = td.repeat(torch.tensor(2))
        assert out is not td
        assert td.batch_size == torch.Size([4])
        assert out.batch_size == torch.Size([8])
        assert (out == ref).all()
        td2 = self._build_nested_td()
        with pytest.raises(ValueError, match="number of repeat elements"):
            td2.repeat(torch.tensor(2))

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
    @pytest.mark.parametrize("dim", [0, -1])
    @pytest.mark.parametrize(
        "repeats", [torch.tensor(2), torch.tensor([2])], ids=["0d", "one-element"]
    )
    def test_repeat_interleave_one_element_tensor(self, repeats, dim, inplace):
        # a 0-d or one-element repeats is broadcast to the size of dim, as in
        # torch
        td = self._build_nested_td()
        ref = self._build_nested_td().repeat_interleave(2, dim=dim)
        out = td.repeat_interleave(repeats, dim=dim, inplace=inplace)
        assert (out is td) is inplace
        assert (
            out.batch_size
            == torch.empty(3, 4).repeat_interleave(repeats, dim=dim).shape
        )
        assert (out == ref).all()

    @pytest.mark.parametrize(
        "op",
        [
            "repeat_interleave",
            "flatten",
            "stack",
            "stack_wraps",
            "maybe_dense_stack",
            "cat",
            "cat_lazy",
            "lazy_stack",
        ],
    )
    def test_dim_past_batch_dims_warns(self, op):
        # A dim outside the batch dims is used on the dims of the leaves. It
        # warns, and will raise an IndexError in 0.17, as for a tensor (#2148).
        td = TensorDict({"a": torch.arange(24.0).view(2, 3, 4)}, [2, 3])
        x = td["a"]
        tds = list(td.unbind(0))  # batch size [3], leaves [3, 4]
        lazy = lazy_stack(tds)
        # leaves that make valid results, which TD_CHECK_INVARIANTS accepts
        y = torch.arange(12.0).view(2, 3, 2)
        tds_y = list(TensorDict({"a": y}, [2, 3]).unbind(0))  # leaves [3, 2]
        z = torch.arange(6.0).view(2, 3)
        tds_z = list(TensorDict({"a": z}, [2, 3]).unbind(0))  # leaves [3]
        # op: (call with a dim outside the batch dims, its batch size and leaf
        # today, the same op with dims inside the batch dims)
        calls = {
            "repeat_interleave": (
                lambda: td.repeat_interleave(2, dim=2),
                (2, 3),
                x.repeat_interleave(2, dim=2),
                lambda: td.repeat_interleave(2, dim=-2),
            ),
            "flatten": (
                lambda: td.flatten(0, 2),
                (6,),
                x.flatten(0, 2),
                lambda: td.flatten(-2, -1),
            ),
            "stack": (
                lambda: torch.stack(tds_y, 2),
                (3, 2),
                torch.stack([y[0], y[1]], 2),
                lambda: torch.stack(tds_y, -2),
            ),
            "stack_wraps": (
                lambda: torch.stack(tds, -5),
                (2, 3),
                x,
                lambda: torch.stack(tds, 1),
            ),
            "maybe_dense_stack": (
                lambda: LazyStackedTensorDict.maybe_dense_stack(tds, -5),
                (2, 3),
                x,
                lambda: LazyStackedTensorDict.maybe_dense_stack(tds, -1),
            ),
            "cat": (
                lambda: torch.cat(tds_z, -2),
                (6,),
                torch.cat([z[0], z[1]]),
                lambda: torch.cat(tds_z, -1),
            ),
            "cat_lazy": (
                lambda: torch.cat([lazy, lazy], -3),
                (2, 6),
                torch.cat([x, x], 1),
                lambda: torch.cat([lazy, lazy], -1),
            ),
            "lazy_stack": (
                lambda: lazy_stack(tds, -3),
                (3, 2),
                torch.stack([x[0], x[1]], 1),
                lambda: lazy_stack(tds, -2),
            ),
        }
        call, batch_size, leaf, in_range = calls[op]
        with pytest.warns(
            FutureWarning,
            match="Starting with TensorDict 0.17, the default will change to "
            "raising an IndexError",
        ) as record:
            out = call()
        assert record[0].filename == __file__
        assert out.batch_size == batch_size
        assert torch.equal(out["a"], leaf)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            in_range()

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
    def test_roll_feature_dims(self, inplace):
        # dims are batch dims: negative dims and the default (flattened batch)
        # must leave the feature dims of the leaves in place
        a = torch.arange(24).view(2, 3, 4)
        b = torch.arange(12).view(2, 3, 2)
        td = TensorDict({"a": a, "nested": TensorDict({"b": b}, [2, 3, 2])}, [2, 3])

        out = td.clone().roll(1, -1, inplace=inplace)
        assert (out["a"] == a.roll(1, 1)).all()
        assert (out["nested", "b"] == b.roll(1, 1)).all()

        out = td.clone().roll(1, inplace=inplace)
        assert (out["a"] == a.view(6, 4).roll(1, 0).view(2, 3, 4)).all()
        assert (out["nested", "b"] == b.view(6, 2).roll(1, 0).view(2, 3, 2)).all()

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("dims", [None, ()])
    def test_roll_flat_batch_edge_cases(self, dims, inplace):
        # empty dims roll the flattened batch like None, as in torch, and an
        # empty feature dim does not make the flattened batch ambiguous
        b = torch.arange(12).view(2, 3, 2)
        td = TensorDict(
            {"z": torch.zeros(2, 3, 0), "nested": TensorDict({"b": b}, [2, 3, 2])},
            [2, 3],
        )
        out = td.clone().roll(1, dims, inplace=inplace)
        assert out["z"].shape == (2, 3, 0)
        assert (out["nested", "b"] == b.view(6, 2).roll(1, 0).view(2, 3, 2)).all()

    @pytest.mark.parametrize("inplace", [True, False])
    def test_roll_flat_batch_keeps_nested_names(self, inplace):
        # rolling the flattened batch reshapes each nested tensordict, which
        # must not drop its dim names
        nested = TensorDict(
            {"b": torch.zeros(2, 3, 2)}, [2, 3, 2], names=["x", "y", "z"]
        )
        td = TensorDict({"nested": nested}, [2, 3], names=["x", "y"])
        out = td.roll(1, inplace=inplace)
        assert out.names == ["x", "y"]
        assert out["nested"].names == ["x", "y", "z"]

    def test_roll_flat_batch_inplace_rebinds_nested(self):
        # like the other in-place paths, roll(inplace=True) replaces the nested
        # leaves instead of writing into them: a tensor the caller still holds
        # is left untouched, and an expanded leaf can be rolled
        b = torch.arange(12).view(2, 3, 2)
        e = torch.arange(2).expand(2, 3, 2)
        td = TensorDict({"nested": TensorDict({"b": b, "e": e}, [2, 3, 2])}, [2, 3])
        td.roll(1, inplace=True)
        assert (b == torch.arange(12).view(2, 3, 2)).all()
        assert (td["nested", "b"] == b.view(6, 2).roll(1, 0).view(2, 3, 2)).all()
        assert (td["nested", "e"] == e).all()

    @pytest.mark.parametrize(
        "dims,batch_size,leaf_reps",
        [
            ((2,), [2, 6], (1, 2, 1)),
            ((2, 1), [4, 3], (2, 1, 1)),
            ((3, 1, 1), [3, 2, 3], (3, 1, 1, 1)),
        ],
    )
    def test_tile_feature_dims(self, dims, batch_size, leaf_reps):
        # dims are aligned with the batch dims, feature dims are not tiled
        a = torch.arange(24).view(2, 3, 4)
        td = TensorDict(
            {"a": a, "nested": TensorDict({"b": a.clone()}, [2, 3, 4])}, [2, 3]
        )
        out = td.tile(dims)
        assert out.batch_size == torch.Size(batch_size)
        assert (out["a"] == a.tile(leaf_reps)).all()
        assert (out["nested", "b"] == a.tile(leaf_reps)).all()

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

    @pytest.mark.parametrize("locked", [False, True])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_detach__unbatched(self, locked, lazy):
        # A tensordict stores an UnbatchedTensor as an alias, which torch
        # cannot detach in place: detach_ replaces it with its detach(), which
        # shares the storage (#2565)
        x = torch.ones(5, requires_grad=True) * 2

        def build():
            return TensorDict(
                a=torch.zeros(3, requires_grad=True) * 1,
                u=tensordict.UnbatchedTensor(x),
                nested=TensorDict(u=tensordict.UnbatchedTensor(x), batch_size=[3]),
                batch_size=[3],
            )

        td = lazy_stack([build(), build()]) if lazy else build()
        if locked:
            td.lock_()
        members = td.tensordicts if lazy else [td]
        ptrs = [
            (m.get("u").data_ptr(), m.get(("nested", "u")).data_ptr()) for m in members
        ]
        assert td.detach_() is td
        assert td.is_locked is locked
        for member, (ptr, nested_ptr) in zip(members, ptrs):
            u, nested_u = member.get("u"), member.get(("nested", "u"))
            assert isinstance(u, tensordict.UnbatchedTensor)
            assert isinstance(nested_u, tensordict.UnbatchedTensor)
            assert not u.requires_grad and not nested_u.requires_grad
            assert not member["a"].requires_grad
            assert u.data_ptr() == ptr and nested_u.data_ptr() == nested_ptr
            assert u.shape == (5,)

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("mask_dims", [1, 2])
    def test_masked_select_inplace(self, inplace, mask_dims):
        @tensorclass
        class _Data:
            x: torch.Tensor

        def build():
            td = self._build_nested_td(batch_size=(3, 4))
            td["tc"] = _Data(x=torch.arange(12).view(3, 4), batch_size=[3, 4])
            td["u"] = tensordict.UnbatchedTensor(torch.zeros(7))
            td.set_non_tensor("s", "x")
            return td

        mask = torch.tensor([[True, False, True, False], [False] * 4, [True] * 4])
        if mask_dims == 1:
            mask = mask[:, 0]
        ref = build().masked_select(mask)
        td = build()
        nested, tc = td["nested"], td["tc"]
        out = td.masked_select(mask, inplace=inplace)
        assert (out is td) is inplace
        if inplace:
            assert out["nested"] is nested
            assert out["tc"] is tc
        assert out.batch_size == ref.batch_size
        assert out["tc"].batch_size == ref.batch_size
        assert out["u"].batch_size == ref.batch_size
        assert out["s"] == "x"
        assert (out == ref).all()

    def test_masked_select_inplace_names(self):
        td = TensorDict(a=torch.zeros(3, 4), batch_size=[3, 4], names=["t", "f"])
        td.masked_select(torch.ones(3, 4, dtype=torch.bool), inplace=True)
        assert td.batch_size == torch.Size([12])
        assert td.names == [None]

    def test_masked_select_inplace_releases_storage(self):
        td = TensorDict({"a": torch.zeros(4, 5, 8)}, batch_size=[4, 5])
        ref = weakref.ref(td["a"])
        td.masked_select(torch.ones(4, 5, dtype=torch.bool), inplace=True)
        gc.collect()
        assert ref() is None

    def test_masked_select_inplace_lazy_stack_raises(self):
        lst = lazy_stack(
            [TensorDict({"x": torch.zeros(4)}, [4]) for _ in range(3)], dim=0
        )
        with pytest.raises(NotImplementedError, match="masked_select"):
            lst.masked_select(torch.ones(3, 4, dtype=torch.bool), inplace=True)

    def test_masked_select_inplace_sub_tensordict_raises(self):
        sub = TensorDict({"x": torch.zeros(4, 3)}, [4])._get_sub_tensordict(slice(0, 2))
        with pytest.raises(
            NotImplementedError,
            match="masked_select.inplace=True. is not supported on _SubTensorDict",
        ):
            sub.masked_select(torch.ones(2, dtype=torch.bool), inplace=True)
        assert sub.batch_size == (2,)

    @pytest.mark.parametrize(
        "op",
        [
            "reshape",
            "repeat",
            "gather",
            "gather_negative_dim",
            "gather_smaller_index_dim0",
            "gather_smaller_index_dim1",
        ],
    )
    @pytest.mark.parametrize("nested_type", ["tensordict", "tensorclass"])
    def test_inplace_shape_op_nested_extra_batch_dims(self, op, nested_type):
        # The nested entry has more batch dims than its parent and comes after
        # a tensor leaf in key order, so a failure on it would leave the
        # parent half rewritten. The gather_smaller_index cases use an index
        # smaller than the tensordict on the dim that is not gathered.
        @tensorclass
        class Sub:
            y: torch.Tensor

        def make():
            y = torch.arange(24.0).view(2, 3, 4)
            if nested_type == "tensordict":
                sub = TensorDict(y=y, batch_size=[2, 3, 4])
            else:
                sub = Sub(y=y, batch_size=[2, 3, 4])
            return TensorDict(
                x=torch.arange(6.0).view(2, 3), sub=sub, batch_size=[2, 3]
            )

        index = torch.tensor([[0, 2], [1, 0]])
        ops = {
            "reshape": lambda td, **kwargs: td.reshape(3, 2, **kwargs),
            "repeat": lambda td, **kwargs: td.repeat(2, 1, **kwargs),
            "gather": lambda td, **kwargs: td.gather(1, index, **kwargs),
            "gather_negative_dim": lambda td, **kwargs: td.gather(-1, index, **kwargs),
            "gather_smaller_index_dim0": lambda td, **kwargs: td.gather(
                1, torch.tensor([[2, 0]]), **kwargs
            ),
            "gather_smaller_index_dim1": lambda td, **kwargs: td.gather(
                0, torch.tensor([[1, 0], [0, 0]]), **kwargs
            ),
        }
        ref = ops[op](make())
        td = make()
        sub = td["sub"]
        out = ops[op](td, inplace=True)
        assert out is td
        assert td["sub"] is sub
        assert td.batch_size == ref.batch_size
        assert sub.batch_size == ref["sub"].batch_size
        for key in ref.keys(include_nested=True, leaves_only=True):
            assert torch.equal(td[key], ref[key]), key

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("with_out", [False, True])
    @pytest.mark.parametrize(
        "dim,index",
        [
            (1, [[2, 0]]),  # size 1 on dim 0
            (1, [[2, 0], [1, 3]]),  # size 2 on dim 0
            (1, [[3, 0, 1, 2, 0]]),  # also longer than the input on dim 1
            (0, [[1, 0], [2, 2]]),  # size 2 on dim 1
        ],
        ids=["size1", "size2", "longer", "dim0"],
    )
    def test_gather_smaller_index(self, dim, index, with_out, lazy):
        # as with torch.gather, the index may be smaller than the input on the
        # dims that are not gathered, and the leaves follow its shape
        index = torch.tensor(index)
        a = torch.arange(24).view(3, 4, 2)
        b = torch.arange(12).view(3, 4)
        c = torch.arange(72).view(3, 4, 6)
        td = TensorDict(
            a=a,
            nested=TensorDict(b=b, batch_size=[3, 4]),
            extra=TensorDict(c=c, batch_size=[3, 4, 6]),
            batch_size=[3, 4],
        )
        if lazy:
            td = lazy_stack(list(td.unbind(0)))
        expected = TensorDict(
            a=a.gather(dim, index.unsqueeze(-1).expand(*index.shape, 2)),
            nested=TensorDict(b=b.gather(dim, index), batch_size=index.shape),
            extra=TensorDict(
                c=c.gather(dim, index.unsqueeze(-1).expand(*index.shape, 6)),
                batch_size=[*index.shape, 6],
            ),
            batch_size=index.shape,
        )
        out = torch.full_like(expected, -1) if with_out else None
        result = td.gather(dim, index, out=out)
        if with_out:
            assert result is out
        assert result.batch_size == index.shape
        assert result["extra"].batch_size == expected["extra"].batch_size
        for key in expected.keys(include_nested=True, leaves_only=True):
            assert torch.equal(result[key], expected[key]), key

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

    @pytest.mark.parametrize(
        "op",
        [
            lambda td, **kwargs: td.reshape(6, **kwargs),
            lambda td, **kwargs: td.reshape(3, 2, **kwargs),
            lambda td, **kwargs: td.flatten(0, 1, **kwargs),
            lambda td, **kwargs: td.unflatten(1, (3, 1), **kwargs),
            lambda td, **kwargs: td.unflatten(0, (1, 2), **kwargs),
        ],
        ids=["reshape", "reshape-2d", "flatten", "unflatten", "unflatten-dim0"],
    )
    @pytest.mark.parametrize("names", [["a", "b"], [None, "b"], None])
    def test_inplace_shape_op_names(self, op, names):
        # inplace=True gives the dim names that the call without it gives
        def make():
            # the nested dim beyond the batch dims of td has a name of its own
            return TensorDict(
                x=torch.zeros(2, 3, 4),
                sub=TensorDict(
                    y=torch.zeros(2, 3, 4),
                    batch_size=[2, 3, 4],
                    names=[*(names or [None, None]), "z"],
                ),
                batch_size=[2, 3],
                names=names,
            )

        ref = op(make())
        td = make()
        op(td, inplace=True)
        assert td.batch_size == ref.batch_size
        assert td.names == ref.names
        assert td["sub"].names == ref["sub"].names

    @pytest.mark.parametrize(
        "op,call",
        [
            ("reshape", lambda td: td.reshape(3, 2, inplace=True)),
            ("flatten", lambda td: td.flatten(inplace=True)),
            ("unflatten", lambda td: td.unflatten(1, (3, 1), inplace=True)),
            ("repeat", lambda td: td.repeat(2, 1, inplace=True)),
            (
                "repeat_interleave",
                lambda td: td.repeat_interleave(2, dim=0, inplace=True),
            ),
            (
                "gather",
                lambda td: td.gather(
                    1, torch.tensor([[2, 1, 0], [0, 2, 1]]), inplace=True
                ),
            ),
            (
                "masked_select",
                lambda td: td.masked_select(
                    torch.tensor([[True, False, True], [False, True, True]]),
                    inplace=True,
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("stack_type", ["non_tensor_stack", "lazy_stack"])
    @pytest.mark.parametrize(
        "container", ["tensordict", "nested_tensordict", "nested_tensorclass"]
    )
    def test_inplace_shape_op_nested_lazy_stack_unchanged(
        self, op, call, stack_type, container
    ):
        # The tensor leaf "x" comes before the lazy entry "s" in key order, so
        # a check that runs only when "s" is reached would rewrite "x" first.
        # The nested containers also hold a tensor leaf "z" before "s".
        @tensorclass
        class Holder:
            z: torch.Tensor
            s: Any

        if stack_type == "non_tensor_stack":
            s = NonTensorStack.from_list([["a", "b", "c"], ["d", "e", "f"]])
        else:
            s = lazy_stack(
                [
                    TensorDict(y=torch.arange(3) + 10 * i, batch_size=[3])
                    for i in range(2)
                ]
            )
        x = torch.arange(6).view(2, 3)
        if container == "tensordict":
            td = TensorDict(x=x, s=s, batch_size=[2, 3])
            holder = td
            key = "s"
        else:
            z = torch.arange(6).view(2, 3) + 10
            cls = TensorDict if container == "nested_tensordict" else Holder
            td = TensorDict(x=x, n=cls(z=z, s=s, batch_size=[2, 3]), batch_size=[2, 3])
            holder = td.get("n")
            if container == "nested_tensorclass":
                holder = holder._tensordict
            key = ("n", "s")
        assert list(td.keys()) == ["x", "s" if container == "tensordict" else "n"]
        ref = td.clone()
        with pytest.raises(NotImplementedError, match=rf"{op}\(inplace=True\)") as err:
            call(td)
        assert td.batch_size == torch.Size([2, 3])
        assert torch.equal(td["x"], ref["x"])
        if container != "tensordict":
            assert holder.batch_size == torch.Size([2, 3])
            assert torch.equal(holder["z"], z)
        assert holder.get("s") is s
        assert s.batch_size == torch.Size([2, 3])
        if stack_type == "non_tensor_stack":
            assert s.tolist() == [["a", "b", "c"], ["d", "e", "f"]]
        else:
            assert torch.equal(s["y"], torch.tensor([[0, 1, 2], [10, 11, 12]]))
        assert f"found a {type(s).__name__} at key {key!r}" in str(err.value)
        assert "inplace=False" in str(err.value)

    @legacy_lazy_mode()
    @pytest.mark.parametrize(
        "op,call",
        [
            ("reshape", lambda td: td.reshape(3, 2, inplace=True)),
            ("flatten", lambda td: td.flatten(inplace=True)),
            ("unflatten", lambda td: td.unflatten(1, (3, 1), inplace=True)),
            ("repeat", lambda td: td.repeat(2, 1, inplace=True)),
            (
                "repeat_interleave",
                lambda td: td.repeat_interleave(2, dim=0, inplace=True),
            ),
            (
                "gather",
                lambda td: td.gather(
                    1, torch.tensor([[2, 1, 0], [0, 2, 1]]), inplace=True
                ),
            ),
            (
                "masked_select",
                lambda td: td.masked_select(
                    torch.tensor([[True, False, True], [False, True, True]]),
                    inplace=True,
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("view", ["permute", "transpose"])
    def test_inplace_shape_op_nested_legacy_lazy_view(self, op, call, view):
        # The legacy lazy views support reshape and repeat_interleave in place,
        # and refuse the other four ops before anything is rewritten.
        src = TensorDict(y=torch.arange(6).view(3, 2), batch_size=[3, 2])
        x = torch.arange(6).view(2, 3)
        td = TensorDict(x=x, batch_size=[2, 3])
        td["s"] = src.permute(1, 0) if view == "permute" else src.transpose(1, 0)
        s_type = type(td.get("s"))
        ref = td.to_tensordict()
        if op in ("reshape", "repeat_interleave"):
            assert call(td) is td
            expected = call(ref)
            assert td.batch_size == expected.batch_size
            assert torch.equal(td["x"], expected["x"])
            assert type(td.get("s")) is s_type
            assert torch.equal(td["s", "y"], expected["s", "y"])
        else:
            with pytest.raises(NotImplementedError, match=rf"{op}\(inplace=True\)"):
                call(td)
            assert td.batch_size == torch.Size([2, 3])
            assert torch.equal(td["x"], x)
            assert torch.equal(td["s", "y"], ref["s", "y"])

    @pytest.mark.parametrize("inplace", [True, False])
    def test_unflatten_minus_one(self, inplace):
        # The -1 is inferred, as in torch.unflatten
        td = TensorDict(
            {"a": torch.zeros(2, 6, 4), "b": {"c": torch.zeros(2, 6)}}, [2, 6]
        )
        out = td.unflatten(1, (3, -1), inplace=inplace)
        assert out.batch_size == out["b"].batch_size == (2, 3, 2)
        assert out["a"].shape == (2, 3, 2, 4)
        with pytest.raises(RuntimeError, match="don't multiply up"):
            TensorDict({}, [2, 6]).unflatten(1, (4, -1))
        # the -1 of a ragged dim stays
        assert TensorDict({}, [2, -1]).unflatten(1, (1, -1)).batch_size == (2, 1, -1)

    @pytest.mark.parametrize(
        "td_type", ["empty", "nested", "non_tensor", "lazy", "non_tensor_stack"]
    )
    @pytest.mark.parametrize("inplace", [True, False])
    def test_unflatten_wrong_element_count(self, td_type, inplace):
        # Without tensor leaves, nothing used to check the element count
        if td_type == "empty":
            td = TensorDict(batch_size=[3, 12])
        elif td_type == "nested":
            td = TensorDict(n=TensorDict(batch_size=[3, 12]), batch_size=[3, 12])
        elif td_type == "non_tensor":
            td = NonTensorData("x", batch_size=[3, 12])
        elif td_type == "lazy":
            td = lazy_stack([TensorDict(batch_size=[12]) for _ in range(3)])
        else:
            td = NonTensorStack(*[NonTensorData("x", batch_size=[12])] * 3)
        if inplace and td_type in ("lazy", "non_tensor_stack"):
            pytest.skip("lazy stacks have no in-place unflatten")
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "unflatten: Provided sizes [2, 7] don't multiply up to the size "
                "of dim 1 (12)"
            ),
        ):
            td.unflatten(-1, (2, 7), inplace=inplace)
        assert td.batch_size == (3, 12)
        assert td.unflatten(1, (4, 3)).batch_size == (3, 4, 3)
        assert TensorDict(batch_size=[0]).unflatten(0, (2, 0)).batch_size == (2, 0)
        with pytest.raises(RuntimeError, match="don't multiply up"):
            TensorDict(batch_size=[0]).unflatten(0, (2, 3))

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

    def test_contiguous_inplace_locked(self):
        # A locked tensordict whose leaves are contiguous keeps them; one with a
        # leaf to rebind raises
        td = TensorDict(
            {"a": torch.zeros(4, 3), "nested": {"b": torch.zeros(4, 3)}},
            batch_size=[4, 3],
        ).lock_()
        a = td["a"]
        assert td.contiguous(inplace=True) is td
        assert td.contiguous(canonical=True, inplace=True) is td
        assert td["a"] is a
        # an inplace op that returns the same leaves writes nothing either
        assert td.flatten(0, 0, inplace=True) is td
        assert td["a"] is a
        memmap_td = TensorDict(a=torch.zeros(4, 3), batch_size=[4, 3]).memmap_()
        assert memmap_td.flatten(0, 0, inplace=True) is memmap_td
        td = TensorDict(a=torch.zeros(3, 4).T, batch_size=[4]).lock_()
        with pytest.raises(RuntimeError, match="Cannot modify locked TensorDict"):
            td.contiguous(inplace=True)

    @pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
    def test_contiguous_inplace_persistent(self, tmp_path):
        h5 = PersistentTensorDict.from_dict(
            TensorDict(a=torch.zeros(3, 2), batch_size=[3]),
            filename=tmp_path / "file.h5",
        )
        with pytest.raises(
            NotImplementedError,
            match=r"contiguous\(inplace=True\) is not supported on PersistentTensorDict",
        ):
            h5.contiguous(inplace=True)
        h5.close()

    def test_pad_sequence_nontensor(self):
        d1 = TensorDict({"a": torch.tensor([1, 1]), "b": "asd"})
        d2 = TensorDict({"a": torch.tensor([2]), "b": "efg"})
        d = pad_sequence([d1, d2])
        assert (d["a"] == torch.tensor([[1, 1], [2, 0]])).all()
        assert d["b"] == ["asd", "efg"]

    @pytest.mark.parametrize("return_mask", [False, True])
    def test_pad_sequence_nontensor_batch_dim(self, return_mask):
        # pad_dim is a batch dim: non-tensor entries are padded as pad() pads
        # them, with None in the pad slots.
        d1 = TensorDict(
            {"a": torch.ones(3, 2), "b": "asd", ("c", "d"): "x"}, batch_size=[3]
        )
        d2 = TensorDict(
            {"a": torch.ones(5, 2), "b": "efg", ("c", "d"): "y"}, batch_size=[5]
        )
        d = pad_sequence([d1, d2], return_mask=return_mask)
        assert d.batch_size == torch.Size([2, 5])
        assert d.get("b").batch_size == torch.Size([2, 5])
        assert d.get("b").tolist() == [["asd"] * 3 + [None] * 2, ["efg"] * 5]
        assert d[0].get("b").tolist() == pad(d1, [0, 2]).get("b").tolist()
        assert d.get(("c", "d")).tolist() == [["x"] * 3 + [None] * 2, ["y"] * 5]
        if return_mask:
            assert d["masks", "a"].tolist() == [[True] * 3 + [False] * 2, [True] * 5]

    def test_pad_sequence_nontensor_out(self):
        # Non-tensor entries are padded only up to the batch size of out
        d1 = TensorDict({"a": torch.ones(3, 2), "b": "asd"}, batch_size=[3])
        d2 = TensorDict({"a": torch.ones(5, 2), "b": "efg"}, batch_size=[5])
        d = pad_sequence([d1, d2], out=TensorDict(batch_size=[2, 5]))
        assert d.get("b").tolist() == [["asd"] * 3 + [None] * 2, ["efg"] * 5]
        # an out without the pad dim keeps the stack as it is
        d = pad_sequence([d1, d2], out=TensorDict(batch_size=[2]))
        assert d.get("b").tolist() == [["asd"] * 3, ["efg"] * 5]
        # no tensor entries: nothing to pad to
        d = pad_sequence(
            [d1.exclude("a")[:3], d2.exclude("a")[:3]],
            out=TensorDict(batch_size=[2, 3]),
        )
        assert d.get("b").tolist() == [["asd"] * 3, ["efg"] * 3]

    @pytest.mark.parametrize("pad_dim", [0, -1])
    def test_pad_sequence_nontensor_only(self, pad_dim):
        # https://github.com/pytorch/tensordict/issues/2490
        d1 = TensorDict({"b": "asd", ("c", "d"): "x"}, batch_size=[3])
        d2 = TensorDict({"b": "efg", ("c", "d"): "y"}, batch_size=[5])
        d = pad_sequence([d1, d2], pad_dim=pad_dim)
        assert d.batch_size == torch.Size([2, 5])
        assert d.get("b").tolist() == [["asd"] * 3 + [None] * 2, ["efg"] * 5]
        assert d.get(("c", "d")).tolist() == [["x"] * 3 + [None] * 2, ["y"] * 5]

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

    def test_collate_dataloader(self):
        @tensorclass
        class Sample:
            x: torch.Tensor

        def load(samples):
            dl = torch.utils.data.DataLoader(  # noqa: TOR401
                samples, batch_size=4, collate_fn=collate
            )
            return next(iter(dl))

        # tensordicts with matching shapes stack densely
        tds = [
            TensorDict(a=torch.full((3,), i), nested=TensorDict(b=torch.zeros(2)))
            for i in range(8)
        ]
        batch = load(tds)
        assert type(batch) is TensorDict
        assert batch.batch_size == (4,)
        assert batch["nested", "b"].shape == (4, 2)
        assert (batch["a"][:, 0] == torch.arange(4)).all()
        # tensorclasses keep their type
        batch = load([Sample(x=torch.randn(3)) for _ in range(8)])
        assert type(batch) is Sample
        assert batch.x.shape == (4, 3)
        batch = load([Sample(x=torch.randn(n + 1)) for n in range(8)])
        assert isinstance(batch, Sample)
        assert [x.shape for x in batch.get("x", as_list=True)] == [
            (1,),
            (2,),
            (3,),
            (4,),
        ]
        # ragged tensordicts stack lazily
        batch = load([TensorDict(points=torch.randn(n + 1, 3)) for n in range(8)])
        assert isinstance(batch, LazyStackedTensorDict)
        assert [td["points"].shape for td in batch.tensordicts] == [
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 3),
        ]
        # tuples, namedtuples and dicts recurse
        td, label = load([(TensorDict(a=torch.randn(3)), i) for i in range(8)])
        assert td.batch_size == (4,)
        assert (label == torch.arange(4)).all()
        Pair = namedtuple("Pair", ["td", "label"])
        batch = load([Pair(TensorDict(a=torch.randn(3)), i) for i in range(8)])
        assert type(batch) is Pair
        assert batch.td.batch_size == (4,)
        batch = load(
            [{"td": TensorDict(a=torch.randn(3)), "x": torch.ones(2)} for _ in range(8)]
        )
        assert batch["td"].batch_size == (4,)
        assert batch["x"].shape == (4, 2)
        # anything that is not a non-empty list is returned as is
        td = TensorDict(a=torch.zeros(4), batch_size=[4])
        assert collate(td) is td
        assert collate([]) == []

    @pytest.mark.parametrize("batch_dims", [1, 2])
    @pytest.mark.parametrize("pad_dim", [2, -2])
    @pytest.mark.parametrize("padding_side", ["right", "left"])
    @pytest.mark.parametrize("masks_key", ["masks", ("valid", "masks")])
    def test_pad_sequence_feature_dim_masks(
        self, batch_dims, pad_dim, padding_side, masks_key
    ):
        batch_size = [2, 3][:batch_dims]
        samples = [
            TensorDict(
                {
                    "keys": torch.full((2, 3, length, 4), float(length)),
                    ("nested", "values"): torch.full((2, 3, length, 5), float(length)),
                },
                batch_size=batch_size,
            )
            for length in (2, 5)
        ]
        padded = pad_sequence(
            samples,
            pad_dim=pad_dim,
            return_mask=True if masks_key == "masks" else masks_key,
            padding_side=padding_side,
        )
        valid = [True] * 2 + [False] * 3
        if padding_side == "left":
            valid.reverse()
        expected = (
            torch.tensor([valid, [True] * 5])
            .reshape(2, *([1] * batch_dims), 5)
            .expand(2, *batch_size, 5)
        )
        masks = padded.get(masks_key)
        for key in samples[0].keys(True, True):
            torch.testing.assert_close(masks.get(key), expected)
        self._assert_unpadded(
            unpad_sequence(padded.exclude(masks_key), masks, pad_dim=pad_dim),
            samples,
        )

    @pytest.mark.parametrize("batch_dims", [1, 2])
    @pytest.mark.parametrize("padding_side", ["right", "left"])
    def test_pad_sequence_negative_batch_pad_dim(self, batch_dims, padding_side):
        # https://github.com/pytorch/tensordict/issues/2938
        samples = []
        for length in (2, 5):
            batch_size = [2, length][-batch_dims:]
            samples.append(
                TensorDict(
                    {
                        "x": torch.randn(*batch_size, 4),
                        ("n", "y"): torch.randn(*batch_size, 3),
                    },
                    batch_size=batch_size,
                )
            )
        padded = pad_sequence(
            samples, pad_dim=-2, return_mask=True, padding_side=padding_side
        )
        expected = pad_sequence(
            samples,
            pad_dim=batch_dims - 1,
            return_mask=True,
            padding_side=padding_side,
        )
        assert padded.batch_size == expected.batch_size == (2, *[2, 5][-batch_dims:])
        assert (padded == expected).all()
        self._assert_unpadded(unpad_sequence(padded, pad_dim=-2), samples)

    def test_pad_sequence_negative_pad_dim_mixed_ranks(self):
        # https://github.com/pytorch/tensordict/issues/2938
        # pad_dim=-1 is the batch dim of "done" and a feature dim of "x": with
        # equal batch sizes, only "x" needs padding
        samples = [
            TensorDict(
                {"done": torch.ones(3, dtype=torch.bool), "x": torch.ones(3, n)}, [3]
            )
            for n in (2, 5)
        ]
        padded = pad_sequence(samples, pad_dim=-1, return_mask=True)
        assert padded.batch_size == (2, 3)
        assert (padded["done"] == 1).all()
        assert padded["x"].shape == (2, 3, 5)
        assert padded["masks", "done"].shape == (2, 3)
        assert padded["masks", "x"][0, 0].tolist() == [True] * 2 + [False] * 3
        self._assert_unpadded(unpad_sequence(padded, pad_dim=-1), samples)
        # pad_dim=-2 is batch dim 0 of "a" and batch dim 1 of "b"
        samples = [
            TensorDict({"a": torch.randn(3, 3), "b": torch.randn(3, 3, 4)}, [3, 3])
            for _ in range(2)
        ]
        padded = pad_sequence(samples, pad_dim=-2)
        assert (padded == torch.stack(samples)).all()
        # with different batch sizes, no padding is consistent with every entry
        samples = [
            TensorDict({"done": torch.ones(n), "x": torch.ones(n, 4)}, [n])
            for n in (2, 5)
        ]
        with pytest.raises(ValueError, match="Pass a non-negative pad_dim"):
            pad_sequence(samples, pad_dim=-1)
        assert pad_sequence(samples, pad_dim=0).batch_size == (2, 5)
        with pytest.raises(IndexError, match="pad_dim=-2 is out of range"):
            pad_sequence(samples, pad_dim=-2)

    def test_pad_sequence_masks_padded_once(self):
        # https://github.com/pytorch/tensordict/issues/2939
        samples = [
            TensorDict({"a": torch.ones(n), ("b", "c"): torch.ones(n, 2)}, [n])
            for n in range(1, 6)
        ]
        with mock.patch(
            "torch.nn.utils.rnn.pad_sequence", wraps=torch.nn.utils.rnn.pad_sequence
        ) as padder:
            padded = pad_sequence(samples, return_mask=True)
        # one call per entry and one per mask, whatever the number of samples
        assert padder.call_count == 4
        assert padded["masks", "b", "c"].tolist() == [
            [True] * n + [False] * (5 - n) for n in range(1, 6)
        ]

    def test_pad_sequence_padding_side(self):
        # https://github.com/pytorch/tensordict/issues/2808
        d1 = TensorDict(
            {"a": torch.ones(2, 3), ("b", "c"): torch.ones(2), "s": "x"}, [2]
        )
        d2 = TensorDict(
            {"a": torch.full((4, 3), 2.0), ("b", "c"): torch.full((4,), 2.0)}, [4]
        )
        d2["s"] = "y"
        right = pad_sequence([d1, d2], return_mask=True)
        assert (
            pad_sequence([d1, d2], return_mask=True, padding_side="right") == right
        ).all()
        left = pad_sequence([d1, d2], return_mask=True, padding_side="left")
        assert left.batch_size == torch.Size([2, 4])
        assert left["b", "c"].tolist() == [[0, 0, 1, 1], [2, 2, 2, 2]]
        assert (left["a"][0, :2] == 0).all() and (left["a"][0, 2:] == 1).all()
        assert left["masks", "a"].tolist() == [[False, False, True, True], [True] * 4]
        assert left["masks", "b", "c"].tolist() == left["masks", "a"].tolist()
        assert left.get("s").tolist() == [[None, None, "x", "x"], ["y"] * 4]
        # pad dim after the batch dims
        left = pad_sequence(
            [TensorDict(a=torch.ones(2, n), batch_size=[2]) for n in (1, 3)],
            pad_dim=1,
            return_mask=True,
            padding_side="left",
        )
        assert left.batch_size == torch.Size([2, 2])
        assert left["a"][0].tolist() == [[0, 0, 1], [0, 0, 1]]
        assert left["masks", "a"][0].tolist() == [[False, False, True]] * 2
        with pytest.raises(ValueError, match="padding_side must be"):
            pad_sequence([d1, d2], padding_side="top")

    @staticmethod
    def _assert_unpadded(samples, tds):
        assert isinstance(samples, LazyStackedTensorDict)
        assert samples.stack_dim == 0
        assert len(samples.tensordicts) == len(tds)
        expected = lazy_stack(tds)
        assert set(samples.keys(True, True)) == set(expected.keys(True, True))
        for sample, td in zip(samples.tensordicts, tds):
            assert sample.batch_size == td.batch_size
            for key in td.keys(True, True, is_leaf=is_leaf_nontensor):
                value = td.get(key)
                if is_non_tensor(value):
                    assert sample.get(key).tolist() == value.tolist()
                else:
                    assert torch.equal(sample.get(key), value), key

    @pytest.mark.parametrize("padding_side", ["right", "left"])
    @pytest.mark.parametrize("pad_dim", [0, 1, -1])
    def test_unpad_sequence(self, padding_side, pad_dim):
        # https://github.com/pytorch/tensordict/issues/2808
        if pad_dim == 0:
            # the batch dim is padded: non-tensor entries are padded too
            tds = [
                TensorDict(
                    {"a": torch.randn(n, 3), ("b", "c"): torch.randn(n), "s": "x"},
                    [n],
                )
                for n in (2, 5, 3)
            ]
        elif pad_dim == 1:
            tds = [
                TensorDict(
                    {
                        "a": torch.randn(4, n, 3),
                        ("b", "c"): torch.randn(4, n),
                        "s": "x",
                    },
                    [4],
                )
                for n in (2, 5, 3)
            ]
        else:
            tds = [
                TensorDict(
                    {"a": torch.randn(3, n), ("b", "c"): torch.randn(2, n), "s": "x"}
                )
                for n in (2, 5, 3)
            ]
        padded = pad_sequence(
            tds, pad_dim=pad_dim, return_mask=True, padding_side=padding_side
        )
        samples = unpad_sequence(padded, pad_dim=pad_dim)
        assert "masks" not in samples.keys()
        self._assert_unpadded(samples, tds)
        # the masks can be passed: everything in padded is unpadded
        samples = unpad_sequence(
            padded.exclude("masks"), padded["masks"], pad_dim=pad_dim
        )
        self._assert_unpadded(samples, tds)

    def test_unpad_sequence_mask(self):
        # https://github.com/pytorch/tensordict/issues/2808
        tds = [
            TensorDict({"a": torch.randn(n, 3), "b": torch.randn(n)}) for n in (2, 4)
        ]
        padded = pad_sequence(tds, return_mask="valid", padding_side="left")
        with pytest.raises(ValueError, match="needs a mask"):
            unpad_sequence(padded)
        mask = padded["valid"]
        padded = padded.exclude("valid")
        self._assert_unpadded(unpad_sequence(padded, mask), tds)
        # one tensor mask for every entry
        self._assert_unpadded(unpad_sequence(padded, mask["b"]), tds)
        with pytest.raises(ValueError, match="shape"):
            unpad_sequence(padded, mask["b"][0])
        with pytest.raises(ValueError, match="shape"):
            unpad_sequence(padded, mask["b"][:1])
        # a second dim that is not the padded length
        with pytest.raises(ValueError, match="shape"):
            unpad_sequence(padded, mask["b"][:, 1:])
        with pytest.raises(ValueError, match="shape"):
            unpad_sequence(padded, torch.cat([mask["b"], mask["b"]], -1))
        # the same when the batch was padded
        batch_padded = pad_sequence(
            [TensorDict(a=torch.randn(n, 3), batch_size=[n]) for n in (2, 4)]
        )
        with pytest.raises(ValueError, match="shape"):
            unpad_sequence(batch_padded, mask["b"][:, 1:])
        with pytest.raises(ValueError, match="shape"):
            unpad_sequence(batch_padded, torch.cat([mask["b"], mask["b"]], -1))
        with pytest.raises(ValueError, match="batch dimension"):
            unpad_sequence(padded[0], mask)
        with pytest.raises(KeyError, match="'a'"):
            unpad_sequence(padded, mask.exclude("a"))
        # marked positions that are not one run per row
        holes = torch.tensor([[True, False, True, False], [True] * 4])
        with pytest.raises(ValueError, match="contiguous"):
            unpad_sequence(padded, holes)
        with pytest.raises(ValueError, match="contiguous"):
            unpad_sequence(padded, TensorDict(a=holes, b=holes, batch_size=[2]))
        # a lazy stack is not changed
        stack = lazy_stack(list(padded.unbind(0)))
        self._assert_unpadded(unpad_sequence(stack, mask), tds)
        assert stack.tensordicts[0]["a"].shape == torch.Size([4, 3])

    @pytest.mark.parametrize("batch_size", [[], [0]])
    def test_unpad_sequence_empty(self, batch_size):
        # https://github.com/pytorch/tensordict/issues/2808
        # every sample has length 0 along pad_dim
        tds = [TensorDict(a=torch.zeros(0, 3), batch_size=batch_size)] * 2
        samples = unpad_sequence(pad_sequence(tds, return_mask=True))
        self._assert_unpadded(samples, tds)

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
    @legacy_lazy_mode()
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
    @legacy_lazy_mode()
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

    def test_atleast_1d_context_manager_keeps_empty_batch(self):
        # 0-dim input: values written inside the block come back, batch stays [].
        td = TensorDict({"a": torch.tensor(3.0)}, batch_size=[])
        with td.atleast_1d() as expanded:
            assert expanded.batch_size == torch.Size([1])
            expanded["a"] = torch.tensor([5.0])
        assert td.batch_size == torch.Size([])
        assert td["a"].shape == torch.Size([])
        assert td["a"].item() == 5.0

    @pytest.mark.parametrize("nd", [2, 3])
    def test_atleast_nd_context_manager_keeps_batch(self, nd):
        # 1-dim input: exit is clean, the original batch is unchanged, and the
        # in-block mutation is visible on the original.
        td = TensorDict({"a": torch.arange(3)}, batch_size=[3])
        expanded_batch = torch.Size((1,) * (nd - 1) + (3,))
        with getattr(td, f"atleast_{nd}d")() as expanded:
            assert expanded.batch_size == expanded_batch
            expanded["a"] = (torch.arange(3) + 1).reshape(expanded_batch)
        assert td.batch_size == torch.Size([3])
        assert td["a"].shape == torch.Size([3])
        assert torch.equal(td["a"], torch.arange(3) + 1)

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

    def test_batch_dims_setter_raises(self):
        td = TensorDict(batch_size=[3, 2])
        lazy = lazy_stack([td, td.clone()])
        for value in (td, lazy):
            # Python 3.10 says "can't set attribute", later versions "has no setter"
            with pytest.raises(
                AttributeError, match="has no setter|can't set attribute"
            ):
                value.batch_dims = 1
        assert td.batch_dims == 2

    def test_subtd_lock_unlock_raise(self):
        # A _SubTensorDict has the lock of its source: lock_ and unlock_ raise
        # when they would change it, and return the view otherwise.
        source = TensorDict(a=torch.zeros(3, 2), batch_size=[3, 2])
        sub = _SubTensorDict(source, (slice(0, 2),))
        with pytest.raises(
            RuntimeError,
            match="^Cannot lock a _SubTensorDict. Lock the parent tensordict instead.$",
        ):
            sub.lock_()
        assert not sub.is_locked
        assert not source.is_locked
        assert sub.unlock_() is sub
        source.lock_()
        assert sub.is_locked
        assert sub.lock_() is sub
        with pytest.raises(
            RuntimeError,
            match="^Cannot unlock a _SubTensorDict. Unlock the parent tensordict "
            "instead.$",
        ):
            sub.unlock_()
        with pytest.raises(RuntimeError, match="Unlock the parent tensordict"):
            sub.is_locked = False
        assert sub.is_locked
        assert source.is_locked
        source.unlock_()
        assert not sub.is_locked

    def test_unlock_holding_subtd(self):
        # unlock_ of a tensordict that holds a _SubTensorDict leaves the view,
        # whose lock is its source's, alone
        source = TensorDict(a=torch.zeros(4, 3), batch_size=[4, 3])
        td = TensorDict(batch_size=[2])
        td.set("s", _SubTensorDict(source, (slice(0, 2),)))
        for locked in (False, True):
            if locked:
                source.lock_()
            assert td.unlock_() is td
            assert not td.is_locked
            assert td.get("s").is_locked is locked
            with pytest.raises(RuntimeError, match="Cannot lock a _SubTensorDict"):
                td.lock_()

    def test_typed_unlock_in_locked_graph_raises(self):
        source = TensorDict(
            a=torch.zeros(3, 5),
            b=torch.zeros(3, 10),
            c=torch.zeros(3, 3, dtype=torch.long),
            batch_size=[3],
        )
        typed = _TypedTD.from_tensordict(source)
        # lock_ and unlock_ reach the source
        assert typed.lock_() is typed
        assert source.is_locked
        assert typed.unlock_() is typed
        assert not source.is_locked
        parent = TensorDict(s=source, batch_size=[3]).lock_()
        assert parent.get("s") is source
        assert typed.is_locked
        with pytest.raises(RuntimeError, match="part of a locked graph"):
            typed.unlock_()
        assert source.is_locked
        assert typed.is_locked

    @pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
    def test_view_state_mirrors_source(self, tmp_path):
        # _SubTensorDict and TypedTensorDict report the state of their source
        data = TensorDict(
            a=torch.zeros(3, 5),
            b=torch.zeros(3, 10),
            c=torch.zeros(3, 3, dtype=torch.long),
            batch_size=[3],
            device="cpu",
        )
        PersistentTensorDict.from_dict(data, filename=tmp_path / "file.h5").close()
        source = PersistentTensorDict(
            filename=tmp_path / "file.h5", mode="r", batch_size=[3], device="cpu"
        )
        source.names = ["n"]
        views = [
            _SubTensorDict(source, (slice(0, 2),)),
            _TypedTD.from_tensordict(source),
        ]
        for view in views:
            assert view.device == torch.device("cpu")
            assert view._has_names()
            assert view._storage_is_read_only()
            assert not view.is_locked
            assert not view._is_shared
            assert not view._is_memmap
            assert not view._has_non_tensor
            with pytest.raises(RuntimeError, match="device cannot be set"):
                view.device = "cpu"
        source.lock_()
        assert all(view.is_locked for view in views)
        source.unlock_()
        assert not any(view.is_locked for view in views)
        source.close()

        source = data.clone().share_memory_()
        source.unlock_()
        source._is_shared = True
        source.set_non_tensor("s", "a string")
        views = [
            _SubTensorDict(source, (slice(0, 2),)),
            _TypedTD.from_tensordict(source),
        ]
        for view in views:
            assert view._is_shared
            assert not view._is_memmap
            assert view._has_non_tensor
            assert not view._storage_is_read_only()

    @pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
    def test_persistent_names_reach_cached_nested(self, tmp_path):
        data = TensorDict(
            a=torch.zeros(3, 2),
            nested=TensorDict(x=torch.zeros(3, 2, 4), batch_size=[3, 2]),
            batch_size=[3],
        )
        h5 = PersistentTensorDict.from_dict(data, filename=tmp_path / "file.h5")
        nested = h5.get("nested")
        h5.names = ["n"]
        assert nested.names == ["n", None]
        assert h5.get("nested").names == ["n", None]
        h5.rename_(None)
        assert nested.names == [None, None]
        h5.close()

    def test_rename_key_nested(self):
        td = TensorDict(a={"b": {"c": 0}})
        td.rename_key_(("a", "b", "c"), ("a", "b"))
        assert td["a", "b"] == 0

    @staticmethod
    def _rename_keys_td():
        return TensorDict(
            a=torch.zeros(3),
            b=torch.ones(3),
            raw=TensorDict(
                p=torch.full((3,), 2.0), q=torch.full((3,), 3.0), batch_size=[3]
            ),
            s=NonTensorData("hi", batch_size=[3]),
            batch_size=[3],
            names=["n"],
        )

    def test_rename_keys_swap_and_chain(self):
        td = self._rename_keys_td()
        keys = set(td.keys(True))
        out = td.rename_keys({"a": "b", "b": "a"})
        assert type(out) is TensorDict
        assert out["a"] is td["b"]
        assert out["b"] is td["a"]
        # non-tensor data is shared, not copied
        assert out.get("s") is td.get("s")
        # a new tree of sub-tensordicts
        assert out.get("raw") is not td.get("raw")
        assert out["raw", "p"] is td["raw", "p"]
        assert out.batch_size == td.batch_size
        assert out.names == ["n"]
        assert set(td.keys(True)) == keys
        out = td.rename_keys({"a": "b", "b": "c"})
        assert out["b"] is td["a"]
        assert out["c"] is td["b"]
        assert "a" not in out.keys()
        assert set(td.keys(True)) == keys

    def test_rename_keys_nested(self):
        td = self._rename_keys_td()
        # emptied sub-tensordicts are removed
        out = td.rename_keys({("raw", "p"): "p", ("raw", "q"): "q"})
        assert set(out.keys(True)) == {"a", "b", "p", "q", "s"}
        # a leaf can move onto its parent once the parent is emptied
        out = td.rename_keys({("raw", "p"): "raw", ("raw", "q"): "q"})
        assert out["raw"] is td["raw", "p"]
        # a whole sub-tensordict
        out = td.rename_keys({"raw": ("x", "solution")})
        assert set(out.keys(True, True, is_leaf=is_leaf_nontensor)) == {
            "a",
            "b",
            "s",
            ("x", "solution", "p"),
            ("x", "solution", "q"),
        }
        # into the source's own sub-key
        out = td.rename_keys({"a": ("a", "b")})
        assert out["a", "b"] is td["a"]
        # empty sub-tensordicts that the move did not empty are kept
        td["empty"] = TensorDict(batch_size=[3])
        td["raw", "e"] = TensorDict(batch_size=[3])
        out = td.rename_keys({("raw", "p"): "p", ("raw", "q"): "q"})
        assert set(out.keys(True)) == {
            "a",
            "b",
            "p",
            "q",
            "s",
            "empty",
            "raw",
            ("raw", "e"),
        }

    @pytest.mark.parametrize(
        "mapping,error,match",
        [
            ({"a": "b"}, ValueError, "already present"),
            ({("raw", "p"): "raw"}, ValueError, "already present"),
            ({"a": ("b", "x")}, ValueError, "not a tensordict"),
            ({"a": ("s", "x")}, ValueError, "not a tensordict"),
            ({"a": "x", "b": "x"}, ValueError, "destination more than once"),
            ({"a": "x", ("a",): "y"}, ValueError, "source more than once"),
            ({"raw": "r", ("raw", "p"): "p"}, ValueError, "lie inside one another"),
            ({"a": ("x", "y"), "b": "x"}, ValueError, "lie inside one another"),
            ({"a": "a", "b": "a"}, ValueError, "destination more than once"),
            ({"zz": "y"}, KeyError, "zz"),
            ({"a": "a", "zz": "zz"}, KeyError, "zz"),
            ({"a": ()}, KeyError, "NestedKeys"),
            ({1: "x"}, TypeError, "string or a tuple"),
        ],
    )
    def test_rename_keys_errors(self, mapping, error, match):
        td = self._rename_keys_td()
        keys = set(td.keys(True))
        with pytest.raises(error, match=match):
            td.rename_keys(mapping)
        assert set(td.keys(True)) == keys

    def test_rename_keys_strict(self):
        td = self._rename_keys_td()
        out = td.rename_keys({"zz": "y", ("raw", "p"): "p"}, strict=False)
        assert "y" not in out.keys()
        assert out["p"] is td["raw", "p"]
        assert td.rename_keys({"a": "a"})["a"] is td["a"]

    def test_rename_keys_docstring(self):
        td = TensorDict(a=torch.zeros(()), b=torch.ones(()))
        out = td.rename_keys({"a": "b", "b": "a"})
        assert out["a"] == 1 and td["a"] == 0
        td = TensorDict(raw=TensorDict(p=torch.zeros(3), u=torch.ones(3)))
        out = td.rename_keys({("raw", "p"): "pressure", ("raw", "u"): "velocity"})
        assert sorted(out.keys(include_nested=True)) == ["pressure", "velocity"]
        out = td.rename_keys({"raw": "solution"})
        assert sorted(out.keys(include_nested=True, leaves_only=True)) == [
            ("solution", "p"),
            ("solution", "u"),
        ]
        out = td.rename_keys({("raw", "zz"): "z", ("raw", "p"): "p"}, strict=False)
        assert (out["p"] == 0).all()

    def test_rename_keys_locked(self):
        td = self._rename_keys_td().lock_()
        out = td.rename_keys({"a": "z"})
        assert not out.is_locked
        assert td.is_locked
        assert "a" in td.keys()
        assert out["z"] is td["a"]

    def test_rename_keys_lazy_stack(self):
        t1 = TensorDict(
            a=torch.zeros(()), x=torch.zeros(()), raw=TensorDict(p=torch.zeros(()))
        )
        t2 = TensorDict(
            a=torch.ones(()),
            y=torch.ones(()),
            raw=TensorDict(p=torch.zeros(()), q=torch.zeros(())),
        )
        ls = lazy_stack([t1, t2])
        before = [set(td.keys(True)) for td in ls.tensordicts]
        out = ls.rename_keys({"a": "b"})
        assert isinstance(out, LazyStackedTensorDict)
        assert out.tensordicts[0]["b"] is t1["a"]
        assert out.tensordicts[1]["b"] is t2["a"]
        # an exclusive key is missing in some members
        with pytest.raises(KeyError, match="x"):
            ls.rename_keys({"x": "z"})
        out = ls.rename_keys({"x": "z"}, strict=False)
        assert set(out.tensordicts[0].keys()) == {"a", "z", "raw"}
        assert set(out.tensordicts[1].keys()) == {"a", "y", "raw"}
        # raw is emptied in the first member only: it is kept
        out = ls.rename_keys({("raw", "p"): "p"})
        assert set(out.tensordicts[0].keys(True)) == {"a", "p", "x", "raw"}
        assert set(out.tensordicts[1].keys(True)) == {
            "a",
            "p",
            "y",
            "raw",
            ("raw", "q"),
        }
        # raw is emptied in every member: it is removed
        out = ls.rename_keys({("raw", "p"): "p", ("raw", "q"): "q"}, strict=False)
        assert set(out.tensordicts[0].keys(True)) == {"a", "p", "x"}
        assert set(out.tensordicts[1].keys(True)) == {"a", "p", "q", "y"}
        assert [set(td.keys(True)) for td in ls.tensordicts] == before

    def test_rename_keys_params(self):
        params = TensorDictParams(
            TensorDict(
                w=nn.Parameter(torch.zeros(2)),
                sub=TensorDict(v=nn.Parameter(torch.zeros(2))),
            )
        )
        out = params.rename_keys({"w": "w2", ("sub", "v"): "v"})
        # a TensorDict, as for select and exclude, so that it compiles
        assert type(out) is TensorDict
        assert set(out.keys(True)) == {"w2", "v"}
        assert out.get("w2") is params.get("w")
        assert isinstance(out.get("v"), nn.Parameter)
        assert set(params.keys(True)) == {"w", "sub", ("sub", "v")}

    @pytest.mark.skipif(not _has_h5py, reason="h5py needed for persistent tensordicts")
    def test_rename_keys_persistent(self, tmp_path):
        td = PersistentTensorDict.from_dict(
            TensorDict(a=torch.zeros(3), batch_size=[3]), tmp_path / "file.h5"
        )
        with pytest.raises(NotImplementedError, match="to_tensordict"):
            td.rename_keys({"a": "b"})
        assert set(td.keys()) == {"a"}
        td.close()

    @pytest.mark.parametrize("num_threads", [0, 2])
    def test_memmap_like_no_prefix_new_files(self, tmpdir, num_threads):
        # Without a prefix, memmap_like must not reuse the files of a source
        # memory-mapped under a prefix.
        td = TensorDict(
            a=torch.ones(3), nested=TensorDict(b=torch.ones(3, 2), batch_size=[3])
        ).memmap(tmpdir)
        like = td.memmap_like(num_threads=num_threads)
        assert like.is_memmap()
        assert (like == 0).all()
        like["a"].fill_(5)
        like["nested", "b"].fill_(5)
        assert (td == 1).all()
        assert (TensorDict.load_memmap(tmpdir) == 1).all()

    @pytest.mark.parametrize("method", ["memmap", "memmap_", "memmap_like"])
    @pytest.mark.parametrize("return_early", [False, True])
    def test_memmap_threads_raise_worker_error(
        self, tmpdir, monkeypatch, method, return_early
    ):
        # An error in a worker thread is raised, as with num_threads=0
        populate_memmap = tensordict._td._populate_memmap

        def failing_populate_memmap(*args, key, **kwargs):
            if key == "b":
                raise OSError("failed to write b")
            return populate_memmap(*args, key=key, **kwargs)

        monkeypatch.setattr(tensordict._td, "_populate_memmap", failing_populate_memmap)
        td = TensorDict(a=torch.zeros(3), b=torch.ones(3))
        with pytest.raises(OSError, match="failed to write b"):
            out = getattr(td, method)(tmpdir, num_threads=2, return_early=return_early)
            if return_early:
                out.result()

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

    @pytest.mark.parametrize(
        "key", [("x", "q"), ("x", "q", "r"), ("n", "y", "q")], ids=str
    )
    @pytest.mark.parametrize(
        "container",
        ["td", "lazy", "tensorclass", "td_of_lazy", "td_of_tensorclass"],
    )
    def test_get_pop_default_through_leaf(self, container, key):
        # With a default, get and pop return it for a key that runs through a
        # leaf; without one, get, pop and indexing raise the error that names
        # the leaf (#2678)
        @tensorclass
        class Leaves:
            x: torch.Tensor
            n: TensorDict

        td = TensorDict(
            x=torch.zeros(3),
            n=TensorDict(y=torch.zeros(3), batch_size=[3]),
            batch_size=[3],
        )
        if container.endswith("lazy"):
            td = lazy_stack(list(td.unbind(0)))
        elif container.endswith("tensorclass"):
            td = Leaves(x=td["x"], n=td["n"], batch_size=[3])
        if container.startswith("td_of"):
            td = TensorDict(t=td, batch_size=[3])
            key = ("t", *key)
        keys = set(td.keys(True))
        assert td.get(key, None) is None
        assert td.get(key, default="D") == "D"
        assert td.pop(key, "D") == "D"
        assert td.pop(key, default=None) is None
        assert set(td.keys(True)) == keys
        reads = [lambda: td.pop(key)]
        if container != "tensorclass":
            # a tensorclass is indexed by batch index only
            reads.append(lambda: td[key])
        if container != "tensorclass" or key[0] != "x":
            # a tensorclass returns None for a field that is a tensor
            reads.append(lambda: td.get(key))
        match = f"'{'x' if 'x' in key else 'y'}' is a Tensor, not a tensordict"
        for read in reads:
            with pytest.raises(ValueError, match=match):
                read()
        assert set(td.keys(True)) == keys

    @pytest.mark.parametrize("inplace", [True, False])
    def test_select_exclude_nested_through_leaf(self, inplace):
        # A nested key through a leaf is missing, as ("x", "q") in td says (#2617)
        def make():
            return TensorDict(x=torch.zeros(3), n=TensorDict(y=torch.zeros(3)))

        td = make()
        with pytest.raises(KeyError, match="'x' is a Tensor, not a tensordict"):
            td.select(("x", "q"), inplace=inplace)
        with pytest.raises(KeyError, match="'y' is a Tensor, not a tensordict"):
            td.select(("n", "y", "q"), inplace=inplace)
        assert set(td.keys(True)) == {"x", "n", ("n", "y")}
        out = make().select(("x", "q"), "n", strict=False, inplace=inplace)
        assert set(out.keys(True)) == {"n", ("n", "y")}
        out = make().select("x", ("x", "q"), strict=False, inplace=inplace)
        assert set(out.keys(True)) == {"x"}
        out = make().select(("n", "y"), ("n", "y", "q"), strict=False, inplace=inplace)
        assert set(out.keys(True)) == {"n", ("n", "y")}
        td = make()
        out = td.exclude(("x", "q"), ("n", "y", "q"), inplace=inplace)
        assert set(out.keys(True)) == {"x", "n", ("n", "y")}
        assert out["x"] is td["x"]

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

    @pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "value",
        [
            (1.0, 2.0),
            (1, 2.0),
            ((1.0, 2.0), (3.0, 4.0)),
            (1j, 2.0),
            (),
            (1, 2),
            (True, False),
            (_Pair(1.0, 2.0), _Pair(3.0, 4.0)),
        ],
    )
    def test_set_tuple_of_numbers_dtype(self, value, default_dtype):
        # A tuple of numbers keeps NumPy's dtype until 0.16, with a
        # FutureWarning where torch.tensor gives another dtype (#2679)
        old = torch.as_tensor(np.asarray(value)).dtype
        default = torch.get_default_dtype()
        torch.set_default_dtype(default_dtype)
        try:
            new = torch.tensor(value).dtype
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                td = TensorDict(a=value)
                td["b"] = value
        finally:
            torch.set_default_dtype(default)
        assert td["a"].dtype == td["b"].dtype == old
        assert (td["a"] == torch.tensor(value)).all()
        record = [w for w in record if issubclass(w.category, FutureWarning)]
        if default_dtype == torch.float64 or old in (torch.int64, torch.bool):
            assert old == new
            assert not record
        else:
            assert old != new
            assert len(record) == 2
            for w in record:
                assert w.filename == __file__
                assert (
                    f"stored as a {old} tensor today. Starting with TensorDict "
                    f"0.16, the default will change to the tensor that "
                    f"torch.tensor(value) gives ({new})"
                ) in str(w.message)

    def test_set_tuple_no_warning(self):
        # Tuples that do not hold only Python numbers, and lists, do not warn (#2679)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            td = TensorDict(
                a=("a", "b"),
                b=(torch.ones(2), torch.ones(2)),
                c=(np.float32(1.0), np.float32(2.0)),
                d=[1.0, 2.0],
                e=np.array([1.0, 2.0]),
                f=(np.float64(1.0), 2.0),
                g=(np.float32(1.0), 2.0),
                h=(np.int64(1), np.uint64(1)),
            )
        assert td["b"].dtype == td["c"].dtype == td["d"].dtype == torch.float32
        for key in "efgh":
            assert td[key].dtype == torch.float64
        # torch has no complex dtype for a bfloat16 default: store as on main
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            td["i"] = (1j, 2.0)
        finally:
            torch.set_default_dtype(default)
        assert td["i"].dtype == torch.complex128

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

    @pytest.mark.parametrize(
        "idx",
        [slice(1, 3), torch.tensor([0, 2]), torch.tensor([True, False, True, False])],
    )
    @pytest.mark.parametrize("lazy", [False, True])
    def test_sub_td_inplace_methods(self, idx, lazy):
        # The entries of a sub-tensordict are copies for an index tensor, a
        # mask or a lazy stack source: the in-place methods write them back
        td = TensorDict(
            a=torch.arange(8.0).view(4, 2),
            nested=TensorDict(b=torch.arange(4.0), batch_size=[4]),
            batch_size=[4],
        )
        expected = td.clone()
        if lazy:
            td = lazy_stack(list(td.unbind(0)))
        sub_td = td._get_sub_tensordict(idx)
        assert sub_td.add_(1.0) is sub_td
        sub_td *= 2
        assert sub_td.clamp_max_(9.0) is sub_td
        expected[idx] = expected[idx].add(1.0).mul(2).clamp_max(9.0)
        assert (td.to_tensordict() == expected).all()

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
        with legacy_lazy_mode() if lazy_leg else set_lazy_legacy(False):
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

            # Indexing propagates the shared status if it gives a view
            td0 = td[0]
            assert_shared(td0)

            # torch reads a bool as a 0-d mask, which copies
            for index in (True, False, (0, True), torch.tensor(True)):
                td0 = td[index]
                assert_not_shared(td0)

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

    @staticmethod
    def _split_keys_rollout():
        return TensorDict(
            observation=torch.randn(4, 3),
            action=torch.randn(4, 2),
            done=torch.zeros(4, 1, dtype=torch.bool),
            step_count=torch.zeros(4, 1, dtype=torch.int64),
            collector=TensorDict(traj_ids=torch.zeros(4, dtype=torch.int64)),
            next=TensorDict(
                observation=torch.randn(4, 3),
                reward=torch.randn(4, 1),
                done=torch.zeros(4, 1, dtype=torch.bool),
            ),
            info=NonTensorData("a string"),
            batch_size=[4],
        )

    def test_split_keys_callable(self):
        td = self._split_keys_rollout()
        seen = []

        def is_float(key, val):
            seen.append(key)
            assert isinstance(key, tuple)
            assert isinstance(val, torch.Tensor)
            return val.is_floating_point()

        floats, rest = td.split_keys(is_float)
        # tensor leaves only, keys always tuples
        assert set(seen) == {
            ("observation",),
            ("action",),
            ("done",),
            ("step_count",),
            ("collector", "traj_ids"),
            ("next", "observation"),
            ("next", "reward"),
            ("next", "done"),
        }
        assert set(floats.keys(True, True)) == {
            "observation",
            "action",
            ("next", "observation"),
            ("next", "reward"),
        }
        assert set(rest.keys(True, True)) == {
            "done",
            "step_count",
            ("collector", "traj_ids"),
            ("next", "done"),
        }
        assert rest["info"] == "a string"
        assert "info" not in floats.keys()
        assert floats["observation"].data_ptr() == td["observation"].data_ptr()
        assert floats.batch_size == td.batch_size
        assert "observation" in td

    def test_split_keys_callable_first_match_wins(self):
        td = self._split_keys_rollout()
        nxt, done, floats, rest = td.split_keys(
            lambda k, v: k[0] == "next",
            lambda k, v: k[-1] == "done",
            lambda k, v: v.is_floating_point(),
        )
        assert set(nxt.keys(True, True)) == {
            ("next", "observation"),
            ("next", "reward"),
            ("next", "done"),
        }
        assert set(done.keys(True, True)) == {"done"}
        assert set(floats.keys(True, True)) == {"observation", "action"}
        assert set(rest.keys(True, True)) == {
            "step_count",
            ("collector", "traj_ids"),
        }
        assert "info" in rest.keys()

    def test_split_keys_callable_rank(self):
        td = TensorDict(
            s1=torch.randn(10),
            s2=torch.randn(10),
            v=torch.randn(10, 3),
            m=torch.randn(10, 4, 4),
            batch_size=[10],
        )
        nb = td.batch_dims
        scalars, vectors, rest = td.split_keys(
            lambda k, v: v.ndim == nb, lambda k, v: v.ndim == nb + 1
        )
        assert set(scalars.keys()) == {"s1", "s2"}
        assert set(vectors.keys()) == {"v"}
        assert set(rest.keys()) == {"m"}

    @pytest.mark.parametrize("reproduce_struct", [False, True])
    def test_split_keys_callable_mixed(self, reproduce_struct):
        td = self._split_keys_rollout()
        collector, bools, rest = td.split_keys(
            ["collector"],
            lambda k, v: v.dtype == torch.bool,
            reproduce_struct=reproduce_struct,
        )
        assert set(collector.keys(True, True)) == {("collector", "traj_ids")}
        assert set(bools.keys(True, True)) == {"done", ("next", "done")}
        assert "collector" not in rest
        assert "done" not in rest
        assert ("next", "done") not in rest
        assert ("next", "reward") in rest
        if reproduce_struct:
            assert "next" in collector.keys()
            assert "collector" in bools.keys()
        else:
            assert "next" not in collector.keys()
            assert "collector" not in bools.keys()

    def test_split_keys_callable_inplace(self):
        td = self._split_keys_rollout()
        bools, rest = td.split_keys(lambda k, v: v.dtype == torch.bool, inplace=True)
        assert rest is td
        assert set(bools.keys(True, True)) == {"done", ("next", "done")}
        assert "done" not in td
        assert ("next", "done") not in td
        assert ("next", "reward") in td

    def test_split_keys_callable_lazy_stack(self):
        td = lazy_stack(
            [
                TensorDict(a=torch.zeros(3), b=torch.zeros(3, dtype=torch.long)),
                TensorDict(a=torch.ones(3), b=torch.ones(3, dtype=torch.long)),
            ]
        )
        floats, rest = td.split_keys(lambda k, v: v.is_floating_point())
        assert isinstance(floats, LazyStackedTensorDict)
        assert set(floats.keys()) == {"a"}
        assert set(rest.keys()) == {"b"}
        assert (floats["a"] == td["a"]).all()

    def test_split_keys_callable_tensorclass(self):
        class TC(TensorClass):
            x: torch.Tensor
            y: torch.Tensor

        tc = TC(x=torch.zeros(3), y=torch.zeros(3, dtype=torch.long), batch_size=[3])
        floats, rest = tc.split_keys(lambda k, v: v.is_floating_point())
        assert isinstance(floats, TC)
        assert isinstance(rest, TC)
        assert floats.x is not None and floats.y is None
        assert rest.x is None and rest.y is not None

    def test_split_keys_enum_class(self):
        # An Enum class is callable but also iterable: it is a list of keys.
        class K(str, enum.Enum):
            a = "a"

        td = TensorDict(a=torch.zeros(3), b=torch.zeros(3), batch_size=[3])
        td_a, rest = td.split_keys(K)
        assert set(td_a.keys()) == {"a"}
        assert set(rest.keys()) == {"b"}

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("reproduce_struct", [False, True])
    def test_split_keys_non_tensor(self, reproduce_struct, lazy):
        # A non-tensor entry goes only to the output whose key set names it,
        # or to the last output, with or without reproduce_struct.
        td = TensorDict(
            a=torch.zeros(3),
            b=torch.ones(3),
            info=NonTensorData("run 7", batch_size=[3]),
            n=TensorDict(
                c=torch.zeros(3),
                s=NonTensorData("label", batch_size=[3]),
                batch_size=[3],
            ),
            batch_size=[3],
        )
        if lazy:
            td = LazyStackedTensorDict(*td.unbind(0))

        def leaves(part):
            return set(part.keys(True, True, is_leaf=is_leaf_nontensor))

        td_a, td_b, rest = td.split_keys(
            ["a"], ["b"], reproduce_struct=reproduce_struct
        )
        assert leaves(td_a) == {"a"}
        assert leaves(td_b) == {"b"}
        assert leaves(rest) == {"info", ("n", "c"), ("n", "s")}
        assert rest[0]["info"] == "run 7"
        td_a, td_info, rest = td.split_keys(
            ["a"], ["info", ("n", "s")], reproduce_struct=reproduce_struct
        )
        assert leaves(td_a) == {"a"}
        assert leaves(td_info) == {"info", ("n", "s")}
        assert td_info[0]["n", "s"] == "label"
        assert leaves(rest) == {"b", ("n", "c")}
        if reproduce_struct:
            # the sub-tensordicts are kept, even without leaves
            assert "n" in td_a.keys()
            assert td_a["n"].is_empty()
        else:
            assert "n" not in td_a.keys()

    @pytest.mark.parametrize("reproduce_struct", [False, True])
    def test_split_keys_tensorclass_non_tensor(self, reproduce_struct):
        # A non-tensor tensorclass field goes only to the output whose key set
        # names it, or to the last output: the other outputs hold None.
        @tensorclass
        class MyClass:
            x: torch.Tensor
            y: torch.Tensor
            s: str

        tc = MyClass(x=torch.zeros(3), y=torch.ones(3), s="hello", batch_size=[3])
        parts = tc.split_keys(["x"], ["y"], reproduce_struct=reproduce_struct)
        assert all(isinstance(part, MyClass) for part in parts)
        assert [part.s for part in parts] == [None, None, "hello"]
        parts = tc.split_keys(["x"], ["s"], reproduce_struct=reproduce_struct)
        assert all(isinstance(part, MyClass) for part in parts)
        assert [part.s for part in parts] == [None, "hello", None]
        if not reproduce_struct:
            # the nested tensorclass is only kept with reproduce_struct=True
            return
        td = TensorDict(a=torch.zeros(3), t=tc, batch_size=[3])
        parts = td.split_keys(["a"], [("t", "x")], reproduce_struct=True)
        assert all(isinstance(part["t"], MyClass) for part in parts)
        assert [part["t"].s for part in parts] == [None, None, "hello"]
        parts = td.split_keys(["a"], [("t", "s")], reproduce_struct=True)
        assert all(isinstance(part["t"], MyClass) for part in parts)
        assert [part["t"].s for part in parts] == [None, "hello", None]

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
        with pytest.raises(TypeError, match="must be int or list of ints"):
            td.split(torch.tensor([1.0, 2.0]), 0)
        # integer sizes in a tensor or a numpy array
        for sizes in (torch.tensor([1, 2]), np.array([1, 2])):
            assert [t.shape[0] for t in td.split(sizes, 0)] == [1, 2]

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

    def test_tensor_split_invalid_tensor(self):
        td = TensorDict({"a": torch.zeros(5)}, [5])
        for indices in (
            torch.tensor([1.0]),
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([[1]]),
        ):
            with pytest.raises(ValueError, match="long tensor on the CPU"):
                td.tensor_split(indices)

    @pytest.mark.parametrize("indices", [[2], [3, 1]], ids=["asc", "desc"])
    def test_tensor_split_locked_views(self, indices):
        td = TensorDict({"a": torch.arange(5)}, [5]).lock_()
        chunks = td.tensor_split(indices)
        assert all(chunk.is_locked for chunk in chunks)
        chunks[-1]["a"].zero_()
        assert (td["a"][indices[-1] :] == 0).all()

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

    def test_reshape_view_zero_size_dim(self):
        td = TensorDict(
            {"a": torch.zeros(0, 3, 4), "b": {"c": torch.zeros(0, 3)}}, [0, 3]
        )
        ref = torch.zeros(0, 3)
        for shape in ((-1,), (-1, 3), (3, -1), (1, -1, 3)):
            for out in (td.reshape(*shape), td.view(*shape)):
                assert out.batch_size == ref.reshape(*shape).shape
                assert out["a"].shape == (*out.batch_size, 4)
                assert out["b", "c"].shape == out.batch_size
        assert TensorDict({}, [0, 3]).reshape(-1).batch_size == torch.Size([0])
        # -1 next to a 0 is ambiguous: torch raises too. The error is an
        # AssertionError on main and a RuntimeError once #2119 lands.
        with pytest.raises((AssertionError, RuntimeError), match="invalid shape"):
            td.reshape(-1, 0)

    def test_split_chunk_zero_size_dim(self):
        td = TensorDict(
            {"a": torch.zeros(0, 3, 4), "b": {"c": torch.zeros(0, 3)}}, [0, 3]
        )
        ref = torch.zeros(0, 3)
        # along dim 0, of size 0, split gives one piece and chunk gives `chunks`
        for dim in (0, 1):
            for pieces, ref_pieces in (
                (td.split(2, dim), ref.split(2, dim)),
                (td.chunk(2, dim), ref.chunk(2, dim)),
                (td.chunk(3, dim), ref.chunk(3, dim)),
            ):
                assert [p.batch_size for p in pieces] == [r.shape for r in ref_pieces]
                for piece in pieces:
                    assert piece["a"].shape == (*piece.batch_size, 4)
                    assert piece["b", "c"].shape == piece.batch_size
        # split(0) gives one empty piece and split([]) none, as in torch
        for split_size in (0, []):
            pieces = td.split(split_size, 0)
            ref_pieces = ref.split(split_size, 0)
            assert [p.batch_size for p in pieces] == [r.shape for r in ref_pieces]
            for piece in pieces:
                assert piece["a"].shape == (*piece.batch_size, 4)
        # along dim 1, of size 3, both raise as in torch
        with pytest.raises(ValueError, match="split_size must be positive"):
            td.split(0, 1)
        with pytest.raises(RuntimeError, match="Insufficient number of elements"):
            td.split([], 1)

    def test_chunk_zero_size_dim_views(self, tmp_path):
        # sub-tensordicts, legacy lazy views and h5 tensordicts have their own
        # chunk, which must give `chunks` empty pieces like torch.chunk
        big = TensorDict({"a": torch.zeros(2, 0, 4)}, [2, 0])
        cases = [(big._get_sub_tensordict(0), 0)]
        with legacy_lazy_mode():
            cases += [(big.unsqueeze(0), 2), (big.permute(1, 0), 0)]
            if _has_h5py:
                h5 = PersistentTensorDict.from_dict(big, filename=tmp_path / "file.h5")
                cases.append((h5, 1))
            for td, dim in cases:
                for chunks in (2, 3):
                    pieces = td.chunk(chunks, dim)
                    ref_pieces = torch.zeros(td.batch_size).chunk(chunks, dim)
                    assert [p.batch_size for p in pieces] == [
                        r.shape for r in ref_pieces
                    ]
                    for piece in pieces:
                        assert piece["a"].shape == (*piece.batch_size, 4)

    def test_chunk_arguments(self):
        # every class shares one chunk: same checks, and numpy integers work
        big = TensorDict({"a": torch.zeros(2, 5, 4)}, [2, 5])
        cases = [
            big[0],
            big._get_sub_tensordict(0),
            lazy_stack([big[0], big[1]]),
            lazy_stack([big[0], big[1]], 1),
        ]
        for td in cases:
            for chunks in (0, -1):
                with pytest.raises(ValueError, match="strictly positive"):
                    td.chunk(chunks)
            pieces = td.chunk(np.int64(2), -1)
            assert [p.batch_size for p in pieces] == [
                r.shape for r in torch.zeros(td.batch_size).chunk(2, -1)
            ]
            for piece in pieces:
                assert all(type(s) is int for s in piece.batch_size)

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

    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_stack_lazy_views(self, dim):
        # Sub-tensordicts and legacy lazy views are lazy but are not lazy
        # stacks: they are stacked densely
        td = TensorDict(
            {"a": torch.randn(2, 3, 4), "b": {"c": torch.randn(2, 3)}},
            batch_size=[2, 3],
        )
        sub_tds = [td._get_sub_tensordict(0), td._get_sub_tensordict(1)]
        expected = torch.stack([td[0], td[1]], dim)
        for stack in (torch.stack, TensorDict.maybe_dense_stack):
            result = stack(sub_tds, dim)
            assert type(result) is TensorDict
            assert (result == expected).all()
        with legacy_lazy_mode():
            unsqueezed = [td[0].unsqueeze(0), td[1].unsqueeze(0)]
        expected = torch.stack([td[0].unsqueeze(0), td[1].unsqueeze(0)], dim)
        assert (torch.stack(unsqueezed, dim) == expected).all()

    class _StackSlowPathTensorDict(TensorDict):
        """A TensorDict subclass: torch.stack never takes the fast path of TensorDict."""

    @staticmethod
    def _stack_outcome(cls, make, stack, dim):
        """What ``stack`` gives for the tensordicts that ``make(cls)`` builds."""
        tds = make(cls)
        try:
            result = stack(tds, dim)
        except Exception as err:
            return type(err), str(err).replace(cls.__name__, "TensorDict")
        if isinstance(result, LazyStackedTensorDict):
            return (
                type(result).__name__,
                result.stack_dim,
                [td is orig for td, orig in zip(result.tensordicts, tds)],
            )
        state = {}
        for key, leaf in result.items(True, True, is_leaf=is_leaf_nontensor):
            if is_non_tensor(leaf):
                value = leaf.tolist()
            elif leaf.is_nested:
                value = [t.tolist() for t in leaf.unbind()]
            else:
                value = (leaf.shape, leaf.tolist())
            state[key] = (type(leaf).__name__, leaf.dtype, value)
        return (
            type(result).__name__.replace(cls.__name__, "TensorDict"),
            state,
            list(state),
            result.batch_size,
            result.names if result._has_names() else None,
            result.device,
            result.is_locked,
        )

    _STACK_FIXTURES = {
        "flat": lambda cls: [
            cls(a=torch.full((3, 4), i), b=torch.arange(3) > i, batch_size=[3])
            for i in range(3)
        ],
        "nested": lambda cls: [
            cls(
                a=torch.full((3, 4), i),
                n=TensorDict(x=torch.full((3, 2), i), batch_size=[3]),
                batch_size=[3],
            )
            for i in range(3)
        ],
        "0-d": lambda cls: [cls(a=torch.ones(4), b=torch.ones(())) for _ in range(2)],
        "empty": lambda cls: [cls(batch_size=[3]) for _ in range(2)],
        "named": lambda cls: [
            cls(a=torch.zeros(3, 4), batch_size=[3], names=["t"]) for _ in range(2)
        ],
        "names differ": lambda cls: [
            cls(a=torch.zeros(3, 4), batch_size=[3], names=[name])
            for name in ("t", "u")
        ],
        "locked": lambda cls: [
            cls(a=torch.zeros(3, 4), batch_size=[3]).lock_() for _ in range(2)
        ],
        "with a device": lambda cls: [
            cls(a=torch.zeros(3, 4), batch_size=[3], device="cpu") for _ in range(2)
        ],
        "keys in another order": lambda cls: [
            # the first tensordict's keys are not sorted: the result keeps them
            cls(b=torch.ones(3), a=torch.zeros(3), batch_size=[3]),
            cls(a=torch.full((3,), 3.0), b=torch.full((3,), 2.0), batch_size=[3]),
        ],
        "exclusive keys": lambda cls: [
            cls(a=torch.zeros(3), batch_size=[3]),
            cls(b=torch.zeros(3), batch_size=[3]),
        ],
        "one key more": lambda cls: [
            cls(a=torch.zeros(3), batch_size=[3]),
            cls(a=torch.zeros(3), b=torch.zeros(3), batch_size=[3]),
        ],
        "shapes differ": lambda cls: [
            cls(a=torch.zeros(3, 4), b=torch.zeros(3), batch_size=[3]),
            cls(a=torch.zeros(3, 5), b=torch.zeros(3), batch_size=[3]),
        ],
        "dtypes differ": lambda cls: [
            cls(a=torch.zeros(3), batch_size=[3]),
            cls(a=torch.ones(3, dtype=torch.float64), batch_size=[3]),
        ],
        "parameter": lambda cls: [
            cls(a=nn.Parameter(torch.zeros(3)), b=torch.zeros(3), batch_size=[3])
            for _ in range(2)
        ],
        "parameter after a tensor": lambda cls: [
            cls(a=torch.zeros(3), batch_size=[3]),
            cls(a=nn.Parameter(torch.ones(3)), batch_size=[3]),
        ],
        "nested tensor": lambda cls: [
            cls(
                a=torch.nested.nested_tensor([torch.zeros(2), torch.ones(3)]),
                batch_size=[2],
            )
            for _ in range(2)
        ],
        "nested tensor after a tensor": lambda cls: [
            cls(a=torch.zeros(2, 3), batch_size=[2]),
            cls(
                a=torch.nested.nested_tensor([torch.zeros(3), torch.ones(3)]),
                batch_size=[2],
            ),
        ],
        "non-tensor": lambda cls: [
            cls(a=torch.zeros(3), s=NonTensorData(str(i)), batch_size=[3])
            for i in range(2)
        ],
        "unbatched": lambda cls: [
            cls(
                a=torch.zeros(3),
                u=tensordict.UnbatchedTensor(torch.ones(2)),
                batch_size=[3],
            )
            for _ in range(2)
        ],
        "tensor-collection subclass child": lambda cls: [
            cls(
                a=torch.zeros(3),
                p=TensorDictParams(TensorDict(x=torch.zeros(3, 2), batch_size=[3])),
                batch_size=[3],
            )
            for _ in range(2)
        ],
    }

    @pytest.mark.parametrize("fixture", list(_STACK_FIXTURES))
    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize(
        "stack", [torch.stack, TensorDict.maybe_dense_stack], ids=["dense", "maybe"]
    )
    def test_stack_fast_path_matches_general(self, fixture, dim, stack):
        # torch.stack of exact TensorDicts reads the keys and the dense leaves
        # from their dicts, without the per-tensordict calls of the general
        # path, which a subclass of TensorDict takes. Both give the same
        # result, lazy fallback and errors.
        make = self._STACK_FIXTURES[fixture]
        expected = self._stack_outcome(self._StackSlowPathTensorDict, make, stack, dim)
        assert self._stack_outcome(TensorDict, make, stack, dim) == expected

    def test_stack_fast_path_skips_per_td_calls(self, monkeypatch):
        # Exact TensorDicts of dense leaves are stacked without a _check_keys
        # call nor a _get_str call per tensordict and key; a subclass makes them.
        calls = []
        check_keys = tensordict._torch_func._check_keys
        get_str = TensorDict._get_str

        def counting_check_keys(*args, **kwargs):
            calls.append("_check_keys")
            return check_keys(*args, **kwargs)

        def counting_get_str(self, *args, **kwargs):
            calls.append("_get_str")
            return get_str(self, *args, **kwargs)

        monkeypatch.setattr(tensordict._torch_func, "_check_keys", counting_check_keys)
        monkeypatch.setattr(TensorDict, "_get_str", counting_get_str)
        make = self._STACK_FIXTURES["flat"]
        torch.stack(make(TensorDict), 0)
        assert calls == []
        torch.stack(make(self._StackSlowPathTensorDict), 0)
        assert calls.count("_check_keys") == 1
        assert calls.count("_get_str") == 6

    @pytest.mark.parametrize("dim", [0, -1])
    def test_stack_fast_path_tensorclass(self, dim):
        # A tensorclass stacks its tensordicts: exact TensorDicts take the fast
        # path, the subclass the general one.
        @tensorclass
        class TC:
            a: torch.Tensor
            s: str
            n: TensorDict

        def make(cls):
            return [
                TC._from_tensordict(
                    cls(
                        a=torch.full((3, 2), i),
                        s=NonTensorData("x"),
                        n=TensorDict(x=torch.zeros(3)),
                        batch_size=[3],
                    )
                )
                for i in range(2)
            ]

        def outcome(cls):
            result = torch.stack(make(cls), dim)
            return (
                type(result),
                result.batch_size,
                result.s,
                result.a.tolist(),
                type(result._tensordict["n"]),
                result._tensordict["n", "x"].shape,
            )

        assert outcome(TensorDict) == outcome(self._StackSlowPathTensorDict)

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

    def test_cummin_cummax_reduce(self):
        td = TensorDict(
            a=torch.randn(3, 4, 5),
            b=TensorDict(
                c=torch.randn(3, 4, 5), d=torch.randn(3, 4, 5), batch_size=(3, 4, 5)
            ),
            batch_size=(3, 4),
        )
        cummax = td.cummax(reduce=True, dim=0)
        cummin = (-td).cummin(reduce=True, dim=0)
        assert isinstance(cummax, torch.return_types.cummax)
        torch.testing.assert_close(cummax.values, -cummin.values)
        torch.testing.assert_close(cummax.indices, cummin.indices)

    @pytest.mark.parametrize("reduction", ["max", "min"])
    def test_max_min_reduce_dim(self, reduction):
        td = TensorDict(
            a=torch.randn(3, 4, 5),
            b=TensorDict(
                c=torch.randn(3, 4, 5), d=torch.randn(3, 4, 5), batch_size=(3, 4, 5)
            ),
            batch_size=(3, 4),
        )
        agglomerate = torch.cat([td["a"], td["b", "c"], td["b", "d"]], dim=0)
        expected = getattr(torch, reduction)(agglomerate, dim=0)
        result = getattr(td, reduction)(dim=0, reduce=True)
        assert isinstance(result, getattr(torch.return_types, reduction))
        torch.testing.assert_close(result.values, expected.values)
        torch.testing.assert_close(result.indices, expected.indices)

    @pytest.mark.parametrize("reduce", [False, True])
    @pytest.mark.parametrize(
        "reduction",
        [
            "sum",
            "nansum",
            "mean",
            "nanmean",
            "std",
            "var",
            "prod",
            "quantile",
            "amax",
            "amin",
            "max",
            "min",
        ],
    )
    def test_reduction_dim_none(self, reduction, reduce):
        # dim=None reduces over all the elements, as a call without dim does
        td = TensorDict(
            a=torch.randn(3, 4, 5),
            b=TensorDict(c=torch.randn(3, 4, 5, 6), batch_size=(3, 4, 5)),
            batch_size=(3, 4),
        )
        args = (0.5,) if reduction == "quantile" else ()
        result = getattr(td, reduction)(*args, dim=None, reduce=reduce)
        expected = getattr(td, reduction)(*args, reduce=reduce)
        if reduce:
            torch.testing.assert_close(result, expected)
        else:
            assert result.batch_size == torch.Size([])
            assert result["b"].batch_size == torch.Size([])
            assert_allclose_td(result, expected)

    @pytest.mark.parametrize("reduction", ["sum", "mean", "max", "cummax"])
    @pytest.mark.parametrize("dim", ["omitted", 0])
    @pytest.mark.parametrize(
        "td",
        [
            TensorDict(batch_size=[3]),
            TensorDict(s="text", batch_size=[3]),
            NonTensorData("text", batch_size=[3]),
        ],
        ids=["empty", "non-tensor-only", "NonTensorData"],
    )
    def test_reduction_reduce_no_tensor_leaves(self, td, dim, reduction):
        # reduce=True with no tensor leaves names the cause, instead of the
        # error of torch.cat on an empty list
        if dim == "omitted" and reduction == "cummax":
            pytest.skip("cummax needs a dim")
        kwargs = {} if dim == "omitted" else {"dim": dim}
        with pytest.raises(ValueError, match="has no tensor leaves"):
            getattr(td, reduction)(reduce=True, **kwargs)

    @pytest.mark.parametrize(
        "reduction",
        ["sum", "nansum", "mean", "nanmean", "std", "var", "quantile", "amax", "amin"],
    )
    def test_reduction_dim_none_keepdim(self, reduction):
        # With keepdim=True, each leaf is reduced as torch reduces it with dim=None
        td = TensorDict(a=torch.randn(3, 4, 5), batch_size=(3, 4))
        args = (0.5,) if reduction == "quantile" else ()
        result = getattr(td, reduction)(*args, dim=None, keepdim=True)
        assert result.batch_size == torch.Size([1, 1])
        torch.testing.assert_close(
            result["a"], getattr(td["a"], reduction)(*args, dim=None, keepdim=True)
        )

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

    @pytest.mark.parametrize(
        "reduction", ["sum", "mean", "std", "amax", "prod", "quantile"]
    )
    def test_reduction_negative_dim_reduce_warns(self, reduction):
        # With reduce=True, a negative dim counts from the end of the leaves. It
        # warns, and will count from the end of the batch dims in 0.17 (#2044).
        x = torch.rand(3, 4, 5)
        td = TensorDict(
            a=x, b=TensorDict(c=x + 1, batch_size=(3, 4)), batch_size=(3, 4)
        )
        args = (0.5,) if reduction == "quantile" else ()
        leaves = torch.cat([x, x + 1], -1)
        for dim in (-1, (0, -1)) if reduction in ("sum", "mean", "std") else (-1,):
            with pytest.warns(
                FutureWarning,
                match="Starting with TensorDict 0.17, the default will change to "
                "counting it from the end of the batch dims",
            ) as record:
                result = getattr(td, reduction)(*args, dim=dim, reduce=True)
            assert record[0].filename == __file__
            torch.testing.assert_close(
                result, getattr(torch, reduction)(leaves, *args, dim=dim)
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # a non-negative dim, and a negative dim without feature dims
            getattr(td, reduction)(*args, dim=1, reduce=True)
            td_flat = TensorDict(a=x[..., 0], batch_size=(3, 4))
            getattr(td_flat, reduction)(*args, dim=-1, reduce=True)
            getattr(td, reduction)(*args, dim=-1)

    @pytest.mark.parametrize(
        "reduction", ["amax", "amin", "max", "min", "prod", "cummax", "cummin"]
    )
    def test_reduction_tuple_dim_warns(self, reduction):
        # amax, amin, max, min, prod, cummax and cummin use only the first dim of
        # a tuple. amax and amin will reduce every dim in 0.17, the others reject
        # a tuple, as torch does (#2044).
        x = torch.rand(3, 4, 5)
        td = TensorDict(a=x, batch_size=(3, 4))
        category, match = (
            (FutureWarning, "the default will change to reducing every dim")
            if reduction in ("amax", "amin")
            else (DeprecationWarning, "will be removed in TensorDict 0.17")
        )
        kwargs = (
            {"return_indices": False}
            if reduction in ("max", "min", "cummax", "cummin")
            else {}
        )
        tuples = [(0, 1), (-1,)]
        if reduction not in ("amax", "amin"):
            tuples.append((1,))
        for dim in tuples:
            with pytest.warns(category, match=match) as record:
                result = getattr(td, reduction)(dim=dim, **kwargs)
            assert record[0].filename == __file__
            torch_reduction = {"max": "amax", "min": "amin"}.get(reduction, reduction)
            expected = getattr(torch, torch_reduction)(x, dim=dim[0])
            if reduction in ("cummax", "cummin"):
                expected = expected.values
            # the first dim of the tuple, not checked against the batch dims
            torch.testing.assert_close(result["a"], expected)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            getattr(td, reduction)(dim=1, **kwargs)
            getattr(td, reduction)(dim=-1, **kwargs)
            if reduction in ("amax", "amin"):
                # one dim inside the batch dims: the same result as torch
                assert getattr(td, reduction)(dim=(1,)).batch_size == (3,)

    @pytest.mark.parametrize("dim", [0, 1, -2, -1])
    def test_prod_keepdim(self, dim):
        # keepdim=True keeps the reduced batch dim with size 1, as torch.prod does
        td = TensorDict(
            a=torch.rand(2, 3, 4),
            b=TensorDict(c=torch.rand(2, 3, 4, 5), batch_size=(2, 3, 4)),
            batch_size=(2, 3),
        )
        batch_dim = dim % td.batch_dims
        result = td.prod(dim, keepdim=True)
        assert result.batch_size == torch.Size(
            [1 if i == batch_dim else s for i, s in enumerate(td.batch_size)]
        )
        for key in ("a", ("b", "c")):
            torch.testing.assert_close(
                result[key], td[key].prod(batch_dim, keepdim=True)
            )

    @pytest.mark.parametrize("dim", [0, 1])
    def test_prod_keepdim_reduce(self, dim):
        # reduce=True reduces the leaves concatenated along dim, as sum does
        td = TensorDict(a=torch.rand(2, 3, 4), b=torch.rand(2, 3, 4), batch_size=(2, 3))
        torch.testing.assert_close(
            td.prod(dim, keepdim=True, reduce=True),
            torch.cat([td["a"], td["b"]], dim).prod(dim, keepdim=True),
        )

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
        assert multi_quantile.batch_size == torch.Size([3, 4])
        assert multi_quantile["a"].shape == torch.Size([3, 4, 5])

        # Test feature dimension
        quantile_feature = td.quantile(0.5, dim="feature")
        assert quantile_feature.batch_size == torch.Size([3, 4])
        assert quantile_feature["a"].shape == torch.Size([3, 4])

        # Test with keepdim
        quantile_keepdim = td.quantile(0.5, dim=0, keepdim=True)
        assert quantile_keepdim.batch_size == torch.Size([1, 4])
        assert quantile_keepdim["a"].shape == torch.Size([1, 4, 5])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dim": 0},
            {"dim": -1},
            {"dim": 0, "keepdim": True},
            {"keepdim": True},
            {"dim": "feature"},
        ],
        ids=["dim", "neg_dim", "dim_keepdim", "keepdim", "feature"],
    )
    def test_quantile_tensor_q(self, kwargs):
        # With a 1-d tensor q, the result stacks the results of each quantile
        # along a new first dim, as torch.quantile does
        td = TensorDict(
            a=torch.randn(3, 4, 5),
            b=TensorDict(c=torch.randn(3, 4, 5, 6), batch_size=(3, 4, 5)),
            batch_size=(3, 4),
        )
        q = torch.tensor([0.25, 0.5])
        result = td.quantile(q, **kwargs)
        expected = torch.stack([td.quantile(q_i.item(), **kwargs) for q_i in q])
        assert result.batch_size == expected.batch_size
        assert result["b"].batch_size == expected["b"].batch_size
        assert_allclose_td(result, expected)
        assert_allclose_td(result.unbind(0)[1], expected[1])
        # A 0-d tensor q adds no dim
        assert td.quantile(q[0], **kwargs).batch_size == expected.batch_size[1:]

        if "dim" in kwargs:
            td.names = ["x", "y"]
            result = td.quantile(q, **kwargs)
            expected = td.quantile(0.5, **kwargs)
            assert result.names == [None, *expected.names]
            assert result["b"].names == [None, *expected["b"].names]

    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.parametrize("batch_size", [(3, 4), (3, 4, 5)])
    def test_quantile_tuple_dim(self, batch_size, keepdim):
        # torch.quantile takes one dim: a tuple reduces the leaves and the batch
        # size over its first dim only. Passing one is deprecated, and will be
        # rejected in 0.17, as torch does (#2044).
        x = torch.rand(3, 4, 5)
        td = TensorDict(a=x, batch_size=batch_size)
        for dim in [(0, 1), (-1,)]:
            with pytest.warns(
                DeprecationWarning, match="will be removed in TensorDict 0.17"
            ) as record:
                result = td.quantile(0.5, dim=dim, keepdim=keepdim)
            assert record[0].filename == __file__
            expected = td.quantile(0.5, dim=dim[0], keepdim=keepdim)
            assert result.batch_size == expected.batch_size
            assert_allclose_td(result, expected)
            assert result["a"].shape[: result.batch_dims] == result.batch_size
            result.unbind(0)

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

    def test_sub_td_expand_shares_parent(self):
        td = TensorDict(
            {
                "a": torch.arange(24.0).view(2, 4, 3),
                ("n", "b"): torch.zeros(2, 4, 3, 1),
            },
            [2, 4, 3],
        )
        for index in (1, slice(0, 1), torch.tensor([1, 0])):
            sub = td._get_sub_tensordict(index)
            out = sub.expand(5, *sub.batch_size)
            assert type(out) is TensorDict
            assert out.batch_size == (5, *sub.batch_size)
            assert (out[0] == sub).all()
            if not isinstance(index, torch.Tensor):
                assert out["a"].untyped_storage().data_ptr() == (
                    td["a"].untyped_storage().data_ptr()
                )
            assert out.expand(-1, *sub.batch_size).batch_size == out.batch_size
        # names, as for a TensorDict
        td.names = ["x", "y", "z"]
        sub = td._get_sub_tensordict(1)
        out = sub.expand(2, 4, 3)
        assert out.names == sub.to_tensordict().expand(2, 4, 3).names
        assert out.names == [None, "y", "z"]
        assert out["n"].names == [None, "y", "z"]

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
        assert (td_reconstruct == td).all(), (
            f"td and td_reconstruct differ, got {td} and {td_reconstruct}"
        )

        superlist = [stack_td(list(_td), 0, contiguous=False) for _td in td]
        td_reconstruct = stack_td(superlist, 0, contiguous=False)
        assert (td_reconstruct == td).all(), (
            f"td and td_reconstruct differ, got {td == td_reconstruct}"
        )

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

    @pytest.mark.parametrize(
        "memory_format, shape",
        [
            (torch.channels_last, (2, 3, 4, 5)),
            (torch.channels_last_3d, (2, 3, 4, 5, 6)),
        ],
        ids=["channels_last", "channels_last_3d"],
    )
    @pytest.mark.parametrize(
        "args",
        [(), ("cpu",), ("meta",), (torch.float64,)],
        ids=["no-device", "same-device", "other-device", "dtype"],
    )
    @pytest.mark.parametrize("td_type", ["td", "consolidated", "sub_td"])
    def test_to_memory_format(self, memory_format, shape, args, td_type):
        # As in nn.Module.to, the memory format applies to the 4D and 5D leaves
        td = TensorDict(
            a=torch.randn(shape),
            b=torch.randn(2, 3),
            nested=TensorDict(c=torch.randn(shape), batch_size=[2]),
            batch_size=[2],
            device="cpu",
        )
        if td_type == "consolidated":
            td = td.consolidate()
        elif td_type == "sub_td":
            td = td._get_sub_tensordict(slice(None))
        out = td.to(*args, memory_format=memory_format)
        assert out["a"].is_contiguous(memory_format=memory_format)
        assert out["nested", "c"].is_contiguous(memory_format=memory_format)
        assert out["b"].stride() == (3, 1)
        if args != ("meta",):
            assert (out == td).all()

    @staticmethod
    def _cast_filter_td():
        return TensorDict(
            x=torch.zeros(2, 3),
            cells=torch.zeros(2, 3, dtype=torch.long),
            mask=torch.zeros(2, dtype=torch.bool),
            info=NonTensorData("info"),
            point_data=TensorDict(
                p=torch.zeros(2),
                zone=torch.zeros(2, dtype=torch.int32),
                batch_size=[2],
            ),
            batch_size=[2],
        )

    @pytest.mark.parametrize(
        "td_type", ["td", "locked", "lazy", "params", "sub_td", "unsqueezed"]
    )
    @pytest.mark.parametrize(
        "call",
        [
            ("to", (torch.bfloat16,), {}),
            ("to", ("cpu", torch.bfloat16), {}),
            ("to", (), {"dtype": torch.bfloat16, "non_blocking": True}),
            ("to", (torch.zeros((), dtype=torch.bfloat16),), {}),
            ("bfloat16", (), {}),
            ("type", (torch.bfloat16,), {}),
        ],
        ids=["to-dtype", "to-device-dtype", "to-kwargs", "to-tensor", "bf16", "type"],
    )
    def test_to_cast_filter(self, td_type, call):
        # cast_filter selects the leaves the dtype cast applies to; it gets
        # the key as a tuple of str and is never called on non-tensor data
        td = self._cast_filter_td()
        if td_type == "locked":
            td = td.lock_()
        elif td_type == "lazy":
            td = lazy_stack(list(td.unbind(0)))
        elif td_type == "params":
            td = TensorDictParams(td.exclude("info"))
        elif td_type == "sub_td":
            td = td._get_sub_tensordict(slice(None))
        elif td_type == "unsqueezed":
            td = td.unsqueeze(0)
        seen = []

        def cast_filter(key, value):
            assert isinstance(key, tuple) and all(isinstance(k, str) for k in key)
            assert isinstance(value, torch.Tensor)
            seen.append(key)
            return tensordict.is_inexact(value)

        name, args, kwargs = call
        out = getattr(td, name)(*args, cast_filter=cast_filter, **kwargs)
        assert set(seen) == {
            ("x",),
            ("cells",),
            ("mask",),
            ("point_data", "p"),
            ("point_data", "zone"),
        }
        assert out["x"].dtype == torch.bfloat16
        assert out["point_data", "p"].dtype == torch.bfloat16
        assert out["cells"].dtype == torch.long
        assert out["mask"].dtype == torch.bool
        assert out["point_data", "zone"].dtype == torch.int32
        assert (out == td.to(torch.bfloat16)).all()
        if td_type not in ("params", "lazy", "sub_td", "unsqueezed"):
            # The leaves left out are not copied
            assert out["cells"] is td["cells"]
            assert out.get_non_tensor("info") == "info"
        if td_type == "locked" and name != "type":
            # type() does not propagate the lock, with or without cast_filter
            assert out.is_locked
        elif td_type == "lazy":
            assert isinstance(out, LazyStackedTensorDict)
        elif td_type == "params" and name == "to":
            assert isinstance(out, TensorDictParams)

    @pytest.mark.parametrize(
        "method",
        [
            "bool",
            "double",
            "float",
            "half",
            "int",
            "bfloat16",
            "complex64",
            "float16",
            "float32",
            "float64",
            "int16",
            "int32",
            "int64",
            "int8",
            "uint8",
        ],
    )
    def test_dtype_shorthand_cast_filter(self, method):
        td = self._cast_filter_td()
        dtype = getattr(td, method)()["x"].dtype
        # A keyed predicate: leave point_data out
        out = getattr(td, method)(
            cast_filter=lambda k, v: tensordict.is_inexact(v) and k[0] != "point_data"
        )
        assert out["x"].dtype == dtype
        assert out["point_data", "p"].dtype == torch.float32
        assert out["cells"].dtype == torch.long
        assert out["mask"].dtype == torch.bool
        assert out["point_data", "zone"].dtype == torch.int32
        # None casts every leaf, as without the keyword
        out = getattr(td, method)(cast_filter=None)
        assert out["cells"].dtype == out["mask"].dtype == dtype

    def test_to_cast_filter_integer_target(self):
        # An integer target does not raise: the filter does what it says
        td = self._cast_filter_td()
        out = td.to(torch.int64, cast_filter=tensordict.is_inexact)
        assert out["x"].dtype == torch.int64
        assert out["mask"].dtype == torch.bool
        assert out["point_data", "zone"].dtype == torch.int32

    @pytest.mark.parametrize("cast_filter", [None, tensordict.is_inexact])
    def test_to_cast_filter_attrs(self, cast_filter):
        # The attrs give each leaf its dtype: cast_filter does not apply
        td = self._cast_filter_td().exclude("info")
        attrs = td.double().attrs(fields=("device", "dtype"))
        out = td.to(attrs, cast_filter=cast_filter)
        assert (out == td.to(attrs)).all()
        assert out["cells"].dtype == out["mask"].dtype == torch.float64

    def test_to_cast_filter_device_and_memory_format(self):
        # The device and memory format apply to every leaf, the dtype to the
        # selected ones
        td = TensorDict(
            img=torch.randn(2, 3, 4, 5),
            ids=torch.zeros(2, 3, 4, 5, dtype=torch.long),
            batch_size=[2],
        )
        out = td.to(
            torch.float64,
            memory_format=torch.channels_last,
            cast_filter=tensordict.is_inexact,
        )
        assert out["img"].dtype == torch.float64
        assert out["ids"].dtype == torch.long
        for value in out.values():
            assert value.is_contiguous(memory_format=torch.channels_last)
        out = td.to("meta", torch.float64, cast_filter=tensordict.is_inexact)
        assert out.device == torch.device("meta")
        assert out["ids"].device == torch.device("meta")
        assert out["img"].dtype == torch.float64
        assert out["ids"].dtype == torch.long
        # Without a dtype, cast_filter is not called
        out = td.to("meta", cast_filter=lambda k, v: pytest.fail("called"))
        assert out["ids"].device == torch.device("meta")

    def test_to_cast_filter_inplace_and_context_manager(self):
        td = self._cast_filter_td()
        out = td.to(torch.float64, cast_filter=tensordict.is_inexact, inplace=True)
        assert out is td
        assert td["x"].dtype == torch.float64
        assert td["cells"].dtype == torch.long
        td = self._cast_filter_td()
        with td.to(torch.float64, cast_filter=tensordict.is_inexact) as td64:
            assert td64["x"].dtype == torch.float64
            assert td64["cells"].dtype == torch.long
            td64["x"] += 1
        assert td["x"].dtype == torch.float32
        assert (td["x"] == 1).all()
        assert td["cells"].dtype == torch.long

    def test_to_cast_filter_lazy_stack_exclusive_keys(self):
        # The keys are those of the members, without their index
        td = lazy_stack(
            [
                TensorDict(a=torch.zeros(()), n=TensorDict(b=torch.zeros(()))),
                TensorDict(a=torch.zeros(()), c=torch.zeros(())),
            ]
        )
        for locked in (False, True):
            if locked:
                td.lock_()
            seen = []
            out = td.to(
                torch.float64,
                cast_filter=lambda k, v, seen=seen: seen.append(k) or k != ("a",),
            )
            assert sorted(seen) == [("a",), ("a",), ("c",), ("n", "b")]
            assert out.tensordicts[0]["a"].dtype == torch.float32
            assert out.tensordicts[0]["n", "b"].dtype == torch.float64
            assert out.tensordicts[1]["c"].dtype == torch.float64

    def test_dtype_predicates(self):
        tensors = {
            torch.float32: (True, True, False, False),
            torch.bfloat16: (True, True, False, False),
            torch.complex64: (True, False, False, False),
            torch.int64: (False, False, True, False),
            torch.uint8: (False, False, True, False),
            torch.bool: (False, False, False, True),
        }
        predicates = (
            tensordict.is_inexact,
            tensordict.is_floating,
            tensordict.is_integer,
            tensordict.is_bool,
        )
        for dtype, expected in tensors.items():
            t = torch.zeros(2, dtype=dtype)
            # Called as fn(value) and as fn(key, value)
            assert tuple(p(t) for p in predicates) == expected
            assert tuple(p(("a", "b"), t) for p in predicates) == expected
        td = self._cast_filter_td()
        floats = td.apply(lambda t: t if tensordict.is_floating(t) else None)
        assert set(floats.keys(True, True)) == {"x", ("point_data", "p")}

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

    @pytest.mark.parametrize(
        "args,kwargs",
        [
            ((torch.float64,), {}),
            ((), {"dtype": torch.float64}),
            # The tensordict is on cpu already: to() returns a new one all the same
            (("cpu", torch.float64), {}),
            ((torch.zeros((), dtype=torch.float64),), {}),
            ((), {"other": TensorDict(a=torch.zeros((), dtype=torch.float64))}),
            (
                (
                    TensorDict(
                        x=torch.zeros(3, dtype=torch.float64),
                        i=torch.zeros(3, dtype=torch.float64),
                        batch_size=[3],
                    ).attrs(),
                ),
                {},
            ),
            ((torch.float64,), {"non_blocking": None}),
        ],
        ids=[
            "dtype",
            "dtype_kwarg",
            "device_dtype",
            "tensor",
            "other",
            "attrs",
            "non_blocking_none",
        ],
    )
    def test_to_context_manager_restores_dtype(self, args, kwargs):
        td = TensorDict(
            x=torch.zeros(3),
            i=torch.arange(3),
            nested=TensorDict(b=torch.ones(3, dtype=torch.bool), batch_size=[3]),
            batch_size=[3],
            device="cpu",
        )
        with td.to(*args, **kwargs) as td64:
            assert td64 is not td
            assert td64["i"].dtype == torch.float64
            td64["x"] += 0.5
            td64["y"] = td64["x"] * 2
        # On exit, each entry gets its original dtype back
        assert td["x"].dtype == torch.float32
        assert (td["x"] == 0.5).all()
        assert td["i"].dtype == torch.int64
        assert (td["i"] == torch.arange(3)).all()
        assert td["nested", "b"].dtype == torch.bool
        # An entry added in the block keeps its dtype
        assert td["y"].dtype == torch.float64
        assert (td["y"] == 1).all()

    @pytest.mark.parametrize(
        "args,kwargs",
        [
            (("cpu",), {}),
            (
                (
                    TensorDict(i=torch.zeros(3), batch_size=[3]).attrs(
                        fields=("device",)
                    ),
                ),
                {},
            ),
            (("cpu",), {"non_blocking": None}),
        ],
        ids=["device", "attrs", "device_non_blocking_none"],
    )
    def test_to_context_manager_device_only_keeps_dtype(self, args, kwargs):
        # A device-only to() restores the devices only: an entry given another
        # dtype in the block keeps it
        td = TensorDict(x=torch.zeros(3), i=torch.arange(3), batch_size=[3])
        with td.to(*args, **kwargs) as td_cpu:
            assert td_cpu is not td
            td_cpu["i"] = td_cpu["i"] / 2
        assert td["x"].dtype == torch.float32
        assert td["i"].dtype == torch.float32
        assert (td["i"] == torch.arange(3) / 2).all()

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"non_blocking": True}, {"num_threads": 2}, {"other": TensorDict()}],
        ids=["none", "non_blocking", "num_threads", "other"],
    )
    def test_to_sub_tensordict_copies(self, kwargs):
        # without a device, a sub-tensordict returns a TensorDict copy
        td = TensorDict(x=torch.zeros(4, 3), batch_size=[4])
        sub = td._get_sub_tensordict(slice(0, 2))
        out = sub.to(**kwargs)
        assert type(out) is TensorDict
        out["x"] += 1
        assert (td["x"] == 0).all()

    def test_to_context_manager_sub_tensordict_same_device(self):
        # to() on the device of a sub-tensordict returns it, so the block
        # writes to the source and nothing is written back on exit
        td = TensorDict(x=torch.zeros(4, 3), batch_size=[4], device="cpu")
        sub = td._get_sub_tensordict(slice(0, 2))
        with sub.to(sub.device) as same:
            assert same is sub
            same["x"] += 1
        assert (td["x"][:2] == 1).all()
        assert (td["x"][2:] == 0).all()

    @pytest.mark.skipif(not _has_streaming, reason="streaming is not installed")
    def test_to_mds(self, tmpdir):
        td = TensorDict(
            a=[0, 1, 2],
            b=[0, 1, 0],
            c=torch.tensor([[4, 5], [6, 7], [8, 9]]),
            d=["a string!", "another string!", "again!"],
            e=torch.tensor([0.5, 1.5, 2.5]),
            f=torch.tensor([0.25, 1.25, 2.25], dtype=torch.float64),
            batch_size=[3],
        )

        tmpdir = str(tmpdir)
        td.to_mds(out=tmpdir)

        # Load the dataset
        dataset = streaming_dataset(local=tmpdir, remote=None, batch_size=2)
        dl = torch.utils.data.DataLoader(  # noqa: TOR401
            dataset=dataset, batch_size=2, collate_fn=TensorDict.from_list
        )
        batches = list(dl)
        batches = [_batch for batch in batches for _batch in batch.unbind(0)]
        test_td = torch.stack(batches)
        assert_allclose_td(td, test_td)

    @pytest.mark.skipif(not _has_streaming, reason="streaming is not installed")
    def test_to_mds_zero_dim(self, tmpdir):
        # a tensordict without batch dims is written as one row
        td = TensorDict(a=torch.arange(3), s="a string", n=3, x=0.5)

        tmpdir = str(tmpdir)
        td.to_mds(out=tmpdir)

        dataset = streaming_dataset(local=tmpdir, remote=None, batch_size=1)
        assert len(dataset) == 1
        assert_allclose_td(td, TensorDict.from_list([dataset[0]])[0])

    @pytest.mark.skipif(not _has_streaming, reason="streaming is not installed")
    def test_to_mds_zero_dim_writer(self, tmpdir):
        # a writer passed by the caller stays open for more samples
        from streaming import MDSWriter

        tds = [
            TensorDict(a=torch.arange(3) + k, s=f"string {k}", n=k, x=k + 0.5)
            for k in range(3)
        ]

        tmpdir = str(tmpdir)
        writer = MDSWriter(
            out=tmpdir,
            columns={"a": "ndarray", "s": "str", "n": "int", "x": "float32"},
        )
        for td in tds:
            td.to_mds(out=tmpdir, writer=writer)
        writer.finish()

        dataset = streaming_dataset(local=tmpdir, remote=None, batch_size=1)
        assert len(dataset) == 3
        test_td = TensorDict.from_list([dataset[i] for i in range(3)])
        assert_allclose_td(torch.stack(tds), test_td)

    @pytest.mark.skipif(not _has_streaming, reason="streaming is not installed")
    def test_to_mds_zero_dim_writer_pkl(self, tmpdir):
        # a pkl column of the caller's writer stores the tensor itself, and
        # entries that NumPy cannot take are fine outside the converted columns
        from streaming import MDSWriter

        td = TensorDict(
            a=torch.arange(3, dtype=torch.bfloat16),
            b=torch.arange(2),
            g=torch.ones(2, requires_grad=True),
        )

        tmpdir = str(tmpdir)
        with MDSWriter(out=tmpdir, columns={"a": "pkl", "b": "ndarray"}) as writer:
            td.to_mds(out=tmpdir, writer=writer)

        dataset = streaming_dataset(
            local=tmpdir, remote=None, batch_size=1, allow_unsafe_types=True
        )
        row = dataset[0]
        assert isinstance(row["a"], torch.Tensor)
        assert row["a"].dtype == torch.bfloat16
        assert (row["a"] == td["a"]).all()
        assert "g" not in row
        assert isinstance(row["b"], np.ndarray)
        assert (row["b"] == td["b"].numpy()).all()

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

    @pytest.mark.parametrize("use_state_dict", [True, False])
    @pytest.mark.parametrize("kind", ["dict", "tensor", "inplace", "object"])
    def test_to_module_extra_state(self, kind, use_state_dict):
        # to_module loads the extra state that from_module(use_state_dict=True)
        # saves, also for nested modules, and the context manager restores it,
        # also when set_extra_state copies into the tensors get_extra_state
        # returned ("inplace").
        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)
                self.state = torch.zeros(())

            @property
            def count(self):
                return int(self.state)

            @count.setter
            def count(self, value):
                self.state.fill_(value)

            def get_extra_state(self):
                if kind == "dict":
                    return {"count": torch.tensor(self.count), "n": self.count}
                if kind == "tensor":
                    return torch.tensor(self.count)
                if kind == "inplace":
                    return self.state
                return self.count

            def set_extra_state(self, state):
                if kind == "dict":
                    assert isinstance(state["n"], int)
                    self.count = int(state["count"])
                    assert state["n"] == self.count
                elif kind == "tensor":
                    assert isinstance(state, torch.Tensor)
                    self.count = int(state)
                elif kind == "inplace":
                    self.state.copy_(state)
                else:
                    assert isinstance(state, int)
                    self.count = state

        def make():
            return nn.Sequential(nn.Linear(2, 2), nn.Sequential(Counter()))

        src = make()
        src[1][0].count = 7
        params = TensorDict.from_module(src, use_state_dict=True)
        assert (params.clone() == params).all()

        dst = make()
        params.to_module(dst, use_state_dict=use_state_dict)
        assert dst[1][0].count == 7
        assert (dst[0].weight == src[0].weight).all()

        dst = make()
        with params.to_module(dst, use_state_dict=use_state_dict):
            assert dst[1][0].count == 7
            dst[1][0].count = 8
        assert dst[1][0].count == 0

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
        assert td_unbind[0].batch_size == td[:, 0].batch_size, (
            f"got {td_unbind[0].batch_size} and {td[:, 0].batch_size}"
        )

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

        td_dest = TensorDict(a=0, b=1, c=2)
        a, b = td_dest["a"], td_dest["b"]
        td_source = TensorDict(b=3, a=4)
        td_dest.update_(td_source)
        assert td_dest["a"] is a and td_dest["b"] is b
        assert a == 4 and b == 3 and td_dest["c"] == 2

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
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "update() got a source with batch size torch.Size([4]) for a "
                "destination with batch size torch.Size([3]). "
                "Pass update_batch_size=True"
            ),
        ):
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

    @pytest.mark.parametrize("lazy", [False, True])
    def test_update_ignore_lock_deprecated(self, lazy):
        # #2260: ignore_lock=True warns once per call, not once per nested
        # tensordict, and otherwise behaves as before.
        td = TensorDict({"a": torch.zeros(3), "n": {"b": {"c": torch.zeros(3)}}}, [3])
        if lazy:
            td = LazyStackedTensorDict(*td.unbind(0))
        source = TensorDict({"a": torch.ones(3), "n": {"b": {"c": torch.ones(3)}}}, [3])
        with pytest.warns(
            DeprecationWarning,
            match=r"^update\(ignore_lock=True\) is deprecated and will be removed in "
            r"TensorDict 0\.17\. Use 'with td\.unlock_\(\): td\.update\(\.\.\.\)' instead\.$",
        ) as record:
            assert td.update(source, ignore_lock=True) is td
        assert len(record) == 1
        assert record[0].filename == __file__
        assert (td == source).all()

        # On a locked tensordict, the call still raises for a new entry
        td.lock_()
        with (
            pytest.warns(DeprecationWarning, match=r"ignore_lock=True") as record,
            pytest.raises(RuntimeError, match="Cannot modify locked TensorDict"),
        ):
            td.update({"d": torch.ones(3)}, ignore_lock=True)
        assert len(record) == 1
        with pytest.warns(DeprecationWarning, match=r"ignore_lock=True") as record:
            td.update(source.zero_(), inplace=True, ignore_lock=True)
        assert len(record) == 1
        assert (td == 0).all()

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("ignore_lock", [None, False])
    def test_update_ignore_lock_not_deprecated(self, lazy, ignore_lock):
        td = TensorDict({"a": torch.zeros(3), "n": {"b": {"c": torch.zeros(3)}}}, [3])
        if lazy:
            td = LazyStackedTensorDict(*td.unbind(0))
        source = TensorDict({"a": torch.ones(3), "n": {"b": {"c": torch.ones(3)}}}, [3])
        kwargs = {} if ignore_lock is None else {"ignore_lock": ignore_lock}
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            td.update(source, **kwargs)
            td.lock_()
            td.update(source.zero_(), inplace=True, **kwargs)
        assert (td == 0).all()

    @pytest.mark.parametrize("method", ["set", "update", "update_td", "update_kwargs"])
    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("device", [None, "cpu", "meta"])
    def test_eager_tensor_mutation(self, method, nested, device):
        value = torch.ones(2, 3, device=device, requires_grad=True)
        source = {"x": value, "y": value + 1}
        child = TensorDict({"keep": torch.zeros(2)}, [2], device=device)
        td = TensorDict({"child": child}, [2], device=device) if nested else child
        td.rename_("batch")
        keep = child["keep"]
        payload = {"child": source} if nested else source
        if method == "set":
            for key, tensor in source.items():
                td.set(("child", key) if nested else key, tensor)
        elif method == "update_kwargs":
            td.update(**payload)
        elif method == "update_td":
            td.update(TensorDict(payload, [2], device=device))
        else:
            td.update(payload)
        assert child["x"] is value and child["y"] is source["y"]
        assert child["keep"] is keep and child.names == ["batch"]
        if nested:
            assert td["child"] is child
        source["x"] = torch.zeros_like(value)
        assert child["x"] is value
        if device != "meta":
            child["x"].sum().backward()
            torch.testing.assert_close(value.grad, torch.ones_like(value))

    def test_eager_update_fallbacks(self):
        value = torch.ones(2, 3)
        td = TensorDict({"x": torch.zeros_like(value)}, [2], device="cpu")
        with pytest.raises(RuntimeError, match="batch dimension mismatch"):
            td.update({"x": value, "bad": torch.ones(3)})
        assert td["x"] is value and "bad" not in td.keys()
        # TensorDict inputs must still reject incompatible batch metadata.
        with pytest.raises(RuntimeError, match="update_batch_size"):
            td.update(TensorDict({"x": torch.ones(3)}, [3]))
        td.lock_()
        with pytest.raises(RuntimeError, match="locked"):
            td.update({"x": value})
        td.unlock_()
        param = nn.Parameter(value.clone())
        td.update({"x": param, ("nested", "y"): value})
        assert td["x"] is param and td["nested", "y"] is value
        dest = TensorDict({}, [2], device="meta")
        dest.update({"x": value})
        assert dest["x"].device.type == "meta"

    @staticmethod
    def _count_batchfree_validations(monkeypatch):
        calls = []
        for name in (
            "_validate_value_batchfree",
            "_validate_value_batchfree_devicefree",
        ):
            orig = getattr(TensorDictBase, name)

            def counted(self, value, *args, _orig=orig, **kwargs):
                calls.append(value)
                return _orig(self, value, *args, **kwargs)

            monkeypatch.setattr(TensorDictBase, name, counted)
        return calls

    @pytest.mark.parametrize("method", ["set", "setitem", "inplace"])
    @pytest.mark.parametrize(
        "parent_device,value_device,validates",
        [
            (None, None, False),
            (None, "cpu", False),
            ("cpu", "cpu", False),
            ("cpu", None, True),
            ("meta", "meta", False),
            ("meta", "cpu", True),
        ],
    )
    def test_eager_batchfree_td_set(
        self, method, parent_device, value_device, validates, monkeypatch
    ):
        # A plain TensorDict stored in a TensorDict without batch dims skips
        # the validator unless it has to move to the parent's device.
        calls = self._count_batchfree_validations(monkeypatch)
        parent = TensorDict(device=parent_device)
        if method == "inplace":
            parent["sub"] = TensorDict(
                {"x": torch.zeros(3, 2, device=parent_device)}, [3, 2]
            )
            calls.clear()
        value = TensorDict(
            {"x": torch.ones(3, 2, device=value_device)},
            [3, 2],
            device=value_device,
            names=["a", None],
        ).lock_()
        if method == "setitem":
            parent["sub"] = value
        else:
            parent.set("sub", value, inplace=method == "inplace")
        assert len(calls) == int(validates)
        stored = parent["sub"]
        if method == "inplace":
            assert parent_device == "meta" or (stored["x"] == 1).all()
            return
        assert (stored is value) is not validates
        assert type(stored) is TensorDict
        assert stored.batch_size == (3, 2) and stored.names == ["a", None]
        device = parent_device or value_device
        assert stored.device == (device and torch.device(device))
        assert stored["x"].device.type == (parent_device or value_device or "cpu")
        assert stored.is_locked
        assert parent.names == [] and not parent.is_locked
        parent.lock_()
        with pytest.raises(RuntimeError, match="locked"):
            parent.set("other", value)
        assert "other" not in parent.keys()

    def test_eager_batchfree_td_set_fallbacks(self, monkeypatch):
        calls = self._count_batchfree_validations(monkeypatch)

        class SubTD(TensorDict):
            pass

        @tensorclass
        class TC:
            data: TensorDict

        # Subclasses on either side and non-tensor data keep the validator.
        TensorDict().set("a", SubTD())
        SubTD().set("a", TensorDict())
        TensorDict().set("a", NonTensorData("x"))
        assert len(calls) == 3
        calls.clear()
        # A TensorDict field of a tensorclass without batch dims skips it.
        sub = TensorDict({"x": torch.ones(3)}, [3])
        tc = TC(data=sub, batch_size=[])
        tc.data = sub
        assert not calls
        assert tc.data is sub

    class _SlowPathTensorDict(TensorDict):
        """A TensorDict subclass: update() never takes the fast path of TensorDict."""

    @staticmethod
    def _update_outcome(cls, make, call):
        """What ``call`` does to the tensordict that ``make(cls)`` builds."""
        made = []

        def tensor(*shape, value=1.0):
            made.append(torch.full(shape, value))
            return made[-1]

        td = make(cls)
        subs = {
            key: sub
            for key, sub in td.items(True)
            if is_tensor_collection(sub) and not is_non_tensor(sub)
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = call(td, tensor)
            except Exception as err:
                outcome = (type(err), str(err).replace(cls.__name__, "TensorDict"))
            else:
                outcome = result is td
        state = {
            key: (
                type(leaf).__name__,
                leaf.shape,
                (
                    None
                    if isinstance(leaf, torch.Tensor) and leaf.is_meta
                    else leaf.tolist()
                ),
                any(leaf is tensor for tensor in made),
            )
            for key, leaf in td.items(True, True, is_leaf=is_leaf_nontensor)
        }
        warned = [
            (w.category, str(w.message), os.path.basename(w.filename)) for w in caught
        ]
        return (
            outcome,
            state,
            # the order of the keys, and which sub-tensordicts were merged
            # rather than replaced
            list(state),
            {key: td.get(key, None) is sub for key, sub in subs.items()},
            td.is_consolidated(),
            td.batch_size,
            td.names if td._has_names() else None,
            td.is_locked,
            warned,
        )

    _UPDATE_FIXTURES = {
        "flat": lambda cls: cls(a=torch.zeros(3, 4), b=torch.zeros(3), batch_size=[3]),
        "nested": lambda cls: cls(
            a=torch.zeros(3, 4),
            n=TensorDict(x=torch.zeros(3, 2), batch_size=[3]),
            batch_size=[3],
        ),
        "0-d": lambda cls: cls(a=torch.zeros(4), b=torch.zeros(()), batch_size=[]),
        "zero-size": lambda cls: cls(a=torch.zeros(0, 3, 2), batch_size=[0, 3]),
        "named": lambda cls: cls(a=torch.zeros(3, 4), batch_size=[3], names=["t"]),
        "locked": lambda cls: cls(a=torch.zeros(3, 4), batch_size=[3]).lock_(),
        "deep": lambda cls: cls(
            a=torch.zeros(3, 4),
            n=TensorDict(
                x=torch.zeros(3, 2),
                m=TensorDict(y=torch.zeros(3), batch_size=[3]),
                batch_size=[3],
            ),
            batch_size=[3],
        ),
        "two children": lambda cls: cls(
            a=torch.zeros(3, 4),
            n=TensorDict(x=torch.zeros(3, 2), batch_size=[3]),
            n2=TensorDict(x=torch.zeros(3, 2), batch_size=[3]),
            batch_size=[3],
        ),
        "aliased children": lambda cls: cls(
            a=torch.zeros(3, 4),
            n=(child := TensorDict(x=torch.zeros(3, 2), batch_size=[3])),
            n2=child,
            batch_size=[3],
        ),
        "aliased children with a grandchild": lambda cls: cls(
            a=torch.zeros(3, 4),
            n=(
                child := TensorDict(
                    x=torch.zeros(3, 2),
                    m=TensorDict(y=torch.zeros(3), batch_size=[3]),
                    batch_size=[3],
                )
            ),
            n2=child,
            batch_size=[3],
        ),
        "two children with grandchildren": lambda cls: cls(
            a=torch.zeros(3, 4),
            n=TensorDict(
                x=torch.zeros(3, 2),
                m=TensorDict(y=torch.zeros(3), batch_size=[3]),
                batch_size=[3],
            ),
            n2=TensorDict(
                x=torch.ones(3, 2),
                m=TensorDict(y=torch.ones(3), batch_size=[3]),
                batch_size=[3],
            ),
            batch_size=[3],
        ),
        "locked child": lambda cls: cls(
            a=torch.zeros(3, 4),
            n=TensorDict(x=torch.zeros(3, 2), batch_size=[3]).lock_(),
            batch_size=[3],
        ),
        "child with a device": lambda cls: cls(
            a=torch.zeros(3, 4),
            n=TensorDict(x=torch.zeros(3, 2), batch_size=[3], device="cpu"),
            batch_size=[3],
        ),
        "with a device": lambda cls: cls(
            a=torch.zeros(3, 4),
            n=TensorDict(x=torch.zeros(3, 2), batch_size=[3]),
            batch_size=[3],
            device="cpu",
        ),
    }

    _UPDATE_CASES = {
        "new key": lambda td, t: td.update({"new": t(*td.batch_size, 2)}),
        "overwrite": lambda td, t: td.update({"a": t(*td["a"].shape)}),
        "new shape": lambda td, t: td.update({"a": t(*td.batch_size, 7)}),
        "tuple key": lambda td, t: td.update({("m", "x"): t(*td.batch_size)}),
        "tuple key over a tensor": lambda td, t: td.update(
            {("a", "x"): t(*td.batch_size)}
        ),
        "nested dict": lambda td, t: td.update({"n": {"y": t(*td.batch_size)}}),
        "td source": lambda td, t: td.update(
            TensorDict(a=t(*td["a"].shape), batch_size=td.batch_size)
        ),
        "td source, prefix batch": lambda td, t: td.update(
            TensorDict(a=t(*td["a"].shape), batch_size=td.batch_size[:-1])
        ),
        "td source, bad batch": lambda td, t: td.update(
            TensorDict(a=t(5, 4), batch_size=[5])
        ),
        "clone": lambda td, t: td.update({"a": t(*td["a"].shape)}, clone=True),
        "inplace": lambda td, t: td.update({"a": t(*td["a"].shape)}, inplace=True),
        "keys_to_update": lambda td, t: td.update(
            {"a": t(*td["a"].shape), "new": t(*td.batch_size)}, keys_to_update=["a"]
        ),
        "nested keys_to_update": lambda td, t: td.update(
            {"n": {"y": t(*td.batch_size)}, "new": t(*td.batch_size)},
            keys_to_update=[("n", "y")],
        ),
        "empty keys_to_update": lambda td, t: td.update(
            {"new": t(*td.batch_size)}, keys_to_update=[]
        ),
        "kwargs": lambda td, t: td.update(new=t(*td.batch_size)),
        "dict and kwargs": lambda td, t: td.update(
            {"new": t(*td.batch_size)}, new=t(*td.batch_size, value=2.0)
        ),
        "td and kwargs": lambda td, t: td.update(
            TensorDict(new=t(*td.batch_size), batch_size=td.batch_size),
            new2=t(*td.batch_size),
        ),
        "None": lambda td, t: td.update(None),
        "self": lambda td, t: td.update(td),
        "empty dict": lambda td, t: td.update({}),
        "str value": lambda td, t: td.update({"s": "text"}),
        "non-tensor value": lambda td, t: td.update(
            {"s": NonTensorData("text", batch_size=td.batch_size)}
        ),
        "float value": lambda td, t: td.update({"f": 1.5}),
        "empty key": lambda td, t: td.update({(): t(*td.batch_size)}),
        "int key": lambda td, t: td.update({1: t(*td.batch_size)}),
        "parameter": lambda td, t: td.update({"p": nn.Parameter(t(*td.batch_size, 2))}),
        "td value": lambda td, t: td.update(
            {"m": TensorDict(x=t(*td.batch_size), batch_size=td.batch_size)}
        ),
        "update_batch_size": lambda td, t: td.update(
            TensorDict(a=t(2, 4), batch_size=[2]), update_batch_size=True
        ),
        "ignore_lock": lambda td, t: td.update(
            {"a": t(*td["a"].shape)}, ignore_lock=True
        ),
        "ignore_lock, inplace": lambda td, t: td.update(
            {"a": t(*td["a"].shape)}, inplace=True, ignore_lock=True
        ),
        "is_leaf": lambda td, t: td.update(
            {"n": {"y": t(*td.batch_size)}}, is_leaf=lambda cls: True
        ),
        "default is_leaf": lambda td, t: td.update(
            {"new": t(*td.batch_size)}, is_leaf=is_leaf_nontensor
        ),
        "lazy source": lambda td, t: td.update(
            lazy_stack([TensorDict(a=t(*td["a"].shape[1:])) for _ in td.unbind(0)])
            if td.batch_dims
            else lazy_stack([TensorDict(a=t(4)), TensorDict(a=t(4))])
        ),
        "dict with a sub-tensordict last": lambda td, t: td.update(
            {
                "new": t(*td.batch_size),
                "m": TensorDict(x=t(*td.batch_size), batch_size=td.batch_size),
            }
        ),
        "td source, nested": lambda td, t: td.update(
            TensorDict(
                a=t(*td["a"].shape),
                n=TensorDict(x=t(*td.batch_size, 2), y=t(*td.batch_size)),
                batch_size=td.batch_size,
            )
        ),
        "td source, nested only": lambda td, t: td.update(
            TensorDict(n=TensorDict(y=t(*td.batch_size)), batch_size=td.batch_size)
        ),
        "td source, deep": lambda td, t: td.update(
            TensorDict(
                n=TensorDict(m=TensorDict(z=t(*td.batch_size))),
                batch_size=td.batch_size,
            )
        ),
        "nested dict, deep": lambda td, t: td.update(
            {"n": {"m": {"z": t(*td.batch_size)}, "y": t(*td.batch_size)}}
        ),
        "nested values between tensors": lambda td, t: td.update(
            {
                "new": t(*td.batch_size),
                "n": {"y": t(*td.batch_size)},
                "n2": TensorDict(y=t(*td.batch_size), batch_size=td.batch_size),
                "new2": t(*td.batch_size),
            }
        ),
        "nested td, bad batch": lambda td, t: td.update(
            {"n": TensorDict(x=t(5, 2), batch_size=[5])}
        ),
        "nested empty td, bad batch": lambda td, t: td.update(
            {"n": TensorDict(batch_size=[5])}
        ),
        "nested td of a td, bad batch": lambda td, t: td.update(
            {"n": TensorDict(m=TensorDict(batch_size=[5, 2]), batch_size=[5])}
        ),
        "nested td, prefix batch": lambda td, t: td.update(
            {"n": TensorDict(y=t(*td.batch_size))}
        ),
        "nested tensor, bad shape": lambda td, t: td.update({"n": {"y": t(7, 1)}}),
        "nested tensor, bad device": lambda td, t: td.update(
            {"n": {"y": t(*td.batch_size).to("meta")}}
        ),
        "nested non-tensor": lambda td, t: td.update(
            {"n": {"y": t(*td.batch_size), "s": "text"}}
        ),
        "nested tuple key": lambda td, t: td.update(
            {"n": {("m", "w"): t(*td.batch_size)}}
        ),
        "nested lazy stack": lambda td, t: td.update(
            {"n": lazy_stack([TensorDict(y=t()) for _ in range(3)])}
            if td.batch_size == (3,)
            else {"n": {"y": t(*td.batch_size)}}
        ),
        "nested value, then a non-tensor": lambda td, t: td.update(
            {"n": {"y": t(*td.batch_size)}, "s": "text"}
        ),
        "nested value, then a bad shape": lambda td, t: td.update(
            {"n": {"y": t(*td.batch_size)}, "bad": t(5, 1), "after": t(*td.batch_size)}
        ),
        "nested value, then another device": lambda td, t: td.update(
            {"n": {"y": t(*td.batch_size)}, "m2": t(*td.batch_size).to("meta")}
        ),
        "nested value, then a parameter": lambda td, t: td.update(
            {"n": {"y": t(*td.batch_size)}, "p": nn.Parameter(t(*td.batch_size))}
        ),
        "nested value, then a tuple key": lambda td, t: td.update(
            {"n": {"y": t(*td.batch_size)}, ("n", "z"): t(*td.batch_size)}
        ),
        "child into itself": lambda td, t: td.update(
            {"n": td.get("n", {}), "new": t(*td.batch_size)}
        ),
        "child into another child": lambda td, t: td.update({"n2": td.get("n", {})}),
        "child as the source": lambda td, t: td.update(
            td.get("n", TensorDict(batch_size=td.batch_size))
        ),
        "grandchild replaced, then merged into": lambda td, t: td.update(
            {"n": {"m": t(*td.batch_size)}, "n2": {"m": {"z": t(*td.batch_size)}}}
        ),
        "grandchild replaced, then read": lambda td, t: td.update(
            {"n2": {"m": t(*td.batch_size)}, "n": td.get("n2", {})}
        ),
        "child read, then written": lambda td, t: td.update(
            {"n": td.get("n2", {}), "n2": {"y": t(*td.batch_size)}}
        ),
        "child written, then read": lambda td, t: td.update(
            {"n2": {"y": t(*td.batch_size)}, "n": td.get("n2", {})}
        ),
        "same dict twice": lambda td, t: td.update(
            {"n": (same := {"y": t(*td.batch_size)}), "n2": same}
        ),
    }

    @pytest.mark.parametrize("fixture", list(_UPDATE_FIXTURES))
    @pytest.mark.parametrize("case", list(_UPDATE_CASES))
    def test_update_fast_path_matches_general(self, fixture, case):
        # TensorDict.update writes a dict or TensorDict of tensors, and merges
        # its sub-tensordicts, without the per-key loop of the general path,
        # which a subclass of TensorDict takes. Both give the same result,
        # errors, warnings, storage sharing and merged sub-tensordicts.
        make, call = self._UPDATE_FIXTURES[fixture], self._UPDATE_CASES[case]
        expected = self._update_outcome(self._SlowPathTensorDict, make, call)
        assert self._update_outcome(TensorDict, make, call) == expected

    @pytest.mark.parametrize(
        "call",
        [
            lambda td: td.update({"n": td["n"], "new": torch.ones(3)}),
            lambda td: td.update({"n": {"y": torch.ones(3)}}),
            lambda td: td.update({"new": torch.ones(3), "n": {"y": torch.ones(3)}}),
            lambda td: td.update(
                TensorDict(n=TensorDict(x=torch.ones(3, 2)), batch_size=[3])
            ),
        ],
    )
    def test_update_fast_path_consolidated(self, call, monkeypatch):
        # An unpickled consolidated tensordict is not locked: update marks it
        # stale as the general path does.
        def outcome():
            td = pickle.loads(
                pickle.dumps(
                    TensorDict(
                        a=torch.zeros(3, 4),
                        n=TensorDict(x=torch.zeros(3, 2), batch_size=[3]),
                        batch_size=[3],
                    ).consolidate()
                )
            )
            assert td.is_consolidated() and not td.is_locked
            call(td)
            return td.is_consolidated(), td["n"].is_consolidated(), td.to_dict()

        fast = outcome()
        monkeypatch.setattr(TensorDict, "_update_fast", TensorDictBase._update_fast)
        general = outcome()
        assert fast[:2] == general[:2]
        assert (TensorDict(fast[2]) == TensorDict(general[2])).all()

    def test_update_fast_path_source_changed_by_a_merge(self):
        # td.update(source), where merging source["n"] into td["n"] writes a
        # string into source["b"]: like the general path, update then writes
        # the string into td["b"], not the tensor that source["b"] held before.
        def make(cls):
            source = TensorDict(
                n=TensorDict(s=TensorDict(b=NonTensorData("text"))),
                b=torch.zeros(3),
            )
            return cls(n=TensorDict(s=source), b=torch.ones(3))

        def call(td, tensor):
            return td.update(td["n", "s"])

        expected = self._update_outcome(self._SlowPathTensorDict, make, call)
        assert expected[0] is True
        assert self._update_outcome(TensorDict, make, call) == expected
        td = make(TensorDict)
        call(td, None)
        assert td["b"] == "text"

    @tensorclass
    class _KeysUpdateTC:
        x: torch.Tensor
        y: torch.Tensor

    @staticmethod
    def _keys_update_source(t, **entries):
        """A source for the tensordicts of test_update__keys_fast_path_matches_general."""
        source = TensorDict(
            a=t(3, 4),
            n=TensorDict(x=t(3, 2), y=t(3), batch_size=[3]),
            batch_size=[3],
        )
        source.set_non_tensor("s", "source")
        source["c"] = t(3, 2)
        for key, value in entries.items():
            if isinstance(value, str) or value is None:
                source.set_non_tensor(key, value)
            else:
                source[key] = value
        return source

    _KEYS_UPDATE_CASES = {
        "tensors": (["a", ("n", "x"), "c"], {}),
        "sub-tensordict key": (["n", "c"], {}),
        "overlapping keys": (["n", ("n", "y"), "a", "a"], {}),
        "invalid key selects all": ([("a", 0)], {}),
        "keys the source lacks": (["a", "nope", ("n", "z")], {}),
        "non-tensor between tensors": (["a", "s", "c"], {}),
        "non-tensor, then a missing key": (["a", "s", "zz", "c"], {"zz": (3,)}),
        "non-tensor, then a bad shape": (["a", "s", "c"], {"c": (3, 5)}),
        "non-tensor into a tensor": (["a", "c"], {"c": "text"}),
        "tensor into a non-tensor": (["a", "s", "c"], {"s": (3,)}),
        "non-tensor, shared dest": (["a", "s", "c"], {"s": None, "share": True}),
        "requires grad between tensors": (["a", "c", "b"], {"b": "grad"}),
        "requires grad, no_grad": (["a", "b", "c"], {"b": "grad", "no_grad": 1}),
        "parameter dest": (["a", "p", "c"], {"p": "param"}),
        "parameter dest, no_grad": (["a", "p"], {"p": "param", "no_grad": 1}),
        "tensorclass": (["a", "tc", "c"], {"tc": "tc"}),
        "tensorclass leaf key": ([("tc", "y"), "c"], {"tc": "tc"}),
        "tensorclass unselected": (["a", "c"], {"tc": "tc"}),
        "lazy stack in the source": (["a", "n", "c"], {"n": "lazy"}),
        "lazy stack in self, unselected": (["a", "c"], {"lz": "lazy dest"}),
        "lazy stack in self, selected": (["a", ("lz", "x")], {"lz": "lazy dest"}),
        "sub-tensordict into a tensor": (["a", "c"], {"c": "td"}),
        "sub-tensordict into a tensor, unselected": (["a"], {"c": "td"}),
        "tensor into a sub-tensordict": (["a", "n"], {"n": (3,)}),
        "TensorDictParams": (["a", "pr", "c"], {"pr": "params"}),
        "TensorDictParams, no_grad": (["a", "pr"], {"pr": "params", "no_grad": 1}),
        "locked": (["a", "n", "s", "c"], {"lock": True}),
        "locked, a bad shape": (["a", "c"], {"c": (3, 5), "lock": True}),
        "dim names": (["a", ("n", "x"), "c"], {"names": True}),
        "forward AD": (["a", "b", "c"], {"b": "dual"}),
    }

    @pytest.mark.parametrize("case", list(_KEYS_UPDATE_CASES))
    def test_update__keys_fast_path_matches_general(self, case, monkeypatch):
        # update_(keys_to_update=...) between exact TensorDicts copies the
        # selected tensors with _foreach_copy_ and writes each other entry
        # (non-tensors, tensors that require grad, Parameters, tensorclasses,
        # lazy stacks) at its place in the order, as the _apply_nest route
        # does: same result, errors, warnings and autograd graph.
        keys, options = self._KEYS_UPDATE_CASES[case]
        options = dict(options)
        share = options.pop("share", False)
        no_grad = options.pop("no_grad", False)
        lock = options.pop("lock", False)
        names = options.pop("names", False)
        dual = "dual" in options.values()
        tc_cls = self._KeysUpdateTC

        def make(cls):
            td = cls(
                a=torch.zeros(3, 4),
                n=TensorDict(x=torch.zeros(3, 2), y=torch.zeros(3), batch_size=[3]),
                batch_size=[3],
            )
            td.set_non_tensor("s", "dest")
            td["c"] = torch.zeros(3, 2)
            td["b"] = torch.zeros(3)
            td["p"] = nn.Parameter(torch.zeros(3))
            td["tc"] = tc_cls(x=torch.zeros(3, 2), y=torch.zeros(3), batch_size=[3])
            td["lz"] = lazy_stack([TensorDict(x=torch.zeros(2)) for _ in range(3)])
            td["pr"] = TensorDictParams(
                TensorDict(w=nn.Parameter(torch.zeros(3)), batch_size=[3])
            )
            if share:
                td.share_memory_()
            if names:
                td.names = ["batch"]
            if lock:
                td.lock_()
            return td

        def entry(t, value):
            if isinstance(value, tuple):
                return t(*value)
            if value == "grad":
                return t(3).requires_grad_() * 2
            if value == "param":
                return t(3)
            if value == "tc":
                return tc_cls(x=t(3, 2), y=t(3), batch_size=[3])
            if value == "lazy":
                return lazy_stack([TensorDict(x=t(2), y=t()) for _ in range(3)])
            if value == "lazy dest":
                return TensorDict(x=t(3, 2), batch_size=[3])
            if value == "td":
                return TensorDict(z=t(3), batch_size=[3])
            if value == "params":
                return TensorDict(w=t(3), batch_size=[3])
            if value == "dual":
                return forward_ad.make_dual(t(3), torch.full((3,), 2.0))
            return value

        tangents = []

        def call(td, t):
            with contextlib.ExitStack() as stack:
                if dual:
                    stack.enter_context(forward_ad.dual_level())
                entries = {key: entry(t, value) for key, value in options.items()}
                source = self._keys_update_source(t, **entries)
                if names:
                    source.names = ["batch"]
                if no_grad:
                    stack.enter_context(torch.no_grad())
                result = td.update_(source, keys_to_update=keys)
                if dual:
                    tangents.append(
                        {
                            key: forward_ad.unpack_dual(leaf).tangent.tolist()
                            for key, leaf in td.items(True, True)
                            if isinstance(leaf, torch.Tensor)
                            and forward_ad.unpack_dual(leaf).tangent is not None
                        }
                    )
                return result

        if dual:
            # the first make_dual warns, in torch: not in the outcomes
            with warnings.catch_warnings(), forward_ad.dual_level():
                warnings.simplefilter("ignore")
                forward_ad.make_dual(torch.zeros(()), torch.zeros(()))
        made = []

        def outcome():
            def tracked(cls):
                made.append(make(cls))
                return made[-1]

            tangents.clear()
            result = self._update_outcome(TensorDict, tracked, call)
            grads = {
                key: (leaf.requires_grad, type(leaf.grad_fn).__name__)
                for key, leaf in made[-1].items(True, True)
                if isinstance(leaf, torch.Tensor)
            }
            return result, grads, list(tangents)

        fast = outcome()
        monkeypatch.setattr(
            tensordict_base, "_keys_update_pairs", lambda *args, **kwargs: -1
        )
        assert outcome() == fast

    @pytest.mark.parametrize("method", ["update_", "copy_"])
    def test_update__forward_ad(self, method):
        # inside a dual level, update_ and copy_ copy each leaf with copy_,
        # which keeps the tangents: _foreach_copy_ has no forward derivative
        # (#2886)
        fwAD = torch.autograd.forward_ad
        with fwAD.dual_level():
            with warnings.catch_warnings():
                # make_dual loads its decompositions with torch.jit.script
                # once, which warns that it is deprecated
                warnings.simplefilter("ignore", FutureWarning)
                warnings.simplefilter("ignore", DeprecationWarning)
                p = fwAD.make_dual(torch.ones(3), torch.full((3,), 5.0))
            td = TensorDict(a=torch.zeros(3), b=torch.zeros(3))
            getattr(td, method)(TensorDict(a=p * 2, b=torch.ones(3)))
            assert (fwAD.unpack_dual(td["a"]).tangent == 10).all()
            assert fwAD.unpack_dual(td["b"]).tangent is None
            assert (td["a"] == 2).all() and (td["b"] == 1).all()

    @pytest.mark.parametrize("method", ["update_", "copy_"])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_update__autograd(self, method, lazy):
        # update_ and copy_ record the graph of a copy_ of each leaf (#2863):
        # torch._foreach_copy_ records none, and marks every leaf as
        # requiring grad.
        def make():
            if lazy:
                return lazy_stack(
                    [TensorDict(a=torch.zeros(3), b=torch.zeros(3)) for _ in range(2)]
                )
            return TensorDict(a=torch.zeros(2, 3), b=torch.zeros(2, 3))

        x = torch.randn(2, 3, requires_grad=True)
        td = make()
        getattr(td, method)(TensorDict(a=x * 2, b=torch.ones(2, 3)))
        assert td["a"].requires_grad and not td["b"].requires_grad
        td["a"].sum().backward()
        assert (x.grad == 2).all()
        assert (td["a"] == x * 2).all() and (td["b"] == 1).all()
        # a destination that requires grad: the graph of copy_ too
        x.grad = None
        getattr(td, method)(TensorDict(a=torch.ones(2, 3), b=x * 3))
        assert td["a"].requires_grad and td["b"].requires_grad
        td["b"].sum().backward()
        assert (x.grad == 3).all()
        # without grad mode, no graph
        td = make()
        with torch.no_grad():
            getattr(td, method)(TensorDict(a=x * 2, b=torch.ones(2, 3)))
        assert not td["a"].requires_grad and not td["b"].requires_grad
        # a leaf that requires grad raises before anything is written
        td = TensorDict(a=torch.zeros(3), b=nn.Parameter(torch.zeros(3)))
        with pytest.raises(RuntimeError, match="leaf Variable that requires grad"):
            getattr(td, method)(TensorDict(a=x[0] * 2, b=torch.ones(3)))
        assert (td["a"] == 0).all() and (td["b"] == 0).all()

    def test_update__intersection_ignores_unknown(self):
        # Without keys_to_update, update_ writes the keys of the source that
        # self has and ignores the others, as long as one key matches.
        td = TensorDict(a=torch.zeros(3), batch_size=[3])
        td.update_({"a": torch.ones(3), "zz": torch.ones(3)})
        assert (td["a"] == 1).all() and "zz" not in td.keys()
        with pytest.raises(KeyError, match="zz"):
            td.update_({"zz": torch.ones(3)})
        with pytest.raises(KeyError, match="zz"):
            td.update_({"a": torch.ones(3), "zz": torch.ones(3)}, keys_to_update=["zz"])
        with pytest.raises(KeyError):
            td.update_at_({"a": torch.ones(()), "zz": torch.ones(())}, 0)
        # a lazy stack at an int index writes into one member, with update_
        lazy = lazy_stack([TensorDict(a=torch.zeros(3), batch_size=[3])] * 2)
        lazy.update_at_({"a": torch.full((3,), 2.0), "zz": torch.ones(3)}, 0)
        assert (lazy[0]["a"] == 2).all() and "zz" not in lazy.keys()
        lazy.update_({"a": torch.ones(2, 3), "zz": torch.ones(2, 3)})
        assert (lazy["a"] == 1).all() and "zz" not in lazy.keys()
        # a sub-tensordict raises
        sub = TensorDict(a=torch.zeros(2, 3), batch_size=[2, 3])._get_sub_tensordict(0)
        with pytest.raises(KeyError, match="zz"):
            sub.update_({"a": torch.ones(3), "zz": torch.ones(3)})

    @staticmethod
    def _flat_td(value, *keys, cls=TensorDict, **kwargs):
        """A flat tensordict of batch size [3]: the i-th key holds ``value + i``."""
        return cls(
            {key: torch.full((3, 2), float(value + i)) for i, key in enumerate(keys)},
            batch_size=[3],
            **kwargs,
        )

    class _PlanTC(TensorClass):
        a: torch.Tensor
        b: torch.Tensor

    # (dest, source) pairs for update_ and copy_: the first five take the one
    # walk of _update_plan for two trees of TensorDicts, the others the
    # general route (#2859).
    _UPDATE__PLAN_CASES = {
        "same keys": lambda flat: (flat(0, "a", "b"), flat(1, "a", "b")),
        "other key order": lambda flat: (flat(0, "a", "b"), flat(1, "b", "a")),
        "nested": lambda flat: (
            flat(0, "a").set("n", flat(0, "x", "y")).set("b", torch.zeros(3, 2)),
            flat(1, "b").set("n", flat(1, "y", "x")).set("a", torch.ones(3, 2)),
        ),
        "nested twice": lambda flat: (
            flat(0, "a").set("n", flat(0, "x").set("m", flat(0, "z"))),
            flat(1, "a").set("n", flat(1, "x").set("m", flat(1, "z"))),
        ),
        "nested empty": lambda flat: (
            flat(0, "a").set("e", TensorDict(batch_size=[3])),
            flat(1, "a").set("e", TensorDict(batch_size=[3])),
        ),
        "only empty nested": lambda flat: (
            TensorDict(e=TensorDict(batch_size=[3]), batch_size=[3]),
            TensorDict(e=TensorDict(batch_size=[3]), batch_size=[3]),
        ),
        "nested extra source key": lambda flat: (
            flat(0, "a").set("n", flat(0, "x")),
            flat(1, "a").set("n", flat(1, "x", "y")),
        ),
        "nested missing source key": lambda flat: (
            flat(0, "a").set("n", flat(0, "x", "y")),
            flat(1, "a").set("n", flat(1, "x")),
        ),
        "nested other key": lambda flat: (
            flat(0, "a").set("n", flat(0, "x")),
            flat(1, "a").set("n", flat(1, "y")),
        ),
        "nested tensor source": lambda flat: (
            flat(0, "a").set("n", flat(0, "x")),
            flat(1, "a").set("n", torch.ones(3, 2)),
        ),
        "nested tensor dest": lambda flat: (
            flat(0, "a").set("n", torch.zeros(3, 2)),
            flat(1, "a").set("n", flat(1, "x")),
        ),
        "nested named": lambda flat: (
            flat(0, "a").set("n", flat(0, "x", names=["t"])),
            flat(1, "a").set("n", flat(1, "x")),
        ),
        "nested lazy stack source": lambda flat: (
            flat(0, "a").set("n", flat(0, "x")),
            flat(1, "a").set("n", lazy_stack(list(flat(1, "x").unbind(0)))),
        ),
        "nested subclass source": lambda flat: (
            flat(0, "a").set("n", flat(0, "x")),
            flat(1, "a").set("n", flat(1, "x", cls=TestGeneric._SlowPathTensorDict)),
        ),
        "nested non-tensor": lambda flat: (
            flat(0, "a").set("n", flat(0, "x").set_non_tensor("s", "x")),
            flat(1, "a").set("n", flat(1, "x").set_non_tensor("s", "y")),
        ),
        "nested grad source": lambda flat: (
            flat(0, "a").set("n", flat(0, "x")),
            flat(1, "a").set("n", flat(1, "x")).apply(lambda t: t.requires_grad_()),
        ),
        "extra source key": lambda flat: (flat(0, "a", "b"), flat(1, "a", "b", "c")),
        "missing source key": lambda flat: (flat(0, "a", "b"), flat(1, "a")),
        "one other key": lambda flat: (flat(0, "a", "b"), flat(1, "a", "c")),
        "no common key": lambda flat: (flat(0, "a"), flat(1, "c")),
        "empty": lambda flat: (flat(0), flat(1)),
        "parameter source": lambda flat: (
            flat(0, "a", "b"),
            flat(1, "a").set("b", nn.Parameter(torch.full((3, 2), 2.0))),
        ),
        "grad source": lambda flat: (
            flat(0, "a", "b"),
            flat(1, "a", "b").apply(lambda t: t.requires_grad_()),
        ),
        "non-tensor source": lambda flat: (
            flat(0, "a").set_non_tensor("s", "x"),
            flat(1, "a").set_non_tensor("s", "y"),
        ),
        "double source": lambda flat: (flat(0, "a", "b"), flat(1, "a", "b").double()),
        "bad shape source": lambda flat: (
            flat(0, "a", "b"),
            TensorDict(a=torch.ones(3, 5), b=torch.ones(3, 2), batch_size=[3]),
        ),
        "named dest": lambda flat: (
            flat(0, "a", "b", names=["x"]),
            flat(1, "a", "b"),
        ),
        "named source": lambda flat: (
            flat(0, "a", "b"),
            flat(1, "a", "b", names=["x"]),
        ),
        "locked dest": lambda flat: (flat(0, "a", "b").lock_(), flat(1, "a", "b")),
        "subclass dest": lambda flat: (
            flat(0, "a", "b", cls=TestGeneric._SlowPathTensorDict),
            flat(1, "a", "b"),
        ),
        "subclass source": lambda flat: (
            flat(0, "a", "b"),
            flat(1, "a", "b", cls=TestGeneric._SlowPathTensorDict),
        ),
        "tensorclass source": lambda flat: (
            flat(0, "a", "b"),
            TestGeneric._PlanTC(
                a=torch.ones(3, 2), b=torch.full((3, 2), 2.0), batch_size=[3]
            ),
        ),
        "lazy stack source": lambda flat: (
            flat(0, "a", "b"),
            lazy_stack(list(flat(1, "a", "b").unbind(0))),
        ),
        "lazy stack dest": lambda flat: (
            lazy_stack(list(flat(0, "a", "b").unbind(0))),
            flat(1, "a", "b"),
        ),
        "params dest": lambda flat: (
            TensorDictParams(flat(0, "a", "b"), no_convert=True),
            flat(1, "a", "b"),
        ),
        "memmap dest": lambda flat: (flat(0, "a", "b").memmap_(), flat(1, "a", "b")),
    }

    @staticmethod
    def _update__plan_outcome(case, op):
        dest, source = TestGeneric._UPDATE__PLAN_CASES[case](TestGeneric._flat_td)
        ptrs = {key: _leaf_storage(dest, key) for key in dest.keys(True, True)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if op == "row":
                    dest[1].copy_(source[1])
                    result = dest
                else:
                    result = getattr(dest, op)(source)
            except Exception as err:
                outcome = (type(err), str(err))
            else:
                outcome = result is dest
        state = {
            key: (
                type(leaf).__name__,
                leaf.tolist(),
                leaf.requires_grad,
                _leaf_storage(dest, key) == ptrs[key],
            )
            if isinstance(leaf, torch.Tensor)
            else (type(leaf).__name__, leaf.tolist())
            for key, leaf in dest.items(True, True, is_leaf=is_leaf_nontensor)
        }
        return (
            outcome,
            state,
            list(state),
            dest.is_locked,
            dest.names if dest._has_names() else None,
            [(w.category, str(w.message)) for w in caught],
        )

    @pytest.mark.parametrize("op", ["update_", "copy_", "row"])
    @pytest.mark.parametrize("case", list(_UPDATE__PLAN_CASES))
    def test_update__plan_fast_path_matches_general(self, case, op, monkeypatch):
        # _update_plan pairs the leaves of two flat TensorDicts in one walk:
        # update_ and copy_ give the same results, errors and warnings as on
        # the general route, which _is_dense_node gates (#2859).
        fast = self._update__plan_outcome(case, op)
        monkeypatch.setattr(TensorDict, "_is_dense_node", lambda self: False)
        assert self._update__plan_outcome(case, op) == fast

    def test_update__plan_flat_one_walk(self, monkeypatch):
        # copy_ and update_ between two flat TensorDicts of tensors pair the
        # leaves without listing them with _items_list (#2859).
        store = TensorDict(
            policy=torch.zeros(4, 3, 5), critic=torch.zeros(4, 3, 2), batch_size=[4, 3]
        )
        obs = TensorDict(critic=torch.ones(3, 2), policy=torch.full((3, 5), 2.0))
        obs.batch_size = [3]

        def items_list(*args, **kwargs):
            raise AssertionError("_items_list was called")

        monkeypatch.setattr(TensorDictBase, "_items_list", items_list)
        store[1].copy_(obs)
        store[2].update_(obs)
        assert (store[1:3]["critic"] == 1).all() and (store[1:3]["policy"] == 2).all()
        assert (store[0] == 0).all() and (store[3] == 0).all()

    def test_update__plan_nested_one_walk(self, monkeypatch):
        # copy_ and update_ between two trees of TensorDicts of tensors, such
        # as TorchRL transitions with a "next" entry, pair the leaves without
        # listing them with _items_list (#2859).
        def transition(value):
            return TensorDict(
                obs=torch.full((3, 4), value),
                next=TensorDict(
                    obs=torch.full((3, 4), value + 1),
                    reward=torch.full((3, 1), value + 2),
                    batch_size=[3],
                ),
                batch_size=[3],
            )

        dest, source = transition(0.0), transition(10.0)

        def items_list(*args, **kwargs):
            raise AssertionError("_items_list was called")

        monkeypatch.setattr(TensorDictBase, "_items_list", items_list)
        dest.copy_(source)
        assert (dest == transition(10.0)).all()
        dest.update_(transition(20.0))
        assert (dest == transition(20.0)).all()

    def test_sub_td_update_overwrites_in_place(self):
        # A sub-tensordict writes the entries it has into its source, in place,
        # and adds the others to the source
        source = TensorDict(
            a=torch.zeros(2, 3), n=TensorDict(x=torch.zeros(2, 3)), batch_size=[2, 3]
        )
        ptr = source["a"].data_ptr()
        sub = source._get_sub_tensordict(0)
        assert sub.update({"a": torch.ones(3), "new": torch.ones(3)}) is sub
        assert source["a"].data_ptr() == ptr
        assert (source["a"][0] == 1).all() and (source["a"][1] == 0).all()
        assert (source["new"][0] == 1).all() and (source["new"][1] == 0).all()
        assert sub.update(n={"x": torch.ones(3)}, b=torch.ones(3)) is sub
        assert (source["n", "x"][0] == 1).all() and (source["b"][0] == 1).all()
        assert sub.update({("n", "x"): torch.full((3,), 2.0)}) is sub
        assert (source["n", "x"][0] == 2).all()
        sub.update(
            TensorDict(a=torch.full((3,), 3.0), batch_size=[3]),
            n={"x": torch.full((3,), 3.0)},
        )
        assert (source["a"][0] == 3).all() and (source["n", "x"][0] == 3).all()
        assert sub.update(None) is sub and sub.update(sub) is sub
        assert sub.update({"a": torch.zeros(3)}, keys_to_update=[]) is sub
        assert (source["a"][0] == 3).all()
        sub.update({"a": torch.zeros(3), "b": torch.zeros(3)}, keys_to_update=["b"])
        assert (source["a"][0] == 3).all() and (source["b"][0] == 0).all()
        with pytest.raises(ValueError, match="incompatible object"):
            sub.update({"n": torch.zeros(3)})
        with pytest.raises(IndexError):
            sub.update({1: torch.zeros(3)})
        exclusive = lazy_stack(
            [TensorDict(a=torch.zeros(3)), TensorDict(b=torch.zeros(3))]
        )
        with pytest.raises(RuntimeError, match="exclusive keys"):
            exclusive._get_sub_tensordict(0).update({"a": torch.ones(3)})
        source.lock_()
        with pytest.raises(RuntimeError, match="locked"):
            sub.update({"a": torch.ones(3)})
        assert sub.update({"a": torch.ones(3)}, inplace=True) is sub

    def test_sub_td_update_nested_inplace_locked(self):
        # update(inplace=True) writes the nested entries of a locked source,
        # as a sub-tensordict and as a nested key (#2745)
        source = TensorDict(
            a=torch.zeros(2, 3), n=TensorDict(x=torch.zeros(2, 3)), batch_size=[2, 3]
        ).lock_()
        sub = source._get_sub_tensordict(0)
        sub.update({"a": torch.ones(3), "n": {"x": torch.ones(3)}}, inplace=True)
        assert (source["a"][0] == 1).all() and (source["n", "x"][0] == 1).all()
        sub.update({("n", "x"): torch.full((3,), 2.0)}, inplace=True)
        assert (source["n", "x"][0] == 2).all() and (source["n", "x"][1] == 0).all()
        # a lazy stack of views of locked tensordicts writes through them
        parents = [source, source.clone().lock_()]
        stack = lazy_stack([p._get_sub_tensordict(1) for p in parents])
        stack[0] = TensorDict(a=torch.ones(3), n={"x": torch.ones(3)}, batch_size=[3])
        assert (source["a"][1] == 1).all() and (source["n", "x"][1] == 1).all()
        assert (parents[1]["n", "x"][1] == 0).all()

    def test_sub_td_update_at_arguments(self):
        # _SubTensorDict.update_at_ and update_ take the arguments of
        # TensorDictBase, and still discard_idx_attr and input_dict
        source = TensorDict(a=torch.zeros(2, 3), batch_size=[2, 3])
        sub = source._get_sub_tensordict(0)
        assert sub.update_at_({"a": torch.ones(())}, 0, True) is sub
        assert sub.update_at_({"a": torch.ones(())}, 0, clone=True) is sub
        assert source["a"][0, 0] == 1 and source["a"][0, 1] == 0
        with pytest.warns(
            DeprecationWarning,
            match=r"discard_idx_attr=.*removed in TensorDict 0\.17.*get_parent_tensordict",
        ) as record:
            assert (
                sub.update_at_({"a": torch.ones(())}, 1, discard_idx_attr=True) is sub
            )
        assert record[0].filename == __file__
        assert (source["a"][1] == 1).all()
        with pytest.warns(DeprecationWarning, match="discard_idx_attr"):
            sub.update_at_({"a": torch.ones(())}, 0, discard_idx_attr=False)
        assert source["a"][0, 0] == 1 and source["a"][0, 1] == 0
        with pytest.warns(
            DeprecationWarning,
            match=r"update_at_\(input_dict=.*0\.17.*update_at_\(input_dict_or_td=",
        ) as record:
            sub.update_at_(input_dict={"a": torch.ones(())}, idx=0)
        assert record[0].filename == __file__
        assert sub.update_at_({"a": torch.zeros(())}, 0, keys_to_update=[]) is sub
        assert source["a"][0, 0] == 1
        with pytest.raises(TypeError, match="Expected value"):
            sub.update_at_({"a": "text"}, 0)
        with pytest.warns(
            DeprecationWarning,
            match=r"update_\(input_dict=.*0\.17.*update_\(input_dict_or_td=",
        ) as record:
            assert sub.update_(input_dict={"a": torch.full((3,), 2.0)}) is sub
        assert record[0].filename == __file__
        assert (source["a"][0] == 2).all()
        assert sub.update_(a=torch.full((3,), 3.0)) is sub
        assert (source["a"][0] == 3).all()
        assert sub.update_({"a": torch.full((3,), 4.0)}, a=torch.full((3,), 5.0)) is sub
        assert (source["a"][0] == 5).all()
        sub.update_(
            TensorDict(a=torch.full((3,), 6.0), batch_size=[3]), a=torch.ones(3)
        )
        assert (source["a"][0] == 1).all()
        assert sub.update_(None) is sub
        assert sub.update_(input_dict_or_td={"a": torch.full((3,), 7.0)}) is sub
        assert (source["a"][0] == 7).all()
        assert sub.update_at_({"a": torch.full((3,), 8.0)}, ()) is sub
        assert (source["a"][0] == 8).all()
        # a NumPy index is not compared with ()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert sub.update_at_({"a": torch.ones(())}, np.array([0, 1])) is sub
            assert sub.update_at_({"a": torch.ones(())}, np.int64(2)) is sub
        assert (source["a"][0] == 1).all()
        # discard_idx_attr writes leaf by leaf, so a lazy source broadcasts
        lazy = lazy_stack(
            [TensorDict(a=torch.zeros(3, 2), batch_size=[3]) for _ in range(2)], 1
        )
        sub = lazy._get_sub_tensordict((slice(None), 0))
        with pytest.warns(DeprecationWarning, match="discard_idx_attr"):
            sub.update_at_({"a": torch.tensor(4.0)}, 0, discard_idx_attr=True)
        assert (lazy["a"][0] == 4).all() and (lazy["a"][1:] == 0).all()
        # only a _SubTensorDict takes discard_idx_attr
        for td in (source, lazy, source.permute(1, 0)):
            with pytest.raises(TypeError, match="discard_idx_attr"):
                td.update_at_({"a": torch.ones(())}, 0, discard_idx_attr=True)

    def test_eager_mutation_custom_validation(self):
        class ValidatedTD(TensorDict):
            def _validate_value_generic(self, value, **kwargs):
                return super()._validate_value_generic(value, **kwargs) + 1

        td = ValidatedTD({}, [2], device="cpu")
        value = torch.zeros(2)
        td.set("x", value)
        td.update({"y": value})
        assert (td["x"] == 1).all() and (td["y"] == 1).all()

    def test_eager_set_validation(self):
        # Only exact tensors that match the batch size and device are stored
        # without validation; every other value is validated as before.
        value = torch.zeros(2, 3)
        for td in (
            TensorDict(batch_size=[2]),
            TensorDict(batch_size=[2], device="cpu"),
            TensorDict({"x": value}, batch_size=[2]),
        ):
            td.set("x", value)
            param = nn.Parameter(value.clone())
            td.set("p", param)
            nested = torch.nested.nested_tensor([torch.zeros(3), torch.zeros(4)])
            td.set("n", nested)
            assert td["x"] is value and td["p"] is param and td["n"] is nested
            with pytest.raises(RuntimeError, match="batch dimension mismatch"):
                td.set("bad", torch.zeros(3))
            assert "bad" not in td.keys()
            td.set("s", "abc")
            assert td["s"] == "abc" and is_non_tensor(td.get("s"))
            # a tensordict value gives its leading dim names to its parent
            sub = TensorDict({"a": value}, [2, 3], names=["a", "b"])
            td.set("sub", sub)
            assert td["sub"].batch_size == (2, 3) and td.names == ["a"]
            assert td["sub", "a"] is value
        meta = TensorDict(batch_size=[2], device="meta")
        meta.set("x", value)
        assert meta["x"].device.type == "meta"
        batch_free = TensorDict()
        batch_free.set("x", value)
        assert batch_free["x"] is value

    @pytest.mark.parametrize("operand", [None, "abc"])
    def test_comparison_unsupported_operand(self, operand):
        # An operand that is not a tensor, number, dict or tensor collection
        # gives a bool: True for != and ^, False for the other ops
        td = TensorDict(a=torch.zeros(3, dtype=torch.long), batch_size=[3])
        for op in ("__eq__", "__lt__", "__le__", "__gt__", "__ge__", "__or__"):
            assert getattr(td, op)(operand) is False
        for op in ("__ne__", "__xor__"):
            assert getattr(td, op)(operand) is True

    @pytest.mark.parametrize("op", ["__ne__", "__xor__", "__or__"])
    def test_comparison_mismatched_keys_keyerror(self, op):
        td = TensorDict(a=torch.zeros(3, dtype=torch.long), batch_size=[3])
        other = TensorDict(b=torch.zeros(3, dtype=torch.long), batch_size=[3])
        with pytest.raises(KeyError, match="keys in TensorDict.* mismatch, got"):
            getattr(td, op)(other)

    @pytest.mark.parametrize("op", ["__eq__", "__lt__", "__le__", "__gt__", "__ge__"])
    def test_comparison_mismatched_keys(self, op):
        td = TensorDict(a=torch.zeros(3, dtype=torch.long), batch_size=[3])
        other = TensorDict(b=torch.zeros(3, dtype=torch.long), batch_size=[3])
        with pytest.raises(KeyError, match="keys in tensordicts mismatch. The first"):
            getattr(td, op)(other)

    @pytest.mark.parametrize("nested", [False, True])
    def test_where_out_without_leaves(self, nested):
        # A result without leaves leaves out alone, even if it is locked
        td = TensorDict({"a": TensorDict(batch_size=[3])} if nested else {}, [3])
        mask = torch.tensor([True, False, True])
        out = td.clone().lock_()
        assert td.where(mask, 0, out=out) is out

    def test_torch_where_td_other_tensor_input(self):
        # torch.where(mask, tensor, td): the tensor broadcasts against each leaf
        # and the mask over the batch dims, as in torch.where(~mask, td, tensor)
        td = TensorDict(
            a=torch.randn(3, 4, 2),
            nested=TensorDict(b=torch.randn(3, 4, 5, 2), batch_size=[3, 4]),
            batch_size=[3, 4],
        )
        value = torch.randn(2)
        for mask in (torch.tensor([True, False, True]), torch.rand(3, 4) > 0.5):
            result = torch.where(mask, value, td)
            assert result.batch_size == td.batch_size
            for key in ("a", ("nested", "b")):
                leaf = td.get(key)
                expected = torch.where(expand_as_right(mask, leaf), value, leaf)
                assert torch.equal(result.get(key), expected)
            assert_allclose_td(result, torch.where(~mask, td, value))

    def test_apply_out_batch_size_list(self):
        # a batch_size list equal to out.batch_size is accepted
        td = TensorDict(a=torch.ones(3), b={"c": torch.ones(3, 2)}, batch_size=[3])
        out = td.clone()
        assert td.apply(lambda x: x * 2, batch_size=[3], out=out) is out
        assert_allclose_td(out, td * 2)

    @pytest.mark.parametrize("nested_keys", [False, True])
    def test_named_apply_out(self, nested_keys):
        # named_apply writes into out= and returns it, as apply does
        td = TensorDict(a=torch.zeros(3), b={"c": torch.zeros(3, 2)}, batch_size=[3])
        out = td.apply(lambda x: torch.full_like(x, 7.0))
        names = []

        def fn(name, x):
            names.append(name)
            return x + 1

        result = td.named_apply(fn, nested_keys=nested_keys, out=out)
        assert result is out
        assert_allclose_td(out, td.apply(lambda x: x + 1))
        assert set(names) == ({"a", ("b", "c")} if nested_keys else {"a", "c"})

    @pytest.mark.parametrize("named", [False, True])
    def test_apply_is_leaf(self, named):
        # is_leaf is a documented keyword of apply and named_apply: it decides
        # which entries fn is called on, here non-tensor data too
        for cls in (TensorDictBase, TensorDict, TensorDictParams):
            meth = getattr(cls, "named_apply" if named else "apply")
            param = inspect.signature(meth).parameters["is_leaf"]
            assert param.kind is inspect.Parameter.KEYWORD_ONLY
            assert param.default is None

        @tensorclass
        class MyClass:
            x: torch.Tensor
            s: str

        td = TensorDict(a=torch.zeros(3), b="a string", batch_size=[3])
        tc = MyClass(x=torch.zeros(3), s="a string", batch_size=[3])
        params = TensorDictParams(
            TensorDict(a=torch.zeros(3), b={"c": torch.zeros(3)}, batch_size=[3])
        )
        cases = [
            (td, None, ["Tensor"]),
            (td, is_leaf_nontensor, ["Tensor", "NonTensorData"]),
            (
                lazy_stack([td[0], td[1]]),
                is_leaf_nontensor,
                ["Tensor", "NonTensorData"] * 2,
            ),
            (tc, is_leaf_nontensor, ["Tensor", "NonTensorData"]),
            (params, lambda cls: True, ["Parameter", "TensorDict"]),
        ]
        seen = []
        for data, is_leaf, expected in cases:
            seen.clear()
            if named:
                data.named_apply(
                    lambda name, x: seen.append(type(x).__name__), is_leaf=is_leaf
                )
            else:
                data.apply(lambda x: seen.append(type(x).__name__), is_leaf=is_leaf)
            assert seen == expected

    def test_apply_out_batch_size_list_multithread(self):
        td = TensorDict(a=torch.ones(3), b=torch.ones(3, 2), batch_size=[3])
        out = td.clone()
        result = td._fast_apply(lambda x: x * 2, batch_size=[3], out=out, num_threads=2)
        assert result is out
        assert_allclose_td(out, td * 2)

    @pytest.mark.parametrize(
        "named,nested_keys", [[False, False], [True, False], [True, True]]
    )
    @pytest.mark.parametrize("filter_empty", [False, True, None])
    @pytest.mark.parametrize("drop", [False, True])
    @pytest.mark.parametrize("mixed", [False, True])
    def test_fast_apply_dense_tree(
        self, named, nested_keys, filter_empty, drop, mixed, monkeypatch
    ):
        # A tree of TensorDict nodes and tensors is rebuilt without the
        # generic _apply_nest walker, with the same result. Other entries
        # (here a NonTensorData and a lazy stack) take _apply_nest.
        b = TensorDict(
            c=torch.ones(3, 2), d={"e": torch.ones(3, 2, 1)}, batch_size=[3, 2]
        )
        td = TensorDict(a=torch.zeros(3), b=b, empty={}, batch_size=[3], device="cpu")
        if mixed:
            td["b", "s"] = NonTensorData("s", batch_size=[3, 2])
            td["b", "l"] = lazy_stack(
                [TensorDict(f=torch.ones(2, 4), batch_size=[2], device="cpu")] * 3
            )
            assert type(td["b", "l"]) is LazyStackedTensorDict
        td.lock_()
        calls = []

        def fn(*args):
            calls.append(args[:-1])
            return None if drop else args[-1] + 1

        kwargs = {
            "named": named,
            "nested_keys": nested_keys,
            "filter_empty": filter_empty,
            "propagate_lock": True,
        }
        # checked=False takes _apply_nest
        expected = td._fast_apply(fn, checked=False, **kwargs)
        expected_calls, calls[:] = list(calls), []
        neg, mul, add = -td, td * 2, td + td

        prefixes = []

        def record(cls):
            apply_nest = cls._apply_nest

            def record_apply_nest(self, *args, **kwargs):
                prefixes.append(kwargs.get("prefix", ()))
                return apply_nest(self, *args, **kwargs)

            monkeypatch.setattr(cls, "_apply_nest", record_apply_nest)

        record(TensorDictBase)
        record(LazyStackedTensorDict)
        result = td._fast_apply(fn, **kwargs)
        assert calls == expected_calls
        if expected is None:
            assert result is None
            result = expected = td.empty()
        results = [(result, expected), (-td, neg), (td * 2, mul), (td + td, add)]
        # Of the recorded _apply_nest copies (TensorDictBase and lazy stacks),
        # only the lazy stack's ran; the NonTensorData takes its own copy,
        # which is not recorded.
        assert bool(prefixes) is mixed
        assert all(prefix[:2] == ("b", "l") for prefix in prefixes)
        for res, exp in results:
            assert list(res.keys(True)) == list(exp.keys(True))
            nodes = [k for k in res.keys(True) if is_tensor_collection(res[k])]
            for r, e in [(res, exp)] + [(res[k], exp[k]) for k in nodes]:
                assert type(r) is type(e)
                assert r.batch_size == e.batch_size
                assert r.device == e.device
                assert r.names == e.names
                assert r.is_locked == e.is_locked
            assert_allclose_td(res, exp)

    def test_norm_out(self):
        td = TensorDict(a=torch.ones(3), b={"c": torch.ones(3, 2)}, batch_size=[3])
        out = td.norm().zero_()
        assert td.norm(out=out) is out
        assert_allclose_td(out, td.norm())

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
def _path_td_sync(monkeypatch):
    def _sync_td(self):
        global _SYNC_COUNTER
        _SYNC_COUNTER += 1
        super(TensorDict, self)._sync_all()

    # monkeypatch deletes the attribute on teardown, so TensorDict inherits
    # _sync_all again. Assigning the old value back would leave an override
    # in TensorDict.__dict__, which TestOneImplementation then reports.
    monkeypatch.setattr(TensorDict, "_sync_all", _sync_td)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
