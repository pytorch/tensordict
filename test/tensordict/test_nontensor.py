# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import collections
import ctypes
import faulthandler
import importlib.util
import math
import operator
import os
import pickle
import platform
import sys
import threading
import warnings
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from tensordict import (
    _utils_key_json,
    capture_non_tensor_stack,
    lazy_stack,
    LazyStackedTensorDict,
    MemoryMappedTensor,
    set_capture_non_tensor_stack,
    TensorClass,
    tensorclass,
    TensorDict,
    UnbatchedTensor,
)
from tensordict._utils_options import _set_capture_non_tensor_stack
from tensordict.tensorclass import MetaData, NonTensorData, NonTensorStack
from tensordict.utils import (
    _check_recursive_properties,
    _is_non_tensor,
    _pass_through,
    _pass_through_cls,
    is_non_tensor,
    LinkedList,
    set_list_to_stack,
)
from torch import multiprocessing as mp

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.tensordict.test._utils_internal import (
        DummyPicklableClass,
        get_available_devices,
        is_npu_available,
        legacy_lazy_mode,
    )
else:
    IS_FB = False
    from _utils_internal import (
        DummyPicklableClass,
        get_available_devices,
        is_npu_available,
        legacy_lazy_mode,
    )


_has_streaming = importlib.util.find_spec("streaming", None) is not None

try:
    import h5py  # noqa

    _has_h5py = True
except ImportError:
    _has_h5py = False

_has_onnx = importlib.util.find_spec("onnxruntime", None) is not None
_has_orjson = importlib.util.find_spec("orjson", None) is not None

PYTORCH_TEST_FBCODE = os.getenv("PYTORCH_TEST_FBCODE")

_IS_OSX = platform.system() == "Darwin"
_IS_WINDOWS = sys.platform == "win32"

TD_BATCH_SIZE = 4

_PICKLE_LOAD_EXECUTED = False


def _record_pickle_load():
    global _PICKLE_LOAD_EXECUTED
    _PICKLE_LOAD_EXECUTED = True
    return "loaded"


class _PickleLoadMarker:
    def __reduce__(self):
        return _record_pickle_load, ()


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


class TestNonTensorData:
    @tensorclass
    class SomeTensorClass:
        a: str
        b: torch.Tensor

    @pytest.fixture
    def non_tensor_data(self):
        return TensorDict(
            {
                "1": 1,
                "nested": {
                    "int": NonTensorData(3, batch_size=[]),
                    "str": NonTensorData("a string!", batch_size=[]),
                    "bool": NonTensorData(True, batch_size=[]),
                },
            },
            batch_size=[],
        )

    @set_capture_non_tensor_stack(False)
    def test_consolidate_nested(self):
        import pickle

        td = TensorDict(
            a=TensorDict(b=self.SomeTensorClass(a="a string!", b=torch.randn(10))),
            c=TensorDict(d=NonTensorData("another string!")),
        )
        td = lazy_stack([td.clone(), td.clone()])
        td = lazy_stack([td.clone(), td.clone()], -1)

        tdc = td.consolidate()

        assert (tdc == td).all()

        tdr = pickle.loads(pickle.dumps(td))
        assert (tdr == td).all()

        tdcr = pickle.loads(pickle.dumps(tdc))
        assert (tdcr == td).all()

    @pytest.mark.parametrize("canonical", [False, True])
    def test_contiguous_preserves_nontensor_stack(self, canonical):
        # Regression test: contiguous() used to materialize NonTensorStack
        # entries into empty TensorDicts, silently dropping the data.
        tds = [
            TensorDict(query=NonTensorData(f"abc{i}"), x=torch.zeros(()))
            for i in range(3)
        ]
        ls = lazy_stack(tds)
        assert isinstance(ls.get("query"), NonTensorStack)

        c = ls.contiguous(canonical=canonical)
        assert not isinstance(c, LazyStackedTensorDict)
        query = c.get("query")
        assert isinstance(query, NonTensorStack)
        assert query.tolist() == ["abc0", "abc1", "abc2"]
        # indexing round-trip
        for i in range(3):
            assert c[i]["query"] == f"abc{i}"

        # a NonTensorStack is its own contiguous form
        stack = NonTensorStack("a", "b")
        assert stack.contiguous(canonical=canonical) is stack

        # nested case
        ls_nested = lazy_stack(
            [
                TensorDict(sub=TensorDict(q=NonTensorData(f"s{i}"), y=torch.ones(2)))
                for i in range(2)
            ]
        )
        c_nested = ls_nested.contiguous(canonical=canonical)
        assert c_nested["sub", "q"] == ["s0", "s1"]

    @pytest.mark.parametrize(
        "container",
        ["stack", "stack_dim1", "dense", "dense_dim1", "lazy", "lazy_dim1"],
    )
    @pytest.mark.parametrize(
        "op",
        [
            lambda x: x.flip(1),
            lambda x: torch.flip(x, [0, 1]),
            lambda x: x.roll(1, 1),
            lambda x: torch.rot90(x, 1, [0, 1]),
            lambda x: x.narrow(1, 1, 2),
            lambda x: torch.tile(x, (2, 1)),
            lambda x: x.broadcast_to((2, 2, 3)),
            lambda x: x.reshape(3, 2),
            lambda x: torch.reshape(x, (3, 2)),
        ],
        ids=[
            "flip",
            "torch.flip",
            "roll",
            "rot90",
            "narrow",
            "tile",
            "broadcast_to",
            "reshape",
            "torch.reshape",
        ],
    )
    def test_shape_ops_preserve_non_tensor_values(self, op, container):
        values = [[0, 1, 2], [3, 4, 5]]
        # stacking the columns along dim=1 yields the same values but a stack
        # whose stack_dim is not the first dimension
        columns = [list(col) for col in zip(*values)]
        expected = op(torch.tensor(values))
        if container == "stack":
            result = op(NonTensorStack.from_list(values))
            assert isinstance(result, NonTensorStack)
            assert result.tolist() == expected.tolist()
            return
        if container == "stack_dim1":
            stack = torch.stack(
                [NonTensorStack.from_list(col) for col in columns], dim=1
            )
            assert stack.stack_dim == 1
            result = op(stack)
            assert isinstance(result, NonTensorStack)
            assert result.tolist() == expected.tolist()
            return
        if container == "lazy":
            td = lazy_stack(
                [
                    TensorDict(
                        query=NonTensorStack.from_list(row),
                        x=torch.tensor(row),
                        batch_size=[3],
                    )
                    for row in values
                ]
            )
        elif container in ("dense_dim1", "lazy_dim1"):
            stack_fn = lazy_stack if container == "lazy_dim1" else torch.stack
            td = stack_fn(
                [
                    TensorDict(
                        query=NonTensorStack.from_list(col),
                        x=torch.tensor(col),
                        batch_size=[2],
                    )
                    for col in columns
                ],
                dim=1,
            )
            assert td.get("query").stack_dim == 1
        else:
            td = TensorDict(
                query=NonTensorStack.from_list(values),
                x=torch.tensor(values),
                batch_size=[2, 3],
            )
        result = op(td)
        assert isinstance(result.get("query"), NonTensorStack)
        assert result.get("query").tolist() == expected.tolist()
        assert (result.get("x") == expected).all()

    def test_shape_ops_non_tensor_stack_legacy_lazy(self):
        # the entries move in the legacy lazy mode too, where permute is lazy
        values = [[0, 1, 2], [3, 4, 5]]
        columns = [NonTensorStack.from_list(list(col)) for col in zip(*values)]
        stack = torch.stack(columns, dim=1)
        assert stack.stack_dim == 1
        with legacy_lazy_mode():
            assert stack.flip(1).tolist() == torch.tensor(values).flip(1).tolist()

    def test_shape_ops_non_tensor_stack_semantics(self):
        stack = NonTensorStack("walk", "jump", "stand")
        # flip copies the entries, as it copies tensors
        flipped = stack.flip(0)
        flipped[0] = "hop"
        assert flipped.tolist() == ["hop", "jump", "walk"]
        assert stack.tolist() == ["walk", "jump", "stand"]
        # narrow is a slice, so the entries are shared like a view
        narrowed = stack.narrow(0, -2, 2)
        narrowed[0] = "hop"
        assert stack.tolist() == ["walk", "hop", "stand"]
        with pytest.raises(RuntimeError, match="exceeds dimension size"):
            stack.narrow(0, 2, 2)

        # an in-place roll moves the non-tensor entries along with the tensors
        td = TensorDict(
            query=NonTensorStack("a", "b", "c"), x=torch.arange(3), batch_size=[3]
        )
        assert td.roll(1, 0, inplace=True) is td
        assert td.get("query").tolist() == ["c", "a", "b"]
        assert td.get("x").tolist() == [2, 0, 1]
        # the same when the stack dim of the entries is not the first one
        td = torch.stack(
            [
                TensorDict(
                    query=NonTensorStack("a", "b"),
                    x=torch.tensor([0, 1]),
                    batch_size=[2],
                ),
                TensorDict(
                    query=NonTensorStack("c", "d"),
                    x=torch.tensor([2, 3]),
                    batch_size=[2],
                ),
            ],
            dim=1,
        )
        assert td.get("query").stack_dim == 1
        assert td.roll(1, 0, inplace=True) is td
        assert td.get("query").tolist() == [["b", "d"], ["a", "c"]]
        assert td.get("x").tolist() == [[1, 3], [0, 2]]
        # and in a nested td with an extra batch dim, when the flattened batch
        # is rolled
        td = TensorDict(
            nested=TensorDict(
                query=NonTensorStack.from_list([["a", "b"], ["c", "d"], ["e", "f"]]),
                x=torch.arange(6).view(3, 2),
                batch_size=[3, 2],
            ),
            batch_size=[3],
        )
        assert td.roll(1, inplace=True) is td
        assert td.get(("nested", "query")).tolist() == [
            ["e", "f"],
            ["a", "b"],
            ["c", "d"],
        ]
        assert td.get(("nested", "x")).tolist() == [[4, 5], [0, 1], [2, 3]]

        grid = NonTensorStack.from_list([["a", "b", "c"], ["d", "e", "f"]])
        assert grid.reshape(2, 3) is grid
        assert grid.reshape(6, 1).tolist() == [["a"], ["b"], ["c"], ["d"], ["e"], ["f"]]
        with pytest.raises(
            RuntimeError, match=r"invalid shape \[4, 2\] for a batch of 6"
        ):
            grid.reshape(4, 2)

    def test_non_tensor_stack_shape_op_types(self):
        values = [["a", "b", "c"], ["d", "e", "f"]]
        ref = np.array(values, dtype=object)
        calls = {
            "flip": (lambda x: x.flip(1), lambda a: np.flip(a, 1)),
            "roll": (lambda x: x.roll(1, 1), lambda a: np.roll(a, 1, 1)),
            "roll()": (lambda x: x.roll(1), lambda a: np.roll(a, 1)),
            "rot90": (lambda x: x.rot90(1, (0, 1)), lambda a: np.rot90(a, 1, (0, 1))),
            "tile": (lambda x: x.tile((2, 1)), lambda a: np.tile(a, (2, 1))),
            "narrow": (lambda x: x.narrow(1, 1, 2), lambda a: a[:, 1:3]),
            "broadcast_to": (
                lambda x: x.broadcast_to((2, 2, 3)),
                lambda a: np.broadcast_to(a, (2, 2, 3)),
            ),
            "expand": (
                lambda x: x.expand(2, 2, 3),
                lambda a: np.broadcast_to(a, (2, 2, 3)),
            ),
            "reshape(3, 2)": (lambda x: x.reshape(3, 2), lambda a: a.reshape(3, 2)),
            "reshape(6)": (lambda x: x.reshape(6), lambda a: a.reshape(6)),
            "reshape(2, 3, 1)": (
                lambda x: x.reshape(2, 3, 1),
                lambda a: a.reshape(2, 3, 1),
            ),
        }
        for name, (call, ref_call) in calls.items():
            stack = NonTensorStack.from_list(values)
            out = call(stack)
            assert isinstance(out, NonTensorStack), name
            assert out.tolist() == ref_call(ref).tolist(), name
            assert stack.tolist() == values, name

        # narrow and a reshape that regroups the stack share the entries;
        # the other ops make new ones, which hold the same data
        def leaves(x):
            if isinstance(x, LazyStackedTensorDict):
                return [leaf for m in x.tensordicts for leaf in leaves(m)]
            return [x]

        stack = NonTensorStack.from_list(values)
        ids = {id(leaf) for leaf in leaves(stack)}
        data_ids = {id(leaf.data) for leaf in leaves(stack)}
        for name, (call, _) in calls.items():
            out = leaves(call(stack))
            shared = sum(id(leaf) in ids for leaf in out)
            regroups = ("narrow", "reshape(6)", "reshape(2, 3, 1)")
            assert shared == (len(out) if name in regroups else 0), name
            assert all(id(leaf.data) in data_ids for leaf in out), name

        # broadcast_to expands the members, as expand does: a locked stack
        # broadcast to its own shape gives a locked stack
        stack = NonTensorStack.from_list(values).lock_()
        assert stack.broadcast_to((-1, -1)).is_locked

    def test_non_tensor_stack_reshape_inplace_raises(self):
        stack = NonTensorStack.from_list([["a", "b", "c"], ["d", "e", "f"]])
        with pytest.raises(
            NotImplementedError,
            match="reshape.inplace=True. is not supported on NonTensorStack",
        ):
            stack.reshape(6, inplace=True)
        assert stack.batch_size == (2, 3)

    def test_non_tensor_stack_lazy_stack_types(self):
        out = NonTensorStack.lazy_stack([NonTensorData("a"), NonTensorData("b")])
        assert type(out) is NonTensorStack
        assert out.tolist() == ["a", "b"]
        out = NonTensorStack.lazy_stack(
            [NonTensorStack("a", "b"), NonTensorStack("c", "d")], 1
        )
        assert type(out) is NonTensorStack
        assert out.tolist() == [["a", "c"], ["b", "d"]]
        out = LazyStackedTensorDict.lazy_stack([NonTensorData("a"), NonTensorData("b")])
        assert type(out) is NonTensorStack
        # tensordicts give a lazy stack, as LazyStackedTensorDict.lazy_stack
        out = NonTensorStack.lazy_stack([TensorDict(), TensorDict()])
        assert type(out) is LazyStackedTensorDict

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize(
        "shape",
        [(1, -1), (1, 0), (2, 0), (0, 1), (3, 0, 1), (2, 3, 0), (1, 0, 3), (-1,)],
    )
    def test_reshape_non_tensor_stack_zero_size(self, shape, stack_dim):
        # A stack with no entries takes the shape torch gives, and tolist()
        # nests like torch's tolist()
        stack = NonTensorStack(
            *[NonTensorData("a", batch_size=[0]) for _ in range(3)],
            stack_dim=stack_dim,
        )
        expected = torch.zeros(stack.batch_size).reshape(shape)
        result = stack.reshape(shape)
        assert isinstance(result, NonTensorStack)
        assert result.batch_size == expected.shape
        assert result.tolist() == expected.tolist()
        assert result.device == stack.device
        assert not result.is_locked
        assert stack.lock_().reshape(shape).is_locked

    @pytest.mark.parametrize(
        "call,shape",
        [
            (lambda x: x.flatten(), (0,)),
            (lambda x: x.unflatten(1, (3, 1)), (0, 3, 1, 0)),
            (lambda x: x.unflatten(0, (2, 0)), (2, 0, 3, 0)),
        ],
        ids=["flatten", "unflatten_3_1", "unflatten_2_0"],
    )
    def test_flatten_unflatten_non_tensor_stack_zero_size(self, call, shape):
        # flatten and unflatten give the stack that reshape gives, locked
        # like the input, which they leave as it is
        def make():
            return NonTensorStack(*[NonTensorStack(batch_size=[0])] * 3, stack_dim=1)

        stack = make()
        assert stack.batch_size == (0, 3, 0)
        out = call(stack)
        expected = make().reshape(shape)
        assert isinstance(out, NonTensorStack)
        assert out.batch_size == expected.batch_size
        assert out.tolist() == expected.tolist()
        assert not out.is_locked
        assert not stack.is_locked
        assert call(make().lock_()).is_locked

    @pytest.mark.parametrize(
        "shape", [(3, 0), (0, 3), (0, 3, 1), (1, 0), (0, 0, 3), (-1, 3)]
    )
    @pytest.mark.parametrize("member", ["stack", "data"])
    def test_view_non_tensor_stack_zero_size(self, shape, member):
        # view gives torch's shape, and the input and result stay unlocked
        def make():
            if member == "stack":
                members = [NonTensorStack(batch_size=[0])] * 3
            else:
                members = [NonTensorData("a", batch_size=[0])] * 3
            return NonTensorStack(*members, stack_dim=1)

        stack = make()
        out = stack.view(shape)
        expected = torch.zeros(stack.batch_size).view(shape)
        assert isinstance(out, NonTensorStack)
        assert out.batch_size == expected.shape
        assert out.tolist() == expected.tolist()
        assert not out.is_locked
        assert not stack.is_locked
        assert make().lock_().view(shape).is_locked

    def test_comparison(self, non_tensor_data):
        non_tensor_data = non_tensor_data.exclude(("nested", "str"))
        assert (non_tensor_data | non_tensor_data).get_non_tensor(("nested", "bool"))
        assert not (non_tensor_data ^ non_tensor_data).get_non_tensor(
            ("nested", "bool")
        )
        assert (non_tensor_data == non_tensor_data).get_non_tensor(("nested", "bool"))
        assert not (non_tensor_data != non_tensor_data).get_non_tensor(
            ("nested", "bool")
        )

    @pytest.mark.parametrize("data_first", [True, False])
    def test_comparison_data_and_stack(self, data_first):
        # A NonTensorData and a NonTensorStack of the same batch size are
        # compared element by element, as two stacks are
        def ordered(data, stack):
            return (data, stack) if data_first else (stack, data)

        data = NonTensorData("a", batch_size=[2])
        lhs, rhs = ordered(data, NonTensorStack("a", "b"))
        assert (lhs == rhs).tolist() == [True, False]
        assert (lhs != rhs).tolist() == [False, True]
        data = NonTensorData("a", batch_size=[2, 2])
        lhs, rhs = ordered(data, NonTensorStack.from_list([["a", "b"], ["a", "a"]]))
        assert (lhs == rhs).tolist() == [[True, False], [True, True]]
        assert (lhs != rhs).tolist() == [[False, True], [False, False]]
        # values that cannot be compared keep the empty result
        data = NonTensorData({"k": np.zeros(2)}, batch_size=[2])
        lhs, rhs = ordered(data, NonTensorStack({"k": np.zeros(2)}, {"k": np.zeros(2)}))
        assert (lhs == rhs).is_empty()
        assert (lhs != rhs).is_empty()

        # so tensordict comparisons see the non-tensor values in either form
        td_data = TensorDict(s=NonTensorData("a", batch_size=[2]), batch_size=[2])
        td_diff = TensorDict(s=NonTensorStack("a", "b"), batch_size=[2])
        td_same = TensorDict(s=NonTensorStack("a", "a"), batch_size=[2])
        lhs, rhs = ordered(td_data, td_diff)
        assert not (lhs == rhs).all()
        assert (lhs != rhs).any()
        lhs, rhs = ordered(td_data, td_same)
        assert (lhs == rhs).all()
        assert not (lhs != rhs).any()
        # with no elements, they stay equal
        lhs, rhs = ordered(td_data[:0], td_diff[:0])
        assert (lhs == rhs).all()
        assert not (lhs != rhs).any()

    @pytest.mark.parametrize("dest", ["data", "stack"])
    def test_gather_non_tensor_data_out(self, dest):
        td = TensorDict(x=torch.zeros(2, 3), batch_size=[2, 3])
        td["query"] = NonTensorData("new", batch_size=[2, 3])
        out = TensorDict(x=torch.zeros(2, 3), batch_size=[2, 3])
        if dest == "data":
            out["query"] = NonTensorData("old", batch_size=[2, 3])
        else:
            out["query"] = NonTensorStack.from_list([["old"] * 3] * 2)
        td.gather(1, torch.tensor([[1, 0, 1], [0, 0, 1]]), out=out)
        assert out.get("query").tolist() == [["new"] * 3] * 2

    @pytest.mark.parametrize("capture", [False, True])
    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize("layout", ["dense", "lazy", "stack_dim1"])
    @pytest.mark.parametrize("with_out", [False, True])
    def test_gather_preserves_non_tensor_values(self, capture, dim, layout, with_out):
        values = [[0, 1, 2], [3, 4, 5]]
        if dim == 0:
            index = torch.tensor([[1, 0, 1], [0, 0, 1]])
        else:
            index = torch.tensor([[2, 0, 1], [1, 1, 0]])
        expected = torch.gather(torch.tensor(values), dim, index)
        with _set_capture_non_tensor_stack(capture):
            if layout == "lazy":
                td = lazy_stack(
                    [
                        TensorDict(
                            query=NonTensorStack.from_list(row),
                            x=torch.tensor(row),
                            batch_size=[3],
                        )
                        for row in values
                    ]
                )
            elif layout == "stack_dim1":
                # stacking the columns along dim=1 gives a non-tensor stack
                # whose stack_dim is not the first dimension
                td = torch.stack(
                    [
                        TensorDict(
                            query=NonTensorStack.from_list(list(col)),
                            x=torch.tensor(col),
                            batch_size=[2],
                        )
                        for col in zip(*values)
                    ],
                    dim=1,
                )
                assert td.get("query").stack_dim == 1
            else:
                td = TensorDict(
                    query=NonTensorStack.from_list(values),
                    x=torch.tensor(values),
                    batch_size=[2, 3],
                )
            out = None
            if with_out:
                out = TensorDict(
                    query=NonTensorStack.from_list(
                        torch.full_like(expected, -1).tolist()
                    ),
                    x=torch.full_like(expected, -1),
                    batch_size=expected.shape,
                )
                out_query = out.get("query")
            result = td.gather(dim, index, out=out)
        if with_out:
            assert result is out
            # the entries are written into the destination stack, as the
            # tensors are written into the destination storage
            assert result.get("query") is out_query
        assert isinstance(result.get("query"), NonTensorStack)
        assert result.get("query").tolist() == expected.tolist()
        assert (result.get("x") == expected).all()

    @pytest.mark.parametrize("with_out", [False, True])
    def test_gather_non_tensor_stack_copies(self, with_out):
        stack = NonTensorStack("walk", "jump", "stand")
        out = NonTensorStack("", "", "") if with_out else None
        gathered = torch.gather(stack, 0, torch.tensor([1, 0, 0]), out=out)
        if with_out:
            assert gathered is out
        assert gathered.tolist() == ["jump", "walk", "walk"]
        # the gathered entries are copies, as with advanced indexing
        gathered[1] = "hop"
        assert gathered.tolist() == ["jump", "hop", "walk"]
        assert stack.tolist() == ["walk", "jump", "stand"]

    @pytest.mark.parametrize("invalid", [-1, 3])
    def test_gather_non_tensor_stack_invalid_index(self, invalid):
        # torch.gather rejects negative and out-of-range indices
        stack = NonTensorStack.from_list([[0, 1, 2], [3, 4, 5]])
        index = torch.tensor([[invalid, 0, 1], [0, 0, 0]])
        with pytest.raises(RuntimeError, match="out of bounds"):
            torch.gather(stack, 1, index)

    def test_gather_non_tensor_stack_smaller_index(self):
        # as with torch.gather, the index may be smaller than the input on the
        # dims that are not gathered
        values = [[0, 1, 2], [3, 4, 5]]
        stack = NonTensorStack.from_list(values)
        for dim, index in ((1, torch.tensor([[2, 0]])), (0, torch.tensor([[1, 0]]))):
            expected = torch.gather(torch.tensor(values), dim, index)
            gathered = torch.gather(stack, dim, index)
            assert gathered.batch_size == expected.shape
            assert gathered.tolist() == expected.tolist()

    @pytest.mark.parametrize("capture", [False, True])
    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize("stacked", [(False, False), (False, True), (True, False)])
    @pytest.mark.parametrize("with_out", [False, True])
    def test_cat_preserves_non_tensor_values(self, capture, dim, stacked, with_out):
        items = [
            (
                NonTensorStack.from_list([[value, value + 1], [value + 2, value + 3]])
                if as_stack
                else NonTensorData(value, batch_size=[2, 2])
            )
            for value, as_stack in zip((0, 4), stacked)
        ]
        expected = torch.cat([torch.tensor(item.tolist()) for item in items], dim)
        out = NonTensorStack.from_list(torch.full_like(expected, -1).tolist())
        with _set_capture_non_tensor_stack(capture):
            result = torch.cat(items, dim=dim, out=out if with_out else None)
        if with_out:
            assert result is out
        assert result.batch_size == expected.shape
        assert result.tolist() == expected.tolist()

    @pytest.mark.parametrize("capture", [False, True])
    def test_cat_non_tensor_data_out(self, capture):
        items = [NonTensorData("value", batch_size=[2])] * 2
        out = NonTensorData("old", batch_size=[4])
        with _set_capture_non_tensor_stack(capture):
            assert torch.cat(items, out=out) is out
        assert out.tolist() == ["value"] * 4

    def test_cat_non_tensor_data_of_equal_tensorclasses(self):
        # A tensorclass defined after tensordict is imported compares like any
        # other accepted class, so equal data stays one NonTensorData.
        @tensorclass
        class Data:
            x: torch.Tensor

        data = Data(x=torch.zeros(3), batch_size=[3])
        items = [
            NonTensorData(data, batch_size=[2]),
            NonTensorData(data.clone(), batch_size=[2]),
        ]
        result = torch.cat(items)
        assert isinstance(result, NonTensorData)
        assert result.batch_size == (4,)
        assert (result.data == data).all()

    @pytest.mark.parametrize("capture", [False, True])
    def test_cat_pads_nested_non_tensor_values(self, capture):
        with _set_capture_non_tensor_stack(capture):
            batch = lazy_stack(
                [
                    TensorDict(
                        observation=torch.tensor([float(index)]),
                        metadata=TensorDict(index=NonTensorData(index)),
                    )
                    for index in range(2)
                ]
            ).contiguous()
            result = torch.cat([batch, batch[-1:].expand(2)])
        assert result.batch_size == (4,)
        assert result.get(("metadata", "index")).tolist() == [0, 1, 1, 1]
        torch.testing.assert_close(
            result["observation"], torch.tensor([[0.0], [1.0], [1.0], [1.0]])
        )
        assert batch.get(("metadata", "index")).tolist() == [0, 1]

    @pytest.mark.parametrize("cls", [NonTensorData, MetaData])
    @pytest.mark.parametrize("old_names", [None, ["m"]])
    def test_empty_names(self, cls, old_names):
        # empty() applies the names it is given, as TensorDict.empty() does
        x = cls(data="x", batch_size=[2], names=old_names)
        assert x.empty().names == (old_names or [None])
        assert x.empty(batch_size=[2]).names == [None]
        for out in (
            x.empty(names=["n"]),
            x.empty(batch_size=[2], names=["n"]),
            TensorDict(batch_size=[2]).empty(names=["n"]),
        ):
            assert out.names == ["n"]
        out = x.empty(batch_size=[2, 3], names=["n", "o"])
        assert type(out) is cls
        assert out.batch_size == (2, 3)
        assert out.names == ["n", "o"]
        assert out.data == "x"

    def test_expand(self):
        d = NonTensorData(0, batch_size=(3,))
        d_expand = d.expand((2, 3))
        assert d_expand.shape == (2, 3)
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        assert isinstance(d_expand, NonTensorStack)
        d = NonTensorData(0, batch_size=(1,))
        d_expand = d.expand((2, 3))
        assert d_expand.shape == (2, 3)
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        assert isinstance(d_expand, NonTensorStack)

    def test_expand_nested(self):
        d = TensorDict(foo=NonTensorData(0, batch_size=(3,)), batch_size=(3,))
        d_expand = d.expand((2, 3)).get("foo")
        assert d_expand.shape == (2, 3)
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        d = TensorDict(foo=NonTensorData(0, batch_size=(1,)), batch_size=(1,))
        d_expand = d.expand((2, 3)).get("foo")
        assert d_expand.shape == (2, 3)
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]

    def test_expand_minus_one(self):
        # A -1 keeps the size of its dim, as in torch.Tensor.expand
        d = NonTensorData(0, batch_size=(1, 3))
        for shape in [(2, -1, 3), (2, -1, -1), (-1, -1)]:
            assert d.expand(*shape).shape == torch.zeros(1, 3).expand(shape).shape
        assert d.expand(2, -1, 3).tolist() == [[[0, 0, 0]], [[0, 0, 0]]]
        stack = NonTensorStack(
            NonTensorData(0, batch_size=(3,)), NonTensorData(1, batch_size=(3,))
        )
        assert stack.expand(4, -1, -1).tolist() == [[[0] * 3, [1] * 3]] * 4
        assert stack.broadcast_to((4, -1, 3)).shape == (4, 2, 3)

    def test_expand_rejected_shape_warns(self):
        # The shapes that torch.Tensor.expand rejects warn, and give the same
        # result as before, until they raise in 0.17 (#2175)
        d = NonTensorData(0, batch_size=(1, 3))
        for shape, batch_size in [((2, 4), (4, 2, 3)), ((3,), (1, 3))]:
            with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
                assert d.expand(*shape).batch_size == batch_size
        for shape in [(5, 3), (2, 1, 3), (-1, 3), (1, 3), (2, 4, 3)]:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = d.expand(*shape)
            assert result.batch_size == torch.zeros(1, 3).expand(shape).shape
            assert result.data == 0
        stack = NonTensorStack(
            *[NonTensorStack(*[NonTensorData(0) for _ in range(5)]) for _ in range(3)]
        )
        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
            stack[1] = NonTensorData(1, batch_size=(3,))
        assert stack.tolist() == [[0] * 5, [1] * 5, [0] * 5]

    @pytest.mark.parametrize(
        "in_out", [(None, None), (0, -1), (None, 0), (0, None), (0, 0)]
    )
    def test_flatten_empty(self, in_out):
        ntd = NonTensorData(0, batch_size=())
        assert ntd.shape == ()
        ntdflat = ntd.flatten(in_out[0], in_out[1])
        assert ntdflat.shape == (1,)
        assert ntdflat.tolist() == [0]

    def test_from_list(self):
        nd = NonTensorStack.from_list(
            [[True, "b", torch.randn(())], ["another", 0, NonTensorData("final")]]
        )
        assert isinstance(nd, NonTensorStack)
        assert nd.shape == (2, 3)
        assert nd[0, 0].data
        assert nd[0, 1].data == "b"
        assert isinstance(nd[0, 2].data, torch.Tensor)
        assert nd[1, 0].data == "another"
        assert nd[1, 1].data == 0
        assert nd[1, 2].data == "final"

    @pytest.mark.parametrize("cls", [NonTensorData, MetaData])
    def test_init(self, cls):
        # NonTensorData and MetaData fill _non_tensordict without going
        # through set(): the result must be the one set() gives.
        x = cls(data="x", batch_size=[3], device="cpu", names=["n"], lock=True)
        assert x.data == "x"
        assert x.batch_size == (3,)
        assert x.device == torch.device("cpu")
        assert x.names == ["n"]
        assert x.is_locked
        assert x._non_tensordict == {
            "data": "x",
            "_metadata": None,
            "_is_non_tensor": True,
        }
        assert x._tensordict.is_empty()
        # A NonTensorData value is unwrapped, as set() does.
        assert cls(cls(NonTensorData("y"))).data == "y"
        assert cls("z", {"k": 1})._metadata == {"k": 1}
        assert cls("z", _metadata={"k": 1})._metadata == {"k": 1}
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            cls()
        with pytest.raises(AttributeError, match="Cannot set the attribute 'foo'"):
            cls(data=1, foo=2)

    def test_is_non_tensor(self):
        from tensordict.tensorclass import is_non_tensor as tc_is_non_tensor

        non_tensors = [
            NonTensorData("a"),
            MetaData("a"),
            NonTensorStack(NonTensorData("a"), NonTensorData("b")),
        ]
        others = [
            torch.zeros(()),
            TensorDict(),
            self.SomeTensorClass(a="a", b=torch.zeros(())),
            "a",
            None,
        ]
        for obj in non_tensors:
            assert tc_is_non_tensor(obj)
            assert is_non_tensor(obj)
        for obj in others:
            assert not tc_is_non_tensor(obj)
            assert not is_non_tensor(obj)

    @pytest.mark.parametrize("memo", ["cold", "warm"])
    def test_is_non_tensor_exact_tensor(self, memo, monkeypatch):
        # An exact torch.Tensor is answered before is_compiling() and the memo;
        # tensor subclasses and other values take the general path.
        class FlaggedTensor(torch.Tensor):
            _is_non_tensor = True

        # value -> (is_non_tensor, _pass_through)
        cases = [
            (torch.zeros(()), False, False),
            (torch.nn.Parameter(torch.zeros(())), False, False),
            (MemoryMappedTensor.from_tensor(torch.zeros(())), False, False),
            (torch.zeros(()).as_subclass(FlaggedTensor), True, True),
            (UnbatchedTensor(torch.zeros(())), False, True),
            (NonTensorData("a"), True, True),
            (TensorDict(), False, False),
            (self.SomeTensorClass(a="a", b=torch.zeros(())), False, False),
            ("a", False, False),
        ]
        if memo == "cold":
            monkeypatch.setattr("tensordict.utils._NON_TENSOR_MEMO", {})
            monkeypatch.setattr("tensordict.utils._PASSTHROUGH_MEMO", {})
        for _ in range(2):
            for value, non_tensor, pass_through in cases:
                assert is_non_tensor(value) is non_tensor
                assert _is_non_tensor(type(value)) is non_tensor
                assert _pass_through(value) is pass_through
                assert _pass_through_cls(type(value)) is pass_through

        def is_compiling():
            raise AssertionError("called is_compiling() for an exact Tensor")

        monkeypatch.setattr("tensordict.utils.is_compiling", is_compiling)
        assert is_non_tensor(torch.zeros(())) is False
        assert _pass_through(torch.zeros(())) is False
        with pytest.raises(AssertionError, match="is_compiling"):
            is_non_tensor(torch.nn.Parameter(torch.zeros(())))

    @set_list_to_stack(True)
    def test_linked_list(self):
        td = TensorDict(a=["foo", "bar"], batch_size=(2,))
        assert isinstance(td["a"], LinkedList)
        td["a"][0] = "baz"
        assert td["a"] == ["baz", "bar"]
        td["a"][:2] = ["baz", "qux"]
        assert td["a"] == ["baz", "qux"]

        result = td["a"]
        assert result._td() is td.get("a")
        result.extend(["baz", "qux"])
        assert result._td() is None

    @pytest.mark.parametrize("value", ["b", NonTensorData("b")])
    def test_linked_list_nontensordata_write_raises(self, value):
        # Regression test for #2978: a NonTensorData holds one value, so the
        # write raises with a message that says why, and the list does not
        # change before the write fails.
        data = NonTensorData("a", batch_size=[2])
        values = data.tolist(as_linked_list=True)
        assert isinstance(values, LinkedList)
        with pytest.raises(ValueError, match="holds one value"):
            values[0] = value
        assert values == ["a", "a"]
        assert data.tolist() == ["a", "a"]

    @pytest.mark.parametrize(
        "key,value,error",
        [
            # the list write fails, so the tensordict must not be written
            (slice(0, 2), 5, TypeError),
            # the list write works, the tensordict write fails
            (slice(1, 2), ["x", "y", "z"], RuntimeError),
        ],
    )
    def test_linked_list_failed_write_unchanged(self, key, value, error):
        stack = NonTensorStack("a", "b", "c")
        values = stack.tolist(as_linked_list=True)
        with pytest.raises(error):
            values[key] = value
        assert values == ["a", "b", "c"]
        assert stack.tolist() == ["a", "b", "c"]

    @set_list_to_stack(True)
    @pytest.mark.parametrize(
        "rows,batch_size",
        [
            ([["a"], ["b"]], (2, 1)),
            ([["a"], ["b", "c"]], (2,)),
            ([[{"role": "user"}], [{"role": "user"}, {"role": "bot"}]], (2,)),
        ],
    )
    @pytest.mark.parametrize("batch_dims", [None, 1])
    def test_from_dict_auto_batch_size_list_of_lists(
        self, rows, batch_size, batch_dims
    ):
        # Regression test for #2542: the nested NonTensorStack is not resized,
        # and a ragged dim (-1) does not enter the batch size
        if batch_dims is not None:
            batch_size = batch_size[:batch_dims]
        td = TensorDict.from_dict(
            {"q": rows}, auto_batch_size=True, batch_dims=batch_dims
        )
        assert type(td) is TensorDict
        assert td.batch_size == batch_size
        assert isinstance(td.get("q"), NonTensorStack)
        assert td["q"] == rows
        td = TensorDict(q=rows).auto_batch_size_(batch_dims=batch_dims)
        assert td.batch_size == batch_size

    def test_tensordict_stacks_a_list_of_equal_shaped_tensors(self):
        # One shared shape is a stacked tensor, not a NonTensorStack.
        td = TensorDict({"a": [torch.ones(2), torch.ones(2)]}, batch_size=[2])
        value = td.get("a")
        assert isinstance(value, torch.Tensor)
        assert not isinstance(value, NonTensorStack)
        assert value.shape == (2, 2)
        assert (value == 1).all()

    def test_tensordict_ragged_tensor_list_is_non_tensor_stack(self):
        # Differently shaped tensors cannot be stacked. Each tensor stays the value
        # of one batch element, as NonTensorStack(t0, t1) does, without a copy.
        tensors = [torch.ones(2), torch.ones(3)]
        td = TensorDict({"a": tensors}, batch_size=[2])
        value = td.get("a")
        assert isinstance(value, NonTensorStack)
        assert value.batch_size == torch.Size([2])
        assert value.tolist()[0] is tensors[0]
        assert value.tolist()[1] is tensors[1]
        assert (td[0]["a"] == tensors[0]).all()
        assert (td[1]["a"] == tensors[1]).all()

    @pytest.mark.parametrize("device", [None, "cpu"])
    def test_lazy_stack_ragged_tensor_list_sets_tensors(self, device):
        # A lazy stack writes each tensor of the list into its own member.
        td = lazy_stack(
            [
                TensorDict(x0=torch.zeros(1), device=device),
                TensorDict(x1=torch.zeros(2), device=device),
            ]
        )
        tensors = [torch.ones(2), torch.ones(3)]
        td.set("a", tensors)
        assert td.tensordicts[0]._get_str("a", None) is tensors[0]
        assert td.tensordicts[1]._get_str("a", None) is tensors[1]

    def test_lazy_stack_non_tensor_list_keeps_non_tensor_data(self):
        # A list of NonTensorData is not unwrapped into plain tensors.
        td = lazy_stack([TensorDict(x0=torch.zeros(1)), TensorDict(x1=torch.zeros(2))])
        tensors = [torch.ones(2), torch.ones(3)]
        td.set("a", [NonTensorData(tensors[0]), NonTensorData(tensors[1])])
        for member, tensor in zip(td.tensordicts, tensors):
            value = member._get_str("a", None)
            assert isinstance(value, NonTensorData)
            assert value.data is tensor

    def test_new_empty_nontensorstack(self):
        td = TensorDict(a=NonTensorStack("a", "b").unsqueeze(-1), batch_size=(2,))
        assert isinstance(td.new_empty((4,), empty_lazy=True).get("a"), NonTensorStack)
        assert isinstance(td.new_empty((1,), empty_lazy=True).get("a"), NonTensorStack)

    def test_new_empty_setitem(self):
        td = TensorDict(
            a=TensorDict(
                b=NonTensorStack("a", "b", "c").unsqueeze(-1), batch_size=(3,)
            ),
            batch_size=(3,),
        ).to_lazystack()
        tdz = td.new_zeros((4,), empty_lazy=True)
        tdz[torch.tensor([True, True, False, True])] = td
        assert tdz.get(("a", "b")).tolist() == [["a"], ["b"], ["a"], ["c"]]

    def test_new_empty_setitem_2(self):
        td = TensorDict(
            a=TensorDict(b=NonTensorStack("a"), batch_size=(1,)), batch_size=(1,)
        ).to_lazystack()
        tdz = td.new_zeros((4,), empty_lazy=True)
        td["a", "b"] = "new"
        tdz[torch.tensor([False, False, False, True])] = td
        assert tdz["a", "b"][-1] == "new"

    def test_non_tensor_call(self):
        td0 = TensorDict({"a": 0, "b": 0})
        td1 = TensorDict({"a": 1, "b": 1})
        td_func = TensorDict({"a": lambda x, y: x - y, "b": lambda x, y: x + y})
        td = td0.apply(lambda x, y, func: func(x, y), td1, td_func)
        assert td["a"] == -1
        assert td["b"] == 1

    def test_non_tensor_from_list(self):
        class X(TensorClass):
            non_tensor: str = None

        x = X(batch_size=3)
        x.non_tensor = NonTensorStack.from_list(["a", "b", "c"])
        assert x[0].non_tensor == "a"
        assert x[1].non_tensor == "b"

        x = X(non_tensor=NonTensorStack("a", "b", "c"), batch_size=3)
        assert x[0].non_tensor == "a"
        assert x[1].non_tensor == "b"

    @pytest.mark.parametrize("lazy", [False, True])
    def test_non_tensor_items(self, lazy):
        td = TensorDict(
            a=torch.zeros(3),
            nested=TensorDict(
                b=torch.zeros(3), s=NonTensorData("y", batch_size=[3]), batch_size=[3]
            ),
            s=NonTensorData("x", batch_size=[3]),
            batch_size=[3],
        )
        expected = [["y"] * 3, ["x"] * 3]
        if lazy:
            td = lazy_stack([td, td.clone()])
            expected = [[value] * 2 for value in expected]
        assert [key for key, _ in td.non_tensor_items()] == ["s"]
        # include_nested=True used to raise AttributeError on the tensor leaves
        items = td.non_tensor_items(include_nested=True)
        assert [key for key, _ in items] == [("nested", "s"), "s"]
        assert [val.tolist() for _, val in items] == expected

    def test_nontensor_dict(self, non_tensor_data):
        assert (
            TensorDict.from_dict(non_tensor_data.to_dict(), auto_batch_size=True)
            == non_tensor_data
        ).all()

    def test_nontensor_tensor(self):
        t1 = torch.tensor([1, 2, 3], dtype=torch.float)
        t2 = torch.tensor([1, 2, 3, 4], dtype=torch.float)
        stack = NonTensorStack(NonTensorData(t1), NonTensorData(t2))  # this works fine
        assert all(isinstance(t, torch.Tensor) for t in stack.tolist())
        stack = torch.stack(
            [NonTensorData(t1), NonTensorData(t2)]
        )  # this triggers an exception
        assert all(isinstance(t, torch.Tensor) for t in stack.tolist())

    def test_repeat(self):
        stack = NonTensorStack(
            NonTensorData("a", batch_size=(3,)),
            NonTensorData("b", batch_size=(3,)),
            stack_dim=1,
        )
        assert stack.shape == (3, 2)
        r = stack.repeat(1, 2)
        assert r.shape == (3, 4)
        assert r[0].tolist() == ["a", "b", "a", "b"]
        assert r[:, 0].tolist() == ["a", "a", "a"]
        assert r[:, -1].tolist() == ["b", "b", "b"]

    def test_repeat_interleave(self):
        stack = NonTensorStack(
            NonTensorData("a", batch_size=(3,)),
            NonTensorData("b", batch_size=(3,)),
            stack_dim=1,
        )
        assert stack.shape == (3, 2)
        r = stack.repeat_interleave(3, dim=1)
        assert isinstance(r, NonTensorStack)
        assert r[0].tolist() == ["a", "a", "a", "b", "b", "b"]

    def test_set(self, non_tensor_data):
        non_tensor_data.set(("nested", "another_string"), "another string!")
        assert (
            non_tensor_data.get(("nested", "another_string")).data == "another string!"
        )
        assert (
            non_tensor_data.get_non_tensor(("nested", "another_string"))
            == "another string!"
        )

    @pytest.mark.parametrize(
        "write,key,expected",
        [
            (lambda td: td.__setitem__("a", td["a"]), "a", True),
            (lambda td: td.set("a", 1), "a", 1),
            (lambda td: td.update({"a": 1.5}), "a", 1.5),
            (lambda td: td.set(("nested", "a"), False), ("nested", "a"), False),
            (lambda td: td.set("a", np.bool_(False)), "a", False),
            (lambda td: td.set("a", np.int64(1)), "a", 1),
            (lambda td: td.set("a", np.float32(1.5)), "a", 1.5),
            (lambda td: td.set("a", np.float64(1.5)), "a", 1.5),
        ],
        ids=[
            "setitem",
            "set",
            "update",
            "nested",
            "np-bool",
            "np-int64",
            "np-float32",
            "np-float64",
        ],
    )
    def test_set_scalar_over_non_tensor_warns(self, write, key, expected):
        # A Python or NumPy scalar written over a NonTensorData entry is a 0-d tensor
        # today, and stays NonTensorData in 0.17 (gh-1199)
        td = TensorDict(a=NonTensorData(True), nested=TensorDict(a=NonTensorData(True)))
        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17") as w:
            write(td)
        assert w[0].filename == __file__
        value = td.get(key)
        assert type(value) is torch.Tensor
        assert value.ndim == 0
        assert value == torch.tensor(expected)

    @pytest.mark.parametrize(
        "write",
        [
            lambda td: td.set("a", "text"),
            lambda td: td.set("a", torch.tensor(1)),
            lambda td: td.set("a", NonTensorData(1)),
            lambda td: td.set("b", 1),
            lambda td: td.set("c", 1),
            lambda td: td.set("s", 1),
            lambda td: td.update(TensorDict(a=1)),
        ],
        ids=["str", "tensor", "non-tensor", "tensor-entry", "new", "stack", "td"],
    )
    def test_set_over_non_tensor_no_warning(self, write):
        td = TensorDict(
            a=NonTensorData(True), b=torch.zeros(()), s=NonTensorStack("x", "y")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            write(td)

    @pytest.mark.parametrize(
        "write,batch_size",
        [
            (lambda td: td.set("a", 1), [2]),
            (lambda td: td.set_("a", 1), []),
            (lambda td: td.update_({"a": 1}), []),
            (lambda td: td.lock_().set("a", 1), []),
        ],
        ids=["batched", "set_", "update_", "locked"],
    )
    def test_set_scalar_over_non_tensor_raises_no_warning(self, write, batch_size):
        # These writes raise today: they do not warn
        td = TensorDict(
            a=NonTensorData(True, batch_size=batch_size), batch_size=batch_size
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises((RuntimeError, ValueError)):
                write(td)

    @pytest.mark.parametrize("lazy", [False, True], ids=["dense", "lazy"])
    @pytest.mark.parametrize("index", [(), 1, slice(0, 2)], ids=str)
    @pytest.mark.parametrize(
        "value",
        [[1, 2], 1, True, 1.5, np.bool_(True), np.int64(1), np.float32(1.5)],
        ids=lambda value: type(value).__name__,
    )
    def test_set_at_cast_over_non_tensor_warns(self, value, index, lazy):
        # set_at_ casts a list or a scalar written into a non-tensor entry to
        # a tensor today, and stores it as non-tensor data in 0.17 (gh-2522).
        # A lazy stack warns once, not once per member.
        td = TensorDict(a=NonTensorData("s", batch_size=[3]), batch_size=[3])
        if lazy:
            td = lazy_stack(td.unbind(0))
        with pytest.warns(
            FutureWarning,
            match="Starting with TensorDict 0.17, the default will change to "
            "storing it as non-tensor data",
        ) as w:
            td.set_at_("a", value, index)
        assert len(w) == 1
        assert w[0].filename == __file__
        assert torch.equal(td[1]["a"], torch.as_tensor(value))

    @pytest.mark.parametrize("lazy", [False, True], ids=["dense", "lazy"])
    @pytest.mark.parametrize(
        "value", [[1, 2], 0, np.float32(1.5)], ids=lambda value: type(value).__name__
    )
    def test_setitem_cast_over_non_tensor_warns(self, value, lazy):
        # td[index] = value casts a list or a scalar to a tensor in the
        # non-tensor entries today, and leaves them unchanged in 0.17, as a
        # TensorDictStore does (gh-2522). The call warns once, whatever the
        # number of non-tensor entries and members.
        td = TensorDict(
            a=NonTensorData("s", batch_size=[3]),
            b=torch.zeros(3),
            nested=TensorDict(c=NonTensorData("t", batch_size=[3]), batch_size=[3]),
            batch_size=[3],
        )
        if lazy:
            td = lazy_stack(td.unbind(0))
        with pytest.warns(
            FutureWarning,
            match="Starting with TensorDict 0.17, the default will change to "
            "leaving the non-tensor entries unchanged",
        ) as w:
            td[:2] = value
        assert len(w) == 1
        assert w[0].filename == __file__
        expected = torch.as_tensor(value)
        assert torch.equal(td[1]["a"], expected)
        assert torch.equal(td[1]["nested", "c"], expected)
        assert td[2]["a"] == "s"

    @pytest.mark.parametrize("lazy", [False, True], ids=["dense", "lazy"])
    @pytest.mark.parametrize("nested", ["tensorclass", "tensordict"])
    @pytest.mark.parametrize(
        "value", [3, np.bool_(True), [1, 2]], ids=lambda value: type(value).__name__
    )
    def test_set_at_cast_into_nested_non_tensor_warns(self, value, nested, lazy):
        # set_at_ of a list or a scalar on a nested tensordict writes it into
        # each entry, as td[index] = value does, and warns once (gh-2522)
        @tensorclass
        class Named:
            x: torch.Tensor
            s: str

        if nested == "tensorclass":
            inner = Named(x=torch.zeros(3), s="v", batch_size=[3])
        else:
            inner = TensorDict(
                x=torch.zeros(3), s=NonTensorData("v", batch_size=[3]), batch_size=[3]
            )
        td = TensorDict(n=inner, batch_size=[3])
        if lazy:
            td = lazy_stack(td.unbind(0))
        with pytest.warns(
            FutureWarning, match="leaving the non-tensor entries unchanged"
        ) as w:
            td.set_at_("n", value, slice(0, 2))
        assert len(w) == 1
        assert w[0].filename == __file__
        assert torch.equal(td[0]["n", "s"], torch.as_tensor(value))
        assert td[2]["n", "s"] == "v"

    def test_set_at_non_tensor_stack_data(self):
        # The members of a NonTensorStack are NonTensorData: set_at_ writes
        # the value into the first one, which holds it as it is
        stack = NonTensorStack("a", "b", "c")
        stack.set_at_("data", 3, 0)
        assert stack.tolist() == [3, "b", "c"]
        td = TensorDict(s=NonTensorStack("a", "b", "c"), batch_size=[3])
        td.set_at_(("s", "data"), 3, 0)
        assert td.get("s").tolist() == [3, "b", "c"]

    @pytest.mark.parametrize("lazy", [False, True], ids=["dense", "lazy"])
    @pytest.mark.parametrize(
        "key,value",
        [
            ("a", torch.tensor(1)),
            ("a", "text"),
            ("a", NonTensorData(1)),
            ("b", 1),
            ("b", [1.0, 2.0]),
            ("b", torch.ones(2)),
        ],
        ids=[
            "tensor",
            "str",
            "non-tensor",
            "tensor-entry-scalar",
            "tensor-entry-list",
            "tensor-entry",
        ],
    )
    def test_set_at_without_cast_no_warning(self, key, value, lazy):
        td = TensorDict(
            a=NonTensorData("s", batch_size=[3]), b=torch.zeros(3, 2), batch_size=[3]
        )
        if lazy:
            td = lazy_stack(td.unbind(0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            td.set_at_(key, value, 1)

    @pytest.mark.parametrize("lazy", [False, True], ids=["dense", "lazy"])
    @pytest.mark.parametrize("kind", ["shared", "memmap"])
    @pytest.mark.parametrize("value", [[1, 2], 1], ids=type)
    def test_set_at_cast_raises_no_warning(self, value, kind, lazy):
        # Shared and memmap containers refuse the write: it does not warn
        td = TensorDict(a=NonTensorData("s", batch_size=[3]), batch_size=[3])
        if lazy:
            td = lazy_stack(td.unbind(0))
        td = td.share_memory_() if kind == "shared" else td.memmap_()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(RuntimeError):
                td.set_at_("a", value, slice(0, 2))

    @pytest.mark.parametrize("stacked", [False, True])
    @pytest.mark.parametrize("array", [np.array, torch.tensor], ids=["numpy", "torch"])
    def test_set_at_array_data(self, array, stacked):
        # Arrays and tensors compare elementwise, so comparing the old and new
        # contents does not give a bool. Writing them used to raise.
        if stacked:
            data = NonTensorStack(*[NonTensorData(array([0, 0])) for _ in range(3)])
        else:
            data = NonTensorData(array([0, 0]), batch_size=[3])
        td = TensorDict(a=data, b=torch.zeros(3), batch_size=[3])
        td.set_at_("a", NonTensorData(array([0, 0])), 0)
        td.set_at_("a", NonTensorData(array([1, 2, 3])), 1)
        td[2] = TensorDict(a=NonTensorData(array([4, 5])), b=torch.ones(()))
        assert [x.tolist() for x in td.get("a").tolist()] == [[0, 0], [1, 2, 3], [4, 5]]
        td[:2] = TensorDict(
            a=NonTensorData(array([6, 7]), batch_size=[2]),
            b=torch.ones(2),
            batch_size=[2],
        )
        assert [x.tolist() for x in td.get("a").tolist()] == [[6, 7], [6, 7], [4, 5]]
        assert td["b"].tolist() == [1.0, 1.0, 1.0]

    @pytest.mark.parametrize("batch_size", [(), (3,), (3, 2)])
    def test_set_at_empty_index(self, batch_size):
        # The index () selects the whole entry, as for a tensor entry. A
        # NonTensorData cannot be indexed with it, which used to raise.
        expected = TensorDict(a="s1", batch_size=batch_size).get("a").tolist()
        td = TensorDict(a="s0", batch_size=batch_size)
        assert td.set_at_("a", "s1", ()) is td
        assert td.get("a").tolist() == expected
        td = TensorDict(a="s0", batch_size=batch_size)
        td[()] = TensorDict(a="s1", batch_size=batch_size)
        assert td.get("a").tolist() == expected

    @pytest.mark.parametrize(
        "value", [3, 2.5, True, np.int64(3), np.float32(2.5)], ids=type
    )
    @pytest.mark.parametrize(
        "index", [1, slice(0, 2), torch.tensor([0, 2])], ids=["int", "slice", "tensor"]
    )
    @pytest.mark.parametrize("device", [None, "cpu"])
    def test_lazy_set_at_scalar(self, value, index, device):
        # A scalar written into a non-tensor entry of a lazy stack gives what a
        # dense tensordict gives. Each member writes it at the index (), which
        # used to raise AttributeError: stack_dim.
        def make():
            return TensorDict(
                a=NonTensorData("s0", batch_size=[3]),
                b=torch.zeros(3),
                batch_size=[3],
                device=device,
            )

        dense = make()
        lazy = lazy_stack(list(make().unbind(0)))
        # the scalar is cast to a tensor until 0.17 (gh-2522)
        with pytest.warns(FutureWarning, match="storing it as non-tensor data"):
            dense.set_at_("a", value, index)
        with pytest.warns(FutureWarning, match="storing it as non-tensor data"):
            lazy.set_at_("a", value, index)
        expected = dense.get("a").tolist()
        result = lazy.get("a").tolist()
        assert [type(x) for x in result] == [type(x) for x in expected]
        for x, y in zip(result, expected):
            if isinstance(y, torch.Tensor):
                assert x.dtype == y.dtype
                assert x == y
            else:
                assert x == y == "s0"
        assert lazy["b"].tolist() == [0.0, 0.0, 0.0]
        # each member keeps the device of its entries
        for member in lazy.tensordicts:
            assert member.get("a").device == member.device
        assert lazy.to_tensordict().get("a").tolist() == result

    def test_set_at_scalar_no_batch_dims(self):
        # A member of a lazy stack writes a scalar into its non-tensor entry
        # at (), which selects the whole entry
        td = TensorDict(a=NonTensorData("s0"))
        # the scalar is cast to a tensor until 0.17 (gh-2522)
        with pytest.warns(FutureWarning, match="storing it as non-tensor data"):
            td.set_at_("a", 3, ())
        assert type(td.get("a")) is NonTensorData
        assert td.get("a").data == torch.tensor(3)

    @pytest.mark.parametrize("cls", [NonTensorData, MetaData])
    @pytest.mark.parametrize("locked", [False, True])
    @pytest.mark.parametrize("index", [..., (...,)], ids=["ellipsis", "tuple"])
    @pytest.mark.parametrize("key", ["a", ("nested", "a")], ids=["flat", "nested"])
    def test_set_at_ellipsis_no_batch_dims(self, cls, locked, index, key):
        # The Ellipsis selects the whole entry of a tensordict without batch
        # dims. set_at_ used to leave a non-tensor entry unchanged.
        td = TensorDict()
        td.set(key, cls("s0"))
        if locked:
            td.lock_()
        td.set_at_(key, "s1", index)
        assert type(td.get(key)) is cls
        assert td.get(key).data == "s1"
        if locked and isinstance(key, tuple):
            # update_at_ writes a nested tensordict with td[()] = other, which
            # raises on a locked tensordict, as set_ does
            return
        td = TensorDict()
        td.set(key, cls("s0"))
        if locked:
            td.lock_()
        other = TensorDict()
        other.set(key, "s1")
        td.update_at_(other, index)
        assert type(td.get(key)) is cls
        assert td.get(key).data == "s1"

    @pytest.mark.parametrize(
        "index", [torch.tensor(True), True, None], ids=["tensor", "bool", "none"]
    )
    def test_set_at_true_no_batch_dims(self, index):
        # A 0-dim True, True or None index on a non-tensor entry with no batch
        # dims used to raise IndexError: squeeze(0) rejects dim 0 on shape [].
        td = TensorDict({"a": NonTensorData("hello")}, [])
        td.set_at_("a", NonTensorData("y"), index)
        assert type(td.get("a")) is NonTensorData
        assert td.get("a").data == "y"
        assert td.get("a").batch_size == ()
        assert td.batch_size == ()

    @pytest.mark.parametrize(
        "index", [torch.tensor(True), True, None], ids=["tensor", "bool", "none"]
    )
    def test_setitem_true_no_batch_dims(self, index):
        # Same index on a batchless NonTensorData. The value has no batch dim,
        # so it is written as it is.
        data = NonTensorData("hello")
        data[index] = NonTensorData("y")
        assert type(data) is NonTensorData
        assert data.data == "y"
        assert data.batch_size == ()

    def test_setitem_true_squeezes_batched_value(self):
        # A 0-dim True index on a batchless tensorclass still squeezes a value
        # that has a leading batch dim. The result is x == tensor(1.) and the
        # batch size stays [].
        @tensorclass
        class M:
            x: torch.Tensor

        data = M(x=torch.zeros(()), batch_size=[])
        data[torch.tensor(True)] = M(x=torch.ones(1), batch_size=[1])
        assert type(data) is M
        assert data.batch_size == ()
        assert torch.equal(data.x, torch.tensor(1.0))

    @pytest.mark.parametrize(
        "index", [torch.tensor(True), True, None], ids=["tensor", "bool", "none"]
    )
    def test_setitem_true_no_batch_dims_tensorclass(self, index):
        # On a batchless tensorclass, a value with no batch dims is written as
        # by [...]: a compatible value writes its fields, and a value of
        # another class raises instead of adding fields to the tensorclass.
        @tensorclass
        class M:
            x: torch.Tensor

        @tensorclass
        class S:
            s: str

        data = M(x=torch.zeros(()), batch_size=[])
        data[index] = M(x=torch.ones(()), batch_size=[])
        assert torch.equal(data.x, torch.tensor(1.0))

        for dest, value in [
            (M(x=torch.zeros(()), batch_size=[]), NonTensorData("y")),
            (S(s="hello", batch_size=[]), M(x=torch.ones(()), batch_size=[])),
        ]:
            with pytest.raises(ValueError, match="compatible class"):
                dest[...] = value
            with pytest.raises(ValueError, match="compatible class"):
                dest[index] = value
            assert set(dest.to_dict()) == set(type(dest).__expected_keys__)
            dest.clone()

        # [...] adds the keys of a TensorDict value that are not fields.
        dest = S(s="hello", batch_size=[])
        with pytest.raises(ValueError, match="not fields"):
            dest[index] = TensorDict(x=torch.ones(()))
        assert set(dest.to_dict()) == {"s"}

    @pytest.mark.parametrize(
        "index",
        [np.array([1, 0]), np.array(1), [1, 0], torch.tensor([1, 0])],
        ids=["ndarray", "ndarray_0d", "list", "tensor"],
    )
    def test_update_at_partial_index(self, index):
        # Only () and slice(None) replace the whole value. Any other index,
        # including an ndarray, is a partial update and raises RuntimeError.
        data = NonTensorData("h", batch_size=[4])
        with pytest.raises(
            RuntimeError, match=r"Cannot update a part of a NonTensorDataBase\."
        ):
            data.update_at_(NonTensorData("x"), index)
        assert data.data == "h"

    @pytest.mark.parametrize(
        "index", [(), slice(None)], ids=["empty_tuple", "slice_none"]
    )
    def test_update_at_whole_index(self, index):
        # () and slice(None) select every element, so update_at_ replaces the
        # whole value through update_.
        data = NonTensorData("h", batch_size=[4])
        assert data.data == "h"
        assert data.update_at_(NonTensorData("x"), index) is data
        assert data.data == "x"

    @pytest.mark.parametrize("array", [np.array, torch.tensor], ids=["numpy", "torch"])
    def test_subtd_set_at_array_data(self, array):
        # Arrays and tensors compare elementwise, so comparing the old and new
        # contents does not give a bool. Writing them through a
        # _SubTensorDict used to raise.
        td = TensorDict(a=NonTensorData(array([0, 0]), batch_size=[4]), batch_size=[4])
        sub = td._get_sub_tensordict(slice(0, 2))
        sub.set_at_("a", NonTensorData(array([1, 2])), 0)
        sub.set_at_("a", NonTensorData(array([3, 4, 5])), 1)
        assert [x.tolist() for x in td.get("a").tolist()] == [
            [1, 2],
            [3, 4, 5],
            [0, 0],
            [0, 0],
        ]

    @pytest.mark.parametrize("cls", [NonTensorData, MetaData])
    def test_set_at_failed_keeps_data(self, cls):
        # The data is not an entry of the wrapped TensorDict, so set_at_ raises.
        # It used to delete the data first, after which reading it raised.
        data = cls("a", batch_size=[3])
        with pytest.raises(KeyError):
            data.set_at_("data", "b", 0)
        assert data.data == "a"
        assert data.tolist() == ["a", "a", "a"]

    @pytest.mark.parametrize(
        "index",
        [torch.tensor([1, 3]), slice(1, 3), [0, 2], torch.tensor([1, 0, 1, 0]) == 1],
        ids=["tensor", "slice", "list", "mask"],
    )
    @pytest.mark.parametrize("ndim", [1, 2])
    def test_set_at_writes_into_stack(self, index, ndim):
        # A write into a NonTensorStack entry writes into the stack. It used to
        # rebuild the whole stack first, which cost O(size) per write.
        def make(prefix):
            if ndim == 1:
                return NonTensorStack.from_list([f"{prefix}{i}" for i in range(4)])
            return NonTensorStack.from_list(
                [[f"{prefix}{i}{j}" for j in range(2)] for i in range(4)]
            )

        td = TensorDict(a=make("s"), batch_size=[4, 2][:ndim])
        stack = td.get("a")
        value = make("v")[index]
        expected = make("s")
        expected[index] = value
        td[index] = TensorDict(a=value, batch_size=value.batch_size)
        assert td.get("a") is stack
        assert stack.tolist() == expected.tolist()
        value = make("w")[index]
        expected[index] = value
        td.set_at_("a", value, index)
        assert td.get("a") is stack
        assert stack.tolist() == expected.tolist()

    def test_set_at_into_stack_of_batched_members(self):
        # The NonTensorData members of this stack have batch dims: a write
        # into one element of a member still turns that member into a stack
        stack = NonTensorStack(*[NonTensorData(c, batch_size=[2]) for c in "abc"])
        td = TensorDict(a=stack, batch_size=[3, 2])
        td.set_at_("a", "z", 1)
        assert td.get("a").tolist() == [["a", "a"], ["z", "z"], ["c", "c"]]
        td.set_at_("a", NonTensorStack("x", "y"), 0)
        assert td.get("a").tolist() == [["x", "y"], ["z", "z"], ["c", "c"]]
        td.lock_()
        td.set_at_("a", "q", (2, 1))
        assert td.get("a").tolist() == [["x", "y"], ["z", "z"], ["c", "q"]]

    def test_setitem_edge_case(self):
        s = NonTensorStack("a string")
        t = NonTensorStack("another string")
        s[0][True] = t
        assert s[0].data == "another string"
        for i in (None, True):
            s = NonTensorStack("0", "1")
            t = NonTensorStack(NonTensorStack("2", "3"), stack_dim=1)
            assert t.batch_size == (2, 1)
            s[:, i] = t
            assert s.tolist() == ["2", "3"]

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_setitem_into_members(self, stack_dim, tmpdir):
        # A write that changes part of a NonTensorData member, along a dim
        # other than the stack dim, turns the member into a stack.
        def make(value="s0"):
            n = 2 if stack_dim == 0 else 3
            return NonTensorStack(
                *[NonTensorData(value, batch_size=[5 - n]) for _ in range(n)],
                stack_dim=stack_dim,
            )

        s = make()
        assert s.batch_size == (2, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s[1] = NonTensorData("x", batch_size=[3])
        assert s.tolist() == [["s0"] * 3, ["x"] * 3]
        s = make()
        s[1, 0] = NonTensorData("x")
        assert s.tolist() == [["s0"] * 3, ["x", "s0", "s0"]]
        s = make()
        s[0, 1:] = NonTensorStack("a", "b")
        assert s.tolist() == [["s0", "a", "b"], ["s0"] * 3]
        # an equal value leaves the members as they are
        s = make()
        s[1, 0] = NonTensorData("s0")
        assert all(type(td) is NonTensorData for td in s.tensordicts)
        assert s.tolist() == [["s0"] * 3] * 2
        s = make().lock_()
        with pytest.raises(RuntimeError, match="Cannot modify locked"):
            s[1, 0] = NonTensorData("x")
        assert s.tolist() == [["s0"] * 3] * 2
        # a memmap stack writes the value and saves it
        s = make().memmap_(tmpdir)
        s[1, 0] = NonTensorData("x")
        assert s.tolist() == [["s0"] * 3, ["x", "s0", "s0"]]
        assert TensorDict.load_memmap(tmpdir).tolist() == s.tolist()
        # a locked stack takes a value it cannot compare, as before
        s = make(np.zeros(2)).lock_()
        with pytest.warns(UserWarning, match="may or may not be equal"):
            s[1, 0] = NonTensorData(np.zeros(2))

    def test_setitem_other_value_raises(self):
        # One NonTensorData holds one value: it cannot take another value at
        # an index
        data = NonTensorData("s0", batch_size=[3])
        with pytest.raises(ValueError, match="one value for all its elements"):
            data[1] = NonTensorData("x")
        with pytest.raises(ValueError, match="one value for all its elements"):
            data[1:] = NonTensorStack("s0", "x")
        assert data.tolist() == ["s0"] * 3
        data[1] = NonTensorData("s0")
        data[1:] = NonTensorStack("s0", "s0")
        assert data.tolist() == ["s0"] * 3

    def test_nts_setitem_scalar_and_oob(self):
        # a NonTensorStack takes a value, or a list of values, at an index
        stack = NonTensorStack("s0", "s1", "s2")
        stack[0] = "x"
        assert stack.tolist() == ["x", "s1", "s2"]
        with pytest.raises(IndexError, match="index 99 is out of bounds"):
            stack[99] = "y"
        assert stack.tolist() == ["x", "s1", "s2"]
        stack[1:3] = ["p", "q"]
        stack[-1] = 5
        assert stack.tolist() == ["x", "p", 5]
        stack[..., 0] = "e"
        stack[torch.tensor([False, True, True])] = "m"
        assert stack.tolist() == ["e", "m", "m"]

    def test_nts_empty_batch_size_keeps_stack(self):
        # the members of a NonTensorStack stay non-tensors: a batch size
        # with other dims from the stack dim on is not given to them
        stack = NonTensorStack(
            *[NonTensorStack(*[f"s{i}{j}" for j in range(3)]) for i in range(4)]
        )
        for batch_size in ([4], [], [5, 3]):
            empty = stack.empty(batch_size=batch_size, names=None)
            assert type(empty) is NonTensorStack
            assert empty.batch_size == torch.Size([4, 3])
            assert empty.tolist() == stack.tolist()
        # along dim 1, a list is read as a torch.Size: the same leading dims
        # give a NonTensorStack, other leading dims a TensorDict
        stack = lazy_stack(
            [NonTensorStack("a", "b", "c"), NonTensorStack("d", "e", "f")], 1
        )
        assert type(stack) is NonTensorStack and stack.stack_dim == 1
        for batch_size in ([3, 2], torch.Size([3, 2]), [3, 5]):
            empty = stack.empty(batch_size=batch_size)
            assert type(empty) is NonTensorStack
            assert empty.batch_size == torch.Size([3, 2])
            assert empty.tolist() == stack.tolist()
        empty = stack.empty(batch_size=[4, 2])
        assert type(empty) is TensorDict
        assert empty.batch_size == torch.Size([4, 2])

    def test_nontensorstack_repr_and_to_tensordict(self):
        stack = NonTensorStack.from_list(["a", "b"])
        assert repr(stack) == (
            "NonTensorStack(\n"
            "    ['a', 'b'],\n"
            "    batch_size=torch.Size([2]),\n"
            "    device=None)"
        )
        # the data is cut after 50 characters
        long = NonTensorStack.from_list(["x" * 30, "y" * 30])
        assert repr(long).splitlines()[1] == (
            "    ['" + "x" * 30 + "', '" + "y" * 14 + "...,"
        )
        # a NonTensorStack is already the dense form of its data
        assert stack.to_tensordict() is stack
        assert stack.to_tensordict(retain_none=True) is stack

    def test_nontensorstack_tolist_convert_nodes(self):
        stack = NonTensorStack.from_list(["a", "b"])
        assert stack.tolist(convert_nodes=True) == ["a", "b"]
        assert stack.tolist(convert_nodes=False) == ["a", "b"]

    @pytest.mark.parametrize(
        "index,rows",
        [
            (torch.tensor([0, 2]), [0, 2]),
            (torch.tensor([True, False, True, False]), [0, 2]),
            (slice(0, 2), [0, 1]),
        ],
    )
    @pytest.mark.parametrize("source", ["stack", "data", "nested"])
    def test_setitem_new_key(self, index, rows, source):
        # A non-tensor stack that the destination lacks gets the full batch
        # size, with None in the rows not written, as a tensor key gets zeros
        if source == "data":
            # A NonTensorData keeps its one value for all rows
            text, values, rows = NonTensorData("a", batch_size=[2]), ["a"] * 4, range(4)
        else:
            text, values = NonTensorStack("a", "b"), ["a", "b"]
        src = TensorDict(obs=torch.ones(2), text=text, batch_size=[2])
        key = "text"
        if source == "nested":
            src = TensorDict(nested=src, batch_size=[2])
            key = ("nested", "text")
        out = TensorDict(obs=torch.zeros(4), batch_size=[4])
        out[index] = src
        expected = [None] * 4
        for row, value in zip(rows, values):
            expected[row] = value
        assert out.get(key).batch_size == out.batch_size
        assert out.get(key).tolist() == expected

    def test_stack(self, non_tensor_data):
        assert (
            LazyStackedTensorDict.lazy_stack([non_tensor_data, non_tensor_data], 0).get(
                ("nested", "int")
            )
            == NonTensorData(3, batch_size=[2])
        ).all()
        assert torch.stack([non_tensor_data, non_tensor_data], 0).get_non_tensor(
            ("nested", "int")
        ) == [3, 3]
        with pytest.warns(
            DeprecationWarning,
            match=r"set_capture_non_tensor_stack\(True\) is deprecated",
        ):
            capture = set_capture_non_tensor_stack(True)
        with capture:
            assert capture_non_tensor_stack()
            assert (
                torch.stack([non_tensor_data, non_tensor_data], 0).get_non_tensor(
                    ("nested", "int")
                )
                == 3
            )
        with set_capture_non_tensor_stack(False):
            assert not capture_non_tensor_stack()
            assert torch.stack([non_tensor_data, non_tensor_data], 0).get_non_tensor(
                ("nested", "int")
            ) == [3, 3]

        assert capture_non_tensor_stack(allow_none=True) is None
        assert isinstance(
            torch.stack([non_tensor_data, non_tensor_data], 0).get(("nested", "int")),
            NonTensorStack,
        )
        with set_capture_non_tensor_stack(False):
            assert isinstance(
                torch.stack([non_tensor_data, non_tensor_data], 0).get(
                    ("nested", "int")
                ),
                NonTensorStack,
            )

        non_tensor_copy = non_tensor_data.clone()
        non_tensor_copy.get(("nested", "int")).data = 4
        assert isinstance(
            torch.stack([non_tensor_data, non_tensor_copy], 0).get(("nested", "int")),
            LazyStackedTensorDict,
        )

    @pytest.mark.parametrize("key", ["a", ("nested", "a")])
    def test_stack_consolidate(self, key):
        td = torch.stack(
            [
                TensorDict({key: "a string", "b": "b string"}),
                TensorDict({key: "another string", "b": "bnother string"}),
            ]
        )
        tdc = td.consolidate()
        assert (tdc == td).all()
        assert tdc[key] == ["a string", "another string"]
        replacement = torch.stack(
            [TensorDict({key: "reset", "b": "reset"}) for _ in range(2)]
        )
        td.where(torch.tensor([True, False]), replacement, out=td)
        assert td[key] == ["a string", "reset"]
        assert tdc[key] == ["a string", "another string"]
        assert tdc.is_locked and tdc.is_consolidated()

    def test_assign_non_tensor(self):
        data = TensorDict({}, [1, 10])

        data[0, 0] = TensorDict({"a": 0, "b": "a string!"}, [])

        assert data["b"] == "a string!"
        assert data.get("b").tolist() == [["a string!"] * 10]
        data[0, 1] = TensorDict({"a": 0, "b": "another string!"}, [])
        assert data.get("b").tolist() == [
            ["a string!"] + ["another string!"] + ["a string!"] * 8
        ]

        data = TensorDict({}, [1, 10])

        data[0, 0] = TensorDict({"a": 0, "b": "a string!"}, [])

        data[0, 5:] = TensorDict({"a": torch.zeros(5), "b": "another string!"}, [5])
        assert data.get("b").tolist() == [["a string!"] * 5 + ["another string!"] * 5]

        data = TensorDict({}, [1, 10])

        data[0, 0] = TensorDict({"a": 0, "b": "a string!"}, [])

        data[0, 0::2] = TensorDict(
            {"a": torch.zeros(5, dtype=torch.long), "b": "another string!"}, [5]
        )
        assert data.get("b").tolist() == [["another string!", "a string!"] * 5]

        data = TensorDict({}, [1, 10])

        data[0, 0] = TensorDict({"a": 0, "b": "a string!"}, [])

        data[0] = TensorDict(
            {"a": torch.zeros(10, dtype=torch.long), "b": "another string!"}, [10]
        )
        assert data.get("b").tolist() == [["another string!"] * 10]

    @pytest.mark.parametrize("consolidated", [False, True])
    @pytest.mark.parametrize(
        "write", ["set_", "update_", "update_keys", "copy_", "copy_select"]
    )
    def test_locked_inplace_write(self, consolidated, write):
        # In-place writes of non-tensor entries work on a locked tensordict, as
        # for tensor entries, and a pickle of a consolidated one sees them (#2288)
        td = TensorDict(
            x=torch.zeros(2),
            s=NonTensorData("hello"),
            sub=TensorDict(t=NonTensorStack("a", "b"), batch_size=[2]),
            batch_size=[2],
        ).lock_()
        if consolidated:
            td = td.consolidate(metadata=True)
        source = TensorDict(
            x=torch.ones(2),
            s=NonTensorData("world"),
            sub=TensorDict(t=NonTensorStack("c", "d"), batch_size=[2]),
            batch_size=[2],
        )
        if write == "set_":
            td.set_("s", "world")
            td.set_(("sub", "t"), NonTensorStack("c", "d"))
        elif write == "update_":
            td.update_(source)
        elif write == "update_keys":
            td.update_(source, keys_to_update=["s", ("sub", "t")])
        elif write == "copy_":
            td.copy_(source)
        else:
            td.copy_(source.select("s", ("sub", "t")))
        assert td.is_locked
        assert td["s"] == "world"
        assert td["sub", "t"] == ["c", "d"]
        td_rt = pickle.loads(pickle.dumps(td))
        assert td_rt["s"] == "world"
        assert td_rt["sub", "t"] == ["c", "d"]
        # The consolidated metadata holds the old values
        assert not td.is_consolidated()

    @pytest.mark.parametrize("layout", ["1d", "nd_members", "stack_dim1", "nested"])
    @pytest.mark.parametrize("mode", ["plain", "locked", "memmap", "memmap_shared"])
    @pytest.mark.parametrize("write", ["update_", "copy_"])
    def test_update__data_into_stack(self, layout, mode, write, tmpdir):
        # A NonTensorData written into a NonTensorStack goes to every member,
        # also when the members are NonTensorData with batch dims (#2263)
        if layout == "1d":
            s = NonTensorStack("a", "b", "c")
        elif layout == "nd_members":
            s = torch.stack([NonTensorData(f"m{i}", batch_size=[3]) for i in range(4)])
        elif layout == "stack_dim1":
            s = NonTensorStack(
                *[NonTensorData(f"m{i}", batch_size=[4]) for i in range(3)],
                stack_dim=1,
            )
        else:
            s = NonTensorStack(
                *[NonTensorStack(*[f"m{i}{j}" for j in range(3)]) for i in range(4)]
            )
        assert isinstance(s, NonTensorStack)
        td = TensorDict(s=s, batch_size=s.batch_size)
        if mode == "locked":
            td.lock_()
        elif mode != "plain":
            td.memmap_(tmpdir, share_non_tensor=mode == "memmap_shared")
        source = TensorDict(
            x=torch.ones(s.batch_size),
            s=NonTensorData("w", batch_size=s.batch_size),
            batch_size=s.batch_size,
        )
        getattr(td, write)(source.select("s"))
        expected = NonTensorData("w", batch_size=s.batch_size).tolist()
        assert td.get("s").tolist() == expected
        if mode.startswith("memmap"):
            assert TensorDict.load_memmap(tmpdir).get("s").tolist() == expected

    @pytest.mark.parametrize("layout", ["1d", "nd_members", "stack_dim1", "nested"])
    @pytest.mark.parametrize("locked", [False, True])
    @pytest.mark.parametrize("write", ["update_", "copy_"])
    def test_update__data_into_stack_with_tensor(self, layout, locked, write):
        # With a tensor entry in the source, update_ takes its fast route,
        # which writes the NonTensorData into every member of the stack (#2263)
        if layout == "1d":
            s = NonTensorStack("a", "b", "c")
        elif layout == "nd_members":
            s = torch.stack([NonTensorData(f"m{i}", batch_size=[3]) for i in range(4)])
        elif layout == "stack_dim1":
            s = NonTensorStack(
                *[NonTensorData(f"m{i}", batch_size=[4]) for i in range(3)],
                stack_dim=1,
            )
        else:
            s = NonTensorStack(
                *[NonTensorStack(*[f"m{i}{j}" for j in range(3)]) for i in range(4)]
            )
        assert isinstance(s, NonTensorStack)
        td = TensorDict(x=torch.zeros(s.batch_size), s=s, batch_size=s.batch_size)
        if locked:
            td.lock_()
        source = TensorDict(
            x=torch.ones(s.batch_size),
            s=NonTensorData("w", batch_size=s.batch_size),
            batch_size=s.batch_size,
        )
        getattr(td, write)(source)
        assert (td["x"] == 1).all()
        assert isinstance(td.get("s"), NonTensorStack)
        assert (
            td.get("s").tolist() == NonTensorData("w", batch_size=s.batch_size).tolist()
        )

    def test_update__stack_into_data_still_skipped(self):
        # A NonTensorStack in the source stays skipped where the destination
        # holds a NonTensorData
        td = TensorDict(x=torch.zeros(2), s=NonTensorData("hello"), batch_size=[2])
        source = TensorDict(x=torch.ones(2), s=NonTensorStack("a", "b"), batch_size=[2])
        td.update_(source)
        assert (td["x"] == 1).all()
        assert td["s"] == "hello"

    @pytest.mark.parametrize("mode", ["shared", "memmap", "memmap_shared"])
    @pytest.mark.parametrize("stack", [False, True])
    @pytest.mark.parametrize("write", ["update_", "copy_"])
    def test_shared_inplace_write_skips_non_tensor(self, mode, stack, write):
        # update_ and copy_ write the tensor leaves of a shared or memmapped
        # tensordict, and skip with a warning the non-tensor values that its
        # shared buffers cannot store, such as None
        def make(x):
            s = NonTensorStack(None, None) if stack else NonTensorData(None)
            return TensorDict(x=torch.full((2,), x), s=s, batch_size=[2])

        td = make(0.0)
        if mode == "shared":
            td.share_memory_()
        else:
            td.memmap_(share_non_tensor=mode == "memmap_shared")
        if mode == "memmap":
            # without shared buffers, the memmapped values are rewritten
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                getattr(td, write)(make(1.0))
        else:
            with pytest.warns(
                UserWarning, match=r"skipped the non-tensor entries \['s'\]"
            ) as record:
                getattr(td, write)(make(1.0))
            assert record[0].filename == __file__
        assert (td["x"] == 1).all()
        assert td.get("s").tolist() == [None, None]

    @pytest.mark.parametrize("mode", ["shared", "memmap", "memmap_shared"])
    @pytest.mark.parametrize("with_tensor", [False, True])
    @pytest.mark.parametrize("keys_to_update", [False, True])
    def test_shared_update__non_tensor(self, mode, with_tensor, keys_to_update):
        # update_ on a shared or memmapped tensordict writes the non-tensor
        # values that its storage can store, and skips the others with a
        # warning, whether or not the source also holds a tensor (#2487)
        td = TensorDict(a=torch.zeros(3), batch_size=[3])
        td.set_non_tensor("s", "hello")
        td.set_non_tensor("i", 3)
        td.set_non_tensor("f", 1.5)
        td.set_non_tensor("by", b"xx")
        td.set_non_tensor("n", None)
        td.set_non_tensor("j", 4)
        td.set_non_tensor("l", [1])
        td.set_non_tensor("d", {"x": 1})
        if mode == "shared":
            td.share_memory_()
        else:
            td.memmap_(share_non_tensor=mode == "memmap_shared")

        def update_(entries):
            source = TensorDict(batch_size=[3])
            for key, value in entries.items():
                source.set_non_tensor(key, value)
            if with_tensor:
                source["a"] = torch.ones(3)
            kwargs = {"keys_to_update": list(source.keys())} if keys_to_update else {}
            return td.update_(source, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # a shared list takes any iterable and a shared dict any pairs,
            # as their extend and update do
            update_({"s": "bye", "i": 7, "f": 2, "l": "ab", "d": [("k", 2)]})
        assert td["s"] == "bye" and td["i"] == 7 and td["f"] == 2
        if mode == "memmap":
            assert td["l"] == "ab" and td["d"] == [("k", 2)]
        else:
            assert td["l"] == ["a", "b"] and td["d"] == {"k": 2}
        assert (td["a"] == float(with_tensor)).all()
        entries = {"s": 5, "i": "x", "f": "y", "by": b"yy", "n": "z", "l": 5, "d": "ab"}
        # an array converts to a scalar only if it has one element
        array = np.array([1, 2])
        if mode == "memmap":
            # without shared buffers, any value is stored
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                update_({**entries, "j": array})
            assert {key: td[key] for key in entries} == entries
            assert td["j"] is array
        else:
            with pytest.warns(
                UserWarning, match="skipped the non-tensor entries"
            ) as record:
                update_({**entries, "j": array})
            assert len(record) == 1
            assert record[0].filename == __file__
            assert f"entries {[*entries, 'j']} of" in str(record[0].message)
            assert td["s"] == "bye" and td["i"] == 7 and td["f"] == 2
            assert td["by"] == b"xx" and td["n"] is None and td["j"] == 4
            assert td["l"] == ["a", "b"] and td["d"] == {"k": 2}
        if mode == "shared":
            # a shallow copy shares the values
            td.copy().update_({"s": "copied", "a": torch.full((3,), 2.0)})
            assert td["s"] == "copied" and (td["a"] == 2).all()

    @pytest.mark.parametrize("mode", ["shared", "memmap_shared"])
    @pytest.mark.parametrize("with_tensor", [False, True])
    def test_shared_update__iterator(self, mode, with_tensor):
        # The manager of a shared list or dict cannot pickle a generator: it is
        # skipped with a warning and not read. Other iterators are read once.
        td = TensorDict(a=torch.zeros(3), batch_size=[3])
        td.set_non_tensor("l", [1])
        td.set_non_tensor("d", {"k": 1})
        if mode == "shared":
            td.share_memory_()
        else:
            td.memmap_(share_non_tensor=True)

        def update_(entries):
            source = TensorDict(batch_size=[3])
            for key, value in entries.items():
                source.set_non_tensor(key, value)
            if with_tensor:
                source["a"] = torch.ones(3)
            td.update_(source)

        generators = [(x for x in [1]), (x for x in [("k", 2)])]
        with pytest.warns(
            UserWarning, match=r"skipped the non-tensor entries \['l', 'd'\]"
        ) as record:
            update_({"l": generators[0], "d": generators[1]})
        assert record[0].filename == __file__
        assert [list(gen) for gen in generators] == [[1], [("k", 2)]]
        assert td["l"] == [1] and td["d"] == {"k": 1}
        assert (td["a"] == float(with_tensor)).all()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            update_({"l": ["ab", "cd"], "d": iter([("k", 2)])})
        assert td["l"] == ["ab", "cd"] and td["d"] == {"k": 2}

    @pytest.mark.parametrize("share_non_tensor", [False, True])
    @pytest.mark.parametrize("index", [1, slice(1, 3)])
    @pytest.mark.parametrize("write", ["update_", "copy_"])
    def test_memmap_stack_view_update__non_tensor(
        self, share_non_tensor, index, write, tmp_path
    ):
        # The members of a memmapped NonTensorStack are written only through
        # the stack: a view at an index skips them with a warning, and still
        # writes the tensors
        td = TensorDict(a=torch.zeros(4), batch_size=[4])
        td["s"] = NonTensorStack("a", "b", "c", "d")
        td.memmap_(tmp_path, share_non_tensor=share_non_tensor)
        view = td[index]
        source = TensorDict(a=torch.ones(view.shape), batch_size=view.shape)
        source.set_non_tensor("s", "z")
        with pytest.warns(
            UserWarning, match=r"skipped the non-tensor entries \['s'\]"
        ) as record:
            getattr(view, write)(source)
        assert record[0].filename == __file__
        expected = torch.zeros(4)
        expected[index] = 1
        assert (td["a"] == expected).all()
        assert td.get("s").tolist() == ["a", "b", "c", "d"]
        loaded = TensorDict.load_memmap(tmp_path)
        assert loaded.get("s").tolist() == ["a", "b", "c", "d"]

    @pytest.mark.parametrize(
        "route", ["update_", "update_no_tensor", "copy_", "keys_to_update"]
    )
    @pytest.mark.parametrize("unpicklable", ["lock", "ctypes_pointer"])
    def test_memmap_update__unpicklable(self, route, unpicklable, tmp_path):
        # A value that cannot be pickled is one that a memmapped tensordict
        # cannot store: update_ skips it with a warning and writes the others.
        # Pickling a lock raises a TypeError, a ctypes pointer a ValueError.
        td = TensorDict(a=torch.zeros(3), batch_size=[3])
        td.set_non_tensor("o", 1)
        td.set_non_tensor("s", "old")
        td.memmap_(tmp_path)
        source = TensorDict(batch_size=[3])
        if unpicklable == "lock":
            source.set_non_tensor("o", threading.Lock())
        else:
            source.set_non_tensor("o", ctypes.pointer(ctypes.c_int(1)))
        source.set_non_tensor("s", "new")
        if route != "update_no_tensor":
            source["a"] = torch.ones(3)
        with pytest.warns(
            UserWarning, match=r"skipped the non-tensor entries \['o'\]"
        ) as record:
            if route == "copy_":
                td.copy_(source)
            elif route == "keys_to_update":
                td.update_(source, keys_to_update=list(source.keys()))
            else:
                td.update_(source)
        assert record[0].filename == __file__
        expected_a = 0.0 if route == "update_no_tensor" else 1.0
        assert (td["a"] == expected_a).all() and td["s"] == "new"
        assert td["o"] == 1 and td.get("o").is_memmap()
        loaded = TensorDict.load_memmap(tmp_path, allow_pickle=True)
        assert loaded["o"] == 1 and loaded["s"] == "new"
        assert (loaded["a"] == expected_a).all()

    def test_ignore_lock(self):
        td = TensorDict({"a": {"b": "1"}}, batch_size=[10])
        td.lock_()
        td[0] = TensorDict({"a": {"b": "0"}}, [])
        assert td.is_locked
        assert td["a"].is_locked
        assert td[0]["a", "b"] == "0"
        assert td[1]["a", "b"] == "1"

    PAIRS = [
        ("something", "something else"),
        (0, 1),
        (0.0, 1.0),
        ([0, "something", 2], [9, "something else", 11]),
        ({"key1": 1, 2: 3}, {"key1": 4, 5: 6}),
    ]

    @pytest.mark.parametrize("pair", PAIRS)
    @pytest.mark.parametrize("strategy", ["shared", "memmap"])
    @pytest.mark.parametrize("update", ["update_", "update-inplace", "update"])
    def test_shared_memmap_single(self, pair, strategy, update, tmpdir):
        val0, val1 = pair
        td = TensorDict({"val": NonTensorData(data=val0, batch_size=[])}, [])
        if strategy == "shared":
            td.share_memory_()
        elif strategy == "memmap":
            td.memmap_(tmpdir)
        else:
            raise RuntimeError

        # Test that the Value is unpacked
        assert "data" not in td.get("val").__dict__
        assert td.get("val").data == val0
        assert td["val"] == val0

        # Check shared status
        if strategy == "shared":
            assert td._is_shared
            assert td.get("val")._is_shared
            assert td.get("val")._tensordict._is_shared
        elif strategy == "memmap":
            assert td._is_memmap
            assert td.get("val")._is_memmap
            assert td.get("val")._tensordict._is_memmap

            # check that the json has been updated
            td_load = TensorDict.load_memmap(tmpdir, allow_pickle=True)
            assert td["val"] == td_load["val"]
            # with open(Path(tmpdir) / "val" / "meta.json") as file:
            #     print(json.load(file))

        # Update in place
        if update == "setitem":
            td["val"] = val1
        elif update == "update_":
            td.get("val").update_(
                NonTensorData(data=val1, batch_size=[]), non_blocking=False
            )
        elif update == "update-inplace":
            td.get("val").update(
                NonTensorData(data=val1, batch_size=[]),
                inplace=True,
                non_blocking=False,
            )
        elif update == "update":
            with pytest.raises(RuntimeError, match="lock"):
                td.get("val").update(
                    NonTensorData(data="something else", batch_size=[])
                )
            return

        # Test that the Value is unpacked
        assert td.get("val").data == val1
        assert td["val"] == val1

        # Check shared status
        if strategy == "shared":
            assert td._is_shared
            assert td.get("val")._is_shared
            assert td.get("val")._tensordict._is_shared
        elif strategy == "memmap":
            assert td._is_memmap
            assert td.get("val")._is_memmap
            assert td.get("val")._tensordict._is_memmap

            # check that the json has been updated
            td_load = TensorDict.load_memmap(tmpdir, allow_pickle=True)
            assert td["val"] == td_load["val"]
            # with open(Path(tmpdir) / "val" / "meta.json") as file:
            #     print(json.load(file))

    @staticmethod
    def _run_worker(td, val1, update):
        # print the child's Python stack if it dies on a signal (#2669)
        faulthandler.enable()
        set_list_to_stack(True).set()
        # Update in place
        if update == "setitem":
            td["val"] = NonTensorData(val1)
        elif update == "update_":
            td.get("val").update_(NonTensorData(data=val1), non_blocking=False)
        elif update == "update-inplace":
            td.get("val").update(
                NonTensorData(data=val1),
                inplace=True,
                non_blocking=False,
            )
        else:
            raise NotImplementedError
        # Test that the Value is unpacked
        assert td.get("val").data == val1
        assert td["val"] == val1

    @pytest.mark.slow
    @set_list_to_stack(True)
    @pytest.mark.parametrize("pair", PAIRS)
    @pytest.mark.parametrize("strategy", ["shared", "memmap"])
    @pytest.mark.parametrize("update", ["update_", "update-inplace"])
    def test_shared_memmap_mult(self, pair, strategy, update, tmpdir):
        from tensordict.tensorclass import _from_shared_nontensor

        val0, val1 = pair
        td = TensorDict({"val": NonTensorData(data=val0, batch_size=[])}, [])
        if strategy == "shared":
            td.share_memory_()
        elif strategy == "memmap":
            td.memmap_(tmpdir, share_non_tensor=True)
        else:
            raise RuntimeError

        # Test that the Value is unpacked
        assert td.get("val").data == val0
        assert td["val"] == val0

        # Check shared status
        if strategy == "shared":
            assert td._is_shared
            assert td.get("val")._is_shared
            assert td.get("val")._tensordict._is_shared
        elif strategy == "memmap":
            assert td._is_memmap
            assert td.get("val")._is_memmap
            assert td.get("val")._tensordict._is_memmap

            # check that the json has been updated
            td_load = TensorDict.load_memmap(tmpdir, allow_pickle=True)
            assert td["val"] == td_load["val"]
            # with open(Path(tmpdir) / "val" / "meta.json") as file:
            #     print(json.load(file))

        proc = mp.get_context(mp_ctx).Process(
            target=self._run_worker, args=(td, val1, update)
        )
        proc.start()
        proc.join()
        # a crashed child must show as a crash, not as a missing write (#2669)
        assert proc.exitcode == 0

        # Test that the Value is unpacked
        assert _from_shared_nontensor(td.get("val")._non_tensordict["data"]) == val1
        assert td.get("val").data == val1
        assert td["val"] == val1

        # Check shared status
        if strategy == "shared":
            assert td._is_shared
            assert td.get("val")._is_shared
            assert td.get("val")._tensordict._is_shared
        elif strategy == "memmap":
            assert td._is_memmap
            assert td.get("val")._is_memmap
            assert td.get("val")._tensordict._is_memmap

            # check that the json has been updated
            td_load = TensorDict.load_memmap(tmpdir, allow_pickle=True)
            assert td["val"] == td_load["val"]
            # with open(Path(tmpdir) / "val" / "meta.json") as file:
            #     print(json.load(file))

    @pytest.mark.parametrize("json_serializable", [True, False])
    @pytest.mark.parametrize("device", [None, *get_available_devices()])
    def test_memmap_stack(self, tmpdir, json_serializable, device):
        if json_serializable:
            data = torch.stack(
                [
                    NonTensorData(data=0, device=device),
                    NonTensorData(data=1, device=device),
                ]
            )

        else:
            data = torch.stack(
                [
                    NonTensorData(data=DummyPicklableClass(0), device=device),
                    NonTensorData(data=DummyPicklableClass(1), device=device),
                ]
            )
        data = torch.stack([data] * 3)
        data_memmap = data.memmap(tmpdir)
        device_str = "null" if device is None else f'"{device}"'
        with open(f"{tmpdir}/meta.json") as f:
            if json_serializable:
                assert (
                    f.read()
                    == f'{{"_type":"<class \'tensordict.tensorclass.NonTensorStack\'>","stack_dim":0,"device":{device_str},"data":[[0,1],[0,1],[0,1]]}}'
                )
            else:
                assert (
                    f.read()
                    == f'{{"_type":"<class \'tensordict.tensorclass.NonTensorStack\'>","stack_dim":0,"device":{device_str},"data":"pickle.pkl"}}'
                )
        if not json_serializable:
            with pytest.raises(RuntimeError, match="allow_pickle=True"):
                TensorDict.load_memmap(tmpdir, allow_pickle=False)
        data_recon = TensorDict.load_memmap(tmpdir, allow_pickle=not json_serializable)
        assert data_recon.batch_size == data.batch_size
        assert data_recon.device == data.device
        assert data_recon.tolist() == data.tolist()
        assert data_memmap[0].is_memmap()
        assert data_memmap.is_memmap()
        assert data_memmap._is_memmap

    def test_memmap_stack_overwrite_removes_stale_pickle(self, tmp_path):
        pickled = torch.stack(
            [
                NonTensorData(data=1 + 2j, batch_size=[]),
                NonTensorData(data=3 + 4j, batch_size=[]),
            ]
        )
        pickled.memmap(tmp_path)
        pickle_path = tmp_path / "pickle.pkl"
        assert pickle_path.exists()

        json_data = torch.stack(
            [
                NonTensorData(data="a", batch_size=[]),
                NonTensorData(data="b", batch_size=[]),
            ]
        )
        json_data.memmap(tmp_path)

        assert not pickle_path.exists()
        loaded = TensorDict.load_memmap(tmp_path, allow_pickle=False)
        assert loaded.tolist() == ["a", "b"]

    @pytest.mark.parametrize("stack", [False, True])
    @pytest.mark.parametrize("old", [1, {1, 2}])
    def test_memmap_unpicklable_keeps_files(self, tmp_path, stack, old):
        # memmap_ over an existing memmap with a value that cannot be pickled
        # raises and leaves the files that are already there as they were
        def make(value):
            if stack:
                return NonTensorStack(value, value)
            return NonTensorData(value)

        def files():
            return {
                path: path.read_bytes()
                for path in tmp_path.rglob("*")
                if path.is_file()
            }

        make(old).memmap_(tmp_path)
        before = files()
        with pytest.raises(TypeError, match="pickle"):
            make(threading.Lock()).memmap_(tmp_path)
        assert files() == before
        loaded = TensorDict.load_memmap(tmp_path, allow_pickle=True)
        assert loaded.tolist() == make(old).tolist()

    @pytest.mark.parametrize("route", ["save", "memmap", "memmap_", "consolidate"])
    def test_save_non_finite_floats(self, tmp_path, route):
        # gh-2913: NaN and infinite floats in non-tensor data load back as they were
        td = TensorDict(
            best_loss=NonTensorData(float("inf")),
            low=NonTensorData(float("-inf")),
            lr=NonTensorData(float("nan")),
            config=NonTensorData({"lr": 1e-3, "best": float("inf")}),
        )
        if route == "save":
            td.save(tmp_path)
            loaded = TensorDict.load(tmp_path)
        elif route == "memmap":
            td.memmap(tmp_path)
            loaded = TensorDict.load_memmap(tmp_path)
        elif route == "memmap_":
            td.memmap_(tmp_path)
            loaded = TensorDict.load_memmap(tmp_path)
        else:
            td.consolidate(filename=tmp_path / "file.mmap")
            loaded = TensorDict.from_consolidated(tmp_path / "file.mmap")
        assert loaded["best_loss"] == float("inf")
        assert loaded["low"] == float("-inf")
        assert math.isnan(loaded["lr"])
        assert loaded["config"] == {"lr": 1e-3, "best": float("inf")}

    @pytest.mark.parametrize("value", [{1, 2}, {"a": {1, 2}}, [{1, 2}]])
    def test_memmap_set_is_pickled(self, tmp_path, value):
        td = TensorDict(x=torch.zeros(3), batch_size=[3])
        td.set_non_tensor("v", value)
        td.memmap(tmp_path)
        with pytest.raises(RuntimeError, match="allow_pickle=True"):
            TensorDict.load_memmap(tmp_path, allow_pickle=False)
        loaded = TensorDict.load_memmap(tmp_path, allow_pickle=True)
        assert loaded["v"] == value

    def test_memmap_stack_of_sets_is_pickled(self, tmp_path):
        data = NonTensorStack({1, 2}, {3})
        data.memmap(tmp_path)
        assert (tmp_path / "pickle.pkl").exists()
        loaded = TensorDict.load_memmap(tmp_path, allow_pickle=True)
        assert loaded.tolist() == [{1, 2}, {3}]

    @pytest.mark.parametrize(
        "backend",
        [
            "json",
            pytest.param(
                "orjson",
                marks=pytest.mark.skipif(not _has_orjson, reason="orjson not found"),
            ),
        ],
    )
    def test_memmap_int_outside_64_bit(self, tmp_path, backend, monkeypatch):
        # orjson only writes ints that fit in 64 bits
        monkeypatch.setattr(_utils_key_json, "_JSON_BACKEND", backend)
        td = TensorDict(n=NonTensorData(2**64), m=NonTensorData(-(2**63) - 1))
        td.memmap_(tmp_path / "td")
        loaded = TensorDict.load_memmap(tmp_path / "td", allow_pickle=False)
        assert loaded["n"] == 2**64
        assert loaded["m"] == -(2**63) - 1
        NonTensorStack(2**64, 1).memmap(tmp_path / "stack")
        loaded = TensorDict.load_memmap(tmp_path / "stack", allow_pickle=False)
        assert loaded.tolist() == [2**64, 1]

    def test_load_memmap_refreshes_non_tensor(self, tmp_path):
        def make(version):
            return TensorDict(
                {
                    "a": torch.full((2,), float(version)),
                    "data": NonTensorData(f"v{version}", batch_size=[2]),
                    "stack": NonTensorStack(f"s{version}", f"t{version}"),
                    "nested": {"data": NonTensorData(f"w{version}", batch_size=[2])},
                },
                batch_size=[2],
            )

        make(2).memmap(tmp_path / "src")
        dest = make(1).memmap(tmp_path / "dest")
        dest.load_memmap_(tmp_path / "src", allow_pickle=False)
        assert (dest["a"] == 2).all()
        assert dest["data"] == "v2"
        assert list(dest["stack"]) == ["s2", "t2"]
        assert dest["nested", "data"] == "w2"
        assert dest.is_memmap()

    def test_memmap_update_writes_json_only(self, tmp_path):
        td = TensorDict(
            x=torch.zeros(3), s=NonTensorData("hello"), batch_size=[3]
        ).memmap(tmp_path / "src")
        td.update_(TensorDict(s=NonTensorData("world"), batch_size=[3]))
        assert sorted(os.listdir(tmp_path / "src" / "s")) == ["meta.json"]
        loaded = TensorDict.load_memmap(tmp_path / "src", allow_pickle=False)
        assert loaded["s"] == "world"

        td.memmap(tmp_path / "dest", copy_existing=True)
        assert sorted(os.listdir(tmp_path / "dest" / "s")) == ["meta.json"]
        loaded = TensorDict.load_memmap(tmp_path / "dest", allow_pickle=False)
        assert loaded["s"] == "world"

    @pytest.mark.parametrize("allow_pickle", [0, 1, np.bool_(False), np.bool_(True)])
    def test_memmap_pickle_policy_requires_bool(self, tmp_path, allow_pickle):
        td = TensorDict(
            {"value": NonTensorData(data=1 + 2j, batch_size=[])},
            batch_size=[],
        )
        td.memmap(tmp_path)

        with pytest.raises(TypeError, match="allow_pickle must be a bool or None"):
            TensorDict.load_memmap(tmp_path, allow_pickle=allow_pickle)

    def test_memmap_pickle_rejected_before_execution(self, tmp_path):
        global _PICKLE_LOAD_EXECUTED

        td = TensorDict(
            {
                "tensor": torch.zeros(1),
                "value": NonTensorData(data=_PickleLoadMarker(), batch_size=[]),
            },
            batch_size=[],
        )
        td.memmap_(tmp_path)
        _PICKLE_LOAD_EXECUTED = False

        with pytest.raises(RuntimeError, match="allow_pickle=True"):
            TensorDict.load_memmap(tmp_path, allow_pickle=False)
        assert not _PICKLE_LOAD_EXECUTED

        with pytest.raises(RuntimeError, match="allow_pickle=True"):
            td.memmap_refresh_(allow_pickle=False)
        assert not _PICKLE_LOAD_EXECUTED

        td = TensorDict.load_memmap(tmp_path, allow_pickle=True)
        assert _PICKLE_LOAD_EXECUTED
        assert td["value"] == "loaded"

    def test_nts_update_kwargs_typeerror(self):
        # A NonTensorStack is a value: update and update_ take no entries as
        # keyword arguments
        stack = NonTensorStack("a", "b")
        with pytest.raises(TypeError, match="keyword-argument"):
            stack.update(x=NonTensorData("c"))
        with pytest.raises(TypeError, match="keyword-argument"):
            stack.update_(x=NonTensorData("c"))
        assert stack.update(None) is stack and stack.update_(None) is stack
        with pytest.warns(DeprecationWarning, match="ignore_lock") as record:
            assert stack.update(NonTensorStack("c", "d"), ignore_lock=True) is stack
        assert record[0].filename == __file__
        assert stack.tolist() == ["c", "d"]
        with pytest.warns(DeprecationWarning, match=r"update_at_\(index=") as record:
            assert stack.update_at_(NonTensorData("e"), index=1) is stack
        assert record[0].filename == __file__
        assert stack.tolist() == ["c", "e"]

    def test_update_at_idx_keyword(self):
        # NonTensorData.update_at_ names the index idx, as
        # TensorDict.update_at_ does, and still takes index= until 0.17
        data = NonTensorData("a")
        assert data.update_at_(NonTensorData("b"), ()) is data
        assert data.data == "b"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert data.update_at_(NonTensorData("c"), idx=()) is data
        assert data.data == "c"
        with pytest.warns(DeprecationWarning, match=r"update_at_\(index=") as record:
            assert data.update_at_(NonTensorData("d"), index=()) is data
        assert record[0].filename == __file__
        assert data.data == "d"

    @pytest.mark.skipif(IS_FB, reason="deactivating on fbcode")
    def test_memmap_stack_updates(self, tmpdir):
        data = torch.stack([NonTensorData(data=0), NonTensorData(data=1)], 0)
        assert is_non_tensor(data)
        data = torch.stack([data] * 3)
        assert is_non_tensor(data), data
        data = data.clone()
        assert is_non_tensor(data)
        data.memmap_(tmpdir)
        data_recon = TensorDict.load_memmap(tmpdir)
        assert data.tolist() == data_recon.tolist()
        assert data.is_memmap()
        assert data._is_memmap
        assert data[0, 0]._is_memmaped_from_above()

        data_other = torch.stack([NonTensorData(data=2), NonTensorData(data=3)], 0)
        data_other = torch.stack([data_other] * 3)
        with pytest.raises(RuntimeError, match="locked"):
            data.update(data_other)
        data.update(data_other, inplace=True, non_blocking=False)
        assert data[0, 0]._is_memmaped_from_above()
        assert data.is_memmap()
        assert data._is_memmap
        assert data.tolist() == [[2, 3]] * 3
        assert data.tolist() == TensorDict.load_memmap(tmpdir).tolist()

        data_other = torch.stack([NonTensorData(data=4), NonTensorData(data=5)], 0)
        data_other = torch.stack([data_other] * 3)
        data.update_(data_other)
        assert data[0, 0]._is_memmaped_from_above()
        assert data.is_memmap()
        assert data._is_memmap
        assert data.tolist() == [[4, 5]] * 3
        assert data.tolist() == TensorDict.load_memmap(tmpdir).tolist()

        data.update(NonTensorData(data=6), inplace=True, non_blocking=False)
        assert data.is_memmap()
        assert data._is_memmap
        assert data.tolist() == [[6] * 2] * 3
        assert data.tolist() == TensorDict.load_memmap(tmpdir).tolist()

        data.update_(NonTensorData(data=7))
        assert data.is_memmap()
        assert data._is_memmap
        assert data.tolist() == [[7] * 2] * 3
        assert data.tolist() == TensorDict.load_memmap(tmpdir).tolist()

        assert data[0, 0]._is_memmaped_from_above()
        # Should raise an exception
        assert isinstance(data[0, 0], NonTensorData)
        with pytest.raises(
            RuntimeError,
            match="Cannot update a leaf NonTensorDataBase from a memmaped parent NonTensorStack",
        ):
            data[0, 0].update(NonTensorData(data=1), inplace=True, non_blocking=False)

        # Should raise an exception
        with pytest.raises(
            RuntimeError,
            match="Cannot update a leaf NonTensorDataBase from a memmaped parent NonTensorStack",
        ):
            data[0].update(NonTensorData(data=1), inplace=True, non_blocking=False)

        # should raise an exception
        with pytest.raises(
            ValueError,
            match="Cannot update a NonTensorDataBase object with a NonTensorStack",
        ):
            out = NonTensorData(data=1).update(data, inplace=True, non_blocking=False)
        # as suggested by the error message this works
        out = (
            NonTensorData(data=1, batch_size=data.batch_size)
            .maybe_to_stack()
            .update(data, inplace=True, non_blocking=False)
        )
        assert out.tolist() == data.tolist()

        data[0, 0] = NonTensorData(data=99)
        assert data.tolist() == [[99, 7], [7, 7], [7, 7]]
        assert data.tolist() == TensorDict.load_memmap(tmpdir).tolist(), (
            TensorDict.load_memmap(tmpdir).tolist()
        )

        data.update_at_(NonTensorData(data=99), (0, 1))
        assert data.tolist() == [[99, 99], [7, 7], [7, 7]], data.tolist()
        assert data.tolist() == TensorDict.load_memmap(tmpdir).tolist(), (
            TensorDict.load_memmap(tmpdir).tolist()
        )

    def test_from_nontensordata_is_deprecated(self):
        data = NonTensorData(data={"x": 1}, batch_size=[2, 3])
        with pytest.warns(
            DeprecationWarning,
            match=r"^NonTensorStack\.from_nontensordata\(\) is deprecated and will be "
            r"removed in TensorDict 0\.17\. Use NonTensorData\.maybe_to_stack\(\) "
            r"instead\.$",
        ) as record:
            stack = NonTensorStack.from_nontensordata(data)
        assert record[0].filename == __file__
        assert isinstance(stack, NonTensorStack)
        assert stack.batch_size == torch.Size([2, 3])
        assert stack.tolist() == data.maybe_to_stack().tolist()
        # every element holds its own copy of the data
        assert stack[0, 0].data is not stack[0, 1].data

    def test_shared_limitations(self):
        # Sharing a special type works but it's locked for writing
        @dataclass
        class MyClass:
            string: str

        val0 = MyClass(string="a string!")

        td = TensorDict({"val": NonTensorData(data=val0, batch_size=[])}, [])
        td.share_memory_()

        # with pytest.raises(RuntimeError)
        val1 = MyClass(string="another string!")
        with pytest.raises(ValueError, match="Failed to update 'val' in tensordict"):
            td.update(
                TensorDict({"val": NonTensorData(data=val1, batch_size=[])}, []),
                inplace=True,
                non_blocking=False,
            )
        # update_ skips what the shared value cannot store (#2487)
        with pytest.warns(
            UserWarning, match=r"skipped the non-tensor entries \['val'\]"
        ):
            td.update_(TensorDict({"val": NonTensorData(data=val1, batch_size=[])}, []))
        assert td["val"] == val0

        # We can update a batched NonTensorData to a NonTensorStack if it's not already shared
        td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
        td[1::2] = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])
        assert td.get("val").tolist() == [0, 1] * 5
        td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
        td.share_memory_()
        with pytest.raises(
            RuntimeError,
            match="You're attempting to update a leaf in-place with a shared",
        ):
            td[1::2] = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])

        # A shared bool only takes what a shared int takes
        td = TensorDict({"val": NonTensorData(data=True, batch_size=[])}, [])
        td.share_memory_()
        with pytest.warns(
            UserWarning, match=r"skipped the non-tensor entries \['val'\]"
        ):
            td.update_(TensorDict({"val": NonTensorData(data=None, batch_size=[])}, []))
        assert td["val"] is True
        with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
            td.get("val").copy_(NonTensorData(data=None, batch_size=[]))

    @pytest.mark.parametrize("strategy", ["shared", "memmap"])
    @pytest.mark.parametrize(
        "val0,val1",
        [
            (2**40, 2**41),
            (True, False),
            ("x" * 150, "y" * 150),
            # Kept unshared, like the unsupported types: no in-place update
            (b"abc", None),
        ],
        ids=["int64", "bool", "long_str", "bytes"],
    )
    def test_shared_values(self, val0, val1, strategy, tmpdir):
        # Sharing keeps the type and the whole value
        td = TensorDict({"val": NonTensorData(data=val0, batch_size=[])}, [])
        if strategy == "shared":
            td.share_memory_()
        else:
            td.memmap_(tmpdir, share_non_tensor=True)
        assert type(td["val"]) is type(val0)
        assert td["val"] == val0
        if val1 is not None:
            td.update_(TensorDict({"val": NonTensorData(data=val1, batch_size=[])}, []))
            assert type(td["val"]) is type(val1)
            assert td["val"] == val1

    @pytest.mark.parametrize(
        "data",
        [
            "a string",
            3,
            None,
            [1, 2],
            {"a": [1]},
            [torch.zeros(1), np.zeros(1)],
            {"t": torch.zeros(1), "a": (np.zeros(1),)},
        ],
    )
    def test_copy(self, data, monkeypatch):
        nt = NonTensorData(data, batch_size=[2], names=["n"], device="cpu")
        is_leaf = not isinstance(data, (list, dict))

        # A shallow copy skips the pytree pass over the non-tensor fields: of
        # a value that is not a container (#2554), and of dicts and lists
        # (#2862)
        def tree_map(*args, **kwargs):
            raise AssertionError("tree_map was called")

        monkeypatch.setattr(sys.modules[NonTensorData.__module__], "tree_map", tree_map)
        for copy in (nt.copy(), nt._clone(recurse=False)):
            assert type(copy) is NonTensorData
            assert copy._tensordict is not nt._tensordict
            assert copy.batch_size == nt.batch_size
            assert copy.names == nt.names
            assert copy.device == nt.device
            assert copy.data == data
            # containers are copied, other values are not
            assert (copy.data is nt.data) is is_leaf

    @pytest.mark.parametrize(
        "data",
        [
            "s",
            None,
            b"b",
            [{"role": "user", "content": "q"}],
            {"a": 1, "b": [1.5, {"c": (True, "d")}], "o": object()},
            (1, "x", [2]),
            collections.OrderedDict(a=[1]),
            collections.namedtuple("NT", "a b")(1, [2]),
            torch.arange(3),
            np.arange(3),
            TensorDict(a=torch.zeros(2)),
            {"td": TensorDict(a=torch.zeros(2)), "l": [torch.ones(1)]},
        ],
        ids=lambda data: type(data).__name__,
    )
    @pytest.mark.parametrize("cls", ["NonTensorData", "MetaData", "subclass"])
    @pytest.mark.parametrize("locked", [False, True])
    def test_copy_matches_general_path(self, data, cls, locked):
        # The fast shallow copy of a NonTensorData gives what
        # TensorClass._clone gives: new dicts, lists and tuples, the same
        # leaves, and the same attributes
        if cls == "subclass":

            class NonTensorDataSubclass(NonTensorData):
                pass

            cls = NonTensorDataSubclass
        else:
            cls = {"NonTensorData": NonTensorData, "MetaData": MetaData}[cls]
        nt = cls(data, batch_size=[2], names=["n"], device="cpu", _metadata={"k": [1]})
        if locked:
            nt.lock_()

        def structure(src, value):
            # the type of each container and whether it is the source's
            result = [type(value), value is src]
            if type(value) in (dict, collections.OrderedDict):
                result += [(k, structure(src[k], v)) for k, v in value.items()]
            elif isinstance(value, (list, tuple)):
                result += [structure(s, v) for s, v in zip(src, value)]
            return result

        expected = TensorClass._clone(nt, recurse=False)
        for copy in (nt.copy(), nt._clone(recurse=False)):
            assert type(copy) is type(expected)
            assert list(copy.__dict__) == list(expected.__dict__)
            assert type(copy._tensordict) is type(expected._tensordict)
            assert copy._tensordict is not nt._tensordict
            assert copy.batch_size == expected.batch_size
            assert copy.names == expected.names
            assert copy.device == expected.device
            assert copy.is_locked == expected.is_locked
            assert list(copy._non_tensordict) == list(expected._non_tensordict)
            for key, value in copy._non_tensordict.items():
                assert structure(nt._non_tensordict[key], value) == structure(
                    nt._non_tensordict[key], expected._non_tensordict[key]
                )

    def test_shared_copy_non_tensor_data(self):
        # copy() of a shared NonTensorData reads the value, not the shared
        # wrapper, as clone(recurse=False) does
        nt = NonTensorData("hello").share_memory_()
        for copy in (nt.copy(), nt.clone(recurse=False)):
            assert copy.data == "hello"

    def test_shared_copy(self):
        # A shallow copy of a shared tensordict reads the values, not the
        # shared wrappers, and keeps sharing them like the tensors
        td = TensorDict(
            a=torch.zeros(3),
            s="hello",
            n=NonTensorData(3.5, batch_size=[3]),
            sub=TensorDict(s="x", batch_size=[3]),
            batch_size=[3],
        ).share_memory_()
        for copy in (td.copy(), td.clone(recurse=False)):
            assert copy["s"] == "hello"
            assert copy["n"] == 3.5
            assert copy["sub", "s"] == "x"
            assert copy.get("s").is_shared()
        copy = td.copy()
        # update_ with tensor entries in the source skips the non-tensor
        # leaves of a shared tensordict (#2487), so the str and the tensor are
        # written in two calls
        copy.update_({"s": "bye"})
        copy.update_({"a": torch.ones(3)})
        assert td["s"] == "bye"
        assert (td["a"] == 1).all()

    def test_memmap_share_non_tensor_copy(self, tmpdir):
        # A shallow copy of a tensordict memory-mapped with share_non_tensor
        # reads the values, not the shared wrappers
        td = TensorDict(
            a=torch.zeros(3),
            s="hello",
            n=NonTensorData(3.5, batch_size=[3]),
            sub=TensorDict(s="x", batch_size=[3]),
            batch_size=[3],
        ).memmap_(tmpdir, share_non_tensor=True)
        for copy in (td.copy(), td.clone(recurse=False)):
            assert copy["s"] == "hello"
            assert copy["n"] == 3.5
            assert copy["sub", "s"] == "x"

    @pytest.mark.parametrize("strategy", ["shared", "memmap"])
    @pytest.mark.parametrize(
        "val1,expected",
        [
            ("y" * 99 + "\u00e9", "y" * 99),
            ("y" * 98 + "\u20ac", "y" * 98),
            ("y" * 97 + "\U0001f600", "y" * 97),
        ],
        ids=["2-byte", "3-byte", "4-byte"],
    )
    def test_shared_str_cut_on_char(self, val1, expected, strategy, tmpdir):
        # A shared str has a fixed size: a longer new value is cut on a
        # character boundary, so it can still be read back
        td = TensorDict({"val": NonTensorData(data="abc", batch_size=[])}, [])
        if strategy == "shared":
            td.share_memory_()
        else:
            td.memmap_(tmpdir, share_non_tensor=True)
        with pytest.warns(UserWarning, match="holds at most 100 bytes"):
            td.update_(TensorDict({"val": NonTensorData(data=val1, batch_size=[])}, []))
        assert td["val"] == expected

    @pytest.mark.parametrize("strategy", ["shared", "memmap"])
    def test_shared_value_lost_warns(self, strategy, tmpdir):
        # Writes that do not fit the shared value warn instead of losing data
        # silently (#2163); writes that fit do not warn.
        td = TensorDict(s=NonTensorData("abc"), i=NonTensorData(3))
        if strategy == "shared":
            td.share_memory_()
        else:
            td.memmap_(tmpdir, share_non_tensor=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            td.update_(
                TensorDict(s=NonTensorData("y" * 100), i=NonTensorData(2**63 - 1))
            )
            td.update_(TensorDict(i=NonTensorData(-(2**63))))
        with pytest.warns(UserWarning, match="the new 120-byte value was cut"):
            td.update_(TensorDict(s=NonTensorData("y" * 120)))
        assert td["s"] == "y" * 100
        with pytest.warns(UserWarning, match="it is stored as 0"):
            td.update_(TensorDict(i=NonTensorData(2**64)))
        assert td["i"] == 0
        with pytest.warns(UserWarning, match="stored as 9223372036854775807"):
            td.set_("i", NonTensorData(-(2**63) - 1))
        assert td["i"] == 2**63 - 1
        if strategy == "shared":
            td = TensorDict(i=NonTensorData(2**64))
            with pytest.warns(UserWarning, match="it is stored as 0"):
                td.share_memory_()
            assert td["i"] == 0

    def _update_stack(self, td):
        # print the child's Python stack if it dies on a signal (#2669)
        faulthandler.enable()
        td[1::2] = TensorDict({"val": NonTensorData(data=3, batch_size=[5])}, [5])

    @pytest.mark.slow
    @pytest.mark.parametrize("update", ["update_at_", "slice"])
    @pytest.mark.parametrize(
        "strategy,share_non_tensor",
        [["shared", None], ["memmap", True], ["memmap", False]],
    )
    def test_shared_stack(self, strategy, update, share_non_tensor, tmpdir):
        td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
        newdata = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])
        if update == "slice":
            td[1::2] = newdata
        elif update == "update_at_":
            td.update_at_(newdata, slice(1, None, 2))
        else:
            raise NotImplementedError
        if strategy == "shared":
            td.share_memory_()
        elif strategy == "memmap":
            td.memmap_(tmpdir, share_non_tensor=share_non_tensor)
        else:
            raise NotImplementedError
        assert td.get("val").tolist() == [0, 1] * 5

        newdata = TensorDict({"val": NonTensorData(data=2, batch_size=[5])}, [5])
        if update == "slice":
            td[1::2] = newdata
        elif update == "update_at_":
            td.update_at_(newdata, slice(1, None, 2))
        else:
            raise NotImplementedError

        assert td.get("val").tolist() == [0, 2] * 5
        if strategy == "memmap":
            assert TensorDict.load_memmap(tmpdir).get("val").tolist() == [0, 2] * 5

        proc = mp.get_context(mp_ctx).Process(target=self._update_stack, args=(td,))
        proc.start()
        proc.join()
        # a crashed child must show as a crash, not as a missing write (#2669)
        assert proc.exitcode == 0
        if share_non_tensor in (True, None):
            assert td.get("val").tolist() == [0, 3] * 5
        else:
            assert td.get("val").tolist() == [0, 2] * 5

        if strategy == "memmap":
            assert TensorDict.load_memmap(tmpdir).get("val").tolist() == [0, 3] * 5

    def test_squeeze_unsqueeze(self):
        td = TensorDict(foo=NonTensorData(3))
        tdu = td.unsqueeze(0)
        assert isinstance(tdu.get("foo"), NonTensorStack)
        assert tdu["foo"] == [3]
        assert isinstance(tdu.squeeze()["foo"], int)
        assert isinstance(tdu.squeeze().get("foo"), NonTensorData)
        assert isinstance(tdu.squeeze(0)["foo"], int)
        assert isinstance(tdu.squeeze(0).get("foo"), NonTensorData)

    def test_to_dict_numpy_namedtuple_non_unique_stack(self):
        # retain_none=False drops a stack of None values and keeps a stack of
        # different values, whose .data raises AttributeError
        td = TensorDict(
            a=torch.zeros(2),
            s=NonTensorStack("x", "y"),
            n=NonTensorStack(None, None),
            batch_size=[2],
        )
        as_dict = td.to_dict(retain_none=False)
        assert list(as_dict) == ["a", "s"]
        assert as_dict["s"] == ["x", "y"]
        as_numpy = td.numpy()
        assert list(as_numpy) == ["a", "s"]
        assert isinstance(as_numpy["a"], np.ndarray)
        assert as_numpy["s"] == ["x", "y"]
        as_namedtuple = td.to_namedtuple()
        assert as_namedtuple._fields == ("a", "s")
        assert as_namedtuple.s == ["x", "y"]

        # from_pytree stores a list of tensors in a namedtuple as a
        # NonTensorStack, and to_pytree rebuilds the namedtuple with
        # to_namedtuple
        NT = collections.namedtuple("NT", ["a", "b"])
        nt = NT(a=torch.zeros(3), b=[torch.zeros(2), torch.ones(3)])
        nt_recon = TensorDict.from_pytree(nt).to_pytree()
        assert type(nt_recon) is NT
        assert (nt_recon.a == nt.a).all()
        assert len(nt_recon.b) == 2
        assert all((x == y).all() for x, y in zip(nt_recon.b, nt.b))

    def test_view(self):
        td = NonTensorStack(*[str(i) for i in range(60)])
        tdv = td.view(3, 4, 5)
        assert isinstance(tdv, NonTensorStack)
        assert isinstance(tdv[0], NonTensorStack)
        assert isinstance(tdv[0, 0], NonTensorStack)
        assert tdv.shape == (3, 4, 5)
        assert tdv.view(60).shape == (60,)
        assert tdv.view(60).tolist() == [str(i) for i in range(60)]
        assert tdv.flatten().tolist() == [str(i) for i in range(60)]

    def test_view_drop_size_one_dim(self):
        strings = [["00", "01", "02"], ["10", "11", "12"]]
        stack = NonTensorStack.from_list(strings).unsqueeze(1)
        assert stack.shape == (2, 1, 3)
        for out in (stack.view(2, 3), stack.reshape(2, 3)):
            assert isinstance(out, NonTensorStack)
            assert out.shape == (2, 3)
            assert out.tolist() == strings
        td = TensorDict(
            {"i": torch.arange(6).view(2, 3), "s": NonTensorStack.from_list(strings)},
            [2, 3],
        )
        out = td.unsqueeze(1).squeeze()
        assert out.get("s").shape == (2, 3)
        assert out["s"] == strings

    def test_where_copies_entries(self):
        # The result has its own entries, which hold the input data as they are
        class NoDeepCopy:
            def __deepcopy__(self, memo):
                raise TypeError("cannot deep-copy")

        condition = torch.tensor([True, False])
        tensor = NonTensorStack("a", "b")
        other = NonTensorStack("x", "y")
        result = tensor.where(condition, other)
        assert result.tolist() == ["a", "y"]
        result[0] = "z"
        result[1] = "z"
        assert tensor.tolist() == ["a", "b"]
        assert other.tolist() == ["x", "y"]
        data = [NoDeepCopy() for _ in range(4)]
        result = NonTensorStack(*data[:2]).where(condition, NonTensorStack(*data[2:]))
        assert result.tolist() == [data[0], data[3]]

    def test_where(self):
        condition = torch.tensor([True, False])
        tensor = NonTensorStack(
            *[NonTensorData(["a"]), NonTensorData(["a"])], batch_size=(2,)
        )
        other = NonTensorStack(
            *[NonTensorData(["b"]), NonTensorData(["b"])], batch_size=(2,)
        )
        out = NonTensorStack(
            *[NonTensorData(["none"]), NonTensorData(["none"])], batch_size=(2,)
        )
        result = tensor.where(condition=condition, other=other, out=out, pad=0)
        assert result.tolist() == [["a"], ["b"]]
        condition = torch.tensor([True, False])
        tensor = NonTensorStack(
            *[NonTensorData(["a"]), NonTensorData(["a"])], batch_size=(2,)
        )
        other = NonTensorStack(
            *[NonTensorData(["a"]), NonTensorData(["a"])], batch_size=(2,)
        )
        out = NonTensorStack(
            *[NonTensorData(["a"]), NonTensorData(["a"])], batch_size=(2,)
        )
        result = tensor.where(condition=condition, other=other, out=out, pad=0)
        assert result.tolist() == [["a"], ["a"]]

    def test_torch_where_non_tensor_data(self):
        # torch.where is not passed through to NonTensorData.where
        condition = torch.tensor([True, False])
        with pytest.raises(TypeError, match="Multiple dispatch failed"):
            torch.where(
                condition,
                NonTensorData("a", batch_size=(2,)),
                NonTensorData("b", batch_size=(2,)),
            )

    def test_where_non_tensor_data(self):
        # The condition picks the value of each element (#2071)
        condition = torch.tensor([True, False, True, False])
        a = NonTensorData("a", batch_size=(4,))
        b = NonTensorData("b", batch_size=(4,))
        result = a.where(condition, b)
        assert isinstance(result, NonTensorStack)
        assert result.tolist() == ["a", "b", "a", "b"]
        assert a.where(~condition, b).tolist() == ["b", "a", "b", "a"]
        result = a.where(condition, NonTensorStack("w", "x", "y", "z"))
        assert result.tolist() == ["a", "x", "a", "z"]
        assert a.where(condition, NonTensorData("b")).tolist() == ["a", "b"] * 2
        result = NonTensorData("a", batch_size=(4, 2)).where(
            condition, NonTensorData("b", batch_size=(4, 2))
        )
        assert result.tolist() == [["a"] * 2, ["b"] * 2] * 2
        # A uniform result stays a NonTensorData
        for cond, other, value in [
            (torch.ones(4, dtype=torch.bool), b, "a"),
            (torch.zeros(4, dtype=torch.bool), b, "b"),
            (condition, a.clone(), "a"),
            (condition, NonTensorStack("x", "a", "y", "a"), "a"),
        ]:
            result = a.where(cond, other)
            assert isinstance(result, NonTensorData)
            assert result.batch_size == (4,)
            assert result.data == value
        assert NonTensorData("a").where(torch.tensor(False), b[0]).data == "b"
        # out receives the result if it can hold it
        out = NonTensorData("c", batch_size=(4,))
        assert a.where(torch.zeros(4, dtype=torch.bool), b, out=out) is out
        assert out.data == "b"
        out = NonTensorStack("1", "2", "3", "4")
        assert a.where(condition, b, out=out) is out
        assert out.tolist() == ["a", "b", "a", "b"]
        # through TensorDict.where, also with out=, as TorchRL's partial reset
        td = TensorDict(x=torch.zeros(4), s=a, batch_size=(4,))
        other = TensorDict(x=torch.ones(4), s=b, batch_size=(4,))
        assert td.where(condition, other).get("s").tolist() == ["a", "b"] * 2
        td.where(condition, other, out=td, pad=0, update_batch_size=True)
        assert td["x"].tolist() == [0.0, 1.0, 0.0, 1.0]
        assert td.get("s").tolist() == ["a", "b", "a", "b"]
        # an other that does not expand to the batch size raises
        with pytest.raises(ValueError):
            a.where(condition, NonTensorData("b", batch_size=(3, 4)))

    def test_linked_list_str(self):
        td = TensorDict(a=NonTensorStack("foo", "bar"), batch_size=(2,))
        # str must not double-wrap: list has no __str__ of its own, so a
        # naive super().__str__() would bounce back to LinkedList.__repr__.
        assert str(td["a"]) == "LinkedList(['foo', 'bar'])"
        assert repr(td["a"]) == "LinkedList(['foo', 'bar'])"

    def test_nontensorstack_boolean_mask(self):
        td = TensorDict(
            {
                "x": torch.arange(3),
                "instr": NonTensorStack("a", "b", "c"),
            },
            batch_size=[3],
        )
        mask = torch.tensor([True, False, True])
        masked = td[mask]
        assert isinstance(masked.get("instr"), NonTensorStack)
        assert masked.get("instr").batch_size == torch.Size([2])
        assert masked.get("instr").tolist() == ["a", "c"]
        assert masked["instr"] == ["a", "c"]
        # all-False mask
        empty = td[torch.zeros(3, dtype=torch.bool)]
        assert empty.batch_size == torch.Size([0])
        assert empty.get("instr").batch_size == torch.Size([0])
        assert empty.get("instr").tolist() == []

    def test_nontensorstack_boolean_mask_2d(self):
        instr = torch.stack(
            [NonTensorStack(*[f"{i}{j}" for j in range(2)]) for i in range(3)]
        )
        td = TensorDict(
            {"instr": instr, "x": torch.arange(6).view(3, 2)}, batch_size=[3, 2]
        )
        # mask along dim 0 of a 2D NonTensorStack used to raise an IndexError
        mask = torch.tensor([True, False, True])
        masked = td[mask]
        assert masked.get("instr").tolist() == [["00", "01"], ["20", "21"]]
        assert masked["x"].tolist() == [[0, 1], [4, 5]]
        # full-shape mask flattens
        full_mask = torch.tensor([[True, False], [False, True], [True, True]])
        assert td[full_mask].get("instr").tolist() == ["00", "11", "20", "21"]
        # boolean setitem with nested stacks
        value = TensorDict(
            {
                "instr": torch.stack(
                    [NonTensorStack("p", "q"), NonTensorStack("r", "s")]
                ),
                "x": torch.zeros(2, 2, dtype=torch.long),
            },
            batch_size=[2, 2],
        )
        td[mask] = value
        assert td.get("instr").tolist() == [["p", "q"], ["10", "11"], ["r", "s"]]

    def test_nontensorstack_scalar_bool_index(self):
        stack = NonTensorStack("a", "b")
        assert stack[torch.tensor(True)].tolist() == [["a", "b"]]
        assert stack[torch.tensor(False)].batch_size == torch.Size([0, 2])
        indexed = stack[(torch.tensor(True),)]
        assert isinstance(indexed, NonTensorStack)
        assert indexed.batch_size == torch.Size([1, 2])
        assert indexed.tolist() == [["a", "b"]]
        # masking to empty keeps the NonTensorStack type
        empty = stack[torch.zeros(2, dtype=torch.bool)]
        assert isinstance(empty, NonTensorStack)
        assert empty.batch_size == torch.Size([0])
        assert empty.tolist() == []

    def test_nontensorstack_advanced_index_copies(self):
        stack = NonTensorStack("walk", "jump", "stand")
        # advanced indexing returns independent entries, as it does for
        # tensors: neither the source nor the repeated entries alias each other
        gathered = stack[torch.tensor([1, 0, 0, 1, 1])]
        assert gathered.tolist() == ["jump", "walk", "walk", "jump", "jump"]
        gathered[torch.tensor([True, False, False, False, True])] = NonTensorStack(
            "walk", "stand"
        )
        assert gathered.tolist() == ["walk", "walk", "walk", "jump", "stand"]
        assert stack.tolist() == ["walk", "jump", "stand"]
        masked = stack[torch.tensor([True, False, True])]
        masked[0] = "hop"
        assert masked.tolist() == ["hop", "stand"]
        assert stack.tolist() == ["walk", "jump", "stand"]
        listed = stack[[2, 2]]
        listed[1] = "hop"
        assert listed.tolist() == ["stand", "hop"]
        assert stack.tolist() == ["walk", "jump", "stand"]
        # basic indexing keeps view semantics
        view = stack[1:]
        view[0] = "hop"
        assert stack.tolist() == ["walk", "hop", "stand"]

    @staticmethod
    def _advanced_index_stack(kind):
        if kind == "str":
            return NonTensorStack("a", "b", "c", "d")
        if kind == "dict":
            return NonTensorStack.from_list([NonTensorData({"i": i}) for i in range(4)])
        if kind == "nested":
            return NonTensorStack.from_list(
                [[f"{i}{j}" for j in range(2)] for i in range(4)]
            )
        if kind == "locked":
            return NonTensorStack("a", "b", "c", "d").lock_()
        if kind == "named":
            stack = NonTensorStack("a", "b", "c", "d")
            stack.names = ["n"]
            return stack
        if kind == "metadata":
            return NonTensorStack(*[MetaData(f"m{i}") for i in range(4)])
        if kind == "mixed":
            return NonTensorStack("a", MetaData("b"), "c", MetaData("d"))
        if kind == "stack_dim_1":
            return NonTensorStack(
                *[NonTensorStack(*[f"{i}{j}" for j in range(4)]) for i in range(2)],
                stack_dim=1,
            )
        if kind == "shared":
            return NonTensorStack("a", "b", "c", "d").share_memory_()
        raise ValueError(kind)

    @staticmethod
    def _describe_index_result(result, source):
        if not isinstance(result, NonTensorStack):
            return type(result), result
        members = [
            (type(td), td.batch_size, td.is_locked, td._is_shared, td.names)
            for td in result.tensordicts
        ]
        source_members = {id(td) for td in source.tensordicts}
        return (
            type(result),
            result.tolist(),
            result.batch_size,
            result.names,
            result.is_locked,
            result._is_shared,
            result.stack_dim,
            members,
            # the members are copies, distinct from each other
            [id(td) in source_members for td in result.tensordicts],
            len({id(td) for td in result.tensordicts}),
        )

    @pytest.mark.parametrize(
        "kind",
        [
            "str",
            "dict",
            "nested",
            "locked",
            "named",
            "metadata",
            "mixed",
            "stack_dim_1",
            "shared",
        ],
    )
    @pytest.mark.parametrize(
        "index",
        [
            torch.tensor([2, 0, 0, -1]),
            (torch.tensor([1, 3]),),
            [3, 1, 1],
            np.array([0, 2]),
            torch.tensor([True, False, True, True]),
            torch.zeros(4, dtype=torch.bool),
            torch.tensor([], dtype=torch.long),
            torch.tensor([0, 4]),
            torch.tensor([True, False]),
            torch.tensor([[0, 1], [2, 3]]),
            torch.tensor([0.0, 1.0]),
        ],
        ids=[
            "tensor",
            "tuple",
            "list",
            "ndarray",
            "mask",
            "empty_mask",
            "empty_tensor",
            "out_of_range",
            "short_mask",
            "2d",
            "float",
        ],
    )
    @pytest.mark.parametrize("capture", [False, True])
    def test_nontensorstack_advanced_index_fast_path(
        self, kind, index, capture, monkeypatch
    ):
        # the fast path of an advanced index gives what the general path gives
        def get(stack):
            with _set_capture_non_tensor_stack(capture):
                try:
                    result = stack[index]
                except Exception as err:
                    return type(err), str(err)
            return self._describe_index_result(result, stack)

        fast_stack = self._advanced_index_stack(kind)
        fast = get(fast_stack)
        locked = fast_stack.is_locked
        monkeypatch.setattr(NonTensorStack, "_getitem_copies", lambda self, index: None)
        general_stack = self._advanced_index_stack(kind)
        assert get(general_stack) == fast
        assert general_stack.is_locked == locked

    @pytest.mark.parametrize("kind", ["str", "dict", "nested", "locked", "named"])
    def test_nontensorstack_advanced_index_fast_path_taken(self, kind):
        stack = self._advanced_index_stack(kind)
        for index in (torch.tensor([2, 0, 0]), [1, 2], torch.tensor([True] * 4)):
            result = stack._getitem_copies(index)
            assert result is not None
            assert type(result) is NonTensorStack
            assert result.tolist() == stack[index].tolist()
        # members of another type, and capture_non_tensor_stack, take the
        # general path
        assert self._advanced_index_stack("metadata")._getitem_copies([0]) is None
        with _set_capture_non_tensor_stack(True):
            assert stack._getitem_copies([0]) is None

    def test_tensorclass_nontensor_field_gather_then_assign(self):
        @tensorclass
        class Task:
            weight: torch.Tensor
            instruction: str

        library = torch.stack(
            [
                Task(torch.tensor([0.0]), "walk", batch_size=[]),
                Task(torch.tensor([1.0]), "jump", batch_size=[]),
                Task(torch.tensor([2.0]), "stand", batch_size=[]),
            ]
        )
        per_env = library[torch.tensor([1, 0, 0, 1, 1])]
        per_env[torch.tensor([True, False, False, False, True])] = library[
            torch.tensor([0, 2])
        ]
        assert per_env.weight.squeeze(-1).tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]
        assert list(per_env.instruction) == ["walk", "walk", "walk", "jump", "stand"]
        assert list(library.instruction) == ["walk", "jump", "stand"]


class TestMetaData:
    def test_typed_metadata(self):
        d = MetaData[int](0, batch_size=(3,))
        assert d.data == 0
        assert isinstance(d, MetaData[int])
        with pytest.raises(TypeError, match="Expected data of type int, got str"):
            MetaData[int]("a string")
        cls = MetaData[int]
        assert issubclass(cls, MetaData)
        d = cls(0, batch_size=(3,))
        assert isinstance(d, MetaData)
        # Test caching
        assert isinstance(d, MetaData[int])

    def test_expand(self):
        d = MetaData(0, batch_size=(3,))
        d_expand = d.expand((2, 3))
        assert d_expand.shape == (2, 3)
        assert d_expand.data == 0
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        assert isinstance(d_expand, MetaData)
        d = MetaData(0, batch_size=(1,))
        d_expand = d.expand((2, 3))
        assert d_expand.shape == (2, 3)
        assert d_expand.data == 0
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        assert isinstance(d_expand, MetaData)

    def test_expand_nested(self):
        d = TensorDict(foo=MetaData(0, batch_size=(3,)), batch_size=(3,))
        d_expand = d.expand((2, 3)).get("foo")
        assert d_expand.shape == (2, 3)
        assert d_expand.data == 0
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        assert isinstance(d_expand, MetaData)
        d = TensorDict(foo=MetaData(0, batch_size=(1,)), batch_size=(1,))
        d_expand = d.expand((2, 3)).get("foo")
        assert d_expand.shape == (2, 3)
        assert d_expand.data == 0
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        assert isinstance(d_expand, MetaData)

    def test_metadata(self):
        foo = MetaData(3)
        assert is_non_tensor(foo)
        assert _pass_through(foo)

    def test_squeeze_unsqueeze(self):
        td = TensorDict(foo=MetaData(3))
        tdu = td.unsqueeze(0)
        assert isinstance(tdu.get("foo"), MetaData)
        assert tdu["foo"] == 3
        assert isinstance(tdu.squeeze()["foo"], int)
        assert isinstance(tdu.squeeze().get("foo"), MetaData)
        assert isinstance(tdu.squeeze(0)["foo"], int)
        assert isinstance(tdu.squeeze(0).get("foo"), MetaData)

    def test_stack(self):
        data = "a string"
        td0 = TensorDict(foo=MetaData(data))
        td1 = TensorDict(foo=MetaData(data))
        assert isinstance(td0.get("foo"), MetaData)
        assert isinstance(td1.get("foo"), MetaData)
        tds = torch.stack([td0, td1])
        foo = tds.get("foo")
        assert isinstance(foo, MetaData)
        assert foo.batch_size == (2,)
        another_data = "another string"
        td1 = TensorDict(foo=MetaData(another_data))
        tds = torch.stack([td0, td1])
        foo = tds.get("foo")
        assert isinstance(foo, NonTensorStack)
        assert foo.batch_size == (2,)

    @pytest.mark.parametrize(
        "index",
        [0, slice(0, 2), torch.tensor([True, False, True, False])],
        ids=["int", "slice", "mask"],
    )
    def test_setitem_number_leaves_metadata(self, index):
        # A number written at an index goes into the tensors, and a MetaData,
        # which has no value per element, stays as it is (#2794)
        td = TensorDict(a=torch.zeros(4), batch_size=[4])
        td["meta"] = MetaData("cfg")
        expected = torch.zeros(4)
        expected[index] = 1
        with warnings.catch_warnings():
            # the warning of the cast into the non-tensor entries
            warnings.simplefilter("ignore", FutureWarning)
            td[index] = 1
        assert (td["a"] == expected).all()
        assert isinstance(td.get("meta"), MetaData)
        assert td["meta"] == "cfg"

    def test_auto_batch_size_scalar_metadata(self):
        # Regression test for #1696: a scalar MetaData entry must not
        # constrain auto_batch_size_ to an empty batch size.
        td = TensorDict(data=torch.zeros(16, 10), meta=MetaData(slice(-1)))
        td.auto_batch_size_(2)
        assert td.batch_size == torch.Size([16, 10])
        td = TensorDict(data=torch.zeros(16, 10), meta=MetaData(slice(-1)))
        td.auto_batch_size_()
        assert td.batch_size == torch.Size([16, 10])
        # scalar NonTensorData behaves the same way
        td = TensorDict(data=torch.zeros(16, 10), meta=NonTensorData("hi"))
        td.auto_batch_size_(2)
        assert td.batch_size == torch.Size([16, 10])


class TestUnbatchedTensor:
    @staticmethod
    def _result_or_exception(fun, arg):
        try:
            return False, fun(arg)
        except Exception as err:
            return True, (type(err), str(err))

    @classmethod
    def _assert_same_conversion(cls, fun, tensor):
        expected_is_exc, expected = cls._result_or_exception(fun, tensor)
        actual_is_exc, actual = cls._result_or_exception(fun, UnbatchedTensor(tensor))

        assert actual_is_exc is expected_is_exc
        if expected_is_exc:
            assert actual[0] is expected[0]
            assert actual[1] == expected[1]
        else:
            assert actual == expected
            assert not isinstance(actual, UnbatchedTensor)

    def test_auto_batch_size(self):
        td = TensorDict(a=UnbatchedTensor(0), b=torch.randn(2, 3)).auto_batch_size_(
            batch_dims=2
        )
        assert td.shape == (2, 3)
        assert td["a"] == 0

    def test_unbatched(self):
        assert UnbatchedTensor._pass_through
        td = TensorDict(
            a=UnbatchedTensor(torch.randn(10)),
            b=torch.randn(3),
            batch_size=(3,),
        )
        assert _pass_through(td.get("a"))
        assert isinstance(td["a"], torch.Tensor)
        assert isinstance(td.get("a"), UnbatchedTensor)

    def test_unbatched_shape_ops(self):
        td = TensorDict(
            a=UnbatchedTensor(torch.randn(10)),
            b=torch.randn(3),
            batch_size=(3,),
        )

        def assert_unbatched_metadata(result):
            assert result["a"].data_ptr() == td["a"].data_ptr()
            assert result["a"].batch_size == result.batch_size
            _check_recursive_properties(result)

        # get item
        assert_unbatched_metadata(td[0])
        assert_unbatched_metadata(td[:])

        unbind = td.unbind(0)[0]
        assert unbind.batch_size == ()
        assert_unbatched_metadata(unbind)

        split = td.split(1)[0]
        assert split.batch_size == (1,)
        assert_unbatched_metadata(split)
        assert_unbatched_metadata(td.split((2, 1))[0])

        reshape = td.reshape((1, 3))
        assert reshape.batch_size == (1, 3)
        assert_unbatched_metadata(reshape)
        transpose = reshape.transpose(0, 1)
        assert transpose.batch_size == (3, 1)
        assert_unbatched_metadata(transpose)
        permute = reshape.permute(1, 0)
        assert permute.batch_size == (3, 1)
        assert_unbatched_metadata(permute)
        squeeze = reshape.squeeze()
        assert squeeze.batch_size == (3,)
        assert_unbatched_metadata(squeeze)

        view = td.view((1, 3))
        assert view.batch_size == (1, 3)
        assert_unbatched_metadata(view)
        unsqueeze = td.unsqueeze(0)
        assert unsqueeze.batch_size == (1, 3)
        assert_unbatched_metadata(unsqueeze)
        gather = td.gather(0, torch.tensor((0,)))
        assert gather.batch_size == (1,)
        assert_unbatched_metadata(gather)

        unflatten = td.unflatten(0, (1, 3))
        assert unflatten.batch_size == (1, 3)
        assert_unbatched_metadata(unflatten)

        flatten = unflatten.flatten(0, 1)
        assert flatten.batch_size == (3,)
        assert_unbatched_metadata(flatten)

    def test_unbatched_torch_func(self):
        td = TensorDict(
            a=UnbatchedTensor(torch.randn(10)),
            b=torch.randn(3),
            batch_size=(3,),
        )
        unbound = torch.unbind(td, 0)[0]
        assert unbound["a"].data_ptr() == td["a"].data_ptr()
        assert unbound["a"].batch_size == unbound.batch_size
        _check_recursive_properties(unbound)
        stacked = torch.stack([td, td], 0)
        assert stacked[0]["a"].data_ptr() == td["a"].data_ptr()
        assert stacked["a"].batch_size == torch.Size([2, 3])
        _check_recursive_properties(stacked)
        catted = torch.cat([td, td], 0)
        assert catted[0]["a"].data_ptr() == td["a"].data_ptr()
        assert catted["a"].batch_size == torch.Size([6])
        _check_recursive_properties(catted)
        assert (torch.ones_like(td)["a"] == 1).all()
        unsqueezed = torch.unsqueeze(td, 0)
        assert unsqueezed["a"].data_ptr() == td["a"].data_ptr()
        assert unsqueezed["a"].batch_size == unsqueezed.batch_size
        _check_recursive_properties(unsqueezed)
        squeezed = torch.squeeze(td)
        assert squeezed["a"].data_ptr() == td["a"].data_ptr()
        assert squeezed["a"].batch_size == squeezed.batch_size
        _check_recursive_properties(squeezed)
        unflatten = torch.unflatten(td, 0, (1, 3))
        assert unflatten["a"].data_ptr() == td["a"].data_ptr()
        assert unflatten["a"].batch_size == unflatten.batch_size
        _check_recursive_properties(unflatten)
        flatten = torch.flatten(unflatten, 0, 1)
        assert flatten["a"].data_ptr() == td["a"].data_ptr()
        assert flatten["a"].batch_size == flatten.batch_size
        _check_recursive_properties(flatten)
        permute = torch.permute(unflatten, (1, 0))
        assert permute["a"].data_ptr() == td["a"].data_ptr()
        assert permute["a"].batch_size == permute.batch_size
        _check_recursive_properties(permute)
        transpose = torch.transpose(unflatten, 1, 0)
        assert transpose["a"].data_ptr() == td["a"].data_ptr()
        assert transpose["a"].batch_size == transpose.batch_size
        _check_recursive_properties(transpose)

    def test_unbatched_other_ops(self):
        td = TensorDict(
            a=UnbatchedTensor(torch.randn(10)),
            b=torch.randn(3),
            c_d=UnbatchedTensor(torch.randn(10)),
            batch_size=(3,),
        )
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif is_npu_available():
            device = torch.device("npu:0")
        else:
            device = torch.device("cpu")
        assert td.copy()["a"] is td["a"]
        assert td.int()["a"].dtype == torch.int
        assert td.to(device)["a"].device == device
        assert td.select("a")["a"] is td["a"]
        assert td.exclude("b")["a"] is td["a"]
        assert td.unflatten_keys(separator="_")["c", "d"] is td["c_d"]
        assert td.unflatten_keys(separator="_").flatten_keys()["c.d"] is td["c_d"]

    @pytest.mark.parametrize(
        "op",
        [
            lambda td: td.exp(),
            lambda td: td.abs(),
            lambda td: td.sqrt(),
            lambda td: td.sigmoid(),
            lambda td: td.reciprocal(),
            lambda td: td.floor(),
            lambda td: td - td,
            lambda td: 1 - td,
            lambda td: td / 2,
            lambda td: td**2,
            lambda td: td.lerp(td, 0.5),
            lambda td: td.addcmul(td, td),
            lambda td: td.addcdiv(td, td),
        ],
    )
    def test_unbatched_pointwise_keeps_batched_leaves(self, op):
        # Regression test for #2352: the _foreach ops wrapped every leaf in an
        # UnbatchedTensor and gave each unbatched leaf the same batch size.
        td = TensorDict(
            a=torch.ones(3, 2),
            u=UnbatchedTensor(torch.ones(5)),
            n=TensorDict(v=UnbatchedTensor(torch.ones(7)), batch_size=[3, 2]),
            batch_size=[3],
        )
        out = op(td)
        assert type(out.get("a")) is torch.Tensor
        assert out[0]["a"].shape == (2,)
        assert isinstance(out.get("u"), UnbatchedTensor)
        assert out.get("u").batch_size == (3,)
        assert isinstance(out.get(("n", "v")), UnbatchedTensor)
        assert out.get(("n", "v")).batch_size == (3, 2)

    def test_unbatched_pointwise_norm(self):
        td = TensorDict(
            a=torch.ones(3, 2), u=UnbatchedTensor(torch.ones(5)), batch_size=[3]
        )
        out = td.norm()
        assert type(out.get("a")) is torch.Tensor
        assert out["a"].shape == ()

    def test_unbatched_to_batch_size(self):
        # https://github.com/pytorch/tensordict/issues/2499
        td = TensorDict(
            a=torch.ones(3, 2),
            u=UnbatchedTensor(torch.ones(5)),
            n=TensorDict(v=UnbatchedTensor(torch.ones(7)), batch_size=[3, 2]),
            batch_size=[3, 2],
        )
        with pytest.warns(FutureWarning, match="batch_size"):
            out = td.to(torch.float64, batch_size=[3])
        for key in ("u", ("n", "v")):
            assert isinstance(out.get(key), UnbatchedTensor)
            assert out.get(key).dtype == torch.float64
            assert out.get(key).batch_size == (3,)
        out = td.to(torch.float64, batch_size=[3, 2])
        assert out.get("u").dtype == torch.float64

        # Shape ops still leave the payload unchanged
        out = td.reshape(6)
        assert out["u"].shape == (5,)
        assert out.get("u").batch_size == (6,)

    def test_unbatched_apply_batch_size(self):
        # https://github.com/pytorch/tensordict/issues/2499
        # apply with a batch_size is a shape op: fn is not called on the
        # UnbatchedTensor, which only takes the new batch size.
        td = TensorDict(
            a=torch.ones(3, 2), u=UnbatchedTensor(torch.ones(5)), batch_size=[3, 2]
        )
        out = td.apply(lambda x: x * 2, batch_size=[3])
        assert (out["a"] == 2).all()
        assert isinstance(out.get("u"), UnbatchedTensor)
        assert (out["u"] == 1).all()
        assert out.get("u").batch_size == (3,)
        out = td.named_apply(lambda name, x: x * 2, batch_size=[3])
        assert (out["u"] == 1).all()
        out = TensorDict(
            a=torch.zeros(3, 2), u=UnbatchedTensor(torch.zeros(5)), batch_size=[3]
        )
        assert td.apply(lambda x: x * 2, batch_size=[3], out=out) is out
        assert (out["a"] == 2).all()
        assert (out["u"] == 1).all()
        # A shape function works because it is not called on the entry
        out = td.apply(lambda x: x.reshape(6, *x.shape[2:]), batch_size=[6])
        assert out["u"].shape == (5,)
        assert out.get("u").batch_size == (6,)

        # Without a batch_size, fn is called on the entry
        out = td.apply(lambda x: x * 2)
        assert isinstance(out.get("u"), UnbatchedTensor)
        assert (out["u"] == 2).all()
        assert out.get("u").batch_size == (3, 2)

    @pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul])
    def test_unbatched_pointwise_other_unbatched(self, op):
        td = TensorDict(a=torch.ones(3, 2), u=torch.ones(3), batch_size=[3])
        other = TensorDict(
            a=torch.ones(3, 2), u=UnbatchedTensor(torch.ones(3)), batch_size=[3]
        )
        out = op(td, other)
        assert type(out.get("a")) is torch.Tensor
        assert out[0]["a"].shape == (2,)

    def test_unbatched_view_base(self):
        view = UnbatchedTensor(torch.arange(6.0)).clone().reshape(2, 3)
        base = view._base
        assert base is not None
        assert view._base is base
        assert base._base is None
        base.zero_()
        assert not view.any()

    def test_unbatched_getitem_returns_unbatched(self):
        data = torch.randn(7, 11)
        td = TensorDict(
            a=UnbatchedTensor(data),
            b=torch.randn(3, 4),
            batch_size=(3, 4),
        )
        result = td["a"]
        assert isinstance(result, torch.Tensor)
        assert isinstance(result, UnbatchedTensor)
        assert result.data_ptr() == data.data_ptr()

    @pytest.mark.parametrize("value_type", ["tensordict", "dict"])
    @pytest.mark.parametrize("index", [None, True, torch.tensor(True)])
    def test_unbatched_setitem_new_dim(self, index, value_type):
        # None and True add a dim of size 1: the value is written to all of td,
        # as with td[:] = value
        data = torch.arange(5.0)
        td = TensorDict(a=torch.zeros(4, 3), u=UnbatchedTensor(data), batch_size=[4, 3])
        value = {"a": torch.ones(1, 4, 3), "u": UnbatchedTensor(torch.full((5,), 7.0))}
        if value_type == "tensordict":
            value = TensorDict(value, batch_size=[1, 4, 3])
        td[index] = value
        assert (td["a"] == 1).all()
        u = td.get("u")
        assert isinstance(u, UnbatchedTensor)
        assert u.data_ptr() == data.data_ptr()
        assert (u == 7).all()

    @pytest.mark.parametrize(
        "index,same_as",
        [
            ((0, Ellipsis), 0),
            (Ellipsis, slice(None)),
            ((Ellipsis,), slice(None)),
            ((None, Ellipsis), None),
        ],
    )
    @pytest.mark.parametrize("value", ["scalar", "tensordict"])
    def test_unbatched_setitem_ellipsis(self, index, same_as, value):
        # the Ellipsis covers the batch dims and the trailing ':' are dropped,
        # so these indices write as the index without the Ellipsis does, also
        # into an UnbatchedTensor of fewer dims than the batch dims
        def make():
            return TensorDict(
                a=torch.zeros(4, 3),
                u=UnbatchedTensor(torch.zeros(5)),
                batch_size=[4, 3],
            )

        td, expected = make(), make()
        if value == "scalar":
            td[index] = 2.0
            expected[same_as] = 2.0
        else:
            source = make()[same_as] + 2
            td[index] = source
            expected[same_as] = source
        assert (td == expected).all()
        assert (td.get("u") == expected.get("u")).all()

    @pytest.mark.parametrize(
        "index",
        [
            0,
            slice(0, 2),
            (slice(None), slice(0, 2)),
            ...,
            torch.tensor([0, 2]),
            torch.tensor([True, False, True, False]),
        ],
    )
    def test_unbatched_setitem_partial_index(self, index):
        # #2251: the whole batch shares the payload, so td[index] = value
        # writes it whole, as td[:] = value does
        data = torch.zeros(5)
        td = TensorDict(
            a=torch.zeros(4, 3),
            u=UnbatchedTensor(data),
            nested=TensorDict(u=UnbatchedTensor(torch.zeros(2)), batch_size=[4, 3]),
            batch_size=[4, 3],
        )
        value = td[index].clone()
        value["a"] += 1
        value["u"] += 7
        value["nested", "u"] += 3
        td[index] = value
        assert (td["a"][index] == 1).all()
        u = td.get("u")
        assert isinstance(u, UnbatchedTensor)
        assert u.data_ptr() == data.data_ptr()
        assert (u == 7).all()
        assert (td["nested", "u"] == 3).all()

    def test_unbatched_setitem_other_shape(self):
        # A payload of another shape is indexed by the batch index, as before
        td = TensorDict(
            a=torch.zeros(3, 4), u=UnbatchedTensor(torch.zeros(3, 5)), batch_size=[3, 4]
        )
        td[0:2] = TensorDict(
            a=torch.ones(2, 4), u=UnbatchedTensor(torch.ones(2, 5)), batch_size=[2, 4]
        )
        assert (td.get("u")[:2] == 1).all()
        assert (td.get("u")[2] == 0).all()
        td[2] = TensorDict(
            a=torch.ones(4), u=UnbatchedTensor(torch.full((5,), 2.0)), batch_size=[4]
        )
        assert (td.get("u")[:2] == 1).all()
        assert (td.get("u")[2] == 2).all()

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize(
        "method", ["memmap", "memmap_", "memmap_threads", "save", "consolidate"]
    )
    def test_unbatched_memmap_consolidate_raise(self, method, nested, tmpdir):
        # Not supported yet: these used to write the entry as a plain tensor
        # that no longer matches the batch size.
        td = TensorDict(
            a=torch.randn(2, 3),
            config=UnbatchedTensor(torch.arange(3.0)),
            batch_size=(2, 3),
        )
        if nested:
            td = TensorDict(nested=td, b=torch.randn(2), batch_size=(2,))
        match = "memory-mapped" if method != "consolidate" else "consolidated"
        with pytest.raises(NotImplementedError, match=f"cannot be {match} yet"):
            if method == "memmap":
                td.memmap(tmpdir)
            elif method == "memmap_":
                td.memmap_(tmpdir)
            elif method == "memmap_threads":
                td.memmap(tmpdir, num_threads=2)
            elif method == "save":
                td.save(tmpdir)
            else:
                td.consolidate()
        if method == "memmap_" and not nested:
            assert not td.is_memmap()
            assert isinstance(td.get("config"), UnbatchedTensor)

    def test_unbatched_stack_same_data_no_warning(self):
        data = torch.randn(5)
        td1 = TensorDict(a=torch.ones(3), ub=UnbatchedTensor(data), batch_size=[3])
        td2 = TensorDict(a=torch.ones(3), ub=UnbatchedTensor(data), batch_size=[3])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            stacked = torch.stack([td1, td2])
        assert stacked["ub"].data_ptr() == data.data_ptr()
        _check_recursive_properties(stacked)

    def test_unbatched_stack_different_data_warns(self):
        td1 = TensorDict(
            a=torch.ones(3), ub=UnbatchedTensor(torch.zeros(5)), batch_size=[3]
        )
        td2 = TensorDict(
            a=torch.ones(3), ub=UnbatchedTensor(torch.ones(5)), batch_size=[3]
        )
        with pytest.warns(UserWarning, match="different data storage"):
            torch.stack([td1, td2])

    @pytest.mark.parametrize(
        "batch_size, dim, expected_batch_size",
        [
            ([], 0, [3]),
            ([2], 0, [3, 2]),
            ([2], 1, [2, 3]),
            ([2, 1], -1, [2, 1, 3]),
        ],
    )
    def test_unbatched_stack_updates_batch_size_metadata(
        self, batch_size, dim, expected_batch_size
    ):
        data = torch.tensor(1)
        items = [
            TensorDict(
                x=torch.zeros(batch_size),
                ub=UnbatchedTensor(data),
                batch_size=batch_size,
            )
            for _ in range(3)
        ]
        result = torch.stack(items, dim)
        assert result.batch_size == torch.Size(expected_batch_size)
        assert result["ub"].batch_size == result.batch_size
        assert result["ub"].shape == torch.Size([])
        assert result["ub"].data_ptr() == data.data_ptr()
        assert all(item["ub"].batch_size == torch.Size(batch_size) for item in items)
        _check_recursive_properties(result)

    def test_unbatched_stack_out_updates_batch_size_metadata(self):
        data = torch.tensor(1)
        items = [
            TensorDict(
                x=torch.zeros(2),
                ub=UnbatchedTensor(data),
                batch_size=[2],
            )
            for _ in range(3)
        ]
        out = TensorDict(
            x=torch.empty(3, 2),
            ub=UnbatchedTensor(data),
            batch_size=[3, 2],
        )
        result = torch.stack(items, 0, out=out)
        assert result is out
        assert result["ub"].batch_size == torch.Size([3, 2])
        assert result["ub"].shape == torch.Size([])
        assert result["ub"].data_ptr() == data.data_ptr()
        _check_recursive_properties(result)

    @pytest.mark.parametrize("with_out", [False, True])
    def test_unbatched_cat_updates_batch_size_metadata(self, with_out):
        data = torch.tensor(1)
        items = [
            TensorDict(
                x=torch.zeros(2),
                ub=UnbatchedTensor(data),
                batch_size=[2],
            )
            for _ in range(3)
        ]
        out = None
        if with_out:
            out = TensorDict(
                x=torch.empty(6),
                ub=UnbatchedTensor(data),
                batch_size=[6],
            )
        result = torch.cat(items, 0, out=out)
        if with_out:
            assert result is out
        assert result.batch_size == torch.Size([6])
        assert result["ub"].batch_size == result.batch_size
        assert result["ub"].shape == torch.Size([])
        assert result["ub"].data_ptr() == data.data_ptr()
        _check_recursive_properties(result)

    def test_lazy_stack_unbatched_updates_batch_size_metadata(self):
        data = torch.tensor(1)
        items = [
            TensorDict(
                x=torch.zeros(2),
                ub=UnbatchedTensor(data),
                batch_size=[2],
            )
            for _ in range(3)
        ]
        result = lazy_stack(items, 0)
        assert result.batch_size == torch.Size([3, 2])
        assert result["ub"].batch_size == result.batch_size
        assert result["ub"].shape == torch.Size([])
        assert result["ub"].data_ptr() == data.data_ptr()
        _check_recursive_properties(result)

    @pytest.mark.parametrize(
        "op",
        [
            "index",
            "slice",
            "gather",
            "split",
            "chunk",
            "masked_select",
        ],
    )
    def test_unbatched_indexing_updates_batch_size_metadata(self, op):
        data = torch.tensor(1)
        td = TensorDict(
            x=torch.zeros(2, 3),
            ub=UnbatchedTensor(data),
            batch_size=[2, 3],
        )

        if op == "index":
            results = (td[0],)
        elif op == "slice":
            results = (td[:, :2],)
        elif op == "gather":
            results = (td.gather(0, torch.zeros(1, 3, dtype=torch.long)),)
        elif op == "split":
            results = td.split(1, dim=0)
        elif op == "chunk":
            results = td.chunk(2, dim=0)
        elif op == "masked_select":
            mask = torch.tensor([[True, False, True], [False, True, False]])
            results = (td.masked_select(mask),)
        else:
            raise AssertionError(op)

        for result in results:
            assert result["ub"].batch_size == result.batch_size
            assert result["ub"].shape == torch.Size([])
            assert result["ub"].data_ptr() == data.data_ptr()
            _check_recursive_properties(result)

    @pytest.mark.skipif(PYTORCH_TEST_FBCODE, reason="vmap now working in fbcode")
    def test_unbatched_vmap(self):
        from torch import vmap

        data = torch.randn(7, 11)
        td = TensorDict(
            a=torch.randn(4, 3, 5),
            unbatched=UnbatchedTensor(data),
            batch_size=[4, 3],
        )
        result = vmap(lambda x: x)(td)
        assert result.batch_size == torch.Size([4, 3])
        assert isinstance(result.get("unbatched"), UnbatchedTensor)
        assert result.get("unbatched").data_ptr() == data.data_ptr()
        assert result.get("unbatched").batch_size == torch.Size([4, 3])
        assert result["a"].shape == (4, 3, 5)

    @pytest.mark.skipif(PYTORCH_TEST_FBCODE, reason="vmap now working in fbcode")
    def test_unbatched_vmap_return_unbatched_tensor(self):
        from torch import vmap

        data = torch.randn(7, 11)
        td = TensorDict(
            a=torch.randn(4, 3, 5),
            unbatched=UnbatchedTensor(data),
            batch_size=[4, 3],
        )
        result = vmap(lambda x: x["unbatched"])(td)
        assert isinstance(result, UnbatchedTensor)
        assert result.data_ptr() == data.data_ptr()
        assert result.batch_size == torch.Size([4, 3])

    @pytest.mark.skipif(PYTORCH_TEST_FBCODE, reason="vmap now working in fbcode")
    def test_unbatched_vmap_direct_input(self):
        from torch import vmap

        data = torch.randn(7, 11)
        unbatched = UnbatchedTensor(data, batch_size=[4])
        result = vmap(lambda x: x)(unbatched)
        assert isinstance(result, UnbatchedTensor)
        assert result.data_ptr() == data.data_ptr()
        assert result.batch_size == torch.Size([4])

    def test_unbatched_tensorclass_attr_returns_unbatched(self):
        @tensorclass
        class MyClass:
            config: torch.Tensor
            value: torch.Tensor

        data = torch.tensor([1.0, 2.0])
        tc = MyClass(
            config=UnbatchedTensor(data),
            value=torch.randn(3),
            batch_size=[3],
        )
        assert isinstance(tc.config, torch.Tensor)
        assert isinstance(tc.config, UnbatchedTensor)
        assert tc.config.data_ptr() == data.data_ptr()

        result = tc.get("config")
        assert isinstance(result, torch.Tensor)
        assert isinstance(result, UnbatchedTensor)
        assert result.data_ptr() == data.data_ptr()

    def test_unbatched_tensor_is_tensor(self):
        data = torch.randn(5, 3)
        ut = UnbatchedTensor(data)
        assert isinstance(ut, torch.Tensor)
        assert isinstance(ut, UnbatchedTensor)
        assert ut.shape == data.shape
        assert ut.data_ptr() == data.data_ptr()

    def test_unbatched_tolist(self):
        data = torch.tensor([1, 2, 3])
        ut = UnbatchedTensor(data)
        assert ut.tolist() == [1, 2, 3]

    @pytest.mark.parametrize(
        "data",
        [
            torch.tensor(3),
            torch.tensor([3]),
            torch.tensor([[3]]),
            torch.tensor(True),
            torch.tensor(1.25),
            torch.tensor(1 + 2j),
            torch.tensor([1, 2]),
            torch.empty(0),
        ],
    )
    def test_unbatched_python_conversions_match_tensor(self, data):
        conversion_funs = [
            str,
            format,
            lambda x: format(x, ".2f"),
            bool,
            int,
            float,
            complex,
            operator.index,
            round,
            lambda x: round(x, 1),
            math.trunc,
            math.floor,
            math.ceil,
            lambda x: x.item(),
        ]
        for fun in conversion_funs:
            self._assert_same_conversion(fun, data)

    def test_unbatched_index_protocol(self):
        data = torch.tensor(1)
        ut = UnbatchedTensor(data)
        assert [10, 20, 30][ut] == [10, 20, 30][data]
        assert hex(ut) == hex(data)

    def test_unbatched_numpy_array_conversion(self):
        data = torch.tensor([[1, 2], [3, 4]])
        ut = UnbatchedTensor(data)
        np.testing.assert_array_equal(np.asarray(ut), np.asarray(data))
        np.testing.assert_array_equal(
            np.asarray(ut, dtype=np.float64), np.asarray(data, dtype=np.float64)
        )

    def test_unbatched_string_representation(self):
        data = torch.tensor([1.0, 2.0])
        ut = UnbatchedTensor(data)
        assert str(ut) == str(data)
        assert repr(ut) == f"UnbatchedTensor({data!r})"

    def test_auto_batch_size_nontensor_not_excluded(self):
        td = TensorDict.from_dict(
            {"query": ["str1", "str2", "str3", "str4"]},
            auto_batch_size=True,
            batch_dims=1,
        )
        assert td.batch_size == torch.Size([4])
        assert "query" in td.keys()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
