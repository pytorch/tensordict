# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import contextlib
import dataclasses
import importlib.util
import inspect
import platform
import sys
import warnings
import weakref
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch
from _utils_internal import is_npu_available, legacy_lazy_mode
from packaging import version
from tensordict import (
    assert_close,
    from_dataclass,
    is_inexact,
    is_leaf_nontensor,
    is_tensor_collection,
    lazy_stack,
    LazyStackedTensorDict,
    MemoryMappedTensor,
    MetaData,
    NonTensorData,
    NonTensorStack,
    tensorclass,
    TensorDict,
    TensorDictBase,
    TensorDictParams,
    TypedTensorDict,
)
from tensordict._unbatched import _HAS_WRAPPER_SUBCLASS_FIX, UnbatchedTensor
from tensordict.base import _get_defaults_to_none, _set_get_defaults_to_none
from tensordict.nn import (
    CompositeDistribution,
    CudaGraphModule,
    functional_vmap,
    InteractionType,
    ProbabilisticTensorDictModule as Prob,
    ProbabilisticTensorDictSequential as ProbSeq,
    set_composite_lp_aggregate,
    set_interaction_type,
    TensorDictModule,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from tensordict.nn.functional_modules import (
    PYTREE_REGISTERED_LAZY_TDS,
    PYTREE_REGISTERED_TDS,
)
from tensordict.store._utils import _prepare_indexed_value
from tensordict.tensorclass import TensorClass
from tensordict.utils import (
    _unravel_key_to_tuple,
    _unravel_keys,
    unravel_key,
    unravel_key_list,
)
from torch._dynamo.testing import CompileCounterWithBackend, EagerAndRecordGraphs
from torch._dynamo.utils import counters
from torch._inductor.utils import fresh_cache
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._pytree import (
    SUPPORTED_NODES,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_unflatten,
)

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)

_has_onnx = importlib.util.find_spec("onnxruntime", None) is not None


_IS_OSX = platform.system() == "Darwin"

npu_device_count = 0
if torch.cuda.is_available():
    cur_device = "cuda"
elif is_npu_available():
    cur_device = "npu"
    npu_device_count = torch.npu.device_count()


@pytest.mark.parametrize("is_tensordict_module", [False, True])
def test_cudagraph_module_is_released_without_gc(is_tensordict_module):
    if is_tensordict_module:
        module = TensorDictModule(lambda x: x, in_keys=["x"], out_keys=["y"])
    else:

        def module(x):
            return x

    with (
        pytest.warns(UserWarning)
        if not torch.cuda.is_available()
        else contextlib.nullcontext()
    ):
        wrapper = CudaGraphModule(module)

    wrapper_ref = weakref.ref(wrapper)
    del wrapper
    assert wrapper_ref() is None


@pytest.fixture(autouse=True)
def _reset_dynamo_code_caches():
    # Tests in this file all exercise torch.compile; a fresh Dynamo code cache
    # between tests keeps recompile/guard-count assertions reliable and prevents
    # one test's frame from leaking into the next.
    torch._dynamo.reset_code_caches()


@pytest.mark.parametrize("value_shape", [(), (3,)])
def test_store_indexed_value_compile(value_shape):
    idx = torch.tensor([[0, 2], [1, 3]])
    value = torch.full(value_shape, 7.0, requires_grad=True)
    compiled = torch.compile(_prepare_indexed_value, fullgraph=True)
    actual = compiled(value, [5, 3], torch.float32, idx)
    torch.testing.assert_close(actual, torch.full((2, 2, 3), 7.0))
    actual.sum().backward()
    torch.testing.assert_close(
        value.grad, torch.full_like(value, actual.numel() / value.numel())
    )


def test_vmap_compile():
    # Since we monkey patch vmap we need to make sure compile is happy with it
    def func(x, y):
        return x + y

    x = torch.randn(3, 4)
    y = torch.randn(3)
    funcv = torch.vmap(func, (1, None))
    funcv(x, y)
    funcv_c = torch.compile(funcv, fullgraph=True)
    funcv_c(x, y)


# TensorDict methods that call torch._foreach_*, which has no vmap batching rule
_VMAP_TD_OPS = {
    "mul": lambda t: t * 2,
    "add": lambda t: t + t,
    "exp": lambda t: t.exp(),
    "clamp_min": lambda t: t.clamp_min(0.0),
    "norm": lambda t: t.norm(),
    "add_": lambda t: t.clone().add_(1),
    "update_": lambda t: t.clone().update_(t * 2),
    "grad": torch.func.grad(lambda t: (t * 2).exp().sum(reduce=True)),
}


@pytest.mark.parametrize("op", sorted(_VMAP_TD_OPS))
def test_vmap_compile_td_ops(op):
    fn = _VMAP_TD_OPS[op]
    td = TensorDict(a=torch.randn(4, 3), b={"c": torch.randn(4, 2)}, batch_size=[4])
    expected = torch.stack([fn(td[i]) for i in range(4)])
    fn_c = torch.compile(torch.vmap(fn), fullgraph=True)
    assert_close(fn_c(td), expected)


def test_compile_td_mul_keeps_foreach():
    # Outside a torch.func transform the graph keeps the fused _foreach op
    td = TensorDict(a=torch.randn(4, 3), b={"c": torch.randn(4, 2)}, batch_size=[4])
    backend = EagerAndRecordGraphs()
    fn_c = torch.compile(lambda t: t * 2, fullgraph=True, backend=backend)
    assert_close(fn_c(td), td * 2)
    targets = [node.target for node in backend.graphs[0].graph.nodes]
    assert torch._foreach_mul in targets


@pytest.mark.parametrize(
    "key",
    ["a", ("b",), ("a", "b", "action"), ("c", ("d",))],
    ids=["str", "single_tuple", "nested_tuple", "nested_wrapped"],
)
def test_unravel_keys_compile(key):
    """Test that unravel_keys returns consistent results under torch.compile."""
    eager = _unravel_keys(key)
    torch._dynamo.reset()
    compiled = torch.compile(_unravel_keys, backend="eager")(key)
    assert eager == compiled, (
        f"unravel_keys mismatch for {key!r}: eager={eager!r}, compiled={compiled!r}"
    )


_UNRAVEL_VALID_KEYS = [
    "a",
    ("a",),
    ("a", "b"),
    (("a", "b"), "c"),
    ("a", ("b", ("c",))),
    ((("a",),),),
]
# These unravel to () in both modes.
_UNRAVEL_INVALID_TUPLE_KEYS = [
    ("a", 1),
    (("a", 1), "b"),
    ("a", ()),
    (),
    ((),),
    (slice(None), 0),
    (0, Ellipsis),
]


@pytest.mark.parametrize(
    "fn", [_unravel_key_to_tuple, unravel_key], ids=lambda fn: fn.__name__
)
@pytest.mark.parametrize(
    "key", _UNRAVEL_VALID_KEYS + _UNRAVEL_INVALID_TUPLE_KEYS, ids=repr
)
def test_unravel_key_fullgraph(fn, key):
    eager = fn(key)
    torch._dynamo.reset()

    def f(x):
        return x + 1, fn(key)

    compiled = torch.compile(f, fullgraph=True, backend="eager")(torch.zeros(()))[1]
    assert compiled == eager


def test_unravel_key_list_fullgraph():
    eager = unravel_key_list(_UNRAVEL_VALID_KEYS)
    eager_keys = _unravel_keys(*_UNRAVEL_VALID_KEYS)
    torch._dynamo.reset()

    def f(x):
        return (
            x + 1,
            unravel_key_list(_UNRAVEL_VALID_KEYS),
            _unravel_keys(*_UNRAVEL_VALID_KEYS),
        )

    _, compiled, compiled_keys = torch.compile(f, fullgraph=True, backend="eager")(
        torch.zeros(())
    )
    assert compiled == eager
    assert compiled_keys == eager_keys


@pytest.mark.parametrize(
    "fn,key",
    [
        (unravel_key, 1),
        (unravel_key, None),
        (unravel_key_list, ["a", 1]),
        (unravel_key_list, ["a", ("a", 1)]),
        (unravel_key_list, ["a", ()]),
    ],
    ids=["unravel_key-int", "unravel_key-None", "list-int", "list-mixed", "list-empty"],
)
def test_unravel_key_invalid_fullgraph(fn, key):
    msg = "key should be a Sequence<NestedKey>"
    with pytest.raises(RuntimeError, match=msg):
        fn(key)
    torch._dynamo.reset()

    def f(x):
        try:
            fn(key)
        except RuntimeError as err:
            return x + 1, str(err)
        return x, None

    _, compiled_msg = torch.compile(f, fullgraph=True, backend="eager")(torch.zeros(()))
    assert compiled_msg == msg


@pytest.mark.parametrize("values", ["aaaa", "abcd"])
def test_tree_leaves_nontensor_stack(values):
    # Flattening a tensordict does not read the data of its NonTensorStack
    # entries, which Dynamo cannot trace.
    def fn(td):
        return [leaf + 1 for leaf in tree_leaves(td) if isinstance(leaf, torch.Tensor)]

    td = TensorDict(a=torch.zeros(4, 3), batch_size=[4])
    td.set("s", NonTensorStack(*values))
    (out,) = torch.compile(fn, fullgraph=True, backend="eager")(td)
    assert (out == 1).all()


def test_where_non_tensor_data():
    # With equal values, NonTensorData.where does not read the condition, and
    # compiles without a graph break. With differing values, it reads the
    # condition and breaks the graph (#2071).
    def fn(td, other, cond):
        return td.where(cond, other)

    cond = torch.tensor([True, False, True, False])
    td = TensorDict(
        x=torch.zeros(4), s=NonTensorData("a", batch_size=[4]), batch_size=[4]
    )
    same = TensorDict(
        x=torch.ones(4), s=NonTensorData("a", batch_size=[4]), batch_size=[4]
    )
    other = TensorDict(
        x=torch.ones(4), s=NonTensorData("b", batch_size=[4]), batch_size=[4]
    )
    result = torch.compile(fn, fullgraph=True, backend="eager")(td, same, cond)
    assert result["x"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert result.get("s").tolist() == ["a"] * 4
    torch._dynamo.reset()
    result = torch.compile(fn, backend="eager")(td, other, cond)
    assert result["x"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert result.get("s").tolist() == ["a", "b", "a", "b"]


@pytest.mark.parametrize("lazy", [False, True], ids=["dense", "lazy"])
@pytest.mark.parametrize("value", [3, [1, 2]], ids=type)
@pytest.mark.parametrize("setitem", [False, True], ids=["set_at_", "setitem"])
def test_set_at_cast_non_tensor(setitem, value, lazy):
    # The FutureWarning for a value cast to a tensor in a non-tensor entry
    # (gh-2522) is skipped under compile, which traces no warnings.warn
    def fn(td):
        if setitem:
            td[:2] = value
        else:
            td.set_at_("a", value, slice(0, 2))
        return td["b"] + 1

    td = TensorDict(
        a=NonTensorData("s", batch_size=[3]), b=torch.zeros(3), batch_size=[3]
    )
    if lazy:
        td = lazy_stack(td.unbind(0))
    torch.compile(fn, fullgraph=True, backend="eager")(td)
    assert torch.equal(td[1]["a"], torch.as_tensor(value))


@pytest.mark.parametrize("lazy", [False, True], ids=["dense", "lazy"])
@pytest.mark.parametrize("non_tensor", [False, True], ids=["tensors", "non-tensor"])
@pytest.mark.parametrize("value", ["number", "tensordict"])
def test_setitem_compiles(value, non_tensor, lazy):
    # td[index] = value decides from Python types whether to warn of a cast
    # in a non-tensor entry (gh-2522), which adds no graph break
    def make(batch_size):
        td = TensorDict(b=torch.zeros(batch_size), n={"e": torch.zeros(batch_size, 2)})
        td.batch_size = [batch_size]
        if non_tensor:
            td["a"] = NonTensorData("s", batch_size=[batch_size])
        return td

    def fn(td, other):
        td[:2] = 0.5 if value == "number" else other
        return td["b"] + 1

    td = make(3)
    if lazy:
        td = lazy_stack(td.unbind(0))
    other = make(2)
    other["b"] = torch.full((2,), 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = torch.compile(fn, fullgraph=True, backend="eager")(td, other)
    expected = 1.5 if value == "number" else 3.0
    assert out.tolist() == [expected, expected, 1.0]


@pytest.mark.parametrize("stack_dim", [0, 1])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_lazy_softmax_compiles(stack_dim, dim):
    # softmax on a lazy stack, along its stack dim too, gives the values of
    # the dense copy without a graph break (#2359)
    dense = TensorDict(
        a=torch.randn(2, 3, 4, 5), n={"b": torch.randn(2, 3, 4)}, batch_size=[2, 3, 4]
    )
    td = lazy_stack(dense.unbind(stack_dim), stack_dim)
    out = torch.compile(lambda td: td.softmax(dim), fullgraph=True, backend="eager")(td)
    for key, value in dense.softmax(dim).items(True, True):
        torch.testing.assert_close(out.get(key), value)


def test_lazy_softmax_ragged_compiles():
    # members whose entries have different shapes keep their result along
    # the stack dim, without a graph break (#2359)
    td = lazy_stack([TensorDict(a=torch.randn(3, 4)), TensorDict(a=torch.randn(3, 5))])
    out = torch.compile(lambda td: td.softmax(0), fullgraph=True, backend="eager")(td)
    for member, expected in zip(out.tensordicts, td.tensordicts):
        torch.testing.assert_close(member["a"], expected["a"].softmax(0))


@pytest.mark.parametrize(
    "name", ["td == 0", "td != td", "td ^ True", "td.all(0)", "td.where(mask, 0)"]
)
def test_comparison_where_fullgraph(name):
    fn = {
        "td == 0": lambda td, mask: td == 0,
        "td != td": lambda td, mask: td != td.clone(),
        "td ^ True": lambda td, mask: td.bool() ^ True,
        "td.all(0)": lambda td, mask: td.all(0),
        "td.where(mask, 0)": lambda td, mask: td.where(mask, 0),
    }[name]
    td = TensorDict(
        a=torch.arange(8.0).view(4, 2),
        nested=TensorDict(b=torch.arange(4), batch_size=[4]),
        batch_size=[4],
    )
    mask = torch.tensor([True, False, True, False])
    result = torch.compile(fn, fullgraph=True, backend="eager")(td, mask)
    assert (result == fn(td, mask)).all()


@pytest.mark.xfail(
    reason="td < tensor breaks the graph: all __torch_function__ overrides "
    "return NotImplemented",
    strict=True,
)
def test_comparison_tensor_fullgraph():
    def fn(td, other):
        return td < other

    td = TensorDict(a=torch.arange(8.0).view(4, 2), batch_size=[4])
    other = torch.ones(4)
    result = torch.compile(fn, fullgraph=True, backend="eager")(td, other)
    assert (result == fn(td, other)).all()


_NESTED_DEVICE_CASES = {
    # #2647: a nested tensordict without a device, in a tensordict on CPU
    "nested_td": lambda x: TensorDict(
        {"s": TensorDict({"a": x, "b": x + 1}, [3]), "c": x * 2}, [3], device="cpu"
    ),
    "non_blocking_false": lambda x: TensorDict(
        {"s": TensorDict({"a": x}, [3]), "c": x},
        [3],
        device="cpu",
        non_blocking=False,
    ),
    "nested_dict": lambda x: TensorDict(
        {"s": {"a": x, "b": {"c": x}}}, [3], device="cpu"
    ),
    "three_levels": lambda x: TensorDict(
        {"s": TensorDict({"t": TensorDict({"a": x}, [3])}, [3])}, [3], device="cpu"
    ),
    "names": lambda x: TensorDict(
        {"s": TensorDict({"a": x}, [3])}, [3], device="cpu", names=["n"]
    ),
    "expanded": lambda x: TensorDict(
        {"g": TensorDict({"u": x[0]}).expand(3), "c": x}, [3], device="cpu"
    ),
    "to_cpu": lambda x: TensorDict({"s": TensorDict({"a": x}, [3])}, [3]).to("cpu"),
}


@pytest.mark.parametrize("name", list(_NESTED_DEVICE_CASES))
def test_nested_device_construction_fullgraph(name):
    # Moving a nested tensordict to the device of the outer one compiles into
    # one graph, without the device-recorder sync (#2647).
    fn = _NESTED_DEVICE_CASES[name]

    def describe(td):
        out = {"": (td.device, td.batch_size, td.names if td._has_names() else None)}
        for key, value in td.items(include_nested=True):
            if is_tensor_collection(value):
                out[key] = (value.device, value.batch_size)
            else:
                out[key] = (value.device, value.shape, value.tolist())
        return out

    x = torch.arange(6.0).reshape(3, 2)
    expected = describe(fn(x))
    torch._dynamo.reset()
    result = torch.compile(fn, fullgraph=True, backend="eager")(x)
    assert describe(result) == expected


@pytest.mark.parametrize("device", [None, "cpu"])
def test_construction_empty_then_full_source(device):
    # A graph traced with an empty source is not reused for a full one: the
    # result kept no entries (#2648).
    def fn(x, source):
        return TensorDict(source, batch_size=[3], device=device)

    compiled = torch.compile(fn, backend="eager")
    x = torch.arange(3.0)
    assert compiled(x, {}).is_empty()
    result = compiled(x, {"a": x, "b": {"c": x}})
    assert set(result.keys(True, True)) == {"a", ("b", "c")}
    assert (result["b", "c"] == x).all()


@pytest.mark.parametrize("mode", [None, "reduce-overhead"])
class TestTD:
    def test_tensor_output(self, mode):
        def add_one(td):
            return td["a", "b"] + 1

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": 0}})
        assert add_one(data) == 1
        assert add_one_c(data) == 1
        assert add_one_c(data + 1) == 2

    def test_td_construct(self, mode):
        def fn(a, b):
            td = TensorDict({"a": a, "b": b}, batch_size=[3])
            return td["a"] + td["b"]

        fn_c = torch.compile(fn, fullgraph=True, mode=mode)
        a = torch.randn(3)
        b = torch.randn(3)
        torch.testing.assert_close(fn(a, b), fn_c(a, b))

    def test_td_construct_nested(self, mode):
        def fn(a, b):
            td = TensorDict(
                {"a": a, "nested": TensorDict({"b": b}, batch_size=[3])},
                batch_size=[3],
            )
            return td["a"] + td["nested", "b"]

        fn_c = torch.compile(fn, fullgraph=True, mode=mode)
        a = torch.randn(3)
        b = torch.randn(3)
        torch.testing.assert_close(fn(a, b), fn_c(a, b))

    def test_td_output(self, mode):
        def add_one(td):
            td["a", "c"] = td["a", "b"] + 1
            return td

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": 0}})
        assert add_one(data.clone())["a", "c"] == 1
        assert add_one_c(data.clone())["a", "c"] == 1
        assert add_one_c(data) is data

    @pytest.mark.parametrize("index_type", ["slice", "tensor", "int"])
    def test_td_index(self, index_type, mode):
        if index_type == "slice":

            def add_one(td):
                return td[:2] + 1

        elif index_type == "tensor":

            def add_one(td):
                return td[torch.tensor([0, 1])] + 1

        elif index_type == "int":

            def add_one(td):
                return td[0] + 1

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        if index_type == "int":
            assert (add_one(data)["a", "b"] == 1).all()
            assert (add_one_c(data)["a", "b"] == 1).all()
            assert add_one_c(data).shape == torch.Size([])
        else:
            assert (add_one(data)["a", "b"] == torch.arange(1, 3)).all()
            assert (add_one_c(data)["a", "b"] == torch.arange(1, 3)).all()
            assert add_one_c(data).shape == torch.Size([2])

    def test_td_index_empty_slice(self, mode):
        def index(td):
            return td[:0]

        index_c = torch.compile(index, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        result = index_c(data)
        assert result.shape == torch.Size([0])
        assert result["a", "b"].shape == torch.Size([0])

    @pytest.mark.parametrize("device", [None, "cpu"])
    def test_td_index_bool_mask(self, mode, device):
        def add_one(td, mask):
            return td[mask] + 1

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(3)}}, [3], device=device)
        result = add_one_c(data, torch.tensor([True, False, True]))
        assert result.shape == torch.Size([2])
        assert (result["a", "b"] == torch.tensor([1, 3])).all()

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "name", ["td[mask]", "td[mask[:, 0], 0]", "td.masked_select(mask)"]
    )
    def test_td_select_rows_bool_mask(self, mode, name, dynamic):
        # The batch size and the leaves count the mask the same way, so that
        # they share one size (#2785)
        fn = {
            "td[mask]": lambda td, mask: td[mask],
            "td[mask[:, 0], 0]": lambda td, mask: td[mask[:, 0], 0],
            "td.masked_select(mask)": lambda td, mask: td.masked_select(mask),
        }[name]
        td = TensorDict(
            a=torch.randn(4, 3, 5), b=TensorDict(c=torch.randn(4, 3)), batch_size=[4, 3]
        )
        mask = torch.tensor(
            [[True, False, True], [False, True, True], [True, True, False], [0, 0, 1]],
            dtype=torch.bool,
        )
        fn_c = torch.compile(fn, fullgraph=True, mode=mode, dynamic=dynamic)
        result = fn_c(td, mask)
        expected = fn(td, mask)
        assert result.batch_size == expected.batch_size
        assert (result == expected).all()

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("value", [1, torch.tensor(2.0)], ids=["int", "tensor"])
    @pytest.mark.parametrize("mask_dims", [1, 2])
    @pytest.mark.parametrize("exclusive", [False, True])
    def test_lazy_stack_write_rows_bool_mask(
        self, mode, dynamic, stack_dim, value, mask_dims, exclusive
    ):
        # Each member takes its slice of the mask (#2785)
        def fill(td, mask):
            td[mask] = value
            return td

        def make():
            tds = [
                TensorDict(a=torch.zeros(4, 5), b={"c": torch.zeros(4)}, batch_size=[4])
                for _ in range(4)
            ]
            if exclusive:
                # Eager writes only the keys that every member has
                tds[1]["x"] = torch.zeros(4, 2)
            return LazyStackedTensorDict(*tds, stack_dim=stack_dim)

        if exclusive:
            # The stack cannot be indexed in one graph here
            fill_c = torch.compile(fill, mode=mode, dynamic=dynamic)
        else:
            fill_c = torch.compile(fill, fullgraph=True, mode=mode, dynamic=dynamic)

        mask = torch.tensor(
            [[True, False, True, True], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.bool,
        )
        if mask_dims == 1:
            mask = mask[:, 0]
        result = fill_c(make(), mask)
        expected = fill(make(), mask)
        assert expected["a"].sum() > 0
        for r, e in zip(result.tensordicts, expected.tensordicts):
            assert (r == e).all()

    def test_stack(self, mode):
        def stack_tds(td0, td1):
            # return TensorDict.stack([td0, td1])
            return torch.stack([td0, td1])

        stack_tds_c = torch.compile(stack_tds, fullgraph=True, mode=mode)
        data0 = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        data1 = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        assert (stack_tds(data0, data1) == stack_tds_c(data0, data1)).all()

    def test_stack_refine_names_nested(self, mode):
        def stack_and_refine(td0, td1):
            out = torch.stack([td0, td1], 0)
            out.refine_names("time")
            return out

        stack_and_refine_c = torch.compile(stack_and_refine, fullgraph=True, mode=mode)

        td0 = TensorDict({"params": TensorDict({"g": torch.tensor(1.0)}, [])}, [])
        td1 = TensorDict({"params": TensorDict({"g": torch.tensor(2.0)}, [])}, [])

        out = stack_and_refine_c(td0, td1)
        assert out.names == ["time"]
        assert out["params"].names == ["time"]
        torch.testing.assert_close(out["params", "g"], torch.tensor([1.0, 2.0]))

    def test_cat(self, mode):
        def cat_tds(td0, td1):
            # return TensorDict.cat([td0, td1])
            return torch.cat([td0, td1])

        cat_tds_c = torch.compile(cat_tds, fullgraph=True, mode=mode)
        data0 = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        data1 = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        assert (cat_tds(data0, data1) == cat_tds_c(data0, data1)).all()

    @pytest.mark.parametrize("dim", [-1, 1])
    def test_pack(self, mode, dim):
        data = TensorDict(
            {"a": torch.randn(2, 4), "b": {"c": torch.randn(2, 4, 3, 2)}}, [2, 4]
        )
        _, layout = data.pack(dim=dim)

        def pack_unpack(td):
            flat, _ = td.pack(dim=dim, layout=layout)
            return layout.unpack(flat * 2)

        pack_unpack_c = torch.compile(pack_unpack, fullgraph=True, mode=mode)
        out = pack_unpack_c(data)
        assert (out == data * 2).all()
        assert out["b"].batch_size == (2, 4)
        out = pack_unpack_c(data)
        assert (out == data * 2).all()

    @pytest.mark.parametrize("dim", [-1, 1])
    def test_pack_graph_movedim(self, mode, dim):
        # Packing along the last dim traces no movedim (#2891).
        if mode is not None:
            pytest.skip("checks the traced graph once")
        data = TensorDict(
            {"a": torch.randn(2, 4), "b": {"c": torch.randn(2, 4, 3, 2)}}, [2, 4]
        )
        backend = EagerAndRecordGraphs()
        pack_c = torch.compile(
            lambda td: td.pack(dim=dim)[0], fullgraph=True, backend=backend
        )
        torch.testing.assert_close(pack_c(data), data.pack(dim=dim)[0])
        targets = [str(node.target) for g in backend.graphs for node in g.graph.nodes]
        assert any("movedim" in target for target in targets) == (dim != -1)

    def test_reshape(self, mode):
        def reshape(td):
            return td.reshape(2, 2)

        reshape_c = torch.compile(reshape, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        data_reshape = reshape(data)
        _ = reshape_c(data)
        data_reshape_c = reshape_c(data)
        assert (data_reshape == data_reshape_c).all()

    def test_torch_reshape_chunk(self, mode):
        def reshape_chunk(td):
            return torch.chunk(torch.reshape(td, (2, 2)), 2, 1)

        reshape_chunk_c = torch.compile(reshape_chunk, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        chunks = reshape_chunk(data)
        chunks_c = reshape_chunk_c(data)
        assert len(chunks) == len(chunks_c) == 2
        for chunk, chunk_c in zip(chunks, chunks_c):
            assert chunk_c.batch_size == (2, 1)
            assert (chunk == chunk_c).all()

    def test_view(self, mode):
        def view(td):
            out = td.view(2, 2).clear_refs_for_compile_()
            return out

        view_c = torch.compile(view, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        data_view = view(data)
        _ = view_c(data)
        data_view_c = view_c(data)
        assert (data_view == data_view_c).all()

    def test_transpose(self, mode):
        def transpose(td):
            return td.transpose(0, 1).clear_refs_for_compile_()

        transpose_c = torch.compile(transpose, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(6).view(2, 3)}}, [2, 3])
        data_transpose = transpose(data)
        _ = transpose_c(data)
        data_transpose_c = transpose_c(data)
        assert (data_transpose == data_transpose_c).all()

    @pytest.mark.parametrize("dims", [(1, 0), (0, 1)])
    @pytest.mark.parametrize("legacy", [False, True])
    def test_permute(self, dims, legacy, mode):
        def permute(td):
            return td.permute(*dims)["a", "b"]

        def permute_twice(td):
            # In legacy mode, the second permute is applied to a permuted td.
            return td.permute(*dims).permute(*dims)["a", "b"]

        permute_c = torch.compile(permute, fullgraph=True, mode=mode)
        permute_twice_c = torch.compile(permute_twice, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(6).view(2, 3)}}, [2, 3])
        with legacy_lazy_mode() if legacy else contextlib.nullcontext():
            data_permute = permute(data)
            _ = permute_c(data)
            data_permute_c = permute_c(data)
            data_permute_twice = permute_twice(data)
            data_permute_twice_c = permute_twice_c(data)
        torch.testing.assert_close(data_permute_c, data_permute)
        torch.testing.assert_close(data_permute_twice_c, data_permute_twice)

    def test_lazy_stack_contains_is_empty(self, mode):
        def contains_is_empty(td):
            return "a" in td.keys(), "c" in td.keys(), td.is_empty(), td["a"] + 1

        contains_is_empty_c = torch.compile(
            contains_is_empty, fullgraph=True, mode=mode
        )
        data = lazy_stack([TensorDict(a=torch.randn(3)) for _ in range(2)])
        has_a, has_c, is_empty, a = contains_is_empty_c(data)
        assert has_a
        assert not has_c
        assert not is_empty
        torch.testing.assert_close(a, data["a"] + 1)

    def test_lazy_stack_keys_entry_class(self, mode):
        def read(td):
            assert list(td.keys(True, True)) == ["a"]
            assert td.entry_class("a") is torch.Tensor
            return td["a"] + 1

        read_c = torch.compile(read, fullgraph=True, mode=mode)
        data = lazy_stack(
            [
                TensorDict(a=torch.zeros(3), batch_size=[3]),
                TensorDict(a=torch.ones(3), batch_size=[3]),
            ]
        )
        torch.testing.assert_close(read_c(data), read(data))

    def test_lazy_stack_set_inplace_pop(self, mode):
        def set_inplace_pop(td):
            td.set_("b", td["b"] + 1)
            return td.pop("a")

        set_inplace_pop_c = torch.compile(set_inplace_pop, fullgraph=True, mode=mode)
        data = lazy_stack(
            [TensorDict(a=torch.randn(3), b=torch.randn(3)) for _ in range(2)]
        )
        data_c = data.clone()
        a = set_inplace_pop(data)
        a_c = set_inplace_pop_c(data_c)
        torch.testing.assert_close(a_c, a)
        assert "a" not in data_c.keys()
        assert (data_c == data).all()

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_lazy_stack_permute(self, dynamic, mode):
        # #2786: the new stack dim was a NumPy value
        def permute(td):
            return td.permute(1, 0)

        permute_c = torch.compile(permute, fullgraph=True, dynamic=dynamic, mode=mode)
        data = lazy_stack(
            [TensorDict(a=torch.randn(3, 2), batch_size=[3]) for _ in range(4)]
        )
        result = permute_c(data)
        assert isinstance(result, LazyStackedTensorDict)
        assert result.stack_dim == 1
        assert (result == permute(data)).all()

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_lazy_stack_get_through_leaf_default(self, dynamic, mode):
        # #2786: a key that runs through a tensor returns the default
        def get(td):
            return td.get(("a", "x"), None), td["a"] + 1

        get_c = torch.compile(get, fullgraph=True, dynamic=dynamic, mode=mode)
        data = lazy_stack([TensorDict(a=torch.randn(3)) for _ in range(2)])
        x, a = get_c(data)
        assert x is None
        torch.testing.assert_close(a, data["a"] + 1)

    @pytest.mark.parametrize("out_dim", [0, 1])
    def test_lazy_stack_apply_out(self, out_dim, mode):
        # #2982: out stacked along another dim than the source
        def apply(td, out):
            return td.apply(lambda x: x + 1, out=out)

        apply_c = torch.compile(apply, fullgraph=True, mode=mode)
        data = TensorDict(x=torch.arange(6).reshape(2, 3), batch_size=[2, 3])
        source = lazy_stack(data.unbind(0), 0)
        out = lazy_stack(torch.zeros_like(data).unbind(out_dim), out_dim)
        result = apply_c(source, out)
        assert (result is out) == (out_dim == 1)
        assert out.stack_dim == out_dim
        torch.testing.assert_close(out.to_tensordict(), data + 1)

    @pytest.mark.parametrize("kwarg", ["as_list", "as_nested_tensor"])
    def test_dense_get_ignored_kwargs(self, kwarg, mode):
        # a dense get warns about these keywords in eager mode only (#2551)
        def get(td):
            return td.get("x", **{kwarg: True}) + 1

        get_c = torch.compile(get, fullgraph=True, mode=mode)
        data = torch.stack([TensorDict(x=torch.arange(4.0) + i) for i in range(2)])
        with pytest.warns(FutureWarning):
            x = get(data)
        torch.testing.assert_close(get_c(data), x)

    def test_unbind(self, mode):
        def unbind(td):
            return td.unbind(0)

        unbind_c = torch.compile(unbind, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        assert (unbind(data)[-1] == unbind_c(data)[-1]).all()

    @staticmethod
    def _unbind_split(td, op):
        if op == "unbind":
            return td.unbind(0)
        if op == "iter":
            return list(td)
        if op == "chunk":
            return td.chunk(2, 0)
        return td.split(2, 0)

    @pytest.mark.parametrize("op", ["unbind", "iter", "chunk", "split"])
    def test_unbind_split_shared(self, op, mode):
        # A shared tensordict can be unbound, iterated, chunked and split in
        # a compiled graph (#2788).
        fn_c = torch.compile(self._unbind_split, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(4.0)}}, [4]).share_memory_()
        expected = self._unbind_split(data, op)
        result = fn_c(data, op)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.batch_size == e.batch_size
            assert r.is_shared() == e.is_shared()
            assert (r == e).all()

    @pytest.mark.parametrize("op", ["unbind", "iter", "chunk", "split"])
    @pytest.mark.parametrize("container", ["td", "tc"])
    def test_unbind_split_memmap(self, op, container, mode, tmpdir):
        # A memory-mapped tensordict or tensorclass can be unbound, iterated,
        # chunked and split in a compiled graph (#2788).
        fn_c = torch.compile(self._unbind_split, fullgraph=True, mode=mode)
        if container == "td":
            data = TensorDict({"a": {"b": torch.arange(4.0)}}, [4])
        else:
            data = MyClass(
                a=MyClass(a=None, b=torch.arange(4.0), batch_size=[4]), batch_size=[4]
            )
        data.memmap_(tmpdir)
        expected = self._unbind_split(data, op)
        result = fn_c(data, op)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert type(r) is type(e)
            assert r.batch_size == e.batch_size
            assert r.is_memmap()
            assert (r == e).all()
            # Dynamo cannot build a MemoryMappedTensor, so compiled code
            # returns plain tensors over the same file, as compiled indexing
            # does.
            assert type(r.get(("a", "b"))) is torch.Tensor
            assert isinstance(e.get(("a", "b")), MemoryMappedTensor)
        # Writes through a result reach the file.
        result[0].get(("a", "b")).add_(10)
        assert data.get(("a", "b"))[0] == 10
        assert type(data).load_memmap(tmpdir).get(("a", "b"))[0] == 10

    def test_split_across_batch_sizes(self, mode):
        # From the second batch size on, the batch size is dynamic: the split
        # sizes must be computed from a symbolic size under fullgraph=True.
        def split(td):
            return td.split(2, 0)

        split_c = torch.compile(split, fullgraph=True, mode=mode)
        for n in (4, 6, 7, 8):
            data = TensorDict({"a": {"b": torch.arange(n)}}, [n])
            expected = split(data)
            result = split_c(data)
            assert [r.batch_size for r in result] == [e.batch_size for e in expected]
            for r, e in zip(result, expected):
                assert (r["a", "b"] == e["a", "b"]).all()

    @pytest.mark.parametrize("dynamic", [False, True])
    @pytest.mark.parametrize(
        "op",
        ["chunk", "split", "cat", "stack", "or", "update", "update_at_", "setitem"],
    )
    def test_unbatched_new_batch_size(self, op, dynamic, mode):
        # These operations give an unbatched entry a new batch size. With
        # dynamic shapes it is an expression such as s0 // 2, which Dynamo
        # must read back from the entry (#2789); with static shapes the entry
        # must not keep the batch size of the input.
        def setitem(td):
            td[0] = td.to_tensordict()[0] * 0
            return td

        fn = {
            "chunk": lambda td: td.chunk(2, 0),
            "split": lambda td: td.split([1, 3], 0),
            "cat": lambda td: (torch.cat([td, td], 0),),
            "stack": lambda td: (torch.stack([td, td], 1),),
            "or": lambda td: ((td > 0) | (td < 1),),
            "update": lambda td: (td.update(td.to_tensordict() + 1),),
            "update_at_": lambda td: (
                td.update_at_(td.to_tensordict()[1:3] * 0, slice(1, 3)),
            ),
            "setitem": lambda td: (setitem(td),),
        }[op]
        fn_c = torch.compile(fn, fullgraph=True, dynamic=dynamic, mode=mode)
        a, u = torch.randn(4, 3, 2, 1, 5), torch.randn(7, 11)

        def make():
            return TensorDict(
                a=a.clone(), u=UnbatchedTensor(u.clone()), batch_size=[4, 3, 2, 1]
            )

        expected = fn(make())
        result = fn_c(make())
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.batch_size == e.batch_size
            assert isinstance(r["u"], UnbatchedTensor)
            assert r["u"].batch_size == e["u"].batch_size
            assert (r["a"] == e["a"]).all()
            assert (r["u"] == e["u"]).all()

    def test_stack_distinct_unbatched(self, mode):
        # Stacking distinct unbatched entries compares their data pointers,
        # which breaks the graph. With the fallback UnbatchedTensor (torch
        # 2.13 and older), the Dynamo of torch 2.11 recursed without end there.
        def make(value):
            return TensorDict(
                a=torch.full((3, 2), value),
                u=UnbatchedTensor(torch.full((5,), value)),
                batch_size=[3],
            )

        fn_c = torch.compile(lambda x, y: torch.stack([x, y], 0), mode=mode)
        with pytest.warns(UserWarning, match="different data storage"):
            result = fn_c(make(0.0), make(1.0))
        assert result.batch_size == (2, 3)
        assert (result["a"][1] == 1).all()
        assert isinstance(result["u"], UnbatchedTensor)
        assert (result["u"] == 0).all()

    @pytest.mark.parametrize("named", [False, True])
    def test_apply_is_leaf(self, named, mode):
        # With is_leaf=is_leaf_nontensor, fn is called on the non-tensor entry
        # and drops it by returning None
        def fn(x):
            return x * 2 if isinstance(x, torch.Tensor) else None

        def apply(td):
            if named:
                return td.named_apply(lambda name, x: fn(x), is_leaf=is_leaf_nontensor)
            return td.apply(fn, is_leaf=is_leaf_nontensor)

        apply_c = torch.compile(apply, fullgraph=True, mode=mode)
        for n in (4, 6):
            td = TensorDict(
                a=torch.ones(n), b={"c": torch.ones(n, 2)}, s="a string", batch_size=[n]
            )
            expected = apply(td)
            result = apply_c(td)
            assert set(expected.keys(True, True)) == {"a", ("b", "c")}
            assert set(result.keys(True, True)) == {"a", ("b", "c")}
            assert_close(result, expected)

    @pytest.mark.parametrize("dtype", [None, torch.float64])
    def test_new_tensor_dtype(self, dtype, mode):
        # Each entry takes its own dtype, or dtype if given, as in eager mode;
        # not the dtype of the data.
        def new_tensor(td, data):
            return td.new_tensor(data, dtype=dtype)

        new_tensor_c = torch.compile(new_tensor, fullgraph=True, mode=mode)
        for n in (4, 6, 8):
            td = TensorDict(
                {"a": torch.zeros(n), "b": {"c": torch.zeros(n, dtype=torch.int32)}},
                [n],
            )
            data = torch.arange(n)
            with pytest.warns(UserWarning, match="To copy construct from a tensor"):
                expected = new_tensor(td, data)
            result = new_tensor_c(td, data)
            assert result.batch_size == expected.batch_size
            for key in (("a",), ("b", "c")):
                assert result[key].dtype == expected[key].dtype
                assert (result[key] == expected[key]).all()

    def test_iter(self, mode):
        def iterate(td):
            return torch.stack([t["a", "b"] + 1 for t in td])

        iterate_c = torch.compile(iterate, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        assert (iterate(data) == iterate_c(data)).all()

    def test_items(self, mode):
        def items(td):
            keys, vals = zip(*td.items(True, True))
            return keys, vals

        items_c = torch.compile(items, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        keys, vals = items(data)
        keys_c, vals_c = items_c(data)

        def assert_eq(x, y):
            assert (x == y).all()

        assert keys == keys_c
        torch.utils._pytree.tree_map(assert_eq, vals, vals_c)

    def test_items_nested(self, mode):
        # Without leaves_only, TensorDict.items goes through TensorDictBase.items.
        def items(td):
            keys, vals = zip(*td.items(True))
            return keys, [val + 1 for val in vals]

        items_c = torch.compile(items, fullgraph=True, mode=mode)
        data = TensorDict(
            {"a": {"b": {"c": torch.arange(4)}, "d": torch.arange(4)}}, [4]
        )
        keys, vals = items(data)
        keys_c, vals_c = items_c(data)
        assert keys == keys_c
        assert keys == ("a", ("a", "b"), ("a", "b", "c"), ("a", "d"))
        for val, val_c in zip(vals, vals_c):
            assert (val == val_c).all()

    def test_keys_nested(self, mode):
        # Iterating, counting and sorting the nested keys compiles to one
        # frame, without a recompile (#2618).
        def keys(td):
            out = 0
            for key in td.keys(True, True):
                out = out + td.get(key).sum()
            return (
                out,
                len(td.keys(True, False)),
                list(td.keys(True, True, sort=True)),
            )

        data = TensorDict(
            {
                "a": torch.arange(4.0),
                "b": {"c": torch.ones(4), "d": {"e": torch.ones(4)}},
            },
            [4],
        )
        for td in (
            data,
            data.clone().lock_(),
            MyClass(a=MyClass(a=None, b=torch.ones(())), b=torch.zeros(())),
        ):
            torch._dynamo.reset_code_caches()
            cnt = CompileCounterWithBackend("eager")
            if mode is None:
                keys_c = torch.compile(keys, fullgraph=True, backend=cnt)
            else:
                keys_c = torch.compile(keys, fullgraph=True, mode=mode)
            expected = keys(td)
            for _ in range(2):
                out, length, sorted_keys = keys_c(td)
                torch.testing.assert_close(out, expected[0])
                assert (length, sorted_keys) == expected[1:]
            if mode is None:
                assert cnt.frame_count == 1

    @pytest.mark.parametrize("recurse", [True, False])
    @pytest.mark.parametrize("lock", [True, False])
    def test_clone(self, recurse, lock, mode):
        def clone(td: TensorDict):
            return td.clone(recurse=recurse)

        clone_c = torch.compile(clone, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": 0, "c": 1}})
        if lock:
            data = data.lock_()
        data_c = clone(data)
        _ = clone_c(data)
        data_c_c = clone_c(data)
        assert_close(data_c, data_c_c)
        assert clone_c(data) is not data
        data_c_c.rename_key_(("a", "b"), ("a", "renamed"))
        assert ("a", "b") in data.keys(include_nested=True)
        if recurse:
            assert clone_c(data)["a", "c"] is not data["a", "c"]
        else:
            assert clone_c(data)["a", "c"] is data["a", "c"]

    @pytest.mark.parametrize("recurse", [True, False])
    def test_flatten_keys(self, recurse, mode):
        def flatten_keys(td: TensorDict):
            return td.flatten_keys().clear_refs_for_compile_()

        flatten_keys_c = torch.compile(flatten_keys, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": 0, "c": 1}})
        data_f = flatten_keys(data)
        _ = flatten_keys(data)
        data_f_c = flatten_keys(data)
        assert_close(data_f, data_f_c)
        assert flatten_keys_c(data) is not data
        assert flatten_keys_c(data)["a.b"] is data["a", "b"]

    @pytest.mark.parametrize("recurse", [True, False])
    def test_unflatten_keys(self, recurse, mode):
        def unflatten_keys(td: TensorDict):
            return td.unflatten_keys().clear_refs_for_compile_()

        unflatten_keys_c = torch.compile(unflatten_keys, fullgraph=True, mode=mode)
        data = TensorDict({"a.b": 0, "a.c": 1})
        data_t = unflatten_keys(data)
        _ = unflatten_keys_c(data)
        data_t_c = unflatten_keys_c(data)
        assert_close(data_t, data_t_c)
        assert unflatten_keys_c(data) is not data
        assert unflatten_keys_c(data)["a", "b"] is data["a.b"]

    def test_pop(self, mode):
        def pop_existing(td: TensorDict):
            return td.pop("a")

        def pop_missing_with_default(td: TensorDict):
            return td.pop("missing", None)

        pop_existing_c = torch.compile(pop_existing, fullgraph=True, mode=mode)
        pop_missing_c = torch.compile(
            pop_missing_with_default, fullgraph=True, mode=mode
        )

        # Test pop existing key
        data = TensorDict({"a": torch.tensor(1), "b": torch.tensor(2)})
        result = pop_existing(data.clone())
        assert result == 1

        data = TensorDict({"a": torch.tensor(1), "b": torch.tensor(2)})
        result_c = pop_existing_c(data.clone())
        assert result_c == 1

        # Verify key is removed
        data = TensorDict({"a": torch.tensor(1), "b": torch.tensor(2)})
        _ = pop_existing_c(data)
        assert "a" not in data.keys()
        assert "b" in data.keys()

        # Test pop missing key with default
        data = TensorDict({"a": torch.tensor(1)})
        result = pop_missing_with_default(data.clone())
        assert result is None

        data = TensorDict({"a": torch.tensor(1)})
        result_c = pop_missing_c(data.clone())
        assert result_c is None

    def test_select_strict_false(self, mode):
        def select_keys(td: TensorDict):
            return td.select("a", "missing_key", strict=False)

        select_keys_c = torch.compile(select_keys, fullgraph=True, mode=mode)

        # Test select with strict=False
        data = TensorDict({"a": torch.tensor(1), "b": torch.tensor(2)})
        result = select_keys(data)
        assert "a" in result.keys()
        assert "missing_key" not in result.keys()
        assert "b" not in result.keys()

        result_c = select_keys_c(data)
        assert "a" in result_c.keys()
        assert "missing_key" not in result_c.keys()
        assert "b" not in result_c.keys()

    @pytest.mark.parametrize("batch_size", [[3], [5]])
    def test_rename_keys(self, mode, batch_size):
        def rename_keys(td: TensorDict):
            return td.rename_keys({"a": "b", "b": "a", ("raw", "p"): "pressure"})

        rename_keys_c = torch.compile(rename_keys, fullgraph=True, mode=mode)

        data = TensorDict(
            a=torch.zeros(batch_size),
            b=torch.ones(batch_size),
            raw=TensorDict(p=torch.full(batch_size, 2.0), batch_size=batch_size),
            batch_size=batch_size,
        )
        result = rename_keys(data)
        result_c = rename_keys_c(data)
        assert set(result_c.keys(True)) == set(result.keys(True))
        assert set(result_c.keys(True)) == {"a", "b", "pressure"}
        assert (result_c == result).all()
        assert (result_c["a"] == 1).all()
        assert set(data.keys(True)) == {"a", "b", "raw", ("raw", "p")}

    def test_rename_keys_params(self, mode):
        # TensorDictParams.rename_keys returns a TensorDict, which compiles
        def rename_keys(params):
            return params.rename_keys({"a": "b", "b": "a"}).rename_keys(
                {"missing": "c"}, strict=False
            )

        rename_keys_c = torch.compile(rename_keys, fullgraph=True, mode=mode)
        params = TensorDictParams(
            TensorDict(a=torch.zeros(3), b=torch.ones(3), batch_size=[3])
        )
        result = rename_keys(params)
        result_c = rename_keys_c(params)
        assert type(result_c) is type(result) is TensorDict
        assert (result_c == result).all()
        assert (result_c["a"] == 1).all()
        assert result_c["a"] is params["b"]

    def test_exclude(self, mode):
        def exclude_keys(td: TensorDict):
            return td.exclude("b")

        exclude_keys_c = torch.compile(exclude_keys, fullgraph=True, mode=mode)

        data = TensorDict({"a": torch.tensor(1), "b": torch.tensor(2)})
        result = exclude_keys(data)
        assert "a" in result.keys()
        assert "b" not in result.keys()

        result_c = exclude_keys_c(data)
        assert "a" in result_c.keys()
        assert "b" not in result_c.keys()

    def test_split_keys_callable(self, mode):
        def split(td: TensorDict):
            floats, rest = td.split_keys(lambda k, v: v.is_floating_point())
            return floats + 1, rest

        split_c = torch.compile(split, fullgraph=True, mode=mode)

        for batch in (3, 5):
            data = TensorDict(
                a=torch.zeros(batch),
                b=torch.zeros(batch, dtype=torch.long),
                nested=TensorDict(c=torch.zeros(batch, 2), batch_size=[batch]),
                batch_size=[batch],
            )
            floats, rest = split(data)
            floats_c, rest_c = split_c(data)
            assert set(floats_c.keys(True, True)) == {"a", ("nested", "c")}
            assert set(rest_c.keys(True, True)) == {"b"}
            assert (floats_c == floats).all()
            assert (rest_c == rest).all()

    def test_all_any(self, mode):
        def call_all(td: TensorDict):
            return td.all(dim=0)

        def call_any(td: TensorDict):
            return td.any(dim=0)

        call_all_c = torch.compile(call_all, fullgraph=True, mode=mode)
        call_any_c = torch.compile(call_any, fullgraph=True, mode=mode)

        data = TensorDict(
            {"a": torch.tensor([[True, False], [True, True]])},
            batch_size=[2, 2],
        )

        result_all = call_all(data)
        result_all_c = call_all_c(data)
        assert (result_all["a"] == result_all_c["a"]).all()
        assert result_all_c.shape == torch.Size([2])

        result_any = call_any(data)
        result_any_c = call_any_c(data)
        assert (result_any["a"] == result_any_c["a"]).all()
        assert result_any_c.shape == torch.Size([2])

    def test_squeeze_unsqueeze(self, mode):
        def call_squeeze(td: TensorDict):
            return td.squeeze(0)

        def call_unsqueeze(td: TensorDict):
            return td.unsqueeze(0)

        call_squeeze_c = torch.compile(call_squeeze, fullgraph=True, mode=mode)
        call_unsqueeze_c = torch.compile(call_unsqueeze, fullgraph=True, mode=mode)

        data = TensorDict({"a": torch.randn(1, 3)}, batch_size=[1, 3])

        result_squeeze = call_squeeze(data)
        result_squeeze_c = call_squeeze_c(data)
        assert result_squeeze.shape == result_squeeze_c.shape
        assert result_squeeze_c.shape == torch.Size([3])

        result_unsqueeze = call_unsqueeze(result_squeeze)
        result_unsqueeze_c = call_unsqueeze_c(result_squeeze_c)
        assert result_unsqueeze.shape == result_unsqueeze_c.shape
        assert result_unsqueeze_c.shape == torch.Size([1, 3])

    def test_names(self, mode):
        def make_td_with_names(data):
            return TensorDict(data, batch_size=[1, 2], names=["d0", "d1"])

        data_dict = {
            "a": torch.randn(1, 2, 3),
            "b": torch.zeros(1, 2, 3, dtype=torch.bool),
        }
        make_td_with_names_c = torch.compile(
            make_td_with_names, fullgraph=True, mode=mode
        )
        make_td_with_names(data_dict)
        td = make_td_with_names_c(data_dict)
        assert td.names == ["d0", "d1"]

    def test_names_kept_by_ops(self, mode):
        # TensorDict._new_unsafe calls TensorDict(..., names=names) under
        # compile, so every op that rebuilds a tensordict goes through the
        # names argument of __init__.
        def ops(td):
            nested = TensorDict(
                {"a": td["a"], "sub": {"b": td["a"]}}, batch_size=[3], names=["n"]
            )
            return (
                td.clone(),
                td.copy(),
                td.clone(False),
                td.select("a"),
                td + 1,
                td[:2],
                torch.stack([td, td], 1),
                nested,
            )

        ops_c = torch.compile(ops, fullgraph=True, mode=mode)
        td = TensorDict(a=torch.zeros(3, 2), batch_size=[3, 2], names=["x", "y"])
        clone, copy, shallow_clone, select, add, index, stack, nested = ops_c(td)
        assert clone.names == ["x", "y"]
        assert copy.names == ["x", "y"]
        assert shallow_clone.names == ["x", "y"]
        assert select.names == ["x", "y"]
        assert add.names == ["x", "y"]
        assert index.names == ["x", "y"]
        assert stack.names == ["x", None, "y"]
        assert nested.names == ["n"]
        assert nested["sub"].names == ["n"]

    def test_to_memory_format(self, mode):
        def to_channels_last(td):
            return td.to(memory_format=torch.channels_last)

        td = TensorDict({"a": torch.randn(1, 2, 3, 4)}, batch_size=[1])
        to_channels_last_c = torch.compile(to_channels_last, fullgraph=True, mode=mode)
        td_c = to_channels_last_c(td)
        assert td_c["a"].is_contiguous(memory_format=torch.channels_last)
        torch.testing.assert_close(td_c["a"], td["a"])

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="cuda required to test device casting"
    )
    @pytest.mark.parametrize("has_device", [True, False])
    def test_to(self, has_device, mode):
        device = f"{cur_device}:0"

        def test_to_device(td):
            return td.to(device)

        td = TensorDict(
            {"a": torch.randn(1, 2, 3), "b": torch.zeros(1, 2, 3, dtype=torch.bool)},
            batch_size=[1, 2],
            device="cpu" if has_device else None,
        )
        test_to_device_c = torch.compile(test_to_device, fullgraph=True, mode=mode)
        # td_device = test_to_device(td)
        _ = test_to_device_c(td)
        td_device_c = test_to_device_c(td)
        assert td_device_c.batch_size == td.batch_size
        assert td_device_c.device == torch.device(device)

    def test_to_cpu(self, mode):
        def to_cpu(td):
            return td.to("cpu")

        td = TensorDict({"a": torch.randn(1, 2, 3)}, batch_size=[1, 2])
        to_cpu_c = torch.compile(to_cpu, fullgraph=True, mode=mode)
        td_cpu_c = to_cpu_c(td)
        assert td_cpu_c.device == torch.device("cpu")
        torch.testing.assert_close(td_cpu_c["a"], td["a"])

    @pytest.mark.parametrize("batch_size", [[2], [2, 3]])
    def test_to_batch_size_same_device(self, mode, batch_size):
        def to_cpu(td):
            return td.to("cpu", batch_size=batch_size)

        td = TensorDict(a=torch.zeros(2, 3), batch_size=[2], device="cpu")
        to_cpu_c = torch.compile(to_cpu, fullgraph=True, mode=mode)
        result = to_cpu_c(td)
        assert result is td
        assert result.batch_size == torch.Size(batch_size)
        torch.testing.assert_close(result["a"], torch.zeros(2, 3))

    @pytest.mark.parametrize("batch_size", [[2], [2, 3], [5]])
    def test_to_batch_size_new_device(self, mode, batch_size):
        def to_meta(td):
            return td.to("meta", batch_size=batch_size)

        td = TensorDict(
            nested=TensorDict(a=torch.zeros(2, 3), batch_size=[2]), batch_size=[2]
        )
        to_meta_c = torch.compile(to_meta, fullgraph=True, mode=mode)
        if batch_size == [5]:
            with pytest.raises(RuntimeError, match="incompatible with the batch-size"):
                to_meta_c(td)
            return
        result = to_meta_c(td)
        assert result.batch_size == torch.Size(batch_size)
        assert result["nested"].batch_size == torch.Size(batch_size)
        assert result["nested", "a"].device == torch.device("meta")

    def test_to_batch_size_new_device_future_warning(self, mode):
        # eager mode warns on this call; compiled code skips the warning
        def to_meta(td):
            return td.to("meta", batch_size=[2, 3])

        td = TensorDict(
            a=torch.zeros(2, 3, 4),
            nested=TensorDict(b=torch.zeros(2, 3, 4), batch_size=[2, 3, 4]),
            batch_size=[2],
            names=["r"],
        )
        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
            expected = to_meta(td)
        to_meta_c = torch.compile(to_meta, fullgraph=True, mode=mode)
        result = to_meta_c(td)
        assert result.batch_size == expected.batch_size
        assert result["nested"].batch_size == expected["nested"].batch_size
        assert result.names == expected.names
        assert result["nested", "b"].device == torch.device("meta")

    @pytest.mark.parametrize("batch_size", [[2], [2, 3]])
    def test_to_dtype_batch_size_locked_nested_lazy(self, mode, batch_size):
        # #2374
        def to_double(td):
            return td.to(torch.float64, batch_size=batch_size)

        td = TensorDict(
            lazy=lazy_stack([TensorDict(x=torch.zeros(3)) for _ in range(2)]),
            y=torch.zeros(2, 3),
            batch_size=[2],
        ).lock_()
        to_double_c = torch.compile(to_double, fullgraph=True, mode=mode)
        result = to_double_c(td)
        assert result.batch_size == torch.Size(batch_size)
        assert result.is_locked
        assert result["lazy", "x"].dtype == torch.float64
        assert result["y"].dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
    def test_to_cpu_from_cuda(self, mode):
        def to_cpu(td):
            return (td + 1).to("cpu")

        td = TensorDict({"a": torch.randn(1, 2, 3)}, batch_size=[1, 2], device="cuda")
        to_cpu_c = torch.compile(to_cpu, fullgraph=True, mode=mode)
        td_cpu = to_cpu(td)
        td_cpu_c = to_cpu_c(td)
        assert td_cpu_c.device == torch.device("cpu")
        assert td_cpu_c["a"].device == torch.device("cpu")
        torch.testing.assert_close(td_cpu_c["a"], td_cpu["a"])

    def test_to_dtype(self, mode):
        def to_dtype(td):
            return td.to(torch.float64)

        def to_tensor(td, other):
            return td.to(other)

        td = TensorDict({"a": torch.randn(1, 2, 3)}, batch_size=[1, 2])
        to_dtype_c = torch.compile(to_dtype, fullgraph=True, mode=mode)
        to_tensor_c = torch.compile(to_tensor, fullgraph=True, mode=mode)
        assert to_dtype_c(td)["a"].dtype == torch.float64
        other = torch.zeros((), dtype=torch.float64)
        td_other_c = to_tensor_c(td, other)
        assert td_other_c["a"].dtype == torch.float64
        assert td_other_c.device == other.device

    def test_to_dtype_cast_filter(self, mode):
        def to_bf16(td):
            return td.to(torch.bfloat16, cast_filter=is_inexact)

        def half_keyed(td):
            return td.half(cast_filter=lambda k, v: is_inexact(v) and k[0] != "keep")

        to_bf16_c = torch.compile(to_bf16, fullgraph=True, mode=mode)
        half_keyed_c = torch.compile(half_keyed, fullgraph=True, mode=mode)
        for batch in (2, 5):
            td = TensorDict(
                a=torch.randn(batch, 3),
                done=torch.zeros(batch, dtype=torch.bool),
                keep=TensorDict(b=torch.randn(batch), batch_size=[batch]),
                ids=torch.arange(batch),
                batch_size=[batch],
            )
            out = to_bf16_c(td)
            assert_close(out, to_bf16(td))
            assert out["a"].dtype == torch.bfloat16
            assert out["keep", "b"].dtype == torch.bfloat16
            assert out["done"].dtype == torch.bool
            assert out["ids"].dtype == torch.long
            out = half_keyed_c(td)
            assert_close(out, half_keyed(td))
            assert out["a"].dtype == torch.half
            assert out["keep", "b"].dtype == torch.float32
            assert out["ids"].dtype == torch.long

    @pytest.mark.skipif(
        is_npu_available(),
        reason="torch.device in torch.compile is not supported on NPU currently.",
    )
    def test_lock(self, mode):
        def locked_op(td):
            # Adding stuff uses cache, check that this doesn't break
            td2 = td + 1
            td3 = td + td2
            return td3.clear_refs_for_compile_()

        td = TensorDict(
            {"a": torch.randn(1, 2, 3), "b": torch.zeros(1, 2, 3, dtype=torch.bool)},
            batch_size=[1, 2],
            device="cpu",
            lock=True,
        )
        locked_op_c = torch.compile(locked_op, fullgraph=True, mode=mode)
        td_op = locked_op(td)
        # no warning the second time this is run
        with (
            pytest.warns(UserWarning, match="Using lock_")
            if mode is None
            else contextlib.nullcontext()
        ):
            _ = locked_op_c(td)
        td_op_c = locked_op_c(td)
        assert (td_op == td_op_c).all()

    def test_lock_inplace(self, mode):
        def locked_op(td):
            # Adding stuff uses cache, check that this doesn't break
            td += 1
            td += td
            return td

        td = TensorDict(
            {"a": torch.randn(1, 2, 3), "b": torch.ones(1, 2, 3, dtype=torch.int64)},
            batch_size=[1, 2],
            device="cpu",
            lock=True,
        )
        locked_op_c = torch.compile(locked_op, fullgraph=True, mode=mode)
        td_op = locked_op(td)
        # no warning the second time this is run
        _ = locked_op_c(td)
        td_op_c = locked_op_c(td)
        assert (td_op == td_op_c).all()

    def test_inplace_broadcast_tensor(self, mode):
        def add_(td, other):
            return td.add_(other)

        td = TensorDict(
            {"a": torch.zeros(1, 2, 3), "b": torch.zeros(1, 2, dtype=torch.int64)},
            batch_size=[1, 2],
            lock=True,
        )
        add_c = torch.compile(add_, fullgraph=True, mode=mode)
        other = torch.ones(1, 2, dtype=torch.int64)
        assert add_c(td, other) is td
        assert add_c(td, other) is td
        assert (td["a"] == 2).all()

    def test_inplace_methods(self, mode):
        # The in-place methods trace without a graph break: the check for
        # containers that read copies of their data is a class attribute, and
        # the one for nested lazy stacks that read copies (#2592) is a try
        def inplace(td, other):
            td.zero_()
            td.abs_()
            td.add_(other, alpha=2.0)
            td.mul_(2.0)
            td += 1
            td.lerp_(other, 0.5)
            td.clamp_max_(other)
            td.update_(other * 2, keys_to_update=["a"])
            return td.copy_(td * 0.5)

        td = TensorDict(a=torch.randn(4, 3), b={"c": torch.randn(4, 2)}, batch_size=[4])
        other = td.apply(lambda x: torch.rand_like(x) + 1)
        expected = inplace(td.clone(), other)
        inplace_c = torch.compile(inplace, fullgraph=True, mode=mode)
        result = td.clone()
        assert inplace_c(result, other) is result
        assert_close(result, expected)

    def test_update__keys_to_update(self, mode):
        # update_(keys_to_update=...) traces when it copies the selected
        # tensors with _foreach_copy_ and writes a tensor that requires grad
        # and a nested tensorclass on their own, and backpropagates as eager
        def update_(td, other):
            td.update_(other, keys_to_update=["a", ("n", "x"), "b", "tc"])
            return td["b"].sum() + td["a"].sum()

        def make():
            x = torch.ones(4, requires_grad=True)
            td = TensorDict(
                a=torch.zeros(4, 3),
                n=TensorDict(x=torch.zeros(4, 2), y=torch.zeros(4)),
                b=torch.zeros(4),
                tc=MyClass(a=None, b=torch.zeros(4), batch_size=[4]),
                batch_size=[4],
            )
            other = td.apply(lambda t: t + 1)
            other["b"] = x * 2
            return x, td, other

        x, td, other = make()
        update_(td, other).backward()
        update_c = torch.compile(update_, fullgraph=True, mode=mode)
        x_c, td_c, other_c = make()
        update_c(td_c, other_c).backward()
        assert_close(td_c, td)
        torch.testing.assert_close(x_c.grad, x.grad)

    @pytest.mark.parametrize("method", ["update_", "copy_"])
    def test_update__autograd(self, method, mode):
        # update_ and copy_ from a source that requires grad trace the copy_
        # of each leaf, which has a derivative, unlike _foreach_copy_ (#2863)
        def update_(td, x):
            getattr(td, method)(TensorDict(a=x * 2, b=torch.ones(3)))
            return td["a"].sum()

        update_c = torch.compile(update_, fullgraph=True, mode=mode)
        x = torch.randn(3, requires_grad=True)
        td = TensorDict(a=torch.zeros(3), b=torch.zeros(3))
        update_c(td, x).backward()
        assert (x.grad == 2).all()
        assert not td["b"].requires_grad

    def test_update_batch_size_prefix_future_warning(self, mode):
        # eager mode warns on this call (#2957); compiled code skips the warning
        def update(x):
            td = TensorDict(x=x.clone())
            td.update(TensorDict(x=x * 0, batch_size=[2]), update_batch_size=True)
            return td

        x = torch.ones(2, 3)
        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
            expected = update(x)
        update_c = torch.compile(update, fullgraph=True, mode=mode)
        result = update_c(x)
        assert result.batch_size == expected.batch_size == ()
        assert_close(result, expected)

    def test_compare_unsupported_operand_future_warning(self, mode):
        # eager mode warns on these calls (#2979); compiled code skips the warning
        def compare(x):
            td = TensorDict(a=x)
            return td ^ "x", td < None, x + 1

        x = torch.ones(3, dtype=torch.bool)
        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
            expected = compare(x)
        compare_c = torch.compile(compare, fullgraph=True, mode=mode)
        result = compare_c(x)
        assert result[:2] == expected[:2] == (True, False)
        torch.testing.assert_close(result[2], expected[2])

    @pytest.mark.parametrize("lazy", [False, True])
    def test_detach_no_grad(self, lazy, mode):
        # detach_() traces when no leaf requires grad: Dynamo cannot trace
        # Tensor.detach_() on a graph input, so those leaves are skipped (#2792)
        def detach_(td):
            return td.detach_()

        td = TensorDict(a=torch.randn(4, 3), b={"c": torch.randn(4, 2)}, batch_size=[4])
        if lazy:
            td = LazyStackedTensorDict(*[t.clone() for t in td.unbind(0)], stack_dim=0)
        expected = td.clone()
        detach_c = torch.compile(detach_, fullgraph=True, mode=mode)
        assert detach_c(td) is td
        assert_close(td, expected)

    @pytest.mark.parametrize("after_empty", ["tensor", "nested"])
    def test_tree_map_empty_nested_first(self, after_empty, mode):
        # The batch size of the result is not read from the empty nested td,
        # https://github.com/pytorch/tensordict/issues/2072
        def add_one(td):
            return tree_map(lambda x: x + 1, td)

        if after_empty == "tensor":
            other = torch.zeros(4, 3)
        else:
            other = TensorDict(x=torch.zeros(4, 2, 3), batch_size=[4, 2])
        td = TensorDict(empty=TensorDict(batch_size=[4]), other=other, batch_size=[4])
        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        out = add_one_c(td)
        assert out.batch_size == torch.Size([4])
        assert out["empty"].batch_size == torch.Size([4])
        assert out["other"].shape == other.shape
        assert (out == add_one(td)).all()

    def test_tree_map_non_tensor_first(self, mode):
        # The batch size of the result is not read from the non-tensor entry,
        # https://github.com/pytorch/tensordict/issues/2207
        def add_one(td):
            return tree_map(lambda x: x + 1, td)

        td = TensorDict(s="a string", a=torch.zeros(4, 3), batch_size=[4])
        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        out = add_one_c(td)
        assert out.batch_size == torch.Size([4])
        assert list(out.keys()) == ["s", "a"]
        assert out.get("s").batch_size == torch.Size([4])
        assert out.get_non_tensor("s") == "a string"
        assert (out["a"] == 1).all()

    # Memmap is currently not supported
    # def test_memmap(self, mode, tmpdir):
    #     def locked_op(td):
    #         # Adding stuff uses cache, check that this doesn't break
    #         return td.apply(lambda x: x+1)
    #
    #     td = TensorDict(
    #         {"a": torch.randn(1, 2, 3), "b": torch.ones(1, 2, 3, dtype=torch.int64)},
    #         batch_size=[1, 2],
    #         device="cpu",
    #     ).memmap_(tmpdir)
    #     locked_op_c = torch.compile(locked_op, fullgraph=True, mode=mode)
    #     td_op = locked_op(td)
    #     # no warning the second time this is run
    #     _ = locked_op_c(td)
    #     td_op_c = locked_op_c(td)
    #     assert (td_op == td_op_c).all()


class _TTDState(TypedTensorDict):
    a: torch.Tensor
    b: torch.Tensor


class _TTDOptionalState(TypedTensorDict):
    a: torch.Tensor
    b: torch.Tensor | None = None


@pytest.mark.parametrize("mode", [None, "reduce-overhead"])
class TestTTD:
    def test_tensor_output(self, mode):
        def add_one(td):
            return td["a"] + 1

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.zeros(3), b=torch.ones(3), batch_size=[3])
        assert (add_one(data) == 1).all()
        assert (add_one_c(data) == 1).all()
        assert (add_one_c(data + 1) == 2).all()

    def test_tensor_output_attr(self, mode):
        def add_one(td):
            return td.a + 1

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.zeros(3), b=torch.ones(3), batch_size=[3])
        assert (add_one(data) == 1).all()
        assert (add_one_c(data) == 1).all()

    def test_construct(self, mode):
        def fn(a, b):
            td = _TTDState(a=a, b=b, batch_size=[3])
            return td["a"] + td["b"]

        fn_c = torch.compile(fn, fullgraph=True, mode=mode)
        a = torch.randn(3)
        b = torch.randn(3)
        torch.testing.assert_close(fn(a, b), fn_c(a, b))

    def test_empty_batch_size(self, mode):
        # sets the batch size and names of a TypedTensorDict built in the graph
        def fn(td):
            return td.empty(batch_size=[3], names=["x"])

        fn_c = torch.compile(fn, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.zeros(3, 2), b=torch.ones(3, 2), batch_size=[3, 2])
        out = fn_c(data)
        assert type(out) is _TTDState
        assert out.batch_size == torch.Size([3])
        assert out.names == ["x"]
        assert out._source.batch_size == torch.Size([3])

    def test_optional_default(self, mode):
        def add_optional(td):
            if td.b is None:
                return td.a
            return td.a + td.b

        add_optional_c = torch.compile(add_optional, fullgraph=True, mode=mode)
        data = _TTDOptionalState(a=torch.ones(3), batch_size=[3])
        torch.testing.assert_close(add_optional(data), add_optional_c(data))

    def test_td_output(self, mode):
        def add_one(td):
            td["a"] = td["a"] + 1
            return td

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.zeros(3), b=torch.ones(3), batch_size=[3])
        assert add_one(data.clone())["a"].eq(1).all()
        assert add_one_c(data.clone())["a"].eq(1).all()
        assert add_one_c(data) is data

    @pytest.mark.parametrize("index_type", ["slice", "tensor", "int"])
    def test_index(self, index_type, mode):
        if index_type == "slice":

            def index_fn(td):
                return td[:2] + 1

        elif index_type == "tensor":

            def index_fn(td):
                return td[torch.tensor([0, 1])] + 1

        elif index_type == "int":

            def index_fn(td):
                return td[0] + 1

        index_fn_c = torch.compile(index_fn, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.arange(3), b=torch.arange(3), batch_size=[3])
        if index_type == "int":
            assert (index_fn_c(data)["a"] == 1).all()
            assert index_fn_c(data).shape == torch.Size([])
        else:
            assert (index_fn_c(data)["a"] == torch.arange(1, 3)).all()
            assert index_fn_c(data).shape == torch.Size([2])

    def test_stack(self, mode):
        def stack_tds(td0, td1):
            return torch.stack([td0, td1])

        stack_tds_c = torch.compile(stack_tds, fullgraph=True, mode=mode)
        d0 = _TTDState(a=torch.arange(3), b=torch.arange(3), batch_size=[3])
        d1 = _TTDState(a=torch.arange(3), b=torch.arange(3), batch_size=[3])
        assert (stack_tds(d0, d1) == stack_tds_c(d0, d1)).all()

    def test_cat(self, mode):
        def cat_tds(td0, td1):
            return torch.cat([td0, td1])

        cat_tds_c = torch.compile(cat_tds, fullgraph=True, mode=mode)
        d0 = _TTDState(a=torch.arange(3), b=torch.arange(3), batch_size=[3])
        d1 = _TTDState(a=torch.arange(3), b=torch.arange(3), batch_size=[3])
        assert (cat_tds(d0, d1) == cat_tds_c(d0, d1)).all()

    @pytest.mark.parametrize("mutate", [False, True])
    def test_reshape(self, mode, mutate):
        def reshape(td):
            if mutate:
                td["a"] = td["a"] + 1
            return td.reshape(2, 2)

        reshape_c = torch.compile(reshape, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.arange(4), b=torch.arange(4), batch_size=[4])
        data_reshape = reshape(data.clone())
        _ = reshape_c(data.clone())
        data_reshape_c = reshape_c(data.clone())
        assert isinstance(data_reshape_c, _TTDState)
        assert (data_reshape == data_reshape_c).all()

    def test_view(self, mode):
        def view(td):
            return td.view(2, 2).clear_refs_for_compile_()

        view_c = torch.compile(view, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.arange(4), b=torch.arange(4), batch_size=[4])
        data_view = view(data)
        _ = view_c(data)
        data_view_c = view_c(data)
        assert (data_view == data_view_c).all()

    def test_transpose(self, mode):
        def transpose(td):
            return td.transpose(0, 1).clear_refs_for_compile_()

        transpose_c = torch.compile(transpose, fullgraph=True, mode=mode)
        data = _TTDState(
            a=torch.arange(6).view(2, 3),
            b=torch.arange(6).view(2, 3),
            batch_size=[2, 3],
        )
        data_t = transpose(data)
        _ = transpose_c(data)
        data_t_c = transpose_c(data)
        assert (data_t == data_t_c).all()

    def test_unbind(self, mode):
        def unbind(td):
            return td.unbind(0)

        unbind_c = torch.compile(unbind, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.arange(4), b=torch.arange(4), batch_size=[4])
        assert (unbind(data)[-1] == unbind_c(data)[-1]).all()

    def test_items(self, mode):
        def items(td):
            keys, vals = zip(*td.items(True, True))
            return keys, vals

        items_c = torch.compile(items, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.arange(4), b=torch.arange(4), batch_size=[4])
        keys, vals = items(data)
        keys_c, vals_c = items_c(data)
        assert keys == keys_c

        def assert_eq(x, y):
            assert (x == y).all()

        torch.utils._pytree.tree_map(assert_eq, vals, vals_c)

    @pytest.mark.parametrize("recurse", [True, False])
    @pytest.mark.parametrize("lock", [True, False])
    def test_clone(self, recurse, lock, mode):
        def clone(td):
            return td.clone(recurse=recurse)

        clone_c = torch.compile(clone, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.zeros(3), b=torch.ones(3), batch_size=[3])
        if lock:
            data = data.lock_()
        data_c = clone(data)
        _ = clone_c(data)
        data_c_c = clone_c(data)
        assert_close(data_c, data_c_c)
        assert clone_c(data) is not data
        if recurse:
            assert clone_c(data)["a"] is not data["a"]
        else:
            assert clone_c(data)["a"] is data["a"]

    def test_flatten_keys(self, mode):
        def flatten_keys(td):
            return td.flatten_keys().clear_refs_for_compile_()

        flatten_keys_c = torch.compile(flatten_keys, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.zeros(3), b=torch.ones(3), batch_size=[3])
        data_f = flatten_keys(data)
        _ = flatten_keys_c(data)
        data_f_c = flatten_keys_c(data)
        assert_close(data_f, data_f_c)

    def test_pop(self, mode):
        def pop_a(td):
            return td.pop("a")

        pop_a_c = torch.compile(pop_a, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.tensor(1), b=torch.tensor(2), batch_size=[])
        result = pop_a(data.clone())
        assert result == 1
        result_c = pop_a_c(data.clone())
        assert result_c == 1

    def test_select(self, mode):
        def select_a(td):
            return td.select("a", strict=False)

        select_a_c = torch.compile(select_a, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.tensor(1), b=torch.tensor(2), batch_size=[])
        result_c = select_a_c(data)
        assert "a" in result_c.keys()
        assert "b" not in result_c.keys()

    def test_exclude(self, mode):
        def exclude_b(td):
            return td.exclude("b")

        exclude_b_c = torch.compile(exclude_b, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.tensor(1), b=torch.tensor(2), batch_size=[])
        result_c = exclude_b_c(data)
        assert "a" in result_c.keys()
        assert "b" not in result_c.keys()

    def test_all_any(self, mode):
        def call_all(td):
            return td.all(dim=0)

        def call_any(td):
            return td.any(dim=0)

        call_all_c = torch.compile(call_all, fullgraph=True, mode=mode)
        call_any_c = torch.compile(call_any, fullgraph=True, mode=mode)
        data = _TTDState(
            a=torch.tensor([[True, False], [True, True]]),
            b=torch.tensor([[False, False], [True, True]]),
            batch_size=[2, 2],
        )
        result_all = call_all(data)
        result_all_c = call_all_c(data)
        assert (result_all["a"] == result_all_c["a"]).all()

        result_any = call_any(data)
        result_any_c = call_any_c(data)
        assert (result_any["a"] == result_any_c["a"]).all()

    def test_squeeze_unsqueeze(self, mode):
        def call_squeeze(td):
            return td.squeeze(0)

        def call_unsqueeze(td):
            return td.unsqueeze(0)

        call_squeeze_c = torch.compile(call_squeeze, fullgraph=True, mode=mode)
        call_unsqueeze_c = torch.compile(call_unsqueeze, fullgraph=True, mode=mode)
        data = _TTDState(a=torch.randn(1, 3), b=torch.randn(1, 3), batch_size=[1, 3])
        result_squeeze_c = call_squeeze_c(data)
        assert result_squeeze_c.shape == torch.Size([3])
        result_unsqueeze_c = call_unsqueeze_c(result_squeeze_c)
        assert result_unsqueeze_c.shape == torch.Size([1, 3])

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="cuda required to test device casting"
    )
    @pytest.mark.parametrize("has_device", [True, False])
    def test_to(self, has_device, mode):
        device = f"{cur_device}:0"

        def test_to_device(td):
            return td.to(device)

        data = _TTDState(
            a=torch.randn(3),
            b=torch.randn(3),
            batch_size=[3],
            device="cpu" if has_device else None,
        )
        test_to_device_c = torch.compile(test_to_device, fullgraph=True, mode=mode)
        _ = test_to_device_c(data)
        td_device_c = test_to_device_c(data)
        assert td_device_c.batch_size == data.batch_size
        assert td_device_c.device == torch.device(device)

    @pytest.mark.skipif(
        is_npu_available(),
        reason="torch.device in torch.compile is not supported on NPU currently.",
    )
    def test_lock(self, mode):
        def locked_op(td):
            td2 = td + 1
            td3 = td + td2
            return td3.clear_refs_for_compile_()

        data = _TTDState(
            a=torch.randn(3),
            b=torch.randn(3),
            batch_size=[3],
            lock=True,
        )
        locked_op_c = torch.compile(locked_op, fullgraph=True, mode=mode)
        td_op = locked_op(data)
        with (
            pytest.warns(UserWarning, match="Using lock_")
            if mode is None
            else contextlib.nullcontext()
        ):
            _ = locked_op_c(data)
        td_op_c = locked_op_c(data)
        assert (td_op == td_op_c).all()

    def test_arithmetic(self, mode):
        def add_one(td):
            return td + 1

        data = _TTDState(a=torch.zeros(3), b=torch.ones(3), batch_size=[3])
        eager = add_one(data.clone())
        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        compiled = add_one_c(data.clone())
        assert (eager["a"] == compiled["a"]).all()
        assert (eager["b"] == compiled["b"]).all()

    # #2787: Dynamo reads the name of a function from its code from torch 2.16
    # on, so the name that functools.wraps gives add_ and mul_ was lost.
    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.16.0"),
        reason="Dynamo gives the name that functools.wraps sets before torch 2.16",
    )
    @pytest.mark.parametrize("op", ["add_", "mul_"])
    def test_inplace_arithmetic_lazy_stack(self, op, mode):
        def inplace(td):
            return getattr(td, op)(2)

        def make():
            return _TTDState.from_tensordict(
                lazy_stack(
                    [
                        TensorDict(a=torch.ones(3), b=torch.ones(3), batch_size=[3])
                        for _ in range(2)
                    ]
                )
            )

        inplace_c = torch.compile(inplace, fullgraph=True, mode=mode)
        data = make()
        result = inplace_c(data)
        assert result is data
        assert (data == inplace(make())).all()

    def test_arithmetic_self(self, mode):
        def add_self(td):
            return td + td

        data = _TTDState(a=torch.ones(3), b=torch.ones(3), batch_size=[3])
        eager = add_self(data.clone())
        add_self_c = torch.compile(add_self, fullgraph=True, mode=mode)
        compiled = add_self_c(data.clone())
        assert (eager["a"] == compiled["a"]).all()
        assert (eager["b"] == compiled["b"]).all()


class TestTTDDynamoCompatibility:
    """Tests that probe known Dynamo limitations we work around.

    Tests marked ``xfail(strict=True)`` use the ideal code path that we would
    prefer, but that is not yet traceable by every supported torch version.
    When an xfail starts passing in the oldest supported torch version, the
    corresponding workaround can be simplified.
    """

    @pytest.mark.xfail(
        TORCH_VERSION < version.parse("2.12.0"),
        strict=True,
        reason=(
            "The Dynamo of torch 2.11 cannot trace frozenset - dict.keys(): "
            "TypedTensorDict.__init__ uses frozenset.difference(dict) instead."
        ),
    )
    def test_frozenset_sub_dict_keys(self):
        """frozenset - dict.keys() should be traceable by Dynamo."""
        required = frozenset({"a", "b"})

        def fn(**kwargs):
            missing = required - kwargs.keys()
            if missing:
                raise TypeError(f"missing: {missing}")
            return kwargs["a"] + kwargs["b"]

        fn_c = torch.compile(fn, fullgraph=True)
        result = fn_c(a=torch.tensor(1.0), b=torch.tensor(2.0))
        assert result == 3.0

    @pytest.mark.xfail(
        TORCH_VERSION < version.parse("2.14.0"),
        strict=True,
        reason=(
            "TensorDict subclass arithmetic hits _has_mps graph break "
            "without explicit pytree registration before torch 2.14."
        ),
    )
    def test_td_subclass_arithmetic_without_registration(self):
        """A plain TensorDict subclass should work under compile without pytree registration."""

        class BareSubclass(TensorDict):
            pass

        # deliberately do NOT register:
        # _register_tensor_class(BareSubclass)
        # _register_td_node(BareSubclass)

        def add_one(td):
            return td + 1

        # device="cpu" is required to trigger the _has_mps graph break
        # (without a device, empty() takes a different path that avoids it)
        data = BareSubclass({"a": torch.zeros(3)}, batch_size=[3], device="cpu")
        add_one_c = torch.compile(add_one, fullgraph=True)
        result = add_one_c(data)
        assert (result["a"] == 1).all()


@tensorclass
class MyClass:
    a: "MyClass"
    b: Any = None
    c: Any = None


@tensorclass(cast_filter=is_inexact)
class _CastFilterTC:
    points: torch.Tensor
    cells: torch.Tensor


@pytest.mark.parametrize("mode", [None, "reduce-overhead"])
class TestTC:
    def test_tc_to_dtype_cast_filter(self, mode):
        def to_bf16(tc):
            return tc.to(torch.bfloat16)

        to_bf16_c = torch.compile(to_bf16, fullgraph=True, mode=mode)
        for batch in (2, 5):
            data = _CastFilterTC(
                points=torch.randn(batch, 3),
                cells=torch.zeros(batch, 3, dtype=torch.long),
                batch_size=[batch],
            )
            out = to_bf16_c(data)
            assert isinstance(out, _CastFilterTC)
            assert out.points.dtype == torch.bfloat16
            assert out.cells.dtype == torch.long
            assert_close(out, to_bf16(data))

    def test_tc_tensor_output(self, mode):
        def add_one(td):
            return td.a.b + 1

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = MyClass(MyClass(a=None, b=torch.zeros(())))
        assert add_one(data) == 1
        assert add_one_c(data) == 1
        assert add_one_c(data + 1) == 2

    def test_tc_items(self, mode):
        def items(td):
            keys, vals = zip(*td.items(True, True))
            return keys, vals

        items_c = torch.compile(items, fullgraph=True, mode=mode)
        data = MyClass(MyClass(a=None, b=torch.zeros(())))
        keys, vals = items(data)
        keys_c, vals_c = items_c(data)

        def assert_eq(x, y):
            assert (x == y).all()

        assert keys == keys_c
        torch.utils._pytree.tree_map(assert_eq, vals, vals_c)

    def test_tc_output(self, mode):
        def add_one(td):
            td.a.c = td.a.b + 1
            return td

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = MyClass(a=MyClass(a=None, b=torch.zeros(())))
        assert add_one(data.clone()).a.c == 1
        assert add_one_c(data.clone()).a.c == 1
        assert add_one_c(data) is data

    def test_tc_tensor_only_assign_none(self, mode):
        class TensorOnly(TensorClass["tensor_only"]):
            x: torch.Tensor
            y: torch.Tensor | None = None

        def clear(tc):
            tc.y = None
            return tc.x + 1

        clear_c = torch.compile(clear, fullgraph=True, mode=mode)
        data = TensorOnly(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
        assert (clear_c(data) == 1).all()
        assert data.y is None

    def test_tc_tensor_only_del_none(self, mode):
        class TensorOnly(TensorClass["tensor_only"]):
            x: torch.Tensor
            y: torch.Tensor | None = None

        def clear(tc):
            tc.del_("y")
            return tc.x + 1

        def read(tc):
            if tc.y is None:
                return tc.x
            return tc.x + tc.y

        clear_c = torch.compile(clear, fullgraph=True, mode=mode)
        read_c = torch.compile(read, fullgraph=True, mode=mode)
        data = TensorOnly(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
        assert (clear_c(data) == 1).all()
        assert data.y is None
        assert (read_c(data) == 0).all()
        # The compiled read sees the field once it holds a tensor again
        data.y = torch.full((3,), 2.0)
        assert (read_c(data) == 2).all()

    @pytest.mark.parametrize("shadow", [False, True])
    def test_tc_deleted_field_read(self, mode, shadow):
        # #2147: a compiled read of a deleted field gives None, and sees the
        # field again once it holds a tensor
        @tensorclass(shadow=shadow)
        class Regular:
            x: torch.Tensor
            y: torch.Tensor = None

        def read(tc):
            if tc.y is None:
                return tc.x
            return tc.x + tc.y

        read_c = torch.compile(read, fullgraph=True, mode=mode)
        data = Regular(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
        del data.y
        assert (read_c(data) == 0).all()
        data.y = torch.full((3,), 2.0)
        assert (read_c(data) == 2).all()

    @pytest.mark.parametrize("form", ["bracket", "decorator"])
    def test_tc_tensor_only_init(self, mode, form):
        # The generated __init__ of the bracket form sets _tensordict
        # through the __setattr__ of its tensor_only parent class.
        if form == "bracket":

            class TensorOnly(TensorClass["tensor_only"]):
                x: torch.Tensor

        else:

            @tensorclass(tensor_only=True)
            class TensorOnly:
                x: torch.Tensor

        def build(x):
            data = TensorOnly(x=x, batch_size=[3])
            return data, data.x + 1

        build_c = torch.compile(build, fullgraph=True, mode=mode)
        data, out = build_c(torch.zeros(3))
        assert isinstance(data, TensorOnly)
        assert data.batch_size == torch.Size([3])
        assert (data.x == 0).all()
        assert (out == 1).all()

    def test_tc_arithmetic(self, mode):
        def add_one(td):
            return td + 1

        data = MyClass(a=MyClass(a=None, b=torch.zeros(())))

        eager = add_one(data.clone())

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        compiled = add_one_c(data.clone())

        assert isinstance(eager.a, MyClass)
        assert eager.a.b == 1

        assert isinstance(compiled.a, MyClass)
        # TODO: breaks because a is not cast to a MyClass but is a dict
        assert compiled.a.b == 1
        assert add_one_c(data) is not data

    def test_tc_arithmetic_other_tc(self, mode):
        def add_self(td):
            return td + td

        data = MyClass(a=MyClass(a=None, b=torch.ones(())))

        eager = add_self(data.clone())

        add_self_c = torch.compile(add_self, fullgraph=True, mode=mode)
        compiled = add_self_c(data.clone())

        assert isinstance(eager.a, MyClass)
        assert eager.a.b == 2

        assert isinstance(compiled.a, MyClass)
        # TODO: breaks because a is not cast to a MyClass but is a dict
        assert compiled.a.b == 2
        assert add_self_c(data) is not data

    @pytest.mark.parametrize("locked", [False, True])
    def test_tc_from_tensordict_nested(self, mode, locked):
        def from_td(td):
            return MyClass.from_tensordict(td)

        from_td_c = torch.compile(from_td, fullgraph=True, mode=mode)
        td = MyClass(
            a=MyClass(a=MyClass(a=None, b=torch.zeros(())), b=torch.zeros(())),
            b=torch.ones(()),
        ).to_tensordict()
        if locked:
            td.lock_()
        compiled = from_td_c(td)
        assert isinstance(compiled.a, MyClass)
        assert isinstance(compiled.a.a, MyClass)
        assert compiled.a.b == 0
        assert compiled.is_locked is locked
        assert isinstance(td["a"], TensorDict)

    @pytest.mark.parametrize("index_type", ["slice", "tensor", "int"])
    def test_tc_index(self, index_type, mode):
        if index_type == "slice":

            def index(td):
                return td[:2]

        elif index_type == "tensor":

            def index(td):
                return td[torch.tensor([0, 1])]

        elif index_type == "int":

            def index(td):
                return td[0]

        index_c = torch.compile(index, fullgraph=True, mode=mode)
        data = MyClass(
            a=MyClass(a=None, b=torch.arange(3), batch_size=[3]), batch_size=[3]
        )

        indexed_data_eager = index(data)
        indexed_data_compile = index_c(data)
        if index_type == "int":
            assert (indexed_data_eager.a.b == 0).all()
            assert (indexed_data_compile.a.b == 0).all()

            assert isinstance(indexed_data_eager, MyClass)
            assert isinstance(indexed_data_compile, MyClass)

            assert isinstance(indexed_data_eager.a, MyClass)
            assert isinstance(indexed_data_compile.a, MyClass)

            assert indexed_data_eager.shape == torch.Size([])
            assert indexed_data_compile.shape == torch.Size([])

        else:
            assert (indexed_data_eager.a.b == torch.arange(0, 2)).all()
            assert (indexed_data_compile.a.b == torch.arange(0, 2)).all()
            assert isinstance(indexed_data_eager, MyClass)
            assert isinstance(indexed_data_compile, MyClass)
            assert isinstance(indexed_data_eager.a, MyClass)
            assert isinstance(indexed_data_compile.a, MyClass)
            assert indexed_data_eager.shape == torch.Size([2])
            assert indexed_data_compile.shape == torch.Size([2])

    def test_tc_stack(self, mode):
        def stack_tds(td0, td1):
            # return TensorDict.stack([td0, td1])
            return torch.stack([td0, td1])

        data0 = MyClass(
            a=MyClass(a=None, b=torch.arange(3), batch_size=[3]), batch_size=[3]
        )
        data1 = MyClass(
            a=MyClass(a=None, b=torch.arange(3, 6), batch_size=[3]), batch_size=[3]
        )
        stack_eager = stack_tds(data0, data1)

        stack_tds_c = torch.compile(stack_tds, fullgraph=True, mode=mode)
        stack_compile = stack_tds_c(data0, data1)

        assert (stack_eager == stack_compile).all()

    def test_tc_stack_names(self, mode):
        # TensorDict.__init__ used to skip the names under compile, with the
        # comment "this breaks when stacking tensorclasses with dynamo".
        def stack_named(b):
            inner = MyClass(a=None, b=b, batch_size=[3], names=["n"])
            data = MyClass(a=inner, batch_size=[3], names=["n"])
            return data, torch.stack([data, data.clone()])

        def stack_inputs(data0, data1):
            return torch.stack([data0, data1])

        stack_named_c = torch.compile(stack_named, fullgraph=True, mode=mode)
        data, stacked = stack_named_c(torch.arange(3))
        assert data.names == ["n"]
        assert data.a.names == ["n"]
        assert stacked.names == [None, "n"]
        assert stacked.a.names == [None, "n"]
        assert (stacked.a.b == torch.arange(3).expand(2, 3)).all()

        stack_inputs_c = torch.compile(stack_inputs, fullgraph=True, mode=mode)
        stacked = stack_inputs_c(data, data.clone())
        assert stacked.names == [None, "n"]
        assert stacked.a.names == [None, "n"]

    def test_tc_cat(self, mode):
        def cat_tds(td0, td1):
            return torch.cat([td0, td1])

        cat_tds_c = torch.compile(cat_tds, fullgraph=True, mode=mode)
        data0 = MyClass(
            a=MyClass(a=None, b=torch.arange(3), batch_size=[3]), batch_size=[3]
        )
        data1 = MyClass(
            a=MyClass(a=None, b=torch.arange(3, 6), batch_size=[3]), batch_size=[3]
        )
        assert (cat_tds(data0, data1) == cat_tds_c(data0, data1)).all()

    def test_tc_reshape(self, mode):
        def reshape(td):
            return td.reshape(2, 2)

        reshape_c = torch.compile(reshape, fullgraph=True, mode=mode)
        data = MyClass(
            a=MyClass(a=None, b=torch.arange(4), batch_size=[4]), batch_size=[4]
        )
        assert (reshape(data) == reshape_c(data)).all()

    def test_tc_view_cm_unset_field(self, mode):
        # Setting the unset field c inside the context drops its None placeholder.
        def set_in_view(tc):
            with tc.view(-1) as flat:
                flat.c = flat.b + 1
            return tc.c

        set_in_view_c = torch.compile(set_in_view, fullgraph=True, mode=mode)
        data = MyClass(a=None, b=torch.zeros(2, 2), batch_size=[2, 2])
        torch.testing.assert_close(set_in_view_c(data), torch.ones(2, 2))
        torch.testing.assert_close(data.to_dict()["c"], torch.ones(2, 2))

    def test_tc_torch_where(self, mode):
        def where(mask, td0, td1):
            return torch.reshape(torch.where(mask, td0, td1), (2, 2))

        where_c = torch.compile(where, fullgraph=True, mode=mode)
        mask = torch.tensor([True, False, True, False])
        data0 = MyClass(
            a=MyClass(a=None, b=torch.arange(4), batch_size=[4]), batch_size=[4]
        )
        data1 = MyClass(
            a=MyClass(a=None, b=torch.arange(4, 8), batch_size=[4]), batch_size=[4]
        )
        result_c = where_c(mask, data0, data1)
        assert isinstance(result_c, MyClass)
        assert result_c.batch_size == (2, 2)
        assert (where(mask, data0, data1) == result_c).all()

    def test_tc_get_defaults_to_none(self, mode):
        # The AttributeError is caught in the compiled function: with
        # fullgraph=True, dynamo reports an uncaught one as Unsupported.
        def get_missing(td):
            try:
                return td.get("missing")
            except AttributeError:
                return "AttributeError"

        get_missing_c = torch.compile(get_missing, fullgraph=True, mode=mode)
        data = MyClass(a=None, b=torch.zeros(()))
        set_back = _get_defaults_to_none()
        try:
            _set_get_defaults_to_none(True)
            assert get_missing_c(data) is None
            _set_get_defaults_to_none(False)
            assert get_missing_c(data) == "AttributeError"
        finally:
            _set_get_defaults_to_none(set_back)

    def test_tc_get_through_tensor_field(self, mode):
        # eager get warns that this raises from 0.17 (#2979); compiled, it
        # returns None as on main, since dynamo cannot trace the warning
        def get(td):
            return td.get(("b", "y"))

        get_c = torch.compile(get, fullgraph=True, mode=mode)
        data = MyClass(a=None, b=torch.zeros(3))
        set_back = _get_defaults_to_none()
        try:
            _set_get_defaults_to_none(True)
            with pytest.warns(FutureWarning, match="TensorDict 0.17"):
                assert get(data) is None
            assert get_c(data) is None
        finally:
            _set_get_defaults_to_none(set_back)

    def test_tc_unbind(self, mode):
        def unbind(td):
            return td.unbind(0)

        unbind_c = torch.compile(unbind, fullgraph=True, mode=mode)
        data = MyClass(
            a=MyClass(a=None, b=torch.arange(4), batch_size=[4]), batch_size=[4]
        )
        assert (unbind(data)[-1] == unbind_c(data)[-1]).all()

    def test_tc_max(self, mode):
        def max_values(td):
            result = td.max(dim=0)
            return result.values, result.indices

        max_values_c = torch.compile(max_values, fullgraph=True, mode=mode)
        data = MyClass(a=None, b=torch.randn(3, 4), batch_size=[3, 4])
        values, indices = max_values_c(data)
        expected = data.b.max(dim=0)
        assert isinstance(values, MyClass)
        assert isinstance(indices, MyClass)
        assert (values.b == expected.values).all()
        assert (indices.b == expected.indices).all()

    def test_tc_iter(self, mode):
        def iterate(tc):
            return torch.stack([t.a.b + 1 for t in tc])

        iterate_c = torch.compile(iterate, fullgraph=True, mode=mode)
        data = MyClass(
            a=MyClass(a=None, b=torch.arange(4), batch_size=[4]), batch_size=[4]
        )
        assert (iterate(data) == iterate_c(data)).all()

    @pytest.mark.parametrize("recurse", [True, False])
    def test_tc_clone(self, recurse, mode):
        def clone(td: TensorDict):
            return td.clone(recurse=recurse)

        clone_c = torch.compile(clone, fullgraph=True, mode=mode)
        data = MyClass(
            a=MyClass(a=None, b=torch.arange(4), batch_size=[4]), batch_size=[4]
        )
        assert_close(clone_c(data), clone(data))
        assert clone_c(data) is not data
        if recurse:
            assert clone_c(data).a.b is not data.a.b
        else:
            assert clone_c(data).a.b is data.a.b

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="cuda required to test device casting"
    )
    @pytest.mark.parametrize("has_device", [True, False])
    def test_tc_to(self, has_device, mode):
        device = f"{cur_device}:0"

        def test_to_device(tc):
            return tc.to(device)

        data = MyClass(
            a=MyClass(a=None, b=torch.arange(4), batch_size=[4]),
            batch_size=[4],
            device="cpu" if has_device else None,
        )
        test_to_device_c = torch.compile(test_to_device, fullgraph=True, mode=mode)
        # tc_device = test_to_device(tc)
        _ = test_to_device_c(data)
        tc_device_c = test_to_device_c(data)
        assert tc_device_c.batch_size == data.batch_size
        assert tc_device_c.device == torch.device(device)

    @pytest.mark.skipif(
        is_npu_available(),
        reason="torch.device in torch.compile is not supported on NPU currently.",
    )
    def test_tc_lock(self, mode):
        def locked_op(tc):
            # Adding stuff uses cache, check that this doesn't break
            tc2 = tc + 1
            tc3 = tc + tc2
            return tc3

        data = MyClass(
            a=MyClass(a=None, b=torch.arange(4), batch_size=[4]),
            batch_size=[4],
            device="cpu",
        ).lock_()
        locked_op_c = torch.compile(locked_op, fullgraph=True, mode=mode)
        tc_op = locked_op(data)
        # no warning the second time this is run
        with (
            pytest.warns(UserWarning, match="Using lock_")
            if mode is None
            else contextlib.nullcontext()
        ):
            _ = locked_op_c(data)
        tc_op_c = locked_op_c(data)
        assert (tc_op == tc_op_c).all()

    def test_tc_shadow_clone(self, mode):
        """Shadow-mode TensorClass clone should not cause graph breaks (gh-1547)."""

        class ShadowState(TensorClass["shadow"]):
            x: torch.Tensor
            v: torch.Tensor

        def step(s):
            clone = s.clone(recurse=False)
            clone.x = s.x + s.v
            return clone

        s = ShadowState(x=torch.randn(3), v=torch.randn(3), batch_size=[])
        step_c = torch.compile(step, fullgraph=True, mode=mode)
        eager_result = step(s)
        compiled_result = step_c(s)
        assert_close(eager_result, compiled_result)
        assert compiled_result is not s
        # v should be preserved (shallow clone)
        assert compiled_result.v is s.v

    def test_tc_shadow_replace(self, mode):
        """Shadow-mode TensorClass replace should not cause graph breaks (gh-1547)."""

        class ShadowState(TensorClass["shadow"]):
            x: torch.Tensor
            v: torch.Tensor

        def step(s):
            return s.replace(x=s.x + s.v)

        s = ShadowState(x=torch.randn(3), v=torch.randn(3), batch_size=[])
        step_c = torch.compile(step, fullgraph=True, mode=mode)
        eager_result = step(s)
        compiled_result = step_c(s)
        assert_close(eager_result, compiled_result)

    def test_td_replace_no_recompile(self, mode):
        """replace() with many distinct kwarg patterns must not recompile."""

        class State(TensorClass["nocast"]):
            x: torch.Tensor
            y: torch.Tensor
            z: torch.Tensor
            w: torch.Tensor
            v: torch.Tensor

        def step(s: State) -> State:
            s = s.replace(x=s.x + 1)
            s = s.replace(y=s.y + 2)
            s = s.replace(z=s.z + 3)
            s = s.replace(x=s.x * 0.9, y=s.y * 0.9)
            s = s.replace(w=s.w + s.x)
            s = s.replace(v=s.v - 1, w=s.w + 1)
            s = s.replace(x=s.x + s.v, y=s.y + s.w, z=s.z + 0.1)
            s = s.replace(v=torch.zeros_like(s.v))
            s = s.replace(w=torch.ones_like(s.w))
            s = s.replace(x=s.x + s.y + s.z)
            return s

        s = State(
            x=torch.randn(4),
            y=torch.randn(4),
            z=torch.randn(4),
            w=torch.randn(4),
            v=torch.randn(4),
            batch_size=[4],
        )
        step_c = torch.compile(step, fullgraph=True, mode=mode)
        eager_result = step(s)
        compiled_result = step_c(s)
        assert_close(eager_result, compiled_result)

    @pytest.mark.xfail(
        reason="Dynamo cannot symbolically trace TensorClass._tensordict "
        "access inside while_loop's pytree flatten (gh-1547). "
        "Requires Dynamo-side support for custom pytree nodes in "
        "higher-order ops.",
    )
    def test_tc_while_loop(self, mode):
        """TensorClass as carry in while_loop should not crash (gh-1547)."""
        from torch._higher_order_ops.while_loop import while_loop

        @tensorclass
        class CarryState:
            val: torch.Tensor
            count: torch.Tensor

        def cond(state):
            return state.count < 5

        def body(state):
            return (
                CarryState(
                    val=state.val + 1,
                    count=state.count + 1,
                    batch_size=[],
                ),
            )

        init = CarryState(val=torch.tensor(0.0), count=torch.tensor(0), batch_size=[])

        def fn():
            return while_loop(cond, body, (init,))

        fn_c = torch.compile(fn, mode=mode)
        (result,) = fn_c()
        assert result.val.item() == 5.0
        assert result.count.item() == 5

    def test_td_new_unsafe(self, mode):

        class MyTd(TensorDict):
            pass

        def func_td():
            return TensorDict._new_unsafe(a=torch.randn(3), batch_size=torch.Size(()))

        @torch.compile(fullgraph=True, mode=mode)
        def func_c_td():
            return TensorDict._new_unsafe(a=torch.randn(3), batch_size=torch.Size(()))

        def func_mytd():
            return MyTd._new_unsafe(a=torch.randn(3), batch_size=torch.Size(()))

        # This will graph break
        @torch.compile(mode=mode)
        def func_c_mytd():
            return MyTd._new_unsafe(a=torch.randn(3), batch_size=torch.Size(()))

        assert type(func_td()) is type(func_c_td())
        assert type(func_mytd()) is type(func_c_mytd())


@set_composite_lp_aggregate(False)
def test_composite_custom_log_prob_keys():
    module = Prob(
        in_keys=["params"],
        distribution_class=CompositeDistribution,
        distribution_kwargs={
            "distribution_map": {"action": torch.distributions.Normal}
        },
        default_interaction_type=InteractionType.MODE,
        return_log_prob=True,
        log_prob_keys=["lp"],
    )
    loc, scale = torch.zeros(2), torch.ones(2)
    td = TensorDict(
        params={"action": {"loc": loc, "scale": scale}},
        lp=torch.full((2,), 123.0),
        batch_size=[2],
    )
    result = torch.compile(module, fullgraph=True, backend="eager")(td)
    expected = torch.distributions.Normal(loc, scale).log_prob(result["action"])
    torch.testing.assert_close(result["lp"], expected)
    assert "action_log_prob" not in result


@pytest.mark.parametrize("fullgraph", [True, False])
def test_composite_seq_log_prob_aggregate_changed_after_init(fullgraph):
    # Modules built while composite_lp_aggregate() was True keep the default
    # names when the sequence is scored with it False.
    with set_composite_lp_aggregate(True):
        seq = ProbSeq(
            Prob(
                in_keys=["params"],
                out_keys=["a", "b"],
                distribution_class=CompositeDistribution,
                distribution_kwargs={
                    "distribution_map": {
                        "a": torch.distributions.Normal,
                        "b": torch.distributions.Normal,
                    }
                },
                default_interaction_type=InteractionType.MODE,
                return_log_prob=True,
            ),
            return_composite=True,
        )
        params = TensorDict(
            {k: {"loc": torch.zeros(2), "scale": torch.ones(2)} for k in "ab"},
            batch_size=[2],
        )
        td = seq(TensorDict(params=params, batch_size=[2]))
    with set_composite_lp_aggregate(False):
        expected = seq.log_prob(td)
        result = torch.compile(
            lambda t: seq.log_prob(t), fullgraph=fullgraph, backend="eager"
        )(td)
    assert set(result.keys()) == set(expected.keys()) == {"a_log_prob", "b_log_prob"}
    for key in expected.keys():
        torch.testing.assert_close(result[key], expected[key])


@pytest.mark.parametrize("output_type", [dict, TensorDict])
@pytest.mark.parametrize("selected", [[0], [1, 0]])
def test_tdmodule_dict_output_selected(output_type, selected):
    out_keys = [("result", "b"), ("result", "c")]

    def func(x):
        return output_type({out_keys[0]: x + 1, out_keys[1]: x * 10})

    module = Mod(func, in_keys=["x"], out_keys=out_keys)
    selected = [out_keys[index] for index in selected]
    module.select_out_keys(*selected)
    compiled = torch.compile(module, fullgraph=True, backend="eager")
    x = torch.arange(3.0)
    result = compiled(TensorDict(x=x, batch_size=[3]))
    assert set(result.keys(True, True)) == {"x", *selected}
    expected = {out_keys[0]: x + 1, out_keys[1]: x * 10}
    for key in selected:
        torch.testing.assert_close(result[key], expected[key])


@pytest.mark.parametrize("mode", [None, "reduce-overhead"])
class TestNN:
    def test_func(self, mode):
        td = TensorDict({"a": 0})
        module = Mod(
            lambda x: x + 1, in_keys=[(((("a",),),),)], out_keys=[(((("a",),),),)]
        )
        module_compile = torch.compile(module, fullgraph=True, mode=mode)
        module_compile(td)
        assert_close(module(td), module_compile(td))

    def test_linear(self, mode):
        net = torch.nn.Linear(4, 5)
        module = Mod(net, in_keys=[(((("a",),),),)], out_keys=[("c", "d")])
        module_compile = torch.compile(module, fullgraph=True, mode=mode)
        td = TensorDict({"a": torch.randn(32, 4)}, [32])
        assert_close(module(td), module_compile(td))

    def test_seq(self, mode):
        net0 = torch.nn.Linear(4, 5)
        module0 = Mod(net0, in_keys=["a"], out_keys=["hidden"])
        net1 = torch.nn.Linear(5, 6)
        module1 = Mod(net1, in_keys=["hidden"], out_keys=[("c", "d")])
        module = Seq(module0, module1)
        module_compile = torch.compile(module, fullgraph=True, mode=mode)
        td = TensorDict({"a": torch.randn(32, 4)}, [32])
        assert_close(module(td), module_compile(td))

        assert module_compile(td) is td

    def test_named_sequence_shared_module(self, mode):
        key = ("nested", "x")
        shared = Mod(lambda x: x + 1, in_keys=[key], out_keys=[key])
        module = Seq(
            {
                "first": shared,
                "middle": Mod(lambda x: x * 10, in_keys=[key], out_keys=[key]),
                "last": shared,
            }
        )
        compiled = torch.compile(module, fullgraph=True, mode=mode)
        result = compiled(TensorDict({key: torch.zeros(3)}, batch_size=[3]))
        torch.testing.assert_close(result[key], torch.full((3,), 11.0))

    def test_seq_lmbda(self, mode):
        net0 = torch.nn.Linear(4, 5)
        module0 = Mod(net0, in_keys=["a"], out_keys=["hidden"])
        net1 = torch.nn.Linear(5, 6)
        module1 = Mod(net1, in_keys=["hidden"], out_keys=[("c", "d")])

        def remove_hidden(td):
            del td["hidden"]
            return td

        module = Seq(lambda td: td.copy(), module0, module1, remove_hidden)
        module_compile = torch.compile(module, fullgraph=True, mode=mode)
        td = TensorDict({"a": torch.randn(32, 4)}, [32])
        module_compile(td)
        assert_close(module(td), module_compile(td))
        assert module_compile(td) is not td

    def test_dispatch_nontensor(self, mode):
        # Non tensor
        x = torch.randn(3)
        y = None
        mod = Seq(
            Mod(lambda x, y: x[y, :], in_keys=["x", "y"], out_keys=["_z"]),
            Mod(lambda x, z: z * x, in_keys=["x", "_z"], out_keys=["out"]),
        )
        assert mod(x=x, y=y)[-1].shape == torch.Size((1, 3))
        mod_compile = torch.compile(mod, fullgraph=True, mode=mode)
        torch.testing.assert_close(mod(x=x, y=y), mod_compile(x=x, y=y))

    def test_dispatch_tensor(self, mode):
        x = torch.randn(3)
        y = torch.randn(3)
        mod = Seq(
            Mod(lambda x, y: x + y, in_keys=["x", "y"], out_keys=["z"]),
            Mod(lambda x, z: z * x, in_keys=["x", "z"], out_keys=["out"]),
        )
        mod(x=x, y=y)
        mod_compile = torch.compile(mod, fullgraph=True, mode=mode)
        torch.testing.assert_close(mod(x=x, y=y), mod_compile(x=x, y=y))

    def test_dispatch_tensordict_keyword(self, mode):
        # module(tensordict=td) compiles like module(td) (#3199)
        module0 = Mod(torch.nn.Linear(4, 5), in_keys=["a"], out_keys=["hidden"])
        module1 = Mod(torch.nn.Linear(5, 6), in_keys=["hidden"], out_keys=["c"])
        for module in (module0, Seq(module0, module1)):
            module_compile = torch.compile(module, fullgraph=True, mode=mode)
            td = TensorDict({"a": torch.randn(32, 4)}, [32])
            assert module_compile(tensordict=td) is td
            assert_close(module(td.copy()), td)

    @set_composite_lp_aggregate(False)
    def test_prob_module_with_kwargs(self, mode):
        kwargs = TensorDictParams(
            TensorDict(scale=1.0, validate_args=NonTensorData(False)), no_convert=True
        )
        dist_cls = torch.distributions.Normal
        mod = Mod(torch.nn.Linear(3, 3), in_keys=["inp"], out_keys=["loc"])
        prob_mod = Seq(
            mod,
            Prob(
                in_keys=["loc"],
                out_keys=["sample"],
                return_log_prob=True,
                distribution_class=dist_cls,
                distribution_kwargs=kwargs,
                default_interaction_type=InteractionType.RANDOM,
            ),
        )
        # check that the scale is in the buffers
        assert len(list(prob_mod.buffers())) == 1
        prob_mod(TensorDict(inp=torch.randn(3)))
        prob_mod_c = torch.compile(prob_mod, fullgraph=True, mode=mode)
        prob_mod_c(TensorDict(inp=torch.randn(3)))

    @set_composite_lp_aggregate(False)
    def test_prob_module_joint_sample_and_log_prob(self, mode):
        # With return_log_prob=True under RANDOM, the module samples and scores
        # in one call when the distribution defines rsample_and_log_prob. The
        # compiled module must take that path without graph breaks.
        class JointNormal(torch.distributions.Normal):
            def rsample_and_log_prob(self, sample_shape=()):
                sample = self.loc.expand(torch.Size(sample_shape) + self.loc.shape)
                # log_prob(sample) would give about -0.92 here
                return sample, torch.full_like(sample, 7.0)

        prob_mod = Prob(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=JointNormal,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=True,
        )
        td = TensorDict(loc=torch.randn(3), scale=torch.ones(3))
        prob_mod_c = torch.compile(prob_mod, fullgraph=True, mode=mode)
        out = prob_mod_c(td.copy())
        torch.testing.assert_close(out["action"], td["loc"])
        torch.testing.assert_close(out["action_log_prob"], torch.full((3,), 7.0))

    @set_composite_lp_aggregate(False)
    def test_prob_module_default_out_key(self, mode):
        # Without out_keys, the sample is written at "_"; the FutureWarning
        # about it must not break a fullgraph compile (#2971)
        prob_mod = Prob(
            in_keys=["loc", "scale"],
            distribution_class=torch.distributions.Normal,
            return_log_prob=True,
        )
        td = TensorDict(loc=torch.randn(3), scale=torch.ones(3))
        prob_mod_c = torch.compile(prob_mod, fullgraph=True, mode=mode)
        out = prob_mod_c(td.copy())
        assert set(out.keys()) == {"_", "__log_prob", "loc", "scale"}

    @pytest.mark.parametrize("mean_raises", [False, True])
    def test_prob_module_mean(self, mode, mean_raises):
        class NoMeanNormal(torch.distributions.Normal):
            @property
            def mean(self):
                raise NotImplementedError

        dist_cls = NoMeanNormal if mean_raises else torch.distributions.Normal
        prob_mod = Prob(
            in_keys=["loc", "scale"],
            out_keys=["sample"],
            distribution_class=dist_cls,
            default_interaction_type=InteractionType.MEAN,
            n_empirical_estimate=8,
        )
        td = TensorDict(loc=torch.randn(3), scale=torch.ones(3))
        prob_mod_c = torch.compile(prob_mod, fullgraph=True, mode=mode)
        sample = prob_mod_c(td.copy())["sample"]
        assert sample.shape == td["loc"].shape
        if not mean_raises:
            torch.testing.assert_close(sample, td["loc"])

    def test_prob_module_interaction_type_change(self, mode):
        # The interaction type is read from a global mode object: the compiled
        # module must follow a change of that mode between calls.
        prob_mod = Prob(
            in_keys=["loc", "scale"],
            out_keys=["sample"],
            distribution_class=torch.distributions.Normal,
        )
        td = TensorDict(loc=torch.zeros(1000), scale=torch.ones(1000))
        prob_mod_c = torch.compile(prob_mod, fullgraph=True, mode=mode)
        with set_interaction_type(InteractionType.MEAN):
            sample = prob_mod_c(td.copy())["sample"]
        torch.testing.assert_close(sample, td["loc"])
        with set_interaction_type(InteractionType.RANDOM):
            sample = prob_mod_c(td.copy())["sample"]
        assert (sample != td["loc"]).all()
        with set_interaction_type(InteractionType.MEAN):
            sample = prob_mod_c(td.copy())["sample"]
        torch.testing.assert_close(sample, td["loc"])

    def test_prob_module_thread_local_interaction_type(self, mode):
        # A thread_local interaction type is read from a threading.local slot:
        # the compiled module must follow it, and fall back to the
        # process-wide one outside of the block.
        prob_mod = Prob(
            in_keys=["loc", "scale"],
            out_keys=["sample"],
            distribution_class=torch.distributions.Normal,
        )
        td = TensorDict(loc=torch.zeros(1000), scale=torch.ones(1000))
        prob_mod_c = torch.compile(prob_mod, fullgraph=True, mode=mode)
        with set_interaction_type(InteractionType.MEAN):
            sample = prob_mod_c(td.copy())["sample"]
            torch.testing.assert_close(sample, td["loc"])
            with set_interaction_type(InteractionType.RANDOM, thread_local=True):
                sample = prob_mod_c(td.copy())["sample"]
            assert (sample != td["loc"]).all()
            sample = prob_mod_c(td.copy())["sample"]
            torch.testing.assert_close(sample, td["loc"])
            with set_interaction_type(InteractionType.RANDOM, thread_local=True):
                sample = prob_mod_c(td.copy())["sample"]
            assert (sample != td["loc"]).all()

    @pytest.mark.parametrize("method", [False, True])
    def test_tdparams_gather(self, mode, method):
        # Gathering from a TensorDictParams gives a plain TensorDict with the
        # batch size, device and names of eager mode, also once the batch
        # size is symbolic (from the second size on).
        def gather(params, index):
            if method:
                return params.gather(0, index)
            return torch.gather(params, 0, index)

        gather_c = torch.compile(gather, fullgraph=True, mode=mode)
        for n in (4, 6, 8):
            params = TensorDictParams(
                TensorDict(
                    a=torch.randn(n, 3),
                    b={"c": torch.randn(n)},
                    batch_size=[n],
                    device="cpu",
                    names=["x"],
                )
            )
            index = torch.arange(0, n, 2)
            expected = gather(params, index)
            result = gather_c(params, index)
            assert type(result) is type(expected) is TensorDict
            assert result.batch_size == expected.batch_size == (n // 2,)
            assert result.device == expected.device
            assert result.names == expected.names == ["x"]
            assert_close(result, expected)

    @pytest.mark.parametrize("op", ["clone", "init", "unsqueeze"])
    def test_tdparams_new(self, mode, op):
        # Building a new TensorDictParams in the compiled region (gh-2791).
        def fn(params):
            if op == "clone":
                return params.clone(recurse=False)
            if op == "init":
                return TensorDictParams(params)
            return params._legacy_unsqueeze(0)

        fn_c = torch.compile(fn, fullgraph=True, mode=mode)
        params = TensorDictParams(
            TensorDict(a=torch.randn(4, 3), b={"c": torch.randn(4)}, batch_size=[4])
        )
        expected = fn(params)
        result = fn_c(params)
        assert type(result) is type(expected) is TensorDictParams
        assert result.batch_size == expected.batch_size
        for named in ("named_parameters", "named_buffers"):
            result_named = dict(getattr(result, named)())
            expected_named = dict(getattr(expected, named)())
            assert result_named.keys() == expected_named.keys()
            for key, value in result_named.items():
                assert type(value) is type(expected_named[key])
        assert_close(result, expected)

    @pytest.mark.parametrize("lock", [False, True])
    def test_tdparams_update(self, mode, lock):
        # update() on a TensorDictParams passed into the compiled region
        # (gh-2791), with an unlocked or a locked content.
        def update(params, other):
            params.update(other)
            return params

        update_c = torch.compile(update, fullgraph=True, mode=mode)
        params = TensorDictParams(
            TensorDict(a=torch.randn(4, 3), b={"c": torch.randn(4)}, batch_size=[4]),
            lock=lock,
        )
        other = TensorDictParams(
            TensorDict(a=torch.zeros(4, 3), b={"d": torch.ones(4)}, batch_size=[4])
        )
        result = update_c(params, other)
        assert result is params
        assert params._param_td.is_locked is lock
        assert params["a"] is other["a"]
        assert params["b", "d"] is other["b", "d"]
        assert dict(params.named_parameters()).keys() == {"a", "b.c", "b.d"}


@pytest.mark.parametrize("mode", [None, "reduce-overhead"])
class TestFunctional:
    # in-place modif raises an error even if fullgraph=False
    @pytest.mark.parametrize("modif_param", [False])
    def test_functional(self, modif_param, mode):

        # TODO: UNTESTED
        class MessUpParams(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros(()))

            def forward(self, x):
                self.param.data.add_(1)
                return x * 1

        module = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 5),
        )
        if modif_param:
            module.append(MessUpParams())

        orig_params = list(module.parameters())
        td = TensorDict.from_module(module)
        td_zero = TensorDictParams(td.data.clone())
        td_zero.zero_()

        def call(x, td):
            with td.to_module(module, preserve_module_state=False):
                y = module(x)
            td.clear_refs_for_compile_()
            return y

        call_compile = torch.compile(call, fullgraph=True, mode=mode)
        x = torch.randn(2, 3)
        assert (call(x, td_zero) == 0).all()
        assert all(
            p_new is p_orig for p_new, p_orig in zip(module.parameters(), orig_params)
        )
        assert (call(x, td_zero) == 0).all()
        assert all(
            p_new is p_orig for p_new, p_orig in zip(module.parameters(), orig_params)
        )
        if modif_param:
            assert td_zero["3", "param"] == 2
        else:
            assert (td_zero == 0).all()
        # torch.testing.assert_close(call_compile(x, td_zero), module(x))

        td.to_module(module, preserve_module_state=False)
        call_compile(x, td_zero)
        assert (call_compile(x, td_zero) == 0).all()
        assert all(
            p_new is p_orig for p_new, p_orig in zip(module.parameters(), orig_params)
        )
        assert (call_compile(x, td_zero) == 0).all()
        assert all(
            p_new is p_orig for p_new, p_orig in zip(module.parameters(), orig_params)
        )
        if modif_param:
            assert td_zero["3", "param"] == 4
        else:
            assert (td_zero == 0).all()

    def test_to_module_restores_on_error(self, mode):
        # The module gets its original parameters back also when the body raises
        module = torch.nn.Linear(3, 4)
        orig_params = list(module.parameters())
        td_zero = TensorDict.from_module(module).data.clone().zero_()

        def call(x, td):
            try:
                with td.to_module(module, preserve_module_state=False):
                    module(x)
                    raise ValueError("error in the body")
            except ValueError:
                pass
            return module(x)

        call_compile = torch.compile(call, fullgraph=True, mode=mode)
        x = torch.randn(2, 3)
        torch.testing.assert_close(call_compile(x, td_zero), module(x))
        assert all(
            p_new is p_orig for p_new, p_orig in zip(module.parameters(), orig_params)
        )

    # in-place modif raises an error even if fullgraph=False
    @pytest.mark.parametrize("preserve_module_state", [False, True])
    def test_vmap_functional(self, mode, preserve_module_state):
        module = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 5),
        )

        td = TensorDict.from_module(module)
        td_zero = TensorDictParams(td.data.expand(10).clone().zero_())

        def call(x, td):
            with td.to_module(module, preserve_module_state=preserve_module_state):
                result = module(x)
            return result

        vmap_call = torch.vmap(call, (None, 0))
        call_compile = torch.compile(vmap_call, fullgraph=True, mode=mode)
        x = torch.randn(2, 3)

        assert (vmap_call(x, td_zero) == 0).all()
        assert (TensorDict.from_module(module) == td).all()
        assert (td_zero == 0).all()

        call_compile(x, td_zero)
        assert (TensorDict.from_module(module) == td).all()
        assert (call_compile(x, td_zero) == 0).all()
        assert (TensorDict.from_module(module) == td).all()
        assert (td_zero == 0).all()

    @pytest.mark.parametrize("loop", [False, True])
    def test_functional_vmap(self, mode, loop):
        """Stacked params over two batch dims, with a mapped and a shared input."""

        class Add(torch.nn.Linear):
            def forward(self, x, y):
                return super().forward(x) + y

        nets = [Add(3, 2) for _ in range(6)]
        params = TensorDictParams(TensorDict.from_modules(*nets).reshape(2, 3))
        x1, x2 = torch.randn(2, 3, 5, 3), torch.randn(5, 2)

        def call(params, x1, x2):
            return functional_vmap(
                nets[0],
                params=params,
                kwargs={"x": x1, "y": x2},
                in_dims={"x": 0},
                loop=loop,
            )

        call_compile = torch.compile(call, fullgraph=True, mode=mode)
        expected = torch.stack([n(x, x2) for n, x in zip(nets, x1.reshape(6, 5, 3))])
        torch.testing.assert_close(call_compile(params, x1, x2), call(params, x1, x2))
        torch.testing.assert_close(
            call_compile(params, x1, x2), expected.reshape(2, 3, 5, 2)
        )


class _ExportStrTC(TensorClass):
    x: torch.Tensor
    name: str


class _ExportNestedStrTC(TensorClass):
    x: torch.Tensor
    inner: _ExportStrTC


class TestExport:
    def test_export_module(self):
        tdm = Mod(lambda x, y: x * y, in_keys=["x", "y"], out_keys=["z"])
        x = torch.randn(3)
        y = torch.randn(3)
        out = torch.export.export(tdm, args=(), kwargs={"x": x, "y": y})
        assert (out.module()(x=x, y=y) == tdm(x=x, y=y)).all()

    def test_export_seq(self):
        tdm = Seq(
            Mod(lambda x, y: x * y, in_keys=["x", "y"], out_keys=["z"]),
            Mod(lambda z, x: z + x, in_keys=["z", "x"], out_keys=["out"]),
        )
        x = torch.randn(3)
        y = torch.randn(3)
        out = torch.export.export(tdm, args=(), kwargs={"x": x, "y": y})
        torch.testing.assert_close(out.module()(x=x, y=y), tdm(x=x, y=y))

    # This tests passes but there are various things that need to be fixed:
    #  - we cannot use vmap directly
    #  - if we use strict=True, there's an error due to the fact that export ignores
    #    the replacement of the params (ie, params are still on "meta" and the values
    #    after the call on the exported module don't match the original ones).
    # Currently only works with strict=False, because export fails to see that
    #  the params in the module have changed and are not 'meta' anymore => this
    #  is symptomatic of export failing to see the functional call
    def test_export_dynamic_batch_size_scalar(self):
        """Scalar SymInt batch_size (x.shape[0]) is accepted during export.

        Regression test for https://github.com/pytorch/tensordict/issues/1003.
        _parse_batch_size did not handle torch.SymInt (which does not subclass
        numbers.Number), causing ValueError("batch size was not specified").
        """

        class Mod(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> TensorDict:
                return TensorDict({"x": x, "y": y}, batch_size=x.shape[0])

        m = Mod()
        x, y = torch.zeros(2, 100), torch.zeros(2, 100)
        ep = torch.export.export(
            m,
            args=(x, y),
            strict=False,
            dynamic_shapes={
                "x": {0: torch.export.Dim("batch"), 1: torch.export.Dim("time")},
                "y": {0: torch.export.Dim("batch"), 1: torch.export.Dim("time")},
            },
        )
        out = ep.module()(torch.zeros(5, 100), torch.zeros(5, 100))
        assert out.batch_size == torch.Size([5])

    def test_export_dynamic_batch_size_multi_dim(self):
        """Multi-dim SymInt batch_size ([b, t]) is serializable in pytree context.

        Regression test for https://github.com/pytorch/tensordict/issues/1003.
        _tensordict_flatten stored batch_size (torch.Size that may contain SymInts)
        in the pytree context. torch.export cannot serialize SymInts in the output
        pytree spec, raising AsPythonConstantNotImplementedError.
        Fix: store batch_dims (int) and reconstruct batch_size from tensor shapes.
        """

        class Mod(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> TensorDict:
                b, t = x.shape[0], x.shape[1]
                return TensorDict({"x": x, "y": x * 2}, batch_size=[b, t])

        m = Mod()
        inp = torch.randn(2, 6, 8)
        ep = torch.export.export(
            m,
            args=(inp,),
            dynamic_shapes=({0: torch.export.Dim("batch", min=1)},),
            strict=True,
        )
        out = ep.module()(torch.randn(4, 6, 8))
        assert out.batch_size == torch.Size([4, 6])
        torch.testing.assert_close(out["y"], out["x"] * 2)

    class _TDInOutModule(torch.nn.Module):
        def forward(self, td: TensorDict) -> TensorDict:
            return TensorDict(c=td["a"] * 2 + td["b"], batch_size=td.batch_size)

    @pytest.mark.parametrize("strict", [False, True])
    def test_export_td_input(self, strict):
        td = TensorDict(a=torch.randn(4, 3), b=torch.randn(4, 1), batch_size=[4])
        ep = torch.export.export(self._TDInOutModule(), (td,), strict=strict)
        out = ep.module()(td)
        assert out.batch_size == torch.Size([4])
        torch.testing.assert_close(out["c"], td["a"] * 2 + td["b"])

    @pytest.mark.parametrize("strict", [False, True])
    def test_export_td_input_dynamic_batch_size(self, strict):
        td = TensorDict(a=torch.randn(4, 3), b=torch.randn(4, 1), batch_size=[4])
        batch = torch.export.Dim("batch", min=2)
        ep = torch.export.export(
            self._TDInOutModule(),
            (td,),
            strict=strict,
            # One entry per leaf of the tensordict, in key order.
            dynamic_shapes=([{0: batch}, {0: batch}],),
        )
        td = TensorDict(a=torch.randn(5, 3), b=torch.randn(5, 1), batch_size=[5])
        out = ep.module()(td)
        assert out.batch_size == torch.Size([5])
        torch.testing.assert_close(out["c"], td["a"] * 2 + td["b"])

    @pytest.mark.parametrize("strict", [False, True])
    def test_export_td_input_non_tensor_dynamic_batch_size(self, strict, monkeypatch):
        # The non-tensor entry takes [] in dynamic_shapes. Export rebuilds the
        # input with the symbolic batch size: the entry must get that batch
        # size before TD_CHECK_INVARIANTS checks the new tensordict, and strict
        # export must not specialize the batch size.
        monkeypatch.setattr("tensordict._td._CHECK_INVARIANTS", True)

        class Mod(torch.nn.Module):
            def forward(self, td):
                return td["a"] * 2

        def make(batch):
            return TensorDict(a=torch.randn(batch, 3), s="a string", batch_size=[batch])

        batch = torch.export.Dim("batch", min=2)
        ep = torch.export.export(
            Mod(), (make(4),), strict=strict, dynamic_shapes=([{0: batch}, []],)
        )
        td = make(5)
        torch.testing.assert_close(ep.module()(td), td["a"] * 2)

    @pytest.mark.parametrize("strict", [False, True])
    def test_export_td_input_empty_nested(self, strict):
        # The nested td has no tensor: its batch size comes from the spec.
        class Mod(torch.nn.Module):
            def forward(self, td):
                return td["a"] * 2, td.batch_size, td["nested"].batch_size

        td = TensorDict(
            nested=TensorDict(batch_size=[4]), a=torch.randn(4, 3), batch_size=[4]
        )
        ep = torch.export.export(Mod(), (td,), strict=strict)
        out, batch_size, nested_batch_size = ep.module()(td)
        torch.testing.assert_close(out, td["a"] * 2)
        assert tuple(batch_size) == (4,)
        assert tuple(nested_batch_size) == (4,)

    @pytest.mark.parametrize("strict", [False, True])
    def test_export_td_input_names(self, strict):
        # Export traces with is_compiling() True. The input spec of non-strict
        # export must record the dim names, so that the exported module
        # accepts the named tensordict it was exported with, and the clone
        # built in forward must keep them.
        class Mod(torch.nn.Module):
            def forward(self, td):
                return td.clone()

        td = TensorDict(a=torch.randn(4, 3), batch_size=[4], names=["n"])
        ep = torch.export.export(Mod(), (td,), strict=strict)
        out = ep.module()(td)
        assert out.names == ["n"]
        torch.testing.assert_close(out["a"], td["a"])

    @pytest.mark.parametrize("strict", [False])  # , True])
    def test_export_with_td_params(self, strict):
        module = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 5),
        )
        module_td = TensorDictParams(
            TensorDict.from_module(module).data.expand(2).clone()
        )
        assert all(
            isinstance(p, torch.nn.Parameter) for p in module_td.values(True, True)
        )

        class MyModule(torch.nn.Module):
            def __init__(self, td_params):
                super().__init__()
                self.tdparams = td_params
                self.arch = torch.nn.Sequential(
                    torch.nn.Linear(3, 4, device="meta"),
                    torch.nn.Linear(4, 5, device="meta"),
                )

            def forward(self, x):
                # vmap with params currently fails
                #  return torch.vmap(self.batch_forward, (0, None))(self.tdparams, x)
                return torch.stack(
                    [self.batch_forward(p, x) for p in self.tdparams.unbind(0)]
                )

            def batch_forward(self, params, x):
                with params.to_module(self.arch, preserve_module_state=False):
                    return self.arch(x)
                # This could be an option but dynamo doesn't know how to trace through state_dict ops
                # sd = self.arch.state_dict()
                # try:
                #     self.arch.load_state_dict(params.flatten_keys().to_dict(), assign=True)
                #     return self.arch(x)
                # finally:
                #     self.arch.load_state_dict(sd, assign=True)

        m = MyModule(module_td)
        x = torch.randn(3)
        assert m(x).shape == (2, 5)
        exported_module = torch.export.export(
            m,
            args=(),
            kwargs={"x": x},
            strict=strict,
        )
        torch.testing.assert_close(exported_module.module()(x=x), m(x))

    @pytest.mark.parametrize("strict", [False, True])
    def test_export_td_input_non_tensor(self, strict):
        # The exported module compares the spec of its input with the spec of
        # the example input, which holds a batched non-tensor entry.
        class Mod(torch.nn.Module):
            def forward(self, td):
                return td["a"] * 2

        def make():
            return TensorDict(a=torch.randn(4, 3), s="a string", batch_size=[4])

        ep = torch.export.export(Mod(), (make(),), strict=strict)
        td = make()
        torch.testing.assert_close(ep.module()(td), td["a"] * 2)

    def test_export_td_output_non_tensor(self):
        # The non-tensor entry of the output is rebuilt with the batch size of
        # the output. Not with strict=True: there _tensordict_unflatten first
        # builds the entry with batch size [0], which TD_CHECK_INVARIANTS
        # rejects before the entry is set again with the right batch size.
        class Mod(torch.nn.Module):
            def forward(self, td):
                return td.apply(lambda x: x * 2)

        def make():
            return TensorDict(a=torch.randn(4, 3), s="a string", batch_size=[4])

        ep = torch.export.export(Mod(), (make(),), strict=False)
        td = make()
        out = ep.module()(td)
        torch.testing.assert_close(out["a"], td["a"] * 2)
        assert out.get("s").batch_size == torch.Size([4])
        assert out.get_non_tensor("s") == "a string"

    @pytest.mark.parametrize("strict", [False, True])
    @pytest.mark.parametrize("nested", [False, True])
    def test_export_tc_input_str_dynamic_batch_size(self, nested, strict, monkeypatch):
        # The str field is held in the TensorDict of the tensorclass, as a
        # non-tensor entry that the pytree rebuilds with its stored batch
        # size: export would specialize the dynamic batch dim to it.
        # https://github.com/pytorch/tensordict/issues/3163
        # Checked with TD_CHECK_INVARIANTS on, as CI runs: the pytree builds
        # the entry with the right batch size before the tensordict.
        monkeypatch.setattr("tensordict._td._CHECK_INVARIANTS", True)

        class Mod(torch.nn.Module):
            def forward(self, tc):
                return tc.apply(lambda x: x * 2)

        def make(n):
            tc = _ExportStrTC(x=torch.randn(n, 3), name="a string", batch_size=[n])
            if nested:
                tc = _ExportNestedStrTC(x=torch.randn(n, 2), inner=tc, batch_size=[n])
            return tc

        batch = torch.export.Dim("batch", min=2)
        # The str field is a pytree node without leaves: its entry is [].
        dynamic_shapes = [{0: batch}, []]
        if nested:
            dynamic_shapes = [{0: batch}, dynamic_shapes]
        ep = torch.export.export(
            Mod(), (make(4),), dynamic_shapes=(dynamic_shapes,), strict=strict
        )
        tc = make(5)
        out = ep.module()(tc)
        assert type(out) is type(tc)
        assert out.batch_size == torch.Size([5])
        torch.testing.assert_close(out.x, tc.x * 2)
        if nested:
            out, tc = out.inner, tc.inner
            assert out.batch_size == torch.Size([5])
            torch.testing.assert_close(out.x, tc.x * 2)
        assert out.name == "a string"
        assert out._tensordict.get("name").batch_size == torch.Size([5])


@pytest.mark.skipif(not _has_onnx, reason="ONNX is not available")
class TestONNXExport:
    def test_onnx_export_module(self, tmpdir):
        tdm = Mod(lambda x, y: x * y, in_keys=["x", "y"], out_keys=["z"])
        x = torch.randn(3)
        y = torch.randn(3)
        torch_input = {"x": x, "y": y}
        onnx_program = torch.onnx.export(tdm, kwargs=torch_input, dynamo=True)

        path = Path(tmpdir) / "file.onnx"
        onnx_program.save(str(path))
        import onnxruntime

        ort_session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )

        def to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

        onnxruntime_input = {k: to_numpy(v) for k, v in torch_input.items()}

        onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
        torch.testing.assert_close(
            torch.as_tensor(onnxruntime_outputs[0]), tdm(x=x, y=y)
        )

    def test_onnx_export_seq(self, tmpdir):
        tdm = Seq(
            Mod(lambda x, y: x * y, in_keys=["x", "y"], out_keys=["z"]),
            Mod(lambda z, x: z + x, in_keys=["z", "x"], out_keys=["out"]),
        )
        x = torch.randn(3)
        y = torch.randn(3)
        torch_input = {"x": x, "y": y}
        torch.onnx.export(tdm, kwargs=torch_input, dynamo=True)
        onnx_program = torch.onnx.export(tdm, kwargs=torch_input, dynamo=True)

        path = Path(tmpdir) / "file.onnx"
        onnx_program.save(str(path))
        import onnxruntime

        ort_session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )

        def to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

        onnxruntime_input = {k: to_numpy(v) for k, v in torch_input.items()}

        onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
        torch.testing.assert_close(
            tree_map(torch.as_tensor, onnxruntime_outputs), tdm(x=x, y=y)
        )


@pytest.mark.parametrize("compiled", [False, True])
class TestCudaGraphs:
    @pytest.fixture(scope="class", autouse=True)
    def _set_cuda_device(self):
        device = torch.get_default_device()
        if torch.cuda.is_available():
            torch.set_default_device("cuda:0")
        yield
        # With no default device set, get_default_device() returns cpu, and
        # set_default_device(cpu) would leave a device mode on for later tests.
        torch.set_default_device(None if device.type == "cpu" else device)

    def test_cudagraphs_keeps_pytree_registry(self, compiled):
        # pytorch/tensordict#2076: creating and calling a CudaGraphModule used to
        # remove the tensordict classes from the pytree registry for the whole
        # process, which broke torch.export and torch.func on tensordicts.
        td_types = PYTREE_REGISTERED_TDS + PYTREE_REGISTERED_LAZY_TDS
        assert all(tdtype in SUPPORTED_NODES for tdtype in td_types)

        def func(td, x):
            return td["a"] + x, td.apply(lambda t: t * 2)

        tdmodule = self._make_cudagraph(
            TensorDictModule(lambda a: a + 1, in_keys=["a"], out_keys=["b"]),
            compiled,
        )
        graphed_func = self._make_cudagraph(func, compiled)
        for i in range(4):
            td = TensorDict(a=torch.full((3,), float(i)), batch_size=[3])
            assert (tdmodule(td.clone())["b"] == i + 1).all()
            y, td_out = graphed_func(td, torch.ones(3))
            assert (y == i + 1).all()
            assert (td_out["a"] == 2 * i).all()
        assert all(tdtype in SUPPORTED_NODES for tdtype in td_types)
        td = TensorDict(a=torch.ones(3), batch_size=[3])
        leaves = tree_leaves(td)
        assert len(leaves) == 1 and leaves[0] is td["a"]
        grad = torch.func.grad(lambda td: (td["a"] ** 2).sum())(td)
        torch.testing.assert_close(grad["a"], 2 * td["a"])

    def test_cudagraphs_random(self, compiled):
        def func(x):
            return x + torch.randn_like(x)

        if compiled:
            func = torch.compile(func)

        with (
            pytest.warns(UserWarning)
            if not torch.cuda.is_available()
            else contextlib.nullcontext()
        ):
            func = CudaGraphModule(func)

        x = torch.randn(10)
        for _ in range(10):
            func(x)
        assert isinstance(func(torch.zeros(10)), torch.Tensor)
        assert (func(torch.zeros(10)) != 0).any()
        y0 = func(x)
        y1 = func(x + 1)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(y0, y1 + 1)

    def test_cudagraphs_module_rewriting_its_input_key(self, compiled):
        """A module that writes an output under one of its input keys.

        The set() rebinds that entry of the captured input during capture, so
        the replay must copy new inputs into the leaves the graph reads, not
        into the rebound entry.
        """

        class Recurrent(torch.nn.Module):
            def forward(self, x, h):
                h = torch.tanh(h + x)
                return h.sum(-1, keepdim=True), h

        module = TensorDictModule(Recurrent(), in_keys=["x", "h"], out_keys=["y", "h"])
        graphed = self._make_cudagraph(module, compiled, warmup=2)

        def make(h):
            return TensorDict({"x": torch.ones(4, 3), "h": torch.full((4, 3), h)}, [4])

        for _ in range(3):
            graphed(make(0.0))
        for h in (1.0, -1.0):
            expected = module(make(h))
            result = graphed(make(h))
            torch.testing.assert_close(result["y"], expected["y"])
            torch.testing.assert_close(result["h"], expected["h"])

    @pytest.mark.parametrize(
        "in_keys,out_keys,warns",
        [
            (["z"], ["w"], ["in_keys", "out_keys"]),
            (["z"], None, ["in_keys"]),
            ([], [("w",), "v"], ["out_keys"]),
            ([], [], []),
            (None, None, []),
            ([("x",)], ["y"], []),
            ("z", "y", ["in_keys"]),
            (("z", "x"), {"y"}, ["in_keys"]),
            ({"x"}, {"w"}, ["out_keys"]),
            (np.array(["z", "w"]), torch.tensor([1, 2]), []),
            (torch.tensor([0, 1]), np.array(["v"]), []),
        ],
    )
    def test_cudagraphs_ignored_keys_warn(self, compiled, in_keys, out_keys, warns):
        # pytorch/tensordict#2973: the module's own in_keys/out_keys win over the
        # arguments. Non-empty arguments that differ from them warn; empty
        # lists (TorchRL's collector passes them) and equal keys stay silent.
        # Arguments that are not a str, list, tuple or set (an array, a tensor)
        # are ignored silently, as on main.
        module = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
        if compiled:
            module = torch.compile(module)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            graphed = CudaGraphModule(module, in_keys=in_keys, out_keys=out_keys)
        ignored = [w for w in record if "ignores" in str(w.message)]
        assert len(ignored) == len(warns)
        for name, w in zip(warns, ignored):
            assert str(w.message).startswith(f"CudaGraphModule ignores {name}=")
            assert w.filename == __file__
        assert graphed.in_keys == ["x"]
        assert graphed.out_keys == ["y"]

    @staticmethod
    def _make_cudagraph(
        func: Callable, compiled: bool, *args, **kwargs
    ) -> CudaGraphModule:
        if compiled:
            func = torch.compile(func)
        with (
            pytest.warns(UserWarning)
            if not torch.cuda.is_available()
            else contextlib.nullcontext()
        ):
            func = CudaGraphModule(func, *args, **kwargs)
        return func

    @staticmethod
    def check_types(func, *args, **kwargs):
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for param_name, param in signature.parameters.items():
            arg_value = bound_args.arguments[param_name]
            if param.annotation != param.empty:
                if not isinstance(arg_value, param.annotation):
                    raise TypeError(
                        f"Argument '{param_name}' should be of type {param.annotation}, but is of type {type(arg_value)}"
                    )

    def test_signature(self, compiled):
        if compiled:
            pytest.skip()

        def func(x: torch.Tensor):
            return x + torch.randn_like(x)

        with pytest.raises(TypeError):
            self.check_types(func, "a string")
        self.check_types(func, torch.ones(()))

    def test_backprop(self, compiled):
        x = torch.nn.Parameter(torch.ones(3))
        y = torch.nn.Parameter(torch.ones(3))
        optimizer = torch.optim.SGD([x, y], lr=1)

        def func():
            optimizer.zero_grad()
            z = x + y
            z = z.sum()
            z.backward()
            optimizer.step()

        func = self._make_cudagraph(func, compiled, warmup=4)

        for i in range(1, 11):
            torch.compiler.cudagraph_mark_step_begin()
            func()

            assert (x == 1 - i).all(), i
            assert (y == 1 - i).all(), i
            # assert (x.grad == 1).all()
            # assert (y.grad == 1).all()

    def test_tdmodule(self, compiled):
        tdmodule = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
        tdmodule = self._make_cudagraph(tdmodule, compiled)
        assert tdmodule._is_tensordict_module
        for i in range(10):
            td = TensorDict(x=torch.randn(()))
            tdmodule(td)
            assert td["y"] == td["x"] + 1, i

        tdmodule = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
        tdmodule = self._make_cudagraph(tdmodule, compiled)
        assert tdmodule._is_tensordict_module
        for _ in range(10):
            x = torch.randn(())
            y = tdmodule(x=x)
            assert y == x + 1

        tdmodule = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
        tdmodule = self._make_cudagraph(tdmodule, compiled)
        assert tdmodule._is_tensordict_module
        for _ in range(10):
            td = TensorDict(x=torch.randn(()))
            tdout = TensorDict()
            tdmodule(td, tensordict_out=tdout)
            assert tdout is not td
            assert "x" not in tdout
            assert tdout["y"] == td["x"] + 1

        tdmodule = lambda td: td.set("y", td.get("x") + 1)
        tdmodule = self._make_cudagraph(tdmodule, compiled, in_keys=[], out_keys=[])
        assert tdmodule._is_tensordict_module
        for i in range(10):
            td = TensorDict(x=torch.randn(()))
            tdmodule(td)
            assert tdmodule._out_matches_in
            if i >= tdmodule._warmup and torch.cuda.is_available():
                assert tdmodule._selected_keys == ["y"]
            assert td["y"] == td["x"] + 1

        tdmodule = lambda td: td.set("y", td.get("x") + 1)
        tdmodule = self._make_cudagraph(
            tdmodule, compiled, in_keys=["x"], out_keys=["y"]
        )
        assert tdmodule._is_tensordict_module
        for _ in range(10):
            td = TensorDict(x=torch.randn(()))
            tdmodule(td)
            assert td["y"] == td["x"] + 1

        tdmodule = lambda td: td.copy().set("y", td.get("x") + 1)
        tdmodule = self._make_cudagraph(tdmodule, compiled, in_keys=[], out_keys=[])
        assert tdmodule._is_tensordict_module
        for _ in range(10):
            td = TensorDict(x=torch.randn(()))
            tdout = tdmodule(td)
            assert tdout is not td
            assert "y" not in td
            assert tdout["y"] == td["x"] + 1

    def test_tdmodule_outputs_do_not_alias(self, compiled):
        # Every replay writes into the same input and output buffers: the
        # tensordicts returned by earlier calls, the capture call included,
        # must keep their own values.
        tdmodule = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
        tdmodule = self._make_cudagraph(tdmodule, compiled)
        tds = [
            tdmodule(TensorDict(x=torch.full((3,), float(i)), batch_size=[3]))
            for i in range(6)
        ]
        for i, td in enumerate(tds):
            torch.testing.assert_close(td["x"], torch.full((3,), float(i)))
            torch.testing.assert_close(td["y"], torch.full((3,), float(i + 1)))

    def test_tdmodule_structure_change_raises(self, compiled):
        if not torch.cuda.is_available():
            pytest.skip("CudaGraphModule only replays graphs on CUDA")
        tdmodule = TensorDictModule(
            lambda x, z: x + z, in_keys=["x", "z"], out_keys=["y"]
        )
        tdmodule = self._make_cudagraph(tdmodule, compiled)
        for _ in range(4):
            tdmodule(
                TensorDict(
                    x=torch.randn(3), z=torch.randn(3), w=torch.randn(3), batch_size=[3]
                )
            )
        # A key that is not an in_key may be missing.
        td = tdmodule(TensorDict(x=torch.ones(3), z=torch.ones(3), batch_size=[3]))
        torch.testing.assert_close(td["y"], torch.full((3,), 2.0))
        with pytest.raises(KeyError, match="missing the in_keys \\['z'\\]"):
            tdmodule(TensorDict(x=torch.randn(3), batch_size=[3]))
        with pytest.raises(ValueError, match="captured with batch_size"):
            tdmodule(TensorDict(x=torch.randn(1), z=torch.randn(1), batch_size=[1]))

    def test_tdmodule_cpu_input_copied_before_return(self, compiled):
        # The caller may overwrite a pinned CPU input as soon as the call returns:
        # the copy to the graph's buffers must be done by then.
        if not torch.cuda.is_available():
            pytest.skip("CudaGraphModule only replays graphs on CUDA")
        tdmodule = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
        tdmodule = self._make_cudagraph(tdmodule, compiled)
        for _ in range(4):
            tdmodule(TensorDict(x=torch.zeros(3), batch_size=[3]))
        x = torch.ones(3, device="cpu").pin_memory()
        # Keep the stream busy so that a pending copy would run after the write.
        torch.cuda._sleep(100_000_000)
        td = tdmodule(TensorDict(x=x, batch_size=[3]))
        x.fill_(100.0)
        torch.testing.assert_close(td["y"], torch.full((3,), 2.0))

    def test_tdmodule_entry_shape_change_raises(self, compiled):
        # Same batch size, entry of another shape: it must not be broadcast into
        # the captured buffer, or fail in the copy.
        if not torch.cuda.is_available():
            pytest.skip("CudaGraphModule only replays graphs on CUDA")
        tdmodule = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
        tdmodule = self._make_cudagraph(tdmodule, compiled)
        for _ in range(4):
            tdmodule(TensorDict(x=torch.zeros(3, 4), w=torch.zeros(3), batch_size=[3]))
        match = r"entry 'x' of shape torch.Size\(\[3, 4\]\) but got shape"
        for shape in ((3, 1), (3, 5)):
            # Every captured entry is present.
            with pytest.raises(ValueError, match=match):
                tdmodule(
                    TensorDict(x=torch.zeros(shape), w=torch.zeros(3), batch_size=[3])
                )
            # A captured entry that is not an in_key is missing.
            with pytest.raises(ValueError, match=match):
                tdmodule(TensorDict(x=torch.zeros(shape), batch_size=[3]))
        # Captured on a lazy stack, replayed on a dense tensordict.
        tdmodule = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
        tdmodule = self._make_cudagraph(tdmodule, compiled)
        for _ in range(4):
            tdmodule(
                LazyStackedTensorDict(
                    *TensorDict(x=torch.zeros(3, 4), batch_size=[3]).unbind(0)
                )
            )
        with pytest.raises(ValueError, match=match):
            tdmodule(TensorDict(x=torch.zeros(3, 1), batch_size=[3]))

    def test_tdmodule_uncaptured_in_key_missing(self, compiled):
        # An in_key that the captured input did not hold is not reported missing.
        if not torch.cuda.is_available():
            pytest.skip("CudaGraphModule only replays graphs on CUDA")
        func = self._make_cudagraph(
            lambda td: td.set("y", td["x"] + 1),
            compiled,
            in_keys=["x", "unused"],
            out_keys=["y"],
        )
        for _ in range(4):
            func(TensorDict(x=torch.zeros(3), w=torch.zeros(3), batch_size=[3]))
        td = func(TensorDict(x=torch.ones(3), batch_size=[3]))
        torch.testing.assert_close(td["y"], torch.full((3,), 2.0))

    @pytest.mark.parametrize("capture_lazy", [True, False])
    def test_tdmodule_lazy_and_dense_inputs(self, compiled, capture_lazy):
        # Captured on a lazy stack and replayed on a dense tensordict, or the
        # reverse: the leaves have other keys, and must still be copied.
        if not torch.cuda.is_available():
            pytest.skip("CudaGraphModule only replays graphs on CUDA")

        def make(value, lazy):
            td = TensorDict(x=torch.full((2, 3), value), batch_size=[2])
            return LazyStackedTensorDict(*td.unbind(0), stack_dim=0) if lazy else td

        tdmodule = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
        tdmodule = self._make_cudagraph(tdmodule, compiled)
        for _ in range(4):
            tdmodule(make(0.0, capture_lazy))
        td = tdmodule(make(5.0, not capture_lazy))
        torch.testing.assert_close(td["y"], torch.full((2, 3), 6.0))

    def test_non_tdmodule_shape_change_raises(self, compiled):
        if not torch.cuda.is_available():
            pytest.skip("CudaGraphModule only replays graphs on CUDA")
        func = self._make_cudagraph(lambda x: x + 1, compiled)
        for _ in range(4):
            func(torch.randn(3))
        with pytest.raises(ValueError, match="captured with an input of shape"):
            func(torch.randn(1))
        func = self._make_cudagraph(lambda td: td["x"] + 1, compiled)
        for _ in range(4):
            func(TensorDict(x=torch.randn(3), batch_size=[3]))
        with pytest.raises(ValueError, match="captured with batch_size"):
            func(TensorDict(x=torch.randn(1), batch_size=[1]))

    def test_td_input_non_tdmodule_writes_input(self, compiled):
        # The function adds a key to its input: the capture must run on a
        # tensordict with the warmup structure, or a compiled function
        # recompiles during capture.
        def func(td):
            return td.set("y", td.get("x") + 1)

        func = self._make_cudagraph(func, compiled)
        for _ in range(4):
            td = TensorDict(x=torch.randn(3), batch_size=[3])
            out = func(td)
            torch.testing.assert_close(out["y"], td["x"] + 1)

    def test_repr(self, compiled):
        func = self._make_cudagraph(lambda x: x + 1, compiled, warmup=3)
        assert "warmup=3" in repr(func)

    def test_td_input_non_tdmodule(self, compiled):
        func = lambda x: x + 1
        func = self._make_cudagraph(func, compiled)
        for i in range(10):
            td = TensorDict(a=1)
            func(td)
            if i == 5:
                assert not func._is_tensordict_module

    def test_td_input_non_tdmodule_nontensor(self, compiled):
        func = lambda x, y: x + y
        func = self._make_cudagraph(func, compiled)
        for i in range(10):
            assert func(torch.zeros(()), 1.0) == 1.0
            if i == 5:
                assert not func._is_tensordict_module
        if torch.cuda.is_available():
            with pytest.raises(
                ValueError, match="Varying inputs must be torch.Tensor subclasses."
            ):
                func(torch.zeros(()), 2.0)

    def test_td_input_nontensor_arg_change_warns(self, compiled):
        # pytorch/tensordict#2075: the replay of a function that takes a
        # tensordict reads the captured non-tensor arguments. A call with other
        # values warns that it will raise in 0.17, and keeps today's result.
        if not torch.cuda.is_available():
            pytest.skip("CudaGraphModule only replays graphs on CUDA")

        def update(td, flag, scale=1.0):
            out = td["x"] + 100 if flag else td["x"] + 1
            return TensorDict(y=out * scale)

        func = self._make_cudagraph(update, compiled, in_keys=[], out_keys=[])
        td = TensorDict(x=torch.zeros(()))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for _ in range(4):
                assert func(td, True)["y"] == 100
            assert func(td, flag=True)["y"] == 100
            assert func(td, 1)["y"] == 100
        for args, kwargs in [
            ((False,), {}),
            ((True,), {"scale": 2.0}),
            ((True, 2.0), {}),
        ]:
            with pytest.warns(
                FutureWarning, match="Starting with TensorDict 0.17"
            ) as record:
                assert func(td, *args, **kwargs)["y"] == 100
            assert {w.filename for w in record if w.category is FutureWarning} == {
                __file__
            }

    def test_interaction_type_change_warns(self, compiled):
        # pytorch/tensordict#2547: the replay repeats the branch that the
        # probabilistic modules took at capture, whatever the interaction type.
        if not torch.cuda.is_available():
            pytest.skip("CudaGraphModule only replays graphs on CUDA")
        policy = Prob(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=torch.distributions.Normal,
            distribution_kwargs={"validate_args": False},
        )
        td = TensorDict(loc=torch.zeros(3), scale=torch.ones(3), batch_size=[3])
        random = self._make_cudagraph(policy, compiled, warmup=3)
        default = self._make_cudagraph(policy, compiled, warmup=3)
        linear = self._make_cudagraph(
            Mod(torch.nn.Linear(3, 3), in_keys=["loc"], out_keys=["y"]), compiled
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("error", "CudaGraphModule was captured")
            with set_interaction_type("random"):
                for _ in range(4):
                    random(td.clone())
                    linear(td.clone())
            for _ in range(4):
                default(td.clone())
            # The default interaction type of the policy is "deterministic".
            with set_interaction_type("deterministic"):
                default(td.clone())
                linear(td.clone())
        with set_interaction_type("deterministic"):
            with pytest.warns(
                UserWarning, match="Capture one CudaGraphModule per interaction type"
            ) as record:
                random(td.clone())
            assert [w.filename for w in record] == [__file__]
            with warnings.catch_warnings():
                warnings.filterwarnings("error", "CudaGraphModule was captured")
                random(td.clone())
        with set_interaction_type("random"), pytest.warns(UserWarning):
            default(td.clone())

    def test_state_dict(self, compiled):
        # Create a linear layer and wrap it in CudaGraphModule
        linear = torch.nn.Linear(3, 4)
        linear = self._make_cudagraph(linear, compiled)

        # Run some warmup iterations
        x = torch.randn(10, 3)
        for _ in range(10):
            linear(x)

        # Get state dict
        state_dict = linear.state_dict()
        if compiled:
            state_dict_get = TensorDict(state_dict)
            state_dict_get = state_dict_get.unflatten_keys(".")["_orig_mod"]
        else:
            state_dict_get = state_dict

        assert "weight" in state_dict_get
        assert "bias" in state_dict_get
        assert state_dict_get["weight"].shape == (4, 3)
        assert state_dict_get["bias"].shape == (4,)

        # Create a new instance and load state
        linear2 = torch.nn.Linear(3, 4)
        linear2 = self._make_cudagraph(linear2, compiled)
        linear2.load_state_dict(state_dict)

        # Test that both modules produce the same output
        y1 = linear(x)
        y2 = linear2(x)
        torch.testing.assert_close(y1, y2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
class TestCompileNontensor:
    # Same issue with the decorator @tensorclass version
    @pytest.fixture(scope="class")
    def data(self):
        return torch.zeros((4, 3), device=cur_device)

    class TensorClassWithNonTensorData(TensorClass["nocast"]):
        tensor: torch.Tensor
        non_tensor_data: int

    def fn_no_device_no_batch_size(self, data):
        a = self.TensorClassWithNonTensorData(tensor=data, non_tensor_data=1)
        return a.tensor

    def fn_no_device(self, data):
        a = self.TensorClassWithNonTensorData(
            tensor=data, non_tensor_data=1, batch_size=[4]
        )
        return a.tensor

    def fn_with_device(self, data):
        a = self.TensorClassWithNonTensorData(
            tensor=data, non_tensor_data=1, batch_size=[4], device=cur_device
        )
        return a.tensor

    def fn_with_device_without_batch_size(self, data):
        a = self.TensorClassWithNonTensorData(
            tensor=data, non_tensor_data=1, device=cur_device
        )
        return a.tensor

    def test_nontensor_no_device_no_batch_size(self, data):
        torch.compile(self.fn_no_device_no_batch_size)(data)

    def test_nontensor_no_device(self, data):
        torch.compile(self.fn_no_device)(data)

    def test_nontensor_with_device(self, data):
        torch.compile(self.fn_with_device)(data)

    def test_nontensor_with_device_without_batch_size(self, data):
        torch.compile(self.fn_with_device_without_batch_size)(data)


class TestTCNonTensorInit:
    """TensorClass with non-tensor fields must be constructible under torch.compile without graph breaks."""

    class MyTC(TensorClass):
        x: torch.Tensor
        label: str

    def test_tc_nontensor_init_fullgraph(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(a):
            tc = self.MyTC(x=a, label="hello", batch_size=[3])
            return tc.x

        result = fn(torch.randn(3))
        assert result.shape == (3,)

    def test_tc_nontensor_init_roundtrip(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(a):
            tc = self.MyTC(x=a, label="hello", batch_size=[3])
            return tc.x + 1

        inp = torch.randn(3)
        result = fn(inp)
        torch.testing.assert_close(result, inp + 1)

    def test_tc_nontensor_init_with_device(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(a):
            tc = self.MyTC(x=a, label="world", batch_size=[3], device="cpu")
            return tc.x * 2

        inp = torch.randn(3)
        result = fn(inp)
        torch.testing.assert_close(result, inp * 2)

    def test_tc_positional_init_fullgraph(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(a):
            tc = self.MyTC(a, "hello", batch_size=[3])
            return tc.x + 1, tc.label

        inp = torch.randn(3)
        result, label = fn(inp)
        torch.testing.assert_close(result, inp + 1)
        assert label == "hello"

    def test_nontensordata_positional_init_fullgraph(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(a):
            return a + 1, NonTensorData("hello", batch_size=[3])

        _, data = fn(torch.randn(3))
        assert isinstance(data, NonTensorData)
        assert data.data == "hello"
        assert data.batch_size == (3,)

    def test_metadata_positional_init_fullgraph(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(a):
            return a + 1, MetaData({"a": 1})

        _, data = fn(torch.randn(3))
        assert isinstance(data, MetaData)
        assert data.data == {"a": 1}


@tensorclass
class _PostInitTC:
    x: torch.Tensor

    def __post_init__(self):
        self.x = self.x * 2


@tensorclass(skip_post_init_on_rebuild=True)
class _SkipRebuildPostInitTC:
    x: torch.Tensor

    def __post_init__(self):
        self.x = self.x * 2


@tensorclass
class _InitVarPostInitTC:
    x: torch.Tensor
    scale: dataclasses.InitVar[float] = 1.0

    def __post_init__(self, scale):
        self.x = self.x * scale


with warnings.catch_warnings():
    # a shared tensor default, which 0.17 copies for each instance (#3095)
    warnings.filterwarnings("ignore", "The Tensor default", FutureWarning)

    @tensorclass
    class _ConcreteDefaultTC:
        cache: torch.Tensor = torch.zeros(3)


@tensorclass
class _FactoryDefaultTC:
    cache: torch.Tensor = dataclasses.field(default_factory=lambda: torch.zeros(3))


@dataclasses.dataclass
class _FactoryDefaultDataClass:
    cache: torch.Tensor = dataclasses.field(default_factory=lambda: torch.zeros(3))


_FactoryDefaultFromDataclassTC = from_dataclass(_FactoryDefaultDataClass)


@tensorclass
class _NoneDefaultTC:
    x: torch.Tensor
    y: torch.Tensor = None


class TestTCPostInitCompile:
    """__post_init__ must run under torch.compile to match eager semantics (gh-1708)."""

    def test_post_init_runs_under_compile(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return _PostInitTC(x=x).x

        inp = torch.ones(3)
        torch.testing.assert_close(_PostInitTC(x=inp).x, torch.full((3,), 2.0))
        torch.testing.assert_close(fn(inp), torch.full((3,), 2.0))

    def test_init_var_under_compile(self):
        # __init__ passes the InitVar to __post_init__, and from_tensordict
        # its default
        def fn(x):
            td = TensorDict(x=x, batch_size=())
            return (
                _InitVarPostInitTC(x=x, scale=3.0).x,
                _InitVarPostInitTC(x, 4.0).x,
                _InitVarPostInitTC(x=x).x,
                _InitVarPostInitTC.from_tensordict(td).x,
            )

        inp = torch.ones(3)
        expected = fn(inp)
        torch.testing.assert_close(expected, (inp * 3, inp * 4, inp, inp))
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(inp), expected)

    def test_post_init_runs_under_compile_from_tensordict(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            td = TensorDict(x=x, batch_size=())
            return _PostInitTC._from_tensordict(td).x

        inp = torch.ones(3)
        torch.testing.assert_close(fn(inp), torch.full((3,), 2.0))

    @pytest.mark.parametrize("cls", [_PostInitTC, _SkipRebuildPostInitTC])
    def test_skip_post_init_on_rebuild_under_compile(self, cls):
        # Compiled code makes the same __post_init__ calls as eager code: the
        # flag is a class attribute, which Dynamo reads as a constant
        def fn(x):
            tc = cls(x=x)
            td = TensorDict(x=x, batch_size=())
            return tc.clone().x, tc[...].x, cls.from_tensordict(td).x

        inp = torch.ones(3)
        expected = fn(inp)
        doubled = torch.full((3,), 2.0)
        rebuilt = doubled if cls is _SkipRebuildPostInitTC else doubled * 2
        torch.testing.assert_close(expected, (rebuilt, rebuilt, doubled))
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(inp), expected)


class TestTCDefaultsCompile:
    """@tensorclass field defaults must be applied under torch.compile (gh-1710)."""

    def test_concrete_default_applied_under_compile(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return _ConcreteDefaultTC().cache

        torch.testing.assert_close(_ConcreteDefaultTC().cache, torch.zeros(3))
        torch.testing.assert_close(fn(), torch.zeros(3))

    def test_default_factory_applied_under_compile(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return _FactoryDefaultTC().cache

        torch.testing.assert_close(_FactoryDefaultTC().cache, torch.zeros(3))
        torch.testing.assert_close(fn(), torch.zeros(3))

    def test_from_dataclass_default_factory_under_compile(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return _FactoryDefaultFromDataclassTC().cache

        torch.testing.assert_close(
            _FactoryDefaultFromDataclassTC().cache, torch.zeros(3)
        )
        torch.testing.assert_close(fn(), torch.zeros(3))

    def test_omitted_none_default_under_compile(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return _NoneDefaultTC(x=x).y

        inp = torch.ones(3)
        assert _NoneDefaultTC(x=inp).y is None
        assert fn(inp) is None

    def test_from_dict_and_update_under_compile(self):
        # from_dict() checks the fields with a default (#2079) without
        # breaking the graph, and a compiled call does not warn.
        @tensorclass
        class WithDefaults:
            a: torch.Tensor
            s: str = "hi"

        @torch.compile(backend="eager", fullgraph=True)
        def from_dict(x, full):
            if full:
                return WithDefaults.from_dict({"a": x, "s": "q"}, batch_size=[3])
            return WithDefaults.from_dict({"a": x}, batch_size=[3])

        @torch.compile(backend="eager", fullgraph=True)
        def update(c, x):
            c.update({"a": x})
            return c

        inp = torch.ones(3)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert from_dict(inp, True).s == "q"
            assert from_dict(inp, False).s is None
            c = update(WithDefaults(a=torch.zeros(3), s="q", batch_size=[3]), inp)
        assert (c.a == 1).all()
        assert c.s == "q"


class TestTCCustomInitCompile:
    """A user-defined __init__ runs under torch.compile (gh-1822)."""

    @pytest.mark.parametrize("tensor_only", [False, True])
    def test_custom_init(self, tensor_only):
        class Data(TensorClass, tensor_only=tensor_only):
            x: torch.Tensor

            def __init__(self, x, **options):
                self.x = x * options["scale"]

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return Data(x, scale=2.0, batch_size=[3]).x + Data(x=x, scale=3.0).x

        inp = torch.ones(3)
        torch.testing.assert_close(fn(inp), inp * 5)

    def test_custom_init_inherited(self):
        class Parent(TensorClass):
            x: torch.Tensor

            def __init__(self, x):
                self.x = x * 2

        with pytest.warns(FutureWarning, match="Child.z is shared"):

            class Child(Parent):
                y: torch.Tensor = None
                z: torch.Tensor = torch.zeros(3)
                w: torch.Tensor = dataclasses.field(
                    default_factory=lambda: torch.ones(3)
                )

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            child = Child(x, batch_size=[3])
            return child.x, child.y, child.z, child.w

        inp = torch.ones(3)
        x, y, z, w = fn(inp)
        torch.testing.assert_close(x, inp * 2)
        assert y is None
        torch.testing.assert_close(z, torch.zeros(3))
        torch.testing.assert_close(w, torch.ones(3))

    def test_custom_init_super(self):
        class Base(TensorClass):
            x: torch.Tensor

        class Child(Base):
            y: torch.Tensor

            def __init__(self, x):
                self.y = x + 1
                super().__init__(x=x, batch_size=x.shape[:1])

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            child = Child(x)
            return child.x + child.y, child.batch_size

        inp = torch.ones(3)
        out, batch_size = fn(inp)
        torch.testing.assert_close(out, inp * 3)
        assert batch_size == torch.Size([3])

    @pytest.mark.parametrize(
        "tensor_only",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.xfail(
                    version.parse("2.12.0") <= TORCH_VERSION < version.parse("2.13.0"),
                    strict=True,
                    reason=(
                        "The Dynamo of torch 2.12 guards the setter of a property "
                        "(a field of a tensor_only class) that object.__setattr__ "
                        "calls through type(property).__set__, a slot wrapper."
                    ),
                ),
            ),
        ],
    )
    def test_custom_init_frozen(self, tensor_only):
        class Data(TensorClass, frozen=True, tensor_only=tensor_only):
            x: torch.Tensor

            def __init__(self, x):
                object.__setattr__(self, "x", x * 2)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return Data(x, batch_size=[3])

        inp = torch.ones(3)
        data = fn(inp)
        torch.testing.assert_close(data.to_tensordict()["x"], inp * 2)
        assert "x" not in data.__dict__
        assert data.is_locked


# Module level: under compile, appends to a list read from the closure of a
# function set as a method are lost.
_TRACED_CALLS = []


def _count_compiles(fn, *args):
    """Compile fn, run it twice, return (frame_count_first, frame_count_second).

    Uses CompileCounterWithBackend("eager") so the function actually executes.
    """
    torch._dynamo.reset_code_caches()
    cnt = CompileCounterWithBackend("eager")
    compiled = torch.compile(fn, backend=cnt)
    compiled(*args)
    first = cnt.frame_count
    compiled(*args)
    second = cnt.frame_count
    return first, second


class TestGuardCount:
    @pytest.mark.parametrize("tensor_only", [False, True])
    def test_tc_construction_eager_and_compiled(self, tensor_only):
        @tensorclass(tensor_only=tensor_only)
        class Data:
            x: torch.Tensor
            y: torch.Tensor

        def build(x, y):
            return Data(x=x, y=y, batch_size=[3], device="cpu")

        def use(data):
            return data.x + data.y

        x, y = torch.randn(3), torch.randn(3)
        eager = build(x, y)
        compiled = torch.compile(build, backend="eager", fullgraph=True)(x, y)
        counter = CompileCounterWithBackend("eager")
        compiled_use = torch.compile(use, backend=counter, fullgraph=True)
        torch.testing.assert_close(compiled_use(eager), x + y)
        torch.testing.assert_close(compiled_use(compiled), x + y)
        assert counter.frame_count == 1

    """Tests that verify compile guard/recompile counts for optimized paths."""

    @pytest.mark.parametrize(
        "variant", ["start", "stop", "start_stop", "dim0", "batch_size"]
    )
    @pytest.mark.parametrize("tensorclass_", [False, True])
    def test_slice_dynamic_no_recompile(self, variant, tensorclass_):
        # gh-2889: a slice with a symbolic bound, or of a symbolic batch size,
        # compiles once, as slicing the leaves does
        @tensorclass
        class Data:
            x: torch.Tensor

        def fn(td, marker):
            a = marker.shape[0]
            if variant == "start":
                out = td[:, a:]
            elif variant == "stop":
                out = td[:, :a]
            elif variant == "start_stop":
                out = td[:, a : a + 4]
            elif variant == "dim0":
                out = td.transpose(0, 1)[a:]
            else:
                out = td[:, 3:]
            return out.get("x").sum(), out.batch_size

        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)
        for n in (5, 7, 9):
            width = 32 + n if variant == "batch_size" else 32
            x = torch.randn(2, width, 3)
            if tensorclass_:
                td = Data(x=x, batch_size=[2, width])
            else:
                td = TensorDict({"x": x}, batch_size=[2, width])
            marker = torch.zeros(n)
            out, batch_size = compiled(td, marker)
            expected, expected_batch_size = fn(td, marker)
            torch.testing.assert_close(out, expected)
            assert batch_size == expected_batch_size
        assert counter.frame_count == 1, f"Recompilation: {counter.frame_count} frames"

    @pytest.mark.parametrize(
        "index", [slice(None, None, -1), (slice(None), slice(2, 5, -1))]
    )
    def test_slice_dynamic_negative_step(self, index):
        # gh-2889: a TensorDict without tensor leaves takes a negative step, and
        # compiled gives the batch size that eager gives
        def fn(td):
            return td[index].batch_size

        compiled = torch.compile(fn, backend="eager", dynamic=True, fullgraph=True)
        for n in (5, 7, 9):
            td = TensorDict({}, batch_size=[n, 6])
            assert compiled(td) == fn(td)

    @pytest.mark.parametrize(
        "variant",
        ["start", "stop", "member", "stack_dim1", "nested", "in_tensordict", "stack"],
    )
    def test_slice_dynamic_lazy_stack(self, variant):
        # gh-2896: a slice of a lazy stack along a dim of its members, with a
        # symbolic bound, compiles once. A bound along the stack dim picks the
        # members, so that slice compiles once for every bound.
        def make():
            members = [
                TensorDict({"x": torch.randn(32, 3)}, batch_size=[32]) for _ in range(6)
            ]
            if variant == "stack_dim1":
                return lazy_stack(members, 1)
            if variant == "nested":
                return lazy_stack([lazy_stack(members[:3]), lazy_stack(members[3:])])
            td = lazy_stack(members)
            if variant == "in_tensordict":
                return TensorDict({"td": td}, batch_size=[6, 32])
            return td

        def fn(td, marker):
            a = marker.shape[0]
            if variant == "start":
                out = td[:, a:]
            elif variant == "stop":
                out = td[:, :a]
            elif variant == "member":
                out = td[2, a:]
            elif variant == "stack_dim1":
                out = td[a:]
            elif variant == "nested":
                out = td[:, :, a:]
            elif variant == "in_tensordict":
                out = td[:, a:]["td"]
            else:
                out = td[a - 4 :]
            return out.get("x").sum(), out.batch_size

        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)
        for n in (5, 7, 9):
            td = make()
            marker = torch.zeros(n)
            out, batch_size = compiled(td, marker)
            expected, expected_batch_size = fn(td, marker)
            torch.testing.assert_close(out, expected)
            assert batch_size == expected_batch_size
        expected_frames = 3 if variant == "stack" else 1
        assert counter.frame_count == expected_frames, (
            f"{counter.frame_count} frames, expected {expected_frames}"
        )

    def test_clone_recurse_false_no_recompile(self):
        def fn(td):
            c = td.clone(recurse=False)
            return c["a"] + 1

        td = TensorDict(
            {"a": torch.randn(4), **{f"key_{i}": torch.randn(4) for i in range(20)}},
            batch_size=[4],
        )
        first, second = _count_compiles(fn, td)
        assert first == 1, f"Expected 1 compile frame, got {first}"
        assert second == 1, f"Recompilation detected: {second} frames"

    def test_get_nested_no_recompile(self):
        @tensorclass
        class TC:
            x: torch.Tensor

        def fn(td):
            hit = td.get(("a", "b"), None)
            miss = td.get(("a", "z"), None)
            return hit + td.get(("a", "b")) + td["tc", "x"], miss

        td = TensorDict(
            {"a": {"b": torch.zeros(3)}, "tc": TC(x=torch.ones(3), batch_size=[3])},
            batch_size=[3],
        )
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for _ in range(2):
            out, miss = compiled(td)
            torch.testing.assert_close(out, torch.ones(3))
            assert miss is None
        assert counter.frame_count == 1, f"Recompilation: {counter.frame_count} frames"

    @pytest.mark.parametrize("variant", ["flat", "nested", "tensorclass"])
    def test_get_missing_key_guards(self, variant):
        # gh-2980: a get that misses guards on the key being absent, so the
        # compiled function does not keep the default once the key appears
        @tensorclass
        class TC:
            cache: TensorDict

        def fn(td):
            if variant == "flat":
                v = td["cache"].get("c", None)
            elif variant == "nested":
                v = td.get(("cache", "c"), None)
            else:
                v = td.cache.get("c", None)
            if v is None:
                return torch.full((), -1.0)
            return v.sum()

        def make(keys):
            cache = TensorDict({k: torch.ones(3) for k in keys})
            if variant == "nested":
                # the first level is missing whenever "c" is
                return TensorDict({"cache" if "c" in keys else "other": cache})
            if variant == "tensorclass":
                return TC(cache=cache)
            return TensorDict(cache=cache)

        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for keys in [[], ["c"], ["a"], ["c", "a"]]:
            td = make(keys)
            torch.testing.assert_close(compiled(td), fn(td))
        assert counter.frame_count == 2, f"{counter.frame_count} frames"

    @pytest.mark.parametrize("kind", ["memmap", "shared"])
    def test_slice_dynamic_memmap_shared_no_recompile(self, kind, tmpdir):
        # gh-2899: a memory-mapped or shared tensordict also compiles a slice
        # with a symbolic bound once, and keeps its flags as in eager
        def fn(td, marker):
            out = td[marker.shape[0] :]
            return out.get("x").sum(), out._is_memmap, out._is_shared

        td = TensorDict({"x": torch.randn(32, 3)}, batch_size=[32])
        if kind == "memmap":
            td = td.memmap_(tmpdir)
        else:
            td = td.share_memory_()
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)
        for n in (5, 7, 9):
            marker = torch.zeros(n)
            out, is_memmap, is_shared = compiled(td, marker)
            expected, expected_memmap, expected_shared = fn(td, marker)
            torch.testing.assert_close(out, expected)
            assert (is_memmap, is_shared) == (expected_memmap, expected_shared)
        assert counter.frame_count == 1, f"Recompilation: {counter.frame_count} frames"

    def test_contains_key_order_no_recompile(self):
        # `key in td` guards on the key, not on the order of the keys (#2892)
        def fn(td):
            out = td["x"]
            if "w" in td:
                out = out * td["w"]
            if ("n", "w") in td:
                out = out + td["n", "w"]
            if ("n", "m", "w") in td:
                out = out + td["n", "m", "w"]
            if "z" in td or ("n", "z") in td:
                out = out - 1
            return out

        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for order in ["xwy", "wxy", "yxw", "xyw"]:
            m = {k: torch.ones(3) for k in order}
            n = {**m, "m": m}
            td = TensorDict({**m, "n": n}, batch_size=[3])
            torch.testing.assert_close(compiled(td), fn(td))
        assert counter.frame_count == 1, f"Recompilation: {counter.frame_count} frames"

    def test_tc_getattr_no_recompile(self):
        class BigTC(TensorClass["nocast"]):
            a: torch.Tensor
            b: torch.Tensor
            c: torch.Tensor
            d: torch.Tensor
            e: torch.Tensor

        def fn(tc):
            return tc.a + tc.b + tc.c + tc.d + tc.e

        tc = BigTC(
            a=torch.randn(4),
            b=torch.randn(4),
            c=torch.randn(4),
            d=torch.randn(4),
            e=torch.randn(4),
            batch_size=[4],
        )
        first, second = _count_compiles(fn, tc)
        assert first == 1, f"Expected 1 compile frame, got {first}"
        assert second == 1, f"Recompilation detected: {second} frames"

    def test_replace_no_recompile(self):
        class State(TensorClass["nocast"]):
            x: torch.Tensor
            y: torch.Tensor
            z: torch.Tensor

        def fn(s):
            s = s.replace(x=s.x + 1)
            s = s.replace(y=s.y + 2)
            s = s.replace(x=s.x + s.y, z=s.z + 1)
            return s

        s = State(
            x=torch.randn(4),
            y=torch.randn(4),
            z=torch.randn(4),
            batch_size=[4],
        )
        first, second = _count_compiles(fn, s)
        assert first == 1, f"Expected 1 compile frame, got {first}"
        assert second == 1, f"Recompilation detected: {second} frames"

    @pytest.mark.parametrize("method", ["update_", "update", "set"])
    def test_update_inplace_no_recompile(self, method):
        def fn(td, src):
            # Replacement must not change aliasing between the two inputs.
            src = src + 1
            if method == "set":
                td.set("a", src["a"])
                td.set("b", src["b"])
            else:
                getattr(td, method)(src)
            return td["a"] + 0

        td = TensorDict(
            {"a": torch.randn(4), "b": torch.randn(4)},
            batch_size=[4],
        )
        src = TensorDict(
            {"a": torch.ones(4), "b": torch.ones(4)},
            batch_size=[4],
        )
        first, second = _count_compiles(fn, td, src)
        assert first == 1, f"Expected 1 compile frame, got {first}"
        assert second == 1, f"Recompilation detected: {second} frames"

    def test_unbatched_clone_no_recompile(self):
        def fn(td):
            c = td.clone()
            return c["a"] + 0

        td = TensorDict(
            {
                "a": torch.randn(4, 3),
                "unbatched": UnbatchedTensor(torch.randn(5)),
            },
            batch_size=[4],
        )
        first, second = _count_compiles(fn, td)
        assert first == 1, f"Expected 1 compile frame, got {first}"
        assert second == 1, f"Recompilation detected: {second} frames"

    def test_unbatched_vmap_clone_preserves_semantics(self):
        """Compiled vmap preserves unbatched metadata and clones the payload."""

        def fn(td):
            cloned = td.clone()
            return cloned

        td = TensorDict(
            {
                "a": torch.randn(4, 3),
                "unbatched": UnbatchedTensor(torch.randn(5)),
            },
            batch_size=[4],
        )
        fn_c = torch.compile(torch.vmap(fn), fullgraph=True)
        result = fn_c(td)
        ut_orig = td.get("unbatched")
        ut_clone = result.get("unbatched")
        torch.testing.assert_close(result["a"], td["a"])
        torch.testing.assert_close(ut_clone, ut_orig)
        assert ut_clone.batch_size == td.batch_size
        assert ut_clone.data_ptr() != ut_orig.data_ptr(), (
            "clone() must produce independent data"
        )

    @pytest.mark.skipif(
        not _HAS_WRAPPER_SUBCLASS_FIX,
        reason="The fallback UnbatchedTensor is not a wrapper subclass.",
    )
    def test_unbatched_aot_autograd_cache_key(self):
        """UnbatchedTensor gives the AOTAutograd cache a key without a warning.

        Equal inputs hit the cache and another batch size misses it.
        """

        def fn(u):
            return u * 2

        def compile_and_count(u):
            torch._dynamo.reset()
            counters.clear()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = torch.compile(fn)(u)
            torch.testing.assert_close(out, u * 2)
            assert not [
                w for w in caught if "_stable_hash_for_caching" in str(w.message)
            ]
            aot_counters = counters["aot_autograd"]
            return (
                aot_counters["autograd_cache_hit"],
                aot_counters["autograd_cache_miss"],
            )

        data = torch.randn(5)
        with (
            fresh_cache(),
            torch._functorch.config.patch(enable_autograd_cache=True),
        ):
            first = compile_and_count(UnbatchedTensor(data, batch_size=[4]))
            equal = compile_and_count(UnbatchedTensor(data.clone(), batch_size=[4]))
            other_batch_size = compile_and_count(UnbatchedTensor(data, batch_size=[6]))
        assert first == (0, 1)
        assert equal == (1, 0)
        assert other_batch_size == (0, 1)

    @pytest.mark.skipif(
        not _HAS_WRAPPER_SUBCLASS_FIX,
        reason="The fallback UnbatchedTensor is not a wrapper subclass.",
    )
    def test_unbatched_aot_autograd_cache_key_subclass_payload(self):
        """The cache key of an UnbatchedTensor covers a wrapper subclass payload."""

        class TaggedTwoTensor(TwoTensor):
            def _stable_hash_for_caching(self):
                return self.tag

        def key(data):
            return UnbatchedTensor(data, batch_size=[4])._stable_hash_for_caching()

        def tagged(tag, a):
            out = TaggedTwoTensor(a, a.clone())
            out.tag = tag
            return out

        a = torch.randn(5)
        two = key(TwoTensor(a, a.clone()))
        assert two == key(TwoTensor(torch.randn(5), torch.randn(5)))
        assert two != key(TwoTensor(a.double(), a.double()))
        assert two != key(a)
        # A payload with its own stable hash is keyed by that hash.
        assert key(tagged("x", a)) == key(tagged("x", a.double()))
        assert key(tagged("x", a)) != key(tagged("y", a))

    def test_lock_inside_compile_no_weakref_leftover(self):
        """``lock_()`` called inside a compiled region must not leave a
        ``weakref`` behind in ``_last_op``.

        The ``_as_context_manager`` decorator that wraps ``lock_`` would
        normally create a ``weakref.ref(self)`` and write it to
        ``_last_op`` on the locked TD. Under compile, that ``WeakRef``
        leaks into the returned TD and trips
        ``Unsupported: reconstruct: WeakRefVariable()`` patterns. The
        decorator must use a strong-ref closure under ``is_compiling()``
        so the same bookkeeping still works without weakrefs.
        """
        import weakref as _wref

        def fn(td):
            # Force actual compute so Dynamo doesn't trivially return td.
            td["c"] = td["a"] + td["b"]
            td.lock_()
            return td

        td = TensorDict({"a": torch.randn(4), "b": torch.randn(4)}, batch_size=[4])

        torch._dynamo.reset_code_caches()
        compiled = torch.compile(fn, fullgraph=True, backend="eager")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            out = compiled(td)
        assert out.is_locked
        # If `_last_op` was written at all, it must not contain a weakref
        # (which Dynamo can't reconstruct cleanly across compile
        # boundaries). A strong-ref callable closure is fine.
        last_op = out.__dict__.get("_last_op")
        if last_op is not None:
            _, (_, _, ref) = last_op
            assert not isinstance(ref, _wref.ref), (
                f"weakref leaked into _last_op under compile: {ref}"
            )
            assert callable(ref) and ref() is out, (
                "strong-ref closure must still resolve to the locked TD"
            )

    def test_locked_td_no_recompile(self):
        """A TD locked in eager mode that flows through compile must
        produce stable compile frames.
        """

        def fn(td):
            return td["a"] + td["b"]

        td1 = TensorDict({"a": torch.randn(4), "b": torch.randn(4)}, batch_size=[4])
        td1.lock_()
        td2 = TensorDict({"a": torch.randn(4), "b": torch.randn(4)}, batch_size=[4])
        td2.lock_()

        torch._dynamo.reset_code_caches()
        cnt = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=cnt, fullgraph=True)
        compiled(td1)
        first = cnt.frame_count
        compiled(td2)
        second = cnt.frame_count
        assert first == 1, f"Expected 1 compile frame, got {first}"
        assert second == 1, f"Recompilation detected: {second} frames"

    def test_getitem_str_key_guards(self):
        """``td["k"]`` guards little more than reading the entry from the dict."""

        def read_td(td):
            return td["a"] + td["sub"]["b"]

        def read_dict(td):
            return td._tensordict["a"] + td._tensordict["sub"]._tensordict["b"]

        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        def guards(fn, td):
            torch._dynamo.reset()
            cnt = CompileCounterWithBackend("eager")
            torch.testing.assert_close(
                torch.compile(fn, backend=cnt, fullgraph=True)(td), fn(td)
            )
            assert cnt.frame_count == 1
            (entry,) = _debug_get_cache_entry_list(fn.__code__)
            tree = str(entry.guard_manager)
            leaves = [
                line
                for line in tree.splitlines()
                if "+- " in line and "GuardManager" not in line.split("+- ")[1][:30]
            ]
            return len(leaves), tree

        td = TensorDict(
            a=torch.zeros(3),
            sub=TensorDict(b=torch.ones(3), batch_size=[3]),
            batch_size=[3],
        )
        n_td, tree = guards(read_td, td)
        n_dict, _ = guards(read_dict, td)
        # The generic path guards each method it calls (_get_tuple_maybe_non_tensor,
        # _get_tuple, _pass_through...). torch 2.14: 22 guards for the dict
        # reads, 49 on the generic path, 33 here.
        assert "_get_tuple_maybe_non_tensor" not in tree
        assert n_td <= n_dict + 18, (n_td, n_dict)

    def test_locked_clone_recurse_false_schema_fast_path(self):
        """A locked TD's ``clone(recurse=False)`` must:

        - take the schema-driven fast path inside compile (i.e. produce a
          shallow clone whose leaves are *the same tensors* as the
          source), and
        - not trigger recompiles across calls with the same schema.

        The fast path is what avoids the
        ``len(_tensordict) != N`` / ``list(dict.keys(_tensordict))[i] == ...``
        guards that the dict-walking path would otherwise produce.
        """

        def fn(td):
            c = td.clone(recurse=False)
            return c["a"] + c["b"] + c["c"]

        td1 = TensorDict(
            {"a": torch.randn(4), "b": torch.randn(4), "c": torch.randn(4)},
            batch_size=[4],
        )
        td1.lock_()
        # The schema must be populated by lock_().
        assert td1._locked_schema is not None
        assert td1._locked_schema.keys == ("a", "b", "c")

        td2 = TensorDict(
            {"a": torch.randn(4), "b": torch.randn(4), "c": torch.randn(4)},
            batch_size=[4],
        )
        td2.lock_()

        torch._dynamo.reset_code_caches()
        cnt = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=cnt, fullgraph=True)
        compiled(td1)
        first = cnt.frame_count
        compiled(td2)
        second = cnt.frame_count
        assert first == 1, f"Expected 1 compile frame, got {first}"
        assert second == 1, f"Recompilation detected: {second} frames"

    def test_locked_schema_cleared_on_unlock(self):
        """unlock_() must drop the locked schema; relock rebuilds it."""

        td = TensorDict({"a": torch.randn(4), "b": torch.randn(4)}, batch_size=[4])
        td.lock_()
        first = td._locked_schema
        assert first is not None
        td.unlock_()
        assert td._locked_schema is None
        td.lock_()
        # New schema instance, same keys.
        assert td._locked_schema is not None
        assert td._locked_schema.keys == first.keys

    def test_td_dim_names_always_on_instance(self):
        """`_td_dim_names` must always live in instance ``__dict__``.

        A TD constructed via the public ``TensorDict(...)`` constructor
        inside a compiled region (the hot path in torchrl env stepping,
        where the env builds a fresh ``next`` TD on every step) must end
        up with ``_td_dim_names`` in its instance ``__dict__``. Otherwise,
        when that TD flows back into a compiled region on the next
        iteration, Dynamo recompiles on
        ``not ___dict_contains('_td_dim_names', __dict__)``.
        """

        def make_inside_compile(seed):
            return TensorDict(
                {"a": seed + 1, "b": seed + 2},
                batch_size=seed.shape[:1],
            )

        seed = torch.randn(4)
        compiled = torch.compile(make_inside_compile, fullgraph=True, backend="eager")
        td_from_compile = compiled(seed)

        # The TD that came out of compile must have _td_dim_names on its
        # instance dict, exactly like a TD built in eager mode.
        td_from_eager = TensorDict(
            {"a": seed + 1, "b": seed + 2},
            batch_size=seed.shape[:1],
        )
        assert "_td_dim_names" in td_from_compile.__dict__, (
            "_td_dim_names must live on instance dict (got from compile-time __init__)"
        )
        assert "_td_dim_names" in td_from_eager.__dict__

        # And then feeding that compile-built TD back into a compiled
        # function must not trigger a recompile relative to the eager TD.
        def use_td(td):
            return td["a"] + td["b"]

        torch._dynamo.reset_code_caches()
        cnt = CompileCounterWithBackend("eager")
        compiled_use = torch.compile(use_td, backend=cnt)
        compiled_use(td_from_eager)
        first = cnt.frame_count
        compiled_use(td_from_compile)
        second = cnt.frame_count
        assert first == 1, f"Expected 1 compile frame, got {first}"
        assert second == 1, (
            f"Mixing eager-built and compile-built TDs recompiled: {second} frames"
        )

    def test_new_type_in_eager_no_recompile(self):
        # The type predicates must not read their memo under Dynamo: a lookup
        # that misses guards on all the keys of the memo, which grows each
        # time eager code checks a new type.
        def fn(td):
            return td.apply(lambda x: x + 1)

        td = TensorDict(a=torch.randn(4), b=torch.randn(4, 3), batch_size=[4])
        torch._dynamo.reset_code_caches()
        cnt = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=cnt, fullgraph=True)
        compiled(td)
        for i in range(3):
            is_tensor_collection(type(f"_NewType{i}", (), {}))
            compiled(td)
        assert cnt.frame_count == 1, f"Recompilation detected: {cnt.frame_count}"

    @pytest.mark.parametrize("op", ["set_scalar", "update_at_", "autocast"])
    def test_new_tensorclass_in_eager_no_recompile(self, op):
        # Defining a tensorclass rebinds tensordict.base._ACCEPTED_CLASSES,
        # which compiled frames must not guard on.
        @tensorclass(autocast=True)
        class AutoCast:
            x: torch.Tensor
            y: float

        def fn(obj):
            if op == "set_scalar":
                obj["c"] = 3.0
                return obj["a"] + obj["c"]
            if op == "update_at_":
                obj.update_at_({"a": torch.ones(())}, 0)
                return obj["a"] + 1
            obj.x = [1.0, 2.0, 3.0]
            obj.y = 2
            return obj.x + 1

        def make():
            if op == "set_scalar":
                return TensorDict(a=torch.zeros(3))
            if op == "update_at_":
                return TensorDict(a=torch.zeros(3), batch_size=[3])
            return AutoCast(x=torch.zeros(3), y=1.0)

        torch._dynamo.reset_code_caches()
        cnt = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=cnt, fullgraph=True)
        compiled(make())
        for i in range(2):
            tensorclass(
                type(f"_NewTC{i}", (), {"__annotations__": {"x": torch.Tensor}})
            )
            compiled(make())
        assert cnt.frame_count == 1, f"Recompilation detected: {cnt.frame_count}"

    def test_autocast_new_type_in_eager_no_recompile(self):
        # Autocasting a field to a type that was never cast to before must not
        # recompile the compiled frames that autocast.
        class NewFloat(float):
            pass

        @tensorclass(autocast=True)
        class AutoCast:
            x: torch.Tensor

        @tensorclass(autocast=True)
        class AutoCastNewFloat:
            z: NewFloat

        def fn(obj):
            obj.x = [1.0, 2.0, 3.0]
            return obj.x + 1

        obj = AutoCast(x=torch.zeros(3))
        torch._dynamo.reset_code_caches()
        cnt = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=cnt, fullgraph=True)
        compiled(obj)
        assert type(AutoCastNewFloat(z=1.0).z) is NewFloat
        compiled(obj)
        assert cnt.frame_count == 1, f"Recompilation detected: {cnt.frame_count}"

    def test_autocast_tensorclass_annotation_eager_and_compiled(self):
        # The bare TensorClass base is an accepted class in eager mode as under
        # compile, so a field annotated with it stores the instance in both.
        class Inner(TensorClass):
            x: torch.Tensor

        @tensorclass(autocast=True)
        class Outer:
            inner: TensorClass
            y: torch.Tensor

        def fn(obj, value):
            obj.inner = value
            return obj.y + 1

        for compiled in (False, True):
            obj = Outer(inner=Inner(x=torch.zeros(3)), y=torch.zeros(3))
            value = Inner(x=torch.ones(3))
            if compiled:
                torch._dynamo.reset_code_caches()
                torch.compile(fn, backend="eager", fullgraph=True)(obj, value)
            else:
                fn(obj, value)
            assert obj.inner is value
            assert "inner" in obj._tensordict.keys()

    def test_eager_flatten_of_new_td_type_no_recompile(self):
        # The pytree flatten of a td picks its constructor without a module-level
        # dict that eager flattens of new td types would grow.
        class FlatState(TypedTensorDict):
            x: torch.Tensor

        def fn(td):
            return tree_map(lambda x: x + 1, td)["a"]

        td = TensorDict(a=torch.zeros(4), batch_size=[4])
        torch._dynamo.reset_code_caches()
        cnt = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=cnt, fullgraph=True)
        compiled(td)
        leaves, spec = tree_flatten(FlatState(x=torch.zeros(4), batch_size=[4]))
        assert type(tree_unflatten(leaves, spec)) is FlatState
        compiled(td)
        assert cnt.frame_count == 1, f"Recompilation detected: {cnt.frame_count}"

    def test_lazy_del_new_stack_no_recompile(self):
        # del_ on a lazy stack must not guard on the id() of its members.
        def make():
            return lazy_stack(
                [
                    TensorDict(a=torch.zeros(3), b=torch.zeros(3), batch_size=[3])
                    for _ in range(2)
                ]
            )

        def fn(td):
            del td["a"]
            return td["b"] + 1

        torch._dynamo.reset_code_caches()
        cnt = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=cnt, fullgraph=True)
        for _ in range(3):
            td = make()
            compiled(td)
            assert "a" not in td.keys()
        assert cnt.frame_count == 1, f"Recompilation detected: {cnt.frame_count}"

    @staticmethod
    def _build_and_set():
        @tensorclass(tensor_only=True)
        class Data:
            a: torch.Tensor
            td: TensorDict

        def fn(x, td, extra):
            batch_size = x.shape[:1]
            out = TensorDict(
                {"a": x, "nested": TensorDict({"b": x + 1}, batch_size)}, batch_size
            )
            out["c"] = x * 2
            out["nested", "d"] = x - 1
            for key, value in extra.items():
                out[key] = value
            data = Data(a=x, td=out, batch_size=batch_size)
            data.a = x + 3
            data.td = extra
            td["cache", "y"] = td["x"] * 2
            return out, data, td

        def make(n, full, batched=True):
            x = torch.arange(float(n))
            batch_size = [n] if batched else []
            td = TensorDict(
                {"x": x, "cache": TensorDict(batch_size=batch_size)}, batch_size
            )
            extra = {"e": x * 5, "f": {"g": x}} if full else {}
            return x, td, extra

        def flat(result):
            out, data, td = result
            return [
                (type(obj).__name__, tuple(obj.batch_size), key, val.tolist())
                for obj in (out, data.td, td)
                for key, val in obj.items(True, True)
            ] + [data.a.tolist(), type(data).__name__]

        return fn, make, flat

    def test_set_matching_tensor_skips_validation(self, monkeypatch):
        # Exact tensors whose shape and device match are stored without
        # tracing the validators, as in eager mode, and a tensorclass
        # attribute write traces neither _convert_mapping_for_field nor
        # is_non_tensor.
        tc_module = sys.modules["tensordict.tensorclass"]
        calls = _TRACED_CALLS
        calls.clear()
        validate = TensorDictBase._validate_value_generic
        convert = tc_module._convert_mapping_for_field
        non_tensor = tc_module.is_non_tensor

        def record_validate(self, value, *args, **kwargs):
            _TRACED_CALLS.append(("validate", type(value).__name__, kwargs.get("key")))
            return validate(self, value, *args, **kwargs)

        def record_convert(key, *args):
            _TRACED_CALLS.append(("convert", key))
            return convert(key, *args)

        def record_non_tensor(obj):
            _TRACED_CALLS.append(("is_non_tensor",))
            return non_tensor(obj)

        monkeypatch.setattr(TensorDictBase, "_validate_value_generic", record_validate)
        monkeypatch.setattr(tc_module, "_convert_mapping_for_field", record_convert)
        monkeypatch.setattr(tc_module, "is_non_tensor", record_non_tensor)
        fn, make, flat = self._build_and_set()
        expected = flat(fn(*make(3, True)))
        torch._dynamo.reset_code_caches()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        calls.clear()
        assert flat(compiled(*make(3, True))) == expected
        # Only the tensorclass constructor validates its tensors
        assert [call for call in calls if call[:2] == ("validate", "Tensor")] == [
            ("validate", "Tensor", "a")
        ]
        assert [call for call in calls if call[0] != "validate"] == [
            ("convert", "a"),
            ("convert", "td"),
        ]
        calls.clear()

    def test_set_no_recompile_across_sizes(self):
        fn, make, flat = self._build_and_set()
        torch._dynamo.reset_code_caches()
        cnt = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=cnt, fullgraph=True, dynamic=None)
        frames = []
        for n in (3, 4, 5):
            assert flat(compiled(*make(n, True))) == flat(fn(*make(n, True)))
            frames.append(cnt.frame_count)
        # The second size makes the graph dynamic, the third reuses it
        assert frames == [1, 2, 2]

    def test_set_empty_and_full_sources(self):
        # Results match eager when the sources and batch sizes alternate
        # between empty and non-empty.
        fn, make, flat = self._build_and_set()
        torch._dynamo.reset_code_caches()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        for n, full, batched in (
            (3, False, True),
            (4, True, True),
            (5, False, False),
            (6, True, False),
            (3, True, True),
            (4, False, True),
            (5, True, False),
            (6, False, False),
        ):
            args = make(n, full, batched)
            assert flat(compiled(*args)) == flat(fn(*make(n, full, batched)))

    @pytest.mark.parametrize(
        "value",
        ["tensor", "parameter", "short", "mismatch", "meta", "td", "str", "dict"],
    )
    @pytest.mark.parametrize("device", [None, "cpu"])
    def test_set_validation_compiled_matches_eager(self, value, device):
        # Values that need validation are still validated under compile
        def fn(x, param):
            td = TensorDict(batch_size=[2, 3], device=device, names=["a", None])
            if value == "tensor":
                td.set("v", x)
            elif value == "parameter":
                td.set("v", param)
            elif value == "short":
                td.set("v", x[0])
            elif value == "mismatch":
                td.set("v", x.transpose(0, 1))
            elif value == "meta":
                td.set("v", x.to("meta"))
            elif value == "td":
                td.set("v", TensorDict({"y": x.unsqueeze(-1)}, [2, 3, 1]))
            elif value == "str":
                td.set("v", "abc")
            else:
                td.set("v", {"y": x})
            td["w"] = x + 1
            return td

        def describe(f, x, param):
            try:
                td = f(x, param)
            except Exception as err:
                return type(err)
            return [
                (key, type(val).__name__, tuple(val.shape), str(val.device))
                for key, val in td.items(True, True, is_leaf=is_leaf_nontensor)
            ] + [td.names, td["v"].names if value == "td" else None]

        x, param = torch.zeros(2, 3), torch.nn.Parameter(torch.zeros(2, 3))
        compiled = torch.compile(fn, backend="eager")
        assert describe(compiled, x, param) == describe(fn, x, param)


class TestNestedCompileRegion:
    def test_nested_compile_region_td(self):
        """TensorDict can be passed through nested_compile_region (gh-1667)."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, td):
                acc = torch.zeros(td["x"].shape[0])
                for _ in range(3):
                    td, step = self._step(td)
                    acc = acc + step
                return acc

            @torch.compiler.nested_compile_region
            def _step(self, td):
                x = self.linear(td["x"])
                td = td.clone()
                td["x"] = x
                return td, x.sum(dim=-1)

        model = Model()
        td = TensorDict({"x": torch.randn(2, 4)}, batch_size=[2])

        eager_result = model(td)
        compiled = torch.compile(model)
        compiled_result = compiled(td)
        torch.testing.assert_close(eager_result, compiled_result)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
