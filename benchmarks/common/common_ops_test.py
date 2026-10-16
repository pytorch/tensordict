# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import itertools

import pytest
import torch
from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    NonTensorStack,
    TensorDict,
)
from tensordict.utils import _unravel_key_to_tuple, unravel_key, unravel_key_list


@pytest.fixture
def a():
    return torch.zeros(3, 4, 5)


@pytest.fixture
def b():
    return torch.zeros(3, 4, 5)


@pytest.fixture
def c():
    return torch.zeros(3, 4, 5)


@pytest.fixture
def td(a, b):
    return TensorDict({"a": a, "b": {"b1": b}}, [3, 4])


@pytest.mark.parametrize("num_fields", [4, 16, 64, 256])
@pytest.mark.parametrize("batch_size", [(), (32,)])
@pytest.mark.parametrize("device", [None, "cpu"])
def test_construct_tensor_leaves(benchmark, num_fields, batch_size, device):
    source = {f"f{i}": torch.empty(32, 8) for i in range(num_fields)}
    benchmark(TensorDict, source, batch_size=batch_size, device=device)


@pytest.mark.parametrize("num_fields", [16, 256])
@pytest.mark.parametrize("source_type", ["dict", "tuple_keys", "ready_children"])
@pytest.mark.parametrize("method", ["constructor", "set", "update"])
def test_build_nested_tensor_leaves(benchmark, num_fields, source_type, method):
    leaves = {f"f{i}": torch.empty(32, 8) for i in range(num_fields)}
    if source_type == "tuple_keys":
        source = {
            (f"g{i // 4}", key): value for i, (key, value) in enumerate(leaves.items())
        }
    else:
        items = list(leaves.items())
        source = {f"g{i // 4}": dict(items[i : i + 4]) for i in range(0, num_fields, 4)}
        if source_type == "ready_children":
            source = {
                key: TensorDict(value, [32], device="cpu")
                for key, value in source.items()
            }

    if method == "constructor":
        result = benchmark(TensorDict, source, batch_size=[32], device="cpu")
    else:

        def build():
            # Construct an empty destination each time so warmup cannot turn
            # this into a benchmark of updating an already populated tree.
            result = TensorDict({}, [32], device="cpu")
            if method == "set":
                for key, value in source.items():
                    result.set(key, value)
            else:
                result.update(source)
            return result

        result = benchmark(build)
    assert result["g0", "f0"] is leaves["f0"]


def big_td():
    return (
        (TensorDict({str(i): torch.zeros(3, 4) + i for i in range(100)}, [3, 4]),),
        {},
    )


def big_nested_td(size=(3, 4)):
    return (
        (
            TensorDict(
                {
                    ".".join([str(j) for j in range(i)] + ["t"]): torch.zeros(size) + i
                    for i in range(1, 20)
                },
                size,
            ).unflatten_keys("."),
        ),
        {},
    )


def big_nested_td_locked():
    return ((big_nested_td()[0][0].lock_(),), {})


def big_nested_stacked_td():
    return (
        (
            torch.stack(
                TensorDict(
                    {
                        ".".join([str(j) for j in range(i)] + ["t"]): torch.zeros(3, 10)
                        + i
                        for i in range(1, 20)
                    },
                    batch_size=[3, 10],
                )
                .unflatten_keys(".")
                .unbind(1),
                1,
            ),
        ),
        {},
    )


def big_nested_stacked_td_locked():
    return ((big_nested_stacked_td()[0][0].lock_(),), {})


def big_nested_lazy_stacked_td():
    # big_nested_stacked_td is a TensorDict (torch.stack stacks densely)
    return (
        (LazyStackedTensorDict.lazy_stack(big_nested_td((3, 10))[0][0].unbind(1), 1),),
        {},
    )


def test_plain_set_nested(benchmark):
    td = big_nested_td()[0][0]
    key = tuple(str(j) for j in range(20, 0, -1))
    tensor = torch.zeros(3, 4)
    benchmark(lambda: td.set(key, tensor))


def test_plain_set_stack_nested(benchmark):
    td = big_nested_stacked_td()[0][0]
    key = tuple(str(j) for j in range(20, 0, -1))
    tensor = torch.zeros(td.shape)
    benchmark(lambda: td.set(key, tensor))


def test_plain_set_nested_inplace(benchmark):
    td = big_nested_td()[0][0]
    key = tuple(str(j) for j in range(20, 0, -1))
    tensor = torch.zeros(3, 4)
    td.set(key, tensor)
    benchmark(lambda: td.set(key, tensor, inplace=True))


def test_plain_set_stack_nested_inplace(benchmark):
    td = big_nested_stacked_td()[0][0]
    key = tuple(str(j) for j in range(20, 0, -1))
    tensor = torch.zeros(td.shape)
    td.set(key, tensor)
    benchmark(lambda: td.set(key, tensor, inplace=True))


def test_plain_set_td_batchfree(benchmark):
    # the storage of a tensorclass: no batch dims and no device
    td = TensorDict()
    sub = TensorDict({"x": torch.zeros(1000)}, [1000])
    benchmark(lambda: td.set("sub", sub))


def test_items(benchmark):
    td = big_td()[0][0]
    benchmark(lambda: list(td.items()))


def test_items_nested(benchmark):
    td = big_nested_td()[0][0]
    benchmark(lambda: list(td.items(True)))


def test_items_nested_locked(benchmark):
    td = big_nested_td_locked()[0][0]
    list(td.items(True))
    benchmark(lambda: list(td.items(True)))


def test_items_nested_leaf(benchmark):
    td = big_nested_td()[0][0]
    benchmark(lambda: list(td.items(True, True)))


def test_items_stack_nested(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark(lambda: list(td.items(True)))


def test_items_stack_nested_leaf(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark(lambda: list(td.items(True, True)))


def test_items_stack_nested_locked(benchmark):
    td = big_nested_stacked_td_locked()[0][0]
    list(td.items(True))
    benchmark(lambda: list(td.items(True)))


def test_keys(benchmark):
    td = big_td()[0][0]
    benchmark(lambda: list(td.keys()))


def test_keys_nested(benchmark):
    td = big_nested_td()[0][0]
    benchmark(lambda: list(td.keys(True)))


def test_keys_nested_locked(benchmark):
    td = big_nested_td_locked()[0][0]
    list(td.keys(True))
    benchmark(lambda: list(td.keys(True)))


def test_keys_nested_leaf(benchmark):
    td = big_nested_td()[0][0]
    benchmark(lambda: list(td.keys(True, True)))


def test_keys_nested_leaf_loop(benchmark):
    td = big_nested_td()[0][0]
    benchmark(lambda: [key for key in td.keys(True, True)])  # noqa: C416


def _small_nested_td():
    # A few leaves on three levels, as in the data of a mesh
    return TensorDict(
        {
            "a": torch.zeros(3),
            "b": torch.zeros(3),
            "nested": {"c": torch.zeros(3), "d": {"e": torch.zeros(3)}},
        },
        batch_size=[3],
    )


def test_keys_nested_leaf_small(benchmark):
    td = _small_nested_td()
    benchmark(lambda: list(td.keys(True, True)))


def test_keys_nested_leaf_small_len(benchmark):
    td = _small_nested_td()
    benchmark(lambda: len(td.keys(True, True)))


def test_keys_stack_nested(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark(lambda: list(td.keys(True)))


def test_keys_stack_nested_leaf(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark(lambda: list(td.keys(True, True)))


def test_keys_stack_nested_locked(benchmark):
    td = big_nested_stacked_td_locked()[0][0]
    list(td.keys(True))
    benchmark(lambda: list(td.keys(True)))


def test_keys_lazy_stack_nested(benchmark):
    td = big_nested_lazy_stacked_td()[0][0]
    benchmark(lambda: list(td.keys(True)))


def test_keys_lazy_stack_nested_leaf(benchmark):
    td = big_nested_lazy_stacked_td()[0][0]
    benchmark(lambda: list(td.keys(True, True)))


def test_keys_lazy_stack_nested_leaf_loop(benchmark):
    td = big_nested_lazy_stacked_td()[0][0]
    benchmark(lambda: [key for key in td.keys(True, True)])  # noqa: C416


def test_values(benchmark):
    td = big_td()[0][0]
    benchmark(lambda: list(td.values()))


def test_values_nested(benchmark):
    td = big_nested_td()[0][0]
    benchmark(lambda: list(td.values(True)))


def test_values_nested_locked(benchmark):
    td = big_nested_td_locked()[0][0]
    list(td.values(True))
    benchmark(lambda: list(td.values(True)))


def test_values_nested_leaf(benchmark):
    td = big_nested_td()[0][0]
    benchmark(lambda: list(td.values(True, True)))


def test_values_stack_nested(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark(lambda: list(td.values(True)))


def test_values_stack_nested_leaf(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark(lambda: list(td.values(True, True)))


def test_values_stack_nested_locked(benchmark):
    td = big_nested_stacked_td_locked()[0][0]
    list(td.values(True))
    benchmark(lambda: list(td.values(True)))


def test_membership(benchmark):
    td = big_td()[0][0]
    benchmark(lambda: "a" in td.keys())


def test_membership_nested(benchmark):
    td = big_nested_td()[0][0]
    benchmark(lambda: ("a",) in td.keys(True))


def test_membership_nested_leaf(benchmark):
    td = big_nested_td()[0][0]
    benchmark(lambda: ("a",) in td.keys(True, True))


def test_membership_stacked_nested(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark(lambda: ("a",) in td.keys(True))


def test_membership_stacked_nested_leaf(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark(lambda: ("a",) in td.keys(True, True))


def test_membership_nested_last(benchmark):
    td = big_nested_td()[0][0]
    subtd = td
    key = []
    while True:
        for _key, value in subtd.items():
            key += [_key]
            if is_tensor_collection(value):
                subtd = value
                break
            else:
                subtd = None
                break
        if subtd is None:
            break
    key = tuple(key)
    benchmark(lambda: key in td.keys(True))


def test_membership_nested_leaf_last(benchmark):
    td = big_nested_td()[0][0]
    subtd = td
    key = []
    while True:
        for _key, value in subtd.items():
            key += [_key]
            if is_tensor_collection(value):
                subtd = value
                break
            else:
                subtd = None
                break
        if subtd is None:
            break
    key = tuple(key)
    benchmark(lambda: key in td.keys(True, True))


def test_membership_stacked_nested_last(benchmark):
    td = big_nested_stacked_td()[0][0]
    subtd = td
    key = []
    while True:
        for _key, value in subtd.items():
            key += [_key]
            if is_tensor_collection(value):
                subtd = value
                break
            else:
                subtd = None
                break
        if subtd is None:
            break
    key = tuple(key)
    benchmark(lambda: key in td.keys(True))


def test_membership_stacked_nested_leaf_last(benchmark):
    td = big_nested_stacked_td()[0][0]
    subtd = td
    key = []
    while True:
        for _key, value in subtd.items():
            key += [_key]
            if is_tensor_collection(value):
                subtd = value
                break
            else:
                subtd = None
                break
        if subtd is None:
            break
    key = tuple(key)
    benchmark(lambda: key in td.keys(True, True))


def test_nested_getleaf(benchmark):
    td = big_nested_td()[0][0]
    key = tuple(str(i) for i in range(19)) + ("t",)
    benchmark(lambda: td.get(key))


def test_nested_get(benchmark):
    td = big_nested_td()[0][0]
    key = tuple(str(i) for i in range(19))
    benchmark(lambda: td.get(key))


def test_stacked_getleaf(benchmark):
    td = big_nested_stacked_td()[0][0]
    key = tuple(str(i) for i in range(19)) + ("t",)
    benchmark(lambda: td.get(key))


def test_stacked_get(benchmark):
    td = big_nested_stacked_td()[0][0]
    key = tuple(str(i) for i in range(19))
    benchmark(lambda: td.get(key))


def test_nested_getitemleaf(benchmark):
    td = big_nested_td()[0][0]
    key = tuple(str(i) for i in range(19)) + ("t",)
    benchmark(lambda: td[key])


def test_nested_getitem(benchmark):
    td = big_nested_td()[0][0]
    key = tuple(str(i) for i in range(19))
    benchmark(lambda: td[key])


def test_stacked_getitemleaf(benchmark):
    td = big_nested_stacked_td()[0][0]
    key = tuple(str(i) for i in range(19)) + ("t",)
    benchmark(lambda: td[key])


def test_stacked_getitem(benchmark):
    td = big_nested_stacked_td()[0][0]
    key = tuple(str(i) for i in range(19))
    benchmark(lambda: td[key])


def test_lock_nested(benchmark):
    benchmark.pedantic(
        lambda td: list(td.lock_()),
        setup=big_nested_td,
        warmup_rounds=10,
        rounds=1000,
        iterations=1,
    )


def test_lock_stack_nested(benchmark):
    benchmark.pedantic(
        lambda td: list(td.lock_()),
        setup=big_nested_stacked_td,
        warmup_rounds=10,
        rounds=1000,
        iterations=1,
    )


def test_unlock_nested(benchmark):
    benchmark.pedantic(
        lambda td: list(td.unlock_()),
        setup=big_nested_td_locked,
        warmup_rounds=10,
        rounds=1000,
        iterations=1,
    )


def test_unlock_stack_nested(benchmark):
    benchmark.pedantic(
        lambda td: list(td.unlock_()),
        setup=big_nested_stacked_td_locked,
        warmup_rounds=10,
        rounds=1000,
        iterations=1,
    )


def test_flatten_speed(benchmark):
    td = big_nested_td()[0][0]
    benchmark(lambda: td.flatten_keys())


def test_unflatten_speed(benchmark):
    td = big_nested_td()[0][0].flatten_keys()
    benchmark(lambda: td.unflatten_keys())


def test_unflatten_wide_speed(benchmark):
    # 20 dotted keys in 4 groups, as in a flat result that is nested afterwards
    td = TensorDict(
        {f"g{i // 5}.f{i}": torch.zeros(1000) for i in range(20)}, batch_size=[1000]
    )
    benchmark(lambda: td.unflatten_keys())


def test_common_ops(benchmark):
    benchmark(main)


def test_creation(benchmark):
    benchmark(TensorDict, {}, [3, 4])


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_creation_device(benchmark, device):
    benchmark(TensorDict, {}, [3, 4], device=device)


def test_creation_empty(benchmark, a, b):
    benchmark(TensorDict, {"a": a, "b": b}, [3, 4])


def test_creation_nested_1(benchmark, a, b):
    benchmark(TensorDict, {"a": a, ("b", "b1"): b}, [3, 4])


def test_creation_nested_2(benchmark, a, b):
    benchmark(TensorDict, {"a": a, "b": {"b1": b}}, [3, 4])


@pytest.mark.parametrize("num_keys", [10, 50, 100])
def test_creation_many_keys(benchmark, num_keys):
    tensors = {str(i): torch.zeros(3, 4, 5) for i in range(num_keys)}
    benchmark(TensorDict, tensors, [3, 4])


@pytest.mark.parametrize("num_keys", [10, 50])
def test_creation_nested_many_keys(benchmark, num_keys):
    tensors = {str(i): torch.zeros(3, 4, 5) for i in range(num_keys)}
    tensors["nested"] = {str(i): torch.zeros(3, 4, 5) for i in range(num_keys)}
    benchmark(TensorDict, tensors, [3, 4])


def test_clone(benchmark, td):
    benchmark(td.clone)


def test_clone_device(benchmark, td):
    benchmark(td.to("cpu").clone)


@pytest.mark.parametrize(
    "index",
    [
        "int",
        "slice_int",
        "range",
        "tuple",
        "list",
        "tensor",
        "mask",
        "ellipsis",
        "none",
        "separated_tensors",
    ],
)
def test_getitem(benchmark, td, c, index):
    if index == "int":
        index = 1
    elif index == "slice_int":
        index = (slice(None), 1)
    elif index == "range":
        index = range(2)
    elif index == "tuple":
        index = (2, 1)
    elif index == "list":
        index = [0, 1]
    elif index == "tensor":
        index = torch.tensor([0, 2])
    elif index == "mask":
        index = torch.tensor([True, False, True])
    elif index == "ellipsis":
        index = (..., 1)
    elif index == "none":
        index = (None, slice(None), 1)
    elif index == "separated_tensors":
        index = (torch.tensor([0, 2]), None, torch.tensor([1, 0]))
    else:
        raise NotImplementedError

    def exec_getitem():
        _ = td[index]

    benchmark(exec_getitem)


@pytest.mark.parametrize(
    "device", [None, "cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_getitem_mask_many_leaves(benchmark, device):
    # the cells of a mesh: several leaves, some nested, filtered by a mask
    leaf_device = "cpu" if device is None else device
    td = TensorDict(
        {
            "a": torch.randn(1000, 3, device=leaf_device),
            "b": torch.randn(1000, device=leaf_device),
            "nest": {
                "c": torch.randn(1000, 4, device=leaf_device),
                "d": {"e": torch.randn(1000, device=leaf_device)},
            },
        },
        [1000],
        device=device,
    )
    mask = torch.rand(1000, device=leaf_device) > 0.5
    benchmark(td.__getitem__, mask)


@pytest.mark.parametrize("index", ["int", "slice_int", "range", "tuple"])
def test_setitem_dim(benchmark, td, c, index):
    if index == "int":
        index = 1
    elif index == "slice_int":
        index = (slice(None), 1)
    elif index == "range":
        index = range(2)
    elif index == "tuple":
        index = (2, 1)
    else:
        raise NotImplementedError

    def setup():
        td_index = td[index].clone().zero_()
        return ((td, td_index), {})

    def exec_setitem(td, td_index):
        td[index] = td_index

    benchmark.pedantic(exec_setitem, setup=setup, warmup_rounds=10, rounds=1000)


@pytest.fixture(scope="module")
def big_non_tensor_td():
    n = 100_000
    return TensorDict(
        obs=torch.zeros(n, 3),
        text=NonTensorStack.from_list([f"x{i}" for i in range(n)]),
        batch_size=[n],
    )


def test_setitem_non_tensor_rows(benchmark, big_non_tensor_td):
    # 4-row writes into a 100k-row tensordict with a string entry, as in a
    # replay buffer extend; each write brings new strings to new rows
    td = big_non_tensor_td
    counter = itertools.count()

    def setup():
        i = next(counter)
        chunk = TensorDict(
            obs=torch.ones(4, 3),
            text=NonTensorStack(*[f"v{i}_{j}" for j in range(4)]),
            batch_size=[4],
        )
        return ((td, chunk, torch.arange(4) + 4 * i), {})

    def exec_setitem(td, chunk, index):
        td[index] = chunk

    benchmark.pedantic(exec_setitem, setup=setup, warmup_rounds=2, rounds=20)


def test_setitem(benchmark, td, c):
    def exec_setitem():
        tdc = td.clone()
        tdc["c"] = c

    benchmark(exec_setitem)


def test_set(benchmark, td, c):
    def exec_set():
        tdc = td.clone()
        tdc.set("c", c)

    benchmark(exec_set)


def test_set_shared(benchmark, td):
    def exec_set_shared():
        tdc = td.clone()
        tdc.share_memory_()

    benchmark(exec_set_shared)


def test_update(benchmark, a, b):
    td = TensorDict({"a": a, "b": b}, [3, 4])
    td2 = td.clone()

    def exec_update():
        tdc = td.clone()
        tdc.update(td2)

    benchmark(exec_update)


def test_update_nested(benchmark, td):
    td2 = td.clone()

    def exec_update_nested():
        tdc = td.clone()
        tdc.update(td2)

    benchmark(exec_update_nested)


def test_update__nested(benchmark, td):
    td2 = td.clone()

    def exec_update__nested():
        tdc = td.clone()
        tdc.update_(td2)

    benchmark(exec_update__nested)


@pytest.mark.parametrize("n", [16, 256, 1024])
@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_update_many_leaves(benchmark, n, as_dict, nested):
    source = TensorDict({str(i): torch.ones(32, 8) for i in range(n)}, [32])
    if nested:
        source = TensorDict({"nested": source}, [32])
    dest = source.clone()
    benchmark(dest.update_, source.to_dict() if as_dict else source)


@pytest.mark.parametrize("n", [16, 256])
@pytest.mark.parametrize("source_type", ["dict", "td", "kwargs"])
@pytest.mark.parametrize("nested", [False, True])
def test_update_tensor_leaves(benchmark, n, source_type, nested):
    source = {f"f{i}": torch.ones(32, 8) for i in range(n)}
    if nested:
        items = list(source.items())
        source = {f"g{i}": dict(items[i : i + 4]) for i in range(0, n, 4)}
    td = TensorDict(source, [32], device="cpu").clone()
    if source_type == "kwargs":
        benchmark(td.update, **source)
    else:
        if source_type == "td":
            source = TensorDict(source, [32], device="cpu")
        benchmark(td.update, source)


@pytest.mark.parametrize("batch_size", [(), (32,)])
@pytest.mark.parametrize("device", [None, "cpu"])
@pytest.mark.parametrize("nested", [False, True])
def test_set_tensor_leaf(benchmark, batch_size, device, nested):
    value = torch.ones(32, 8)
    td = TensorDict({}, batch_size=batch_size, device=device)
    key = ("child", "x") if nested else "x"
    td.set(key, value)
    benchmark(td.set, key, value)


@pytest.mark.parametrize("batch_size", [(), (32,)])
@pytest.mark.parametrize("device", [None, "cpu"])
@pytest.mark.parametrize("nested", [False, True])
def test_set_tensor_leaf_inplace(benchmark, batch_size, device, nested):
    value = torch.ones(32, 8)
    key = ("child", "x") if nested else "x"
    td = TensorDict(
        {key: torch.zeros_like(value)}, batch_size=batch_size, device=device
    )
    storage = td[key].data_ptr()
    benchmark(td.set_, key, value)
    assert td[key].data_ptr() == storage != value.data_ptr()
    torch.testing.assert_close(td[key], value)


def test_set_nested(benchmark, td, b):
    def exec_set_nested():
        tdc = td.clone()
        tdc["b", "b1"] = b

    benchmark(exec_set_nested)


def test_set_nested_new(benchmark, td, c):
    def exec_set_nested_new():
        tdc = td.clone()
        tdc["c", "c", "c"] = c

    benchmark(exec_set_nested_new)


_KEYS = {
    "str": "x",
    "tuple": ("a", "b", "c"),
    "nested_tuple": ("next", ("agents", "reward")),
}


@pytest.mark.parametrize("kind", list(_KEYS))
def test_unravel_key(benchmark, kind):
    benchmark(unravel_key, _KEYS[kind])


@pytest.mark.parametrize("kind", ["str", "tuple", "mixed"])
def test_unravel_key_list(benchmark, kind):
    keys = {
        "str": ["a", "b", "c"],
        "tuple": [("a", "b"), ("a", "c"), ("next", "reward")],
        "mixed": list(_KEYS.values()),
    }[kind]
    benchmark(unravel_key_list, keys)


@pytest.mark.parametrize("kind", list(_KEYS))
def test_unravel_key_to_tuple(benchmark, kind):
    benchmark(_unravel_key_to_tuple, _KEYS[kind])


@pytest.mark.parametrize("kind", list(_KEYS))
def test_get_by_key(benchmark, kind):
    value = torch.zeros(3)
    td = TensorDict(
        {
            "x": value,
            ("a", "b", "c"): value,
            ("next", "agents", "reward"): value,
        },
        batch_size=[3],
    )
    benchmark(td.get, _KEYS[kind])


@pytest.mark.parametrize("key", ["x", "sub", "text", ("sub", "y")])
def test_getitem_by_key(benchmark, key):
    td = TensorDict(
        x=torch.zeros(3),
        sub=TensorDict(y=torch.zeros(3), batch_size=[3]),
        batch_size=[3],
    )
    td.set_non_tensor("text", "value")
    benchmark(td.__getitem__, key)


@pytest.mark.parametrize("key", [("nest", "c"), ("nest", "zz")], ids=["hit", "miss"])
def test_get_nested_with_default(benchmark, key):
    # reading an optional nested entry, such as a cached value
    td = TensorDict({"a": torch.zeros(3), "nest": {"c": torch.zeros(3)}}, [3])
    benchmark(td.get, key, None)


def test_select(benchmark, td, c):
    def exec_select():
        tdc = td.clone()
        tdc["c", "c", "c"] = c
        tdc.select("a", "z", ("c", "c", "c"), strict=False)

    benchmark(exec_select)


def test_select_nested(benchmark):
    td = big_nested_td()[0][0]
    key = list(td.keys(True, True))[-1]

    def func():
        td.select(key)

    benchmark(func)


def test_exclude_nested(benchmark):
    td = big_nested_td()[0][0]
    key = list(td.keys(True, True))[-1]

    def func():
        td.exclude(key)

    benchmark(func)


@pytest.mark.parametrize("op", ["exclude", "select", "step"])
def test_exclude_select_update_step(benchmark, op):
    # One env step of a nested (multi-agent style) env turned into the next
    # root, as TorchRL's step_mdp does (#2558).
    def leaf(dtype=torch.float32):
        return torch.zeros(3, 1, dtype=dtype)

    td = TensorDict(
        data=TensorDict(
            states=leaf(torch.int32),
            action=leaf(),
            done=leaf(torch.bool),
            terminated=leaf(torch.bool),
            batch_size=[3],
        ),
        next=TensorDict(
            data=TensorDict(
                states=leaf(torch.int32),
                done=leaf(torch.bool),
                terminated=leaf(torch.bool),
                reward=leaf(),
                batch_size=[3],
            )
        ),
    )
    excl = [
        ("next",),
        ("data", "done"),
        ("data", "terminated"),
        ("data", "_reset"),
        ("data", "reward"),
    ]
    keep = [("data", "states"), ("data", "done"), ("data", "terminated")]
    nxt = td.get("next")
    func = {
        "exclude": lambda: td.exclude(*excl),
        "select": lambda: nxt.select(*keep, strict=False),
        "step": lambda: td.exclude(*excl).update(nxt.select(*keep, strict=False)),
    }[op]
    benchmark(func)


@pytest.mark.parametrize("recurse", [True, False])
def test_empty(benchmark, recurse):
    td = big_nested_td()[0][0]

    def func(recurse=recurse):
        td.empty(recurse=recurse)

    benchmark(func)


def test_to_same_device(benchmark):
    # A device-less env step whose leaves are already on the CPU
    td = TensorDict(
        observation=torch.randn(4),
        action=torch.randn(2),
        done=torch.zeros(1, dtype=torch.bool),
        next=TensorDict(
            observation=torch.randn(4),
            reward=torch.randn(1),
            done=torch.zeros(1, dtype=torch.bool),
            terminated=torch.zeros(1, dtype=torch.bool),
        ),
        collector=TensorDict(traj_ids=torch.zeros((), dtype=torch.long)),
    )
    benchmark(td.to, "cpu")


@pytest.mark.skipif(not torch.cuda.device_count(), reason="No cuda device")
def test_to(benchmark, td):
    benchmark(td.to, "cuda:0")


@pytest.mark.skipif(not torch.cuda.device_count(), reason="No cuda device")
def test_to_nonblocking(benchmark, td):
    benchmark(td.to, "cuda:0", non_blocking=True)


def main():
    # creation
    td = TensorDict({}, [3, 4])

    # creation empty
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": b}, [3, 4])

    # creation nested 1
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, ("b", "b1"): b}, [3, 4])

    # creation nested 2
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])

    # clone
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    tdc = td.clone()

    # __setitem__
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    c = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    tdc = td.clone()
    tdc["c"] = c

    # set
    c = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    tdc = td.clone()
    tdc.set("c", c)

    # set shared
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    tdc = td.clone()
    tdc.share_memory_()

    # update
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": b}, [3, 4])
    td2 = td.clone()
    tdc = td.clone()
    tdc.update(td2)

    # update nested
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    td2 = td.clone()
    tdc = td.clone()
    tdc.update(td2)

    # set nested
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    tdc = td.clone()
    tdc["b", "b1"] = b

    # set nested new
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    c = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    tdc = td.clone()
    tdc["c", "c", "c"] = c

    # select
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    c = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    tdc = td.clone()
    tdc["c", "c", "c"] = c


def test_unbind_speed(benchmark):
    (td,), _ = big_nested_td()
    benchmark(lambda td: td.unbind(0), td)


def test_unbind_speed_stack0(benchmark):
    (td,), _ = big_nested_stacked_td()
    benchmark(lambda td: td.unbind(0), td)


def test_unbind_speed_stack1(benchmark):
    (td,), _ = big_nested_stacked_td()
    benchmark(lambda td: td.unbind(1), td)


def test_stack_nested(benchmark):
    td = big_nested_td()[0][0]
    benchmark(torch.stack, [td] * 4, 0)


def test_stack_rollout_steps(benchmark):
    # a rollout stacks one flat tensordict per step
    n = 64
    tds = [
        TensorDict(
            obs=torch.randn(n, 48),
            done=torch.zeros(n, dtype=torch.bool),
            value=torch.randn(n),
            action=torch.randn(n, 12),
            logprob=torch.randn(n),
            reward=torch.randn(n),
            batch_size=[n],
        )
        for _ in range(24)
    ]
    benchmark(torch.stack, tds, 0)


def test_cat_nested(benchmark):
    td = big_nested_td()[0][0]
    benchmark(torch.cat, [td] * 4, 0)


@pytest.mark.parametrize("op", ["pack", "unpack"])
def test_pack_nested(benchmark, op):
    td = TensorDict(
        {
            "a": torch.randn(64, 3),
            "b": torch.randn(64),
            "n": {"c": torch.randn(64, 3, 3)},
        },
        [64],
    )
    flat, layout = td.pack()
    if op == "pack":
        benchmark(td.pack)
    else:
        benchmark(layout.unpack, flat)


def test_apply_nested(benchmark):
    td = big_nested_td()[0][0]
    benchmark(td.apply, lambda x: x)


@pytest.mark.parametrize("op", ["neg", "mul_scalar", "add"])
def test_pointwise_nested(benchmark, op):
    td = TensorDict(
        {"a": torch.randn(8, 3), "b": torch.randn(8), "n": {"c": torch.randn(8, 4)}},
        [8],
    )
    fn = {
        "neg": lambda td: -td,
        "mul_scalar": lambda td: td * 2.0,
        "add": lambda td: td + td,
    }[op]
    benchmark(fn, td)


def test_split(benchmark):
    (td,), _ = big_nested_td(size=(3, 20))
    benchmark(lambda td: td.split(2, dim=1), td)


def test_chunk(benchmark):
    (td,), _ = big_nested_td(size=(3, 20))
    benchmark(lambda td: td.chunk(10, dim=1), td)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda device required")
@pytest.mark.parametrize("op", ["add", "sub"])
def test_add_cuda_scalar_tensor(benchmark, op):
    td = TensorDict(
        {f"k{i}": torch.ones(64, 3, device="cuda") for i in range(10)},
        batch_size=[64],
    )
    t = torch.tensor(0.5, device="cuda")
    benchmark(getattr(td, op), t)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
