import argparse

import pytest
import torch

from tensordict import is_tensor_collection, TensorDict


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


def test_common_ops(benchmark):
    benchmark(main)


def test_creation(benchmark):
    benchmark(TensorDict, {}, [3, 4])


def test_creation_empty(benchmark, a, b):
    benchmark(TensorDict, {"a": a, "b": b}, [3, 4])


def test_creation_nested_1(benchmark, a, b):
    benchmark(TensorDict, {"a": a, ("b", "b1"): b}, [3, 4])


def test_creation_nested_2(benchmark, a, b):
    benchmark(TensorDict, {"a": a, "b": {"b1": b}}, [3, 4])


def test_clone(benchmark, td):
    benchmark(td.clone)


@pytest.mark.parametrize("index", ["int", "slice_int", "range", "tuple", "list"])
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
    else:
        raise NotImplementedError

    def exec_getitem():
        _ = td[index]

    benchmark(exec_getitem)


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


@pytest.mark.parametrize("recurse", [True, False])
def test_empty(benchmark, recurse):
    td = big_nested_td()[0][0]

    def func(recurse=recurse):
        td.empty(recurse=recurse)

    benchmark(func)


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


def test_split(benchmark):
    (td,), _ = big_nested_td(size=(3, 20))
    benchmark(lambda td: td.split(2, dim=1), td)


def test_chunk(benchmark):
    (td,), _ = big_nested_td(size=(3, 20))
    benchmark(lambda td: td.chunk(10, dim=1), td)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
