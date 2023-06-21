import argparse

import pytest
import torch

from tensordict import TensorDict


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


def big_nested_td():
    return (
        (
            TensorDict(
                {
                    ".".join([str(j) for j in range(i)] + ["t"]): torch.zeros(3, 4) + i
                    for i in range(1, 20)
                },
                [3, 4],
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
                    [3, 10],
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


def test_items(benchmark):
    td = big_td()[0][0]
    benchmark.pedantic(lambda: list(td.items()), rounds=1000, iterations=1)


def test_items_nested(benchmark):
    td = big_nested_td()[0][0]
    benchmark.pedantic(lambda: list(td.items(True)), rounds=1000, iterations=1)


def test_items_nested_locked(benchmark):
    td = big_nested_td_locked()[0][0]
    list(td.items(True))
    benchmark.pedantic(lambda: list(td.items(True)), rounds=1000, iterations=1)


def test_items_nested_leaf(benchmark):
    td = big_nested_td()[0][0]
    benchmark.pedantic(
        lambda: list(td.items(True, True)),
        rounds=1000,
        iterations=1,
    )


def test_items_stack_nested(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark.pedantic(
        lambda: list(td.items(True)),
        rounds=1000,
        iterations=1,
    )


def test_items_stack_nested_leaf(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark.pedantic(
        lambda: list(td.items(True, True)),
        rounds=1000,
        iterations=1,
    )


def test_items_stack_nested_locked(benchmark):
    td = big_nested_stacked_td_locked()[0][0]
    list(td.items(True))
    benchmark.pedantic(lambda: list(td.items(True)), rounds=1000, iterations=1)


def test_keys(benchmark):
    td = big_td()[0][0]
    benchmark.pedantic(lambda: list(td.keys()), rounds=1000, iterations=1)


def test_keys_nested(benchmark):
    td = big_nested_td()[0][0]
    benchmark.pedantic(lambda: list(td.keys(True)), rounds=1000, iterations=1)


def test_keys_nested_locked(benchmark):
    td = big_nested_td_locked()[0][0]
    list(td.keys(True))
    benchmark.pedantic(lambda: list(td.keys(True)), rounds=1000, iterations=1)


def test_keys_nested_leaf(benchmark):
    td = big_nested_td()[0][0]
    benchmark.pedantic(
        lambda: list(td.keys(True, True)),
        rounds=1000,
        iterations=1,
    )


def test_keys_stack_nested(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark.pedantic(
        lambda: list(td.keys(True)),
        rounds=1000,
        iterations=1,
    )


def test_keys_stack_nested_leaf(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark.pedantic(
        lambda: list(td.keys(True, True)),
        rounds=1000,
        iterations=1,
    )


def test_keys_stack_nested_locked(benchmark):
    td = big_nested_stacked_td_locked()[0][0]
    list(td.keys(True))
    benchmark.pedantic(lambda: list(td.keys(True)), rounds=1000, iterations=1)


def test_values(benchmark):
    td = big_td()[0][0]
    benchmark.pedantic(lambda: list(td.values()), rounds=1000, iterations=1)


def test_values_nested(benchmark):
    td = big_nested_td()[0][0]
    benchmark.pedantic(lambda: list(td.values(True)), rounds=1000, iterations=100)


def test_values_nested_locked(benchmark):
    td = big_nested_td_locked()[0][0]
    list(td.values(True))
    benchmark.pedantic(lambda: list(td.values(True)), rounds=1000, iterations=1)


def test_values_nested_leaf(benchmark):
    td = big_nested_td()[0][0]
    benchmark.pedantic(
        lambda: list(td.values(True, True)),
        rounds=1000,
        iterations=1,
    )


def test_values_stack_nested(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark.pedantic(
        lambda: list(td.values(True)),
        rounds=1000,
        iterations=1,
    )


def test_values_stack_nested_leaf(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark.pedantic(
        lambda: list(td.values(True, True)),
        rounds=1000,
        iterations=1,
    )


def test_values_stack_nested_locked(benchmark):
    td = big_nested_stacked_td_locked()[0][0]
    list(td.values(True))
    benchmark.pedantic(lambda: list(td.values(True)), rounds=1000, iterations=1)


def test_membership(benchmark):
    td = big_td()[0][0]
    benchmark.pedantic(lambda: "a" in td.keys(), rounds=1000, iterations=1)


def test_membership_nested(benchmark):
    td = big_nested_td()[0][0]
    benchmark.pedantic(
        lambda: ("a",) in td.keys(True),
        rounds=1000,
        iterations=1,
    )


def test_membership_nested_leaf(benchmark):
    td = big_nested_td()[0][0]
    benchmark.pedantic(
        lambda: ("a",) in td.keys(True, True),
        rounds=1000,
        iterations=1,
    )


def test_membership_stacked_nested(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark.pedantic(
        lambda: ("a",) in td.keys(True),
        rounds=1000,
        iterations=1,
    )


def test_membership_stacked_nested_leaf(benchmark):
    td = big_nested_stacked_td()[0][0]
    benchmark.pedantic(
        lambda: ("a",) in td.keys(True, True),
        rounds=1000,
        iterations=1,
    )


def test_stacked_getleaf(benchmark):
    td = big_nested_stacked_td()[0][0]
    key = tuple(str(i) for i in range(19)) + ("t",)
    benchmark.pedantic(
        lambda: td.get(key),
        rounds=1000,
        iterations=1,
    )


def test_stacked_get(benchmark):
    td = big_nested_stacked_td()[0][0]
    key = tuple(str(i) for i in range(19))
    benchmark.pedantic(
        lambda: td.get(key),
        rounds=1000,
        iterations=1,
    )


def test_lock_nested(benchmark):
    benchmark.pedantic(
        lambda td: list(td.lock_()), setup=big_nested_td, rounds=1000, iterations=1
    )


def test_lock_stack_nested(benchmark):
    benchmark.pedantic(
        lambda td: list(td.lock_()),
        setup=big_nested_stacked_td,
        rounds=1000,
        iterations=1,
    )


def test_unlock_nested(benchmark):
    benchmark.pedantic(
        lambda td: list(td.unlock_()),
        setup=big_nested_td_locked,
        rounds=1000,
        iterations=1,
    )


def test_unlock_stack_nested(benchmark):
    benchmark.pedantic(
        lambda td: list(td.unlock_()),
        setup=big_nested_stacked_td_locked,
        rounds=1000,
        iterations=1,
    )


def test_flatten_speed(benchmark):
    td = big_nested_td()[0][0]
    benchmark.pedantic(
        lambda: td.flatten_keys(),
        rounds=1000,
        iterations=1,
    )


def test_unflatten_speed(benchmark):
    td = big_nested_td()[0][0].flatten_keys()
    benchmark.pedantic(
        lambda: td.unflatten_keys(),
        rounds=1000,
        iterations=1,
    )


def test_common_ops(benchmark):
    benchmark.pedantic(main, iterations=100, rounds=100)


def test_creation(benchmark):
    benchmark.pedantic(TensorDict, args=({}, [3, 4]), iterations=100, rounds=100)


def test_creation_empty(benchmark, a, b):
    benchmark.pedantic(
        TensorDict, args=({"a": a, "b": b}, [3, 4]), iterations=100, rounds=100
    )


def test_creation_nested_1(benchmark, a, b):
    benchmark.pedantic(
        TensorDict, args=({"a": a, ("b", "b1"): b}, [3, 4]), iterations=100, rounds=100
    )


def test_creation_nested_2(benchmark, a, b):
    benchmark.pedantic(
        TensorDict, args=({"a": a, "b": {"b1": b}}, [3, 4]), iterations=100, rounds=100
    )


def test_clone(benchmark, td):
    benchmark.pedantic(td.clone, iterations=100, rounds=100)


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

    benchmark.pedantic(exec_getitem, iterations=1000, rounds=1000)


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

    benchmark.pedantic(exec_setitem, setup=setup, iterations=1, rounds=10000)


def test_setitem(benchmark, td, c):
    def exec_setitem():
        tdc = td.clone()
        tdc["c"] = c

    benchmark.pedantic(exec_setitem, iterations=100, rounds=100)


def test_set(benchmark, td, c):
    def exec_set():
        tdc = td.clone()
        tdc.set("c", c)

    benchmark.pedantic(exec_set, iterations=100, rounds=100)


def test_set_shared(benchmark, td):
    def exec_set_shared():
        tdc = td.clone()
        tdc.share_memory_()

    benchmark.pedantic(exec_set_shared, iterations=100, rounds=100)


def test_update(benchmark, a, b):
    td = TensorDict({"a": a, "b": b}, [3, 4])
    td2 = td.clone()

    def exec_update():
        tdc = td.clone()
        tdc.update(td2)

    benchmark.pedantic(exec_update, iterations=100, rounds=100)


def test_update_nested(benchmark, td):
    td2 = td.clone()

    def exec_update_nested():
        tdc = td.clone()
        tdc.update(td2)

    benchmark.pedantic(exec_update_nested, iterations=100, rounds=100)


def test_set_nested(benchmark, td, b):
    def exec_set_nested():
        tdc = td.clone()
        tdc["b", "b1"] = b

    benchmark.pedantic(exec_set_nested, iterations=100, rounds=100)


def test_set_nested_new(benchmark, td, c):
    def exec_set_nested_new():
        tdc = td.clone()
        tdc["c", "c", "c"] = c

    benchmark.pedantic(exec_set_nested_new, iterations=100, rounds=100)


def test_select(benchmark, td, c):
    def exec_select():
        tdc = td.clone()
        tdc["c", "c", "c"] = c
        tdc.select("a", "z", ("c", "c", "c"), strict=False)

    benchmark.pedantic(exec_select, iterations=100, rounds=100)


@pytest.mark.skipif(not torch.cuda.device_count(), reason="No cuda device")
def test_to(benchmark, td):
    benchmark.pedantic(td.to, args=("cuda:0",), iterations=100, rounds=1000)


@pytest.mark.skipif(not torch.cuda.device_count(), reason="No cuda device")
def test_to_nonblocking(benchmark, td):
    benchmark.pedantic(
        td.to,
        args=("cuda:0",),
        kwargs={"non_blocking": True},
        iterations=100,
        rounds=1000,
    )


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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
