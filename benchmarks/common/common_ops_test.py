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


def test_common_ops(benchmark):
    benchmark.pedantic(main, iterations=10000)


def test_creation(benchmark):
    benchmark.pedantic(TensorDict, args=({}, [3, 4]), iterations=10000)


def test_creation_empty(benchmark, a, b):
    benchmark.pedantic(TensorDict, args=({"a": a, "b": b}, [3, 4]), iterations=10000)


def test_creation_nested_1(benchmark, a, b):
    benchmark.pedantic(
        TensorDict, args=({"a": a, ("b", "b1"): b}, [3, 4]), iterations=10000
    )


def test_creation_nested_2(benchmark, a, b):
    benchmark.pedantic(
        TensorDict, args=({"a": a, "b": {"b1": b}}, [3, 4]), iterations=10000
    )


def test_clone(benchmark, td):
    benchmark.pedantic(td.clone, iterations=10000)


def test_setitem(benchmark, td, c):
    def exec_setitem():
        tdc = td.clone()
        tdc["c"] = c

    benchmark.pedantic(exec_setitem, iterations=10000)


def test_set(benchmark, td, c):
    def exec_set():
        tdc = td.clone()
        tdc.set("c", c)

    benchmark.pedantic(exec_set, iterations=10000)


def test_set_shared(benchmark, td):
    def exec_set_shared():
        tdc = td.clone()
        tdc.share_memory_()

    benchmark.pedantic(exec_set_shared, iterations=10000)


def test_update(benchmark, a, b):
    td = TensorDict({"a": a, "b": b}, [3, 4])
    td2 = td.clone()

    def exec_update():
        tdc = td.clone()
        tdc.update(td2)

    benchmark.pedantic(exec_update, iterations=10000)


def test_update_nested(benchmark, td):
    td2 = td.clone()

    def exec_update_nested():
        tdc = td.clone()
        tdc.update(td2)

    benchmark.pedantic(exec_update_nested, iterations=10000)


def test_set_nested(benchmark, td, b):
    def exec_set_nested():
        tdc = td.clone()
        tdc["b", "b1"] = b

    benchmark.pedantic(exec_set_nested, iterations=10000)


def test_set_nested_new(benchmark, td, c):
    def exec_set_nested_new():
        tdc = td.clone()
        tdc["c", "c", "c"] = c

    benchmark.pedantic(exec_set_nested_new, iterations=10000)


def test_select(benchmark, td, c):
    def exec_select():
        tdc = td.clone()
        tdc["c", "c", "c"] = c
        tdc.select("a", "z", ("c", "c", "c"), strict=False)

    benchmark.pedantic(exec_select, iterations=10000)

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
