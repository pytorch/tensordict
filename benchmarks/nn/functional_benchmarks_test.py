# we use deepcopy as our implementation modifies the modules in-place
import argparse
from copy import deepcopy

import pytest
import torch
from functorch import make_functional_with_buffers as functorch_make_functional

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from tensordict.nn.functional_modules import make_functional
from torch import nn


def make_net():
    return nn.Sequential(
        nn.Linear(2, 2),
        nn.Linear(2, 2),
        nn.Linear(2, 2),
        nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)),
    )


@pytest.fixture
def net():
    return make_net()


def _functorch_make_functional(net):
    functorch_make_functional(deepcopy(net))


def _make_functional(net):
    make_functional(deepcopy(net))


def make_tdmodule():
    return (
        (
            TensorDictModule(lambda x: x, in_keys=["x"], out_keys=["y"]),
            TensorDict({"x": torch.zeros(())}, []),
        ),
        {},
    )


def test_tdmodule(benchmark):
    benchmark.pedantic(
        lambda net, td: net(td),
        setup=make_tdmodule,
        iterations=1,
        rounds=10_000,
        warmup_rounds=1000,
    )


def make_tdmodule_dispatch():
    return (
        (
            TensorDictModule(lambda x: x, in_keys=["x"], out_keys=["y"]),
            torch.zeros(()),
        ),
        {},
    )


def test_tdmodule_dispatch(benchmark):
    benchmark.pedantic(
        lambda net, x: net(x),
        setup=make_tdmodule_dispatch,
        iterations=1,
        rounds=10_000,
        warmup_rounds=1000,
    )


def make_tdseq():
    class MyModule(TensorDictModuleBase):
        in_keys = ["x"]
        out_keys = ["y"]

        def forward(self, tensordict):
            return tensordict.set("y", tensordict.get("x"))

    return (
        (
            TensorDictSequential(MyModule()),
            TensorDict({"x": torch.zeros(())}, []),
        ),
        {},
    )


def test_tdseq(benchmark):
    benchmark.pedantic(
        lambda net, td: net(td),
        setup=make_tdseq,
        iterations=1,
        rounds=10_000,
        warmup_rounds=1000,
    )


def make_tdseq_dispatch():
    class MyModule(TensorDictModuleBase):
        in_keys = ["x"]
        out_keys = ["y"]

        def forward(self, tensordict):
            return tensordict.set("y", tensordict.get("x"))

    return (
        (
            TensorDictSequential(MyModule()),
            torch.zeros(()),
        ),
        {},
    )


def test_tdseq_dispatch(benchmark):
    benchmark.pedantic(
        lambda net, x: net(x),
        setup=make_tdseq_dispatch,
        iterations=1,
        rounds=10_000,
        warmup_rounds=1000,
    )


# Creation
def test_instantiation_functorch(benchmark, net):
    benchmark.pedantic(
        _functorch_make_functional, args=(net,), iterations=10, rounds=100
    )


def test_instantiation_td(benchmark, net):
    benchmark.pedantic(_make_functional, args=(net,), iterations=10, rounds=100)


# Execution
def test_exec_functorch(benchmark, net):
    x = torch.randn(2, 2)
    fmodule, params, buffers = functorch_make_functional(net)
    benchmark.pedantic(fmodule, args=(params, buffers, x), iterations=100, rounds=100)


def test_exec_td(benchmark, net):
    x = torch.randn(2, 2)
    fmodule = net
    params = make_functional(fmodule)
    benchmark.pedantic(
        fmodule, args=(x,), kwargs={"params": params}, iterations=100, rounds=100
    )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
