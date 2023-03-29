# we use deepcopy as our implementation modifies the modules in-place
from copy import deepcopy

import pytest
import torch
from functorch import make_functional_with_buffers as functorch_make_functional

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
