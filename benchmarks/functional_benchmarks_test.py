import pytest

# we use deepcopy as our implementation modifies the modules in-place
from copy import deepcopy

import torch
from functorch import make_functional_with_buffers as functorch_make_functional
from tensordict.nn.functional_modules import make_functional
from torch import nn


@pytest.fixture
def net():
    return nn.Sequential(
        nn.Linear(2, 2),
        nn.Linear(2, 2),
        nn.Linear(2, 2),
        nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)),
    )
# Creation
def test_instantiation_functorch(benchmark, net):
    benchmark.pedantic(functorch_make_functional, args=(deepcopy(net),), iterations=1000)

def test_instantiation_td(benchmark, net):
    benchmark.pedantic(make_functional, args=(deepcopy(net),), iterations=1000)

# Execution
def test_exec_functorch(benchmark, net):
    x = torch.randn(2, 2)
    fmodule, params, buffers = functorch_make_functional(deepcopy(net))
    benchmark.pedantic(fmodule, args=(params, buffers, x), iterations=10000)

def test_exec_td(benchmark, net):
    x = torch.randn(2, 2)
    fmodule = deepcopy(net)
    params = make_functional(fmodule)
    benchmark.pedantic(fmodule, args=(x,), kwargs={'params' : params}, iterations=10000)
