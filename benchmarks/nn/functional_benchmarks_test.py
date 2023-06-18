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

try:
    from torch import vmap
except ImportError:
    try:
        from functorch import vmap
    except ImportError:
        raise RuntimeError("vmap couldn't be found, check pytorch version.")


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


@torch.no_grad()
@pytest.mark.parametrize("stack", [True, False])
@pytest.mark.parametrize("tdmodule", [True, False])
def test_vmap_mlp_speed(benchmark, stack, tdmodule):
    # tests speed of vmapping over a transformer
    device = "cuda" if torch.cuda.device_count() else "cpu"
    t = nn.Sequential(
        nn.Linear(64, 64, device=device),
        nn.ReLU(),
        nn.Linear(64, 64, device=device),
        nn.ReLU(),
        nn.Linear(64, 64, device=device),
        nn.ReLU(),
        nn.Linear(64, 64, device=device),
        nn.ReLU(),
    )
    if tdmodule:
        t = TensorDictModule(t, in_keys=["x"], out_keys=["y"])

    x = torch.randn(1, 1, 64)
    t.eval()
    params = make_functional(t)
    if not stack:
        params = params.expand(2).to_tensordict()
    else:
        params = torch.stack([params, params.clone()], 0)
    if tdmodule:
        fun = vmap(t, (None, 0))
        data = TensorDict({"x": x}, [])
        fun(data, params)
        benchmark.pedantic(fun, args=(data, params), rounds=100, iterations=100)
    else:
        fun = vmap(t, (None, 0))
        fun(x, params)
        benchmark.pedantic(fun, args=(x, params), rounds=100, iterations=100)


@torch.no_grad()
@pytest.mark.parametrize("stack", [True, False])
@pytest.mark.parametrize("tdmodule", [True, False])
def test_vmap_transformer_speed(benchmark, stack, tdmodule):
    # tests speed of vmapping over a transformer
    device = "cuda" if torch.cuda.device_count() else "cpu"
    t = torch.nn.Transformer(
        64,
        nhead=4,
        num_decoder_layers=3,
        num_encoder_layers=3,
        dim_feedforward=64,
        device=device,
    )
    if tdmodule:
        t = TensorDictModule(t, in_keys=["x", "x"], out_keys=["y"])

    x = torch.randn(1, 1, 64)
    t.eval()
    params = make_functional(t)
    if not stack:
        params = params.expand(2).to_tensordict()
    else:
        params = torch.stack([params, params.clone()], 0)
    if tdmodule:
        fun = vmap(t, (None, 0))
        data = TensorDict({"x": x}, [])
        fun(data, params)
        benchmark.pedantic(fun, args=(data, params), rounds=100, iterations=100)
    else:
        fun = vmap(t, (None, None, 0))
        fun(x, x, params)
        benchmark.pedantic(fun, args=(x, x, params), rounds=100, iterations=100)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
