# we use deepcopy as our implementation modifies the modules in-place
import argparse
from copy import deepcopy

import pytest
import torch
from functorch import make_functional_with_buffers as functorch_make_functional

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from tensordict.nn.functional_modules import make_functional

from torch import nn, vmap


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
        warmup_rounds=10,
        rounds=1000,
        iterations=1,
    )


def make_tdmodule_dispatch():
    return (
        (TensorDictModule(lambda x: x, in_keys=["x"], out_keys=["y"]), torch.zeros(())),
        {},
    )


def test_tdmodule_dispatch(benchmark):
    benchmark.pedantic(
        lambda net, x: net(x),
        setup=make_tdmodule_dispatch,
        warmup_rounds=10,
        rounds=1000,
        iterations=1,
    )


def make_tdseq():
    class MyModule(TensorDictModuleBase):
        in_keys = ["x"]
        out_keys = ["y"]

        def forward(self, tensordict):
            return tensordict.set("y", tensordict.get("x"))

    return (
        (TensorDictSequential(MyModule()), TensorDict({"x": torch.zeros(())}, [])),
        {},
    )


def test_tdseq(benchmark):
    benchmark.pedantic(
        lambda net, td: net(td), setup=make_tdseq, warmup_rounds=10, rounds=1000
    )


def make_tdseq_dispatch():
    class MyModule(TensorDictModuleBase):
        in_keys = ["x"]
        out_keys = ["y"]

        def forward(self, tensordict):
            return tensordict.set("y", tensordict.get("x"))

    return ((TensorDictSequential(MyModule()), torch.zeros(())), {})


def test_tdseq_dispatch(benchmark):
    benchmark.pedantic(
        lambda net, x: net(x), setup=make_tdseq_dispatch, warmup_rounds=10, rounds=1000
    )


# Creation
def test_instantiation_functorch(benchmark, net):
    benchmark(_functorch_make_functional, net)


def test_instantiation_td(benchmark, net):
    benchmark(_make_functional, net)


# Execution
def test_exec_functorch(benchmark, net):
    x = torch.randn(2, 2)
    sd = net.state_dict()

    def fun(x, sd):
        torch.func.functional_call(net, sd, x)

    benchmark(fun, x, sd)


def test_exec_functional_call(benchmark, net):
    x = torch.randn(2, 2)
    fmodule, params, buffers = functorch_make_functional(net)
    benchmark(fmodule, params, buffers, x)


def test_exec_td(benchmark, net):
    x = torch.randn(2, 2)
    fmodule = net
    params = make_functional(fmodule)
    benchmark(fmodule, x, params=params)


def test_exec_td_decorator(benchmark, net):
    x = torch.randn(2, 2)
    fmodule = net
    params = TensorDict.from_module(fmodule)

    def fun(x, params):
        with params.to_module(net):
            net(x)

    benchmark(fun, x, params)


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

    x = torch.randn(1, 1, 64, device=device)
    t.eval()
    params = make_functional(t)
    if not stack:
        params = params.expand(2).to_tensordict().lock_()
    else:
        params = torch.stack([params, params.clone()], 0).lock_()
    if tdmodule:
        fun = vmap(t, (None, 0))
        data = TensorDict({"x": x}, [])
        fun(data, params)
        benchmark(fun, data, params)
    else:
        fun = vmap(t, (None, 0))
        fun(x, params)
        benchmark(fun, x, params)


@torch.no_grad()
@pytest.mark.parametrize("stack", [True, False])
@pytest.mark.parametrize("tdmodule", [True, False])
def test_vmap_mlp_speed_decorator(benchmark, stack, tdmodule):
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

    x = torch.randn(1, 1, 64, device=device)
    t.eval()
    params = TensorDict.from_module(t)
    if not stack:
        params = params.expand(2).to_tensordict().lock_()
    else:
        params = torch.stack([params, params.clone()], 0).lock_()

    def fun(x, params):
        with params.to_module(t):
            return t(x)

    vfun = vmap(fun, (None, 0))

    if tdmodule:
        data = TensorDict({"x": x}, [])
        vfun(data, params)
        benchmark(vfun, data, params)
    else:
        vfun(x, params)
        benchmark(vfun, x, params)


@torch.no_grad()
@pytest.mark.skipif(
    not torch.cuda.device_count(), reason="cuda device required for test"
)
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
        batch_first=False,
    )
    if tdmodule:
        t = TensorDictModule(t, in_keys=["x", "x"], out_keys=["y"])

    x = torch.randn(1, 20, 64, device=device)
    t.eval()
    params = make_functional(t)
    if not stack:
        params = params.expand(2).to_tensordict().lock_()
    else:
        params = torch.stack([params, params.clone()], 0).lock_()
    if tdmodule:
        fun = vmap(t, (None, 0))
        data = TensorDict({"x": x}, [])
        fun(data, params)
        benchmark(fun, data, params)
    else:
        fun = vmap(t, (None, None, 0))
        fun(x, x, params)
        benchmark(fun, x, x, params)


@torch.no_grad()
@pytest.mark.skipif(
    not torch.cuda.device_count(), reason="cuda device required for test"
)
@pytest.mark.parametrize("stack", [True, False])
@pytest.mark.parametrize("tdmodule", [True, False])
def test_vmap_transformer_speed_decorator(benchmark, stack, tdmodule):
    # tests speed of vmapping over a transformer
    device = "cuda" if torch.cuda.device_count() else "cpu"
    t = torch.nn.Transformer(
        8,
        dim_feedforward=8,
        device=device,
        batch_first=False,
    )
    if tdmodule:
        t = TensorDictModule(t, in_keys=["x", "x"], out_keys=["y"])

    x = torch.randn(2, 2, 8, device=device)
    t.eval()
    params = TensorDict.from_module(t)
    if not stack:
        params = params.expand(2).to_tensordict().lock_()
    else:
        params = torch.stack([params, params.clone()], 0).lock_()

    if tdmodule:

        def fun(x, params):
            with params.to_module(t):
                return t(x)

        vfun = vmap(fun, (None, 0))
        data = TensorDict({"x": x}, [])
        vfun(data, params)
        benchmark(vfun, data, params)
    else:

        def fun(x, params):
            with params.to_module(t):
                return t(x, x)

        vfun = vmap(fun, (None, 0))
        vfun(x, params)
        benchmark(vfun, x, params)


@pytest.mark.parametrize("tdparams", [True, False])
def test_to_module_speed(benchmark, tdparams):
    module = torch.nn.Transformer()
    params = TensorDict.from_module(module, as_module=tdparams)

    def func(params=params, module=module):
        with params.to_module(module):
            pass
        return

    benchmark(func)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
