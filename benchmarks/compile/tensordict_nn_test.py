# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import functools
import gc
import sys

import pytest
import torch

from packaging import version
from tensordict import TensorDict, TensorDictParams

from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)

sys.setrecursionlimit(10000)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="function", autouse=True)
def auto_device():
    device = torch.get_default_device()
    torch.set_default_device(DEVICE)
    yield
    torch.set_default_device(device)


compile = functools.partial(torch.compile, fullgraph=True)
compile_overhead = functools.partial(
    torch.compile, fullgraph=True, mode="reduce-overhead"
)


@pytest.fixture(scope="function", autouse=True)
def reset_dynamo():
    # Start a fresh compile for each parameter of the test case
    try:
        torch.compiler.reset()
    except AttributeError:
        torch._dynamo.reset()
    gc.collect()
    yield


def mlp(device, depth=2, num_cells=32, feature_dim=3):
    return torch.nn.Sequential(
        torch.nn.Linear(feature_dim, num_cells, device=device),
        torch.nn.ReLU(),
        *[
            torch.nn.Sequential(
                torch.nn.Linear(num_cells, num_cells, device=device), torch.nn.ReLU()
            )
            for _ in range(depth)
        ],
        torch.nn.Linear(num_cells, 4, device=device),
    )


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", ["eager", "compile", "compile-overhead"])
def test_mod_add(mode, benchmark):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td = TensorDict({"a": 0}, device=device)
    module = Mod(lambda x: x + 1, in_keys=["a"], out_keys=[("c", "d")])
    if mode == "compile":
        module = compile(module)
    elif mode == "compile-overhead":
        module = compile_overhead(module)
    module(td)
    module(td)
    benchmark(module, td)


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", ["eager", "compile", "compile-overhead"])
def test_mod_wrap(mode, benchmark):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = mlp(device)
    td = TensorDict({"a": torch.zeros(32, 3, device=device)}, device=device)
    module = Mod(net, in_keys=["a"], out_keys=[("c", "d")])
    if mode == "compile":
        module = compile(module)
    elif mode == "compile-overhead":
        module = compile_overhead(module)
    module(td)
    module(td)
    benchmark(module, td)


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", ["eager", "compile", "compile-overhead"])
def test_mod_wrap_and_backward(mode, benchmark):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = mlp(device, num_cells=1024, depth=5)
    td = TensorDict({"a": torch.zeros(32, 3, device=device)}, device=device)
    module = Mod(net, in_keys=["a"], out_keys=[("c", "d")])
    if mode == "compile":
        module = compile(module)
    elif mode == "compile-overhead":
        module = compile_overhead(module)

    def module_exec(td):
        if torch.cuda.is_available():
            torch.compiler.cudagraph_mark_step_begin()
        module.zero_grad()
        module(td)
        td["c", "d"].mean().backward()

    module_exec(td)
    module_exec(td)
    benchmark(module_exec, td)


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", ["eager", "compile", "compile-overhead"])
def test_seq_add(mode, benchmark):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td = TensorDict({"a": 0}, device=device)

    def delhidden(td):
        del td["hidden"]
        return td

    module = Seq(
        lambda td: td.copy(),
        Mod(lambda x: x + 1, in_keys=["a"], out_keys=["hidden"]),
        Mod(lambda x: x + 1, in_keys=["hidden"], out_keys=[("c", "d")]),
        delhidden,
    )
    if mode == "compile":
        module = compile(module)
    elif mode == "compile-overhead":
        module = compile_overhead(module)
    module(td)
    module(td)
    benchmark(module, td)


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", ["eager", "compile", "compile-overhead"])
def test_seq_wrap(mode, benchmark):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = mlp(device)
    td = TensorDict({"a": torch.zeros(32, 3, device=device)}, device=device)

    def delhidden(td):
        del td["hidden"]
        return td

    module = Seq(
        lambda td: td.copy(),
        *[
            Mod(
                layer,
                in_keys=["a" if i == 0 else "hidden"],
                out_keys=[("c", "d") if i == len(net) - 1 else "hidden"],
            )
            for i, layer in enumerate(net)
        ],
        delhidden,
    )
    if mode == "compile":
        module = compile(module)
    elif mode == "compile-overhead":
        module = compile_overhead(module)
    module(td)
    module(td)
    benchmark(module, td)


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.slow
@pytest.mark.parametrize("mode", ["eager", "compile", "compile-overhead"])
def test_seq_wrap_and_backward(mode, benchmark):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = mlp(device, num_cells=1024, depth=5)
    td = TensorDict({"a": torch.zeros(32, 3, device=device)}, device=device)

    def delhidden(td):
        del td["hidden"]
        return td

    module = Seq(
        lambda td: td.copy(),
        *[
            Mod(
                layer,
                in_keys=["a" if i == 0 else "hidden"],
                out_keys=[("c", "d") if i == len(net) - 1 else "hidden"],
            )
            for i, layer in enumerate(net)
        ],
        delhidden,
    )
    if mode == "compile":
        module = compile(module)
    elif mode == "compile-overhead":
        module = compile_overhead(module)

    def module_exec(td):
        module.zero_grad()
        td = module(td.copy())
        td["c", "d"].mean().backward()
        return

    module_exec(td)
    module_exec(td)
    benchmark(module_exec, td)


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", ["eager", "compile", "compile-overhead"])
@pytest.mark.parametrize("functional", [False, True])
def test_func_call_runtime(mode, functional, benchmark):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = mlp(device=device, depth=10, num_cells=16, feature_dim=16)
    # module = torch.nn.Transformer(16, dim_feedforward=64, device=device)
    if functional:
        td = TensorDict.from_module(module)
        td = TensorDictParams(td.clone())

        def call(x, td):
            # with needs registering
            params = td.to_module(module, return_swap=True)
            result = module(x)
            params.to_module(module, return_swap=False)
            return result

    else:
        call = module

    if mode == "compile":
        call = compile(call)
    elif mode == "compile-overhead":
        call = compile_overhead(call)

    x = torch.randn(2, 2, 16)
    if functional:
        call(x, td)
        call(x, td)
        benchmark(call, x, td)
    else:
        call(x)
        call(x)
        benchmark(call, x)


@pytest.mark.parametrize("mode", ["eager", "compile", "compile-overhead"])
@pytest.mark.parametrize("functional", [False, True])
def test_func_call_cm_runtime(mode, functional, benchmark):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = mlp(device=device, depth=10, num_cells=16, feature_dim=16)
    # module = torch.nn.Transformer(16, dim_feedforward=64, device=device)
    if functional:
        td = TensorDict.from_module(module)
        td = TensorDictParams(td.clone())

        def call(x, td):
            # with needs registering
            with td.to_module(module):
                return module(x)

    else:
        call = module

    if mode == "compile":
        call = torch.compile(call)
    elif mode == "compile-overhead":
        call = torch.compile(call, mode="reduce-overhead")

    x = torch.randn(2, 2, 16)
    if functional:
        call(x, td)
        call(x, td)
        benchmark(call, x, td)
    else:
        call(x)
        call(x)
        benchmark(call, x)


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.slow
@pytest.mark.parametrize("mode", ["eager", "compile", "compile-overhead"])
@pytest.mark.parametrize(
    "functional,plain_decorator", [[False, None], [True, False], [True, True]]
)
def test_func_call_runtime_and_backward(mode, functional, plain_decorator, benchmark):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = mlp(device=device, depth=10, num_cells=16, feature_dim=16)
    # module = torch.nn.Transformer(16, dim_feedforward=64, device=device)
    if functional:
        td = TensorDict.from_module(module)
        td = TensorDictParams(td.data.clone())
        if not plain_decorator:

            def call(x, td):
                if torch.cuda.is_available():
                    torch.compiler.cudagraph_mark_step_begin()
                # with needs registering
                params = td.to_module(module, return_swap=True)
                result = module(x)
                params.to_module(module, return_swap=False)
                return result

        else:

            def call(x, td):
                if torch.cuda.is_available():
                    torch.compiler.cudagraph_mark_step_begin()
                # with needs registering
                with td.to_module(module):
                    return module(x)

    else:
        call = module

    if mode == "compile":
        call = torch.compile(call, fullgraph=not plain_decorator)
    elif mode == "compile-overhead":
        call = torch.compile(
            call, fullgraph=not plain_decorator, mode="reduce-overhead"
        )

    def call_with_backward(*args):
        call(*args).mean().backward()

    x = torch.randn(2, 2, 16)
    if functional:
        call_with_backward(x, td)
        call_with_backward(x, td)
        benchmark(call_with_backward, x, td)
    else:
        call_with_backward(x)
        call_with_backward(x)
        benchmark(call_with_backward, x)


@pytest.mark.parametrize("mode", ["eager", "compile", "compile-overhead"])
def test_vmap_func_call_cm_runtime(mode, benchmark):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = mlp(device=device, depth=10, num_cells=16, feature_dim=16)
    # module = torch.nn.Transformer(16, dim_feedforward=64, device=device)
    td = TensorDict.from_module(module)
    td = TensorDictParams(td.data.expand(10).clone().zero_())

    def call(x, td):
        # with needs registering
        with td.to_module(module):
            return module(x)

    call_vmap = torch.vmap(call, (None, 0))
    if mode == "compile":
        call_vmap = torch.compile(call_vmap)
    elif mode == "compile-overhead":
        call_vmap = torch.compile(call_vmap, mode="reduce-overhead")

    x = torch.randn(2, 2, 16)
    call_vmap(x, td)
    call_vmap(x, td)
    benchmark(call_vmap, x, td)


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.slow
@pytest.mark.parametrize("mode", ["eager", "compile", "compile-overhead"])
@pytest.mark.parametrize("plain_decorator", [None, False, True])
def test_vmap_func_call_runtime_and_backward(mode, plain_decorator, benchmark):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = mlp(device=device, depth=10, num_cells=16, feature_dim=16)
    # module = torch.nn.Transformer(16, dim_feedforward=64, device=device)
    td = TensorDict.from_module(module)
    td = TensorDictParams(td.data.expand(10).clone().zero_())
    if not plain_decorator:

        def call(x, td):
            if torch.cuda.is_available():
                torch.compiler.cudagraph_mark_step_begin()
            # with needs registering
            params = td.to_module(module, return_swap=True)
            result = module(x)
            params.to_module(module, return_swap=False)
            return result

    else:

        def call(x, td):
            if torch.cuda.is_available():
                torch.compiler.cudagraph_mark_step_begin()
            # with needs registering
            with td.to_module(module):
                return module(x)

    call_vmap = torch.vmap(call, (None, 0))
    if mode == "compile":
        call_vmap = torch.compile(call_vmap)
    elif mode == "compile-overhead":
        call_vmap = torch.compile(call_vmap, mode="reduce-overhead")

    if mode == "compile":
        call_vmap = torch.compile(call_vmap, fullgraph=not plain_decorator)
    elif mode == "compile-overhead":
        call_vmap = torch.compile(
            call_vmap, fullgraph=not plain_decorator, mode="reduce-overhead"
        )

    def call_with_backward(*args):
        call_vmap(*args).mean().backward()

    x = torch.randn(2, 2, 16)
    call_with_backward(x, td)
    call_with_backward(x, td)
    benchmark(call_with_backward, x, td)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
