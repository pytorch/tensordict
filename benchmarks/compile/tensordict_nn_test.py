# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch
from tensordict import TensorDict, TensorDictParams

from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq


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


@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_mod_add(mode, benchmark):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td = TensorDict({"a": 0}, device=device)
    module = Mod(lambda x: x + 1, in_keys=["a"], out_keys=[("c", "d")])
    if mode == "compile":
        module = torch.compile(module, fullgraph=True)
    module(td)
    benchmark(module, td)


@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_mod_wrap(mode, benchmark):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = mlp(device)
    td = TensorDict({"a": torch.zeros(32, 3, device=device)}, device=device)
    module = Mod(net, in_keys=["a"], out_keys=[("c", "d")])
    if mode == "compile":
        module = torch.compile(module, fullgraph=True)
    module(td)
    benchmark(module, td)


@pytest.mark.parametrize("mode", ["eager", "compile"])
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
        module = torch.compile(module, fullgraph=True)
    module(td)
    benchmark(module, td)


@pytest.mark.parametrize("mode", ["eager", "compile"])
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
                out_keys=["hidden" if i < len(net) - 1 else ("c", "d")],
            )
            for i, layer in enumerate(net)
        ],
        delhidden,
    )
    if mode == "compile":
        module = torch.compile(module, fullgraph=True)
    module(td)
    benchmark(module, td)


@pytest.mark.parametrize("mode", ["eager", "compile"])
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
        call = torch.compile(call, fullgraph=True)

    x = torch.randn(2, 2, 16)
    if functional:
        call(x, td)
        benchmark(call, x, td)
    else:
        call(x)
        benchmark(call, x)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
