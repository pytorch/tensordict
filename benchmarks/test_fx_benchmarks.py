# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

try:
    from tensordict.prototype.fx import symbolic_trace

    _HAS_FX = True
except ImportError:
    _HAS_FX = False

pytestmark = pytest.mark.skipif(
    not _HAS_FX, reason="tensordict.prototype.fx not available"
)


class Net(nn.Module):
    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class Masker(nn.Module):
    def forward(self, x, mask):
        return torch.softmax(x * mask, dim=1)


class FCLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.relu(self.fc(x))


class Output(nn.Module):
    def __init__(self, input_size, output_size=10):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)


@pytest.fixture()
def sequential_modules():
    net = TensorDictModule(
        Net(), in_keys=[("input", "x")], out_keys=[("intermediate", "x")]
    )
    masker = TensorDictModule(
        Masker(),
        in_keys=[("intermediate", "x"), ("input", "mask")],
        out_keys=[("output", "probabilities")],
    )
    module = TensorDictSequential(net, masker)
    graph_module = symbolic_trace(module)
    td = TensorDict(
        {
            "input": TensorDict(
                {"x": torch.rand(32, 100), "mask": torch.randint(2, size=(32, 10))},
                batch_size=[32],
            )
        },
        batch_size=[32],
    )
    return module, graph_module, td


@pytest.fixture()
def nested_modules():
    tdmodule1 = TensorDictModule(FCLayer(100, 50), ["input"], ["x"])
    tdmodule2 = TensorDictModule(FCLayer(50, 40), ["x"], ["x"])
    tdmodule3 = TensorDictModule(Output(40, 10), ["x"], ["probabilities"])
    module = TensorDictSequential(TensorDictSequential(tdmodule1, tdmodule2), tdmodule3)
    graph_module = symbolic_trace(module)
    td = TensorDict({"input": torch.rand(32, 100)}, [32])
    return module, graph_module, td


class TestFxSequential:
    def test_sequential_tensordict(self, sequential_modules, benchmark):
        module, _, td = sequential_modules
        benchmark(module, td)

    def test_sequential_graph_module(self, sequential_modules, benchmark):
        _, graph_module, td = sequential_modules
        benchmark(graph_module, td)


class TestFxNested:
    def test_nested_tensordict(self, nested_modules, benchmark):
        module, _, td = nested_modules
        benchmark(module, td)

    def test_nested_graph_module(self, nested_modules, benchmark):
        _, graph_module, td = nested_modules
        benchmark(graph_module, td)
