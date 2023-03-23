# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pytest
import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.prototype.fx import symbolic_trace


def test_tensordictmodule_trace_consistency():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.LazyLinear(1)

        def forward(self, x):
            logits = self.linear(x)
            return logits, torch.sigmoid(logits)

    module = TensorDictModule(
        Net(),
        in_keys=["input"],
        out_keys=[("outputs", "logits"), ("outputs", "probabilities")],
    )
    graph_module = symbolic_trace(module)

    tensordict = TensorDict({"input": torch.randn(32, 100)}, [32])

    module_out = TensorDict({}, [])
    graph_module_out = TensorDict({}, [])

    module(tensordict, tensordict_out=module_out)
    graph_module(tensordict, tensordict_out=graph_module_out)

    assert (
        module_out["outputs", "logits"] == graph_module_out["outputs", "logits"]
    ).all()
    assert (
        module_out["outputs", "probabilities"]
        == graph_module_out["outputs", "probabilities"]
    ).all()


def test_tensordictsequential_trace_consistency():
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

    tensordict = TensorDict(
        {
            "input": TensorDict(
                {"x": torch.rand(32, 100), "mask": torch.randint(2, size=(32, 10))},
                batch_size=[32],
            )
        },
        batch_size=[32],
    )

    module_out = TensorDict({}, [])
    graph_module_out = TensorDict({}, [])

    module(tensordict, tensordict_out=module_out)
    graph_module(tensordict, tensordict_out=graph_module_out)

    assert (
        graph_module_out["intermediate", "x"] == module_out["intermediate", "x"]
    ).all()
    assert (
        graph_module_out["output", "probabilities"]
        == module_out["output", "probabilities"]
    ).all()


def test_nested_tensordictsequential_trace_consistency():
    class Net(nn.Module):
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

    module1 = Net(100, 50)
    module2 = Net(50, 40)
    module3 = Output(40, 10)

    tdmodule1 = TensorDictModule(module1, ["input"], ["x"])
    tdmodule2 = TensorDictModule(module2, ["x"], ["x"])
    tdmodule3 = TensorDictModule(module3, ["x"], ["probabilities"])

    tdmodule = TensorDictSequential(
        TensorDictSequential(tdmodule1, tdmodule2), tdmodule3
    )
    graph_module = symbolic_trace(tdmodule)

    tensordict = TensorDict({"input": torch.rand(32, 100)}, [32])

    module_out = TensorDict({}, [])
    graph_module_out = TensorDict({}, [])

    tdmodule(tensordict, tensordict_out=module_out)
    graph_module(tensordict, tensordict_out=graph_module_out)

    assert (module_out["x"] == graph_module_out["x"]).all()
    assert (module_out["probabilities"] == graph_module_out["probabilities"]).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
