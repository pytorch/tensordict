# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import inspect

import pytest
import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from tensordict.prototype.fx import symbolic_trace


def test_fx():
    seq = Seq(
        Mod(lambda x: x + 1, in_keys=["x"], out_keys=["y"]),
        Mod(lambda x, y: (x * y).sqrt(), in_keys=["x", "y"], out_keys=["z"]),
        Mod(lambda z, x: z - z, in_keys=["z", "x"], out_keys=["a"]),
    )
    symbolic_trace(seq)


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, td: TensorDict) -> torch.Tensor:
        vals = td.values()  # pyre-ignore[6]
        return torch.cat([val._values for val in vals], dim=0)


def test_td_scripting() -> None:
    for cls in (TensorDict,):
        for name in dir(cls):
            method = inspect.getattr_static(cls, name)
            if isinstance(method, classmethod):
                continue
            elif isinstance(method, staticmethod):
                continue
            elif not callable(method):
                continue
            elif not name.startswith("__") or name in ("__init__", "__setitem__"):
                setattr(cls, name, torch.jit.unused(method))

    m = TestModule()
    td = TensorDict(
        a=torch.nested.nested_tensor([torch.ones((1,))], layout=torch.jagged)
    )
    m(td)
    m = torch.jit.script(m, example_inputs=(td,))
    m.code


def test_tensordictmodule_trace_consistency():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.LazyLinear(1)

        def forward(self, x):
            logits = self.linear(x)
            return logits, torch.sigmoid(logits)

    module = Mod(
        Net(),
        in_keys=["input"],
        out_keys=[("outputs", "logits"), ("outputs", "probabilities")],
    )
    graph_module = symbolic_trace(module)

    tensordict = TensorDict({"input": torch.randn(32, 100)}, [32])

    module_out = TensorDict()
    graph_module_out = TensorDict()

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

    net = Mod(Net(), in_keys=[("input", "x")], out_keys=[("intermediate", "x")])
    masker = Mod(
        Masker(),
        in_keys=[("intermediate", "x"), ("input", "mask")],
        out_keys=[("output", "probabilities")],
    )
    module = Seq(net, masker)
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

    module_out = TensorDict()
    graph_module_out = TensorDict()

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

    tdmodule1 = Mod(module1, ["input"], ["x"])
    tdmodule2 = Mod(module2, ["x"], ["x"])
    tdmodule3 = Mod(module3, ["x"], ["probabilities"])

    tdmodule = Seq(Seq(tdmodule1, tdmodule2), tdmodule3)
    graph_module = symbolic_trace(tdmodule)

    tensordict = TensorDict({"input": torch.rand(32, 100)}, [32])

    module_out = TensorDict()
    graph_module_out = TensorDict()

    tdmodule(tensordict, tensordict_out=module_out)
    graph_module(tensordict, tensordict_out=graph_module_out)

    assert (module_out["x"] == graph_module_out["x"]).all()
    assert (module_out["probabilities"] == graph_module_out["probabilities"]).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
