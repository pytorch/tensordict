import timeit

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.prototype.fx import symbolic_trace


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


if __name__ == "__main__":
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

    print(
        "forward, TensorDictSequential",
        timeit.timeit(
            "module(tensordict)",
            globals={"tensordict": tensordict, "module": module},
            number=10_000,
        ),
    )

    print(
        "forward, GraphModule",
        timeit.timeit(
            "module(tensordict)",
            globals={"tensordict": tensordict, "module": graph_module},
            number=10_000,
        ),
    )
