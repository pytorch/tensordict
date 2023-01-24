# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import timeit

# we use deepcopy as our implementation modifies the modules in-place
from copy import deepcopy

import torch
from functorch import make_functional_with_buffers as functorch_make_functional
from tensordict.nn.functional_modules import make_functional
from torch import nn

if __name__ == "__main__":
    # Creation
    net = nn.Sequential(
        nn.Linear(2, 2),
        nn.Linear(2, 2),
        nn.Linear(2, 2),
        nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)),
    )
    print(
        "instantiation, functorch:",
        timeit.timeit(
            "functorch_make_functional(deepcopy(net))",
            globals={
                "functorch_make_functional": functorch_make_functional,
                "net": net,
                "deepcopy": deepcopy,
            },
            number=1000,
        ),
    )

    print(
        "instantiation, tensordict:",
        timeit.timeit(
            "make_functional(deepcopy(net))",
            globals={
                "make_functional": make_functional,
                "net": net,
                "deepcopy": deepcopy,
            },
            number=1000,
        ),
    )

    # Execution
    x = torch.randn(2, 2)
    fmodule, params, buffers = functorch_make_functional(deepcopy(net))
    print(
        "exec, functorch:",
        timeit.timeit(
            "fmodule(params, buffers, x)",
            globals={"fmodule": fmodule, "params": params, "buffers": buffers, "x": x},
            number=10000,
        ),
    )

    fmodule = deepcopy(net)
    params = make_functional(fmodule)
    print(
        "exec, tensordict:",
        timeit.timeit(
            "fmodule(x, params=params)",
            globals={"fmodule": fmodule, "params": params, "x": x},
            number=10000,
        ),
    )
