# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pytest
import torch
from _utils_internal import expand_list
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.functional_modules import make_functional, repopulate_module
from torch import nn
from torch.nn import Linear

try:
    from functorch import (
        make_functional_with_buffers as functorch_make_functional_with_buffers,
        vmap,
    )

    _has_functorch = True
    FUNCTORCH_ERR = ""
except ImportError as err:
    _has_functorch = False
    FUNCTORCH_ERR = str(err)


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
)
@pytest.mark.parametrize(
    "moduletype,batch_params",
    [
        ["linear", False],
        ["bn1", True],
        ["linear", True],
    ],
)
def test_vmap_patch(moduletype, batch_params):
    if moduletype == "linear":
        module = nn.Linear(3, 4)
    elif moduletype == "bn1":
        module = nn.BatchNorm1d(3)
    else:
        raise NotImplementedError
    if moduletype == "linear":
        params = make_functional(module)
        fmodule = module
        x = torch.randn(10, 1, 3)
        if batch_params:
            params = params.expand(10, *params.batch_size)
            y = vmap(fmodule, (0, 0))(x, params)
        else:
            y = vmap(fmodule, (0, None))(x, params)
        assert y.shape == torch.Size([10, 1, 4])
    elif moduletype == "bn1":
        params = make_functional(module)
        fmodule = module
        x = torch.randn(10, 2, 3)
        if batch_params:
            params = params.expand(10, *params.batch_size).contiguous()
            # buffers = buffers.expand(10, *buffers.batch_size).contiguous()
            y = vmap(fmodule, (0, 0))(x, params)
        else:
            raise NotImplementedError
        assert y.shape == torch.Size([10, 2, 3])


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
)
@pytest.mark.parametrize(
    "moduletype,batch_params",
    [
        ["linear", False],
        ["bn1", True],
        ["linear", True],
    ],
)
def test_vmap_tdmodule_functorch(moduletype, batch_params):
    if moduletype == "linear":
        module = nn.Linear(3, 4)
    elif moduletype == "bn1":
        module = nn.BatchNorm1d(3)
    else:
        raise NotImplementedError
    if moduletype == "linear":
        tdmodule = TensorDictModule(module, in_keys=["x"], out_keys=["y"])
        tdmodule, params, buffers = functorch_make_functional_with_buffers(tdmodule)
        x = torch.randn(10, 1, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = expand_list(params, 10)
            buffers = expand_list(buffers, 10)
            td = vmap(tdmodule, (0, 0, 0))(params, buffers, td)
        else:
            td = vmap(tdmodule, (None, None, 0))(params, buffers, td)
        y = td["y"]
        assert y.shape == torch.Size([10, 1, 4])
    elif moduletype == "bn1":
        tdmodule = TensorDictModule(module, in_keys=["x"], out_keys=["y"])
        tdmodule, params, buffers = functorch_make_functional_with_buffers(tdmodule)
        x = torch.randn(10, 2, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = expand_list(params, 10)
            buffers = expand_list(buffers, 10)
            td = vmap(tdmodule, (0, 0, 0))(params, buffers, td)
        else:
            raise NotImplementedError
        y = td["y"]
        assert y.shape == torch.Size([10, 2, 3])


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
)
@pytest.mark.parametrize(
    "moduletype,batch_params",
    [
        ["linear", False],
        ["bn1", True],
        ["linear", True],
    ],
)
def test_vmap_tdmodule_nativebuilt(moduletype, batch_params):
    if moduletype == "linear":
        module = nn.Linear(3, 4)
    elif moduletype == "bn1":
        module = nn.BatchNorm1d(3)
    else:
        raise NotImplementedError
    if moduletype == "linear":
        tdmodule = TensorDictModule(module, in_keys=["x"], out_keys=["y"])
        params = make_functional(tdmodule)
        x = torch.randn(10, 1, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = params.expand(10, *params.batch_size)
            td = vmap(tdmodule, (0, 0))(td, params)
        else:
            td = vmap(tdmodule, (0, None))(td, params)
        y = td["y"]
        assert y.shape == torch.Size([10, 1, 4])
    elif moduletype == "bn1":
        tdmodule = TensorDictModule(module, in_keys=["x"], out_keys=["y"])
        params = make_functional(tdmodule)
        x = torch.randn(10, 2, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = params.expand(10, *params.batch_size).contiguous()
            td = vmap(tdmodule, (0, 0))(td, params)
        else:
            raise NotImplementedError
        y = td["y"]
        assert y.shape == torch.Size([10, 2, 3])


def test_swap():
    def zero_grad(p):
        p.grad = torch.zeros_like(p.grad)

    net = nn.Sequential(
        nn.Linear(2, 2),
        nn.Linear(2, 2),
        nn.Linear(2, 2),
        nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)),
    )
    x = torch.randn(2, 2)
    params = make_functional(net)
    assert len(list(net.parameters())) == 0
    assert len(list(net.buffers())) == 0
    for _ in range(2):
        y = net(x, params)
        assert len(list(net.parameters())) == 0
        assert len(list(net.buffers())) == 0
        y.sum().backward()
        assert all(
            p.grad.pow(2).sum() > 0 if p.requires_grad else True
            for p in params.flatten_keys().values()
        )
        assert params.requires_grad
        params.apply_(zero_grad)
        assert params.requires_grad


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
)
@pytest.mark.parametrize(
    "moduletype,batch_params",
    [
        ["linear", False],
        ["bn1", True],
        ["linear", True],
    ],
)
def test_vmap_tdsequence_functorch(moduletype, batch_params):
    if moduletype == "linear":
        module1 = nn.Linear(3, 4)
        module2 = nn.Linear(4, 5)
    elif moduletype == "bn1":
        module1 = nn.BatchNorm1d(3)
        module2 = nn.BatchNorm1d(3)
    else:
        raise NotImplementedError
    if moduletype == "linear":
        tdmodule1 = TensorDictModule(module1, in_keys=["x"], out_keys=["y"])
        tdmodule2 = TensorDictModule(module2, in_keys=["y"], out_keys=["z"])
        tdmodule = TensorDictSequential(tdmodule1, tdmodule2)
        tdmodule, params, buffers = functorch_make_functional_with_buffers(tdmodule)
        x = torch.randn(10, 1, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = expand_list(params, 10)
            buffers = expand_list(buffers, 10)
            td = vmap(tdmodule, (0, 0, 0))(params, buffers, td)
        else:
            td = vmap(tdmodule, (None, None, 0))(params, buffers, td)
        z = td["z"]
        assert z.shape == torch.Size([10, 1, 5])
    elif moduletype == "bn1":
        tdmodule1 = TensorDictModule(module1, in_keys=["x"], out_keys=["y"])
        tdmodule2 = TensorDictModule(module2, in_keys=["y"], out_keys=["z"])
        tdmodule = TensorDictSequential(tdmodule1, tdmodule2)
        tdmodule, params, buffers = functorch_make_functional_with_buffers(tdmodule)
        x = torch.randn(10, 2, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = expand_list(params, 10)
            buffers = expand_list(buffers, 10)
            td = vmap(tdmodule, (0, 0, 0))(params, buffers, td)
        else:
            raise NotImplementedError
        z = td["z"]
        assert z.shape == torch.Size([10, 2, 3])


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
)
@pytest.mark.parametrize(
    "moduletype,batch_params",
    [
        ["linear", False],
        ["bn1", True],
        ["linear", True],
    ],
)
def test_vmap_tdsequence_nativebuilt(moduletype, batch_params):
    if moduletype == "linear":
        module1 = nn.Linear(3, 4)
        module2 = nn.Linear(4, 5)
    elif moduletype == "bn1":
        module1 = nn.BatchNorm1d(3)
        module2 = nn.BatchNorm1d(3)
    else:
        raise NotImplementedError
    if moduletype == "linear":
        tdmodule1 = TensorDictModule(module1, in_keys=["x"], out_keys=["y"])
        tdmodule2 = TensorDictModule(module2, in_keys=["y"], out_keys=["z"])
        tdmodule = TensorDictSequential(tdmodule1, tdmodule2)
        params = make_functional(tdmodule)
        assert {"0", "1"} == set(params["module"].keys())
        x = torch.randn(10, 1, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = params.expand(10, *params.batch_size)
            td = vmap(tdmodule, (0, 0))(td, params)
        else:
            td = vmap(tdmodule, (0, None))(td, params)
        z = td["z"]
        assert z.shape == torch.Size([10, 1, 5])
    elif moduletype == "bn1":
        tdmodule1 = TensorDictModule(module1, in_keys=["x"], out_keys=["y"])
        tdmodule2 = TensorDictModule(module2, in_keys=["y"], out_keys=["z"])
        tdmodule = TensorDictSequential(tdmodule1, tdmodule2)
        params = make_functional(tdmodule)
        assert {"0", "1"} == set(params["module"].keys())
        x = torch.randn(10, 2, 3)
        td = TensorDict({"x": x}, [10])
        if batch_params:
            params = params.expand(10, *params.batch_size).contiguous()
            td = vmap(tdmodule, (0, 0))(td, params)
        else:
            raise NotImplementedError
        z = td["z"]
        assert z.shape == torch.Size([10, 2, 3])


def test_repopulate():
    module = nn.ModuleList(
        [
            nn.Linear(3, 4),
            nn.BatchNorm1d(10),
            nn.Sequential(nn.GELU(), nn.Conv2d(2, 3, 4)),
        ]
    )
    params = set(module.named_parameters())
    buffers = set(module.named_buffers())
    assert len(params)
    assert len(buffers)

    params_td = make_functional(module)
    assert len(list(module.parameters())) == 0
    assert len(list(module.buffers())) == 0
    new_module = repopulate_module(module, params_td)
    assert new_module is module
    new_params = set(new_module.named_parameters())
    new_buffers = set(new_module.named_buffers())
    assert len(new_params)
    assert len(new_buffers)


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
)
def test_nested_modules():
    class LinearWithKwargs(Linear):
        """Checks that modules with kwargs work equally well."""

        def forward(self, x, stuff="ha"):
            return super().forward(x)

    net = nn.Sequential(
        Linear(3, 4),
        LinearWithKwargs(4, 5),
    )
    params = make_functional(net)

    x = torch.zeros(3)
    z = net(x, params=params)
    assert z.shape == torch.Size([5])

    y = net[0](x, params=params["0"])
    assert y.shape == torch.Size([4])
    z_bis = net[1](y, params=params["1"])
    assert z_bis.shape == torch.Size([5])

    assert torch.allclose(z, z_bis)

    y = vmap(net[0], (0, None))(x.expand(10, 3), params["0"])
    assert y.shape == torch.Size([10, 4])
    y = vmap(net[0], (None, 0))(x, params["0"].expand(10))
    assert y.shape == torch.Size([10, 4])


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
)
class TestNativeFunctorch:
    def test_vamp_basic(self):
        class MyModule(torch.nn.Module):
            def forward(self, tensordict):
                a = tensordict["a"]
                return TensorDict(
                    {"a": a}, tensordict.batch_size, device=tensordict.device
                )

        tensordict = TensorDict({"a": torch.randn(3)}, []).expand(4)
        out = vmap(MyModule(), (0,))(tensordict)
        assert out.shape == torch.Size([4])
        assert out["a"].shape == torch.Size([4, 3])

    def test_vamp_composed(self):
        class MyModule(torch.nn.Module):
            def forward(self, tensordict, tensor):
                a = tensordict["a"]
                return (
                    TensorDict(
                        {"a": a}, tensordict.batch_size, device=tensordict.device
                    ),
                    tensor,
                )

        tensor = torch.randn(3)
        tensordict = TensorDict({"a": torch.randn(3, 1)}, [3]).expand(4, 3)
        out = vmap(MyModule(), (0, None))(tensordict, tensor)

        assert out[0].shape == torch.Size([4, 3])
        assert out[1].shape == torch.Size([4, 3])
        assert out[0]["a"].shape == torch.Size([4, 3, 1])

    def test_vamp_composed_flipped(self):
        class MyModule(torch.nn.Module):
            def forward(self, tensordict, tensor):
                a = tensordict["a"]
                return (
                    TensorDict(
                        {"a": a}, tensordict.batch_size, device=tensordict.device
                    ),
                    tensor,
                )

        tensor = torch.randn(3).expand(4, 3)
        tensordict = TensorDict({"a": torch.randn(3, 1)}, [3])
        out = vmap(MyModule(), (None, 0))(tensordict, tensor)

        assert out[0].shape == torch.Size([4, 3])
        assert out[1].shape == torch.Size([4, 3])
        assert out[0]["a"].shape == torch.Size([4, 3, 1])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
