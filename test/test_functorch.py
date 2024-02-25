# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pytest
import torch

from _utils_internal import expand_list, get_available_devices, TestTensorDictsBase

from tensordict import LazyStackedTensorDict, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.functional_modules import (
    get_functional,
    make_functional,
    repopulate_module,
)
from tensordict.utils import implement_for
from torch import nn
from torch.nn import Linear
from torch.utils._pytree import tree_map

try:
    from functorch import (
        make_functional_with_buffers as functorch_make_functional_with_buffers,
    )

    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    _has_functorch = True
    FUNCTORCH_ERR = ""
except ImportError as err:
    _has_functorch = False
    FUNCTORCH_ERR = str(err)


class TestVmap:
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
    def test_vmap_patch(self, moduletype, batch_params):
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
                params = params.expand(10, *params.batch_size).contiguous().lock_()
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
    def test_vmap_tdmodule_functorch(self, moduletype, batch_params):
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
    def test_vmap_tdmodule_nativebuilt(self, moduletype, batch_params):
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
                params = params.expand(10, *params.batch_size).lock_()
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
                params = params.expand(10, *params.batch_size).contiguous().lock_()
                td = vmap(tdmodule, (0, 0))(td, params)
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
    def test_vmap_tdsequence_functorch(self, moduletype, batch_params):
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
    def test_vmap_tdsequence_nativebuilt(self, moduletype, batch_params):
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
                params = params.expand(10, *params.batch_size).contiguous().lock_()
                td = vmap(tdmodule, (0, 0))(td, params)
            else:
                raise NotImplementedError
            z = td["z"]
            assert z.shape == torch.Size([10, 2, 3])

    def test_vmap_names(self):
        def fun(a, b):
            b["c"] = a["a"] + b["b"]
            return b

        a = TensorDict({"a": torch.randn(3, 4)}, [3])
        b = TensorDict({"b": torch.randn(3, 5, 4)}, [3, 5])

        a.names = ["0"]
        b.names = ["A", "B"]

        c = vmap(fun, (None, 1))(a, b)
        assert c.names == [None, "A"]

        a = TensorDict({"a": torch.randn(5, 4)}, [5])
        b = TensorDict({"b": torch.randn(3, 5, 4)}, [3, 5])

        a.names = ["0"]
        b.names = ["A", "B"]

        c = vmap(fun, (None, 0))(a, b)
        assert c.names == [None, "B"]

    @pytest.mark.parametrize("out_dim", [0, 1])
    @pytest.mark.parametrize("in_dim", [0, 1])
    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("lock_x", [True, False])
    @pytest.mark.parametrize("lock_y", [True, False])
    @pytest.mark.parametrize("key", ["a", ("a", "b")])
    def test_vmap_write_lazystack(
        self, in_dim, out_dim, stack_dim, lock_x, lock_y, key
    ):
        fun = vmap(
            lambda x, y: x.set(key, y.get(key) + x.get(key)),
            (in_dim, in_dim),
            (out_dim,),
        )
        td0 = TensorDict({key: [1.0]}, [1])
        td1 = TensorDict({key: [2.0]}, [1])
        x = LazyStackedTensorDict.lazy_stack([td0, td0.clone()], stack_dim)
        y = LazyStackedTensorDict.lazy_stack([td1, td1.clone()], stack_dim)
        if lock_x:
            x.lock_()
        if lock_y:
            y.lock_()
        out = fun(x, y)
        assert isinstance(out, LazyStackedTensorDict)
        if out_dim == 0:
            assert out.shape[out_dim] == x.shape[in_dim]
        else:
            assert out.shape[out_dim] == x.shape[in_dim]


class TestFunctionalization:
    @torch.no_grad()
    def test_setattr(self):
        # some modules (LSTM) rewrite __setattr__ which may break the logic
        # these are tested here
        x = torch.randn(2, 3, 10)
        lstm = nn.LSTM(10, 11)
        y0, (h0, c0) = lstm(x)
        params = make_functional(lstm)
        y1, (h1, c1) = lstm(x, params=params)
        y2, (h2, c2) = lstm(x, (h1, c1), params=params)
        torch.testing.assert_close(y1, y0)
        params.apply_(lambda p: p.data.zero_())
        y1, (h1, c1) = lstm(x, params=params)
        assert not torch.isclose(y0, y1).all()

    def test_swap(self):
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
            params.apply_(zero_grad, filter_empty=True)
            assert params.requires_grad

    def test_repopulate(self):
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

    def test_functional_restitute(self):
        module = nn.Transformer(32, nhead=4)
        params = make_functional(module, keep_params=True)
        params_clone = params.clone().zero_()
        data = torch.ones(1, 1, 32)
        y = module(data, data, params=params_clone)
        assert (y == 0).all()
        params_and_buffers = (
            TensorDict(dict(module.named_parameters()), [])
            .update(TensorDict(dict(module.named_buffers()), []))
            .unflatten_keys(".")
        )
        params = TensorDict({key: value for key, value in params.items(True, True)}, [])
        assert (params_and_buffers == params).all()


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
    y = vmap(net[0], (None, 0))(x, params["0"].expand(10).lock_())
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


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
)
def test_outputsize_vmap():
    a = TensorDict(
        {
            "a": torch.rand(7, 3, 6),
            "b": torch.rand(7, 3, 2),
        },
        batch_size=[7, 3],
    )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(6, 5)
            self.b = nn.Linear(2, 6)

        def forward(self, a, b):
            return self.a(a), self.b(b)

    # Not testing this as it will be deprecated soon
    # model = TensorDictModule(Model(), in_keys=["a", "b"],
    #                          out_keys=["out.a", "out.b"])
    # # option 1
    # fmodel, params = make_functional(model)
    # out = vmap(fmodel, in_dims=(None, 1), out_dims=1)(params, a)
    # assert out.shape == torch.Size([7, 3])

    # option 2
    model = TensorDictModule(Model(), in_keys=["a", "b"], out_keys=["out.a", "out.b"])
    params = make_functional(model)
    out = vmap(model, in_dims=(1, None), out_dims=1)(a, params)
    assert out.shape == torch.Size([7, 3])

    # option 2
    model = TensorDictModule(Model(), in_keys=["a", "b"], out_keys=["out.a", "out.b"])
    a = TensorDict(
        {
            "a": torch.rand(3, 4, 6),
            "b": torch.rand(3, 4, 2),
        },
        batch_size=[3, 4],
    )
    params = make_functional(model)
    params = params.expand(3, *params.shape).lock_()
    out = vmap(model, (0, 0))(a, params)
    assert out.shape == torch.Size([3, 4])


class TestGetFunctional:
    def test_get_functional(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(6, 5)
                self.b = nn.Linear(2, 6)

            def forward(self, a, b):
                return self.a(a), self.b(b)

        model = Model()
        get_functional(model)
        params = TensorDict(
            {
                "a": {"weight": torch.randn(5, 6), "bias": torch.randn(5)},
                "b": {"weight": torch.randn(6, 2), "bias": torch.randn(6)},
            },
            [],
        )
        a = torch.randn(6)
        b = torch.randn(2)
        v1a, v1b = model(a, b)
        _ = model(a, b, params=params)
        v2a, v2b = model(a, b)
        # check error
        with pytest.raises(
            TypeError,
            match="It seems you tried to provide the parameters",
        ):
            model(a, b, params)
        v3a, v3b = model(a, b)
        assert (v1a == v2a).all()
        assert (v1a == v3a).all()
        assert (v1b == v2b).all()
        assert (v1b == v3b).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)


class TestPyTree(TestTensorDictsBase):
    def test_pytree_map(self):
        td = TensorDict({"a": {"b": {"c": 1}, "d": 1}, "e": 1}, [])
        td = tree_map(lambda x: x + 1, td)
        assert (td == 2).all()

    def test_pytree_map_batch(self):
        td = TensorDict(
            {
                "a": TensorDict(
                    {
                        "b": TensorDict({"c": torch.ones(2, 3, 4)}, [2, 3]),
                        "d": torch.ones(2),
                    },
                    [2],
                ),
                "e": 1,
            },
            [],
        )
        td = tree_map(lambda x: x + 1, td)
        assert (td == 2).all()
        assert td.shape == torch.Size([])
        assert td["a"].shape == torch.Size([2])
        assert td["a", "b"].shape == torch.Size([2, 3])
        assert td["a", "b", "c"].shape == torch.Size([2, 3, 4])

    def test_pytree_vs_apply(self):
        td = TensorDict(
            {
                "a": TensorDict(
                    {
                        "b": TensorDict({"c": torch.ones(2, 3, 4)}, [2, 3]),
                        "d": torch.ones(2),
                    },
                    [2],
                ),
                "e": 1,
            },
            [],
        )
        td_pytree = tree_map(lambda x: x + 1, td)
        td_apply = td.apply(lambda x: x + 1)
        assert (td_apply == td_pytree).all()
        for v1, v2 in zip(td_pytree.values(True), td_apply.values(True)):
            # recursively checks the shape, including for the nested tensordicts
            assert v1.shape == v2.shape

    @implement_for("torch", "2.3")
    def test_map_with_path(self):
        def assert_path(path, tensor):
            assert path[0].key == "a"
            assert path[1].key == "b"
            assert path[2].key == "c"
            return tensor

        td = TensorDict({"a": {"b": {"c": [1]}}}, [1])
        torch.utils._pytree.tree_map_with_path(assert_path, td)

    @implement_for("torch", None, "2.3")
    def test_map_with_path(self):  # noqa: F811
        pytest.skip(reason="tree_map_with_path not implemented")

    @pytest.mark.parametrize("dest", get_available_devices())
    def test_device_map(self, dest):
        td = TensorDict({"a": {"b": {"c": [1]}, "d": [2]}}, [1], device="cpu")
        td_device = tree_map(lambda x: x.to(dest), td)
        if dest == torch.device("cpu"):
            assert td_device.device == torch.device("cpu")
        else:
            assert td_device.device is None

    def test_shape_map(self):
        td = TensorDict({"a": {"b": {"c": [1]}, "d": [2]}}, [1])
        td_no_shape = tree_map(lambda x: x.squeeze(), td)
        assert td_no_shape.shape == torch.Size([])
