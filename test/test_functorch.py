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
from tensordict.utils import implement_for
from torch import nn
from torch.utils._pytree import tree_map

try:
    from functorch import (
        make_functional_with_buffers as functorch_make_functional_with_buffers,
    )

    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap  # noqa: TOR103

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
    @pytest.mark.parametrize("lock_x", [False, True])
    @pytest.mark.parametrize("lock_y", [False, True])
    @pytest.mark.parametrize("key", ["a", ("a", "b")])
    def test_vmap_write_lazystack(
        self, in_dim, out_dim, stack_dim, lock_x, lock_y, key
    ):
        def func(x, y):
            return x.set(key, y.get(key) + x.get(key))

        fun = vmap(
            func,
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
        if lock_x:
            with pytest.raises(RuntimeError, match="Cannot modify"):
                fun(x, y)
            return
        else:
            out = fun(x, y)
        assert (out[key] == 3).all()
        assert isinstance(out, LazyStackedTensorDict)
        if out_dim == 0:
            assert out.shape[out_dim] == x.shape[in_dim]
        else:
            assert out.shape[out_dim] == x.shape[in_dim]


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

    def test_pytree_lazy(self):
        td0 = TensorDict(
            {
                "a": torch.zeros(3),
                "b": torch.zeros(4),
                "c": {"d": 0, "e": "a string!"},
                "f": "another string",
            },
            [],
        )
        td1 = TensorDict(
            {
                "b": torch.zeros(5),
                "c": {"d": 0, "e": "a string!"},
                "f": "another string",
                "a": torch.zeros(3),
            },
            [],
        )
        td = TensorDict.lazy_stack([td0, td1])
        assert (tree_map(lambda x: x + 1, td) == td + 1).all()
        # With exclusive keys
        del td0["a"]
        assert (tree_map(lambda x: x + 1, td) == td + 1).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
