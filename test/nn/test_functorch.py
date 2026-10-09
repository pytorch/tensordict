# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import subprocess
import sys

import pytest
import tensordict
import tensordict.nn.functional_modules
import torch
from _utils_internal import expand_list, get_available_devices, TestTensorDictsBase
from functorch import (
    make_functional_with_buffers as functorch_make_functional_with_buffers,
)
from tensordict import LazyStackedTensorDict, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn, vmap
from torch.utils._pytree import tree_flatten_with_path, tree_map, tree_structure


class TestVmap:
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


class TestSetTensor:
    def test_set_tensor_deprecation(self):
        module = nn.Linear(2, 3)
        module.register_buffer("buf", torch.zeros(()))
        weight = nn.Parameter(torch.ones(3, 2))
        with pytest.warns(
            DeprecationWarning,
            match=r"^tensordict\.nn\.functional_modules\.set_tensor\(\) is "
            r"deprecated and will be removed in TensorDict 0\.17\.$",
        ) as record:
            tensordict.nn.functional_modules.set_tensor(module, "weight", weight)
        assert record[0].filename == __file__
        assert module._parameters["weight"] is weight
        buf = torch.ones(())
        with pytest.warns(DeprecationWarning, match="set_tensor"):
            tensordict.nn.functional_modules.set_tensor(module, "buf", buf)
        assert module._buffers["buf"] is buf
        bias = torch.zeros(3)
        with pytest.warns(DeprecationWarning, match="set_tensor"):
            tensordict.nn.functional_modules.set_tensor(module, "bias", bias)
        assert "bias" not in module._parameters
        assert module.__dict__["bias"] is bias

    def test_set_tensor_dict_deprecation(self):
        module = nn.Linear(2, 3)
        module.register_buffer("buf", torch.zeros(()))
        weight = nn.Parameter(torch.ones(3, 2))
        with pytest.warns(
            DeprecationWarning,
            match=r"^tensordict\.nn\.functional_modules\.set_tensor_dict\(\) is "
            r"deprecated and will be removed in TensorDict 0\.17\.$",
        ) as record:
            tensordict.nn.functional_modules.set_tensor_dict(
                module.__dict__, module, "weight", weight
            )
        assert record[0].filename == __file__
        assert module._parameters["weight"] is weight
        buf = torch.ones(())
        with pytest.warns(DeprecationWarning, match="set_tensor_dict"):
            tensordict.nn.functional_modules.set_tensor_dict(
                module.__dict__, module, "buf", buf
            )
        assert module._buffers["buf"] is buf
        bias = torch.zeros(3)
        with pytest.warns(DeprecationWarning, match="set_tensor_dict"):
            tensordict.nn.functional_modules.set_tensor_dict(
                module.__dict__, module, "bias", bias
            )
        assert "bias" not in module._parameters
        assert module.__dict__["bias"] is bias


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

    def test_map_with_path(self):
        def assert_path(path, tensor):
            assert path[0].key == "a"
            assert path[1].key == "b"
            assert path[2].key == "c"
            return tensor

        td = TensorDict({"a": {"b": {"c": [1]}}}, [1])
        torch.utils._pytree.tree_map_with_path(assert_path, td)

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

    def test_spec_batch_size(self):
        # The batch size is not part of the tree structure, the number of
        # batch dims is. The batch size is kept to rebuild a tensordict without
        # tensors.
        def spec(*batch_size):
            return tree_structure(
                TensorDict(a=torch.zeros(*batch_size, 2), batch_size=batch_size)
            )

        assert spec(4) == spec(6)
        assert spec(4) != spec(4, 2)
        _, spec_with_path = tree_flatten_with_path(
            TensorDict(a=torch.zeros(4, 2), batch_size=[4])
        )
        assert spec_with_path == spec(4)
        td = tree_map(lambda x: x, TensorDict(batch_size=[3]))
        assert td.batch_size == torch.Size([3])

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


# The names that ``from tensordict._pytree import *`` used to copy into the
# tensordict namespace.
_LEAKED_PYTREE_NAMES = [
    "Any",
    "Context",
    "Dict",
    "List",
    "MappingKey",
    "PYTREE_REGISTERED_LAZY_TDS",
    "PYTREE_REGISTERED_TDS",
    "Tuple",
    "cls",
    "defaultdict",
    "implement_for",
    "is_compiling",
    "register_pytree_node",
    "torch",
]


class TestPyTreeNamespace:
    @pytest.mark.parametrize("name", _LEAKED_PYTREE_NAMES)
    def test_leaked_name_not_in_namespace(self, name):
        assert name not in vars(tensordict)

    @pytest.mark.parametrize("name", _LEAKED_PYTREE_NAMES)
    def test_leaked_name_is_deprecated(self, name):
        with pytest.warns(
            DeprecationWarning,
            match=f"tensordict.{name} is deprecated and will be removed in TensorDict 0.17",
        ) as record:
            obj = getattr(tensordict, name)
        assert obj is getattr(tensordict._pytree, name)
        assert record[0].filename == __file__

    def test_unknown_attribute_raises(self):
        with pytest.raises(
            AttributeError, match="module 'tensordict' has no attribute 'not_a_name'"
        ):
            tensordict.not_a_name

    def test_import_does_not_warn(self):
        subprocess.run(
            [
                sys.executable,
                "-W",
                "error::DeprecationWarning",
                "-c",
                "import tensordict",
            ],
            check=True,
        )


def test_is_batchedtensor_is_deprecated():
    assert "is_batchedtensor" not in vars(tensordict)
    assert "is_batchedtensor" not in tensordict.__all__
    with pytest.warns(
        DeprecationWarning,
        match=r"^tensordict\.is_batchedtensor is deprecated and will be removed in "
        r"TensorDict 0\.17\. Use torch\._C\._functorch\.is_batchedtensor instead\.$",
    ) as record:
        from tensordict import is_batchedtensor
    assert record[0].filename == __file__
    assert is_batchedtensor is torch._C._functorch.is_batchedtensor
    assert not is_batchedtensor(torch.zeros(3))
    seen = []

    def func(x):
        seen.append(is_batchedtensor(x))
        return x

    vmap(func)(torch.zeros(2, 3))
    assert seen == [True]


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
