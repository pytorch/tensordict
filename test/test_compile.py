# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import contextlib
import importlib.util
import inspect
import platform
from pathlib import Path
from typing import Any, Callable

import pytest

import torch
from packaging import version

from tensordict import (
    assert_close,
    NonTensorData,
    PYTREE_REGISTERED_LAZY_TDS,
    PYTREE_REGISTERED_TDS,
    tensorclass,
    TensorDict,
    TensorDictParams,
)
from tensordict.nn import (
    CudaGraphModule,
    InteractionType,
    ProbabilisticTensorDictModule as Prob,
    set_composite_lp_aggregate,
    TensorDictModule,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)

from tensordict.nn.functional_modules import _exclude_td_from_pytree

from torch.utils._pytree import SUPPORTED_NODES, tree_map

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)

_has_onnx = importlib.util.find_spec("onnxruntime", None) is not None

_v2_5 = TORCH_VERSION >= version.parse("2.5.0")

_IS_OSX = platform.system() == "Darwin"


def test_vmap_compile():
    # Since we monkey patch vmap we need to make sure compile is happy with it
    def func(x, y):
        return x + y

    x = torch.randn(3, 4)
    y = torch.randn(3)
    funcv = torch.vmap(func, (1, None))
    funcv(x, y)
    funcv_c = torch.compile(funcv, fullgraph=True)
    funcv_c(x, y)


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", [None, "reduce-overhead"])
class TestTD:
    def test_tensor_output(self, mode):
        def add_one(td):
            return td["a", "b"] + 1

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": 0}})
        assert add_one(data) == 1
        assert add_one_c(data) == 1
        assert add_one_c(data + 1) == 2

    def test_td_output(self, mode):
        def add_one(td):
            td["a", "c"] = td["a", "b"] + 1
            return td

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": 0}})
        assert add_one(data.clone())["a", "c"] == 1
        assert add_one_c(data.clone())["a", "c"] == 1
        assert add_one_c(data) is data

    @pytest.mark.parametrize("index_type", ["slice", "tensor", "int"])
    def test_td_index(self, index_type, mode):
        if index_type == "slice":

            def add_one(td):
                return td[:2] + 1

        elif index_type == "tensor":

            def add_one(td):
                return td[torch.tensor([0, 1])] + 1

        elif index_type == "int":

            def add_one(td):
                return td[0] + 1

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        if index_type == "int":
            assert (add_one(data)["a", "b"] == 1).all()
            assert (add_one_c(data)["a", "b"] == 1).all()
            assert add_one_c(data).shape == torch.Size([])
        else:
            assert (add_one(data)["a", "b"] == torch.arange(1, 3)).all()
            assert (add_one_c(data)["a", "b"] == torch.arange(1, 3)).all()
            assert add_one_c(data).shape == torch.Size([2])

    def test_stack(self, mode):
        def stack_tds(td0, td1):
            # return TensorDict.stack([td0, td1])
            return torch.stack([td0, td1])

        stack_tds_c = torch.compile(stack_tds, fullgraph=True, mode=mode)
        data0 = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        data1 = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        assert (stack_tds(data0, data1) == stack_tds_c(data0, data1)).all()

    def test_cat(self, mode):
        def cat_tds(td0, td1):
            # return TensorDict.cat([td0, td1])
            return torch.cat([td0, td1])

        cat_tds_c = torch.compile(cat_tds, fullgraph=True, mode=mode)
        data0 = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        data1 = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        assert (cat_tds(data0, data1) == cat_tds_c(data0, data1)).all()

    def test_reshape(self, mode):
        def reshape(td):
            return td.reshape(2, 2)

        reshape_c = torch.compile(reshape, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        data_reshape = reshape(data)
        _ = reshape_c(data)
        data_reshape_c = reshape_c(data)
        assert (data_reshape == data_reshape_c).all()

    def test_view(self, mode):
        def view(td):
            out = td.view(2, 2).clear_refs_for_compile_()
            return out

        view_c = torch.compile(view, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        data_view = view(data)
        _ = view_c(data)
        data_view_c = view_c(data)
        assert (data_view == data_view_c).all()

    def test_transpose(self, mode):
        def transpose(td):
            return td.transpose(0, 1).clear_refs_for_compile_()

        transpose_c = torch.compile(transpose, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(6).view(2, 3)}}, [2, 3])
        data_transpose = transpose(data)
        _ = transpose_c(data)
        data_transpose_c = transpose_c(data)
        assert (data_transpose == data_transpose_c).all()

    def test_unbind(self, mode):
        def unbind(td):
            return td.unbind(0)

        unbind_c = torch.compile(unbind, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        assert (unbind(data)[-1] == unbind_c(data)[-1]).all()

    def test_items(self, mode):
        def items(td):
            keys, vals = zip(*td.items(True, True))
            return keys, vals

        items_c = torch.compile(items, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        keys, vals = items(data)
        keys_c, vals_c = items_c(data)

        def assert_eq(x, y):
            assert (x == y).all()

        assert keys == keys_c
        torch.utils._pytree.tree_map(assert_eq, vals, vals_c)

    @pytest.mark.parametrize("recurse", [True, False])
    @pytest.mark.parametrize("lock", [True, False])
    def test_clone(self, recurse, lock, mode):
        def clone(td: TensorDict):
            return td.clone(recurse=recurse)

        clone_c = torch.compile(clone, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": 0, "c": 1}})
        if lock:
            data = data.lock_()
        data_c = clone(data)
        _ = clone_c(data)
        data_c_c = clone_c(data)
        assert_close(data_c, data_c_c)
        assert clone_c(data) is not data
        if recurse:
            assert clone_c(data)["a", "b"] is not data["a", "b"]
        else:
            assert clone_c(data)["a", "b"] is data["a", "b"]

    @pytest.mark.parametrize("recurse", [True, False])
    def test_flatten_keys(self, recurse, mode):
        def flatten_keys(td: TensorDict):
            return td.flatten_keys().clear_refs_for_compile_()

        flatten_keys_c = torch.compile(flatten_keys, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": 0, "c": 1}})
        data_f = flatten_keys(data)
        _ = flatten_keys(data)
        data_f_c = flatten_keys(data)
        assert_close(data_f, data_f_c)
        assert flatten_keys_c(data) is not data
        assert flatten_keys_c(data)["a.b"] is data["a", "b"]

    @pytest.mark.parametrize("recurse", [True, False])
    def test_unflatten_keys(self, recurse, mode):
        def unflatten_keys(td: TensorDict):
            return td.unflatten_keys().clear_refs_for_compile_()

        unflatten_keys_c = torch.compile(unflatten_keys, fullgraph=True, mode=mode)
        data = TensorDict({"a.b": 0, "a.c": 1})
        data_t = unflatten_keys(data)
        _ = unflatten_keys_c(data)
        data_t_c = unflatten_keys_c(data)
        assert_close(data_t, data_t_c)
        assert unflatten_keys_c(data) is not data
        assert unflatten_keys_c(data)["a", "b"] is data["a.b"]

    def test_names(self, mode):
        import torch._dynamo.exc

        def make_td_with_names(data):
            return TensorDict(data, batch_size=[1, 2], names=["d0", "d1"])

        data_dict = {
            "a": torch.randn(1, 2, 3),
            "b": torch.zeros(1, 2, 3, dtype=torch.bool),
        }
        make_td_with_names_c = torch.compile(
            make_td_with_names, fullgraph=True, mode=mode
        )
        make_td_with_names(data_dict)
        # with pytest.raises(torch._dynamo.exc.Unsupported):
        make_td_with_names_c(data_dict)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="cuda required to test device casting"
    )
    @pytest.mark.parametrize("has_device", [True, False])
    def test_to(self, has_device, mode):
        device = "cuda:0"

        def test_to_device(td):
            return td.to(device)

        td = TensorDict(
            {"a": torch.randn(1, 2, 3), "b": torch.zeros(1, 2, 3, dtype=torch.bool)},
            batch_size=[1, 2],
            device="cpu" if has_device else None,
        )
        test_to_device_c = torch.compile(test_to_device, fullgraph=True, mode=mode)
        # td_device = test_to_device(td)
        _ = test_to_device_c(td)
        td_device_c = test_to_device_c(td)
        assert td_device_c.batch_size == td.batch_size
        assert td_device_c.device == torch.device(device)

    def test_lock(self, mode):
        def locked_op(td):
            # Adding stuff uses cache, check that this doesn't break
            td2 = td + 1
            td3 = td + td2
            return td3.clear_refs_for_compile_()

        td = TensorDict(
            {"a": torch.randn(1, 2, 3), "b": torch.zeros(1, 2, 3, dtype=torch.bool)},
            batch_size=[1, 2],
            device="cpu",
            lock=True,
        )
        locked_op_c = torch.compile(locked_op, fullgraph=True, mode=mode)
        td_op = locked_op(td)
        # no warning the second time this is run
        with (
            pytest.warns(UserWarning, match="Using lock_")
            if mode is None
            else contextlib.nullcontext()
        ):
            _ = locked_op_c(td)
        td_op_c = locked_op_c(td)
        assert (td_op == td_op_c).all()

    def test_lock_inplace(self, mode):
        def locked_op(td):
            # Adding stuff uses cache, check that this doesn't break
            td += 1
            td += td
            return td

        td = TensorDict(
            {"a": torch.randn(1, 2, 3), "b": torch.ones(1, 2, 3, dtype=torch.int64)},
            batch_size=[1, 2],
            device="cpu",
            lock=True,
        )
        locked_op_c = torch.compile(locked_op, fullgraph=True, mode=mode)
        td_op = locked_op(td)
        # no warning the second time this is run
        _ = locked_op_c(td)
        td_op_c = locked_op_c(td)
        assert (td_op == td_op_c).all()

    # Memmap is currently not supported
    # def test_memmap(self, mode, tmpdir):
    #     def locked_op(td):
    #         # Adding stuff uses cache, check that this doesn't break
    #         return td.apply(lambda x: x+1)
    #
    #     td = TensorDict(
    #         {"a": torch.randn(1, 2, 3), "b": torch.ones(1, 2, 3, dtype=torch.int64)},
    #         batch_size=[1, 2],
    #         device="cpu",
    #     ).memmap_(tmpdir)
    #     locked_op_c = torch.compile(locked_op, fullgraph=True, mode=mode)
    #     td_op = locked_op(td)
    #     # no warning the second time this is run
    #     _ = locked_op_c(td)
    #     td_op_c = locked_op_c(td)
    #     assert (td_op == td_op_c).all()


@tensorclass
class MyClass:
    a: "MyClass"
    b: Any = None
    c: Any = None


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", [None, "reduce-overhead"])
class TestTC:
    def test_tc_tensor_output(self, mode):
        def add_one(td):
            return td.a.b + 1

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = MyClass(MyClass(a=None, b=torch.zeros(())))
        assert add_one(data) == 1
        assert add_one_c(data) == 1
        assert add_one_c(data + 1) == 2

    def test_tc_items(self, mode):
        def items(td):
            keys, vals = zip(*td.items(True, True))
            return keys, vals

        items_c = torch.compile(items, fullgraph=True, mode=mode)
        data = MyClass(MyClass(a=None, b=torch.zeros(())))
        keys, vals = items(data)
        keys_c, vals_c = items_c(data)

        def assert_eq(x, y):
            assert (x == y).all()

        assert keys == keys_c
        torch.utils._pytree.tree_map(assert_eq, vals, vals_c)

    def test_tc_output(self, mode):
        def add_one(td):
            td.a.c = td.a.b + 1
            return td

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        data = MyClass(a=MyClass(a=None, b=torch.zeros(())))
        assert add_one(data.clone()).a.c == 1
        assert add_one_c(data.clone()).a.c == 1
        assert add_one_c(data) is data

    def test_tc_arithmetic(self, mode):
        def add_one(td):
            return td + 1

        data = MyClass(a=MyClass(a=None, b=torch.zeros(())))

        eager = add_one(data.clone())

        add_one_c = torch.compile(add_one, fullgraph=True, mode=mode)
        compiled = add_one_c(data.clone())

        assert isinstance(eager.a, MyClass)
        assert eager.a.b == 1

        assert isinstance(compiled.a, MyClass)
        # TODO: breaks because a is not cast to a MyClass but is a dict
        assert compiled.a.b == 1
        assert add_one_c(data) is not data

    def test_tc_arithmetic_other_tc(self, mode):
        def add_self(td):
            return td + td

        data = MyClass(a=MyClass(a=None, b=torch.ones(())))

        eager = add_self(data.clone())

        add_self_c = torch.compile(add_self, fullgraph=True, mode=mode)
        compiled = add_self_c(data.clone())

        assert isinstance(eager.a, MyClass)
        assert eager.a.b == 2

        assert isinstance(compiled.a, MyClass)
        # TODO: breaks because a is not cast to a MyClass but is a dict
        assert compiled.a.b == 2
        assert add_self_c(data) is not data

    @pytest.mark.parametrize("index_type", ["slice", "tensor", "int"])
    def test_tc_index(self, index_type, mode):
        if index_type == "slice":

            def index(td):
                return td[:2]

        elif index_type == "tensor":

            def index(td):
                return td[torch.tensor([0, 1])]

        elif index_type == "int":

            def index(td):
                return td[0]

        index_c = torch.compile(index, fullgraph=True, mode=mode)
        data = MyClass(
            a=MyClass(a=None, b=torch.arange(3), batch_size=[3]), batch_size=[3]
        )

        indexed_data_eager = index(data)
        indexed_data_compile = index_c(data)
        if index_type == "int":
            assert (indexed_data_eager.a.b == 0).all()
            assert (indexed_data_compile.a.b == 0).all()

            assert isinstance(indexed_data_eager, MyClass)
            assert isinstance(indexed_data_compile, MyClass)

            assert isinstance(indexed_data_eager.a, MyClass)
            assert isinstance(indexed_data_compile.a, MyClass)

            assert indexed_data_eager.shape == torch.Size([])
            assert indexed_data_compile.shape == torch.Size([])

        else:
            assert (indexed_data_eager.a.b == torch.arange(0, 2)).all()
            assert (indexed_data_compile.a.b == torch.arange(0, 2)).all()
            assert isinstance(indexed_data_eager, MyClass)
            assert isinstance(indexed_data_compile, MyClass)
            assert isinstance(indexed_data_eager.a, MyClass)
            assert isinstance(indexed_data_compile.a, MyClass)
            assert indexed_data_eager.shape == torch.Size([2])
            assert indexed_data_compile.shape == torch.Size([2])

    def test_tc_stack(self, mode):
        def stack_tds(td0, td1):
            # return TensorDict.stack([td0, td1])
            return torch.stack([td0, td1])

        data0 = MyClass(
            a=MyClass(a=None, b=torch.arange(3), batch_size=[3]), batch_size=[3]
        )
        data1 = MyClass(
            a=MyClass(a=None, b=torch.arange(3, 6), batch_size=[3]), batch_size=[3]
        )
        stack_eager = stack_tds(data0, data1)

        stack_tds_c = torch.compile(stack_tds, fullgraph=True, mode=mode)
        stack_compile = stack_tds_c(data0, data1)

        assert (stack_eager == stack_compile).all()

    def test_tc_cat(self, mode):
        def cat_tds(td0, td1):
            return torch.cat([td0, td1])

        cat_tds_c = torch.compile(cat_tds, fullgraph=True, mode=mode)
        data0 = MyClass(
            a=MyClass(a=None, b=torch.arange(3), batch_size=[3]), batch_size=[3]
        )
        data1 = MyClass(
            a=MyClass(a=None, b=torch.arange(3, 6), batch_size=[3]), batch_size=[3]
        )
        assert (cat_tds(data0, data1) == cat_tds_c(data0, data1)).all()

    def test_tc_reshape(self, mode):
        def reshape(td):
            return td.reshape(2, 2)

        reshape_c = torch.compile(reshape, fullgraph=True, mode=mode)
        data = MyClass(
            a=MyClass(a=None, b=torch.arange(4), batch_size=[4]), batch_size=[4]
        )
        assert (reshape(data) == reshape_c(data)).all()

    def test_tc_unbind(self, mode):
        def unbind(td):
            return td.unbind(0)

        unbind_c = torch.compile(unbind, fullgraph=True, mode=mode)
        data = MyClass(
            a=MyClass(a=None, b=torch.arange(4), batch_size=[4]), batch_size=[4]
        )
        assert (unbind(data)[-1] == unbind_c(data)[-1]).all()

    @pytest.mark.parametrize("recurse", [True, False])
    def test_tc_clone(self, recurse, mode):
        def clone(td: TensorDict):
            return td.clone(recurse=recurse)

        clone_c = torch.compile(clone, fullgraph=True, mode=mode)
        data = MyClass(
            a=MyClass(a=None, b=torch.arange(4), batch_size=[4]), batch_size=[4]
        )
        assert_close(clone_c(data), clone(data))
        assert clone_c(data) is not data
        if recurse:
            assert clone_c(data).a.b is not data.a.b
        else:
            assert clone_c(data).a.b is data.a.b

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="cuda required to test device casting"
    )
    @pytest.mark.parametrize("has_device", [True, False])
    def test_tc_to(self, has_device, mode):
        device = "cuda:0"

        def test_to_device(tc):
            return tc.to(device)

        data = MyClass(
            a=MyClass(a=None, b=torch.arange(4), batch_size=[4]),
            batch_size=[4],
            device="cpu" if has_device else None,
        )
        test_to_device_c = torch.compile(test_to_device, fullgraph=True, mode=mode)
        # tc_device = test_to_device(tc)
        _ = test_to_device_c(data)
        tc_device_c = test_to_device_c(data)
        assert tc_device_c.batch_size == data.batch_size
        assert tc_device_c.device == torch.device(device)

    def test_tc_lock(self, mode):
        def locked_op(tc):
            # Adding stuff uses cache, check that this doesn't break
            tc2 = tc + 1
            tc3 = tc + tc2
            return tc3

        data = MyClass(
            a=MyClass(a=None, b=torch.arange(4), batch_size=[4]),
            batch_size=[4],
            device="cpu",
        ).lock_()
        locked_op_c = torch.compile(locked_op, fullgraph=True, mode=mode)
        tc_op = locked_op(data)
        # no warning the second time this is run
        with (
            pytest.warns(UserWarning, match="Using lock_")
            if mode is None
            else contextlib.nullcontext()
        ):
            _ = locked_op_c(data)
        tc_op_c = locked_op_c(data)
        assert (tc_op == tc_op_c).all()

    def test_td_new_unsafe(self, mode):

        class MyTd(TensorDict):
            pass

        def func_td():
            return TensorDict._new_unsafe(a=torch.randn(3), batch_size=torch.Size(()))

        @torch.compile(fullgraph=True, mode=mode)
        def func_c_td():
            return TensorDict._new_unsafe(a=torch.randn(3), batch_size=torch.Size(()))

        def func_mytd():
            return MyTd._new_unsafe(a=torch.randn(3), batch_size=torch.Size(()))

        # This will graph break
        @torch.compile(mode=mode)
        def func_c_mytd():
            return MyTd._new_unsafe(a=torch.randn(3), batch_size=torch.Size(()))

        assert type(func_td()) is type(func_c_td())
        assert type(func_mytd()) is type(func_c_mytd())


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", [None, "reduce-overhead"])
class TestNN:
    def test_func(self, mode):
        td = TensorDict({"a": 0})
        module = Mod(
            lambda x: x + 1, in_keys=[(((("a",),),),)], out_keys=[(((("a",),),),)]
        )
        module_compile = torch.compile(module, fullgraph=True, mode=mode)
        module_compile(td)
        assert_close(module(td), module_compile(td))

    def test_linear(self, mode):
        net = torch.nn.Linear(4, 5)
        module = Mod(net, in_keys=[(((("a",),),),)], out_keys=[("c", "d")])
        module_compile = torch.compile(module, fullgraph=True, mode=mode)
        td = TensorDict({"a": torch.randn(32, 4)}, [32])
        assert_close(module(td), module_compile(td))

    def test_seq(self, mode):
        net0 = torch.nn.Linear(4, 5)
        module0 = Mod(net0, in_keys=["a"], out_keys=["hidden"])
        net1 = torch.nn.Linear(5, 6)
        module1 = Mod(net1, in_keys=["hidden"], out_keys=[("c", "d")])
        module = Seq(module0, module1)
        module_compile = torch.compile(module, fullgraph=True, mode=mode)
        td = TensorDict({"a": torch.randn(32, 4)}, [32])
        assert_close(module(td), module_compile(td))

        assert module_compile(td) is td

    def test_seq_lmbda(self, mode):
        net0 = torch.nn.Linear(4, 5)
        module0 = Mod(net0, in_keys=["a"], out_keys=["hidden"])
        net1 = torch.nn.Linear(5, 6)
        module1 = Mod(net1, in_keys=["hidden"], out_keys=[("c", "d")])

        def remove_hidden(td):
            del td["hidden"]
            return td

        module = Seq(lambda td: td.copy(), module0, module1, remove_hidden)
        module_compile = torch.compile(module, fullgraph=True, mode=mode)
        td = TensorDict({"a": torch.randn(32, 4)}, [32])
        module_compile(td)
        assert_close(module(td), module_compile(td))
        assert module_compile(td) is not td

    @pytest.mark.skipif(not _v2_5, reason="requires torch 2.5 or higher")
    def test_dispatch_nontensor(self, mode):
        torch._dynamo.reset_code_caches()

        # Non tensor
        x = torch.randn(3)
        y = None
        mod = Seq(
            Mod(lambda x, y: x[y, :], in_keys=["x", "y"], out_keys=["_z"]),
            Mod(lambda x, z: z * x, in_keys=["x", "_z"], out_keys=["out"]),
        )
        assert mod(x=x, y=y)[-1].shape == torch.Size((1, 3))
        mod_compile = torch.compile(mod, fullgraph=_v2_5, mode=mode)
        torch.testing.assert_close(mod(x=x, y=y), mod_compile(x=x, y=y))

    @pytest.mark.skipif(not _v2_5, reason="requires torch 2.5 or higher")
    def test_dispatch_tensor(self, mode):
        torch._dynamo.reset_code_caches()

        x = torch.randn(3)
        y = torch.randn(3)
        mod = Seq(
            Mod(lambda x, y: x + y, in_keys=["x", "y"], out_keys=["z"]),
            Mod(lambda x, z: z * x, in_keys=["x", "z"], out_keys=["out"]),
        )
        mod(x=x, y=y)
        mod_compile = torch.compile(mod, fullgraph=_v2_5, mode=mode)
        torch.testing.assert_close(mod(x=x, y=y), mod_compile(x=x, y=y))

    @set_composite_lp_aggregate(False)
    def test_prob_module_with_kwargs(self, mode):
        kwargs = TensorDictParams(
            TensorDict(scale=1.0, validate_args=NonTensorData(False)), no_convert=True
        )
        dist_cls = torch.distributions.Normal
        mod = Mod(torch.nn.Linear(3, 3), in_keys=["inp"], out_keys=["loc"])
        prob_mod = Seq(
            mod,
            Prob(
                in_keys=["loc"],
                out_keys=["sample"],
                return_log_prob=True,
                distribution_class=dist_cls,
                distribution_kwargs=kwargs,
                default_interaction_type=InteractionType.RANDOM,
            ),
        )
        # check that the scale is in the buffers
        assert len(list(prob_mod.buffers())) == 1
        prob_mod(TensorDict(inp=torch.randn(3)))
        prob_mod_c = torch.compile(prob_mod, fullgraph=True, mode=mode)
        prob_mod_c(TensorDict(inp=torch.randn(3)))


@pytest.mark.skipif(
    TORCH_VERSION <= version.parse("2.4.0"), reason="requires torch>2.4"
)
@pytest.mark.parametrize("mode", [None, "reduce-overhead"])
class TestFunctional:
    def test_functional_error(self, mode):
        TORCHDYNAMO_INLINE_INBUILT_NN_MODULES = (
            torch._dynamo.config.inline_inbuilt_nn_modules
        )
        torch._dynamo.config.inline_inbuilt_nn_modules = True
        module = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 5),
        )
        td = TensorDict.from_module(module)
        td_zero = TensorDictParams(td.data.clone())
        td_zero.zero_()

        torch._dynamo.config.inline_inbuilt_nn_modules = False
        try:

            def call(x, td):
                with td.to_module(module):
                    return module(x)

            call_compile = torch.compile(call, fullgraph=True, mode=mode)
            x = torch.randn(2, 3)
            with pytest.raises(
                RuntimeError, match="torch._dynamo.config.inline_inbuilt_nn_modules"
            ):
                call_compile(x, td_zero)
        finally:
            if torch._dynamo.config.inline_inbuilt_nn_modules is not None:
                torch._dynamo.config.inline_inbuilt_nn_modules = (
                    TORCHDYNAMO_INLINE_INBUILT_NN_MODULES
                )

    # in-place modif raises an error even if fullgraph=False
    @pytest.mark.parametrize("modif_param", [False])
    @pytest.mark.skipif(
        TORCH_VERSION <= version.parse("2.5.0"), reason="requires torch>2.5"
    )
    def test_functional(self, modif_param, mode):

        # TODO: UNTESTED
        class MessUpParams(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros(()))

            def forward(self, x):
                self.param.data.add_(1)
                return x * 1

        module = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 5),
        )
        if modif_param:
            module.append(MessUpParams())

        orig_params = list(module.parameters())
        td = TensorDict.from_module(module)
        td_zero = TensorDictParams(td.data.clone())
        td_zero.zero_()

        def call(x, td):
            with td.to_module(module):
                y = module(x)
            td.clear_refs_for_compile_()
            return y

        call_compile = torch.compile(call, fullgraph=True, mode=mode)
        x = torch.randn(2, 3)
        assert (call(x, td_zero) == 0).all()
        assert all(
            p_new is p_orig for p_new, p_orig in zip(module.parameters(), orig_params)
        )
        assert (call(x, td_zero) == 0).all()
        assert all(
            p_new is p_orig for p_new, p_orig in zip(module.parameters(), orig_params)
        )
        if modif_param:
            assert td_zero["3", "param"] == 2
        else:
            assert (td_zero == 0).all()
        # torch.testing.assert_close(call_compile(x, td_zero), module(x))

        td.to_module(module)
        call_compile(x, td_zero)
        assert (call_compile(x, td_zero) == 0).all()
        assert all(
            p_new is p_orig for p_new, p_orig in zip(module.parameters(), orig_params)
        )
        assert (call_compile(x, td_zero) == 0).all()
        assert all(
            p_new is p_orig for p_new, p_orig in zip(module.parameters(), orig_params)
        )
        if modif_param:
            assert td_zero["3", "param"] == 4
        else:
            assert (td_zero == 0).all()

    # in-place modif raises an error even if fullgraph=False
    @pytest.mark.skipif(
        TORCH_VERSION <= version.parse("2.5.0"), reason="requires torch>2.5"
    )
    def test_vmap_functional(self, mode):
        module = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 5),
        )

        td = TensorDict.from_module(module)
        td_zero = TensorDictParams(td.data.expand(10).clone().zero_())

        def call(x, td):
            with td.to_module(module):
                result = module(x)
            return result

        vmap_call = torch.vmap(call, (None, 0))
        call_compile = torch.compile(vmap_call, fullgraph=True, mode=mode)
        x = torch.randn(2, 3)

        assert (vmap_call(x, td_zero) == 0).all()
        assert (TensorDict.from_module(module) == td).all()
        assert (td_zero == 0).all()

        call_compile(x, td_zero)
        assert (TensorDict.from_module(module) == td).all()
        assert (call_compile(x, td_zero) == 0).all()
        assert (TensorDict.from_module(module) == td).all()
        assert (td_zero == 0).all()


@pytest.mark.skipif(not _v2_5, reason="Requires PT>=2.5")
class TestExport:
    def test_export_module(self):
        torch._dynamo.reset_code_caches()
        tdm = Mod(lambda x, y: x * y, in_keys=["x", "y"], out_keys=["z"])
        x = torch.randn(3)
        y = torch.randn(3)
        out = torch.export.export(tdm, args=(), kwargs={"x": x, "y": y})
        assert (out.module()(x=x, y=y) == tdm(x=x, y=y)).all()

    def test_export_seq(self):
        torch._dynamo.reset_code_caches()
        tdm = Seq(
            Mod(lambda x, y: x * y, in_keys=["x", "y"], out_keys=["z"]),
            Mod(lambda z, x: z + x, in_keys=["z", "x"], out_keys=["out"]),
        )
        x = torch.randn(3)
        y = torch.randn(3)
        out = torch.export.export(tdm, args=(), kwargs={"x": x, "y": y})
        torch.testing.assert_close(out.module()(x=x, y=y), tdm(x=x, y=y))

    # This tests passes but there are various things that need to be fixed:
    #  - we cannot use vmap directly
    #  - if we use strict=True, there's an error due to the fact that export ignores
    #    the replacement of the params (ie, params are still on "meta" and the values
    #    after the call on the exported module don't match the original ones).
    # Currently only works with strict=False, because export fails to see that
    #  the params in the module have changed and are not 'meta' anymore => this
    #  is symptomatic of export failing to see the functional call
    @pytest.mark.parametrize("strict", [False])  # , True])
    def test_export_with_td_params(self, strict):
        module = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 5),
        )
        module_td = TensorDictParams(
            TensorDict.from_module(module).data.expand(2).clone()
        )
        assert all(
            isinstance(p, torch.nn.Parameter) for p in module_td.values(True, True)
        )

        class MyModule(torch.nn.Module):
            def __init__(self, td_params):
                super().__init__()
                self.tdparams = td_params
                self.arch = torch.nn.Sequential(
                    torch.nn.Linear(3, 4, device="meta"),
                    torch.nn.Linear(4, 5, device="meta"),
                )

            def forward(self, x):
                # vmap with params currently fails
                #  return torch.vmap(self.batch_forward, (0, None))(self.tdparams, x)
                return torch.stack(
                    [self.batch_forward(p, x) for p in self.tdparams.unbind(0)]
                )

            def batch_forward(self, params, x):
                with params.to_module(self.arch):
                    return self.arch(x)
                # This could be an option but dynamo doesn't know how to trace through state_dict ops
                # sd = self.arch.state_dict()
                # try:
                #     self.arch.load_state_dict(params.flatten_keys().to_dict(), assign=True)
                #     return self.arch(x)
                # finally:
                #     self.arch.load_state_dict(sd, assign=True)

        m = MyModule(module_td)
        x = torch.randn(3)
        assert m(x).shape == (2, 5)
        exported_module = torch.export.export(
            m,
            args=(),
            kwargs={"x": x},
            strict=strict,
        )
        torch.testing.assert_close(exported_module.module()(x=x), m(x))


@pytest.mark.skipif(not _has_onnx, reason="ONNX is not available")
class TestONNXExport:
    def test_onnx_export_module(self, tmpdir):
        tdm = Mod(lambda x, y: x * y, in_keys=["x", "y"], out_keys=["z"])
        x = torch.randn(3)
        y = torch.randn(3)
        torch_input = {"x": x, "y": y}
        onnx_program = torch.onnx.dynamo_export(tdm, **torch_input)

        onnx_input = onnx_program.adapt_torch_inputs_to_onnx(**torch_input)

        path = Path(tmpdir) / "file.onnx"
        onnx_program.save(str(path))
        import onnxruntime

        ort_session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )

        def to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

        onnxruntime_input = {
            k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)
        }

        onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
        torch.testing.assert_close(
            torch.as_tensor(onnxruntime_outputs[0]), tdm(x=x, y=y)
        )

    def test_onnx_export_seq(self, tmpdir):
        tdm = Seq(
            Mod(lambda x, y: x * y, in_keys=["x", "y"], out_keys=["z"]),
            Mod(lambda z, x: z + x, in_keys=["z", "x"], out_keys=["out"]),
        )
        x = torch.randn(3)
        y = torch.randn(3)
        torch_input = {"x": x, "y": y}
        torch.onnx.dynamo_export(tdm, x=x, y=y)
        onnx_program = torch.onnx.dynamo_export(tdm, **torch_input)

        onnx_input = onnx_program.adapt_torch_inputs_to_onnx(**torch_input)

        path = Path(tmpdir) / "file.onnx"
        onnx_program.save(str(path))
        import onnxruntime

        ort_session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )

        def to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

        onnxruntime_input = {
            k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)
        }

        onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
        torch.testing.assert_close(
            tree_map(torch.as_tensor, onnxruntime_outputs), tdm(x=x, y=y)
        )


@pytest.mark.skipif(
    TORCH_VERSION <= version.parse("2.4.1"), reason="requires torch>=2.5"
)
@pytest.mark.skipif(
    (TORCH_VERSION <= version.parse("2.7.0")) and _IS_OSX,
    reason="requires torch>=2.7 ons OSX",
)
@pytest.mark.parametrize("compiled", [False, True])
class TestCudaGraphs:
    @pytest.fixture(scope="class", autouse=True)
    def _set_cuda_device(self):
        device = torch.get_default_device()
        do_unset = False
        for tdtype in PYTREE_REGISTERED_TDS + PYTREE_REGISTERED_LAZY_TDS:
            if tdtype in SUPPORTED_NODES:
                do_unset = True
                excluder = _exclude_td_from_pytree()
                excluder.set()
                break
        if torch.cuda.is_available():
            torch.set_default_device("cuda:0")
        yield
        if do_unset:
            excluder.unset()
        torch.set_default_device(device)

    def test_cudagraphs_random(self, compiled):
        def func(x):
            return x + torch.randn_like(x)

        if compiled:
            func = torch.compile(func)

        with (
            pytest.warns(UserWarning)
            if not torch.cuda.is_available()
            else contextlib.nullcontext()
        ):
            func = CudaGraphModule(func)

        x = torch.randn(10)
        for _ in range(10):
            func(x)
        assert isinstance(func(torch.zeros(10)), torch.Tensor)
        assert (func(torch.zeros(10)) != 0).any()
        y0 = func(x)
        y1 = func(x + 1)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(y0, y1 + 1)

    @staticmethod
    def _make_cudagraph(
        func: Callable, compiled: bool, *args, **kwargs
    ) -> CudaGraphModule:
        if compiled:
            func = torch.compile(func)
        with (
            pytest.warns(UserWarning)
            if not torch.cuda.is_available()
            else contextlib.nullcontext()
        ):
            func = CudaGraphModule(func, *args, **kwargs)
        return func

    @staticmethod
    def check_types(func, *args, **kwargs):
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for param_name, param in signature.parameters.items():
            arg_value = bound_args.arguments[param_name]
            if param.annotation != param.empty:
                if not isinstance(arg_value, param.annotation):
                    raise TypeError(
                        f"Argument '{param_name}' should be of type {param.annotation}, but is of type {type(arg_value)}"
                    )

    def test_signature(self, compiled):
        if compiled:
            pytest.skip()

        def func(x: torch.Tensor):
            return x + torch.randn_like(x)

        with pytest.raises(TypeError):
            self.check_types(func, "a string")
        self.check_types(func, torch.ones(()))

    def test_backprop(self, compiled):
        x = torch.nn.Parameter(torch.ones(3))
        y = torch.nn.Parameter(torch.ones(3))
        optimizer = torch.optim.SGD([x, y], lr=1)

        def func():
            optimizer.zero_grad()
            z = x + y
            z = z.sum()
            z.backward()
            optimizer.step()

        func = self._make_cudagraph(func, compiled, warmup=4)

        for i in range(1, 11):
            torch.compiler.cudagraph_mark_step_begin()
            func()

            assert (x == 1 - i).all(), i
            assert (y == 1 - i).all(), i
            # assert (x.grad == 1).all()
            # assert (y.grad == 1).all()

    def test_tdmodule(self, compiled):
        tdmodule = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
        tdmodule = self._make_cudagraph(tdmodule, compiled)
        assert tdmodule._is_tensordict_module
        for i in range(10):
            td = TensorDict(x=torch.randn(()))
            tdmodule(td)
            assert td["y"] == td["x"] + 1, i

        tdmodule = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
        tdmodule = self._make_cudagraph(tdmodule, compiled)
        assert tdmodule._is_tensordict_module
        for _ in range(10):
            x = torch.randn(())
            y = tdmodule(x=x)
            assert y == x + 1

        tdmodule = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
        tdmodule = self._make_cudagraph(tdmodule, compiled)
        assert tdmodule._is_tensordict_module
        for _ in range(10):
            td = TensorDict(x=torch.randn(()))
            tdout = TensorDict()
            tdmodule(td, tensordict_out=tdout)
            assert tdout is not td
            assert "x" not in tdout
            assert tdout["y"] == td["x"] + 1

        tdmodule = lambda td: td.set("y", td.get("x") + 1)
        tdmodule = self._make_cudagraph(tdmodule, compiled, in_keys=[], out_keys=[])
        assert tdmodule._is_tensordict_module
        for i in range(10):
            td = TensorDict(x=torch.randn(()))
            tdmodule(td)
            assert tdmodule._out_matches_in
            if i >= tdmodule._warmup and torch.cuda.is_available():
                assert tdmodule._selected_keys == ["y"]
            assert td["y"] == td["x"] + 1

        tdmodule = lambda td: td.set("y", td.get("x") + 1)
        tdmodule = self._make_cudagraph(
            tdmodule, compiled, in_keys=["x"], out_keys=["y"]
        )
        assert tdmodule._is_tensordict_module
        for _ in range(10):
            td = TensorDict(x=torch.randn(()))
            tdmodule(td)
            assert td["y"] == td["x"] + 1

        tdmodule = lambda td: td.copy().set("y", td.get("x") + 1)
        tdmodule = self._make_cudagraph(tdmodule, compiled, in_keys=[], out_keys=[])
        assert tdmodule._is_tensordict_module
        for _ in range(10):
            td = TensorDict(x=torch.randn(()))
            tdout = tdmodule(td)
            assert tdout is not td
            assert "y" not in td
            assert tdout["y"] == td["x"] + 1

    def test_td_input_non_tdmodule(self, compiled):
        func = lambda x: x + 1
        func = self._make_cudagraph(func, compiled)
        for i in range(10):
            td = TensorDict(a=1)
            func(td)
            if i == 5:
                assert not func._is_tensordict_module

    def test_td_input_non_tdmodule_nontensor(self, compiled):
        func = lambda x, y: x + y
        func = self._make_cudagraph(func, compiled)
        for i in range(10):
            assert func(torch.zeros(()), 1.0) == 1.0
            if i == 5:
                assert not func._is_tensordict_module
        if torch.cuda.is_available():
            with pytest.raises(
                ValueError, match="Varying inputs must be torch.Tensor subclasses."
            ):
                func(torch.zeros(()), 2.0)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
