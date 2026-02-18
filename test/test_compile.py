# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import contextlib
import importlib.util
import inspect
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

import torch

from _utils_internal import is_npu_available
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

from tensordict.tensorclass import TensorClass

from torch.utils._pytree import SUPPORTED_NODES, tree_map

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)

_has_onnx = importlib.util.find_spec("onnxruntime", None) is not None

_v2_5 = TORCH_VERSION >= version.parse("2.5.0")
_v2_6 = TORCH_VERSION >= version.parse("2.6.0")
_v2_7 = TORCH_VERSION >= version.parse("2.7.0")

_IS_OSX = platform.system() == "Darwin"

npu_device_count = 0
if torch.cuda.is_available():
    cur_device = "cuda"
elif is_npu_available():
    cur_device = "npu"
    npu_device_count = torch.npu.device_count()

pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="torch.compile is not supported on python 3.14+ ",
)


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

    def test_stack_refine_names_nested(self, mode):
        def stack_and_refine(td0, td1):
            out = torch.stack([td0, td1], 0)
            out.refine_names("time")
            return out

        stack_and_refine_c = torch.compile(stack_and_refine, fullgraph=True, mode=mode)

        td0 = TensorDict({"params": TensorDict({"g": torch.tensor(1.0)}, [])}, [])
        td1 = TensorDict({"params": TensorDict({"g": torch.tensor(2.0)}, [])}, [])

        out = stack_and_refine_c(td0, td1)
        assert out.names == ["time"]
        assert out["params"].names == ["time"]
        torch.testing.assert_close(out["params", "g"], torch.tensor([1.0, 2.0]))

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

    def test_pop(self, mode):
        def pop_existing(td: TensorDict):
            return td.pop("a")

        def pop_missing_with_default(td: TensorDict):
            return td.pop("missing", None)

        pop_existing_c = torch.compile(pop_existing, fullgraph=True, mode=mode)
        pop_missing_c = torch.compile(
            pop_missing_with_default, fullgraph=True, mode=mode
        )

        # Test pop existing key
        data = TensorDict({"a": torch.tensor(1), "b": torch.tensor(2)})
        result = pop_existing(data.clone())
        assert result == 1

        data = TensorDict({"a": torch.tensor(1), "b": torch.tensor(2)})
        result_c = pop_existing_c(data.clone())
        assert result_c == 1

        # Verify key is removed
        data = TensorDict({"a": torch.tensor(1), "b": torch.tensor(2)})
        _ = pop_existing_c(data)
        assert "a" not in data.keys()
        assert "b" in data.keys()

        # Test pop missing key with default
        data = TensorDict({"a": torch.tensor(1)})
        result = pop_missing_with_default(data.clone())
        assert result is None

        data = TensorDict({"a": torch.tensor(1)})
        result_c = pop_missing_c(data.clone())
        assert result_c is None

    def test_select_strict_false(self, mode):
        def select_keys(td: TensorDict):
            return td.select("a", "missing_key", strict=False)

        select_keys_c = torch.compile(select_keys, fullgraph=True, mode=mode)

        # Test select with strict=False
        data = TensorDict({"a": torch.tensor(1), "b": torch.tensor(2)})
        result = select_keys(data)
        assert "a" in result.keys()
        assert "missing_key" not in result.keys()
        assert "b" not in result.keys()

        result_c = select_keys_c(data)
        assert "a" in result_c.keys()
        assert "missing_key" not in result_c.keys()
        assert "b" not in result_c.keys()

    def test_exclude(self, mode):
        def exclude_keys(td: TensorDict):
            return td.exclude("b")

        exclude_keys_c = torch.compile(exclude_keys, fullgraph=True, mode=mode)

        data = TensorDict({"a": torch.tensor(1), "b": torch.tensor(2)})
        result = exclude_keys(data)
        assert "a" in result.keys()
        assert "b" not in result.keys()

        result_c = exclude_keys_c(data)
        assert "a" in result_c.keys()
        assert "b" not in result_c.keys()

    def test_all_any(self, mode):
        def call_all(td: TensorDict):
            return td.all(dim=0)

        def call_any(td: TensorDict):
            return td.any(dim=0)

        call_all_c = torch.compile(call_all, fullgraph=True, mode=mode)
        call_any_c = torch.compile(call_any, fullgraph=True, mode=mode)

        data = TensorDict(
            {"a": torch.tensor([[True, False], [True, True]])},
            batch_size=[2, 2],
        )

        result_all = call_all(data)
        result_all_c = call_all_c(data)
        assert (result_all["a"] == result_all_c["a"]).all()
        assert result_all_c.shape == torch.Size([2])

        result_any = call_any(data)
        result_any_c = call_any_c(data)
        assert (result_any["a"] == result_any_c["a"]).all()
        assert result_any_c.shape == torch.Size([2])

    def test_squeeze_unsqueeze(self, mode):
        def call_squeeze(td: TensorDict):
            return td.squeeze(0)

        def call_unsqueeze(td: TensorDict):
            return td.unsqueeze(0)

        call_squeeze_c = torch.compile(call_squeeze, fullgraph=True, mode=mode)
        call_unsqueeze_c = torch.compile(call_unsqueeze, fullgraph=True, mode=mode)

        data = TensorDict({"a": torch.randn(1, 3)}, batch_size=[1, 3])

        result_squeeze = call_squeeze(data)
        result_squeeze_c = call_squeeze_c(data)
        assert result_squeeze.shape == result_squeeze_c.shape
        assert result_squeeze_c.shape == torch.Size([3])

        result_unsqueeze = call_unsqueeze(result_squeeze)
        result_unsqueeze_c = call_unsqueeze_c(result_squeeze_c)
        assert result_unsqueeze.shape == result_unsqueeze_c.shape
        assert result_unsqueeze_c.shape == torch.Size([1, 3])

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
        device = f"{cur_device}:0"

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

    @pytest.mark.skipif(
        is_npu_available(),
        reason="torch.device in torch.compile is not supported on NPU currently.",
    )
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
        device = f"{cur_device}:0"

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

    @pytest.mark.skipif(
        is_npu_available(),
        reason="torch.device in torch.compile is not supported on NPU currently.",
    )
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

    def test_tc_shadow_clone(self, mode):
        """Shadow-mode TensorClass clone should not cause graph breaks (gh-1547)."""

        class ShadowState(TensorClass["shadow"]):
            x: torch.Tensor
            v: torch.Tensor

        def step(s):
            clone = s.clone(recurse=False)
            clone.x = s.x + s.v
            return clone

        s = ShadowState(x=torch.randn(3), v=torch.randn(3), batch_size=[])
        step_c = torch.compile(step, fullgraph=True, mode=mode)
        eager_result = step(s)
        compiled_result = step_c(s)
        assert_close(eager_result, compiled_result)
        assert compiled_result is not s
        # v should be preserved (shallow clone)
        assert compiled_result.v is s.v

    def test_tc_shadow_replace(self, mode):
        """Shadow-mode TensorClass replace should not cause graph breaks (gh-1547)."""

        class ShadowState(TensorClass["shadow"]):
            x: torch.Tensor
            v: torch.Tensor

        def step(s):
            return s.replace(x=s.x + s.v)

        s = ShadowState(x=torch.randn(3), v=torch.randn(3), batch_size=[])
        step_c = torch.compile(step, fullgraph=True, mode=mode)
        eager_result = step(s)
        compiled_result = step_c(s)
        assert_close(eager_result, compiled_result)

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.6.0"),
        reason="while_loop requires torch>=2.6",
    )
    @pytest.mark.xfail(
        reason="Dynamo cannot symbolically trace TensorClass._tensordict "
        "access inside while_loop's pytree flatten (gh-1547). "
        "Requires Dynamo-side support for custom pytree nodes in "
        "higher-order ops.",
    )
    def test_tc_while_loop(self, mode):
        """TensorClass as carry in while_loop should not crash (gh-1547)."""
        from torch._higher_order_ops.while_loop import while_loop

        @tensorclass
        class CarryState:
            val: torch.Tensor
            count: torch.Tensor

        def cond(state):
            return state.count < 5

        def body(state):
            return (
                CarryState(
                    val=state.val + 1,
                    count=state.count + 1,
                    batch_size=[],
                ),
            )

        init = CarryState(val=torch.tensor(0.0), count=torch.tensor(0), batch_size=[])

        def fn():
            return while_loop(cond, body, (init,))

        fn_c = torch.compile(fn, mode=mode)
        (result,) = fn_c()
        assert result.val.item() == 5.0
        assert result.count.item() == 5

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
@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="torch.export has compatibility issues with Python 3.14 (networkx/dataclasses)",
)
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
    @pytest.mark.skipif(not _v2_7, reason="Requires PT>=2.7")
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
        onnx_program = torch.onnx.export(tdm, kwargs=torch_input, dynamo=True)

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

        onnxruntime_input = {k: to_numpy(v) for k, v in torch_input.items()}

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
        torch.onnx.export(tdm, kwargs=torch_input, dynamo=True)
        onnx_program = torch.onnx.export(tdm, kwargs=torch_input, dynamo=True)

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

        onnxruntime_input = {k: to_numpy(v) for k, v in torch_input.items()}

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

    def test_state_dict(self, compiled):
        # Create a linear layer and wrap it in CudaGraphModule
        linear = torch.nn.Linear(3, 4)
        linear = self._make_cudagraph(linear, compiled)

        # Run some warmup iterations
        x = torch.randn(10, 3)
        for _ in range(10):
            linear(x)

        # Get state dict
        state_dict = linear.state_dict()
        if compiled:
            state_dict_get = TensorDict(state_dict)
            state_dict_get = state_dict_get.unflatten_keys(".")["_orig_mod"]
        else:
            state_dict_get = state_dict

        assert "weight" in state_dict_get
        assert "bias" in state_dict_get
        assert state_dict_get["weight"].shape == (4, 3)
        assert state_dict_get["bias"].shape == (4,)

        # Create a new instance and load state
        linear2 = torch.nn.Linear(3, 4)
        linear2 = self._make_cudagraph(linear2, compiled)
        linear2.load_state_dict(state_dict)

        # Test that both modules produce the same output
        y1 = linear(x)
        y2 = linear2(x)
        torch.testing.assert_close(y1, y2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
class TestCompileNontensor:
    # Same issue with the decorator @tensorclass version
    @pytest.fixture(scope="class")
    def data(self):
        return torch.zeros((4, 3), device=cur_device)

    class TensorClassWithNonTensorData(TensorClass["nocast"]):
        tensor: torch.Tensor
        non_tensor_data: int

    def fn_no_device_no_batch_size(self, data):
        a = self.TensorClassWithNonTensorData(tensor=data, non_tensor_data=1)
        return a.tensor

    def fn_no_device(self, data):
        a = self.TensorClassWithNonTensorData(
            tensor=data, non_tensor_data=1, batch_size=[4]
        )
        return a.tensor

    def fn_with_device(self, data):
        a = self.TensorClassWithNonTensorData(
            tensor=data, non_tensor_data=1, batch_size=[4], device=cur_device
        )
        return a.tensor

    def fn_with_device_without_batch_size(self, data):
        a = self.TensorClassWithNonTensorData(
            tensor=data, non_tensor_data=1, device=cur_device
        )
        return a.tensor

    def test_nontensor_no_device_no_batch_size(self, data):
        torch._dynamo.reset_code_caches()
        torch.compile(self.fn_no_device_no_batch_size)(data)

    def test_nontensor_no_device(self, data):
        torch._dynamo.reset_code_caches()
        torch.compile(self.fn_no_device)(data)

    def test_nontensor_with_device(self, data):
        torch._dynamo.reset_code_caches()
        torch.compile(self.fn_with_device)(data)

    def test_nontensor_with_device_without_batch_size(self, data):
        torch._dynamo.reset_code_caches()
        torch.compile(self.fn_with_device_without_batch_size)(data)


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
class TestSchemaFreeze:
    """Tests for freeze_schema_() functionality for torch.compile optimization."""

    def test_freeze_schema_basic(self):
        """Test basic schema freezing and unfreezing."""
        td = TensorDict(
            {"a": torch.randn(3, 4), "b": torch.zeros(3, 2)},
            batch_size=[3],
        )
        assert not td.has_frozen_schema
        assert td._frozen_schema is None
        assert td._frozen_schema_hash is None

        # Freeze schema
        td.freeze_schema_()
        assert td.has_frozen_schema
        assert td._frozen_schema is not None
        assert td._frozen_schema_hash is not None

        # Check schema content
        schema = td._frozen_schema
        assert schema.keys == ("a", "b")
        assert schema.batch_dims == 1
        assert len(schema.tensor_specs) == 2

        # Unfreeze
        td.unfreeze_schema_()
        assert not td.has_frozen_schema
        assert td._frozen_schema is None

    def test_freeze_schema_blocks_new_keys(self):
        """Test that frozen schema blocks adding new keys."""
        td = TensorDict({"a": torch.randn(3, 4)}, batch_size=[3])
        td.freeze_schema_()

        with pytest.raises(RuntimeError, match="Cannot add key"):
            td.set("b", torch.randn(3, 2))

        with pytest.raises(RuntimeError, match="Cannot add key"):
            td["c"] = torch.randn(3, 2)

    def test_freeze_schema_allows_existing_keys(self):
        """Test that frozen schema allows updating existing keys."""
        td = TensorDict({"a": torch.randn(3, 4)}, batch_size=[3])
        td.freeze_schema_()

        # Should work - updating existing key
        td.set("a", torch.randn(3, 4))
        td["a"] = torch.randn(3, 4)
        td.set_("a", torch.randn(3, 4))

    def test_freeze_schema_blocks_delete(self):
        """Test that frozen schema blocks deleting keys."""
        td = TensorDict({"a": torch.randn(3, 4), "b": torch.randn(3, 2)}, batch_size=[3])
        td.freeze_schema_()

        with pytest.raises(RuntimeError, match="Cannot delete key"):
            td.del_("a")

        with pytest.raises(RuntimeError, match="Cannot delete key"):
            del td["b"]

    def test_freeze_schema_blocks_rename(self):
        """Test that frozen schema blocks renaming keys."""
        td = TensorDict({"a": torch.randn(3, 4)}, batch_size=[3])
        td.freeze_schema_()

        with pytest.raises(RuntimeError, match="Cannot rename key"):
            td.rename_key_("a", "b")

    def test_freeze_schema_nested(self):
        """Test that freeze_schema_ propagates to nested TensorDicts."""
        td = TensorDict(
            {
                "a": torch.randn(3, 4),
                "nested": TensorDict({"b": torch.randn(3, 2)}, batch_size=[3]),
            },
            batch_size=[3],
        )

        td.freeze_schema_()

        # Both should be frozen
        assert td.has_frozen_schema
        assert td["nested"].has_frozen_schema

        # Both should block new keys
        with pytest.raises(RuntimeError, match="Cannot add key"):
            td.set("c", torch.randn(3, 1))

        with pytest.raises(RuntimeError, match="Cannot add key"):
            td["nested"].set("d", torch.randn(3, 1))

        # Unfreeze propagates too
        td.unfreeze_schema_()
        assert not td.has_frozen_schema
        assert not td["nested"].has_frozen_schema

    def test_freeze_schema_with_batch_dims(self):
        """Test schema with explicit batch_dims."""
        td = TensorDict(
            {"obs": torch.randn(32, 10, 84, 84)},  # batch, time, H, W
            batch_size=[32, 10],
        )

        # Freeze with batch_dims=2: first 2 dims are dynamic
        td.freeze_schema_(batch_dims=2)

        schema = td._frozen_schema
        assert schema.batch_dims == 2

        # Check that shape_suffix captures non-batch dims
        obs_spec = dict(schema.tensor_specs)["obs"]
        assert obs_spec.shape_suffix == (84, 84)

    def test_freeze_schema_hash_stability(self):
        """Test that schema hash is deterministic for same structure."""
        td1 = TensorDict(
            {"a": torch.randn(3, 4), "b": torch.zeros(3, 2)},
            batch_size=[3],
        )
        td2 = TensorDict(
            {"a": torch.randn(3, 4), "b": torch.zeros(3, 2)},
            batch_size=[3],
        )

        td1.freeze_schema_()
        td2.freeze_schema_()

        # Same structure should have same hash
        assert td1._frozen_schema_hash == td2._frozen_schema_hash
        assert td1._frozen_schema == td2._frozen_schema

    def test_freeze_schema_hash_different_for_different_structure(self):
        """Test that schema hash differs for different structures."""
        td1 = TensorDict({"a": torch.randn(3, 4)}, batch_size=[3])
        td2 = TensorDict({"b": torch.randn(3, 4)}, batch_size=[3])
        td3 = TensorDict({"a": torch.randn(3, 4, 5)}, batch_size=[3])

        td1.freeze_schema_()
        td2.freeze_schema_()
        td3.freeze_schema_()

        # Different keys -> different hash
        assert td1._frozen_schema_hash != td2._frozen_schema_hash

        # Different shape suffix -> different hash
        assert td1._frozen_schema_hash != td3._frozen_schema_hash

    @pytest.mark.parametrize("mode", [None, "reduce-overhead"])
    def test_freeze_schema_compile_basic(self, mode):
        """Test that frozen schema TensorDict can be compiled."""

        def process(td):
            td["a"] = td["a"] + 1
            return td

        td = TensorDict({"a": torch.randn(3, 4)}, batch_size=[3])
        td.freeze_schema_()

        process_c = torch.compile(process, fullgraph=True, mode=mode)
        result = process_c(td.clone())

        torch.testing.assert_close(result["a"], td["a"] + 1)

    @pytest.mark.parametrize("mode", [None, "reduce-overhead"])
    def test_freeze_schema_compile_nested(self, mode):
        """Test that compiled function works with nested frozen TensorDicts."""

        def process(td):
            td["a"] = td["a"] + 1
            td["nested", "b"] = td["nested", "b"] * 2
            return td

        td = TensorDict(
            {
                "a": torch.randn(3, 4),
                "nested": TensorDict({"b": torch.randn(3, 2)}, batch_size=[3]),
            },
            batch_size=[3],
        )
        td.freeze_schema_()

        process_c = torch.compile(process, fullgraph=True, mode=mode)
        td_clone = td.clone()
        result = process_c(td_clone)

        torch.testing.assert_close(result["a"], td["a"] + 1)
        torch.testing.assert_close(result["nested", "b"], td["nested", "b"] * 2)

    def test_pytree_context_includes_schema_hash(self):
        """Test that pytree flatten includes schema_hash when frozen."""
        from tensordict._pytree import _tensordict_flatten

        td = TensorDict({"a": torch.randn(3, 4)}, batch_size=[3])

        # Without freeze, no schema_hash in context
        _, context = _tensordict_flatten(td)
        assert "schema_hash" not in context

        # With freeze, schema_hash is in context
        td.freeze_schema_()
        _, context = _tensordict_flatten(td)
        assert "schema_hash" in context
        assert context["schema_hash"] == td._frozen_schema_hash


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
