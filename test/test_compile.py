# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import contextlib

import pytest

import torch

from tensordict import assert_close, TensorDict

TORCH_VERSION = torch.__version__


@pytest.mark.skipif(TORCH_VERSION < "2.4", reason="requires torch>2.4")
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
            return td.view(2, 2)

        view_c = torch.compile(view, fullgraph=True, mode=mode)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        data_view = view(data)
        _ = view_c(data)
        data_view_c = view_c(data)
        assert (data_view == data_view_c).all()

    def test_transpose(self, mode):
        def transpose(td):
            return td.transpose(0, 1)

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
            return td.flatten_keys()

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
            return td.unflatten_keys()

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
        with pytest.raises(torch._dynamo.exc.Unsupported):
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
            return td3

        td = TensorDict(
            {"a": torch.randn(1, 2, 3), "b": torch.zeros(1, 2, 3, dtype=torch.bool)},
            batch_size=[1, 2],
            device="cpu",
            lock=True,
        )
        locked_op_c = torch.compile(locked_op, fullgraph=True, mode=mode)
        td_op = locked_op(td)
        # no warning the second time this is run
        with pytest.warns(
            UserWarning, match="Using lock_"
        ) if mode is None else contextlib.nullcontext():
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
