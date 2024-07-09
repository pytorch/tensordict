# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest

import torch

from tensordict import assert_close, TensorDict


class TestTD:
    def test_tensor_output(self):
        def add_one(td):
            return td["a", "b"] + 1

        add_one_c = torch.compile(add_one, fullgraph=True)
        data = TensorDict({"a": {"b": 0}})
        assert add_one(data) == 1
        assert add_one_c(data) == 1
        assert add_one_c(data + 1) == 2

    def test_td_output(self):
        def add_one(td):
            td["a", "c"] = td["a", "b"] + 1
            return td

        add_one_c = torch.compile(add_one, fullgraph=True)
        data = TensorDict({"a": {"b": 0}})
        assert add_one(data.clone())["a", "c"] == 1
        assert add_one_c(data.clone())["a", "c"] == 1
        assert add_one_c(data) is data

    @pytest.mark.parametrize("index_type", ["slice", "tensor", "int"])
    def test_td_index(self, index_type):
        if index_type == "slice":

            def add_one(td):
                return td[:2] + 1

        elif index_type == "tensor":

            def add_one(td):
                return td[torch.tensor([0, 1])] + 1

        elif index_type == "int":

            def add_one(td):
                return td[0] + 1

        add_one_c = torch.compile(add_one, fullgraph=True)
        data = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        if index_type == "int":
            assert (add_one(data)["a", "b"] == 1).all()
            assert (add_one_c(data)["a", "b"] == 1).all()
            assert add_one_c(data).shape == torch.Size([])
        else:
            assert (add_one(data)["a", "b"] == torch.arange(1, 3)).all()
            assert (add_one_c(data)["a", "b"] == torch.arange(1, 3)).all()
            assert add_one_c(data).shape == torch.Size([2])

    def test_stack(self):
        def stack_tds(td0, td1):
            return TensorDict.stack([td0, td1])
            # return torch.stack([td0, td1])

        stack_tds_c = torch.compile(stack_tds, fullgraph=True)
        data0 = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        data1 = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        assert (stack_tds(data0, data1) == stack_tds_c(data0, data1)).all()

    def test_cat(self):
        def cat_tds(td0, td1):
            return TensorDict.cat([td0, td1])

        cat_tds_c = torch.compile(cat_tds, fullgraph=True)
        data0 = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        data1 = TensorDict({"a": {"b": torch.arange(3)}}, [3])
        assert (cat_tds(data0, data1) == cat_tds_c(data0, data1)).all()

    def test_reshape(self):
        def reshape(td):
            return td.reshape(2, 2)

        reshape_c = torch.compile(reshape, fullgraph=True)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        assert (reshape(data) == reshape_c(data)).all()

    def test_unbind(self):
        def unbind(td):
            return td.unbind(0)

        unbind_c = torch.compile(unbind, fullgraph=True)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        assert (unbind(data)[-1] == unbind_c(data)[-1]).all()

    def test_items(self):
        def items(td):
            keys, vals = zip(*td.items(True, True))
            return keys, vals

        items_c = torch.compile(items, fullgraph=True)
        data = TensorDict({"a": {"b": torch.arange(4)}}, [4])
        keys, vals = items(data)
        keys_c, vals_c = items_c(data)

        def assert_eq(x, y):
            assert (x == y).all()

        assert keys == keys_c
        torch.utils._pytree.tree_map(assert_eq, vals, vals_c)

    @pytest.mark.parametrize("recurse", [True, False])
    def test_clone(self, recurse):
        def clone(td: TensorDict):
            return td.clone(recurse=recurse)

        clone_c = torch.compile(clone, fullgraph=True)
        data = TensorDict({"a": {"b": 0, "c": 1}})
        assert_close(clone_c(data), clone(data))
        assert clone_c(data) is not data
        if recurse:
            assert clone_c(data)["a", "b"] is not data["a", "b"]
        else:
            assert clone_c(data)["a", "b"] is data["a", "b"]


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
