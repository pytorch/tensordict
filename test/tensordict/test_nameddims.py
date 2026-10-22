# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
import warnings

import pytest
import tensordict
import torch
from tensordict import lazy_stack, LazyStackedTensorDict, TensorAttrs, TensorDict

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.tensordict.test._utils_internal import (
        get_available_devices,
        is_npu_available,
        TestTensorDictsBase,
    )
else:
    IS_FB = False
    from _utils_internal import (
        get_available_devices,
        is_npu_available,
        TestTensorDictsBase,
    )


_has_streaming = importlib.util.find_spec("streaming", None) is not None

try:
    import h5py  # noqa

    _has_h5py = True
except ImportError:
    _has_h5py = False

_has_onnx = importlib.util.find_spec("onnxruntime", None) is not None

PYTORCH_TEST_FBCODE = os.getenv("PYTORCH_TEST_FBCODE")

_IS_OSX = platform.system() == "Darwin"
_IS_WINDOWS = sys.platform == "win32"

TD_BATCH_SIZE = 4

# Capture all warnings
pytestmark = [
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings(
        "ignore:There is a performance drop because we have not yet implemented the batching rule"
    ),
    pytest.mark.filterwarnings(
        "ignore:A destination should be provided when cloning a PersistentTensorDict"
    ),
    pytest.mark.filterwarnings(
        "ignore:Replacing an array with another one is inefficient"
    ),
    pytest.mark.filterwarnings(
        "ignore:Indexing an h5py.Dataset object with a boolean mask that needs broadcasting does not work directly"
    ),
    pytest.mark.filterwarnings(
        "ignore:The PyTorch API of nested tensors is in prototype"
    ),
    pytest.mark.filterwarnings(
        "ignore:Lazy modules are a new feature under heavy development so changes to the API or functionality"
    ),
    pytest.mark.filterwarnings(
        "ignore:The content of the stacked NonTensorData objects matched in value but not identity"
    ),
    pytest.mark.filterwarnings(
        "ignore:No PYTORCH_KERNEL_CACHE_PATH or HOME environment variable set"
    ),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]

mp_ctx = "spawn"
cur_device = "cpu"
npu_device_count = 0
if torch.cuda.is_available():
    cur_device = "cuda"
elif is_npu_available():
    cur_device = "npu"
    npu_device_count = torch.npu.device_count()


class TestNamedDims(TestTensorDictsBase):
    def test_all(self):
        td = TensorDict(batch_size=[3, 4, 1, 6], names=["a", "b", "c", "d"])
        tda = td.all(2)
        assert tda.names == ["a", "b", "d"]
        tda = td.any(2)
        assert tda.names == ["a", "b", "d"]

    def test_apply(self):
        td = TensorDict(batch_size=[3, 4, 1, 6], names=["a", "b", "c", "d"])
        tda = td.apply(lambda x: x + 1)
        assert tda.names == ["a", "b", "c", "d"]
        tda = td.apply(lambda x: x.squeeze(2), batch_size=[3, 4, 6])
        # no way to tell what the names have become, in general
        assert tda.names == [None] * 3

    def test_apply_batch_size_lazy_stack_names(self):
        # https://github.com/pytorch/tensordict/issues/2366: a lazy stack kept its
        # names when the batch size changed, even if their count no longer fit
        td = TensorDict(a=torch.zeros(2, 3), batch_size=[2, 3])
        stack = LazyStackedTensorDict(*td.unbind(0), stack_dim=0)
        stack.names = ["x", "y"]
        result = stack.apply(lambda x: x[0], batch_size=[3])
        assert result.batch_size == torch.Size([3])
        assert result.names == [None]
        result = stack.apply(lambda x: x[0], batch_size=[3], names=None)
        assert result.names == [None]
        result = stack.tile((2, 1, 1))
        assert result.batch_size == torch.Size([2, 2, 3])
        assert result.names == [None] * 3
        # the number of dims is unchanged: the names are kept
        result = stack.tile((2, 1))
        assert result.batch_size == torch.Size([4, 3])
        assert result.names == ["x", "y"]
        # names given are used as given
        result = stack.apply(lambda x: x[0], batch_size=[3], names=["z"])
        assert result.names == ["z"]
        result = stack.apply(lambda x: x[0, 0], batch_size=[], names=[])
        assert result.batch_size == torch.Size([])
        assert result.names == []

    def test_cat(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=None)
        tdc = torch.cat([td, td], -1)
        assert tdc.names == [None] * 4
        td = TensorDict(batch_size=[3, 4, 5, 6], names=["a", "b", "c", "d"])
        tdc = torch.cat([td, td], -1)
        assert tdc.names == ["a", "b", "c", "d"]

    def test_change_batch_size(self):
        td = TensorDict(batch_size=[3, 4, 1, 6], names=["a", "b", "c", "z"])
        td.batch_size = [3, 4, 1, 6, 1]
        assert td.names == ["a", "b", "c", "z", None]
        td.batch_size = []
        assert td.names == []
        td.batch_size = [3, 4]
        assert td.names == [None, None]
        td.names = ["a", None]
        td.batch_size = [3]
        assert td.names == ["a"]

    def test_clone(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=None)
        td.names = ["a", "b", "c", "d"]
        tdc = td.clone()
        assert tdc.names == ["a", "b", "c", "d"]
        tdc = td.clone(False)
        assert tdc.names == ["a", "b", "c", "d"]

    def test_detach(self):
        td = TensorDict(batch_size=[3, 4, 1, 6], names=["a", "b", "c", "d"])
        td[""] = torch.zeros(td.shape, requires_grad=True)
        tdd = td.detach()
        assert tdd.names == ["a", "b", "c", "d"]

    def test_error_similar(self):
        with pytest.raises(ValueError):
            td = TensorDict(batch_size=[3, 4, 1, 6], names=["a", "b", "c", "a"])
        with pytest.raises(ValueError):
            td = TensorDict(
                {},
                batch_size=[3, 4, 1, 6],
            )
            td.names = ["a", "b", "c", "a"]
        with pytest.raises(ValueError):
            td = TensorDict(
                {},
                batch_size=[3, 4, 1, 6],
            )
            td.refine_names("a", "a", ...)
        with pytest.raises(ValueError):
            td = TensorDict(batch_size=[3, 4, 1, 6], names=["a", "b", "c", "z"])
            td.rename_(a="z")

    def test_expand(self):
        td = TensorDict(batch_size=[3, 4, 1, 6], names=["a", "b", "c", "d"])
        tde = td.expand(2, 3, 4, 5, 6)
        assert tde.names == [None, "a", "b", "c", "d"]

    def test_flatten(self):
        td = TensorDict(batch_size=[3, 4, 1, 6], names=["a", "b", "c", "d"])
        tdf = td.flatten(1, 3)
        assert tdf.names == ["a", None]
        tdu = tdf.unflatten(1, (4, 1, 6))
        assert tdu.names == ["a", None, None, None]
        tdf = td.flatten(1, 2)
        assert tdf.names == ["a", None, "d"]
        tdu = tdf.unflatten(1, (4, 1))
        assert tdu.names == ["a", None, None, "d"]
        tdf = td.flatten(0, 2)
        assert tdf.names == [None, "d"]
        tdu = tdf.unflatten(0, (3, 4, 1))
        assert tdu.names == [None, None, None, "d"]

    def test_fullname(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=["a", "b", "c", "d"])
        assert td.names == ["a", "b", "c", "d"]

    def test_gather(self):
        td = TensorDict(batch_size=[3, 4, 1, 6], names=["a", "b", "c", "d"])
        idx = torch.randint(6, (3, 4, 1, 18))
        tdg = td.gather(dim=-1, index=idx)
        assert tdg.names == ["a", "b", "c", "d"]

    @pytest.mark.skipif(not _has_h5py, reason="h5py not installed")
    def test_h5(self, tmpdir):
        td = TensorDict(
            {"a": torch.zeros(3, 4, 1, 6)},
            batch_size=[3, 4, 1, 6],
            names=["a", "b", "c", "d"],
        )
        tdm = td.to_h5(filename=tmpdir / "file.h5")
        assert tdm.names == ["a", "b", "c", "d"]

    @pytest.mark.skipif(not _has_h5py, reason="h5py not installed")
    def test_h5_td(self):
        td = self.td_h5("cpu")
        td.names = list("abcd")
        assert td.rename(c="g").names == list("abgd")
        assert td.names == list("abcd")
        td.rename_(c="g")
        assert td.names == list("abgd")

    @pytest.mark.skipif(not _has_h5py, reason="h5py not installed")
    def test_h5_names_setter_keeps_nested_trailing_name(self, tmpdir):
        td = TensorDict(
            {"agents": TensorDict({"obs": torch.zeros(2, 3, 4)}, [2, 3])}, [2]
        )
        tdh5 = td.to_h5(filename=tmpdir / "file.h5")
        tdh5["agents"].names = [None, "agent"]
        tdh5.names = ["batch"]
        assert tdh5["agents"].names == ["batch", "agent"]
        tdh5.names = None
        assert tdh5["agents"].names == [None, "agent"]

    def test_index(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=["a", "b", "c", "d"])
        assert td[0].names == ["b", "c", "d"]
        assert td[:, 0].names == ["a", "c", "d"]
        assert td[0, :].names == ["b", "c", "d"]
        assert td[0, :1].names == ["b", "c", "d"]
        assert td[..., -1].names == ["a", "b", "c"]
        assert td[0, ..., -1].names == ["b", "c"]
        assert td[0, ..., [-1]].names == ["b", "c", "d"]
        assert td[0, ..., torch.tensor([-1])].names == ["b", "c", "d"]
        assert td[0, ..., torch.tensor(-1)].names == ["b", "c"]
        assert td[0, ..., :-1].names == ["b", "c", "d"]
        assert td[:1, ..., :-1].names == ["a", "b", "c", "d"]
        # a 1-D mask keeps the name of its dim, an N-D mask gives one unnamed dim
        tdbool = td[torch.ones(3, dtype=torch.bool)]
        assert tdbool.names == ["a", "b", "c", "d"]
        assert tdbool.ndim == 4
        tdbool = td[torch.ones(3, 4, dtype=torch.bool)]
        assert tdbool.names == [None, "c", "d"]
        assert tdbool.ndim == 3

    def test_index_scalar_bool_and_nd_mask(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=["a", "b", "c", "d"])
        # a scalar bool or a 0-d mask adds an unnamed dim
        assert td[True].names == [None, "a", "b", "c", "d"]
        assert td[True, 0, 0, 0].names == [None, "d"]
        assert td[:, False].names == ["a", None, "b", "c", "d"]
        assert td[torch.tensor(True)].names == [None, "a", "b", "c", "d"]
        # an N-D mask merges the dims it consumes into an unnamed dim
        mask = torch.ones(4, 5, dtype=torch.bool)
        assert td[:, mask].names == ["a", None, "d"]
        assert td[..., mask, :].names == ["a", None, "d"]
        assert td[[[True] * 4] * 3].names == [None, "c", "d"]

    def test_index_advanced_names(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=["a", "b", "c", "d"])
        rows, cols = torch.tensor([0, 2]), torch.tensor([1, 3])
        # advanced indices that broadcast together give unnamed dims
        assert td[rows, cols].names == [None, "c", "d"]
        assert td[rows, :, cols].names == [None, "b", "d"]
        assert td[:, rows, None, cols].names == [None, "a", None, "d"]
        # unless only one of them varies along a dim
        assert td[rows[:, None], cols].names == ["a", "b", "c", "d"]
        assert td[rows, :, [1]].names == ["a", "b", "d"]
        assert td[rows, True].names == ["a", "b", "c", "d"]
        assert td[True, :, rows].names == ["b", "a", "c", "d"]
        # an N-D index names only the dim of the block along which it varies,
        # as a name can't be repeated
        assert td[rows[None]].names == [None, "a", "b", "c", "d"]
        assert td[torch.tensor([[0, 1], [1, 2]])].names == [None, None, "b", "c", "d"]
        assert td[rows[None]].to_tensordict().names == [None, "a", "b", "c", "d"]

    @pytest.mark.parametrize("stack_dim", [None, 0, 1, 2])
    @pytest.mark.parametrize(
        "index,names",
        [
            (torch.tensor([True, False, True]), ["a", "b", "c"]),
            ([True, False, True], ["a", "b", "c"]),
            ((slice(None), torch.tensor([True, False, True, False])), ["a", "b", "c"]),
            ((..., torch.tensor([True, False, True, True, False])), ["a", "b", "c"]),
            ((0, torch.tensor([True, False, True, False])), ["b", "c"]),
            (torch.tensor([[True, False, True, False]] * 3), [None, "c"]),
            ((slice(None), torch.ones(4, 5, dtype=torch.bool)), ["a", None]),
        ],
    )
    def test_index_mask_names(self, stack_dim, index, names):
        # a 1-D mask keeps the name of its dim and an N-D mask gives one unnamed
        # dim, for tensordicts and lazy stacks alike
        td = TensorDict({"x": torch.zeros(3, 4, 5)}, [3, 4, 5], names=["a", "b", "c"])
        if stack_dim is not None:
            td = LazyStackedTensorDict.lazy_stack(
                list(td.unbind(stack_dim)), stack_dim
            ).refine_names("a", "b", "c")
        assert td[index].names == names

    @pytest.mark.parametrize("stack_dim", [None, 0, 1, 2])
    @pytest.mark.parametrize(
        "op,names",
        [
            pytest.param(lambda td: td.all(0), ["b", "c"], id="all"),
            pytest.param(lambda td: td.all(-1), ["a", "b"], id="all-neg-dim"),
            pytest.param(lambda td: td.any(1), ["a", "c"], id="any"),
            pytest.param(
                lambda td: td.where(torch.zeros(3, 4, 5, dtype=torch.bool), td.clone()),
                ["a", "b", "c"],
                id="where",
            ),
            pytest.param(
                lambda td: td.expand(2, 3, 4, 5), [None, "a", "b", "c"], id="expand"
            ),
            pytest.param(lambda td: td.memmap(), ["a", "b", "c"], id="memmap"),
            pytest.param(
                lambda td: td.memmap_like(), ["a", "b", "c"], id="memmap_like"
            ),
            pytest.param(lambda td: td.empty(), ["a", "b", "c"], id="empty"),
            pytest.param(
                lambda td: td.empty(recurse=True),
                ["a", "b", "c"],
                id="empty-recurse",
            ),
            pytest.param(
                lambda td: td.empty(batch_size=[3, 4, 5, 2]),
                ["a", "b", "c", None],
                id="empty-batch-size",
            ),
            pytest.param(
                lambda td: td.empty(batch_size=[2, 4, 5]),
                ["a", "b", "c"],
                id="empty-batch-size-leading",
            ),
        ],
    )
    def test_lazy_stack_op_names(self, stack_dim, op, names):
        # the result keeps the names of the dims, for tensordicts and lazy
        # stacks alike
        td = TensorDict(
            {"x": torch.zeros(3, 4, 5, dtype=torch.bool)},
            [3, 4, 5],
            names=["a", "b", "c"],
        )
        if stack_dim is not None:
            td = LazyStackedTensorDict.lazy_stack(
                list(td.unbind(stack_dim)), stack_dim
            ).refine_names("a", "b", "c")
        assert op(td).names == names

    def test_lazy_stack_op_names_members_differ(self):
        # a dim that the members name differently, or a stack without members,
        # gives an unnamed dim
        td = LazyStackedTensorDict(
            TensorDict({}, [3], names=["x"]),
            TensorDict({}, [3], names=["y"]),
            stack_dim=1,
        )
        assert td.all(0).names == [None]
        assert td.any(1).names == [None]
        td = LazyStackedTensorDict(stack_dim=0, batch_size=[3])
        assert td.all(1).names == [None]
        assert td.any(0).names == [None]
        # names that repeat once a dim is reduced: lazy_stack accepts a stack
        # dim name that names a member dim, with a warning until 0.17 (#2690)
        with pytest.warns(FutureWarning, match="stack dim name 'x'"):
            td = LazyStackedTensorDict.lazy_stack(
                [
                    TensorDict({"a": torch.zeros(3, 4, 2)}, [3, 4], names=["x", "y"])
                    for _ in range(2)
                ],
                2,
                stack_dim_name="x",
            )
        assert td.all(1).names == [None, None]
        assert td.any(1).names == [None, None]
        # a nested lazy stack whose members have different names can't be
        # renamed, which must not make expand fail
        td = LazyStackedTensorDict(
            *[
                TensorDict(
                    {
                        "sub": TensorDict(
                            {"c": torch.zeros(3, 2, n)}, [3, 2], names=["x", name]
                        )
                    },
                    [3],
                    names=["x"],
                )
                for n, name in ((4, "p"), (5, "q"))
            ],
            stack_dim=1,
            stack_dim_name="s",
        )
        assert td.expand(5, 3, 2).batch_size == (5, 3, 2)

    def test_masked_fill(self):
        td = TensorDict(batch_size=[3, 4, 1, 6], names=["a", "b", "c", "d"])
        tdm = td.masked_fill(torch.zeros(3, 4, 1, dtype=torch.bool), 1.0)
        assert tdm.names == ["a", "b", "c", "d"]

    def test_memmap_like(self, tmpdir):
        td = TensorDict(
            {"a": torch.zeros(3, 4, 1, 6)},
            batch_size=[3, 4, 1, 6],
            names=["a", "b", "c", "d"],
        )
        tdm = td.memmap_like(prefix=tmpdir)
        assert tdm.names == ["a", "b", "c", "d"]
        assert tdm.is_memmap()

    @pytest.mark.parametrize(
        "save", ["memmap", "memmap_", "save", "archive", "append_memmap"]
    )
    def test_memmap_load_keeps_names(self, save, tmp_path):
        # #3262: the directory loaded back had no dim names.
        def save_and_load(td, name):
            path = tmp_path / name
            if save == "memmap":
                td.memmap(path)
            elif save == "memmap_":
                td.clone().memmap_(path)
            elif save == "save":
                td.save(path)
            elif save == "archive":
                path = tmp_path / f"{name}.tdz"
                td.memmap(path)
            else:
                td[:1].memmap(path)
                tensordict.append_memmap(path, td[1:])
            return tensordict.load_memmap(path)

        # no nested tensordict, whose names would spread to the top level
        td = TensorDict(a=torch.zeros(3, 4), batch_size=[3, 4], names=["time", "env"])
        assert save_and_load(td, "flat").names == ["time", "env"]
        td["nested"] = TensorDict(b=torch.ones(3, 4, 2), batch_size=[3, 4, 2])
        td["nested"].names = ["time", "env", "agent"]
        loaded = save_and_load(td, "nested")
        assert loaded.names == ["time", "env"]
        assert loaded["nested"].names == ["time", "env", "agent"]
        # a key named like the entry of the names keeps its entry
        td = TensorDict(a=torch.zeros(3), _names=torch.ones(3), batch_size=[3])
        loaded = save_and_load(td, "key")
        assert (loaded["_names"] == 1).all()
        assert set(loaded.keys()) == {"a", "_names"}
        # without names, the loaded tensordict has none
        td = TensorDict(a=torch.zeros(3), batch_size=[3])
        assert not save_and_load(td, "unnamed")._has_names()

    def test_memmap_load_keeps_lazy_stack_dim_name(self, tmp_path):
        # #3262
        td = TensorDict(a=torch.zeros(3), batch_size=[3], names=["time"])
        lazy_stack([td, td.clone()], 0, stack_dim_name="env").memmap(tmp_path)
        assert tensordict.load_memmap(tmp_path).names == ["env", "time"]

    def test_memmap_load_keeps_tensorclass_names(self, tmp_path):
        # #3262: a tensorclass saves its tensors through the same code
        from tensordict import tensorclass

        @tensorclass
        class Frames:
            a: torch.Tensor

        tc = Frames(a=torch.zeros(3, 4), batch_size=[3, 4], names=["time", "env"])
        tc.memmap(tmp_path)
        loaded = tensordict.load_memmap(tmp_path)
        assert type(loaded) is Frames
        assert loaded.names == ["time", "env"]

    def test_memmap_load_names_subclass_without_names_arg(self, tmp_path):
        # #3262: loading must not pass names or stack_dim_name to the
        # __init__ of a subclass that does not take them.
        class MyTD(TensorDict):
            def __init__(self, source=None, batch_size=None, device=None):
                super().__init__(source, batch_size=batch_size, device=device)

        class MyStack(LazyStackedTensorDict):
            def __init__(self, *tensordicts, stack_dim=0):
                super().__init__(*tensordicts, stack_dim=stack_dim)

        td = MyTD({"a": torch.zeros(3)}, batch_size=[3])
        td.names = ["time"]
        td.memmap(tmp_path / "td")
        loaded = MyTD.load_memmap(tmp_path / "td")
        assert type(loaded) is MyTD
        assert loaded.names == ["time"]
        member = TensorDict(a=torch.zeros(3), batch_size=[3])
        stack = MyStack(member, member.clone())
        stack.names = ["env", None]
        stack.memmap(tmp_path / "stack")
        loaded = MyStack.load_memmap(tmp_path / "stack")
        assert type(loaded) is MyStack
        assert loaded.names == ["env", None]

    def test_memmap_td(self):
        td = self.memmap_td("cpu")
        td.names = list("abcd")
        assert td.rename(c="g").names == list("abgd")
        assert td.names == list("abcd")
        td.rename_(c="g")
        assert td.names == list("abgd")
        assert td.clone().names == list("abgd")

    def test_nested(self):
        td = TensorDict(batch_size=[3, 4, 1, 6], names=["a", "b", "c", "d"])
        td["a"] = TensorDict(batch_size=[3, 4, 1, 6])
        assert td["a"].names == td.names
        td["a"] = TensorDict()
        assert td["a"].names == td.names
        td = TensorDict(batch_size=[3, 4, 1, 6], names=None)
        td["a"] = TensorDict(batch_size=[3, 4, 1, 6])
        td.names = ["a", "b", None, None]
        assert td["a"].names == td.names
        td.set_("a", TensorDict(batch_size=[3, 4, 1, 6]))
        assert td["a"].names == td.names

    def test_nested_indexing(self):
        td = TensorDict(
            {"": TensorDict({}, [3, 4], names=["c", "d"])}, [3], names=["c"]
        )
        assert td[0][""].names == td[""][0].names == ["d"]

    def test_nested_indexing_extra_dims(self):
        """Regression test: indexing nested TensorDicts with more batch dims than parent.

        When a nested TensorDict has more batch dimensions than its parent and names
        are set on the parent, the nested TensorDict should get properly extended names.
        Indexing should not raise IndexError due to mismatched name list length.
        """
        # Structure: parent [10, 5] -> next [10, 5] -> data [10, 5, 3]
        parent = TensorDict(
            {
                "next": TensorDict(
                    {
                        "data": TensorDict(
                            {"truncated": torch.ones(10, 5, 3, dtype=torch.bool)},
                            batch_size=[10, 5, 3],
                        )
                    },
                    batch_size=[10, 5],
                )
            },
            batch_size=[10, 5],
        )
        parent.names = ["time", "batch"]

        # Verify names are properly extended to nested TensorDicts
        assert parent["next"]._td_dim_names == ["time", "batch"]
        assert parent["next"]["data"]._td_dim_names == ["time", "batch", None]

        # This should not raise IndexError: list index out of range
        result = parent[..., -1]
        assert result.batch_size == torch.Size([10])
        assert result.names == ["time"]
        assert result["next"].batch_size == torch.Size([10])
        assert result["next"].names == ["time"]
        assert result["next", "data"].batch_size == torch.Size([10, 3])
        assert result["next", "data"].names == ["time", None]
        assert result["next", "data", "truncated"].all()

        # Test with integer indexing that removes first dimension
        result2 = parent[0]
        assert result2.batch_size == torch.Size([5])
        assert result2.names == ["batch"]
        assert result2["next"].batch_size == torch.Size([5])
        assert result2["next"].names == ["batch"]
        assert result2["next", "data"].batch_size == torch.Size([5, 3])
        assert result2["next", "data"].names == ["batch", None]

    def test_nested_stacked_td(self):
        td = self.nested_stacked_td("cpu")
        td.names = list("abcd")
        assert td.names == list("abcd")
        assert td[:, 1].names == list("acd")
        assert td["my_nested_td"][:, 1].names == list("acd")
        assert td[:, 1]["my_nested_td"].names == list("acd")
        tdr = td.rename(c="z")
        assert td.names == list("abcd")
        assert tdr.names == list("abzd")
        td.rename_(c="z")
        assert td.names == list("abzd")
        assert td[:, 1].names == list("azd")
        assert td["my_nested_td"][:, 1].names == list("azd")
        assert td[:, 1]["my_nested_td"].names == list("azd")
        assert td.contiguous().names == list("abzd")
        assert td[:, 1].contiguous()["my_nested_td"].names == list("azd")

    def test_nested_tc(self):
        nested_td = self.nested_tensorclass("cpu")
        nested_td.names = list("abcd")
        assert nested_td.rename(c="g").names == list("abgd")
        assert nested_td.names == list("abcd")
        nested_td.rename_(c="g")
        assert nested_td.names == list("abgd")
        assert nested_td.get("my_nested_tc").names == list("abgd")
        assert nested_td.contiguous().names == list("abgd")
        assert nested_td.contiguous().get("my_nested_tc").names == list("abgd")

    def test_nested_td(self):
        nested_td = self.nested_td("cpu")
        nested_td.names = list("abcd")
        assert nested_td.rename(c="g").names == list("abgd")
        assert nested_td.names == list("abcd")
        nested_td.rename_(c="g")
        assert nested_td.names == list("abgd")
        assert nested_td["my_nested_td"].names == list("abgd")
        assert nested_td.contiguous().names == list("abgd")
        assert nested_td.contiguous()["my_nested_td"].names == list("abgd")

    def test_noname(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=None)
        assert td.names == [None] * 4

    def test_partial_name(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=["a", None, None, "d"])
        assert td.names == ["a", None, None, "d"]

    def test_partial_set(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=None)
        td.names = ["a", None, None, "d"]
        assert td.names == ["a", None, None, "d"]
        td.names = ["a", "b", "c", "d"]
        assert td.names == ["a", "b", "c", "d"]
        with pytest.raises(
            ValueError,
            match="the length of the dimension names must equate the tensordict batch_dims",
        ):
            td.names = ["a", "b", "c"]

    def test_permute(self):
        td = TensorDict({"sub": {}}, batch_size=[3, 4, 5, 6], names=None, lock=True)
        td.names = ["a", "b", "c", "d"]
        tdp = td.permute(-1, -2, -3, -4)
        assert tdp.names == list("dcba")
        tdp = td.permute(-1, 1, 2, -4)
        assert tdp.names == list("dbca")
        assert tdp.is_locked
        assert tdp["sub"].is_locked

    def test_permute_leading_dims(self):
        # an order for the leading dims only leaves the other dims, and their
        # names, in place
        td = TensorDict({"sub": {}}, batch_size=[2, 3, 4], names=["x", "y", "z"])
        tdp = td.permute(1, 0)
        assert tdp.batch_size == torch.Size([3, 2, 4])
        assert tdp.names == ["y", "x", "z"]
        assert tdp["sub"].names == ["y", "x", "z"]

    def test_permute_td(self):
        td = self.unsqueezed_td("cpu")
        with pytest.raises(
            RuntimeError, match="Names of a lazy tensordict cannot be modified"
        ):
            td.names = list("abcd")

    def test_refine_names(self):
        td = TensorDict(batch_size=[3, 4, 5, 6])
        tdr = td.refine_names(None, None, None, "d")
        assert tdr.names == [None, None, None, "d"]
        tdr = tdr.refine_names(None, None, "c", "d")
        assert tdr.names == [None, None, "c", "d"]
        with pytest.raises(
            RuntimeError, match="refine_names: cannot coerce TensorDict"
        ):
            tdr.refine_names(None, None, "d", "d")
        tdr = td.refine_names(..., "d")
        assert tdr.names == [None, None, "c", "d"]
        tdr = td.refine_names("a", ..., "d")
        assert tdr.names == ["a", None, "c", "d"]

    def test_rename(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=None)
        td.names = ["a", None, None, "d"]
        td.rename_(a="c")
        assert td.names == ["c", None, None, "d"]
        td.rename_(d="z")
        assert td.names == ["c", None, None, "z"]
        td.rename_(*list("mnop"))
        assert td.names == ["m", "n", "o", "p"]
        td2 = td.rename(p="q")
        assert td.names == ["m", "n", "o", "p"]
        assert td2.names == ["m", "n", "o", "q"]
        td2 = td.rename(*list("wxyz"))
        assert td.names == ["m", "n", "o", "p"]
        assert td2.names == ["w", "x", "y", "z"]

    @pytest.mark.parametrize("method", ["rename", "rename_"])
    @pytest.mark.parametrize("batch_size", [[], [3], [3, 4]])
    def test_rename_none(self, method, batch_size):
        # a single None clears every name, for any number of batch dims
        names = list("ab")[: len(batch_size)] or None
        sub = TensorDict(batch_size=batch_size + [5], names=names and names + ["z"])
        td = TensorDict({"sub": sub}, batch_size, names=names)
        with pytest.raises(ValueError, match="Passed both a name map"):
            getattr(td, method)(None, a="x")
        assert td.names == (names or [])
        out = getattr(td, method)(None)
        assert out is td if method == "rename_" else out is not td
        assert out.names == [None] * len(batch_size)
        assert out["sub"].names == [None] * len(batch_size) + ["z" if names else None]
        assert td.names == (
            [None] * len(batch_size) if method == "rename_" else names or []
        )

    def test_select(self):
        td = TensorDict(batch_size=[3, 4, 1, 6], names=["a", "b", "c", "d"])
        tds = td.select()
        assert tds.names == ["a", "b", "c", "d"]
        tde = td.exclude()
        assert tde.names == ["a", "b", "c", "d"]
        td[""] = torch.zeros(td.shape)
        td["*"] = torch.zeros(td.shape)
        tds = td.select("")
        assert tds.names == ["a", "b", "c", "d"]

    def test_set_at(self):
        td = TensorDict(
            {"": TensorDict({}, [3, 4, 1, 6])},
            batch_size=[3, 4, 1, 6],
            names=["a", "b", "c", "d"],
        )
        td.set_at_("", TensorDict({}, [4, 1, 6]), 0)
        assert td.names == ["a", "b", "c", "d"]
        assert td[""].names == ["a", "b", "c", "d"]

    def test_set_item_populate_names(self):
        td = TensorDict({}, [3])
        td["a"] = TensorDict({}, [3, 4], names=["a", "b"])
        assert td.names == ["a"]
        assert td["a"].names == ["a", "b"]

    def test_init_keeps_nested_trailing_name(self):
        sub = TensorDict({"obs": torch.zeros(2, 3, 4)}, [2, 3], names=[None, "agent"])
        td = TensorDict({"agents": sub}, [2])
        assert td["agents"].names == [None, "agent"]
        assert sub.names == [None, "agent"]

    def test_names_setter_none_keeps_nested_trailing_name(self):
        sub = TensorDict({"obs": torch.zeros(2, 3, 4)}, [2, 3], names=[None, "agent"])
        td = TensorDict({"agents": sub}, [2], names=["batch"])
        assert td["agents"].names == ["batch", "agent"]
        td.names = None
        assert td.names == [None]
        assert td["agents"].names == [None, "agent"]

    def test_names_setter_none_erases_unnamed_nested(self):
        td = TensorDict({"sub": TensorDict({}, [2, 3])}, [2], names=["batch"])
        assert td["sub"].names == ["batch", None]
        td.names = None
        assert not td["sub"]._has_names()

    def test_names_setter_rejects_nested_trailing_name_clash(self):
        # the trailing name now survives, so a parent name equal to it would
        # leave the child with duplicate names
        sub = TensorDict({"obs": torch.zeros(2, 3, 4)}, [2, 3], names=[None, "batch"])
        td = TensorDict({"agents": sub}, [2])
        with pytest.raises(ValueError, match="non-unique"):
            td.names = ["batch"]
        td.names = ["time"]
        assert td["agents"].names == ["time", "batch"]

    def test_names_setter_none_keeps_lazy_stack_names(self):
        # the members disagree on names, so the stack has no names of its own
        stack = LazyStackedTensorDict(
            TensorDict({}, [3], names=["x"]), TensorDict({}, [3], names=["y"])
        )
        td = TensorDict({"s": stack}, [])
        td.names = None
        assert [member.names for member in td["s"].tensordicts] == [["x"], ["y"]]

    def test_names_wrong_length_with_none_entries_warns(self):
        # a list one entry too long, with as many None entries as batch dims,
        # is accepted and dropped until 0.17, where it raises (#2600)
        match = r"Got the dimension names \['s', None\] for 1 batch dims"
        with pytest.warns(FutureWarning, match=match) as record:
            td = TensorDict(a=torch.zeros(3), batch_size=[3], names=["s", None])
        assert record[0].filename == __file__
        assert td.names == [None]
        td = TensorDict(a=torch.zeros(3), batch_size=[3], names=["prev"])
        with pytest.warns(FutureWarning, match=match) as record:
            td.names = ["s", None]
        assert record[0].filename == __file__
        assert td.names == [None]
        with pytest.warns(FutureWarning, match=match):
            assert td.rename_("s", None).names == [None]
        with pytest.warns(FutureWarning, match=match):
            td = TensorDict.from_dict(
                {"a": torch.zeros(3)}, batch_size=[3], names=["s", None]
            )
        assert td.names == [None]

        tc = self.nested_tensorclass("cpu").get("my_nested_tc")
        with pytest.warns(FutureWarning, match="for 4 batch dims") as record:
            tc.names = ["s", None, None, None, None]
        assert record[0].filename == __file__
        assert tc.names == [None] * 4

        stack = LazyStackedTensorDict(
            *[TensorDict(a=torch.zeros(3), batch_size=[3]) for _ in range(2)]
        )
        with pytest.warns(FutureWarning, match="for 2 batch dims") as record:
            stack.names = ["s", "t", None]
        # the members get the list, but the user's call warns once
        assert len(record) == 1
        assert record[0].filename == __file__
        assert stack.names == ["s", None]

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Got the dimension names")
            # lists of the right length, and from_list, which names the result
            # once its batch size is known, do not warn
            td = TensorDict(a=torch.zeros(3, 4), batch_size=[3, 4])
            td.names = ["s", None]
            assert td.names == ["s", None]
            td.names = [None, None]
            assert td.names == [None, None]
            td = TensorDict.from_list(
                [{"a": torch.zeros(3)}, {"a": torch.zeros(3)}],
                names=["x", "y"],
                auto_batch_size=True,
            )
            assert td.names == ["x", "y"]
            # a list of the wrong length that raises keeps raising
            with pytest.raises(ValueError, match="the length of the dimension names"):
                TensorDict(a=torch.zeros(3), batch_size=[3], names=["s", "t"])

    def test_names_setter_clash_leaves_names_unchanged(self):
        td = TensorDict(
            {
                "a": TensorDict({}, [2, 3], names=[None, "x"]),
                "b": TensorDict({}, [2, 3], names=[None, "batch"]),
            },
            [2],
        )
        with pytest.raises(ValueError, match=r"at key \('b',\)"):
            td.names = ["batch"]
        assert td.names == [None]
        assert td["a"].names == [None, "x"]
        assert td["b"].names == [None, "batch"]

        # a clash in a grandchild leaves the child unchanged too
        gc = TensorDict({}, [2, 3, 4], names=[None, None, "batch"])
        child = TensorDict({"gc": gc}, [2, 3], names=["time", "agent"])
        td = TensorDict({"c": child}, [2], names=["time"])
        with pytest.raises(ValueError, match=r"at key \('c', 'gc'\)"):
            td.names = ["batch"]
        assert td.names == ["time"]
        assert td["c"].names == ["time", "agent"]
        assert td["c", "gc"].names == ["time", "agent", "batch"]

    @pytest.mark.parametrize("device", [None, "cpu"])
    def test_init_unnamed_parent_skips_rename(self, device, monkeypatch):
        # children whose shared dims are unnamed must not make the parent
        # re-walk every child once per child
        source = {
            f"k{i}": TensorDict({}, [2, 3], names=[None, "agent"]) for i in range(10)
        }
        rename_calls = [0]
        orig_rename_subtds = TensorDict._rename_subtds

        def _spy(self, names):
            rename_calls[0] += 1
            return orig_rename_subtds(self, names)

        monkeypatch.setattr(TensorDict, "_rename_subtds", _spy)
        td = TensorDict(source, [2], device=device)
        # only the constructor's own _set_names(None)
        assert rename_calls[0] <= 1
        assert td["k0"].names == [None, "agent"]

    def test_split(self):
        td = TensorDict(
            {}, batch_size=[3, 4, 1, 6], names=["a", "b", "c", "d"], lock=True
        )
        _, tdu = td.split(dim=-1, split_size=[3, 3])
        assert tdu.names == ["a", "b", "c", "d"]
        _, tdu = td.split(dim=1, split_size=[1, 3])
        assert tdu.names == ["a", "b", "c", "d"]
        # assert tdu.is_locked

    def test_squeeze(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=None)
        td.names = ["a", "b", "c", "d"]
        tds = td.squeeze(0)
        assert tds.names == ["a", "b", "c", "d"]
        td = TensorDict(batch_size=[3, 1, 5, 6], names=None)
        td.names = ["a", "b", "c", "d"]
        tds = td.squeeze(1)
        assert tds.names == ["a", "c", "d"]

    def test_squeeze_all_singleton_dims(self):
        # squeeze() of a named tensordict whose dims all have size 1, or of the
        # 0-dim tensordict that squeeze(dim) leaves, gives a 0-dim tensordict
        td = TensorDict(
            {"a": torch.zeros(1, 1, 2)}, batch_size=[1, 1], names=["x", "y"]
        )
        for tds in (
            td.squeeze(),
            torch.squeeze(td),
            td.squeeze(1).squeeze(0).squeeze(),
        ):
            assert tds.batch_size == torch.Size([])
            assert tds.names == []
            assert tds["a"].shape == torch.Size([2])
        # the same when it is nested in a tensordict that is squeezed
        parent = TensorDict({"td": td}, batch_size=[1], names=["x"])
        tds = parent.squeeze()
        assert tds.batch_size == torch.Size([])
        assert tds.names == []
        assert tds["td", "a"].shape == torch.Size([1, 2])

    def test_squeeze_td(self):
        td = self.squeezed_td("cpu")
        with pytest.raises(
            RuntimeError, match="Names of a lazy tensordict cannot be modified"
        ):
            td.names = list("abcd")

    def test_stack(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=["a", "b", "c", "d"])
        tds = LazyStackedTensorDict.lazy_stack([td, td], 0)
        assert tds.names == [None, "a", "b", "c", "d"]
        tds = LazyStackedTensorDict.lazy_stack([td, td], -1)
        assert tds.names == ["a", "b", "c", "d", None]
        tds = LazyStackedTensorDict.lazy_stack([td, td], 2)
        tds.names = list("mnopq")
        assert tds.names == list("mnopq")
        assert td.names == ["m", "n", "p", "q"]

    def test_stack_assign(self):
        td = TensorDict(
            {"": TensorDict({}, [3, 4], names=["c", "d"])}, [3], names=["c"]
        )
        tds = LazyStackedTensorDict.lazy_stack([td, td], -1)
        assert tds.names == ["c", None]
        assert tds[""].names == ["c", None, "d"]
        with pytest.raises(ValueError):
            tds.names = ["c", "d"]
        tds.names = ["c", "e"]
        assert tds.names == ["c", "e"]
        assert tds[""].names == ["c", "e", "d"]
        assert tds[0].names == ["e"]
        assert tds[0][""].names == tds[""][0].names == ["e", "d"]

    def test_lazy_stack_set_names_failure_leaves_names_unchanged(self):
        # A clash on the stack dim, or a names list of the wrong length, is
        # rejected and leaves the stack dim name and every member unchanged.
        members = [
            TensorDict({"x": torch.zeros(3)}, [3], names=["inner"]),
            TensorDict({"x": torch.zeros(3)}, [3], names=["inner"]),
        ]
        stack = LazyStackedTensorDict(*members, stack_dim=0)
        assert stack.names == [None, "inner"]
        with pytest.raises(ValueError, match="already taken"):
            stack.names = ["inner", "inner"]
        assert stack.names == [None, "inner"]
        assert [member.names for member in stack.tensordicts] == [
            ["inner"],
            ["inner"],
        ]

        # The clash is only on the second member. The first must not stay
        # renamed.
        first = TensorDict({"x": torch.zeros(3, 4)}, [3], names=["inner"])
        second = TensorDict(
            {"nested": TensorDict({}, [3, 4], names=["inner", "taken"])},
            [3],
            names=["inner"],
        )
        stack = LazyStackedTensorDict(first, second, stack_dim=0)
        assert stack.names == [None, "inner"]
        with pytest.raises(ValueError, match="already taken"):
            stack.names = ["taken", "other"]
        assert first.names == ["inner"]
        assert second.names == ["inner"]
        assert second["nested"].names == ["inner", "taken"]
        assert stack.names == [None, "inner"]

        stack = LazyStackedTensorDict(
            TensorDict({}, [3], names=["feat"]),
            TensorDict({}, [3], names=["feat"]),
            stack_dim=0,
        )
        assert stack.ndim == 2
        assert stack.names == [None, "feat"]
        for bad_names, error in ((["batch"], ValueError), ([], IndexError)):
            with pytest.raises(error):
                stack.names = bad_names
            assert stack.names == [None, "feat"]
            assert [member.names for member in stack.tensordicts] == [
                ["feat"],
                ["feat"],
            ]

    def test_lazy_stack_set_names_members_disagree(self):
        # The members of a nested lazy stack may disagree on names, so that
        # its names getter raises. Setting the names of the outer stack still
        # works, and a failure restores each member's own names.
        def disagree():
            return LazyStackedTensorDict(
                TensorDict({}, [3], names=["p"]), TensorDict({}, [3], names=["q"])
            )

        outer = LazyStackedTensorDict(disagree(), disagree())
        outer.names = [None, None, None]
        assert outer.names == [None, None, None]

        clash = TensorDict(
            {"nested": TensorDict({}, [3, 4], names=["inner", "taken"])},
            [3],
            names=["inner"],
        )
        outer = LazyStackedTensorDict(
            disagree(), LazyStackedTensorDict(clash, clash.clone())
        )
        with pytest.raises(ValueError, match="non-unique"):
            outer.names = [None, None, "taken"]
        assert [member.names for member in outer.tensordicts[0].tensordicts] == [
            ["p"],
            ["q"],
        ]
        assert outer.tensordicts[1].names == [None, "inner"]
        assert clash["nested"].names == ["inner", "taken"]

        # The same member twice: its second snapshot is taken after its first
        # rename, so the rollback must end on the first one.
        td = TensorDict({}, [3], names=["q"])
        bad = TensorDict(
            {"n": TensorDict({}, [3, 4], names=["q", "taken"])}, [3], names=["q"]
        )
        stack = LazyStackedTensorDict(td, td, bad)
        with pytest.raises(ValueError, match="non-unique"):
            stack.names = ["s", "taken"]
        assert td.names == ["q"]
        assert stack.names == [None, "q"]

    @pytest.mark.filterwarnings("error::FutureWarning")
    def test_lazy_stack_set_names_checks_names_after_rename(self):
        # The stack dim name is checked against the names the members have
        # after the rename, as on a TensorDict (#2690).
        def stack_of(names):
            return LazyStackedTensorDict(
                *[TensorDict({"x": torch.zeros(3)}, [3], names=names) for _ in "ab"]
            )

        # the members drop the name the stack dim takes
        stack = stack_of(["inner"])
        stack.names = ["inner", "other"]
        assert stack.names == ["inner", "other"]
        assert [member.names for member in stack.tensordicts] == [
            ["other"],
            ["other"],
        ]
        nested = TensorDict({"n": TensorDict({}, [3, 4])}, [3], names=["inner"])
        stack = LazyStackedTensorDict(nested, nested.clone())
        stack.names = ["inner", "other"]
        assert stack.names == ["inner", "other"]
        assert stack.tensordicts[0]["n"].names == ["other", None]

        # the members keep the name: raises, and leaves the names unchanged
        stack = stack_of(["inner"])
        with pytest.raises(ValueError, match="already taken"):
            stack.names = ["inner", "inner"]
        assert stack.names == [None, "inner"]
        assert [member.names for member in stack.tensordicts] == [
            ["inner"],
            ["inner"],
        ]

        # the members take the name in the rename: accepted until 0.17
        stack = stack_of(None)
        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
            stack.names = ["a", "a"]
        assert stack.names == ["a", "a"]

        # names that do not clash stay silent
        stack = stack_of(["inner"])
        stack.names = ["s", "inner"]
        assert stack.names == ["s", "inner"]
        stack.names = [None, "other"]
        assert stack.names == [None, "other"]

    @pytest.mark.filterwarnings("error::FutureWarning")
    def test_lazy_stack_stack_dim_name_taken_warns(self):
        # A stack dim name that names a member dim gives duplicate names:
        # accepted until 0.17 (#2690).
        members = [TensorDict({"x": torch.zeros(3)}, [3], names=["a"]) for _ in "ab"]
        with pytest.warns(FutureWarning, match="stack dim name 'a'"):
            stack = LazyStackedTensorDict(*members, stack_dim_name="a")
        assert stack.names == ["a", "a"]
        with pytest.warns(FutureWarning, match="stack dim name 'a'"):
            stack = lazy_stack(members, stack_dim_name="a")
        assert stack.names == ["a", "a"]

        # an unnamed or unique stack dim name stays silent
        assert LazyStackedTensorDict(*members).names == [None, "a"]
        assert lazy_stack(members).names == [None, "a"]
        assert lazy_stack(members, stack_dim_name="s").names == ["s", "a"]
        assert LazyStackedTensorDict(*members, stack_dim_name="s").names == [
            "s",
            "a",
        ]

    def test_sub_td(self):
        td = self.sub_td("cpu")
        with pytest.raises(
            RuntimeError, match="Names of a subtensordict cannot be modified"
        ):
            td.names = list("abcd")
        td = self.sub_td2("cpu")
        with pytest.raises(
            RuntimeError, match="Names of a subtensordict cannot be modified"
        ):
            td.names = list("abcd")

    def test_subtd(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=["a", "b", "c", "d"])
        assert td._get_sub_tensordict(0).names == ["b", "c", "d"]
        assert td._get_sub_tensordict((slice(None), 0)).names == ["a", "c", "d"]
        assert td._get_sub_tensordict((0, slice(None))).names == ["b", "c", "d"]
        assert td._get_sub_tensordict((0, slice(None, 1))).names == ["b", "c", "d"]
        assert td._get_sub_tensordict((..., -1)).names == ["a", "b", "c"]
        assert td._get_sub_tensordict((0, ..., -1)).names == ["b", "c"]
        assert td._get_sub_tensordict((0, ..., [-1])).names == ["b", "c", "d"]
        assert td._get_sub_tensordict((0, ..., torch.tensor([-1]))).names == [
            "b",
            "c",
            "d",
        ]
        assert td._get_sub_tensordict((0, ..., torch.tensor(-1))).names == ["b", "c"]
        assert td._get_sub_tensordict((0, ..., slice(None, -1))).names == [
            "b",
            "c",
            "d",
        ]
        assert td._get_sub_tensordict((slice(None, 1), ..., slice(None, -1))).names == [
            "a",
            "b",
            "c",
            "d",
        ]
        tdbool = td._get_sub_tensordict(torch.ones(3, dtype=torch.bool))
        assert tdbool.names == ["a", "b", "c", "d"]
        assert tdbool.ndim == 4
        tdbool = td._get_sub_tensordict(torch.ones(3, 4, dtype=torch.bool))
        assert tdbool.names == [None, "c", "d"]
        assert tdbool.ndim == 3
        with pytest.raises(
            RuntimeError, match="Names of a subtensordict cannot be modified"
        ):
            tdbool.names = "All work and no play makes Jack a dull boy"

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "non_blocking_pin",
        (
            [False]
            if not torch.cuda.is_available() and not is_npu_available()
            else [False, True]
        ),
    )
    @pytest.mark.parametrize("num_threads", [0, 1, 4, None])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_to(self, device, non_blocking_pin, num_threads, inplace):
        td = TensorDict(
            {"": TensorDict({}, [3, 4, 1, 6])},
            batch_size=[3, 4, 1, 6],
            names=["a", "b", "c", "d"],
        )
        tdt = td.to(
            device,
            non_blocking_pin=non_blocking_pin,
            num_threads=num_threads,
            inplace=inplace,
        )
        assert tdt.names == ["a", "b", "c", "d"]
        assert tdt.device == device
        for v in tdt.values(True, True):
            assert v.device == device
        if inplace:
            assert tdt is td
        else:
            assert tdt is not td

    def test_attrs_basic(self):
        td = TensorDict(
            {
                "a": torch.zeros(3, 4, dtype=torch.float32),
                "b": torch.ones(3, dtype=torch.int64),
            },
            batch_size=[3],
        )
        attrs = td.attrs()
        assert isinstance(attrs["a"], TensorAttrs)
        assert isinstance(attrs["b"], TensorAttrs)
        assert attrs["a"].tgt_device == td["a"].device
        assert attrs["a"].tgt_dtype == torch.float32
        assert attrs["a"].tgt_shape == torch.Size([3, 4])
        assert attrs["b"].tgt_dtype == torch.int64
        assert attrs["b"].tgt_shape == torch.Size([3])

    def test_attrs_fields_subset(self):
        td = TensorDict({"a": torch.zeros(3, dtype=torch.float32)}, batch_size=[3])
        attrs = td.attrs(fields=("device",))
        assert attrs["a"].tgt_device == td["a"].device
        assert attrs["a"].tgt_dtype is None
        assert attrs["a"].tgt_shape is None

    def test_attrs_nested(self):
        td = TensorDict(
            {
                "a": torch.zeros(3, dtype=torch.float32),
                "sub": TensorDict(
                    {"b": torch.zeros(3, dtype=torch.int32)}, batch_size=[3]
                ),
            },
            batch_size=[3],
        )
        attrs = td.attrs()
        assert isinstance(attrs["a"], TensorAttrs)
        assert isinstance(attrs[("sub", "b")], TensorAttrs)
        assert attrs[("sub", "b")].tgt_dtype == torch.int32

    def test_to_attrs_td_dtype(self):
        td = TensorDict(
            {
                "a": torch.zeros(3, dtype=torch.float32),
                "b": torch.zeros(3, dtype=torch.float64),
                "sub": {"c": torch.zeros(3, dtype=torch.float32)},
            },
            batch_size=[3],
        )
        spec = TensorDict(
            {
                "a": TensorAttrs(tgt_dtype=torch.int32, batch_size=()),
                "b": TensorAttrs(tgt_dtype=torch.int64, batch_size=()),
                "sub": {"c": TensorAttrs(tgt_dtype=torch.int16, batch_size=())},
            },
            batch_size=[],
        )
        out = td.to(spec)
        assert out["a"].dtype == torch.int32
        assert out["b"].dtype == torch.int64
        assert out["sub", "c"].dtype == torch.int16
        assert out["a"].device == td["a"].device

    def test_to_attrs_td_missing_keys_passthrough(self):
        td = TensorDict(
            {
                "a": torch.zeros(3, dtype=torch.float32),
                "b": torch.zeros(3, dtype=torch.float32),
                "c": torch.zeros(3, dtype=torch.float32),
            },
            batch_size=[3],
        )
        spec = TensorDict(
            {"a": TensorAttrs(tgt_dtype=torch.int32, batch_size=())},
            batch_size=[],
        )
        out = td.to(spec)
        assert out["a"].dtype == torch.int32
        assert out["b"].dtype == torch.float32
        assert out["c"].dtype == torch.float32

    def test_to_attrs_td_roundtrip(self):
        td = TensorDict(
            {
                "a": torch.zeros(3, dtype=torch.float32),
                "b": torch.zeros(3, dtype=torch.int64),
            },
            batch_size=[3],
        )
        out = td.to(td.attrs())
        assert out["a"].dtype == td["a"].dtype
        assert out["b"].dtype == td["b"].dtype
        assert out["a"].device == td["a"].device
        assert out["b"].device == td["b"].device

    @pytest.mark.parametrize("num_threads", [0, 2])
    def test_to_attrs_td_inplace_nested(self, num_threads):
        # https://github.com/pytorch/tensordict/issues/2498
        td = TensorDict(
            a=torch.zeros(3),
            nested=TensorDict(b=torch.zeros(3), batch_size=[3]),
            batch_size=[3],
        )
        nested = td["nested"]
        attrs = td.to(torch.float64).attrs()
        out = td.to(attrs, inplace=True, num_threads=num_threads)
        assert out is td
        assert set(td.keys()) == {"a", "nested"}
        assert td["nested"] is nested
        assert td["a"].dtype == torch.float64
        assert td["nested", "b"].dtype == torch.float64

        # The same with a separate out
        out = td.clone()
        td._fast_apply(lambda x: x + 1, out=out, num_threads=num_threads)
        assert set(out.keys()) == {"a", "nested"}
        assert out["nested"] is not out
        assert set(out["nested"].keys()) == {"b"}
        assert (out["nested", "b"] == 1).all()

    def test_to_attrs_td_extra_positional_rejected(self):
        td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
        spec = TensorDict(
            {"a": TensorAttrs(tgt_dtype=torch.int32, batch_size=())}, batch_size=[]
        )
        with pytest.raises(TypeError, match="does not accept additional positional"):
            td.to(spec, "cpu")

    def test_to_attrs_td_non_blocking_pin_rejected(self):
        td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
        spec = TensorDict(
            {"a": TensorAttrs(tgt_dtype=torch.int32, batch_size=())}, batch_size=[]
        )
        with pytest.raises(NotImplementedError, match="non_blocking_pin"):
            td.to(spec, non_blocking_pin=True)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires cuda for device casting"
    )
    def test_to_attrs_td_heterogeneous_devices(self):
        td = TensorDict(
            {
                "a": torch.zeros(3, device="cuda:0"),
                "b": torch.zeros(3, device="cpu"),
            },
            batch_size=[3],
        )
        spec = TensorDict(
            {
                "a": TensorAttrs(tgt_device=torch.device("cpu"), batch_size=()),
                "b": TensorAttrs(tgt_device=torch.device("cuda:0"), batch_size=()),
            },
            batch_size=[],
        )
        out = td.to(spec)
        assert out["a"].device.type == "cpu"
        assert out["b"].device.type == "cuda"

    def test_to_attrs_td_sync_toggle(self, monkeypatch):
        td = TensorDict(
            {"a": torch.zeros(3), "b": torch.zeros(3, dtype=torch.float32)},
            batch_size=[3],
        )

        sync_calls = [0]
        orig_sync = TensorDict._sync_all

        def _spy(self):
            sync_calls[0] += 1
            return orig_sync(self)

        monkeypatch.setattr(TensorDict, "_sync_all", _spy)

        # Dtype-only spec: no device transfer at all — no sync.
        spec_dtype_only = TensorDict(
            {"a": TensorAttrs(tgt_dtype=torch.int32, batch_size=())},
            batch_size=[],
        )
        td.to(spec_dtype_only)
        assert sync_calls[0] == 0

        # Source is CPU, target is CPU: not a D2H transfer — no sync needed.
        spec_cpu_to_cpu = TensorDict(
            {"a": TensorAttrs(tgt_device=torch.device("cpu"), batch_size=())},
            batch_size=[],
        )
        sync_calls[0] = 0
        td.to(spec_cpu_to_cpu)
        assert sync_calls[0] == 0

        # non_blocking=True disables the sync regardless of direction.
        sync_calls[0] = 0
        td.to(spec_cpu_to_cpu, non_blocking=True)
        assert sync_calls[0] == 0

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="D2H sync path requires a non-CPU source device",
    )
    def test_to_attrs_td_sync_d2h_only(self, monkeypatch):
        td = TensorDict(
            {
                "a": torch.zeros(3, device="cuda:0"),
                "b": torch.zeros(3, device="cuda:0"),
            },
            batch_size=[3],
        )

        sync_calls = [0]
        orig_sync = TensorDict._sync_all

        def _spy(self):
            sync_calls[0] += 1
            return orig_sync(self)

        monkeypatch.setattr(TensorDict, "_sync_all", _spy)

        # D2H: source cuda, target cpu — must sync.
        spec_d2h = TensorDict(
            {"a": TensorAttrs(tgt_device=torch.device("cpu"), batch_size=())},
            batch_size=[],
        )
        td.to(spec_d2h)
        assert sync_calls[0] == 1

        # H2D: build a CPU source and send to cuda — should not sync.
        td_cpu = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
        spec_h2d = TensorDict(
            {"a": TensorAttrs(tgt_device=torch.device("cuda:0"), batch_size=())},
            batch_size=[],
        )
        sync_calls[0] = 0
        td_cpu.to(spec_h2d)
        assert sync_calls[0] == 0

    def test_unbind(self):
        td = TensorDict(batch_size=[3, 4, 1, 6], names=["a", "b", "c", "d"])
        *_, tdu = td.unbind(-1)
        assert tdu.names == ["a", "b", "c"]
        *_, tdu = td.unbind(-2)
        assert tdu.names == ["a", "b", "d"]

    def test_unsqueeze(self):
        td = TensorDict(batch_size=[3, 4, 5, 6], names=None)
        td.names = ["a", "b", "c", "d"]
        tdu = td.unsqueeze(0)
        assert tdu.names == [None, "a", "b", "c", "d"]
        tdu = td.unsqueeze(-1)
        assert tdu.names == ["a", "b", "c", "d", None]
        tdu = td.unsqueeze(2)
        assert tdu.names == ["a", "b", None, "c", "d"]
        # Reducing the only dim of a named tensordict leaves its names empty
        td = TensorDict(a=torch.zeros(3), batch_size=[3], names=["a"]).sum(0)
        tdu = td.unsqueeze(0)
        assert tdu.batch_size == torch.Size([1])
        assert tdu.names == [None]

    @pytest.mark.parametrize(
        "batch_size,names",
        [
            ([3], ["a"]),
            ([2, 3], ["a", "b"]),
            ([3, 1], ["a", "b"]),
            ([2, 3], [None, "b"]),
        ],
    )
    def test_prod_keepdim_names(self, batch_size, names):
        # keepdim=True keeps the name of the reduced dim, as sum and mean do
        td = TensorDict(
            a=torch.rand(*batch_size, 4), batch_size=batch_size, names=names
        )
        for dim in range(-len(batch_size), len(batch_size)):
            result = td.prod(dim, keepdim=True)
            assert result.names == names
            assert result.names == td.sum(dim, keepdim=True).names
        # the result stacks with a slice of the input, as for sum
        stacked = LazyStackedTensorDict.lazy_stack([td.prod(0, keepdim=True), td[:1]])
        assert stacked.to_tensordict().names == [None, *names]

    @pytest.mark.parametrize(
        "reduction", ["amax", "amin", "nansum", "nanmean", "std", "var", "quantile"]
    )
    def test_reduction_keepdim_no_dim_names(self, reduction):
        # https://github.com/pytorch/tensordict/issues/2326: with no dim, keepdim=True
        # keeps every batch dim, so every name
        td = TensorDict(a=torch.rand(2, 3), batch_size=[2, 3], names=["x", "y"])
        args = (0.5,) if reduction == "quantile" else ()
        result = getattr(td, reduction)(*args, keepdim=True)
        assert result.batch_size == torch.Size([1, 1])
        assert result.names == ["x", "y"]

    @pytest.mark.parametrize("reduction", ["amax", "amin", "max", "min"])
    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_reduction_dim_names(self, reduction, dim):
        td = TensorDict(a=torch.rand(2, 3), batch_size=[2, 3], names=["x", "y"])
        kept = [name for i, name in enumerate(td.names) if i != dim % 2]
        result = getattr(td, reduction)(dim)
        result_keepdim = getattr(td, reduction)(dim, keepdim=True)
        if reduction in ("max", "min"):
            for res in result:
                assert res.names == kept
            for res in result_keepdim:
                assert res.names == ["x", "y"]
        else:
            assert result.names == kept
            assert result_keepdim.names == ["x", "y"]
            assert result_keepdim.batch_size == td.sum(dim, keepdim=True).batch_size

    @pytest.mark.parametrize("td_type", ["td", "lazy", "sub"])
    @pytest.mark.parametrize("dim", [0, 1, -1, (0, 1), None])
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_logsumexp_names(self, td_type, dim, keepdim):
        # https://github.com/pytorch/tensordict/issues/2332: logsumexp keeps the
        # names of the dims that survive the reduction, as sum does
        td = TensorDict(
            a=torch.rand(2, 3),
            nested=TensorDict(b=torch.rand(2, 3, 4), batch_size=[2, 3]),
            batch_size=[2, 3],
            names=["x", "y"],
        )
        if td_type == "lazy":
            td = LazyStackedTensorDict(*td.unbind(0), stack_dim=0)
            td.names = ["x", "y"]
        elif td_type == "sub":
            td = td._get_sub_tensordict(slice(0, 2))
        expected = td.sum(dim, keepdim=keepdim)
        result = td.logsumexp(dim, keepdim=keepdim)
        assert result.batch_size == expected.batch_size
        assert result.names == expected.names
        assert result["nested"].names == expected["nested"].names
        out = td.logsumexp(dim, keepdim=keepdim).zero_()
        result = td.logsumexp(dim, keepdim=keepdim, out=out)
        assert result is out
        assert result.names == expected.names
        assert (result == td.logsumexp(dim, keepdim=keepdim)).all()

    @pytest.mark.parametrize("reduction", ["cummax", "cummin"])
    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_cumulative_reduction_names(self, reduction, dim):
        # cummax and cummin keep every batch dim, so every name
        td = TensorDict(a=torch.rand(2, 3), batch_size=[2, 3], names=["x", "y"])
        result = getattr(td, reduction)(dim)
        for res in result:
            assert res.batch_size == td.batch_size
            assert res.names == ["x", "y"]
        assert getattr(td, reduction)(dim, return_indices=False).names == ["x", "y"]

    def test_unsqueeze_td(self):
        td = self.unsqueezed_td("cpu")
        with pytest.raises(
            RuntimeError, match="Names of a lazy tensordict cannot be modified"
        ):
            td.names = list("abcd")


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
