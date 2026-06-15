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

import pytest
import torch
from packaging import version
from tensordict import LazyStackedTensorDict, TensorAttrs, TensorDict

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
TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)

_has_onnx = importlib.util.find_spec("onnxruntime", None) is not None

_v2_5 = TORCH_VERSION >= version.parse("2.5.0")
PYTORCH_TEST_FBCODE = os.getenv("PYTORCH_TEST_FBCODE")

_IS_OSX = platform.system() == "Darwin"
_IS_WINDOWS = sys.platform == "win32"

TD_BATCH_SIZE = 4
HAS_NESTED_TENSOR = (
    getattr(torch, "_nested_compute_contiguous_strides_offsets", None) is not None
)

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
        tdbool = td[torch.ones(3, dtype=torch.bool)]
        assert tdbool.names == [None, "b", "c", "d"]
        assert tdbool.ndim == 4
        tdbool = td[torch.ones(3, 4, dtype=torch.bool)]
        assert tdbool.names == [None, "c", "d"]
        assert tdbool.ndim == 3

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
        assert tdbool.names == [None, "b", "c", "d"]
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
            },
            batch_size=[3],
        )
        spec = TensorDict(
            {
                "a": TensorAttrs(tgt_dtype=torch.int32, batch_size=()),
                "b": TensorAttrs(tgt_dtype=torch.int64, batch_size=()),
            },
            batch_size=[],
        )
        out = td.to(spec)
        assert out["a"].dtype == torch.int32
        assert out["b"].dtype == torch.int64
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

    def test_unsqueeze_td(self):
        td = self.unsqueezed_td("cpu")
        with pytest.raises(
            RuntimeError, match="Names of a lazy tensordict cannot be modified"
        ):
            td.names = list("abcd")


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
