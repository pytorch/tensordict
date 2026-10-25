# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import tensordict.persistent as tensordict_persistent
import torch
from tensordict import (
    from_zarr,
    lazy_stack,
    NonTensorData,
    NonTensorStack,
    PersistentTensorDict,
    TensorDict,
)
from tensordict._td import _SubTensorDict
from tensordict.base import _is_leaf_nontensor
from tensordict.utils import _NUMPY_TO_TORCH_DTYPE_DICT, is_non_tensor
from torch import multiprocessing as mp
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

TIMEOUT = 100


def _keep_batch(batch):
    return batch


class _CustomNonTensor:
    def __eq__(self, other):
        return isinstance(other, _CustomNonTensor)


try:
    import zarr

    _has_zarr = int(zarr.__version__.split(".")[0]) >= 3
except ImportError:
    _has_zarr = False

# dtypes that both torch and zarr v3 support natively
_ZARR_DTYPES = [
    torch_dtype
    for np_dtype, torch_dtype in _NUMPY_TO_TORCH_DTYPE_DICT.items()
    if np_dtype not in (np.dtype("uint16"), np.dtype("uint32"), np.dtype("uint64"))
    or torch_dtype in (torch.uint16, torch.uint32, torch.uint64)
]


@pytest.mark.skipif(not _has_zarr, reason="zarr>=3.0 not found.")
class TestZarrRoundTrip:
    def test_to_from_zarr(self, tmp_path):
        td = TensorDict(
            {
                "a": torch.arange(12, dtype=torch.float32).view(3, 4),
                "b": TensorDict(
                    {"c": torch.arange(60).view(3, 4, 5)},
                    batch_size=[3, 4, 5],
                ),
            },
            batch_size=[3, 4],
        )
        ptd = td.to_zarr(tmp_path / "store.zarr")
        assert isinstance(ptd, PersistentTensorDict)
        td_recon = TensorDict.from_zarr(tmp_path / "store.zarr")
        assert td_recon.batch_size == torch.Size([3, 4])
        assert td_recon["b"].batch_size == torch.Size([3, 4, 5])
        assert (td_recon["a"] == td["a"]).all()
        assert (td_recon["b", "c"] == td["b", "c"]).all()
        assert (td_recon.to_tensordict() == td).all()

    def test_empty_batch_size(self, tmp_path):
        td = TensorDict({"a": torch.randn(3)}, batch_size=[])
        td.to_zarr(tmp_path / "s.zarr")
        td_recon = TensorDict.from_zarr(tmp_path / "s.zarr")
        assert td_recon.batch_size == torch.Size([])
        assert (td_recon["a"] == td["a"]).all()

    @pytest.mark.parametrize("dtype", _ZARR_DTYPES, ids=[str(d) for d in _ZARR_DTYPES])
    def test_dtypes(self, tmp_path, dtype):
        if dtype.is_floating_point:
            value = torch.rand(3).to(dtype)
        elif dtype.is_complex:
            value = torch.complex(torch.rand(3), torch.rand(3)).to(dtype)
        elif dtype == torch.bool:
            value = torch.tensor([True, False, True])
        else:
            value = torch.arange(3).to(dtype)
        td = TensorDict({"x": value}, batch_size=[3])
        td.to_zarr(tmp_path / "s.zarr")
        td_recon = TensorDict.from_zarr(tmp_path / "s.zarr")
        assert td_recon["x"].dtype == dtype
        assert (td_recon["x"] == value).all()

    def test_bfloat16_raises(self, tmp_path):
        td = TensorDict({"x": torch.zeros(3, dtype=torch.bfloat16)}, batch_size=[3])
        with pytest.raises((TypeError, RuntimeError, ValueError)):
            td.to_zarr(tmp_path / "s.zarr")

    @pytest.mark.parametrize(
        "dtype", [torch.bfloat16, torch.complex32, torch.float8_e4m3fn], ids=str
    )
    def test_unsupported_dtype_error_names_entry(self, tmp_path, dtype):
        # #3265: the error named no key
        td = TensorDict(
            obs=TensorDict(w=torch.zeros(2, dtype=dtype), batch_size=[2]),
            batch_size=[2],
        )
        with pytest.raises(TypeError, match=r"\('obs', 'w'\).*zarr.*memmap"):
            td.to_zarr(tmp_path / "d.zarr")
        ztd = PersistentTensorDict(
            filename=tmp_path / "s.zarr", backend="zarr", mode="w"
        )
        with pytest.raises(TypeError, match=r"\('x', 'y'\).*zarr"):
            ztd["x", "y"] = torch.zeros(2, dtype=dtype)
        ztd.close()

    @pytest.mark.parametrize("layout", [torch.jagged, torch.strided], ids=str)
    def test_nested_tensor_error_names_entry(self, tmp_path, layout):
        # #3265: the error named no key
        nt = torch.nested.nested_tensor([torch.zeros(2), torch.zeros(3)], layout=layout)
        td = TensorDict(obs=TensorDict(j=nt, batch_size=[2]), batch_size=[2])
        with pytest.raises(RuntimeError, match=r"\('obs', 'j'\).*zarr.*nested"):
            td.to_zarr(tmp_path / "d.zarr")

    @pytest.mark.parametrize(
        "make, err, hint",
        [
            (
                lambda: torch.zeros(2, dtype=torch.complex64).conj(),
                RuntimeError,
                "resolve_conj",
            ),
            (lambda: torch.zeros(2).to_sparse(), TypeError, "to_dense"),
        ],
        ids=["conj", "sparse"],
    )
    def test_other_numpy_error_names_entry(self, tmp_path, make, err, hint):
        # #3265: a numpy dtype that fails for another reason keeps torch's advice only
        ztd = PersistentTensorDict(
            filename=tmp_path / "s.zarr", backend="zarr", mode="w"
        )
        with pytest.raises(err, match=rf"\('x', 'y'\).*zarr.*{hint}") as info:
            ztd["x", "y"] = make()
        assert "numpy dtypes only" not in str(info.value)
        ztd.close()

    def test_names_roundtrip(self, tmp_path):
        td = TensorDict({"a": torch.randn(3, 4)}, batch_size=[3, 4], names=["x", "y"])
        td.to_zarr(tmp_path / "s.zarr")
        td_recon = TensorDict.from_zarr(tmp_path / "s.zarr")
        assert td_recon.names == ["x", "y"]

    def test_explicit_batch_size(self, tmp_path):
        td = TensorDict({"a": torch.randn(3, 4)}, batch_size=[3, 4])
        td.to_zarr(tmp_path / "s.zarr")
        td_recon = TensorDict.from_zarr(tmp_path / "s.zarr", batch_size=torch.Size([3]))
        assert td_recon.batch_size == torch.Size([3])

    def test_batch_size_setter_incompatible(self, tmp_path):
        # as on a TensorDict, an incompatible batch size raises and the old
        # one stays
        td = TensorDict({"a": torch.zeros(3, 4)}, batch_size=[3])
        ztd = td.to_zarr(tmp_path / "s.zarr")
        with pytest.raises(RuntimeError, match="batch size and array size mismatch"):
            ztd.batch_size = [5]
        assert ztd.batch_size == (3,)
        with pytest.raises(RuntimeError, match="batch size and array size mismatch"):
            ztd.to(batch_size=[5])
        assert ztd.batch_size == (3,)
        ztd.batch_size = [3, 4]
        assert ztd.batch_size == (3, 4)

    def test_foreign_store_inference(self, tmp_path):
        # a store written by raw zarr, without tensordict metadata
        root = zarr.open_group(str(tmp_path / "foreign.zarr"), mode="w")
        arr = root.create_array("x", shape=(3, 4), dtype="float32")
        arr[...] = np.random.randn(3, 4)
        g = root.create_group("sub")
        arr2 = g.create_array("y", shape=(3, 4, 5), dtype="int64")
        arr2[...] = np.arange(60).reshape(3, 4, 5)
        td = TensorDict.from_zarr(tmp_path / "foreign.zarr")
        assert td.batch_size == torch.Size([3, 4])
        assert td["sub"].batch_size == torch.Size([3, 4, 5])

    def test_auto_batch_size(self, tmp_path):
        td = TensorDict({"a": torch.randn(3, 4)}, batch_size=[3, 4])
        td.to_zarr(tmp_path / "s.zarr")
        td_recon = TensorDict.from_zarr(
            tmp_path / "s.zarr", auto_batch_size=True, batch_dims=1
        )
        assert td_recon.batch_size == torch.Size([3])

    def test_auto_batch_size_persistent(self, tmp_path):
        td = TensorDict({"a": torch.randn(3, 4)}, batch_size=[3, 4])
        td.to_zarr(tmp_path / "s.zarr")
        td_recon = PersistentTensorDict.from_zarr(
            tmp_path / "s.zarr", auto_batch_size=True, batch_dims=1
        )
        assert type(td_recon) is PersistentTensorDict
        assert td_recon.batch_size == torch.Size([3])

    def test_free_function(self, tmp_path):
        from tensordict import from_zarr

        td = TensorDict({"a": torch.randn(3)}, batch_size=[3])
        td.to_zarr(tmp_path / "s.zarr")
        td_recon = from_zarr(tmp_path / "s.zarr")
        assert (td_recon["a"] == td["a"]).all()


@pytest.mark.skipif(not _has_zarr, reason="zarr>=3.0 not found.")
class TestZarrNonTensor:
    @pytest.mark.parametrize("allow_pickle", [None, True, False])
    @pytest.mark.parametrize("stack", [False, True])
    @pytest.mark.parametrize("factory", ["constructor", "persistent", "tensor", "free"])
    def test_pickle_read_policy(
        self, tmp_path, monkeypatch, allow_pickle, stack, factory
    ):
        value = (
            NonTensorStack(_CustomNonTensor(), {1, 2})
            if stack
            else NonTensorData(_CustomNonTensor(), batch_size=[2])
        )
        path = tmp_path / "policy.zarr"
        TensorDict({"obj": value, "json": "text", "x": torch.zeros(2)}, [2]).to_zarr(
            path
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            if factory == "constructor":
                td = PersistentTensorDict(
                    filename=path,
                    backend="zarr",
                    batch_size=[2],
                    allow_pickle=allow_pickle,
                )
            else:
                loader = {
                    "persistent": PersistentTensorDict.from_zarr,
                    "tensor": TensorDict.from_zarr,
                    "free": from_zarr,
                }[factory]
                td = loader(path, allow_pickle=allow_pickle)
            assert td["json"] == "text"
            assert td["x"].shape == (2,)
        if allow_pickle is False:

            def unexpected_loads(*args, **kwargs):
                pytest.fail("pickle.loads must not be called")

            monkeypatch.setattr("tensordict.persistent.pickle.loads", unexpected_loads)
            with pytest.raises(RuntimeError, match="allow_pickle=True"):
                td["obj"]
        elif allow_pickle is None:
            with pytest.warns(
                FutureWarning, match="0.17.*allow_pickle=False"
            ) as record:
                td["obj"]
            assert record[0].filename == __file__
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                actual = td["obj"]
                assert actual == (value.tolist() if stack else value.data)

    @pytest.mark.parametrize("allow_pickle", [None, True, False])
    @pytest.mark.parametrize(
        "operation",
        ["nested", "index", "get_at", "clone", "copy", "to", "pickle", "locked"],
    )
    def test_pickle_policy_propagation(self, tmp_path, allow_pickle, operation):
        path = tmp_path / "policy.zarr"
        TensorDict(
            {("nested", "obj"): NonTensorData(_CustomNonTensor(), batch_size=[2])},
            [2],
        ).to_zarr(path)
        td = PersistentTensorDict.from_zarr(path, allow_pickle=allow_pickle)
        # Exercise both cached nested views and those reconstructed from metadata.
        td["nested"]
        key = ("nested", "obj")
        if operation == "nested":
            td, key = td["nested"], "obj"
        elif operation == "index":
            td = td[0]
        elif operation == "clone":
            td = td.clone(recurse=False)
        elif operation == "copy":
            td = td.clone(newfile=tmp_path / "copy.zarr")
        elif operation == "to":
            td = td.to("cpu")
        elif operation == "pickle":
            td = pickle.loads(pickle.dumps(td))
        elif operation == "locked":
            td.lock_()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            if allow_pickle is False:
                with pytest.raises(RuntimeError, match="allow_pickle=True"):
                    td.get_at(key, 0) if operation == "get_at" else td[key]
            elif allow_pickle is None:
                with pytest.warns(FutureWarning, match="0.17.*allow_pickle=False"):
                    td.get_at(key, 0) if operation == "get_at" else td[key]
            else:
                result = td.get_at(key, 0) if operation == "get_at" else td[key]
                assert (
                    result.data if is_non_tensor(result) else result
                ) == _CustomNonTensor()

    def test_pickle_policy_old_state(self, tmp_path):
        path = tmp_path / "old.zarr"
        TensorDict({"obj": _CustomNonTensor()}, []).to_zarr(path)
        td = PersistentTensorDict.from_zarr(path)
        state = td.__getstate__()
        del state["_backend"].allow_pickle
        restored = PersistentTensorDict.__new__(PersistentTensorDict)
        restored.__setstate__(state)
        with pytest.warns(FutureWarning, match="0.17.*allow_pickle=False"):
            assert restored["obj"] == _CustomNonTensor()

    @pytest.mark.parametrize(
        "marker",
        [{}, {"encoding": "other"}, {"encoding": None}, {"encoding": {}}, "pickle"],
    )
    @pytest.mark.parametrize("allow_pickle", [None, True, False])
    def test_unknown_non_tensor_encoding(
        self, tmp_path, monkeypatch, marker, allow_pickle
    ):
        path = tmp_path / "encoding.zarr"
        TensorDict({"obj": _CustomNonTensor()}, []).to_zarr(path)
        td = PersistentTensorDict.from_zarr(path, mode="r+", allow_pickle=allow_pickle)
        td.file["obj"].attrs["__tensordict_non_tensor__"] = marker

        def unexpected_loads(*args, **kwargs):
            pytest.fail("pickle.loads must not be called for an unknown encoding")

        monkeypatch.setattr("tensordict.persistent.pickle.loads", unexpected_loads)
        with pytest.raises(ValueError, match="Unknown zarr non-tensor encoding"):
            td["obj"]

    @pytest.mark.parametrize("writer", ["to_zarr", "from_dict"])
    @pytest.mark.parametrize("allow_pickle", [None, True, False])
    def test_pickle_policy_writer(self, tmp_path, writer, allow_pickle):
        path = tmp_path / "written.zarr"
        td = TensorDict({"obj": _CustomNonTensor()}, [])
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            if writer == "to_zarr":
                written = td.to_zarr(path, allow_pickle=allow_pickle)
            else:
                written = PersistentTensorDict.from_dict(
                    td, path, backend="zarr", allow_pickle=allow_pickle
                )
            if allow_pickle is False:
                with pytest.raises(RuntimeError, match="allow_pickle=True"):
                    written["obj"]
            else:
                assert written["obj"] == _CustomNonTensor()
        # Neither an independent reader nor append mode inherits writer trust.
        for mode in ("r", "a"):
            reopened = from_zarr(path, mode=mode)
            with pytest.warns(FutureWarning, match="0.17.*allow_pickle=False"):
                assert reopened["obj"] == _CustomNonTensor()

    @pytest.mark.parametrize(
        "factory", ["constructor", "from_zarr", "free", "from_dict", "to_zarr"]
    )
    @pytest.mark.parametrize("invalid", [0, 1, "yes", [], np.bool_(True)])
    def test_pickle_policy_validation(self, tmp_path, factory, invalid):
        path = tmp_path / "invalid.zarr"
        td = TensorDict({"x": torch.zeros(2)}, [2])
        with pytest.raises(TypeError, match="allow_pickle must be a bool or None"):
            if factory == "constructor":
                PersistentTensorDict(
                    filename=path, backend="zarr", mode="w", allow_pickle=invalid
                )
            elif factory == "from_zarr":
                TensorDict.from_zarr(path, allow_pickle=invalid)
            elif factory == "free":
                from_zarr(path, allow_pickle=invalid)
            elif factory == "from_dict":
                PersistentTensorDict.from_dict(
                    td, path, backend="zarr", allow_pickle=invalid
                )
            else:
                td.to_zarr(path, allow_pickle=invalid)
        assert not path.exists()

    def test_zarr_nontensor(self, tmpdir):
        path = Path(tmpdir) / "store.zarr"
        td = TensorDict(
            {
                "a": 0,
                "b": 1,
                "c": "a string!",
                ("d", "e"): "another string!",
            },
            [],
        )
        td = td.expand(10)
        ztd = PersistentTensorDict.from_dict(td, filename=path, backend="zarr")
        assert "c" in ztd.keys(is_leaf=_is_leaf_nontensor)
        assert "c" in ztd.keys()
        assert "c" in ztd
        # unlike h5, strings round-trip as str, not bytes
        assert ztd["c"] == "a string!"
        assert ztd.get("c").batch_size == (10,)
        assert ("d", "e") in ztd.keys(True, True, is_leaf=_is_leaf_nontensor)
        assert ("d", "e") in ztd
        assert ztd["d", "e"] == "another string!"
        assert ztd.get(("d", "e")).batch_size == (10,)

        ztd.set("f", NonTensorData(1, batch_size=[10]))
        assert ztd["f"] == 1
        ztd.set(("g", "h"), NonTensorData(1, batch_size=[10]))
        assert ztd["g", "h"] == 1

        td_recover = ztd.to_tensordict()
        assert is_non_tensor(td_recover.get("c"))
        assert is_non_tensor(td_recover.get(("d", "e")))
        assert is_non_tensor(td_recover.get("f"))
        assert is_non_tensor(td_recover.get(("g", "h")))

    def test_pickle_encoded_object(self, tmp_path):
        td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
        ztd = td.to_zarr(tmp_path / "s.zarr")
        ztd.set("obj", NonTensorData(_CustomNonTensor(), batch_size=[3]))
        assert ztd["obj"] == _CustomNonTensor()

    def test_nontensor_is_marked(self, tmp_path):
        td = TensorDict({"c": "a string!"}, [])
        td.to_zarr(tmp_path / "s.zarr")
        root = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
        assert root["c"].attrs.get("__tensordict_non_tensor__") is not None
        assert root["c"].dtype == np.dtype("uint8")

    def test_nontensor_stack(self, tmp_path):
        # one value per row is written as an array of the stack's shape (a
        # payload if the values are not strings) and read back as a
        # NonTensorStack (#2544)
        grid = NonTensorStack(*(NonTensorStack(f"{i}0", f"{i}1") for i in range(3)))
        td = TensorDict(
            {
                "x": torch.zeros(3),
                "text": NonTensorStack("a", "b", "c"),
                ("n", "y"): torch.zeros(3, 2),
                ("n", "grid"): grid,
                "dicts": NonTensorStack({"a": 1}, {"b": 2}, {"c": 3}),
            },
            [3],
        )
        ztd = td.to_zarr(tmp_path / "s.zarr")
        # a stack is written whole: an in-place write raises
        with pytest.raises(RuntimeError, match="in place"):
            ztd.set_at_("text", "q", 0)
        ztd.close()
        root = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
        assert root["text"].shape == (3,)
        assert root["text"].attrs["__tensordict_stack__"] == [3]
        assert root["n/grid"].shape == (3, 2)
        ztd = PersistentTensorDict.from_zarr(tmp_path / "s.zarr", mode="r+")
        assert ztd.entry_class("text") is NonTensorStack
        text = ztd.get("text")
        assert isinstance(text, NonTensorStack)
        assert text.batch_size == (3,)
        assert text.tolist() == ["a", "b", "c"]
        assert ztd[1]["text"] == "b"
        assert ztd.get(("n", "grid")).batch_size == (3, 2)
        assert ztd.get(("n", "grid")).tolist() == grid.tolist()
        assert ztd[2]["n", "grid"] == ["20", "21"]
        assert ztd.get("dicts").tolist() == [{"a": 1}, {"b": 2}, {"c": 3}]
        assert ztd.to_tensordict()["text"] == ["a", "b", "c"]
        ztd["text"] = NonTensorStack("x", "yy", "z")
        assert ztd.get("text").tolist() == ["x", "yy", "z"]
        ztd.close()

    def test_foreign_column_as_stack(self, tmp_path):
        # a string column that another tool wrote, one string per row, reads
        # as a NonTensorStack of its rows with non_tensor_as_stack=True (#2545)
        path = str(tmp_path / "s.zarr")
        root = zarr.open_group(path, mode="w")
        root.create_array("pos", shape=(4,), dtype="float32")[:] = 0
        root.create_array("mission", shape=(4,), dtype=str)[:] = ["a", "b", "c", "d"]
        ztd = PersistentTensorDict.from_zarr(path)
        assert ztd.entry_class("mission") is NonTensorData
        ztd.close()
        ztd = PersistentTensorDict.from_zarr(path, mode="r+", non_tensor_as_stack=True)
        assert ztd.batch_size == (4,)
        assert ztd.entry_class("mission") is NonTensorStack
        assert ztd[1:3].get("mission").tolist() == ["b", "c"]
        assert ztd[2]["mission"] == "c"
        # a value that tensordict wrote is a payload of bytes, not rows
        ztd.set("n", NonTensorData(1234, batch_size=[4]))
        assert ztd.file["n"].shape == (4,)
        assert ztd.entry_class("n") is NonTensorData
        assert ztd[1]["n"] == 1234
        ztd.close()

    @pytest.mark.parametrize(
        "idx",
        [
            2,
            -1,
            slice(1, 5, 2),
            [0, 3],
            torch.tensor([4, 1]),
            torch.tensor([True, False, True, False, False, True]),
            (slice(1, 3),),
        ],
    )
    def test_foreign_column_as_stack_reads_rows_at(self, tmp_path, idx, monkeypatch):
        # an indexed read of a column read as rows builds only the indexed
        # rows, with the result of indexing the whole column (#2814)
        path = str(tmp_path / "s.zarr")
        root = zarr.open_group(path, mode="w")
        root.create_array("pos", shape=(6,), dtype="float32")[:] = 0
        root.create_array("mission", shape=(6,), dtype=str)[:] = [
            f"m{i}" for i in range(6)
        ]
        ztd = PersistentTensorDict.from_zarr(path, non_tensor_as_stack=True)
        expected = ztd.get("mission")[idx]
        num_rows = torch.zeros(6)[idx].numel()
        from_list = NonTensorStack._from_list.__func__
        sizes = []

        def spy(cls, datalist, *args, **kwargs):
            sizes.append(len(datalist))
            return from_list(cls, datalist, *args, **kwargs)

        monkeypatch.setattr(NonTensorStack, "_from_list", classmethod(spy))
        for out in (ztd.get_at("mission", idx), ztd[idx].get("mission")):
            assert type(out) is type(expected)
            assert out.batch_size == expected.batch_size
            assert out.tolist() == expected.tolist()
        assert max(sizes, default=0) <= num_rows
        ztd.close()


@pytest.mark.skipif(not _has_zarr, reason="zarr>=3.0 not found.")
class TestZarrIndexing:
    @pytest.fixture
    def ztd_and_td(self, tmp_path):
        td = TensorDict(
            {
                "a": torch.arange(12, dtype=torch.float32).view(3, 4),
                "b": {"c": torch.arange(60, dtype=torch.float32).view(3, 4, 5)},
            },
            batch_size=[3, 4],
        )
        ztd = td.to_zarr(tmp_path / "s.zarr")
        return ztd, td.to_tensordict()

    @pytest.mark.parametrize(
        "index",
        [
            0,
            slice(1, 3),
            (slice(None), 1),
            torch.tensor([0, 2]),
            [0, 2],
            range(2),
            torch.tensor([True, False, True]),
            (torch.tensor([0, 2]), torch.tensor([1, 3])),
        ],
        ids=[
            "int",
            "slice",
            "tuple_slice_int",
            "tensor",
            "list",
            "range",
            "bool_mask",
            "coordinates",
        ],
    )
    def test_get_at(self, ztd_and_td, index):
        ztd, td = ztd_and_td
        expected = td["a"][index]
        result = ztd.get_at("a", index)
        assert (result == expected).all(), (result, expected)

    def test_get_at_broadcast_mask_fallback(self, ztd_and_td):
        ztd, td = ztd_and_td
        mask = torch.zeros(3, 4, dtype=torch.bool)
        mask[0, 1] = True
        mask[2, 3] = True
        expected = td["a"][mask]
        result = ztd.get_at("a", mask)
        assert (result == expected).all()

    def test_getitem_subtd(self, ztd_and_td):
        ztd, td = ztd_and_td
        sub = ztd[1:3]
        assert (sub["a"] == td["a"][1:3]).all()
        assert (sub["b", "c"] == td["b", "c"][1:3]).all()

    def test_set_at(self, ztd_and_td):
        ztd, td = ztd_and_td
        ztd.set_at_("a", torch.zeros(4), 0)
        assert (ztd.get_at("a", 0) == 0).all()
        ztd.set_at_("a", torch.ones(2, 4), slice(1, 3))
        assert (ztd.get_at("a", slice(1, 3)) == 1).all()

    def test_set_at_mask(self, ztd_and_td):
        ztd, td = ztd_and_td
        mask = torch.tensor([True, False, True])
        ztd.set_at_("a", torch.zeros(2, 4), mask)
        assert (ztd.get_at("a", mask) == 0).all()
        assert (ztd.get_at("a", 1) == td["a"][1]).all()

    def test_setitem_index(self, ztd_and_td):
        ztd, td = ztd_and_td
        value = TensorDict(
            {"a": torch.zeros(4), "b": {"c": torch.zeros(4, 5)}}, batch_size=[4]
        )
        ztd[0] = value
        assert (ztd.get_at("a", 0) == 0).all()
        assert (ztd.get_at(("b", "c"), 0) == 0).all()

    @pytest.mark.parametrize(
        "index",
        [0, (slice(None), 1), torch.tensor([0, 2])],
        ids=["int", "tuple", "tensor"],
    )
    def test_setitem_scalar(self, ztd_and_td, index):
        # a scalar is written into every entry at the index
        ztd, td = ztd_and_td
        ztd[index] = 7
        td[index] = 7
        for key in td.keys(True, True):
            assert (ztd.get(key) == td.get(key)).all(), key


@pytest.mark.skipif(not _has_zarr, reason="zarr>=3.0 not found.")
class TestZarrWriteOps:
    @pytest.fixture
    def ztd(self, tmp_path):
        td = TensorDict(
            {"a": torch.randn(3, 4), "b": {"c": torch.randn(3, 4, 5)}},
            batch_size=[3, 4],
        )
        return td.to_zarr(tmp_path / "s.zarr")

    def test_set_new_key(self, ztd):
        ztd["new"] = torch.rand(3, 4)
        assert "new" in ztd.keys()
        assert ztd["new"].shape == torch.Size([3, 4])

    def test_set_replace_warns(self, ztd):
        with pytest.warns(UserWarning, match="Replacing an array"):
            ztd.set("a", torch.rand(3, 4), inplace=False)

    def test_fill_zero(self, ztd):
        ztd.fill_("a", 1.0)
        assert (ztd["a"] == 1).all()
        ztd.zero_()
        assert (ztd["a"] == 0).all()
        assert (ztd["b", "c"] == 0).all()

    def test_inplace_methods_write(self, ztd, tmp_path):
        # #2242: the in-place methods write their result to the store
        expected = ztd.to_tensordict()
        other = expected.apply(lambda x: torch.rand_like(x) + 1)
        for td in (ztd, expected):
            assert td.abs_() is td
            assert td.add_(other, alpha=2.0) is td
            td *= 2
            assert td.clamp_max_(other) is td
            assert td.update_(other.select("a") + 1) is td
            assert td.update_(other * 3, keys_to_update=[("b", "c")]) is td
        reopened = PersistentTensorDict.from_zarr(tmp_path / "s.zarr")
        torch.testing.assert_close(reopened.to_tensordict(), expected)
        ztd.copy_(other)
        torch.testing.assert_close(reopened.to_tensordict(), other)

    def test_inplace_methods_write_lazy_stack(self, ztd, tmp_path):
        # #2570: the in-place methods of a lazy stack write to its zarr members
        expected = ztd.to_tensordict()
        dense = expected.clone()
        stack = lazy_stack([ztd, dense])
        assert stack.add_(1.0) is stack
        assert stack.lerp_(stack.to_tensordict() * 0, 0.5) is stack
        expected.add_(1.0).lerp_(expected * 0, 0.5)
        reopened = PersistentTensorDict.from_zarr(tmp_path / "s.zarr")
        torch.testing.assert_close(reopened.to_tensordict(), expected)
        torch.testing.assert_close(dense, expected)
        assert stack.zero_() is stack
        assert (reopened["a"] == 0).all()
        assert (reopened["b", "c"] == 0).all()
        # a read-only member is left unchanged, with a warning
        stack = lazy_stack([ztd, PersistentTensorDict.from_zarr(tmp_path / "s.zarr")])
        with pytest.warns(FutureWarning, match="add_.*leaves the file unchanged"):
            stack.add_(1.0)
        assert (reopened["a"] == 1).all()

    def test_inplace_methods_write_nested_lazy_stack(self, ztd, tmp_path):
        # #2592: the in-place methods of a tensordict write to the zarr
        # members of a lazy stack that it holds
        expected = ztd.to_tensordict()
        dense = expected.clone()
        data = TensorDict({"n": {"s": lazy_stack([ztd, dense])}}, batch_size=[2])
        assert data.add_(1.0) is data
        assert data.mul_(torch.tensor(2.0)) is data
        expected.add_(1.0).mul_(2.0)
        reopened = PersistentTensorDict.from_zarr(tmp_path / "s.zarr")
        torch.testing.assert_close(reopened.to_tensordict(), expected)
        torch.testing.assert_close(dense, expected)
        # a read-only member is left unchanged, with a warning
        stack = lazy_stack([ztd, PersistentTensorDict.from_zarr(tmp_path / "s.zarr")])
        data = TensorDict({"s": stack}, batch_size=[2])
        with pytest.warns(FutureWarning, match="add_.*leaves the file unchanged"):
            data.add_(1.0)
        torch.testing.assert_close(reopened["a"], expected["a"] + 1)

    def test_inplace_read_only_warns(self, ztd, tmp_path):
        # The in-place methods leave a read-only store unchanged, as before
        # they wrote to stores, and warn that this will raise
        ro = PersistentTensorDict.from_zarr(tmp_path / "s.zarr")
        before = ro.to_tensordict()
        match = r"^Calling add_\(\) on a tensordict stored in a file opened read-only.*0\.17"
        with pytest.warns(FutureWarning, match=match) as record:
            assert ro.add_(1.0) is ro
        assert record[0].filename == __file__
        with pytest.warns(FutureWarning, match="update_"):
            ro.update_(before + 1)
        torch.testing.assert_close(ro.to_tensordict(), before)
        # masked_fill_ raises on the rows of the store too, as it did before
        mask = torch.tensor([True, False])
        for call in (
            lambda: ro[0:2].masked_fill_(mask, 0.0),
            lambda: lazy_stack([ro[0], ro[1]]).masked_fill_(mask, 0.0),
        ):
            with pytest.raises(ValueError, match="read-only"):
                call()
        torch.testing.assert_close(ro.to_tensordict(), before)

    def test_masked_fill_(self, ztd):
        ztd.zero_()
        ztd.masked_fill_(torch.tensor([True, False, False]), 3.0)
        assert (ztd["a"][0] == 3).all()
        assert (ztd["a"][1:] == 0).all()

    def test_missing_key_reads_no_data(self, ztd, monkeypatch):
        # A membership test for a missing key, and the one in set, must not
        # read the arrays of the store
        reads = []
        getitem = zarr.Array.__getitem__

        def counting_getitem(self, *args, **kwargs):
            reads.append(self.name)
            return getitem(self, *args, **kwargs)

        monkeypatch.setattr(zarr.Array, "__getitem__", counting_getitem)
        assert "missing" not in ztd.keys()
        assert ("b", "missing") not in ztd.keys(True)
        assert "a" in ztd.keys()
        ztd.set(("new_group", "x"), torch.zeros(3, 4))
        assert reads == []
        with pytest.raises(KeyError, match="key missing not found"):
            ztd._get_array("missing")
        assert reads == []

    def test_fill_zero_masked_fill_no_read(self, ztd, tmp_path, monkeypatch):
        # fill_, zero_, masked_fill_ and detach_ write to the store without
        # reading it
        td = ztd.to_tensordict()

        def no_read(*args, **kwargs):
            raise AssertionError("the store was read")

        backend = type(ztd._backend)
        with monkeypatch.context() as patch:
            patch.setattr(backend, "read_full", no_read)
            patch.setattr(backend, "read_at", no_read)
            assert ztd.detach_() is ztd
            mask = torch.tensor([True, False, True])
            for dest in (ztd, td):
                assert dest.fill_("a", 2.0) is dest
                assert dest.fill_("b", 3.0) is dest
                assert dest.masked_fill_(mask, -1.0) is dest
        reopened = PersistentTensorDict.from_zarr(tmp_path / "s.zarr")
        torch.testing.assert_close(reopened.to_tensordict(), td)
        with monkeypatch.context() as patch:
            patch.setattr(backend, "read_full", no_read)
            patch.setattr(backend, "read_at", no_read)
            assert ztd.zero_() is ztd
        reopened = PersistentTensorDict.from_zarr(tmp_path / "s.zarr")
        torch.testing.assert_close(reopened.to_tensordict(), td.zero_())

    def test_del(self, ztd):
        ztd.del_("a")
        assert "a" not in ztd.keys()

    @pytest.mark.parametrize("key", ["a", "b", ("b", "c")])
    def test_del_existing_key_does_not_warn(self, ztd, key):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ztd.del_(key)
        assert key not in ztd.keys(True)

    @pytest.mark.parametrize("key", ["missing", ("b", "missing")])
    def test_del_missing_key_warns(self, ztd, key):
        # https://github.com/pytorch/tensordict/issues/2062
        with pytest.warns(
            FutureWarning,
            match=r"del_ of a missing key .* Starting with TensorDict 0\.17, the "
            r"default will change to raising a KeyError",
        ) as record:
            assert ztd.del_(key) is ztd
        assert len(record) == 1
        assert record[0].filename == __file__
        assert set(ztd.keys(True, True)) == {"a", ("b", "c")}

    def test_rename_key(self, ztd):
        ztd.rename_key_("a", "a2")
        assert "a2" in ztd.keys()
        assert "a" not in ztd.keys()
        with pytest.raises(KeyError, match="already present"):
            ztd.rename_key_("a2", ("b", "c"), safe=True)
        # safe=False replaces the existing entry, as in TensorDict
        a2 = ztd["a2"].clone()
        assert ztd.rename_key_("a2", ("b", "c")) is ztd
        assert set(ztd.keys(True, True)) == {("b", "c")}
        torch.testing.assert_close(ztd["b", "c"], a2)
        # an entry replaces the key that holds it, as in TensorDict: it raised
        # a KeyError
        ztd.rename_key_(("b", "c"), "b")
        assert set(ztd.keys(True, True)) == {"b"}
        torch.testing.assert_close(ztd["b"], a2)

    def test_rename_key_locked_warns(self, ztd):
        # a locked PersistentTensorDict renamed its keys without a check:
        # it still does, with a warning, until 0.17
        ztd.lock_()
        with pytest.warns(
            FutureWarning,
            match=r"rename_key_ renames keys of a locked tensordict today\. "
            r"Starting with TensorDict 0\.17",
        ) as record:
            assert ztd.rename_key_("a", "a2") is ztd
        assert len(record) == 1
        assert record[0].filename == __file__
        assert "a2" in ztd.keys()

    def test_rename_nested_group(self, ztd):
        ztd.rename_key_("b", "b2")
        assert ("b2", "c") in ztd.keys(True, True)

    def test_rename_key_inside_old_key(self, ztd):
        # a key moved into one of its own entries moves there, as in
        # TensorDict: a group raised RecursionError, and an array TypeError
        expected = ztd.to_tensordict()
        ztd.rename_key_("b", ("b", "b"))
        ztd.rename_key_("a", ("a", "x", "y"))
        assert set(ztd.keys(True, True)) == {("a", "x", "y"), ("b", "b", "c")}
        torch.testing.assert_close(ztd["a", "x", "y"], expected["a"])
        torch.testing.assert_close(ztd["b", "b", "c"], expected["b", "c"])

    def test_pop_nested(self, ztd):
        # https://github.com/pytorch/tensordict/issues/2089
        c = ztd["b", "c"].clone()
        out = ztd.pop("b")
        assert isinstance(out, TensorDict)
        assert set(out.keys()) == {"c"}
        assert (out["c"] == c).all()
        assert "b" not in ztd.keys()
        ztd["b"] = TensorDict(z=torch.zeros(3, 4), batch_size=[3, 4])
        assert set(out.keys()) == {"c"}

        ztd["d", "e", "f"] = torch.ones(3, 4)
        out = ztd.pop(("d", "e"))
        assert isinstance(out, TensorDict)
        assert (out["f"] == 1).all()
        assert ("d", "e") not in ztd.keys(True)

    def test_create_nested(self, ztd):
        ztd.create_nested("nested")
        assert "nested" in ztd.keys()
        ztd["nested", "x"] = torch.zeros(3, 4)
        assert (ztd["nested", "x"] == 0).all()

    def test_create_nested_existing(self, ztd):
        # A tuple key keeps the groups already on the path, and a string key
        # replaces the existing entry (#3008).
        ztd.create_nested(("b", "d"))
        assert set(ztd.keys(True)) == {"a", "b", ("b", "c"), ("b", "d")}
        # a group read before it is replaced is not served from the cache
        ztd["b", "d", "e"] = torch.zeros(3, 4)
        assert set(ztd["b", "d"].keys()) == {"e"}
        ztd.create_nested(("b", "d"))
        assert set(ztd["b", "d"].keys()) == set()
        ztd.create_nested("b")
        assert set(ztd.keys(True)) == {"a", "b"}

    def test_create_nested_read_only(self, ztd, tmp_path):
        # a read-only store raises the errors of zarr's create_group at the
        # first level of the key, as before #3008's fix
        ro = PersistentTensorDict.from_zarr(tmp_path / "s.zarr")
        for key in ("b", ("b", "c"), ("b", "d")):
            with pytest.raises(zarr.errors.ContainsGroupError):
                ro.create_nested(key)
        with pytest.raises(zarr.errors.ContainsArrayError):
            ro.create_nested(("a", "x"))
        with pytest.raises(ValueError, match="read-only"):
            ro.create_nested(("new", "y"))
        assert set(ro.keys(True)) == {"a", "b", ("b", "c")}

    @pytest.mark.parametrize("op", ["del_", "pop", "create_nested", "rename_key_"])
    def test_replace_group_through_child(self, tmp_path, op):
        # a group that a child deletes or replaces is not served from the
        # cache of the tensordicts above it (#3181)
        e = TensorDict(f=torch.zeros(3, 4), batch_size=[3, 4])
        td = TensorDict(
            n=TensorDict(d=TensorDict(e=e, batch_size=[3]), batch_size=[3]),
            batch_size=[3],
        )
        ztd = td.to_zarr(tmp_path / "t.zarr")
        # the root and its child cache their own copies of the group
        assert ztd["n", "d", "e"].batch_size == (3, 4)
        assert ztd["n"]["d", "e"].batch_size == (3, 4)
        child = ztd["n"]["d"]
        new = TensorDict(g=torch.ones(3), batch_size=[3])
        if op == "del_":
            child.del_("e")
            child.set("e", new)
        elif op == "pop":
            child.pop("e")
            child.set("e", new)
        elif op == "create_nested":
            child.create_nested("e")
            child.set(("e", "g"), torch.ones(3))
        else:
            child.set("h", new)
            child.rename_key_("h", "e", safe=False)
        for sub in (ztd["n", "d", "e"], ztd["n"]["d", "e"], child["e"]):
            assert set(sub.keys()) == {"g"}
            assert sub.batch_size == (3,)

    def test_kwargs_passthrough(self, tmp_path):
        td = TensorDict({"a": torch.randn(64, 64)}, batch_size=[64])
        td.to_zarr(tmp_path / "s.zarr", chunks=(16, 64))
        root = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
        assert root["a"].chunks == (16, 64)

    def test_kwargs_passthrough_nested(self, tmp_path):
        # create-kwargs must reach the leaves of nested tensordicts too
        # https://github.com/pytorch/tensordict/issues/1758
        from zarr.codecs import ZstdCodec

        td = TensorDict(
            {
                "a": torch.zeros(64, 3),
                "b": {"c": torch.zeros(64, 3), "d": {"e": torch.zeros(64, 3)}},
            },
            batch_size=[64],
        )
        td.to_zarr(tmp_path / "s.zarr", chunks=(16,), compressors=ZstdCodec(level=3))
        root = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
        for key in ("a", "b/c", "b/d/e"):
            assert root[key].chunks == (16, 3), key
            assert root[key].compressors, key
        td_back = TensorDict.from_zarr(tmp_path / "s.zarr")
        assert (td_back == td).all()

    def test_to_context_manager_writes_in_place(self, tmp_path, monkeypatch):
        # the write-back of to() reads nothing and keeps the layout of the
        # arrays, also for a store reopened without its creation arguments
        td = TensorDict(
            {"a": torch.arange(8.0), "b": {"c": torch.zeros(8)}}, batch_size=[8]
        )
        td.to_zarr(tmp_path / "s.zarr", chunks=(2,))
        ztd = PersistentTensorDict.from_zarr(tmp_path / "s.zarr", mode="r+")
        reads = []
        getitem = zarr.Array.__getitem__

        def counting_getitem(self, *args, **kwargs):
            reads.append(self.path)
            return getitem(self, *args, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with ztd.to(torch.float64) as td64:
                td64["a"] += 1
                td64["b", "d"] = torch.ones(8)
                monkeypatch.setattr(zarr.Array, "__getitem__", counting_getitem)
        monkeypatch.undo()
        assert reads == []
        root = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
        assert root["a"].chunks == (2,)
        assert root["b/c"].chunks == (2,)
        assert ztd["a"].dtype == torch.float32
        assert (ztd["a"] == torch.arange(8.0) + 1).all()
        assert (ztd["b", "d"] == 1).all()

    def test_to_context_manager_changes_entry_type(self, tmp_path):
        # as on a TensorDict, a group that the block replaces with a tensor is
        # replaced, and a tensor replaced with a tensordict raises
        td = TensorDict(
            {"a": torch.arange(8.0), "b": {"c": torch.zeros(8)}}, batch_size=[8]
        )
        td.to_zarr(tmp_path / "s.zarr")
        ztd = PersistentTensorDict.from_zarr(tmp_path / "s.zarr", mode="r+")
        with ztd.to(torch.float64) as td64:
            td64["b"] = torch.ones(8)
        assert set(ztd.keys(True, True)) == {"a", "b"}
        assert (ztd["b"] == 1).all()
        with pytest.raises(RuntimeError, match="cannot set a tensor collection"):
            with ztd.to(torch.float64) as td64:
                td64["a"] = TensorDict(c=torch.ones(8), batch_size=[8])
        assert set(ztd.keys(True, True)) == {"a", "b"}
        assert (ztd["a"] == torch.arange(8.0)).all()

    def test_chunks_heterogeneous_ranks(self, tmp_path):
        # a single chunks spec constrains the leading dims of every leaf,
        # whatever its rank, and leaves non-tensor payloads untouched
        td = TensorDict(
            {
                "images": torch.randn(128, 3, 8, 8),
                "labels": torch.randint(10, (128,)),
                "note": "some text",
            },
            batch_size=[128],
        )
        td.to_zarr(tmp_path / "s.zarr", chunks=(16,))
        root = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
        assert root["images"].chunks == (16, 3, 8, 8)
        assert root["labels"].chunks == (16,)
        td_back = TensorDict.from_zarr(tmp_path / "s.zarr")
        assert (td_back["images"] == td["images"]).all()
        assert td_back["note"] == "some text"

    def test_consolidated_metadata_written(self, tmp_path):
        td = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3)}}, [3])
        td.to_zarr(tmp_path / "s.zarr")
        root = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
        assert root.metadata.consolidated_metadata is not None

    def test_consolidated_metadata_stays_fresh(self, tmp_path):
        # structural mutations must not leave stale consolidated metadata behind
        td = TensorDict({"a": torch.randn(3)}, batch_size=[3])
        td.to_zarr(tmp_path / "s.zarr")
        ztd = PersistentTensorDict(
            filename=str(tmp_path / "s.zarr"), mode="r+", batch_size=[3], backend="zarr"
        )
        ztd["new"] = torch.rand(3)
        ztd.del_("a")
        reread = TensorDict.from_zarr(tmp_path / "s.zarr")
        assert "new" in reread.keys()
        assert "a" not in reread.keys()

    def test_default_layout(self, tmp_path):
        td = TensorDict({"a": torch.randn(8, 8)}, batch_size=[8])
        td.to_zarr(tmp_path / "s.zarr")
        root = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
        # single chunk, no compression by default
        assert root["a"].chunks == (8, 8)
        assert not root["a"].compressors

    def test_compression(self, tmp_path):
        from zarr.codecs import ZstdCodec

        td = TensorDict({"a": torch.zeros(1024, 1024)}, batch_size=[1024])
        td.to_zarr(tmp_path / "s.zarr", compressors=ZstdCodec())
        td_recon = TensorDict.from_zarr(tmp_path / "s.zarr")
        assert (td_recon["a"] == 0).all()
        root = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
        assert root["a"].compressors


@pytest.mark.skipif(not _has_zarr, reason="zarr>=3.0 not found.")
class TestZarrStores:
    @pytest.mark.parametrize("call", ["to_zarr", "from_zarr", "persistent"])
    def test_zarr_2_raises_import_error(self, tmp_path, monkeypatch, call):
        # #3298: every zarr path names the zarr version it needs; to_zarr
        # failed inside zarr 2 with a TypeError about use_consolidated
        import zarr

        td = TensorDict(a=torch.zeros(3), batch_size=[3])
        td.to_zarr(tmp_path / "a.zarr")
        open_group = zarr.open_group

        def zarr_2_open_group(*args, **kwargs):
            # zarr 2's open_group has no use_consolidated
            if "use_consolidated" in kwargs:
                raise TypeError(
                    "open_group() got an unexpected keyword argument 'use_consolidated'"
                )
            return open_group(*args, **kwargs)

        monkeypatch.setattr(zarr, "__version__", "2.18.3")
        monkeypatch.setattr(zarr, "open_group", zarr_2_open_group)
        calls = {
            "to_zarr": lambda: td.to_zarr(tmp_path / "b.zarr"),
            "from_zarr": lambda: TensorDict.from_zarr(tmp_path / "a.zarr"),
            "persistent": lambda: PersistentTensorDict(
                filename=str(tmp_path / "c.zarr"),
                backend="zarr",
                mode="w",
                batch_size=[3],
            ),
        }
        with pytest.raises(
            ImportError, match=r"requires zarr>=3\.1\.1, got zarr==2\.18\.3"
        ):
            calls[call]()

    @pytest.mark.parametrize(
        "version,ok",
        [
            ("3.0.8", False),
            ("3.1.0", False),
            ("3.1.1rc1", False),
            ("3.1.1.dev3", False),
            ("3.1.1", True),
            ("3.1.2.dev5+g1234abc", True),
            ("3.2.0", True),
            ("4.0.0", True),
        ],
    )
    def test_zarr_below_3_1_1_raises_import_error(
        self, tmp_path, monkeypatch, version, ok
    ):
        # #3352: the zarr extra needs zarr>=3.1.1 (3.0.0-3.0.7 are yanked,
        # 3.0.8-3.1.0 warn on every string array); the check accepted any zarr 3
        import zarr

        monkeypatch.setattr(zarr, "__version__", version)
        backend = tensordict_persistent._ZarrBackend()
        if ok:
            backend.check_available()
            return
        with pytest.raises(
            ImportError,
            match=rf"requires zarr>=3\.1\.1, got zarr=={re.escape(version)}\. "
            r"Upgrade with `pip install 'zarr>=3\.1\.1'`",
        ):
            backend.check_available()
        with pytest.raises(ImportError, match=r"requires zarr>=3\.1\.1"):
            PersistentTensorDict(
                filename=str(tmp_path / "a.zarr"),
                backend="zarr",
                mode="w",
                batch_size=[3],
            )

    def test_explicit_store(self, tmp_path):
        from zarr.storage import LocalStore

        td = TensorDict({"a": torch.randn(3)}, batch_size=[3])
        store = LocalStore(str(tmp_path / "s.zarr"))
        td.to_zarr(store)
        td_recon = TensorDict.from_zarr(
            LocalStore(str(tmp_path / "s.zarr"), read_only=True)
        )
        assert (td_recon["a"] == td["a"]).all()

    def test_zip_store(self, tmp_path):
        from zarr.storage import ZipStore

        td = TensorDict(
            {"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}, batch_size=[3]
        )
        store = ZipStore(str(tmp_path / "s.zarr.zip"), mode="w")
        ztd = td.to_zarr(store)
        ztd.close()

        store_read = ZipStore(str(tmp_path / "s.zarr.zip"), mode="r")
        td_recon = TensorDict.from_zarr(store_read)
        assert td_recon.batch_size == torch.Size([3])
        assert (td_recon["a"] == td["a"]).all()
        assert (td_recon["b", "c"] == td["b", "c"]).all()
        td_recon.close()

    def test_zip_store_nontensor_written_once(self, tmp_path):
        # a zip entry cannot be replaced: the non-tensor marker must be
        # written with the array, not in a second zarr.json entry
        import zipfile

        from zarr.storage import ZipStore

        path = tmp_path / "s.zarr.zip"
        td = TensorDict({"a": torch.randn(3), "s": "a string"}, batch_size=[3])
        store = ZipStore(str(path), mode="w")
        td.to_zarr(store).close()

        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        assert names.count("s/zarr.json") == 1
        td_recon = TensorDict.from_zarr(ZipStore(str(path), mode="r"))
        assert td_recon["s"] == "a string"
        td_recon.close()

    @pytest.mark.parametrize("named", [False, True])
    def test_zip_store_groups_written_once(self, tmp_path, named):
        # each group's zarr.json is written with its attributes, once
        import zipfile

        from zarr.storage import ZipStore

        path = tmp_path / "s.zarr.zip"
        td = TensorDict(
            {"a": torch.randn(4, 3), "b": {"c": {"d": torch.randn(4, 2, 5)}}},
            batch_size=[4],
        )
        td["b"].batch_size = [4, 2]
        if named:
            td.names = ["t"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            store = ZipStore(str(path), mode="w")
            td.to_zarr(store).close()

        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        assert len(names) == len(set(names))
        td_recon = TensorDict.from_zarr(ZipStore(str(path), mode="r"))
        assert td_recon.batch_size == torch.Size([4])
        assert td_recon["b"].batch_size == torch.Size([4, 2])
        assert td_recon["b", "c"].batch_size == torch.Size([4, 2])
        assert td_recon.names == (["t"] if named else [None])
        assert (td_recon["b", "c", "d"] == td["b", "c", "d"]).all()
        td_recon.close()

    def test_zip_store_from_dict_named_groups_written_once(self, tmp_path):
        # the nested groups of a named source keep their names, so that
        # from_dict writes each group's zarr.json once (#2252)
        import zipfile

        from zarr.storage import ZipStore

        path = tmp_path / "s.zarr.zip"
        td = TensorDict(
            {
                "meta": {"m": torch.randn(4, 3)},
                "b": {"c": {"d": torch.randn(4, 2, 5)}},
                "e": TensorDict({}, [4, 3]),
            },
            batch_size=[4],
            names=["t"],
        )
        # no warnings filter: the metadata rewrite is best effort and would
        # swallow the duplicate-entry warning raised as an error
        store = ZipStore(str(path), mode="w")
        ptd = PersistentTensorDict.from_dict(td, store, backend="zarr")
        assert ptd["meta"].names == ["t"]
        assert ptd["b"]["c"].names == ["t"]
        ptd["s"] = TensorDict(
            {"y": torch.zeros(4, 2)}, batch_size=[4, 2], names=["t", "u"]
        )
        assert ptd["s"].names == ["t", "u"]
        ptd.close()

        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        assert len(names) == len(set(names))
        td_recon = TensorDict.from_zarr(ZipStore(str(path), mode="r"))
        assert td_recon["meta"].names == ["t"]
        assert (td_recon["b", "c", "d"] == td["b", "c", "d"]).all()
        td_recon.close()

    def test_rename_clashing_with_nested_names_warns(self, tmp_path):
        # a new name that clashes with a kept nested name leaves that nested
        # dim unnamed, as before #2252, and warns that it will raise
        sub = TensorDict({"y": torch.zeros(4, 2)}, [4, 2], names=["t", "x"])
        td = TensorDict({"s": sub}, [4], names=["t"])
        with pytest.warns(FutureWarning, match="TensorDict 0.17"):
            ptd = PersistentTensorDict.from_dict(
                td, tmp_path / "a.zarr", backend="zarr", names=["x"]
            )
        assert ptd.names == ["x"]
        assert ptd["s"].names == ["x", None]
        ptd.close()
        ptd = PersistentTensorDict.from_dict(td, tmp_path / "b.zarr", backend="zarr")
        with pytest.warns(FutureWarning, match="TensorDict 0.17"):
            ptd.names = ["x"]
        assert ptd["s"].names == ["x", None]
        ptd.close()

    def test_group_kwarg(self, tmp_path):
        td = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3)}}, [3])
        td.to_zarr(tmp_path / "s.zarr")
        root = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
        sub = PersistentTensorDict(group=root["b"], backend="zarr", batch_size=[3])
        assert (torch.as_tensor(sub["c"]) == td["b", "c"]).all()

    def test_from_schema(self, tmp_path):
        td = TensorDict.from_schema(
            {"obs": ([4], torch.float32), "reward": ([], torch.int64)},
            batch_size=[8],
            storage="zarr",
            filename=str(tmp_path / "s.zarr"),
        )
        assert isinstance(td, PersistentTensorDict)
        assert td["obs"].shape == torch.Size([8, 4])
        assert (td["obs"] == 0).all()
        td[0] = TensorDict(
            {"obs": torch.ones(4), "reward": torch.ones((), dtype=torch.int64)}, []
        )
        assert (td["obs"][0] == 1).all()
        assert (td["obs"][1:] == 0).all()

    def test_from_schema_unknown_storage_lists_zarr(self):
        with pytest.raises(ValueError, match="zarr"):
            TensorDict.from_schema({"a": ([1], torch.float32)}, storage="nope")


@pytest.mark.skipif(not _has_zarr, reason="zarr>=3.0 not found.")
class TestZarrSerialization:
    @classmethod
    def worker(cls, ztd, q1, q2):
        assert isinstance(ztd, PersistentTensorDict)
        q1.put(ztd["base", "sub"])
        assert q2.get(timeout=TIMEOUT) == "checked"
        val = ztd["base", "sub", "default"] + 1
        q1.put(val)
        assert q2.get(timeout=TIMEOUT) == "checked"
        q1.close()
        q2.close()

    def test_zarr_multiprocessing(self, tmp_path):
        arr = np.random.randn(1000)
        td = TensorDict(
            {("base", "sub", "default"): torch.as_tensor(arr)}, batch_size=[]
        )
        td.to_zarr(tmp_path / "groups.zarr")
        persistent_td = PersistentTensorDict(
            filename=str(tmp_path / "groups.zarr"), batch_size=[], backend="zarr"
        )
        q1 = mp.Queue(1)
        q2 = mp.Queue(1)
        p = mp.Process(target=self.worker, args=(persistent_td, q1, q2))
        p.start()
        try:
            val = q1.get(timeout=TIMEOUT)
            assert (torch.tensor(arr) == val["default"]).all()
            q2.put("checked")
            val = q1.get(timeout=TIMEOUT)
            assert (torch.tensor(arr) + 1 == val).all()
            q2.put("checked")
            q1.close()
            q2.close()
        finally:
            p.join()

    def test_pickle_roundtrip(self, tmp_path):
        td = TensorDict(
            {"a": torch.randn(3, 4), "b": {"c": torch.randn(3, 4, 5)}},
            batch_size=[3, 4],
        )
        ztd = td.to_zarr(tmp_path / "s.zarr")
        ztd2 = pickle.loads(pickle.dumps(ztd))
        assert ztd2.batch_size == ztd.batch_size
        assert (ztd2["a"] == ztd["a"]).all()
        assert (ztd2["b", "c"] == ztd["b", "c"]).all()

    def test_pickle_locked_after_nested_access(self, tmp_path):
        # gh-2920: locking after reading a nested view left a weakref in the
        # pickled state
        td = TensorDict(
            {"a": torch.randn(3), ("b", "c"): torch.randn(3), ("b", "s"): "text"},
            batch_size=[3],
        )
        ztd = td.to_zarr(tmp_path / "s.zarr")
        ztd["b"]
        ztd.lock_()
        ztd2 = pickle.loads(pickle.dumps(ztd))
        assert ztd2.is_locked
        assert ztd2["b"].is_locked
        assert (ztd2["a"] == ztd["a"]).all()
        assert (ztd2["b", "c"] == ztd["b", "c"]).all()
        assert ztd2["b", "s"] == "text"

    @staticmethod
    def _loader_data(tmp_path):
        td = TensorDict(
            {
                "x": torch.randn(300, 4),
                ("nested", "y"): torch.arange(300),
                "s": "text",
            },
            batch_size=[300],
        )
        return td.to_zarr(tmp_path / "s.zarr")

    # gh-3218: the workers sent a lazy view of the store, so the main process
    # read every batch
    @pytest.mark.timeout(120)
    def test_dataloader_workers_read_rows(self, tmp_path, monkeypatch):
        ztd = self._loader_data(tmp_path)
        reads = []
        read_array_at = PersistentTensorDict._read_array_at

        def logged_read_array_at(self, array, idx):
            reads.append(array.name)
            return read_array_at(self, array, idx)

        monkeypatch.setattr(
            PersistentTensorDict, "_read_array_at", logged_read_array_at
        )
        batch_sampler = BatchSampler(SequentialSampler(range(300)), 64, False)
        loader = DataLoader(  # noqa: TOR401
            ztd, batch_size=64, num_workers=2, collate_fn=_keep_batch
        )
        batches = list(loader)
        assert len(batches) == 5
        for idx, batch in zip(batch_sampler, batches):
            assert type(batch) is TensorDict
            assert type(batch["nested"]) is TensorDict
            expected = ztd[idx].to_tensordict()
            assert batch.batch_size == expected.batch_size
            assert (batch["x"] == expected["x"]).all()
            assert (batch["nested", "y"] == expected["nested", "y"]).all()
            assert batch["s"] == expected["s"]
        # the workers read the rows, the main process none
        reads.clear()
        for batch in batches:
            batch["x"], batch["nested", "y"]
        assert reads == []

    # gh-3218: in a worker, the batch pickles with its rows, not as a view
    def test_getitems_in_worker(self, tmp_path, monkeypatch):
        ztd = self._loader_data(tmp_path)
        monkeypatch.setattr(
            tensordict_persistent, "get_worker_info", object, raising=False
        )
        batch = pickle.loads(pickle.dumps(ztd.__getitems__([3, 1, 4])))
        assert type(batch) is TensorDict
        assert (batch["x"] == ztd["x"][[3, 1, 4]]).all()
        # a key still gives the entry
        assert (ztd.__getitems__("x") == ztd["x"]).all()
        assert (ztd.__getitems__(("nested", "y")) == ztd["nested", "y"]).all()

    @pytest.mark.timeout(120)
    def test_dataloader_no_workers_gives_view(self, tmp_path):
        ztd = self._loader_data(tmp_path)
        batch = next(iter(DataLoader(ztd, batch_size=64)))  # noqa: TOR401
        assert type(batch) is _SubTensorDict
        # a write to the batch reaches the store
        batch["x"] = torch.zeros(64, 4)
        assert (ztd["x"][:64] == 0).all()
        assert (ztd["x"][64:] != 0).any()


@pytest.mark.skipif(not _has_zarr, reason="zarr>=3.0 not found.")
class TestZarrInterop:
    def test_to_tensordict(self, tmp_path):
        td = TensorDict(
            {"a": torch.randn(3, 4), "b": {"c": torch.randn(3, 4, 5)}},
            batch_size=[3, 4],
        )
        ztd = td.to_zarr(tmp_path / "s.zarr")
        td2 = ztd.to_tensordict()
        assert (td2 == td).all()

    def test_memmap(self, tmp_path):
        td = TensorDict(
            {"a": torch.randn(3, 4), "b": {"c": torch.randn(3, 4, 5)}},
            batch_size=[3, 4],
        )
        ztd = td.to_zarr(tmp_path / "s.zarr")
        mm = ztd.memmap(tmp_path / "memmap")
        assert (mm["a"] == td["a"]).all()
        mm_loaded = TensorDict.load_memmap(tmp_path / "memmap")
        assert (mm_loaded["a"] == td["a"]).all()

    def test_clone(self, tmp_path):
        td = TensorDict(
            {"a": torch.randn(3, 4), "b": {"c": torch.randn(3, 4, 5)}, "d": "a string"},
            batch_size=[3, 4],
        )
        ztd = td.to_zarr(tmp_path / "s.zarr")
        clone = ztd._clone(recurse=True, newfile=str(tmp_path / "clone.zarr"))
        assert (clone["a"] == td["a"]).all()
        assert clone["d"] == "a string"
        # the cloned store is loadable on its own with the right batch size
        reload = TensorDict.from_zarr(tmp_path / "clone.zarr")
        assert reload.batch_size == torch.Size([3, 4])

    def test_device_kwarg(self, tmp_path):
        td = TensorDict({"a": torch.randn(3)}, batch_size=[3])
        td.to_zarr(tmp_path / "s.zarr")
        ztd = PersistentTensorDict(
            filename=str(tmp_path / "s.zarr"),
            backend="zarr",
            batch_size=[3],
            device="cpu",
        )
        assert ztd.device == torch.device("cpu")
        assert ztd["a"].device == torch.device("cpu")

    def test_tensorclass(self, tmp_path):
        from tensordict import tensorclass

        @tensorclass
        class MyData:
            a: torch.Tensor

        data = MyData(a=torch.randn(3), batch_size=[3])
        out = data.to_zarr(tmp_path / "s.zarr")
        assert type(out) is MyData
        assert (out.a == data.a).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
