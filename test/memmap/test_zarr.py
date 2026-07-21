# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import NonTensorData, PersistentTensorDict, TensorDict
from tensordict.base import _is_leaf_nontensor
from tensordict.utils import is_non_tensor, NUMPY_TO_TORCH_DTYPE_DICT
from torch import multiprocessing as mp

TIMEOUT = 100


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
    for np_dtype, torch_dtype in NUMPY_TO_TORCH_DTYPE_DICT.items()
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

    def test_free_function(self, tmp_path):
        from tensordict import from_zarr

        td = TensorDict({"a": torch.randn(3)}, batch_size=[3])
        td.to_zarr(tmp_path / "s.zarr")
        td_recon = from_zarr(tmp_path / "s.zarr")
        assert (td_recon["a"] == td["a"]).all()


@pytest.mark.skipif(not _has_zarr, reason="zarr>=3.0 not found.")
class TestZarrNonTensor:
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

    def test_masked_fill_(self, ztd):
        ztd.zero_()
        ztd.masked_fill_(torch.tensor([True, False, False]), 3.0)
        assert (ztd["a"][0] == 3).all()
        assert (ztd["a"][1:] == 0).all()

    def test_del(self, ztd):
        ztd.del_("a")
        assert "a" not in ztd.keys()

    def test_rename_key(self, ztd):
        ztd.rename_key_("a", "a2")
        assert "a2" in ztd.keys()
        assert "a" not in ztd.keys()
        with pytest.raises(KeyError, match="already present"):
            ztd.rename_key_("a2", ("b", "c"))

    def test_rename_nested_group(self, ztd):
        ztd.rename_key_("b", "b2")
        assert ("b2", "c") in ztd.keys(True, True)

    def test_create_nested(self, ztd):
        ztd.create_nested("nested")
        assert "nested" in ztd.keys()
        ztd["nested", "x"] = torch.zeros(3, 4)
        assert (ztd["nested", "x"] == 0).all()

    def test_kwargs_passthrough(self, tmp_path):
        td = TensorDict({"a": torch.randn(64, 64)}, batch_size=[64])
        td.to_zarr(tmp_path / "s.zarr", chunks=(16, 64))
        root = zarr.open_group(str(tmp_path / "s.zarr"), mode="r")
        assert root["a"].chunks == (16, 64)

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
