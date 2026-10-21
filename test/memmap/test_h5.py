# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import itertools
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
import tensordict.persistent as tensordict_persistent
import torch
from tensordict import (
    lazy_stack,
    MemoryMappedTensor,
    NonTensorData,
    NonTensorStack,
    PersistentTensorDict,
    TensorClass,
    TensorDict,
)
from tensordict.base import _is_leaf_nontensor
from tensordict.utils import is_non_tensor
from torch import multiprocessing as mp
from torch.utils._pytree import tree_map

TIMEOUT = 100

try:
    import h5py

    _has_h5py = True
except ImportError:
    _has_h5py = False


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
class TestH5Serialization:
    @pytest.mark.parametrize("allow_pickle", [None, True, False])
    def test_zarr_pickle_policy_does_not_change_h5(self, tmp_path, allow_pickle):
        path = tmp_path / "policy.h5"
        with h5py.File(path, "w") as file:
            file.create_dataset("x", data=np.arange(2))
            file.create_dataset("text", data=b"text")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            td = PersistentTensorDict(filename=path, allow_pickle=allow_pickle)
            assert td["text"] == b"text"
            assert torch.equal(td["x"], torch.arange(2))
            clone = td.clone(recurse=False)
            assert clone["text"] == b"text"
            clone.close()
            td.close()

    @classmethod
    def worker(cls, cyberbliptronics, q1, q2):
        assert isinstance(cyberbliptronics, PersistentTensorDict)
        assert cyberbliptronics.file.filename.endswith("groups.hdf5")
        q1.put(cyberbliptronics["Base_Group"]["Sub_Group"])
        assert q2.get(timeout=TIMEOUT) == "checked"
        val = cyberbliptronics["Base_Group", "Sub_Group", "default"] + 1
        q1.put(val)
        assert q2.get(timeout=TIMEOUT) == "checked"
        q1.close()
        q2.close()

    def test_h5_serialization(self, tmp_path):
        arr = np.random.randn(1000)
        fn = tmp_path / "groups.hdf5"
        with h5py.File(fn, "w") as f:
            g = f.create_group("Base_Group")
            gg = g.create_group("Sub_Group")

            _ = g.create_dataset("default", data=arr)
            _ = gg.create_dataset("default", data=arr)

        persistent_td = PersistentTensorDict(filename=fn, batch_size=[])
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

    def test_h5_nontensor(self, tmpdir):
        file = Path(tmpdir) / "file.h5"
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
        h5td = PersistentTensorDict.from_dict(td, filename=file)
        assert "c" in h5td.keys(is_leaf=_is_leaf_nontensor)
        assert "c" in h5td.keys()
        assert "c" in h5td
        assert h5td["c"] == b"a string!"
        assert h5td.get("c").batch_size == (10,)
        assert ("d", "e") in h5td.keys(True, True, is_leaf=_is_leaf_nontensor)
        assert ("d", "e") in h5td
        assert h5td["d", "e"] == b"another string!"
        assert h5td.get(("d", "e")).batch_size == (10,)

        h5td.set("f", NonTensorData(1, batch_size=[10]))
        assert h5td["f"] == 1
        h5td.set(("g", "h"), NonTensorData(1, batch_size=[10]))
        assert h5td["g", "h"] == 1

        td_recover = h5td.to_tensordict()
        assert is_non_tensor(td_recover.get("c"))
        assert is_non_tensor(td_recover.get(("d", "e")))
        assert is_non_tensor(td_recover.get("f"))
        assert is_non_tensor(td_recover.get(("g", "h")))

    def test_h5_str_read_as_bytes_warns(self, tmp_path):
        # strings that tensordict writes are marked; reading one still gives
        # bytes and warns that it will give str
        td = TensorDict(
            {
                "label": "cat",
                "a": torch.zeros(3),
                ("n", "s"): "dog",
                "i": NonTensorData(1, batch_size=[3]),
            },
            [3],
        )
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        match = (
            r"Starting with TensorDict 0\.17, the default will change to "
            r"reading it as str"
        )
        for call, expected in (
            (lambda: h5td["label"], b"cat"),
            (lambda: h5td.get("label").data, b"cat"),
            (lambda: h5td["n", "s"], b"dog"),
            (lambda: h5td.to_tensordict()["label"], b"cat"),
            (lambda: h5td.clone(newfile=tmp_path / "copy.h5")["label"], b"cat"),
        ):
            with pytest.warns(FutureWarning, match=match) as record:
                assert call() == expected
            # one warning per string read, which points to the caller
            assert {r.filename for r in record} == {__file__}
        # other entries do not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (h5td["a"] == 0).all()
            assert h5td["i"] == 1
            h5td["label"] = "pig"
        with pytest.warns(FutureWarning, match=match):
            assert h5td["label"] == b"pig"
        h5td.close()

        # strings that another writer stored stay bytes, without a warning
        with h5py.File(tmp_path / "other.h5", "w") as f:
            f["s"] = "dog"
            f["x"] = np.zeros(3)
        h5td = PersistentTensorDict.from_h5(tmp_path / "other.h5")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert h5td["s"] == b"dog"
            assert h5td.to_tensordict()["s"] == b"dog"
        h5td.close()

    def test_h5_nontensor_stack(self, tmp_path):
        # one value per row is written as a dataset of the stack's shape and
        # read back as a NonTensorStack (#2544)
        grid = NonTensorStack(*(NonTensorStack(f"{i}0", f"{i}1") for i in range(3)))
        td = TensorDict(
            {
                "x": torch.zeros(3, 2),
                "text": NonTensorStack("a", "b", "c"),
                ("n", "y"): torch.zeros(3, 2),
                ("n", "grid"): grid,
                ("info", "mission"): NonTensorStack("u", "v", "w"),
                "ints": NonTensorStack(1, 2, 3),
                "bytes": NonTensorStack(b"a", b"bb", b"c"),
            },
            [3],
        )
        PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5").close()
        with h5py.File(tmp_path / "file.h5") as f:
            assert h5py.check_string_dtype(f["text"].dtype)
            assert f["text"].shape == (3,)
            assert f["n/grid"].shape == (3, 2)
        # the batch size is inferred from the stacks too
        h5td = PersistentTensorDict.from_h5(tmp_path / "file.h5")
        assert h5td.batch_size == (3,)
        assert h5td["info"].batch_size == (3,)
        with warnings.catch_warnings():
            # the strings of a stack are read as str
            warnings.simplefilter("error")
            assert h5td.entry_class("text") is NonTensorStack
            text = h5td.get("text")
            assert isinstance(text, NonTensorStack)
            assert text.batch_size == (3,)
            assert text.tolist() == ["a", "b", "c"]
            assert h5td[1]["text"] == "b"
            assert h5td.get(("n", "grid")).batch_size == (3, 2)
            assert h5td.get(("n", "grid")).tolist() == grid.tolist()
            assert h5td[2]["n", "grid"] == ["20", "21"]
            assert h5td.get("ints").tolist() == [1, 2, 3]
            assert h5td.get("bytes").tolist() == [b"a", b"bb", b"c"]
            assert h5td.to_tensordict()["text"] == ["a", "b", "c"]
            assert h5td.to_tensordict()["info", "mission"] == ["u", "v", "w"]
            copy = h5td.clone(newfile=tmp_path / "copy.h5")
            assert copy.get("text").tolist() == ["a", "b", "c"]
        # a stack is written whole: an in-place write raises, a whole one works
        with pytest.raises(RuntimeError, match="in place"):
            copy.set_at_("text", "q", 0)
        assert copy.get("text").tolist() == ["a", "b", "c"]
        copy["text"] = NonTensorStack("x", "yy", "z")
        assert copy.get("text").tolist() == ["x", "yy", "z"]
        # values that h5 cannot keep exactly raise
        with pytest.raises(TypeError, match="single type"):
            copy.set("mixed", NonTensorStack("a", 1, "c"))
        with pytest.raises(ValueError, match="NUL"):
            copy.set("nul", NonTensorStack("a\x00b", "c", "d"))
        assert "nul" not in copy.keys()
        copy.close()
        h5td.close()

    def test_h5_foreign_column_as_stack(self, tmp_path):
        # a string column that another tool wrote, one string per row, reads
        # as a NonTensorStack of its rows with non_tensor_as_stack=True (#2545)
        with h5py.File(tmp_path / "file.h5", "w") as f:
            f["pos"] = np.zeros(4, dtype=np.float32)
            strings = ["go left", "go right", "stop", "wait"]
            f.create_dataset("mission", data=strings, dtype=h5py.string_dtype())
            f.create_dataset("name", data="episode", dtype=h5py.string_dtype())
            f["info/x"] = np.zeros((4, 2), dtype=np.float32)
            f["info/s"] = np.array([b"a", b"b", b"c", b"d"])
            f["obs/y"] = np.zeros(4, dtype=np.float32)
            f["obs/s"] = np.array([b"a", b"b", b"c", b"d"])
            # how tensordict writes a NonTensorData of a list
            f["l"] = np.arange(4).astype(h5py.opaque_dtype(np.dtype("int64")))
        h5td = PersistentTensorDict.from_h5(tmp_path / "file.h5")
        # the default is unchanged: one value holds the whole column
        assert h5td.entry_class("mission") is NonTensorData
        assert h5td[1:3].get("mission").data.tolist() == [s.encode() for s in strings]
        h5td.close()

        h5td = PersistentTensorDict.from_h5(
            tmp_path / "file.h5", non_tensor_as_stack=True
        )
        assert h5td.batch_size == (4,)
        assert h5td.entry_class("mission") is NonTensorStack
        mission = h5td.get("mission")
        assert isinstance(mission, NonTensorStack)
        assert mission.batch_size == (4,)
        # decoded as the NonTensorData would be: bytes
        assert mission.tolist() == [b"go left", b"go right", b"stop", b"wait"]
        assert h5td[1:3].get("mission").tolist() == [b"go right", b"stop"]
        assert h5td[2]["mission"] == b"stop"
        assert h5td.get_at("mission", 1).data == b"go right"
        # a column that does not have the rows of the batch stays one value
        assert h5td.entry_class("name") is NonTensorData
        assert h5td["name"] == b"episode"
        # a value that tensordict wrote stays one value
        assert h5td.entry_class("l") is NonTensorData
        assert h5td[1]["l"].tolist() == [0, 1, 2, 3]
        # the rows of a nested column are those of its own tensordict
        assert h5td["info"].batch_size == (4, 2)
        assert h5td.entry_class(("info", "s")) is NonTensorData
        assert isinstance(h5td.get(("info", "s")), NonTensorData)
        assert h5td.entry_class(("obs", "s")) is NonTensorStack
        assert h5td["obs"][1]["s"] == b"b"
        assert h5td[1]["obs", "s"] == b"b"
        assert h5td.to_tensordict()[1:3]["mission"] == [b"go right", b"stop"]
        assert h5td.clone(False)[3]["mission"] == b"wait"
        h5td.close()

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
    def test_h5_foreign_column_as_stack_reads_rows_at(self, tmp_path, idx, monkeypatch):
        # an indexed read of a column read as rows builds only the indexed
        # rows, with the result of indexing the whole column (#2814)
        with h5py.File(tmp_path / "file.h5", "w") as f:
            f["pos"] = np.zeros(6, dtype=np.float32)
            strings = [f"mission {i}" for i in range(6)]
            f.create_dataset("mission", data=strings, dtype=h5py.string_dtype())
            f["obs/y"] = np.zeros(6, dtype=np.float32)
            f["obs/s"] = np.array([f"s{i}".encode() for i in range(6)])
        h5td = PersistentTensorDict.from_h5(
            tmp_path / "file.h5", non_tensor_as_stack=True
        )
        expected = {key: h5td.get(key)[idx] for key in ("mission", ("obs", "s"))}
        num_rows = torch.zeros(6)[idx].numel()
        from_list = NonTensorStack._from_list.__func__
        sizes = []

        def spy(cls, datalist, *args, **kwargs):
            sizes.append(len(datalist))
            return from_list(cls, datalist, *args, **kwargs)

        monkeypatch.setattr(NonTensorStack, "_from_list", classmethod(spy))
        for key, value in expected.items():
            for out in (h5td.get_at(key, idx), h5td[idx].get(key)):
                assert type(out) is type(value)
                assert out.batch_size == value.batch_size
                assert out.tolist() == value.tolist()
        assert max(sizes, default=0) <= num_rows
        h5td.close()

    def test_h5_foreign_column_as_stack_locked_get_at_cached(self, tmp_path):
        # a locked td indexes the cached stack of the whole column, so
        # repeated reads return its members (#2814)
        with h5py.File(tmp_path / "file.h5", "w") as f:
            f["pos"] = np.zeros(6, dtype=np.float32)
            strings = [f"mission {i}" for i in range(6)]
            f.create_dataset("mission", data=strings, dtype=h5py.string_dtype())
        h5td = PersistentTensorDict.from_h5(
            tmp_path / "file.h5", non_tensor_as_stack=True
        )
        h5td.lock_()
        out = h5td.get_at("mission", 1)
        assert out is h5td.get_at("mission", 1)
        assert out.tolist() == h5td.get("mission")[1].tolist()
        h5td.unlock_()
        h5td.close()

    @pytest.mark.parametrize("non_tensor_as_stack", [False, True])
    @pytest.mark.parametrize("with_prefix", [False, True])
    def test_h5_memmap_enum(self, tmp_path, non_tensor_as_stack, with_prefix):
        # an enum column reads as a non-tensor: memmap() keeps it as one,
        # as to_tensordict().memmap() does (#2781)
        enum = h5py.enum_dtype({"A": 0, "B": 1}, basetype="i1")
        with h5py.File(tmp_path / "file.h5", "w") as f:
            f["pos"] = np.arange(4, dtype=np.float32)
            f.create_dataset("color", data=np.array([0, 1, 1, 0]), dtype=enum)
            f.create_dataset("sub/e", data=np.array([1, 0, 0, 1]), dtype=enum)
        h5td = PersistentTensorDict.from_h5(
            tmp_path / "file.h5", non_tensor_as_stack=non_tensor_as_stack
        )
        expected = h5td.to_tensordict().memmap()
        prefix = tmp_path / "memmap" if with_prefix else None
        out = h5td.memmap(prefix)
        h5td.close()
        assert isinstance(out.get("pos"), MemoryMappedTensor)
        assert (out.get("pos") == torch.arange(4)).all()
        cls = NonTensorStack if non_tensor_as_stack else NonTensorData
        for key in ("color", ("sub", "e")):
            assert type(out.get(key)) is cls
            assert type(expected.get(key)) is cls
            assert np.array_equal(out.get(key).tolist(), expected.get(key).tolist())

    def test_h5_memmap_non_tensor_list(self, tmp_path):
        # tensordict writes a NonTensorData that is not a str as an opaque
        # dataset, whose metadata also has a torch dtype (#2781)
        td = TensorDict({"a": torch.zeros(3), "l": NonTensorData([1, 2, 3])}, [3])
        PersistentTensorDict.from_dict(td, tmp_path / "file.h5").close()
        h5td = PersistentTensorDict.from_h5(tmp_path / "file.h5")
        expected = h5td.to_tensordict().memmap()
        out = h5td.memmap(tmp_path / "memmap")
        h5td.close()
        assert isinstance(out.get("a"), MemoryMappedTensor)
        assert type(out.get("l")) is NonTensorData
        assert type(expected.get("l")) is NonTensorData
        assert np.array_equal(out.get("l").data, expected.get("l").data)
        assert np.array_equal(out.get("l").data, [1, 2, 3])

    def test_rename_key_missing(self, tmp_path):
        # the error names the missing source key, not the free destination
        td = TensorDict({"a": torch.zeros(3), ("b", "c"): torch.zeros(3)}, [3])
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        with pytest.raises(KeyError, match="key nope not found"):
            h5td.rename_key_("nope", "x")
        with pytest.raises(KeyError, match="key b/nope not found"):
            h5td.rename_key_(("b", "nope"), "x")
        with pytest.raises(KeyError, match="key a already present"):
            h5td.rename_key_("nope", "a")
        assert set(h5td.keys(True, True)) == {"a", ("b", "c")}
        h5td.close()

    def test_rename_key_overwrite_read_group(self, tmp_path):
        # the overwritten group was read first: reads and writes after the
        # rename go to the new node, not to the deleted group
        td = TensorDict({"a": {"x": torch.zeros(3)}, "b": {"y": torch.ones(3)}}, [3])
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        _ = h5td["b"], h5td["a"]
        h5td.rename_key_("a", "b")
        assert set(h5td["b"].keys()) == {"x"}
        h5td["b", "z"] = torch.full((3,), 2.0)
        assert set(h5td.keys(True, True)) == {("b", "x"), ("b", "z")}
        # a new "a" is not the moved group
        h5td["a"] = TensorDict(w=torch.zeros(3), batch_size=[3])
        assert set(h5td.keys(True, True)) == {("a", "w"), ("b", "x"), ("b", "z")}
        h5td.close()

    def test_rename_key_into_own_child(self, tmp_path):
        # an entry replaces the key that holds it, as in TensorDict: the file
        # raised a KeyError
        td = TensorDict(
            {
                "n": {"sub": {"x": torch.full((3,), 2.0)}, "c": torch.ones(3)},
                "n_": torch.ones(3),
            },
            [3],
        )
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        with pytest.raises(KeyError, match="already present"):
            h5td.rename_key_(("n", "sub"), "n", safe=True)
        h5td.rename_key_(("n", "sub"), "n")
        assert set(h5td.keys(True, True)) == {("n", "x"), "n_"}
        h5td.rename_key_(("n", "x"), "n")
        assert set(h5td.keys(True, True)) == {"n", "n_"}
        assert (h5td["n"] == 2).all()
        assert (h5td["n_"] == 1).all()
        del h5td["n"]
        h5td["n"] = TensorDict(sub=torch.zeros(3), c=torch.ones(3), batch_size=[3])
        # a key moved into one of its own entries moves there, as in
        # TensorDict: the file raised, or lost the key
        h5td.rename_key_("n", ("n", "sub"))
        assert set(h5td.keys(True, True)) == {
            ("n", "sub", "sub"),
            ("n", "sub", "c"),
            "n_",
        }
        assert (h5td["n", "sub", "sub"] == 0).all()
        assert (h5td["n", "sub", "c"] == 1).all()
        h5td.rename_key_(("n", "sub", "c"), ("n", "sub", "c", "d", "e"))
        assert (h5td["n", "sub", "c", "d", "e"] == 1).all()
        h5td.close()

    def test_del_read_group_then_set(self, tmp_path):
        # a sub-tensordict read before del_ must not catch the new entry at its key
        td = TensorDict(
            a=torch.zeros(3),
            nested=TensorDict(d=torch.ones(3), batch_size=[3]),
            batch_size=[3],
        )
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        h5td.get("nested")
        h5td.del_("nested")
        h5td.set("nested", TensorDict(z=torch.zeros(3), batch_size=[3]))
        assert set(h5td.keys(True, True)) == {"a", ("nested", "z")}
        assert set(h5td["nested"].keys()) == {"z"}
        # the same one level down
        h5td.get("nested")
        h5td.del_(("nested", "z"))
        h5td.set(("nested", "w"), torch.ones(3))
        assert set(h5td.keys(True, True)) == {"a", ("nested", "w")}
        h5td.close()

    def test_create_nested_existing(self, tmp_path):
        # A string key replaces the existing entry. A tuple key keeps the
        # groups already on the path, as in TensorDict (#3008).
        td = TensorDict(
            a=torch.zeros(3),
            nested=TensorDict(d=torch.ones(3), batch_size=[3]),
            batch_size=[3],
        )
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        assert h5td.create_nested(("nested", "b")) is h5td
        assert set(h5td.keys(True)) == {"a", "nested", ("nested", "d"), ("nested", "b")}
        assert (h5td["nested", "d"] == 1).all()
        # a group read before it is replaced is not served from the cache
        h5td["nested", "b", "c"] = torch.zeros(3)
        assert set(h5td["nested", "b"].keys()) == {"c"}
        h5td.create_nested(("nested", "b"))
        assert set(h5td["nested", "b"].keys()) == set()
        h5td.create_nested(("a", "b"))
        h5td.create_nested("nested")
        assert set(h5td.keys(True)) == {"a", ("a", "b"), "nested"}
        assert h5td["nested"].batch_size == (3,)
        h5td.close()

    def test_create_nested_read_only(self, tmp_path):
        # a read-only file raises the ValueError of h5py's create_group, for a
        # string key and for a tuple key, as before #3008's fix
        td = TensorDict(
            a=torch.zeros(3),
            nested=TensorDict(d=torch.ones(3), batch_size=[3]),
            batch_size=[3],
        )
        PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5").close()
        h5td = PersistentTensorDict.from_h5(tmp_path / "file.h5", mode="r")
        for key in ("nested", ("a", "b"), ("nested", "d")):
            with pytest.raises(ValueError):
                h5td.create_nested(key)
        assert set(h5td.keys(True)) == {"a", "nested", ("nested", "d")}
        h5td.close()

    def test_create_nested_keeps_cached_and_locked_children(self, tmp_path):
        td = TensorDict(a=TensorDict(x=torch.zeros(3), batch_size=[3]), batch_size=[3])
        PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5").close()
        # a create_nested that raises on a read-only file leaves the cached
        # sub-tensordict and its lock as they were
        h5td = PersistentTensorDict.from_h5(tmp_path / "file.h5", mode="r")
        sub = h5td["a"].lock_()
        with pytest.raises(ValueError):
            h5td.create_nested(("a", "b"))
        assert h5td["a"] is sub
        assert sub.is_locked
        h5td.close()
        # a locked sub-tensordict is not written into: the call raises for
        # the existing group, as before #3008's fix
        h5td = PersistentTensorDict.from_h5(tmp_path / "file.h5", mode="r+")
        sub = h5td["a"].lock_()
        with pytest.raises(ValueError):
            h5td.create_nested(("a", "b"))
        assert set(sub.keys()) == {"x"}
        sub.unlock_()
        h5td.create_nested(("a", "b"))
        assert set(h5td["a"].keys()) == {"x", "b"}
        h5td.close()

    def test_device_setter(self, tmp_path):
        # as on a TensorDict, the device changes only through to()
        td = TensorDict({"a": torch.zeros(3)}, [3])
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        with pytest.raises(RuntimeError, match="device cannot be set"):
            h5td.device = "cpu"
        assert h5td.device is None
        h5td.close()

    def test_set_named_sub_td_keeps_names(self, tmp_path):
        # as on a TensorDict, a nested tensordict keeps its names (#2252)
        td = TensorDict({"a": torch.zeros(4), "meta": {"m": torch.zeros(4, 3)}}, [4])
        td.names = ["t"]
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        assert h5td["meta"].names == ["t"]
        h5td["s"] = TensorDict(
            {"y": torch.zeros(4, 2)}, batch_size=[4, 2], names=["t", "u"]
        )
        assert h5td["s"].names == ["t", "u"]
        h5td.close()

    def test_rename_clashing_with_nested_names_warns(self, tmp_path):
        # a new name that clashes with a kept nested name leaves that nested
        # dim unnamed, as before #2252, and warns that it will raise
        sub = TensorDict({"y": torch.zeros(4, 2)}, [4, 2], names=["t", "x"])
        td = TensorDict({"s": sub}, [4], names=["t"])
        with pytest.warns(FutureWarning, match="TensorDict 0.17"):
            h5td = PersistentTensorDict.from_dict(
                td, filename=tmp_path / "a.h5", names=["x"]
            )
        assert h5td.names == ["x"]
        assert h5td["s"].names == ["x", None]
        h5td.close()
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "b.h5")
        with pytest.warns(FutureWarning, match="TensorDict 0.17"):
            h5td.names = ["x"]
        assert h5td["s"].names == ["x", None]
        h5td.close()

    @pytest.mark.parametrize(
        "op_name,args,same_args",
        [
            ("reshape", (6,), (2, 3)),
            ("flatten", (0, 1), (0, 0)),
            ("repeat", (2, 1), (1, 1)),
            ("unflatten", (1, (3, 1)), (1, (3,))),
            ("repeat_interleave", (2, 0), (1, 0)),
            ("gather", (1, torch.zeros(2, 1, dtype=torch.long)), None),
            ("masked_select", (torch.ones(2, 3, dtype=torch.bool),), None),
        ],
    )
    def test_inplace_shape_op_raises_before_writing(
        self, tmp_path, op_name, args, same_args
    ):
        # The file cannot change its batch size in place: the call raises
        # before it rewrites any dataset
        td = TensorDict(
            {"x": torch.zeros(2, 3, 4), "n": {"y": torch.zeros(2, 3)}}, [2, 3]
        )
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        with pytest.raises(
            NotImplementedError, match="changes the batch size.*to_tensordict"
        ):
            getattr(h5td, op_name)(*args, inplace=True)
        assert h5td.batch_size == (2, 3)
        assert h5td["x"].shape == (2, 3, 4)
        assert h5td["n", "y"].shape == (2, 3)
        if same_args is not None:
            # a call that keeps the batch size still works
            assert getattr(h5td, op_name)(*same_args, inplace=True) is h5td
        h5td.close()

    @pytest.mark.parametrize(
        "op",
        [
            lambda td: td.flatten(0, 1, inplace=True),
            lambda td: td.reshape(6, inplace=True),
            lambda td: td.pad([0, 1], inplace=True),
        ],
        ids=["flatten", "reshape", "pad"],
    )
    def test_inplace_shape_op_in_parent_raises_before_writing(self, tmp_path, op):
        # A tensordict that holds a file raises before it rewrites its own
        # leaves, which come before the file
        h5td = PersistentTensorDict.from_dict(
            TensorDict({"y": torch.zeros(2, 3)}, [2, 3]),
            filename=tmp_path / "file.h5",
        )
        td = TensorDict({"a": torch.arange(6.0).view(2, 3), "h": h5td}, [2, 3])
        with pytest.raises(
            NotImplementedError, match="PersistentTensorDict at key 'h'"
        ):
            op(td)
        assert td.batch_size == (2, 3)
        assert td["a"].shape == (2, 3)
        assert h5td.batch_size == (2, 3)
        assert h5td["y"].shape == (2, 3)
        h5td.close()

    def test_batch_size_setter_incompatible(self, tmp_path):
        # as on a TensorDict, an incompatible batch size raises and the old
        # one stays
        td = TensorDict({"a": torch.zeros(2, 3)}, [2])
        h5td = PersistentTensorDict.from_dict(
            td, filename=tmp_path / "file.h5", device="cpu"
        )
        with pytest.raises(RuntimeError, match="batch size and array size mismatch"):
            h5td.batch_size = [5]
        assert h5td.batch_size == (2,)
        with pytest.raises(RuntimeError, match="batch size and array size mismatch"):
            h5td.to(batch_size=[5])
        with pytest.raises(RuntimeError, match="batch size and array size mismatch"):
            h5td.to("cpu", batch_size=[5])
        assert h5td.batch_size == (2,)
        h5td.batch_size = [2, 3]
        assert h5td.batch_size == (2, 3)
        h5td.close()

    def test_replace_does_not_write_into_arg(self, tmp_path):
        # https://github.com/pytorch/tensordict/issues/2984
        h5td = PersistentTensorDict.from_dict(
            TensorDict(a=torch.zeros(3), nested={"x": torch.zeros(3)}, batch_size=[3]),
            filename=tmp_path / "file.h5",
        )
        td = TensorDict(a=torch.ones(3), batch_size=[3])
        out = td.replace(h5td, b=torch.ones(3), nested={"y": torch.ones(3)})
        assert set(h5td.keys(True)) == {"a", "nested", ("nested", "x")}
        assert type(out) is TensorDict
        assert set(out.keys(True)) == {
            "a",
            "b",
            "nested",
            ("nested", "x"),
            ("nested", "y"),
        }
        assert (out["a"] == 0).all()
        assert (out["b"] == 1).all()
        assert (out["nested", "y"] == 1).all()
        # An indexed file is a view that shares the file too.
        for index in (slice(0, 2), 0, torch.tensor([0, 2])):
            sub = h5td[index]
            out = TensorDict(batch_size=sub.batch_size).replace(
                sub, b=torch.ones(sub.batch_size)
            )
            assert set(h5td.keys()) == {"a", "nested"}
            assert type(out) is TensorDict
            assert out.batch_size == sub.batch_size
            assert set(out.keys()) == {"a", "b", "nested"}
            assert (out["b"] == 1).all()
        h5td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_auto_batch_size(tmpdir):
    tmpdir = Path(tmpdir)
    td = TensorDict(
        {
            "a": torch.arange(12).view((3, 4)),
            "b": TensorDict(
                {
                    "c": torch.arange(60).view(3, 4, 5),
                    "d": "a string!",
                },
                batch_size=[3, 4, 5],
            ),
            "e": "another string!",
        },
        batch_size=[3, 4],
    )
    td.to_h5(tmpdir / "file.h5")
    td_recon = TensorDict.from_h5(tmpdir / "file.h5")
    assert td_recon.batch_size == torch.Size([3, 4])
    assert td_recon["b"].batch_size == torch.Size([3, 4, 5])

    assert (td_recon["a"] == td["a"]).all()
    assert (td_recon["b", "c"] == td["b", "c"]).all()
    # This breaks because str are loaded as bytes
    # assert (td_recon == td).all(), (td == td_recon).to_dict()

    td_dict = td.to_dict()
    td_recon_dict = td_recon.to_dict()

    # Checks that all items match
    def check(x, y):
        if isinstance(x, torch.Tensor):
            assert (x == y).all()
            return
        assert str(x) == y.decode("utf-8")

    tree_map(check, td_dict, td_recon_dict)


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
@pytest.mark.parametrize("batch_dims", [None, 1, 2])
def test_from_any_batch_size(tmpdir, batch_dims):
    filename = Path(tmpdir) / "file.h5"
    with h5py.File(filename, "w") as f:
        f["a"] = np.zeros((3, 4, 5))
        f["b/c"] = np.zeros((3, 4, 5, 6))
        f["b/d"] = np.zeros((3, 4, 5))
    expected = torch.Size([3, 4, 5][:batch_dims])
    with h5py.File(filename, "r") as f:
        td = TensorDict.from_any(f, auto_batch_size=True, batch_dims=batch_dims)
        assert td.batch_size == expected
        assert td["b"].batch_size == expected
        td = TensorDict.from_any(f, batch_size=expected)
        assert td.batch_size == expected
        assert td["b"].batch_size == expected
        assert TensorDict.from_any(f, batch_size=3).batch_size == (3,)


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_kwargs_passthrough_nested(tmpdir):
    # create_dataset kwargs must reach the leaves of nested tensordicts too
    # https://github.com/pytorch/tensordict/issues/1758
    tmpdir = Path(tmpdir)
    td = TensorDict(
        {
            "a": torch.zeros(64, 3),
            "b": {"c": torch.zeros(64, 5), "d": {"e": torch.zeros(64, 7)}},
        },
        batch_size=[64],
    )
    td.to_h5(tmpdir / "file.h5", compression="gzip", compression_opts=9)
    with h5py.File(tmpdir / "file.h5", "r") as f:
        for key in ("a", "b/c", "b/d/e"):
            assert f[key].compression == "gzip", key
            assert f[key].compression_opts == 9, key
    td_recon = TensorDict.from_h5(tmpdir / "file.h5")
    for key in (("a",), ("b", "c"), ("b", "d", "e")):
        assert (td_recon[key] == td[key]).all(), key


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_inplace_methods_write(tmp_path):
    # #2217: the in-place methods, update_ and copy_ write to the file
    td = TensorDict(
        {"a": torch.full((3,), -1.0), "nested": {"b": torch.full((3, 2), -1.0)}},
        batch_size=[3],
    )
    filename = tmp_path / "file.h5"
    h5td = PersistentTensorDict.from_dict(td, filename=filename)
    other = td.apply(lambda x: torch.rand_like(x) + 1)
    for data in (h5td, td):
        assert data.abs_() is data
        assert data.add_(1.0) is data
        data += 1
        assert data.pow_(other) is data
        assert data.update_(other.select("a") * 3) is data
        assert data.update_(other, keys_to_update=[("nested", "b")]) is data
    h5td.close()
    with h5py.File(filename, "r") as file:
        torch.testing.assert_close(torch.as_tensor(file["a"][()]), td["a"])
        torch.testing.assert_close(
            torch.as_tensor(file["nested/b"][()]), td["nested", "b"]
        )


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_inplace_methods_write_lazy_stack(tmp_path):
    # #2570: the in-place methods of a lazy stack write to its h5 members, also
    # beside a dense member and in a stack of stacks
    def make():
        return TensorDict(
            {"a": torch.ones(3), "nested": {"b": torch.ones(3, 2)}}, batch_size=[3]
        )

    files = [
        PersistentTensorDict.from_dict(make(), filename=tmp_path / f"{i}.h5")
        for i in range(4)
    ]
    dense = make()
    expected = lazy_stack([make() for _ in range(3)])
    for data in (lazy_stack([files[0], dense, files[1]]), expected):
        assert data.add_(1.0) is data
        assert data.lerp_(expected.to_tensordict() * 0 + 10, 0.5) is data
        assert data.mul_(torch.arange(3.0).unsqueeze(-1).expand(3, 3)) is data
    for td, member in zip((files[0], dense, files[1]), expected.tensordicts):
        torch.testing.assert_close(td.to_tensordict(), member)
    stack = lazy_stack([lazy_stack(files[:2]), lazy_stack(files[2:])], dim=1)
    assert stack.zero_() is stack
    for td in files:
        assert (td["a"] == 0).all()
        assert (td["nested", "b"] == 0).all()
    # masked_fill_ writes as before
    assert stack.masked_fill_(torch.ones(2, 2, 3, dtype=torch.bool), 5.0) is stack
    for td in files:
        assert (td["a"] == 5).all()
        assert (td["nested", "b"] == 5).all()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_inplace_methods_write_nested_lazy_stack(tmp_path):
    # #2592: the in-place methods of a tensordict write to the h5 members of a
    # lazy stack that it holds, also beside a dense member and two levels deep
    def make():
        return TensorDict(
            {"a": torch.ones(3), "nested": {"b": torch.ones(3, 2)}}, batch_size=[3]
        )

    def outer(first, second):
        return TensorDict(
            {"x": torch.ones(2, 3), "s": first, "n": {"s": second}}, batch_size=[2]
        )

    files = [
        PersistentTensorDict.from_dict(make(), filename=tmp_path / f"{i}.h5")
        for i in range(3)
    ]
    dense = make()
    data = outer(lazy_stack(files[:2]), lazy_stack([files[2], dense]))
    expected = outer(lazy_stack([make(), make()]), lazy_stack([make(), make()]))
    end = expected * 0 + 10
    for td in (data, expected):
        assert td.add_(1.0) is td
        assert td.lerp_(end, 0.5) is td
        assert td.mul_(torch.tensor(2.0)) is td
        assert td.addcmul_(end, end, value=0.01) is td
    members = expected["s"].tensordicts + expected["n", "s"].tensordicts
    for td, member in zip(files + [dense], members):
        torch.testing.assert_close(td.to_tensordict(), member)
    torch.testing.assert_close(data["x"], expected["x"])
    # zero_, fill_, masked_fill_, apply_ and detach_ write as before
    assert data.zero_() is data
    assert data.fill_(("s", "a"), 2.0) is data
    assert data.masked_fill_(torch.tensor([True, False]), 3.0) is data
    assert data.apply_(lambda x: x.add_(1)) is data
    assert data.detach_() is data
    for td, a, b in zip(files + [dense], (4, 3, 4, 1), (4, 1, 4, 1)):
        assert (td["a"] == a).all()
        assert (td["nested", "b"] == b).all()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_inplace_methods_write_nested_file(tmp_path):
    # #2775: copy_, update_ and the in-place math write to the h5 files that a
    # tensordict holds, also in the members of a lazy stack, and to the lazy
    # stacks of files that those members hold
    files = []

    def h5():
        files.append(
            PersistentTensorDict.from_dict(
                TensorDict(x=torch.ones(3), batch_size=[3]),
                filename=tmp_path / f"{len(files)}.h5",
            )
        )
        return files[-1]

    def member(n):
        return TensorDict(a=torch.ones(3), n=n, batch_size=[3])

    layouts = (
        lambda: member(h5()),
        lambda: member(TensorDict(m=h5(), batch_size=[3])),
        lambda: lazy_stack([member(h5()), member(h5())]),
        lambda: lazy_stack([member(lazy_stack([h5(), h5()], 1)) for _ in range(2)]),
    )
    calls = (
        lambda td: td.add_(1.0),
        lambda td: td.mul_(torch.tensor(3.0)),
        lambda td: td.copy_(td.to_tensordict() * 0 + 2),
        lambda td: td.update_(td.to_tensordict() * 0 + 5),
        lambda td: td.update_(td.to_tensordict().apply(lambda x: x * 0 + 4).to_dict()),
    )
    for make in layouts:
        for call in calls:
            start = len(files)
            data = make()
            expected = data.to_tensordict()
            assert call(data) is data
            call(expected)
            torch.testing.assert_close(data.to_tensordict(), expected)
            for file in files[start:]:
                assert (file["x"] != 1).all()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
@pytest.mark.parametrize("idx", [1, slice(1, 3), (slice(1, 3), 0)])
def test_sub_td_inplace_methods_tensor_write_nested_file(tmp_path, idx):
    # #3049: through a sub-tensordict, the in-place math with a tensor operand
    # writes to the h5 file that the tensordict holds, as with a number
    expected = TensorDict(
        h=TensorDict(
            x=torch.arange(12.0).view(4, 3),
            g=TensorDict(z=torch.arange(24.0).view(4, 3, 2), batch_size=[4, 3]),
            batch_size=[4, 3],
        ),
        y=torch.arange(12.0).view(4, 3),
        batch_size=[4, 3],
    )
    td = expected.clone()
    path = tmp_path / "file.h5"
    td["h"] = PersistentTensorDict.from_dict(td["h"], filename=path)
    for data in (td, expected):
        sub_td = data._get_sub_tensordict(idx)
        w = torch.arange(1.0, 1.0 + sub_td.batch_size.numel()).view(sub_td.batch_size)
        sub_td.add_(w)
        sub_td.mul_(w[..., :1])
        sub_td.clamp_min_(w * 3)
        sub_td.lerp_(sub_td.to_tensordict() * 0, w / 10)
        sub_td.addcmul_(w, w)
    torch.testing.assert_close(td.to_tensordict(), expected)
    td["h"].close()
    h5td = PersistentTensorDict.from_h5(path)
    torch.testing.assert_close(h5td.to_tensordict(), expected["h"])
    h5td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_update_keys_to_update_writes_nested_file(tmp_path):
    # #2780: update_ with keys_to_update writes the h5 files that a tensordict
    # holds, as without keys_to_update, also for a key that names a
    # tensordict, and raises before it writes for a key that self lacks
    files = []

    def h5():
        files.append(
            PersistentTensorDict.from_dict(
                TensorDict(x=torch.ones(3), batch_size=[3]),
                filename=tmp_path / f"{len(files)}.h5",
            )
        )
        return files[-1]

    def member(n):
        return TensorDict(a=torch.ones(3), n=n, batch_size=[3])

    layouts = (
        lambda: member(h5()),
        lambda: member(TensorDict(m=h5(), batch_size=[3])),
        lambda: member(lazy_stack([h5(), h5()], 1)),
        lambda: lazy_stack([member(h5()), member(h5())]),
        lambda: lazy_stack([member(lazy_stack([h5(), h5()], 1)) for _ in range(2)]),
    )
    for make in layouts:
        for keys in (["n"], [key for key in make().keys(True, True) if "x" in key]):
            start = len(files)
            data = make()
            expected = data.to_tensordict()
            source = expected * 0 + 5
            assert data.update_(source, keys_to_update=keys) is data
            expected.update_(source, keys_to_update=keys)
            torch.testing.assert_close(data.to_tensordict(), expected)
            assert (data["a"] == 1).all()
            for file in files[start:]:
                assert (file["x"] == 5).all()
            # from a file
            source = PersistentTensorDict.from_dict(
                expected * 0 + 6, filename=tmp_path / f"src{start}.h5"
            )
            assert data.update_(source, keys_to_update=keys) is data
            for file in files[start:]:
                assert (file["x"] == 6).all()
            source = expected * 0 + 5
            source["n", "y"] = torch.zeros(source["n"].batch_size)
            with pytest.raises(KeyError, match="not found in the dest"):
                data.update_(source * 0, keys_to_update=["n"])
            for file in files[start:]:
                assert (file["x"] == 6).all()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_inplace_read_only_warns(tmp_path):
    # from_h5 opens files read-only by default: the in-place methods leave the
    # file unchanged, as before they wrote to files, and warn that this will
    # raise. Writes such as zero_() still raise.
    td = TensorDict({"a": torch.full((3,), -1.0)}, batch_size=[3])
    td.to_h5(tmp_path / "file.h5").close()
    h5td = PersistentTensorDict.from_h5(tmp_path / "file.h5")
    match = (
        r"^Calling abs_\(\) on a tensordict stored in a file opened read-only, "
        r"which leaves the file unchanged, is deprecated and will be removed in "
        r"TensorDict 0\.17\."
    )
    with pytest.warns(FutureWarning, match=match) as record:
        assert h5td.abs_() is h5td
    assert record[0].filename == __file__

    def iadd():
        nonlocal h5td
        h5td += 1

    for call in (
        lambda: h5td.mul_(2.0),
        iadd,
        lambda: h5td[0:2].add_(1.0),
        lambda: h5td[0:2].zero_(),
        # #2570: one warning for a lazy stack of rows of the file
        lambda: lazy_stack([h5td[0], h5td[1]]).add_(1.0),
        lambda: lazy_stack([h5td[0], h5td[1]]).zero_(),
        lambda: lazy_stack([h5td, h5td]).zero_(),
        # #2592: and for a lazy stack of rows held by a tensordict
        lambda: TensorDict({"s": lazy_stack([h5td[0], h5td[1]])}, [2]).add_(1.0),
        # #2775: and for the file held by a tensordict
        lambda: TensorDict({"f": h5td}, [3]).add_(1.0),
        lambda: TensorDict({"f": h5td}, [3]).update_({"f": {"a": torch.zeros(3)}}),
        lambda: h5td.update_({"a": torch.zeros(3)}),
        lambda: h5td.copy_(td.clone().zero_()),
    ):
        with pytest.warns(FutureWarning, match="leaves the file unchanged") as record:
            call()
        # one warning, which points to the caller
        assert len(record) == 1
        assert record[0].filename == __file__
    assert (h5td["a"] == -1).all()
    with pytest.raises(OSError):
        h5td.zero_()
    # #2570: zero_ on a lazy stack writes the writable files and leaves the
    # read-only ones unchanged, with one warning, also in nested stacks
    nested = TensorDict({"a": torch.ones(3), "n": {"b": torch.ones(3)}}, [3])
    nested.to_h5(tmp_path / "nested.h5").close()
    read_only = PersistentTensorDict.from_h5(tmp_path / "nested.h5")
    writable = PersistentTensorDict.from_dict(nested, filename=tmp_path / "w.h5")
    stack = lazy_stack(
        [lazy_stack([writable, read_only]), lazy_stack([read_only, writable])]
    )
    with pytest.warns(
        FutureWarning, match="zero_.*leaves the file unchanged"
    ) as record:
        assert stack.zero_() is stack
    assert len(record) == 1
    assert record[0].filename == __file__
    torch.testing.assert_close(writable.to_tensordict(), nested.zero_())
    assert (read_only["a"] == 1).all()
    assert (read_only["n", "b"] == 1).all()
    # masked_fill_ raises on the rows of the file too, as it did before
    mask = torch.tensor([True, False])
    for call in (
        lambda: h5td[0:2].masked_fill_(mask, 0.0),
        lambda: lazy_stack([h5td[0], h5td[1]]).masked_fill_(mask, 0.0),
    ):
        with pytest.raises(OSError):
            call()
    assert (h5td["a"] == -1).all()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
@pytest.mark.parametrize("file_state", ["closed", "read_only"])
@pytest.mark.parametrize(
    "write,read_only_error",
    [
        (lambda td: td.__setitem__("new", torch.ones(2)), ValueError),
        (lambda td: td.__setitem__("x", torch.ones(2)), OSError),
        (lambda td: td.set(("n", "new"), torch.ones(2)), ValueError),
        (lambda td: td.set(("q", "r", "s"), torch.ones(2)), ValueError),
        (lambda td: td.update({"new": torch.ones(2)}), ValueError),
        # replacing an entry of a read-only file fails to delete the old one
        (lambda td: td.update({"x": torch.ones(2)}), KeyError),
        (lambda td: td.set("x", torch.ones(2), inplace=True), OSError),
    ],
    ids=[
        "setitem_new",
        "setitem",
        "set_nested_new",
        "set_nested_deep_new",
        "update_new",
        "update",
        "set_inplace",
    ],
)
def test_write_that_cannot_be_made_raises(tmp_path, file_state, write, read_only_error):
    # #2964: these writes went silently missing from the file
    path = tmp_path / "file.h5"
    td = TensorDict(x=torch.zeros(2), n=TensorDict(y=torch.zeros(2)), batch_size=[2])
    h5td = td.to_h5(path)
    h5td.close()
    if file_state == "read_only":
        h5td = PersistentTensorDict.from_h5(path)
    # h5py's error; a closed file rejects every write with a ValueError
    error = ValueError if file_state == "closed" else read_only_error
    with pytest.raises(error):
        write(h5td)
    h5td.close()
    h5td = PersistentTensorDict.from_h5(path)
    torch.testing.assert_close(h5td.to_tensordict(), td)
    h5td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_write_rejected_dataset_kwargs_raises(tmp_path):
    # #2964: h5py rejects the chunks of a 2-element array, and the entry was
    # not written, without an error
    h5td = PersistentTensorDict(
        filename=tmp_path / "file.h5", batch_size=[], mode="w", chunks=(4,)
    )
    with pytest.raises(ValueError, match="Chunk shape"):
        h5td["x"] = torch.zeros(2)
    h5td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_inplace_methods_keep_key_order(tmp_path):
    # lerp_ pairs the leaves of its operand with those of the file by key, also
    # when a nested key comes first
    td = TensorDict(
        nested=TensorDict(d=torch.zeros(4, 3), batch_size=[4]),
        a=torch.zeros(4, 2),
        batch_size=[4],
    )
    h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
    for data in (h5td, td._get_sub_tensordict(torch.tensor([0, 2]))):
        end = data.to_tensordict()
        end["a"] = torch.full_like(end["a"], 10.0)
        end["nested", "d"] = torch.full_like(end["nested", "d"], -10.0)
        assert data.lerp_(end, 1.0) is data
        assert (data["a"] == 10).all()
        assert (data["nested", "d"] == -10).all()


def _h5_with_nested(tmp_path):
    td = TensorDict(
        a=torch.arange(12.0).view(4, 3),
        nested=TensorDict(b=torch.zeros(4, 3, 2), batch_size=[4, 3]),
        batch_size=[4, 3],
    )
    return PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_to_device_is_lazy(tmp_path, monkeypatch):
    # to(device) returns a PersistentTensorDict on the file, without reading it;
    # a dtype gives a TensorDict copy
    h5td = _h5_with_nested(tmp_path)

    def no_read(*args, **kwargs):
        raise AssertionError("the file was read")

    monkeypatch.setattr(type(h5td._backend), "read_full", staticmethod(no_read))
    monkeypatch.setattr(type(h5td._backend), "read_at", staticmethod(no_read))
    meta = h5td.to("meta")
    assert type(meta) is PersistentTensorDict and meta is not h5td
    assert meta.device == torch.device("meta")
    assert meta.batch_size == h5td.batch_size
    assert meta.to("meta") is meta
    meta_bs = h5td.to("meta", batch_size=[4])
    assert type(meta_bs) is PersistentTensorDict and meta_bs.batch_size == (4,)
    assert h5td.batch_size == (4, 3)
    monkeypatch.undo()
    assert meta["nested"].device == torch.device("meta")
    assert meta["a"].device == torch.device("meta")
    assert meta["nested", "b"].device == torch.device("meta")
    double = h5td.to(torch.float64)
    assert type(double) is TensorDict
    assert double["a"].dtype == torch.float64
    assert (double["a"] == h5td["a"]).all()
    # batch size alone: in place
    assert h5td.to(batch_size=[4]) is h5td
    assert h5td.batch_size == (4,)
    h5td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_to_raises(tmp_path):
    h5td = _h5_with_nested(tmp_path)
    with pytest.raises(
        TypeError, match=r"Cannot use inplace=True with PersistentTensorDict.to\(\)"
    ):
        h5td.to(torch.float64, inplace=True)
    with pytest.raises(
        RuntimeError, match="Cannot use non_blocking_pin=True PersistentTensorDict"
    ):
        h5td.to("cpu", non_blocking_pin=True)
    h5td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_to_memory_format(tmp_path):
    td = TensorDict(a=torch.arange(120.0).view(2, 3, 4, 5), batch_size=[2])
    h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
    # without a dtype, the memory format is ignored: the file itself is
    # returned, with a warning
    match = r"to\(memory_format=\.\.\.\) without a dtype ignores the memory format"
    with pytest.warns(FutureWarning, match=match) as record:
        out = h5td.to(memory_format=torch.channels_last)
    assert record[0].filename == __file__
    assert out is h5td
    assert not out["a"].is_contiguous(memory_format=torch.channels_last)
    with pytest.warns(FutureWarning, match=match):
        meta = h5td.to("meta", memory_format=torch.channels_last)
    assert type(meta) is PersistentTensorDict and meta.device == torch.device("meta")
    # with a dtype, the dense copy takes the memory format
    out = h5td.to(torch.float64, memory_format=torch.channels_last)
    assert type(out) is TensorDict
    assert out["a"].is_contiguous(memory_format=torch.channels_last)
    assert (out["a"] == td["a"]).all()
    h5td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_to_context_manager(tmp_path):
    h5td = _h5_with_nested(tmp_path)
    # the dense copy is written back to the file on exit, as on TensorDict
    # in place: no dataset is replaced, so nothing warns
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with h5td.to(torch.float64) as td64:
            assert type(td64) is TensorDict
            td64["a"] += 1
            td64["new"] = torch.ones(4, 3)
    assert h5td["a"].dtype == torch.float32
    assert (h5td["a"] == torch.arange(12.0).view(4, 3) + 1).all()
    assert (h5td["new"] == 1).all()
    # the device-only copy and the file itself write to the file in the block,
    # so nothing is written back
    with h5td.to("meta") as meta:
        assert meta.device == torch.device("meta")
    # a copy on meta holds no data to write back
    with h5td.to("meta", torch.float64) as meta:
        assert type(meta) is TensorDict
        meta["a"] += 1
    assert (h5td["a"] == torch.arange(12.0).view(4, 3) + 1).all()
    with h5td.to(batch_size=[4, 3]) as same:
        assert same is h5td
        same["a"] = same["a"] + 2
    assert (h5td["a"] == torch.arange(12.0).view(4, 3) + 3).all()
    h5td.close()
    # a file opened read-only stays unchanged, with a warning
    h5td = PersistentTensorDict.from_h5(tmp_path / "file.h5")
    with pytest.warns(
        FutureWarning,
        match=r"^Calling to\(\) on a tensordict stored in a file opened read-only",
    ) as record:
        with h5td.to(torch.float64) as td64:
            td64["a"] += 1
    assert record[0].filename == __file__
    assert (h5td["a"] == torch.arange(12.0).view(4, 3) + 3).all()
    h5td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_to_context_manager_writes_in_place(tmp_path, monkeypatch):
    # the write-back reads nothing and keeps the layout of the datasets, also
    # for a file reopened without its creation arguments
    td = TensorDict(
        a=torch.arange(4.0),
        nested=TensorDict(b=torch.zeros(4), batch_size=[4]),
        batch_size=[4],
    )
    PersistentTensorDict.from_dict(
        td, filename=tmp_path / "file.h5", compression="gzip"
    ).close()
    h5td = PersistentTensorDict.from_h5(tmp_path / "file.h5", mode="r+")
    reads = []
    getitem = h5py.Dataset.__getitem__

    def counting_getitem(self, *args, **kwargs):
        reads.append(self.name)
        return getitem(self, *args, **kwargs)

    with h5td.to(torch.float64) as td64:
        td64["a"] += 1
        td64["nested", "c"] = torch.ones(4)
        monkeypatch.setattr(h5py.Dataset, "__getitem__", counting_getitem)
    monkeypatch.undo()
    assert reads == []
    assert h5td.file["a"].compression == "gzip"
    assert h5td.file["nested/b"].compression == "gzip"
    assert h5td["a"].dtype == torch.float32
    assert (h5td["a"] == torch.arange(4.0) + 1).all()
    assert (h5td["nested", "c"] == 1).all()
    # an entry whose shape the block changes replaces its dataset
    with (
        pytest.warns(UserWarning, match="Replacing an array"),
        h5td.to(torch.float64) as td64,
    ):
        td64["nested", "b"] = torch.ones(4, 2)
    assert (h5td["nested", "b"] == torch.ones(4, 2)).all()
    h5td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_to_context_manager_changes_entry_type(tmp_path):
    # as on a TensorDict, a group that the block replaces with a tensor is
    # replaced, and a tensor replaced with a tensordict raises
    td = TensorDict(
        a=torch.arange(4.0),
        nested=TensorDict(b=torch.zeros(4), batch_size=[4]),
        batch_size=[4],
    )
    h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
    with h5td.to(torch.float64) as td64:
        td64["nested"] = torch.ones(4)
    assert set(h5td.keys(True, True)) == {"a", "nested"}
    assert (h5td["nested"] == 1).all()
    with pytest.raises(RuntimeError, match="cannot set a tensor collection"):
        with h5td.to(torch.float64) as td64:
            td64["a"] = TensorDict(c=torch.ones(4), batch_size=[4])
    assert set(h5td.keys(True, True)) == {"a", "nested"}
    assert (h5td["a"] == torch.arange(4.0)).all()
    h5td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
@pytest.mark.parametrize("dest", ["meta", "cpu", torch.float64])
def test_to_context_manager_tensorclass(tmp_path, dest):
    # a tensorclass over a file writes back as the file does
    class H5Data(TensorClass):
        a: torch.Tensor

    td = TensorDict(a=torch.arange(4.0), batch_size=[4])
    h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
    tc = H5Data.from_tensordict(h5td)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with tc.to(dest) as out:
            assert type(out) is H5Data
            if dest is torch.float64:
                out.a += 1
    assert h5td["a"].dtype == torch.float32
    assert (h5td["a"] == torch.arange(4.0) + (dest is torch.float64)).all()
    h5td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_from_h5_auto_batch_size(tmp_path):
    TensorDict(a=torch.zeros(4, 3), batch_size=[4, 3]).to_h5(tmp_path / "f.h5").close()
    h5td = PersistentTensorDict.from_h5(
        tmp_path / "f.h5", auto_batch_size=True, batch_dims=1
    )
    assert type(h5td) is PersistentTensorDict
    assert h5td.batch_size == (4,)
    h5td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_from_dict_batch_dims(tmp_path):
    h5td = PersistentTensorDict.from_dict(
        {"a": torch.zeros(4, 3)}, filename=tmp_path / "f.h5", batch_dims=1
    )
    assert h5td.batch_size == (4,)
    h5td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
@pytest.mark.parametrize("keys_to_update", [None, ["a"]])
def test_update_from_h5_reads(tmp_path, monkeypatch, keys_to_update):
    # update_ from an h5 file reads it as often as a TensorDict destination does
    td = TensorDict(
        {"a": torch.ones(3), "nested": {"b": torch.ones(3, 2)}}, batch_size=[3]
    )
    source = PersistentTensorDict.from_dict(td, filename=tmp_path / "src.h5")
    dest = PersistentTensorDict.from_dict(td * 0, filename=tmp_path / "dest.h5")
    reads = []
    read_full = source._backend.read_full

    def counting_read_full(node):
        reads.append(node.file.filename)
        return read_full(node)

    monkeypatch.setattr(
        type(source._backend), "read_full", staticmethod(counting_read_full)
    )
    (td * 0).update_(source, keys_to_update=keys_to_update)
    expected = reads.count(source.file.filename)
    assert expected
    reads.clear()
    dest.update_(source, keys_to_update=keys_to_update)
    assert reads.count(source.file.filename) == expected
    assert (dest["a"] == 1).all()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_fill_zero_empty(tmp_path):
    # fill_ and zero_ of a file with batch size [0]
    td = TensorDict({"a": torch.ones(0, 3), "n": {"b": torch.ones(0)}}, [0])
    h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
    assert h5td.fill_("a", 2.0) is h5td
    assert h5td.fill_("n", 1.0) is h5td
    assert h5td.zero_() is h5td
    assert h5td["a"].shape == (0, 3)


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_fill_zero_masked_fill_no_read(tmp_path, monkeypatch):
    # fill_, zero_, masked_fill_ and detach_ write to the file without reading it
    td = TensorDict(
        {"a": torch.ones(3, 2), "nested": {"b": torch.ones(3, 2, 4)}}, batch_size=[3]
    )
    h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")

    def no_read(*args, **kwargs):
        raise AssertionError("the file was read")

    backend = type(h5td._backend)
    with monkeypatch.context() as patch:
        patch.setattr(backend, "read_full", no_read)
        patch.setattr(backend, "read_at", no_read)
        assert h5td.detach_() is h5td
        mask = torch.tensor([True, False, True])
        for dest in (h5td, td):
            assert dest.fill_("a", 2.0) is dest
            assert dest.fill_("nested", 3.0) is dest
            assert dest.masked_fill_(mask, -1.0) is dest
    torch.testing.assert_close(h5td.to_tensordict(), td)
    with monkeypatch.context() as patch:
        patch.setattr(backend, "read_full", no_read)
        patch.setattr(backend, "read_at", no_read)
        # #2570: so does zero_ on a lazy stack of the file and a dense member
        stack = lazy_stack([h5td, td.clone()])
        assert stack.zero_() is stack
        # #2592: and on a tensordict that holds the stack
        holder = TensorDict({"n": {"s": stack}}, batch_size=[2])
        assert holder.zero_() is holder
        assert h5td.zero_() is h5td
    h5td.close()
    reopened = PersistentTensorDict.from_h5(tmp_path / "file.h5")
    torch.testing.assert_close(reopened.to_tensordict(), td.zero_())
    reopened.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_fill_write_blocks(tmp_path, monkeypatch):
    # fill_ writes a dataset in blocks of rows, a 0-d dataset at once, and
    # nothing into an empty dataset
    monkeypatch.setattr(tensordict_persistent, "_WRITE_FULL_BLOCK_BYTES", 100)
    td = TensorDict(
        {
            "a": torch.zeros(7, 3, dtype=torch.int64),
            "big": torch.zeros(2, 40),
            "s": 0.0,
            "empty": torch.zeros(0, 3),
        },
        batch_size=[],
    )
    h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
    for key in td.keys():
        assert h5td.fill_(key, 2.5) is h5td
        td.fill_(key, 2.5)
    torch.testing.assert_close(h5td.to_tensordict(), td)
    assert h5td.zero_() is h5td
    torch.testing.assert_close(h5td.to_tensordict(), td.zero_())
    # a value that is not a scalar is broadcast into the whole dataset, as
    # before, also when it is larger than one block
    value = torch.arange(21).view(7, 3)
    assert h5td.fill_("a", value) is h5td
    assert (h5td["a"] == value).all()
    assert h5td.fill_("a", torch.arange(3)) is h5td
    assert (h5td["a"] == torch.arange(3)).all()
    with pytest.raises(TypeError, match="Can't broadcast"):
        h5td.fill_("a", torch.arange(7))
    # the value is cast as h5py casts it: a complex value raises for a real
    # dataset, even an empty one
    for key in ("a", "empty"):
        with pytest.raises(TypeError, match="complex"):
            h5td.fill_(key, 1 + 2j)


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_kwargs_is_deprecated(tmpdir):
    filename = Path(tmpdir) / "file.h5"
    td = PersistentTensorDict(
        filename=filename, batch_size=[3], mode="w", compression="gzip"
    )
    with pytest.warns(
        DeprecationWarning,
        match=r"^PersistentTensorDict\.kwargs is deprecated and will be removed in "
        r"TensorDict 0\.17\.$",
    ) as record:
        assert td.kwargs == {"compression": "gzip"}
    assert record[0].filename == __file__
    td["a"] = torch.zeros(3)
    td["b", "c"] = torch.zeros(3, 2)
    assert td.file["b/c"].compression == "gzip"
    td.close()
    td = PersistentTensorDict(
        filename=filename, batch_size=[3], mode="r", compression="gzip"
    )
    # pickles made with TensorDict 0.14 store the dataset options under "kwargs"
    state = td.__getstate__()
    state["kwargs"] = state.pop("_dataset_kwargs")
    loaded = PersistentTensorDict.__new__(PersistentTensorDict)
    loaded.__setstate__(state)
    assert "kwargs" not in vars(loaded)
    assert loaded._dataset_kwargs == {"compression": "gzip"}
    assert (loaded["b", "c"] == 0).all()
    td.close()
    loaded.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_pickle_locked_after_nested_access(tmp_path):
    # gh-2920: locking after reading a nested view left a weakref in the
    # pickled state
    td = PersistentTensorDict.from_dict(
        TensorDict({"a": torch.randn(3), ("b", "c"): torch.randn(3)}, batch_size=[3]),
        filename=tmp_path / "x.h5",
    )
    td["b"]
    td.lock_()
    loaded = pickle.loads(pickle.dumps(td))
    assert loaded.is_locked
    assert loaded["b"].is_locked
    assert (loaded["b", "c"] == td["b", "c"]).all()
    td.close()
    loaded.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
def test_vmap_raises(tmp_path):
    td = PersistentTensorDict.from_dict(
        TensorDict(a=torch.zeros(3, 4), batch_size=[3]), filename=tmp_path / "x.h5"
    )
    with pytest.raises(RuntimeError, match="cannot be used with vmap"):
        torch.vmap(lambda x: x)(td)
    # A function that returns it used to give None, or an entry holding None
    with pytest.raises(RuntimeError, match="cannot be used with vmap"):
        torch.vmap(lambda x: td)(torch.zeros(2))
    with pytest.raises(RuntimeError, match="cannot be used with vmap"):
        torch.vmap(lambda x: TensorDict(h=td))(torch.zeros(2))
    td.close()


@pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
class TestH5Indexing:
    @pytest.fixture
    def data(self, tmp_path):
        td = TensorDict(
            {
                "a": torch.arange(60.0).view(10, 6),
                "b": torch.randint(0, 2, (10, 3), dtype=torch.bool),
                "nested": {"c": torch.randn(10, 2)},
                "s": "a string!",
            },
            batch_size=[10],
        )
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        return td, h5td

    @pytest.mark.parametrize(
        "idx",
        [
            0,
            -1,
            slice(2, 7),
            slice(None, None, 3),
            torch.tensor([1, 4, 7]),
            # indices h5py cannot read directly: unsorted, repeated, negative
            torch.tensor([7, 1, 4]),
            torch.tensor([2, 2, 5]),
            torch.tensor([-1, 0]),
            [0, 3],
            range(1, 4),
            np.int64(2),
            torch.tensor(3),
            torch.tensor([], dtype=torch.long),
            torch.tensor([True, False] * 5),
            torch.zeros(10, dtype=torch.bool),
            # h5py reads uint8 as integer indices, torch as a mask
            torch.tensor([1, 0] * 5, dtype=torch.uint8),
            (None,),
        ],
    )
    def test_index_matches_tensordict(self, data, idx):
        td, h5td = data
        result = h5td[idx]
        expected = td[idx]
        assert result.batch_size == expected.batch_size
        for key in expected.keys(True, True):
            assert result.get(key).shape == expected.get(key).shape, key
            assert result.get(key).dtype == expected.get(key).dtype, key
            assert (result.get(key) == expected.get(key)).all(), key
        assert result["s"] == b"a string!"

    @pytest.mark.parametrize(
        "idx",
        [
            (torch.tensor([0, 2]), torch.tensor([1, 3])),
            (slice(1, 4), torch.tensor([5, 0])),
            (Ellipsis, torch.tensor([1, 2])),
            (None, torch.tensor([0, 3])),
            (-1, -2),
            (torch.tensor(1), torch.tensor(2)),
            torch.tensor([[0, 1], [2, 3]]),
            torch.rand(10, 6) > 0.5,
        ],
    )
    def test_index_multidim_batch(self, tmp_path, idx):
        td = TensorDict(
            {
                "a": torch.arange(180.0).view(10, 6, 3),
                "nested": {"b": torch.arange(60).view(10, 6)},
            },
            batch_size=[10, 6],
        )
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        result = h5td[idx]
        expected = td[idx]
        assert result.batch_size == expected.batch_size
        for key in expected.keys(True, True):
            assert result.get(key).shape == expected.get(key).shape, key
            assert (result.get(key) == expected.get(key)).all(), key

    @pytest.mark.parametrize("idx", [True, False])
    def test_index_bool(self, data, idx):
        # h5py reads True / False as the integers 1 / 0
        td, h5td = data
        for key in ("a", ("nested", "c")):
            expected = td.get(key)[idx]
            assert h5td[idx].get(key).shape == expected.shape, key
            assert (h5td[idx].get(key) == expected).all(), key

    @pytest.mark.parametrize(
        "idx",
        [
            1,
            slice(1, 3),
            torch.tensor([0, 2]),
            torch.tensor([True, False, True, False]),
        ],
    )
    @pytest.mark.parametrize("value", [7, 0.1, True])
    def test_index_setitem_scalar(self, tmp_path, idx, value):
        # h5td[idx] = scalar writes the scalar into every tensor entry, in the
        # dtype of the entry, and leaves the non-tensor entries as they are
        td = TensorDict(
            {
                "a": torch.arange(8.0, dtype=torch.float64).view(4, 2),
                "b": torch.zeros(4, dtype=torch.bool),
                "nested": TensorDict(c=torch.arange(12).view(4, 3), batch_size=[4, 3]),
                "s": "a string!",
            },
            batch_size=[4],
        )
        td.set_non_tensor("f", 1)
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        h5td[idx] = value
        expected = td.exclude("s", "f")
        expected[idx] = value
        for key in expected.keys(True, True):
            assert h5td.get(key).dtype == expected.get(key).dtype, key
            assert (h5td.get(key) == expected.get(key)).all(), key
        assert h5td["s"] == b"a string!"
        assert h5td["f"] == 1

    def test_h5_setitem_key_in_place(self, data):
        # h5td[key] = value writes into the existing array: the dtype of the
        # file is kept, and a new shape raises. A new key takes the value.
        td, h5td = data
        h5td["a"] = torch.ones(10, 6, dtype=torch.float64)
        assert h5td["a"].dtype == torch.float32
        assert (h5td["a"] == 1).all()
        with pytest.raises(TypeError, match="Can't broadcast"):
            h5td["a"] = torch.zeros(10, 7)
        h5td["new"] = torch.ones(10, dtype=torch.float64)
        assert h5td["new"].dtype == torch.float64
        h5td["nested", "c"] = torch.zeros(10, 2)
        assert (h5td["nested", "c"] == 0).all()
        # set_ of a missing nested key names its path in the file
        with pytest.raises(
            KeyError, match='key "zz/y" not found in PersistentTensorDict'
        ):
            h5td.set_(("zz", "y"), torch.zeros(10))

    def test_h5_setitem_index_values(self, data):
        # a list index, a dict, a tensordict that does not broadcast, an Ellipsis
        td, h5td = data
        h5td[[0, 3]] = td[[0, 3]].exclude("s") * 0
        assert (h5td["a"][[0, 3]] == 0).all()
        assert (h5td["a"][1] == td["a"][1]).all()
        h5td[1:3] = {"a": torch.full((6,), -1.0)}
        assert (h5td["a"][1:3] == -1).all()
        assert (h5td["nested", "c"][1:3] == td["nested", "c"][1:3]).all()
        with pytest.raises(RuntimeError, match="Cannot broadcast the tensordict"):
            h5td[0:2] = TensorDict(a=torch.ones(3, 6), batch_size=[3])
        h5td[..., 4] = 5
        assert (h5td["a"][4] == 5).all()
        assert (h5td["a"][5:] == td["a"][5:]).all()
        h5td[...] = 2
        assert (h5td["a"] == 2).all()
        assert h5td["s"] == b"a string!"

    def test_h5_empty_reads_nothing(self, data, monkeypatch):
        td, h5td = data
        reads = []
        backend = type(h5td._backend)
        read_full = backend.read_full

        def counting_read_full(self, array):
            reads.append(array.name)
            return read_full(self, array)

        monkeypatch.setattr(backend, "read_full", counting_read_full)
        empty = h5td.empty()
        assert list(empty.keys()) == []
        empty = h5td.empty(recurse=True, batch_size=[10], names=["x"])
        assert type(empty) is TensorDict
        assert empty.names == ["x"]
        assert list(empty.keys(True)) == ["nested", "s"]
        assert empty["s"] == b"a string!"
        assert reads == []

    @pytest.mark.parametrize("idx", [slice(0, 2), torch.tensor([1, 3])])
    def test_index_masked_fill_(self, tmp_path, idx):
        # h5td[idx] writes to the file: only the masked rows are filled
        td = TensorDict(
            {"a": torch.arange(8.0).view(4, 2), "nested": {"b": torch.arange(4)}},
            batch_size=[4],
        )
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        mask = torch.tensor([True, False])
        h5td[idx].masked_fill_(mask, -1)
        td[idx] = td[idx].masked_fill(mask, -1)
        for key in td.keys(True, True):
            assert (h5td.get(key) == td.get(key)).all(), key

    @pytest.mark.parametrize("idx", [slice(0, 2), torch.tensor([1, 3])])
    def test_index_inplace_methods(self, tmp_path, idx):
        # h5td[idx] writes to the file: the in-place methods change only the
        # indexed rows (#2217)
        td = TensorDict(
            {"a": torch.arange(8.0).view(4, 2), "nested": {"b": torch.arange(4.0)}},
            batch_size=[4],
        )
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        sub = h5td[idx]
        sub.add_(1.0)
        sub *= 2
        sub.update_(sub.to_tensordict().neg())
        td[idx] = td[idx].add(1.0).mul(2).neg()
        for key in td.keys(True, True):
            assert (h5td.get(key) == td.get(key)).all(), key

    def test_stack_items(self, data):
        # h5td[idx] is a lazy view of the file, not a lazy stack
        td, h5td = data
        result = torch.stack([h5td[0], h5td[3]])
        expected = torch.stack([td[0], td[3]])
        assert type(result) is TensorDict
        assert result.batch_size == expected.batch_size
        for key in expected.keys(True, True):
            assert (result.get(key) == expected.get(key)).all(), key

    def test_index_reads_only_selected_rows(self, data, monkeypatch):
        # Slicing must not load whole datasets from storage
        _, h5td = data

        def read_full(node):
            raise AssertionError(f"full read of {node.name}")

        monkeypatch.setattr(h5td._backend, "read_full", read_full)
        result = h5td[2:5]
        assert (result["a"] == torch.arange(12.0, 30.0).view(3, 6)).all()
        assert result["nested", "c"].shape == (3, 2)
        assert result.to_tensordict().batch_size == (3,)

    @pytest.mark.parametrize(
        "idx",
        [
            torch.arange(0, 1000, 2),
            torch.tensor([900, 10, 10, -1]),
            torch.arange(1000) % 3 == 0,
        ],
    )
    def test_fancy_index_reads_a_slice(self, tmp_path, monkeypatch, idx):
        # Integer indices and masks are read as a single slice: h5py point
        # selection is quadratic in the number of selected rows
        td = TensorDict({"a": torch.randn(1000, 4)}, batch_size=[1000])
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        backend = h5td._backend
        read_at = backend.read_at
        indices = []

        def recording_read_at(node, index, device):
            indices.append(index)
            return read_at(node, index, device)

        def read_full(node):
            raise AssertionError(f"full read of {node.name}")

        monkeypatch.setattr(backend, "read_at", recording_read_at)
        monkeypatch.setattr(backend, "read_full", read_full)
        assert (h5td[idx]["a"] == td[idx]["a"]).all()
        assert indices and all(isinstance(index, slice) for index in indices)

    def test_get_at(self, data):
        td, h5td = data
        assert (h5td.get_at("a", torch.tensor([7, 1])) == td["a"][[7, 1]]).all()
        assert h5td.get_at("s", 0).data == b"a string!"
        assert h5td.get_at("s", slice(2, 5)).batch_size == (3,)
        assert h5td.get_at(("nested", "c"), 3).shape == (2,)
        assert h5td.get_at("missing", 0, None) is None
        # a missing key returns None, as on TensorDict
        assert h5td.get_at("missing", 0) is None

    def test_get_at_index_kwarg(self, data):
        td, h5td = data
        assert (h5td.get_at("a", index=1) == td["a"][1]).all()
        with pytest.raises(TypeError, match="index argument missing"):
            h5td.get_at("a")

    def test_get_nested_through_tensor_default(self, data):
        # a path through a tensor gives the default, as on TensorDict (#2701)
        td, h5td = data
        assert h5td.get(("a", "x"), None) is None
        assert td.get(("a", "x"), None) is None

    @pytest.mark.parametrize("op", ["flatten_keys", "unflatten_keys"])
    def test_flatten_inplace_raises_before_read(self, data, op, monkeypatch):
        td, h5td = data
        keys = list(h5td.keys(True, True))
        reads = []
        monkeypatch.setattr(
            type(h5td), "to_tensordict", lambda self, **kw: reads.append(1)
        )
        with pytest.raises(ValueError, match=f"Cannot call {op} in_place"):
            getattr(h5td, op)(inplace=True)
        assert not reads
        assert list(h5td.keys(True, True)) == keys
        assert (h5td["a"] == td["a"]).all()
        monkeypatch.undo()
        flat = h5td.flatten_keys()
        assert type(flat) is TensorDict
        assert set(flat.keys()) == {"a", "b", "nested.c", "s"}

        def is_leaf(cls):
            return cls is TensorDict

        flat = h5td.flatten_keys(is_leaf=is_leaf)
        assert list(flat.keys()) == list(td.flatten_keys(is_leaf=is_leaf).keys())

    @pytest.mark.parametrize("idx", [slice(1, 3), 1, torch.tensor([1, 2])])
    def test_setitem_nested_on_locked_h5(self, data, idx):
        # writing rows into a locked file writes the nested group too, as on a
        # locked TensorDict, instead of raising after the top-level entries
        # (#2745)
        td, h5td = data
        td = td.exclude("s").lock_()
        h5td.lock_()
        value = td[idx].apply(torch.zeros_like)
        td[idx] = value
        h5td[idx] = value
        for key in td.keys(True, True):
            assert (h5td.get(key) == td.get(key)).all(), key
        assert (h5td["nested", "c"][idx] == 0).all()

    def test_subtd_rename_on_locked_h5(self, data):
        # a sub-tensordict renames in its locked h5 source, with the source's
        # warning
        td, h5td = data
        h5td.lock_()
        sub = h5td._get_sub_tensordict(0)
        assert sub.is_locked
        with pytest.warns(FutureWarning, match="renames keys of a locked"):
            assert sub.rename_key_("a", "z") is sub
        assert "z" in h5td.keys()
        assert "a" not in h5td.keys()
        h5td.unlock_()
        with pytest.raises(KeyError, match="already present"):
            sub.rename_key_("z", "b", safe=True)

    def test_select_exclude_unsupported(self, data):
        td, h5td = data
        with pytest.raises(
            NotImplementedError,
            match="Cannot call select on a PersistentTensorDict. Call to_tensordict",
        ):
            h5td.select("a")
        with pytest.raises(
            NotImplementedError,
            match="Cannot call exclude on a PersistentTensorDict. Call to_tensordict",
        ):
            h5td.exclude("a")

    def test_popitem(self, data):
        td, h5td = data
        key, value = h5td.popitem()
        assert key == "s"
        assert list(h5td.keys()) == ["a", "b", "nested"]

    def test_tensorclass_iter(self, tmp_path):
        class H5Data(TensorClass):
            a: torch.Tensor

        td = TensorDict(a=torch.arange(3.0), batch_size=[3])
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        tc = H5Data.from_tensordict(h5td)
        # Indexing past the end of a persistent tensordict does not raise, so
        # iteration must stop at the batch size
        elements = list(itertools.islice(tc, 4))
        assert len(elements) == 3
        for i, element in enumerate(elements):
            assert type(element) is H5Data
            assert element.a == td["a"][i]

    @pytest.mark.parametrize("method", ["update_at_", "sub_update_"])
    def test_update_at_nested_keys_to_update(self, tmp_path, method):
        td = TensorDict(
            a=torch.ones(3, 2),
            n=TensorDict(b=torch.ones(3, 2), c=torch.ones(3), batch_size=[3]),
            batch_size=[3],
        )
        h5td = PersistentTensorDict.from_dict(td, filename=tmp_path / "file.h5")
        dest = td.clone().zero_()
        if method == "update_at_":
            dest.update_at_(h5td, slice(0, 3), keys_to_update=[("n", "b")])
        else:
            dest._get_sub_tensordict(slice(0, 3)).update_(
                h5td, keys_to_update=[("n", "b")]
            )
        assert (dest["n", "b"] == 1).all()
        assert (dest["n", "c"] == 0).all()
        assert (dest["a"] == 0).all()

    def test_keys_contains(self, data):
        _, h5td = data
        assert "a" in h5td.keys()
        assert "nested" in h5td.keys()
        assert "nested" not in h5td.keys(True, True)
        assert ("nested", "c") in h5td.keys(True)
        assert ("nested", "c") not in h5td.keys()
        assert "s" in h5td.keys()
        assert "s" in h5td.keys(True, True, is_leaf=_is_leaf_nontensor)
        # "/" is the storage separator, not a valid key character
        assert "nested/c" not in h5td.keys(True)
        # path components the storage library would resolve
        for key in (".", "..", ("nested", "."), ("nested", ""), ("nested", "..")):
            assert key not in h5td.keys(True), key
        assert "missing" not in h5td.keys()
        for include_nested, leaves_only in (
            (False, False),
            (True, False),
            (True, True),
        ):
            keys = h5td.keys(include_nested, leaves_only)
            assert all(key in keys for key in keys)

    def test_missing_key_reads_no_data(self, data, monkeypatch):
        # A membership test for a missing key, and the one in set, must not
        # read the datasets of the file
        _, h5td = data
        reads = []
        getitem = h5py.Dataset.__getitem__

        def counting_getitem(self, *args, **kwargs):
            reads.append(self.name)
            return getitem(self, *args, **kwargs)

        monkeypatch.setattr(h5py.Dataset, "__getitem__", counting_getitem)
        assert "missing" not in h5td.keys()
        assert ("nested", "missing") not in h5td.keys(True)
        assert "a" in h5td.keys()
        h5td.set(("new_group", "x"), torch.zeros(10))
        assert reads == []
        with pytest.raises(KeyError, match="key missing not found"):
            h5td._get_array("missing")
        assert reads == []

    def test_entry_class(self, data):
        _, h5td = data
        assert h5td.entry_class("a") is torch.Tensor
        assert h5td.entry_class("nested") is PersistentTensorDict
        assert h5td.entry_class("s") is NonTensorData
        for key in h5td.keys(True):
            assert h5td.entry_class(key) is type(h5td.get(key)), key


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
