# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import NonTensorData, PersistentTensorDict, TensorClass, TensorDict
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
        with pytest.raises(KeyError):
            h5td.get_at("missing", 0)

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
