# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import NonTensorData, PersistentTensorDict, TensorDict
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
            torch.tensor([True, False] * 5),
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
