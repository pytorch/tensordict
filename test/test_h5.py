# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import numpy as np
import pytest
import torch

from tensordict import PersistentTensorDict
from torch import multiprocessing as mp

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
        q1.put(
            cyberbliptronics["Base_Group"][
                "Sub_Group",
            ]
        )
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
