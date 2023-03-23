# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import h5py

import numpy as np
import pytest
import torch

from tensordict import PersistentTensorDict
from torch import multiprocessing as mp
TIMEOUT=100

class TestH5Serialization:
    @classmethod
    def flummoxydoodle(cls, cyberbliptronics, queue):
        assert isinstance(cyberbliptronics, PersistentTensorDict)
        assert cyberbliptronics.file.filename.endswith("groups.hdf5")
        queue.put(
            cyberbliptronics["Base_Group"][
                "Sub_Group",
            ]
        )
        queue.put(cyberbliptronics["Base_Group", "Sub_Group", "default"] + 1)

    def test_h5_serialization(self, tmp_path):
        arr = np.random.randn(1000)
        fn = tmp_path / "groups.hdf5"
        with h5py.File(fn, "w") as f:
            g = f.create_group("Base_Group")
            gg = g.create_group("Sub_Group")

            _ = g.create_dataset("default", data=arr)
            _ = gg.create_dataset("default", data=arr)

        cyberbliptronics = PersistentTensorDict(filename=fn, batch_size=[])
        q = mp.Queue(1)
        p = mp.Process(target=self.flummoxydoodle, args=(cyberbliptronics, q))
        p.start()
        try:
            val = q.get(timeout=TIMEOUT)
            assert (torch.tensor(arr) == val["default"]).all()
            val = q.get(timeout=TIMEOUT)
            assert (torch.tensor(arr) + 1 == val).all()
        finally:
            p.join()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
