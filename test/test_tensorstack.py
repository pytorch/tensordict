# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch

from tensordict.tenosrstack import TensorStack


@pytest.fixture
def _tensorstack():
    torch.manual_seed(0)
    x = torch.randint(10, (3, 1, 5))
    y = torch.randint(10, (3, 2, 5))
    z = torch.randint(10, (3, 3, 5))
    t = TensorStack.from_tensors([x, y, z])
    return t, (x, y, z)


class TestTensorStack:
    def test_indexing_int(self, _tensorstack):
        t, (x, y, z) = _tensorstack
        assert (t[0] == x).all()
        assert (t[1] == y).all()
        assert (t[2] == z).all()

    def test_indexing_slice(self, _tensorstack):
        t, (x, y, z) = _tensorstack

        assert (t[:3][0] == x).all()
        assert (t[:3][1] == y).all()
        assert (t[:3][2] == z).all()
        assert (t[-3:][0] == x).all()
        assert (t[-3:][1] == y).all()
        assert (t[-3:][2] == z).all()
        assert (t[::-1][0] == z).all()
        assert (t[::-1][1] == y).all()
        assert (t[::-1][2] == x).all()

    def test_indexing_range(self, _tensorstack):
        t, (x, y, z) = _tensorstack
        assert (t[range(3)][0] == x).all()
        assert (t[range(3)][1] == y).all()
        assert (t[range(3)][2] == z).all()
        assert (t[range(1, 3)][0] == y).all()
        assert (t[range(1, 3)][1] == z).all()

    def test_indexing_tensor(self, _tensorstack):
        t, (x, y, z) = _tensorstack
        assert (t[torch.tensor([0, 2])][0] == x).all()
        assert (t[torch.tensor([0, 2])][1] == z).all()
        assert (t[torch.tensor([0, 2, 0, 2])][2] == x).all()
        assert (t[torch.tensor([0, 2, 0, 2])][3] == z).all()
        assert (t[torch.tensor([[0, 2], [0, 2]])][0][0] == x).all()
        assert (t[torch.tensor([[0, 2], [0, 2]])][0][1] == z).all()
        assert (t[torch.tensor([[0, 2], [0, 2]])][1][0] == x).all()
        assert (t[torch.tensor([[0, 2], [0, 2]])][1][1] == z).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
