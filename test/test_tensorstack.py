# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch

from tensordict.tensorstack import TensorStack


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
        # this breaks because the shape backend is a tensor, which cannot be indexed with neg steps
        # assert (t[::-1][0] == z).all()
        # assert (t[::-1][1] == y).all()
        # assert (t[::-1][2] == x).all()

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

        assert (t[torch.tensor([[0, 2], [0, 2]])][0, 0] == x).all()
        assert (t[torch.tensor([[0, 2], [0, 2]])][0, 1] == z).all()
        assert (t[torch.tensor([[0, 2], [0, 2]])][1, 0] == x).all()
        assert (t[torch.tensor([[0, 2], [0, 2]])][1, 1] == z).all()

    def test_indexing_composite(self, _tensorstack):
        _, (x, y, z) = _tensorstack
        t = TensorStack.from_tensors([[x, y, z], [x, y, z]])
        assert (t[0, 0] == x).all()
        assert (t[torch.tensor([0]), torch.tensor([0])] == x).all()
        assert (t[torch.tensor([0]), torch.tensor([1])] == y).all()
        assert (t[torch.tensor([0]), torch.tensor([2])] == z).all()
        assert (t[:, torch.tensor([0])] == x).all()
        assert (t[:, torch.tensor([1])] == y).all()
        assert (t[:, torch.tensor([2])] == z).all()
        assert (
            t[torch.tensor([0]), torch.tensor([1, 2])]
            == TensorStack.from_tensors([y, z])
        ).all()
        with pytest.raises(IndexError, match="Cannot index along"):
            assert (
                t[..., torch.tensor([1, 2]), :, :, :]
                == TensorStack.from_tensors([y, z])
            ).all()

    @pytest.mark.parametrize(
        "op",
        ["__add__", "__truediv__", "__mul__", "__sub__", "__mod__", "__eq__", "__ne__"],
    )
    def test_elementwise(self, _tensorstack, op):
        t, (x, y, z) = _tensorstack
        t2 = getattr(t, op)(2)
        torch.testing.assert_close(t2[0], getattr(x, op)(2))
        torch.testing.assert_close(t2[1], getattr(y, op)(2))
        torch.testing.assert_close(t2[2], getattr(z, op)(2))
        t2 = getattr(t, op)(torch.ones(5) * 2)
        torch.testing.assert_close(t2[0], getattr(x, op)(torch.ones(5) * 2))
        torch.testing.assert_close(t2[1], getattr(y, op)(torch.ones(5) * 2))
        torch.testing.assert_close(t2[2], getattr(z, op)(torch.ones(5) * 2))
        # check broadcasting
        assert t2[0].shape == x.shape
        v = torch.ones(2, 1, 1, 1, 5) * 2
        t2 = getattr(t, op)(v)
        assert t2.shape == torch.Size([2, 3, 3, -1, 5])
        torch.testing.assert_close(t2[:, 0], getattr(x, op)(v[:, 0]))
        torch.testing.assert_close(t2[:, 1], getattr(y, op)(v[:, 0]))
        torch.testing.assert_close(t2[:, 2], getattr(z, op)(v[:, 0]))
        # check broadcasting
        assert t2[:, 0].shape == torch.Size((2, *x.shape))

    def test_permute(self):
        w = torch.randint(10, (3, 5, 5))
        x = torch.randint(10, (3, 4, 5))
        y = torch.randint(10, (3, 5, 5))
        z = torch.randint(10, (3, 4, 5))
        ts = TensorStack.from_tensors([[w, x], [y, z]])
        tst = ts.permute(1, 0, 2, 3, 4)
        assert (tst[0, 1] == ts[1, 0]).all()
        assert (tst[1, 0] == ts[0, 1]).all()
        assert (tst[1, 1] == ts[1, 1]).all()
        assert (tst[0, 0] == ts[0, 0]).all()

    def test_transpose(self):
        w = torch.randint(10, (3, 5, 5))
        x = torch.randint(10, (3, 4, 5))
        y = torch.randint(10, (3, 5, 5))
        z = torch.randint(10, (3, 4, 5))
        ts = TensorStack.from_tensors([[w, x], [y, z]])
        tst = ts.transpose(1, 0)
        assert (tst[0, 1] == ts[1, 0]).all()
        assert (tst[1, 0] == ts[0, 1]).all()
        assert (tst[1, 1] == ts[1, 1]).all()
        assert (tst[0, 0] == ts[0, 0]).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
