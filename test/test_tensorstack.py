# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch

from tensordict.tensorstack import LazyStackedTensors as TensorStack


def _tensorstack(stack_dim, nt, init="randint"):
    torch.manual_seed(0)
    if init == "randint":
        x = torch.randint(10, (3, 1, 5))
        y = torch.randint(10, (3, 2, 5))
        z = torch.randint(10, (3, 3, 5))
    elif init == "zeros":
        x = torch.zeros((3, 1, 5))
        y = torch.zeros((3, 2, 5))
        z = torch.zeros((3, 3, 5))
    elif init == "ones":
        x = torch.ones((3, 1, 5))
        y = torch.ones((3, 2, 5))
        z = torch.ones((3, 3, 5))
    if not nt:
        t = TensorStack([x, y, z], stack_dim=stack_dim)
    else:
        t = TensorStack(torch.nested.nested_tensor([x, y, z]), stack_dim=stack_dim)
    return t, (x, y, z)


class TestTensorStack:
    @pytest.mark.parametrize("stack_dim", [0, 1, 2, 3, -1, -2, -3, -4])
    @pytest.mark.parametrize("nt", [True, False])
    def test_indexing_int(self, stack_dim, nt):
        t, (x, y, z) = _tensorstack(stack_dim, nt)
        sd = stack_dim if stack_dim >= 0 else 4 + stack_dim
        init_slice = (slice(None),) * sd
        assert (t[init_slice + (0,)] == x).all()
        assert (t[init_slice + (1,)] == y).all()
        assert (t[init_slice + (2,)] == z).all()

    @pytest.mark.parametrize("stack_dim", [0, 1, 2, 3, -1, -2, -3, -4])
    @pytest.mark.parametrize("nt", [True, False])
    def test_all(self, stack_dim, nt):
        t, (x, y, z) = _tensorstack(stack_dim, nt, "zeros")
        # sd = stack_dim if stack_dim >= 0 else 4 + stack_dim
        # init_slice = (slice(None),) * sd
        assert not t.all()
        assert not t.any()
        t, (x, y, z) = _tensorstack(stack_dim, nt, "ones")
        # sd = stack_dim if stack_dim >= 0 else 4 + stack_dim
        # init_slice = (slice(None),) * sd
        assert t.all()
        assert t.any()

    @pytest.mark.parametrize("nt", [False, True])
    @pytest.mark.parametrize("stack_dim", [0, 1, 2, 3, -1, -2, -3, -4])
    def test_indexing_slice(self, stack_dim, nt):
        t, (x, y, z) = _tensorstack(stack_dim, nt)
        sd = stack_dim if stack_dim >= 0 else 4 + stack_dim
        init_slice = (slice(None),) * sd
        assert (t[init_slice + (slice(1),)][init_slice + (0,)] == x).all(), (
            t[init_slice + (slice(3),)][0],
            x,
        )
        assert (t[init_slice + (slice(2),)][init_slice + (1,)] == y).all()
        assert (t[init_slice + (slice(3),)][init_slice + (2,)] == z).all()
        assert (t[init_slice + (slice(-3, None),)][init_slice + (0,)] == x).all()
        assert (t[init_slice + (slice(-2, None),)][init_slice + (0,)] == y).all()
        assert (t[init_slice + (slice(-1, None),)][init_slice + (0,)] == z).all()

        assert (
            TensorStack([x, y], stack_dim=t.stack_dim) == t[init_slice + (slice(2),)]
        ).all()
        assert (
            TensorStack([y, z], stack_dim=t.stack_dim)
            == t[init_slice + (slice(-2, None),)]
        ).all()
        assert (
            TensorStack([x, z], stack_dim=t.stack_dim)
            == t[init_slice + (slice(0, 3, 2),)]
        ).all()

    @pytest.mark.parametrize("nt", [False, True])
    @pytest.mark.parametrize("stack_dim", [0, 1, 2, 3, -1, -2, -3, -4])
    def test_indexing_range(self, stack_dim, nt):
        t, (x, y, z) = _tensorstack(stack_dim, nt)
        sd = stack_dim if stack_dim >= 0 else 4 + stack_dim
        init_slice = (slice(None),) * sd
        assert (t[init_slice + (slice(1),)][init_slice + (0,)] == x).all(), (
            t[init_slice + (slice(3),)][0],
            x,
        )
        assert (t[init_slice + (range(2),)][init_slice + (1,)] == y).all()
        assert (t[init_slice + (range(3),)][init_slice + (2,)] == z).all()
        assert (t[init_slice + (range(-3, 1),)][init_slice + (0,)] == x).all()
        assert (t[init_slice + (range(-2, 1),)][init_slice + (0,)] == y).all()
        assert (t[init_slice + (range(-1, 1),)][init_slice + (0,)] == z).all()

        assert (
            TensorStack([x, y], stack_dim=t.stack_dim) == t[init_slice + (range(2),)]
        ).all()
        assert (
            TensorStack([y, z], stack_dim=t.stack_dim)
            == t[init_slice + (range(-2, 0),)]
        ).all()
        assert (
            TensorStack([x, z], stack_dim=t.stack_dim)
            == t[init_slice + (range(0, 3, 2),)]
        ).all()

    @pytest.mark.parametrize("nt", [False, True])
    @pytest.mark.parametrize("stack_dim", [0, 1, 2, 3, -1, -2, -3, -4])
    def test_indexing_tensor(self, stack_dim, nt):
        t, (x, y, z) = _tensorstack(stack_dim, nt)
        sd = stack_dim if stack_dim >= 0 else 4 + stack_dim
        init_slice = (slice(None),) * sd
        assert (t[init_slice + (slice(1),)][init_slice + (0,)] == x).all(), (
            t[init_slice + (slice(3),)][0],
            x,
        )
        assert (t[init_slice + (torch.tensor([0, 2]),)][init_slice + (0,)] == x).all()
        assert (t[init_slice + (torch.tensor([0, 2]),)][init_slice + (1,)] == z).all()
        assert (
            t[init_slice + (torch.tensor([0, 2, 0, 2]),)][init_slice + (2,)] == x
        ).all()
        assert (
            t[init_slice + (torch.tensor([0, 2, 0, 2]),)][init_slice + (3,)] == z
        ).all()

        assert (
            t[init_slice + (torch.tensor([[0, 2], [0, 2]]),)][init_slice + (0, 0)] == x
        ).all()
        assert (
            t[init_slice + (torch.tensor([[0, 2], [0, 2]]),)][init_slice + (0, 1)] == z
        ).all()
        assert (
            t[init_slice + (torch.tensor([[0, 2], [0, 2]]),)][init_slice + (1, 0)] == x
        ).all()
        assert (
            t[init_slice + (torch.tensor([[0, 2], [0, 2]]),)][init_slice + (1, 1)] == z
        ).all()

    @pytest.mark.parametrize(
        "transpose",
        [(0, 1), (0, -1), (-1, 0), (1, 3), (1, 2), (2, 1), (0, 2), (2, 0), (2, 2)],
    )
    @pytest.mark.parametrize("het", [False, True])
    @pytest.mark.parametrize("nt", [False, True])
    def test_transpose(self, het, transpose, nt):
        torch.manual_seed(0)
        x = torch.randn(6, 5, 4, 3)
        if het:
            y = torch.randn(6, 5, 2, 3)
        else:
            y = torch.randn(6, 5, 4, 3)
        if nt:
            t = TensorStack(torch.nested.nested_tensor([x, y]), stack_dim=2)
        else:
            t = TensorStack([x, y], stack_dim=2)

        tt = t.transpose(transpose)
        with pytest.raises(ValueError):
            t.transpose(transpose, 0)
        if transpose == (1, 2) or transpose == (2, 1):
            assert (tt[:, 0] == x).all()
            assert (tt[:, 1] == y).all()
        elif transpose == (0, 2) or transpose == (2, 0):
            assert (tt[0] == x.permute(1, 0, 2, 3)).all()
            assert (tt[1] == y.permute(1, 0, 2, 3)).all()
        elif transpose == (2, 2):
            assert (t[:, :, 0] == x).all()
            assert (t[:, :, 1] == y).all()
        elif transpose == (0, 1):
            assert (tt[:, :, 0] == x.transpose(0, 1)).all()
            assert (tt[:, :, 1] == y.transpose(0, 1)).all()
        elif transpose == (0, -1):
            assert (tt[:, :, 0] == x.transpose(0, -1)).all()
            assert (tt[:, :, 1] == y.transpose(0, -1)).all()
        elif transpose == (1, 3):
            assert (tt[:, :, 0] == x.transpose(1, 2)).all()
            assert (tt[:, :, 1] == y.transpose(1, 2)).all()

    def test_permute(self):
        torch.manual_seed(0)
        x = torch.zeros(6, 5, 4, 3)
        y = torch.zeros(6, 5, 4, 3)
        t = TensorStack((x, y), stack_dim=2)
        with pytest.raises(ValueError, match="Got incompatible argument permute_dims"):
            t.permute((1, 2, 3), 0)
        with pytest.raises(ValueError, match="Got incompatible argument permute_dims"):
            t.permute((1, 2, 3, 4, 10))
        with pytest.raises(ValueError, match="permute_dims must have the same length"):
            t.permute((1, 2, 3, 4))
        stack = torch.stack([x, y], 2)
        for _ in range(128):
            pdim = torch.randperm(5).tolist()
            tp = t.permute(pdim)
            assert tp.shape == stack.permute(pdim).shape
            assert (tp == stack.permute(pdim)).all()

    @pytest.mark.parametrize("unbind", range(5))
    @pytest.mark.parametrize("nt", [False, True])
    def test_permute(self, unbind, nt):
        torch.manual_seed(0)
        x = torch.zeros(6, 5, 4, 3)
        y = torch.zeros(6, 5, 4, 3)
        if nt:
            t = TensorStack(torch.nested.nested_tensor([x, y]), stack_dim=2)
        else:
            t = TensorStack([x, y], stack_dim=2)
        stack = torch.stack([x, y], 2)
        for v1, v2 in zip(t.unbind(unbind), stack.unbind(unbind)):
            assert (v1 == v2).all()

    @pytest.mark.parametrize(
        "op",
        ["__add__", "__truediv__", "__mul__", "__sub__", "__mod__", "__eq__", "__ne__"],
    )
    @pytest.mark.parametrize("nt", [False, True])
    @pytest.mark.parametrize("stack_dim", [0])
    def test_indexing_tensor(self, stack_dim, nt, op):
        if nt and op in ("__eq__", "__ne__", "__mod__"):
            # not implemented
            return
        t, (x, y, z) = _tensorstack(stack_dim, nt)
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
        v = torch.ones(17, 1, 1, 1, 5) * 2
        t2 = getattr(t, op)(v)
        assert t2.shape == torch.Size([17, 3, 3, -1, 5])
        torch.testing.assert_close(t2[:, 0], getattr(x, op)(v[:, 0]))
        torch.testing.assert_close(t2[:, 1], getattr(y, op)(v[:, 0]))
        torch.testing.assert_close(t2[:, 2], getattr(z, op)(v[:, 0]))
        # check broadcasting
        assert t2[:, 0].shape == torch.Size((17, *x.shape))

    @pytest.mark.parametrize("nt", [False, True])
    @pytest.mark.parametrize("stack_dim", [0, 1, 2, 3, -3, -2, -1])
    @pytest.mark.parametrize("dim", [0, 1, 2, 3, -3, -2, -1])
    def test_split(self,stack_dim, nt, dim):
        t, (x, y, z) = _tensorstack(stack_dim, nt)
        tsplit = t.split(3, dim)
        assert sum(ts.numel() for ts in tsplit) == t.numel()
        uniques = set()
        for ts in tsplit:
            uniques = uniques.union(ts.unique().tolist())
        assert uniques == set(t.unique().tolist())

    @pytest.mark.parametrize("nt", [False, True])
    @pytest.mark.parametrize("stack_dim", [0, 1, 2, 3, -3, -2, -1])
    def test_reshape(self, stack_dim, nt, dim):
        ...
    @pytest.mark.parametrize("nt", [False, True])
    @pytest.mark.parametrize("stack_dim", [0, 1, 2, 3, -3, -2, -1])
    def test_unique(self, stack_dim, nt, dim):
        ...
    @pytest.mark.parametrize("nt", [False, True])
    @pytest.mark.parametrize("stack_dim", [0, 1, 2, 3, -3, -2, -1])
    def test_view(self, stack_dim, nt, dim):
        ...

if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
