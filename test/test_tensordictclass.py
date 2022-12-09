from __future__ import annotations

import argparse
from typing import Optional

import pytest
import torch

from tensordict import LazyStackedTensorDict, TensorDict
from tensordict.prototype import tensordictclass
from tensordict.tensordict import _PermutedTensorDict, _ViewedTensorDict


@tensordictclass
class MyData:
    X: torch.tensor
    y: torch.tensor

    def stuff(self):
        return self.X + self.y


def test_type():

    data = MyData(
        X=torch.ones(3, 4, 5),
        y=torch.zeros(3, 4, 5, dtype=torch.bool),
        batch_size=[3, 4],
    )
    assert isinstance(data, MyData)


def test_attributes():

    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    tensordict = TensorDict(
        {
            "X": X,
            "y": y,
        },
        batch_size=[3, 4],
    )

    data = MyData(X=X, y=y, batch_size=batch_size)

    equality_tensordict = data.tensordict == tensordict

    assert torch.equal(data.X, X)
    assert torch.equal(data.y, y)
    assert data.batch_size == batch_size
    assert equality_tensordict.all()
    assert equality_tensordict.batch_size == torch.Size(batch_size)


def test_stack():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    data1 = MyData(X=X, y=y, batch_size=batch_size)
    data2 = MyData(X=X, y=y, batch_size=batch_size)
    stacked_tdc = torch.stack([data1, data2], 0)
    assert stacked_tdc.X.shape == torch.Size([2, 3, 4, 5])
    assert (stacked_tdc.X == 1).all()
    assert isinstance(stacked_tdc.tensordict, LazyStackedTensorDict)


def test_cat():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    data1 = MyData(X=X, y=y, batch_size=batch_size)
    data2 = MyData(X=X, y=y, batch_size=batch_size)
    stacked_tdc = torch.cat([data1, data2], 0)
    assert stacked_tdc.X.shape == torch.Size([6, 4, 5])
    assert (stacked_tdc.X == 1).all()
    assert isinstance(stacked_tdc.tensordict, TensorDict)


def test_reshape():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    data = MyData(X=X, y=y, batch_size=batch_size)
    stacked_tdc = data.reshape(-1)
    assert stacked_tdc.X.shape == torch.Size([12, 5])
    assert (stacked_tdc.X == 1).all()
    assert isinstance(stacked_tdc.tensordict, TensorDict)


def test_view():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    data = MyData(X=X, y=y, batch_size=batch_size)
    stacked_tdc = data.view(-1)
    assert stacked_tdc.X.shape == torch.Size([12, 5])
    assert (stacked_tdc.X == 1).all()
    assert isinstance(stacked_tdc.tensordict, _ViewedTensorDict)


def test_permute():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    data = MyData(X=X, y=y, batch_size=batch_size)
    stacked_tdc = data.permute(1, 0)
    assert stacked_tdc.X.shape == torch.Size([4, 3, 5])
    assert (stacked_tdc.X == 1).all()
    assert isinstance(stacked_tdc.tensordict, _PermutedTensorDict)


def test_nested():
    @tensordictclass
    class MyDataNested:
        X: torch.tensor
        batch_size: torch.Size
        y: Optional[MyDataNested] = None

    X = torch.ones(3, 4, 5)
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, batch_size=batch_size)
    assert isinstance(data.y, MyDataNested), type(data.y)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
