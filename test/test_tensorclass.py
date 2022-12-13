from __future__ import annotations

import argparse
import dataclasses
import re
from typing import Any, Optional, Union

import pytest
import torch

from tensordict import LazyStackedTensorDict, TensorDict
from tensordict.prototype import tensorclass
from tensordict.tensordict import _PermutedTensorDict, _ViewedTensorDict, TensorDictBase
from torch import Tensor


@tensorclass
class MyData:
    X: torch.Tensor
    y: torch.Tensor

    def stuff(self):
        return self.X + self.y


def test_dataclass():

    data = MyData(
        X=torch.ones(3, 4, 5),
        y=torch.zeros(3, 4, 5, dtype=torch.bool),
        batch_size=[3, 4],
    )
    assert dataclasses.is_dataclass(data)


def test_type():

    data = MyData(
        X=torch.ones(3, 4, 5),
        y=torch.zeros(3, 4, 5, dtype=torch.bool),
        batch_size=[3, 4],
    )
    assert isinstance(data, MyData)


def test_banned_types():
    @tensorclass
    class MyAnyClass:
        subclass: Any = None

    data = MyAnyClass(subclass=torch.ones(3, 4), batch_size=[3])
    assert data.subclass is not None

    @tensorclass
    class MyOptAnyClass:
        subclass: Optional[Any] = None

    data = MyOptAnyClass(subclass=torch.ones(3, 4), batch_size=[3])
    assert data.subclass is not None

    @tensorclass
    class MyUnionAnyClass:
        subclass: Union[Any] = None

    data = MyUnionAnyClass(subclass=torch.ones(3, 4), batch_size=[3])
    assert data.subclass is not None

    @tensorclass
    class MyUnionAnyTDClass:
        subclass: Union[Any, TensorDict] = None

    data = MyUnionAnyTDClass(subclass=torch.ones(3, 4), batch_size=[3])
    assert data.subclass is not None

    @tensorclass
    class MyOptionalClass:
        subclass: Optional[TensorDict] = None

    data = MyOptionalClass(subclass=TensorDict({}, [3]), batch_size=[3])
    assert data.subclass is not None

    data = MyOptionalClass(subclass=torch.ones(3), batch_size=[3])
    assert data.subclass is not None

    @tensorclass
    class MyUnionClass:
        subclass: Union[MyOptionalClass, TensorDict] = None

    data = MyUnionClass(
        subclass=MyUnionClass(_tensordict=TensorDict({}, [3])), batch_size=[3]
    )
    with pytest.raises(TypeError, match="can't be deterministically cast."):
        assert data.subclass is not None


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


def test_indexing():
    data = MyData(
        X=torch.ones(3, 4, 5),
        y=torch.zeros(3, 4, 5, dtype=torch.bool),
        batch_size=[3, 4],
    )

    assert data[:2].batch_size == torch.Size([2, 4])
    assert data[1].batch_size == torch.Size([4])
    assert data[1][1].batch_size == torch.Size([])

    with pytest.raises(
        RuntimeError,
        match="indexing a tensordict with td.batch_dims==0 is not permitted",
    ):
        data[1][1][1]

    with pytest.raises(ValueError, match="Invalid indexing arguments."):
        data["X"]


def test_setitem():
    data = MyData(
        X=torch.ones(3, 4, 5),
        y=torch.zeros(3, 4, 5, dtype=torch.bool),
        batch_size=[3, 4],
    )
    data[:2] = data[:2].clone()
    data[[1, 2]] = data[[1, 2]].clone()
    data[0] = data[0].clone()
    data[:, 0] = data[:, 0].clone()
    data[:, [1, 2]] = data[:, [1, 2]].clone()
    with pytest.raises(
        RuntimeError, match="indexed destination TensorDict batch size is"
    ):
        data[:, [1, 2]] = data.clone()


def test_stack():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    data1 = MyData(X=X, y=y, batch_size=batch_size)
    data2 = MyData(X=X, y=y, batch_size=batch_size)
    stacked_tdc = torch.stack([data1, data2], 0)
    assert type(stacked_tdc) is type(data1)
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
    assert type(stacked_tdc) is type(data1)
    assert stacked_tdc.X.shape == torch.Size([6, 4, 5])
    assert (stacked_tdc.X == 1).all()
    assert isinstance(stacked_tdc.tensordict, TensorDict)


def test_unbind():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    data = MyData(X=X, y=y, batch_size=batch_size)
    unbind_tdcs = torch.unbind(data, 0)
    assert type(unbind_tdcs[0]) is type(data)
    assert len(unbind_tdcs) == 3
    assert torch.all(torch.eq(unbind_tdcs[0].X, torch.ones(4, 5)))
    assert unbind_tdcs[0].batch_size == torch.Size([4])


def test_full_like():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    data = MyData(X=X, y=y, batch_size=batch_size)
    full_like_tdc = torch.full_like(data, 9.0)
    assert type(full_like_tdc) is type(data)
    assert full_like_tdc.batch_size == torch.Size(data.batch_size)
    assert full_like_tdc.X.size() == data.X.size()
    assert full_like_tdc.y.size() == data.y.size()
    assert (full_like_tdc.X == 9).all()
    assert full_like_tdc.y.all()


def test_clone():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    data = MyData(X=X, y=y, batch_size=batch_size)
    clone_tdc = torch.clone(data)
    assert clone_tdc.batch_size == torch.Size(data.batch_size)
    assert torch.all(torch.eq(clone_tdc.X, data.X))
    assert torch.all(torch.eq(clone_tdc.y, data.y))


def test_squeeze():
    X = torch.ones(1, 4, 5)
    y = torch.zeros(1, 4, 5, dtype=torch.bool)
    batch_size = [1, 4]
    data = MyData(X=X, y=y, batch_size=batch_size)
    squeeze_tdc = torch.squeeze(data)
    assert squeeze_tdc.batch_size == torch.Size([4])
    assert squeeze_tdc.X.shape == torch.Size([4, 5])
    assert squeeze_tdc.y.shape == torch.Size([4, 5])


def test_unsqueeze():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    data = MyData(X=X, y=y, batch_size=batch_size)
    unsqueeze_tdc = torch.unsqueeze(data, dim=1)
    assert unsqueeze_tdc.batch_size == torch.Size([3, 1, 4])
    assert unsqueeze_tdc.X.shape == torch.Size([3, 1, 4, 5])
    assert unsqueeze_tdc.y.shape == torch.Size([3, 1, 4, 5])


def test_split():
    X = torch.ones(3, 6, 5)
    y = torch.zeros(3, 6, 5, dtype=torch.bool)
    batch_size = [3, 6]
    data = MyData(X=X, y=y, batch_size=batch_size)
    split_tdcs = torch.split(data, split_size_or_sections=[3, 2, 1], dim=1)
    assert type(split_tdcs[1]) is type(data)
    assert split_tdcs[0].batch_size == torch.Size([3, 3])
    assert split_tdcs[1].batch_size == torch.Size([3, 2])
    assert split_tdcs[2].batch_size == torch.Size([3, 1])
    assert torch.all(torch.eq(split_tdcs[0].X, torch.ones(3, 3, 5)))
    assert torch.all(torch.eq(split_tdcs[2].y, torch.zeros(3, 1, 5, dtype=torch.bool)))


def test_reshape():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    data = MyData(X=X, y=y, batch_size=batch_size)
    stacked_tdc = data.reshape(-1)
    assert stacked_tdc.X.shape == torch.Size([12, 5])
    assert stacked_tdc.shape == torch.Size([12])
    assert (stacked_tdc.X == 1).all()
    assert isinstance(stacked_tdc.tensordict, TensorDict)


def test_view():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    data = MyData(X=X, y=y, batch_size=batch_size)
    stacked_tdc = data.view(-1)
    assert stacked_tdc.X.shape == torch.Size([12, 5])
    assert stacked_tdc.shape == torch.Size([12])
    assert (stacked_tdc.X == 1).all()
    assert isinstance(stacked_tdc.tensordict, _ViewedTensorDict)


def test_permute():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    data = MyData(X=X, y=y, batch_size=batch_size)
    stacked_tdc = data.permute(1, 0)
    assert stacked_tdc.X.shape == torch.Size([4, 3, 5])
    assert stacked_tdc.shape == torch.Size([4, 3])
    assert (stacked_tdc.X == 1).all()
    assert isinstance(stacked_tdc.tensordict, _PermutedTensorDict)


def test_nested():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        y: MyDataNested = None

    X = torch.ones(3, 4, 5)
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, batch_size=batch_size)
    assert isinstance(data.y, MyDataNested), type(data.y)


@pytest.mark.parametrize("any_to_td", [True, False])
def test_nested_heterogeneous(any_to_td):
    @tensorclass
    class MyDataNest:
        X: torch.Tensor

    @tensorclass
    class MyDataParent:
        W: Any
        X: Tensor
        z: TensorDictBase
        y: MyDataNest

    batch_size = [3, 4]
    if any_to_td:
        W = TensorDict({}, batch_size)
    else:
        W = torch.zeros(*batch_size, 1)
    X = torch.ones(3, 4, 5)
    data_nest = MyDataNest(X=X, batch_size=batch_size)
    td = TensorDict({}, batch_size)
    data = MyDataParent(X=X, y=data_nest, z=td, W=W, batch_size=batch_size)
    assert isinstance(data.y, MyDataNest)
    assert isinstance(data.y.X, Tensor)
    assert isinstance(data.X, Tensor)
    if not any_to_td:
        assert isinstance(data.W, Tensor)
    else:
        assert isinstance(data.W, TensorDict)
    assert isinstance(data, MyDataParent)
    assert isinstance(data.z, TensorDict)
    assert isinstance(data.tensordict["y"], TensorDict)


@pytest.mark.parametrize("any_to_td", [True, False])
def test_setattr(any_to_td):
    @tensorclass
    class MyDataNest:
        X: torch.Tensor

    @tensorclass
    class MyDataParent:
        W: Any
        X: Tensor
        z: TensorDictBase
        y: MyDataNest

    batch_size = [3, 4]
    if any_to_td:
        W = TensorDict({}, batch_size)
    else:
        W = torch.zeros(*batch_size, 1)
    X = torch.ones(3, 4, 5)
    X_clone = X.clone()
    td = TensorDict({}, batch_size)
    td_clone = td.clone()
    data_nest = MyDataNest(X=X, batch_size=batch_size)
    data = MyDataParent(X=X, y=data_nest, z=td, W=W, batch_size=batch_size)
    data_nest_clone = data_nest.clone()
    assert type(data_nest_clone) is type(data_nest)
    data.y = data_nest_clone
    assert data.tensordict["y"] is not data_nest.tensordict
    assert data.tensordict["y"] is data_nest_clone.tensordict, (
        type(data.tensordict["y"]),
        type(data_nest.tensordict),
    )
    data.X = X_clone
    assert data.tensordict["X"] is X_clone
    data.z = td_clone
    assert data.tensordict["z"] is td_clone
    # check that you can't mess up the batch_size
    with pytest.raises(
        RuntimeError, match=re.escape("the tensor smth has shape torch.Size([1]) which")
    ):
        data.z = TensorDict({"smth": torch.zeros(1)}, [])
    # check that you can't write any attribute
    with pytest.raises(AttributeError, match=re.escape("Cannot set the attribute")):
        data.newattr = TensorDict({"smth": torch.zeros(1)}, [])


def test_default():
    @tensorclass
    class MyData:
        X: torch.Tensor = None  # TODO: do we want to allow any default, say an integer?
        y: torch.Tensor = torch.ones(3, 4, 5)

    data = MyData(batch_size=[3, 4])
    assert data.__dict__["y"] is None
    assert (data.y == 1).all()
    assert data.X is None
    data.X = torch.zeros(3, 4, 1)
    assert (data.X == 0).all()

    MyData(batch_size=[3])
    MyData(batch_size=[])
    with pytest.raises(RuntimeError, match="batch_size are incongruent"):
        MyData(batch_size=[4])


def test_defaultfactory():
    @tensorclass
    class MyData:
        X: torch.Tensor = None  # TODO: do we want to allow any default, say an integer?
        y: torch.Tensor = dataclasses.field(default_factory=torch.ones(3, 4, 5))

    data = MyData(batch_size=[3, 4])
    assert data.__dict__["y"] is None
    assert (data.y == 1).all()
    assert data.X is None
    data.X = torch.zeros(3, 4, 1)
    assert (data.X == 0).all()

    MyData(batch_size=[3])
    MyData(batch_size=[])
    with pytest.raises(RuntimeError, match="batch_size are incongruent"):
        MyData(batch_size=[4])


def test_kjt():
    try:
        from torchrec import KeyedJaggedTensor
    except ImportError:
        pytest.skip("TorchRec not installed.")

    def _get_kjt():
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0])
        keys = ["index_0", "index_1", "index_2"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8, 9, 10, 11])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )
        return jag_tensor

    kjt = _get_kjt()

    @tensorclass
    class MyData:
        X: torch.Tensor
        y: KeyedJaggedTensor

    data = MyData(X=torch.zeros(3, 1), y=kjt, batch_size=[3])
    subdata = data[:2]
    assert (
        subdata.y["index_0"].to_padded_dense() == torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    ).all()

    subdata = data[[0, 2]]
    assert (
        subdata.y["index_0"].to_padded_dense() == torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    ).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
