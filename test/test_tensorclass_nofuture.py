import argparse
import dataclasses
import inspect
import os
import pickle
import re
from multiprocessing import Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Union

import pytest
import torch
import torchsnapshot

from _utils_internal import get_available_devices

from tensordict import LazyStackedTensorDict, MemmapTensor, TensorDict
from tensordict.prototype import is_tensorclass, tensorclass
from tensordict.tensordict import (
    _PermutedTensorDict,
    _ViewedTensorDict,
    assert_allclose_td,
    TensorDictBase,
)
from torch import Tensor


class MyData:
    X: torch.Tensor
    y: torch.Tensor
    z: str

    def stuff(self):
        return self.X + self.y


# this slightly convoluted construction of MyData allows us to check that instances of
# the tensorclass are instances of the original class.
MyDataUndecorated, MyData = MyData, tensorclass(MyData)


@tensorclass
class MyData2:
    X: torch.Tensor
    y: torch.Tensor
    z: list


def test_dataclass():
    data = MyData(
        X=torch.ones(3, 4, 5),
        y=torch.zeros(3, 4, 5, dtype=torch.bool),
        z="test_tensorclass",
        batch_size=[3, 4],
    )
    assert dataclasses.is_dataclass(data)


def test_type():
    data = MyData(
        X=torch.ones(3, 4, 5),
        y=torch.zeros(3, 4, 5, dtype=torch.bool),
        z="test_tensorclass",
        batch_size=[3, 4],
    )
    assert isinstance(data, MyData)
    assert is_tensorclass(data)
    assert is_tensorclass(MyData)
    # we get an instance of the user defined class, not a dynamically defined subclass
    assert type(data) is MyDataUndecorated


def test_signature():
    sig = inspect.signature(MyData)
    assert list(sig.parameters) == ["X", "y", "z", "batch_size", "device"]

    with pytest.raises(TypeError, match="missing 3 required positional arguments"):
        MyData(batch_size=[10])

    with pytest.raises(TypeError, match="missing 2 required positional argument"):
        MyData(X=torch.rand(10), batch_size=[10])

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        MyData(X=torch.rand(10), y=torch.rand(10), batch_size=[10], device="cpu")

    # if all positional arguments are specified, ommitting batch_size gives error
    with pytest.raises(
        TypeError, match="missing 1 required keyword-only argument: 'batch_size'"
    ):
        MyData(X=torch.rand(10), y=torch.rand(10))

    # all positional arguments + batch_size is fine
    MyData(X=torch.rand(10), y=torch.rand(10), z="test_tensorclass", batch_size=[10])


@pytest.mark.parametrize("device", get_available_devices())
def test_device(device):
    data = MyData(
        X=torch.ones(3, 4, 5),
        y=torch.zeros(3, 4, 5, dtype=torch.bool),
        z="test_tensorclass",
        batch_size=[3, 4],
        device=device,
    )
    assert data.device == device
    assert data.X.device == device
    assert data.y.device == device

    with pytest.raises(AttributeError, match="'str' object has no attribute 'device'"):
        assert data.z.device == device

    with pytest.raises(
        RuntimeError, match="device cannot be set using tensorclass.device = device"
    ):
        data.device = torch.device("cpu")


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
        subclass=MyUnionClass._from_tensordict(TensorDict({}, [3])), batch_size=[3]
    )
    assert data.subclass is not None


def test_attributes():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    z = "test_tensorclass"
    tensordict = TensorDict(
        {
            "X": X,
            "y": y,
        },
        batch_size=[3, 4],
    )

    data = MyData(X=X, y=y, z=z, batch_size=batch_size)

    equality_tensordict = data._tensordict == tensordict

    assert torch.equal(data.X, X)
    assert torch.equal(data.y, y)
    assert data.batch_size == torch.Size(batch_size)
    assert equality_tensordict.all()
    assert equality_tensordict.batch_size == torch.Size(batch_size)
    assert data.z == z


def test_disallowed_attributes():
    with pytest.raises(
        AttributeError,
        match="Attribute name reshape can't be used with @tensorclass",
    ):

        @tensorclass
        class MyInvalidClass:
            x: torch.Tensor
            y: torch.Tensor
            reshape: torch.Tensor


def test_batch_size():
    myc = MyData(
        X=torch.rand(2, 3, 4),
        y=torch.rand(2, 3, 4, 5),
        z="test_tensorclass",
        batch_size=[2, 3],
    )

    assert myc.batch_size == torch.Size([2, 3])
    assert myc.X.shape == torch.Size([2, 3, 4])

    myc.batch_size = torch.Size([2])

    assert myc.batch_size == torch.Size([2])
    assert myc.X.shape == torch.Size([2, 3, 4])


def test_len():
    myc = MyData(
        X=torch.rand(2, 3, 4),
        y=torch.rand(2, 3, 4, 5),
        z="test_tensorclass",
        batch_size=[2, 3],
    )
    assert len(myc) == 2

    myc2 = MyData(
        X=torch.rand(2, 3, 4),
        y=torch.rand(2, 3, 4, 5),
        z="test_tensorclass",
        batch_size=[],
    )
    assert len(myc2) == 0


def test_indexing():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: list
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = ["a", "b", "c"]
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)

    assert data[:2].batch_size == torch.Size([2, 4])
    assert data[:2].X.shape == torch.Size([2, 4, 5])
    assert (data[:2].X == X[:2]).all()
    assert isinstance(data[:2].y, type(data_nest))

    # Nested tensors all get indexed
    assert (data[:2].y.X == X[:2]).all()
    assert data[:2].y.batch_size == torch.Size([2, 4])
    assert data[1].batch_size == torch.Size([4])
    assert data[1][1].batch_size == torch.Size([])

    # Non-tensor data won't get indexed
    assert data[1].z == data[2].z == data[:2].z == z

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
        y=torch.zeros(3, 4, 5),
        z="test_tensorclass",
        batch_size=[3, 4],
    )

    x = torch.randn(3, 4, 5)
    y = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data2 = MyData(X=x, y=y, z=z, batch_size=batch_size)
    data3 = MyData(X=y, y=x, z=z, batch_size=batch_size)

    # Testing the data before setting
    assert (data[:2].X == torch.ones(2, 4, 5)).all()
    assert (data[:2].y == torch.zeros(2, 4, 5)).all()
    assert data[:2].z == "test_tensorclass"
    assert (data[[1, 2]].X == torch.ones(5)).all()

    # Setting the item and testing post setting the item
    data[:2] = data2[:2].clone()
    assert (data[:2].X == data2[:2].X).all()
    assert (data[:2].y == data2[:2].y).all()
    assert data[:2].z == z

    data[[1, 2]] = data3[[1, 2]].clone()
    assert (data[[1, 2]].X == data3[[1, 2]].X).all()
    assert (data[[1, 2]].y == data3[[1, 2]].y).all()
    assert data[[1, 2]].z == z

    data[:, [1, 2]] = data2[:, [1, 2]].clone()
    assert (data[:, [1, 2]].X == data2[:, [1, 2]].X).all()
    assert (data[:, [1, 2]].y == data[:, [1, 2]].y).all()
    assert data[:, [1, 2]].z == z

    with pytest.raises(
        RuntimeError, match="indexed destination TensorDict batch size is"
    ):
        data[:, [1, 2]] = data.clone()

    # Negative testcase for non-tensor data
    z = "test_bluff"
    data2 = MyData(X=x, y=y, z=z, batch_size=batch_size)
    with pytest.warns(
        UserWarning,
        match="Meta data at 'z' may or may not be equal, this may result in undefined behaviours",
    ):
        data[1] = data2[1]

    # Validating nested test cases
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: list
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.randn(3, 4, 5)
    z = ["a", "b", "c"]
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    X2 = torch.ones(3, 4, 5)
    data_nest2 = MyDataNested(X=X2, z=z, batch_size=batch_size)
    data2 = MyDataNested(X=X2, y=data_nest2, z=z, batch_size=batch_size)
    data[:2] = data2[:2].clone()
    assert (data[:2].X == data2[:2].X).all()
    assert (data[:2].y.X == data2[:2].y.X).all()
    assert data[:2].z == z

    # Negative Scenario
    data3 = MyDataNested(X=X2, y=data_nest2, z=["e", "f"], batch_size=batch_size)
    with pytest.warns(
        UserWarning,
        match="Meta data at 'z' may or may not be equal, this may result in undefined behaviours",
    ):
        data[:2] = data3[:2]


def test_setitem_memmap():
    # regression test PR #203
    # We should be able to set tensors items with MemmapTensors and viceversa
    @tensorclass
    class MyDataMemMap1:
        x: torch.Tensor
        y: MemmapTensor

    data1 = MyDataMemMap1(
        x=torch.zeros(3, 4, 5),
        y=MemmapTensor.from_tensor(torch.zeros(3, 4, 5)),
        batch_size=[3, 4],
    )

    data2 = MyDataMemMap1(
        x=MemmapTensor.from_tensor(torch.ones(3, 4, 5)),
        y=torch.ones(3, 4, 5),
        batch_size=[3, 4],
    )

    data1[:2] = data2[:2]
    assert (data1[:2] == 1).all()
    assert (data1.x[:2] == 1).all()
    assert (data1.y[:2] == 1).all()
    data2[2:] = data1[2:]
    assert (data2[2:] == 0).all()
    assert (data2.x[2:] == 0).all()
    assert (data2.y[2:] == 0).all()


def test_setitem_other_cls():
    @tensorclass
    class MyData1:
        x: torch.Tensor
        y: MemmapTensor

    data1 = MyData1(
        x=torch.zeros(3, 4, 5),
        y=MemmapTensor.from_tensor(torch.zeros(3, 4, 5)),
        batch_size=[3, 4],
    )

    # Set Item should work for other tensorclass
    @tensorclass
    class MyData2:
        x: MemmapTensor
        y: torch.Tensor

    data_other_cls = MyData2(
        x=MemmapTensor.from_tensor(torch.ones(3, 4, 5)),
        y=torch.ones(3, 4, 5),
        batch_size=[3, 4],
    )
    data1[:2] = data_other_cls[:2]
    data_other_cls[2:] = data1[2:]

    # Set Item should raise if other tensorclass with different members
    @tensorclass
    class MyData3:
        x: MemmapTensor
        z: torch.Tensor

    data_wrong_cls = MyData3(
        x=MemmapTensor.from_tensor(torch.ones(3, 4, 5)),
        z=torch.ones(3, 4, 5),
        batch_size=[3, 4],
    )
    with pytest.raises(
        ValueError,
        match="__setitem__ is only allowed for same-class or compatible class .* assignment",
    ):
        data1[:2] = data_wrong_cls[:2]
    with pytest.raises(
        ValueError,
        match="__setitem__ is only allowed for same-class or compatible class .* assignment",
    ):
        data_wrong_cls[2:] = data1[2:]

@pytest.mark.parametrize(
    "val2broadcast", 
    [0, torch.zeros(4, 5), TensorDict(X=torch.zeros(2, 4, 5), batch_size=[2, 4]), MemmapTensor.from_tensor(torch.zeros(4, 5))],
)
def test_setitem_broadcast():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: list
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = ["a", "b", "c"]
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)

    # scalar
    data[:2] = val2broadcast
    assert (data[:2] == 0).all()
    assert (data.X[:2] == 0).all()
    assert (data.y.X[:2] == 0).all()


def test_stack():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data1 = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    data2 = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)

    stacked_tc = torch.stack([data1, data2], 0)
    assert type(stacked_tc) is type(data1)
    assert isinstance(stacked_tc.y, type(data1.y))
    assert stacked_tc.X.shape == torch.Size([2, 3, 4, 5])
    assert stacked_tc.y.X.shape == torch.Size([2, 3, 4, 5])
    assert (stacked_tc.X == 1).all()
    assert (stacked_tc.y.X == 1).all()
    assert isinstance(stacked_tc._tensordict, LazyStackedTensorDict)
    assert isinstance(stacked_tc.y._tensordict, LazyStackedTensorDict)
    assert stacked_tc.z == stacked_tc.y.z == z

    # Testing negative scenarios
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    data3 = MyData(X=X, y=y, z=z, batch_size=batch_size)

    with pytest.raises(
        TypeError,
        match=(
            "no implementation found for 'torch.stack' on types that implement "
            "__torch_function__"
        ),
    ):
        torch.stack([data1, data3], dim=0)


def test_cat():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data1 = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    data2 = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)

    catted_tc = torch.cat([data1, data2], 0)
    assert type(catted_tc) is type(data1)
    assert isinstance(catted_tc.y, type(data1.y))
    assert catted_tc.X.shape == torch.Size([6, 4, 5])
    assert catted_tc.y.X.shape == torch.Size([6, 4, 5])
    assert (catted_tc.X == 1).all()
    assert (catted_tc.y.X == 1).all()
    assert isinstance(catted_tc._tensordict, TensorDict)
    assert catted_tc.z == catted_tc.y.z == z

    # Testing negative scenarios
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    data3 = MyData(X=X, y=y, z=z, batch_size=batch_size)

    with pytest.raises(
        TypeError,
        match=(
            "no implementation found for 'torch.cat' on types that implement "
            "__torch_function__"
        ),
    ):
        torch.cat([data1, data3], dim=0)


def test_unbind():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    unbind_tcs = torch.unbind(data, 0)
    assert type(unbind_tcs[1]) is type(data)
    assert type(unbind_tcs[0].y[0]) is type(data)
    assert len(unbind_tcs) == 3
    assert torch.all(torch.eq(unbind_tcs[0].X, torch.ones(4, 5)))
    assert torch.all(torch.eq(unbind_tcs[0].y[0].X, torch.ones(4, 5)))
    assert unbind_tcs[0].batch_size == torch.Size([4])
    assert unbind_tcs[0].z == unbind_tcs[1].z == unbind_tcs[2].z == z


def test_full_like():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    full_like_tc = torch.full_like(data, 9.0)
    assert type(full_like_tc) is type(data)
    assert full_like_tc.batch_size == torch.Size(data.batch_size)
    assert full_like_tc.X.size() == data.X.size()
    assert isinstance(full_like_tc.y, type(data.y))
    assert full_like_tc.y.X.size() == data.y.X.size()
    assert (full_like_tc.X == 9).all()
    assert (full_like_tc.y.X == 9).all()
    assert full_like_tc.z == data.z == z


def test_clone():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    clone_tc = torch.clone(data)
    assert clone_tc.batch_size == torch.Size(data.batch_size)
    assert torch.all(torch.eq(clone_tc.X, data.X))
    assert isinstance(clone_tc.y, MyDataNested)
    assert torch.all(torch.eq(clone_tc.y.X, data.y.X))
    assert clone_tc.z == data.z == z


def test_squeeze():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(1, 4, 5)
    z = "test_tensorclass"
    batch_size = [1, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    squeeze_tc = torch.squeeze(data)
    assert squeeze_tc.batch_size == torch.Size([4])
    assert squeeze_tc.X.shape == torch.Size([4, 5])
    assert squeeze_tc.y.X.shape == torch.Size([4, 5])
    assert squeeze_tc.z == squeeze_tc.y.z == z


def test_unsqueeze():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    unsqueeze_tc = torch.unsqueeze(data, dim=1)
    assert unsqueeze_tc.batch_size == torch.Size([3, 1, 4])
    assert unsqueeze_tc.X.shape == torch.Size([3, 1, 4, 5])
    assert unsqueeze_tc.y.X.shape == torch.Size([3, 1, 4, 5])
    assert unsqueeze_tc.z == unsqueeze_tc.y.z == z


def test_split():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None

    X = torch.ones(3, 6, 5)
    z = "test_tensorclass"
    batch_size = [3, 6]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyData(X=X, y=data_nest, z=z, batch_size=batch_size)
    split_tcs = torch.split(data, split_size_or_sections=[3, 2, 1], dim=1)
    assert type(split_tcs[1]) is type(data)
    assert split_tcs[0].batch_size == torch.Size([3, 3])
    assert split_tcs[1].batch_size == torch.Size([3, 2])
    assert split_tcs[2].batch_size == torch.Size([3, 1])
    assert split_tcs[0].y[0].batch_size == torch.Size([3, 3])
    assert split_tcs[0].y[1].batch_size == torch.Size([3, 2])
    assert split_tcs[0].y[2].batch_size == torch.Size([3, 1])
    assert torch.all(torch.eq(split_tcs[0].X, torch.ones(3, 3, 5)))
    assert torch.all(torch.eq(split_tcs[0].y[0].X, torch.ones(3, 3, 5)))
    assert split_tcs[0].z == split_tcs[1].z == split_tcs[2].z == z
    assert split_tcs[0].y[0].z == split_tcs[0].y[1].z == split_tcs[0].y[2].z == z


def test_reshape():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    stacked_tc = data.reshape(-1)
    assert stacked_tc.X.shape == torch.Size([12, 5])
    assert stacked_tc.y.X.shape == torch.Size([12, 5])
    assert stacked_tc.shape == torch.Size([12])
    assert (stacked_tc.X == 1).all()
    assert isinstance(stacked_tc._tensordict, TensorDict)
    assert stacked_tc.z == stacked_tc.y.z == z


def test_view():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    stacked_tc = data.view(-1)
    assert stacked_tc.X.shape == torch.Size([12, 5])
    assert stacked_tc.y.X.shape == torch.Size([12, 5])
    assert stacked_tc.shape == torch.Size([12])
    assert (stacked_tc.X == 1).all()
    assert isinstance(stacked_tc._tensordict, _ViewedTensorDict)
    assert stacked_tc.z == stacked_tc.y.z == z


def test_permute():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    stacked_tc = data.permute(1, 0)
    assert stacked_tc.X.shape == torch.Size([4, 3, 5])
    assert stacked_tc.y.X.shape == torch.Size([4, 3, 5])
    assert stacked_tc.shape == torch.Size([4, 3])
    assert (stacked_tc.X == 1).all()
    assert isinstance(stacked_tc._tensordict, _PermutedTensorDict)
    assert stacked_tc.z == stacked_tc.y.z == z


def test_nested():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    assert isinstance(data.y, MyDataNested), type(data.y)
    assert data.z == data_nest.z == data.y.z == z


def test_nested_eq():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    data_nest2 = MyDataNested(X=X, z=z, batch_size=batch_size)
    data2 = MyDataNested(X=X, y=data_nest2, z=z, batch_size=batch_size)
    assert (data == data2).all()
    assert (data == data2).X.all()
    assert (data == data2).z is None
    assert (data == data2).y.X.all()
    assert (data == data2).y.z is None


def test_nested_ne():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: "MyDataNested" = None  # future: drop quotes

    X = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    data_nest2 = MyDataNested(X=X, z=z, batch_size=batch_size)
    z = "test_bluff"
    data2 = MyDataNested(X=X + 1, y=data_nest2, z=z, batch_size=batch_size)
    assert (data != data2).any()
    assert (data != data2).X.all()
    assert (data != data2).z is None
    assert not (data != data2).y.X.any()
    assert (data != data2).y.z is None


def test_args():
    @tensorclass
    class MyData:
        D: torch.Tensor
        B: torch.Tensor
        A: torch.Tensor
        C: torch.Tensor
        E: str

    D = torch.ones(3, 4, 5)
    B = torch.ones(3, 4, 5)
    A = torch.ones(3, 4, 5)
    C = torch.ones(3, 4, 5)
    E = "test_tensorclass"
    data1 = MyData(D, B=B, A=A, C=C, E=E, batch_size=[3, 4])
    data2 = MyData(D, B, A=A, C=C, E=E, batch_size=[3, 4])
    data3 = MyData(D, B, A, C=C, E=E, batch_size=[3, 4])
    data4 = MyData(D, B, A, C, E=E, batch_size=[3, 4])
    data5 = MyData(D, B, A, C, E, batch_size=[3, 4])
    data = torch.stack([data1, data2, data3, data4, data5], 0)
    assert (data.A == A).all()
    assert (data.B == B).all()
    assert (data.C == C).all()
    assert (data.D == D).all()
    assert data.E == E


@pytest.mark.parametrize("any_to_td", [True, False])
def test_nested_heterogeneous(any_to_td):
    @tensorclass
    class MyDataNest:
        X: torch.Tensor
        v: str

    @tensorclass
    class MyDataParent:
        W: Any
        X: Tensor
        z: TensorDictBase
        y: MyDataNest
        v: str

    batch_size = [3, 4]
    if any_to_td:
        W = TensorDict({}, batch_size)
    else:
        W = torch.zeros(*batch_size, 1)
    X = torch.ones(3, 4, 5)
    data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
    td = TensorDict({}, batch_size)
    v = "test_tensorclass"
    data = MyDataParent(X=X, y=data_nest, z=td, W=W, v=v, batch_size=batch_size)
    assert isinstance(data.y, MyDataNest)
    assert isinstance(data.y.X, Tensor)
    assert isinstance(data.X, Tensor)
    if not any_to_td:
        assert isinstance(data.W, Tensor)
    else:
        assert isinstance(data.W, TensorDict)
    assert isinstance(data, MyDataParent)
    assert isinstance(data.z, TensorDict)
    assert data.v == v
    assert data.y.v == "test_nested"
    # Testing nested indexing
    assert isinstance(data[0], type(data))
    assert isinstance(data[0].y, type(data.y))
    assert data[0].y.X.shape == torch.Size([4, 5])


@pytest.mark.parametrize("any_to_td", [True, False])
def test_getattr(any_to_td):
    @tensorclass
    class MyDataNest:
        X: torch.Tensor
        v: str

    @tensorclass
    class MyDataParent:
        W: Any
        X: Tensor
        z: TensorDictBase
        y: MyDataNest
        v: str

    batch_size = [3, 4]
    if any_to_td:
        W = TensorDict({}, batch_size)
    else:
        W = torch.zeros(*batch_size, 1)
    X = torch.ones(3, 4, 5)
    td = TensorDict({}, batch_size)
    data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
    v = "test_tensorclass"
    data = MyDataParent(X=X, y=data_nest, z=td, W=W, v=v, batch_size=batch_size)
    assert isinstance(data.y, type(data_nest))
    assert (data.X == X).all()
    assert data.batch_size == torch.Size(batch_size)
    assert data.v == v
    assert (data.z == td).all()
    assert (data.W == W).all()

    # Testing nested tensor class
    assert data.y._tensordict is data_nest._tensordict
    assert (data.y.X == X).all()
    assert data.y.v == "test_nested"
    assert data.y.batch_size == torch.Size(batch_size)


@pytest.mark.parametrize("any_to_td", [True, False])
def test_setattr(any_to_td):
    @tensorclass
    class MyDataNest:
        X: torch.Tensor
        v: str

    @tensorclass
    class MyDataParent:
        W: Any
        X: Tensor
        z: TensorDictBase
        y: MyDataNest
        v: Any

    batch_size = [3, 4]
    if any_to_td:
        W = TensorDict({}, batch_size)
    else:
        W = torch.zeros(*batch_size, 1)
    X = torch.ones(3, 4, 5)
    td = TensorDict({}, batch_size)
    data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
    data = MyDataParent(
        X=X, y=data_nest, z=td, W=W, v="test_tensorclass", batch_size=batch_size
    )
    assert isinstance(data.y, type(data_nest))
    assert data.y._tensordict is data_nest._tensordict
    data.X = torch.zeros(3, 4, 5)
    assert (data.X == torch.zeros(3, 4, 5)).all()
    v_new = "test_bluff"
    data.v = v_new
    assert data.v == v_new
    # check that you can't mess up the batch_size
    with pytest.raises(
        RuntimeError, match=re.escape("the tensor smth has shape torch.Size([1]) which")
    ):
        data.z = TensorDict({"smth": torch.zeros(1)}, [])
    # check that you can't write any attribute
    with pytest.raises(AttributeError, match=re.escape("Cannot set the attribute")):
        data.newattr = TensorDict({"smth": torch.zeros(1)}, [])
    # Testing nested cases
    data_nest.X = torch.zeros(3, 4, 5)
    assert (data_nest.X == torch.zeros(3, 4, 5)).all()
    assert (data.y.X == torch.zeros(3, 4, 5)).all()
    assert data.y.v == "test_nested"
    data.y.v = "test_nested_new"
    assert data.y.v == data_nest.v == "test_nested_new"
    data_nest.v = "test_nested"
    assert data_nest.v == data.y.v == "test_nested"

    # Testing if user can override the type of the attribute
    data.v = torch.ones(3, 4, 5)
    assert (data.v == torch.ones(3, 4, 5)).all()
    assert "v" in data._tensordict.keys()
    assert "v" not in data._non_tensordict.keys()

    data.v = "test"
    assert data.v == "test"
    assert "v" not in data._tensordict.keys()
    assert "v" in data._non_tensordict.keys()


def test_pre_allocate():
    @tensorclass
    class M1:
        X: Any

    @tensorclass
    class M2:
        X: Any

    @tensorclass
    class M3:
        X: Any

    m1 = M1(M2(M3(X=None, batch_size=[4]), batch_size=[4]), batch_size=[4])
    m2 = M1(M2(M3(X=torch.randn(2), batch_size=[]), batch_size=[]), batch_size=[])
    assert m1.X.X.X is None
    m1[0] = m2
    assert (m1[0].X.X.X == m2.X.X.X).all()


def test_post_init():
    @tensorclass
    class MyDataPostInit:
        X: torch.Tensor
        y: torch.Tensor

        def __post_init__(self):
            assert (self.X > 0).all()
            assert self.y.abs().max() <= 10
            self.y = self.y.abs()

    y = torch.clamp(torch.randn(3, 4), min=-10, max=10)
    data = MyDataPostInit(X=torch.rand(3, 4), y=y, batch_size=[3, 4])
    assert (data.y == y.abs()).all()

    # initialising from tensordict is fine
    data = MyDataPostInit._from_tensordict(
        TensorDict({"X": torch.rand(3, 4), "y": y}, batch_size=[3, 4])
    )

    with pytest.raises(AssertionError):
        MyDataPostInit(X=-torch.ones(2), y=torch.rand(2), batch_size=[2])

    with pytest.raises(AssertionError):
        MyDataPostInit._from_tensordict(
            TensorDict({"X": -torch.ones(2), "y": torch.rand(2)}, batch_size=[2])
        )


def test_default():
    @tensorclass
    class MyData:
        X: torch.Tensor = None  # TODO: do we want to allow any default, say an integer?
        y: torch.Tensor = torch.ones(3, 4, 5)

    data = MyData(batch_size=[3, 4])
    assert (data.y == 1).all()
    assert data.X is None
    data.X = torch.zeros(3, 4, 1)
    assert (data.X == 0).all()

    MyData(batch_size=[3])
    MyData(batch_size=[])
    with pytest.raises(RuntimeError, match="batch dimension mismatch"):
        MyData(batch_size=[4])


def test_defaultfactory():
    @tensorclass
    class MyData:
        X: torch.Tensor = None  # TODO: do we want to allow any default, say an integer?
        y: torch.Tensor = dataclasses.field(default_factory=lambda: torch.ones(3, 4, 5))

    data = MyData(batch_size=[3, 4])
    assert (data.y == 1).all()
    assert data.X is None
    data.X = torch.zeros(3, 4, 1)
    assert (data.X == 0).all()

    MyData(batch_size=[3])
    MyData(batch_size=[])
    with pytest.raises(RuntimeError, match="batch dimension mismatch"):
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
        z: str

    z = "test_tensorclass"
    data = MyData(X=torch.zeros(3, 1), y=kjt, z=z, batch_size=[3])
    subdata = data[:2]
    assert (
        subdata.y["index_0"].to_padded_dense() == torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    ).all()

    subdata = data[[0, 2]]
    assert (
        subdata.y["index_0"].to_padded_dense() == torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    ).all()
    assert subdata.z == data.z == z


def test_pickle():
    data = MyData(
        X=torch.ones(3, 4, 5),
        y=torch.zeros(3, 4, 5, dtype=torch.bool),
        z="test_tensorclass",
        batch_size=[3, 4],
    )

    with TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)

        with open(tempdir / "test.pkl", "wb") as f:
            pickle.dump(data, f)

        with open(tempdir / "test.pkl", "rb") as f:
            data2 = pickle.load(f)

    assert_allclose_td(data.to_tensordict(), data2.to_tensordict())
    assert isinstance(data2, MyData)
    assert data2.z == data.z


def _make_data(shape):
    return MyData(
        X=torch.rand(*shape),
        y=torch.rand(*shape),
        z="test_tensorclass",
        batch_size=shape[:1],
    )


def test_multiprocessing():
    with Pool(os.cpu_count()) as p:
        catted = torch.cat(p.map(_make_data, [(i, 2) for i in range(1, 9)]), dim=0)

    assert catted.batch_size == torch.Size([36])
    assert catted.z == "test_tensorclass"


def test_torchsnapshot(tmp_path):
    @tensorclass
    class MyClass:
        x: torch.Tensor
        z: str
        y: "MyClass" = None  # future: drop quotes

    z = "test_tensorclass"
    tc = MyClass(
        x=torch.randn(3),
        z=z,
        y=MyClass(x=torch.randn(3), z=z, batch_size=[]),
        batch_size=[],
    )
    tc.memmap_()
    assert isinstance(tc.y.x, MemmapTensor)
    assert tc.z == z

    app_state = {"state": torchsnapshot.StateDict(tensordict=tc.state_dict())}
    snapshot = torchsnapshot.Snapshot.take(app_state=app_state, path=str(tmp_path))

    tc_dest = MyClass(
        x=torch.randn(3),
        z="other",
        y=MyClass(x=torch.randn(3), z=z, batch_size=[]),
        batch_size=[],
    )
    tc_dest.memmap_()
    assert isinstance(tc_dest.y.x, MemmapTensor)
    app_state = {"state": torchsnapshot.StateDict(tensordict=tc_dest.state_dict())}
    snapshot.restore(app_state=app_state)

    assert (tc_dest == tc).all()
    assert tc_dest.y.batch_size == tc.y.batch_size
    assert isinstance(tc_dest.y.x, MemmapTensor)
    # torchsnapshot does not support updating strings and such
    assert tc_dest.z != z

    tc_dest = MyClass(
        x=torch.randn(3),
        z="other",
        y=MyClass(x=torch.randn(3), z=z, batch_size=[]),
        batch_size=[],
    )
    tc_dest.memmap_()
    tc_dest.load_state_dict(tc.state_dict())
    assert (tc_dest == tc).all()
    assert tc_dest.y.batch_size == tc.y.batch_size
    assert isinstance(tc_dest.y.x, MemmapTensor)
    # load_state_dict outperforms snapshot in this case
    assert tc_dest.z == z


def test_statedict_errors():
    @tensorclass
    class MyClass:
        x: torch.Tensor
        z: str
        y: "MyClass" = None  # future: drop quotes

    z = "test_tensorclass"
    tc = MyClass(
        x=torch.randn(3),
        z=z,
        y=MyClass(x=torch.randn(3), z=z, batch_size=[]),
        batch_size=[],
    )

    sd = tc.state_dict()
    sd["a"] = None
    with pytest.raises(KeyError, match="Key 'a' wasn't expected in the state-dict"):
        tc.load_state_dict(sd)
    del sd["a"]
    sd["_tensordict"]["a"] = None
    with pytest.raises(KeyError, match="Key 'a' wasn't expected in the state-dict"):
        tc.load_state_dict(sd)
    del sd["_tensordict"]["a"]
    sd["_non_tensordict"]["a"] = None
    with pytest.raises(KeyError, match="Key 'a' wasn't expected in the state-dict"):
        tc.load_state_dict(sd)
    del sd["_non_tensordict"]["a"]
    sd["_non_tensordict"]["y"]["_tensordict"]["a"] = None
    with pytest.raises(KeyError, match="Key 'a' wasn't expected in the state-dict"):
        tc.load_state_dict(sd)


def test_equal():
    @tensorclass
    class MyClass1:
        x: torch.Tensor
        z: str
        y: "MyClass1" = None  # future: drop quotes

    @tensorclass
    class MyClass2:
        x: torch.Tensor
        z: str
        y: "MyClass2" = None  # future: drop quotes

    a = MyClass1(
        torch.zeros(3),
        "z0",
        MyClass1(
            torch.ones(3),
            "z1",
            None,
            batch_size=[3],
        ),
        batch_size=[3],
    )
    b = MyClass2(
        torch.zeros(3),
        "z0",
        MyClass2(
            torch.ones(3),
            "z1",
            None,
            batch_size=[3],
        ),
        batch_size=[3],
    )
    c = TensorDict({"x": torch.zeros(3), "y": {"x": torch.ones(3)}}, batch_size=[3])

    assert (a == a.clone()).all()
    assert (a != 1.0).any()
    assert (a[:2] != 1.0).any()

    assert (a.y == 1).all()
    assert (a[:2].y == 1).all()
    assert (a.y[:2] == 1).all()

    assert (a != torch.ones([])).any()
    assert (a.y == torch.ones([])).all()

    assert (a == b).all()
    assert (b == a).all()
    assert (b[:2] == a[:2]).all()

    assert (a == c).all()
    assert (a[:2] == c[:2]).all()

    assert (c == a).all()
    assert (c[:2] == a[:2]).all()

    assert (a != c.clone().zero_()).any()
    assert (c != a.clone().zero_()).any()


def test_all_any():
    @tensorclass
    class MyClass1:
        x: torch.Tensor
        z: str
        y: "MyClass1" = None  # future: drop quotes

    # with all 0
    x = MyClass1(
        torch.zeros(3, 1),
        "z",
        MyClass1(torch.zeros(3, 1), "z", batch_size=[3, 1]),
        batch_size=[3, 1],
    )
    assert not x.all()
    assert not x.any()
    assert isinstance(x.all(), bool)
    assert isinstance(x.any(), bool)
    for dim in [0, 1, -1, -2]:
        assert isinstance(x.all(dim=dim), MyClass1)
        assert isinstance(x.any(dim=dim), MyClass1)
        assert not x.all(dim=dim).all()
        assert not x.any(dim=dim).any()
    # with all 1
    x = x.apply(lambda x: x.fill_(1.0))
    assert isinstance(x, MyClass1)
    assert x.all()
    assert x.any()
    assert isinstance(x.all(), bool)
    assert isinstance(x.any(), bool)
    for dim in [0, 1]:
        assert isinstance(x.all(dim=dim), MyClass1)
        assert isinstance(x.any(dim=dim), MyClass1)
        assert x.all(dim=dim).all()
        assert x.any(dim=dim).any()

    # with 0 and 1
    x.y.x.fill_(0.0)
    assert not x.all()
    assert x.any()
    assert isinstance(x.all(), bool)
    assert isinstance(x.any(), bool)
    for dim in [0, 1]:
        assert isinstance(x.all(dim=dim), MyClass1)
        assert isinstance(x.any(dim=dim), MyClass1)
        assert not x.all(dim=dim).all()
        assert x.any(dim=dim).any()

    assert not x.y.all()
    assert not x.y.any()


@pytest.mark.parametrize("from_torch", [True, False])
def test_gather(from_torch):
    @tensorclass
    class MyClass:
        x: torch.Tensor
        z: str
        y: "MyClass" = None  # future: drop quotes

    c = MyClass(
        torch.randn(3, 4),
        "foo",
        MyClass(torch.randn(3, 4, 5), "bar", None, batch_size=[3, 4, 5]),
        batch_size=[3, 4],
    )
    dim = -1
    index = torch.arange(3).expand(3, 3)
    if from_torch:
        c_gather = torch.gather(c, index=index, dim=dim)
    else:
        c_gather = c.gather(index=index, dim=dim)
    assert c_gather.x.shape == torch.Size([3, 3])
    assert c_gather.y.shape == torch.Size([3, 3, 5])
    assert c_gather.y.x.shape == torch.Size([3, 3, 5])
    assert c_gather.y.z == "bar"
    assert c_gather.z == "foo"
    c_gather_zero = c_gather.clone().zero_()
    if from_torch:
        c_gather2 = torch.gather(c, index=index, dim=dim, out=c_gather_zero)
    else:
        c_gather2 = c.gather(index=index, dim=dim, out=c_gather_zero)

    assert (c_gather2 == c_gather).all()


def test_to_tensordict():
    @tensorclass
    class MyClass:
        x: torch.Tensor
        z: str
        y: "MyClass" = None  # future: drop quotes

    c = MyClass(
        torch.randn(3, 4),
        "foo",
        MyClass(torch.randn(3, 4, 5), "bar", None, batch_size=[3, 4, 5]),
        batch_size=[3, 4],
    )

    ctd = c.to_tensordict()
    assert isinstance(ctd, TensorDictBase)
    assert "x" in ctd.keys()
    assert "z" not in ctd.keys()
    assert "y" in ctd.keys()
    assert ("y", "x") in ctd.keys(True)


def test_memmap_():
    @tensorclass
    class MyClass:
        x: torch.Tensor
        z: str
        y: "MyClass" = None  # future: drop quotes

    c = MyClass(
        torch.randn(3, 4),
        "foo",
        MyClass(torch.randn(3, 4, 5), "bar", None, batch_size=[3, 4, 5]),
        batch_size=[3, 4],
    )

    cmemmap = c.memmap_()
    assert cmemmap is c
    assert isinstance(c.x, MemmapTensor)
    assert isinstance(c.y.x, MemmapTensor)
    assert c.z == "foo"


def test_memmap_like():
    @tensorclass
    class MyClass:
        x: torch.Tensor
        z: str
        y: "MyClass" = None  # future: drop quotes

    c = MyClass(
        torch.randn(3, 4),
        "foo",
        MyClass(torch.randn(3, 4, 5), "bar", None, batch_size=[3, 4, 5]),
        batch_size=[3, 4],
    )

    cmemmap = c.memmap_like()
    assert cmemmap is not c
    assert cmemmap.y is not c.y
    assert (cmemmap == 0).all()
    assert isinstance(cmemmap.x, MemmapTensor)
    assert isinstance(cmemmap.y.x, MemmapTensor)
    assert cmemmap.z == "foo"


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
