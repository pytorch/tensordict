from __future__ import annotations

import argparse
import dataclasses
import os
import pickle
import re
from multiprocessing import Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Union

import pytest
import torch

from _utils_internal import get_available_devices

from tensordict import LazyStackedTensorDict, TensorDict
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


@tensorclass
class MyData3:
    X: torch.Tensor
    y: torch.Tensor
    z: str


@tensorclass
class MyData4:
    X: torch.Tensor
    y: torch.Tensor


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
        subclass=MyUnionClass(_tensordict=TensorDict({}, [3])), batch_size=[3]
    )
    with pytest.raises(TypeError, match="can't be deterministically cast."):
        assert data.subclass is not None


def test_attributes():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    z = "test_tensorclass"
    batch_size = [3, 4]
    tensordict = TensorDict(
        {
            "X": X,
            "y": y,
        },
        batch_size=[3, 4],
    )

    data = MyData(X=X, y=y, z=z, batch_size=batch_size)

    equality_tensordict = data.tensordict == tensordict

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
    myc = MyData(X=torch.rand(2, 3, 4), y=torch.rand(2, 3, 4, 5), batch_size=[2, 3])
    assert len(myc) == 2

    myc2 = MyData(X=torch.rand(2, 3, 4), y=torch.rand(2, 3, 4, 5), batch_size=[])
    assert len(myc2) == 0


def test_indexing():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    z = ["a", "b", "c"]
    data = MyData2(X=X, y=y, z=z, batch_size=[3, 4])

    assert data[:2].batch_size == torch.Size([2, 4])
    assert data[1].batch_size == torch.Size([4])
    assert data[1][1].batch_size == torch.Size([])
    assert data[1].z == data[2].z == data[:2].z == z
    assert data[:2].X.shape == torch.Size([2, 4, 5])

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
        z="test_tensorclass",
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

    # Test cases for non-tensor data
    x = torch.randn(4, 5)
    y = torch.zeros(4, 5, dtype=torch.bool)
    z = "test_tensorclass"
    batch_size = [4]
    data[1] = MyData(X=x, y=y, z=z, batch_size=batch_size)
    assert (data[1].X == x).all()
    assert (data[1].y == y).all()
    assert data[1].z == z

    # Negative testcase for non-tensor data
    z = "test_bluff"
    with pytest.raises(
        ValueError, match=f"Expecting {repr(data.z)} for z instead got {repr(z)}"
    ):
        data[1] = MyData(X=x, y=y, z=z, batch_size=batch_size)


def test_stack():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    z = "test_tensorclass"
    batch_size = [3, 4]

    data1 = MyData(X=X, y=y, z=z, batch_size=batch_size)
    data2 = MyData(X=X, y=y, z=z, batch_size=batch_size)
    data3 = MyData3(X=X, y=y, z=z, batch_size=batch_size)
    data4 = MyData(X=X, y=y, z="test_bluff", batch_size=batch_size)

    stacked_tc = torch.stack([data1, data2], 0)
    assert type(stacked_tc) is type(data1)
    assert stacked_tc.X.shape == torch.Size([2, 3, 4, 5])
    assert (stacked_tc.X == 1).all()
    assert isinstance(stacked_tc.tensordict, LazyStackedTensorDict)
    assert stacked_tc.z == data1.z == z

    with pytest.raises(
        TypeError,
        match=re.escape(
            "no implementation found for 'torch.stack' on types that implement "
            "__torch_function__: [<class 'test_tensorclass.MyData'>, "
            "<class 'test_tensorclass.MyData3'>]"
        ),
    ):
        torch.stack([data1, data3], dim=0)

    with pytest.raises(
        ValueError,
        match=f"{repr(data4.z)} and {repr(data1.z)} for the attribute 'z' are not matching",
    ):
        assert torch.stack([data1, data4], dim=0)


def test_cat():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    z = "test_tensorclass"
    batch_size = [3, 4]

    data1 = MyData(X=X, y=y, z=z, batch_size=batch_size)
    data2 = MyData(X=X, y=y, z=z, batch_size=batch_size)
    data3 = MyData3(X=X, y=y, z=z, batch_size=batch_size)
    data4 = MyData(X=X, y=y, z="test_bluff", batch_size=batch_size)

    catted_tc = torch.cat([data1, data2], 0)
    assert type(catted_tc) is type(data1)
    assert catted_tc.X.shape == torch.Size([6, 4, 5])
    assert (catted_tc.X == 1).all()
    assert isinstance(catted_tc.tensordict, TensorDict)
    assert catted_tc.z == data1.z == z

    with pytest.raises(
        TypeError,
        match=re.escape(
            "no implementation found for 'torch.cat' on types that implement "
            "__torch_function__: [<class 'test_tensorclass.MyData'>, "
            "<class 'test_tensorclass.MyData3'>]"
        ),
    ):
        torch.cat([data1, data3], dim=0)

    with pytest.raises(
        ValueError,
        match=f"{repr(data4.z)} and {repr(data1.z)} for the attribute 'z' are not matching",
    ):
        assert torch.cat([data1, data4], dim=0)


def test_unbind():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    z = ["a", "b", "c"]
    batch_size = [3, 4]
    data = MyData2(X=X, y=y, z=z, batch_size=batch_size)
    unbind_tcs = torch.unbind(data, 0)
    assert type(unbind_tcs[0]) is type(data)
    assert len(unbind_tcs) == 3
    assert torch.all(torch.eq(unbind_tcs[0].X, torch.ones(4, 5)))
    assert unbind_tcs[0].batch_size == torch.Size([4])
    assert unbind_tcs[0].z == unbind_tcs[1].z == unbind_tcs[2].z == z


def test_full_like():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data = MyData(X=X, y=y, z=z, batch_size=batch_size)
    full_like_tc = torch.full_like(data, 9.0)
    assert type(full_like_tc) is type(data)
    assert full_like_tc.batch_size == torch.Size(data.batch_size)
    assert full_like_tc.X.size() == data.X.size()
    assert full_like_tc.y.size() == data.y.size()
    assert (full_like_tc.X == 9).all()
    assert full_like_tc.y.all()
    assert full_like_tc.z == data.z == z


def test_clone():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data = MyData(X=X, y=y, z=z, batch_size=batch_size)
    clone_tc = torch.clone(data)
    assert clone_tc.batch_size == torch.Size(data.batch_size)
    assert torch.all(torch.eq(clone_tc.X, data.X))
    assert torch.all(torch.eq(clone_tc.y, data.y))
    assert clone_tc.z == data.z == z


def test_squeeze():
    X = torch.ones(1, 4, 5)
    y = torch.zeros(1, 4, 5, dtype=torch.bool)
    z = "test_tensorclass"
    batch_size = [1, 4]
    data = MyData(X=X, y=y, z=z, batch_size=batch_size)
    squeeze_tc = torch.squeeze(data)
    assert squeeze_tc.batch_size == torch.Size([4])
    assert squeeze_tc.X.shape == torch.Size([4, 5])
    assert squeeze_tc.y.shape == torch.Size([4, 5])
    assert squeeze_tc.z == data.z == z


def test_unsqueeze():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    z = ["a", "b", "c"]
    batch_size = [3, 4]
    data = MyData2(X=X, y=y, z=z, batch_size=batch_size)
    unsqueeze_tc = torch.unsqueeze(data, dim=1)
    assert unsqueeze_tc.batch_size == torch.Size([3, 1, 4])
    assert unsqueeze_tc.X.shape == torch.Size([3, 1, 4, 5])
    assert unsqueeze_tc.y.shape == torch.Size([3, 1, 4, 5])
    assert unsqueeze_tc.z == data.z == z


def test_split():
    X = torch.ones(3, 6, 5)
    y = torch.zeros(3, 6, 5, dtype=torch.bool)
    z = "test_tensorclass"
    batch_size = [3, 6]
    data = MyData(X=X, y=y, z=z, batch_size=batch_size)
    split_tcs = torch.split(data, split_size_or_sections=[3, 2, 1], dim=1)
    assert type(split_tcs[1]) is type(data)
    assert split_tcs[0].batch_size == torch.Size([3, 3])
    assert split_tcs[1].batch_size == torch.Size([3, 2])
    assert split_tcs[2].batch_size == torch.Size([3, 1])
    assert torch.all(torch.eq(split_tcs[0].X, torch.ones(3, 3, 5)))
    assert torch.all(torch.eq(split_tcs[2].y, torch.zeros(3, 1, 5, dtype=torch.bool)))
    assert split_tcs[0].z == split_tcs[1].z == split_tcs[2].z == z


def test_reshape():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data = MyData(X=X, y=y, z=z, batch_size=batch_size)
    stacked_tc = data.reshape(-1)
    assert stacked_tc.X.shape == torch.Size([12, 5])
    assert stacked_tc.shape == torch.Size([12])
    assert (stacked_tc.X == 1).all()
    assert isinstance(stacked_tc.tensordict, TensorDict)
    assert stacked_tc.z == data.z == z


def test_view():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data = MyData(X=X, y=y, z=z, batch_size=batch_size)
    stacked_tc = data.view(-1)
    assert stacked_tc.X.shape == torch.Size([12, 5])
    assert stacked_tc.shape == torch.Size([12])
    assert (stacked_tc.X == 1).all()
    assert isinstance(stacked_tc.tensordict, _ViewedTensorDict)
    assert stacked_tc.z == data.z == z


def test_permute():
    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data = MyData(X=X, y=y, z=z, batch_size=batch_size)
    stacked_tc = data.permute(1, 0)
    assert stacked_tc.X.shape == torch.Size([4, 3, 5])
    assert stacked_tc.shape == torch.Size([4, 3])
    assert (stacked_tc.X == 1).all()
    assert isinstance(stacked_tc.tensordict, _PermutedTensorDict)
    assert stacked_tc.z == data.z == z


def test_nested():
    @tensorclass
    class MyDataNested:
        X: torch.Tensor
        z: str
        y: MyDataNested = None

    X = torch.ones(3, 4, 5)
    z = "test_tensorclass"
    batch_size = [3, 4]
    data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
    data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
    assert isinstance(data.y, MyDataNested), type(data.y)
    assert data.z == data_nest.z == z


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
    assert isinstance(data.tensordict["y"], TensorDict)
    assert data.v == v
    assert data.y.v == "test_nested"


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
        v: str

    batch_size = [3, 4]
    if any_to_td:
        W = TensorDict({}, batch_size)
    else:
        W = torch.zeros(*batch_size, 1)
    X = torch.ones(3, 4, 5)
    X_clone = X.clone()
    td = TensorDict({}, batch_size)
    td_clone = td.clone()
    data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
    v = "test_tensorclass"
    data = MyDataParent(X=X, y=data_nest, z=td, W=W, v=v, batch_size=batch_size)
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
    v_new = "test_bluff"
    data.v = v_new
    assert data.v == v_new
    assert data.y.v == "test_nested"
    data.y.v = "test_nested_new"
    assert data.y.v == "test_nested_new"
    data_nest.v = "test_nested"
    assert data.y.v == "test_nested"
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
        y: torch.Tensor = dataclasses.field(default_factory=lambda: torch.ones(3, 4, 5))

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


def _make_data(shape):
    return MyData4(X=torch.rand(*shape), y=torch.rand(*shape), batch_size=shape[:1])


def test_multiprocessing():
    with Pool(os.cpu_count()) as p:
        catted = torch.cat(p.map(_make_data, [(i, 2) for i in range(1, 9)]), dim=0)

    assert catted.batch_size == torch.Size([36])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
