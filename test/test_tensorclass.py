# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import inspect
import os
import pathlib
import pickle
import re
import sys
import weakref
from dataclasses import field
from multiprocessing import Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Tuple, Union

import numpy as np
import pytest
import tensordict.utils
import torch

from _utils_internal import get_available_devices

from tensordict import (
    assert_allclose_td,
    is_tensorclass,
    lazy_legacy,
    LazyStackedTensorDict,
    MemoryMappedTensor,
    set_capture_non_tensor_stack,
    set_list_to_stack,
    tensorclass,
    TensorClass,
    TensorDict,
    TensorDictBase,
)
from tensordict._lazy import _PermutedTensorDict, _ViewedTensorDict
from tensordict.base import _GENERIC_NESTED_ERR
from tensordict.tensorclass import from_dataclass
from torch import Tensor

try:
    import torchsnapshot

    _has_torchsnapshot = True
    TORCHSNAPSHOT_ERR = ""
except ImportError as err:
    _has_torchsnapshot = False
    TORCHSNAPSHOT_ERR = str(err)

# Capture all warnings
pytestmark = [
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings(
        "ignore:type_hints are none, cannot perform auto-casting"
    ),
    pytest.mark.filterwarnings(
        "ignore:You are using `torch.load` with `weights_only=False`"
    ),
]


def _get_methods_from_pyi(file_path):
    """
    Reads a .pyi file and returns a set of method names.

    Args:
        file_path (str): Path to the .pyi file.

    Returns:
        set: A set of method names.
    """
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())

    methods = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for child_node in node.body:
                if isinstance(child_node, ast.FunctionDef):
                    methods.add(child_node.name)

    return methods


def _get_methods_from_class(cls):
    """
    Returns a set of method names from a given class.

    Args:
        cls (class): The class to get methods from.

    Returns:
        set: A set of method names.
    """
    methods = set()
    for name in dir(cls):
        attr = getattr(cls, name)
        if (
            inspect.isfunction(attr)
            or inspect.ismethod(attr)
            or isinstance(attr, property)
        ):
            methods.add(name)

    return methods


def test_tensorclass_stub_methods():
    tensorclass_pyi_path = (
        pathlib.Path(__file__).parent.parent / "tensordict/tensorclass.pyi"
    )
    tensorclass_methods = _get_methods_from_pyi(str(tensorclass_pyi_path))

    from tensordict import TensorDict

    tensordict_methods = _get_methods_from_class(TensorDict)

    missing_methods = tensordict_methods - tensorclass_methods
    missing_methods = [
        method for method in missing_methods if (not method.startswith("_"))
    ]

    if missing_methods:
        raise Exception(
            f"Missing methods in tensorclass.pyi: {sorted(missing_methods)}"
        )


def test_tensorclass_instance_methods():
    @tensorclass
    class X:
        x: torch.Tensor

    tensorclass_pyi_path = (
        pathlib.Path(__file__).parent.parent / "tensordict/tensorclass.pyi"
    )
    tensorclass_abstract_methods = _get_methods_from_pyi(str(tensorclass_pyi_path))

    tensorclass_methods = _get_methods_from_class(X)

    missing_methods = (
        tensorclass_abstract_methods - tensorclass_methods - {"data", "grad"}
    )
    missing_methods = [
        method for method in missing_methods if (not method.startswith("_"))
    ]

    if missing_methods:
        raise Exception(
            f"Missing methods in tensorclass.pyi: {sorted(missing_methods)}"
        )


def test_sorted_methods():
    from tensordict.tensorclass import (
        _FALLBACK_METHOD_FROM_TD,
        _FALLBACK_METHOD_FROM_TD_FORCE,
        _FALLBACK_METHOD_FROM_TD_NOWRAP,
        _METHOD_FROM_TD,
    )

    lists_to_check = [
        _FALLBACK_METHOD_FROM_TD_NOWRAP,
        _METHOD_FROM_TD,
        _FALLBACK_METHOD_FROM_TD_FORCE,
        _FALLBACK_METHOD_FROM_TD,
    ]
    # Check that each list is sorted and has unique elements
    for lst in lists_to_check:
        assert lst == sorted(lst), f"List {lst} is not sorted"
        assert len(lst) == len(set(lst)), f"List {lst} has duplicate elements"
    # Check that no two lists share any elements
    for i, lst1 in enumerate(lists_to_check):
        for j, lst2 in enumerate(lists_to_check):
            if i != j:
                shared_elements = set(lst1) & set(lst2)
                assert (
                    not shared_elements
                ), f"Lists {lst1} and {lst2} share elements: {shared_elements}"


def _make_data(shape):
    return MyData(
        X=torch.rand(*shape),
        y=torch.rand(*shape),
        z="test_tensorclass",
        batch_size=shape[:1],
    )


class MyData:
    X: torch.Tensor
    y: torch.Tensor
    z: str

    def stuff(self):
        return self.X + self.y


PY8 = sys.version_info >= (3, 8) and sys.version_info < (3, 9)
PY9 = sys.version_info >= (3, 9) and sys.version_info < (3, 10)
PY10 = sys.version_info >= (3, 10)

# this slightly convoluted construction of MyData allows us to check that instances of
# the tensorclass are instances of the original class.
MyDataUndecorated, MyData = MyData, tensorclass(MyData)


@tensorclass
class MyData2:
    X: torch.Tensor
    y: torch.Tensor
    z: list


@dataclasses.dataclass
class MyDataClass:
    a: int
    b: torch.Tensor
    c: str


try:
    MyTensorClass_autocast = from_dataclass(MyDataClass, autocast=True)
    MyTensorClass_nocast = from_dataclass(MyDataClass, nocast=True)
    MyTensorClass = from_dataclass(MyDataClass)
except Exception:
    MyTensorClass_autocast = MyTensorClass_nocast = MyTensorClass = None


@tensorclass
class TCStrings:
    a: str
    b: str


class TestTensorClass:
    def test_get_default(self):
        @tensorclass
        class Data:
            td: TensorDict
            a: torch.Tensor

        data = Data(td=TensorDict(), a=torch.zeros(()))
        assert data.get("a") is not None
        assert data.get("b") is None
        assert data.get("b", "else") == "else"

        with pytest.raises(KeyError, match=_GENERIC_NESTED_ERR.format(())):
            data.get(("td", str))  # something unexpected!

        assert data.get(("td", "missing"), "else") == "else"
        assert data.get(("td", "missing")) is None

        data = data.expand(10)
        assert data.get_at("a", 0) is not None
        assert data.get_at("b", 0) is None
        assert data.get_at("b", 0, "else") == "else"

        assert data.get_at(("td", "missing"), 0, "else") == "else"
        assert data.get_at(("td", "missing"), 0) is None

    def test_decorator(self):
        @tensorclass
        class MyClass:
            X: torch.Tensor
            y: Any

        obj = MyClass(X=torch.zeros(2), y="a string!", batch_size=[])
        assert not obj.is_locked
        with obj.lock_():
            assert obj.is_locked
            with obj.unlock_():
                assert not obj.is_locked
            assert obj.is_locked
        assert not obj.is_locked

    def test_to_dict(self):
        @tensorclass
        class TestClass:
            my_tensor: torch.Tensor
            my_str: str

        test_class = TestClass(
            my_tensor=torch.tensor([1, 2, 3]), my_str="hello", batch_size=[3]
        )

        assert (
            test_class
            == TestClass.from_dict(test_class.to_dict(), auto_batch_size=True)
        ).all()

        # Currently we don't test non-tensor in __eq__ because __eq__ can break with arrays and such
        # test_class2 = TestClass(
        #     my_tensor=torch.tensor([1, 2, 3]), my_str="goodbye", batch_size=[3]
        # )
        #
        # assert not (test_class == TestClass.from_dict(test_class2.to_dict())).all()

        test_class3 = TestClass(
            my_tensor=torch.tensor([1, 2, 0]), my_str="hello", batch_size=[3]
        )

        assert not (
            test_class
            == TestClass.from_dict(test_class3.to_dict(), auto_batch_size=True)
        ).all()

    def test_all_any(self):
        @tensorclass
        class MyClass1:
            x: torch.Tensor
            z: str
            y: "MyClass1" = None

        # with all 0
        x = MyClass1(
            torch.zeros(3, 1),
            "z",
            MyClass1(torch.zeros(3, 1), "z", batch_size=[3, 1]),
            batch_size=[3, 1],
        )
        assert x.shape == x.batch_size
        assert x.batch_size == (3, 1)
        assert x.ndim == 2
        assert x.batch_dims == 2
        assert x.numel() == 3

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

    def test_args(self):
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
        with set_capture_non_tensor_stack(True):
            data = torch.stack([data1, data2, data3, data4, data5], 0)
        assert (data.A == A).all()
        assert (data.B == B).all()
        assert (data.C == C).all()
        assert (data.D == D).all()
        assert data.E == E

    def test_attributes(self):
        X = torch.ones(3, 4, 5)
        y = torch.zeros(3, 4, 5, dtype=torch.bool)
        batch_size = [3, 4]
        z = "test_tensorclass"
        tensordict = TensorDict(
            {
                "X": X,
                "y": y,
                "z": z,
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

    def test_property(self):
        @tensorclass
        class MyData:
            a: torch.Tensor
            b: str
            _c: Any = None

            @property
            def c(self):
                return getattr(self, "_c", None)

            @c.setter
            def c(self, value):
                self._c = value

        data = MyData(a=torch.ones(()), b="a string!")
        assert data.c is None
        data.c = "1"
        assert data.c == "1"
        assert isinstance(data.c, str)

    def test_banned_types(self):
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

    def test_batch_size(self):
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
        myc.batch_size = [2]
        assert isinstance(myc.batch_size, torch.Size)

        assert myc.X.shape == torch.Size([2, 3, 4])

    def test_cat(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data1 = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        data2 = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)

        tc_cat = torch.cat([data1, data2], 0)
        assert type(tc_cat) is type(data1)
        assert isinstance(tc_cat.y, type(data1.y))
        assert tc_cat.X.shape == torch.Size([6, 4, 5])
        assert tc_cat.y.X.shape == torch.Size([6, 4, 5])
        assert (tc_cat.X == 1).all()
        assert (tc_cat.y.X == 1).all()
        assert isinstance(tc_cat._tensordict, TensorDict)
        assert tc_cat.z == tc_cat.y.z == z

        # Testing negative scenarios
        y = torch.zeros(3, 4, 5, dtype=torch.bool)
        data3 = MyData(X=X, y=y, z=z, batch_size=batch_size)

        with pytest.raises(
            TypeError,
            match=("Multiple dispatch failed|no implementation found"),
        ):
            torch.cat([data1, data3], dim=0)

    def test_clone(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

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

    @pytest.mark.parametrize("file", [True, False])
    def test_consolidate(self, file, tmpdir):
        data = MyData2(
            torch.ones((2,)), torch.ones((2, 3)) * 2, "a string!", batch_size=[2]
        )
        if file:
            filename = Path(tmpdir) / "file.mmap"
        else:
            filename = None
        data_c = data.consolidate(filename=filename)
        assert data_c.z == "a string!"
        assert isinstance(data_c, MyData2)
        assert hasattr(data_c, "_consolidated")

        # test pickle
        f = Path(tmpdir) / "data.pkl"
        torch.save(data, f)
        data_load = torch.load(f, weights_only=False)
        assert isinstance(data_load, MyData2)
        assert data_load.z == "a string!"
        assert data_load.batch_size == data.batch_size
        assert (data_load == data).all()

        # with consolidated data
        f = Path(tmpdir) / "data.pkl"
        torch.save(data_c, f)
        data_load = torch.load(f, weights_only=False)
        assert isinstance(data_load, MyData2)
        assert data_load.z == "a string!"
        assert data_load.batch_size == data.batch_size
        assert (data_load == data).all()

    def test_dataclass(self):
        data = MyData(
            X=torch.ones(3, 4, 5),
            y=torch.zeros(3, 4, 5, dtype=torch.bool),
            z="test_tensorclass",
            batch_size=[3, 4],
        )
        assert dataclasses.is_dataclass(data)

    def test_default(self):
        @tensorclass
        class MyData:
            X: torch.Tensor = (
                None  # TODO: do we want to allow any default, say an integer?
            )
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

    def test_defaultfactory(self):
        @tensorclass
        class MyData:
            X: torch.Tensor = (
                None  # TODO: do we want to allow any default, say an integer?
            )
            y: torch.Tensor = dataclasses.field(
                default_factory=lambda: torch.ones(3, 4, 5)
            )

        data = MyData(batch_size=[3, 4])
        assert (data.y == 1).all()
        assert data.X is None
        data.X = torch.zeros(3, 4, 1)
        assert (data.X == 0).all()

        MyData(batch_size=[3])
        MyData(batch_size=[])
        with pytest.raises(RuntimeError, match="batch dimension mismatch"):
            MyData(batch_size=[4])

    @pytest.mark.parametrize("device", get_available_devices())
    def test_device(self, device):
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
            AttributeError, match="'str' object has no attribute 'device'"
        ):
            assert data.z.device == device

        with pytest.raises(
            RuntimeError, match="device cannot be set using tensorclass.device = device"
        ):
            data.device = torch.device("cpu")

    def test_disallowed_attributes(self):
        with pytest.raises(
            AttributeError,
            match="Attribute name reshape can't be used with @tensorclass",
        ):

            @tensorclass
            class MyInvalidClass:
                x: torch.Tensor
                y: torch.Tensor
                reshape: torch.Tensor

    def test_equal(self):
        @tensorclass
        class MyClass1:
            x: torch.Tensor
            z: str
            y: "MyClass1" = None

        @tensorclass
        class MyClass2:
            x: torch.Tensor
            z: str
            y: "MyClass2" = None

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
        assert not a._non_tensordict
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
        assert not b._non_tensordict
        c = TensorDict(
            {"x": torch.zeros(3), "z": "z0", "y": {"x": torch.ones(3), "z": "z1"}},
            batch_size=[3],
        )

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

    def test_field(self):
        class Cls(TensorClass):
            a: torch.Tensor
            b: str
            c: dict = field(default_factory=dict)

        obj = Cls(a=torch.arange(3), b="abc", batch_size=[3])
        assert obj[0].a == obj[1].a - 1
        assert obj[0].b == obj[1].b
        assert obj[0].c is obj[1].c

    def test_from_dataclass_exec(self):
        # Check that everything runs fine
        from_dataclass(MyDataClass, autocast=True)
        from_dataclass(MyDataClass, nocast=True)
        from_dataclass(MyDataClass)

    def test_from_dataclass(self):
        assert is_tensorclass(MyTensorClass_autocast)
        assert MyTensorClass_nocast is not MyDataClass
        assert MyTensorClass_autocast._autocast
        x = MyTensorClass_autocast(a=0, b=0, c=0)
        assert isinstance(x.a, int)
        assert isinstance(x.b, torch.Tensor)
        assert isinstance(x.c, str)

        assert is_tensorclass(MyTensorClass_nocast)
        assert MyTensorClass_nocast is not MyTensorClass_autocast
        assert MyTensorClass_nocast._nocast

        x = MyTensorClass_nocast(a=0, b=0, c=0)
        assert is_tensorclass(MyTensorClass)
        assert not MyTensorClass._autocast
        assert not MyTensorClass._nocast
        assert isinstance(x.a, int)
        assert isinstance(x.b, int)
        assert isinstance(x.c, int)

        x = MyTensorClass(a=0, b=0, c=0)
        assert isinstance(x.a, torch.Tensor)
        assert isinstance(x.b, torch.Tensor)
        assert isinstance(x.c, torch.Tensor)

        x = TensorDict.from_dataclass(MyTensorClass(a=0, b=0, c=0))
        assert isinstance(x, TensorDict)
        assert isinstance(x["a"], torch.Tensor)
        assert isinstance(x["b"], torch.Tensor)
        assert isinstance(x["c"], torch.Tensor)

        x = from_dataclass(MyTensorClass(a=0, b=0, c=0))
        assert is_tensorclass(x)
        assert isinstance(x.a, torch.Tensor)
        assert isinstance(x.b, torch.Tensor)
        assert isinstance(x.c, torch.Tensor)

        @dataclasses.dataclass
        class MyOtherDataClass:
            a: int = 0
            b: int = 0
            c: int = 0

        cls = from_dataclass(MyOtherDataClass)
        x = from_dataclass(MyOtherDataClass(), dest_cls=cls)
        assert is_tensorclass(x)
        assert type(x) is cls
        assert isinstance(x.a, torch.Tensor)
        assert isinstance(x.b, torch.Tensor)
        assert isinstance(x.c, torch.Tensor)

    def test_from_dict(self):
        td = TensorDict(
            {
                ("a", "b", "c"): 1,
                ("a", "d"): 2,
            },
            [],
        ).expand(10)
        d = td.to_dict()

        @tensorclass
        class MyClass:
            a: TensorDictBase

        tc = MyClass.from_dict(d, auto_batch_size=True)
        assert isinstance(tc, MyClass)
        assert isinstance(tc.a, TensorDict)
        assert tc.batch_size == torch.Size([10])

    def test_full_like(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

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

    def test_frozen(self):

        @tensorclass(frozen=True, autocast=True)
        class X:
            y: torch.Tensor

        x = X(y=1)
        assert isinstance(x.y, torch.Tensor)
        _ = {x: 0}
        assert x.is_locked
        with pytest.raises((RuntimeError, dataclasses.FrozenInstanceError)):
            x.y = 0

        @tensorclass(frozen=False, autocast=True)
        class X:
            y: torch.Tensor

        x = X(y=1)
        assert isinstance(x.y, torch.Tensor)
        with pytest.raises(TypeError, match="unhashable"):
            _ = {x: 0}
        assert not x.is_locked
        x.y = 0

        @tensorclass(frozen=True, autocast=False)
        class X:
            y: torch.Tensor

        x = X(y="a string!")
        assert isinstance(x.y, str)
        _ = {x: 0}
        assert x.is_locked
        with pytest.raises((RuntimeError, dataclasses.FrozenInstanceError)):
            x.y = 0

        @tensorclass(frozen=False, autocast=False)
        class X:
            y: torch.Tensor

        x = X(y="a string!")
        assert isinstance(x.y, str)
        with pytest.raises(TypeError, match="unhashable"):
            _ = {x: 0}
        assert not x.is_locked
        x.y = 0

    @pytest.mark.parametrize("from_torch", [True, False])
    def test_gather(self, from_torch):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            z: str
            y: "MyClass" = None

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
        assert isinstance(c_gather, type(c))
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

    def test_get(self):
        @tensorclass
        class MyDataNest:
            X: torch.Tensor
            v: str

        @tensorclass
        class MyDataParent:
            X: Tensor
            z: TensorDictBase
            y: MyDataNest
            v: str
            k: Optional[Tensor] = None

        batch_size = [3, 4]
        X = torch.ones(3, 4, 5)
        td = TensorDict({}, batch_size)
        data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
        v = "test_tensorclass"
        data = MyDataParent(X=X, y=data_nest, z=td, v=v, batch_size=batch_size)
        assert isinstance(data.y, type(data_nest))
        assert (data.get("X") == X).all()
        assert data.get("batch_size") == torch.Size(batch_size)
        assert data.get("v") == v
        assert (data.get("z") == td).all()

        # Testing nested tensor class
        assert data.get("y")._tensordict is data_nest._tensordict
        assert (data.get("y").X == X).all()
        assert (data.get(("y", "X")) == X).all()
        assert data.get("y").v == "test_nested"
        assert data.get(("y", "v")) == "test_nested"
        assert data.get("y").batch_size == torch.Size(batch_size)

        # ensure optional fields are there
        assert data.get("k") is None

        # ensure default works
        assert data.get("foo", "working") == "working"
        assert data.get(("foo", "foo2"), "working") == "working"
        assert data.get(("X", "foo2"), "working") == "working"

        assert (data.get("X", "working") == X).all()
        assert data.get("v", "working") == v

    def test_get_lazystack(self):
        lazystack = LazyStackedTensorDict(
            TensorDict(X=torch.ones(3), y=0, z="a string"),
            TensorDict(X=torch.ones(2) * 2, y=0, z="a string"),
        )
        obj = MyData2.from_tensordict(lazystack)
        a = obj.get("X", as_list=True)
        assert isinstance(a, list)
        a = obj.get("X", as_nested_tensor=True)
        assert a.is_nested
        a = obj.get("X", as_padded_tensor=True)
        assert a.shape == (2, 3)
        assert a[1, -1] == 0

    @pytest.mark.parametrize("any_to_td", [True, False])
    def test_getattr(self, any_to_td):
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

    @pytest.mark.parametrize("list_to_stack", [True, False])
    def test_indexing(self, list_to_stack):
        with set_list_to_stack(list_to_stack):

            @tensorclass
            class MyDataNested:
                X: torch.Tensor
                z: list
                y: "MyDataNested" = None

            X = torch.ones(3, 4, 5)
            z = ["a", "b", "c"]
            batch_size = [3, 4]
            with (
                pytest.raises(RuntimeError, match="batch dimension mismatch")
                if list_to_stack
                else contextlib.nullcontext()
            ):
                data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
                data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
            if list_to_stack:
                return
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
        assert data[1].z == data[2].z, (data[1].z, data[2].z)
        assert data[1].z == data[:2].z
        assert data[1].z == z

        with pytest.raises(
            RuntimeError,
            match="indexing a tensordict with td.batch_dims==0 is not permitted",
        ):
            data[1][1][1]

        with pytest.raises(ValueError, match="Invalid indexing arguments."):
            data["X"]

    def test_grad(self):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            y: str
            z: torch.Tensor | None = None

        a = MyClass(
            x=torch.randn(3, requires_grad=True),
            y="a string!",
            z=torch.randn(2),
        )
        assert a.requires_grad
        b = a + 1
        b.sum().x.backward()
        assert (a == a.data).all()
        assert not a.data.requires_grad
        assert a.grad.x is not None
        assert a.grad.z is None

    def test_len(self):
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

    def test_multiprocessing(self):
        with Pool(os.cpu_count()) as p:
            catted = torch.cat(p.map(_make_data, [(i, 2) for i in range(1, 9)]), dim=0)

        assert catted.batch_size == torch.Size([36])
        assert catted.z == "test_tensorclass"

    def test_nested(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        assert isinstance(data.y, MyDataNested), type(data.y)
        assert data.z == data_nest.z == data.y.z == z

    def test_nested_eq(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        data_nest2 = MyDataNested(X=X, z=z, batch_size=batch_size)
        data2 = MyDataNested(X=X, y=data_nest2, z=z, batch_size=batch_size)
        assert (data == data2).all()
        assert (data == data2).X.all()
        assert (data == data2).z.all()
        assert (data == data2).y.X.all()
        assert (data == data2).y.z.all()

    @pytest.mark.parametrize("any_to_td", [True, False])
    def test_nested_heterogeneous(self, any_to_td):
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

    def test_nested_ne(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

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
        assert (data != data2).z.all()
        assert not (data != data2).y.X.any()
        assert not (data != data2).y.z.any()

    def test_permute(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        permuted_data = data.permute(1, 0)
        assert permuted_data.X.shape == torch.Size([4, 3, 5])
        assert permuted_data.y.X.shape == torch.Size([4, 3, 5])
        assert permuted_data.shape == torch.Size([4, 3])
        assert (permuted_data.X == 1).all()
        if lazy_legacy():
            assert isinstance(permuted_data._tensordict, _PermutedTensorDict)
        assert permuted_data.z == permuted_data.y.z == z

    def test_pickle(self):
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

        assert_allclose_td(
            data.to_tensordict(retain_none=False),
            data2.to_tensordict(retain_none=False),
        )
        assert isinstance(data2, MyData)
        assert data2.z == data.z

    @pytest.mark.parametrize("consolidate", [False, True])
    def test_pickle_consolidate(self, consolidate):
        with set_capture_non_tensor_stack(False):

            tc = TCStrings(a="a", b="b")

            tcstack = TensorDict(tc=torch.stack([tc, tc.clone()]))
            if consolidate:
                tcstack = tcstack.consolidate()
            assert isinstance(tcstack["tc"], TCStrings)
            loaded = pickle.loads(pickle.dumps(tcstack))
            assert isinstance(loaded["tc"], TCStrings)
            assert loaded["tc"].a == tcstack["tc"].a
            assert loaded["tc"].b == tcstack["tc"].b

    def test_post_init(self):
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

    def test_pre_allocate(self):
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

    def test_repeat(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        assert (data.repeat(2, 3) == torch.cat([torch.cat([data] * 2, 0)] * 3, 1)).all()

    def test_repeat_interleave(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        assert data.repeat_interleave(2, dim=1).shape == torch.Size((3, 8))

    def test_reshape(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

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

    def test_set(self):
        @tensorclass
        class MyDataNest:
            X: torch.Tensor
            v: str

        @tensorclass
        class MyDataParent:
            X: Tensor
            z: TensorDictBase
            y: MyDataNest
            v: str
            k: Optional[Tensor] = None

        batch_size = [3, 4]
        X = torch.ones(3, 4, 5)
        td = TensorDict({}, batch_size)
        data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
        data = MyDataParent(
            X=X, y=data_nest, z=td, v="test_tensorclass", batch_size=batch_size
        )

        assert isinstance(data.y, type(data_nest))
        assert data.y._tensordict is data_nest._tensordict
        data.set("X", torch.zeros(3, 4, 5))
        assert (data.X == torch.zeros(3, 4, 5)).all()
        v_new = "test_bluff"
        data.set("v", v_new)
        assert data.v == v_new
        # check that you can't mess up the batch_size
        with pytest.raises(
            RuntimeError,
            match=re.escape("the Tensor smth has shape torch.Size([1]) which"),
        ):
            data.set("z", TensorDict({"smth": torch.zeros(1)}, []))
        # check that you can't write any attribute
        with pytest.raises(AttributeError, match=re.escape("Cannot set the attribute")):
            data.set("newattr", TensorDict({"smth": torch.zeros(1)}, []))

        # Testing nested cases
        data_nest.set("X", torch.zeros(3, 4, 5))
        assert (data_nest.X == torch.zeros(3, 4, 5)).all()
        assert (data.y.X == torch.zeros(3, 4, 5)).all()
        assert data.y.v == "test_nested"
        data.set(("y", "v"), "test_nested_new")
        assert data.y.v == data_nest.v == "test_nested_new"
        data_nest.set("v", "test_nested")
        assert data_nest.v == data.y.v == "test_nested"

        data.set(("y", ("v",)), "this time another string")
        assert data.y.v == data_nest.v == "this time another string"

        # Testing if user can override the type of the attribute
        vorig = torch.ones(3, 4, 5)
        data.set("v", vorig)
        assert (data.v == torch.ones(3, 4, 5)).all()
        assert "v" in data._tensordict.keys()
        assert "v" not in data._non_tensordict.keys()

        data.set("v", torch.zeros(3, 4, 5), inplace=True)
        assert (vorig == 0).all()
        with pytest.raises(RuntimeError, match="Cannot update an existing"):
            data.set("v", "les chaussettes", inplace=True)

        data.set("v", "test")
        assert data.v == "test"
        assert "v" in data._tensordict.keys()

        with pytest.raises(ValueError, match="Failed to update 'v'"):
            data.set("v", vorig, inplace=True)

        # ensure optional fields are writable
        data.set("k", torch.zeros(3, 4, 5))

    def test_select(self):

        @tensorclass
        class Data:
            a: torch.Tensor
            b: torch.Tensor

        data = Data(a=1, b=1)
        assert isinstance(data.a, torch.Tensor)
        assert isinstance(data.b, torch.Tensor)
        assert (data == 1).all()
        data_select = data.select("a")
        assert isinstance(data.a, torch.Tensor)
        assert isinstance(data.b, torch.Tensor)
        assert (data == 1).all()
        assert isinstance(data_select.a, torch.Tensor)
        assert data_select.b is None
        assert "a" in data_select._tensordict
        assert "b" not in data_select._tensordict
        assert (data_select == 1).all()
        assert "a" in data_select._tensordict

    @set_list_to_stack(True)
    def test_set_list_in_constructor(self):
        obj = MyTensorClass(
            a=["a string", "another string"],
            b=[torch.randn(3), torch.zeros(3)],
            c="smth completly different",
            batch_size=2,
        )
        assert obj.shape == (2,)
        assert obj[0].a == "a string"
        assert obj[1].a == "another string"
        assert (obj[0].b != 0).all()
        assert (obj[1].b == 0).all()
        assert obj.c == obj[0].c

    def test_set_dict(self):
        @tensorclass(autocast=True)
        class MyClass:
            x: torch.Tensor
            y: MyClass = None

        c = MyClass(x=torch.zeros((10,)), y={"x": torch.ones((10,))}, batch_size=[10])

        assert isinstance(c.y, MyClass)
        assert c.y.batch_size == c.batch_size

    @pytest.mark.parametrize("any_to_td", [True, False])
    def test_setattr(self, any_to_td):
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
            k: Optional[Tensor] = None

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
            RuntimeError,
            match=re.escape("the Tensor smth has shape torch.Size([1]) which"),
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
        assert "v" in data._tensordict.keys()
        assert "v" not in data._non_tensordict.keys()

        # ensure optional fields are writable
        data.k = torch.zeros(3, 4, 5)

    @pytest.mark.parametrize("list_to_stack", [True, False])
    def test_setitem(self, list_to_stack):
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
        data[1] = data2[1]
        assert (data[1] == data2[1]).all()
        assert data[1].z == ["test_bluff"] * data[1].numel()

        # Validating nested test cases
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: list
            y: "MyDataNested" = None

        X = torch.randn(3, 4, 5)
        z = ["a", "b", "c"]
        batch_size = [3, 4]
        with set_list_to_stack(list_to_stack), (
            pytest.raises(RuntimeError, match="batch dimension mismatch")
            if list_to_stack
            else contextlib.nullcontext()
        ):
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
            data3 = MyDataNested(
                X=X2, y=data_nest2, z=["e", "f"], batch_size=batch_size
            )
            data[:2] = data3[:2]
            assert data[:2].z == data3[:2]._get_str("z", None).tolist()

    @pytest.mark.parametrize(
        "broadcast_type",
        ["scalar", "tensor", "tensordict", "maptensor"],
    )
    @pytest.mark.parametrize("list_to_stack", [True, False])
    def test_setitem_broadcast(self, broadcast_type, list_to_stack):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: list
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = ["a", "b", "c"]
        batch_size = [3, 4]
        with set_list_to_stack(list_to_stack), (
            pytest.raises(RuntimeError, match="batch dimension mismatch")
            if list_to_stack
            else contextlib.nullcontext()
        ):
            data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
            data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)

            if broadcast_type == "scalar":
                val = 0
            elif broadcast_type == "tensor":
                val = torch.zeros(4, 5)
            elif broadcast_type == "tensordict":
                val = TensorDict({"X": torch.zeros(2, 4, 5)}, batch_size=[2, 4])
            elif broadcast_type == "maptensor":
                val = MemoryMappedTensor.from_tensor(torch.zeros(4, 5))

            data[:2] = val
            assert (data[:2] == 0).all()
            assert (data.X[:2] == 0).all()
            assert (data.y.X[:2] == 0).all()

    def test_setitem_memmap(self):
        # regression test PR #203
        # We should be able to set tensors items with MemoryMappedTensors and viceversa
        @tensorclass
        class MyDataMemMap1:
            x: torch.Tensor
            y: MemoryMappedTensor

        data1 = MyDataMemMap1(
            x=torch.zeros(3, 4, 5),
            y=MemoryMappedTensor.from_tensor(torch.zeros(3, 4, 5)),
            batch_size=[3, 4],
        )

        data2 = MyDataMemMap1(
            x=MemoryMappedTensor.from_tensor(torch.ones(3, 4, 5)),
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

    def test_setitem_other_cls(self):
        @tensorclass
        class MyData1:
            x: torch.Tensor
            y: MemoryMappedTensor

        data1 = MyData1(
            x=torch.zeros(3, 4, 5),
            y=MemoryMappedTensor.from_tensor(torch.zeros(3, 4, 5)),
            batch_size=[3, 4],
        )

        # Set Item should work for other tensorclass
        @tensorclass
        class MyData2:
            x: MemoryMappedTensor
            y: torch.Tensor

        data_other_cls = MyData2(
            x=MemoryMappedTensor.from_tensor(torch.ones(3, 4, 5)),
            y=torch.ones(3, 4, 5),
            batch_size=[3, 4],
        )
        data1[:2] = data_other_cls[:2]
        data_other_cls[2:] = data1[2:]

        # Set Item should raise if other tensorclass with different members
        @tensorclass
        class MyData3:
            x: MemoryMappedTensor
            z: torch.Tensor

        data_wrong_cls = MyData3(
            x=MemoryMappedTensor.from_tensor(torch.ones(3, 4, 5)),
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

    def test_signature(self):
        sig = inspect.signature(MyData)
        assert list(sig.parameters) == ["X", "y", "z", "batch_size", "device", "names"]

        with pytest.raises(TypeError, match="missing 3 required positional arguments"):
            MyData(batch_size=[10])

        with pytest.raises(TypeError, match="missing 2 required positional argument"):
            MyData(X=torch.rand(10), batch_size=[10])

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            MyData(X=torch.rand(10), y=torch.rand(10), batch_size=[10], device="cpu")

        # No batch_size is empty batch size
        assert MyData(
            X=torch.rand(10), y=torch.rand(10), z="str"
        ).batch_size == torch.Size([])

        # all positional arguments + batch_size is fine
        MyData(
            X=torch.rand(10), y=torch.rand(10), z="test_tensorclass", batch_size=[10]
        )

    def test_split(self):
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

        assert split_tcs[0].y.batch_size == torch.Size([3, 3])
        assert split_tcs[1].y.batch_size == torch.Size([3, 2])
        assert split_tcs[2].y.batch_size == torch.Size([3, 1])

        assert torch.all(torch.eq(split_tcs[0].X, torch.ones(3, 3, 5)))
        assert torch.all(torch.eq(split_tcs[0].y[0].X, torch.ones(3, 3, 5)))
        assert split_tcs[0].z == split_tcs[1].z == split_tcs[2].z == z
        assert split_tcs[0].y[0].z == split_tcs[0].y[1].z == split_tcs[0].y[2].z == z

    def test_update(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

            @classmethod
            def get_data(cls, shift):
                X = torch.zeros(1, 4, 5) + shift
                z = f"test_tensorclass{shift}"
                batch_size = [1, 4]
                data_nest = cls(X=X, z=z, batch_size=batch_size)
                data = cls(X=X, y=data_nest, z=z, batch_size=batch_size)
                return data

        data1 = MyDataNested.get_data(1)
        # for _data1 in (data1, data1.to_dict(), data1.to_tensordict()):
        data0 = MyDataNested.get_data(0)
        data0.update(data1)
        assert (data0.X == 1).all()
        assert data0.z == "test_tensorclass1"
        assert (data0.y.X == 1).all()
        assert data0.y.z == "test_tensorclass1"
        data0 = MyDataNested.get_data(0)
        data0.update(data1.to_dict(retain_none=False))
        assert (data0.X == 1).all()
        assert data0.z == "test_tensorclass1", data0.z
        assert (data0.y.X == 1).all()
        assert data0.y.z == "test_tensorclass1"

        data0 = MyDataNested.get_data(0)
        data0.update(data1.to_tensordict(retain_none=False))
        assert (data0.X == 1).all()
        assert data0.z == "test_tensorclass1"
        assert (data0.y.X == 1).all()
        assert data0.y.z == "test_tensorclass1"

    def test_replace(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(1, 4, 5)
        z = "test_tensorclass"
        batch_size = [1, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)

        replacement = data.clone().zero_()
        replacement.z = "replacement"
        replacement.y.z = "replacement"
        assert data.z == "test_tensorclass"
        assert data.y.z == "test_tensorclass"
        data_replace = data.replace(replacement)

        assert isinstance(data_replace, MyDataNested)
        assert isinstance(data_replace.y, MyDataNested)
        assert data.z == "test_tensorclass"
        assert data.y.z == "test_tensorclass"
        assert data_replace.z == "replacement"
        assert data_replace.y.z == "replacement"

        assert (data.X == 1).all()
        assert (data.y.X == 1).all()
        assert (data_replace.X == 0).all()
        assert (data_replace.y.X == 0).all()

    def test_squeeze(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

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

    @set_capture_non_tensor_stack(False)
    @pytest.mark.parametrize("lazy", [True, False, "maybe"])
    def test_stack(self, lazy):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        if lazy:
            Xb = torch.randn(3, 4, 4)
        else:
            Xb = X.clone()
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data_nest_b = MyDataNested(X=Xb, z=z, batch_size=batch_size)
        data1 = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        data2 = MyDataNested(X=Xb, y=data_nest_b, z=z, batch_size=batch_size)

        if lazy is True:
            stacked_tc = LazyStackedTensorDict.lazy_stack([data1, data2], 0)
        elif lazy == "maybe":
            stacked_tc = LazyStackedTensorDict.maybe_dense_stack([data1, data2], 0)
        else:
            with set_capture_non_tensor_stack(True):
                stacked_tc = torch.stack([data1, data2], 0)
        assert type(stacked_tc) is type(data1)
        assert isinstance(stacked_tc.y, type(data1.y))
        if not lazy:
            assert stacked_tc.X.shape == torch.Size([2, 3, 4, 5])
            assert stacked_tc.y.X.shape == torch.Size([2, 3, 4, 5])

            assert (stacked_tc.X == 1).all()
            assert (stacked_tc.y.X == 1).all()
        else:
            assert stacked_tc[0].X.shape == torch.Size([3, 4, 5])
            assert stacked_tc[0].y.X.shape == torch.Size([3, 4, 5])
            assert stacked_tc[1].X.shape == torch.Size([3, 4, 4])
            assert stacked_tc[1].y.X.shape == torch.Size([3, 4, 4])
            assert (stacked_tc[0].X == 1).all()
            assert (stacked_tc[0].y.X == 1).all()

        if lazy_legacy() or lazy:
            assert isinstance(stacked_tc._tensordict, LazyStackedTensorDict)
            assert isinstance(stacked_tc.y._tensordict, LazyStackedTensorDict)
        zlist = z
        if lazy:
            for d in range(stacked_tc.ndim - 1, -1, -1):
                zlist = [zlist] * stacked_tc.batch_size[d]
        assert stacked_tc.z == stacked_tc.y.z
        assert stacked_tc.z == zlist

        # Testing negative scenarios
        y = torch.zeros(3, 4, 5, dtype=torch.bool)
        data3 = MyData(X=X, y=y, z=z, batch_size=batch_size)

        with pytest.raises(
            TypeError,
            match=("Multiple dispatch failed|no implementation found"),
        ):
            torch.stack([data1, data3], dim=0)

    def test_stack_keyorder(self):

        class MyTensorClass(TensorClass):
            foo: Tensor
            bar: Tensor

        tc1 = MyTensorClass(foo=torch.zeros((1,)), bar=torch.ones((1,)))

        for _ in range(10000):
            assert list(torch.stack([tc1, tc1], dim=0)._tensordict.keys()) == [
                "foo",
                "bar",
            ]

    def test_statedict_errors(self):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            z: str
            y: "MyClass" = None

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
        sd["_tensordict"]["y"]["_tensordict"]["a"] = None
        with pytest.raises(KeyError, match="Key 'a' wasn't expected in the state-dict"):
            tc.load_state_dict(sd)

    def test_tensorclass_get_at(self):
        @tensorclass
        class MyDataNest:
            X: torch.Tensor
            v: str

        @tensorclass
        class MyDataParent:
            X: Tensor
            z: TensorDictBase
            y: MyDataNest
            v: str
            k: Optional[Tensor] = None

        batch_size = [3, 4]
        X = torch.ones(3, 4, 5)
        td = TensorDict({}, batch_size)
        data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
        v = "test_tensorclass"
        data = MyDataParent(X=X, y=data_nest, z=td, v=v, batch_size=batch_size)

        assert (data.get("X")[2:3] == data.get_at("X", slice(2, 3))).all()
        assert (data.get(("y", "X"))[2:3] == data.get_at(("y", "X"), slice(2, 3))).all()

        # check default
        assert data.get_at(("y", "foo"), slice(2, 3), "working") == "working"
        assert data.get_at("foo", slice(2, 3), "working") == "working"

    def test_tensorclass_set_at_(self):
        @tensorclass
        class MyDataNest:
            X: torch.Tensor
            v: str

        @tensorclass
        class MyDataParent:
            X: Tensor
            z: TensorDictBase
            y: MyDataNest
            v: str
            k: Optional[Tensor] = None

        batch_size = [3, 4]
        X = torch.ones(3, 4, 5)
        td = TensorDict({}, batch_size)
        data_nest = MyDataNest(X=X, v="test_nested", batch_size=batch_size)
        v = "test_tensorclass"
        data = MyDataParent(X=X, y=data_nest, z=td, v=v, batch_size=batch_size)

        data.set_at_("X", 5, slice(2, 3))
        data.set_at_(("y", "X"), 5, slice(2, 3))
        assert (data.get_at("X", slice(2, 3)) == 5).all()
        assert (data.get_at(("y", "X"), slice(2, 3)) == 5).all()
        # assert other not changed
        assert (data.get_at("X", slice(0, 2)) == 1).all()
        assert (data.get_at(("y", "X"), slice(0, 2)) == 1).all()
        assert (data.get_at("X", slice(3, 5)) == 1).all()
        assert (data.get_at(("y", "X"), slice(3, 5)) == 1).all()

    def test_to_tensordict(self):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            z: str
            y: "MyClass" = None

        c = MyClass(
            torch.randn(3, 4),
            "foo",
            MyClass(torch.randn(3, 4, 5), "bar", None, batch_size=[3, 4, 5]),
            batch_size=[3, 4],
        )

        ctd = c.to_tensordict(retain_none=False)
        assert isinstance(ctd, TensorDictBase)
        assert "x" in ctd.keys()
        assert "z" in ctd.keys()
        assert "y" in ctd.keys()
        assert ("y", "x") in ctd.keys(True)

    @pytest.mark.skipif(
        not _has_torchsnapshot,
        reason=f"torchsnapshot not found: err={TORCHSNAPSHOT_ERR}",
    )
    def test_torchsnapshot(self, tmp_path):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            z: str
            y: "MyClass" = None

        z = "test_tensorclass"
        tc = MyClass(
            x=torch.randn(3),
            z=z,
            y=MyClass(x=torch.randn(3), z=z, batch_size=[]),
            batch_size=[],
        )
        tc.memmap_()
        assert isinstance(tc.y.x, MemoryMappedTensor)
        assert tc.z == z

        app_state = {
            "state": torchsnapshot.StateDict(tensordict=tc.state_dict(keep_vars=True))
        }
        snapshot = torchsnapshot.Snapshot.take(app_state=app_state, path=str(tmp_path))

        tc_dest = MyClass(
            x=torch.randn(3),
            z="other",
            y=MyClass(x=torch.randn(3), z=z, batch_size=[]),
            batch_size=[],
        )
        tc_dest.memmap_()
        assert isinstance(tc_dest.y.x, MemoryMappedTensor)
        app_state = {
            "state": torchsnapshot.StateDict(
                tensordict=tc_dest.state_dict(keep_vars=True)
            )
        }
        snapshot.restore(app_state=app_state)

        assert (tc_dest == tc).all()
        assert tc_dest.y.batch_size == tc.y.batch_size
        assert isinstance(tc_dest.y.x, MemoryMappedTensor)
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
        assert isinstance(tc_dest.y.x, MemoryMappedTensor)
        # load_state_dict outperforms snapshot in this case
        assert tc_dest.z == z

    def test_type(self):
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

    def test_unbind(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

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

    def test_unsqueeze(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

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

    def test_view(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        viewed_td = data.view(-1)
        assert viewed_td.X.shape == torch.Size([12, 5])
        assert viewed_td.y.X.shape == torch.Size([12, 5])
        assert viewed_td.shape == torch.Size([12])
        assert (viewed_td.X == 1).all()
        if lazy_legacy():
            assert isinstance(viewed_td._tensordict, _ViewedTensorDict)
        assert viewed_td.z == viewed_td.y.z == z

    def test_weakref_attr(self):
        @tensorclass
        class Y:
            _z: weakref.ref

            @property
            def z(self) -> torch.Tensor:
                return self._z()

        obj = torch.ones(())
        y0 = Y(weakref.ref(obj), batch_size=[1])
        y1 = Y(weakref.ref(obj), batch_size=[1])
        y = torch.cat([y0, y1])
        assert y.z.shape == torch.Size(())
        with set_capture_non_tensor_stack(True):
            y = torch.stack([y0, y1])
        assert y.z.shape == torch.Size(())


class TestMemmap:
    def test_from_memmap(self, tmpdir):
        td = TensorDict(
            {
                ("a", "b", "c"): 1,
                ("a", "d"): 2,
            },
            [],
        ).expand(10)

        @tensorclass
        class MyClass:
            a: TensorDictBase

        MyClass._from_tensordict(td).memmap_(tmpdir)

        tc = MyClass.load_memmap(tmpdir)
        assert isinstance(tc.a, TensorDict)
        assert tc.batch_size == torch.Size([10])

    def test_load_scenarios(self, tmpdir):
        @tensorclass
        class MyClass:
            X: torch.Tensor
            td: TensorDict
            integer: int
            string: str
            dictionary: dict

        @tensorclass
        class MyOtherClass:
            Y: torch.Tensor

        data = MyClass(
            X=torch.randn(10, 3),
            td=TensorDict({"y": torch.randn(10)}, batch_size=[10]),
            integer=3,
            string="a string",
            dictionary={"some_data": "a"},
            batch_size=[],
        )

        data.memmap_(tmpdir)
        data2 = MyClass.load_memmap(tmpdir)
        assert (data2 == data).all()
        data.apply_(lambda x: x + 1)
        assert (data2 == data).all()
        data3 = MyOtherClass.load_memmap(tmpdir)
        assert isinstance(data3, MyClass)

    def test_memmap_(self):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            z: str
            y: "MyClass" = None

        c = MyClass(
            torch.randn(3, 4),
            "foo",
            MyClass(torch.randn(3, 4, 5), "bar", None, batch_size=[3, 4, 5]),
            batch_size=[3, 4],
        )

        cmemmap = c.memmap_()
        assert cmemmap is c
        assert isinstance(c.x, MemoryMappedTensor)
        assert isinstance(c.y.x, MemoryMappedTensor)
        assert c.z == "foo"

    def test_memmap_like(self):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            z: str
            y: "MyClass" = None

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
        assert isinstance(cmemmap.x, MemoryMappedTensor)
        assert isinstance(cmemmap.y.x, MemoryMappedTensor)
        assert cmemmap.z == "foo"
        assert cmemmap.is_memmap()


class TestNesting:
    @tensorclass
    class TensorClass:
        tens: torch.Tensor
        order: Tuple[str]
        test: str

    def get_nested(self):
        c = self.TensorClass(torch.ones(1), ("a", "b", "c"), "Hello", batch_size=[])

        with set_capture_non_tensor_stack(True):
            td = torch.stack(
                [
                    TensorDict({"t": torch.ones(1), "c": c}, batch_size=[])
                    for _ in range(3)
                ]
            )
        return td

    def test_apply(self):
        td = self.get_nested()
        td = td.apply(lambda x: x + 1)
        assert isinstance(td.get("c")[0], self.TensorClass)

    def test_chunk(self):
        td = self.get_nested()
        td, _ = td.chunk(2, dim=0)
        assert isinstance(td.get("c")[0], self.TensorClass)

    def test_idx(self):
        td = self.get_nested()[0]
        assert isinstance(td.get("c"), self.TensorClass)

    def test_split(self):
        td = self.get_nested()
        td, _ = td.split([2, 1], dim=0)
        assert isinstance(td.get("c")[0], self.TensorClass)

    def test_to(self):
        td = self.get_nested()
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu:1")
        td_device = td.to(device)
        assert isinstance(td_device.get("c")[0], self.TensorClass)
        assert td_device is not td
        assert td_device.device == device

        td_device = td.to(device, inplace=True)
        assert td_device is td
        assert td_device.device == device

        td_cpu = td_device.to("cpu", inplace=True)
        assert td_cpu.device == torch.device("cpu")

        td_double = td.to(torch.float64, inplace=True)
        assert td_double is td
        assert td_double.dtype == torch.double
        assert td_double.device == torch.device("cpu")


@tensorclass(autocast=True)
class AutoCast:
    tensor: torch.Tensor
    non_tensor: str
    td: TensorDict
    tc: AutoCast


@tensorclass(autocast=True)
class AutoCastOr:
    tensor: torch.Tensor
    non_tensor: str
    td: TensorDict
    tc: AutoCast | None = None


@tensorclass(autocast=True)
class AutoCastOptional:
    tensor: torch.Tensor
    non_tensor: str
    td: TensorDict
    # DO NOT CHANGE Optional
    tc: Optional[AutoCast] = None


@tensorclass(autocast=True)
class AutoCastTensor:
    tensor: torch.Tensor
    integer: int
    string: str
    floating: float
    numpy_array: np.ndarray
    anything: Any


class TestNoCasting:
    def test_nocast_int(self):
        @tensorclass(nocast=False)
        class X:
            a: int  # type is irrelevant

        assert isinstance(X(1).a, torch.Tensor)

        @tensorclass(nocast=True)
        class X:
            a: int  # type is irrelevant

        assert isinstance(X(1).a, int)

    def test_nocast_np(self):
        @tensorclass(nocast=False)
        class X:
            a: int  # type is irrelevant

        assert isinstance(X(np.array([1])).a, torch.Tensor)

        @tensorclass(nocast=True)
        class X:
            a: int  # type is irrelevant

        assert isinstance(X(np.array([1])).a, np.ndarray)

    def test_nocast_bool(self):
        @tensorclass(nocast=False)
        class X:
            a: int  # type is irrelevant

        assert isinstance(X(True).a, torch.Tensor)

        @tensorclass(nocast=True)
        class X:
            a: int  # type is irrelevant

        assert isinstance(X(False).a, bool)

    def test_exclusivity(self):
        with pytest.raises(ValueError, match="exclusive"):

            @tensorclass(nocast=True, autocast=True)
            class X:
                a: int  # type is irrelevant


class TestAutoCasting:
    @tensorclass(autocast=True)
    class ClsAutoCast:
        tensor: torch.Tensor
        non_tensor: str
        td: TensorDict
        tc: "ClsAutoCast"  # noqa: F821
        tc_global: AutoCast

    def test_autocast_attr(self):
        @tensorclass(autocast=False)
        class T:
            X: torch.Tensor

        assert not T._autocast

        @tensorclass
        class T:
            X: torch.Tensor

        assert not T._autocast

        @tensorclass(autocast=True)
        class T:
            X: torch.Tensor

        assert T._autocast

    def test_autocast_simple(self):
        obj = AutoCastTensor(
            tensor=1,
            integer=1,
            string=1,
            floating=1,
            numpy_array=1,
            anything=1,
        )
        assert isinstance(obj.tensor, torch.Tensor)
        assert isinstance(obj.integer, int)
        assert isinstance(obj.string, str), type(obj.string)
        assert isinstance(obj.floating, float)
        assert isinstance(obj.numpy_array, np.ndarray)
        assert isinstance(obj.anything, torch.Tensor)
        obj.tensor = 1.0
        assert isinstance(obj.tensor, torch.Tensor)
        with pytest.raises(TypeError):
            obj.tensor = "str"
        obj.anything = 1.0
        assert isinstance(obj.anything, torch.Tensor)
        obj.anything = "str"

    def test_autocast(self):
        # Autocasting is implemented only for tensordict / tensorclasses.
        # Since some type annotations are not supported such as `Tensor | None`,
        # we don't want to encourage this feature too much, as it will break
        # in many cases.
        obj = AutoCast(
            tensor=torch.zeros(()),
            non_tensor="x",
            td={"a": 0.0},
            tc={
                "tensor": torch.zeros(()),
                "non_tensor": "y",
                "td": {"b": 0.0},
                "tc": None,
            },
        )

        assert isinstance(obj, AutoCast), type(obj)
        assert isinstance(obj.tensor, torch.Tensor)
        assert isinstance(obj.non_tensor, str)
        assert isinstance(obj.td, TensorDict)
        assert isinstance(obj.tc, AutoCast), (type(obj.tc), type(obj))

        assert isinstance(obj.tc.tensor, torch.Tensor)
        assert isinstance(obj.tc.non_tensor, str)
        assert isinstance(obj.tc.td, TensorDict)
        assert obj.tc.tc is None

    def test_autocast_cls(self):
        obj = self.ClsAutoCast(
            tensor=torch.zeros(()),
            non_tensor="x",
            td={"a": 0.0},
            tc={
                "tensor": torch.zeros(()),
                "non_tensor": "y",
                "td": {"b": 0.0},
                "tc": None,
            },
            tc_global=AutoCast(
                tensor=torch.zeros(()), non_tensor="x", td={"a": 0.0}, tc=None
            ),
        )

        assert isinstance(obj.tensor, torch.Tensor)
        assert isinstance(obj.non_tensor, str)
        assert isinstance(obj.td, TensorDict)
        if not PY8:
            assert isinstance(obj.tc, self.ClsAutoCast), (type(obj.tc), type(obj))
        else:
            assert isinstance(obj.tc, dict), (type(obj.tc), type(obj))

        assert isinstance(obj.tc_global, AutoCast), (type(obj.tc), type(obj))

        if not PY8:
            assert isinstance(obj.tc.tensor, torch.Tensor)
            assert isinstance(obj.tc.non_tensor, str)
            assert isinstance(obj.tc.td, TensorDict)
            assert obj.tc.tc is None

    def test_autocast_or(self):
        with (
            pytest.warns(
                UserWarning, match="This may be caused by annotations that use plain"
            )
            if not PY10
            else contextlib.nullcontext()
        ):
            obj = AutoCastOr(
                tensor=torch.zeros(()),
                non_tensor="x",
                td={"a": 0.0},
                tc={
                    "tensor": torch.zeros(()),
                    "non_tensor": "y",
                    "td": {"b": 0.0},
                    "tc": None,
                },
            )

        assert isinstance(obj.tensor, torch.Tensor)
        assert isinstance(obj.non_tensor, str)
        if not PY10:
            assert not isinstance(obj.td, TensorDict)
        else:
            assert isinstance(obj.td, TensorDict)
        assert not isinstance(obj.tc, AutoCast), (type(obj.tc), type(obj))

        assert isinstance(obj.tc["tensor"], torch.Tensor)
        assert isinstance(obj.tc["non_tensor"], str)
        assert not isinstance(obj.tc["td"], TensorDict)
        assert obj.tc["tc"] is None

    def test_autocast_optional(self):
        obj = AutoCastOptional(
            tensor=torch.zeros(()),
            non_tensor="x",
            td={"a": 0.0},
            tc={
                "tensor": torch.zeros(()),
                "non_tensor": "y",
                "td": {"b": 0.0},
                "tc": None,
            },
        )

        assert isinstance(obj.tensor, torch.Tensor)
        assert isinstance(obj.non_tensor, str)
        # With Optional, no error is raised
        assert isinstance(obj.td, TensorDict)
        assert not isinstance(obj.tc, AutoCast), (type(obj.tc), type(obj))

        assert isinstance(obj.tc["tensor"], torch.Tensor)
        assert isinstance(obj.tc["non_tensor"], str)
        assert not isinstance(obj.tc["td"], TensorDict)
        assert obj.tc["tc"] is None

    def test_autocast_func(self):
        @tensorclass(autocast=True)
        class FuncAutoCast:
            tensor: torch.Tensor
            non_tensor: str
            td: TensorDict
            tc: FuncAutoCast
            tc_global: AutoCast
            tc_cls: TestAutoCasting.ClsAutoCast

        obj = FuncAutoCast(
            tensor=torch.zeros(()),
            non_tensor="x",
            td={"a": 0.0},
            tc={
                "tensor": torch.zeros(()),
                "non_tensor": "y",
                "td": {"b": 0.0},
                "tc": None,
            },
            tc_global={
                "tensor": torch.zeros(()),
                "non_tensor": "x",
                "td": {"a": 0.0},
                "tc": None,
            },
            tc_cls={
                "tensor": torch.zeros(()),
                "non_tensor": "x",
                "td": {"a": 0.0},
                "tc": None,
                "tc_global": {
                    "tensor": torch.zeros(()),
                    "non_tensor": "x",
                    "td": {"a": 0.0},
                    "tc": None,
                },
            },
        )

        assert isinstance(obj.tensor, torch.Tensor)
        assert isinstance(obj.non_tensor, str)
        assert isinstance(obj.td, TensorDict)
        assert isinstance(obj.tc, FuncAutoCast), (type(obj.tc), type(obj))
        assert isinstance(obj.tc_cls, self.ClsAutoCast), (type(obj.tc), type(obj))
        assert isinstance(obj.tc_global, AutoCast), (type(obj.tc), type(obj))

        assert isinstance(obj.tc.tensor, torch.Tensor)
        assert isinstance(obj.tc.non_tensor, str)
        assert isinstance(obj.tc.td, TensorDict)
        assert obj.tc.tc is None


class TestShadow:
    def test_no_shadow(self):
        with pytest.raises(AttributeError):

            @tensorclass
            class MyClass:
                x: str
                y: int
                batch_size: Any

        with pytest.raises(AttributeError):

            @tensorclass
            class MyClass:  # noqa: F811
                x: str
                y: int
                names: Any

        with pytest.raises(AttributeError):

            @tensorclass
            class MyClass:  # noqa: F811
                x: str
                y: int
                device: Any

        @tensorclass(shadow=True)
        class MyClass:  # noqa: F811
            x: str
            y: int
            batch_size: Any
            names: Any
            device: Any

    def test_shadow_values_dec(self):

        @tensorclass(shadow=True)
        class MyClass:
            batch_size: Any
            names: Any
            device: Any

        c = MyClass(batch_size=0, names=0, device=0)
        assert c.batch_size == 0
        assert c.names == 0
        assert c.device == 0
        c.batch_size = 1
        assert c.batch_size == 1

    def test_shadow_values_dec_subcls(self):
        @tensorclass(shadow=True)
        class MyClass:
            batch_size: Any
            names: Any
            device: Any

        class MyClsSubcls(MyClass): ...

        c = MyClsSubcls(batch_size=0, names=0, device=0)
        assert c.batch_size == 0
        assert c.names == 0
        assert c.device == 0
        c.batch_size = 1
        assert c.batch_size == 1

    def test_shadow_values_subcls(self):

        class MyClassSbcls(TensorClass, shadow=True):
            batch_size: Any
            names: Any
            device: Any

        c = MyClassSbcls(batch_size=0, names=0, device=0)
        assert c.batch_size == 0
        assert c.names == 0
        assert c.device == 0

    def test_shadow_values_subcls_idx(self):

        class MyClassSbcls(TensorClass["shadow"]):
            batch_size: Any
            names: Any
            device: Any

        c = MyClassSbcls(batch_size=0, names=0, device=0)
        assert c.batch_size == 0
        assert c.names == 0
        assert c.device == 0

    def test_shadow_repr(self):
        @tensorclass(shadow=True)
        class MyClass:
            batch_size: Any
            names: Any
            device: Any

        c = MyClass(batch_size=0, names=0, device=0)
        assert (
            repr(c)
            == """MyClass(
    batch_size=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
    device=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
    names=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
    shape=torch.Size([]),
    is_shared=False)"""
        )


class TestVMAP:
    def test_regular_vmap(self):
        @tensorclass
        class VmappableClass:
            x: torch.Tensor
            y: torch.Tensor

        data = VmappableClass(
            x=torch.zeros(3, 4), y=torch.ones(3, 4), batch_size=[3, 4]
        )

        def assert_is_data(x):
            assert isinstance(x, VmappableClass)
            return x

        data_bis = torch.vmap(assert_is_data)(data)
        assert isinstance(data_bis, VmappableClass)
        assert (data_bis == data).all()

    def test_non_tensor_vmap(self):
        @tensorclass
        class VmappableClass:
            x: torch.Tensor
            y: torch.Tensor
            non_tensor: Any

        data = VmappableClass(
            x=torch.zeros(3, 4),
            y=torch.ones(3, 4),
            non_tensor="a string!",
            batch_size=[3, 4],
        )

        def assert_is_data(x):
            assert isinstance(x, VmappableClass)
            return x

        data_bis = torch.vmap(assert_is_data)(data)
        assert isinstance(data_bis, VmappableClass)
        assert "non_tensor" in data_bis._tensordict
        assert "non_tensor" not in data_bis._non_tensordict
        assert (data_bis == data).all()
        assert data_bis.non_tensor == "a string!"


class TestSerialization:
    def test_save_load(self, tmpdir):
        myc = MyData(
            X=torch.rand(2, 3, 4),
            y=torch.rand(2, 3, 4, 5),
            z="test_tensorclass",
            batch_size=[2, 3],
        )
        myc.save(tmpdir)
        tensordict.utils.print_directory_tree(tmpdir)
        myc_load = TensorDict.load(tmpdir)
        assert myc_load.z == "test_tensorclass"
        assert (myc == myc_load).all()


class TestPointWise:
    def test_pointwise(self):
        @tensorclass
        class X:
            a: torch.Tensor
            b: str

        x = X(torch.zeros(()), "a string")
        assert (x + 1).b == "a string"
        x += 1
        assert x.a == 1
        assert (x.add(1) == (x + 1)).all()
        assert (x.mul(2) == (x * 2)).all()
        assert (x.div(2) == (x / 2)).all()

    def test_logic_and_right_ops(self):
        @tensorclass
        class MyClass:
            x: str

        c = MyClass(torch.randn(10))
        _ = c < 0
        _ = c > 0
        _ = c <= 0
        _ = c >= 0
        _ = c != 0

        _ = c.bool() ^ True
        _ = True ^ c.bool()

        _ = c.bool() | False
        _ = False | c.bool()

        _ = c.bool() & False
        _ = False & c.bool()

        _ = abs(c)

        _ = c + 1
        _ = 1 + c
        c += 1

        _ = c * 1
        _ = 1 * c

        _ = c - 1
        _ = 1 - c
        c -= 1

        _ = c / 1
        _ = 1 / c

        _ = c**1
        # not implemented
        # 1 ** c


class TestSubClassing:
    def test_subclassing(self):
        class SubClass(TensorClass):
            a: int

        assert is_tensorclass(SubClass)
        assert not SubClass._autocast
        assert not SubClass._nocast
        assert issubclass(SubClass, TensorClass)

    def test_subclassing_autocast(self):
        class SubClass(TensorClass, autocast=True):
            a: int

        assert is_tensorclass(SubClass)
        assert SubClass._autocast
        assert not SubClass._nocast
        assert issubclass(SubClass, TensorClass)
        assert isinstance(SubClass(torch.ones(())).a, int)

        class SubClass(TensorClass["autocast"]):
            a: int

        assert not TensorClass._autocast
        assert is_tensorclass(SubClass)
        assert SubClass._autocast
        assert not SubClass._nocast
        assert issubclass(SubClass, TensorClass)
        assert isinstance(SubClass(torch.ones(())).a, int)

    def test_subclassing_nocast(self):
        class SubClass(TensorClass, nocast=True):
            a: int

        assert is_tensorclass(SubClass)
        assert not SubClass._autocast
        assert SubClass._nocast
        assert issubclass(SubClass, TensorClass)
        assert isinstance(SubClass(1).a, int)

        class SubClass(TensorClass["nocast"]):
            a: int

        assert not TensorClass._nocast
        assert is_tensorclass(SubClass)
        assert not SubClass._autocast
        assert SubClass._nocast
        assert issubclass(SubClass, TensorClass)
        assert isinstance(SubClass(1).a, int)

    def test_subclassing_mult(self):
        class SubClass(TensorClass, nocast=True, frozen=True):
            a: int

        assert is_tensorclass(SubClass)
        assert not SubClass._autocast
        assert SubClass._nocast
        assert SubClass._frozen
        assert issubclass(SubClass, TensorClass)
        s = SubClass(1)
        assert isinstance(s.a, int)
        with pytest.raises((RuntimeError, dataclasses.FrozenInstanceError)):
            s.a = 2

        class SubClass(TensorClass["nocast", "frozen"]):
            a: int

        assert not TensorClass._nocast
        assert not TensorClass._frozen
        assert is_tensorclass(SubClass)
        assert SubClass._nocast
        assert SubClass._frozen
        assert issubclass(SubClass, TensorClass)
        s = SubClass(1)
        assert isinstance(s.a, int)
        with pytest.raises((RuntimeError, dataclasses.FrozenInstanceError)):
            s.a = 2

    def test_subclassing_super_call(self):
        class SubClass(TensorClass, nocast=True):
            a: int
            b: int

            def __setattr__(self, key, value):
                if key == "b":
                    return super().__setattr__("b", value + 1)
                return super().__setattr__("a", value - 1)

        s = SubClass(a=torch.zeros(3), b=torch.zeros(3))
        assert (s.a == -1).all()
        assert (s.b == 1).all()
        s.a = torch.ones(())
        s.b = torch.ones(())
        assert (s.a == 0).all()
        assert (s.b == 2).all()


class TestTensorOnly:
    class TensorOnly(TensorClass["tensor_only"]):
        a: torch.Tensor
        b: torch.Tensor
        c: torch.Tensor | None = None

    def test_tensor_only_base(self):
        x = self.TensorOnly(1, 2, 3)
        assert x.a == 1
        assert x.b == 2
        assert x.c == 3
        assert isinstance(x.a, torch.Tensor)
        assert isinstance(x.b, torch.Tensor)
        assert isinstance(x.c, torch.Tensor)

    def test_tensor_only_none(self):
        x = self.TensorOnly(1, 2)
        assert x.a == 1
        assert x.b == 2
        assert x.c is None
        x.c = 3
        assert x.c == 3
        assert isinstance(x.c, torch.Tensor)
        delattr(x, "c")
        assert not hasattr(x, "c")

    @pytest.mark.skipif(PY9, reason="3.9 not supported for type checks")
    def test_wrong_tensor_only(self):
        class TensorOnly(TensorClass["tensor_only"]):
            a: torch.IntTensor
            b: torch.LongTensor
            c: torch.Tensor | None = None
            d: torch.Tensor | Union[torch.IntTensor, torch.LongTensor] | None = (
                None  # noqa
            )
            e: Optional[torch.IntTensor] = None  # noqa
            f: Optional[torch.IntTensor | None] = None  # noqa
            g: TensorDict | None = None
            h: MyTensorClass | None = None

        with pytest.raises(
            TypeError,
            match="tensor_only requires types to be Tensor, Tensor-subtrypes or None",
        ):

            class TensorOnlyAny(TensorClass["tensor_only"]):
                a: torch.Tensor
                b: Any
                c: torch.Tensor | None = None

        with pytest.raises(
            TypeError,
            match="tensor_only requires types to be Tensor, Tensor-subtrypes or None",
        ):

            class TensorOnlyStr(TensorClass["tensor_only"]):
                a: torch.Tensor
                b: torch.Tensor | str
                c: torch.Tensor | None = None

        with pytest.raises(
            TypeError,
            match="tensor_only requires types to be Tensor, Tensor-subtrypes or None",
        ):

            class TensorOnlyStrUnion(TensorClass["tensor_only"]):
                a: torch.Tensor
                b: torch.Tensor
                c: torch.Tensor | Union[torch.IntTensor, str] | None = None  # noqa


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
