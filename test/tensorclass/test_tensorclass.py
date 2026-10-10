# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import dataclasses
import importlib.util
import inspect
import json
import os
import pathlib
import pickle
import re
import sys
import textwrap
import weakref
from collections import UserDict
from dataclasses import field
from multiprocessing import Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    ClassVar,
    Generic,
    get_args,
    get_origin,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pytest
import tensordict.utils
import torch
from _utils_internal import is_npu_available
from tensordict import (
    assert_allclose_td,
    is_tensorclass,
    lazy_legacy,
    LazyStackedTensorDict,
    MemoryMappedTensor,
    MetaData,
    NonTensorData,
    set_capture_non_tensor_stack,
    set_get_defaults_to_none,
    set_list_to_stack,
    TensorClass,
    tensorclass,
    TensorDict,
    TensorDictBase,
)
from tensordict._lazy import _PermutedTensorDict, _ViewedTensorDict
from tensordict._td import lazy_stack
from tensordict._utils_options import _set_capture_non_tensor_stack, _set_list_to_stack
from tensordict.base import (
    _GENERIC_NESTED_ERR,
    _get_defaults_to_none,
    _set_get_defaults_to_none,
)
from tensordict.tensorclass import from_dataclass
from tensordict.utils import _check_recursive_properties
from torch import Tensor

_has_streaming = importlib.util.find_spec("streaming", None) is not None

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.tensordict.test._utils_internal import get_available_devices
else:
    IS_FB = False
    from _utils_internal import get_available_devices

# Capture all warnings
pytestmark = [
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings(
        "ignore:type_hints are none, cannot perform auto-casting"
    ),
    pytest.mark.filterwarnings(
        "ignore:You are using `torch.load` with `weights_only=False`"
    ),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]

IS_FB = os.getenv("PYTORCH_TEST_FBCODE")


_TENSORDICT_DIR = pathlib.Path(__file__).parents[2] / "tensordict"


def _get_class_attrs_from_pyi(file_path, class_name):
    """
    Reads a .pyi file and returns the names that one of its classes declares.

    Args:
        file_path (str): Path to the .pyi file.
        class_name (str): Name of the class in the .pyi file.

    Returns:
        set: The names of the methods, properties and attributes of the class.
    """
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())

    (class_node,) = (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    attrs = set()
    for child_node in class_node.body:
        if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            attrs.add(child_node.name)
        elif isinstance(child_node, ast.AnnAssign):
            attrs.add(child_node.target.id)
        elif isinstance(child_node, ast.Assign):
            attrs.update(target.id for target in child_node.targets)
    return attrs


def _get_module_names_from_pyi(file_path):
    """
    Reads a .pyi file and returns the names that the module exports.

    Args:
        file_path (str): Path to the .pyi file.

    Returns:
        set: The names of the top-level classes, functions and variables, and
        of the ``from x import y as y`` re-exports.
    """
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())

    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets)
        elif isinstance(node, ast.ImportFrom):
            names.update(
                alias.name for alias in node.names if alias.asname == alias.name
            )
    return names


def _check_stub_class(stub_attrs, cls, exclusions):
    """
    Checks that a class stub declares every public attribute of ``cls``, and
    nothing that ``cls`` lacks, except for the names in ``exclusions``.
    """
    runtime_attrs = set(dir(cls))
    stale_exclusions = exclusions - (stub_attrs ^ runtime_attrs)
    assert not stale_exclusions, f"Stale exclusions: {sorted(stale_exclusions)}"
    missing = {
        name
        for name in runtime_attrs - stub_attrs - exclusions
        if not name.startswith("_")
    }
    assert not missing, (
        f"Public attributes of {cls.__name__} missing from its stub: {sorted(missing)}"
    )
    extra = stub_attrs - runtime_attrs - exclusions
    assert not extra, (
        f"Stub attributes that {cls.__name__} lacks at runtime: {sorted(extra)}"
    )


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


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
def test_init_stub_exports():
    # Type checkers read tensordict/__init__.pyi instead of __init__.py.
    with open(_TENSORDICT_DIR / "__init__.pyi", "r") as f:
        tree = ast.parse(f.read())

    stub_all = None
    stub_names = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = importlib.import_module(node.module)
            module_path = _TENSORDICT_DIR.parent / node.module.replace(".", "/")
            module_stub = next(
                (
                    path
                    for path in (
                        module_path.with_suffix(".pyi"),
                        module_path / "__init__.pyi",
                    )
                    if path.exists()
                ),
                None,
            )
            module_stub_names = (
                _get_module_names_from_pyi(module_stub)
                if module_stub is not None
                else None
            )
            for alias in node.names:
                assert hasattr(module, alias.name), (node.module, alias.name)
                if module_stub_names is not None:
                    assert alias.name in module_stub_names, (
                        module_stub.name,
                        alias.name,
                    )
                stub_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.AnnAssign):
            stub_names.add(node.target.id)
        elif isinstance(node, ast.Assign) and node.targets[0].id == "__all__":
            stub_all = ast.literal_eval(node.value)

    assert sorted(stub_all) == sorted(tensordict.__all__)
    unbound = set(stub_all) - stub_names
    assert not unbound, f"__init__.pyi exports undefined names: {sorted(unbound)}"


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
def test_tensorcollection_stub_methods():
    # TensorCollection is an empty base class at runtime; its stub declares
    # the interface of TensorDictBase.
    stub_attrs = _get_class_attrs_from_pyi(
        str(_TENSORDICT_DIR / "_tensorcollection.pyi"), "TensorCollection"
    )
    _check_stub_class(stub_attrs, TensorDictBase, exclusions=set())


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
def test_tensorclass_stub_methods():
    tensorclass_methods = _get_class_attrs_from_pyi(
        str(_TENSORDICT_DIR / "tensorclass.pyi"), "TensorClass"
    )

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


# Names on which the TensorClass stub and runtime tensorclasses differ on purpose.
_TENSORCLASS_STUB_EXCLUSIONS = {
    # Forwards to LazyStackedTensorDict.extend, so it works only when the
    # tensorclass wraps a lazy stack; TensorDictBase has no extend.
    "extend",
    # Iteration goes through __getitem__ at runtime; the stub declares
    # __iter__ so that type checkers know what a loop yields.
    "__iter__",
}


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.parametrize("form", ["decorator", "subclass"])
def test_tensorclass_instance_methods(form):
    exclusions = set(_TENSORCLASS_STUB_EXCLUSIONS)
    if form == "decorator":

        @tensorclass
        class X:
            x: torch.Tensor

        # TensorClass["nocast"] and the like exist on TensorClass only.
        exclusions.add("__class_getitem__")
    else:

        class X(TensorClass):
            x: torch.Tensor

    stub_attrs = _get_class_attrs_from_pyi(
        str(_TENSORDICT_DIR / "tensorclass.pyi"), "TensorClass"
    )
    _check_stub_class(stub_attrs, X, exclusions)


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.parametrize(
    "path,class_name",
    [
        ("_tensorcollection.pyi", "TensorCollection"),
        ("tensorclass.pyi", "TensorClass"),
        ("_base/device.py", "_DeviceOps"),
    ],
)
def test_to_overloads_accept_str_device(path, class_name):
    # Type checkers must accept td.to("cpu") and td.to(device="cuda").
    with open(_TENSORDICT_DIR / path, "r") as f:
        tree = ast.parse(f.read())
    (class_node,) = (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    device_annotations = [
        arg.annotation
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "to"
        for arg in node.args.args + node.args.kwonlyargs
        if arg.arg == "device"
    ]
    assert device_annotations
    for annotation in device_annotations:
        device_type = eval(ast.unparse(annotation), vars(tensordict.utils))
        assert str in get_args(device_type), ast.unparse(annotation)


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
                assert not shared_elements, (
                    f"Lists {lst1} and {lst2} share elements: {shared_elements}"
                )


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


@tensorclass(frozen=True)
class MyDataFrozen:
    X: torch.Tensor
    z: str


@tensorclass(tensor_only=True)
class MyDataTensorOnly:
    X: torch.Tensor


class TestTensorClass:
    @pytest.mark.parametrize("tensor_only", [False, True])
    @pytest.mark.parametrize("frozen", [False, True])
    @pytest.mark.parametrize("device", [None, "cpu", "cpu:0", "meta"])
    def test_tensor_schema_initialization(self, tensor_only, frozen, device):
        @tensorclass(tensor_only=tensor_only, frozen=frozen)
        class Data:
            x: torch.Tensor
            y: torch.Tensor

        x = torch.randn(3, requires_grad=True)
        y = x.square()
        data = Data(x, y=y, batch_size=[3], device=device)
        assert data.is_locked == frozen
        assert data.device == (torch.device(device) if device is not None else None)
        if device != "meta":
            if device == "cpu:0":
                assert data.x is not x and data.y is not y
            else:
                assert data.x is x and data.y is y
            (data.x + data.y).sum().backward()
            torch.testing.assert_close(x.grad, 1 + 2 * x.detach())
        else:
            assert data.x.device.type == data.y.device.type == "meta"
            assert x.device.type == "cpu"
        with pytest.raises(RuntimeError, match="batch dimension mismatch"):
            Data(x=x, y=torch.zeros(2), batch_size=[3], device=device)
        with pytest.raises(TypeError, match="torch.Size"):
            Data(x=x, y=y, batch_size="bad")
        with pytest.raises(RuntimeError, match="batch dimension mismatch"):
            Data(x=x, y=y, batch_size=[2], device="cuda")
        with pytest.raises(TypeError, match="missing.*y"):
            Data(x=x)
        with pytest.raises(ValueError, match="already set"):
            Data(x, x=x, y=y)
        if not tensor_only:
            with pytest.raises(AttributeError, match="expected attributes"):
                Data(x=x, y=y, extra=x)
        # None values must still enter non-tensor storage.
        assert Data(x=x, y=None, batch_size=[3]).y is None

    def test_recursive_properties_common_ops(self):
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

        def check(result):
            _check_recursive_properties(result)
            return result

        check(data)
        check(data._tensordict)
        check(data.clone())
        check(data[:2])
        check(data.reshape(-1))
        check(data.apply(lambda x: x))
        check(data.unsqueeze(0))
        check(torch.stack([data, data.clone()], 0))
        check(LazyStackedTensorDict.lazy_stack([data, data.clone()], 0))

        updated = data.clone()
        updated.X = torch.zeros_like(updated.X)
        check(updated)

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

    @pytest.mark.parametrize("defaults_to_none", [True, False])
    def test_get_default_set_get_defaults_to_none(self, defaults_to_none):
        @tensorclass
        class Data:
            a: torch.Tensor

        data = Data(a=torch.zeros(3), batch_size=[3])
        set_back = _get_defaults_to_none()
        try:
            with pytest.warns(
                DeprecationWarning, match="set_get_defaults_to_none.*0.17"
            ):
                set_get_defaults_to_none(defaults_to_none)
            if defaults_to_none:
                assert data.get("b") is None
                assert data.get_at("b", 0) is None
            else:
                with pytest.raises(AttributeError):
                    data.get("b")
                with pytest.raises(AttributeError):
                    data.get_at("b", 0)
        finally:
            _set_get_defaults_to_none(set_back)

    def test_backward(self):
        @tensorclass
        class Losses:
            actor: torch.Tensor
            critic: torch.Tensor

        x = torch.randn(3, requires_grad=True)
        losses = Losses(actor=x.sum(), critic=x.pow(2).sum(), batch_size=[])
        losses.backward()
        assert torch.allclose(x.grad, 1 + 2 * x.detach())

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
        with _set_capture_non_tensor_stack(True):
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

        tc_cat = tensordict.cat([data1, data2], 0)
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

    def test_classvar(self):
        @tensorclass
        class MyData:
            X: torch.Tensor
            y: ClassVar[float] = 0.5

        # Test with keyword arguments
        data = MyData(X=torch.ones(3, 4, 5), batch_size=[3, 4])
        assert data.y == 0.5
        assert MyData.y == 0.5
        assert "y" not in data.__expected_keys__

        # Test with positional arguments
        data_pos = MyData(torch.ones(3, 4, 5), batch_size=[3, 4])
        assert data_pos.y == 0.5
        assert data_pos.X.shape == (3, 4, 5)
        assert "y" not in data_pos.__expected_keys__

        # Test with multiple fields and ClassVar in the middle
        @tensorclass
        class MyDataComplex:
            a: torch.Tensor
            b: ClassVar[str] = "constant"
            c: torch.Tensor
            d: ClassVar[int] = 42

        # Test with keyword arguments
        data_complex = MyDataComplex(
            a=torch.ones(2, 3), c=torch.zeros(2, 3), batch_size=[2, 3]
        )
        assert data_complex.b == "constant"
        assert data_complex.d == 42
        assert "b" not in data_complex.__expected_keys__
        assert "d" not in data_complex.__expected_keys__
        assert "a" in data_complex.__expected_keys__
        assert "c" in data_complex.__expected_keys__

        # Test with positional arguments
        data_complex_pos = MyDataComplex(
            torch.ones(2, 3), torch.zeros(2, 3), batch_size=[2, 3]
        )
        assert data_complex_pos.b == "constant"
        assert data_complex_pos.d == 42
        assert data_complex_pos.a.shape == (2, 3)
        assert data_complex_pos.c.shape == (2, 3)

        # Test that passing too many positional arguments (including ClassVar fields) raises an error
        # This should behave the same as regular dataclass
        with pytest.raises(
            TypeError, match="takes 3 positional arguments but 5 were given"
        ):
            # This should fail because we're passing 4 args (a, b, c, d) but only 2 are actual fields
            MyDataComplex(
                torch.ones(2, 3), "constant", torch.zeros(2, 3), 42, batch_size=[2, 3]
            )

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

    def test_defaultfactory_given_value(self):
        calls = []

        def factory():
            calls.append(None)
            return torch.ones(3)

        @tensorclass
        class MyData:
            X: torch.Tensor = dataclasses.field(default_factory=factory)

        assert (MyData(X=torch.zeros(3)).X == 0).all()
        assert MyData(X=None).X is None
        assert not calls
        assert (MyData().X == 1).all()
        assert len(calls) == 1

    @pytest.mark.parametrize("pre_dataclass", [False, True])
    def test_defaultfactory_init_false_setattr(self, pre_dataclass):
        # With a custom __setattr__, the dataclass __init__ sets the fields. It
        # calls the default factories of init=False fields itself.
        class MyData:
            X: torch.Tensor = dataclasses.field(
                init=False, default_factory=lambda: torch.ones(3)
            )

            def __setattr__(self, key, value):
                super().__setattr__(key, value)

        if pre_dataclass:
            MyData = dataclasses.dataclass(MyData)
        MyData = tensorclass(MyData)
        torch.testing.assert_close(MyData().X, torch.ones(3))

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

    @pytest.mark.parametrize(
        "conversion", ["from_dataclass", "inplace", "from_instance", "decorator"]
    )
    def test_from_dataclass_fields(self, conversion):
        # The tensorclass keeps the field() declarations of the dataclass.
        @dataclasses.dataclass
        class MyFieldsDataClass:
            a: torch.Tensor = dataclasses.field(
                default_factory=lambda: torch.ones(3), repr=False, metadata={"c": 0}
            )
            b: torch.Tensor = dataclasses.field(default=None, kw_only=True)

        if conversion == "from_dataclass":
            cls = from_dataclass(MyFieldsDataClass)
        elif conversion == "inplace":
            cls = from_dataclass(MyFieldsDataClass, inplace=True)
        elif conversion == "from_instance":
            cls = type(from_dataclass(MyFieldsDataClass()))
        else:
            cls = tensorclass(MyFieldsDataClass)

        data0, data1 = cls(), cls()
        torch.testing.assert_close(data0.a, torch.ones(3))
        assert data0.a is not data1.a
        assert data0.b is None
        a, b = dataclasses.fields(cls)
        assert not a.repr
        assert a.metadata == {"c": 0}
        assert b.kw_only

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
        with _set_list_to_stack(list_to_stack):

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

    def test_pickle_copy_frozen(self):
        data = MyDataFrozen(
            X=torch.ones(3, 4, 5), z="test_tensorclass", batch_size=[3, 4]
        )
        for data2 in (pickle.loads(pickle.dumps(data)), copy.copy(data)):
            assert isinstance(data2, MyDataFrozen)
            assert (data2.X == data.X).all()
            assert data2.z == data.z
            assert data2.batch_size == data.batch_size

    def test_pickle_tensor_only_while_compiling(self):
        # torch.compile sets this process-wide flag for as long as it compiles,
        # so other threads, such as a DataLoader's pin-memory thread, see it.
        data = MyDataTensorOnly(X=torch.ones(3, 4), batch_size=[3])
        payload = pickle.dumps(data)
        with torch.compiler._compile_session_context():
            data2 = pickle.loads(payload)
        assert isinstance(data2, MyDataTensorOnly)
        assert (data2.X == data.X).all()
        assert data2.batch_size == data.batch_size

    @pytest.mark.parametrize("frozen", [False, True])
    def test_copy_deepcopy(self, frozen):
        if frozen:
            data = MyDataFrozen(X=torch.ones(3, 4, 5), z="z", batch_size=[3, 4])
        else:
            data = MyData(
                X=torch.ones(3, 4, 5), y=torch.zeros(3, 4), z="z", batch_size=[3, 4]
            )
        shallow = copy.copy(data)
        assert type(shallow) is type(data)
        assert shallow._tensordict is not data._tensordict
        assert shallow.X.data_ptr() == data.X.data_ptr()
        deep = copy.deepcopy(data)
        assert type(deep) is type(data)
        assert deep is not data
        assert deep._tensordict is not data._tensordict
        assert deep.X.data_ptr() != data.X.data_ptr()
        assert (deep.X == data.X).all()
        assert deep.z == data.z
        if not frozen:
            shallow.z = "other"
            shallow.batch_size = [3]
            assert data.z == "z"
            assert data.batch_size == torch.Size([3, 4])

    def test_copy_deepcopy_user_defined(self):
        @tensorclass
        class MyDataCopy:
            X: torch.Tensor

            def __copy__(self):
                return "copy"

            def __deepcopy__(self, memo):
                return "deepcopy"

        @tensorclass
        class MyDataCopyChild(MyDataCopy):
            y: torch.Tensor = None

        for cls in (MyDataCopy, MyDataCopyChild):
            data = cls(X=torch.ones(3), batch_size=[3])
            assert copy.copy(data) == "copy"
            assert copy.deepcopy(data) == "deepcopy"

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

    def test_repeat_interleave_tensor(self):
        class MyDataNested(TensorClass):
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data_nest = MyDataNested(X=X, z=z, batch_size=batch_size)
        data = MyDataNested(X=X, y=data_nest, z=z, batch_size=batch_size)
        repeated = data.repeat_interleave(
            torch.tensor([2, 3, 4, 5], device=data.device), dim=1
        )
        assert repeated.shape == torch.Size((3, 14))
        assert (
            repeated.X
            == X.repeat_interleave(
                torch.tensor([2, 3, 4, 5], device=data.device), dim=1
            )
        ).all()

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

    def test_is_non_tensor_key(self):
        @tensorclass
        class Transition:
            obs: torch.Tensor
            label: str

        t = Transition(obs=torch.randn(4), label="cat", batch_size=[])

        # TensorClass: is_non_tensor(key) checks internal representation
        assert t.is_non_tensor("label")
        assert not t.is_non_tensor("obs")

        # Confirm the standalone is_non_tensor gives False on unwrapped value
        from tensordict import is_non_tensor

        assert not is_non_tensor(t.get("label"))

        # Plain TensorDict: consistent behaviour
        td = TensorDict({"obs": torch.randn(4)}, [])
        td.set_non_tensor("label", "cat")
        assert td.is_non_tensor("label")
        assert not td.is_non_tensor("obs")

        # Nested key support
        nested_td = TensorDict({"a": TensorDict({}, [])}, [])
        nested_td["a"].set_non_tensor("x", 42)
        nested_td["a", "y"] = torch.zeros(3)
        assert nested_td.is_non_tensor(("a", "x"))
        assert not nested_td.is_non_tensor(("a", "y"))

    def test_select_as_tensordict(self):
        @tensorclass
        class Transition:
            obs: torch.Tensor
            action: torch.Tensor
            label: str

        t = Transition(
            obs=torch.randn(4),
            action=torch.tensor([1.0]),
            label="cat",
            batch_size=[],
        )

        # Default: returns TensorClass with unselected fields as None
        result_tc = t.select("obs", "action")
        assert type(result_tc).__name__ == "Transition"
        assert result_tc.label is None

        # as_tensordict=True: returns plain TensorDict without unselected fields
        result_td = t.select("obs", "action", as_tensordict=True)
        assert type(result_td) is TensorDict
        assert set(result_td.keys()) == {"obs", "action"}
        assert "label" not in result_td.keys()
        assert (result_td["obs"] == t.obs).all()
        assert (result_td["action"] == t.action).all()

        # as_tensordict on a plain TensorDict is a no-op
        td = TensorDict({"a": torch.zeros(3), "b": torch.ones(3)}, [])
        td_sel = td.select("a", as_tensordict=True)
        assert type(td_sel) is TensorDict
        assert set(td_sel.keys()) == {"a"}

    @set_list_to_stack(True)
    def test_set_list_in_constructor(self):
        obj = MyTensorClass(
            a=["a string", "another string"],
            b=[torch.randn(3), torch.zeros(3)],
            c="smth completely different",
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
        with (
            _set_list_to_stack(list_to_stack),
            (
                pytest.raises(RuntimeError, match="batch dimension mismatch")
                if list_to_stack
                else contextlib.nullcontext()
            ),
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
        with (
            _set_list_to_stack(list_to_stack),
            (
                pytest.raises(RuntimeError, match="batch dimension mismatch")
                if list_to_stack
                else contextlib.nullcontext()
            ),
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

    def test_tensor_split(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        data_in = MyDataNested(
            X=torch.ones(3, 6, 5), z="test_tensorclass", batch_size=[3, 6]
        )
        data_out = MyDataNested(
            X=torch.ones(3, 6, 5), z="test_tensorclass", y=data_in, batch_size=[3, 6]
        )

        data_split = data_out.tensor_split((1, 4, 5), 1)
        assert len(data_split) == 4
        assert data_split[0].batch_size == torch.Size([3, 1])
        assert data_split[1].batch_size == torch.Size([3, 3])
        assert data_split[2].batch_size == torch.Size([3, 1])
        assert data_split[2].batch_size == torch.Size([3, 1])

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

    def test_update_kwargs(self):
        @tensorclass
        class TC:
            a: torch.Tensor
            b: torch.Tensor

        tc = TC(a=torch.zeros(3), b=torch.zeros(3), batch_size=[3])
        tc.update(a=torch.ones(3))
        assert (tc.a == 1).all()

        tc.update_(b=torch.ones(3) * 2)
        assert (tc.b == 2).all()

        # merge: positional + kwargs, kwargs win
        tc2 = TC(a=torch.zeros(3), b=torch.zeros(3), batch_size=[3])
        other = TC(a=torch.ones(3) * 5, b=torch.ones(3) * 7, batch_size=[3])
        tc2.update(other, b=torch.ones(3) * 9)
        assert (tc2.a == 5).all()
        assert (tc2.b == 9).all()

        # empty call is a no-op
        tc3 = TC(a=torch.zeros(3), b=torch.zeros(3), batch_size=[3])
        assert tc3.update() is tc3
        assert tc3.update_() is tc3

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
            with _set_capture_non_tensor_stack(True):
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

    def test_stack_metadata(self):
        class MyTensorClass(TensorClass):
            custom_data: MetaData

        my_tc = MyTensorClass(custom_data=MetaData("abc"))
        assert isinstance(my_tc._tensordict._tensordict.get("custom_data"), MetaData)
        stacked_tc = torch.stack([my_tc, my_tc], dim=0)
        assert isinstance(
            stacked_tc._tensordict._tensordict.get("custom_data"), MetaData
        )
        assert stacked_tc.custom_data == "abc"
        stacked_tc = lazy_stack([my_tc, my_tc], dim=0)
        assert isinstance(stacked_tc._tensordict.get("custom_data"), MetaData)
        assert stacked_tc.custom_data == "abc"

    def test_stack_metadata_typed(self):
        class MyTensorClass(TensorClass):
            custom_data: MetaData[str]

        my_tc = MyTensorClass(custom_data=MetaData[str]("abc"))
        assert isinstance(my_tc._tensordict._tensordict.get("custom_data"), MetaData)
        stacked_tc = torch.stack([my_tc, my_tc], dim=0)
        assert isinstance(
            stacked_tc._tensordict._tensordict.get("custom_data"), MetaData[str]
        )
        assert stacked_tc.custom_data == "abc"
        stacked_tc = lazy_stack([my_tc, my_tc], dim=0)
        assert isinstance(stacked_tc._tensordict.get("custom_data"), MetaData[str])
        assert stacked_tc.custom_data == "abc"

    def test_td_classmethods_return_subclass(self):
        # TensorDict classmethods exposed on TensorClass subclasses must
        # accept the classmethod calling convention and return the subclass
        # (they are re-wrapped per subclass; the instance-method fallbacks
        # inherited from the TensorClass base would bind the first argument
        # to self and crash).
        class MC(TensorClass):
            foo: Any
            bar: Any

        class MCChild(MC):
            pass

        out = MC.from_list([{"foo": 1, "bar": 2}, {"foo": 3, "bar": 4}])
        assert type(out) is MC
        assert type(MCChild.from_list([{"foo": 1, "bar": 2}])) is MCChild
        assert type(MC.fromkeys(["foo", "bar"], 0)) is MC
        mc = MC(foo=torch.zeros(3), bar=torch.ones(3), batch_size=[3])
        for name in ("stack", "cat", "lazy_stack", "maybe_dense_stack"):
            assert type(getattr(MC, name)([mc, mc])) is MC, name

    def test_to_lazystack(self):
        class MyTensorClass(TensorClass):
            foo: Tensor
            bar: Tensor

        tc = MyTensorClass(
            foo=torch.zeros((1, 2)), bar=torch.ones((1, 2)), batch_size=(1, 2)
        )
        tc2 = tc.to_lazystack(1)
        assert isinstance(tc2, MyTensorClass)
        assert isinstance(tc2._tensordict, LazyStackedTensorDict)

    @pytest.mark.skipif(not _has_streaming, reason="streaming is not installed")
    def test_to_mds(self, tmpdir):
        td = LazyStackedTensorDict(
            TensorDict(a=0, b=1, c=torch.randn(2), d="a string"),
            TensorDict(a=1, b=1, c=torch.randn(3), d="another string"),
            TensorDict(a=2, b=1, c=torch.randn(3), d="yet another string"),
        )

        class MC(TensorClass):
            a: Any
            b: Any
            c: Any
            d: Any

        td = MC.from_tensordict(td)

        tmpdir = str(tmpdir)
        td.to_mds(out=tmpdir)

        # Create a dataloader
        from streaming import StreamingDataset

        # Load the dataset
        dataset = StreamingDataset(local=tmpdir, remote=None, batch_size=2)
        dl = torch.utils.data.DataLoader(  # noqa: TOR401
            dataset=dataset, batch_size=2, collate_fn=MC.from_list
        )
        batches = list(dl)
        batches = [_batch for batch in batches for _batch in batch.unbind(0)]
        test_td = TensorDict.lazy_stack(batches)
        assert isinstance(test_td, MC)
        assert isinstance(test_td._tensordict, LazyStackedTensorDict)
        assert_allclose_td(td, test_td)

    def test_stack_names(self):
        class MyTensorClass(TensorClass):
            foo: Tensor
            bar: Tensor

        tc = MyTensorClass(
            foo=torch.zeros((1, 2)),
            bar=torch.ones((1, 2)),
            names=["first", "second"],
            batch_size=(1, 2),
        )
        tc2 = torch.stack([tc, tc], dim=0)
        assert tc2.names == [None, "first", "second"]
        tc2 = torch.stack([tc, tc], dim=1)
        assert tc2.names == ["first", None, "second"]
        tc2 = torch.stack([tc, tc], dim=-1)
        assert tc2.names == ["first", "second", None]

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
        # Unexpected key in flat state_dict
        sd["a"] = None
        with pytest.raises(KeyError, match="Key 'a' wasn't expected in the state-dict"):
            tc.load_state_dict(sd)
        del sd["a"]
        # Unexpected key in _metadata non_tensor data
        sd._metadata[""]["_non_tensor"]["a"] = None
        with pytest.raises(KeyError, match="Key 'a' wasn't expected in the state-dict"):
            tc.load_state_dict(sd)
        del sd._metadata[""]["_non_tensor"]["a"]
        # Nested format: unexpected key in nested tensorclass state_dict
        sd_nested = tc.state_dict(flatten=False)
        sd_nested["y"]["a"] = None
        with pytest.raises(
            (KeyError, RuntimeError),
        ):
            tc.load_state_dict(sd_nested)

    def test_statedict_flat_roundtrip(self):
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
        assert set(sd.keys()) == {"x", "y.x"}
        assert hasattr(sd, "_metadata")
        assert "MyClass" in sd._metadata[""]["_type"]
        assert sd._metadata[""]["_non_tensor"]["z"] == z
        assert sd._metadata["y"]["_non_tensor"]["z"] == z

        tc2 = tc.clone()
        tc2.x = torch.zeros(3)
        tc2.y.x = torch.zeros(3)
        tc2.load_state_dict(sd)
        assert torch.allclose(tc2.x, tc.x)
        assert torch.allclose(tc2.y.x, tc.y.x)
        assert tc2.z == z
        assert tc2.y.z == z

    def test_statedict_nested_roundtrip(self):
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
        sd = tc.state_dict(flatten=False)
        assert set(sd.keys()) == {"x", "y"}
        assert isinstance(sd["y"], dict)

        tc2 = tc.clone()
        tc2.x = torch.zeros(3)
        tc2.y.x = torch.zeros(3)
        tc2.load_state_dict(sd)
        assert torch.allclose(tc2.x, tc.x)
        assert torch.allclose(tc2.y.x, tc.y.x)

    def test_statedict_legacy_compat(self):
        import collections

        @tensorclass
        class MyClass:
            x: torch.Tensor
            y: torch.Tensor

        tc = MyClass(x=torch.randn(3), y=torch.randn(3), batch_size=[])
        # Build a legacy-format state_dict matching the old tensorclass output
        legacy_sd = collections.OrderedDict()
        legacy_sd["_tensordict"] = collections.OrderedDict()
        legacy_sd["_tensordict"]["x"] = torch.tensor([1.0, 2.0, 3.0])
        legacy_sd["_tensordict"]["y"] = torch.tensor([4.0, 5.0, 6.0])
        legacy_sd["_tensordict"]["__batch_size"] = torch.Size([])
        legacy_sd["_tensordict"]["__device"] = None
        legacy_sd["_non_tensordict"] = {}
        tc.load_state_dict(legacy_sd)
        assert torch.allclose(tc.x, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(tc.y, torch.tensor([4.0, 5.0, 6.0]))

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
        # This is expected to fail since unsqueeze now returns lists when a non tensor data
        #  is used.
        # assert unsqueeze_tc.z == unsqueeze_tc.y.z == [z]

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

    def test_view_cm(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str

        X = torch.ones(3, 4, 5)
        z = "test_tensorclass"
        batch_size = [3, 4]
        data = MyDataNested(X=X, z=z, batch_size=batch_size)
        with data.view(-1) as viewed_td:
            assert isinstance(viewed_td, MyDataNested)
            viewed_td.X *= 0
        assert (data.X == 0).all()

        data = MyDataNested(X=None, z=z, batch_size=batch_size)
        with data.view(-1) as viewed_td:
            assert isinstance(viewed_td, MyDataNested)
            viewed_td.X = torch.zeros(12, 5)
        assert data.X is not None
        assert (data.X == 0).all()

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
        with _set_capture_non_tensor_stack(True):
            y = torch.stack([y0, y1])
        assert y.z.shape == torch.Size(())

    def test_autograd_grad(self):
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
        b = a + 1
        inputs = a.select("x")
        outputs = b.select("x")
        grads = torch.autograd.grad(outputs, inputs, torch.ones_like(outputs))
        assert (grads == 1).all()


class TestMemmap:
    def test_empty_tensor_roundtrip(self, tmp_path):
        @tensorclass
        class MyClass:
            vector: torch.Tensor
            matrix: torch.Tensor

        data = MyClass(
            vector=torch.empty(0, dtype=torch.int32),
            matrix=torch.empty(2, 0, dtype=torch.float64),
            batch_size=[],
        )
        data.memmap_(tmp_path)

        loaded = MyClass.load_memmap(tmp_path)

        assert loaded.vector.shape == (0,)
        assert loaded.vector.dtype is torch.int32
        assert loaded.matrix.shape == (2, 0)
        assert loaded.matrix.dtype is torch.float64

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

    @pytest.mark.parametrize("mode", [None, "r", "r+"])
    def test_load_mode(self, tmp_path, mode):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            td: TensorDict

        MyClass(
            x=torch.zeros(3),
            td=TensorDict(y=torch.zeros(3), batch_size=[3]),
            batch_size=[3],
        ).memmap(tmp_path)
        loaded = MyClass.load(tmp_path, mode=mode)
        loaded.x.add_(1)
        loaded.td["y"].add_(1)
        # "r" keeps in-place writes in memory
        expected = 0 if mode == "r" else 1
        on_disk = MyClass.load(tmp_path)
        assert (on_disk.x == expected).all()
        assert (on_disk.td["y"] == expected).all()

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
        data3 = MyOtherClass.load_memmap(tmpdir, allow_pickle=True)
        assert isinstance(data3, MyClass)

    def test_load_memmap_out(self, tmp_path):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            string: str
            other: Any = None

        MyClass(x=torch.ones(3), string="new", batch_size=[3]).memmap(tmp_path)
        out = MyClass(x=torch.zeros(3), string="old", other="stale", batch_size=[3])
        loaded = MyClass.load_memmap(tmp_path, out=out)
        assert loaded is out
        assert (out.x == 1).all()
        assert out.string == "new"
        assert out.other is None

    @pytest.mark.parametrize("memmap", [False, True])
    def test_load_memmap_(self, tmp_path, memmap):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            string: str
            other: Any = None

        MyClass(x=torch.ones(3), string="new", other="set", batch_size=[3]).memmap(
            tmp_path / "src"
        )
        dest = MyClass(x=torch.zeros(3), string="old", batch_size=[3])
        if memmap:
            dest.memmap_(tmp_path / "dest")
        assert dest.load_memmap_(tmp_path / "src") is dest
        assert (dest.x == 1).all()
        assert dest.string == "new"
        assert dest.other == "set"
        assert dest.is_memmap() is memmap
        if memmap:
            assert dest.saved_path == tmp_path / "src" / "_tensordict"
            MyClass(x=torch.full((3,), 2.0), string="newer", batch_size=[3]).memmap(
                tmp_path / "src"
            )
            dest.memmap_refresh_()
            assert (dest.x == 2).all()
            assert dest.string == "newer"

    def test_load_memmap_redefined_class(self, tmp_path, monkeypatch):
        # TensorDict.load_memmap looks the saved class up by name, so the
        # lookup can return another class with the same name, e.g. one that is
        # redefined in a notebook. An instance of the redefined class passed as
        # out is still loaded in place.
        def make_class():
            @tensorclass
            class MyClass:
                x: torch.Tensor
                string: str

            return MyClass

        saved_cls, dest_cls = make_class(), make_class()
        saved_cls(x=torch.ones(3), string="new", batch_size=[3]).memmap(tmp_path)
        # make the lookup by name return saved_cls
        monkeypatch.setattr(
            tensordict.base,
            "_ACCEPTED_CLASSES",
            (saved_cls,) + tensordict.base._ACCEPTED_CLASSES,
        )
        dest = dest_cls(x=torch.zeros(3), string="old", batch_size=[3])
        assert TensorDict.load_memmap(tmp_path, out=dest) is dest
        assert (dest.x == 1).all()
        assert dest.string == "new"

    def test_load_memmap_other_class(self, tmp_path):
        @tensorclass
        class MyClass:
            x: torch.Tensor

        @tensorclass
        class OtherClass:
            x: torch.Tensor

        MyClass(x=torch.ones(3), batch_size=[3]).memmap(tmp_path)
        dest = OtherClass(x=torch.zeros(3), batch_size=[3])
        with pytest.raises(ValueError, match="Cannot load a saved MyClass in place"):
            dest.load_memmap_(tmp_path)
        assert (dest.x == 0).all()

    @pytest.mark.parametrize("form", ["decorator", "subclass"])
    def test_load_memmap_class_saved_from_main(self, tmp_path, form):
        # A class saved from __main__ (a script or a notebook) is recorded as
        # __main__.<qualname>, which another program cannot find by name. The
        # class load_memmap is called on is used if its qualname matches.
        if form == "decorator":

            @tensorclass
            class MyClass:
                x: torch.Tensor
                string: str

            @tensorclass
            class OtherClass:
                x: torch.Tensor
                string: str

        else:

            class MyClass(TensorClass):
                x: torch.Tensor
                string: str

            class OtherClass(TensorClass):
                x: torch.Tensor
                string: str

        MyClass(x=torch.ones(3), string="saved", batch_size=[3]).memmap(tmp_path)
        saved_name = f"__main__.{MyClass.__qualname__}"
        meta_path = tmp_path / "meta.json"
        metadata = json.loads(meta_path.read_text())
        metadata["_type"] = f"<class '{saved_name}'>"
        meta_path.write_text(json.dumps(metadata))

        loaded = MyClass.load_memmap(tmp_path)
        assert type(loaded) is MyClass
        assert (loaded.x == 1).all()
        assert loaded.string == "saved"
        assert type(MyClass.load(tmp_path)) is MyClass
        dest = MyClass(x=torch.zeros(3), string="old", batch_size=[3])
        assert dest.load_memmap_(tmp_path) is dest
        assert (dest.x == 1).all()
        assert dest.string == "saved"

        # TensorDict.load_memmap does not guess the class, and a class with
        # another name is not used.
        for other_cls in (TensorDict, OtherClass):
            msg = (
                f"Could not find the class {saved_name} saved in {tmp_path}. "
                "Import the module that defines it, or call MyClass.load_memmap() "
                f"instead of {other_cls.__qualname__}.load_memmap()."
            )
            with pytest.raises(RuntimeError, match=re.escape(msg)):
                other_cls.load_memmap(tmp_path)

        # The qualname must match: a class nested in another class of
        # __main__ is not MyClass.
        metadata["_type"] = f"<class '__main__.Outer.{MyClass.__qualname__}'>"
        meta_path.write_text(json.dumps(metadata))
        with pytest.raises(
            RuntimeError, match=re.escape("Could not find the class __main__.Outer.")
        ):
            MyClass.load_memmap(tmp_path)

    def test_load_memmap_tensordict_as_subclass(self, tmp_path):
        # A TensorClass subclass wraps a loaded plain TensorDict in the class.
        class MyClass(TensorClass):
            x: torch.Tensor

        TensorDict(x=torch.ones(3), batch_size=[3]).memmap(tmp_path)
        for loaded in (MyClass.load_memmap(tmp_path), MyClass.load(tmp_path)):
            assert type(loaded) is MyClass
            assert (loaded.x == 1).all()

    def test_memmap_overwrite_removes_stale_pickle(self, tmp_path):
        @tensorclass
        class MyClass:
            value: Any

        MyClass(value=1 + 2j, batch_size=[]).memmap(tmp_path)
        pickle_path = tmp_path / "_tensordict" / "value" / "other.pickle"
        assert pickle_path.exists()

        MyClass(value="json", batch_size=[]).memmap(tmp_path)
        assert not pickle_path.exists()
        assert MyClass.load_memmap(tmp_path, allow_pickle=False).value == "json"

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

    def test_memmap_nested_tensorclass_roundtrip(self, tmpdir):
        # A tensorclass nested (at depth >= 2) inside a plain TensorDict must
        # come back from a memmap_ / load_memmap roundtrip with its class
        # identity intact -- for both the decorator and the TensorClass
        # subclass flavors.
        @tensorclass
        class MyDecorated:
            a: torch.Tensor
            b: torch.Tensor

        class MySubclassed(TensorClass["nocast"]):
            a: torch.Tensor
            meta: str | None = None

        decorated = MyDecorated(a=torch.zeros(5), b=torch.ones(5), batch_size=[5])
        subclassed = MySubclassed(
            a=torch.arange(5, dtype=torch.float), meta="hello", batch_size=[5]
        )
        td = TensorDict(
            {
                "obs": {
                    "decorated": decorated,
                    "deep": {"subclassed": subclassed},
                },
                "x": torch.zeros(5),
            },
            batch_size=[5],
        )
        td.memmap_(tmpdir)
        loaded = TensorDict.load_memmap(tmpdir)
        assert isinstance(loaded["obs", "decorated"], MyDecorated)
        assert isinstance(loaded["obs", "deep", "subclassed"], MySubclassed)
        assert (loaded["obs", "decorated"].a == 0).all()
        assert (loaded["obs", "decorated"].b == 1).all()
        assert (
            loaded["obs", "deep", "subclassed"].a == torch.arange(5, dtype=torch.float)
        ).all()
        assert loaded["obs", "deep", "subclassed"].meta == "hello"

    def test_memmap_tensorclass_subclass_roundtrip(self, tmpdir):
        # A top-level TensorClass subclass must also keep its identity, both
        # through the generic TensorDict.load_memmap entry point and through
        # cls.load_memmap (TensorClass subclasses used to delegate _memmap_ /
        # _load_memmap to TensorDict, dropping the class information).
        class MySubclassed(TensorClass["nocast"]):
            a: torch.Tensor
            meta: str | None = None

        data = MySubclassed(a=torch.zeros(3), meta="hello", batch_size=[3])
        data.memmap_(tmpdir)
        loaded = TensorDict.load_memmap(tmpdir)
        assert isinstance(loaded, MySubclassed)
        assert (loaded.a == 0).all()
        assert loaded.meta == "hello"
        loaded = MySubclassed.load_memmap(tmpdir)
        assert isinstance(loaded, MySubclassed)
        assert (loaded.a == 0).all()
        assert loaded.meta == "hello"

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

        with _set_capture_non_tensor_stack(True):
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
        elif is_npu_available():
            device = torch.device("npu:0")
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


class NestedPose(TensorClass):
    q: torch.Tensor


class NestedObs(TensorClass):
    a: NestedPose
    x: torch.Tensor


class NestedObsOptional(TensorClass):
    a: Optional[NestedPose] = None
    x: torch.Tensor | None = None


class NestedObsAutocast(TensorClass["autocast"]):
    a: NestedPose
    x: torch.Tensor


class NestedObsDeep(TensorClass):
    obs: NestedObs
    y: torch.Tensor


class NestedPoseDerived(NestedPose):
    extra: torch.Tensor


class NestedAnyTensorClass(TensorClass):
    a: TensorClass


class NestedLabeled(TensorClass):
    x: torch.Tensor
    label: NonTensorData


_NestedT = TypeVar("_NestedT")


class NestedGenericPose(TensorClass, Generic[_NestedT]):
    q: torch.Tensor


class NestedObsGeneric(TensorClass):
    a: NestedGenericPose[int]


def _nested_obs_td():
    return TensorDict(
        a=TensorDict(q=torch.randn(5, 4), batch_size=[5]),
        x=torch.zeros(5),
        batch_size=[5],
    )


class TestFromTensorDictNested:
    """from_tensordict builds the fields annotated with a tensorclass (gh-1929)."""

    @pytest.mark.parametrize("cls", [NestedObs, NestedObsOptional, NestedObsAutocast])
    def test_from_tensordict_nested(self, cls):
        td = _nested_obs_td()
        obs = cls.from_tensordict(td)
        assert type(obs.a) is NestedPose
        assert type(obs.a.q) is torch.Tensor
        assert (obs.a.q == td["a", "q"]).all()

    def test_from_tensordict_nested_round_trip(self):
        obs = NestedObs(
            a=NestedPose(q=torch.randn(5, 4), batch_size=[5]),
            x=torch.zeros(5),
            batch_size=[5],
        )
        deep = NestedObsDeep(obs=obs, y=torch.ones(5), batch_size=[5])
        back = NestedObsDeep.from_tensordict(deep.to_tensordict())
        assert type(back.obs) is NestedObs
        assert type(back.obs.a) is NestedPose
        assert (back == deep).all()

    def test_from_tensordict_nested_input_unchanged(self):
        td = _nested_obs_td()
        obs = NestedObs.from_tensordict(td)
        assert type(obs.a) is NestedPose
        # The input keeps its TensorDict entry and shares the leaves.
        assert type(td["a"]) is TensorDict
        assert td["a"]["q"] is obs.a.q
        assert td["x"] is obs.x
        obs.a.q.add_(1)
        assert (td["a", "q"] == obs.a.q).all()

    def test_from_tensordict_nested_locked(self):
        td = _nested_obs_td().lock_()
        obs = NestedObs.from_tensordict(td)
        assert type(obs.a) is NestedPose
        assert obs.is_locked

    def test_from_tensordict_no_nested_entry_aliases_input(self):
        td = TensorDict(x=torch.zeros(5), batch_size=[5])
        obs = NestedObsOptional.from_tensordict(td)
        assert obs._tensordict is td
        assert obs.a is None

    def test_from_tensordict_nested_lazy_stack(self):
        # Only TensorDict inputs are rebuilt: other backends are wrapped as
        # they are.
        td = lazy_stack([_nested_obs_td(), _nested_obs_td()])
        obs = NestedObs.from_tensordict(td)
        assert obs._tensordict is td
        assert isinstance(obs.a, LazyStackedTensorDict)

    def test_from_dict_nested(self):
        obs = NestedObs.from_dict(_nested_obs_td().to_dict(), batch_size=[5])
        assert type(obs.a) is NestedPose

    def test_from_tensordict_nested_generic(self):
        obs = NestedObsGeneric.from_tensordict(_nested_obs_td().exclude("x"))
        assert type(obs.a) is NestedGenericPose

    def test_from_tensordict_deep_copies_once(self, monkeypatch):
        deep = NestedObsDeep(
            obs=NestedObs(
                a=NestedPose(q=torch.randn(5, 4), batch_size=[5]),
                x=torch.zeros(5),
                batch_size=[5],
            ),
            y=torch.ones(5),
            batch_size=[5],
        )
        copies = []
        copy = TensorDict.copy

        def counting_copy(self):
            copies.append(self)
            return copy(self)

        monkeypatch.setattr(TensorDict, "copy", counting_copy)
        back = NestedObsDeep.from_tensordict(deep.to_tensordict())
        assert type(back.obs.a) is NestedPose
        assert len(copies) == 1

    def test_from_tensordict_undeclared_keys(self):
        # A field annotated with NestedPose holds a subclass with more fields:
        # the entry stays a TensorDict.
        obs = NestedObs(
            a=NestedPoseDerived(
                q=torch.zeros(5, 4), extra=torch.ones(5), batch_size=[5]
            ),
            x=torch.zeros(5),
            batch_size=[5],
        )
        back = NestedObs.from_tensordict(obs.to_tensordict())
        assert type(back.a) is TensorDict
        assert (back.a["extra"] == 1).all()
        auto = NestedObsAutocast(a=back.a, x=torch.zeros(5), batch_size=[5])
        assert type(auto.a) is TensorDict

    def test_from_tensordict_base_tensorclass_annotation(self):
        # TensorClass declares no fields: the entry stays a TensorDict.
        td = _nested_obs_td().exclude("x")
        obs = NestedAnyTensorClass.from_tensordict(td)
        assert obs._tensordict is td
        assert type(obs.a) is TensorDict

    def test_from_tensordict_non_tensor_stack(self):
        stacked = torch.stack(
            [
                NestedLabeled(x=torch.zeros(()), label="a"),
                NestedLabeled(x=torch.zeros(()), label="b"),
            ]
        )
        back = NestedLabeled.from_tensordict(stacked.to_tensordict())
        assert back.label == ["a", "b"]


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
        assert isinstance(obj.tc, self.ClsAutoCast), (type(obj.tc), type(obj))

        assert isinstance(obj.tc_global, AutoCast), (type(obj.tc), type(obj))

        assert isinstance(obj.tc.tensor, torch.Tensor)
        assert isinstance(obj.tc.non_tensor, str)
        assert isinstance(obj.tc.td, TensorDict)
        assert obj.tc.tc is None

    def test_autocast_or(self):
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

    def test_autocast_tensordict_for_tensorclass_field(self):
        q = torch.randn(5, 4)
        obs = NestedObsAutocast(
            a=TensorDict(q=q, batch_size=[5]), x=torch.zeros(5), batch_size=[5]
        )
        assert type(obs.a) is NestedPose
        assert obs.a.q is q
        obs.a = TensorDict(q=q + 1, batch_size=[5])
        assert type(obs.a) is NestedPose
        assert (obs.a.q == q + 1).all()

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

    def test_shadow_non_tensor_values(self):
        # Non-tensor values are wrapped in NonTensorData, which must take the
        # batch size and device of the TensorDict, not the shadowed fields.
        @tensorclass(shadow=True, nocast=True)
        class MyClass:
            x: torch.Tensor
            batch_size: Any
            device: Any
            name: str

        c = MyClass(
            torch.zeros(10, 4), batch_size=4, device="not-a-device", name="graph"
        )
        assert c.batch_size == 4
        assert c.device == "not-a-device"
        assert c.name == "graph"
        assert c._tensordict.batch_size == torch.Size([])
        assert c._tensordict.device is None
        c.batch_size = 5
        c.device = "other"
        assert c.batch_size == 5
        assert c.device == "other"

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

    # A TensorClass subclass inherits TensorClass methods and properties, which
    # dataclass() used to take as the defaults of fields with the same name.
    @pytest.mark.parametrize("subclass", [False, True])
    def test_no_shadow_method_name(self, subclass):
        with pytest.raises(AttributeError, match="Attribute name sum can't"):
            if subclass:

                class MyClass(TensorClass):
                    sum: torch.Tensor
                    other: torch.Tensor

            else:

                @tensorclass
                class MyClass:  # noqa: F811
                    sum: torch.Tensor
                    other: torch.Tensor

    # Every member of a tensorclass is a reserved field name, except "data" and
    # "fields", which a field replaces, and the "_is_non_tensor" field of
    # NonTensorData.
    @pytest.mark.parametrize(
        "make_cls",
        [
            lambda ann: tensorclass(type("MyClass", (), {"__annotations__": ann})),
            lambda ann: tensorclass(tensor_only=True)(
                type("MyClass", (), {"__annotations__": ann})
            ),
            lambda ann: type("MyClass", (TensorClass,), {"__annotations__": ann}),
            lambda ann: type(
                "MyClass", (TensorClass["frozen"],), {"__annotations__": ann}
            ),
            lambda ann: type(
                "MyClass", (TensorClass["tensor_only"],), {"__annotations__": ann}
            ),
        ],
        ids=["decorator", "decorator-tensor_only", "subclass", "frozen", "tensor_only"],
    )
    def test_reserved_field_names_cover_members(self, make_cls):
        from tensordict.tensorclass import _is_reserved_field_name

        MyClass = make_cls({"x": torch.Tensor})
        c = MyClass(x=torch.zeros(3), batch_size=[3])
        names = {
            name
            for name in set(dir(MyClass)).union(vars(c))
            if not (name.startswith("__") and name.endswith("__"))
        }
        names -= {"x", "data", "fields", "_is_non_tensor"}
        assert {name for name in names if not _is_reserved_field_name(name)} == set()

    @pytest.mark.parametrize(
        "name",
        [
            "from_tensordict",
            "extend",
            "_tensordict",
            "_non_tensordict",
            "_type_hints",
            "_set_dict_warn_msg",
            "_get_non_tensor",
            "_send",
            "_transform_keys",
        ],
    )
    @pytest.mark.parametrize("subclass", [False, True])
    def test_no_shadow_tensorclass_member_name(self, name, subclass):
        # The first six exist on tensorclasses but not on TensorDict. The others
        # are private TensorDict methods that TensorDict code calls on nested
        # tensor collections, which a tensorclass forwards to its TensorDict.
        annotations = {name: torch.Tensor, "other": torch.Tensor}
        with pytest.raises(
            AttributeError,
            match=rf"Attribute name {name} can't be used .* pass shadow=True",
        ):
            if subclass:
                type("MyClass", (TensorClass,), {"__annotations__": annotations})
            else:
                tensorclass(type("MyClass", (), {"__annotations__": annotations}))

    @pytest.mark.parametrize("tensor_only", [False, True])
    def test_private_tensordict_name_as_field(self, tensor_only):
        # "_cache" is a private TensorDict attribute that only the cache decorator
        # of TensorDict methods reads, on the TensorDict, so it is a valid field
        # name.
        @tensorclass(tensor_only=tensor_only)
        class MyClass:
            x: torch.Tensor
            _cache: torch.Tensor

        c = MyClass(x=torch.zeros(3), _cache=torch.ones(3), batch_size=[3])
        assert (c._cache == 1).all()
        assert (torch.stack([c, c])[1, 0]._cache == 1).all()
        c._cache = torch.full((3,), 2.0)
        assert (c.get("_cache") == 2).all()

    @pytest.mark.parametrize("name", ["data", "fields"])
    @pytest.mark.parametrize(
        "base",
        [
            None,
            "tensor_only",
            TensorClass,
            TensorClass["frozen"],
            TensorClass["shadow"],
        ],
        ids=["decorator", "decorator-tensor_only", "subclass", "frozen", "shadow"],
    )
    def test_field_replaces_member(self, name, base):
        annotations = {name: torch.Tensor, "other": torch.Tensor}
        if base is None or base == "tensor_only":
            MyClass = tensorclass(tensor_only=base == "tensor_only")(
                type("MyClass", (), {"__annotations__": annotations})
            )
        else:
            MyClass = type("MyClass", (base,), {"__annotations__": annotations})
        assert [f.name for f in dataclasses.fields(MyClass)] == [name, "other"]
        c = MyClass(**{name: torch.ones(3), "other": torch.zeros(3)}, batch_size=[3])
        assert (getattr(c, name) == 1).all()
        assert getattr(c[0], name) == 1
        assert (getattr(torch.stack([c, c]), name) == 1).all()

    @pytest.mark.parametrize("subclass", [False, True])
    @pytest.mark.parametrize("frozen", [False, True])
    def test_shadow_method_name(self, subclass, frozen):
        if subclass:
            base = TensorClass["shadow", "frozen"] if frozen else TensorClass["shadow"]

            class MyClass(base):
                sum: torch.Tensor  # a method
                shape: torch.Tensor  # a property
                other: torch.Tensor

        else:

            @tensorclass(shadow=True, frozen=frozen)
            class MyClass:
                sum: torch.Tensor
                shape: torch.Tensor
                other: torch.Tensor

        with pytest.raises(TypeError, match="sum"):
            MyClass(shape=torch.ones(3), other=torch.zeros(3), batch_size=[3])
        c = MyClass(
            sum=torch.ones(3), shape=torch.ones(3), other=torch.zeros(3), batch_size=[3]
        )
        for name in ("sum", "shape"):
            assert (getattr(c, name) == 1).all()
            assert getattr(c[0], name) == 1
            if frozen:
                with pytest.raises(dataclasses.FrozenInstanceError):
                    setattr(c, name, torch.full((3,), 2.0))
            else:
                setattr(c, name, torch.full((3,), 2.0))
                assert (getattr(c, name) == 2).all()
                assert (c.get(name) == 2).all()


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
        is_called = False

        class SubClass(TensorClass):
            a: int

            def __setattr__(self, name, val):
                nonlocal is_called
                is_called = True
                super().__setattr__(name, val)

        assert is_tensorclass(SubClass)
        assert not SubClass._autocast
        assert not SubClass._nocast
        assert issubclass(SubClass, TensorClass)
        obj = SubClass(a=0)
        assert is_called
        is_called = False
        obj.a = 1
        assert obj.a == 1
        assert is_called

    def test_subclassing_autocast(self):
        is_called = False

        class SubClass(TensorClass, autocast=True):
            a: int

            def __setattr__(self, name, val):
                nonlocal is_called
                is_called = True
                super().__setattr__(name, val)

        assert is_tensorclass(SubClass)
        assert SubClass._autocast
        assert not SubClass._nocast
        assert issubclass(SubClass, TensorClass)
        assert isinstance(SubClass(torch.ones(())).a, int)

        obj = SubClass(a=0)
        assert is_called
        is_called = False
        obj.a = 1
        assert obj.a == 1
        assert is_called

        class SubClass(TensorClass["autocast"]):
            a: int

        assert not TensorClass._autocast
        assert is_tensorclass(SubClass)
        assert SubClass._autocast
        assert not SubClass._nocast
        assert issubclass(SubClass, TensorClass)
        assert isinstance(SubClass(torch.ones(())).a, int)

    def test_subclassing_nocast(self):
        is_called = False

        class SubClass(TensorClass, nocast=True):
            a: int

            def __setattr__(self, name, val):
                nonlocal is_called
                is_called = True
                super().__setattr__(name, val)

        assert is_tensorclass(SubClass)
        assert not SubClass._autocast
        assert SubClass._nocast
        assert issubclass(SubClass, TensorClass)
        assert isinstance(SubClass(1).a, int)

        obj = SubClass(a=0)
        assert is_called
        is_called = False
        obj.a = 1
        assert obj.a == 1
        assert is_called

        is_called = False

        class SubClass(TensorClass["nocast"]):
            a: int

            def __setattr__(self, name, val):
                nonlocal is_called
                is_called = True
                super().__setattr__(name, val)

        assert not TensorClass._nocast
        assert is_tensorclass(SubClass)
        assert not SubClass._autocast
        assert SubClass._nocast
        assert issubclass(SubClass, TensorClass)
        assert isinstance(SubClass(1).a, int)

        obj = SubClass(a=0)
        assert is_called
        is_called = False
        obj.a = 1
        assert obj.a == 1
        assert is_called

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
        is_called = False

        class SubClass(TensorClass, nocast=True):
            a: int
            b: int

            def __setattr__(self, key, value):
                nonlocal is_called
                is_called = True
                if key == "b":
                    return super().__setattr__("b", value + 1)
                return super().__setattr__("a", value - 1)

        s = SubClass(a=torch.zeros(3), b=torch.zeros(3))
        assert is_called
        is_called = False
        assert (s.a == -1).all()
        assert (s.b == 1).all()
        s.a = torch.ones(())
        assert is_called
        is_called = False
        s.b = torch.ones(())
        assert is_called
        assert (s.a == 0).all()
        assert (s.b == 2).all()

    # Regression test for GitHub issue #1469: the metaclass __getitem__ used to
    # read every subscript as a list of flags, so a generic TensorClass could not
    # be subscripted with types.
    def test_subclassing_generic(self):
        T = TypeVar("T")

        class Base(TensorClass, Generic[T]):
            x: torch.Tensor

        class Child(Base[T]):
            y: torch.Tensor

        class Concrete(Base[int]):
            y: torch.Tensor

        class Quoted(Base["int"]):
            y: torch.Tensor

        assert get_origin(Base[int]) is Base
        assert Child.__parameters__ == (T,)
        assert Concrete.__orig_bases__ == (Base[int],)
        for cls in (Child, Child[float], Concrete, Quoted):
            obj = cls(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
            assert isinstance(obj, Base)
            assert (obj[0].y == 1).all()

        # flags still configure the class
        class NoCast(Base["nocast"]):
            z: int

        assert isinstance(NoCast(x=torch.zeros(()), z=1).z, int)

        # other subscripts of a non-generic class are rejected
        with pytest.raises(TypeError, match="only accepts the flags"):
            TensorClass["autocst"]
        with pytest.raises(TypeError, match="only accepts the flags"):
            TensorClass[int]

    @pytest.mark.skipif(
        sys.version_info < (3, 12), reason="PEP 695 syntax requires Python 3.12"
    )
    def test_subclassing_generic_pep695(self):
        # exec keeps this file parseable on Python < 3.12
        namespace = {"__name__": __name__, "TensorClass": TensorClass, "torch": torch}
        exec(
            textwrap.dedent(
                """
                class Base[T: int](TensorClass):
                    x: torch.Tensor

                class Child[T: int](Base[T]):
                    y: torch.Tensor

                class Concrete(Base[int]):
                    y: torch.Tensor

                class Quoted(Base["int"]):
                    y: torch.Tensor
                """
            ),
            namespace,
        )
        Base, Child = namespace["Base"], namespace["Child"]
        assert Child.__parameters__ == Child.__type_params__
        for cls in (Child, Child[int], namespace["Concrete"], namespace["Quoted"]):
            obj = cls(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
            assert isinstance(obj, Base)
            assert (obj[0].y == 1).all()


class TestCustomInit:
    """A user-defined __init__ runs, with its own signature (gh-1822)."""

    @staticmethod
    def make_class(api, init, **flags):
        if api == "inheritance":

            class Data(TensorClass, **flags):
                x: torch.Tensor
                __init__ = init

            return Data

        class Data:
            x: torch.Tensor
            __init__ = init

        if api == "dataclass":
            Data = dataclasses.dataclass(Data)
        return tensorclass(Data, **flags)

    @pytest.mark.parametrize("api", ["decorator", "dataclass", "inheritance"])
    @pytest.mark.parametrize("tensor_only", [False, True])
    def test_custom_init(self, api, tensor_only):
        def init(self, x, scale=2.0):
            # The container exists when __init__ runs.
            assert self.batch_size == torch.Size([2])
            self.x = x * scale

        Data = self.make_class(api, init, tensor_only=tensor_only)
        x = torch.ones(2, 3)
        data = Data(x, batch_size=[2], device="cpu", names=["n"])
        torch.testing.assert_close(data.x, x * 2)
        assert data.device == torch.device("cpu")
        assert data.names == ["n"]
        torch.testing.assert_close(Data(x=x, scale=3.0, batch_size=[2]).x, x * 3)
        assert list(inspect.signature(Data).parameters) == [
            "x",
            "scale",
            "batch_size",
            "device",
            "names",
        ]

    @pytest.mark.parametrize("api", ["decorator", "inheritance"])
    def test_custom_init_var_kwargs(self, api):
        def init(self, x, **options):
            self.x = x * options.pop("scale")
            assert not options

        # Defining the class used to fail: the signature put the keyword-only
        # batch_size, device and names after **options.
        Data = self.make_class(api, init)
        x = torch.ones(2)
        data = Data(x=x, scale=3.0, batch_size=[2], names=["n"])
        torch.testing.assert_close(data.x, x * 3)
        assert data.names == ["n"]
        assert list(inspect.signature(Data).parameters) == [
            "x",
            "batch_size",
            "device",
            "names",
            "options",
        ]

    def test_custom_init_super(self):
        class Base(TensorClass):
            x: torch.Tensor

        class Forward(Base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        x = torch.ones(3)
        data = Forward(x=x, batch_size=[3], device="cpu", names=["n"], lock=True)
        torch.testing.assert_close(data.x, x)
        assert data.batch_size == torch.Size([3])
        assert data.device == torch.device("cpu")
        assert data.names == ["n"]
        assert data.is_locked

        class Child(Base):
            y: torch.Tensor

            def __init__(self, x):
                # y is kept, and the batch_size and names passed to
                # super().__init__() apply.
                self.y = x + 1
                super().__init__(x=x, batch_size=x.shape[:1], names=["n"])

        data = Child(x)
        torch.testing.assert_close(data.x, x)
        torch.testing.assert_close(data.y, x + 1)
        assert data.batch_size == torch.Size([3])
        assert data.names == ["n"]

    @pytest.mark.parametrize("api", ["decorator", "inheritance"])
    @pytest.mark.parametrize("tensor_only", [False, True])
    def test_custom_init_frozen(self, api, tensor_only):
        def init(self, x):
            object.__setattr__(self, "x", x * 2)

        Data = self.make_class(api, init, frozen=True, tensor_only=tensor_only)
        x = torch.ones(2)
        data = Data(x, batch_size=[2])
        torch.testing.assert_close(data.x, x * 2)
        torch.testing.assert_close(data.to_tensordict()["x"], x * 2)
        assert "x" not in data.__dict__
        assert data.is_locked
        with pytest.raises(dataclasses.FrozenInstanceError):
            data.x = x

    @pytest.mark.parametrize("tensor_only", [False, True])
    def test_custom_init_super_frozen(self, tensor_only):
        class Base(TensorClass, frozen=True, tensor_only=tensor_only):
            x: torch.Tensor

        class Child(Base):
            y: torch.Tensor

            def __init__(self, x):
                super().__init__(x=x)
                object.__setattr__(self, "y", self.x + 1)

        x = torch.ones(2)
        data = Child(x, batch_size=[2])
        torch.testing.assert_close(data.x, x)
        torch.testing.assert_close(data.y, x + 1)
        assert not set(data.__dict__) & {"x", "y"}
        assert data.is_locked

    def test_custom_init_super_no_args(self):
        class Data(TensorClass):
            x: torch.Tensor

            def __init__(self, x):
                super().__init__()
                self.x = x * 2

        torch.testing.assert_close(Data(torch.ones(2)).x, torch.full((2,), 2.0))
        with pytest.raises(NotImplementedError):
            TensorClass()

    @pytest.mark.parametrize("frozen", [False, True])
    def test_custom_init_super_lock(self, frozen):
        class Base(TensorClass, frozen=frozen):
            x: torch.Tensor

        class Child(Base):
            y: torch.Tensor = None

            def __init__(self, x):
                super().__init__(x=x * 2, lock=True)

        data = Child(torch.ones(2))
        torch.testing.assert_close(data.x, torch.full((2,), 2.0))
        assert data.y is None
        assert data.is_locked

    def test_custom_init_declared_metadata(self):
        received = []

        class Data(TensorClass):
            x: torch.Tensor

            def __init__(self, x, batch_size=None, device=None):
                received.append((batch_size, device))
                self.x = x

        data = Data(torch.ones(2), batch_size=[2], device="cpu")
        assert received == [([2], "cpu")]
        assert data.batch_size == torch.Size([2])
        assert data.device == torch.device("cpu")

        # With shadow=True, a field can have such a name: it is not metadata.
        class Shadow(TensorClass, shadow=True):
            batch_size: torch.Tensor

            def __init__(self, batch_size):
                self.batch_size = batch_size

        data = Shadow(batch_size=torch.ones(2))
        torch.testing.assert_close(data.batch_size, torch.ones(2))
        assert data.to_tensordict().batch_size == torch.Size([])

    @pytest.mark.parametrize("api", ["decorator", "inheritance"])
    def test_custom_init_inherited(self, api):
        def init(self, x):
            self.x = x * 2

        Parent = self.make_class(api, init)

        # Without an __init__ of their own, subclasses run Parent's. Fields it
        # does not set get their default.
        class Child(Parent):
            y: torch.Tensor = None
            z: torch.Tensor = dataclasses.field(default_factory=lambda: torch.zeros(2))

        class GrandChild(Child):
            pass

        if api == "decorator":
            Child = tensorclass(Child)
            GrandChild = tensorclass(GrandChild)
        x = torch.ones(2)
        for cls in (Child, GrandChild):
            data = cls(x, batch_size=[2])
            torch.testing.assert_close(data.x, x * 2)
            assert data.y is None
            torch.testing.assert_close(data.z, torch.zeros(2))
            assert list(inspect.signature(cls).parameters) == [
                "x",
                "batch_size",
                "device",
                "names",
            ]

    def test_custom_init_unset_field(self):
        class Data(TensorClass):
            x: torch.Tensor
            y: torch.Tensor
            z: torch.Tensor = dataclasses.field(init=False)

            def __init__(self, x):
                self.x = x

        with pytest.raises(TypeError, match="did not set the required field 'y'"):
            Data(torch.ones(2))

        class Parent(TensorClass):
            x: torch.Tensor

            def __init__(self, x):
                self.x = x

        class Child(Parent):
            y: torch.Tensor
            z: torch.Tensor = dataclasses.field(init=False)

        with pytest.raises(TypeError, match="did not set the required field 'y'"):
            Child(torch.ones(2))

        # init=False fields without a default get None, as with a generated
        # __init__.
        class Full(Data):
            def __init__(self, x):
                self.x = self.y = x

        assert Full(torch.ones(2)).z is None

    def test_custom_init_post_init(self):
        calls = []

        class Base(TensorClass):
            x: torch.Tensor

            def __post_init__(self):
                calls.append(type(self).__name__)

        class Custom(Base):
            def __init__(self, x):
                self.x = x

        class Calls(Base):
            def __init__(self, x):
                self.x = x
                self.__post_init__()

        class Super(Base):
            def __init__(self, x):
                super().__init__(x=x)

        # As with dataclasses, a custom __init__ does not call __post_init__
        # but the generated one, reached through super().__init__(), does.
        for cls in (Custom, Calls, Super):
            cls(torch.ones(2))
        assert calls == ["Calls", "Super"]

    def test_double_decoration(self):
        @tensorclass
        class Generated(TensorClass):
            x: torch.Tensor

        @tensorclass
        class Custom(TensorClass):
            x: torch.Tensor

            def __init__(self, x):
                self.x = x * 2

        @tensorclass
        class Decorated:
            x: torch.Tensor

        x = torch.ones(2)
        for cls, expected in (
            (Generated, x),
            (Custom, x * 2),
            (tensorclass(Decorated), x),
        ):
            data = cls(x=x, batch_size=[2], device="cpu", names=["n"])
            torch.testing.assert_close(data.x, expected)
            assert data.device == torch.device("cpu")
            assert data.names == ["n"]


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

    @pytest.mark.parametrize("mapping_type", [dict, UserDict])
    def test_tensor_only_tensordict_mapping(self, mapping_type):
        @tensorclass(tensor_only=True)
        class TensorOnlyMapping:
            data: TensorDict

        value = mapping_type({"tensor": torch.ones(())})
        tc = TensorOnlyMapping(data=value)

        assert isinstance(tc.data, TensorDict)
        assert tc.data["tensor"] == 1

        tc.data = mapping_type({"tensor": torch.zeros(())})
        assert isinstance(tc.data, TensorDict)
        assert tc.data["tensor"] == 0

        tc.set("data", mapping_type({"tensor": torch.ones(())}))
        assert isinstance(tc.data, TensorDict)
        assert tc.data["tensor"] == 1

    def test_tensor_only_tensordict_nested_mapping(self):
        class TensorOnlyMapping(TensorClass["tensor_only"]):
            data: TensorDict

        tc = TensorOnlyMapping(
            data=UserDict({"nested": UserDict({"tensor": torch.ones(())})})
        )

        assert isinstance(tc.data["nested"], TensorDict)
        assert tc.data["nested", "tensor"] == 1

    @pytest.mark.parametrize("set_method", ["constructor", "attribute", "set"])
    def test_tensor_only_preserves_tensordict(self, set_method):
        class TensorOnlyMapping(TensorClass["tensor_only"]):
            data: TensorDict

        value = TensorDict({"tensor": torch.ones(())}).lock_()
        if set_method == "constructor":
            tc = TensorOnlyMapping(data=value)
        else:
            tc = TensorOnlyMapping(data=TensorDict())
            if set_method == "attribute":
                tc.data = value
            else:
                tc.set("data", value)

        assert tc.data is value
        assert tc.data.is_locked

    def test_tensor_only_preserves_nested_tensordict(self):
        class TensorOnlyMapping(TensorClass["tensor_only"]):
            data: TensorDict

        value = TensorDict({"tensor": torch.ones(())}).lock_()
        tc = TensorOnlyMapping(data=UserDict({"nested": value}))

        assert tc.data["nested"] is value
        assert tc.data["nested"].is_locked

    @pytest.mark.parametrize("from_type", [False, True])
    def test_tensor_only_mapping_from_dataclass(self, from_type):
        @dataclasses.dataclass
        class Data:
            data: TensorDict

        value = UserDict({"tensor": torch.ones(())})
        if from_type:
            TensorOnlyData = from_dataclass(Data, tensor_only=True)
            tc = TensorOnlyData(data=value)
        else:
            tc = from_dataclass(Data(data=value), tensor_only=True)

        assert isinstance(tc.data, TensorDict)
        assert tc.data["tensor"] == 1

    @pytest.mark.parametrize("nested", [False, True])
    def test_tensor_only_from_dataclass_preserves_tensordict(self, nested):
        @dataclasses.dataclass
        class Data:
            data: TensorDict

        value = TensorDict({"tensor": torch.ones(())}).lock_()
        data = UserDict({"nested": value}) if nested else value

        tc = from_dataclass(Data(data=data), tensor_only=True)
        result = tc.data["nested"] if nested else tc.data

        assert result is value
        assert result.is_locked

    def test_mapping_with_any_annotation_stays_non_tensor(self):
        class NonTensorMapping(TensorClass):
            data: Any

        value = UserDict({"metadata": "value"})
        tc = NonTensorMapping(data=value)

        assert tc.data is value

    def test_tensor_only_non_tensor_mapping_stays_non_tensor(self):
        class NonTensorMapping(TensorClass["tensor_only"]):
            data: NonTensorData

        value = UserDict({"metadata": "value"})
        tc = NonTensorMapping(data=value)

        assert isinstance(tc.data, NonTensorData)
        assert tc.data.data is value

    def test_tensor_only_autocast_nocast(self):
        @tensorclass(tensor_only=True, autocast=False)
        class TensorOnly:
            a: torch.Tensor
            b: torch.Tensor
            c: torch.Tensor | None = None

        with pytest.raises(TypeError, match="tensor_only"):

            @tensorclass(tensor_only=True, nocast=True)
            class TensorOnlyNocast:
                a: torch.Tensor
                b: torch.Tensor
                c: torch.Tensor | None = None

        with pytest.raises(TypeError, match="tensor_only"):

            @tensorclass(tensor_only=True, autocast=True)
            class TensorOnlyAutocast:
                a: torch.Tensor
                b: torch.Tensor
                c: torch.Tensor | None = None

    def test_wrong_tensor_only(self):
        class TensorOnly(TensorClass["tensor_only"]):
            a: torch.IntTensor
            b: torch.LongTensor
            c: torch.Tensor | None = None
            d: torch.Tensor | Union[torch.IntTensor, torch.LongTensor] | None = None  # noqa
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

    def test_tensor_only_parameterized_generic(self):
        # Regression test for GitHub issue #1658:
        # tensor_only=True should accept parameterized generics like TensorDict[str, Tensor]
        @tensorclass(tensor_only=True)
        class TensorOnlyGeneric:
            a: torch.Tensor
            b: TensorDict[str, torch.Tensor]
            c: TensorDict[str, torch.Tensor] | None = None

        tc = TensorOnlyGeneric(
            a=torch.zeros(()),
            b=UserDict({"tensor": torch.ones(())}),
            c=UserDict({"tensor": torch.zeros(())}),
        )
        assert isinstance(tc.b, TensorDict)
        assert isinstance(tc.c, TensorDict)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
