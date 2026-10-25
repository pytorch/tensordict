# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import copy
import dataclasses
import importlib.util
import inspect
import itertools
import json
import os
import pathlib
import pickle
import re
import runpy
import shutil
import subprocess
import sys
import textwrap
import warnings
import weakref
import zipfile
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
    get_type_hints,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pytest
import tensordict.base as tensordict_base
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
    NonTensorDataBase,
    NonTensorStack,
    pad_sequence,
    set_capture_non_tensor_stack,
    set_get_defaults_to_none,
    set_list_to_stack,
    set_printoptions,
    TensorClass,
    tensorclass,
    TensorDict,
    TensorDictBase,
    UnbatchedTensor,
)
from tensordict._lazy import _PermutedTensorDict, _ViewedTensorDict
from tensordict._td import lazy_stack
from tensordict._utils_options import _set_capture_non_tensor_stack, _set_list_to_stack
from tensordict.base import (
    _GENERIC_NESTED_ERR,
    _get_defaults_to_none,
    _set_get_defaults_to_none,
)
from tensordict.tensorclass import _TensorClassMeta, from_dataclass
from tensordict.utils import _check_recursive_properties
from torch import Tensor

_has_streaming = importlib.util.find_spec("streaming", None) is not None
_has_mypy = importlib.util.find_spec("mypy") is not None
_has_pyright = importlib.util.find_spec("pyright") is not None
_has_jaxtyping = importlib.util.find_spec("jaxtyping") is not None

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.tensordict.test._utils_internal import (
        get_available_devices,
        streaming_dataset,
    )
else:
    IS_FB = False
    from _utils_internal import get_available_devices, streaming_dataset

# Capture all warnings
pytestmark = [
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings(
        "ignore:type_hints are none, cannot perform auto-casting"
    ),
    pytest.mark.filterwarnings(
        "ignore:You are using `torch.load` with `weights_only=False`"
    ),
    # On free-threaded Python, urllib3 imports brotli inside the streaming tests,
    # and brotli enables the GIL. Other modules that enable it still fail.
    pytest.mark.filterwarnings(
        r"ignore:The global interpreter lock \(GIL\) has been enabled to load module '_brotli':RuntimeWarning"
    ),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]

IS_FB = os.getenv("PYTORCH_TEST_FBCODE")


# The stubs of the tensordict under test: the source tree in a checkout, and the
# installed package in the nightly wheel tests, which delete the source tree.
_TENSORDICT_DIR = pathlib.Path(tensordict.__file__).parent


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
    # Deprecated (removed in TensorDict 0.17). It forwards to
    # LazyStackedTensorDict.extend, so it works only when the tensorclass wraps
    # a lazy stack; TensorDictBase has no extend.
    "extend",
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

    # The fields of X are its own attributes, not TensorClass API.
    exclusions |= {field.name for field in dataclasses.fields(X)}
    stub_attrs = _get_class_attrs_from_pyi(
        str(_TENSORDICT_DIR / "tensorclass.pyi"), "TensorClass"
    )
    _check_stub_class(stub_attrs, X, exclusions)


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.parametrize("cls", [NonTensorDataBase, NonTensorStack])
def test_non_tensor_stub_maybe_to_stack(cls):
    # The deprecation warning of NonTensorStack.from_nontensordata points to
    # NonTensorData.maybe_to_stack.
    assert hasattr(cls, "maybe_to_stack")
    stub_attrs = _get_class_attrs_from_pyi(
        str(_TENSORDICT_DIR / "tensorclass.pyi"), cls.__name__
    )
    assert "maybe_to_stack" in stub_attrs


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
def test_tensorclass_stub_constructor():
    # The stub types the constructor of a subclass with dataclass_transform:
    # its fields, then the keyword-only arguments of _TensorClassInitArgs,
    # which the runtime constructor must accept.
    with open(_TENSORDICT_DIR / "tensorclass.pyi", "r") as f:
        tree = ast.parse(f.read())
    (init_args_class,) = (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_TensorClassInitArgs"
    )
    init_args = {
        node.target.id
        for node in init_args_class.body
        if isinstance(node, ast.AnnAssign)
    }

    class X(TensorClass):
        x: torch.Tensor

    init_values = {"batch_size": [3], "device": "cpu", "names": ["n"], "lock": True}
    assert init_args == init_values.keys()
    x = X(torch.zeros(3), **init_values)
    assert x.batch_size == (3,)
    assert x.device == torch.device("cpu")
    assert x.names == ["n"]
    assert x.is_locked


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
def test_tensorclass_stub_constructor_static(tmp_path, monkeypatch):
    from mypy import api

    # Check tensorclass.pyi alone: the modules it imports are reduced to the
    # names these checks need, or to Any. Mypy also searches the working
    # directory, which must not hold the tensordict sources.
    monkeypatch.chdir(tmp_path)
    stubs = tmp_path / "stubs"
    package = stubs / "tensordict"
    package.mkdir(parents=True)
    torch_package = stubs / "torch"
    torch_package.mkdir()
    torch_package.joinpath("__init__.pyi").write_text(
        """from typing import Any

class Tensor: ...
class device: ...

def __getattr__(name: str) -> Any: ...
"""
    )
    package.joinpath("__init__.pyi").write_text(
        "from .tensorclass import TensorClass as TensorClass\n"
    )
    package.joinpath("utils.pyi").write_text(
        """from typing import Any, TypeAlias

import torch

DeviceType: TypeAlias = torch.device | str | int

def __getattr__(name: str) -> Any: ...
"""
    )
    package.joinpath("tensorclass.pyi").write_text(
        (_TENSORDICT_DIR / "tensorclass.pyi").read_text()
    )

    config = tmp_path / "mypy.ini"
    config.write_text(
        f"""[mypy]
python_version = 3.11
show_error_codes = True
mypy_path = {stubs}
no_site_packages = True
ignore_missing_imports = True
"""
    )

    def user_errors(source):
        user = tmp_path / "user.py"
        user.write_text(source)
        stdout, stderr, status = api.run(["--config-file", str(config), str(user)])
        assert not stderr, stderr
        return [line for line in stdout.splitlines() if line.startswith("user.py:")]

    # The runtime calls __post_init__() without arguments, so a plain
    # __post_init__(self) must type-check. Type checkers also know the class
    # option kw_only and InitVars.
    valid = """from dataclasses import InitVar
import torch
from tensordict import TensorClass

class Obs(TensorClass):
    a: torch.Tensor
    label: str = "x"

class Child(Obs):
    b: torch.Tensor | None = None

class Post(TensorClass):
    a: torch.Tensor

    def __post_init__(self) -> None: ...

class KwOnly(TensorClass, kw_only=True):
    a: torch.Tensor

class WithInitVar(TensorClass):
    a: torch.Tensor
    scale: InitVar[float] = 1.0

    def __post_init__(self, scale: float) -> None: ...

t = torch.Tensor()
Obs(t, "s", batch_size=[3], device="cpu", names=["n"], lock=False)
Obs(a=t, batch_size=3)
Child(t, "s", None, batch_size=[3])
Post(a=t)
KwOnly(a=t)
WithInitVar(t, 2.0)
WithInitVar(a=t, scale=2.0, batch_size=[3])
"""
    assert user_errors(valid) == []

    invalid = """import torch
from tensordict import TensorClass

class Obs(TensorClass):
    a: torch.Tensor

class KwOnly(TensorClass, kw_only=True):
    a: torch.Tensor

Obs(batch_size=[3])
Obs(a="not a tensor")
Obs(a=torch.Tensor(), extra=1)
Obs(a=torch.Tensor(), non_blocking=True)
KwOnly(torch.Tensor())
"""
    errors = "\n".join(user_errors(invalid))
    assert 'Missing positional argument "a"' in errors
    assert 'incompatible type "str"' in errors
    assert 'Unexpected keyword argument "extra"' in errors
    assert 'Unexpected keyword argument "non_blocking"' in errors
    assert 'Too many positional arguments for "KwOnly"' in errors


@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
def test_tensorclass_stub_python310_static(tmp_path, monkeypatch):
    # #3290: on Python 3.10, typing has no dataclass_transform or Self, so the
    # stub must take them from typing_extensions for tensorclasses to get
    # their fields.
    from mypy import api

    monkeypatch.chdir(tmp_path)
    stubs = tmp_path / "stubs"
    package = stubs / "tensordict"
    package.mkdir(parents=True)
    torch_package = stubs / "torch"
    torch_package.mkdir()
    torch_package.joinpath("__init__.pyi").write_text(
        """from typing import Any

class Tensor: ...
class device: ...

def __getattr__(name: str) -> Any: ...
"""
    )
    package.joinpath("__init__.pyi").write_text(
        "from .tensorclass import TensorClass as TensorClass\n"
        "from .tensorclass import tensorclass as tensorclass\n"
    )
    package.joinpath("utils.pyi").write_text(
        """from typing import Any, TypeAlias

import torch

DeviceType: TypeAlias = torch.device | str | int

def __getattr__(name: str) -> Any: ...
"""
    )
    package.joinpath("tensorclass.pyi").write_text(
        (_TENSORDICT_DIR / "tensorclass.pyi").read_text()
    )
    config = tmp_path / "mypy.ini"
    config.write_text(
        f"""[mypy]
python_version = 3.10
show_error_codes = True
mypy_path = {stubs}
no_site_packages = True
ignore_missing_imports = True
"""
    )
    user = tmp_path / "user.py"
    user.write_text(
        """import torch
from tensordict import TensorClass, tensorclass

class Observation(TensorClass):
    pixels: torch.Tensor

@tensorclass
class Step:
    obs: torch.Tensor

t = torch.Tensor()
Observation(pixels=t, batch_size=[4])
Observation(t, batch_size=[4])
Step(obs=t)
"""
    )
    stdout, stderr, status = api.run(["--config-file", str(config), str(user)])
    assert not stderr, stderr
    assert [line for line in stdout.splitlines() if line.startswith("user.py:")] == []


@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
def test_tensordict_self_python310_static(tmp_path, monkeypatch):
    # #3304: on Python 3.10, typing has no Self, so the .py modules must take
    # it from typing_extensions for methods typed -> Self to keep their type.
    from mypy import api

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source"
    shutil.copytree(
        _TENSORDICT_DIR,
        source / "tensordict",
        ignore=shutil.ignore_patterns("__pycache__", "*.so"),
    )
    stubs = tmp_path / "stubs"
    torch_package = stubs / "torch"
    torch_package.mkdir(parents=True)
    torch_package.joinpath("__init__.pyi").write_text(
        """from typing import Any

class Tensor: ...
class device: ...

def __getattr__(name: str) -> Any: ...
"""
    )
    config = tmp_path / "mypy.ini"
    config.write_text(
        f"""[mypy]
python_version = 3.10
show_error_codes = True
mypy_path = {stubs},{source}
no_site_packages = True
ignore_missing_imports = True
follow_imports = silent
"""
    )
    user = tmp_path / "user.py"
    user.write_text(
        """import torch
from tensordict import TensorDict

td = TensorDict(a=torch.zeros(4), batch_size=[4])
reveal_type(td.clone())
reveal_type(td == td)
"""
    )
    stdout, stderr, status = api.run(["--config-file", str(config), str(user)])
    assert not stderr, stderr
    assert [line for line in stdout.splitlines() if line.startswith("user.py:")] == [
        'user.py:5: note: Revealed type is "tensordict._td.TensorDict"',
        'user.py:6: note: Revealed type is "tensordict._td.TensorDict"',
    ]


# The names that typing has only from Python 3.11 or later.
_TYPING_NAMES_AFTER_PY310 = frozenset(
    {
        "assert_never",
        "assert_type",
        "clear_overloads",
        "dataclass_transform",
        "get_overloads",
        "get_protocol_members",
        "is_protocol",
        "LiteralString",
        "Never",
        "NoDefault",
        "NotRequired",
        "override",
        "ReadOnly",
        "Required",
        "reveal_type",
        "Self",
        "TypeAliasType",
        "TypeIs",
        "TypeVarTuple",
        "Unpack",
    }
)


def test_no_typing_import_newer_than_python310():
    # #3304: on Python 3.10, typing lacks the names above, so type checkers
    # can't resolve them there. Code that only type checkers read (the .pyi
    # stubs and the TYPE_CHECKING blocks) must import them from
    # typing_extensions. A runtime import with a fallback, inside try or under
    # a sys.version_info check, is fine.
    offenders = []
    for path in sorted(_TENSORDICT_DIR.rglob("*.py*")):
        if path.suffix not in (".py", ".pyi"):
            continue
        tree = ast.parse(path.read_text())
        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) or (
                isinstance(node, ast.If) and "version_info" in ast.unparse(node.test)
            ):
                guarded.update(id(child) for child in ast.walk(node))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "typing"
                and id(node) not in guarded
            ):
                names = {alias.name for alias in node.names} & _TYPING_NAMES_AFTER_PY310
                if names:
                    offenders.append(
                        f"{path.relative_to(_TENSORDICT_DIR.parent)}:{node.lineno}: "
                        f"{sorted(names)}"
                    )
    assert not offenders, "\n".join(offenders)


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
def test_tensorclass_stub_init_subclass():
    # The class keywords (class X(TensorClass, autocast=True)) go to the
    # metaclass at runtime; the stub declares them on __init_subclass__.
    with open(_TENSORDICT_DIR / "tensorclass.pyi", "r") as f:
        tree = ast.parse(f.read())
    (stub_class,) = (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TensorClass"
    )
    (init_subclass,) = (
        node
        for node in stub_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init_subclass__"
    )
    class_kwargs = {arg.arg for arg in init_subclass.args.kwonlyargs}

    meta_params = inspect.signature(_TensorClassMeta.__new__).parameters.values()
    assert class_kwargs == {
        param.name for param in meta_params if param.default is not param.empty
    }


# Short, correct user code over tensordict's common API (#3237). The
# installed-wheel test below runs mypy on it as a user would run it.
_REPO_ROOT = pathlib.Path(__file__).parents[2]
_TYPING_CORPUS = _REPO_ROOT / "test" / "typing_corpus"
_TYPING_CORPUS_FILES = sorted(path.name for path in _TYPING_CORPUS.glob("*.py"))
# Snippets that mypy still rejects because of filed annotation bugs.
_TYPING_CORPUS_MYPY_XFAILS = {
    "index_ops.py": "#3075: apply is typed Self | None",
    "torch_functions.py": "#3271: torch.stack and torch.cat of tensordicts are "
    "typed Tensor",
}


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory):
    # Build a wheel of the repository without build isolation and unpack it,
    # which is how pip installs a pure-Python wheel.
    setuptools = pytest.importorskip("setuptools")
    if int(setuptools.__version__.split(".")[0]) < 77:
        pytest.skip("pyproject.toml's build-system needs setuptools>=77")
    root = tmp_path_factory.mktemp("wheel")
    source = root / "source"
    shutil.copytree(
        _REPO_ROOT / "tensordict",
        source / "tensordict",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("setup.py", "pyproject.toml", "version.txt", "README.md", "LICENSE"):
        shutil.copy2(_REPO_ROOT / name, source / name)
    env = dict(
        os.environ,
        SETUPTOOLS_SCM_PRETEND_VERSION=(source / "version.txt").read_text().strip(),
    )
    build = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from setuptools import build_meta; "
            "print(build_meta.build_wheel(sys.argv[1]))",
            str(root / "dist"),
        ],
        cwd=source,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheel = root / "dist" / build.stdout.splitlines()[-1]
    site = root / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    return site


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
def test_wheel_ships_type_information(installed_wheel):
    # Without py.typed, mypy ignores the annotations of the installed package;
    # without the stubs, it reads the untyped sources instead.
    source = {
        path.relative_to(_REPO_ROOT)
        for pattern in ("*.pyi", "py.typed")
        for path in (_REPO_ROOT / "tensordict").rglob(pattern)
    }
    shipped = {
        path.relative_to(installed_wheel)
        for pattern in ("*.pyi", "py.typed")
        for path in (installed_wheel / "tensordict").rglob(pattern)
    }
    assert pathlib.Path("tensordict", "py.typed") in source
    assert source == shipped, f"missing from the wheel: {sorted(source - shipped)}"


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.parametrize("snippet", _TYPING_CORPUS_FILES)
def test_typing_corpus_runs(snippet):
    # The corpus must be correct code, so that a mypy error in it is a wrong
    # annotation in tensordict.
    runpy.run_path(str(_TYPING_CORPUS / snippet), run_name="__main__")


@pytest.fixture(scope="module")
def installed_wheel_mypy_errors(installed_wheel, tmp_path_factory):
    # Mypy reads the unpacked wheel as an installed package (PYTHONPATH
    # entries count as site-packages), with no configuration file: the
    # repository's mypy.ini must not apply. One run checks the whole corpus
    # and gives the errors of each file.
    if not _has_mypy:
        pytest.skip("mypy is not installed")
    root = tmp_path_factory.mktemp("mypy")
    corpus = root / "corpus"
    shutil.copytree(_TYPING_CORPUS, corpus)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("MYPYPATH", "PYTHONPATH")
    }
    env["PYTHONPATH"] = str(installed_wheel)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file=",
            # Writes no cache, which would hold about 100 MB of torch.
            f"--cache-dir={os.devnull}",
            *_TYPING_CORPUS_FILES,
        ],
        cwd=corpus,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert result.returncode in (0, 1), result.stdout + result.stderr
    errors = collections.defaultdict(list)
    for line in result.stdout.splitlines():
        if ": error: " in line:
            errors[line.split(":", 1)[0]].append(line)
    return errors


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
@pytest.mark.parametrize(
    "snippet",
    [
        pytest.param(
            snippet,
            marks=pytest.mark.xfail(
                strict=True, reason=_TYPING_CORPUS_MYPY_XFAILS[snippet]
            ),
        )
        if snippet in _TYPING_CORPUS_MYPY_XFAILS
        else snippet
        for snippet in _TYPING_CORPUS_FILES
    ],
)
def test_typing_corpus_installed_wheel_mypy(installed_wheel_mypy_errors, snippet):
    # #3237: with py.typed, mypy at default settings must accept correct user
    # code.
    errors = installed_wheel_mypy_errors[snippet]
    assert not errors, (
        f"mypy rejects correct code in test/typing_corpus/{snippet} against "
        "the installed wheel:\n" + "\n".join(errors)
    )


# #3305: the comparisons of TensorDictParams and NonTensorStack, an
# InteractionType argument, NonTensorStack built from plain values, and the
# batch_size, device and names of a TypedTensorDict instance.
_ISSUE_3305_SOURCE = """from typing import Any

import torch
from typing_extensions import assert_type
from tensordict import NonTensorData, NonTensorStack, TensorDict, TypedTensorDict
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    set_interaction_type,
    TensorDictParams,
)

params = TensorDictParams(TensorDict(w=torch.nn.Parameter(torch.zeros(3))))
assert_type(params == params, TensorDict)
assert_type(params != params, TensorDict)

stack = NonTensorStack("a", "b")
NonTensorStack(1, 2, stack_dim=0, batch_size=[2], device="cpu")
NonTensorStack(NonTensorData("a"), NonTensorData("b"))
assert_type(stack == stack, Any)
assert_type(stack != stack, Any)
by_number = stack == 1
by_tensor = stack != torch.zeros(2)
by_str = stack == "a"

ProbabilisticTensorDictModule(
    in_keys=["loc", "scale"],
    out_keys=["action"],
    distribution_class=torch.distributions.Normal,
    default_interaction_type=InteractionType.RANDOM,
)
with set_interaction_type(InteractionType.MODE):
    pass
kind: InteractionType = InteractionType.MEAN

class State(TypedTensorDict):
    x: torch.Tensor

state = State(x=torch.zeros(5, 3), batch_size=[5], device="cpu", names=["b"])
assert_type(state.batch_size, torch.Size)
assert_type(state.device, torch.device | None)
names = state.names
state.batch_size = torch.Size([5])
"""


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
def test_issue_3305_source_runs(tmp_path):
    # The checked source must be correct code.
    path = tmp_path / "user.py"
    path.write_text(_ISSUE_3305_SOURCE)
    runpy.run_path(str(path), run_name="__main__")


def _checker_env(installed_wheel):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("MYPYPATH", "PYTHONPATH")
    }
    env["PYTHONPATH"] = str(installed_wheel)
    return env


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
@pytest.mark.parametrize("python_version", ["3.10", "3.12"])
def test_issue_3305_installed_wheel_mypy(installed_wheel, tmp_path, python_version):
    # #3305: mypy typed params == params as TensorDictParams and stack ==
    # stack as NonTensorStack, rejected NonTensorStack("a", "b"), and on
    # Python 3.10 typed InteractionType.RANDOM as a str.
    tmp_path.joinpath("user.py").write_text(_ISSUE_3305_SOURCE)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file=",
            f"--cache-dir={os.devnull}",
            f"--python-version={python_version}",
            "user.py",
        ],
        cwd=tmp_path,
        env=_checker_env(installed_wheel),
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_pyright, reason="pyright is not installed")
@pytest.mark.parametrize("python_version", ["3.10", "3.12"])
def test_issue_3305_installed_wheel_pyright(installed_wheel, tmp_path, python_version):
    # #3305: pyright also rejected InteractionType.RANDOM as an
    # InteractionType on every Python version, and rejected reading
    # batch_size, device and names on a TypedTensorDict instance (init-only
    # fields), which mypy accepted.
    tmp_path.joinpath("user.py").write_text(_ISSUE_3305_SOURCE)
    # Keeps pyright from reading a configuration file above tmp_path.
    tmp_path.joinpath("pyrightconfig.json").write_text("{}")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            "--pythonpath",
            sys.executable,
            "--pythonversion",
            python_version,
            "user.py",
        ],
        cwd=tmp_path,
        env=_checker_env(installed_wheel),
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# #3320: save, dumps, memmap and memmap_like return a TensorDict on
# TensorDictParams and PersistentTensorDict, and memmap_like does on a
# sub-tensordict; TensorDictParams.memmap_like takes the keywords of the base
# class.
_ISSUE_3320_SOURCE = """import os
import tempfile

import torch
from typing_extensions import assert_type
from tensordict import LazyStackedTensorDict, PersistentTensorDict, TensorDict
from tensordict._td import _SubTensorDict
from tensordict.nn import TensorDictParams

with tempfile.TemporaryDirectory() as tmp:
    def path(name: str) -> str:
        return os.path.join(tmp, name)

    params = TensorDictParams(TensorDict(w=torch.nn.Parameter(torch.zeros(3))))
    assert_type(params.save(path("p1")), TensorDict)
    assert_type(params.dumps(path("p2"), archive=True), TensorDict)
    assert_type(params.memmap(path("p3")), TensorDict)
    params.memmap_like(path("p4"), existsok=True)
    params.memmap_like(path("p5.tdz"), archive=True)
    params.memmap_like(path("p6"), False, num_threads=2, return_early=False)
    params.memmap_like(path("p7"), share_non_tensor=False, robust_key=True)

    td = TensorDict(a=torch.zeros(3), batch_size=[3])
    h5: PersistentTensorDict = PersistentTensorDict.from_dict(
        td, filename=path("x.h5")
    )
    assert_type(h5.save(path("h1")), TensorDict)
    assert_type(h5.dumps(path("h2")), TensorDict)
    assert_type(h5.memmap(path("h3.tdz")), TensorDict)
    assert_type(h5.memmap_like(path("h4")), TensorDict)
    h5.close()

    sub = _SubTensorDict(td, slice(0, 2))
    assert_type(sub.memmap_like(path("s1")), TensorDict)
    assert_type(sub.save(path("s2")), _SubTensorDict)
    assert_type(td.save(path("t1")), TensorDict)
    lazy = LazyStackedTensorDict(td, td)
    assert_type(lazy.memmap(path("l1")), LazyStackedTensorDict)
"""


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
def test_issue_3320_source_runs(tmp_path):
    # The checked source must be correct code, and each assert_type must hold
    # at run time.
    pytest.importorskip("h5py")
    source = _ISSUE_3320_SOURCE.replace("assert_type(", "_check_type(")
    path = tmp_path / "user.py"
    path.write_text(source)

    def _check_type(value, cls):
        assert type(value) is cls, (type(value), cls)

    runpy.run_path(
        str(path), init_globals={"_check_type": _check_type}, run_name="__main__"
    )


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
@pytest.mark.parametrize("python_version", ["3.10", "3.12"])
def test_issue_3320_installed_wheel_mypy(installed_wheel, tmp_path, python_version):
    # #3320: mypy typed the save family of TensorDictParams and
    # PersistentTensorDict as the receiver's class.
    tmp_path.joinpath("user.py").write_text(_ISSUE_3320_SOURCE)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file=",
            f"--cache-dir={os.devnull}",
            f"--python-version={python_version}",
            "user.py",
        ],
        cwd=tmp_path,
        env=_checker_env(installed_wheel),
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_pyright, reason="pyright is not installed")
@pytest.mark.parametrize("python_version", ["3.10", "3.12"])
def test_issue_3320_installed_wheel_pyright(installed_wheel, tmp_path, python_version):
    # #3320: pyright also rejected the existsok, archive, return_early,
    # share_non_tensor and robust_key arguments of
    # TensorDictParams.memmap_like, which mypy accepted. It must reject
    # num_threads by position, which raises a TypeError at run time.
    tmp_path.joinpath("user.py").write_text(_ISSUE_3320_SOURCE)
    tmp_path.joinpath("invalid.py").write_text(
        """import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictParams

params = TensorDictParams(TensorDict(w=torch.nn.Parameter(torch.zeros(3))))
params.memmap_like("p", False, 2)
params.memmap_like("p", archive=1.5)
"""
    )
    # Keeps pyright from reading a configuration file above tmp_path.
    tmp_path.joinpath("pyrightconfig.json").write_text("{}")

    def pyright(path):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pyright",
                "--pythonpath",
                sys.executable,
                "--pythonversion",
                python_version,
                path,
            ],
            cwd=tmp_path,
            env=_checker_env(installed_wheel),
            capture_output=True,
            text=True,
            timeout=1200,
        )

    result = pyright("user.py")
    assert result.returncode == 0, result.stdout + result.stderr
    result = pyright("invalid.py")
    errors = [line for line in result.stdout.splitlines() if " - error: " in line]
    assert len(errors) == 2, result.stdout + result.stderr
    assert "invalid.py:6:" in errors[0]
    assert "invalid.py:7:" in errors[1]


# #3353: TensorDictSequential and ProbabilisticTensorDictSequential take their
# modules positionally (a dict, a list, or one argument per module) and their
# options by keyword.
_ISSUE_3353_HEADER = """from collections import OrderedDict

import torch
from tensordict import TensorDict
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)

net = TensorDictModule(
    lambda x: (x, x.abs() + 1), in_keys=["x"], out_keys=["loc", "scale"]
)
prob = ProbabilisticTensorDictModule(
    in_keys=["loc", "scale"],
    out_keys=["y"],
    distribution_class=torch.distributions.Normal,
)
mods = [net, prob]
"""
_ISSUE_3353_SOURCE = (
    _ISSUE_3353_HEADER
    + """ProbabilisticTensorDictSequential(net, prob)
ProbabilisticTensorDictSequential(
    net,
    prob,
    partial_tolerant=True,
    return_composite=True,
    selected_out_keys=["y"],
    inplace=False,
)
ProbabilisticTensorDictSequential(*mods)
ProbabilisticTensorDictSequential([net, prob])
ProbabilisticTensorDictSequential(
    [net, prob],
    partial_tolerant=True,
    return_composite=False,
    selected_out_keys=["y"],
    inplace=True,
)
ProbabilisticTensorDictSequential({"net": net, "prob": prob})
ProbabilisticTensorDictSequential(
    OrderedDict(net=net, prob=prob), partial_tolerant=True, inplace=None
)
seq = TensorDictSequential(net, prob)
TensorDictSequential(net, prob, partial_tolerant=True, selected_out_keys=["y"])
TensorDictSequential(*mods, inplace=True)
TensorDictSequential([net, prob], inplace="empty")
TensorDictSequential({"net": net, "prob": prob}, partial_tolerant=True)
TensorDictSequential(OrderedDict(net=net, prob=prob), inplace=False)
seq(TensorDict(x=torch.zeros(3)))
"""
)
# Each call raises a TypeError at run time, so type checkers must reject it.
_ISSUE_3353_INVALID = [
    "ProbabilisticTensorDictSequential(modules=[net, prob])",
    'ProbabilisticTensorDictSequential(modules={"net": net, "prob": prob})',
    "TensorDictSequential(modules=[net, prob])",
    'TensorDictSequential(modules={"net": net, "prob": prob})',
]


def _issue_3353_write_sources(tmp_path):
    tmp_path.joinpath("user.py").write_text(_ISSUE_3353_SOURCE)
    tmp_path.joinpath("invalid.py").write_text(
        _ISSUE_3353_HEADER + "\n".join(_ISSUE_3353_INVALID) + "\n"
    )
    first = _ISSUE_3353_HEADER.count("\n") + 1
    return list(range(first, first + len(_ISSUE_3353_INVALID)))


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
def test_issue_3353_source_runs(tmp_path):
    # The accepted calls must run, and each rejected call must raise a
    # TypeError at run time.
    _issue_3353_write_sources(tmp_path)
    runpy.run_path(str(tmp_path / "user.py"), run_name="__main__")
    for call in _ISSUE_3353_INVALID:
        path = tmp_path / "call.py"
        path.write_text(_ISSUE_3353_HEADER + call + "\n")
        with pytest.raises(TypeError, match="unexpected keyword argument 'modules'"):
            runpy.run_path(str(path), run_name="__main__")


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
@pytest.mark.parametrize("python_version", ["3.10", "3.12"])
def test_issue_3353_installed_wheel_mypy(installed_wheel, tmp_path, python_version):
    # #3353: mypy accepted the modules of both sequences by keyword, which
    # raises a TypeError at run time.
    expected = _issue_3353_write_sources(tmp_path)

    def mypy(path):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "--config-file=",
                f"--cache-dir={os.devnull}",
                f"--python-version={python_version}",
                path,
            ],
            cwd=tmp_path,
            env=_checker_env(installed_wheel),
            capture_output=True,
            text=True,
            timeout=1200,
        )

    result = mypy("user.py")
    assert result.returncode == 0, result.stdout + result.stderr
    result = mypy("invalid.py")
    lines = {
        int(line.split(":")[1])
        for line in result.stdout.splitlines()
        if ": error: " in line
    }
    assert sorted(lines) == expected, result.stdout + result.stderr


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_pyright, reason="pyright is not installed")
@pytest.mark.parametrize("python_version", ["3.10", "3.12"])
def test_issue_3353_installed_wheel_pyright(installed_wheel, tmp_path, python_version):
    # #3353: pyright accepted the modules of both sequences by keyword, which
    # raises a TypeError at run time.
    expected = _issue_3353_write_sources(tmp_path)
    # Keeps pyright from reading a configuration file above tmp_path.
    tmp_path.joinpath("pyrightconfig.json").write_text("{}")

    def pyright(path):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pyright",
                "--pythonpath",
                sys.executable,
                "--pythonversion",
                python_version,
                path,
            ],
            cwd=tmp_path,
            env=_checker_env(installed_wheel),
            capture_output=True,
            text=True,
            timeout=1200,
        )

    result = pyright("user.py")
    assert result.returncode == 0, result.stdout + result.stderr
    result = pyright("invalid.py")
    lines = {
        int(line.split("invalid.py:")[1].split(":")[0])
        for line in result.stdout.splitlines()
        if " - error: " in line
    }
    assert sorted(lines) == expected, result.stdout + result.stderr


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
def test_tensorclass_stub_static_types(tmp_path, monkeypatch):
    from mypy import api

    # Check tensorclass.pyi alone: the modules it imports are reduced to the
    # names these checks need, or to Any. Mypy also searches the working
    # directory, which must not hold the tensordict sources.
    monkeypatch.chdir(tmp_path)
    stubs = tmp_path / "stubs"
    package = stubs / "tensordict"
    package.mkdir(parents=True)
    torch_package = stubs / "torch"
    torch_package.mkdir()
    torch_package.joinpath("__init__.pyi").write_text(
        """from typing import Any

class Tensor: ...
class device: ...

def __getattr__(name: str) -> Any: ...
"""
    )
    package.joinpath("__init__.pyi").write_text(
        "from .tensorclass import TensorClass as TensorClass\n"
    )
    package.joinpath("utils.pyi").write_text(
        """from typing import Any, TypeAlias

import torch

DeviceType: TypeAlias = torch.device | str | int

def __getattr__(name: str) -> Any: ...
"""
    )
    package.joinpath("_tensorcollection.pyi").write_text(
        "class TensorCollection: ...\n"
    )
    package.joinpath("tensorclass.pyi").write_text(
        (_TENSORDICT_DIR / "tensorclass.pyi").read_text()
    )

    config = tmp_path / "mypy.ini"
    config.write_text(
        f"""[mypy]
python_version = 3.11
show_error_codes = True
mypy_path = {stubs}
no_site_packages = True
ignore_missing_imports = True
"""
    )

    valid = tmp_path / "valid.py"
    valid.write_text(
        """from typing import assert_type
import torch
from tensordict import TensorClass

class Obs(TensorClass):
    a: torch.Tensor
    label: str = "x"

class Flagged(TensorClass, autocast=True, tensor_only=False):
    a: torch.Tensor

t = torch.Tensor()
obs = Obs(t, "s", batch_size=[3], device="cpu", names=["n"], lock=False)
Flagged(a=t, batch_size=3)
assert_type(obs.a, torch.Tensor)
assert_type(obs[0], Obs)
assert_type(obs.unbind(0), tuple[Obs, ...])
assert_type(obs.to("cpu"), Obs)
assert_type(obs + 1, Obs)
assert_type(1 + obs, Obs)
assert_type(obs * 2.0, Obs)
assert_type(-obs, Obs)
assert_type(obs == obs, Obs)
for item in obs:
    assert_type(item, Obs)

# apply gives None only when fn returns None for every leaf, inplace or not,
# and filter_empty is not False.
def double(x: torch.Tensor) -> torch.Tensor:
    return x
def drop(x: torch.Tensor) -> None: ...
def maybe_drop(x: torch.Tensor) -> torch.Tensor | None:
    return x
def named_double(key: str, x: torch.Tensor) -> torch.Tensor:
    return x
def named_drop(key: str, x: torch.Tensor) -> None: ...
assert_type(obs.apply(double), Obs)
assert_type(obs.apply(double, inplace=True), Obs)
assert_type(obs.apply(drop), Obs | None)
assert_type(obs.apply(drop, inplace=True), Obs | None)
assert_type(obs.apply(maybe_drop), Obs | None)
assert_type(obs.apply(drop, filter_empty=False), Obs)
assert_type(obs.apply(double, filter_empty=True), Obs | None)
assert_type(obs.named_apply(named_double), Obs)
assert_type(obs.named_apply(named_drop), Obs | None)
assert_type(obs.named_apply(named_drop, filter_empty=False), Obs)
"""
    )
    stdout, stderr, status = api.run(["--config-file", str(config), str(valid)])
    assert status == 0, stdout + stderr


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
def test_tensorclass_stub_update_static_types(tmp_path, monkeypatch):
    # #3237: the update of tensorclass.pyi takes nested keys and a typed dict.
    # tensorclass.pyi is checked alone, as in
    # test_tensorclass_stub_static_types: the modules it imports are reduced
    # to the names these checks need, or to Any. Mypy also searches the
    # working directory, which must not hold the tensordict sources.
    from mypy import api

    monkeypatch.chdir(tmp_path)
    stubs = tmp_path / "stubs"
    package = stubs / "tensordict"
    package.mkdir(parents=True)
    torch_package = stubs / "torch"
    torch_package.mkdir()
    torch_package.joinpath("__init__.pyi").write_text(
        """from typing import Any

class Tensor: ...
class device: ...

def __getattr__(name: str) -> Any: ...
"""
    )
    package.joinpath("__init__.pyi").write_text(
        "from .tensorclass import TensorClass as TensorClass\n"
    )
    package.joinpath("utils.pyi").write_text(
        """from typing import Any, TypeAlias

import torch

DeviceType: TypeAlias = torch.device | str | int

def __getattr__(name: str) -> Any: ...
"""
    )
    package.joinpath("_tensorcollection.pyi").write_text(
        "class TensorCollection: ...\n"
    )
    package.joinpath("tensorclass.pyi").write_text(
        (_TENSORDICT_DIR / "tensorclass.pyi").read_text()
    )

    config = tmp_path / "mypy.ini"
    config.write_text(
        f"""[mypy]
python_version = 3.11
show_error_codes = True
mypy_path = {stubs}
no_site_packages = True
ignore_missing_imports = True
"""
    )

    valid = tmp_path / "valid.py"
    valid.write_text(
        """from typing import assert_type
import torch
from tensordict import TensorClass

class Obs(TensorClass):
    a: torch.Tensor

t = torch.Tensor()
obs = Obs(a=t, batch_size=[3])
data: dict[str, torch.Tensor] = {"a": t}
assert_type(obs.update({("a", "b"): t}), Obs)
assert_type(obs.update(data), Obs)
"""
    )
    stdout, stderr, status = api.run(["--config-file", str(config), str(valid)])
    assert status == 0, stdout + stderr


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
def test_tensordict_stub_source_static_types(tmp_path, monkeypatch):
    from mypy import api

    # Check the source annotations of TensorDict.__init__ and from_dict alone:
    # their signatures are copied from _td.py into a stub, next to the names
    # they use. Mypy also searches the working directory, which must not hold
    # the tensordict sources.
    monkeypatch.chdir(tmp_path)
    with open(_TENSORDICT_DIR / "_td.py", "r") as f:
        tree = ast.parse(f.read())
    (class_node,) = (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TensorDict"
    )
    methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in ("__init__", "from_dict")
    ]
    assert len(methods) == 2
    for method in methods:
        method.body = [ast.Expr(ast.Constant(...))]
    stubs = tmp_path / "stubs"
    package = stubs / "tensordict"
    package.mkdir(parents=True)
    torch_package = stubs / "torch"
    torch_package.mkdir()
    torch_package.joinpath("__init__.pyi").write_text(
        """from typing import Any

class Tensor: ...
class Size: ...

def __getattr__(name: str) -> Any: ...
"""
    )
    package.joinpath("_nestedkey.pyi").write_text(
        (_TENSORDICT_DIR / "_nestedkey.pyi").read_text()
    )
    methods_source = textwrap.indent("\n".join(map(ast.unparse, methods)), "    ")
    package.joinpath("__init__.pyi").write_text(
        f"""from collections.abc import Mapping, Sequence
from typing import Any, Self, TypeVar

import torch
from tensordict._nestedkey import NestedKey

class TensorCollection: ...
class TensorDictBase(TensorCollection): ...

T = TypeVar("T", bound=TensorCollection)
CompatibleType = torch.Tensor | TensorCollection
DeviceType = Any

class TensorDict(TensorDictBase):
{methods_source}
"""
    )

    config = tmp_path / "mypy.ini"
    config.write_text(
        f"""[mypy]
python_version = 3.11
show_error_codes = True
mypy_path = {stubs}
no_site_packages = True
ignore_missing_imports = True
"""
    )

    def user_errors(source):
        user = tmp_path / "user.py"
        user.write_text(source)
        stdout, stderr, status = api.run(["--config-file", str(config), str(user)])
        assert not stderr, stderr
        return [line for line in stdout.splitlines() if line.startswith("user.py:")]

    # A nested dict and a dict[str, Tensor] variable work at runtime.
    valid = """import torch
from tensordict import TensorDict
from tensordict._nestedkey import NestedKey

t = torch.Tensor()
data: dict[str, torch.Tensor] = {"a": t}
nested_key_data: dict[NestedKey, torch.Tensor] = {("a", "b"): t}
TensorDict({"a": t, "b": {"c": t}}, batch_size=[3])
TensorDict(data, batch_size=[3])
TensorDict(nested_key_data)
TensorDict({("a", "b"): t})
TensorDict({"a": t, ("b", "c"): t})
TensorDict(TensorDict(data))
TensorDict.from_dict({"a": t, "b": {"c": t}}, batch_size=[3])
TensorDict.from_dict(data)
"""
    assert user_errors(valid) == []

    invalid = """import torch
from tensordict import TensorDict

TensorDict(3)
TensorDict.from_dict([torch.Tensor()])
"""
    errors = user_errors(invalid)
    assert len(errors) == 2, errors
    assert 'Argument 1 to "TensorDict" has incompatible type "int"' in errors[0]
    assert 'Argument 1 to "from_dict" of "TensorDict"' in errors[1]


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
def test_common_calls_source_static_types(tmp_path, monkeypatch):
    # #3237: with py.typed, mypy must accept td.items(), td.update() with a
    # nested key, TensorDictSequential(m1, m2) and setting out_keys. The
    # annotations of CompatibleType, update, items, values, out_keys and the
    # TensorDictSequential overloads are copied from the sources into a
    # package of their own, with the import cycle of base.py and
    # _tensorcollection.pyi. Mypy also searches the working directory, which
    # must not hold the tensordict sources.
    from mypy import api

    monkeypatch.chdir(tmp_path)

    def members(path, class_name, names, *, overloads_only=False):
        with open(_TENSORDICT_DIR / path, "r") as f:
            tree = ast.parse(f.read())
        (class_node,) = (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        nodes = [
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
            and node.name in names
            and not (
                overloads_only
                and "overload" not in map(ast.unparse, node.decorator_list)
            )
        ]
        for node in nodes:
            node.body = [ast.Expr(ast.Constant(...))]
        return textwrap.indent("\n".join(map(ast.unparse, nodes)), "    ")

    with open(_TENSORDICT_DIR / "base.py", "r") as f:
        (compatible_type,) = (
            ast.unparse(node)
            for node in ast.parse(f.read()).body
            if ast.unparse(node).startswith(("CompatibleType =", "CompatibleType:"))
        )

    stubs = tmp_path / "stubs"
    package = stubs / "tensordict"
    (package / "nn").mkdir(parents=True)
    torch_package = stubs / "torch"
    torch_package.mkdir()
    torch_package.joinpath("__init__.pyi").write_text(
        """from typing import Any

class Tensor:
    shape: Any

def __getattr__(name: str) -> Any: ...
"""
    )
    package.joinpath("_nestedkey.pyi").write_text(
        (_TENSORDICT_DIR / "_nestedkey.pyi").read_text()
    )
    imports = """from collections.abc import Mapping
from typing import (
    Any, Callable, Iterator, List, OrderedDict, overload, Self, Sequence, Tuple,
    Type, TypeAlias, TypeVar, Union,
)

from torch import Tensor
from tensordict._nestedkey import NestedKey
"""
    package.joinpath("__init__.pyi").write_text("")
    package.joinpath("_tensorcollection.pyi").write_text(
        f"""{imports}
from tensordict.base import TensorDictBase

T = TypeVar("T", bound="TensorCollection")
CompatibleType: TypeAlias = Tensor

class TensorCollection:
    def to_tensordict(self) -> TensorDictBase: ...
{members("_tensorcollection.pyi", "TensorCollection", ("update",))}
"""
    )
    # A module, not a stub, as in the sources: the cycle hides
    # TensorCollection from a plain alias in a module only.
    package.joinpath("base.py").write_text(
        f"""# mypy: disable-error-code="empty-body"
from __future__ import annotations

{imports}
from tensordict._tensorcollection import TensorCollection

T = TypeVar("T", bound="TensorCollection")

{compatible_type}

class TensorDictBase(TensorCollection):
{members("base.py", "TensorDictBase", ("update", "items", "values"))}
"""
    )
    module_names = ("in_keys", "out_keys", "out_keys_source")
    package.joinpath("nn", "__init__.pyi").write_text(
        f"""{imports}
from tensordict.base import TensorDictBase

class TensorDictModuleBase:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
{members("nn/common.py", "TensorDictModuleBase", module_names)}

class TensorDictModuleWrapper(TensorDictModuleBase):
{members("nn/common.py", "TensorDictModuleWrapper", module_names)}

class TensorDictSequential(TensorDictModuleBase):
{members("nn/sequence.py", "TensorDictSequential", ("__init__",), overloads_only=True)}
"""
    )

    config = tmp_path / "mypy.ini"
    config.write_text(
        f"""[mypy]
python_version = 3.11
show_error_codes = True
mypy_path = {stubs}
no_site_packages = True
ignore_missing_imports = True
"""
    )

    def mypy_errors(source):
        user = tmp_path / "user.py"
        user.write_text(source)
        stdout, stderr, status = api.run(["--config-file", str(config), str(user)])
        assert not stderr, stderr
        # Errors in the copied annotations count too.
        return [line for line in stdout.splitlines() if ": error: " in line]

    # Entering the cycle at _tensorcollection hid CompatibleType from mypy.
    valid = """from typing import assert_type
import torch
from tensordict._tensorcollection import TensorCollection
from tensordict.base import CompatibleType, TensorDictBase
from tensordict.nn import (
    TensorDictModuleBase, TensorDictModuleWrapper, TensorDictSequential,
)

def check(td: TensorDictBase, tc: TensorCollection, t: torch.Tensor) -> None:
    for key, value in td.items():
        assert_type(value, torch.Tensor | TensorCollection)
    for value in td.values():
        assert_type(value, torch.Tensor | TensorCollection)
    x: CompatibleType = t
    data: dict[str, torch.Tensor] = {"a": t}
    entries: dict[str, CompatibleType] = {"a": t}
    nested: dict[tuple[str, str], torch.Tensor] = {("a", "b"): t}
    td.update(data)
    td.update(entries)
    td.update(nested)
    td.update({("a", "b"): t})
    td.update({"a": t, ("b", "c"): t})
    td.update(td)
    tc.update({("a", "b"): t})
    tc.update(data)

class Module(TensorDictModuleBase):
    def __init__(self) -> None:
        self.in_keys = ["a"]
        self.out_keys = ["b"]

class Wrapper(TensorDictModuleWrapper):
    def __init__(self) -> None:
        self.out_keys = ["b"]

def fn(td: TensorDictBase) -> TensorDictBase:
    return td

TensorDictSequential(Module(), Module())
TensorDictSequential(Module(), fn, selected_out_keys=["b"])
TensorDictSequential([Module(), Module()])
"""
    assert mypy_errors(valid) == []

    invalid = """from tensordict.base import TensorDictBase
from tensordict.nn import TensorDictSequential

def check(td: TensorDictBase) -> None:
    td.update(3)

TensorDictSequential(3)
"""
    errors = mypy_errors(invalid)
    assert len(errors) == 2, errors
    assert 'Argument 1 to "update" of "TensorDictBase"' in errors[0]
    assert "TensorDictSequential" in errors[1]


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
@pytest.mark.parametrize("python_version", ["3.10", "3.12"])
def test_sequential_overloads_source_static_types(
    tmp_path, monkeypatch, python_version
):
    # #3303: mypy must accept ProbabilisticTensorDictSequential(m1, m2) and a
    # plain dict of modules, as documented. The __init__ overloads of both
    # sequences are copied from the sources into a stub. Mypy also searches
    # the working directory, which must not hold the tensordict sources.
    from mypy import api

    monkeypatch.chdir(tmp_path)

    def overloads(path, class_name):
        with open(_TENSORDICT_DIR / path, "r") as f:
            tree = ast.parse(f.read())
        (class_node,) = (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        nodes = [
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "__init__"
            and "overload" in map(ast.unparse, node.decorator_list)
        ]
        for node in nodes:
            node.body = [ast.Expr(ast.Constant(...))]
        return textwrap.indent("\n".join(map(ast.unparse, nodes)), "    ")

    stubs = tmp_path / "stubs"
    package = stubs / "tensordict"
    (package / "nn").mkdir(parents=True)
    package.joinpath("__init__.pyi").write_text("")
    package.joinpath("_nestedkey.pyi").write_text(
        (_TENSORDICT_DIR / "_nestedkey.pyi").read_text()
    )
    package.joinpath("nn", "__init__.pyi").write_text(
        f"""from typing import Any, Callable, List, OrderedDict, overload

from tensordict._nestedkey import NestedKey

class TensorDictBase: ...

class TensorDictModuleBase:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class TensorDictModule(TensorDictModuleBase):
    def __init__(self, module: Any, in_keys: Any, out_keys: Any) -> None: ...

class ProbabilisticTensorDictModule(TensorDictModuleBase):
    def __init__(self, **kwargs: Any) -> None: ...

class TensorDictSequential(TensorDictModuleBase):
{overloads("nn/sequence.py", "TensorDictSequential")}

class ProbabilisticTensorDictSequential(TensorDictSequential):
{overloads("nn/probabilistic.py", "ProbabilisticTensorDictSequential")}
"""
    )

    config = tmp_path / "mypy.ini"
    config.write_text(
        f"""[mypy]
python_version = {python_version}
show_error_codes = True
mypy_path = {stubs}
no_site_packages = True
ignore_missing_imports = True
"""
    )

    def mypy_errors(source):
        user = tmp_path / "user.py"
        user.write_text(source)
        stdout, stderr, status = api.run(["--config-file", str(config), str(user)])
        assert not stderr, stderr
        # Errors in the copied annotations count too.
        return [line for line in stdout.splitlines() if ": error: " in line]

    valid = """import collections
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)

net = TensorDictModule(None, in_keys=["x"], out_keys=["loc", "scale"])
prob = ProbabilisticTensorDictModule(in_keys=["loc", "scale"], out_keys=["action"])
ProbabilisticTensorDictSequential(net, prob)
ProbabilisticTensorDictSequential(net, prob, return_composite=True)
ProbabilisticTensorDictSequential({"net": net, "prob": prob})
ProbabilisticTensorDictSequential(collections.OrderedDict(net=net, prob=prob))
TensorDictSequential({"net": net, "prob": prob})
TensorDictSequential(collections.OrderedDict(net=net, prob=prob))
"""
    assert mypy_errors(valid) == []

    invalid = """from tensordict.nn import (
    ProbabilisticTensorDictSequential,
    TensorDictSequential,
)

ProbabilisticTensorDictSequential(3)
TensorDictSequential({"net": 3})
"""
    errors = mypy_errors(invalid)
    assert len(errors) == 2, errors
    assert errors[0].startswith("user.py:6:"), errors
    assert "ProbabilisticTensorDictSequential" in errors[0]
    assert errors[1].startswith("user.py:7:"), errors


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
@pytest.mark.parametrize("python_version", ["3.10", "3.12"])
def test_sequential_overloads_keyword_only_source_static_types(
    tmp_path, monkeypatch, python_version
):
    # #3332: every option after the modules is keyword-only at runtime, so
    # mypy must reject ProbabilisticTensorDictSequential(modules, True) for a
    # dict or a list of modules. The __init__ overloads of both sequences are
    # copied from the sources into a stub. Mypy also searches the working
    # directory, which must not hold the tensordict sources.
    from mypy import api

    monkeypatch.chdir(tmp_path)

    def overloads(path, class_name):
        with open(_TENSORDICT_DIR / path, "r") as f:
            tree = ast.parse(f.read())
        (class_node,) = (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        nodes = [
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "__init__"
            and "overload" in map(ast.unparse, node.decorator_list)
        ]
        for node in nodes:
            node.body = [ast.Expr(ast.Constant(...))]
        return textwrap.indent("\n".join(map(ast.unparse, nodes)), "    ")

    stubs = tmp_path / "stubs"
    package = stubs / "tensordict"
    (package / "nn").mkdir(parents=True)
    package.joinpath("__init__.pyi").write_text("")
    package.joinpath("_nestedkey.pyi").write_text(
        (_TENSORDICT_DIR / "_nestedkey.pyi").read_text()
    )
    package.joinpath("nn", "__init__.pyi").write_text(
        f"""from typing import Any, Callable, List, OrderedDict, overload

from tensordict._nestedkey import NestedKey

class TensorDictBase: ...

class TensorDictModuleBase:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class TensorDictModule(TensorDictModuleBase):
    def __init__(self, module: Any, in_keys: Any, out_keys: Any) -> None: ...

class ProbabilisticTensorDictModule(TensorDictModuleBase):
    def __init__(self, **kwargs: Any) -> None: ...

class TensorDictSequential(TensorDictModuleBase):
{overloads("nn/sequence.py", "TensorDictSequential")}

class ProbabilisticTensorDictSequential(TensorDictSequential):
{overloads("nn/probabilistic.py", "ProbabilisticTensorDictSequential")}
"""
    )

    config = tmp_path / "mypy.ini"
    config.write_text(
        f"""[mypy]
python_version = {python_version}
show_error_codes = True
mypy_path = {stubs}
no_site_packages = True
ignore_missing_imports = True
"""
    )

    def mypy_errors(source):
        user = tmp_path / "user.py"
        user.write_text(source)
        stdout, stderr, status = api.run(["--config-file", str(config), str(user)])
        assert not stderr, stderr
        # Errors in the copied annotations count too.
        return [line for line in stdout.splitlines() if ": error: " in line]

    header = """from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)

net = TensorDictModule(None, in_keys=["x"], out_keys=["loc", "scale"])
prob = ProbabilisticTensorDictModule(in_keys=["loc", "scale"], out_keys=["action"])
"""
    valid = (
        header
        + """ProbabilisticTensorDictSequential(net, prob)
ProbabilisticTensorDictSequential(net, prob, partial_tolerant=True)
ProbabilisticTensorDictSequential({"net": net, "prob": prob})
ProbabilisticTensorDictSequential(
    {"net": net, "prob": prob}, partial_tolerant=True, return_composite=True
)
ProbabilisticTensorDictSequential([net, prob])
ProbabilisticTensorDictSequential(
    [net, prob], partial_tolerant=True, return_composite=True
)
TensorDictSequential({"net": net, "prob": prob}, partial_tolerant=True)
TensorDictSequential([net, prob], partial_tolerant=True)
"""
    )
    assert mypy_errors(valid) == []

    invalid = [
        'ProbabilisticTensorDictSequential({"net": net, "prob": prob}, True)',
        'ProbabilisticTensorDictSequential({"net": net, "prob": prob}, False, True)',
        "ProbabilisticTensorDictSequential([net, prob], True)",
        "ProbabilisticTensorDictSequential([net, prob], False, True)",
        'TensorDictSequential({"net": net, "prob": prob}, True)',
        "TensorDictSequential([net, prob], True)",
    ]
    errors = mypy_errors(header + "\n".join(invalid) + "\n")
    first = header.count("\n") + 1
    lines = sorted({int(error.split(":")[1]) for error in errors})
    assert lines == list(range(first, first + len(invalid))), errors


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
def test_stack_cat_source_static_types(tmp_path, monkeypatch):
    from mypy import api

    # Check the source annotations of stack, cat and lazy_stack alone: their
    # signatures and the TYPE_CHECKING block that defines their TypeVar are
    # copied from _td_functions.py into a stub. Mypy also searches the working
    # directory, which must not hold the tensordict sources.
    monkeypatch.chdir(tmp_path)
    with open(_TENSORDICT_DIR / "_td_functions.py", "r") as f:
        tree = ast.parse(f.read())
    names = ("stack", "cat", "lazy_stack", "maybe_dense_stack")
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(functions) == len(names)
    for function in functions:
        function.body = [ast.Expr(ast.Constant(...))]
    (type_checking,) = (
        node
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"
    )
    stubs = tmp_path / "stubs"
    package = stubs / "tensordict"
    package.mkdir(parents=True)
    package.joinpath("_tensorcollection.pyi").write_text(
        "class TensorCollection: ...\n"
    )
    package.joinpath("tensorclass.pyi").write_text("class TensorClass: ...\n")
    package.joinpath("__init__.pyi").write_text(
        """from typing import Sequence, TYPE_CHECKING

from tensordict._tensorcollection import TensorCollection as TensorCollection
from tensordict.tensorclass import TensorClass as TensorClass

"""
        + "\n".join(map(ast.unparse, [type_checking, *functions]))
    )

    config = tmp_path / "mypy.ini"
    config.write_text(
        f"""[mypy]
python_version = 3.11
show_error_codes = True
mypy_path = {stubs}
no_site_packages = True
ignore_missing_imports = True
"""
    )

    valid = tmp_path / "valid.py"
    valid.write_text(
        """from typing import assert_type
import tensordict
from tensordict import TensorClass, TensorCollection

class Obs(TensorClass): ...
class TD(TensorCollection): ...

obs = Obs()
td = TD()
assert_type(tensordict.stack([obs, obs]), Obs)
assert_type(tensordict.cat([obs, obs], dim=0), Obs)
assert_type(tensordict.lazy_stack([obs, obs]), Obs)
assert_type(tensordict.maybe_dense_stack([obs, obs]), Obs)
assert_type(tensordict.stack([td, td]), TD)
"""
    )
    stdout, stderr, status = api.run(["--config-file", str(config), str(valid)])
    assert status == 0, stdout + stderr


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
def test_memmap_save_path_source_static_types(tmp_path, monkeypatch):
    # #3268: with py.typed, mypy must accept a pathlib.Path in memmap,
    # memmap_, save, dumps and memmap_like, as the run time does. The
    # signatures are copied from _tensorcollection.pyi, tensorclass.pyi and
    # the _Serialization mixin of TensorDictBase into a package of their own.
    # Mypy also searches the working directory, which must not hold the
    # tensordict sources.
    from mypy import api

    monkeypatch.chdir(tmp_path)
    names = ("memmap", "memmap_", "save", "dumps", "memmap_like")

    def members(path, class_name):
        with open(_TENSORDICT_DIR / path, "r") as f:
            tree = ast.parse(f.read())
        (class_node,) = (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        nodes = [
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        assert nodes, (path, class_name)
        for node in nodes:
            node.body = [ast.Expr(ast.Constant(...))]
            node.decorator_list = []
        return textwrap.indent("\n".join(map(ast.unparse, nodes)), "    ")

    stubs = tmp_path / "stubs"
    package = stubs / "tensordict"
    package.mkdir(parents=True)
    imports = """from pathlib import Path
from typing import Self
"""
    package.joinpath("__init__.pyi").write_text("")
    package.joinpath("_tensorcollection.pyi").write_text(
        f"""{imports}
class TensorCollection:
{members("_tensorcollection.pyi", "TensorCollection")}
"""
    )
    package.joinpath("tensorclass.pyi").write_text(
        f"""{imports}
class TensorClass:
{members("tensorclass.pyi", "TensorClass")}
"""
    )
    package.joinpath("base.py").write_text(
        f"""# mypy: disable-error-code="empty-body"
from __future__ import annotations

{imports}
from tensordict._tensorcollection import TensorCollection

class TensorDictBase(TensorCollection):
{members("_base/serialization.py", "_Serialization")}
"""
    )

    config = tmp_path / "mypy.ini"
    config.write_text(
        f"""[mypy]
python_version = 3.11
show_error_codes = True
mypy_path = {stubs}
no_site_packages = True
ignore_missing_imports = True
"""
    )

    def user_errors(source):
        user = tmp_path / "user.py"
        user.write_text(source)
        stdout, stderr, status = api.run(["--config-file", str(config), str(user)])
        assert not stderr, stderr
        return [line for line in stdout.splitlines() if line.startswith("user.py:")]

    valid = """from pathlib import Path
from tensordict._tensorcollection import TensorCollection
from tensordict.base import TensorDictBase
from tensordict.tensorclass import TensorClass

def check(
    tc: TensorCollection, td: TensorDictBase, obs: TensorClass, path: Path
) -> None:
    tc.memmap(path)
    tc.memmap_(path)
    tc.save(path)
    tc.dumps(path)
    tc.memmap_like(path)
    td.memmap(path)
    td.memmap_(path)
    td.save(path)
    td.memmap_like(path)
    obs.memmap(path)
    obs.memmap_(path)
    obs.save(path)
    obs.dumps(path)
    obs.memmap_like(path)
    tc.memmap("out")
    tc.memmap()
"""
    assert user_errors(valid) == []

    invalid = """from tensordict._tensorcollection import TensorCollection

def check(tc: TensorCollection) -> None:
    tc.memmap(3)
"""
    errors = user_errors(invalid)
    assert len(errors) == 1, errors
    assert 'Argument 1 to "memmap" of "TensorCollection"' in errors[0]


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
def test_memmap_save_archive_static_types(tmp_path, monkeypatch):
    # #3314: with py.typed, mypy must accept the archive and compression
    # arguments of save, dumps and memmap, and the archive argument of
    # memmap_like, as the run time does. The signatures are copied from
    # _tensorcollection.pyi, tensorclass.pyi and the _Serialization mixin of
    # TensorDictBase into a package of their own. Mypy also searches the
    # working directory, which must not hold the tensordict sources.
    from mypy import api

    monkeypatch.chdir(tmp_path)
    names = ("save", "dumps", "memmap", "memmap_like")

    def members(path, class_name):
        with open(_TENSORDICT_DIR / path, "r") as f:
            tree = ast.parse(f.read())
        (class_node,) = (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        nodes = [
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        assert nodes, (path, class_name)
        for node in nodes:
            node.body = [ast.Expr(ast.Constant(...))]
            node.decorator_list = []
        return textwrap.indent("\n".join(map(ast.unparse, nodes)), "    ")

    stubs = tmp_path / "stubs"
    package = stubs / "tensordict"
    package.mkdir(parents=True)
    imports = """from pathlib import Path
from typing import Self
"""
    package.joinpath("__init__.pyi").write_text("")
    package.joinpath("_tensorcollection.pyi").write_text(
        f"""{imports}
class TensorCollection:
{members("_tensorcollection.pyi", "TensorCollection")}
"""
    )
    package.joinpath("tensorclass.pyi").write_text(
        f"""{imports}
class TensorClass:
{members("tensorclass.pyi", "TensorClass")}
"""
    )
    package.joinpath("base.py").write_text(
        f"""# mypy: disable-error-code="empty-body"
from __future__ import annotations

{imports}
from tensordict._tensorcollection import TensorCollection

class TensorDictBase(TensorCollection):
{members("_base/serialization.py", "_Serialization")}
"""
    )

    config = tmp_path / "mypy.ini"
    config.write_text(
        f"""[mypy]
python_version = 3.11
show_error_codes = True
mypy_path = {stubs}
no_site_packages = True
ignore_missing_imports = True
"""
    )

    def user_errors(source):
        user = tmp_path / "user.py"
        user.write_text(source)
        stdout, stderr, status = api.run(["--config-file", str(config), str(user)])
        assert not stderr, stderr
        return [line for line in stdout.splitlines() if line.startswith("user.py:")]

    calls = """
    {0}.save("x.tdz", archive=True)
    {0}.save("x.tdz", compression="deflated")
    {0}.save("x.tdz", archive=None, compression=None)
    {0}.dumps("x.tdz", archive=True)
    {0}.memmap("x.tdz", archive=True, compression="stored")
    {0}.memmap("x.tdz", archive=True, compression=8)
    {0}.memmap_like("y.tdz", archive=True)"""
    valid = f"""from tensordict._tensorcollection import TensorCollection
from tensordict.base import TensorDictBase
from tensordict.tensorclass import TensorClass

def check(tc: TensorCollection, td: TensorDictBase, obs: TensorClass) -> None:
{"".join(calls.format(name) for name in ("tc", "td", "obs"))}
"""
    assert user_errors(valid) == []

    invalid = """from tensordict._tensorcollection import TensorCollection
from tensordict.tensorclass import TensorClass

def check(tc: TensorCollection, obs: TensorClass) -> None:
    tc.save("x.tdz", compression=1.5)
    obs.memmap_like("y.tdz", compression="stored")
"""
    errors = [line for line in user_errors(invalid) if ": error:" in line]
    assert len(errors) == 2, errors
    assert 'Argument "compression" to "save" of "TensorCollection"' in errors[0]
    assert 'Unexpected keyword argument "compression" for "memmap_like"' in errors[1]


@pytest.mark.skipif(IS_FB, reason="not working on fbcode")
@pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
def test_tensordict_compare_and_split_source_static_types(tmp_path, monkeypatch):
    # #3269 and #3270: with py.typed, mypy must type td == td as a tensordict,
    # not as the bool of Mapping.__eq__, and the pieces of unbind, chunk and
    # split as TensorDictBase, not TensorCollection. Type checkers cannot
    # expand the starred mixin bases of TensorDictBase, so they see
    # MutableMapping, then the TensorCollection stub, and the TYPE_CHECKING
    # declarations of the class body. These are copied into a package of
    # their own. Mypy also searches the working directory, which must not hold
    # the tensordict sources.
    from mypy import api

    monkeypatch.chdir(tmp_path)
    names = ("__eq__", "__ne__", "unbind", "chunk", "split")

    def class_body(path, class_name):
        with open(_TENSORDICT_DIR / path, "r") as f:
            tree = ast.parse(f.read())
        (class_node,) = (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        return class_node.body

    stub_members = [
        node
        for node in class_body("_tensorcollection.pyi", "TensorCollection")
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(stub_members) == len(names)
    type_checking = [
        node
        for node in class_body("base.py", "TensorDictBase")
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"
    ]

    def indent(nodes):
        return textwrap.indent("\n".join(map(ast.unparse, nodes)) or "...", "    ")

    stubs = tmp_path / "stubs"
    package = stubs / "tensordict"
    package.mkdir(parents=True)
    package.joinpath("__init__.pyi").write_text("")
    package.joinpath("_tensorcollection.pyi").write_text(
        f"""from typing import Self

class TensorCollection:
{indent(stub_members)}
"""
    )
    # A module, not a stub, as in the sources.
    package.joinpath("base.py").write_text(
        f"""from __future__ import annotations

from collections.abc import MutableMapping
from typing import Self, TYPE_CHECKING

from tensordict._tensorcollection import TensorCollection

class TensorDictBase(MutableMapping, TensorCollection):
{indent(type_checking)}

class TensorDict(TensorDictBase): ...
"""
    )

    config = tmp_path / "mypy.ini"
    config.write_text(
        f"""[mypy]
python_version = 3.11
show_error_codes = True
mypy_path = {stubs}
no_site_packages = True
ignore_missing_imports = True
"""
    )

    user = tmp_path / "user.py"
    user.write_text(
        """from typing import assert_type
from tensordict.base import TensorDict, TensorDictBase

def check(td: TensorDict, base: TensorDictBase) -> None:
    assert_type(td == td, TensorDict)
    assert_type(td != td, TensorDict)
    assert_type(base == td, TensorDictBase)
    assert_type(td.unbind(0), tuple[TensorDictBase, ...])
    assert_type(td.chunk(2), tuple[TensorDictBase, ...])
    assert_type(td.split(2), tuple[TensorDictBase, ...])
    assert_type(base.split([1, 3], 0), tuple[TensorDictBase, ...])
    first: TensorDictBase = td.unbind(0)[0]
"""
    )
    stdout, stderr, status = api.run(["--config-file", str(config), str(user)])
    assert not stderr, stderr
    errors = [line for line in stdout.splitlines() if line.startswith("user.py:")]
    assert errors == [], stdout


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


_RAGGED_POST_INIT_CALLS = []


@tensorclass
class Ragged:
    offsets: torch.Tensor
    indices: torch.Tensor

    def __post_init__(self):
        _RAGGED_POST_INIT_CALLS.append(self.offsets)
        if self.offsets is None or int(self.offsets[0]) != 0:
            raise ValueError(f"offsets must start at 0, got {self.offsets}")


class _ShapedTensor:
    # Stands in for a jaxtyping annotation such as Float[Tensor, "*batch 3"]: a
    # class that names the array type it wraps in ``array_type`` and that
    # cannot be instantiated.
    array_type = torch.Tensor
    dim_str = "*batch 3"

    def __new__(cls, *args, **kwargs):
        raise RuntimeError("jaxtyping annotations cannot be instantiated")


class _ShapedArray(_ShapedTensor):
    array_type = np.ndarray


class _WithArrayType(TensorClass):
    # An attribute named ``array_type`` alone does not make a shape annotation.
    array_type: ClassVar[type] = np.ndarray
    x: torch.Tensor


_MESH_POST_INIT_CALLS = []


class _ValidatingMesh(TensorClass):
    # __post_init__ reads the shape of a leaf, which raises on a lazy stack of
    # meshes of different sizes and on a shell without fields.
    points: torch.Tensor
    cells: torch.Tensor
    mask: torch.Tensor = None

    def __post_init__(self):
        _MESH_POST_INIT_CALLS.append(self.points.shape)
        if self.points.shape[-1] != 3:
            raise ValueError(f"points must be (..., 3), got {tuple(self.points.shape)}")
        if self.mask is None:
            self.mask = torch.ones(self.points.shape[:-1], dtype=torch.bool)


_REBUILD_CALLS = []


class _Counted(TensorClass):
    # Module level, so that loading resolves these classes. __post_init__
    # records the class of each instance it runs on.
    offsets: torch.Tensor
    indices: torch.Tensor

    def __post_init__(self):
        _REBUILD_CALLS.append(type(self))


class _SkipCounted(_Counted["skip_post_init_on_rebuild"]):
    pass


class _CountedHolder(TensorClass):
    adj: _SkipCounted
    x: torch.Tensor

    def __post_init__(self):
        _REBUILD_CALLS.append(type(self))


# The ops that rebuild an instance from the data of an existing one
_REBUILD_OPS = {
    "getitem": lambda tc: tc[:2],
    "clone": lambda tc: tc.clone(),
    "to": lambda tc: tc.to(torch.float64),
    "apply": lambda tc: tc.apply(lambda t: t),
    "add": lambda tc: tc + 1,
    "detach": lambda tc: tc.detach(),
    "copy": lambda tc: tc.copy(),
    "select": lambda tc: tc.select(next(iter(tc.keys()))),
    "exclude": lambda tc: tc.exclude(next(iter(tc.keys()))),
    "stack": lambda tc: torch.stack([tc, tc]),
    "lazy_stack": lambda tc: lazy_stack([tc, tc]),
    "cat": lambda tc: torch.cat([tc, tc]),
    "stack_classmethod": lambda tc: type(tc).stack([tc, tc]),
    "lazy_stack_classmethod": lambda tc: type(tc).lazy_stack([tc, tc]),
    "cat_classmethod": lambda tc: type(tc).cat([tc, tc]),
    "unbind": lambda tc: tc.unbind(0),
    "consolidate": lambda tc: tc.consolidate(),
    "memmap": lambda tc: tc.memmap(),
}


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


@pytest.mark.parametrize("form", ["subclass", "decorator"])
def test_tensorclass_keeps_its_docstring(form):
    # A tensorclass keeps its own docstring (help(), IDEs and the API docs
    # read it); a class without one documents its constructor signature.
    if form == "subclass":

        class WithDoc(TensorClass):
            """Observations of one step."""

            obs: torch.Tensor

        class WithoutDoc(TensorClass):
            obs: torch.Tensor

    else:

        @tensorclass
        class WithDoc:
            """Observations of one step."""

            obs: torch.Tensor

        @tensorclass
        class WithoutDoc:
            obs: torch.Tensor

    assert WithDoc.__doc__ == "Observations of one step."
    assert WithoutDoc.__doc__.startswith("WithoutDoc(")
    assert "obs" in WithoutDoc.__doc__


@pytest.mark.parametrize("cls", [NonTensorData, MetaData])
def test_non_tensor_classes_keep_their_docstrings(cls):
    # Their docstrings used to be replaced by the generated signature.
    assert not cls.__doc__.startswith(cls.__name__ + "(")


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
        with pytest.raises(
            RuntimeError, match="batch dimension mismatch.* for key 'y'"
        ):
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
            with pytest.raises(
                AttributeError,
                match=re.escape(
                    "Cannot set the attribute 'extra': Data has no such field. "
                    "Its fields are ['x', 'y']."
                ),
            ):
                Data(x=x, y=y, extra=x)
        else:
            # Until TensorDict 0.17, a tensor_only class stores the argument
            # and warns.
            with pytest.warns(
                DeprecationWarning,
                match=re.escape(
                    "Passing arguments that are not fields (['extra']) to the "
                    "tensor_only tensorclass Data"
                ),
            ) as record:
                Data(x=x, y=y, extra=x)
            assert record[0].filename == __file__
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

        # a field that is not a tensor collection gets the default passed on,
        # as before #2678 (dict.get takes no keyword arguments)
        @tensorclass
        class Meta:
            meta: dict

        with pytest.raises(TypeError, match="keyword arguments"):
            Meta(meta={"k": 1}).get(("meta", "k"))

        data = data.expand(10)
        assert data.get_at("a", 0) is not None
        assert data.get_at("b", 0) is None
        assert data.get_at("b", 0, "else") == "else"

        assert data.get_at(("td", "missing"), 0, "else") == "else"
        assert data.get_at(("td", "missing"), 0) is None

    def test_get_through_tensor_field_warns(self):
        # get without a default returns None for a key that runs through a
        # tensor field, where TensorDict.get raises; it raises from 0.17 (#2979)
        @tensorclass
        class Inner:
            a: torch.Tensor

        @tensorclass
        class Data:
            x: torch.Tensor
            name: str
            opt: torch.Tensor | None = None
            inner: Inner = None
            td: TensorDict = None

        data = Data(
            x=torch.zeros(3),
            name="a",
            inner=Inner(a=torch.zeros(3), batch_size=[3]),
            td=TensorDict(t=torch.zeros(3), batch_size=[3]),
            batch_size=[3],
        )
        for key in [("x", "y"), ("x", "y", "z"), ("inner", "a", "y")]:
            with pytest.warns(
                FutureWarning, match="is a Tensor, not a tensordict.*TensorDict 0.17"
            ):
                assert data.get(key) is None
            with pytest.warns(FutureWarning, match="TensorDict 0.17"):
                assert data.get(key, as_list=True) is None
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            # an explicit default is returned, as TensorDict.get does
            assert data.get(("x", "y"), None) is None
            assert data.get(("x", "y"), default="D") == "D"
            assert data.get(("inner", "a", "y"), "D") == "D"
            # TensorDict.get returns None for these too
            assert data.get(("name", "y")) is None
            assert data.get(("opt", "y")) is None
            assert data.get(("z", "y")) is None
            assert data.get(("inner", "q")) is None
            assert data.get(("td", "q")) is None
            assert data.get("x") is data.x
        assert not [w for w in record if issubclass(w.category, FutureWarning)]
        # a TensorDict field raises already
        with pytest.raises(ValueError, match="'t' is a Tensor, not a tensordict"):
            data.get(("td", "t", "y"))

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

    @pytest.mark.parametrize(
        "op,args,kwargs",
        [
            ("squeeze", (0,), {}),
            ("squeeze", (-1,), {}),
            ("flatten", (0, None), {}),
            ("flatten", (), {"end_dim": None}),
            ("transpose", (), {"dim0": 0, "dim1": 1}),
            ("swapaxes", (), {"axis0": 0, "axis1": 1}),
        ],
    )
    def test_shape_op_context_manager(self, op, args, kwargs):
        # the shape op is undone on exit and the values written in the block stay
        @tensorclass
        class MyClass:
            X: torch.Tensor

        obj = MyClass(X=torch.zeros(2, 3), batch_size=[2, 3])
        with getattr(obj, op)(*args, **kwargs) as obj_op:
            obj_op.X += 1
        assert obj.batch_size == (2, 3)
        assert obj.X.shape == (2, 3)
        assert (obj.X == 1).all()

    @pytest.mark.parametrize(
        "shape,op,args",
        [((2, 3, 4), "flatten", (0, 1)), ((6, 4), "unflatten", (0, (2, 3)))],
    )
    def test_shape_op_context_manager_inplace(self, shape, op, args):
        # with inplace=True, the shape is restored on exit too (gh-2700)
        @tensorclass
        class MyClass:
            X: torch.Tensor
            y: TensorDict

        obj = MyClass(
            X=torch.zeros(shape), y=TensorDict(z=torch.zeros(shape)), batch_size=shape
        )
        with getattr(obj, op)(*args, inplace=True) as obj_op:
            assert obj_op is obj
            obj_op.X = obj_op.X + 1
        assert obj.batch_size == shape
        assert obj.X.shape == shape
        assert obj.y.batch_size == shape
        assert (obj.X == 1).all()

    def test_flatten_keys_context_manager_inplace(self):
        # with inplace=True, the nested keys are restored on exit too (gh-2700)
        @tensorclass
        class MyClass:
            X: torch.Tensor
            y: TensorDict

        obj = MyClass(X=torch.zeros(2), y=TensorDict(z=torch.zeros(2)), batch_size=[2])
        with obj.flatten_keys(inplace=True):
            pass
        assert set(obj._tensordict.keys(True, True)) == {"X", ("y", "z")}

    def test_lock_context_manager_restores_on_error(self):
        # lock_() and unlock_() undo their change also when the body raises
        @tensorclass
        class MyClass:
            X: torch.Tensor

        obj = MyClass(X=torch.zeros(2), batch_size=[])
        with pytest.raises(ValueError, match="error in the body"):
            with obj.lock_():
                raise ValueError("error in the body")
        assert not obj.is_locked
        obj.lock_()
        with pytest.raises(ValueError, match="error in the body"):
            with obj.unlock_():
                raise ValueError("error in the body")
        assert obj.is_locked
        assert not obj._tensordict._last_op_queue

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

    def test_bool(self):
        # bool() raises and names the replacements (#1939)
        myc = _make_data([3])
        with pytest.raises(
            RuntimeError,
            match=r"Converting a tensorclass to boolean value is not permitted. "
            r"Use len\(tc.keys\(\)\) to test for keys, tc.is_empty\(\) to test "
            r"for leaves, or \(a == b\).all\(\) / .any\(\) to reduce a comparison.",
        ):
            bool(myc)
        with pytest.raises(RuntimeError, match="is not permitted"):
            assert myc == myc

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

    def test_pack(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str = None
            y: "MyDataNested" = None

        data = MyDataNested(
            X=torch.randn(3, 2),
            y=MyDataNested(X=torch.randn(3, 4), batch_size=[3, 4]),
            batch_size=[3],
        )
        flat, layout = data.pack()
        assert flat.shape == (3, 6)
        assert layout.keys == ("X", ("y", "X"))
        assert layout.cls is MyDataNested
        out = layout.unpack(flat)
        assert type(out) is MyDataNested
        assert type(out.y) is MyDataNested
        assert out.y.batch_size == (3, 4)
        assert out.z is None
        assert out.y.y is None
        assert (out == data).all()
        assert out.X._base is flat

        data.z = "a string"
        with pytest.raises(TypeError, match="non-tensor entry 'z'"):
            data.pack()

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

    @pytest.mark.parametrize("inplace", [False, True])
    def test_consolidate_return_early(self, inplace):
        # #3149: the future gives the tensorclass back, as the eager call does
        data = MyData2(
            torch.ones((2,)), torch.ones((2, 3)) * 2, "a string!", batch_size=[2]
        )
        future = data.consolidate(return_early=True, num_threads=2, inplace=inplace)
        data_c = future.result()
        assert isinstance(data_c, MyData2)
        assert (data_c is data) is inplace
        assert data_c.z == "a string!"
        assert data_c.is_consolidated()
        torch.testing.assert_close(data_c.X, torch.ones((2,)))
        torch.testing.assert_close(data_c.y, torch.ones((2, 3)) * 2)
        assert future.result() is data_c

    def test_consolidate_return_early_post_init_after_writes(self):
        # #3149: the tensorclass is built when result() is called, after the
        # writes, so that __post_init__ sees the data
        calls = []

        @tensorclass
        class CheckedData:
            x: torch.Tensor

            def __post_init__(self):
                calls.append(self.x.clone())

        data = CheckedData(x=torch.ones(3), batch_size=[3])
        calls.clear()
        future = data.consolidate(num_threads=2, return_early=True)
        assert calls == []
        result = future.result()
        assert isinstance(result, CheckedData)
        assert len(calls) == 1
        torch.testing.assert_close(calls[0], torch.ones(3))

    def test_consolidate_nested_post_init(self):
        # consolidate must not run __post_init__ on an empty tensorclass while
        # it collects the leaves of a nested one
        calls = _RAGGED_POST_INIT_CALLS
        adj = Ragged(offsets=torch.tensor([0, 2, 3]), indices=torch.tensor([5, 6, 7]))
        calls.clear()
        adj.consolidate()
        assert len(calls) == 1

        td = TensorDict(adj=adj, x=torch.zeros(4))
        calls.clear()
        td_c = td.consolidate()
        assert len(calls) == 1
        assert isinstance(td_c["adj"], Ragged)
        torch.testing.assert_close(td_c["adj"].offsets, adj.offsets)
        torch.testing.assert_close(td_c["adj"].indices, adj.indices)
        torch.testing.assert_close(td_c["x"], td["x"])

        td_load = pickle.loads(pickle.dumps(td_c))
        assert isinstance(td_load["adj"], Ragged)
        torch.testing.assert_close(td_load["adj"].offsets, adj.offsets)
        torch.testing.assert_close(td_load["adj"].indices, adj.indices)
        torch.testing.assert_close(td_load["x"], td["x"])

    def test_consolidate_file_nested(self, tmp_path, monkeypatch):
        # A tensordict that holds a tensorclass can be written to a file and
        # read back with the nested tensorclass (#3117)
        data = MyData2(
            torch.ones((2,)), torch.ones((2, 3)) * 2, "a string!", batch_size=[2]
        )
        td = TensorDict(tc=data, x=torch.zeros(2), batch_size=[2])
        filename = tmp_path / "file.mmap"
        td_c = td.consolidate(filename=filename)
        assert isinstance(td_c["tc"], MyData2)
        # the in-memory metadata still holds the class itself
        assert td_c._consolidated["metadata"]["tc"]["cls"] is MyData2

        loaded = tensordict.from_consolidated(filename)
        assert loaded.is_consolidated()
        assert isinstance(loaded["tc"], MyData2)
        # the loaded metadata holds the class too, so pickles of it name it
        assert loaded._consolidated["metadata"]["tc"]["cls"] is MyData2
        assert loaded["tc"].z == "a string!"
        assert (loaded == td).all()
        loaded = pickle.loads(pickle.dumps(loaded))
        assert isinstance(loaded["tc"], MyData2)
        assert (loaded == td).all()

        # A tensorclass that is not defined where the file is read cannot be
        # found by name
        monkeypatch.setattr(
            tensordict.base,
            "_ACCEPTED_CLASSES",
            tuple(c for c in tensordict.base._ACCEPTED_CLASSES if c is not MyData2),
        )
        with pytest.raises(RuntimeError, match="Could not find the class .*MyData2"):
            tensordict.from_consolidated(filename)

    def test_dataclass(self):
        data = MyData(
            X=torch.ones(3, 4, 5),
            y=torch.zeros(3, 4, 5, dtype=torch.bool),
            z="test_tensorclass",
            batch_size=[3, 4],
        )
        assert dataclasses.is_dataclass(data)

    def test_dataclass_rebuild(self):
        # gh-3162: the dataclass and list branches of Lightning's
        # apply_to_collection, which its bf16-true and 64-true precision
        # plugins call on the batch, rebuild a tensorclass.
        def apply(data, fn):
            if isinstance(data, torch.Tensor):
                return fn(data)
            if dataclasses.is_dataclass(data) and not isinstance(data, type):
                values = {
                    field.name: getattr(data, field.name)
                    for field in dataclasses.fields(data)
                }
                memo = {id(value): value for value in values.values()}
                result = copy.deepcopy(data, memo=memo)
                for name, value in values.items():
                    setattr(result, name, apply(value, fn))
                for name in vars(type(data)):
                    getattr(type(data), name)
                return result
            if isinstance(data, list):
                return type(data)(apply(item, fn) for item in data)
            return data

        class Inner(TensorClass):
            a: torch.Tensor

        class Batch(TensorClass):
            x: torch.Tensor
            inner: Inner
            name: str

        batch = torch.stack(
            [
                Batch(x=torch.randn(3), inner=Inner(a=torch.randn(2)), name=f"b{i}")
                for i in range(4)
            ]
        )
        assert isinstance(batch.name, tensordict.utils.LinkedList)
        out = apply(batch, lambda tensor: tensor.double())
        assert type(out) is Batch
        assert type(out.inner) is Inner
        assert out.batch_size == (4,)
        assert out.x.dtype == torch.float64
        assert out.inner.a.dtype == torch.float64
        assert out.name == ["b0", "b1", "b2", "b3"]
        torch.testing.assert_close(out.x, batch.x.double())
        assert batch.x.dtype == torch.float32

    def test_default(self):
        with pytest.warns(FutureWarning, match="MyData.y is shared"):

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

    def test_tensor_default_warns(self):
        # A tensor default is shared by every instance; 0.17 copies it (#3095)
        def messages(record):
            return [
                str(w.message)
                for w in record
                if "Starting with TensorDict 0.17" in str(w.message)
            ]

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")

            class Sub(TensorClass):
                step: torch.Tensor = torch.tensor(0)

            @tensorclass
            class Deco:
                weight: torch.Tensor = torch.nn.Parameter(torch.zeros(()))

            class Child(Sub):  # the inherited field warned already
                x: torch.Tensor = None

        found = messages(record)
        assert len(found) == 2, found
        assert found[0].startswith("The Tensor default of the field Sub.step ")
        assert found[1].startswith("The Parameter default of the field Deco.weight ")
        assert "default_factory=lambda: torch.tensor(0)" in found[0]
        (sub_warning,) = (w for w in record if str(w.message) == found[0])
        assert sub_warning.category is FutureWarning
        assert sub_warning.filename == __file__
        a, b = Sub(), Sub()
        a.step.add_(1)
        assert b.step == 1  # unchanged until 0.17

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")

            class Silent(TensorClass):
                a: torch.Tensor = None
                b: int = 1
                c: str = "c"
                d: torch.Tensor = dataclasses.field(
                    default_factory=lambda: torch.tensor(0)
                )
                e: ClassVar[torch.Tensor] = torch.tensor(0)

            @dataclasses.dataclass
            class Cfg:
                x: torch.Tensor = dataclasses.field(default=torch.zeros(2))

            Silent()
            NonTensorData("a")
            MetaData("a")
            # an instance's values are deep-copied: nothing changes in 0.17
            from_dataclass(Cfg())
            TensorDict.from_dataclass(Cfg(), as_tensorclass=True)
        # only Silent() casting b to a 0-d tensor warns (#1199), not a default
        assert [m.split(" is stored")[0] for m in messages(record)] == [
            "The int given to the field 'b' of Silent"
        ]

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

    def test_ignored_field_options_warn(self):
        # Tensorclasses ignore these field options: the class warns once for
        # each, at the class statement (#3073).
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")

            class MyData(TensorClass):
                x: torch.Tensor
                y: torch.Tensor = dataclasses.field(default=None, init=False)
                z: torch.Tensor = dataclasses.field(
                    default=None, repr=False, compare=False
                )
                w: torch.Tensor = dataclasses.field(default=None, kw_only=True)

            # A subclass does not warn again for the inherited fields
            class MyChild(MyData):
                v: torch.Tensor = None

        record = [w for w in record if "tensorclasses ignore" in str(w.message)]
        assert all(w.category is UserWarning for w in record)
        assert all(w.filename == __file__ for w in record)
        messages = [str(w.message) for w in record]
        assert sorted(messages) == sorted(
            f"MyData.{name}: tensorclasses ignore the field option {option}"
            for name, option in [
                ("y", "init=False: __init__ still takes the field as an argument."),
                ("z", "compare=False: == and != still compare the field."),
                (
                    "w",
                    "kw_only=True: __init__ still takes the field as a positional "
                    "argument, and warns when it does.",
                ),
            ]
        )

        # No warning where the class's own methods replace the ones that
        # ignore the options, or where the options take their default.
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")

            class MyOwnMethods(TensorClass):
                x: torch.Tensor = dataclasses.field(
                    default=None, init=False, repr=False, compare=False, kw_only=True
                )

                def __init__(self, x):
                    self.x = x

                def __repr__(self):
                    return "MyOwnMethods"

                def __eq__(self, other):
                    return True

                def __ne__(self, other):
                    return False

            class MyDefaults(TensorClass):
                x: torch.Tensor = dataclasses.field(
                    default=None, init=True, repr=True, compare=True, kw_only=False
                )

            # Nor for the options that tensorclasses honour: repr=False, and
            # kw_only=True with the class option kw_only or with InitVars
            class MyHonoured(TensorClass, kw_only=True):
                x: torch.Tensor = dataclasses.field(default=None, repr=False)
                y: torch.Tensor = dataclasses.field(default=None, kw_only=True)

            class MyInitVar(TensorClass):
                x: torch.Tensor = dataclasses.field(default=None, kw_only=True)
                s: dataclasses.InitVar[float] = 1.0

                def __post_init__(self, s):
                    pass

        assert not [w for w in record if "tensorclasses ignore" in str(w.message)]

    @pytest.mark.parametrize("style", ["subclass", "decorator"])
    def test_positional_args_signature(self, style):
        # Positional arguments go to the fields in order, keyword-only and
        # init=False fields included. A call that this order makes raise
        # follows the signature, and a call that the signature maps
        # differently warns: 0.17 makes the signature the rule (#3100).
        def make(name, annotations, namespace, base=None):
            namespace = {"__annotations__": annotations, **namespace}
            if style == "subclass":
                return type(name, (base or TensorClass,), namespace)
            return tensorclass(type(name, (base,) if base else (), namespace))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*tensorclasses ignore", UserWarning)
            K = make(
                "K",
                {"a": torch.Tensor, "b": torch.Tensor},
                {"a": dataclasses.field(kw_only=True)},
            )
            F = make(
                "F",
                {"a": torch.Tensor, "b": torch.Tensor},
                {"a": dataclasses.field(default=None, init=False), "b": None},
            )
            L = make(
                "L",
                {"a": torch.Tensor, "b": torch.Tensor},
                {"b": dataclasses.field(default=None, kw_only=True)},
            )
        KC = make("KC", {"c": torch.Tensor}, {"c": None}, base=K)
        t0, t1 = torch.zeros(2), torch.ones(2)
        assert list(inspect.signature(K).parameters)[:2] == ["b", "a"]

        def check(data, **expected):
            for name, value in expected.items():
                if value is None:
                    assert getattr(data, name) is None, name
                else:
                    assert (getattr(data, name) == value).all(), name

        # calls that raised and that the signature allows follow it
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check(K(t1, a=t0, batch_size=[2]), a=0, b=1)
            check(F(t1, a=t0, batch_size=[2]), a=0, b=1)
            check(KC(t1, a=t0, batch_size=[2]), a=0, b=1, c=None)
            check(KC(t1, t0, a=t0, batch_size=[2]), a=0, b=1, c=0)
            # calls that map the same either way do not warn
            check(K(b=t1, a=t0, batch_size=[2]), a=0, b=1)
            check(L(t1, batch_size=[2]), a=1, b=None)

        # calls that the signature maps differently keep their result and warn
        for cls, args, expected, moved in [
            (K, (t0, t1), {"a": 0, "b": 1}, ["a", "b"]),
            (F, (t1,), {"a": 1, "b": None}, ["a"]),
            (F, (t1, t0), {"a": 1, "b": 0}, ["a", "b"]),
            (L, (t1, t0), {"a": 1, "b": 0}, ["b"]),
            (KC, (t0, t1, t1), {"a": 0, "b": 1, "c": 1}, ["a", "b", "c"]),
        ]:
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                check(cls(*args, batch_size=[2]), **expected)
            (warning,) = record
            assert warning.category is FutureWarning
            assert warning.filename == __file__
            message = str(warning.message)
            name = cls.__name__
            assert message.startswith(
                f"{name}() assigns positional arguments to the fields in the "
                "order they are declared"
            )
            assert f"this call sets {', '.join(map(repr, moved))} by position" in (
                message
            )
            assert "Starting with TensorDict 0.17, the default will change" in message
            assert message.endswith(
                f"pass these fields by keyword: "
                f"{name}({', '.join(f'{key}=...' for key in moved)})."
            )
        # calls that raise either way raise as before, without a warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(TypeError, match="missing 1 required positional"):
                K(t1, batch_size=[2])
            with pytest.raises(TypeError, match="3 positional arguments but 4"):
                K(t0, t1, t1, batch_size=[2])

        # A class with a custom __setattr__, whose dataclass __init__ rejects
        # the init=False field
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*tensorclasses ignore", UserWarning)

            def __setattr__(self, key, value):
                super(type(self), self).__setattr__(key, value)

            S = make(
                "S",
                {"a": torch.Tensor, "b": torch.Tensor},
                {
                    "a": dataclasses.field(default=None, init=False),
                    "b": None,
                    "__setattr__": __setattr__,
                },
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check(S(t1, batch_size=[2]), a=None, b=1)

        # InitVars are positional arguments of the signature
        scales = []

        def __post_init__(self, scale=2.0):
            scales.append(scale)

        A = make(
            "A",
            {"x": torch.Tensor, "s": dataclasses.InitVar[float], "y": torch.Tensor},
            {"s": 1.0, "y": None, "__post_init__": __post_init__},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            data = A(t1, 3.0, t0, batch_size=[2])
        check(data, x=1, y=0)
        assert scales == [3.0]
        with pytest.warns(FutureWarning, match="this call sets 'y' by position"):
            check(A(t1, t0, batch_size=[2]), x=1, y=0)

    @pytest.mark.parametrize("style", ["subclass", "decorator"])
    def test_class_kw_only(self, style):
        # The class option kw_only makes every field keyword-only (#3103)
        if style == "subclass":

            class MyData(TensorClass, kw_only=True):
                x: torch.Tensor
                y: torch.Tensor = None

        else:

            @tensorclass(kw_only=True)
            class MyData:
                x: torch.Tensor
                y: torch.Tensor = None

        params = inspect.signature(MyData).parameters
        assert params["x"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["y"].kind is inspect.Parameter.KEYWORD_ONLY
        assert all(field.kw_only for field in dataclasses.fields(MyData))
        with pytest.raises(TypeError, match="positional arguments but 2 were given"):
            MyData(torch.zeros(3), batch_size=[3])
        data = MyData(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
        assert (data.y == 1).all()
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            MyData(y=torch.ones(3), batch_size=[3])

        # As in dataclasses, the option applies to the fields of the class
        # that sets it: those of a subclass take positional arguments, which
        # skip the keyword-only fields of the parent.
        class MyChild(MyData):
            z: torch.Tensor = None

        if style == "decorator":
            MyChild = tensorclass(MyChild)
        child = MyChild(torch.ones(3), x=torch.zeros(3), batch_size=[3])
        assert (child.z == 1).all()
        assert (child.x == 0).all()
        assert child.y is None
        with pytest.raises(TypeError, match="positional arguments but 3 were given"):
            MyChild(torch.ones(3), torch.ones(3), x=torch.zeros(3))
        assert (child.clone().z == 1).all()

        # A subclass that is not a tensorclass itself maps as its parent
        if style == "decorator":

            class MyPlain(MyData):
                pass

            with pytest.raises(
                TypeError, match="positional arguments but 2 were given"
            ):
                MyPlain(torch.zeros(3), batch_size=[3])
            assert (MyPlain(x=torch.ones(3), batch_size=[3]).x == 1).all()

    @pytest.mark.parametrize("style", ["subclass", "decorator"])
    def test_init_var(self, style):
        # __init__ passes an InitVar to __post_init__, as dataclasses does,
        # and the other calls to __post_init__ pass its default (#3103)
        def __post_init__(self, scale):
            self.x = self.x * scale

        annotations = {"x": torch.Tensor, "scale": dataclasses.InitVar[float]}
        namespace = {
            "__annotations__": annotations,
            "scale": 1.0,
            "__post_init__": __post_init__,
        }
        if style == "subclass":
            MyInitVarData = type("MyInitVarData", (TensorClass,), namespace)
        else:
            MyInitVarData = tensorclass(type("MyInitVarData", (), namespace))

        assert [field.name for field in dataclasses.fields(MyInitVarData)] == ["x"]
        assert list(inspect.signature(MyInitVarData).parameters)[:2] == ["x", "scale"]
        ones = torch.ones(3)
        torch.testing.assert_close(
            MyInitVarData(x=ones, scale=2.0, batch_size=[3]).x, ones * 2
        )
        torch.testing.assert_close(MyInitVarData(ones, 3.0, batch_size=[3]).x, ones * 3)
        torch.testing.assert_close(MyInitVarData(x=ones, batch_size=[3]).x, ones)
        with pytest.raises(TypeError, match="takes 3 positional arguments"):
            MyInitVarData(ones, 2.0, ones)
        data = MyInitVarData(x=ones, scale=2.0, batch_size=[3])
        assert "scale" not in data.to_dict()
        # The rebuilds pass the default, 1.0
        torch.testing.assert_close(data.clone().x, ones * 2)
        torch.testing.assert_close(data[0].x, torch.tensor(2.0))
        torch.testing.assert_close(torch.stack([data, data]).x, ones.expand(2, 3) * 2)
        lazy = lazy_stack([data, data])
        assert isinstance(lazy._tensordict, LazyStackedTensorDict)
        torch.testing.assert_close(lazy.x, ones.expand(2, 3) * 2)
        torch.testing.assert_close(
            MyInitVarData.from_tensordict(TensorDict(x=ones, batch_size=[3])).x, ones
        )

        # A subclass inherits the InitVar and the __post_init__ that takes it
        class MyChild(MyInitVarData):
            y: torch.Tensor = None

        if style == "decorator":
            MyChild = tensorclass(MyChild)
        child = MyChild(ones, 5.0, ones, batch_size=[3])
        torch.testing.assert_close(child.x, ones * 5)
        torch.testing.assert_close(child.y, ones)

        # So does a subclass that is not a tensorclass itself
        if style == "decorator":

            class MyPlain(MyInitVarData):
                pass

            torch.testing.assert_close(MyPlain(ones, 3.0, batch_size=[3]).x, ones * 3)
            torch.testing.assert_close(
                MyPlain(x=ones, scale=3.0, batch_size=[3]).x, ones * 3
            )
            torch.testing.assert_close(MyPlain(x=ones, batch_size=[3]).x, ones)

            # One that replaces __post_init__ with one that takes no InitVar
            # keeps the behaviour from before #3103
            class MyOverride(MyInitVarData):
                def __post_init__(self):
                    self.x = self.x + 10

            data = MyOverride(x=ones, batch_size=[3])
            torch.testing.assert_close(data.x, ones + 10)
            torch.testing.assert_close(MyOverride(ones, batch_size=[3]).x, ones + 10)
            torch.testing.assert_close(data.clone().x, ones + 20)
            torch.testing.assert_close(data[0].x, torch.tensor(21.0))
            other = MyOverride(x=ones, batch_size=[3])
            torch.testing.assert_close(
                lazy_stack([data, other]).x, ones.expand(2, 3) + 20
            )
            torch.testing.assert_close(
                MyOverride.from_dict({"x": ones}, batch_size=[3]).x, ones + 10
            )

    def test_init_var_required_and_rebuild(self, tmp_path):
        @tensorclass(skip_post_init_on_rebuild=True)
        class MyData:
            x: torch.Tensor
            shift: dataclasses.InitVar[float]

            def __post_init__(self, shift):
                self.x = self.x + shift

        zeros = torch.zeros(3)
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            MyData(x=zeros, batch_size=[3])
        with pytest.raises(TypeError, match="got an unexpected keyword argument 'y'"):
            MyData(y=zeros, shift=1.0)
        data = MyData(x=zeros, shift=1.0, batch_size=[3])
        # Without __post_init__ on rebuilds, a required InitVar is not needed
        torch.testing.assert_close(data.clone().x, zeros + 1)
        torch.testing.assert_close(data[1:].x, zeros[1:] + 1)
        data.memmap(tmp_path)
        # Loads and from_tensordict run __post_init__, which then needs a
        # default for the InitVar
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            MyData.load_memmap(tmp_path)
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            MyData.from_tensordict(TensorDict(x=zeros, batch_size=[3]))

    def test_init_var_not_taken(self):
        # A __post_init__ that does not take the InitVar: as before #3103, the
        # InitVar is not accepted
        class MyData(TensorClass):
            x: torch.Tensor
            scale: dataclasses.InitVar[float] = 1.0

            def __post_init__(self):
                pass

        MyData(x=torch.ones(3), batch_size=[3])
        with pytest.raises(AttributeError, match="has no such field"):
            MyData(x=torch.ones(3), scale=2.0, batch_size=[3])

    def test_init_var_no_post_init(self):
        # Without __post_init__, the InitVar is taken by name and dropped, as
        # in dataclasses. Positional arguments still go to the fields in
        # order, as before #3103.
        class MyNoPostInit(TensorClass):
            x: torch.Tensor
            scale: dataclasses.InitVar[float] = 1.0
            y: torch.Tensor = None

        ones, twos = torch.ones(3), torch.full((3,), 2.0)
        data = MyNoPostInit(x=ones, scale=2.0, batch_size=[3])
        torch.testing.assert_close(data.x, ones)
        assert data.y is None
        assert "scale" not in data.to_dict()
        # the signature would pass twos to scale (#3100)
        with pytest.warns(FutureWarning, match="this call sets 'y' by position"):
            data = MyNoPostInit(ones, twos, batch_size=[3])
        torch.testing.assert_close(data.x, ones)
        torch.testing.assert_close(data.y, twos)

    def test_init_var_post_init_default(self):
        # A __post_init__ that gives the InitVar a default of its own gets
        # the InitVar only when it is given, so the calls that worked before
        # #3103 behave as they did.
        class MyData(TensorClass):
            x: torch.Tensor
            scale: dataclasses.InitVar[float] = 1.0
            y: torch.Tensor = None

            def __post_init__(self, scale=5.0):
                self.x = self.x * scale

        ones, twos = torch.ones(3), torch.full((3,), 2.0)
        data = MyData(x=ones, batch_size=[3])
        torch.testing.assert_close(data.x, ones * 5)
        torch.testing.assert_close(data.clone().x, ones * 25)
        # the signature would pass twos to scale (#3100)
        with pytest.warns(FutureWarning, match="this call sets 'y' by position"):
            data = MyData(ones, twos, batch_size=[3])
        torch.testing.assert_close(data.x, ones * 5)
        torch.testing.assert_close(data.y, twos)
        torch.testing.assert_close(
            MyData(x=ones, scale=3.0, batch_size=[3]).x, ones * 3
        )

    def test_init_var_custom_setattr(self):
        # With a custom __setattr__, the dataclass __init__ passes the InitVar
        # to __post_init__, and positional arguments go to the fields in
        # order, as before #3103
        class MyData(TensorClass):
            x: torch.Tensor
            scale: dataclasses.InitVar[float] = 1.0
            y: torch.Tensor = None

            def __post_init__(self, scale):
                self.x = self.x * scale

            def __setattr__(self, key, value):
                super().__setattr__(key, value)

        ones, twos = torch.ones(3), torch.full((3,), 2.0)
        # the signature would pass twos to scale (#3100)
        with pytest.warns(FutureWarning, match="this call sets 'y' by position"):
            data = MyData(ones, twos, batch_size=[3])
        torch.testing.assert_close(data.x, ones)
        torch.testing.assert_close(data.y, twos)
        torch.testing.assert_close(
            MyData(x=ones, scale=3.0, batch_size=[3]).x, ones * 3
        )

    @pytest.mark.parametrize("inplace", [False, True])
    def test_init_var_from_dataclass(self, inplace):
        def __post_init__(self, scale):
            self.x = self.x * scale

        # Annotations that are not strings: Python 3.10 cannot resolve the
        # string "dataclasses.InitVar[float]" with typing.get_type_hints
        MyDataClass = dataclasses.make_dataclass(
            "MyDataClass",
            [
                ("x", torch.Tensor),
                ("scale", dataclasses.InitVar[float], dataclasses.field(default=1.0)),
            ],
            namespace={"__post_init__": __post_init__},
        )

        ones = torch.ones(3)
        # from an instance: the fields already hold the result of __post_init__
        torch.testing.assert_close(
            from_dataclass(MyDataClass(ones, 2.0), batch_size=[3]).x, ones * 2
        )
        # from the type: the class takes the InitVar. With inplace=False, the
        # new class does not have the methods of the dataclass, so no
        # __post_init__ runs.
        cls = from_dataclass(MyDataClass, inplace=inplace)
        expected = ones * 3 if inplace else ones
        torch.testing.assert_close(cls(x=ones, scale=3.0, batch_size=[3]).x, expected)

    def test_init_var_from_dataclass_positional(self):
        # The class from_dataclass makes has no __post_init__: positional
        # arguments go to the fields in order, as before #3103
        MyDataClass = dataclasses.make_dataclass(
            "MyDataClass",
            [
                ("x", torch.Tensor),
                ("scale", dataclasses.InitVar[float], dataclasses.field(default=1.0)),
                ("y", torch.Tensor, dataclasses.field(default=None)),
            ],
        )
        cls = from_dataclass(MyDataClass)
        ones, twos = torch.ones(3), torch.full((3,), 2.0)
        # the signature would pass twos to scale (#3100)
        with pytest.warns(FutureWarning, match="this call sets 'y' by position"):
            data = cls(ones, twos, batch_size=[3])
        torch.testing.assert_close(data.x, ones)
        torch.testing.assert_close(data.y, twos)
        torch.testing.assert_close(cls(x=ones, scale=3.0, batch_size=[3]).x, ones)

    def test_field_repr_false(self):
        # field(repr=False) leaves the field out of the reprs (#3103)
        class MyData(TensorClass):
            x: torch.Tensor
            hidden: torch.Tensor = dataclasses.field(default=None, repr=False)
            label: str = dataclasses.field(default="secret", repr=False)
            unset: torch.Tensor = dataclasses.field(default=None, repr=False)

        data = MyData(x=torch.zeros(3), hidden=torch.ones(3), batch_size=[3])
        nested = TensorDict(data=data, batch_size=[3])
        for obj in (data, nested, lazy_stack([data, data])):
            with tensordict.set_printoptions(compact=False):
                text = repr(obj)
            assert "x=" in text
            assert "hidden" not in text
            assert "secret" not in text
            assert "label" not in text
            assert "unset" not in text
            # the compact repr is the default
            text = repr(obj)
            assert "x: " in text
            assert "secret" not in text
            assert "hidden" not in text
            assert "label" not in text
            assert "unset" not in text
            html = obj._repr_html_()
            assert "x: " in html
            assert "hidden" not in html
            assert "label" not in html
        # The field is still there
        assert (data.hidden == 1).all()
        assert data.label == "secret"

    def test_del_returns_self(self):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            y: torch.Tensor = None
            td: TensorDict = None

        data = MyClass(
            x=torch.zeros(3),
            td=TensorDict(a=torch.zeros(3), b=torch.zeros(3), batch_size=[3]),
            batch_size=[3],
        )
        # A tensor field, a field that is None and a nested key
        assert data.del_("x") is data
        assert "x" not in data.keys()
        assert data.del_("y") is data
        assert data.y is None
        assert data.del_(("td", "a")) is data
        assert list(data.td.keys()) == ["b"]

    def test_rename_key_inside_old_key(self):
        # a new key inside the old one moves the field there: the field was
        # deleted with everything below it
        @tensorclass
        class Inner:
            x: torch.Tensor

        @tensorclass
        class MyClass:
            td: TensorDict

        inner = Inner(x=torch.ones(3), batch_size=[3])
        data = MyClass(
            td=TensorDict(a=torch.zeros(3), c=inner, batch_size=[3]), batch_size=[3]
        )
        assert data.rename_key_("td", ("td", "td")) is data
        assert set(data.keys(True, True)) == {("td", "td", "a"), ("td", "td", "c", "x")}
        assert (data.td["td", "a"] == 0).all()
        assert data.td["td", "c"] is inner

    def test_rename_keys(self):
        @tensorclass
        class Inner:
            x: torch.Tensor
            y: torch.Tensor = None

        @tensorclass
        class MyClass:
            a: torch.Tensor
            b: torch.Tensor
            raw: TensorDict = None
            inner: Inner = None
            c: torch.Tensor = None

        data = MyClass(
            a=torch.zeros(3),
            b=torch.ones(3),
            raw=TensorDict(p=torch.full((3,), 2.0), batch_size=[3]),
            inner=Inner(x=torch.full((3,), 3.0), batch_size=[3]),
            batch_size=[3],
        )
        out = data.rename_keys({"a": "b", "b": "a"})
        assert type(out) is MyClass
        assert out.a is data.b
        assert out.b is data.a
        assert data.a.sum() == 0
        # a field renamed away reads as None
        out = data.rename_keys({"a": "c"})
        assert out.a is None
        assert out.c is data.a
        # fields and nested tensorclasses that the move empties are kept
        out = data.rename_keys(
            {("raw", "p"): "c", ("inner", "x"): "a", "a": "b", "b": ("raw", "q")}
        )
        assert out.c is data.raw["p"]
        assert out.a is data.inner.x
        assert out.b is data.a
        assert out.raw["q"] is data.b
        out = data.rename_keys({("raw", "p"): "c", ("inner", "x"): ("inner", "y")})
        assert isinstance(out.raw, TensorDict)
        assert out.raw.is_empty()
        assert type(out.inner) is Inner
        assert out.inner.x is None
        assert out.inner.y is data.inner.x
        assert data.raw["p"] is not None
        assert data.inner.y is None
        # a destination that is not a field
        with pytest.raises(AttributeError, match="not fields of MyClass"):
            data.rename_keys({"a": "z"})
        # a tensorclass nested in a TensorDict
        td = TensorDict(data=data, batch_size=[3])
        out = td.rename_keys({("data", "raw", "p"): "p"})
        assert type(out["data"]) is MyClass
        assert out["data"].raw.is_empty()
        assert out["p"] is data.raw["p"]

    def test_deleted_field_reads_as_none(self):
        # #2147: a deleted field reads as None, so hasattr() and
        # getattr(obj, name, default) work after a delete instead of raising
        # KeyError.
        @tensorclass
        class MyClass:
            x: torch.Tensor
            y: torch.Tensor = None

        data = MyClass(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
        del data.y
        assert data.y is None
        assert getattr(data, "y", "default") is None
        assert hasattr(data, "y")
        assert "y" not in data.keys()
        assert data.get("y") is None
        assert data.get("y", "default") is None
        assert data.get_at("y", 0, "default") is None

        # Deleting a field that is None is not an error
        unset = MyClass(x=torch.zeros(3), batch_size=[3])
        del unset.y
        assert unset.y is None
        assert "y" not in unset.keys()
        # ... unless the instance is locked
        locked = MyClass(x=torch.zeros(3), batch_size=[3]).lock_()
        with pytest.raises(RuntimeError, match="locked"):
            del locked.y
        assert "y" in locked.to_dict()

        # del_ and pop remove the field, which reads as None too
        cleared = MyClass(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
        assert cleared.del_("y") is cleared
        assert cleared.y is None
        popped = MyClass(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
        torch.testing.assert_close(popped.pop("y"), torch.ones(3))
        assert popped.y is None

        # A key that no field declares keeps the attribute protocol
        with pytest.raises(AttributeError):
            data.not_a_field
        # A deleted field is a None field: deleting it again is not an error
        del unset.y
        assert unset.y is None

    def test_deleted_field_reads_as_none_shadow(self):
        # #2147, for the __getattr__ of a shadow class
        @tensorclass(shadow=True)
        class Shadow:
            x: torch.Tensor
            y: torch.Tensor = None

        shadow = Shadow(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
        del shadow.y
        assert shadow.y is None
        assert getattr(shadow, "y", "default") is None

    @pytest.mark.parametrize("shadow", [False, True])
    @set_printoptions(compact=False)
    def test_none_field_is_missing_key(self, shadow):
        # #2210: a None field is a key missing from the TensorDict, and
        # _non_tensordict holds no placeholder for it
        @tensorclass(shadow=shadow)
        class Data:
            x: torch.Tensor
            y: torch.Tensor | None = None

        @tensorclass(shadow=shadow)
        class Custom:
            x: torch.Tensor
            y: torch.Tensor | None = None

            def __init__(self, x):
                self.x = x
                self.y = None

        def check(obj, key="y"):
            assert getattr(obj, key) is None
            assert key not in obj._tensordict.keys()
            assert key not in obj._non_tensordict

        data = Data(x=torch.zeros(3), batch_size=[3])
        full = Data(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
        cleared = full.clone()
        cleared.y = None
        restored = Data.__new__(Data)
        # pickled by a version that kept a placeholder
        restored.__setstate__(
            {"tensordict": data._tensordict.clone(), "non_tensordict": {"y": None}}
        )
        for result in (
            data,
            cleared,
            restored,
            Custom(torch.zeros(3)),
            data.clone(),
            data[0],
            torch.stack([data, data]),
            data.apply(lambda t: t + 1),
            Data.from_dict({"x": torch.zeros(3), "y": None}, batch_size=[3]),
            Data.from_tensordict(TensorDict(x=torch.zeros(3), batch_size=[3])),
        ):
            check(result)
        # The outputs list the None field, as they did with a placeholder
        assert data.to_dict()["y"] is None
        assert "y=None" in repr(data)
        assert data.non_tensor_items() == (("y", None),)
        assert data.state_dict()._metadata[""]["_non_tensor"] == {"y": None}
        assert data.__getstate__()["non_tensordict"] == {"y": None}

        # #2230: lazy_stack leaves its inputs as they are
        stack = lazy_stack([full, data])
        assert stack[1].y is None
        assert full._non_tensordict == {}
        assert isinstance(full.to_dict()["y"], torch.Tensor)
        check(data)

        # #2171: writing the field through the TensorDict leaves no placeholder
        for write in (
            lambda d: d.set_non_tensor("y", "v"),
            lambda d: d.setdefault("y", torch.ones(3)),
            lambda d: d.rename_key_("x", "y"),
        ):
            d = Data(x=torch.zeros(3), batch_size=[3])
            write(d)
            assert d.to_dict()["y"] is not None
            assert repr(d).count("y=") == 1
        # the renamed field is None
        check(d, "x")
        assert d.to_dict()["x"] is None

    @pytest.mark.parametrize("kind", ["regular", "tensor_only", "shadow"])
    def test_lazy_stack_field_set_on_some_members(self, kind):
        # #2526: a field that only some members of a lazy stack set reads as
        # the list of the members' values, and setting it to None clears it
        # on every member
        @tensorclass
        class Inner:
            a: torch.Tensor

        @tensorclass(tensor_only=kind == "tensor_only", shadow=kind == "shadow")
        class Data:
            x: torch.Tensor
            y: torch.Tensor | None = None
            inner: Inner | None = None

        def members():
            p = Data(
                x=torch.zeros(3),
                y=torch.ones(3),
                inner=Inner(a=torch.ones(3), batch_size=[3]),
                batch_size=[3],
            )
            return p, Data(x=torch.zeros(3), batch_size=[3])

        p, q = members()
        stack = lazy_stack([p, q])
        for value in (stack.y, stack.get("y")):
            assert isinstance(value, list) and len(value) == 2
            assert (value[0] == 1).all() and value[1] is None
        inner = stack.inner
        assert isinstance(inner[0], Inner) and inner[1] is None
        # a field that every member sets, or none, reads as before
        assert stack.x.shape == (2, 3)
        assert lazy_stack([q, q.clone()]).y is None
        # a stack of stacks reads each member as it reads on its own
        nested = lazy_stack([stack, lazy_stack([p.clone(), p.clone()])])
        assert nested.y[0][1] is None and nested.y[1].shape == (2, 3)

        for clear in (
            lambda s: setattr(s, "y", None),
            lambda s: s.set("y", None),
        ):
            p, q = members()
            r, _ = members()
            stack = lazy_stack([lazy_stack([p, r]), lazy_stack([q, q.clone()])])
            clear(stack)
            assert stack.y is None
            assert p.y is None and r.y is None
            assert stack[0].y is None and stack[0, 0].y is None
            assert "y" not in p._tensordict.keys()
            assert stack.x.shape == (2, 2, 3)
        stack.inner = None
        assert p.inner is None and stack.inner is None

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

    def test_is_meta(self):
        data = MyData(
            X=torch.ones(3, 4, 5),
            y=torch.zeros(3, 4, 5, dtype=torch.bool),
            z="test_tensorclass",
            batch_size=[3, 4],
            device="cpu",
        )
        assert data.is_meta is False
        assert data.to("meta").is_meta is True

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

    def test_from_dataclass_exclusive_flags(self):
        # https://github.com/pytorch/tensordict/issues/2979 (item 17)
        @dataclasses.dataclass
        class Tensors:
            a: torch.Tensor

        x = MyDataClass(a=0, b=torch.zeros(()), c="c")
        with pytest.warns(FutureWarning, match="TensorDict 0.17"):
            cls = from_dataclass(MyDataClass, autocast=True, nocast=True)
        assert cls._autocast and cls._nocast
        with pytest.warns(FutureWarning, match="TensorDict 0.17"):
            assert is_tensorclass(from_dataclass(x, autocast=True, nocast=True))
        for flag in ("autocast", "nocast"):
            with pytest.warns(FutureWarning, match="TensorDict 0.17"):
                cls = from_dataclass(Tensors, tensor_only=True, **{flag: True})
            assert cls._tensor_only and getattr(cls, f"_{flag}")
            # An instance raises already.
            with pytest.raises(TypeError, match="exclusive"):
                from_dataclass(
                    Tensors(torch.zeros(())), tensor_only=True, **{flag: True}
                )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            from_dataclass(MyDataClass, autocast=True)
            from_dataclass(x, nocast=True)
            from_dataclass(Tensors, tensor_only=True)
            # The flags are not used with dest_cls or for a tensorclass type.
            dest_cls = from_dataclass(MyDataClass)
            with pytest.warns(FutureWarning, match="keeping the Python value"):
                assert isinstance(
                    from_dataclass(x, dest_cls=dest_cls, autocast=True, nocast=True),
                    dest_cls,
                )
            assert from_dataclass(dest_cls, tensor_only=True, autocast=True) is dest_cls
            # Inputs that are not dataclasses raise as before.
            with pytest.raises(TypeError, match="Expected a obj input"):
                from_dataclass(3, autocast=True, nocast=True)
            with pytest.raises(AttributeError, match="__dataclass_fields__"):
                from_dataclass(int, autocast=True, nocast=True)

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

        with pytest.warns(FutureWarning, match="keeping the Python value"):
            x = MyTensorClass(a=0, b=0, c=0)
        assert isinstance(x.a, torch.Tensor)
        assert isinstance(x.b, torch.Tensor)
        assert isinstance(x.c, torch.Tensor)

        with pytest.warns(FutureWarning, match="keeping the Python value"):
            x = TensorDict.from_dataclass(MyTensorClass(a=0, b=0, c=0))
        assert isinstance(x, TensorDict)
        assert isinstance(x["a"], torch.Tensor)
        assert isinstance(x["b"], torch.Tensor)
        assert isinstance(x["c"], torch.Tensor)

        with pytest.warns(FutureWarning, match="keeping the Python value") as record:
            x = from_dataclass(MyTensorClass(a=0, b=0, c=0))
        # it points at the call, so Python shows it once per call site, not
        # once per call
        assert {w.filename for w in record} == {__file__}
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
        with pytest.warns(FutureWarning, match="keeping the Python value"):
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

        # The tensorclass ignores repr=False and kw_only=True (#3073)
        with pytest.warns(UserWarning, match="tensorclasses ignore the field option"):
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
        # a scalar over a NonTensorData entry: a tensor until 0.17 (gh-1199)
        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
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

    def test_get_dense_as_list_warns(self):
        # over a dense tensordict, as_list and as_nested_tensor are ignored,
        # and warn as TensorDict.get does (#2551)
        X = torch.arange(6.0).view(2, 3)
        obj = MyData2(X=X, y=torch.zeros(2), z=None, batch_size=[2])
        for name in ("as_list", "as_nested_tensor"):
            with pytest.warns(
                FutureWarning,
                match=rf"^MyData2\.get\(key, {name}=True\) returns the stored "
                "tensor today. Starting with TensorDict 0.17",
            ) as record:
                out = obj.get("X", **{name: True})
            assert len(record) == 1
            assert record[0].filename == __file__
            assert out is X
        # a nested key warns once, from the tensordict that ignores it
        outer = TensorDict(data=obj, batch_size=[2])
        holder = MyData2(X=X, y=obj, z=outer, batch_size=[2])
        for key, cls in (
            (("y", "X"), "MyData2"),
            (("z", "data", "X"), "TensorDict"),
        ):
            with pytest.warns(FutureWarning, match=rf"^{cls}\.get") as record:
                assert holder.get(key, as_list=True) is X
            assert len(record) == 1
        lazy = MyData2.from_tensordict(
            LazyStackedTensorDict(
                TensorDict(X=torch.ones(3)), TensorDict(X=torch.ones(3))
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert obj.get("X") is X
            assert obj.get("X", as_padded_tensor=True) is X
            assert obj.get("missing", 0, as_list=True) == 0
            assert isinstance(lazy.get("X", as_list=True), list)
            assert lazy.get("X", as_nested_tensor=True).is_nested

    @pytest.mark.parametrize("tensor_only", [False, True])
    def test_field_typo_suggests_field(self, tensor_only):
        # #3079: a misspelled field names the fields and the closest one
        @tensorclass(tensor_only=tensor_only)
        class Data:
            obs: torch.Tensor
            reward: torch.Tensor

        x = torch.zeros(3)
        data = Data(obs=x, reward=x, batch_size=[3])
        fields = re.escape("Its fields are ['obs', 'reward'].")
        with pytest.raises(
            AttributeError,
            match=re.escape(
                "'Data' object has no attribute or field 'rewrd'. "
                "Did you mean 'reward'? "
            )
            + fields,
        ):
            data.rewrd
        assert not hasattr(data, "rewrd")
        assert getattr(data, "rewrd", None) is None
        # a private name keeps the bare message
        with pytest.raises(AttributeError, match="^_rewrd$"):
            data._rewrd
        # the unknown keyword comes before the missing field
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Data.__init__() got an unexpected keyword argument 'rewrd'. "
                "Did you mean 'reward'? Its fields are ['obs', 'reward']. It is "
                "also missing 1 required positional argument: 'reward'."
            ),
        ):
            Data(obs=x, rewrd=x, batch_size=[3])
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Data.__init__() got unexpected keyword arguments ['rewrd', 'ob']."
            ),
        ):
            Data(rewrd=x, ob=x, batch_size=[3])
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Data.__init__() missing 1 required positional argument: 'reward'"
            ),
        ):
            Data(obs=x, batch_size=[3])
        if not tensor_only:
            with pytest.raises(
                AttributeError, match=fields + re.escape(" Did you mean 'reward'?")
            ):
                Data(obs=x, reward=x, rewrd=x, batch_size=[3])
        with pytest.raises(
            AttributeError, match=fields + re.escape(" Did you mean 'reward'?")
        ):
            data.rewrd = x
        # the private fields are not listed
        with pytest.raises(AttributeError, match=re.escape("Its fields are ['data'].")):
            NonTensorData("a").foo

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

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Invalid indexing arguments: 'X'. A tensorclass is indexed along its "
                "batch dimensions; to read a field, use the attribute or get('X')."
            ),
        ):
            data["X"]
        with pytest.raises(
            ValueError,
            match=re.escape("to write a field, use the attribute or set('X', value)"),
        ):
            data["X"] = X

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

    @pytest.mark.parametrize("tensor_only", [False, True])
    @set_printoptions(compact=False)
    def test_grad_keeps_self(self, tensor_only):
        # .grad holds None for a field without a grad. Neither that None nor
        # later writes into the grad may reach the instance.
        @tensorclass(tensor_only=tensor_only)
        class MyClass:
            x: torch.Tensor
            y: torch.Tensor | None = None

        a = MyClass(
            x=torch.zeros(3, requires_grad=True), y=torch.ones(3), batch_size=[3]
        )
        (a.x * 2).sum().backward()
        grad = a.grad
        assert (grad.x == 2).all()
        assert grad.y is None
        torch.testing.assert_close(a.to_dict()["y"], torch.ones(3))
        torch.testing.assert_close(
            a.to_tensordict(retain_none=True)["y"], torch.ones(3)
        )
        assert repr(a).count("y=") == 1
        grad.x = None
        assert grad.x is None
        assert isinstance(a.to_dict()["x"], torch.Tensor)

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

    def test_iter(self):
        myc = MyData(
            X=torch.arange(6).view(3, 2),
            y=torch.rand(3, 2, 4),
            z="test_tensorclass",
            batch_size=[3],
        )
        elements = list(myc)
        assert len(elements) == 3
        for i, element in enumerate(elements):
            assert type(element) is MyData
            assert element.batch_size == torch.Size([])
            assert (element.X == myc.X[i]).all()
            assert element.z == "test_tensorclass"

        myc0d = myc[0]
        with pytest.raises(TypeError, match="iteration over a 0-d tensordict"):
            iter(myc0d)

        # NonTensorData indexing has no bound check: iteration must stop at the batch size
        ntd = NonTensorData("a string", batch_size=[3])
        assert len(list(itertools.islice(ntd, 4))) == 3

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

    def test_non_tensor_items(self):
        @tensorclass
        class Inner:
            z: torch.Tensor
            u: str

        @tensorclass
        class MyData:
            x: torch.Tensor
            s: str
            y: torch.Tensor = None
            inner: Inner = None

        inner = Inner(z=torch.zeros(3), u="u", batch_size=[3])
        data = MyData(x=torch.zeros(3), s="s", inner=inner, batch_size=[3])
        # as for a TensorDict: a tuple, with the non-tensor fields
        items = data.non_tensor_items()
        assert isinstance(items, tuple)
        assert [key for key, _ in items] == ["y", "s"]
        assert items[0][1] is None
        assert items[1][1].data == "s"
        # include_nested=True used to raise TypeError
        items = data.non_tensor_items(include_nested=True)
        assert [key for key, _ in items] == ["y", "s", ("inner", "u")]
        assert items[2][1].data == "u"
        # the pytree context keeps the fields held outside the TensorDict
        leaves, spec = torch.utils._pytree.tree_flatten(data)
        assert spec.context["non_tensor_data"] == {"y": None}
        rebuilt = torch.utils._pytree.tree_unflatten(leaves, spec)
        assert rebuilt.s == "s" and rebuilt.y is None and rebuilt.inner.u == "u"

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

    def test_pickle_tensor_only_while_compiling(self, monkeypatch):
        # torch.compile sets this process-wide flag for as long as it compiles,
        # so other threads, such as a DataLoader's pin-memory thread, see it.
        data = MyDataTensorOnly(X=torch.ones(3, 4), batch_size=[3])
        payload = pickle.dumps(data)
        if hasattr(torch.compiler, "_compile_session_context"):
            with torch.compiler._compile_session_context():
                data2 = pickle.loads(payload)
        else:
            # Before torch 2.13, only torch.export sets the flag
            monkeypatch.setattr(torch.compiler, "_is_compiling_flag", True)
            assert torch.compiler.is_compiling()
            data2 = pickle.loads(payload)
            monkeypatch.undo()
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

    @pytest.mark.skipif(
        sys.version_info < (3, 13), reason="copy.replace requires Python 3.13"
    )
    @pytest.mark.parametrize("spelling", ["decorator", "subclass"])
    def test_copy_replace(self, spelling):
        # copy.replace keeps the batch size, device and names (#3072)
        if spelling == "decorator":

            @tensorclass
            class MyDataReplace:
                X: torch.Tensor
                z: str = "z"
                y: MyDataReplace = None

        else:

            class MyDataReplace(TensorClass):
                X: torch.Tensor
                z: str = "z"
                y: MyDataReplace = None

        nested = MyDataReplace(X=torch.zeros(3, 2), batch_size=[3])
        data = MyDataReplace(
            X=torch.zeros(3, 2), y=nested, batch_size=[3], names=["n"], device="cpu"
        )
        for changes in ({}, {"X": torch.ones(3, 2)}, {"z": "other"}):
            out = copy.replace(data, **changes)
            assert type(out) is MyDataReplace
            assert out is not data
            assert out.batch_size == torch.Size([3])
            assert out.names == ["n"]
            assert out.device == torch.device("cpu")
            assert type(out.y) is MyDataReplace
            assert out.y.batch_size == torch.Size([3])
        assert (out.X == 0).all()
        assert out.z == "other"
        assert data.z == "z"
        assert (copy.replace(data, X=torch.ones(3, 2)).X == 1).all()

        # The metadata can still be passed, as to __init__
        out = copy.replace(data, names=["m"], y=None)
        assert out.batch_size == torch.Size([3])
        assert out.names == ["m"]
        out = copy.replace(data, batch_size=[3, 2], y=None)
        assert out.batch_size == torch.Size([3, 2])
        assert out.names == [None, None]
        with pytest.raises(AttributeError, match="no such field"):
            copy.replace(data, other=1)

    @pytest.mark.skipif(
        sys.version_info < (3, 13), reason="copy.replace requires Python 3.13"
    )
    def test_copy_replace_init(self):
        # copy.replace builds the copy with __init__, as dataclasses.replace
        class MyDataInit(TensorClass):
            X: torch.Tensor

            def __init__(self, X, scale=1.0):
                self.X = X * scale

        data = MyDataInit(torch.ones(3), batch_size=[3])
        out = copy.replace(data, scale=2.0)
        assert out.batch_size == torch.Size([3])
        assert (out.X == 2).all()

        # A batch size that __init__ or __post_init__ sets from the data wins
        class MyDataInitBatch(TensorClass):
            X: torch.Tensor

            def __init__(self, X):
                self.X = X
                self.batch_size = X.shape[:1]

        @tensorclass
        class MyDataPostInitBatch:
            X: torch.Tensor

            def __post_init__(self):
                self.batch_size = self.X.shape[:1]

        class MyDataAutoBatch(TensorClass):
            X: torch.Tensor

            def __post_init__(self):
                self.auto_batch_size_(1)

        for cls in (MyDataInitBatch, MyDataPostInitBatch, MyDataAutoBatch):
            data = cls(X=torch.zeros(3, 2))
            assert data.batch_size == torch.Size([3])
            assert copy.replace(data).batch_size == torch.Size([3])
            out = copy.replace(data, X=torch.ones(5, 2))
            assert out.batch_size == torch.Size([5])
            assert (out.X == 1).all()
            out = copy.replace(data, X=torch.ones(5, 2), names=["m"])
            assert out.batch_size == torch.Size([5])
            assert out.names == ["m"]

        @tensorclass
        class MyDataPostInit:
            X: torch.Tensor

            def __post_init__(self):
                self.X = self.X.float()

        data = MyDataPostInit(X=torch.zeros(3), batch_size=[3])
        out = copy.replace(data, X=torch.ones(3, dtype=torch.long))
        assert out.batch_size == torch.Size([3])
        assert out.X.dtype == torch.float32

        data = NonTensorData("a", batch_size=[3])
        out = copy.replace(data, data="b")
        assert type(out) is NonTensorData
        assert out.batch_size == torch.Size([3])
        assert out.data == "b"

    @pytest.mark.skipif(
        sys.version_info < (3, 13), reason="copy.replace requires Python 3.13"
    )
    def test_copy_replace_no_fit(self):
        # Without batch dims to keep, or when the new values do not fit them,
        # the copy keeps the batch size that dataclasses.replace gives it
        class MyDataNoFit(TensorClass):
            X: torch.Tensor
            z: str = "z"

        data = MyDataNoFit(X=torch.zeros(3), batch_size=[3], names=["n"])
        out = copy.replace(data, X=torch.zeros(5))
        assert out.batch_size == torch.Size([])
        assert out.names == []
        assert out.X.shape == torch.Size([5])
        z = NonTensorData("other")
        out = copy.replace(data, z=z)
        assert out.batch_size == torch.Size([])
        assert z.batch_size == torch.Size([])

        # names= goes through the names setter of the copy
        for source, changes in (
            (MyDataNoFit(X=torch.zeros(3)), {}),
            (data, {"X": torch.zeros(5)}),
        ):
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                out = copy.replace(source, names=["m"], **changes)
            assert any(
                "Got the dimension names" in str(w.message)
                and w.category is FutureWarning
                for w in record
            )
            assert out.batch_size == torch.Size([])
            assert out.names == []

        # Names that do not match the batch dims raise
        with pytest.raises(ValueError, match="dimension names"):
            copy.replace(data, names=["m", "k"])

        # Non-tensor values with other batch dims do not fit either
        z = NonTensorData("a", batch_size=[5])
        out = copy.replace(data, z=z)
        assert out.batch_size == torch.Size([])
        assert z.batch_size == torch.Size([5])
        out = copy.replace(data, z=["a"] * 5)
        assert out.z == ["a"] * 5

    @pytest.mark.skipif(
        sys.version_info < (3, 13), reason="copy.replace requires Python 3.13"
    )
    def test_copy_replace_shared(self):
        # The sub-tensordicts of the caller and of the instance keep their
        # batch size and names
        class MyDataShared(TensorClass):
            X: torch.Tensor
            z: Any = None

        named = MyDataShared(X=torch.zeros(3), batch_size=[3], names=["n"])
        td = TensorDict(a=torch.zeros(3), batch_size=[3])
        z = NonTensorData("a", batch_size=[3])
        for value in (td, z):
            out = copy.replace(named, z=value)
            assert out.names == ["n"]
            assert value.batch_size == torch.Size([3])
            assert value.names == [None]

        data = MyDataShared(X=torch.zeros(3), batch_size=[3])
        td = TensorDict(a=torch.zeros(3), batch_size=[3], names=["k"])
        out = copy.replace(data, z=td)
        assert out.batch_size == torch.Size([3])
        assert td.names == ["k"]

        # With no names to set, the copy shares the sub-tensordict, as on main
        sub = TensorDict(a=torch.zeros(3), batch_size=[3])
        data = MyDataShared(X=torch.zeros(3), z=sub, batch_size=[3])
        out = copy.replace(data)
        assert out.batch_size == torch.Size([3])
        assert out.z is data.z

        sub = TensorDict(a=torch.zeros(3), batch_size=[3])
        data = MyDataShared(X=torch.zeros(3), z=sub, batch_size=[3])
        out = copy.replace(data, names=["q"])
        assert out.names == ["q"]
        assert out.z.names == ["q"]
        assert data.z.names == [None]
        assert sub.names == [None]

    @pytest.mark.skipif(
        sys.version_info < (3, 13), reason="copy.replace requires Python 3.13"
    )
    def test_copy_replace_named_replace(self):
        # __replace__ reserves no field or method name
        @tensorclass(shadow=True)
        class MyDataField:
            X: torch.Tensor
            replace: torch.Tensor

        data = MyDataField(X=torch.zeros(3), replace=torch.zeros(3), batch_size=[3])
        out = copy.replace(data, replace=torch.ones(3))
        assert out.batch_size == torch.Size([3])
        assert (out.replace == 1).all()

        class MyDataMethod(TensorClass):
            X: torch.Tensor

            def replace(self, **kwargs):
                return "replace"

        data = MyDataMethod(X=torch.zeros(3), batch_size=[3])
        assert data.replace() == "replace"
        out = copy.replace(data, X=torch.ones(3))
        assert out.batch_size == torch.Size([3])
        assert (out.X == 1).all()

        class MyDataOwn(TensorClass):
            X: torch.Tensor

            def __replace__(self, **changes):
                return "own"

        class MyDataOwnChild(MyDataOwn):
            y: torch.Tensor = None

        for cls in (MyDataOwn, MyDataOwnChild):
            assert copy.replace(cls(X=torch.zeros(3), batch_size=[3])) == "own"

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

    def test_post_init_lazy_stack(self):
        # __post_init__ runs on each stacked instance, not on the stack
        calls = _MESH_POST_INIT_CALLS
        small = _ValidatingMesh(points=torch.rand(4, 3), cells=torch.zeros(2, 3))
        large = _ValidatingMesh(points=torch.rand(6, 3), cells=torch.zeros(2, 3))
        calls.clear()
        batch = lazy_stack([small, large])
        assert isinstance(batch, _ValidatingMesh)
        assert isinstance(batch._tensordict, LazyStackedTensorDict)
        assert calls == [(4, 3), (6, 3)]
        calls.clear()
        clone = batch.clone()
        assert isinstance(clone, _ValidatingMesh)
        assert calls == [(4, 3), (6, 3)]
        assert clone[1].points.shape == (6, 3)
        # a stack of stacks
        calls.clear()
        assert isinstance(lazy_stack([batch, batch]), _ValidatingMesh)
        assert len(calls) == 4
        # dense stacks keep one call, on the stacked instance
        calls.clear()
        torch.stack([small, small])
        assert calls == [(2, 4, 3)]

        # The members of a raw lazy stack are validated, and the fields set by
        # __post_init__ land in the stack
        raw = lazy_stack(
            [
                TensorDict(points=torch.rand(4, 3), cells=torch.zeros(2, 3)),
                TensorDict(points=torch.rand(6, 3), cells=torch.zeros(2, 3)),
            ]
        )
        calls.clear()
        batch = _ValidatingMesh.from_tensordict(raw)
        assert calls == [(4, 3), (6, 3)]
        assert batch[0].mask.shape == (4,)
        assert batch[1].mask.shape == (6,)
        raw = lazy_stack(
            [
                TensorDict(points=torch.rand(4, 3), cells=torch.zeros(2, 3)),
                TensorDict(points=torch.rand(6), cells=torch.zeros(2, 3)),
            ]
        )
        with pytest.raises(ValueError, match="points must be"):
            _ValidatingMesh.from_tensordict(raw)
        # The stack's keys are those that all the members share: a member may
        # hold one that is not a field
        raw = lazy_stack(
            [
                TensorDict(points=torch.rand(4, 3), cells=torch.zeros(2, 3), foo=1),
                TensorDict(points=torch.rand(6, 3), cells=torch.zeros(2, 3)),
            ]
        )
        calls.clear()
        batch = _ValidatingMesh.from_tensordict(raw)
        assert calls == [(4, 3), (6, 3)]
        calls.clear()
        batch.clone()
        assert calls == [(4, 3), (6, 3)]

    def test_post_init_empty_shell(self):
        # empty() and pad_sequence build shells that they fill afterwards:
        # __post_init__ does not run on them
        calls = _MESH_POST_INIT_CALLS
        small = _ValidatingMesh(points=torch.rand(4, 3), cells=torch.zeros(2, 3))
        large = _ValidatingMesh(points=torch.rand(6, 3), cells=torch.zeros(2, 3))
        calls.clear()
        for shell in (small.empty(), small.empty(recurse=True)):
            assert isinstance(shell, _ValidatingMesh)
            assert shell.points is None
            assert shell.mask is None
        shell = TensorDict(mesh=small).empty(recurse=True)
        assert isinstance(shell["mesh"], _ValidatingMesh)
        assert shell["mesh"].points is None
        assert calls == []
        # With a device, to() rebuilds the shell. The copy that empty() makes
        # first runs __post_init__ on the full mesh.
        shell = TensorDict(mesh=small).empty(recurse=True, device="cpu")
        assert isinstance(shell["mesh"], _ValidatingMesh)
        assert shell["mesh"].points is None
        assert calls == [(4, 3)]
        calls.clear()
        padded = pad_sequence([small, large])
        assert isinstance(padded, _ValidatingMesh)
        assert padded.points.shape == (2, 6, 3)
        assert padded.mask.shape == (2, 6)
        assert calls == []
        # the post-init of the other calls still runs
        small.clone()
        assert calls == [(4, 3)]

    @pytest.mark.parametrize("op", list(_REBUILD_OPS))
    def test_skip_post_init_on_rebuild(self, op):
        calls = _REBUILD_CALLS
        assert not _Counted._skip_post_init_on_rebuild
        assert _SkipCounted._skip_post_init_on_rebuild
        kwargs = {"offsets": torch.arange(3), "indices": torch.arange(3)}
        tc = _Counted(**kwargs, batch_size=[3])
        calls.clear()
        _REBUILD_OPS[op](tc)
        assert calls and set(calls) == {_Counted}
        # The flag skips __post_init__ on rebuilds, and subclasses inherit it
        tc = _SkipCounted(**kwargs, batch_size=[3])
        assert calls[-1] is _SkipCounted
        calls.clear()
        result = _REBUILD_OPS[op](tc)
        assert calls == []
        result = result[0] if op == "unbind" else result
        assert isinstance(result, _SkipCounted)
        # Each class follows its own flag
        holder = _CountedHolder(adj=tc, x=torch.zeros(3), batch_size=[3])
        calls.clear()
        _REBUILD_OPS[op](holder)
        assert calls and set(calls) == {_CountedHolder}

    def test_skip_post_init_on_rebuild_new_data(self, tmp_path):
        # The calls that bring in data from outside still run __post_init__
        calls = _REBUILD_CALLS
        tc = _SkipCounted(offsets=torch.arange(3), indices=torch.arange(3))
        holder = _CountedHolder(adj=tc, x=torch.zeros(()))
        calls.clear()
        _SkipCounted.from_tensordict(tc.to_tensordict())
        assert calls == [_SkipCounted]
        calls.clear()
        _SkipCounted.from_dict(tc.to_dict())
        assert calls == [_SkipCounted]
        # The constructors that a tensorclass takes from TensorDict
        calls.clear()
        _SkipCounted.from_any(tc.to_dict())
        assert calls == [_SkipCounted]
        calls.clear()
        _SkipCounted.fromkeys(["offsets", "indices"], 0)
        assert calls == [_SkipCounted]
        calls.clear()
        _SkipCounted.from_list([{"offsets": 0, "indices": 1}] * 2)
        assert calls == [_SkipCounted]
        calls.clear()
        _CountedHolder.from_tensordict(holder.to_tensordict())
        assert calls == [_SkipCounted, _CountedHolder]
        calls.clear()
        _SkipCounted.from_tensordict(lazy_stack([tc.to_tensordict()] * 2))
        assert calls == [_SkipCounted, _SkipCounted]
        holder.memmap(tmp_path / "memmap")
        holder.save(tmp_path / "data.tdz")
        calls.clear()
        loaded = TensorDict.load_memmap(tmp_path / "memmap")
        assert isinstance(loaded.adj, _SkipCounted)
        assert sorted(calls, key=str) == [_CountedHolder, _SkipCounted]
        calls.clear()
        TensorDict.load(tmp_path / "data.tdz")
        assert sorted(calls, key=str) == [_CountedHolder, _SkipCounted]

    def test_skip_post_init_on_rebuild_decorator(self):
        calls = []

        @tensorclass(skip_post_init_on_rebuild=True)
        class Graph:
            x: torch.Tensor

            def __post_init__(self):
                calls.append(self.x)

        graph = Graph(x=torch.zeros(3), batch_size=[3])
        assert len(calls) == 1
        graph.clone()
        graph[0]
        assert len(calls) == 1
        assert not tensorclass(type("Plain", (), {}))._skip_post_init_on_rebuild

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
        repeated = torch.repeat_interleave(data, 2, 1)
        assert type(repeated) is type(data)
        assert repeated.shape == torch.Size((3, 8))
        assert repeated.y.shape == torch.Size((3, 8))
        assert repeated.z == z
        repeated = torch.repeat_interleave(
            data, repeats=torch.tensor([1, 2, 3, 4]), dim=1, output_size=10
        )
        assert type(repeated) is type(data)
        assert repeated.shape == torch.Size((3, 10))
        assert repeated.y.shape == torch.Size((3, 10))

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

        for reshaped in (
            torch.reshape(data, (4, 3)),
            torch.reshape(data, shape=(4, 3)),
        ):
            assert type(reshaped) is type(data)
            assert reshaped.shape == torch.Size([4, 3])
            assert reshaped.y.shape == torch.Size([4, 3])
            assert reshaped.z == reshaped.y.z == z

    def test_torch_pointwise_functions(self):
        # pointwise torch functions keep the tensorclass type (#1926)
        data = MyData(
            X=-torch.ones(3, 4), y=torch.zeros(3, 4), z="test", batch_size=[3, 4]
        )
        for result, x in (
            (torch.abs(data), torch.ones(3, 4)),
            (torch.add(data, data), -2 * torch.ones(3, 4)),
            (torch.clamp(data, min=-0.5), -0.5 * torch.ones(3, 4)),
            (torch.lerp(data, torch.zeros(3, 4), 0.5), -0.5 * torch.ones(3, 4)),
        ):
            assert type(result) is MyData
            assert result.z == "test"
            torch.testing.assert_close(result.X, x)
        # in-place forms and non-tensor data are not passed through
        with pytest.raises(TypeError):
            torch.abs_(data)
        with pytest.raises(TypeError):
            torch.abs(NonTensorData("x", batch_size=[3]))

    def test_shape_ops_minus_one(self):
        # A -1 keeps the size of its dim in expand and broadcast_to, and is
        # inferred in unflatten, as for tensors
        @tensorclass
        class MyDataMinusOne:
            X: torch.Tensor
            z: str

        data = MyDataMinusOne(X=torch.zeros(1, 6, 4), z="z", batch_size=[1, 6])
        for out in (data.expand(2, -1, 6), data.broadcast_to((2, -1, -1))):
            assert out.shape == (2, 1, 6)
            assert out.X.shape == (2, 1, 6, 4)
        out = data.unflatten(1, (3, -1))
        assert out.shape == (1, 3, 2)
        assert out.X.shape == (1, 3, 2, 4)

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
        with pytest.raises(
            AttributeError,
            match=re.escape(
                "Cannot set the attribute 'newattr': MyDataParent has no such field. "
                "Its fields are ['X', 'k', 'v', 'y', 'z']."
            ),
        ):
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

    def test_select_exclude_nested_through_leaf(self):
        # A nested key through a leaf is missing (#2617)
        @tensorclass
        class Data:
            x: torch.Tensor
            n: TensorDict

        data = Data(x=torch.zeros(3), n=TensorDict(y=torch.zeros(3)), batch_size=[3])
        with pytest.raises(KeyError, match="'x' is a Tensor, not a tensordict"):
            data.select(("x", "q"))
        out = data.select(("x", "q"), "n", strict=False)
        assert isinstance(out, Data)
        assert out.x is None
        assert out.n is data.n
        out = data.exclude(("x", "q"))
        assert isinstance(out, Data)
        assert out.x is data.x

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
        with pytest.raises(
            AttributeError,
            match=re.escape(
                "Cannot set the attribute 'newattr': MyDataParent has no such field. "
                "Its fields are ['W', 'X', 'k', 'v', 'y', 'z']."
            ),
        ):
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

            if broadcast_type == "scalar":
                # z is a non-tensor entry, where the scalar is cast to a
                # tensor until 0.17, which leaves it unchanged (gh-2522). The
                # call warns once, for z and y.z.
                with pytest.warns(
                    FutureWarning, match="leaving the non-tensor entries unchanged"
                ) as w:
                    data[:2] = val
                assert len(w) == 1
            else:
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

    def test_setitem_past_batch_dims_no_batch_dims(self):
        # #3202: on a tensorclass without batch dims, data[0] = value writes
        # into the first row of every field, and warns once that it raises
        # from 0.17, as a tensordict does; data[...] = value does not warn
        @tensorclass
        class MyData:
            x: torch.Tensor
            y: torch.Tensor

        data = MyData(x=torch.zeros(3), y=torch.zeros(3, 4), batch_size=[])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            data[0] = 1.0
        messages = [w for w in record if "runs past the batch dims" in str(w.message)]
        assert len(messages) == 1
        assert messages[0].category is FutureWarning
        assert messages[0].filename == __file__
        assert (data.x == torch.tensor([1.0, 0.0, 0.0])).all()
        assert (data.y[0] == 1).all() and (data.y[1:] == 0).all()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            data[...] = 2.0
        assert (data.x == 2).all() and (data.y == 2).all()

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

        for torch_split in (
            torch.tensor_split(data_out, (1, 4, 5), 1),
            torch.tensor_split(data_out, indices=[1, 4, 5], dim=1),
            torch.tensor_split(data_out, torch.tensor([1, 4, 5]), 1),
        ):
            assert len(torch_split) == 4
            for split, expected in zip(torch_split, data_split):
                assert type(split) is type(data_out)
                assert split.batch_size == expected.batch_size
                assert split.y.batch_size == expected.batch_size
                assert split.z == "test_tensorclass"
        torch_split = torch.tensor_split(data_out, sections=4, dim=1)
        assert [split.batch_size[1] for split in torch_split] == [2, 2, 1, 1]
        assert all(type(split) is type(data_out) for split in torch_split)

    def test_torch_chunk(self):
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
        for chunks in (
            torch.chunk(data_out, 4, 1),
            torch.chunk(data_out, chunks=4, dim=1),
        ):
            assert len(chunks) == len(data_out.chunk(4, 1)) == 3
            for chunk in chunks:
                assert type(chunk) is type(data_out)
                assert chunk.batch_size == torch.Size([3, 2])
                assert chunk.y.batch_size == torch.Size([3, 2])
                assert chunk.z == "test_tensorclass"

    def test_torch_where(self):
        @tensorclass
        class MyDataNested:
            X: torch.Tensor
            z: str
            y: "MyDataNested" = None

        def make(value):
            data_in = MyDataNested(
                X=torch.full((3, 4, 5), value), z="test_tensorclass", batch_size=[3, 4]
            )
            return MyDataNested(
                X=torch.full((3, 4, 5), value),
                z="test_tensorclass",
                y=data_in,
                batch_size=[3, 4],
            )

        data0, data1 = make(0.0), make(1.0)
        mask = torch.zeros(3, 4, dtype=torch.bool)
        mask[0] = True
        for result in (
            torch.where(mask, data1, data0),
            torch.where(mask, input=data1, other=data0),
            torch.where(mask, data1, 0.0),
            # a scalar or tensor input takes the type of other (#2743)
            torch.where(mask, 1.0, data0),
            torch.where(mask, torch.ones(5), data0),
        ):
            assert type(result) is type(data0)
            assert result.batch_size == data0.batch_size
            assert result.z == "test_tensorclass"
            assert (result.X[mask] == 1).all()
            assert (result.X[~mask] == 0).all()
            assert (result.y.X[mask] == 1).all()
            assert (result.y.X[~mask] == 0).all()
        # out is filled, and the result wraps its tensordict
        out = make(2.0)
        result = torch.where(mask, data1, data0, out=out)
        assert result._tensordict is out._tensordict
        assert (out.X[~mask] == 0).all()
        with pytest.raises(TypeError, match="missing 2 required positional"):
            torch.where(data0)

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

    @pytest.mark.parametrize("tensor_only", [False, True])
    def test_set_self_nesting(self, tensor_only):
        # #2741: these calls put the tensorclass inside itself, and the next
        # repr or clone raised a RecursionError
        class SelfNesting(TensorClass, tensor_only=tensor_only):
            a: torch.Tensor
            x: SelfNesting = None

        msg = re.escape("Cannot set a tensordict inside itself (key 'x')")
        tc = SelfNesting(a=torch.zeros(3), batch_size=[3])
        for write in (
            lambda: tc.set("x", tc),
            lambda: tc.set(("x",), tc),
            lambda: setattr(tc, "x", tc),
            lambda: tc.update({"x": tc}),
            lambda: tc.update(x=tc),
            lambda: tc.update(TensorDict(x=tc, batch_size=[3])),
            # as in TensorDict.update, the entries before x are written
            lambda: tc.update(SelfNesting(a=torch.zeros(3), x=tc, batch_size=[3])),
        ):
            with pytest.raises(ValueError, match=msg):
                write()
            assert tc.x is None
            assert (tc.a == 0).all()
        # a key that is not a field keeps its AttributeError
        with pytest.raises(AttributeError, match="no such field"):
            tc.set("zz", tc)
        # an update that leaves out the field, or that writes a copy, works
        tc.update({"x": tc, "a": torch.ones(3)}, keys_to_update=["a"])
        assert tc.x is None
        assert (tc.a == 1).all()
        tc.update({"x": tc}, clone=True)
        assert tc.x is not tc
        assert (tc.x.a == 1).all()
        repr(tc)
        # a locked tensorclass raises its lock error, as before
        tc = SelfNesting(a=torch.zeros(3), batch_size=[3]).lock_()
        with pytest.raises(RuntimeError, match="locked"):
            tc.update({"x": tc}, inplace=True)

    def test_update_self_nesting_check_skips_tensordicts(self, monkeypatch):
        # update() checks whether a value is a tensorclass holding self (#2741).
        # A Tensor or a TensorDict is not a tensorclass: it skips the lookup,
        # which made an update of nested tensordicts about 10% slower than in
        # 0.15.2 (#3346)
        looked_up = []
        is_tensorclass = tensordict_base._is_tensorclass

        def recording(cls):
            looked_up.append(cls)
            return is_tensorclass(cls)

        monkeypatch.setattr(tensordict_base, "_is_tensorclass", recording)
        source = {"a": TensorDict(x=torch.zeros(3), batch_size=[3]), "b": torch.ones(3)}
        td = TensorDict(batch_size=[3])
        td.update(source)
        td.update(source, inplace=True)
        assert TensorDict not in looked_up
        assert torch.Tensor not in looked_up
        assert (td["a", "x"] == 0).all() and (td["b"] == 1).all()

        class SelfNesting(TensorClass):
            a: torch.Tensor
            x: SelfNesting = None

        tc = SelfNesting(a=torch.zeros(3), batch_size=[3])
        with pytest.raises(ValueError, match="inside itself"):
            tc.update({"x": tc})

    def test_set_non_tensor_data_to_itself(self):
        # a NonTensorData set to itself keeps its data, as before #2741's check
        nd = NonTensorData("hi")
        nd.set("data", nd)
        assert nd.data == "hi"
        nd.data = nd
        assert nd.data == "hi"

    def test_update_ignore_lock_deprecated(self):
        # #2260: ignore_lock=True warns once per call, also when the update
        # goes on into the tensordict of the tensorclass and its nested entries
        @tensorclass
        class TC:
            x: torch.Tensor
            y: TensorDict

        def make(value):
            return TC(
                x=torch.full((3,), value),
                y=TensorDict(z={"w": torch.full((3,), value)}, batch_size=[3]),
                batch_size=[3],
            )

        data = make(0.0)
        with pytest.warns(
            DeprecationWarning,
            match=r"^update\(ignore_lock=True\) is deprecated and will be removed in "
            r"TensorDict 0\.17\.",
        ) as record:
            assert data.update(make(1.0), ignore_lock=True) is data
        assert len(record) == 1
        assert record[0].filename == __file__
        assert (data.x == 1).all() and (data.y["z", "w"] == 1).all()

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            data.update(make(2.0))
            data.update(make(3.0), ignore_lock=False)
        assert (data.x == 3).all() and (data.y["z", "w"] == 3).all()

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

    @pytest.mark.parametrize("source", ["dict", "kwargs", "tensorclass"])
    def test_update_unset_field(self, source):
        # update() must drop the None placeholder of a field it writes, as set() does.
        @tensorclass
        class TC:
            x: torch.Tensor
            z: torch.Tensor = None

        tc = TC(x=torch.zeros(3), batch_size=[3])
        z = torch.ones(3)
        if source == "dict":
            tc.update({"z": z})
        elif source == "kwargs":
            tc.update(z=z)
        else:
            tc.update(TC(x=torch.zeros(3), z=z, batch_size=[3]))
        torch.testing.assert_close(tc.z, z)
        torch.testing.assert_close(tc.to_dict()["z"], z)
        torch.testing.assert_close(tc.to_tensordict(retain_none=True)["z"], z)
        assert "z=None" not in repr(tc)
        other = TC(x=torch.zeros(3), z=torch.zeros(3), batch_size=[3])
        other.load_state_dict(tc.state_dict())
        torch.testing.assert_close(other.z, z)

    @pytest.mark.parametrize("tensor_only", [False, True])
    @pytest.mark.parametrize(
        "method",
        [
            "set_non_tensor",
            "setdefault",
            "create_nested",
            "rename_key_",
            "update_td",
            "update_tc",
        ],
    )
    def test_write_non_field_key(self, method, tensor_only):
        # set() rejects keys that are not fields. These methods store them
        # and warn until TensorDict 0.17, and write fields without a warning.
        @tensorclass(tensor_only=tensor_only)
        class TC:
            x: torch.Tensor
            y: torch.Tensor = None
            td: TensorDict = None

        @tensorclass
        class Other:
            x: torch.Tensor
            nope: torch.Tensor

        def write(tc, key):
            if method == "set_non_tensor":
                tc.set_non_tensor(key, "v")
            elif method == "setdefault":
                tc.setdefault(key, torch.ones(3))
            elif method == "create_nested":
                tc.create_nested(key)
            elif method == "rename_key_":
                tc.rename_key_("x", key)
            elif method == "update_td":
                tc.update(
                    TensorDict({key: torch.ones(3), "x": torch.ones(3)}, [3]),
                    keys_to_update=None if key == "nope" else [key],
                )
            elif key == "nope":
                tc.update(Other(x=torch.ones(3), nope=torch.ones(3), batch_size=[3]))
            else:
                tc.update(
                    Other(x=torch.ones(3), nope=torch.ones(3), batch_size=[3]),
                    keys_to_update=[key],
                )

        name = "update" if method.startswith("update") else method
        tc = TC(x=torch.zeros(3), batch_size=[3])
        with pytest.warns(
            DeprecationWarning,
            match=re.escape(
                "Writing keys that are not fields (['nope']) with "
                f"TC.{name}(), which set() rejects with "
                "AttributeError, is deprecated and will be removed in "
                "TensorDict 0.17."
            ),
        ) as record:
            write(tc, "nope")
        assert len(record) == 1
        assert record[0].filename == __file__
        assert "nope" in tc._tensordict.keys()

        # a field, and a key that is not a field left out by keys_to_update
        key = "td" if method == "create_nested" else "x" if "update" in method else "y"
        tc = TC(x=torch.zeros(3), batch_size=[3])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            write(tc, key)
        assert "nope" not in tc._tensordict.keys()
        assert key in tc._tensordict.keys()

    @pytest.mark.parametrize("index", ["int", "slice", "ellipsis", "mask"])
    @pytest.mark.parametrize("nested", [False, True])
    def test_setitem_non_field_key(self, index, nested):
        # set() rejects keys that are not fields. An indexed write of a
        # TensorDict stores them and warns until TensorDict 0.17, as update()
        # does, and writes fields without a warning.
        @tensorclass
        class TC:
            x: torch.Tensor

        idx = {
            "int": 0,
            "slice": slice(None),
            "ellipsis": ...,
            "mask": torch.tensor([True, False, True]),
        }[index]
        batch_size = torch.zeros(3)[idx].shape
        key = ("nope", "a") if nested else "nope"
        tc = TC(x=torch.zeros(3), batch_size=[3])
        value = TensorDict(
            {key: torch.ones(batch_size), "x": torch.ones(batch_size)}, batch_size
        )
        with pytest.warns(
            DeprecationWarning,
            match=re.escape(
                "Writing keys that are not fields (['nope']) with "
                "TC.__setitem__(), which set() rejects with "
                "AttributeError, is deprecated and will be removed in "
                "TensorDict 0.17."
            ),
        ) as record:
            tc[idx] = value
        assert len(record) == 1
        assert record[0].filename == __file__
        assert "nope" in tc._tensordict.keys()

        tc = TC(x=torch.zeros(3), batch_size=[3])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tc[idx] = value.exclude("nope")
            # tensorclass, scalar and tensor values are not checked here
            tc[idx] = TC(x=torch.full(batch_size, 2.0), batch_size=batch_size)
            tc[idx] = 3.0
            tc[idx] = torch.full(batch_size, 4.0)
        assert set(tc._tensordict.keys()) == {"x"}
        assert (tc.x[idx] == 4).all()

        # a tensorclass of another class still raises
        @tensorclass
        class Other:
            nope: torch.Tensor

        with pytest.raises(ValueError, match="compatible class"):
            tc[idx] = Other(nope=torch.ones(batch_size), batch_size=batch_size)

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

        # The keyword entries do not change the positional argument (#2949)
        data_replace = data.replace(replacement, z="keyword")
        assert replacement.z == "replacement"
        assert data_replace.z == "keyword"
        assert isinstance(data_replace.y, MyDataNested)
        assert data_replace.y.z == "replacement"

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

    @pytest.mark.parametrize(
        "flags", [{}, {"tensor_only": True}, {"shadow": True}], ids=str
    )
    def test_td_classmethods_on_decorated_class(self, flags):
        # https://github.com/pytorch/tensordict/issues/2524: on a @tensorclass,
        # MC.stack([...]) bound the list to self and raised AttributeError.
        @tensorclass(**flags)
        class MC:
            foo: torch.Tensor
            bar: torch.Tensor

        class MCChild(MC):
            pass

        out = MC.from_list(
            [{"foo": torch.zeros(()), "bar": 2}, {"foo": 3, "bar": torch.ones(())}]
        )
        assert type(out) is MC
        assert out.batch_size == (2,)
        assert (out.foo == torch.tensor([0, 3])).all()
        assert type(MCChild.from_list([{"foo": 1, "bar": 2}])) is MCChild
        assert type(MC.fromkeys(["foo", "bar"], 0)) is MC
        assert type(MC.from_any({"foo": torch.zeros(3), "bar": 1})) is MC
        mc = MC(foo=torch.zeros(3), bar=torch.ones(3), batch_size=[3])
        for name in ("stack", "cat", "lazy_stack", "maybe_dense_stack"):
            assert type(getattr(MC, name)([mc, mc])) is MC, name
            # the call on an instance keeps working
            assert type(getattr(mc, name)([mc, mc])) is MC, name
        assert MC.stack([mc, mc], 1).batch_size == (3, 2)
        assert MC.cat([mc, mc]).batch_size == (6,)

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

        # Load the dataset
        dataset = streaming_dataset(local=tmpdir, remote=None, batch_size=2)
        dl = torch.utils.data.DataLoader(  # noqa: TOR401
            dataset=dataset, batch_size=2, collate_fn=MC.from_list
        )
        batches = list(dl)
        batches = [_batch for batch in batches for _batch in batch.unbind(0)]
        test_td = TensorDict.lazy_stack(batches)
        assert isinstance(test_td, MC)
        assert isinstance(test_td._tensordict, LazyStackedTensorDict)
        assert_allclose_td(td, test_td)

    @pytest.mark.skipif(not _has_streaming, reason="streaming is not installed")
    def test_to_mds_zero_dim(self, tmpdir):
        # a tensorclass without batch dims is written as one row
        class MC(TensorClass):
            a: torch.Tensor
            s: str

        tc = MC(a=torch.arange(3), s="a string")

        tmpdir = str(tmpdir)
        tc.to_mds(out=tmpdir)

        dataset = streaming_dataset(local=tmpdir, remote=None, batch_size=1)
        assert len(dataset) == 1
        test_tc = MC.from_list([dataset[0]])[0]
        assert isinstance(test_tc, MC)
        assert_allclose_td(tc, test_tc)

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

    def test_tensorclass_get_at_passes_get_kwargs(self):
        # https://github.com/pytorch/tensordict/issues/2830
        @tensorclass
        class A:
            X: torch.Tensor
            k: Optional[Tensor] = None

        a = A(X=torch.arange(6).view(3, 2), batch_size=[3])
        td = TensorDict(X=torch.arange(6).view(3, 2), batch_size=[3])
        expected = torch.tensor([0, 1])
        assert (td.get_at("X", 0, as_list=True) == expected).all()
        # as_list=True reaches get, which warns that it ignores it on a dense
        # tensorclass until TensorDict 0.17 (#2551)
        with pytest.warns(FutureWarning, match=r"A\.get\(key, as_list=True\)"):
            assert (a.get_at("X", 0, as_list=True) == expected).all()
        with pytest.warns(FutureWarning, match=r"A\.get\(key, as_list=True\)"):
            assert (a.get_at("X", 0, default=None, as_list=True) == expected).all()
        with pytest.warns(FutureWarning, match=r"A\.get\(key, as_list=True\)"):
            assert (a.get_at("X", index=0, as_list=True) == expected).all()
        # the forms that worked before keep working
        assert (a.get_at("X", 0) == expected).all()
        assert (a.get_at("X", index=0) == expected).all()
        assert (a.get_at("X", 0, None) == expected).all()
        assert (a.get_at("X", 0, default=None) == expected).all()
        assert a.get_at("k", 0) is None
        assert a.get_at("k", 0, as_list=True) is None
        assert a.get_at("foo", 0, "working") == "working"
        assert a.get_at("foo", 0, default="working", as_list=True) == "working"
        assert a.get_at("foo", 0) is None
        assert a.get_at("foo", 0, as_list=True) is None
        with pytest.raises(TypeError, match="index argument missing"):
            a.get_at("X")

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

    def test_tensorclass_set_at_returns_self_keeps_unset_field(self):
        @tensorclass
        class MyClass:
            x: torch.Tensor
            z: torch.Tensor = None

        data = MyClass(x=torch.zeros(3), batch_size=[3])
        # z is unset, so set_at_ has no entry to write into and raises. It used
        # to delete the field first, after which reading data.z raised.
        with pytest.raises(KeyError):
            data.set_at_("z", torch.tensor(1.0), 0)
        assert data.z is None
        assert data.to_dict()["z"] is None
        assert data.set_at_("x", torch.tensor(1.0), 0) is data
        assert data.x.tolist() == [1.0, 0.0, 0.0]

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

    @pytest.mark.parametrize("source_type", ["tensorclass", "tensordict", "dict"])
    def test_update_at_(self, source_type):
        @tensorclass
        class TC:
            a: torch.Tensor
            b: torch.Tensor

        tc = TC(a=torch.zeros(4, 3), b=torch.zeros(4), batch_size=[4])
        a, b = tc.a, tc.b
        source = TC(a=torch.ones(2, 3), b=torch.ones(2), batch_size=[2])
        if source_type == "tensordict":
            source = source.to_tensordict()
        elif source_type == "dict":
            source = source.to_dict()
        assert tc.update_at_(source, slice(1, 3)) is tc
        # the values are written in place, at the index only
        assert tc.a is a and tc.b is b
        assert (tc.b == torch.tensor([0.0, 1.0, 1.0, 0.0])).all()
        assert (tc.a == tc.b.unsqueeze(-1)).all()

        tc.update_at_(source, slice(2, 4), keys_to_update=["a"])
        assert (tc.a[:, 0] == torch.tensor([0.0, 1.0, 1.0, 1.0])).all()
        assert (tc.b == torch.tensor([0.0, 1.0, 1.0, 0.0])).all()

        # a source without batch dims is written at an int index
        tc = TC(a=torch.zeros(4, 3), b=torch.zeros(4), batch_size=[4])
        a, b = tc.a, tc.b
        source = TC(a=torch.ones(3), b=torch.ones(()), batch_size=[])
        if source_type == "tensordict":
            source = source.to_tensordict()
        elif source_type == "dict":
            source = source.to_dict()
        assert tc.update_at_(source, 1) is tc
        assert tc.a is a and tc.b is b
        assert (tc.b == torch.tensor([0.0, 1.0, 0.0, 0.0])).all()
        assert (tc.a == tc.b.unsqueeze(-1)).all()

        # a source with a prefix of the batch dims, at an index on the last dim
        tc = TC(a=torch.zeros(4, 3), b=torch.zeros(4, 3), batch_size=[4, 3])
        a, b = tc.a, tc.b
        source = TC(a=torch.ones(4), b=torch.ones(4), batch_size=[4])
        if source_type == "tensordict":
            source = source.to_tensordict()
        elif source_type == "dict":
            source = source.to_dict()
        tc.update_at_(source, (slice(None), 1))
        assert tc.a is a and tc.b is b
        assert (tc.a == torch.tensor([0.0, 1.0, 0.0])).all()
        assert (tc.b == tc.a).all()

        if source_type == "tensorclass":
            # a source whose values have the batch size of tc still replaces
            # them, and warns until TensorDict 0.17
            tc = TC(a=torch.zeros(4, 3), b=torch.zeros(4), batch_size=[4])
            source = TC(a=torch.ones(4, 3), b=torch.ones(4), batch_size=[])
            with pytest.warns(FutureWarning, match="whatever the index"):
                tc.update_at_(source, 1)
            assert tc.a is source.a and tc.b is source.b

            # an UnbatchedTensor does not have the batch size, but update()
            # accepts it: the entries are still replaced
            tc = TC(
                a=torch.zeros(4, 3),
                b=UnbatchedTensor(torch.zeros(5)),
                batch_size=[4],
            )
            source = TC(
                a=torch.ones(4, 3), b=UnbatchedTensor(torch.ones(5)), batch_size=[]
            )
            with pytest.warns(FutureWarning, match="whatever the index"):
                tc.update_at_(source, 1)
            assert tc.a is source.a
            assert (tc.b == 1).all()

    def test_update_at_idx_keyword(self):
        # update_at_ names the index idx, as TensorDict.update_at_ does, and
        # still takes index= until TensorDict 0.17
        @tensorclass
        class TC:
            a: torch.Tensor

        source = TC(a=torch.ones(2), batch_size=[2])
        expected = TC(a=torch.zeros(4), batch_size=[4])
        expected.update_at_(source, slice(1, 3))
        tc = TC(a=torch.zeros(4), batch_size=[4])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert tc.update_at_(source, idx=slice(1, 3)) is tc
        assert (tc.a == expected.a).all()
        assert tc.a.tolist() == [0.0, 1.0, 1.0, 0.0]

        tc = TC(a=torch.zeros(4), batch_size=[4])
        with pytest.warns(
            DeprecationWarning,
            match=r"update_at_\(index=.*0\.17.*update_at_\(idx=",
        ) as record:
            assert tc.update_at_(source, index=slice(1, 3)) is tc
        assert record[0].filename == __file__
        assert (tc.a == expected.a).all()

    @pytest.mark.parametrize("source_type", ["tensorclass", "dict"])
    def test_update_at_full_size_source(self, source_type):
        # update_at_ sends a source that update() accepts to update(), which
        # replaces the entries whatever the index. It warns until TensorDict
        # 0.17, which writes the source at the index.
        @tensorclass
        class TC:
            a: torch.Tensor
            b: torch.Tensor = None

        tc = TC(a=torch.zeros(4), batch_size=[4])
        source = TC(a=torch.ones(4), batch_size=[4])
        if source_type == "dict":
            source = source.to_dict()
        with pytest.warns(
            FutureWarning,
            match=re.escape(
                "TC.update_at_() with a source that update() accepts, such as "
                "one with the batch size of the tensorclass, replaces the entries "
                "of the tensorclass by those of the source, whatever the index. "
                "Starting with TensorDict 0.17, the default will change to "
                "writing the source in place at the index"
            ),
        ) as record:
            assert tc.update_at_(source, 0) is tc
        assert len(record) == 1
        assert record[0].filename == __file__
        assert (tc.a == 1).all()
        if source_type == "tensorclass":
            assert tc.a is source.a

        # a source written at the index
        tc = TC(a=torch.zeros(4), b=torch.zeros(4), batch_size=[4])
        source = TC(a=torch.ones(2), b=torch.ones(2), batch_size=[2])
        if source_type == "dict":
            source = source.to_dict()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tc.update_at_(source, slice(1, 3))
        assert tc.a.tolist() == [0.0, 1.0, 1.0, 0.0]

    @pytest.mark.parametrize("source_type", ["tensorclass", "dict"])
    def test_update__keys_to_update(self, source_type):
        # update_ ignores keys_to_update for a tensorclass or dict source. It
        # warns when that changes the result, until TensorDict 0.17, which
        # honours keys_to_update as for a TensorDict source.
        @tensorclass
        class TC:
            a: torch.Tensor
            b: torch.Tensor = None
            c: torch.Tensor = None

        def source(**kwargs):
            source = TC(**kwargs, batch_size=[3])
            if source_type == "dict":
                source = {
                    key: value
                    for key, value in source.to_dict().items()
                    if value is not None
                }
            return source

        # b is written although keys_to_update leaves it out
        tc = TC(a=torch.zeros(3), b=torch.zeros(3), batch_size=[3])
        with pytest.warns(
            FutureWarning,
            match=re.escape(
                "TC.update_() with a tensorclass or dict source ignores "
                "keys_to_update, and writes the entries that it leaves out. "
                "Starting with TensorDict 0.17, the default will change to "
                "writing only the entries in keys_to_update"
            ),
        ) as record:
            tc.update_(source(a=torch.ones(3), b=torch.ones(3)), keys_to_update=["a"])
        assert len(record) == 1
        assert record[0].filename == __file__
        assert (tc.a == 1).all() and (tc.b == 1).all()

        # c is unset in tc: keys_to_update=["c"] will raise a KeyError
        with pytest.warns(FutureWarning, match="raises a KeyError"):
            tc.update_(
                source(a=torch.full((3,), 2.0), c=torch.ones(3)),
                keys_to_update=["a", "c"],
            )
        assert (tc.a == 2).all() and tc.c is None

        # the result does not change: no warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tc.update_(
                source(a=torch.full((3,), 3.0), b=torch.full((3,), 3.0)),
                keys_to_update=["a", "b", "c"],
            )
            tc.update_(source(a=torch.full((3,), 4.0)), keys_to_update=["a"])
            tc.update_(source(a=torch.full((3,), 5.0), c=torch.ones(3)))
        assert (tc.a == 5).all() and (tc.b == 3).all() and tc.c is None

    @pytest.mark.parametrize("source_type", ["tensordict", "dict"])
    def test_update_at_lazy_stack(self, source_type):
        @tensorclass
        class TC:
            a: torch.Tensor

        tc = lazy_stack([TC(a=torch.zeros(3), batch_size=[]) for _ in range(4)])
        source = TensorDict(a=torch.ones(2, 3), batch_size=[2])
        if source_type == "dict":
            source = source.to_dict()
        tc.update_at_(source, slice(1, 3))
        assert (tc.a[:, 0] == torch.tensor([0.0, 1.0, 1.0, 0.0])).all()

    # max(dim=...), min(dim=...), cummax and cummin return the torch.return_types
    # that TensorDict returns. Other tuples, such as the output of split, stay
    # plain tuples, and so does to_namedtuple, whose namedtuple cannot be pickled.
    @pytest.mark.parametrize("subclass", [False, True])
    def test_tuple_results(self, subclass):
        if subclass:

            class MyClass(TensorClass):
                x: torch.Tensor
                y: torch.Tensor

        else:

            @tensorclass
            class MyClass:
                x: torch.Tensor
                y: torch.Tensor

        data = MyClass(x=torch.randn(3, 4), y=torch.randn(3, 4), batch_size=[3, 4])
        td = data.to_tensordict()
        for method, kwargs in [
            ("max", {"dim": 0}),
            ("min", {"dim": 1, "keepdim": True}),
            ("cummax", {"dim": 0}),
            ("cummin", {"dim": 1}),
        ]:
            result = getattr(data, method)(**kwargs)
            expected = getattr(td, method)(**kwargs)
            assert type(result) is type(expected)
            assert type(result.values) is MyClass
            assert type(result.indices) is MyClass
            assert (result.values.to_tensordict() == expected.values).all()
            assert (result.indices.to_tensordict() == expected.indices).all()
        assert type(data.cummax(0, reduce=True)) is torch.return_types.cummax

        assert type(data.to_namedtuple()) is tuple

        splits = data.split(2, dim=1)
        assert type(splits) is tuple
        assert all(type(split) is MyClass for split in splits)

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

    @pytest.mark.parametrize(
        "op,args",
        [
            ("view", (-1,)),
            ("permute", (1, 0)),
            ("unsqueeze", (0,)),
            ("flatten", (0, 1)),
        ],
    )
    def test_shape_cm_unset_field(self, op, args):
        # Setting an unset field inside the context must drop its None
        # placeholder in the original object, as setting it there does.
        @tensorclass
        class TC:
            x: torch.Tensor
            z: torch.Tensor = None
            s: str = None

        tc = TC(x=torch.zeros(2, 3), batch_size=[2, 3])
        with getattr(tc, op)(*args) as tc_op:
            tc_op.z = torch.ones(tc_op.batch_size)
            tc_op.s = "value"
        z = torch.ones(2, 3)
        torch.testing.assert_close(tc.z, z)
        torch.testing.assert_close(tc.to_dict()["z"], z)
        torch.testing.assert_close(tc.to_tensordict(retain_none=True)["z"], z)
        assert tc.s == tc.to_dict()["s"] == "value"
        assert "z=None" not in repr(tc)
        assert "s=None" not in repr(tc)

    def test_shape_cm_locked_cached_result(self):
        # flatten_keys and unflatten_keys of a locked tensorclass are cached.
        # Calling them more times than the recursion limit must not raise.
        @tensorclass
        class TC:
            x: torch.Tensor
            z: torch.Tensor = None

        tc = TC(x=torch.zeros(2, 3), batch_size=[2, 3]).lock_()
        for _ in range(1100):
            with tc.flatten_keys():
                pass
            tc.unflatten_keys()

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


class _ValidatingRagged(TensorClass):
    # Module level, so that loading an archive resolves this class whatever
    # the test order. __post_init__ records the offsets it sees in `seen`.
    offsets: torch.Tensor
    indices: torch.Tensor
    seen: ClassVar[list] = []

    def __post_init__(self):
        type(self).seen.append(self.offsets.tolist())
        if int(self.offsets[-1]) != len(self.indices):
            raise ValueError("last offset must be len(indices)")


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

        with pytest.warns(FutureWarning, match="keeping the Python value"):
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

    @pytest.mark.parametrize("nested", [True, False])
    def test_save_archive_validating_post_init(self, tmp_path, nested):
        # Saving to a .tdz archive must not run __post_init__ on the
        # uninitialized or zero placeholders that it stages before writing
        # the data; __post_init__ runs on the real data when it is loaded.
        Ragged = _ValidatingRagged
        seen = Ragged.seen
        data = Ragged(offsets=torch.tensor([0, 2, 3]), indices=torch.tensor([5, 6, 7]))
        if nested:
            data = TensorDict(adj=data, x=torch.zeros(4))
        seen.clear()
        saved = data.save(tmp_path / "data.tdz")
        assert seen == [[0, 2, 3]]
        seen.clear()
        loaded = TensorDict.load(tmp_path / "data.tdz")
        assert seen == [[0, 2, 3]]
        for result in (saved, loaded):
            assert type(result) is type(data)
            assert_allclose_td(result, data)
            adj = result["adj"] if nested else result
            assert isinstance(adj, Ragged)

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

        with pytest.warns(FutureWarning, match="keeping the Python value"):
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

        with pytest.warns(FutureWarning, match="keeping the Python value"):
            assert isinstance(X(True).a, torch.Tensor)

        @tensorclass(nocast=True)
        class X:
            a: int  # type is irrelevant

        assert isinstance(X(False).a, bool)


class TestCastFilter:
    @staticmethod
    def _mesh(cls):
        return cls(
            points=torch.zeros(4, 3),
            cells=torch.zeros(4, 3, dtype=torch.long),
            point_data=TensorDict(
                p=torch.zeros(4), zone=torch.zeros(4, dtype=torch.int32), batch_size=[4]
            ),
            mask=torch.zeros(4, dtype=torch.bool),
            name="mesh",
            batch_size=[4],
        )

    class Mesh(TensorClass, cast_filter=tensordict.is_inexact):
        points: torch.Tensor
        cells: torch.Tensor
        point_data: TensorDict
        mask: torch.Tensor
        name: str

    def _check_mesh(self, out, cls, dtype):
        assert type(out) is cls
        assert out.points.dtype == dtype
        assert out.point_data["p"].dtype == dtype
        assert out.cells.dtype == torch.long
        assert out.point_data["zone"].dtype == torch.int32
        assert out.mask.dtype == torch.bool
        if out.batch_dims == 1:
            assert out.name == "mesh"

    @pytest.mark.parametrize(
        "call,dtype",
        [
            (("bfloat16", (), {}), torch.bfloat16),
            (("half", (), {}), torch.half),
            (("double", (), {}), torch.double),
            (("int", (), {}), torch.int),
            (("float64", (), {}), torch.float64),
            (("type", (torch.float64,), {}), torch.float64),
            (("to", (torch.bfloat16,), {}), torch.bfloat16),
            (("to", ("cpu", torch.float64), {"non_blocking": True}), torch.float64),
            (("to", (), {"dtype": torch.float64}), torch.float64),
        ],
    )
    def test_cast_filter_class_default(self, call, dtype):
        # The Mesh case: floats cast, int cells and bool masks kept, the
        # tensorclass type kept
        mesh = self._mesh(self.Mesh)
        name, args, kwargs = call
        out = getattr(mesh, name)(*args, **kwargs)
        self._check_mesh(out, self.Mesh, dtype)
        # A lazy stack of tensorclasses takes the class default too
        out = getattr(torch.stack([mesh, mesh]), name)(*args, **kwargs)
        self._check_mesh(out, self.Mesh, dtype)
        # cast_filter=None casts every leaf
        out = getattr(mesh, name)(*args, cast_filter=None, **kwargs)
        assert out.cells.dtype == out.mask.dtype == dtype

    def test_cast_filter_override(self):
        mesh = self._mesh(self.Mesh)
        out = mesh.to(
            torch.bfloat16,
            cast_filter=lambda k, v: tensordict.is_inexact(v) and k[0] != "point_data",
        )
        assert out.points.dtype == torch.bfloat16
        assert out.point_data["p"].dtype == torch.float32
        assert out.cells.dtype == torch.long
        with mesh.to(torch.float64) as mesh64:
            assert mesh64.points.dtype == torch.float64
            assert mesh64.cells.dtype == torch.long
        assert mesh.points.dtype == torch.float32

    def test_type_locked(self):
        # type() of a locked tensorclass is unlocked, unlike to(dtype): a
        # FutureWarning says it stays locked from 0.17 (#2959)
        mesh = self._mesh(self.Mesh).lock_()
        with pytest.warns(FutureWarning, match="type\\(\\) returns an unlocked") as w:
            out = mesh.type(torch.float64)
        assert w[0].filename == __file__
        self._check_mesh(out, self.Mesh, torch.float64)
        assert not out.is_locked
        assert mesh.to(torch.float64).is_locked

    def test_cast_filter_attrs(self):
        # The class default does not stop to(attrs), whose attrs give each
        # leaf its dtype
        mesh = self._mesh(self.Mesh).exclude("name")
        attrs = mesh.double(cast_filter=None).attrs(fields=("device", "dtype"))
        out = mesh.to(attrs)
        assert type(out) is self.Mesh
        assert out.points.dtype == out.cells.dtype == torch.float64
        assert out.point_data["zone"].dtype == torch.float64

    def test_cast_filter_inheritance(self):
        class Sub(self.Mesh):
            pass

        class Flags(self.Mesh["nocast"]):
            pass

        class Reset(self.Mesh, cast_filter=None):
            pass

        class Plain(TensorClass):
            x: torch.Tensor

        assert Sub._cast_filter is tensordict.is_inexact
        assert Flags._cast_filter is tensordict.is_inexact
        assert Reset._cast_filter is None
        assert Plain._cast_filter is None
        self._check_mesh(self._mesh(Sub).half(), Sub, torch.half)
        assert self._mesh(Reset).half().cells.dtype == torch.half
        plain = Plain(x=torch.zeros(3, dtype=torch.long))
        assert plain.half().x.dtype == torch.half
        assert plain.half(cast_filter=tensordict.is_inexact).x.dtype == torch.long

    def test_cast_filter_decorator(self):
        @tensorclass(cast_filter=tensordict.is_inexact)
        class Mesh:
            points: torch.Tensor
            cells: torch.Tensor
            point_data: TensorDict
            mask: torch.Tensor
            name: str

        assert Mesh._cast_filter is tensordict.is_inexact
        self._check_mesh(self._mesh(Mesh).bfloat16(), Mesh, torch.bfloat16)
        assert self._mesh(Mesh).bfloat16(cast_filter=None).cells.dtype == (
            torch.bfloat16
        )


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

    def test_autocast_shape_annotations(self):
        # A shape annotation casts to the array type that it wraps.
        class Shaped(TensorClass["autocast"]):
            a: _ShapedTensor
            b: _ShapedArray

        x = Shaped(a=[1, 2], b=[1, 2], batch_size=[2])
        assert isinstance(x.a, torch.Tensor)
        assert isinstance(x.b, np.ndarray)

        # A tensorclass with an ``array_type`` attribute is not unwrapped.
        class Outer(TensorClass["autocast"]):
            inner: _WithArrayType

        y = Outer(
            inner=_WithArrayType(x=torch.zeros(2), batch_size=[2]), batch_size=[2]
        )
        assert isinstance(y.inner, _WithArrayType)

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
            class MyClass:
                x: str
                y: int
                names: Any

        with pytest.raises(AttributeError):

            @tensorclass
            class MyClass:
                x: str
                y: int
                device: Any

        @tensorclass(shadow=True)
        class MyClass:
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

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("batched", [False, True])
    def test_shadow_state_dict(self, batched, nested):
        # The state dict must store the batch size and device of the
        # TensorDict, not the shadowed fields, and load the fields as entries.
        @tensorclass(shadow=True)
        class MyClass:
            x: torch.Tensor
            batch_size: Any
            device: Any
            name: str

        shape = (10,) if batched else ()
        src = MyClass(
            torch.zeros(10, 4),
            batch_size=torch.full(shape, 4),
            device=torch.zeros(shape),
            name="graph",
        )
        dst = MyClass(
            torch.ones(10, 4),
            batch_size=torch.ones(shape, dtype=torch.long),
            device=torch.ones(shape),
            name="other",
        )
        if batched:
            src._tensordict.batch_size = [10]
            src = src.to("cpu")
        if nested:
            src = TensorDict(c=src)
            dst = TensorDict(c=dst)
        sd = src.state_dict()
        metadata = sd._metadata["c" if nested else ""]
        assert metadata["batch_size"] == torch.Size(shape)
        assert metadata["device"] == (torch.device("cpu") if batched else None)
        dst.load_state_dict(sd)
        if nested:
            dst = dst["c"]
        assert (dst.x == 0).all()
        assert (dst.batch_size == 4).all()
        assert (dst.device == 0).all()
        assert dst.name == "graph"
        assert dst._tensordict.batch_size == torch.Size(shape)
        assert dst._tensordict.device is None

    def test_shadow_update_dict(self):
        # update_ and update_at_ with a dict must use the batch size of the
        # TensorDict, not the shadowed batch_size field.
        @tensorclass(shadow=True)
        class MyClass:
            x: torch.Tensor
            batch_size: Any
            device: Any

        c = MyClass(torch.zeros(10, 4), batch_size="abc", device="cpu")
        c._tensordict.batch_size = [10]
        c.update_({"x": torch.ones(10, 4)})
        assert (c.x == 1).all()
        c.update_at_({"x": torch.full((4,), 2.0)}, 0)
        assert (c.x[0] == 2).all()
        assert (c.x[1:] == 1).all()
        assert c.batch_size == "abc"
        assert c.device == "cpu"
        assert c._tensordict.batch_size == torch.Size([10])

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

    @set_printoptions(compact=False)
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
                class MyClass:
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


class TestRuntimeTypeCheckers:
    # A runtime type-checker that decorates a tensorclass (beartype, typeguard)
    # checks the methods in its namespace, and skips the functions marked with
    # __no_type_check__ (typing.no_type_check). Tensorclass marks the methods
    # that it adds: their annotations name builtins that the class shadows
    # (bool, int, type...).

    @staticmethod
    def _functions(value):
        if isinstance(value, property):
            return [f for f in (value.fget, value.fset, value.fdel) if f is not None]
        if isinstance(value, (classmethod, staticmethod)):
            value = value.__func__
        return [value] if inspect.isfunction(value) else []

    @pytest.mark.parametrize(
        "flags", [{}, {"tensor_only": True}, {"shadow": True, "frozen": True}]
    )
    @pytest.mark.parametrize("subclass", [False, True])
    def test_added_methods_are_not_type_checked(self, subclass, flags):
        if subclass:

            class Data(TensorClass, **flags):
                x: torch.Tensor

                def scaled(self, k: torch.Tensor) -> torch.Tensor:
                    return self.x * k

        else:

            @tensorclass(**flags)
            class Data:
                x: torch.Tensor

                def scaled(self, k: torch.Tensor) -> torch.Tensor:
                    return self.x * k

        unmarked = [
            name
            for name, value in vars(Data).items()
            if name not in ("scaled", "__init__")
            for func in self._functions(value)
            if func.__annotations__ and not getattr(func, "__no_type_check__", False)
        ]
        assert not unmarked
        # The methods of the class body are checked, and so is __init__, which has
        # the annotations of the fields.
        assert not hasattr(Data.scaled, "__no_type_check__")
        assert not hasattr(Data.__init__, "__no_type_check__")
        assert get_type_hints(Data.__init__)["x"] is torch.Tensor
        # Tensorclasses get copies of the TensorDict methods that they reuse, which
        # pickle by reference as these methods do
        for name in ("save", "memmap", "load_memmap"):
            assert not hasattr(getattr(TensorDict, name), "__no_type_check__")
        assert get_type_hints(TensorDict.save)["copy_existing"] is bool
        for name in ("save", "__iter__"):
            method = getattr(Data, name)
            assert pickle.loads(pickle.dumps(method)) is method

    def test_method_annotations_name_shadowed_builtins(self):
        # On Python 3.14+, without `from __future__ import annotations` (which this
        # module has), a method's annotations are evaluated when first read, in the
        # namespace of the class, where a tensorclass has methods named int and set.
        namespace = {"__name__": __name__}
        code = """
            import torch
            from tensordict import tensorclass

            @tensorclass
            class Data:
                x: torch.Tensor

                def count(self, k: int) -> int:
                    return k

                def tags(self) -> set[str]:
                    return {"x"}
        """
        exec(
            compile(textwrap.dedent(code), "<test>", "exec", dont_inherit=True),
            namespace,
        )
        Data = namespace["Data"]
        assert get_type_hints(Data.count) == {"k": int, "return": int}
        assert inspect.signature(Data.tags).return_annotation == set[str]


class TestDeprecations:
    @pytest.mark.parametrize("subclass", [False, True])
    def test_fields(self, subclass):
        if subclass:

            class MyClass(TensorClass):
                x: torch.Tensor

        else:

            @tensorclass
            class MyClass:
                x: torch.Tensor

        c = MyClass(x=torch.zeros(3), batch_size=[3])
        for obj in (MyClass, c):
            with pytest.warns(
                DeprecationWarning, match=r"fields\(\) .* removed in TensorDict 0\.17"
            ) as record:
                assert obj.fields() == dataclasses.fields(MyClass)
            assert record[0].filename == __file__

    def test_extend(self):
        @tensorclass
        class MyClass:
            x: torch.Tensor

        c = MyClass(x=torch.zeros(3), batch_size=[3])
        stack = lazy_stack([c, c])
        with pytest.warns(
            DeprecationWarning, match=r"extend\(\) .* removed in TensorDict 0\.17"
        ) as record:
            stack.extend(lazy_stack([c])._tensordict)
        assert record[0].filename == __file__
        assert stack.batch_size == torch.Size([3, 3])

    @pytest.mark.parametrize(
        "args,kwargs", [((), {"safe": True}), ((None, False), {})], ids=["kw", "pos"]
    )
    def test_from_tensordict_safe(self, args, kwargs):
        @tensorclass
        class MyClass:
            x: torch.Tensor

        td = TensorDict(x=torch.zeros(3), batch_size=[3])
        with pytest.warns(
            DeprecationWarning, match=r"safe=\.\.\.\) .* removed in TensorDict 0\.17"
        ) as record:
            c = MyClass.from_tensordict(td, *args, **kwargs)
        assert record[0].filename == __file__
        assert c._tensordict is td

    @pytest.mark.parametrize(
        "base",
        [None, TensorClass, TensorClass["tensor_only"]],
        ids=["decorator", "subclass", "tensor_only"],
    )
    def test_from_dict_defaults(self, base):
        annotations = {"a": torch.Tensor, "b": torch.Tensor, "t": torch.Tensor}
        namespace = {
            "__annotations__": annotations,
            "b": dataclasses.field(default_factory=lambda: torch.ones(2)),
            "t": None,
        }
        if base is None:
            annotations["s"] = str
            namespace["s"] = "hi"
            MyClass = tensorclass(type("MyClass", (), namespace))
        else:
            MyClass = type("MyClass", (base,), namespace)
        defaults = {"b"} if base is not None else {"b", "s"}
        a = torch.zeros(2)
        with pytest.warns(
            FutureWarning, match=r"Starting with TensorDict 0\.17, the default"
        ) as record:
            c = MyClass.from_dict({"a": a}, batch_size=[2])
        assert record[0].filename == __file__
        assert all(repr(key) in str(record[0].message) for key in defaults)
        assert all(getattr(c, key) is None for key in (*defaults, "t"))

        # no warning when the result stays the same in 0.17
        full = {
            "a": a,
            **{key: getattr(MyClass(a=a, batch_size=[2]), key) for key in defaults},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # every field with a non-None default is given; t defaults to None
            assert MyClass.from_dict(full, batch_size=[2]).t is None
            # a field without a default stays None
            assert (
                MyClass.from_dict(
                    {k: v for k, v in full.items() if k != "a"}, batch_size=[2]
                ).a
                is None
            )
            # the update methods keep the values of the instance
            c = MyClass(a=a, b=torch.full((2,), 3.0), batch_size=[2])
            c.update({"a": torch.ones(2)})
            c.update_({"a": torch.ones(2)})
            c.update_at_({"a": torch.ones(())}, 0)
            assert (c.b == 3).all()

    @pytest.mark.parametrize(
        "annotation,value",
        [
            (int, 3),
            (int, True),
            (float, 1.5),
            (float, 1),
            pytest.param(float, np.float64(1.5), id="float-np.float64"),
            (bool, False),
            (Optional[int], 3),
            (Union[int, float], 1.5),
        ],
    )
    def test_scalar_field(self, annotation, value):
        MyClass = type(
            "MyClass", (TensorClass,), {"__annotations__": {"a": annotation}}
        )
        match = r"Starting with TensorDict 0\.17, the default will change to keeping"
        with pytest.warns(FutureWarning, match=match) as record:
            c = MyClass(a=value)
        assert record[0].filename == __file__
        assert c.a.shape == () and c.a == value
        with pytest.warns(FutureWarning, match=match) as record:
            c.a = value
        assert record[0].filename == __file__

        # with a non-empty batch size, where a 0-d tensor raises, the value is
        # kept already
        c = MyClass(a=value, batch_size=[2])
        assert type(c.a) is type(value) and c.a == value
        assert isinstance(c._tensordict.get("a"), NonTensorData)
        c.a = value
        assert type(c[1].a) is type(value) and c[1].a == value

    def test_scalar_field_default(self):
        class MyClass(TensorClass):
            x: torch.Tensor
            n: int = 3

        with pytest.warns(FutureWarning, match="Starting with TensorDict 0.17"):
            assert MyClass(x=torch.zeros(())).n == torch.tensor(3)
        assert MyClass(x=torch.zeros(4), batch_size=[4]).n == 3

    @pytest.mark.parametrize("entry", ["tensor", "non_tensor"])
    @pytest.mark.parametrize("op", ["set_inplace", "set_", "update", "update_"])
    def test_scalar_field_inplace_raises(self, entry, op):
        # A write in place raises as before, with a non-empty batch size
        class MyClass(TensorClass):
            a: int

        if entry == "tensor":
            a = torch.zeros(2, dtype=torch.long)
        else:
            a = NonTensorData(1, batch_size=[2])
        c = MyClass(a=a, batch_size=[2])
        with pytest.raises(RuntimeError) as err:
            if op == "set_inplace":
                c.set("a", 3, inplace=True)
            elif op == "set_":
                c.set_("a", 3)
            elif op == "update":
                c.update({"a": 3}, inplace=True)
            else:
                c.update_({"a": 3})
        # the exception of the TensorDict, as before #1199
        assert type(err.value) is RuntimeError
        assert str(err.value) == (
            "batch dimension mismatch, got self.batch_size=torch.Size([2]) and "
            "value.shape=torch.Size([]) for key 'a'. The leading dimensions of "
            "a value must match the batch size. Python scalars are stored as "
            "0-dim tensors: to store a scalar in a tensordict with a non-empty "
            "batch size, expand it to the batch size or wrap it in NonTensorData "
            "(in a tensorclass, the nocast option stores scalars as they are)."
        )

    @pytest.mark.parametrize("entry", ["tensor", "non_tensor"])
    @pytest.mark.parametrize("call", ["dict", "kwargs"])
    def test_scalar_field_update(self, entry, call):
        # update() keeps a number given to a scalar field as setattr does, with a
        # non-empty batch size; it raised the batch-size mismatch before
        class MyClass(TensorClass):
            a: int
            x: torch.Tensor

        def make():
            if entry == "tensor":
                a = torch.zeros(2, dtype=torch.long)
            else:
                a = NonTensorData(1, batch_size=[2])
            return MyClass(a=a, x=torch.zeros(2), batch_size=[2])

        c = make()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            if call == "dict":
                out = c.update({"a": 3, "x": torch.ones(2)})
            else:
                out = c.update(a=3, x=torch.ones(2))
        assert out is c
        assert isinstance(c._tensordict.get("a"), NonTensorData)
        assert c.a == 3
        assert (c.x == 1).all()
        d = make()
        d.a = 3
        assert type(d._tensordict.get("a")) is type(c._tensordict.get("a"))
        assert d.a == c.a

    def test_scalar_field_update_keys_to_update(self):
        class MyClass(TensorClass):
            a: int
            x: torch.Tensor

        c = MyClass(a=1, x=torch.zeros(2), batch_size=[2])
        c.update({"a": 3, "x": torch.ones(2)}, keys_to_update=["x"])
        assert c.a == 1
        assert (c.x == 1).all()
        c.update({"a": 4}, keys_to_update=[("a",)])
        assert c.a == 4

    def test_scalar_field_over_non_tensor(self):
        # over a NonTensorData entry, only the warning of TensorDict.set (#2432)
        class MyClass(TensorClass):
            n: int

        c = MyClass(n=NonTensorData(3))
        with pytest.warns(FutureWarning) as record:
            c.n = 4
        assert len(record) == 1
        assert "written over the NonTensorData entry" in str(record[0].message)
        assert c.n == torch.tensor(4)

    def test_scalar_field_unchanged(self):
        # no warning and no change: a value of another type, a field that is not
        # annotated int, float or bool, and the classes that cast by annotation
        class MyClass(TensorClass):
            b: bool = 1
            t: torch.Tensor = 1
            x: Any = 1
            o: Optional[int] = None

        class NoCast(TensorClass["nocast"]):
            a: int = 1

        class AutoCast(TensorClass["autocast"]):
            a: int = 1

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            c = MyClass()
            assert all(isinstance(getattr(c, k), torch.Tensor) for k in "btx")
            assert c.o is None
            assert NoCast().a == 1 and AutoCast().a == 1
            assert NoCast(batch_size=[2]).a == 1 and AutoCast(batch_size=[2]).a == 1
            with pytest.raises(RuntimeError, match="batch dimension mismatch"):
                MyClass(t=torch.zeros(2), x=torch.zeros(2), batch_size=[2])

    # The library code that builds and converts tensorclasses does not call the
    # deprecated members.
    @pytest.mark.parametrize(
        "base", [None, TensorClass, TensorClass["tensor_only"], TensorClass["frozen"]]
    )
    def test_no_warning_internally(self, base):
        from tensordict.nn.tensorclass_module import _tensor_class_keys

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            annotations = {"x": torch.Tensor, "y": torch.Tensor}
            if base is None:
                MyClass = tensorclass(
                    type("MyClass", (), {"__annotations__": annotations})
                )
            else:
                MyClass = type("MyClass", (base,), {"__annotations__": annotations})
            td = TensorDict(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
            c = MyClass.from_tensordict(td)
            MyClass._from_tensordict(td, safe=False)
            MyClass.from_dict(td.to_dict(), auto_batch_size=True)
            torch.stack([c, c])[0].clone()
            assert _tensor_class_keys(MyClass) == [("x",), ("y",)]


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

    @pytest.mark.parametrize("operand", [None, "abc"])
    def test_logic_unsupported_operand(self, operand):
        # ^ and | with an operand that is not a tensor, number, dict or tensor
        # collection return False and warn that they return NotImplemented
        # from 0.17, as on TensorDict (#2979)
        @tensorclass
        class MyClass:
            x: torch.Tensor

        c = MyClass(torch.zeros(3, dtype=torch.bool), batch_size=[3])
        for op in ("__xor__", "__or__"):
            with pytest.warns(FutureWarning, match=f"{op} got an operand") as record:
                assert getattr(c, op)(operand) is False
            record = [w for w in record if "NotImplemented" in str(w.message)]
            assert len(record) == 1
            assert record[0].filename == __file__
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (c == operand) is False
            assert (c != operand) is True
            zeros = torch.zeros(3, dtype=torch.bool)
            for other in (c, c._tensordict, {"x": zeros}, zeros, True):
                assert type(c ^ other) is type(c | other) is MyClass


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
        with pytest.warns(FutureWarning, match="keeping the Python value"):
            obj = SubClass(a=0)
        assert is_called
        is_called = False
        with pytest.warns(FutureWarning, match="keeping the Python value"):
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

    def test_subclassing_redeclared_field(self):
        # The field getters of the base raise on the subclass, so a subclass
        # that declares a field again does not take a default from its base.
        class Base(TensorClass):
            a: torch.Tensor
            s: str

        class SubClass(Base):
            a: torch.Tensor
            t: str

        assert all(
            field.default is dataclasses.MISSING
            for field in dataclasses.fields(SubClass)
        )
        obj = SubClass(a=torch.zeros(3), s="s", t="t", batch_size=[3])
        assert (obj.a == 0).all()
        assert obj.s == "s"
        assert obj.t == "t"

    def test_field_class_access(self):
        # gh-3162: a field read on its class returns its getter, as a property
        # does, so code that reads every name of vars(cls) on the class, such
        # as Lightning's apply_to_collection, works.
        class Base(TensorClass):
            a: torch.Tensor
            s: str

        class SubClass(Base):
            a: torch.Tensor
            t: str

        for cls in (Base, SubClass):
            for name in vars(cls):
                getattr(cls, name)
            assert cls.a is vars(cls)["a"]
        assert SubClass.a is not Base.a

        # On a class that does not define the field, the getter of the parent
        # raises, as before: dataclass() would take it as the default of a
        # field that the class declares again.
        @tensorclass
        class Parent:
            a: torch.Tensor

        class Plain(Parent):
            pass

        assert hasattr(Parent, "a")
        assert not hasattr(Plain, "a")

    @pytest.mark.parametrize("api", ["decorator", "inheritance"])
    def test_dataclass_after_tensorclass(self, api):
        # dataclass() applied to a tensorclass takes no field getter as a
        # default (gh-3162). The getter relies on dataclass() replacing
        # __dataclass_params__ before it reads the defaults: this test fails if
        # a Python release changes that.
        namespace = {
            "__annotations__": {"a": torch.Tensor, "b": torch.Tensor},
            "b": None,
        }
        if api == "inheritance":
            LateDataclass = type("LateDataclass", (TensorClass,), namespace)
        else:
            LateDataclass = tensorclass(type("LateDataclass", (), namespace))
        LateDataclass = dataclasses.dataclass(LateDataclass)
        assert all(
            field.default is dataclasses.MISSING
            for field in dataclasses.fields(LateDataclass)
        )
        obj = LateDataclass(a=torch.zeros(3), batch_size=[3])
        assert (obj.a == 0).all()
        assert obj.b is None
        with pytest.raises(TypeError, match="missing"):
            LateDataclass(batch_size=[3])
        # The tensorclass serves the fields again after a second tensorclass()
        LateDataclass = tensorclass(LateDataclass)
        assert LateDataclass.a is vars(LateDataclass)["a"]
        assert (LateDataclass(a=torch.ones(3), batch_size=[3]).a == 1).all()

    def test_subclassing_setattr_object_setattr(self):
        # A custom __setattr__ that calls object.__setattr__ stores the fields
        # in the instance __dict__, from which they are read.
        class Base(TensorClass):
            a: torch.Tensor
            s: str

        class SubClass(Base):
            def __setattr__(self, key, value):
                object.__setattr__(self, key, value)

        obj = SubClass(a=torch.zeros(3), s="s", batch_size=[3])
        assert (obj.a == 0).all()
        assert obj.s == "s"
        obj.s = "u"
        assert obj.s == "u"

    _OPERATORS = (
        "__repr__",
        "__eq__",
        "__ne__",
        "__or__",
        "__xor__",
        "__bool__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__ror__",
    )

    @pytest.mark.parametrize("api", ["decorator", "dataclass", "inheritance"])
    def test_subclassing_keeps_operators(self, api):
        # gh-2602: a tensorclass keeps the operators that its body or a parent
        # defines, as dataclasses keeps __repr__ and __eq__.
        def make_operator(name):
            def operator(self, *args):
                return False if name == "__bool__" else f"custom {name}"

            return operator

        operators = {name: make_operator(name) for name in self._OPERATORS}
        namespace = {"__annotations__": {"x": torch.Tensor}, **operators}
        if api == "inheritance":
            Parent = type("Parent", (TensorClass,), namespace)
            Child = type("Child", (Parent,), {})
        else:
            Parent = type("Parent", (), namespace)
            if api == "dataclass":
                Parent = dataclasses.dataclass(Parent)
            Parent = tensorclass(Parent)
            Child = tensorclass(type("Child", (Parent,), {}))
        for cls in (Parent, Child):
            for name, operator in operators.items():
                assert getattr(cls, name) is operator, name
            obj = cls(x=torch.zeros(2), batch_size=[2])
            assert repr(obj) == "custom __repr__"
            assert (obj == obj) == "custom __eq__"
            assert (obj < obj) == "custom __lt__"
            assert (1 | obj) == "custom __ror__"
            assert not obj

    @pytest.mark.parametrize("api", ["decorator", "dataclass", "inheritance"])
    @set_printoptions(compact=False)
    def test_subclassing_default_operators(self, api):
        if api == "inheritance":

            class Parent(TensorClass):
                x: torch.Tensor

        else:

            class Parent:
                x: torch.Tensor

            if api == "dataclass":
                Parent = dataclasses.dataclass(Parent)
            Parent = tensorclass(Parent)

        class Child(Parent):
            pass

        if api != "inheritance":
            Child = tensorclass(Child)
        for cls in (Parent, Child):
            obj = cls(x=torch.zeros(2, dtype=torch.bool), batch_size=[2])
            assert repr(obj).startswith(f"{cls.__name__}(\n    x=Tensor(")
            for result in (obj == obj, obj != obj, obj | obj, obj ^ obj, obj < obj):
                assert type(result) is cls
            assert (obj == obj).x.all()
            assert not (obj < obj).x.any()
            with pytest.raises(RuntimeError):
                bool(obj)

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

    # Regression test for GitHub issue #2601: the metaclass __getitem__ hid a
    # __class_getitem__ defined by a subclass.
    def test_subclassing_class_getitem(self):
        class Shape(TensorClass):
            x: torch.Tensor

            @classmethod
            def __class_getitem__(cls, item):
                return (cls, item)

        class SubShape(Shape):
            y: torch.Tensor

        class Super(TensorClass):
            x: torch.Tensor

            def __class_getitem__(cls, item):
                return super().__class_getitem__(item)

        for cls in (Shape, SubShape):
            assert cls[2, 3] == (cls, (2, 3))
            assert cls[2] == (cls, 2)
            assert cls[(2,)] == (cls, (2,))
            assert cls[...] == (cls, ...)
            # flags are handled first, as before
            NoCast = cls["nocast"]
            assert issubclass(NoCast, cls) and NoCast._nocast

        assert Super["nocast"]._nocast
        assert Super.__class_getitem__("nocast")._nocast
        with pytest.raises(TypeError, match="only accepts the flags"):
            Super[2]

        class Plain(TensorClass):
            x: torch.Tensor

        with pytest.raises(TypeError, match="Plain is not a generic class"):
            Plain[2, 3]

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
    def test_custom_init_dict_write(self, tensor_only):
        # A field written straight into the instance __dict__ moves into the
        # tensordict too, also when the field is a property (tensor_only).
        class Data(TensorClass, tensor_only=tensor_only):
            x: torch.Tensor

            def __init__(self, x):
                self.__dict__["x"] = x * 2

        x = torch.ones(2)
        data = Data(x, batch_size=[2])
        torch.testing.assert_close(data.x, x * 2)
        assert "x" not in data.__dict__

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
        # A deleted field is None, as one that was never set
        delattr(x, "c")
        assert x.c is None
        assert x.to_dict()["c"] is None

    def test_tensor_only_assign_none(self):
        # Assigning None clears a field that holds a tensor, as set() does.
        x = self.TensorOnly(
            torch.zeros(3), torch.zeros(3), torch.ones(3), batch_size=[3]
        )
        x.c = None
        assert x.c is None
        assert x.clone().c is None
        assert x[0].c is None
        assert x.to_dict()["c"] is None
        # On a locked instance, it raises and keeps the tensor.
        x.c = torch.ones(3)
        x.lock_()
        with pytest.raises(RuntimeError, match="Cannot modify locked TensorDict"):
            x.c = None
        assert (x.c == 1).all()
        assert (x.to_dict()["c"] == 1).all()

    @staticmethod
    def _check_none_is_missing_key(x, key):
        # A None field of a tensor_only instance is a key missing from its
        # TensorDict, and _non_tensordict holds no placeholder for it
        assert getattr(x, key) is None
        assert key not in x._tensordict.keys()
        assert key not in x._non_tensordict

    @set_printoptions(compact=False)
    def test_tensor_only_none_is_missing_key(self):
        x = self.TensorOnly(torch.zeros(3), torch.zeros(3), batch_size=[3])
        y = self.TensorOnly(
            torch.zeros(3), torch.zeros(3), torch.ones(3), batch_size=[3]
        )
        y.c = None
        z = self.TensorOnly(
            torch.zeros(3), torch.zeros(3), torch.ones(3), batch_size=[3]
        )
        z.set("c", None)
        results = [
            x,
            y,
            z,
            x.clone(),
            x[0],
            x[:2],
            torch.stack([x, x]),
            lazy_stack([x, x]),
            x.to("cpu"),
            x.apply(lambda v: v + 1),
            x.exclude("a"),
            x.select("a", "b"),
            x.unbind(0)[0],
            pickle.loads(pickle.dumps(x)),
            self.TensorOnly.from_tensordict(
                TensorDict(a=torch.zeros(3), b=torch.zeros(3), batch_size=[3])
            ),
        ]
        for result in results:
            self._check_none_is_missing_key(result, "c")
        # The outputs list the None field, as they did with a placeholder
        assert x.to_dict()["c"] is None
        assert "c=None" in repr(x)
        assert x.non_tensor_items() == (("c", None),)
        assert x.state_dict()._metadata[""]["_non_tensor"] == {"c": None}
        assert isinstance(x.to_tensordict(retain_none=True).get("c"), NonTensorData)
        assert "c" not in x.to_tensordict(retain_none=False).keys()
        td = TensorDict(
            a=torch.zeros(3), b=torch.zeros(3), c=torch.ones(3), batch_size=[3]
        )
        assert (x == td).all()
        assert not (x != td).any()

    def test_tensor_only_del_none(self):
        # A deleted field is None, whatever it held before
        x = self.TensorOnly(
            torch.zeros(3), torch.zeros(3), torch.ones(3), batch_size=[3]
        )
        x.del_("c")
        self._check_none_is_missing_key(x, "c")
        x.del_("c")
        self._check_none_is_missing_key(x, "c")
        y = self.TensorOnly(torch.zeros(3), torch.zeros(3), batch_size=[3])
        y.del_("c")
        self._check_none_is_missing_key(y, "c")
        with pytest.raises(KeyError):
            y.del_("d")

    def test_tensor_only_dict_none(self):
        # from_dict and update read a None value as None, as a regular class does
        x = self.TensorOnly.from_dict(
            {"a": torch.zeros(3), "b": torch.zeros(3), "c": None}, batch_size=[3]
        )
        self._check_none_is_missing_key(x, "c")
        y = self.TensorOnly.from_dict(x.to_dict(), batch_size=[3])
        self._check_none_is_missing_key(y, "c")
        for update, warns in (
            (lambda z: z.update({"c": None}), False),
            (lambda z: z.update(c=None), False),
            # update_at_ unsets the field whatever the index, and warns
            (lambda z: z.update_at_({"c": None}, 0), True),
        ):
            z = self.TensorOnly(
                torch.zeros(3), torch.zeros(3), torch.ones(3), batch_size=[3]
            )
            with (
                pytest.warns(FutureWarning, match="whatever the index")
                if warns
                else contextlib.nullcontext()
            ):
                update(z)
            self._check_none_is_missing_key(z, "c")
        # keys_to_update still selects the entries
        z = self.TensorOnly(
            torch.zeros(3), torch.zeros(3), torch.ones(3), batch_size=[3]
        )
        z.update({"a": torch.ones(3), "c": None}, keys_to_update=["a"])
        assert (z.a == 1).all()
        assert (z.c == 1).all()

    def test_tensor_only_lazy_stack_none(self):
        x = self.TensorOnly(
            torch.zeros(3), torch.zeros(3), torch.ones(3), batch_size=[3]
        )
        y = self.TensorOnly(torch.zeros(3), torch.zeros(3), batch_size=[3])
        # Building the stack leaves its inputs as they are
        stack = lazy_stack([x, y])
        assert isinstance(x.to_dict()["c"], torch.Tensor)
        assert x._non_tensordict == {}
        assert stack[1].c is None
        # Setting None clears the field in every member
        for clear in (
            lambda s: setattr(s, "c", None),
            lambda s: s.set("c", None),
            lambda s: s.del_("c"),
        ):
            x = self.TensorOnly(
                torch.zeros(3), torch.zeros(3), torch.ones(3), batch_size=[3]
            )
            y = self.TensorOnly(
                torch.zeros(3), torch.zeros(3), torch.ones(3), batch_size=[3]
            )
            stack = lazy_stack([x, y])
            clear(stack)
            assert stack.c is None
            self._check_none_is_missing_key(x, "c")
            self._check_none_is_missing_key(y, "c")

    def test_tensor_only_custom_init_none(self):
        class Data(TensorClass["tensor_only"]):
            a: torch.Tensor
            b: torch.Tensor
            c: torch.Tensor | None = None

            def __init__(self, a, b=None, set_b=True):
                self.a = a
                if set_b:
                    self.b = b

        # A field without a default that __init__ sets to None is None
        x = Data(torch.zeros(3))
        self._check_none_is_missing_key(x, "b")
        self._check_none_is_missing_key(x, "c")
        assert (Data(torch.zeros(3), torch.ones(3)).b == 1).all()
        # One that __init__ does not set still raises
        with pytest.raises(TypeError, match="did not set the required field 'b'"):
            Data(torch.zeros(3), set_b=False)

        class Updated(TensorClass["tensor_only"]):
            a: torch.Tensor
            b: torch.Tensor

            def __init__(self, a):
                self.a = a
                self.update({"b": None})

        self._check_none_is_missing_key(Updated(torch.zeros(3)), "b")

    def test_tensor_only_load_none_placeholders(self, tmp_path):
        # Earlier versions kept a None placeholder in _non_tensordict, and
        # wrote it in pickles and memmaps
        x = self.TensorOnly(torch.zeros(3), torch.zeros(3), batch_size=[3])
        state = x.__getstate__()
        assert state["non_tensordict"] == {"c": None}
        y = self.TensorOnly.__new__(self.TensorOnly)
        y.__setstate__(
            {"tensordict": x._tensordict.clone(), "non_tensordict": {"c": None}}
        )
        self._check_none_is_missing_key(y, "c")
        y.c = torch.ones(3)
        assert (y.to_dict()["c"] == 1).all()
        x.memmap(tmp_path)
        meta = json.loads((tmp_path / "meta.json").read_text())
        meta["c"] = None
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        self._check_none_is_missing_key(self.TensorOnly.load_memmap(tmp_path), "c")

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

    def test_tensor_only_setattr_mapping(self):
        class TensorOnlyMapping(TensorClass["tensor_only"]):
            data: TensorDict
            meta: NonTensorData
            x: torch.Tensor

        x = torch.zeros(2)
        tc = TensorOnlyMapping(data={}, meta="a", x=x, batch_size=[2])
        tc.data = UserDict({"nested": UserDict({"tensor": x})})
        assert isinstance(tc.data["nested"], TensorDict)
        assert tc.data["nested", "tensor"] is x
        assert tc.data.batch_size == (2,)
        tc.data = {}
        assert tc.data.is_empty() and tc.data.batch_size == (2,)
        # A Mapping in a field that is not a TensorDict is not normalized
        meta = UserDict({"metadata": "value"})
        tc.meta = meta
        assert tc.meta.data is meta
        y = torch.ones(2)
        tc.x = y
        assert tc.x is y
        with pytest.raises(RuntimeError, match="batch dimension mismatch"):
            tc.x = torch.ones(3)
        assert tc.x is y

    def test_non_tensor_data_set(self):
        data = NonTensorData("a", batch_size=[2])
        data.data = NonTensorData("b", batch_size=[2])
        assert data.data == "b"
        with pytest.raises(KeyError, match="only 'data' keys are supported"):
            data._set_str("other", "c", inplace=False, validated=False)
        with pytest.raises(KeyError, match="only 'data' keys are supported"):
            data._set_at_str("other", "c", 0, validated=False)
        assert data.data == "b"

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
            d: torch.Tensor | Union[torch.IntTensor, torch.LongTensor] | None = None
            e: Optional[torch.IntTensor] = None
            f: Optional[torch.IntTensor | None] = None
            g: TensorDict | None = None
            h: MyTensorClass | None = None

        with pytest.raises(
            TypeError,
            match="tensor_only requires types to be Tensor, Tensor-subtypes or None",
        ):

            class TensorOnlyAny(TensorClass["tensor_only"]):
                a: torch.Tensor
                b: Any
                c: torch.Tensor | None = None

        with pytest.raises(
            TypeError,
            match="tensor_only requires types to be Tensor, Tensor-subtypes or None",
        ):

            class TensorOnlyStr(TensorClass["tensor_only"]):
                a: torch.Tensor
                b: torch.Tensor | str
                c: torch.Tensor | None = None

        with pytest.raises(
            TypeError,
            match="tensor_only requires types to be Tensor, Tensor-subtypes or None",
        ):

            class TensorOnlyStrUnion(TensorClass["tensor_only"]):
                a: torch.Tensor
                b: torch.Tensor
                c: torch.Tensor | Union[torch.IntTensor, str] | None = None

    def test_tensor_only_shape_annotations(self):
        # A shape annotation counts as the array type that it wraps, and its
        # shape is not checked.
        class Shaped(TensorClass["tensor_only"]):
            a: _ShapedTensor
            b: _ShapedTensor | None = None

        x = Shaped(a=torch.zeros(2, 3), batch_size=[2])
        assert isinstance(x.a, torch.Tensor)
        assert x.b is None
        x.b = torch.zeros(2, 4)
        assert x.b.shape == (2, 4)

        with pytest.raises(
            TypeError,
            match="tensor_only requires types to be Tensor, Tensor-subtypes or None",
        ):

            class ShapedArray(TensorClass["tensor_only"]):
                a: _ShapedArray

    def test_tensor_only_array_type_attribute(self):
        # A tensorclass with an ``array_type`` attribute is not unwrapped.
        class Outer(TensorClass["tensor_only"]):
            inner: _WithArrayType

        x = Outer(
            inner=_WithArrayType(x=torch.zeros(2), batch_size=[2]), batch_size=[2]
        )
        assert isinstance(x.inner, _WithArrayType)

    @pytest.mark.skipif(not _has_jaxtyping, reason="jaxtyping not installed")
    @pytest.mark.parametrize("flag", ["tensor_only", "autocast"])
    def test_jaxtyping_annotations(self, flag):
        namespace = {"__name__": __name__}
        code = f"""
            import torch
            from jaxtyping import Float
            from tensordict import TensorClass

            class Points(TensorClass["{flag}"]):
                pos: Float[torch.Tensor, "*batch 3"]
                mass: Float[torch.Tensor, "*batch"] | None = None
        """
        exec(
            compile(textwrap.dedent(code), "<test>", "exec", dont_inherit=True),
            namespace,
        )
        points = namespace["Points"](pos=[[0.0, 1.0, 2.0]], batch_size=[1])
        assert isinstance(points.pos, torch.Tensor)
        assert points.mass is None

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
