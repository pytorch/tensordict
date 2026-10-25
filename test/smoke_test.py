# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path


def test_imports_deps():
    print("Importing numpy")  # noqa
    import numpy

    print("Importing torch")  # noqa
    import torch


def test_imports():
    print("Importing tensordict")  # noqa
    from tensordict import TensorDict

    print("Importing tensordict nn")  # noqa
    import tensordict
    from tensordict.nn import TensorDictModule

    print("version", tensordict.__version__)  # noqa


def test_nested_keys():
    from tensordict import TensorDict, unravel_key

    assert unravel_key(("a", ("b",))) == ("a", "b")
    td = TensorDict(a=TensorDict(b=1))
    assert td["a", "b"] == 1


def test_typed():
    # Type checkers such as mypy read the annotations and the stubs of an
    # installed package only if it ships the py.typed marker of PEP 561.
    import tensordict

    package = Path(tensordict.__file__).parent
    assert (package / "py.typed").is_file()
    assert (package / "tensorclass.pyi").is_file()


if __name__ == "__main__":
    test_imports_deps()
    test_imports()
    test_nested_keys()
    test_typed()
