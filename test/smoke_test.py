# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def test_imports_deps():
    print("Importing numpy")  # noqa
    import numpy  # noqa

    print("Importing torch")  # noqa
    import torch  # noqa


def test_imports():
    print("Importing tensordict")  # noqa
    from tensordict import TensorDict  # noqa: F401

    print("Importing tensordict nn")  # noqa
    import tensordict  # noqa
    from tensordict.nn import TensorDictModule  # noqa: F401

    print("version", tensordict.__version__)  # noqa


if __name__ == "__main__":
    test_imports_deps()
    test_imports()
