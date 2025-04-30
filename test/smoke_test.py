# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
import sys
from pathlib import Path

_IS_LINUX = sys.platform.startswith("linux")


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


def test_static_linking():
    if not _IS_LINUX:
        return
    # Locate _C.so
    try:
        import tensordict._C
    except ImportError as e:
        raise RuntimeError(f"Failed to import tensordict._C: {e}")
    # Get the path to _C.so
    _C_path = Path(tensordict._C.__file__)
    if not _C_path.exists():
        raise RuntimeError(f"_C.so not found at {_C_path}")
    # Run ldd on _C.so
    try:
        output = subprocess.check_output(["ldd", str(_C_path)]).decode("utf-8")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run ldd on {_C_path}: {e}")
    # Check if libpython is dynamically linked
    for line in output.splitlines():
        if "libpython" in line and "=>" in line and "not found" not in line:
            raise RuntimeError(f"tensordict/_C.so is dynamically linked against {line.strip()}")
    print("Test passed: tensordict/_C.so does not show dynamic linkage to libpython.")


if __name__ == "__main__":
    test_imports_deps()
    test_imports()
