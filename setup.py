# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import distutils.command.clean
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension,
)

cwd = os.path.dirname(os.path.abspath(__file__))
version_txt = os.path.join(cwd, "version.txt")
with open(version_txt, "r") as f:
    version = f.readline().strip()


ROOT_DIR = Path(__file__).parent.resolve()


try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
except Exception:
    sha = "Unknown"
package_name = "tensordict"

if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
elif sha != "Unknown":
    version += "+" + sha[:7]


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tensordict setup")
    parser.add_argument(
        "--package_name",
        type=str,
        default="tensordict",
        help="the name of this output wheel",
    )
    return parser.parse_known_args(argv)


def write_version_file():
    version_path = os.path.join(cwd, "tensordict", "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))


def _get_pytorch_version():
    # if "PYTORCH_VERSION" in os.environ:
    #     return f"torch=={os.environ['PYTORCH_VERSION']}"
    return "torch"


def _get_packages():
    exclude = [
        "build*",
        "test*",
        "third_party*",
        "tools*",
    ]
    return find_packages(exclude=exclude)


ROOT_DIR = Path(__file__).parent.resolve()


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove tensordict extension
        for path in (ROOT_DIR / "tensordict").glob("**/*.so"):
            print(f"removing '{path}'")
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


def _main(argv):
    args, unknown = parse_args(argv)
    name = args.package_name
    pytorch_package_dep = _get_pytorch_version()
    print("-- PyTorch dependency:", pytorch_package_dep)

    this_directory = Path(__file__).parent
    long_description = (this_directory / "README.md").read_text()
    sys.argv = [sys.argv[0]] + unknown

    setup(
        # Metadata
        name=name,
        version=version,
        author="tensordict contributors",
        author_email="vmoens@fb.com",
        url="https://github.com/pytorch-labs/tensordict",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="BSD",
        # Package info
        packages=find_packages(exclude=("test", "tutorials")),
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
        install_requires=[pytorch_package_dep, "numpy", "packaging", "cloudpickle"],
        extras_require={
            "tests": ["pytest", "pyyaml", "pytest-instafail"],
            "checkpointing": ["torchsnapshot-nightly"],
        },
        zip_safe=False,
    )


if __name__ == "__main__":

    write_version_file()
    print("Building wheel {}-{}".format(package_name, version))
    print(f"BUILD_VERSION is {os.getenv('BUILD_VERSION')}")
    _main(sys.argv[1:])
