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
from datetime import date
from pathlib import Path
from typing import List

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension

ROOT_DIR = Path(__file__).parent.resolve()

try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR)
        .decode("ascii")
        .strip()
    )
except Exception:
    sha = "Unknown"
package_name = "tensordict"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tensordict setup")
    parser.add_argument(
        "--package_name",
        type=str,
        default="tensordict",
        help="the name of this output wheel",
    )
    return parser.parse_known_args(argv)


def get_version():
    version = (ROOT_DIR / "version.txt").read_text().strip()
    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]
    return version


def get_nightly_version():
    return f"{date.today():%Y.%m.%d}"


def write_version_file(version):
    version_path = ROOT_DIR / "tensordict" / "version.py"
    with version_path.open("w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")


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


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove tensordict extension
        for path in (ROOT_DIR / "tensordict").glob("**/*.so"):
            print(f"removing '{path}'")
            path.unlink()
        # Remove build directory
        build_dirs = [ROOT_DIR / "build"]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


def _main(argv):
    args, unknown = parse_args(argv)
    name = args.package_name
    is_nightly = "nightly" in name

    version = get_nightly_version() if is_nightly else get_version()

    write_version_file(version)
    print(f"Building wheel {package_name}-{version}")
    print(f"BUILD_VERSION is {os.getenv('BUILD_VERSION')}")

    pytorch_package_dep = _get_pytorch_version()
    print("-- PyTorch dependency:", pytorch_package_dep)

    long_description = (ROOT_DIR / "README.md").read_text()
    sys.argv = [sys.argv[0], *unknown]

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
        packages=find_packages(exclude=("test", "tutorials", "packaging", "gallery")),
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
        install_requires=[pytorch_package_dep, "numpy", "cloudpickle"],
        extras_require={
            "tests": ["pytest", "pyyaml", "pytest-instafail", "pytest-rerunfailures"],
            "checkpointing": ["torchsnapshot-nightly"],
        },
        zip_safe=False,
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Development Status :: 4 - Beta",
        ],
    )


if __name__ == "__main__":
    _main(sys.argv[1:])
