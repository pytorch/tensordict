# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import distutils.command.clean
import logging
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import List

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = Path(__file__).parent.resolve()


def get_python_executable():
    # Check if we're running in a virtual environment
    if "VIRTUAL_ENV" in os.environ:
        # Get the virtual environment's Python executable
        python_executable = os.path.join(os.environ["VIRTUAL_ENV"], "bin", "python")
    else:
        # Fall back to sys.executable
        python_executable = sys.executable
    return python_executable


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
    if os.getenv("TENSORDICT_BUILD_VERSION"):
        version = os.getenv("TENSORDICT_BUILD_VERSION")
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


def _get_pytorch_version(is_nightly, is_local):
    if "PYTORCH_VERSION" in os.environ:
        return f"torch=={os.environ['PYTORCH_VERSION']}"
    if is_nightly:
        return "torch>=2.7.0.dev"
    if is_local:
        return "torch"
    return "torch>=2.5.0"


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
            logging.info(f"removing '{path}'")
            path.unlink()
        # Remove build directory
        build_dirs = [ROOT_DIR / "build"]
        for path in build_dirs:
            if path.exists():
                logging.info(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={get_python_executable()}",
            f"-DPython3_EXECUTABLE={get_python_executable()}",
        ]
        CONDA_PREFIX = os.environ.get("CONDA_PREFIX")
        if CONDA_PREFIX:
            CMAKE_PREFIX_PATH = os.environ.get("CMAKE_PREFIX_PATH")
            cmake_args.append(f"-DCMAKE_PREFIX_PATH={CONDA_PREFIX}:{CMAKE_PREFIX_PATH}")
        build_args = []
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


def get_extensions():
    extensions_dir = os.path.join(ROOT_DIR, "tensordict", "csrc")
    return [CMakeExtension("tensordict._C", sourcedir=extensions_dir)]


def _main(argv):
    args, unknown = parse_args(argv)
    name = args.package_name
    is_nightly = "nightly" in name

    version = get_nightly_version() if is_nightly else get_version()

    write_version_file(version)
    logging.info(f"Building wheel {package_name}-{version}")
    BUILD_VERSION = os.getenv("TENSORDICT_BUILD_VERSION")
    logging.info(f"TENSORDICT_BUILD_VERSION is {BUILD_VERSION}")
    local_build = BUILD_VERSION is None

    pytorch_package_dep = _get_pytorch_version(is_nightly, local_build)
    logging.info("-- PyTorch dependency:", pytorch_package_dep)

    long_description = (ROOT_DIR / "README.md").read_text(encoding="utf8")
    sys.argv = [sys.argv[0], *unknown]

    setup(
        # Metadata
        name=name,
        version=version,
        author="tensordict contributors",
        author_email="vmoens@fb.com",
        url="https://github.com/pytorch/tensordict",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="BSD",
        # Package info
        packages=find_packages(
            exclude=("test", "tutorials", "packaging", "gallery", "docs")
        ),
        ext_modules=get_extensions(),
        cmdclass={
            "build_ext": CMakeBuild,
            "clean": clean,
        },
        install_requires=[
            pytorch_package_dep,
            "numpy",
            "cloudpickle",
            "orjson",
            "packaging",
        ],
        extras_require={
            "tests": [
                "pytest",
                "pyyaml",
                "pytest-instafail",
                "pytest-rerunfailures",
                "pytest-benchmark",
            ],
            "checkpointing": ["torchsnapshot-nightly"],
            "h5": ["h5py>=3.8"],
            "dev": ["pybind11", "cmake", "ninja"],
        },
        zip_safe=False,
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Development Status :: 4 - Beta",
        ],
    )


if __name__ == "__main__":
    _main(sys.argv[1:])
