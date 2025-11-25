# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import distutils.command.clean
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

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
        is_editable = self.inplace
        if is_editable:
            # For editable installs, place the extension in the source directory
            extdir = os.path.abspath(os.path.join(ROOT_DIR, "tensordict"))
        else:
            # For regular installs, place the extension in the build directory
            extdir = os.path.abspath(os.path.join(self.build_lib, "tensordict"))

        # Try to import torch to obtain its CMake prefix for find_package(Torch)
        cmake_prefix = os.environ.get("CMAKE_PREFIX_PATH", "")
        try:
            import torch  # noqa: F401
            from torch.utils import cmake_prefix_path as torch_cmake_prefix_path  # type: ignore

            # Prepend Torch's cmake prefix so CMake can find Torch
            cmake_prefix = (
                f"{torch_cmake_prefix_path}:{cmake_prefix}" if cmake_prefix else torch_cmake_prefix_path
            )
        except Exception:
            # Torch not importable at build time; rely on environment CMAKE_PREFIX_PATH
            pass

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={get_python_executable()}",
            f"-DPython3_EXECUTABLE={get_python_executable()}",
            # for windows
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        if cmake_prefix:
            cmake_args.append(f"-DCMAKE_PREFIX_PATH={cmake_prefix}")

        build_args = []
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        if sys.platform == "win32":
            build_args += ["--config", "Release"]
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--verbose"] + build_args, cwd=self.build_temp
        )


def get_extensions():
    extensions_dir = os.path.join(ROOT_DIR, "tensordict", "csrc")
    return [CMakeExtension("tensordict._C", sourcedir=extensions_dir)]


@contextlib.contextmanager
def set_version():

    if "SETUPTOOLS_SCM_PRETEND_VERSION" not in os.environ:
        # grab version from local version.py
        sys.path.append(Path(__file__).parent)
        with open("version.txt", "r") as f:
            version_str = f.read()
            os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = version_str
        yield
        del os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"]
        return
    yield


with set_version():
    setup(
        ext_modules=get_extensions(),
        cmdclass={
            "build_ext": CMakeBuild,
            "clean": clean,
        },
        packages=find_packages(
            exclude=("test", "tutorials", "packaging", "gallery", "docs")
        ),
        setup_requires=["setuptools_scm"],
        use_scm_version=True,
    )
