# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={get_python_executable()}",
            f"-DPython3_EXECUTABLE={get_python_executable()}",
            # for windows
            "-DCMAKE_BUILD_TYPE=Release",
        ]

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


def version():
    return {
        "write_to": "tensordict/_version.py",  # Specify the path where the version file should be written
    }


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
    use_scm_version=version(),
)
