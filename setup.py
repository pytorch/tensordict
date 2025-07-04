# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import distutils.command.clean
import logging
import os
import platform
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

from setuptools import Command, Extension, find_packages, setup
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


def check_cmake_version():
    """Check if CMake version is sufficient."""
    try:
        result = subprocess.run(
            ["cmake", "--version"], capture_output=True, text=True, check=True
        )
        version_line = result.stdout.split("\n")[0]
        version_str = version_line.split()[2]
        major, minor = map(int, version_str.split(".")[:2])
        if major < 3 or (major == 3 and minor < 18):
            warnings.warn(
                f"CMake version {version_str} may be too old. Recommended: 3.18+"
            )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        warnings.warn("Could not determine CMake version")
        return False


def is_apple_silicon():
    """Check if running on Apple Silicon (M1/M2)."""
    return (
        sys.platform == "darwin" 
        and platform.machine() in ("arm64", "aarch64")
    )


class clean(Command):
    """Custom clean command to remove tensordict extensions."""

    description = "remove tensordict extensions and build files"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
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
        # Check CMake version before building
        check_cmake_version()
        
        # Log architecture information
        if is_apple_silicon():
            print("Detected Apple Silicon (ARM64) architecture")
        
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
        
        # Add ARM64-specific arguments for Apple Silicon
        if is_apple_silicon():
            cmake_args.extend([
                "-DCMAKE_OSX_ARCHITECTURES=arm64",
            ])

        build_args = []
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        if sys.platform == "win32":
            build_args += ["--config", "Release"]

        try:
            subprocess.check_call(
                ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
            )
            subprocess.check_call(
                ["cmake", "--build", ".", "--verbose"] + build_args, cwd=self.build_temp
            )
        except subprocess.CalledProcessError as e:
            warnings.warn(
                f"Error building extension: {e}\n"
                "This might be due to missing dependencies or incompatible compiler. "
                "Please ensure you have CMake 3.18+ and a C++17 compatible compiler."
            )
            raise


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
