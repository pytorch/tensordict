# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import distutils.command.clean
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = Path(__file__).parent.resolve()
_RELEASE_BRANCH_RE = re.compile(r"^release/v(?P<release_id>.+)$")


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


def _git_output(args) -> str | None:
    try:
        return (
            subprocess.check_output(["git", *args], cwd=str(ROOT_DIR))
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _branch_name() -> str | None:
    for key in (
        "GITHUB_REF_NAME",
        "GIT_BRANCH",
        "BRANCH_NAME",
        "CI_COMMIT_REF_NAME",
    ):
        val = os.environ.get(key)
        if val:
            return val
    branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    if not branch or branch == "HEAD":
        return None
    return branch


def _short_sha() -> str | None:
    return _git_output(["rev-parse", "--short", "HEAD"])


def _version_with_local_sha(base_version: str) -> str:
    branch = _branch_name()
    if branch:
        m = _RELEASE_BRANCH_RE.match(branch)
        if m and m.group("release_id").strip() == base_version.strip():
            return base_version
    sha = _short_sha()
    if not sha:
        return base_version
    return f"{base_version}+g{sha}"


@contextlib.contextmanager
def set_version():

    if "SETUPTOOLS_SCM_PRETEND_VERSION" not in os.environ:
        # grab version from local version.txt
        sys.path.append(str(Path(__file__).parent))
        with open("version.txt", "r") as f:
            base_version = f.read().strip()
        # Compute full version with +g<sha> unless on release branch
        full_version = _version_with_local_sha(base_version)
        os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = full_version
        yield
        del os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"]
        return
    yield


with set_version():
    pretend_version = os.environ.get("SETUPTOOLS_SCM_PRETEND_VERSION")
    _has_setuptools_scm = importlib.util.find_spec("setuptools_scm") is not None

    # If users pass --no-build-isolation, pip will not install build requirements.
    # In that case, setuptools_scm may be absent and setuptools can fall back to
    # a bogus 0.0.0 version. Use version.txt explicitly when scm isn't available.
    if not _has_setuptools_scm and not pretend_version:
        raise RuntimeError(
            "tensordict requires setuptools_scm to build from a git checkout, "
            "unless SETUPTOOLS_SCM_PRETEND_VERSION is set. "
            "Install setuptools_scm or avoid --no-build-isolation."
        )

    setup(
        ext_modules=get_extensions(),
        cmdclass={
            "build_ext": CMakeBuild,
            "clean": clean,
        },
        packages=find_packages(
            exclude=("test", "tutorials", "packaging", "gallery", "docs")
        ),
        **(
            {"setup_requires": ["setuptools_scm"], "use_scm_version": True}
            if _has_setuptools_scm
            # pretend_version already includes +g<sha> (computed in set_version)
            else {"version": pretend_version}
        ),
    )
