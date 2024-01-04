# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import distutils.command.clean
import glob
import logging
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import List

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

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


def _get_pytorch_version(is_nightly):
    # if "PYTORCH_VERSION" in os.environ:
    #     return f"torch=={os.environ['PYTORCH_VERSION']}"
    if is_nightly:
        return "torch>=2.2.0.dev"
    return "torch>=2.1.0"


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


def get_extensions():
    extension = CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-std=c++17",
            "-fdiagnostics-color=always",
        ]
    }
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        logging.info("Compiling in debug mode")
        extra_compile_args = {
            "cxx": [
                "-O0",
                "-fno-inline",
                "-g",
                "-std=c++17",
                "-fdiagnostics-color=always",
            ]
        }
        extra_link_args = ["-O0", "-g"]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "tensordict", "csrc")

    extension_sources = {
        os.path.join(extensions_dir, p)
        for p in glob.glob(os.path.join(extensions_dir, "*.cpp"))
    }
    sources = list(extension_sources)

    ext_modules = [
        extension(
            "tensordict._tensordict",
            sources,
            include_dirs=[this_dir],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


def _main(argv):
    args, unknown = parse_args(argv)
    name = args.package_name
    is_nightly = "nightly" in name

    version = get_nightly_version() if is_nightly else get_version()

    write_version_file(version)
    logging.info(f"Building wheel {package_name}-{version}")
    logging.info(f"BUILD_VERSION is {os.getenv('BUILD_VERSION')}")

    pytorch_package_dep = _get_pytorch_version(is_nightly)
    logging.info("-- PyTorch dependency:", pytorch_package_dep)

    long_description = (ROOT_DIR / "README.md").read_text()
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
        packages=find_packages(exclude=("test", "tutorials", "packaging", "gallery")),
        ext_modules=get_extensions(),
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
        install_requires=[pytorch_package_dep, "numpy", "cloudpickle"],
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
        },
        zip_safe=False,
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Development Status :: 4 - Beta",
        ],
    )


if __name__ == "__main__":
    _main(sys.argv[1:])
