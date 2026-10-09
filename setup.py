# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent.resolve()
_RELEASE_BRANCH_RE = re.compile(r"^release/v(?P<release_id>.+)$")


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
        packages=find_packages(exclude=("test", "tutorials", "packaging", "docs")),
        **(
            {"setup_requires": ["setuptools_scm"], "use_scm_version": True}
            if _has_setuptools_scm
            # pretend_version already includes +g<sha> (computed in set_version)
            else {"version": pretend_version}
        ),
    )
