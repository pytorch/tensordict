# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).parent

TEMPLATE = """from __future__ import annotations

{file_contents}
"""


def pytest_sessionstart(session):
    source = HERE / "test_tensorclass_nofuture.py"
    dest = HERE / "test_tensorclass.py"

    tensorclass_tests = source.read_text()
    # some error message checks refer to filename
    tensorclass_tests = tensorclass_tests.replace("tensorclass_nofuture", "tensorclass")

    lines = tensorclass_tests.split("\n")
    for i, line in enumerate(lines):
        if "# future: drop quotes" in line:
            lines[i] = line.replace('"', "")
    tensorclass_tests = "\n".join(lines)

    dest.touch()
    dest.write_text(TEMPLATE.format(file_contents=tensorclass_tests))


def pytest_sessionfinish(session):
    (HERE / "test_tensorclass.py").unlink()


@pytest.fixture
def tmpdir():
    # creates a pathlib.Path pointing to a temporary directory that exists for the
    # duration of the test
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)
