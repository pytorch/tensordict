# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Configuration of the docstring examples: ``pytest --doctest-modules tensordict``."""

from __future__ import annotations

import importlib.util

import pytest

_has_zarr = importlib.util.find_spec("zarr") is not None


def pytest_collection_modifyitems(config, items):
    skip_zarr = pytest.mark.skip(reason="zarr is not installed")
    for item in items:
        dtest = getattr(item, "dtest", None)
        if dtest is None:
            continue
        # An example runs with the __future__ imports of its module. With
        # ``from __future__ import annotations``, the annotations of a
        # tensorclass defined in an example are strings, and they cannot be
        # resolved, since the classes they name are not module globals. Run the
        # examples as a user who pastes them would: without that import.
        dtest.globs["annotations"] = None
        if not _has_zarr and any(
            "zarr" in example.source for example in dtest.examples
        ):
            item.add_marker(skip_zarr)
