# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deprecated module: the distributed-test fixtures moved to ``tensordict._testing``."""

from __future__ import annotations

from tensordict import _deprecation, _testing

__getattr__ = _deprecation.deprecated_attributes(
    __name__, {"MyDistData": (_testing.MyDistData, None)}, removal="0.17"
)
