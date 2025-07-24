# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypeVar

T = TypeVar("T", bound="TensorCollection")


class TensorCollection:
    """A base class for TensorDictBase and TensorClass."""

    ...
