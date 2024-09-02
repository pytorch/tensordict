# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Tuple, type_check_only

NestedKey = type_check_only(str | Tuple["NestedKeyType", ...])
