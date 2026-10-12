# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The private helpers of the pandas, CSV, Parquet and JSON conversions.

They live in ``tensordict._tabular`` now. This module keeps
``tensordict.tabular`` importable until TensorDict 0.17.
"""

from __future__ import annotations

from typing import Any

from tensordict import _tabular
from tensordict._deprecation import warn_deprecated

warn_deprecated(
    "tensordict.tabular",
    removal="0.17",
    replacement="TensorDict.from_pandas, from_csv, from_parquet, from_json, "
    "to_pandas, to_csv, to_parquet and to_json",
)


def __getattr__(name: str) -> Any:
    return getattr(_tabular, name)
