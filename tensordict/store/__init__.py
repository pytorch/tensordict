# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Key-value store backed TensorDict implementations (Redis, Dragonfly, etc.)."""

from tensordict.store._store import (
    LazyStackedTensorDictStore,
    STORE_BACKENDS,
    TensorDictStore,
)

__all__ = [
    "LazyStackedTensorDictStore",
    "STORE_BACKENDS",
    "TensorDictStore",
]
