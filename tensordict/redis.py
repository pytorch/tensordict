# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Backward-compatible re-exports from ``tensordict.store``.

This module exists so that ``from tensordict.redis import RedisTensorDict``
continues to work.  New code should import from ``tensordict.store`` instead.
"""

from tensordict.store import (  # noqa: F401
    LazyStackedTensorDictStore,
    LazyStackedTensorDictStore as RedisLazyStackedTensorDict,
    TensorDictStore,
    TensorDictStore as RedisTensorDict,
)

__all__ = [
    "RedisTensorDict",
    "RedisLazyStackedTensorDict",
    "TensorDictStore",
    "LazyStackedTensorDictStore",
]
