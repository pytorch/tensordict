# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The nested-key helpers that used to be a C++ extension.

They are written in Python in :mod:`tensordict.utils` now. This module keeps
``tensordict._C`` importable until TensorDict 0.17.
"""

from __future__ import annotations

from tensordict._deprecation import warn_deprecated
from tensordict.utils import _unravel_key_to_tuple, unravel_key, unravel_key_list

warn_deprecated(
    "tensordict._C",
    removal="0.17",
    replacement="unravel_key, unravel_key_list and _unravel_key_to_tuple "
    "from tensordict.utils",
)

# The C++ binding took a single key, as unravel_key does.
unravel_keys = unravel_key

__all__ = ["_unravel_key_to_tuple", "unravel_key", "unravel_key_list", "unravel_keys"]
