# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

from tensordict._C import (  # noqa
    _unravel_key_to_tuple,
    unravel_key,
    unravel_key_list,
    unravel_keys,
)

warnings.warn(
    "tensordict._tensordict will soon be removed in favour of tensordict._C.",
    category=DeprecationWarning,
)
