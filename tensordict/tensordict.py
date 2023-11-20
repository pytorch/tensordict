# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensordict._lazy import LazyStackedTensorDict  # noqa: F401
from tensordict.base import is_tensor_collection, TensorDictBase  # noqa: F401
from tensordict.functional import (  # noqa: F401
    dense_stack_tds,
    make_tensordict,
    merge_tensordicts,
    pad,
    pad_sequence,
)
from tensordict.td import SubTensorDict, TensorDict  # noqa: F401
