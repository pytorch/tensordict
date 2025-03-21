# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensordict._lazy import LazyStackedTensorDict  # noqa: F401
from tensordict._td import TensorDict  # noqa: F401
from tensordict.base import (  # noqa: F401
    is_tensor_collection,
    NO_DEFAULT,
    TensorDictBase,
)
from tensordict.functional import (  # noqa: F401
    dense_stack_tds,
    make_tensordict,
    merge_tensordicts,
    pad,
    pad_sequence,
)
from tensordict.memmap import MemoryMappedTensor  # noqa: F401
from tensordict.utils import (  # noqa: F401
    assert_allclose_td,
    cache,
    convert_ellipsis_to_idx,
    erase_cache,
    expand_as_right,
    expand_right,
    implement_for,
    infer_size_impl,
    int_generator,
    is_nested_key,
    is_seq_of_nested_key,
    is_tensorclass,
    lock_blocked,
    NestedKey,
)
