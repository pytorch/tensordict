# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pyvers
from tensordict import _indexing
from tensordict._deprecation import deprecated_attributes
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
    _cache_while_locked,
    _erase_cache_first,
    _infer_size_impl,
    _int_generator,
    _is_nested_key,
    _is_seq_of_nested_key,
    _lock_blocked,
    assert_allclose_td,
    expand_as_right,
    expand_right,
    is_tensorclass,
    NestedKey,
)

__all__ = [
    "LazyStackedTensorDict",
    "MemoryMappedTensor",
    "NO_DEFAULT",
    "NestedKey",
    "TensorDict",
    "TensorDictBase",
    "assert_allclose_td",
    "dense_stack_tds",
    "expand_as_right",
    "expand_right",
    "is_tensor_collection",
    "is_tensorclass",
    "make_tensordict",
    "merge_tensordicts",
    "pad",
    "pad_sequence",
]

__getattr__ = deprecated_attributes(
    __name__,
    {
        "cache": (_cache_while_locked, None),
        "convert_ellipsis_to_idx": (_indexing.convert_ellipsis_to_idx, None),
        "erase_cache": (_erase_cache_first, None),
        "implement_for": (pyvers.implement_for, "pyvers.implement_for"),
        "infer_size_impl": (_infer_size_impl, None),
        "int_generator": (_int_generator, None),
        "is_nested_key": (
            _is_nested_key,
            "isinstance(key, tensordict.NestedKey) (which rejects lists and "
            "accepts nested tuples)",
        ),
        "is_seq_of_nested_key": (
            _is_seq_of_nested_key,
            "isinstance(key, tensordict.NestedKey) on each key (which rejects "
            "lists and accepts nested tuples)",
        ),
        "lock_blocked": (_lock_blocked, None),
    },
    removal="0.17",
)
