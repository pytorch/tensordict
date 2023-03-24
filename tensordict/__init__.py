# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensordict.memmap import MemmapTensor, set_transfer_ownership
from tensordict.persistent import PersistentTensorDict
from tensordict.tensordict import (
    is_batchedtensor,
    is_memmap,
    is_tensor_collection,
    LazyStackedTensorDict,
    make_tensordict,
    merge_tensordicts,
    pad,
    pad_sequence,
    SubTensorDict,
    TensorDict,
    TensorDictBase,
)

try:
    from tensordict.version import __version__
except ImportError:
    __version__ = None


__all__ = [
    "LazyStackedTensorDict",
    "MemmapTensor",
    "SubTensorDict",
    "TensorDict",
    "TensorDictBase",
    "merge_tensordicts",
    "set_transfer_ownership",
    "pad_sequence",
    "make_tensordict",
    "is_memmap",
    "is_batchedtensor",
    "is_tensor_collection",
    "pad",
    "PersistentTensorDict",
    "is_functional",
]
