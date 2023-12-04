# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensordict._lazy import LazyStackedTensorDict
from tensordict._td import is_tensor_collection, SubTensorDict, TensorDict
from tensordict.base import TensorDictBase
from tensordict.functional import (
    dense_stack_tds,
    make_tensordict,
    merge_tensordicts,
    pad,
    pad_sequence,
)
from tensordict.memmap import MemoryMappedTensor
from tensordict.memmap_deprec import is_memmap, MemmapTensor, set_transfer_ownership
from tensordict.persistent import PersistentTensorDict
from tensordict.tensorclass import tensorclass
from tensordict.utils import assert_allclose_td, is_batchedtensor, is_tensorclass

try:
    from tensordict.version import __version__
except ImportError:
    __version__ = None

from tensordict._pytree import *

from tensordict._tensordict import unravel_key, unravel_key_list

__all__ = [
    "LazyStackedTensorDict",
    "MemmapTensor",
    "SubTensorDict",
    "make_tensordict",
    "assert_allclose_td",
    "TensorDict",
    "TensorDictBase",
    "merge_tensordicts",
    "set_transfer_ownership",
    "pad_sequence",
    "is_memmap",
    "is_batchedtensor",
    "is_tensor_collection",
    "pad",
    "PersistentTensorDict",
    "tensorclass",
    "dense_stack_tds",
]

# from tensordict._pytree import *
