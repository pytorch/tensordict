# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ._extension import _init_extension
from .memmap import MemmapTensor, set_transfer_ownership
from .tensordict import (
    LazyStackedTensorDict,
    merge_tensordicts,
    SubTensorDict,
    TensorDict,
)
from tensordict._tensormap import TensorMap

try:
    from .version import __version__
except ImportError:
    __version__ = None

_init_extension()

__all__ = [
    "LazyStackedTensorDict",
    "MemmapTensor",
    "SubTensorDict",
    "TensorDict",
    "merge_tensordicts",
    "set_transfer_ownership",
    "TensorMap"
]
