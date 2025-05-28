# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tensordict._reductions
from tensordict._lazy import LazyStackedTensorDict
from tensordict._nestedkey import NestedKey
from tensordict._td import (
    cat,
    from_consolidated,
    from_module,
    from_modules,
    from_pytree,
    fromkeys,
    is_tensor_collection,
    lazy_stack,
    load,
    load_memmap,
    maybe_dense_stack,
    memmap,
    save,
    stack,
    TensorDict,
)
from tensordict._unbatched import UnbatchedTensor

from tensordict.base import (
    _default_is_leaf as default_is_leaf,
    _is_leaf_nontensor as is_leaf_nontensor,
    from_any,
    from_dict,
    from_h5,
    from_namedtuple,
    from_struct_array,
    from_tuple,
    get_defaults_to_none,
    set_get_defaults_to_none,
    TensorDictBase,
)
from tensordict.functional import (
    dense_stack_tds,
    make_tensordict,
    merge_tensordicts,
    pad,
    pad_sequence,
)
from tensordict.memmap import MemoryMappedTensor
from tensordict.persistent import PersistentTensorDict
from tensordict.tensorclass import (
    from_dataclass,
    MetaData,
    NonTensorData,
    NonTensorDataBase,
    NonTensorStack,
    tensorclass,
    TensorClass,
)
from tensordict.utils import (
    assert_allclose_td,
    assert_close,
    capture_non_tensor_stack,
    is_batchedtensor,
    is_non_tensor,
    is_tensorclass,
    lazy_legacy,
    list_to_stack,
    parse_tensor_dict_string,
    set_capture_non_tensor_stack,
    set_lazy_legacy,
    set_list_to_stack,
    unravel_key,
    unravel_key_list,
)
from tensordict._pytree import *
from tensordict.nn import as_tensordict_module, TensorDictParams

try:
    from tensordict._version import __version__  # @manual=//pytorch/tensordict:version
except ImportError:
    __version__ = None
