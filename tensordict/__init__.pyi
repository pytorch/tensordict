# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from tensordict.nn import as_tensordict_module, TensorDictParams
from tensordict.persistent import PersistentTensorDict
from tensordict.tensorclass import (
    from_dataclass,
    MetaData,
    NonTensorData,
    NonTensorDataBase,
    NonTensorStack,
    TensorClass,
    tensorclass,
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

__version__: str | None

__all__ = [
    # Core classes
    "TensorDict",
    "TensorDictBase",
    "LazyStackedTensorDict",
    "UnbatchedTensor",
    "TensorClass",
    "MemoryMappedTensor",
    "PersistentTensorDict",
    "NestedKey",
    # Factory functions
    "from_dict",
    "from_any",
    "from_h5",
    "from_namedtuple",
    "from_struct_array",
    "from_tuple",
    "from_dataclass",
    "fromkeys",
    "from_module",
    "from_modules",
    "from_pytree",
    "from_consolidated",
    "make_tensordict",
    # Stacking and concatenation
    "stack",
    "cat",
    "lazy_stack",
    "maybe_dense_stack",
    "dense_stack_tds",
    # Memory mapping
    "memmap",
    "load_memmap",
    # Saving and loading
    "save",
    "load",
    # Merging and padding
    "merge_tensordicts",
    "pad",
    "pad_sequence",
    # Utility functions
    "is_tensor_collection",
    "is_batchedtensor",
    "is_non_tensor",
    "is_tensorclass",
    "assert_close",
    "assert_allclose_td",
    "unravel_key",
    "unravel_key_list",
    "parse_tensor_dict_string",
    # Configuration
    "default_is_leaf",
    "is_leaf_nontensor",
    "get_defaults_to_none",
    "set_get_defaults_to_none",
    "capture_non_tensor_stack",
    "set_capture_non_tensor_stack",
    "lazy_legacy",
    "set_lazy_legacy",
    "list_to_stack",
    "set_list_to_stack",
    # TensorClass components
    "tensorclass",
    "MetaData",
    "NonTensorData",
    "NonTensorDataBase",
    "NonTensorStack",
    # NN imports
    "as_tensordict_module",
    "TensorDictParams",
]
