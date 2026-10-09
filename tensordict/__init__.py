# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings as _warnings

import tensordict._reductions

# Registers the tensordict classes with torch's pytree.
from tensordict import _pytree
from tensordict._archive import (
    is_memmap_archive,
    pack_memmap,
    refresh_archive_checksums,
    unpack_memmap,
)
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
    from_csv,
    from_dict,
    from_h5,
    from_json,
    from_namedtuple,
    from_pandas,
    from_parquet,
    from_struct_array,
    from_tuple,
    from_zarr,
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
from tensordict.nn import (
    as_tensordict_module,
    TensorClassModuleBase,
    TensorClassModuleWrapper,
    TensorDictParams,
)
from tensordict.persistent import PersistentTensorDict
from tensordict.store import LazyStackedTensorDictStore, TensorDictStore
from tensordict.tensorclass import (
    from_dataclass,
    MetaData,
    NonTensorData,
    NonTensorDataBase,
    NonTensorStack,
    TensorAttrs,
    TensorClass,
    tensorclass,
)
from tensordict.typedtensordict import TypedTensorDict
from tensordict.utils import (
    assert_allclose_td,
    assert_close,
    capture_non_tensor_stack,
    get_printoptions,
    is_batchedtensor,
    is_non_tensor,
    is_tensorclass,
    lazy_legacy,
    list_to_stack,
    parse_tensor_dict_string,
    set_capture_non_tensor_stack,
    set_lazy_legacy,
    set_list_to_stack,
    set_printoptions,
    unravel_key,
    unravel_key_list,
)

__version__ = None  # type: ignore
try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("tensordict")
except Exception:
    try:
        from tensordict._version import (
            __version__,
        )  # @manual=//pytorch/tensordict:version
    except ImportError:
        __version__ = None  # type: ignore

__all__ = [
    # Core classes
    "TensorDict",
    "TensorDictBase",
    "LazyStackedTensorDict",
    "UnbatchedTensor",
    "TensorClass",
    "TypedTensorDict",
    "MemoryMappedTensor",
    "PersistentTensorDict",
    "TensorDictStore",
    "LazyStackedTensorDictStore",
    "NestedKey",
    # Factory functions
    "from_csv",
    "from_dict",
    "from_any",
    "from_h5",
    "from_zarr",
    "from_json",
    "from_namedtuple",
    "from_pandas",
    "from_parquet",
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
    "pack_memmap",
    "unpack_memmap",
    "is_memmap_archive",
    "refresh_archive_checksums",
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
    "get_printoptions",
    "set_printoptions",
    # TensorClass components
    "tensorclass",
    "MetaData",
    "NonTensorData",
    "NonTensorDataBase",
    "NonTensorStack",
    "TensorAttrs",
    # NN imports
    "as_tensordict_module",
    "TensorClassModuleBase",
    "TensorClassModuleWrapper",
    "TensorDictParams",
    # Version
    "__version__",
]

# Names that ``from tensordict._pytree import *`` used to leak into this
# namespace, mapped to their replacements.
_DEPRECATED_PYTREE_NAMES = {
    "Any": "typing.Any",
    "Context": "torch.utils._pytree.Context",
    "Dict": "typing.Dict",
    "List": "typing.List",
    "MappingKey": "torch.utils._pytree.MappingKey",
    "PYTREE_REGISTERED_LAZY_TDS": "tensordict.nn.functional_modules.PYTREE_REGISTERED_LAZY_TDS",
    "PYTREE_REGISTERED_TDS": "tensordict.nn.functional_modules.PYTREE_REGISTERED_TDS",
    "Tuple": "typing.Tuple",
    "cls": "tensordict.LazyStackedTensorDict",
    "defaultdict": "collections.defaultdict",
    "implement_for": "pyvers.implement_for",
    "is_compiling": "tensordict.utils.is_compiling",
    "register_pytree_node": "torch.utils._pytree.register_pytree_node",
    "torch": "torch",
}


def __getattr__(name: str) -> object:
    if name in _DEPRECATED_PYTREE_NAMES:
        _warnings.warn(
            f"tensordict.{name} is deprecated and will be removed in TensorDict 0.17. "
            f"Use {_DEPRECATED_PYTREE_NAMES[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_pytree, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
