# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import collections
import concurrent
import contextlib
import ctypes
import dataclasses
import difflib
import functools
import hashlib
import inspect
import json
import multiprocessing.managers
import multiprocessing.sharedctypes
import numbers
import operator
import os
import pickle
import shutil
import sys
import warnings
import weakref
from collections.abc import Iterable, Iterator, Mapping
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from textwrap import indent
from types import FunctionType, NoneType, UnionType
from typing import (
    AbstractSet,
    Any,
    Callable,
    Generic,
    get_args,
    get_origin,
    get_type_hints,
    List,
    Literal,
    Sequence,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import numpy as np
import torch
from tensordict import base as _td_base
from tensordict._deprecation import deprecated, warn_deprecated
from tensordict._lazy import _GATHER, LazyStackedTensorDict
from tensordict._nestedkey import NestedKey
from tensordict._pytree import _register_td_node
from tensordict._td import is_tensor_collection, NO_DEFAULT, TensorDict, TensorDictBase
from tensordict._td_memmap import _format_metadata
from tensordict._tensorcollection import TensorCollection
from tensordict._torch_func import _POINTWISE_FUNCTIONS, TD_HANDLED_FUNCTIONS
from tensordict._utils_options import _set_capture_non_tensor_stack
from tensordict.base import (
    _is_accepted_class,
    _is_leaf_nontensor,
    _is_tensor_collection,
    _key_through_leaf_error,
    _register_tensor_class,
    _rename_keys_pairs,
    _TENSORDICTBASE_MIXINS,
    _UNSET,
    _update_at_aliases,
    _warn_get_kwargs,
    _warn_unsupported_operand,
    CompatibleType,
)
from tensordict.utils import (  # @manual=//pytorch/tensordict:_C
    _compact_repr,
    _expands_to,
    _GENERIC_NESTED_ERR,
    _get_shape_from_args,
    _html_repr,
    _is_dataclass as is_dataclass,
    _is_json_serializable,
    _is_tensorclass,
    _is_unbatched,
    _LOCK_ERROR,
    _REPR_OPTIONS,
    _resolve_expand_shape,
    _SKIP_TENSORCLASS_POST_INIT,
    _strong_ref,
    _td_fields,
    _TENSORCLASS_MEMO,
    _unravel_key_to_tuple,
    _UPDATE_IGNORE_LOCK_WARNED,
    _user_stacklevel,
    _warn_batch_dims_ignored,
    _warn_update_ignore_lock,
    _zip_strict,
    capture_non_tensor_stack,
    DeviceType,
    expand_as_right,
    IndexType,
    is_tensorclass,
    LinkedList,
    list_to_stack,
    TensorDictFuture,
)
from torch import multiprocessing as mp, Tensor
from torch.compiler import is_compiling, is_dynamo_compiling
from torch.multiprocessing import Manager
from torch.utils._pytree import tree_is_leaf, tree_map

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

__all__ = [
    "MetaData",
    "NonTensorData",
    "NonTensorDataBase",
    "NonTensorStack",
    "TensorAttrs",
    "TensorClass",
    "from_dataclass",
    "is_non_tensor",
    "tensorclass",
]


def _identity(cls):
    return cls


_PYTREE_SCALAR_LEAVES = (str, int, float, bool, bytes, type(None))


def _copy_pytree_containers(value):
    """Return ``tree_map(_identity, value)``: new dicts, lists and tuples, the same leaves.

    Copies the containers of the exact types ``dict``, ``list`` and ``tuple``
    itself, which is faster than ``tree_map``, returns other pytree leaves
    (such as tensors and arrays) as they are, and hands any other value to
    ``tree_map``.
    """
    cls = type(value)
    if cls in _PYTREE_SCALAR_LEAVES:
        return value
    if cls is dict:
        return {k: _copy_pytree_containers(v) for k, v in value.items()}
    if cls is list:
        return [_copy_pytree_containers(v) for v in value]
    if cls is tuple:
        return tuple(_copy_pytree_containers(v) for v in value)
    if tree_is_leaf(value):
        return value
    return tree_map(_identity, value)


try:
    from typing import dataclass_transform
except ImportError:

    def dataclass_transform(*args, **kwargs):
        """Returns a no-op decorator, in place of :func:`typing.dataclass_transform` (Python < 3.11)."""
        return _identity


T = TypeVar("T", bound=TensorCollection)
# We use an abstract AnyType instead of Any because Any isn't recognised as a type for python < 3.10
major, minor = sys.version_info[:2]
if (major, minor) < (3, 11):

    class _AnyType:
        def __subclasscheck__(self, subclass):
            return False

else:
    _AnyType = Any
if (major, minor) >= (3, 14):
    import annotationlib

_TensorTypes = (
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.IntTensor,
    torch.LongTensor,
    torch.ByteTensor,
    torch.BoolTensor,
    torch.Tensor,  # The base tensor class
)
_TENSOR_ONLY_TYPE_ERR = TypeError(
    "tensor_only requires types to be Tensor, Tensor-subtypes or None."
)
# raised when an indexed write gives a NonTensorData another value
_NON_TENSOR_ONE_VALUE_MSG = (
    "A NonTensorData holds one value for all its elements, so it "
    "cannot take another value at an index. Call maybe_to_stack() "
    "to get a NonTensorStack, which holds a value per element."
)
# flags accepted by the bracket form TensorClass["autocast", ...]
_TENSORCLASS_FLAGS = (
    "autocast",
    "nocast",
    "frozen",
    "tensor_only",
    "shadow",
    "skip_post_init_on_rebuild",
)
# methods where non_tensordict data should be cleared in the return value
_CLEAR_METADATA = {"all", "any"}
# the methods that take the cast_filter of the class by default
_CAST_METHODS = frozenset(
    (
        "to",
        "type",
        "bool",
        "double",
        "float",
        "half",
        "int",
        "bfloat16",
        "complex128",
        "complex32",
        "complex64",
        "float16",
        "float32",
        "float64",
        "int16",
        "int32",
        "int64",
        "int8",
        "qint32",
        "qint8",
        "quint4x2",
        "quint8",
        "uint16",
        "uint32",
        "uint64",
        "uint8",
    )
)

# The container types that from_pytree records under _pytree_type. memmap
# pickles them to other.pickle, which older versions read, and also saves their
# names to _JSON_TYPES_FILE, so that load_memmap needs no pickle for them.
_JSON_TYPES = {
    "dict": dict,
    "OrderedDict": collections.OrderedDict,
    "list": list,
    "tuple": tuple,
}
_JSON_TYPE_NAMES = {value: key for key, value in _JSON_TYPES.items()}
_JSON_TYPES_FILE = "json_types.json"

# torch functions where we can wrap the corresponding TensorDict version
_TD_PASS_THROUGH = {
    torch.atleast_1d: True,
    torch.atleast_2d: True,
    torch.atleast_3d: True,
    torch.broadcast_to: True,
    torch.cat: True,
    torch.chunk: True,
    torch.clone: True,
    torch.empty_like: True,
    torch.flatten: True,
    torch.flip: True,
    torch.fliplr: True,
    torch.flipud: True,
    torch.full_like: True,
    torch.gather: True,
    torch.movedim: True,
    torch.moveaxis: True,
    torch.narrow: True,
    torch.ones_like: True,
    torch.permute: True,
    torch.rand_like: True,
    torch.randn_like: True,
    torch.repeat_interleave: True,
    torch.reshape: True,
    torch.roll: True,
    torch.rot90: True,
    torch.split: True,
    torch.squeeze: True,
    torch.stack: True,
    torch.swapaxes: True,
    torch.swapdims: True,
    torch.tensor_split: True,
    torch.tile: True,
    torch.unbind: True,
    torch.unflatten: True,
    torch.unsqueeze: True,
    torch.where: True,
    torch.zeros_like: True,
    torch.autograd.grad: True,
}
# pointwise torch functions, without the in-place forms, which would return a
# new tensorclass. NonTensorData does not pass them through.
_POINTWISE_PASS_THROUGH = {
    getattr(torch, name): True
    for name in _POINTWISE_FUNCTIONS
    if not name.endswith("_")
}
_TD_PASS_THROUGH.update(_POINTWISE_PASS_THROUGH)
# Methods to be executed from tensordict, any ref to self means 'tensorclass'
_METHOD_FROM_TD = [
    "__enter__",
    "__iter__",
    "dumps",
    "load_",
    "memmap",
    "memmap_",
    "memmap_like",
    "memmap_refresh_",
    "pack",
    "save",
]
# Methods to be executed from tensordict, any ref to self means 'self._tensordict', no wrap of result
_FALLBACK_METHOD_FROM_TD_NOWRAP = [
    "_change_batch_size",
    "_check_batch_size",
    "_check_device",
    "_check_dim_name",
    "_check_unlock",
    "_default_get",
    "_get_at_str",
    "_get_at_tuple",
    "_get_names_idx",  # no wrap output
    "_get_str",
    "_get_tuple",
    "_get_tuple_maybe_non_tensor",
    "_has_names",
    "_items_list",
    "_maybe_names",
    "_multithread_apply_flat",
    "_multithread_apply_nest",
    "_multithread_rebuild",  # rebuild checks if self is a non tensor
    "_propagate_lock",
    "_propagate_unlock",
    "_reduce_get_metadata",
    "_set_device",
    "_set_names",
    "_sync_all",
    "_values_list",
    "all_gather",
    "all_reduce",
    "arecv",
    "asend",
    "batch_dims",
    "batch_size",
    # "broadcast" is handled separately (tensorclass type injection)
    "bytes",
    "cat_tensors",
    "clear_refs_for_compile_",
    "data_ptr",
    "depth",
    "dim",
    "dtype",
    "entry_class",
    "get_item_shape",
    "get_non_tensor",
    # "init_remote" is handled separately (tensorclass type injection)
    "irecv",
    "is_consolidated",
    "is_contiguous",
    "is_cpu",
    "is_cuda",
    "is_empty",
    "is_floating_point",
    "is_locked",
    "is_memmap",
    "is_meta",
    "is_non_tensor",
    "is_shared",
    "isend",
    "items",
    "keys",
    "make_memmap",
    "make_memmap_from_tensor",
    "names",
    "ndim",
    "ndimension",
    "numel",
    "numpy",
    "param_count",
    "pop",
    "recv",
    "reduce",
    "requires_grad",
    "saved_path",
    "scatter",
    "send",
    "shape",
    "size",
    "sorted_keys",
    "to_csv",
    "to_json",
    "to_mds",
    "to_pandas",
    "to_parquet",
    "to_struct_array",
    "tolist",
    "values",
]

# Methods to be executed from tensordict, any ref to self means 'self._tensordict'
_FALLBACK_METHOD_FROM_TD_FORCE = [
    "__ge__",
    "__gt__",
    "__le__",
    "__lt__",
    "__ror__",
]
_FALLBACK_METHOD_FROM_TD = [
    "__abs__",
    "__add__",
    "__and__",
    "__bool__",
    "__eq__",
    "__iadd__",
    "__imul__",
    "__invert__",
    "__ipow__",
    "__isub__",
    "__itruediv__",
    "__mod__",
    "__mul__",
    "__ne__",
    "__neg__",
    "__or__",
    "__pow__",
    "__radd__",
    "__rand__",
    "__rmul__",
    "__rpow__",
    "__rsub__",
    "__rtruediv__",
    "__rxor__",
    "__sub__",
    "__truediv__",
    "__xor__",
    "_add_batch_dim",
    "_apply_nest",
    "_clone",
    "_clone_recurse",
    "_data",
    "_erase_names",  # TODO: must be specialized
    "_exclude",  # TODO: must be specialized
    "_fast_apply",
    "_flatten_keys_inplace",
    "_flatten_keys_outplace",
    "_get_sub_tensordict",
    "_grad",
    "_map",
    "_maybe_remove_batch_dim",
    # "_memmap_" must NOT be listed here: tensorclasses need the dedicated
    # _memmap_ (which records the class in meta.json) rather than the
    # TD-delegating fallback, otherwise the class identity is lost on
    # load_memmap. See _patch_tc and the explicit assignments below.
    "_permute",
    "_remove_batch_dim",
    "_repeat",
    "_select",  # TODO: must be specialized
    "_set_at_tuple",
    "_set_tuple",
    "_to_module",
    "abs",
    "abs_",
    "acos",
    "acos_",
    "add",
    "add_",
    "addcdiv",
    "addcdiv_",
    "addcmul",
    "addcmul_",
    "all",
    "amax",
    "amin",
    "any",
    "apply",
    "apply_",
    "as_tensor",
    "asin",
    "asin_",
    "atan",
    "atan_",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "attrs",
    "auto_batch_size_",
    "auto_device_",
    "backward",
    "bfloat16",
    "bitwise_and",
    "bool",
    "broadcast_to",
    "cat",
    "cat_from_tensordict",
    "ceil",
    "ceil_",
    "chunk",
    "clamp",
    "clamp_max",
    "clamp_max_",
    "clamp_min",
    "clamp_min_",
    "clear",
    "clear_device_",
    "complex128",
    "complex32",
    "complex64",
    "consolidate",
    "contiguous",
    "copy_",
    "copy_at_",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "cpu",
    "create_nested",
    "cuda",
    "cummax",
    "cummin",
    "densify",
    "detach",
    "detach_",
    "div",
    "div_",
    "double",
    "erf",
    "erf_",
    "erfc",
    "erfc_",
    "exclude",
    "exp",
    "exp_",
    "expand",
    "expand_as",
    "expm1",
    "expm1_",
    "fill_",
    "filter_empty_",
    "filter_non_tensor_data",
    "flatten",
    "flatten_keys",
    "flip",
    "fliplr",
    "flipud",
    "float",
    "float16",
    "float32",
    "float64",
    "floor",
    "floor_",
    "frac",
    "frac_",
    "from_any",
    "from_consolidated",
    "from_dataclass",
    "from_h5",
    "from_list",
    "from_modules",
    "from_namedtuple",
    "from_pytree",
    # "from_remote_init" is handled separately (tensorclass type recovery)
    "from_struct_array",
    "from_tuple",
    "from_zarr",
    "fromkeys",
    "gather",
    "gather_and_stack",
    "half",
    "int",
    "int16",
    "int32",
    "int64",
    "int8",
    "isfinite",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "lazy_stack",
    "lerp",
    "lerp_",
    "lgamma",
    "lgamma_",
    "lock_",
    "log",
    "log10",
    "log10_",
    "log1p",
    "log1p_",
    "log2",
    "log2_",
    "log_",
    "logical_and",
    "logsumexp",
    "make_memmap_from_storage",
    "map",
    "map_iter",
    "masked_fill",
    "masked_fill_",
    "masked_select",
    "max",
    "maximum",
    "maximum_",
    "maybe_dense_stack",
    "mean",
    "min",
    "minimum",
    "minimum_",
    "mod",
    "moveaxis",
    "movedim",
    "mul",
    "mul_",
    "named_apply",
    "nanmean",
    "nansum",
    "narrow",
    "neg",
    "neg_",
    "new_empty",
    "new_full",
    "new_ones",
    "new_tensor",
    "new_zeros",
    "norm",
    "pad",
    "permute",
    "pin_memory",
    "pin_memory_",
    "popitem",
    "pow",
    "pow_",
    "prod",
    "qint32",
    "qint8",
    "quantile",
    "quint4x2",
    "quint8",
    "reciprocal",
    "reciprocal_",
    "record_stream",
    "refine_names",
    "rename",
    "rename_",  # TODO: must be specialized
    "rename_key_",
    "repeat",
    "repeat_interleave",
    "replace",
    "requires_grad_",
    "reshape",
    "roll",
    "rot90",
    "round",
    "round_",
    "rsub",
    "separates",
    "set_",
    "set_non_tensor",
    "setdefault",
    "sigmoid",
    "sigmoid_",
    "sign",
    "sign_",
    "sin",
    "sin_",
    "sinh",
    "sinh_",
    "softmax",
    "split",
    "split_keys",
    "sqrt",
    "sqrt_",
    "squeeze",
    "stack",
    "stack_from_tensordict",
    "stack_tensors",
    "std",
    "sub",
    "sub_",
    "sum",
    "swapaxes",
    "swapdims",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "tensor_split",
    "tile",
    "to",
    "to_h5",
    "to_lazystack",
    "to_module",
    "to_namedtuple",
    "to_padded_tensor",
    "to_pytree",
    "to_store",
    "to_zarr",
    "transpose",
    "trunc",
    "trunc_",
    "type",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "unflatten",
    "unflatten_keys",
    "unlock_",
    "unsqueeze",
    "var",
    "view",
    "where",
    "zero_",
    "zero_grad",
]

# These methods require a copy of the non tensor data
_FALLBACK_METHOD_FROM_TD_COPY = [
    "_clone",  # TODO: must be specialized
    "clone",  # TODO: must be specialized
    "copy",  # TODO: must be specialized
]


def is_non_tensor(obj) -> bool:
    """Whether ``obj`` is non-tensor data: a ``NonTensorDataBase`` or a ``NonTensorStack``.

    A local version of :func:`tensordict.is_non_tensor`. These classes set
    ``_is_non_tensor = True``. Reading it from the type is faster than
    ``isinstance``, whose ABCMeta check runs Python code.
    """
    return getattr(type(obj), "_is_non_tensor", False) is True


class _tensorclass_dec:
    autocast: bool
    frozen: bool
    nocast: bool
    shadow: bool
    tensor_only: bool
    skip_post_init_on_rebuild: bool
    cast_filter: Callable | None
    kw_only: bool

    def __new__(
        cls,
        autocast: bool = False,
        frozen: bool = False,
        nocast: bool = False,
        shadow: bool = False,
        tensor_only: bool = False,
        skip_post_init_on_rebuild: bool = False,
        cast_filter: Callable | None = None,
        kw_only: bool = False,
    ):
        if not isinstance(autocast, bool):
            clz = autocast
            self = super().__new__(cls)
            self.__init__(
                autocast=False,
                frozen=False,
                nocast=False,
                shadow=False,
                tensor_only=False,
            )
            return self.__call__(clz)
        return super().__new__(cls)

    def __init__(
        self,
        autocast: bool = False,
        frozen: bool = False,
        nocast: bool = False,
        shadow: bool = False,
        tensor_only: bool = False,
        skip_post_init_on_rebuild: bool = False,
        cast_filter: Callable | None = None,
        kw_only: bool = False,
    ):
        if autocast and nocast:
            raise TypeError("autocast is exclusive with nocast.")
        if tensor_only and (autocast or nocast):
            raise TypeError(
                "tensor_only and autocast or nocast are exclusive features."
            )
        self.autocast = autocast
        self.frozen = frozen
        self.nocast = nocast
        self.shadow = shadow
        self.tensor_only = tensor_only
        self.skip_post_init_on_rebuild = skip_post_init_on_rebuild
        self.cast_filter = cast_filter
        self.kw_only = kw_only

    @dataclass_transform()
    def __call__(self, cls: T) -> T:
        clz = _tensorclass(
            cls,
            frozen=self.frozen,
            shadow=self.shadow,
            tensor_only=self.tensor_only,
            kw_only=self.kw_only,
        )
        clz._autocast = self.autocast
        clz._nocast = self.nocast
        clz._shadow = self.shadow
        clz._frozen = self.frozen
        clz._tensor_only = self.tensor_only
        clz._skip_post_init_on_rebuild = self.skip_post_init_on_rebuild
        # staticmethod: an instance must not bind the predicate
        clz._cast_filter = (
            None if self.cast_filter is None else staticmethod(self.cast_filter)
        )
        return clz


def from_dataclass(
    obj: Any,
    *,
    dest_cls: Type | None = None,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
    batch_size: torch.Size | None = None,
    frozen: bool = False,
    autocast: bool = False,
    nocast: bool = False,
    inplace: bool = False,
    shadow: bool = False,
    tensor_only: bool = False,
    device: torch.device | None = None,
) -> Any:
    """Converts a dataclass instance to a tensorclass instance, or a dataclass type to a tensorclass type.

    Use it to move code from ``@dataclass`` to a tensorclass. For a type,
    the result is a new tensorclass named ``<name>_tc`` with the same fields,
    or the type itself, made a tensorclass, if ``inplace=True``. A type that
    is already a tensorclass is returned as it is. For an instance, the
    result is an instance of a new ``<name>_tc`` tensorclass, or of
    ``dest_cls``, built from the fields of ``obj``.

    :meth:`TensorDict.from_dataclass <tensordict.TensorDict.from_dataclass>`
    converts a dataclass instance to a :class:`~tensordict.TensorDict`
    instead.

    Args:
        obj (Any): the dataclass instance or type to convert.

    Keyword Args:
        dest_cls (type, optional): for an instance, the tensorclass to build.
            When it is given, the class flags below have no effect. Not used
            for a type. Defaults to ``None``, which creates a new tensorclass.
        auto_batch_size (bool, optional): for an instance, if ``True``, sets
            the batch size to the largest leading shape that all the entries
            share. Exclusive with ``batch_size``. Defaults to ``False``.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Ignored without ``auto_batch_size``,
            with a ``FutureWarning`` when it would change the result.
            Starting with TensorDict 0.17, the default will change to
            turning on ``auto_batch_size`` when ``batch_dims`` is given, as
            ``from_dict`` does. Defaults to ``None``, which allows as many as
            the entries share.
        batch_size (torch.Size or list of int, optional): for an instance, the
            batch size of the result: the leading dimensions that all its
            entries share. Defaults to ``None``, which gives an empty batch
            size.
        frozen (bool, optional): the ``frozen`` class flag: if ``True``,
            assigning a field raises ``dataclasses.FrozenInstanceError``.
            Defaults to ``False``.
        autocast (bool, optional): the ``autocast`` class flag: if ``True``, a
            value set to a field is converted to the annotated type. Defaults
            to ``False``.
        nocast (bool, optional): the ``nocast`` class flag: if ``True``,
            tensor-compatible values (``int``, ``float``, ``np.ndarray``) are
            stored as they are, not cast to tensors. Defaults to ``False``.
        inplace (bool, optional): for a type, if ``True``, turns ``obj``
            itself into a tensorclass instead of creating a new class. Not
            used for an instance. Defaults to ``False``.
        shadow (bool, optional): the ``shadow`` class flag: if ``True``,
            fields can have the name of a tensorclass member (``batch_size``,
            ``sum``, ...) and hide it. Defaults to ``False``.
        tensor_only (bool, optional): the ``tensor_only`` class flag: if
            ``True``, every field annotation must be a tensor type or a tensor
            collection, and reading a field is faster. Defaults to ``False``.
        device (torch.device, str or int, optional): for an instance, the
            device of the result. Defaults to ``None``, which leaves each leaf
            where it is.

    The class flags are described in :class:`~tensordict.TensorClass`. Pass
    at most one of ``autocast``, ``nocast`` and ``tensor_only``, as for
    :func:`~tensordict.tensorclass`. When the flags are applied (not with
    ``dest_cls``, and not for a type that is already a tensorclass),
    ``autocast`` with ``nocast``, or, for a type, ``tensor_only`` with either
    of them, emits a ``FutureWarning`` today. Starting with TensorDict 0.17,
    the default will change to raising the ``TypeError`` that
    :func:`~tensordict.tensorclass` raises.

    Returns:
        A tensorclass type if ``obj`` is a type, a tensorclass instance
        otherwise.

    Raises:
        TypeError: if ``obj`` is not a dataclass instance or type, or if, for
            an instance, ``tensor_only`` is combined with ``autocast`` or
            ``nocast``, or ``auto_batch_size=True`` with ``batch_size``.

    Examples:
        >>> from dataclasses import dataclass
        >>> import torch
        >>> from tensordict.tensorclass import from_dataclass
        >>>
        >>> @dataclass
        ... class X:
        ...     a: int
        ...     b: torch.Tensor
        ...
        >>> x = X(0, 0)
        >>> x2 = from_dataclass(x)
        >>> print(x2)
        X_tc(batch_size=[], fields={a: i64[], b: i64[]})
        >>> X2 = from_dataclass(X, autocast=True)
        >>> print(X2(a=0, b=0))
        X_tc(batch_size=[], fields={a: int, b: i64[]})

    """
    from dataclasses import asdict, make_dataclass

    def _field_specs(source, type_hints):
        return [
            (
                field.name,
                type_hints.get(field.name, field.type),
                copy(field),
            )
            for field in source.__dataclass_fields__.values()
        ]

    def _warn_exclusive_flags():
        # Called only where the flags are applied to a class.
        if (autocast and nocast) or (tensor_only and (autocast or nocast)):
            warnings.warn(
                f"from_dataclass() got autocast={autocast}, nocast={nocast} and "
                f"tensor_only={tensor_only}, flags that @tensorclass rejects as "
                "exclusive. Starting with TensorDict 0.17, the default will "
                "change to raising the TypeError that @tensorclass raises.",
                FutureWarning,
                stacklevel=3,
            )

    if isinstance(obj, type):
        if is_tensorclass(obj):
            return obj
        type_hints = get_type_hints(obj)
        if not inplace:
            cls = make_dataclass(
                obj.__name__ + "_tc",
                fields=_field_specs(obj, type_hints),
                bases=obj.__bases__,
            )
        else:
            cls = obj
        clz = _tensorclass(cls, frozen=frozen, shadow=shadow, tensor_only=tensor_only)
        _set_tensorclass_type_hints(clz, type_hints)
        clz._autocast = autocast
        clz._nocast = nocast
        clz._shadow = shadow
        clz._frozen = frozen
        clz._tensor_only = tensor_only
        _warn_exclusive_flags()
        return clz

    if not is_dataclass(obj):
        raise TypeError(f"Expected a obj input, got a {type(obj)} input instead.")
    name = obj.__class__.__name__ + "_tc"
    if dest_cls is None:
        if tensor_only and (autocast or nocast):
            raise TypeError(
                "tensor_only and autocast or nocast are exclusive features."
            )
        try:
            type_hints = get_type_hints(type(obj))
        except (NameError, TypeError):
            # Preserve the existing fallback for unresolved annotations.
            type_hints = None
        clz = _tensorclass(
            make_dataclass(
                name,
                fields=_field_specs(type(obj), type_hints or {}),
            ),
            frozen=frozen,
            shadow=shadow,
            tensor_only=tensor_only,
            # the result gets deep copies of the instance's values, not the defaults
            _warn_defaults=False,
        )
        if type_hints is not None:
            _set_tensorclass_type_hints(clz, type_hints)
        clz._autocast = autocast
        clz._nocast = nocast
        clz._shadow = shadow
        clz._frozen = frozen
        clz._tensor_only = tensor_only
    else:
        clz = dest_cls
    data = asdict(obj)
    if clz._tensor_only:
        # ``asdict`` recursively deep-copies values, which would discard the
        # identity and lock state of TensorDicts. Keep TensorDict-annotated
        # fields intact and let the tensor-only constructor normalize mappings.
        for key in clz._tensordict_fields:
            data[key] = getattr(obj, key)
    result = clz(**data, batch_size=batch_size, device=device)
    if auto_batch_size:
        if batch_size is not None:
            raise TypeError(
                TensorDictBase._CONFLICTING_BATCH_SIZES.format("from_dataclass")
            )
        result = result.auto_batch_size_(batch_dims=batch_dims)
    elif batch_dims is not None:
        _warn_batch_dims_ignored("from_dataclass", result, batch_dims, batch_size)
    if dest_cls is None:
        _warn_exclusive_flags()
    return result


@dataclass_transform()
def tensorclass(
    cls: T = None,
    /,
    *,
    autocast: bool = False,
    frozen: bool = False,
    nocast: bool = False,
    shadow: bool = False,
    tensor_only: bool = False,
    skip_post_init_on_rebuild: bool = False,
    cast_filter: Callable | None = None,
    kw_only: bool = False,
) -> T | None:
    """A decorator that turns a class with annotated fields into a tensorclass.

    A tensorclass is a dataclass whose instances behave like a
    :class:`~tensordict.TensorDict`: they have a batch size, and they support
    indexing, reshaping, ``to(device)``, stacking and memory-mapping. For new
    code, subclass :class:`~tensordict.TensorClass` instead. It takes the same
    flags, and type checkers understand it without a plugin. This decorator
    is kept for existing code and for code that moves from
    ``@dataclass``. See :ref:`tensorclass-guide`.

    Args:
        cls (type, optional): the class to decorate, passed by
            ``@tensorclass`` without parentheses.

    Keyword Args:
        autocast (bool, optional): if ``True``, when a field is set, the value
            is converted to the annotated type. Exclusive with ``nocast`` and
            ``tensor_only``. Defaults to ``False``.
        frozen (bool, optional): if ``True``, assigning a field raises
            ``dataclasses.FrozenInstanceError``, as with
            ``@dataclass(frozen=True)``. To block writes on one instance only,
            call ``lock_()`` on it or pass ``lock=True`` to the constructor:
            writes then raise a ``RuntimeError``. Defaults to ``False``.
        nocast (bool, optional): if ``True``, Tensor-compatible types such as
            ``int``, ``np.ndarray`` and the like are not cast to a tensor
            type. Exclusive with ``autocast`` and ``tensor_only``. Defaults
            to ``False``.
        shadow (bool, optional): if ``True``, fields can have the name of a
            tensorclass member (``batch_size``, ``sum``, ...), and hide that
            member. If ``False``, such a name raises an ``AttributeError``
            when the class is defined. Defaults to ``False``.
        tensor_only (bool, optional): if ``True``, every field annotation must
            be a tensor type or a tensor collection. Values that can be cast
            to a tensor are cast. Reading a field then skips the lookup in the
            non-tensor storage. Exclusive with ``autocast`` and ``nocast``.
            Defaults to ``False``.
        skip_post_init_on_rebuild (bool, optional): if ``True``,
            ``__post_init__`` runs in the generated ``__init__``,
            ``from_tensordict``, ``from_dict``, the other constructors such
            as ``from_any`` or ``fromkeys``, and when the data is loaded or
            received from another process, but not when tensordict makes a
            new instance from the data of an existing one, as in indexing,
            ``clone``, ``to``, ``apply`` or ``torch.stack``. Defaults to
            ``False``: ``__post_init__`` also runs on these rebuilds, except
            on the shells that ``empty()`` builds. On a lazy stack it runs on
            each stacked instance.
        cast_filter (Callable, optional): the default ``cast_filter`` of
            :meth:`~tensordict.TensorDictBase.to` and of the dtype shorthands
            (``float()``, ``bfloat16()``, ...) on the instances of the class: a
            predicate ``fn(key, value) -> bool`` that selects the tensor leaves
            a dtype cast applies to, ``key`` being the nested key as a tuple of
            str. A ``cast_filter`` passed to the call overrides it, and
            ``cast_filter=None`` there casts every leaf. Subclasses inherit it.
            It applies to calls on the tensorclass itself, not when the
            tensorclass is a value of another tensordict or tensorclass that
            is cast. Defaults to ``None``: every leaf is cast.
        kw_only (bool, optional): if ``True``, the fields of the class are
            keyword-only arguments of ``__init__``, as with
            ``@dataclass(kw_only=True)``. The fields of a subclass are not,
            unless it sets the option too. Defaults to ``False``.

    Returns:
        The tensorclass, or a decorator if the flags are given without a
        class (``@tensorclass(autocast=True)``).

    Examples:
        tensorclass can be used with or without arguments:

        >>> from typing import Any
        >>> import torch
        >>> from tensordict import tensorclass
        >>> @tensorclass
        ... class X:
        ...     y: int
        >>> X(torch.ones(())).y
        tensor(1.)
        >>> @tensorclass(autocast=False)
        ... class X:
        ...     y: int
        >>> X(torch.ones(())).y
        tensor(1.)
        >>> @tensorclass(autocast=True)
        ... class X:
        ...     y: int
        >>> X(torch.ones(())).y
        1
        >>> @tensorclass(nocast=True)
        ... class X:
        ...     y: Any
        >>> X(1).y
        1
        >>> @tensorclass(nocast=False)
        ... class X:
        ...     y: Any
        >>> X(1).y
        tensor(1)

        A tensorclass can have methods. Fields that are not tensors are stored
        as :class:`~tensordict.NonTensorData`:

        >>> @tensorclass
        ... class MyData:
        ...     X: torch.Tensor
        ...     y: torch.Tensor
        ...     z: str
        ...     def expand_and_mask(self):
        ...         X = self.X.unsqueeze(-1).expand_as(self.y)
        ...         X = X[self.y]
        ...         return X
        ...
        >>> data = MyData(
        ...     X=torch.ones(3, 4, 1),
        ...     y=torch.zeros(3, 4, 2, 2, dtype=torch.bool),
        ...     z="test",
        ...     batch_size=[3, 4])
        >>> print(data)
        MyData(batch_size=[3,4], fields={X: f32[1], y: bool[2,2], z: str})
        >>> print(data.expand_and_mask())
        tensor([])

        A tensorclass instance can be a field of another tensorclass:

        >>> @tensorclass
        ... class NestingMyData:
        ...     nested: MyData
        ...
        >>> nesting_data = NestingMyData(nested=data, batch_size=[3, 4])
        >>> # although the data is stored as a TensorDict, the type hint helps us
        >>> # to appropriately cast the data to the right type
        >>> assert isinstance(nesting_data.nested, type(data))

        ``skip_post_init_on_rebuild`` checks the data once, when it comes in:

        >>> @tensorclass(skip_post_init_on_rebuild=True)
        ... class Ragged:
        ...     offsets: torch.Tensor
        ...     indices: torch.Tensor
        ...     def __post_init__(self):
        ...         if int(self.offsets[-1]) != len(self.indices):
        ...             raise ValueError("the last offset must be len(indices)")
        >>> r = Ragged(offsets=torch.tensor([0, 2, 3]), indices=torch.zeros(3))
        >>> r.clone().offsets  # not checked again
        tensor([0, 2, 3])

        ``cast_filter`` keeps the integer and boolean fields out of dtype casts:

        >>> from tensordict import is_inexact
        >>> @tensorclass(cast_filter=is_inexact)
        ... class Mesh:
        ...     points: torch.Tensor
        ...     cells: torch.Tensor
        >>> mesh = Mesh(points=torch.zeros(4, 3), cells=torch.zeros(2, 3, dtype=torch.long))
        >>> mesh.bfloat16().points.dtype, mesh.bfloat16().cells.dtype
        (torch.bfloat16, torch.int64)
        >>> mesh.bfloat16(cast_filter=None).cells.dtype  # cast every leaf
        torch.bfloat16

    """

    def wrap(cls):
        return _tensorclass_dec(
            autocast=autocast,
            frozen=frozen,
            nocast=nocast,
            shadow=shadow,
            tensor_only=tensor_only,
            skip_post_init_on_rebuild=skip_post_init_on_rebuild,
            cast_filter=cast_filter,
            kw_only=kw_only,
        )(cls)

    # See if we're being called as @tensorclass or @tensorclass().
    if cls is None:
        # We're called with parens.
        return wrap

    # We're called as @tensorclass without parens.
    return wrap(cls)


def _own_annotation_names(cls: type) -> list[str]:
    """Return the names annotated in the body of ``cls`` itself."""
    annotations = cls.__dict__.get("__annotations__")
    if annotations is None:
        if (major, minor) < (3, 14):
            return []
        # Python 3.14+ evaluates annotations lazily
        annotations = annotationlib.get_annotations(
            cls, format=annotationlib.Format.FORWARDREF
        )
    return list(annotations)


# The dataclasses.field options that tensorclasses ignore: the value that a
# plain dataclass honours, and what the tensorclass does instead (#3073).
_IGNORED_FIELD_OPTIONS = {
    "init": (False, "__init__ still takes the field as an argument"),
    "compare": (False, "== and != still compare the field"),
    "kw_only": (True, "__init__ still takes the field as a positional argument"),
}


def _warn_ignored_field_options(cls: type, options: list[str]) -> None:
    """Warns once for each of ``options`` that a field declared in ``cls`` sets, as the tensorclass ignores it."""
    own_names = set(_own_annotation_names(cls))
    for field in dataclasses.fields(cls):
        if field.name not in own_names:
            continue
        for option in options:
            value, effect = _IGNORED_FIELD_OPTIONS[option]
            if getattr(field, option) is value:
                warnings.warn(
                    f"{cls.__name__}.{field.name}: tensorclasses ignore the "
                    f"field option {option}={value}: {effect}.",
                    UserWarning,
                    stacklevel=_user_stacklevel(),
                )


def _defines_method(cls: type, name: str, default: Callable | None) -> bool:
    """Whether ``cls`` defines the method ``name``, in its body or in a parent.

    The tensorclass ``default``, the methods that the TensorClass base forwards
    to its TensorDict, those that :func:`dataclasses.dataclass` generates and
    those of ``object`` do not count.
    """
    for base in cls.__mro__:
        if name in base.__dict__:
            method = base.__dict__[name]
            break
    else:
        return False
    if base is object or method is default:
        return False
    if getattr(method, "__module__", None) == "dataclasses":
        # dataclasses.dataclass() adds __replace__ (Python 3.13+)
        return False
    if getattr(method, "__qualname__", "").startswith("_wrap_td_method."):
        return False
    if "__dataclass_fields__" not in base.__dict__:
        return True
    # dataclasses compiles the methods it generates from a string, as for
    # __init__ in _tensorclass
    code = getattr(inspect.unwrap(method), "__code__", None)
    return getattr(code, "co_filename", None) != "<string>"


# The names of the members of a tensorclass and of the attributes that hold its
# state. A tensorclass has or forwards to its TensorDict every TensorDict
# attribute, and library code calls private TensorDict methods on nested tensor
# collections (e.g. _get_non_tensor, _send, _transform_keys), so the names of
# TensorDict are in the set, except the two below. The tuple holds the names
# that only tensorclasses have.
_TENSORCLASS_MEMBER_NAMES = frozenset(dir(TensorDict)).union(
    (
        "extend",
        "fields",
        "from_tensordict",
        "_autocast",
        "_cast_filter",
        "_from_dict_defaults",
        "_from_tensordict",
        "_frozen",
        "_is_tensorclass",
        "_nocast",
        "_scalar_fields",
        "_set_dict_warn_msg",
        "_shadow",
        "_skip_post_init_on_rebuild",
        "_tensor_only",
        "_tensorclass_fields",
        "_tensordict_fields",
        "_type_hints",
        # instance attributes
        "_is_initialized",
        "_non_tensordict",
        "_tensordict",
        # a TensorDict attribute that TensorDict.__enter__ reads from a tensorclass
        "_last_op_queue",
    )
) - {
    # a field of NonTensorData
    "_is_non_tensor",
    # read only by the cache decorator of TensorDict methods, on the TensorDict
    "_cache",
}


def _is_reserved_field_name(name: str) -> bool:
    # A field named "data" (as in NonTensorData) or "fields" replaces the member.
    return name in _TENSORCLASS_MEMBER_NAMES and name not in ("data", "fields")


def _raise_reserved_field_name(name: str) -> None:
    raise AttributeError(
        f"Attribute name {name} can't be used with @tensorclass or TensorClass, as tensorclasses have a "
        f"member of that name. To allow it, pass shadow=True (@tensorclass(shadow=True) or "
        f"TensorClass['shadow'])."
    )


def _warn_shared_defaults(cls: type) -> None:
    """Warns for each field of ``cls`` with a tensor default, which every instance shares until 0.17 (#3095)."""
    own_names = _own_annotation_names(cls)
    for field in dataclasses.fields(cls):
        default = field.default
        if field.name in own_names and (
            isinstance(default, Tensor) or _is_tensor_collection(type(default))
        ):
            warnings.warn(
                f"The {type(default).__name__} default of the field "
                f"{cls.__name__}.{field.name} is shared by every instance of the "
                "class: an in-place change on one instance changes the others "
                "and the default. Starting with TensorDict 0.17, the "
                "default will change to giving each instance its own copy "
                "(a clone) of the default. To get a copy for each instance now, "
                "use dataclasses.field(default_factory=...), as in "
                "`dataclasses.field(default_factory=lambda: torch.tensor(0))`.",
                FutureWarning,
                stacklevel=_user_stacklevel(),
            )


def _attribute_functions(value: Any) -> tuple[FunctionType, ...]:
    """Return the functions behind a method, classmethod, staticmethod or property."""
    if isinstance(value, FunctionType):
        return (value,)
    if isinstance(value, property):
        functions = (value.fget, value.fset, value.fdel)
    elif isinstance(value, (classmethod, staticmethod)):
        functions = (value.__func__,)
    else:
        return ()
    return tuple(func for func in functions if isinstance(func, FunctionType))


@functools.cache
def _unchecked_copy(func: FunctionType) -> FunctionType:
    """Return a copy of a TensorDict method that runtime type-checkers skip.

    Tensorclasses get the copy, so that marking it leaves the TensorDict method as
    is. The copy is named after the attribute of TensorClass that holds it, where
    pickle finds it.
    """
    copied = FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )
    copied.__kwdefaults__ = func.__kwdefaults__
    copied.__dict__.update(func.__dict__)
    copied.__doc__ = func.__doc__
    copied.__annotations__ = dict(func.__annotations__)
    copied.__module__ = __name__
    copied.__qualname__ = f"TensorClass.{func.__name__}"
    copied.__no_type_check__ = True
    return copied


@dataclass_transform()
def _tensorclass(
    cls: T,
    *,
    frozen,
    shadow: bool,
    tensor_only: bool,
    kw_only: bool = False,
    _warn_defaults: bool = True,
) -> T:
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        if func not in _TD_PASS_THROUGH or not all(
            issubclass(t, (Tensor, cls, TensorDictBase)) for t in types
        ):
            from torch._ops import HigherOrderOperator

            if isinstance(func, HigherOrderOperator):
                if kwargs is None:
                    kwargs = {}
                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)
            return NotImplemented

        if kwargs is None:
            kwargs = {}

        # get the output type from the arguments / keyword arguments
        if func is torch.where:
            # torch.where(condition, input, other): the result has the type of
            # input, or of other when input is a scalar or a tensor
            tensorclass_instance = args[1] if len(args) > 1 else kwargs.get("input")
            if not _is_tensor_collection(type(tensorclass_instance)):
                tensorclass_instance = args[2] if len(args) > 2 else kwargs.get("other")
        elif len(args) > 0:
            tensorclass_instance = args[0]
        else:
            tensorclass_instance = kwargs.get("input", kwargs["tensors"])
        if isinstance(tensorclass_instance, (tuple, list)):
            tensorclass_instance = tensorclass_instance[0]
        # torch.clamp(..., out=out) returns out itself, as TensorDict.clamp does
        clamp_out = kwargs.get("out") if func is torch.clamp else None
        args = tuple(_arg_to_tensordict(arg) for arg in args)
        kwargs = {key: _arg_to_tensordict(value) for key, value in kwargs.items()}

        result = TD_HANDLED_FUNCTIONS[func](*args, **kwargs)
        if clamp_out is not None and result is kwargs["out"]:
            return clamp_out
        if isinstance(result, (list, tuple)):
            return type(result)(
                _from_tensordict_with_copy(tensorclass_instance, tensordict_result)
                for tensordict_result in result
            )
        return _from_tensordict_with_copy(tensorclass_instance, result)

    # Python 3.14+ evaluates the annotations of a method when they are first
    # read, and looks their names up in the namespace of the class (PEP 649).
    # Read those of the methods in the class body now, before the methods
    # added below (bool, int, set...) shadow the builtins that they name.
    attrs_before = dict(cls.__dict__)
    for value in attrs_before.values():
        for func in _attribute_functions(value):
            try:
                func.__annotations__  # noqa: B018
            except Exception:
                # e.g. a name defined later in the module: read on first use
                pass

    _is_non_tensor = getattr(cls, "_is_non_tensor", False)

    # dataclass() takes an inherited attribute as the default of the field with
    # the same name, so a field named after a TensorClass method or property
    # (e.g. "sum") would get that attribute as its default.
    inherited_reserved_fields = [
        name
        for name in _own_annotation_names(cls)
        if name not in cls.__dict__ and name in _TENSORCLASS_MEMBER_NAMES
        if hasattr(cls, name)
    ]
    if not shadow:
        for name in inherited_reserved_fields:
            if _is_reserved_field_name(name):
                _raise_reserved_field_name(name)
    for name in inherited_reserved_fields:
        setattr(cls, name, dataclasses.field())

    # If cls is already a dataclass, its first dataclass() pass removed the Field
    # declarations (default factories, init/repr/compare/kw_only flags, metadata)
    # from the class namespace, from which the call below rebuilds the fields.
    # Put copies back, except for the fields reset above, whose first-pass
    # default can be the inherited attribute.
    if "__dataclass_fields__" in cls.__dict__:
        own_names = set(_own_annotation_names(cls)) - set(inherited_reserved_fields)
        for field in dataclasses.fields(cls):
            if field.name in own_names:
                setattr(cls, field.name, copy(field))

    # A user-defined __init__ runs instead of the generated one. As in plain
    # Python, a class without one inherits that of a tensorclass parent.
    init = cls.__dict__.get("__init__")
    if init is None:
        user_init = getattr(cls.__init__, "_user_init", None)
    elif (
        "__dataclass_fields__" in cls.__dict__
        and getattr(getattr(init, "__code__", None), "co_filename", None) == "<string>"
    ):
        # cls was a dataclass already, and dataclass() generated this
        # __init__. That relies on a CPython detail: dataclasses compiles the
        # methods it generates from a string. Should that stop holding, a
        # generated __init__ takes the slower custom path, with the same result
        # except for a dataclass that was not frozen and that frozen=True
        # freezes here: its __init__ would raise FrozenInstanceError.
        user_init = None
    else:
        user_init = init

    # Keep the docstring of the class. A class that was a dataclass already
    # may carry the docstring that dataclass() generated from its signature:
    # treat it as absent, so that it is regenerated below.
    own_doc = cls.__dict__.get("__doc__")
    if own_doc and "__dataclass_fields__" in cls.__dict__:
        try:
            generated_doc = cls.__name__ + str(inspect.signature(cls)).replace(
                " -> None", ""
            )
        except (TypeError, ValueError):
            generated_doc = None
        if own_doc == generated_doc:
            own_doc = None

    # Keep the operators that the class defines, in its body or in a parent.
    # Look for them before dataclass() adds a __repr__ and an __eq__.
    operators = {
        "__repr__": _repr,
        "__eq__": _eq,
        "__ne__": _ne,
        "__or__": _or,
        "__xor__": _xor,
        "__bool__": _bool,
        "__replace__": _replace,
    }
    own_operators = {
        name
        for name in (*operators, *_FALLBACK_METHOD_FROM_TD_FORCE)
        if _defines_method(cls, name, operators.get(name))
    }

    # Breaks some tests, don't do that:
    # if not dataclasses.is_dataclass(cls):
    cls = dataclass(cls, frozen=frozen, kw_only=kw_only)
    # Positional arguments skip the keyword-only fields of a class with the
    # class option kw_only, and of its subclasses (see _init_wrapper).
    cls.__tensorclass_kw_only__ = kw_only or cls.__dict__.get(
        "__tensorclass_kw_only__", False
    )
    for name in own_operators:
        if name in cls.__dict__ and name not in attrs_before:
            # dataclass() replaced an inherited __repr__ or __eq__
            delattr(cls, name)
    _TENSORCLASS_MEMO[cls] = True

    # Use dataclasses.fields() to get the correct field list (excludes ClassVar fields)
    # rather than __dataclass_fields__ which may incorrectly include them
    expected_keys = cls.__expected_keys__ = {
        field.name for field in dataclasses.fields(cls)
    }
    # The fields that from_dict() will fill with their defaults in 0.17. A
    # non-tensor class keeps the input of from_dict() as its data.
    cls._from_dict_defaults = (
        ()
        if _is_non_tensor
        else tuple(
            field.name
            for field in dataclasses.fields(cls)
            if field.default_factory is not dataclasses.MISSING
            or (field.default is not dataclasses.MISSING and field.default is not None)
        )
    )
    if _warn_defaults and "_is_tensorclass" not in cls.__dict__:
        # not on the second pass of a class decorated twice
        _warn_shared_defaults(cls)

    if not shadow:
        for attr in expected_keys:
            if _is_reserved_field_name(attr):
                _raise_reserved_field_name(attr)

    cls.fields = classmethod(_fields)
    for field in dataclasses.fields(cls):
        if hasattr(cls, field.name):
            # if we have used Cls(TensorClass["shadow"]), we have a subclass of Cls(TensorClass)
            #  so we cannot directly delete the attribute
            try:
                delattr(cls, field.name)
            except AttributeError:
                pass

    if not tensor_only and not shadow:
        for field in dataclasses.fields(cls):
            setattr(
                cls,
                field.name,
                _FieldGetter(field.name, _is_non_tensor and field.name == "data"),
            )

    if tensor_only:
        for field in dataclasses.fields(cls):
            name = field.name

            def _make_prop(key):
                def _getter(self):
                    td = self._tensordict
                    if type(td) is TensorDict:
                        # Cheaper than _get_str. Dynamo guards on a key that
                        # dict.get misses: a compiled function that read None
                        # recompiles once the field holds a tensor.
                        out = td._tensordict.get(key, _UNSET)
                    else:
                        out = td._get_str(key, _UNSET)
                        if out is _UNSET and is_compiling() and key in td.keys():
                            # see above: only for the guard, as keys() can be
                            # costly (e.g. on a lazy stack)
                            out = td._get_str(key, None)
                        if out is _UNSET:
                            out = _partial_field(td, key)
                    if out is _UNSET:
                        # A field missing from the TensorDict is None
                        non_tensordict = self._non_tensordict
                        return non_tensordict.get(key) if non_tensordict else None
                    return out

                def _setter(self, value):
                    # Reached through object.__setattr__, as in the __init__
                    # of a frozen class.
                    type(self).set(self, key, value)

                return property(_getter, _setter)

            setattr(cls, name, _make_prop(name))

    _get_type_hints(cls, tensor_only=tensor_only)
    # Detect user-defined __setattr__ that must be called during init.
    # After dataclass(), frozen=True adds a guard __setattr__, which is not
    # user-defined and gets bypassed by the frozen __init__ via
    # object.__setattr__. Only non-frozen classes can have a real custom one.
    _has_custom_setattr = "__setattr__" in cls.__dict__ and not frozen
    init_vars, needs_init_vars = _init_vars(cls)
    # The defaults of the InitVars, that __post_init__ gets outside __init__,
    # and the __post_init__ they are for (see _post_init_args)
    cls.__tensorclass_init_var_defaults__ = (
        getattr(cls, "__post_init__", None),
        tuple(default for _, default in init_vars)
        if needs_init_vars
        and all(default is not dataclasses.MISSING for _, default in init_vars)
        else (),
    )
    if "_is_tensorclass" not in cls.__dict__:
        # Warn about the field options that the class ignores, except where
        # a method of the user's replaces the one that ignores them.
        ignored_options = []
        if user_init is None:
            if not _has_custom_setattr:
                # the dataclass __init__ sets the fields: it honours init
                ignored_options.append("init")
            if not _follows_signature(cls, needs_init_vars and not _has_custom_setattr):
                ignored_options.append("kw_only")
        if not {"__eq__", "__ne__"} <= own_operators:
            ignored_options.append("compare")
        _warn_ignored_field_options(cls, ignored_options)
        # A class that was decorated before (e.g. @tensorclass on a
        # TensorClass subclass) keeps the __init__ wrapper it got then.
        cls.__init__ = _init_wrapper(
            cls.__init__ if user_init is None else user_init,
            cls,
            frozen,
            shadow,
            tensor_only,
            _has_custom_setattr,
            _custom_init=user_init is not None,
            _is_non_tensor=_is_non_tensor,
            _init_vars=init_vars,
            _needs_init_vars=needs_init_vars,
        )
    cls._from_tensordict = classmethod(_from_tensordict)
    cls.from_tensordict = classmethod(_from_tensordict_public)
    if not hasattr(cls, "__torch_function__"):
        cls.__torch_function__ = classmethod(__torch_function__)
    cls.__getstate__ = _getstate
    cls.__setstate__ = _setstate
    if not hasattr(cls, "__copy__"):
        cls.__copy__ = _copy
    if not hasattr(cls, "__deepcopy__"):
        cls.__deepcopy__ = _deepcopy

    if tensor_only:
        cls.__getattr__ = _getattr_tensor_only
    else:
        cls.__getattr__ = _getattr

    if "__setattr__" not in cls.__dict__:
        if not tensor_only:
            cls.__setattr__ = _setattr
        else:
            cls.__setattr__ = _setattr_tensor_only
    if "__getitem__" not in cls.__dict__:
        cls.__getitem__ = _getitem
    if "__getitems__" not in cls.__dict__:
        cls.__getitems__ = _getitem
    if "__setitem__" not in cls.__dict__:
        cls.__setitem__ = _setitem
    if "__len__" not in cls.__dict__:
        cls.__len__ = _len
    for name, method in operators.items():
        if name not in own_operators and not (_is_non_tensor and name == "__repr__"):
            setattr(cls, name, method)
    if not hasattr(cls, "_repr_html_"):
        cls._repr_html_ = _repr_html
    if _is_non_tensor and hasattr(cls, "_patch_operators"):
        # NonTensorDataBase patches its comparison operators
        cls._patch_operators()

    if not hasattr(cls, "_new_unsafe"):
        cls._new_unsafe = classmethod(_new_unsafe)
    if not hasattr(cls, "non_tensor_items") and "non_tensor_items" not in expected_keys:
        cls.non_tensor_items = _non_tensor_items
    if not hasattr(cls, "set") and "set" not in expected_keys:
        cls.set = _set
    if not hasattr(cls, "set_at_") and "set_at_" not in expected_keys:
        cls.set_at_ = _set_at_
    if not hasattr(cls, "_set_str"):
        cls._set_str = _set_str
    if not hasattr(cls, "_set_at_str"):
        cls._set_at_str = _set_at_str
    if not hasattr(cls, "del_") and "del_" not in expected_keys:
        cls.del_ = _del_
    if "__delattr__" not in cls.__dict__:
        cls.__delattr__ = _delattr
    if not hasattr(cls, "get") and "get" not in expected_keys:
        cls.get = _get
    if not hasattr(cls, "get_at") and "get_at" not in expected_keys:
        cls.get_at = _get_at
    if not hasattr(cls, "unbind") and "unbind" not in expected_keys:
        cls.unbind = _unbind
    if not hasattr(cls, "empty"):
        cls.empty = _empty
    cls._unbind = _unbind
    if not hasattr(cls, "state_dict") and "state_dict" not in expected_keys:
        cls.state_dict = _state_dict
    if not hasattr(cls, "load_state_dict") and "load_state_dict" not in expected_keys:
        cls.load_state_dict = _load_state_dict
    if not hasattr(cls, "_memmap_") and "_memmap_" not in expected_keys:
        cls._memmap_ = _memmap_
    if not hasattr(cls, "share_memory_") and "share_memory_" not in expected_keys:
        cls.share_memory_ = _share_memory_
    if not hasattr(cls, "update") and "update" not in expected_keys:
        cls.update = _update
    if not hasattr(cls, "update_") and "update_" not in expected_keys:
        cls.update_ = _update_
    if not hasattr(cls, "update_at_") and "update_at_" not in expected_keys:
        cls.update_at_ = _update_at_
    if not hasattr(cls, "extend"):
        cls.extend = _extend
    if not hasattr(cls, "__exit__"):
        cls.__exit__ = _exit
    for method_name in _METHOD_FROM_TD:
        if not hasattr(cls, method_name):
            setattr(cls, method_name, _unchecked_copy(getattr(TensorDict, method_name)))
    for method_name in _FALLBACK_METHOD_FROM_TD:
        if not hasattr(cls, method_name):
            if inspect.ismethod(getattr(TensorDict, method_name)):
                # classmethod: TC.stack(...) must not take its input as self
                setattr(cls, method_name, _wrap_td_classmethod(method_name))
            else:
                setattr(cls, method_name, _wrap_td_method(method_name))
    for method_name in _FALLBACK_METHOD_FROM_TD_FORCE:
        if method_name not in own_operators:
            setattr(cls, method_name, _wrap_td_method(method_name))
    for method_name in _FALLBACK_METHOD_FROM_TD_NOWRAP:
        if not hasattr(cls, method_name) and method_name not in expected_keys:
            is_property = isinstance(
                getattr(TensorDictBase, method_name, None), property
            )
            setattr(
                cls,
                method_name,
                _wrap_td_method(method_name, no_wrap=True, is_property=is_property),
            )
    for method_name in _FALLBACK_METHOD_FROM_TD_COPY:
        if not hasattr(cls, method_name):
            setattr(
                cls,
                method_name,
                _wrap_td_method(method_name, copy_non_tensor=True),
            )

    # Distributed methods that need tensorclass type injection
    if not hasattr(cls, "init_remote") and "init_remote" not in expected_keys:
        cls.init_remote = _tc_init_remote
    if not hasattr(cls, "broadcast") and "broadcast" not in expected_keys:
        cls.broadcast = _tc_broadcast
    if not hasattr(cls, "from_remote_init") and "from_remote_init" not in expected_keys:
        cls.from_remote_init = classmethod(
            lambda cls_inner, *args, **kwargs: TensorDictBase.from_remote_init(
                *args, **kwargs
            )
        )

    # if not hasattr(cls, "batch_size") and "batch_size" not in expected_keys:
    #     cls.batch_size = property(_batch_size, _batch_size_setter)

    # Memmap. load and load_memmap are bound to cls (here and in the loop
    # below): load_memmap falls back on the class it is called on when the
    # saved class cannot be found by name.
    if not hasattr(cls, "load_memmap") and "load_memmap" not in expected_keys:
        cls.load_memmap = classmethod(
            _unchecked_copy(TensorDictBase.load_memmap.__func__)
        )
    if not hasattr(cls, "load") and "load" not in expected_keys:
        cls.load = classmethod(_unchecked_copy(TensorDictBase.load.__func__))
    if not hasattr(cls, "load_memmap_") and "load_memmap_" not in expected_keys:
        cls.load_memmap_ = _load_memmap_
    if not hasattr(cls, "_load_memmap"):
        cls._load_memmap = classmethod(_load_memmap)
    if not hasattr(cls, "from_dict") and "from_dict" not in expected_keys:
        cls.from_dict = classmethod(_from_dict_public)
    if (
        not hasattr(cls, "from_dict_instance")
        and "from_dict_instance" not in expected_keys
    ):
        cls.from_dict_instance = _from_dict_instance

    td_attrs = set(TensorDict.__dict__).union(
        TensorDictBase.__dict__, *(mixin.__dict__ for mixin in _TENSORDICTBASE_MIXINS)
    )
    for attr in td_attrs:
        # __init_subclass__ is TensorDict's hook for its own subclasses
        if attr in ("__torch_function__", "__init_subclass__"):
            continue
        func = getattr(TensorDict, attr)
        if inspect.ismethod(func) and attr not in cls.__dict__:
            # A genuine classmethod implementation inherited from a
            # tensorclass base (e.g. TensorClass._load_memmap) rebinds `cls`
            # on access and must not be shadowed by the TensorDict
            # classmethod: shadowing _load_memmap with TensorDict._load_memmap
            # used to break load_memmap for TensorClass subclasses (an empty
            # instance was returned because the tensorclass directory layout
            # was read as a plain TensorDict one). Everything else inherited
            # from a base -- _wrap_td_method/_wrap_classmethod wrappers and
            # methods bound to the parent class (e.g. from_list, stack on the
            # TensorClass base) -- hardcodes the wrong class or the wrong
            # calling convention and must be re-wrapped for cls as before.
            inherited = next(
                (
                    base.__dict__[attr]
                    for base in cls.__mro__[1:]
                    if attr in base.__dict__
                ),
                None,
            )
            if isinstance(inherited, (classmethod, staticmethod)):
                continue
            tdcls = func.__self__
            if issubclass(tdcls, TensorDictBase):  # detects classmethods
                if attr in ("load", "load_memmap"):
                    # Bound to cls rather than TensorDict (see "Memmap" above).
                    # Rebind func too: on some Python versions, func.__get__
                    # does not rebind a bound method.
                    tdcls, func = cls, func.__func__.__get__(cls)
                setattr(cls, attr, _wrap_classmethod(tdcls, cls, func))

    if not hasattr(cls, "select") and "select" not in expected_keys:
        cls.select = _tc_select
    if not hasattr(cls, "rename_keys") and "rename_keys" not in expected_keys:
        cls.rename_keys = _tc_rename_keys
    if not hasattr(cls, "to_tensordict") and "to_tensordict" not in expected_keys:
        cls.to_tensordict = _to_tensordict
    if not hasattr(cls, "device") and "device" not in expected_keys:
        cls.device = property(_device, _device_setter)
    if not _is_non_tensor and not hasattr(cls, "data") and "data" not in expected_keys:
        cls.data = property(_data, _data_setter)
    if not hasattr(cls, "grad") and "grad" not in expected_keys:
        cls.grad = property(_grad)
    if not hasattr(cls, "to_dict") and "to_dict" not in expected_keys:
        cls.to_dict = _to_dict

    if not tensor_only:
        # Attribute lookup would find a TensorClass method or property of the
        # same name (e.g. "sum", or the "fields" of a tensorclass base) before
        # the field, which _getattr serves.
        for name in expected_keys:
            if (shadow or name == "fields") and hasattr(cls, name):
                setattr(cls, name, property(functools.partial(_getattr, item=name)))

    # Without a docstring, the class documents its constructor signature.
    cls.__doc__ = own_doc or f"{cls.__name__}{inspect.signature(cls)}"

    # A runtime type-checker that decorates the class (e.g. beartype) checks
    # every method in its namespace. Mark the methods added above with PEP
    # 484's __no_type_check__ so that it skips them: their annotations name
    # builtins that the class shadows (bool, int, type...), and some do not
    # hold for tensorclasses (e.g. the device can be None). __init__ keeps the
    # annotations of the fields (or of a user-defined __init__), which
    # checkers such as jaxtyping check.
    module_globals = globals()
    for name, value in cls.__dict__.items():
        if name == "__init__" or attrs_before.get(name) is value:
            continue
        for func in _attribute_functions(value):
            # A function of this module, or a wrapper of one (e.g. deprecated)
            if func.__annotations__ and (
                func.__globals__ is module_globals or func.__module__ == __name__
            ):
                func.__no_type_check__ = True

    _register_tensor_class(cls)
    try:
        _register_td_node(cls)
    except ValueError:
        # The class may already be registered as a pytree node
        pass

    # faster than doing instance checks
    cls._is_non_tensor = _is_non_tensor
    cls._is_tensorclass = True
    # The tensorclass decorator sets it from its flag
    cls._skip_post_init_on_rebuild = False
    return cls


# def _batch_size(self):
#     return self.__dict__["_tensordict"]._batch_size
# def _batch_size_setter(self, value):
#     self.__dict__["_tensordict"].batch_size = value


def _arg_to_tensordict(arg):
    # if arg is a tensorclass or sequence of tensorclasses, extract the underlying
    # tensordicts and return those instead

    # since arg can be anything (e.g. callable etc) we can't use pytree
    # def convert(x):
    #     if _is_tensorclass(type(x)):
    #         return x._tensordict
    #     return x
    # return torch.utils._pytree.tree_map(convert, arg)

    if _is_tensorclass(type(arg)):
        return arg._tensordict
    elif isinstance(arg, (tuple, list)) and all(
        _is_tensorclass(type(item)) for item in arg
    ):
        return type(arg)(item._tensordict for item in arg)
    return arg


def _tc_tensorclass_type_str(obj):
    """Return the fully qualified name of a tensorclass, e.g. ``'my.module.MyData'``."""
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _tc_init_remote(self, dst=None, group=None, device=None, use_broadcast=False):
    tc_type = _tc_tensorclass_type_str(self)
    self._tensordict.init_remote(
        dst=dst,
        group=group,
        device=device,
        use_broadcast=use_broadcast,
        _tensorclass_type=tc_type,
    )


def _tc_broadcast(self, src, *, group=None, device=None):
    tc_type = _tc_tensorclass_type_str(self)
    result = self._tensordict.broadcast(
        src, group=group, device=device, _tensorclass_type=tc_type
    )
    if not isinstance(result, type(self)):
        return type(self)._from_tensordict(result)
    return result


@deprecated(
    "TensorClass.fields()", removal="0.17", replacement="dataclasses.fields(cls)"
)
def _fields(cls) -> tuple[dataclasses.Field, ...]:
    """Returns the fields of the tensorclass.

    .. deprecated:: 0.15
        Use ``dataclasses.fields(cls)`` instead. It returns the same tuple of
        :class:`dataclasses.Field` objects.
    """
    return dataclasses.fields(cls)


@deprecated(
    "TensorClass.extend()",
    removal="0.17",
    replacement="lazy_stack or torch.cat to build a new instance",
)
def _extend(self, tensordict: list[TensorDictBase] | TensorDictBase) -> None:
    """Appends tensordicts to the lazy stack that the instance holds.

    .. deprecated:: 0.15
        Build a new instance with :func:`tensordict.lazy_stack` or
        :func:`torch.cat` instead.
    """
    self._tensordict.extend(tensordict)


def _from_tensordict_with_copy(tc, tensordict):
    # creates a new tensorclass with the same type as tc, and a copy of the
    # non_tensordict data
    return type(tc)._from_tensordict(
        tensordict=tensordict, non_tensordict=dict(tc._non_tensordict)
    )


def _from_tensordict_with_none(tc, tensordict):
    # creates a new tensorclass with the same type as tc, and all non_tensordict entries
    # set to None
    return type(tc)._from_tensordict(
        tensordict=tensordict,
        non_tensordict={key: None for key in tc._non_tensordict},
    )


def _did_you_mean(name: str, fields: Iterable[str]) -> str:
    """Returns `` Did you mean 'field'?`` for the field closest to ``name``, or ``""``."""
    close = difflib.get_close_matches(name, list(fields), n=1)
    return f" Did you mean {close[0]!r}?" if close else ""


def _no_attribute_error(self, item: str) -> AttributeError:
    """The error of reading ``item``, which is neither a field nor an attribute of ``self``.

    It lists the public fields and suggests the closest one. A private or dunder
    name, which libraries probe with ``hasattr``, keeps the bare message: it
    costs nothing to build. ``obj`` without ``name`` keeps Python from adding
    its own suggestion to the traceback, which would repeat ours.
    """
    if item.startswith("_") or is_dynamo_compiling():
        return AttributeError(item)
    fields = sorted(
        key for key in type(self).__expected_keys__ if not key.startswith("_")
    )
    return AttributeError(
        f"{type(self).__name__!r} object has no attribute or field {item!r}."
        f"{_did_you_mean(item, fields)} Its fields are {fields}.",
        obj=self,
    )


def _warn_unknown_tensor_only_fields(cls, kwargs: dict) -> None:
    # The tensor_only path of __init__ stores the arguments that are not
    # fields. The other paths raise AttributeError for them, and this path
    # will too once the deprecation ends.
    unknown = sorted(key for key in kwargs if key not in cls.__expected_keys__)
    warn_deprecated(
        f"Passing arguments that are not fields ({unknown}) to the tensor_only "
        f"tensorclass {cls.__name__}, which other tensorclasses reject with "
        "AttributeError,",
        removal="0.17",
        replacement=f"only its fields ({sorted(cls.__expected_keys__)})",
        stacklevel=3,
    )


def _pop_instance_attr_traced(obj, name: str) -> Any:
    """Removes ``name`` from the instance ``__dict__`` of ``obj`` under Dynamo, and returns it.

    The Dynamo of torch 2.11 can neither pop from the ``__dict__`` of an
    instance nor call ``object.__delattr__``, but it traces a
    ``super().__delattr__`` that resolves to ``object.__delattr__``. Eager code
    pops from the ``__dict__``, which is faster.
    """
    obj_dict = obj.__dict__
    value = obj_dict[name]
    try:
        super(type(obj).__mro__[-2], obj).__delattr__(name)
    except AttributeError:
        # the field of a tensor_only class is a property without a deleter
        del obj_dict[name]
    return value


def _init_vars(cls) -> tuple[tuple[tuple[str, Any], ...], bool]:
    """Returns the ``(name, default)`` of the InitVars that ``__init__`` takes, and whether ``__post_init__`` needs them.

    ``__init__`` takes the ``dataclasses.InitVar`` fields of ``cls`` if the
    class has no ``__post_init__`` (it drops them), or if ``__post_init__``
    takes one positional argument per InitVar (it passes them on). The
    InitVars are needed if ``__post_init__`` cannot be called without them:
    such a class raised on every construction before #3103. Only then do
    omitted InitVars and rebuilds give ``__post_init__`` the defaults, and
    positional arguments follow the signature. The other classes keep their
    old behaviour for the calls that worked before.
    """
    init_vars = tuple(
        (name, field.default)
        for name, field in cls.__dataclass_fields__.items()
        if field._field_type is dataclasses._FIELD_INITVAR
    )
    post_init = getattr(cls, "__post_init__", None)
    if not init_vars or post_init is None:
        return init_vars, False
    try:
        signature = inspect.signature(post_init)
        signature.bind(None, *(None,) * len(init_vars))
    except (TypeError, ValueError):
        return (), False
    try:
        signature.bind(None)
    except TypeError:
        return init_vars, True
    return init_vars, False


def _follows_signature(cls, needs_init_vars: bool) -> bool:
    """Whether positional arguments follow the signature of the dataclass ``__init__``.

    They do for a class whose ``__post_init__`` needs its InitVars, or with
    the class option ``kw_only`` (in its own definition or in that of a
    parent). The other classes assign them to the fields in order,
    keyword-only fields included (#3100).
    """
    return needs_init_vars or any(
        base.__dict__.get("__tensorclass_kw_only__", False) for base in cls.__mro__
    )


def _post_init_args(cls) -> tuple:
    """The InitVar defaults that ``cls.__post_init__`` gets outside ``__init__``.

    A subclass that is not a tensorclass itself and replaces the
    ``__post_init__`` they are for gets none, as before #3103.
    """
    post_init, defaults = cls.__tensorclass_init_var_defaults__
    return defaults if cls.__post_init__ is post_init else ()


def _init_wrapper(
    __init__: Callable,
    cls,
    frozen: bool,
    shadow: bool,
    tensor_only: bool,
    _has_custom_setattr: bool = False,
    _custom_init: bool = False,
    _is_non_tensor: bool = False,
    _init_vars: tuple = (),
    _needs_init_vars: bool = False,
) -> Callable:
    init_sig = inspect.signature(__init__)
    params = list(init_sig.parameters.values())
    # drop first entry of params which corresponds to self and isn't passed by the user
    required_params = [p.name for p in params[1:] if p.default is inspect._empty]
    declared_params = frozenset(
        p.name for p in params[1:] if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    )
    fields = dataclasses.fields(cls)
    field_names = tuple(field.name for field in fields)
    required_fields = frozenset(field_names)
    init_vars = () if _custom_init else _init_vars
    # The dataclass __init__ of a custom __setattr__ passes the InitVars on
    # itself, and positional arguments go to the fields in order, as before.
    needs_init_vars = _needs_init_vars and not _custom_init and not _has_custom_setattr
    # a subclass that replaces this __post_init__ keeps the old behaviour
    post_init = getattr(cls, "__post_init__", None)
    init_var_names = frozenset(name for name, _ in init_vars)
    signature_names = tuple(
        p.name for p in params[1:] if p.kind is p.POSITIONAL_OR_KEYWORD
    )
    # The InitVars are among the positional arguments that follow the
    # signature, and the keyword-only fields are not.
    positional_names = (
        signature_names
        if not _custom_init and _follows_signature(cls, needs_init_vars)
        else field_names
    )
    old_positional_names = (
        signature_names
        if not _custom_init and _follows_signature(cls, False)
        else field_names
    )
    # Specialize only complete schemas without defaults or user initialization
    # hooks. All other constructors retain the general field-setting path.
    can_init_tensors = (
        not shadow
        and not _has_custom_setattr
        and cls.__mro__[1].__setattr__
        in (object.__setattr__, _setattr, _setattr_tensor_only)
        and all(
            field.init
            and field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
            for field in fields
        )
        and set(required_params) == required_fields
    )
    # Non-tensor carriers (NonTensorData, MetaData) keep every field in
    # _non_tensordict, so their constructor can fill that dict directly
    # instead of routing each field through set(). The defaults are only
    # collected for these classes: Dynamo cannot wrap a closure dict that
    # holds a tensor default (as in ``x: torch.Tensor = torch.zeros(3)``).
    can_init_non_tensor = (
        _is_non_tensor
        and not _custom_init
        and not shadow
        and not tensor_only
        and not _has_custom_setattr
        and not init_vars
        and all(
            field.init and field.default_factory is dataclasses.MISSING
            for field in fields
        )
    )
    if can_init_non_tensor:
        non_tensor_defaults = {
            field.name: field.default
            for field in fields
            if field.default is not dataclasses.MISSING
        }
        non_tensor_required = required_fields - non_tensor_defaults.keys()
    else:
        non_tensor_defaults = non_tensor_required = None
    # if not required_params and hasattr(cls, "__init_parent__"):
    #     init_sig_parent = inspect.signature(cls.__init_parent__)
    #     params_parent = list(init_sig_parent.parameters.values())
    #     # drop first entry of params which corresponds to self and isn't passed by the user
    #     required_params = [p.name for p in params_parent[1:] if p.default is inspect._empty]

    @functools.wraps(__init__)
    def wrapper(
        self,
        *args: Any,
        **kwargs,
    ):
        if _custom_init or (type(self) is not cls and "_tensordict" in self.__dict__):
            # Run a user-defined __init__ (own or inherited), or a parent's
            # __init__ that one reaches through super().__init__(). The
            # outermost call builds the container; nested calls reuse it.
            # __init__ also gets the metadata that it declares as parameters,
            # unless a field has that name (shadow=True).
            batch_size, device, names, lock = [
                None
                if key in required_fields
                else kwargs.get(key)
                if key in declared_params
                else kwargs.pop(key, None)
                for key in ("batch_size", "device", "names", "lock")
            ]
            outer = type(self) is cls or "_tensordict" not in self.__dict__
            if outer:
                if batch_size is None:
                    batch_size = ()
                elif isinstance(batch_size, int):
                    batch_size = (batch_size,)
                td = TensorDict._new_unsafe(
                    {}, batch_size=torch.Size(batch_size), device=device, names=names
                )
                object.__setattr__(self, "_tensordict", td)
                object.__setattr__(self, "_non_tensordict", {})
                object.__setattr__(self, "_is_initialized", True)
                if not type(self)._is_non_tensor:
                    # A None field is a missing key.
                    # Record (as keys) the fields that __init__ sets to None,
                    # so that the loop below can tell them from the fields
                    # that it does not set.
                    self.__dict__["_init_none_fields"] = {}
            elif batch_size is not None or device is not None or names is not None:
                # Metadata passed to super().__init__() replaces the outer
                # call's. Keep the entries set before that call.
                td = self._tensordict
                td = TensorDict(
                    td._tensordict,
                    batch_size=td.batch_size if batch_size is None else batch_size,
                    device=td.device if device is None else device,
                    names=td._maybe_names() if names is None else names,
                )
                object.__setattr__(self, "_tensordict", td)
            __init__(self, *args, **kwargs)
            if outer:
                # Move the fields that __init__ set with object.__setattr__
                # (as a frozen class does) into the container, and give those
                # it did not set their default.
                if self._tensordict.is_locked:
                    # super().__init__(..., lock=True) locked it already.
                    self._tensordict.unlock_()
                    lock = True
                self_dict = self.__dict__
                td_dict = self._tensordict._tensordict
                non_td = self._non_tensordict
                compiling = is_dynamo_compiling()
                if "_init_none_fields" not in self_dict:
                    init_none_fields = ()
                elif compiling:
                    init_none_fields = _pop_instance_attr_traced(
                        self, "_init_none_fields"
                    )
                else:
                    init_none_fields = self_dict.pop("_init_none_fields")
                for key, field in type(self).__dataclass_fields__.items():
                    if key not in required_fields:
                        continue
                    if key in self_dict:
                        if compiling:
                            value = _pop_instance_attr_traced(self, key)
                        else:
                            value = self_dict.pop(key)
                    elif key in td_dict or key in init_none_fields or key in non_td:
                        continue
                    elif field.default is None and not type(self)._is_non_tensor:
                        # A None field is a missing key
                        continue
                    elif field.default_factory is not dataclasses.MISSING:
                        value = field.default_factory()
                    elif field.default is not dataclasses.MISSING:
                        value = field.default
                    elif field.init:
                        raise TypeError(
                            f"{type(self).__name__}.__init__() did not set the "
                            f"required field {key!r}"
                        )
                    else:
                        value = None
                    type(self).set(self, key, value)
                if lock is None:
                    lock = frozen
            if lock:
                self._tensordict.lock_()
            return

        if "batch_size" in required_params:
            batch_size = torch.Size(())
        else:
            batch_size = kwargs.pop("batch_size", torch.Size(()))
        if isinstance(batch_size, int):
            batch_size = (batch_size,)
        elif batch_size is None:
            batch_size = torch.Size(())

        if "names" in required_params:
            names = None
        else:
            names = kwargs.pop("names", None)
        if "device" in required_params:
            device = None
        else:
            device = kwargs.pop("device", None)
        if "lock" in required_params:
            lock = None
        else:
            lock = kwargs.pop("lock", None)
        if lock is None:
            lock = frozen
        uses_init_vars = needs_init_vars and (
            type(self) is cls or type(self).__post_init__ is post_init
        )
        if args:
            # Use __dataclass_fields__ but filter out ClassVar fields to preserve order
            expected_keys_list = (
                (positional_names if uses_init_vars else old_positional_names)
                # a subclass that is not a tensorclass itself maps as cls
                if type(self) is cls
                or type(self).__dataclass_fields__ is cls.__dataclass_fields__
                else [
                    key
                    for key in type(self).__dataclass_fields__
                    if key in self.__expected_keys__
                ]
            )

            # Check that we don't have too many positional arguments
            if len(args) > len(expected_keys_list):
                raise TypeError(
                    f"{type(self).__name__}.__init__() takes {len(expected_keys_list) + 1} positional arguments but {len(args) + 1} were given"
                )

            for value, key in zip(args, expected_keys_list):
                if key in kwargs:
                    raise ValueError(f"The key {key} is already set in kwargs")
                kwargs[key] = value
        if not is_compiling():
            if (
                can_init_tensors
                and type(self) is cls
                and names is None
                and not cls._autocast
                and not cls._is_non_tensor
                and cls.set is _set
                and not hasattr(cls, "__post_init__")
                and kwargs.keys() == required_fields
                and all(type(value) is torch.Tensor for value in kwargs.values())
            ):
                # Construct once instead of routing every field through set().
                # Unindexed accelerator devices retain the general path:
                # TensorDict normalizes them, whereas tensorclass does not.
                batch_size = torch.Size(batch_size)
                device = torch.device(device) if device is not None else None
                if (
                    device is None
                    or device.index is not None
                    or device.type in ("cpu", "meta")
                ):
                    td = TensorDict(
                        kwargs, batch_size=batch_size, device=device, non_blocking=False
                    )
                    object.__setattr__(self, "_tensordict", td)
                    object.__setattr__(self, "_non_tensordict", {})
                    object.__setattr__(self, "_is_initialized", True)
                    if lock:
                        td.lock_()
                    return
            if (
                can_init_non_tensor
                and type(self) is cls
                and not cls._autocast
                and cls.set is _set
                and kwargs.keys() <= required_fields
                and non_tensor_required <= kwargs.keys()
            ):
                # Same result as set() with a non-tensor self: every value,
                # unwrapped from NonTensorData, goes to _non_tensordict.
                non_tensordict = {}
                for key, value in kwargs.items():
                    while is_non_tensor(value):
                        value = value.data
                    non_tensordict[key] = value
                for key, value in non_tensor_defaults.items():
                    non_tensordict.setdefault(key, value)
                td = TensorDict._new_unsafe(
                    {},
                    batch_size=torch.Size(batch_size),
                    device=device,
                    names=names,
                )
                object.__setattr__(self, "_tensordict", td)
                object.__setattr__(self, "_non_tensordict", non_tensordict)
                object.__setattr__(self, "_is_initialized", True)
                if hasattr(cls, "__post_init__"):
                    self.__post_init__()
                if lock:
                    td.lock_()
                return

        # Use `is`/isinstance instead of `in (..., dataclasses.MISSING)`:
        # under torch.compile, Dynamo can't proxy `_MISSING_TYPE` for `==`
        # comparisons against a tensor default value.
        _missing_type = getattr(dataclasses, "_MISSING_TYPE", type(dataclasses.MISSING))
        for key, field in type(self).__dataclass_fields__.items():
            # Only process fields that are in __expected_keys__ (excludes ClassVar fields)
            if key in self.__expected_keys__ and key not in kwargs:
                if field.default_factory is not dataclasses.MISSING and not isinstance(
                    field.default_factory, _missing_type
                ):
                    if _has_custom_setattr and not field.init:
                        # The dataclass __init__ calls this factory itself.
                        continue
                    default = field.default_factory()
                else:
                    default = field.default
                if default is not None and not isinstance(default, _missing_type):
                    kwargs.setdefault(key, default)

        missing_params = [p for p in required_params if p not in kwargs]
        if missing_params:
            n_missing = len(missing_params)
            missing = (
                f"missing {n_missing} required positional "
                f"argument{'' if n_missing == 1 else 's'}: "
                f"""{", ".join(f"'{name}'" for name in missing_params)}"""
            )
            expected_keys = self.__expected_keys__
            unknown = [
                key
                for key in kwargs
                if key not in expected_keys and key not in init_var_names
            ]
            if not unknown:
                msg = f"{type(self).__name__}.__init__() {missing}"
            else:
                # a misspelled field is the likely cause: report it first
                msg = (
                    f"{type(self).__name__}.__init__() got an unexpected keyword "
                    f"argument {unknown[0]!r}."
                    f"{_did_you_mean(unknown[0], expected_keys)}"
                    if len(unknown) == 1
                    else f"{type(self).__name__}.__init__() got unexpected "
                    f"keyword arguments {unknown}."
                )
                msg += f" Its fields are {sorted(expected_keys)}. It is also {missing}."
            raise TypeError(msg)

        super(type(self), self).__setattr__(
            "_tensordict",
            TensorDict._new_unsafe(
                {},
                batch_size=torch.Size(batch_size),
                device=device,
                names=names,
            ),
        )
        # super(type(self), self).__setattr__("_non_tensordict", {})
        # super(type(self), self).__setattr__("_is_initialized", True)
        object.__setattr__(self, "_non_tensordict", {})
        object.__setattr__(self, "_is_initialized", True)

        if _has_custom_setattr:
            # The class defines a custom __setattr__ that must be
            # respected during init. Fall back to the dataclass __init__
            # which routes through __setattr__ for each field.
            __init__(self, **kwargs)
        else:
            # Bypass the dataclass __init__ to avoid per-key __setattr__
            # overhead. The dataclass __init__ just does self.key = value
            # per field, which goes through __setattr__ → set(). Calling
            # set() directly saves the __setattr__ dispatch cost. For
            # frozen dataclasses this also avoids double-setting (dataclass
            # __init__ writes to __dict__ via object.__setattr__, then
            # _setattr re-routes to _tensordict, then __dict__ cleanup).
            #
            # Fields with None defaults are intentionally skipped by the
            # default-handling above, but the dataclass __init__ would
            # still assign them. Ensure every expected key is present.
            # The InitVars go to __post_init__ only: required ones are in
            # kwargs, as the check above shows. If __post_init__ does not
            # need them, it gets them only when one is given.
            init_var_values = (
                [kwargs.pop(name, default) for name, default in init_vars]
                if uses_init_vars
                or (init_vars and any(name in kwargs for name in init_var_names))
                else ()
            )
            for key in self.__expected_keys__:
                kwargs.setdefault(key, None)
            if tensor_only:
                if (
                    len(kwargs) != len(self.__expected_keys__)
                    and not is_dynamo_compiling()
                ):
                    # Dynamo cannot trace warnings.warn, so only eager calls warn.
                    _warn_unknown_tensor_only_fields(type(self), kwargs)
                # Fast path: validate and assign directly to the
                # underlying dict, skipping the set → _set_str →
                # _tensordict._set_str chain entirely.
                _td = self._tensordict
                _td_dict = _td._tensordict
                _validate = _td._validate_value
                _tensordict_fields = type(self)._tensordict_fields
                for key, value in kwargs.items():
                    if value is None:
                        continue
                    else:
                        if _tensordict_fields:
                            value, _ = _convert_mapping_for_field(
                                key, value, _tensordict_fields
                            )
                        _td_dict[key] = _validate(
                            value, check_shape=True, non_blocking=False, key=key
                        )
            else:
                set_value = type(self).set
                for key, value in kwargs.items():
                    set_value(self, key, value)
            if hasattr(type(self), "__post_init__"):
                self.__post_init__(*init_var_values)
        if lock:
            self._tensordict.lock_()

    if not shadow:
        new_params = [
            inspect.Parameter("batch_size", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("device", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("names", inspect.Parameter.KEYWORD_ONLY, default=None),
        ]
    else:
        new_params = []
        for p in params:
            if p._name == "batch_size":
                break
        else:
            new_params.append(
                inspect.Parameter("batch_size", inspect.Parameter.KEYWORD_ONLY)
            )
        for p in params:
            if p._name == "device":
                break
        else:
            new_params.append(
                inspect.Parameter(
                    "device", inspect.Parameter.KEYWORD_ONLY, default=None
                )
            )
        for p in params:
            if p._name == "names":
                break
        else:
            new_params.append(
                inspect.Parameter("names", inspect.Parameter.KEYWORD_ONLY, default=None)
            )
    # A user-defined __init__ may declare these itself or take **kwargs, which
    # must stay last.
    new_params = [p for p in new_params if p.name not in init_sig.parameters]
    var_kwargs = []
    if params and params[-1].kind is inspect.Parameter.VAR_KEYWORD:
        var_kwargs = [params.pop()]
    wrapper.__signature__ = init_sig.replace(
        parameters=params + new_params + var_kwargs
    )
    wrapper._user_init = __init__ if _custom_init else None

    return wrapper


def _cast_func(cls: type) -> Callable:
    """Returns the function that casts a value to the type annotation ``cls``."""
    # Not a module-level dict: Dynamo guards on all the keys of a global dict
    # read with a non-constant key, so a new type cast in eager would recompile.
    if cls is torch.Tensor:
        return torch.as_tensor
    if cls is np.ndarray:
        return np.asarray
    return cls


def _new_unsafe(cls, *args, **kwargs) -> T:
    return _from_tensordict(cls, TensorDict._new_unsafe(*args, **kwargs))


def _get_type_hints(cls, with_locals=False, tensor_only=False):
    #######
    # Set proper type annotations for autocasting to tensordict/tensorclass
    #
    # by updating locals, we can allow this to be used within a function
    # local-cross referencing will not work though
    # def foo():
    #     @tensorclass
    #     class MyOtherClass:
    #         x: torch.Tensor
    #     @tensorclass
    #     class MyClass:
    #         x: MyClass # works
    #         y: MyOtherClass # fails
    #
    # In this case, we will use the get_parent_local function to get the locals
    # from the parent frame and so recursively until we can find the class.

    if with_locals:
        # This function gets the parent frame recursively until we can find the current class.
        # Any exception leads to this to be None and auto-casting will be disabled
        localns = locals()
        localns = copy(localns)

        def get_parent_locals(cls, localns=localns):
            # Get the current frame
            frame = inspect.currentframe()
            try:
                parent_locs = localns
                while cls.__name__ not in parent_locs:
                    # Get the parent frame
                    parent_frame = frame.f_back
                    # Get the locals dictionary of the parent frame
                    parent_locs = parent_frame.f_locals
                    frame = parent_frame
            except Exception:
                localns.setdefault(cls.__name__, cls)
                return localns
            finally:
                # Clean up the frame reference
                del frame
            return copy(parent_locs)

        localns = get_parent_locals(cls)
    else:
        localns = None

    globalns = None

    cls._tensordict_fields = frozenset()
    cls._tensorclass_fields = {}
    cls._scalar_fields = {}
    try:
        type_hints = get_type_hints(
            cls,
            localns=localns,
            # globalns=globals(),
        )
        _set_tensorclass_type_hints(cls, type_hints)
        if tensor_only:

            def is_tensor_or_optional_tensor(type_hint):
                type_hint = _unwrap_array_annotation(type_hint)
                # Check if the type hint is exactly torch.Tensor
                if isinstance(type_hint, type):
                    return issubclass(type_hint, _TensorTypes) or _is_tensor_collection(
                        type_hint
                    )
                if isinstance(type_hint, type(Any)):
                    return False
                if isinstance(type_hint, UnionType):
                    args = get_args(type_hint)
                    return all(
                        t is None or t is NoneType or is_tensor_or_optional_tensor(t)
                        for t in args
                    )
                # Check if the type hint is a Union (e.g., Tensor | None)
                origin = get_origin(type_hint)

                if origin is Union:
                    args = get_args(type_hint)
                    return all(
                        t is None or t is NoneType or is_tensor_or_optional_tensor(t)
                        for t in args
                    )
                # Handle parameterized generics (e.g., TensorDict[str, Tensor])
                if origin is not None and isinstance(origin, type):
                    return issubclass(origin, _TensorTypes) or _is_tensor_collection(
                        origin
                    )
                return False

            for key, val in cls._type_hints.items():
                if key not in cls.__expected_keys__:
                    continue
                if not is_tensor_or_optional_tensor(val):
                    raise _TENSOR_ONLY_TYPE_ERR
        cls._type_hints = {
            key: val if isinstance(val, type) else _AnyType
            for key, val in cls._type_hints.items()
        }
    except NameError:
        if not with_locals:
            return _get_type_hints(cls, with_locals=True, tensor_only=tensor_only)
        cls._set_dict_warn_msg = (
            "A NameError occurred while trying to retrieve a type annotation. "
            "This can occur when a tensorclass references another locally defined "
            "tensorclass. "
            f"As a result type hints cannot be read and {cls}.from_dict(...) "
            f"or `{cls}.set` will not attempt to map dictionaries to "
            "the relevant tensorclass. To resolve this issue, consider defining "
            "your tensorclass globally."
        )
        cls._type_hints = None
    except TypeError as err:
        if err is _TENSOR_ONLY_TYPE_ERR:
            raise err
        # This is a rather common case where type annotation is like
        # class MyClass:
        #     x: int | str
        # in which case get_type_hints doesn't work (it does work
        # however with old-school Optional or Union...)
        # We simply differ the warning till _set() is called
        cls._set_dict_warn_msg = (
            "A TypeError occurred when trying to retrieve a type annotation. "
            "This may be caused by annotations that use plain `|` instead of typing.Union "
            "or typing.Optional which are supported. If you wish to use the feature "
            "of setting dict as attributes with automapping to tensordict/tensorclass "
            "(`my_obj.attr = dict(...)`), consider re-writing the tensorclass with "
            "traditional type annotations."
        )
        cls._type_hints = None


def _unwrap_array_annotation(type_hint: Any) -> Any:
    """Return the array type that a shape annotation such as jaxtyping's ``Float[Tensor, "3"]`` wraps.

    jaxtyping stores it as ``array_type`` and the shape as ``dim_str``; other
    annotations are returned as is.
    """
    if isinstance(type_hint, type) and isinstance(
        getattr(type_hint, "dim_str", None), str
    ):
        array_type = getattr(type_hint, "array_type", None)
        if isinstance(array_type, type):
            return array_type
    return type_hint


def _is_tensordict_annotation(type_hint: Any) -> bool:
    """Return whether an annotation contains a TensorDictBase type."""
    origin = get_origin(type_hint)
    if origin in (Union, UnionType):
        return any(_is_tensordict_annotation(arg) for arg in get_args(type_hint))
    if origin is not None:
        type_hint = origin
    return isinstance(type_hint, type) and issubclass(type_hint, TensorDictBase)


def _tensorclass_annotation(type_hint: Any) -> type | None:
    """Return the tensorclass of a ``T``, ``T[...]`` or ``Optional[T]`` annotation, else ``None``.

    The base ``TensorClass``, which declares no fields, and the non-tensor classes
    (``NonTensorData``, ``MetaData``) do not count.
    """
    if get_origin(type_hint) in (Union, UnionType):
        args = [arg for arg in get_args(type_hint) if arg is not NoneType]
        if len(args) != 1:
            return None
        type_hint = args[0]
    origin = get_origin(type_hint)
    if origin is not None:
        type_hint = origin
    if (
        isinstance(type_hint, type)
        and _is_tensorclass(type_hint)
        and "__expected_keys__" in vars(type_hint)
        and not getattr(type_hint, "_is_non_tensor", False)
    ):
        return type_hint
    return None


def _fits_tensorclass(tensorclass_type: type, entry: Any) -> bool:
    """Return whether ``entry`` is a tensordict whose keys are all fields of ``tensorclass_type``."""
    if not isinstance(entry, TensorDictBase) or is_non_tensor(entry):
        return False
    expected_keys = tensorclass_type.__expected_keys__
    return all(key in expected_keys for key in entry.keys())


def _scalar_annotation_types(type_hint: Any) -> tuple[type, ...] | None:
    """Return the scalar types that an ``int``, ``float`` or ``bool`` annotation (or a union of these, optional or not) accepts, else ``None``."""
    if get_origin(type_hint) in (Union, UnionType):
        args = [arg for arg in get_args(type_hint) if arg is not NoneType]
    else:
        args = [type_hint]
    types = set()
    for arg in args:
        if arg is bool:
            types.add(bool)
        elif arg is int:
            types.add(int)
        elif arg is float:
            # an int is a valid float (PEP 484)
            types.update((int, float))
        else:
            return None
    return tuple(types) or None


def _warn_scalar_field(cls: type, key: str, value: Any) -> None:
    """Warns that ``value``, given to the int, float or bool field ``key``, becomes a 0-d tensor today."""
    warnings.warn(
        f"The {type(value).__name__} given to the field {key!r} of "
        f"{cls.__name__} is stored as a 0-d tensor today. Starting with "
        "TensorDict 0.17, the default will change to "
        "keeping the Python value of fields annotated int/float/bool, as "
        "NonTensorData. Annotate the field torch.Tensor to keep the tensor, or "
        "use the nocast class option to keep the Python value now.",
        FutureWarning,
        stacklevel=_user_stacklevel(),
    )


def _set_tensorclass_type_hints(cls: type, type_hints: dict[str, Any]) -> None:
    """Store resolved hints and cache the TensorDict-, tensorclass- and scalar-annotated fields."""
    cls._tensordict_fields = frozenset(
        key
        for key, val in type_hints.items()
        if key in cls.__expected_keys__ and _is_tensordict_annotation(val)
    )
    tensorclass_fields = {}
    scalar_fields = {}
    for key, val in type_hints.items():
        if key not in cls.__expected_keys__:
            continue
        tensorclass_type = _tensorclass_annotation(val)
        if tensorclass_type is not None:
            tensorclass_fields[key] = tensorclass_type
        scalar_types = _scalar_annotation_types(val)
        if scalar_types is not None:
            scalar_fields[key] = scalar_types
    cls._tensorclass_fields = tensorclass_fields
    cls._scalar_fields = scalar_fields
    cls._type_hints = {
        key: _unwrap_array_annotation(val) for key, val in type_hints.items()
    }


def _normalize_nested_mapping(
    mapping: Mapping[NestedKey, Any],
) -> dict[NestedKey, Any] | TensorDictBase:
    """Materialize nested mappings while preserving TensorDictBase values."""
    if isinstance(mapping, TensorDictBase):
        return mapping
    return {
        key: _normalize_nested_mapping(value) if isinstance(value, Mapping) else value
        for key, value in mapping.items()
    }


def _convert_mapping_for_field(
    key: str, value: Any, tensordict_fields: AbstractSet[str]
) -> tuple[Any, bool]:
    """Normalize a Mapping assigned to a TensorDict-annotated field.

    Returns the normalized value and whether the field/value pair required
    Mapping handling. Existing TensorDict values are returned unchanged.
    """
    is_mapping_field = key in tensordict_fields and isinstance(value, Mapping)
    if is_mapping_field:
        value = _normalize_nested_mapping(value)
    return value, is_mapping_field


def _from_tensordict(
    cls,
    tensordict: TensorDictBase,
    non_tensordict: dict | None = None,
    safe: bool = True,
    *,
    rebuild: bool = True,
) -> Self:  # noqa: D417
    """Tensor class wrapper to instantiate a new tensor class object.

    Args:
        tensordict (TensorDictBase): Dictionary of tensor types
        non_tensordict (dict): Dictionary with non-tensor and nested tensor class objects
        safe (bool): Whether to raise an error if the tensordict is not a TensorDictBase instance
        rebuild (bool): Whether the data comes from an existing instance, as in
            ``clone()`` or indexing, rather than from outside, as in
            ``from_tensordict()`` or a load. A class with ``skip_post_init_on_rebuild``
            does not run ``__post_init__`` on a rebuild.

    """
    if safe and not isinstance(tensordict, TensorDictBase):
        raise RuntimeError(
            f"Expected a TensorDictBase instance but got {type(tensordict)}"
        )
    none_is_missing = not cls._is_non_tensor
    if none_is_missing:
        # A None field is a key missing from the tensordict (except for
        # NonTensorData, which keeps its data in _non_tensordict). Copy the
        # dict: callers may pass the one of an instance.
        non_tensordict = (
            {k: v for k, v in non_tensordict.items() if v is not None}
            if non_tensordict
            else {}
        )
    # Validating keys of tensordict
    # tensordict = tensordict.copy()
    tensor_keys = tensordict.keys()
    # TODO: compile doesn't like set() over an arbitrary object
    if is_compiling():
        tensor_keys = {k for k in tensor_keys}  # noqa: C416
        exp_keys = {k for k in cls.__expected_keys__}  # noqa: C416
        if non_tensordict is not None:
            nontensor_keys = {k for k in non_tensordict.keys()}  # noqa: C416
        else:
            nontensor_keys = set()
            non_tensordict = {}
        # TODO: Makes compile unhappy
        # total_keys = tensor_keys.union(nontensor_keys)
        total_keys = set(tensor_keys)
        total_keys.update(nontensor_keys)
    else:
        tensor_keys = set(tensor_keys)
        exp_keys = set(cls.__expected_keys__)
        if non_tensordict is not None:
            nontensor_keys = set(non_tensordict.keys())
            total_keys = tensor_keys.union(nontensor_keys)
        else:
            nontensor_keys = set()
            non_tensordict = {}
            total_keys = tensor_keys
    for key in nontensor_keys:
        if key not in tensor_keys:
            continue
        if non_tensordict[key] is None:
            del non_tensordict[key]
            continue
        raise KeyError(f"{key} is present in both tensor and non-tensor dicts.")
    if total_keys - exp_keys:
        raise ValueError(
            f"Keys from the tensordict ({set(tensordict.keys())}) must "
            f"correspond to the class attributes ({cls.__expected_keys__}). Got the set of "
            f"keys {{{total_keys - exp_keys}}} which do not belong to the class."
        )
    elif not none_is_missing:
        to_add = exp_keys - total_keys
        for key in to_add:
            non_tensordict[key] = None

    # Bypass __init__ — we already have a tensordict to use as the underlying
    # storage so there is no need to create an empty one and populate it.
    # Under torch.compile the cls(**kwargs) path caused graph breaks because
    # Dynamo cannot trace through the TensorClass __init__ wrapper (property
    # descriptors, shadow-mode fields, etc.).  Using __new__ + direct attribute
    # setting is both faster in eager mode and compile-friendly.
    tc = cls.__new__(cls)
    if not is_compiling():
        tc.__dict__["_tensordict"] = tensordict
        tc.__dict__["_non_tensordict"] = non_tensordict
    else:
        object.__setattr__(tc, "_tensordict", tensordict)
        object.__setattr__(tc, "_non_tensordict", non_tensordict)
    # since we aren't calling the dataclass __init__, we need to manually
    # check whether a __post_init__ method has been defined and invoke it
    if (
        hasattr(cls, "__post_init__")
        and not (rebuild and cls._skip_post_init_on_rebuild)
        and (is_compiling() or not _SKIP_TENSORCLASS_POST_INIT.get())
    ):
        if isinstance(tensordict, LazyStackedTensorDict):
            # Run it on each stacked instance, which sees its own fields: on
            # the stack, reading a field of uneven shape would raise. The
            # fields it sets go to the members, which the stack holds.
            _post_init_members(cls, tensordict, non_tensordict)
        else:
            tc.__post_init__(*_post_init_args(cls))
    return tc


def _post_init_members(cls, tensordict: LazyStackedTensorDict, non_tensordict: dict):
    """Run ``cls.__post_init__`` on an instance built from each member of ``tensordict``.

    The keys were checked on the stack: a member may hold more keys than the
    stack, which keeps those that all the members share.
    """
    for member in tensordict.tensordicts:
        if isinstance(member, LazyStackedTensorDict):
            _post_init_members(cls, member, non_tensordict)
            continue
        tc = cls.__new__(cls)
        object.__setattr__(tc, "_tensordict", member)
        object.__setattr__(tc, "_non_tensordict", dict(non_tensordict))
        tc.__post_init__(*_post_init_args(cls))


def _tensorclass_keys(cls, tensordict: TensorDict) -> list[str]:
    """Return the tensorclass fields of ``cls`` whose entry in ``tensordict`` fits their class."""
    return [
        key
        for key, tensorclass_type in cls._tensorclass_fields.items()
        if _fits_tensorclass(tensorclass_type, tensordict._get_str(key, None))
    ]


def _set_tensorclass_entries(cls, tensordict: TensorDict, keys: list[str]) -> None:
    """Replace the entries ``keys`` of ``tensordict`` with the tensorclasses of these fields of ``cls``.

    ``tensordict`` must own its nested TensorDicts (a ``copy()``): they are modified in place.
    """
    for key in keys:
        tensorclass_type = cls._tensorclass_fields[key]
        entry = tensordict._get_str(key, None)
        if tensorclass_type._tensorclass_fields and isinstance(entry, TensorDict):
            nested_keys = _tensorclass_keys(tensorclass_type, entry)
            if nested_keys:
                _set_tensorclass_entries(tensorclass_type, entry, nested_keys)
        tensordict._set_str(
            key,
            tensorclass_type._from_tensordict(entry, rebuild=False),
            validated=True,
            inplace=False,
            non_blocking=False,
        )


def _from_tensordict_public(
    cls,
    tensordict: TensorDictBase,
    non_tensordict: dict | None = None,
    safe: bool | None = None,
) -> Self:
    """Wraps a tensordict in a new instance of the tensorclass, without copying the leaves.

    If ``tensordict`` is a :class:`~tensordict.TensorDict`, the entries of the fields
    annotated with a tensorclass (or ``Optional`` of one) that are tensordicts become
    instances of that tensorclass, recursively. They are set in a shallow copy of
    ``tensordict``, which keeps its own entries. An entry with keys that the annotated
    class does not declare stays a tensordict. Other backends are wrapped as they are:
    setting a tensorclass in them would write to their storage (e.g. an H5 file).

    Args:
        tensordict (TensorDictBase): the tensordict that holds the tensor fields.
        non_tensordict (dict, optional): the values of the non-tensor fields. The
            fields that neither argument holds are set to ``None``. Defaults
            to ``None``.
        safe (bool, optional): deprecated in 0.15, will be removed in TensorDict 0.17.
            If ``False``, skips the check that ``tensordict`` is a
            :class:`~tensordict.TensorDictBase`. By default, the check runs, and
            another type raises a ``RuntimeError``.

    Returns:
        A new instance of the class, whose fields are the entries of
        ``tensordict``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Pose(TensorClass):
        ...     q: torch.Tensor
        >>> class Obs(TensorClass):
        ...     a: Pose
        >>> obs = Obs(a=Pose(q=torch.zeros(5, 4), batch_size=[5]), batch_size=[5])
        >>> type(Obs.from_tensordict(obs.to_tensordict()).a).__name__
        'Pose'
    """
    if safe is None:
        safe = True
    else:
        warn_deprecated("TensorClass.from_tensordict(safe=...)", removal="0.17")
    # The tensorclass ops rebuild their results with _from_tensordict: their
    # nested entries are already tensorclasses.
    if cls._tensorclass_fields and isinstance(tensordict, TensorDict):
        keys = _tensorclass_keys(cls, tensordict)
        if keys:
            # The copy owns its nested TensorDicts, so the tensorclasses can be
            # set in them, at every level, without another copy.
            is_locked = tensordict.is_locked
            tensordict = tensordict.copy()
            _set_tensorclass_entries(cls, tensordict, keys)
            if is_locked:
                tensordict.lock_()
    return cls._from_tensordict(tensordict, non_tensordict, safe, rebuild=False)


def _memmap_(
    self,
    *,
    prefix: str | None = None,
    copy_existing: bool = False,
    executor=None,
    futures=None,
    inplace=True,
    like=False,
    memmaped: bool = False,
    share_non_tensor: bool = False,
    existsok: bool = True,
    robust_key,
    non_tensordict: dict | None = None,
):
    if non_tensordict is None:
        non_tensordict = self._non_tensordict
    _non_tensordict = dict(non_tensordict)
    cls = type(self)

    if not memmaped and prefix is not None:
        prefix = Path(prefix)
        if not prefix.exists():
            os.makedirs(prefix, exist_ok=True)

        def save_metadata(cls=cls, _non_tensordict=_non_tensordict, prefix=prefix):
            metadata = {"_type": str(cls)}
            to_pickle = {}
            for key, value in _non_tensordict.items():
                value = _from_shared_nontensor(value)
                if _is_json_serializable(value):
                    metadata[key] = value
                else:
                    to_pickle[key] = value
            from tensordict._utils_key_json import json_dumps

            json_str = json_dumps(metadata)
            # Ensure we write bytes to the binary file
            if isinstance(json_str, str):
                json_str = json_str.encode("utf-8")
            # Serialize before opening any file, so that a value that cannot be
            # pickled leaves the files on disk as they were.
            pickled = pickle.dumps(to_pickle) if to_pickle else None
            with open(prefix / "meta.json", "wb") as f:
                f.write(json_str)
            if pickled is not None:
                with open(prefix / "other.pickle", "wb") as pickle_file:
                    pickle_file.write(pickled)
                _save_json_types(prefix, to_pickle, pickled)
            else:
                (prefix / "other.pickle").unlink(missing_ok=True)
                (prefix / _JSON_TYPES_FILE).unlink(missing_ok=True)

        if executor is None:
            save_metadata()
        else:
            futures.append(executor.submit(save_metadata))

        prefix = prefix / "_tensordict"
    new_futures = []
    if not isinstance(self, NonTensorDataBase):
        # TODO: We can't execute this using multiple threads because from_tensordict expects
        #  the tensordict and non_tensordict to be complete
        td = self._tensordict._memmap_(
            prefix=prefix,
            # executor=None,
            # futures=[],
            executor=executor,
            futures=new_futures,
            inplace=inplace,
            like=like,
            copy_existing=copy_existing,
            share_non_tensor=share_non_tensor,
            existsok=existsok,
            robust_key=robust_key,
        )
        if new_futures:
            futures += new_futures
        td._device = torch.device("cpu")
    else:
        # For non-tensor data, we don't create an empty _tensordict dir
        td = self._tensordict.empty()
        td._is_memmap = True
        td._is_locked = True
        td._memmap_prefix = prefix
        if inplace:
            self.__dict__["_tensordict"] = td
    if not inplace:
        if new_futures:
            concurrent.futures.wait(new_futures)
        result = cls._from_tensordict(td, _non_tensordict)
    else:
        result = self
    return result


def _share_memory_(self):
    self._tensordict.share_memory_()
    return self


def _save_json_types(prefix: Path, to_pickle: dict, pickled: bytes) -> None:
    # When other.pickle holds only classes of _JSON_TYPES, save their names
    # with the hash of other.pickle, so that a file that an older version
    # rewrote later is not read with stale names.
    path = prefix / _JSON_TYPES_FILE
    if not all(
        isinstance(value, type) and value in _JSON_TYPE_NAMES
        for value in to_pickle.values()
    ):
        path.unlink(missing_ok=True)
        return
    saved = {
        "types": {key: _JSON_TYPE_NAMES[value] for key, value in to_pickle.items()},
        "pickle_sha256": hashlib.sha256(pickled).hexdigest(),
    }
    with open(path, "w") as f:
        json.dump(saved, f)


def _load_json_types(prefix: Path) -> dict | None:
    # The classes that other.pickle holds, read without pickle, or None if
    # the save did not record them (e.g. files written by older versions).
    path = prefix / _JSON_TYPES_FILE
    if not path.exists() or not (prefix / "other.pickle").exists():
        return None
    with path.open("rb") as f:
        saved = json.loads(f.read())
    with (prefix / "other.pickle").open("rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != saved.get("pickle_sha256"):
            return None
    types = {}
    for key, name in saved["types"].items():
        if name not in _JSON_TYPES:
            raise ValueError(
                f"Unknown type name {name!r} for the field {key!r} in {path}. "
                f"Expected one of {sorted(_JSON_TYPES)}."
            )
        types[key] = _JSON_TYPES[name]
    return types


def _load_pickle_file(path, *, allow_pickle: bool | None):
    if allow_pickle is False:
        raise RuntimeError(
            "Refusing to load pickled non-tensor data from a memory-mapped "
            "TensorDict. Pickle can execute arbitrary code. Pass "
            "allow_pickle=True only when the saved data is trusted."
        )
    if allow_pickle is None:
        warnings.warn(
            "TensorDict.load_memmap is loading a pickled non-tensor field. "
            "This is only safe for trusted data. Starting with TensorDict "
            "0.16, the default will change to allow_pickle=False. Pass "
            "allow_pickle=True to acknowledge trusted data or "
            "allow_pickle=False to reject pickle now.",
            FutureWarning,
            stacklevel=5,
        )
    with path.open("rb") as pickle_file:
        return pickle.load(pickle_file)


def _load_memmap(
    cls,
    prefix: Path,
    metadata: dict,
    *,
    robust_key,
    allow_pickle: bool | None = None,
    **kwargs,
):
    non_tensordict = dict(metadata)
    del non_tensordict["_type"]
    # Path methods (rather than os.path) so that archive paths can be
    # traversed through the same code path as regular directories.
    json_types = _load_json_types(prefix)
    if json_types is not None:
        non_tensordict.update(json_types)
    elif (prefix / "other.pickle").exists():
        non_tensordict.update(
            _load_pickle_file(
                prefix / "other.pickle",
                allow_pickle=allow_pickle,
            )
        )
    out = kwargs.pop("out", None)
    tc_out = None
    if is_tensorclass(out):
        # Only an instance of the saved class can be loaded in place. The
        # saved class is looked up by name, so compare names: a class that is
        # redefined under the same name (e.g. in a notebook) also matches.
        tc_out = out if str(type(out)) == str(cls) else None
        out = tc_out._tensordict if tc_out is not None else None
    if (prefix / "_tensordict").exists():
        td = TensorDict.load_memmap(
            prefix / "_tensordict",
            **kwargs,
            non_blocking=False,
            out=out,
            robust_key=robust_key,
            allow_pickle=allow_pickle,
        )
    else:
        if not issubclass(cls, NonTensorDataBase):
            raise ValueError("The _tensordict directory seems to be missing.")
        td = TensorDict(device="cpu") if tc_out is None else tc_out._tensordict
    result = cls._from_tensordict(td, non_tensordict, rebuild=False)
    if tc_out is not None:
        tc_out._non_tensordict.clear()
        tc_out._non_tensordict.update(result._non_tensordict)
        return tc_out
    return result


def _load_memmap_(
    self,
    prefix: str | Path,
    robust_key: bool | None = True,
    *,
    allow_pickle: bool | None = None,
    mode: str | None = None,
):
    """Loads a memory-mapped tensorclass into this instance, and returns ``self``.

    Works as :meth:`~tensordict.TensorDictBase.load_memmap_`, and also loads
    the non-tensor fields.

    Args:
        prefix (str or Path): the directory, or the ``".tdz"`` archive file,
            to read.
        robust_key (bool, optional): the key encoding used when the data was
            saved. See :meth:`~tensordict.TensorDictBase.memmap`. Defaults to
            ``True``.

    Keyword Args:
        allow_pickle (bool, optional): whether pickled non-tensor fields may
            be loaded. See :meth:`~tensordict.TensorDictBase.load_memmap`.
            Defaults to ``None``, which loads pickled fields with a
            ``FutureWarning``.
        mode (str, optional): the mode of the memory maps. See
            :meth:`~tensordict.TensorDictBase.load_memmap`. Defaults to
            ``None``.
    """
    # Lock and memmap state live on the tensordict, but loading through the
    # tensorclass also refreshes the non-tensor fields.
    td = self._tensordict
    is_memmap = td.is_memmap()
    if is_memmap:
        td.unlock_()
    loaded = type(self).load_memmap(
        prefix,
        device=td.device,
        out=self,
        robust_key=robust_key,
        allow_pickle=allow_pickle,
        mode=mode,
    )
    # A tensordict save (e.g. the ``_tensordict`` directory refreshed by
    # ``memmap_refresh_``) is loaded into ``td``.
    if loaded is not self and loaded is not td:
        raise ValueError(
            f"Cannot load a saved {type(loaded).__name__} in place into an "
            f"instance of {type(self).__name__}."
        )
    if is_memmap and td._memmap_prefix is not None:
        td.memmap_()
    return self


def _getstate(self) -> dict[str, Any]:
    """Returns a state dict which consists of tensor and non_tensor dicts for serialization.

    Returns:
        dictionary of state of tensor class

    """
    non_tensordict = self._non_tensordict
    if not type(self)._is_non_tensor:
        # Pickle the None fields as earlier versions did, so that they can load it
        non_tensordict = {**non_tensordict, **dict.fromkeys(_none_fields(self))}
    return {"tensordict": self._tensordict, "non_tensordict": non_tensordict}


def _setstate(self, state: dict[str, Any]) -> None:  # noqa: D417
    """Used to set the state of an object using state parameter.

    Args:
        state (dict): State parameter to set the object
    """
    if type(self) is NonTensorData:
        # perf: what the code below does for a NonTensorData, as _setattr
        # calls object.__setattr__ while these attributes are missing
        object.__setattr__(self, "_tensordict", state.get("tensordict"))
        object.__setattr__(self, "_non_tensordict", state.get("non_tensordict"))
        return
    non_tensordict = state.get("non_tensordict")
    if not type(self)._is_non_tensor and non_tensordict:
        # Pickled by a version that kept None fields in _non_tensordict
        non_tensordict = {k: v for k, v in non_tensordict.items() if v is not None}
    if type(self)._frozen:
        # Bypass the dataclass guard while preserving custom __setattr__ behavior
        # for non-frozen tensorclasses.
        object.__setattr__(self, "_tensordict", state.get("tensordict"))
        object.__setattr__(self, "_non_tensordict", non_tensordict)
    else:
        self._tensordict = state.get("tensordict")
        self._non_tensordict = non_tensordict


def _none_fields(self) -> list[str]:
    # The None fields of an instance, in field order: the fields
    # missing from its TensorDict (and from _non_tensordict, which is empty
    # unless the instance was built with non-tensor values there).
    td = self._tensordict
    expected_keys = self.__expected_keys__
    if type(td) is TensorDict:
        # A TensorDict is checked with a dict lookup, much cheaper than keys().
        keys = td._tensordict
        if keys.keys() >= expected_keys:
            # no field is None
            return []
    else:
        keys = td.keys()
    non_tensordict = self._non_tensordict
    return [
        key
        for key in type(self).__dataclass_fields__
        if key not in keys and key not in non_tensordict and key in expected_keys
    ]


def _partial_field(td, key: str) -> Any:
    # A lazy stack's keys() lists only the keys every member holds. A field
    # that only some members set reads as the list of the members' values,
    # with None for a member that does not set it, like get(key, as_list=True).
    # Returns _UNSET if no member sets it.
    if not isinstance(td, LazyStackedTensorDict):
        return _UNSET
    values = []
    found = False
    for member in td.tensordicts:
        if type(member) is TensorDict:
            # cheaper than _get_str, which raises a KeyError for a miss
            out = member._tensordict.get(key)
        else:
            out = member._get_str(key, _UNSET)
            if out is _UNSET:
                out = _partial_field(member, key)
                if out is _UNSET:
                    out = None
        if out is not None:
            found = True
            if is_non_tensor(out):
                out = (
                    out.data
                    if not isinstance(out, NonTensorStack)
                    else out.tolist(as_linked_list=True)
                )
        values.append(out)
    return values if found else _UNSET


def _del_partial_field(td, key: str) -> None:
    # Deletes key from the members of a lazy stack that hold it, when the
    # stack's keys() does not list it because other members do not.
    if isinstance(td, LazyStackedTensorDict):
        for member in td.tensordicts:
            if type(member) is TensorDict:
                if key in member._tensordict:
                    member.del_(key)
            elif key in member.keys():
                member.del_(key)
            else:
                _del_partial_field(member, key)


def _non_tensor_keys(self):
    # The fields that the TensorDict of self does not hold
    if not type(self)._is_non_tensor:
        return [*self._non_tensordict, *_none_fields(self)]
    return self._non_tensordict.keys()


def _copy(self) -> Any:
    """Copies the tensorclass without cloning its tensors, like ``self.copy()``."""
    return self.copy()


def _deepcopy(self, memo: dict[int, Any]) -> Any:
    """Copies the tensorclass and clones its tensors, like ``self.clone()``."""
    return self.clone()


def _replace(self, /, **changes: Any) -> Any:
    """Implements :func:`copy.replace`: :func:`dataclasses.replace` that keeps the metadata.

    The device of ``self`` is the default of the call to ``__init__``. Without
    ``batch_size`` in ``changes``, the copy gets the batch size and names of
    ``self`` when ``__init__`` and ``__post_init__`` leave it the default
    batch size and its entries fit that batch size. Otherwise the copy keeps
    the batch size that :func:`dataclasses.replace` gives it. ``names`` are
    set on the copy with the names setter, after the sub-tensordicts that the
    copy shares with ``changes`` or ``self`` are replaced by shallow clones.
    """
    td = self._tensordict
    batch_size = td.batch_size
    fields = self.__dataclass_fields__
    if "device" not in changes and "device" not in fields:
        changes["device"] = td.device
    if "batch_size" in changes or "batch_size" in fields or "names" in fields:
        return dataclasses.replace(self, **changes)
    has_names = "names" in changes
    names = changes.pop("names", None)
    result = dataclasses.replace(self, **changes)
    result_td = result._tensordict
    ndim = len(batch_size)
    # The batch_size setter expands the non-tensor entries in place: only
    # those that the copy built itself with no batch dims may be shorter.
    # __init__ and __post_init__ may have set the batch size.
    fits = (
        batch_size
        and not result_td.batch_size
        and all(
            value.shape[:ndim] == batch_size
            or (
                is_non_tensor(value)
                and not value.batch_size
                and value is not changes.get(key)
            )
            for key, value in result_td.items()
        )
    )
    if fits and not has_names:
        names = td._maybe_names()
        has_names = names is not None and any(name is not None for name in names)
    if not fits and not has_names:
        return result
    if has_names:
        # The names setter renames the sub-tensordicts in place: copy those
        # that the copy shares with the caller or with self, as set() does
        shared = {id(value) for value in changes.values()}
        shared.update(id(value) for value in td.values())
        result_dict = result_td._tensordict
        for key, value in result_dict.items():
            if id(value) in shared and _is_tensor_collection(type(value)):
                result_dict[key] = value.clone(False)
    if fits:
        result_td.batch_size = batch_size
    if has_names:
        result_td.names = names
    return result


def _getattr_tensor_only(self, item: str, **kwargs) -> Any:
    # Guard against infinite recursion when _tensordict/_non_tensordict are
    # not yet set (e.g. during Dynamo tracing of the constructor or pytree
    # unflatten inside while_loop).  Raising AttributeError here is the
    # correct __getattr__ protocol — Python will propagate it as "attribute
    # not found".
    if item in ("_tensordict", "_non_tensordict"):
        raise AttributeError(item)
    # Use _UNSET sentinel instead of try/except for torch.compile compatibility
    out = self._tensordict._get_str(item, _UNSET, **kwargs)
    if out is not _UNSET:
        return out
    out = self._non_tensordict.get(item, _UNSET)
    if out is not _UNSET:
        return out
    out = getattr(self._tensordict, item, NO_DEFAULT)
    if out is not NO_DEFAULT:
        if not callable(out) and not is_non_tensor(out):
            return out
        if is_non_tensor(out):
            return out.data if hasattr(out, "data") else out.tolist(as_linked_list=True)
        return _wrap_method(self, item, out)
    raise _no_attribute_error(self, item)


def _getattr(self, item: str, **kwargs) -> Any:
    # Guard against infinite recursion when _tensordict/_non_tensordict are
    # not yet set (e.g. during Dynamo tracing of the constructor or pytree
    # unflatten inside while_loop).  Raising AttributeError here is the
    # correct __getattr__ protocol — Python will propagate it as "attribute
    # not found".
    if item in ("_tensordict", "_non_tensordict"):
        raise AttributeError(item)

    __dataclass_fields__ = type(self).__expected_keys__

    if item in __dataclass_fields__:
        _non_tensordict = self._non_tensordict
        if _non_tensordict and item not in self._tensordict.keys():
            out = _non_tensordict.get(item, NO_DEFAULT)
            if out is not NO_DEFAULT:
                if (
                    isinstance(self, NonTensorDataBase)
                    and item == "data"
                    and (self._is_shared or self._is_memmap)
                ):
                    return _from_shared_nontensor(out)
                return out
        td = self._tensordict
        if type(td) is TensorDict:
            # as in the tensor_only getter: cheaper than _get_str, and
            # Dynamo guards on a key that dict.get misses
            out = td._tensordict.get(item, _UNSET)
        else:
            out = td._get_str(item, _UNSET, **kwargs)
            if out is _UNSET and is_compiling() and item in td.keys():
                out = td._get_str(item, None, **kwargs)
            if out is _UNSET:
                out = _partial_field(td, item)
        if out is _UNSET:
            # A declared field that neither store holds is None
            return None
        if is_non_tensor(out):
            return (
                out.data
                if not isinstance(out, NonTensorStack)
                else out.tolist(as_linked_list=True)
            )
        return out

    out = getattr(self._tensordict, item, NO_DEFAULT)
    if out is not NO_DEFAULT:
        if not callable(out) and not is_non_tensor(out):
            return out
        if is_non_tensor(out):
            return out.data if hasattr(out, "data") else out.tolist(as_linked_list=True)
        return _wrap_method(self, item, out)
    raise _no_attribute_error(self, item)


# Reads a field of a tensorclass instance, as the field branch of _getattr does.
# Tensorclasses that are neither tensor_only nor shadow have one per field, so
# that x.field does not first fail the normal attribute lookup and then go
# through __getattr__. It defines no __set__, so an entry of the instance
# __dict__ (which a custom __setattr__ can write with object.__setattr__) still
# comes first. On the class it raises AttributeError, as the field had no class
# attribute before: otherwise dataclass() would take it as the default of a
# field that a subclass declares again. It has no docstring, so that Sphinx and
# help() do not show one for every field.
class _FieldGetter:
    __slots__ = ("_check_shared", "_name")

    def __init__(self, name: str, check_shared: bool):
        self._name = name
        # True for the data field of NonTensorData and MetaData.
        self._check_shared = check_shared

    def __get__(self, obj, objtype=None):
        name = self._name
        if obj is None:
            raise AttributeError(
                f"type object {objtype.__name__!r} has no attribute {name!r}"
            )
        _non_tensordict = obj._non_tensordict
        td = obj._tensordict
        if _non_tensordict and name in _non_tensordict and name not in td.keys():
            out = _non_tensordict[name]
            if self._check_shared and (td._is_shared or td._is_memmap):
                return _from_shared_nontensor(out)
            return out
        if type(td) is TensorDict:
            # as in the tensor_only getter: cheaper than _get_str, and
            # Dynamo guards on a key that dict.get misses
            out = td._tensordict.get(name, _UNSET)
        else:
            out = td._get_str(name, _UNSET)
            if out is _UNSET and is_compiling() and name in td.keys():
                out = td._get_str(name, None)
            if out is _UNSET:
                out = _partial_field(td, name)
        if out is _UNSET:
            # A declared field that neither store holds is None
            return None
        if is_non_tensor(out):
            return (
                out.data
                if not isinstance(out, NonTensorStack)
                else out.tolist(as_linked_list=True)
            )
        return out


SET_ATTRIBUTES = (
    "batch_size",
    "device",
    "_locked_tensordicts",
    "names",
    "_is_initialized",
)


def _setattr(self, key: str, value: Any) -> None:  # noqa: D417
    __dict__ = self.__dict__
    if (
        "_tensordict" not in __dict__
        or "_non_tensordict" not in __dict__
        or (
            not self._shadow
            and (
                key in SET_ATTRIBUTES
                # Check the fields first: once a compiled function has
                # modified a dict, the Dynamo of torch 2.12 cannot read
                # type(self).__dict__ (a mappingproxy).
                or (
                    key not in getattr(self, "__expected_keys__", set())
                    and key in type(self).__dict__
                )
            )
        )
    ):
        # if we ever decide to allow anything to be written in a tc
        # or key not in self.__dataclass_fields__):
        # Call object.__setattr__ directly (not via the __setattr_parent__
        # class attribute): dynamo re-dispatches a slot wrapper stored as a
        # class attribute through the custom __setattr__, which recurses.
        return object.__setattr__(self, key, value)

    if key not in self.__expected_keys__:
        raise AttributeError(
            f"Cannot set the attribute {key!r}: {type(self).__name__} has no such "
            f"field. Its fields are {sorted(self.__expected_keys__)}."
            f"{_did_you_mean(key, self.__expected_keys__)}"
        )
    out = self.set(key, value)
    if out is not self:
        raise RuntimeError(
            "Cannot set the attribute on a locked tensorclass, even if "
            "clone_on_set is set to True. Use my_obj.set(...) instead."
        )


def _set_none_field(self, key: str) -> None:
    # A field of an instance was set to None, and is no longer in its
    # TensorDict. _set and _setattr_tensor_only inline this.
    __dict__ = self.__dict__
    if "_init_none_fields" in __dict__:
        # A custom __init__ runs: see _init_wrapper
        __dict__["_init_none_fields"][key] = None
    else:
        non_tensordict = __dict__["_non_tensordict"]
        if non_tensordict and key in non_tensordict:
            # e.g. a value passed to from_tensordict(non_tensordict=...)
            del non_tensordict[key]


def _setattr_tensor_only(self, key: str, value: Any) -> None:  # noqa: D417
    # ``is_compiling()`` stays true in every thread while any thread compiles,
    # so code that runs eagerly (such as unpickling in a DataLoader's
    # pin-memory thread) must not take the branch meant for Dynamo tracing.
    if not is_dynamo_compiling():
        __dict__ = self.__dict__
        if (
            "_tensordict" not in __dict__
            or "_non_tensordict" not in __dict__
            or (
                not self._shadow
                and (
                    key in SET_ATTRIBUTES
                    or (
                        key in type(self).__dict__ and key not in self.__expected_keys__
                    )
                )
            )
        ):
            return object.__setattr__(self, key, value)
    else:
        # The generated __init__ of a subclass sets _tensordict through
        # super().__setattr__, which is this method if the parent class is a
        # tensor_only tensorclass. The key is a constant while Dynamo traces,
        # so other writes skip the __dict__ lookup and add no guard.
        if key in SET_ATTRIBUTES or (
            key == "_tensordict" and "_tensordict" not in self.__dict__
        ):
            return object.__setattr__(self, key, value)
        # read below for a None value only
        __dict__ = None
    if key not in self.__expected_keys__:
        raise AttributeError(
            f"Cannot set the attribute {key!r}: {type(self).__name__} has no such "
            f"field. Its fields are {sorted(self.__expected_keys__)}."
            f"{_did_you_mean(key, self.__expected_keys__)}"
        )
    if value is None:
        # A None field is a key missing from the TensorDict, as in set(). A
        # TensorDict is checked with a dict lookup, which is much cheaper than
        # keys().
        td = self._tensordict
        if key in (td._tensordict if type(td) is TensorDict else td.keys()):
            td.del_(key)
        elif type(td) is not TensorDict:
            _del_partial_field(td, key)
        # _set_none_field, inlined
        if __dict__ is None:
            __dict__ = self.__dict__
        if "_init_none_fields" in __dict__:
            __dict__["_init_none_fields"][key] = None
        else:
            non_tensordict = __dict__["_non_tensordict"]
            if non_tensordict and key in non_tensordict:
                del non_tensordict[key]
        return
    # _convert_mapping_for_field, inlined
    if key in type(self)._tensordict_fields and isinstance(value, Mapping):
        value = _normalize_nested_mapping(value)
    out = self._set_str(key, value, inplace=False, validated=False, ignore_lock=False)
    if out is not self:
        raise RuntimeError(
            "Cannot set attribute on a locked tensorclass, even if "
            "clone_on_set is set to True. Use my_obj.set(...) instead."
        )


def _write_back_dest(td_ref, tc_ref):
    # Returns tc._tensordict, which the context manager writes back into on exit,
    # as td_ref() did. tc_ref is for _exit, which updates tc._non_tensordict.
    return td_ref()


def _record_tensorclass(tc, result, last_op):
    # ``with tc.view(-1) as f:`` writes f back into tc._tensordict on exit, the
    # object that the last op of f refers to. Also refer to tc.
    name, (args, kwargs, td_ref) = last_op
    if td_ref() is tc._tensordict:
        tc_ref = _strong_ref(tc) if is_compiling() else weakref.ref(tc)
        dest = functools.partial(_write_back_dest, td_ref, tc_ref)
        result._last_op = (name, (args, kwargs, dest))


def _exit(self, exc_type, exc_val, exc_tb):
    queue = getattr(self._tensordict, "_last_op_queue", None)
    last_op = queue[-1] if queue else None
    result = TensorDict.__exit__(self, exc_type, exc_val, exc_tb)
    if exc_type is None and last_op is not None:
        dest = last_op[1][2]
        if type(dest) is functools.partial and dest.func is _write_back_dest:
            tc = dest.args[1]()
            if tc is not None and tc.__dict__["_non_tensordict"]:
                # The write-back keeps the values in tc._non_tensordict (from
                # from_tensordict) of the fields it writes. Drop them, as
                # update() does. The fields in self._non_tensordict are not
                # written, so only the others need a check.
                non_tensordict = tc.__dict__["_non_tensordict"]
                source_non_tensordict = self.__dict__["_non_tensordict"]
                maybe_written = [
                    key for key in non_tensordict if key not in source_non_tensordict
                ]
                if maybe_written:
                    keys = tc.__dict__["_tensordict"].keys()
                    for key in maybe_written:
                        if key in keys:
                            del non_tensordict[key]
    return result


# The torch.return_types that TensorDict.max, min, cummax and cummin return
_TD_RETURN_TYPES = (
    torch.return_types.max,
    torch.return_types.min,
    torch.return_types.cummax,
    torch.return_types.cummin,
)


def _tuple_like(result: tuple, items: list) -> tuple:
    """Returns ``items`` in a plain tuple, or in a tuple of the type of ``result``.

    The ``torch.return_types`` of ``max``, ``min``, ``cummax`` and ``cummin`` keep
    their type. ``torch.load(weights_only=True)`` accepts them, see
    ``tensordict/_reductions.py``. Other tuples stay plain tuples, such as the
    namedtuples of ``to_namedtuple``: their classes are made at call time and
    cannot be pickled.
    """
    if isinstance(result, _TD_RETURN_TYPES):
        return type(result)(items)
    return tuple(items)


# The fallback methods that write a key, with the position and the name of
# its argument. See _warn_non_field_keys.
_FIELD_KEY_ARGUMENT = {
    "create_nested": (0, "key"),
    "rename_key_": (1, "new_key"),
    "set_non_tensor": (0, "key"),
    "setdefault": (0, "key"),
}


def _non_field_keys(self, keys) -> set[str]:
    # The first keys of the nested keys in keys that are not fields of
    # type(self). Dynamo cannot trace warnings.warn, so only eager
    # calls look for them and warn.
    expected_keys = type(self).__expected_keys__
    if is_dynamo_compiling() or expected_keys.issuperset(keys):
        return set()
    return {_unravel_key_to_tuple(key)[0] for key in keys} - expected_keys


def _warn_non_field_keys(self, keys: set[str], method: str) -> None:
    # set() and attribute assignment reject keys that are not fields with
    # AttributeError, and so will these methods once the deprecation ends.
    cls = type(self)
    warn_deprecated(
        f"Writing keys that are not fields ({sorted(keys)}) with "
        f"{cls.__name__}.{method}(), which set() rejects with AttributeError,",
        removal="0.17",
        replacement=f"only its fields ({sorted(cls.__expected_keys__)})",
        stacklevel=3,
    )


class _TensorClassFuture(TensorDictFuture):
    """A :class:`~tensordict.TensorDictFuture` whose :meth:`result` gives the tensorclass.

    ``wrap`` builds the tensorclass from the tensordict once the writes are
    done, and only once.
    """

    def __init__(self, future: TensorDictFuture, wrap: Callable) -> None:
        super().__init__(future.futures, future.resulting_td)
        self._wrap = wrap
        self._wrapped = None

    def result(self):
        if self._wrapped is None:
            self._wrapped = self._wrap(super().result())
        return self._wrapped


def _wrap_td_method(
    funcname, *, copy_non_tensor=False, no_wrap=False, is_property=False
):
    def deliver_result(self, result, kwargs):
        if result is None:
            return
        if isinstance(result, TensorDictBase) and kwargs.get("out") is not result:
            if not is_compiling():
                non_tensordict = super(type(self), self).__getattribute__(
                    "_non_tensordict"
                )
            else:
                non_tensordict = self._non_tensordict
            non_tensordict = dict(non_tensordict)
            if copy_non_tensor and non_tensordict:
                # use tree_map to copy
                non_tensordict = tree_map(_identity, non_tensordict)
            if non_tensordict:
                # self has values in _non_tensordict, which _exit may drop
                last_op = result._last_op
                if last_op is not None:
                    _record_tensorclass(self, result, last_op)
            return self._from_tensordict(result, non_tensordict, safe=False)
        return result

    if not is_property:

        def wrapped_func(self, *args, **kwargs):
            if not is_compiling():
                td = super(type(self), self).__getattribute__("_tensordict")
            else:
                td = self._tensordict
            result = getattr(td, funcname)(*args, **kwargs)
            if no_wrap:
                return result

            if isinstance(result, TensorDictFuture):
                # e.g. consolidate(return_early=True): result() gives the
                # class, built after the writes so that __post_init__ sees
                # the data
                return _TensorClassFuture(
                    result,
                    lambda out: (
                        self if out is td else deliver_result(self, out, kwargs)
                    ),
                )

            if result is td:
                return self

            if isinstance(result, tuple):
                return _tuple_like(
                    result, [deliver_result(self, r, kwargs) for r in result]
                )
            return deliver_result(self, result, kwargs)

        if funcname in _CAST_METHODS:
            uncast_func = wrapped_func

            def wrapped_func(self, *args, **kwargs):
                # The class default, unless the call passes cast_filter
                if "cast_filter" not in kwargs:
                    cast_filter = getattr(type(self), "_cast_filter", None)
                    if cast_filter is not None:
                        kwargs["cast_filter"] = cast_filter
                return uncast_func(self, *args, **kwargs)

            return wrapped_func
        if funcname not in _FIELD_KEY_ARGUMENT:
            return wrapped_func
        position, name = _FIELD_KEY_ARGUMENT[funcname]
        unchecked_func = wrapped_func

        def wrapped_func(self, *args, **kwargs):
            result = unchecked_func(self, *args, **kwargs)
            # The call succeeded, so key is a string or a tuple of strings.
            key = args[position] if len(args) > position else kwargs.get(name)
            if key is not None and key not in type(self).__expected_keys__:
                unknown = _non_field_keys(self, (key,))
                if unknown:
                    _warn_non_field_keys(self, unknown, funcname)
            return result

        return wrapped_func

    def wrapped_func(self):
        if not is_compiling():
            td = super(type(self), self).__getattribute__("_tensordict")
        else:
            td = self._tensordict
        result = getattr(td, funcname)

        if no_wrap:
            return result

        if result is td:
            return self

        if isinstance(result, tuple):
            return _tuple_like(result, [deliver_result(self, r, {}) for r in result])
        return deliver_result(self, result, {})

    def wrapped_func_setter(self, value):
        if not is_compiling():
            td = super(type(self), self).__getattribute__("_tensordict")
        else:
            td = self._tensordict
        return setattr(td, funcname, value)

    return property(wrapped_func, wrapped_func_setter)


def _wrap_method(self, attr, func, nowarn=False):
    if not nowarn:
        warnings.warn(
            f"The method {func} wasn't explicitly implemented for tensorclass. "
            f"This fallback will be deprecated in future releases because it is inefficient "
            f"and non-compilable. Please raise an issue in tensordict repo to support this method!"
        )

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        args = tuple(_arg_to_tensordict(arg) for arg in args)
        kwargs = {key: _arg_to_tensordict(value) for key, value in kwargs.items()}
        res = func(*args, **kwargs)
        if isinstance(res, TensorDictBase):
            if attr.endswith("_"):
                # in-place operation, return the current object
                return self
            elif attr in _CLEAR_METADATA:
                # this is an attribute where copying the metadata makes no sense, e.g.
                # .all or .any, so we replace all values with None
                return type(self)._from_tensordict(
                    res, {k: None for k in self._non_tensordict}
                )
            # create a new tensorclass from res and copy the metadata from self
            return type(self)._from_tensordict(res, dict(self._non_tensordict))
        return res

    if not is_compiling():
        wrapped_func = functools.wraps(func)(wrapped_func)

    return wrapped_func


def _dict_none_keys(self, input_dict: dict) -> list[str]:
    # The fields of an instance that a dict passed to update() sets
    # to None. from_dict() leaves them out of its result: a None field is a
    # missing key.
    if type(self)._is_non_tensor:
        return []
    return [
        key
        for key, value in input_dict.items()
        if value is None and key in type(self).__expected_keys__
    ]


def _add_none_entries(tc, keys: list[str]) -> None:
    # Write the None values left out by from_dict() as NonTensorData(None)
    # entries, which the update methods of the TensorDict check and write as
    # they did before None fields became missing keys.
    for key in keys:
        tc._tensordict.set_non_tensor(key, None)


def _clear_none_entries(self, keys: list[str], keys_to_update=None) -> None:
    # Remove the NonTensorData(None) entries that _add_none_entries() made
    # update() write: a None field is a missing key.
    if keys_to_update is not None:
        first_keys = {
            key if isinstance(key, str) else _unravel_key_to_tuple(key)[0]
            for key in keys_to_update
        }
        keys = [key for key in keys if key in first_keys]
    for key in keys:
        self.del_(key)
        # in a custom __init__, record that the field was set
        _set_none_field(self, key)


def _update_non_field_keys(self, keys, keys_to_update) -> set[str]:
    # The keys in keys that are not fields and that update() writes.
    unknown = _non_field_keys(self, keys)
    if unknown and keys_to_update is not None:
        unknown &= {_unravel_key_to_tuple(key)[0] for key in keys_to_update}
    return unknown


def _update_scalar_fields(self, input_dict: dict, keys_to_update) -> dict:
    """Sets the numbers that ``input_dict`` gives to ``int``, ``float`` or ``bool`` fields as ``setattr`` does.

    With a non-empty batch size, ``setattr`` keeps such a number as
    ``NonTensorData`` (#1199), where the dict path of ``update`` would make it
    a 0-d tensor that does not fit the batch size. Returns the other entries.
    """
    scalar_fields = type(self)._scalar_fields
    selected = (
        None
        if keys_to_update is None
        else {key for key in keys_to_update if len(_unravel_key_to_tuple(key)) == 1}
    )
    rest = {}
    for key, value in input_dict.items():
        scalar_types = scalar_fields.get(key)
        if (
            scalar_types is not None
            and isinstance(value, scalar_types)
            and (selected is None or key in selected or (key,) in selected)
        ):
            _set(self, key, value)
        else:
            rest[key] = value
    return rest


def _update(
    self,
    input_dict_or_td: dict[str, CompatibleType] | T | None = None,
    clone: bool = False,
    inplace: bool = False,
    *,
    keys_to_update: Sequence[NestedKey] | None = None,
    non_blocking: bool = False,
    update_batch_size: bool = False,
    ignore_lock: bool = False,
    is_leaf: Callable[[Type], bool] | None = None,
    **kwargs,
):
    if ignore_lock and ignore_lock is not _UPDATE_IGNORE_LOCK_WARNED:
        ignore_lock = _warn_update_ignore_lock()
    if kwargs:
        if input_dict_or_td is None:
            input_dict_or_td = kwargs
        elif isinstance(input_dict_or_td, dict):
            input_dict_or_td = {**input_dict_or_td, **kwargs}
        else:
            _update(
                self,
                input_dict_or_td,
                clone=clone,
                inplace=inplace,
                keys_to_update=keys_to_update,
                non_blocking=non_blocking,
                update_batch_size=update_batch_size,
                ignore_lock=ignore_lock,
                is_leaf=is_leaf,
            )
            input_dict_or_td = kwargs
    elif input_dict_or_td is None:
        return self
    if is_leaf is None:
        is_leaf = _is_leaf_nontensor
    if isinstance(input_dict_or_td, dict):
        if (
            type(self)._scalar_fields
            and not inplace
            and not update_batch_size
            and not ignore_lock
            and not type(self)._nocast
            and not self._is_non_tensor
            and self._tensordict.batch_dims
            and not self._tensordict.is_locked
        ):
            # As setattr does (#1199): these calls raised before
            input_dict_or_td = _update_scalar_fields(
                self, input_dict_or_td, keys_to_update
            )
            if not input_dict_or_td:
                return self
        none_keys = _dict_none_keys(self, input_dict_or_td)
        input_dict_or_td = _from_dict_for_update(
            type(self), input_dict_or_td, auto_batch_size=False
        )
        if none_keys:
            _add_none_entries(input_dict_or_td, none_keys)
            _update(
                self,
                input_dict_or_td,
                clone=clone,
                inplace=inplace,
                keys_to_update=keys_to_update,
                non_blocking=non_blocking,
                update_batch_size=update_batch_size,
                ignore_lock=ignore_lock,
                is_leaf=is_leaf,
            )
            _clear_none_entries(self, none_keys, keys_to_update)
            return self

    if is_tensorclass(input_dict_or_td):
        non_tensordict = {
            k: v
            for k, v in input_dict_or_td.__dict__["_non_tensordict"].items()
            if v is not None
        }
        self._tensordict.update(
            input_dict_or_td.__dict__["_tensordict"],
            clone=clone,
            inplace=inplace,
            keys_to_update=keys_to_update,
            non_blocking=non_blocking,
            update_batch_size=update_batch_size,
            ignore_lock=ignore_lock,
            is_leaf=is_leaf,
        )
        # Drop the _non_tensordict values of the fields just written, as the
        # TensorDict branch below does. The fields in the source's
        # _non_tensordict are not in its TensorDict, so only the other keys
        # need a check.
        self_non_tensordict = self._non_tensordict
        if self_non_tensordict:
            source_non_tensordict = input_dict_or_td.__dict__["_non_tensordict"]
            maybe_written = [
                key for key in self_non_tensordict if key not in source_non_tensordict
            ]
            if maybe_written:
                keys = self._tensordict.keys()
                for key in maybe_written:
                    if key in keys:
                        del self_non_tensordict[key]
        self._non_tensordict.update(non_tensordict)
        # The keys of the dict of a TensorDict are faster to read than its
        # keys(), and valid calls only pay for the issuperset checks.
        source = input_dict_or_td.__dict__["_tensordict"]
        source_keys = (
            source._tensordict if type(source) is TensorDict else source.keys()
        )
        expected_keys = type(self).__expected_keys__
        if not (
            expected_keys.issuperset(source_keys)
            and expected_keys.issuperset(non_tensordict)
        ):
            unknown = _update_non_field_keys(
                self, source_keys, keys_to_update
            ) | _non_field_keys(self, non_tensordict)
            if unknown:
                _warn_non_field_keys(self, unknown, "update")
        return self

    self._tensordict.update(
        input_dict_or_td,
        clone=clone,
        inplace=inplace,
        keys_to_update=keys_to_update,
        non_blocking=non_blocking,
        update_batch_size=update_batch_size,
        ignore_lock=ignore_lock,
        is_leaf=is_leaf,
    )
    # We also need to remove things from non_tensordict
    if self._non_tensordict:
        keys = set(self._tensordict.keys())
        ntd = {k: val for k, val in self._non_tensordict.items() if k not in keys}
        self._non_tensordict.clear()
        self._non_tensordict.update(ntd)
    source_keys = (
        input_dict_or_td._tensordict
        if type(input_dict_or_td) is TensorDict
        else input_dict_or_td.keys()
    )
    if not type(self).__expected_keys__.issuperset(source_keys):
        unknown = _update_non_field_keys(self, source_keys, keys_to_update)
        if unknown:
            _warn_non_field_keys(self, unknown, "update")
    return self


def _update_ignores_keys_to_update(self, source, keys_to_update) -> bool:
    # Whether update_() with the tensorclass source writes other entries than
    # it would with keys_to_update, as for a TensorDict source: an entry of
    # self that keys_to_update leaves out, or none for a selected entry that
    # self does not have, which raises a KeyError.
    keys_to_update = [_unravel_key_to_tuple(key) for key in keys_to_update]

    def selected(key):
        return any(key[: len(prefix)] == prefix for prefix in keys_to_update)

    dest_keys = {
        _unravel_key_to_tuple(key)
        for key in self._tensordict.keys(True, True, is_leaf=_is_leaf_nontensor)
    }
    for key in source._tensordict.keys(True, True, is_leaf=_is_leaf_nontensor):
        key = _unravel_key_to_tuple(key)
        if selected(key) != (key in dest_keys):
            return True
    return any(
        value is not None and not selected((key,))
        for key, value in source._non_tensordict.items()
    )


def _update_(
    self,
    input_dict_or_td: dict[str, CompatibleType] | T | None = None,
    clone: bool = False,
    inplace: bool = False,
    *,
    keys_to_update: Sequence[NestedKey] | None = None,
    non_blocking: bool = False,
    **kwargs,
):
    """Copies the entries of a dictionary, a tensordict or a tensorclass into the existing entries.

    Returns ``self``. Works as :meth:`~tensordict.TensorDictBase.update_`, which describes the
    arguments: the existing tensors keep their storage. The fields can also
    be passed as keyword arguments, ``tc.update_(x=value)``. ``inplace`` has
    no effect.

    A tensorclass or dict source ignores ``clone``, ``non_blocking`` and
    ``keys_to_update``: the entries that ``keys_to_update`` leaves out are
    written too, and the call warns. Starting with TensorDict 0.17, the default
    will change to writing only the entries in ``keys_to_update``, as for a
    TensorDict source.

    Examples:
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Foo(TensorClass):
        ...     x: torch.Tensor
        >>> foo = Foo(x=torch.zeros(3), batch_size=[3])
        >>> x = foo.x
        >>> _ = foo.update_({"x": torch.ones(3)})
        >>> x
        tensor([1., 1., 1.])
    """
    if kwargs:
        if input_dict_or_td is None:
            input_dict_or_td = kwargs
        elif isinstance(input_dict_or_td, dict):
            input_dict_or_td = {**input_dict_or_td, **kwargs}
        else:
            _update_(
                self,
                input_dict_or_td,
                clone=clone,
                inplace=inplace,
                keys_to_update=keys_to_update,
                non_blocking=non_blocking,
            )
            input_dict_or_td = kwargs
    elif input_dict_or_td is None:
        return self
    if isinstance(input_dict_or_td, dict):
        none_keys = _dict_none_keys(self, input_dict_or_td)
        input_dict_or_td = _from_dict_for_update(
            type(self), input_dict_or_td, batch_size=self._tensordict.batch_size
        )
        # update_() raises for them, as it did before
        _add_none_entries(input_dict_or_td, none_keys)

    if is_tensorclass(input_dict_or_td):
        non_tensordict = {
            k: v for k, v in input_dict_or_td._non_tensordict.items() if v is not None
        }
        if (
            keys_to_update is not None
            and not is_dynamo_compiling()
            and _update_ignores_keys_to_update(self, input_dict_or_td, keys_to_update)
        ):
            warnings.warn(
                f"{type(self).__name__}.update_() with a tensorclass or dict "
                "source ignores keys_to_update, and writes the entries that "
                "it leaves out. Starting with TensorDict 0.17, the default will "
                "change to writing only the entries in keys_to_update, as for "
                "a TensorDict source, which raises a KeyError for a selected "
                "entry that the tensorclass does not have.",
                FutureWarning,
                stacklevel=2,
            )
        self._tensordict.update_(input_dict_or_td._tensordict)
        self._non_tensordict.update(non_tensordict)
        return self

    self._tensordict.update_(
        input_dict_or_td,
        clone=clone,
        keys_to_update=keys_to_update,
        non_blocking=non_blocking,
    )
    return self


@_update_at_aliases
def _update_at_(
    self,
    input_dict_or_td: dict[str, CompatibleType] | T,
    idx: IndexType,
    clone: bool = False,
    *,
    keys_to_update: Sequence[NestedKey] | None = None,
    non_blocking: bool = False,
):
    """Writes the entries of a dictionary, a tensordict or a tensorclass in place at an index, and returns ``self``.

    Works as :meth:`~tensordict.TensorDictBase.update_at_`, which describes
    the arguments, with one difference. A tensorclass or dict source that
    :meth:`~tensordict.TensorDictBase.update` accepts, such as one with the
    batch size of the tensorclass, goes to
    :meth:`~tensordict.TensorDictBase.update` instead: its entries replace
    those of the tensorclass whatever the index, and the call warns. Starting
    with TensorDict 0.17, the default will change to writing such a source in
    place at the index, as :meth:`~tensordict.TensorDictBase.update_at_`
    does.
    """
    # A source that update() accepts still goes to update(), which replaces the
    # entries whatever the index, as writing it at the index could raise.
    # Sources that update() rejects for their batch size are written at the index.
    none_keys = ()
    if isinstance(input_dict_or_td, dict):
        try:
            none_keys = _dict_none_keys(self, input_dict_or_td)
            input_dict_or_td = _from_dict_for_update(
                type(self), input_dict_or_td, batch_size=self._tensordict.batch_size
            )
            _add_none_entries(input_dict_or_td, none_keys)
        except RuntimeError:
            # write the dict at the index only if its values do not have the
            # batch size of self
            batch_size = self._tensordict.batch_size
            if not any(
                isinstance(value, torch.Tensor)
                and value.shape[: len(batch_size)] != batch_size
                for value in input_dict_or_td.values()
            ):
                raise

    if is_tensorclass(input_dict_or_td):
        non_tensordict = {
            k: v for k, v in input_dict_or_td._non_tensordict.items() if v is not None
        }
        source = input_dict_or_td._tensordict
        dest = self._tensordict
        if (
            source.batch_dims < dest.batch_dims
            and isinstance(dest, TensorDict)
            and not is_non_tensor(input_dict_or_td)
            and any(
                isinstance(value, torch.Tensor)
                and not value.is_nested
                and not _is_unbatched(value)
                and value.shape[: dest.batch_dims] != dest.batch_size
                for value in source.values()
            )
        ):
            # update() would raise for this source, possibly after replacing
            # some entries: write it at the index instead
            input_dict_or_td = source
        else:
            try:
                self._tensordict.update(source)
            except RuntimeError:
                # TensorDict.update checks the batch size before it changes anything
                batch_size = self._tensordict.batch_size
                if (
                    not isinstance(self._tensordict, TensorDict)
                    or is_non_tensor(input_dict_or_td)
                    or batch_size[: source.batch_dims]
                    == source.batch_size[: len(batch_size)]
                ):
                    raise
                input_dict_or_td = source
            else:
                self._non_tensordict.update(non_tensordict)
                _clear_none_entries(self, none_keys)
                # Dynamo cannot trace warnings.warn, so only eager calls warn.
                if not is_dynamo_compiling() and (
                    non_tensordict or not source.is_empty()
                ):
                    warnings.warn(
                        f"{type(self).__name__}.update_at_() with a source that "
                        "update() accepts, such as one with the batch size of the "
                        "tensorclass, replaces the entries of the tensorclass by "
                        "those of the source, whatever the index. Starting with "
                        "TensorDict 0.17, the default will change to writing the "
                        "source in place at the index, as "
                        "TensorDict.update_at_() does: the entries keep their "
                        "storage, and a source that does not have the shape of "
                        "the index raises.",
                        FutureWarning,
                        # the caller of the _update_at_aliases wrapper
                        stacklevel=3,
                    )
                return self

    self._tensordict.update_at_(
        input_dict_or_td,
        idx,
        clone=clone,
        non_blocking=non_blocking,
        keys_to_update=keys_to_update,
    )
    return self


# The TensorDict classmethods whose inputs are instances: the others, such as
# from_any or fromkeys, build an instance from new data.
_REBUILD_CLASSMETHODS = frozenset({"stack", "cat", "lazy_stack", "maybe_dense_stack"})


def _wrap_classmethod(td_cls, cls, func):
    rebuild = func.__name__ in _REBUILD_CLASSMETHODS

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        res = func.__get__(td_cls)(*args, **kwargs)
        # res = func(*args, **kwargs)
        if isinstance(res, TensorDictBase):
            # create a new tensorclass from res and copy the metadata from self
            return cls._from_tensordict(res, rebuild=rebuild)
        return res

    return wrapped_func


class _wrap_td_classmethod:
    """A TensorDict classmethod (e.g. ``stack``, ``from_list``) on a tensorclass.

    Accessed on the class, it calls the TensorDict classmethod and returns an
    instance of that class, as ``_wrap_classmethod`` does. Accessed on an
    instance, it calls the method on the instance's tensordict, as
    ``_wrap_td_method`` does.
    """

    def __init__(self, funcname):
        self.funcname = funcname
        self.instance_method = _wrap_td_method(funcname)

    def __get__(self, instance, owner=None):
        if instance is None:
            func = getattr(TensorDict, self.funcname)
            return _wrap_classmethod(TensorDict, owner, func)
        return self.instance_method.__get__(instance, owner)


def _getitem(self, item: NestedKey) -> Tensor | TensorCollection | Any:
    """Retrieve the class object at the given index. Indexing will happen for nested tensors as well.

    Args:
       item (int or any other valid index type): index of the object to retrieve

    Returns:
        Tensor class object at the given index

    """
    if isinstance(item, str) or (
        isinstance(item, tuple) and all(isinstance(_item, str) for _item in item)
    ):
        raise ValueError(
            f"Invalid indexing arguments: {item!r}. A tensorclass is indexed along "
            f"its batch dimensions; to read a field, use the attribute or "
            f"get({item!r})."
        )
    # tensor_res = super(type(self), self).__getattribute__("_tensordict")[item]
    tensor_res = self.__dict__["_tensordict"][item]
    return _from_tensordict_with_copy(self, tensor_res)  # device=res.device)


def _setitem(self, item: NestedKey, value: Any) -> None:  # noqa: D417
    """Set the value of the Tensor class object at the given index. Note that there is no strict validation on non-tensor values.

    Args:
        item (int or any other valid index type): index of the object to set
        value (any): value to set for the item

    """
    istuple = isinstance(item, tuple)
    if istuple or isinstance(item, str):
        # _unravel_key_to_tuple will return an empty tuple if the index isn't a NestedKey
        idx_unravel = _unravel_key_to_tuple(item)
        if idx_unravel:
            raise ValueError(
                f"Invalid indexing arguments: {item!r}. A tensorclass is indexed "
                f"along its batch dimensions; to write a field, use the attribute "
                f"or set({item!r}, value)."
            )

    if istuple and len(item) == 1:
        return _setitem(self, item[0], value)
    if (
        (
            isinstance(item, torch.Tensor)
            and item.dtype == torch.bool
            and not item.shape
            and item
        )
        or (item is True)
        or (item is None)
    ) and self.batch_size == ():
        # Squeeze a value that has a batch dim. squeeze(0) raises IndexError
        # on batch size [], unlike Tensor.squeeze. A tensor has no batch size
        # and is still squeezed.
        value_batch_size = getattr(value, "batch_size", None)
        if value_batch_size is None or value_batch_size:
            return self.update(value.squeeze(0))
        # A NonTensorData takes another NonTensorData here, which [...]
        # refuses. Any other value without a batch dim is written as by
        # [...], which checks the class of a tensorclass value. [...] does not
        # check the keys of a TensorDict value, so check them here.
        if isinstance(self, NonTensorData) and isinstance(value, NonTensorData):
            return self.update(value)
        if isinstance(value, TensorDictBase):
            extra_keys = set(value.keys()) - set(type(self).__expected_keys__)
            if extra_keys:
                raise ValueError(
                    f"Cannot write keys {sorted(extra_keys)} into a "
                    f"{type(self).__name__}: they are not fields of the class."
                )
        item = Ellipsis
    if (
        isinstance(self, NonTensorData)
        and is_non_tensor(value)
        and _non_tensor_value_differs(self, value)
    ):
        raise ValueError(_NON_TENSOR_ONE_VALUE_MSG)

    if not is_tensorclass(value) and not isinstance(
        value, (TensorDictBase, numbers.Number, Tensor)
    ):
        if isinstance(self, NonTensorData):
            # A raw value, such as a string written through the list of
            # NonTensorData.tolist(as_linked_list=True).
            raise ValueError(_NON_TENSOR_ONE_VALUE_MSG)
        raise ValueError(
            f"__setitem__ only supports tensorclasses, tensordicts,"
            f" numeric scalars and tensors. Got {type(value)}"
        )

    if is_tensorclass(value):
        if not isinstance(value, type(self)):
            self_keys = set().union(_non_tensor_keys(self), self._tensordict.keys())
            value_keys = set().union(_non_tensor_keys(value), value._tensordict.keys())
            if self_keys != value_keys:
                # if tensorclass but different class ensure that all keys are equal
                raise ValueError(
                    "__setitem__ is only allowed for same-class or "
                    "compatible class (i.e. same members) assignment"
                )

        # Validating the non-tensor data before setting the item
        for key, val in value._non_tensordict.items():
            # Raise a warning if non_tensor data doesn't match
            if (
                key in self._non_tensordict.keys()
                and val is not self._non_tensordict[key]
            ):
                warnings.warn(
                    f"Meta data at {repr(key)} may or may not be equal, "
                    f"this may result in undefined behaviours",
                    category=UserWarning,
                    stacklevel=2,
                )

        for key in value._tensordict.keys():
            # Making sure that the key-clashes won't happen, if the key is present
            # in tensor data in value we will honor that and remove the key-value
            # pair from non-tensor data
            if key in self._non_tensordict.keys():
                del self._non_tensordict[key]

        self._tensordict[item] = value._tensordict
    else:
        if isinstance(value, TensorDictBase):
            # As in update(): the keys of the dict of a TensorDict are faster
            # to read than its keys(), and valid calls only pay for issuperset.
            keys = value._tensordict if type(value) is TensorDict else value.keys()
            if not type(self).__expected_keys__.issuperset(keys):
                unknown = _non_field_keys(self, keys)
                if unknown:
                    _warn_non_field_keys(self, unknown, "__setitem__")
        # int, float etc.
        self._tensordict[item] = value


def _repr_html(self) -> str | None:
    return _html_repr(self)


def _repr_hidden_fields(cls) -> list[str]:
    """Returns the fields of ``cls`` declared with ``field(repr=False)``, which the reprs leave out."""
    return [name for name, field in cls.__dataclass_fields__.items() if not field.repr]


def _repr(self) -> str:
    """Return a string representation of Tensor class object."""
    if _REPR_OPTIONS["compact"]:
        return _compact_repr(self)
    hidden = _repr_hidden_fields(type(self))
    td = self._tensordict
    non_td = self._non_tensordict
    keys = None
    if hidden:
        keys = [key for key in td.keys() if key not in hidden]
        non_td = {key: value for key, value in non_td.items() if key not in hidden}
    fields = _td_fields(td, keys=keys, sep="=")
    field_str = [fields] if fields else []
    non_tensor_fields = _all_non_td_fields_as_str(non_td)
    if not type(self)._is_non_tensor:
        non_tensor_fields += [
            f"{key}=None" for key in _none_fields(self) if key not in hidden
        ]

    metadata_fields = []

    if _REPR_OPTIONS["show_batch_size"]:
        if "batch_size" not in self.__expected_keys__:
            metadata_fields.append(indent(f"batch_size={self.batch_size}", 4 * " "))
        elif "shape" not in self.__expected_keys__:
            metadata_fields.append(indent(f"shape={self.shape}", 4 * " "))
    if _REPR_OPTIONS["show_device"] and "device" not in self.__expected_keys__:
        metadata_fields.append(indent(f"device={self.device}", 4 * " "))
    if _REPR_OPTIONS["show_is_shared"]:
        metadata_fields.append(indent(f"is_shared={self.is_shared()}", 4 * " "))

    if len(non_tensor_fields) > 0:
        non_tensor_field_str = indent(
            ",\n".join(non_tensor_fields),
            4 * " ",
        )
        if field_str:
            string = ",\n".join(field_str + [non_tensor_field_str, *metadata_fields])
        else:
            string = ",\n".join([non_tensor_field_str, *metadata_fields])
    elif field_str:
        string = ",\n".join(field_str + metadata_fields)
    elif len(metadata_fields) > 0:
        string = ",\n".join(metadata_fields)
    else:
        string = ""
    return f"{type(self).__name__}({string})"


def _len(self) -> int:
    """Returns the length of first dimension, if there is, otherwise 0."""
    return len(self._tensordict)


def _to_dict(self, *, retain_none: bool = True, convert_tensors: bool = False) -> dict:
    td_dict = self._tensordict.to_dict(
        retain_none=retain_none, convert_tensors=convert_tensors
    )
    if retain_none and not type(self)._is_non_tensor:
        td_dict.update(dict.fromkeys(_none_fields(self)))
    if self._non_tensordict:
        if retain_none:
            td_dict.update(self._non_tensordict)
        else:
            td_dict.update(
                {k: v for k, v in self._non_tensordict.items() if v is not None}
            )

    return td_dict


def _from_dict(
    cls,
    input_dict,
    *,
    auto_batch_size: bool | None = None,
    batch_size=None,
    device=None,
    batch_dims=None,
):
    # we pass through a tensordict because keys could be passed as NestedKeys
    # We can't assume all keys are strings, otherwise calling cls(**kwargs)
    # would work ok
    if issubclass(cls, NonTensorDataBase):
        # Note: this won't deal with sub-tensordicts which may or may not be tensorclasses.
        # We don't want to enforce them to be tensorclasses so we can't do much about it...
        return cls.from_tensordict(
            tensordict=TensorDict(
                batch_size=batch_size,
                device=device,
            ),
            non_tensordict=input_dict,
        )
    if not cls._is_non_tensor:
        # A None field is a missing key, not a NonTensorData(None) entry.
        input_dict = {
            key: value
            for key, value in input_dict.items()
            if value is not None or key not in cls.__expected_keys__
        }
    td = TensorDict.from_dict(
        input_dict,
        batch_size=batch_size,
        device=device,
        batch_dims=batch_dims,
        auto_batch_size=auto_batch_size,
    )
    non_tensordict = {}

    return cls.from_tensordict(tensordict=td, non_tensordict=non_tensordict)


def _from_dict_public(
    cls,
    input_dict,
    *,
    auto_batch_size: bool | None = None,
    batch_size=None,
    device=None,
    batch_dims=None,
):
    """Builds an instance of the tensorclass from a dictionary of its fields.

    The fields that ``input_dict`` misses are set to ``None``, also those that
    declare a default or a default factory. Starting with TensorDict 0.17, the
    default will change to filling these fields with their defaults, as the
    constructor does. Until then, a call that misses such a field warns with a
    :class:`FutureWarning`.

    Args:
        input_dict (dict): the values of the fields. Nested dictionaries become
            tensordicts, or tensorclasses for the fields annotated with one.

    Keyword Args:
        auto_batch_size (bool, optional): if ``True``, sets the batch size to
            the largest leading shape that all the entries share. Defaults to
            ``None``, which behaves like ``False``, unless ``batch_dims`` is
            passed: then it behaves like ``True``.
        batch_size (torch.Size or list of int, optional): the batch size of
            the result: the leading dimensions that all its entries share.
            Defaults to ``None``, which gives an empty batch size unless
            ``auto_batch_size`` is on.
        device (torch.device, str or int, optional): the device of the
            result. Leaves on another device are moved to it. Defaults to
            ``None``, which leaves each leaf where it is.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Exclusive with ``batch_size``.
            Defaults to ``None``.

    Returns:
        A new instance of the class.

    Examples:
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Foo(TensorClass):
        ...     x: torch.Tensor
        ...     y: torch.Tensor
        >>> foo = Foo.from_dict({"x": torch.zeros(3, 4), "y": torch.zeros(3)}, auto_batch_size=True)
        >>> foo.batch_size
        torch.Size([3])
    """
    keys = {key[0] if isinstance(key, tuple) else key for key in input_dict}
    missing = [name for name in cls._from_dict_defaults if name not in keys]
    # Dynamo cannot trace warnings.warn: a compiled call stays silent.
    if missing and not is_compiling():
        warnings.warn(
            f"{cls.__name__}.from_dict() sets the fields {missing} that the input "
            "misses to None, not to their declared defaults. Starting with "
            "TensorDict 0.17, the default will change to filling these fields "
            "with their defaults and default factories, as the constructor does. "
            "Pass the fields to from_dict() to choose their values.",
            FutureWarning,
            stacklevel=2,
        )
    return _from_dict(
        cls,
        input_dict,
        auto_batch_size=auto_batch_size,
        batch_size=batch_size,
        device=device,
        batch_dims=batch_dims,
    )


def _from_dict_for_update(cls, input_dict, **kwargs):
    # The update methods keep the fields that input_dict misses, so the
    # defaults that from_dict() fills in 0.17 do not concern them: skip its
    # warning, unless the class defines its own from_dict(). Dynamo traces
    # this attribute read, not an identity check on cls.from_dict.__func__.
    if getattr(cls.from_dict, "_tc_default_from_dict", False):
        return _from_dict(cls, input_dict, **kwargs)
    return cls.from_dict(input_dict, **kwargs)


_from_dict_public._tc_default_from_dict = True


def _from_dict_instance(
    self,
    input_dict,
    *,
    auto_batch_size: bool | None = None,
    batch_size=None,
    device=None,
    batch_dims=None,
):
    if batch_dims is not None and batch_size is not None:
        raise ValueError("Cannot pass both batch_size and batch_dims to `from_dict`.")
    from tensordict import TensorDict

    batch_size_set = torch.Size(()) if batch_size is None else batch_size
    # TODO: this is a bit slow and will be a bottleneck every time td[idx] = dict(subtd)
    # is called when there are non tensor data in it
    if not _is_tensor_collection(type(input_dict)):
        input_tdict = TensorDict.from_dict(input_dict, auto_batch_size=auto_batch_size)
    else:
        input_tdict = input_dict
    trsf_dict = {}
    for key, value in list(input_tdict.items()):
        # cur_value = getattr(self, key, None)
        cur_value = self.get(key)
        if _is_tensor_collection(type(cur_value)):
            trsf_dict[key] = cur_value.from_dict_instance(
                value, batch_size=[], device=device, batch_dims=None
            )
        elif not isinstance(cur_value, torch.Tensor) and is_non_tensor(value):
            trsf_dict[key] = value.data
        elif cur_value is not None and not isinstance(cur_value, torch.Tensor):
            # This is slightly unsafe but will work with bool, float and int
            try:
                trsf_dict[key] = type(cur_value)(value)
            except Exception:
                trsf_dict[key] = input_dict[key]
        else:
            trsf_dict[key] = value
    out = type(self)(
        **trsf_dict,
        batch_size=batch_size_set,
        device=device,
    )
    # check that
    if batch_size is None:
        if auto_batch_size is None and batch_dims is None:
            auto_batch_size = False
        elif auto_batch_size is None:
            auto_batch_size = True
        if auto_batch_size:
            out.auto_batch_size_()
    return out


def _to_tensordict(self, *, retain_none: bool | None = None) -> TensorDict:
    """Returns a :class:`~tensordict.TensorDict` with a copy of the fields of the tensorclass.

    The tensors are copied, and memory-mapped or shared tensors become regular
    tensors. Nested tensorclasses become tensordicts, and non-tensor fields
    become :class:`~tensordict.NonTensorData` entries.

    Keyword Args:
        retain_none (bool, optional): if ``True``, fields set to ``None`` are
            written as ``None`` entries. Defaults to ``None``, which drops
            them.

    Examples:
        >>> from typing import Any
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Foo(TensorClass):
        ...     x: torch.Tensor
        ...     y: Any = None
        >>> foo = Foo(x=torch.zeros(3))
        >>> sorted(foo.to_tensordict().keys())
        ['x']
        >>> sorted(foo.to_tensordict(retain_none=True).keys())
        ['x', 'y']

    """
    td = self._tensordict.to_tensordict(retain_none=retain_none)
    if retain_none and not type(self)._is_non_tensor:
        for key in _none_fields(self):
            td.set_non_tensor(key, None)
    for key, val in self._non_tensordict.items():
        if val is None:
            if retain_none is None:
                retain_none = False
            if retain_none:
                pass
            else:
                continue
        td.set_non_tensor(key, val)
    return td


def _tc_select(
    self, *keys, inplace: bool = False, strict: bool = True, as_tensordict: bool = False
):
    """Returns a tensorclass that holds only the fields of ``keys``, with the other fields set to ``None``.

    Works as :meth:`~tensordict.TensorDictBase.select`, which describes
    ``keys``: the leaves are not copied.

    Keyword Args:
        inplace (bool, optional): if ``True``, sets the other fields of the
            tensorclass to ``None`` and returns ``self``. Defaults to
            ``False``.
        strict (bool, optional): if ``True``, a missing key raises a
            ``KeyError``. If ``False``, missing keys are ignored. Defaults to
            ``True``.
        as_tensordict (bool, optional): if ``True``, returns the selected
            entries in a :class:`~tensordict.TensorDict` instead of a
            tensorclass. Defaults to ``False``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Foo(TensorClass):
        ...     x: torch.Tensor
        ...     y: torch.Tensor
        >>> foo = Foo(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
        >>> print(foo.select("x").y)
        None
        >>> type(foo.select("x", as_tensordict=True)).__name__
        'TensorDict'
    """
    td = self._tensordict
    result = td.select(*keys, inplace=inplace, strict=strict)
    if as_tensordict:
        return result
    if result is td:
        return self
    non_tensordict = dict(self._non_tensordict)
    return type(self)._from_tensordict(result, non_tensordict, safe=False)


def _tc_rename_keys(self, mapping, *, strict: bool = True):
    """Returns a tensorclass with several keys renamed at once.

    Works as :meth:`~tensordict.TensorDictBase.rename_keys`, which describes
    ``mapping`` and ``strict``, with two differences: each destination must be
    a field of the class, or the call raises an ``AttributeError``, and a
    field that the renaming empties is kept.

    Examples:
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Foo(TensorClass):
        ...     x: torch.Tensor
        ...     y: torch.Tensor
        >>> foo = Foo(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
        >>> foo.rename_keys({"x": "y", "y": "x"}).x
        tensor([1., 1., 1.])
    """
    pairs = _rename_keys_pairs(mapping)
    cls = type(self)
    dst_roots = {dst[0] for _, dst in pairs}
    unknown = dst_roots - cls.__expected_keys__
    if unknown:
        raise AttributeError(
            f"The destinations {sorted(unknown)} are not fields of {cls.__name__} "
            f"({sorted(cls.__expected_keys__)})."
        )
    result = self._tensordict._rename_keys(pairs, strict=strict, prune_root=False)
    non_tensordict = {
        key: value
        for key, value in self._non_tensordict.items()
        if key not in dst_roots
    }
    return cls._from_tensordict(result, non_tensordict, safe=False)


def _device(self) -> torch.device:
    """The device of the tensorclass, or ``None`` if its leaves may be on different devices.

    It is read-only: use :meth:`~tensordict.TensorDictBase.to` to move the
    tensorclass to another device.
    """
    return self._tensordict.device


def _device_setter(self, value: DeviceType) -> None:
    raise RuntimeError(
        "device cannot be set using tensorclass.device = device, "
        "because device cannot be updated in-place. To update device, use "
        "tensorclass.to(new_device), which will return a new tensorclass "
        "on the new device."
    )


def _set(
    self, key: NestedKey, value: Any, inplace: bool = False, non_blocking: bool = False
):
    """Sets a field, and returns the tensorclass.

    ``tc.set("x", value)`` is the same as ``tc.x = value``: the value is cast
    according to the class flags (see :class:`~tensordict.TensorClass`).
    Otherwise, it works as :meth:`~tensordict.TensorDictBase.set`.

    Args:
        key (NestedKey): the field to set. A tuple sets an entry of a nested
            field: ``tc.set(("a", "b"), value)`` is ``tc.a.set("b", value)``.
        value (Any): the value of the field.
        inplace (bool, optional): if ``True`` and the field holds a tensor,
            the value is copied into it, which keeps its storage. Defaults to
            ``False``. See :ref:`inplace-conventions`.
        non_blocking (bool, optional): if ``True``, copies between devices
            can run asynchronously with respect to the host. Defaults to
            ``False``.

    Returns:
        ``self``.

    Raises:
        AttributeError: if ``key`` is not a field of the class.
        RuntimeError: if the tensorclass is locked, also with
            ``inplace=True``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Foo(TensorClass):
        ...     x: torch.Tensor
        >>> foo = Foo(x=torch.zeros(3), batch_size=[3])
        >>> foo.set("x", torch.ones(3)).x
        tensor([1., 1., 1.])
        >>> foo.set("z", torch.ones(3))
        Traceback (most recent call last):
            ...
        AttributeError: Cannot set the attribute 'z': Foo has no such field. Its fields are ['x'].
    """
    if isinstance(key, str):
        cls = type(self)
        __dict__ = self.__dict__
        if __dict__["_tensordict"].is_locked:
            raise RuntimeError(_LOCK_ERROR)
        # if key in ("batch_size", "names", "device"):
        #     # handled by setattr
        #     return
        expected_keys = cls.__expected_keys__
        if key not in expected_keys:
            raise AttributeError(
                f"Cannot set the attribute {key!r}: {cls.__name__} has no such "
                f"field. Its fields are {sorted(expected_keys)}."
                f"{_did_you_mean(key, expected_keys)}"
            )

        self_is_non_tensor = self._is_non_tensor
        value_type = type(value)

        def set_tensor(
            key=key,
            value=value,
            inplace=inplace,
            non_blocking=non_blocking,
            non_tensor=False,
        ):
            if self_is_non_tensor:
                while is_non_tensor(value):
                    value = value.data
                self._non_tensordict[key] = value
                return self
            if non_tensor:
                # Read the metadata from the TensorDict: with shadow=True,
                # self.batch_size and self.device can be fields.
                td = self._tensordict
                value = NonTensorData(
                    data=value, batch_size=td.batch_size, device=td.device
                )
            if key in self._non_tensordict:
                del self._non_tensordict[key]
            # Avoiding key clash, honoring the user input to assign tensor type data to the key
            self._tensordict.set(key, value, inplace=inplace, non_blocking=non_blocking)
            return self

        def _is_castable(datatype):
            return issubclass(datatype, (int, float, np.ndarray))

        if cls._tensor_only:
            value, is_mapping_field = _convert_mapping_for_field(
                key, value, cls._tensordict_fields
            )
            if is_mapping_field:
                return set_tensor(value=value)

        if cls._autocast:
            type_hints = cls._type_hints
            if type_hints is not None:
                target_cls = type_hints.get(key, _AnyType)
            else:
                warnings.warn("type_hints are none, cannot perform auto-casting")
                target_cls = _AnyType

            if isinstance(value, dict):
                if _is_tensor_collection(target_cls):
                    cast_val = target_cls.from_dict(value, auto_batch_size=False)
                    self._tensordict.set(
                        key, cast_val, inplace=inplace, non_blocking=non_blocking
                    )
                    return self
                elif type_hints is None:
                    warnings.warn(type(self)._set_dict_warn_msg)
            elif (
                isinstance(value, TensorDictBase)
                and _tensorclass_annotation(target_cls) is not None
            ):
                # A tensordict with keys that the class does not declare is kept.
                if _fits_tensorclass(target_cls, value):
                    value = target_cls.from_tensordict(value)
                return set_tensor(value=value)
            elif value is not None and _is_accepted_class(target_cls):
                try:
                    if not issubclass(value_type, target_cls):
                        if issubclass(target_cls, torch.Tensor):
                            # first convert to tensor to make sure that the dtype is preserved
                            value = torch.as_tensor(value)
                        cast_val = _cast_func(target_cls)(value)
                    else:
                        cast_val = value
                except TypeError:
                    raise TypeError(
                        f"Failed to cast the value {key} to the type annotation {target_cls}."
                    )
                return set_tensor(value=cast_val)
            elif value is not None and target_cls is not _AnyType:
                cast_val = _cast_func(target_cls)(value)
                return set_tensor(value=cast_val, non_tensor=True)
            elif target_cls is _AnyType and _is_castable(value_type):
                return set_tensor()
            non_tensor = not (
                _is_accepted_class(value_type) or _is_tensor_collection(value_type)
            )
        elif issubclass(value_type, torch.Tensor) or _is_tensor_collection(value_type):
            return set_tensor()
        elif not cls._nocast and issubclass(value_type, (int, float, bool, np.ndarray)):
            scalar_types = cls._scalar_fields.get(key)
            if (
                scalar_types is not None
                and not self_is_non_tensor
                and isinstance(value, scalar_types)
            ):
                if self._tensordict.batch_dims:
                    # A 0-d tensor would not fit the batch size and raise:
                    # keep the Python value, as all batch sizes will from 0.17.
                    # A write in place still raises the batch-size mismatch.
                    if not inplace:
                        return set_tensor(non_tensor=True)
                elif not is_dynamo_compiling():
                    dest = self._tensordict._get_str(key, None)
                    # over a NonTensorData entry, TensorDict.set warns (#2432)
                    if inplace or not (is_non_tensor(dest) and is_tensorclass(dest)):
                        _warn_scalar_field(cls, key, value)
            return set_tensor()
        elif issubclass(value_type, list) and list_to_stack():
            # set() will take care of casting to non tensor
            non_tensor = False
        else:
            non_tensor = True

        if self_is_non_tensor or value is None:
            # Avoiding key clash, honoring the user input to assign non-tensor data to the key
            if not self_is_non_tensor and key in self._tensordict.keys():
                if inplace:
                    raise RuntimeError(
                        f"Cannot update an existing entry of type {type(self._tensordict.get(key))} with a value of type {value_type}."
                    )
                self._tensordict.del_(key)
            elif (
                not self_is_non_tensor
                and not inplace
                and type(self._tensordict) is not TensorDict
            ):
                _del_partial_field(self._tensordict, key)
            if self_is_non_tensor:
                self._non_tensordict[key] = value
            # _set_none_field, inlined: a None field is a missing key
            elif "_init_none_fields" in __dict__:
                __dict__["_init_none_fields"][key] = None
            else:
                non_tensordict = __dict__["_non_tensordict"]
                if non_tensordict and key in non_tensordict:
                    del non_tensordict[key]
        else:
            if inplace:
                if key in self._tensordict.keys():
                    raise RuntimeError(
                        f"Cannot update an existing entry of type {type(self._tensordict.get(key))} with a value of type {value_type}."
                    )
            return set_tensor(value=value, non_tensor=non_tensor)
        return self

    if isinstance(key, tuple) and len(key):
        key = _unravel_key_to_tuple(key)
        if len(key) > 1:
            return self.set(key[0], getattr(self, key[0]).set(key[1:], value))
        out = self.set(key[0], value)
        return out
    raise ValueError(
        f"Supported type for key are str and tuple, got {key} of type {type(key)}"
    )


def _set_str(
    self,
    key: NestedKey,
    value: str,
    *,
    inplace: bool,
    validated: bool,
    ignore_lock: bool = False,
    non_blocking: bool = False,
):
    # is_non_tensor(self), inlined: every tensorclass sets _is_non_tensor
    if type(self)._is_non_tensor is True:
        if key != "data":
            raise KeyError(f"only 'data' keys are supported for {type(self).__name__}.")
        while isinstance(value, (NonTensorData, NonTensorStack)):
            value = value.data
        self._non_tensordict[key] = value
        return self
    else:
        if key in self._non_tensordict:
            del self._non_tensordict[key]
    self._tensordict._set_str(
        key,
        value,
        inplace=inplace,
        validated=validated,
        ignore_lock=ignore_lock,
        non_blocking=non_blocking,
    )
    return self


def _set_at_str(
    self,
    key: NestedKey,
    value: str,
    idx,
    *,
    validated: bool,
    non_blocking: bool = False,
):
    # is_non_tensor(self), inlined: every tensorclass sets _is_non_tensor
    if type(self)._is_non_tensor is True:
        if key != "data":
            raise KeyError(f"only 'data' keys are supported for {type(self).__name__}.")
        while isinstance(value, (NonTensorData, NonTensorStack)):
            value = value.data
        self._non_tensordict[key] = value
        return self
    else:
        if key in self._non_tensordict:
            del self._non_tensordict[key]
    self._tensordict._set_at_str(
        key, value, idx, validated=validated, non_blocking=non_blocking
    )
    return self


def _delattr(self, key):
    td = self._tensordict
    if key in self._non_tensordict and key not in td.keys():
        # A value kept outside the TensorDict, e.g. the data of NonTensorData
        if td.is_locked:
            raise RuntimeError(_LOCK_ERROR)
        del self._non_tensordict[key]
    elif (
        not type(self)._is_non_tensor
        and key in self.__expected_keys__
        and key not in td.keys()
    ):
        # A None field is a missing key: there is nothing to delete. Another
        # key in neither store goes to the TensorDict below, which raises
        # KeyError.
        if td.is_locked:
            raise RuntimeError(_LOCK_ERROR)
        _set_none_field(self, key)
    else:
        del td[key]


def _del_(self, key):
    key = _unravel_key_to_tuple(key)
    if len(key) > 1:
        td = self.get(key[0])
        td.del_(key[1:])
        return self
    if key[0] in self._tensordict.keys():
        self._tensordict.del_(key[0])
        # self.set(key[0], None)
    elif not type(self)._is_non_tensor and key[0] in self.__expected_keys__:
        # A None field is a missing key
        _set_none_field(self, key[0])
    elif key[0] in self._non_tensordict.keys():
        self._non_tensordict[key[0]] = None
    else:
        raise KeyError(f"Key {key} could not be found in tensorclass {self}.")
    return self


def _set_at_(
    self, key: NestedKey, value: Any, idx: IndexType, non_blocking: bool = False
):
    in_non_tensordict = key in self._non_tensordict
    self._tensordict.set_at_(key, value, idx, non_blocking=non_blocking)
    # Drop the non-tensor value only after the write succeeded
    if in_non_tensordict:
        del self._non_tensordict[key]
    return self


def _warn_get_through_tensor(self, key: tuple, entry: Tensor) -> None:
    """Warns that ``get`` returned ``None`` for a key that runs through a tensor, which raises from 0.17."""
    warnings.warn(
        f"{_key_through_leaf_error(key, 1, entry)} {type(self).__name__}.get "
        "returns None for this key. Starting with TensorDict 0.17, the "
        "default will change to raising this ValueError, as TensorDict.get "
        "does. Pass a default, as in get(key, None), to keep getting None.",
        FutureWarning,
        stacklevel=_user_stacklevel(),
    )


def _get(self, key: NestedKey, *args, **kwargs):
    """Returns the value of a field, or ``default`` if the key is missing.

    ``tc.get("x")`` is the same as ``tc.x``: a non-tensor field returns the
    Python object, not a :class:`~tensordict.NonTensorData`. The keyword
    arguments of :meth:`~tensordict.TensorDictBase.get` (``as_list``,
    ``as_padded_tensor``, ...) are passed to the tensordict that holds the
    fields.

    A nested key that runs through a tensor field, such as
    ``tc.get(("x", "y"))`` for a tensor ``x``, returns ``None`` and warns.
    Starting with TensorDict 0.17, the default will change to raising the
    ``ValueError`` that :meth:`~tensordict.TensorDictBase.get` raises. A
    default passed explicitly, as in ``tc.get(("x", "y"), None)``, is
    returned without a warning.

    Args:
        key (NestedKey): the key to read. A tuple reads an entry of a nested
            field: ``tc.get(("a", "b"))`` is ``tc.a.get("b")``.
        default (optional): the value returned if the key is missing.
            Defaults to ``None``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Foo(TensorClass):
        ...     x: torch.Tensor
        ...     name: str
        >>> foo = Foo(x=torch.zeros(3), name="a", batch_size=[3])
        >>> foo.get("name")
        'a'
        >>> foo.get("z", "missing")
        'missing'
    """
    key = _unravel_key_to_tuple(key)
    if not key:
        raise KeyError(_GENERIC_NESTED_ERR.format(key))
    # Find what the default is
    passed_default = True
    if args:
        default = args[0]
        if len(args) > 1:
            raise TypeError("Only one arg is allowed in TD.get.")
        elif "default" in kwargs:
            raise TypeError("'default' arg was passed twice.")
    elif "default" in kwargs:
        default = kwargs.pop("default")
        if args:
            raise TypeError("'default' arg was passed twice.")
    elif _td_base._GET_DEFAULTS_TO_NONE:
        default = None
        passed_default = False
    else:
        default = NO_DEFAULT

    try:
        if len(key) > 1:
            entry = _getattr(self, key[0], **kwargs)
            if not passed_default and _is_tensor_collection(type(entry)):
                # the entry's get returns its own default, and keeps raising
                # for a key that runs through a leaf, as get does without one
                return entry.get(key[1:], **kwargs)
            if not passed_default and isinstance(entry, Tensor):
                # TensorDict.get raises here; a tensorclass will from 0.17.
                # Dynamo cannot trace the warning's sys._getframe
                if not is_dynamo_compiling():
                    _warn_get_through_tensor(self, key, entry)
                return default
            return entry.get(key[1:], default=default, **kwargs)
        if kwargs:
            result = _getattr(self, key[0], **kwargs)
            if (
                (
                    ("as_list" in kwargs and kwargs["as_list"])
                    or ("as_nested_tensor" in kwargs and kwargs["as_nested_tensor"])
                )
                and self._tensordict._get_ignores_kwargs
                and not is_compiling()
            ):
                # warn as a dense tensordict's get does (#2551)
                _warn_get_kwargs(self, kwargs, result, default)
            return result
        return getattr(self, key[0])
    except (AttributeError, KeyError):
        if default is NO_DEFAULT:
            raise
        return default


def _get_at(self, key: NestedKey, *args, **kwargs):
    key = _unravel_key_to_tuple(key)
    if not key:
        raise KeyError(_GENERIC_NESTED_ERR.format(key))

    try:
        if len(args):
            index = args[0]
            args = args[1:]
        else:
            index = kwargs.pop("index")
    except KeyError:
        raise TypeError("index argument missing from get_at")

    # Find what the default is
    if args:
        default = args[0]
        if len(args) > 1:
            raise TypeError("only one (keyword) argument is allowed.")
    elif "default" in kwargs:
        default = kwargs.pop("default")
    elif _td_base._GET_DEFAULTS_TO_NONE:
        default = None
    else:
        default = NO_DEFAULT

    try:
        # the other keywords go on to get, as TensorDictBase.get_at does
        out = self.get(key, NO_DEFAULT, **kwargs)
        # A None field is None at every index, as obj[index].field is
        return None if out is None else out[index]
    except (AttributeError, KeyError):
        if default is NO_DEFAULT:
            raise
        return default


def _data(self):
    # We allow data to be a field of the class too
    if "data" in self.__dataclass_fields__:
        data = self._tensordict.get("data")
        if data is None:
            data = self._non_tensordict.get("data")
        return data
    return self._from_tensordict(self._tensordict.data, self._non_tensordict)


def _data_setter(self, new_data):
    if self._is_non_tensor:
        self._non_tensor["data"] = new_data
        return
    if "data" in self.__dataclass_fields__:
        self.set("data", new_data)
        return
    raise AttributeError("property 'data' is read-only.")


def _grad(self):
    grad = self._tensordict._grad
    if grad is None:
        return None
    # Pass a copy: _from_tensordict writes None into it for each field
    # without a grad, and that None must not reach self.
    return self._from_tensordict(self._tensordict.grad, dict(self._non_tensordict))


def _state_dict(
    self, destination=None, prefix="", keep_vars=False, flatten=True
) -> dict[str, Any]:
    """Returns the state dict of the tensorclass.

    Works as :meth:`~tensordict.TensorDictBase.state_dict`, which describes
    the arguments. The tensor fields are the entries of the state dict. The
    non-tensor fields and the name of the class go to the ``_metadata``
    attribute, under the ``"_non_tensor"`` and ``"_type"`` keys of their
    level.

    Examples:
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Foo(TensorClass):
        ...     x: torch.Tensor
        ...     name: str
        >>> foo = Foo(x=torch.zeros(3), name="a", batch_size=[3])
        >>> sd = foo.state_dict()
        >>> list(sd.keys())
        ['x']
        >>> sd._metadata[""]["_non_tensor"]
        {'name': 'a'}
    """
    import collections

    if destination is None:
        destination = collections.OrderedDict()
        destination._metadata = collections.OrderedDict()
    elif not hasattr(destination, "_metadata"):
        destination._metadata = collections.OrderedDict()

    td = super(type(self), self).__getattribute__("_tensordict")
    _non_tensordict = super(type(self), self).__getattribute__("_non_tensordict")

    non_tensor = dict(_non_tensordict)
    if not type(self)._is_non_tensor:
        non_tensor.update(dict.fromkeys(_none_fields(self)))
    for key, item in td.items():
        if is_non_tensor(item):
            non_tensor[key] = item.data
        elif not _is_tensor_collection(type(item)):
            if not keep_vars:
                destination[prefix + key] = item.detach()
            else:
                destination[prefix + key] = item
        elif flatten:
            item.state_dict(
                destination=destination,
                prefix=prefix + key + ".",
                keep_vars=keep_vars,
                flatten=True,
            )
        else:
            destination[prefix + key] = item.state_dict(
                keep_vars=keep_vars, flatten=False
            )

    metadata_key = prefix[:-1] if prefix.endswith(".") else prefix
    # Read the metadata from the TensorDict: with shadow=True,
    # self.batch_size and self.device can be fields.
    destination._metadata[metadata_key] = {
        "batch_size": td.batch_size,
        "device": td.device,
        "_type": type(self).__qualname__,
        "_non_tensor": non_tensor,
    }

    return destination


def _load_state_dict(
    self, state_dict: dict[str, Any], strict=True, assign=False, from_flatten=None
):
    """Loads a state dict made by :meth:`state_dict` into the tensorclass, and returns ``self``.

    Works as :meth:`~tensordict.TensorDictBase.load_state_dict`, which
    describes the arguments, and also loads the non-tensor fields. It also
    reads the state dicts of older versions, which hold ``"_tensordict"`` and
    ``"_non_tensordict"`` entries.

    Examples:
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Foo(TensorClass):
        ...     x: torch.Tensor
        ...     name: str
        >>> sd = Foo(x=torch.zeros(3), name="a", batch_size=[3]).state_dict()
        >>> foo = Foo(x=torch.ones(3), name="b", batch_size=[3])
        >>> _ = foo.load_state_dict(sd)
        >>> foo.x, foo.name
        (tensor([0., 0., 0.]), 'a')
    """
    # Legacy format detection
    if "_tensordict" in state_dict and "_non_tensordict" in state_dict:
        return _load_state_dict_legacy(
            self, state_dict, strict=strict, assign=assign, from_flatten=from_flatten
        )

    if from_flatten is None:
        _metadata = getattr(state_dict, "_metadata", None)
        if _metadata is not None:
            td = super(type(self), self).__getattribute__("_tensordict")
            td_keys = {
                k for k in td.keys() if not is_non_tensor(td.get(k, default=None))
            }
            sd_keys = set(state_dict.keys())
            from_flatten = sd_keys != td_keys
        else:
            from_flatten = False

    if from_flatten:
        from tensordict.base import _unflatten_state_dict

        nested_sd = _unflatten_state_dict(state_dict)
        return _load_state_dict(
            self, nested_sd, strict=strict, assign=assign, from_flatten=False
        )

    _metadata = getattr(state_dict, "_metadata", None)
    if _metadata is not None:
        local_metadata = _metadata.get("", {})
        non_tensor = local_metadata.get("_non_tensor", {})
        _non_tensordict = super(type(self), self).__getattribute__("_non_tensordict")
        td = super(type(self), self).__getattribute__("_tensordict")
        for key, value in non_tensor.items():
            if key not in type(self).__dataclass_fields__:
                raise KeyError(f"Key '{key}' wasn't expected in the state-dict.")
            if key in _non_tensordict:
                _non_tensordict[key] = value
            elif (
                value is None and not type(self)._is_non_tensor and key not in td.keys()
            ):
                # A None field is a missing key
                _set_none_field(self, key)
                continue
            else:
                td.set(
                    key,
                    NonTensorData(
                        data=value, batch_size=td.batch_size, device=td.device
                    ),
                    inplace=not assign,
                )

    td = super(type(self), self).__getattribute__("_tensordict")
    td_keys = {k for k in td.keys() if not is_non_tensor(td.get(k, default=None))}
    non_tensor_keys = set(non_tensor) if _metadata is not None else set()
    expected_keys = td_keys | non_tensor_keys

    if strict:
        for k in state_dict.keys():
            if k not in expected_keys:
                raise KeyError(f"Key '{k}' wasn't expected in the state-dict.")

    _metadata = getattr(state_dict, "_metadata", None)
    if _metadata is not None:
        local_metadata = _metadata.get("", {})
        batch_size = local_metadata.get("batch_size", td.batch_size)
        td.batch_size = batch_size

    for key, value in state_dict.items():
        if key not in td_keys:
            continue
        td_value = td.get(key, default=None)
        if (
            isinstance(value, dict)
            and td_value is not None
            and _is_tensor_collection(type(td_value))
        ):
            td_value.load_state_dict(
                value, strict=strict, assign=assign, from_flatten=from_flatten
            )
            if assign:
                td.set(key, td_value)
        else:
            td.set(key, value, inplace=not assign)

    return self


def _load_state_dict_legacy(
    self, state_dict: dict[str, Any], strict=True, assign=False, from_flatten=False
):
    """Load a legacy-format state_dict with _tensordict/_non_tensordict wrapper."""
    for key, item in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("Only str keys are allowed when calling load_state_dict.")
        if key == "_non_tensordict":
            for sub_key, sub_item in item.items():
                if isinstance(sub_item, dict) and "_non_tensordict" in sub_item:
                    raise RuntimeError(
                        "Loading a saved tensorclass on a uninitialized tensorclass is not allowed"
                    )
                else:
                    if sub_key not in type(self).__dataclass_fields__:
                        raise KeyError(
                            f"Key '{sub_key}' wasn't expected in the state-dict."
                        )
                    if sub_item is None and not type(self)._is_non_tensor:
                        # A None field is a missing key
                        continue
                    super(type(self), self).__getattribute__("_non_tensordict")[
                        sub_key
                    ] = sub_item
        elif key == "_tensordict":
            for sub_key in item.keys():
                if sub_key not in type(self).__dataclass_fields__ and sub_key not in (
                    "__batch_size",
                    "__device",
                ):
                    raise KeyError(
                        f"Key '{sub_key}' wasn't expected in the state-dict."
                    )
            super(type(self), self).__getattribute__("_tensordict").load_state_dict(
                item, strict=strict, assign=assign, from_flatten=from_flatten
            )
        else:
            raise KeyError(f"Key '{key}' wasn't expected in the state-dict.")
    return self


def _eq(self, other: object) -> bool:
    """Compares the tensorclass with another object, element by element, for equality.

    With another tensorclass, the non-tensor fields are compared too, and
    each gives a boolean tensor of the batch size.

    Args:
        other: object to compare to this object. Can be a tensorclass, a
            tensordict or any compatible type (int, float or tensor), in
            which case the equality check will be propagated to the leaves.

    Returns:
        ``False`` if ``other`` is not a tensor collection, a dict, a number or
        a tensor; otherwise a tensorclass of boolean tensors.

    Examples:
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Foo(TensorClass):
        ...     x: torch.Tensor
        ...     name: str
        >>> a = Foo(x=torch.zeros(3), name="a", batch_size=[3])
        >>> b = Foo(x=torch.zeros(3), name="b", batch_size=[3])
        >>> (a == b).x
        tensor([True, True, True])
        >>> (a == b).name
        tensor([False, False, False])

    """
    if not is_tensor_collection(other) and not isinstance(
        other, (dict, numbers.Number, Tensor)
    ):
        return False
    if is_tensorclass(other):
        tensor = self._tensordict == other._tensordict
    else:
        tensor = self._tensordict == (
            other.exclude(*_non_tensor_keys(self))
            if _is_tensor_collection(type(other))
            else other
        )
    if isinstance(self, NonTensorDataBase):
        # Return a plain tensordict containing the eq value
        return tensor
    return _from_tensordict_with_none(self, tensor)


def _ne(self, other: object) -> bool:
    """Compares the tensorclass with another object, element by element, for inequality.

    With another tensorclass, the non-tensor fields are compared too, and
    each gives a boolean tensor of the batch size.

    Args:
        other: object to compare to this object

    Returns:
        ``True`` if ``other`` is not a tensor collection, a dict, a number or
        a tensor; otherwise a tensorclass of boolean tensors.

    Examples:
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Foo(TensorClass):
        ...     x: torch.Tensor
        ...     name: str
        >>> a = Foo(x=torch.zeros(3), name="a", batch_size=[3])
        >>> b = Foo(x=torch.zeros(3), name="b", batch_size=[3])
        >>> (a != b).x
        tensor([False, False, False])
        >>> (a != b).name
        tensor([True, True, True])

    """
    if not is_tensor_collection(other) and not isinstance(
        other, (dict, numbers.Number, Tensor)
    ):
        return True
    if is_tensorclass(other):
        tensor = self._tensordict != other._tensordict
    else:
        tensor = self._tensordict != (
            other.exclude(*_non_tensor_keys(self))
            if _is_tensor_collection(type(other))
            else other
        )
    if isinstance(self, NonTensorDataBase):
        # Return a plain tensordict containing the neq value
        return tensor
    return _from_tensordict_with_none(self, tensor)


def _or(self, other: object) -> bool:
    """Computes the element-wise logical OR of the tensorclass and another object.

    As for a tensordict, the result is not locked and, except with a tensor
    ``other`` of one or more dims, has no dim names.

    Args:
        other: object to compare to this object. Can be a tensorclass, a
            tensordict or any compatible type (int, float or tensor), in
            which case the equality check will be propagated to the leaves.

    Returns:
        ``False`` if ``other`` is not a tensor collection, a dict, a number or
        a tensor, with a ``FutureWarning``: starting with TensorDict 0.17, the
        default will change to returning ``NotImplemented``, so that Python
        raises a ``TypeError``. Otherwise a tensorclass of boolean tensors.

    """
    if not is_tensor_collection(other) and not isinstance(
        other, (dict, numbers.Number, Tensor)
    ):
        _warn_unsupported_operand("__or__", other, False)
        return False
    if is_tensorclass(other):
        tensor = self._tensordict | other._tensordict
    else:
        tensor = self._tensordict | (
            other.exclude(*_non_tensor_keys(self))
            if _is_tensor_collection(type(other))
            else other
        )
    if isinstance(self, NonTensorDataBase):
        # Return a plain tensordict containing the or value
        return tensor
    return _from_tensordict_with_none(self, tensor)


def _xor(self, other: object) -> bool:
    """Computes the element-wise exclusive OR of the tensorclass and another object.

    As for a tensordict, the result is not locked and, except with a tensor
    ``other`` of one or more dims, has no dim names.

    Args:
        other: object to compare to this object. Can be a tensorclass, a
            tensordict or any compatible type (int, float or tensor), in
            which case the equality check will be propagated to the leaves.

    Returns:
        ``False`` if ``other`` is not a tensor collection, a dict, a number or
        a tensor, with a ``FutureWarning``: starting with TensorDict 0.17, the
        default will change to returning ``NotImplemented``, so that Python
        raises a ``TypeError``. Otherwise a tensorclass of boolean tensors.

    """
    if not is_tensor_collection(other) and not isinstance(
        other, (dict, numbers.Number, Tensor)
    ):
        _warn_unsupported_operand("__xor__", other, False)
        return False
    if is_tensorclass(other):
        tensor = self._tensordict ^ other._tensordict
    else:
        tensor = self._tensordict ^ (
            other.exclude(*_non_tensor_keys(self))
            if _is_tensor_collection(type(other))
            else other
        )
    if isinstance(self, NonTensorDataBase):
        # Return a plain tensordict containing the xor value
        return tensor
    return _from_tensordict_with_none(self, tensor)


def _non_tensor_items(self, include_nested=False):
    # The fields held outside the TensorDict, then its non-tensor leaves
    items = tuple(self._non_tensordict.items())
    if not type(self)._is_non_tensor:
        # A None field is a missing key
        items += tuple((key, None) for key in _none_fields(self))
    return items + self._tensordict.non_tensor_items(include_nested)


def _bool(self):
    raise RuntimeError(
        "Converting a tensorclass to boolean value is not permitted. "
        "Use len(tc.keys()) to test for keys, tc.is_empty() to test for "
        "leaves, or (a == b).all() / .any() to reduce a comparison."
    )


def _all_non_td_fields_as_str(src_dict) -> list:
    """Returns a list of string representation of non-tensor key-value pairs.

    Args:
        src_dict (dict): non_tensor_dict

    Returns:
        result (list): list of strings with key-value representation

    """
    result = []
    for key, val in src_dict.items():
        if not is_tensor_collection(val):
            result.append(f"{key}={repr(val)}")

    return result


def _unbind(self, dim: int):
    """Returns a tuple of tensorclasses, one per index along a batch dimension.

    Works as :meth:`~tensordict.TensorDictBase.unbind`: the results share
    the storage of the tensorclass.

    Args:
        dim (int): the batch dimension to remove.

    Examples:
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Foo(TensorClass):
        ...     x: torch.Tensor
        >>> foo = Foo(x=torch.arange(6).reshape(2, 3), batch_size=[2, 3])
        >>> [item.x for item in foo.unbind(1)]
        [tensor([0, 3]), tensor([1, 4]), tensor([2, 5])]
    """
    # TODO: dynamo doesn't like copy, using dict instead
    return tuple(
        type(self)._from_tensordict(td, non_tensordict=dict(self._non_tensordict))
        for td in self._tensordict.unbind(dim)
    )


_empty_and_post_init = _wrap_td_method("empty")


def _empty(self, *args, **kwargs):
    # The result is a shell that the caller fills: __post_init__ would see no
    # fields. Dynamo cannot set a ContextVar, so compiled code still runs it.
    token = None if is_compiling() else _SKIP_TENSORCLASS_POST_INIT.set(True)
    try:
        return _empty_and_post_init(self, *args, **kwargs)
    finally:
        if token is not None:
            _SKIP_TENSORCLASS_POST_INIT.reset(token)


################
# Custom classes
# --------------

_MP_MANAGER = None


def _mp_manager():
    global _MP_MANAGER
    if _MP_MANAGER is None:
        _MP_MANAGER = Manager()
    return _MP_MANAGER


def _patch_tc(cls):
    cls.__setattr__ = _setattr
    cls.__getattr__ = _getattr
    cls.__getitem__ = _getitem
    cls.__setitem__ = _setitem
    cls.__len__ = _len
    cls.__repr__ = _repr
    cls.__eq__ = _eq
    cls.__ne__ = _ne
    cls.__or__ = _or
    cls.__xor__ = _xor
    cls.__bool__ = _bool

    cls.device = property(_device, _device_setter)
    # cls.data = property(_data, _data_setter)
    cls.grad = property(_grad)

    cls._from_tensordict = classmethod(_from_tensordict)
    cls.from_tensordict = _from_tensordict_public
    cls._new_unsafe = classmethod(_new_unsafe)
    cls._load_memmap = classmethod(_load_memmap)
    cls.from_dict = classmethod(_from_dict_public)
    cls._from_dict_defaults = ()

    cls.set = _set
    cls.get = _get
    cls.update = _update
    cls.update_ = _update_
    cls.update_at_ = _update_at_
    cls.to_tensordict = _to_tensordict
    cls.to_dict = _to_dict
    cls.non_tensor_items = _non_tensor_items
    cls.unbind = _unbind
    cls._unbind = _unbind
    cls.empty = _empty
    cls.state_dict = _state_dict
    cls.load_state_dict = _load_state_dict
    cls._memmap_ = _memmap_
    cls.share_memory_ = _share_memory_
    cls.load_memmap = TensorDictBase.load_memmap
    cls.load = TensorDictBase.load
    cls.load_memmap_ = _load_memmap_
    cls.from_dict_instance = _from_dict_instance

    cls.__exit__ = _exit

    # # Methods from lists
    for method_name in _METHOD_FROM_TD:
        setattr(cls, method_name, _unchecked_copy(getattr(TensorDict, method_name)))
    for method_name in _FALLBACK_METHOD_FROM_TD:
        setattr(cls, method_name, _wrap_td_method(method_name))
    for method_name in _FALLBACK_METHOD_FROM_TD_FORCE:
        setattr(cls, method_name, _wrap_td_method(method_name))
    for method_name in _FALLBACK_METHOD_FROM_TD_NOWRAP:
        is_property = isinstance(getattr(TensorDictBase, method_name, None), property)
        # if is_property:
        #     print("method_name", method_name)
        #     continue
        setattr(
            cls,
            method_name,
            _wrap_td_method(method_name, no_wrap=True, is_property=is_property),
        )
    for method_name in _FALLBACK_METHOD_FROM_TD_COPY:
        setattr(
            cls,
            method_name,
            _wrap_td_method(method_name, copy_non_tensor=True),
        )

    # Distributed methods that need tensorclass type injection
    cls.init_remote = _tc_init_remote
    cls.broadcast = _tc_broadcast
    cls.from_remote_init = classmethod(
        lambda cls_inner, *args, **kwargs: TensorDictBase.from_remote_init(
            *args, **kwargs
        )
    )

    return cls
    # _set_methods(TensorClass)


def _is_tensorclass_flags(item: IndexType) -> bool:
    if not isinstance(item, tuple):
        item = (item,)
    return all(isinstance(_item, str) and _item in _TENSORCLASS_FLAGS for _item in item)


class _TensorClassMeta(abc.ABCMeta):
    def __new__(
        mcs,
        name,
        bases,
        namespace,
        autocast=None,
        nocast=None,
        frozen=None,
        tensor_only=None,
        shadow=None,
        skip_post_init_on_rebuild=None,
        cast_filter=NO_DEFAULT,
        kw_only=False,
        **kwargs,
    ):
        # Create the class using the ABCMeta's __new__ method
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Apply the dataclass decorator to the class
        if frozen is None and hasattr(cls, "_frozen"):
            frozen = cls._frozen
        if nocast is None and hasattr(cls, "_nocast"):
            nocast = cls._nocast
        if autocast is None and hasattr(cls, "_autocast"):
            autocast = cls._autocast
        if tensor_only is None and hasattr(cls, "_tensor_only"):
            tensor_only = cls._tensor_only
        if shadow is None and hasattr(cls, "_shadow"):
            shadow = cls._shadow
        if skip_post_init_on_rebuild is None and hasattr(
            cls, "_skip_post_init_on_rebuild"
        ):
            skip_post_init_on_rebuild = cls._skip_post_init_on_rebuild
        if cast_filter is NO_DEFAULT:
            cast_filter = getattr(cls, "_cast_filter", None)
        if name == "TensorClass":
            # if "tensordict.tensorclass" in namespace.get(
            #     "__module__", ""
            # ):
            cls = _patch_tc(cls)
            # ideally we could make it a dataclass but that won't work because:
            # - we cannot have a frozen dataclass inherit from a non frozen one
            # - if we freeze it we cannot set new methods
            # cls = dataclass(cls, init=False)

            delattr(cls, "batch_size")
            delattr(cls, "device")
            delattr(cls, "names")
        else:
            cls = tensorclass(
                frozen=bool(frozen),
                nocast=bool(nocast),
                autocast=bool(autocast),
                tensor_only=bool(tensor_only),
                shadow=bool(shadow),
                skip_post_init_on_rebuild=bool(skip_post_init_on_rebuild),
                cast_filter=cast_filter,
                kw_only=bool(kw_only),
            )(cls)

        return cls

    def __getitem__(cls, item: IndexType) -> Self:
        # Python calls a metaclass __getitem__ before __class_getitem__, so pass
        # anything other than flags or type arguments of a generic class to a
        # __class_getitem__ that a subclass defines, with the item unchanged.
        if not issubclass(cls, Generic) and not _is_tensorclass_flags(item):
            for base in cls.__mro__:
                if base is TensorClass:
                    break
                if "__class_getitem__" in base.__dict__:
                    return cls.__class_getitem__(item)
        return _TensorClassMeta._getitem_flags(cls, item)

    def _getitem_flags(cls, item: IndexType) -> Self:
        # TensorClass[flags] and the type arguments of a generic TensorClass.
        if not isinstance(item, tuple):
            item = (item,)
        if not _is_tensorclass_flags(item):
            # Type arguments of a generic class, as in
            # ``class Foo(TensorClass, Generic[T])`` or ``class Foo[T](TensorClass)``.
            # This metaclass __getitem__ hides Generic.__class_getitem__, so we
            # call it explicitly.
            if issubclass(cls, Generic):
                return Generic.__dict__["__class_getitem__"].__get__(None, cls)(item)
            raise TypeError(
                f"{cls.__name__} is not a generic class, so {cls.__name__}[...] only "
                f"accepts the flags {_TENSORCLASS_FLAGS}. Got {item}."
            )
        name = "_".join(item)  # type: ignore
        cls_name = f"TensorClass_{name}"
        bases = (cls,)
        class_dict = {}
        # Copy the __init__ method from the original class
        result = _TensorClassMeta(  # type: ignore
            cls_name,
            bases,
            class_dict,
            **{_item: True for _item in item},
        )
        # Note: We must destroy any property that is set by the tensorclass decorator.
        #  The result is a base class, so we don't need them, and they will be populated later.
        #  If they are present, the dataclass decorator applied within tensorclass will look for a default value
        #  for these guys (when shadow=True) and it will actually find them (since they're properties of the base
        #  class). This is bad because then we'll be using the property as default value - not what we want.
        if "device" in result.__dict__:
            delattr(result, "device")
        if "batch_size" in result.__dict__:
            delattr(result, "batch_size")
        if "names" in result.__dict__:
            delattr(result, "names")
        return result


class TensorClass(TensorCollection, metaclass=_TensorClassMeta):
    """A base class for dataclass-like containers of tensors.

    Subclass :class:`TensorClass` and annotate fields, as with
    :func:`dataclasses.dataclass`. The instances behave like a
    :class:`~tensordict.TensorDict`: they have a batch size, and they support
    indexing, reshaping, ``to(device)``, :func:`torch.stack`,
    :func:`torch.cat` and memory-mapping. A field can hold a tensor, a
    tensordict, another tensorclass, or any Python object (stored as
    :class:`~tensordict.NonTensorData`). This class replaces the
    :func:`~tensordict.tensorclass` decorator, which is kept for existing code.

    Use a tensorclass instead of a :class:`~tensordict.TensorDict` when the
    data has a fixed set of named fields, when you want editors and type
    checkers to know the fields, or when you want to add methods. Use a
    :class:`~tensordict.TypedTensorDict` instead if you need a
    :class:`~tensordict.TensorDictBase` subclass (``state["key"]``,
    ``**state`` unpacking). :ref:`choosing-a-container` compares the
    containers, and :ref:`tensorclass-guide` is the user guide.

    The constructor of a subclass takes the fields, by name or in order, and
    these keyword arguments: ``Foo(tensor=..., batch_size=[3])``.

    A field declared with :func:`dataclasses.field` keeps its ``default``,
    ``default_factory`` and ``metadata``, and ``repr=False`` leaves it out of
    the repr. The options ``init=False``, ``compare=False`` and
    ``kw_only=True`` are ignored, and the class warns about each when it is
    defined, unless a method of the class (``__init__``, ``__eq__`` and
    ``__ne__``) replaces the one that ignores the option. The class keyword
    ``kw_only=True`` makes the fields of the class keyword-only, as in a
    dataclass. A :class:`dataclasses.InitVar` goes to ``__post_init__``, as
    in a dataclass. See :ref:`tensorclass-field-options`.

    Keyword Args:
        batch_size (torch.Size or list of int, optional): the batch size of
            the instance: the leading dimensions that all its fields share.
            See :ref:`batch-size`. Defaults to ``torch.Size([])``.
        device (torch.device, str or int, optional): the device of the
            instance. Tensors on another device are moved to it. Defaults to
            ``None``, which leaves each tensor where it is.
        names (list of str, optional): the names of the batch dimensions, one
            per dimension. Defaults to ``None``.
        lock (bool, optional): if ``True``, the instance is locked: writing a
            field raises a ``RuntimeError``. See :ref:`keys-locking`.
            Defaults to ``False``.

    The class flags below go to the class definition instead.

    .. rubric:: Class flags

    A flag changes the behavior of every instance of the class. Set it in
    one of three equivalent forms:

        >>> import torch
        >>> from tensordict import TensorClass, tensorclass
        >>> class Foo(TensorClass["autocast"]):     # bracket form
        ...     integer: int
        >>> class Foo(TensorClass, autocast=True):  # keyword form
        ...     integer: int
        >>> @tensorclass(autocast=True)             # legacy decorator form
        ... class Foo:
        ...     integer: int

    To combine flags, list them in the brackets:

        >>> class Foo(TensorClass["nocast", "frozen"]):
        ...     integer: int

    The flags are:

    - No flag: tensor-compatible values (``int``, ``float``, ``np.ndarray``)
      are cast to tensors. Other values are stored as
      :class:`~tensordict.NonTensorData`.

          >>> class Foo(TensorClass):
          ...     x: torch.Tensor
          >>> Foo(x=1).x
          tensor(1)

      A Python scalar given to a field annotated ``int``, ``float`` or
      ``bool`` (or an optional or union of these) is cast too, with a
      ``FutureWarning``: starting with TensorDict 0.17, the default will
      change to keeping the Python value of fields annotated int/float/bool,
      as :class:`~tensordict.NonTensorData`. With a non-empty batch size,
      where a 0-d tensor does not fit, the value is kept already.

    - ``autocast``: when a field is set, the value is converted to the
      annotated type. For example, an ``int`` field set to
      ``torch.ones(())`` holds ``1``.

          >>> class Foo(TensorClass["autocast"]):
          ...     x: int
          >>> Foo(x=torch.ones(())).x
          1

    - ``nocast``: tensor-compatible values are stored as they are, not cast
      to tensors.

          >>> class Foo(TensorClass["nocast"]):
          ...     x: int
          >>> Foo(x=1).x
          1

    - ``tensor_only``: every field annotation must be a tensor type, a tensor
      collection, or an optional or union of these. Other annotations raise
      a ``TypeError`` when the class is defined. Values that can be cast to
      a tensor (Python scalars, numpy arrays) are cast. Reading a field then
      skips the lookup in the non-tensor storage.

          >>> class Foo(TensorClass["tensor_only"]):
          ...     x: torch.Tensor
          >>> Foo(x=1).x
          tensor(1)

    - ``frozen``: assigning a field raises ``FrozenInstanceError``, as with
      ``@dataclass(frozen=True)``. Subclasses of a frozen class are frozen
      too. A frozen class cannot subclass a non-frozen tensorclass: this
      raises a ``TypeError``.

          >>> class Foo(TensorClass["frozen"]):
          ...     x: torch.Tensor
          >>> foo = Foo(x=torch.zeros(3))
          >>> foo.x = torch.ones(3)
          Traceback (most recent call last):
            ...
          dataclasses.FrozenInstanceError: cannot assign to field 'x'

    - ``shadow``: by default, a field whose name is a tensorclass member
      (``batch_size``, ``device``, ``names``, ``sum``, ``apply`` and the
      other TensorDict methods) raises an ``AttributeError`` when the class
      is defined. ``shadow=True`` allows these names. The field then hides
      the member of the same name.

          >>> class Foo(TensorClass["shadow"]):
          ...     sum: torch.Tensor
          >>> Foo(sum=torch.ones(3), batch_size=[3]).sum
          tensor([1., 1., 1.])

    - ``skip_post_init_on_rebuild``: ``__post_init__`` runs on the data that
      comes in (the generated ``__init__``, ``from_tensordict``,
      ``from_dict``, the other constructors such as ``from_any`` or
      ``fromkeys``, and loads), but not when tensordict makes a new instance
      from an existing one (indexing, ``clone``, ``to``, ``apply``, stacking
      and the other operations). Subclasses inherit the flag.

          >>> class Foo(TensorClass["skip_post_init_on_rebuild"]):
          ...     x: torch.Tensor
          ...     def __post_init__(self):
          ...         print("checked")
          >>> foo = Foo(x=torch.zeros(3))
          checked
          >>> foo = foo.clone()

    ``autocast``, ``nocast`` and ``tensor_only`` are exclusive: combining two
    of them raises a ``TypeError``.

    A class keyword that is not a flag, ``cast_filter``, sets the default
    ``cast_filter`` of :meth:`~tensordict.TensorDictBase.to` and of the dtype
    shorthands (``float()``, ``bfloat16()``, ...): a predicate
    ``fn(key, value) -> bool`` that selects the tensor leaves a dtype cast
    applies to. A ``cast_filter`` passed to the call overrides it, and
    ``cast_filter=None`` there casts every leaf. Subclasses inherit it. It
    applies to calls on the tensorclass itself, not when the tensorclass is a
    value of another tensordict or tensorclass that is cast.

        >>> from tensordict import is_inexact
        >>> class Mesh(TensorClass, cast_filter=is_inexact):
        ...     points: torch.Tensor
        ...     cells: torch.Tensor
        >>> mesh = Mesh(points=torch.zeros(4, 3), cells=torch.zeros(2, 3, dtype=torch.long))
        >>> mesh.bfloat16().points.dtype, mesh.bfloat16().cells.dtype
        (torch.bfloat16, torch.int64)

    .. rubric:: Subclassing

    A class with flags is a regular class. You can subclass it and add
    flags:

        >>> class Base(TensorClass["autocast"]):
        ...     x: int
        >>> class Sub(Base, shadow=True):   # autocast inherited, shadow added
        ...     y: float

    .. rubric:: Type checking

    ``TensorClass[...]`` is implemented with :meth:`~object.__class_getitem__`,
    so pyright resolves it to the class with flags, not to a generic
    parameter. Mypy does not evaluate ``__class_getitem__`` in a list of base
    classes and rejects the bracket form. In code that mypy checks, use the
    keyword form: ``class Foo(TensorClass, autocast=True)``. Editors complete
    the annotated fields on instances.

    .. note:: ``TensorClass`` itself is not a tensorclass: only its
        subclasses become dataclasses. The reason is that a non-frozen base
        cannot have a frozen subclass, and the base cannot know whether a
        subclass sets ``frozen``.

    .. seealso::

        - :func:`~tensordict.tensorclass` for the decorator form.
        - :func:`~tensordict.from_dataclass` to migrate an existing ``@dataclass`` to a tensorclass.
        - :class:`~tensordict.nn.TensorClassModuleBase` for typed ``nn.Module`` I/O backed by
          tensorclasses.

    Examples:
        >>> from typing import Any
        >>> import torch
        >>> from tensordict import TensorClass
        >>> class Foo(TensorClass):
        ...     tensor: torch.Tensor
        ...     non_tensor: Any
        ...     nested: Any = None
        >>> foo = Foo(tensor=torch.zeros(3), non_tensor="a string!", batch_size=[3])
        >>> print(foo)
        Foo(batch_size=[3], fields={non_tensor: str, tensor: f32[], nested: None})
        >>> foo[0].non_tensor
        'a string!'
        >>> foo.reshape(1, 3).batch_size
        torch.Size([1, 3])

    """

    _autocast = False
    _nocast = False
    _frozen = False
    _tensor_only = False
    _cast_filter = None

    _is_tensorclass = True
    _is_non_tensor = False

    # Provide typing-friendly indexing support. Static type checkers look for
    # __class_getitem__ on the class rather than a metaclass __getitem__.
    # Delegate to the metaclass implementation that builds the configured subclass.
    def __class_getitem__(cls, item: IndexType) -> Self:  # type: ignore[override]
        return _TensorClassMeta._getitem_flags(cls, item)


# _TensorClassMeta does not pass the base class itself to tensorclass(), which
# registers every subclass.
_register_tensor_class(TensorClass)


def _check_equal(a, b):
    # A util to check that two non-tensor data match
    #  We're replacing this by an identity match, not a value check (which will be faster and easier to handle).
    try:
        if _is_accepted_class(type(a)) or _is_accepted_class(type(b)):
            iseq = (a == b).all() and a.shape == b.shape
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            iseq = (a == b).all() and a.shape == b.shape
        else:
            iseq = bool(a == b)
    except Exception:
        iseq = False
    return iseq


def _compares_by_element(non_tensor, other) -> bool:
    # Whether a NonTensorDataBase is compared with `other` element by element,
    # as two non-tensor stacks are
    return (
        isinstance(other, NonTensorStack)
        and other.batch_size == non_tensor.batch_size
        # the stack comparison needs at least one element
        and non_tensor.batch_size.numel() > 0
        # arrays and tensors are left out: they may hold one value per element
        # (from_struct_array makes such data), which maybe_to_stack ignores
        and not isinstance(non_tensor.data, (np.ndarray, torch.Tensor))
    )


class NonTensorDataBase(TensorClass):
    """A base class for tensordict entries that hold a Python object instead of a tensor.

    There are two subclasses:

    - :class:`~tensordict.NonTensorData` holds one value per batch element.
      Stacking or ``unsqueeze`` gives a :class:`~tensordict.NonTensorStack`.
    - :class:`~tensordict.MetaData` holds one value for the whole batch.
      Stacking items that hold the same object (checked with ``is``) gives
      one :class:`~tensordict.MetaData` with the new batch size. Other items
      give a :class:`~tensordict.NonTensorStack`. ``unsqueeze`` and indexing
      keep a :class:`~tensordict.MetaData`.

    See :ref:`non-tensor-data`.

    Args:
        data (Any): the value. A :class:`~tensordict.NonTensorData` is
            unwrapped: its value is used.

    Keyword Args:
        batch_size (torch.Size or list of int, optional): the batch size of
            the entry, usually that of the tensordict that holds it. Defaults
            to ``torch.Size([])``.
        device (torch.device, str or int, optional): the device of the entry.
            The value itself is not moved. Defaults to ``None``.
        names (list of str, optional): the names of the batch dimensions, one
            per dimension. Defaults to ``None``.

    """

    # Used to carry non-tensor data in a tensordict.
    # The advantage of storing this in a tensorclass is that we don't need
    # to patch tensordict with additional checks that will encur unwanted overhead
    # and all the overhead falls back on this class.
    data: Any
    _metadata: dict | None = None

    _is_non_tensor: bool = True

    def __repr__(self):
        data_str = str(self.data)
        if len(data_str) > 200:
            data_str = data_str[:20] + "  ...  " + data_str[-20:]
        repr_str = f"{type(self).__name__}(data={data_str}"
        if "batch_size" not in self.__expected_keys__:
            repr_str += f", batch_size={self.batch_size}"
        elif "shape" not in self.__expected_keys__:
            repr_str += f", shape={self.shape}"
        if "device" not in self.__expected_keys__:
            repr_str += f", device={self.device}"
        return repr_str + ")"

    def __post_init__(self):
        _tensordict = self.__dict__["_tensordict"]
        _non_tensordict = self.__dict__["_non_tensordict"]
        data = _non_tensordict.get("data", NO_DEFAULT)
        if data is NO_DEFAULT:
            data = _tensordict._get_str("data", default=NO_DEFAULT)
            data_inner = getattr(data, "data", None)
            if data_inner is None:
                # Support for stacks
                data_inner = data.tolist()
            del _tensordict["data"]
            _non_tensordict["data"] = data_inner

    @classmethod
    def _patch_operators(cls):
        # _tensorclass calls this when it builds the class. The first instance
        # did it before, but the Dynamo of torch 2.11 and 2.12 cannot set a
        # class attribute, so a class first instantiated in a compiled
        # function failed.
        old_eq = cls.__eq__
        if old_eq is _eq:

            @functools.wraps(_eq)
            def __eq__(self, other):
                if isinstance(other, NonTensorDataBase):
                    eqval = self.data == other.data
                    if isinstance(eqval, torch.Tensor):
                        return eqval
                    if isinstance(eqval, np.ndarray):
                        return torch.as_tensor(eqval, device=self.device)
                    return torch.full(
                        self.batch_size,
                        bool(eqval),
                        device=self.device,
                    )
                if _compares_by_element(self, other):
                    try:
                        return self.maybe_to_stack() == other
                    except Exception:
                        # values that cannot be compared (dicts holding
                        # arrays, data frames) keep the empty result
                        pass
                # # Handle comparison with scalar values (like 0, 1, etc.)
                # # For non-tensor data, we should return a boolean tensor
                # if isinstance(other, (int, float, bool)) or (isinstance(other, torch.Tensor) and other.numel() == 1):
                #     eqval = self.data == other
                #     if isinstance(eqval, torch.Tensor):
                #         return eqval
                #     if isinstance(eqval, np.ndarray):
                #         return torch.as_tensor(eqval, device=self.device)
                #     return torch.full(
                #         self.batch_size,
                #         bool(eqval),
                #         device=self.device,
                #     )
                return old_eq(self, other)

            cls.__eq__ = __eq__

            _ne = cls.__ne__

            @functools.wraps(_ne)
            def __ne__(self, other):
                if isinstance(other, NonTensorDataBase):
                    neqval = self.data != other.data
                    if isinstance(neqval, torch.Tensor):
                        return neqval
                    if isinstance(neqval, np.ndarray):
                        return torch.as_tensor(neqval, device=self.device)
                    return torch.full(
                        self.batch_size,
                        bool(neqval),
                        device=self.device,
                    )
                if _compares_by_element(self, other):
                    try:
                        return self.maybe_to_stack() != other
                    except Exception:
                        # values that cannot be compared (dicts holding
                        # arrays, data frames) keep the empty result
                        pass
                # # Handle comparison with scalar values (like 0, 1, etc.)
                # # For non-tensor data, we should return a boolean tensor
                # if isinstance(other, (int, float, bool)) or (isinstance(other, torch.Tensor) and other.numel() == 1):
                #     neqval = self.data != other
                #     if isinstance(neqval, torch.Tensor):
                #         return neqval
                #     if isinstance(neqval, np.ndarray):
                #         return torch.as_tensor(neqval, device=self.device)
                #     return torch.full(
                #         self.batch_size,
                #         bool(neqval),
                #         device=self.device,
                #     )
                return _ne(self, other)

            cls.__ne__ = __ne__

            _xor = cls.__xor__

            @functools.wraps(_xor)
            def __xor__(self, other):
                if isinstance(other, NonTensorDataBase):
                    xorval = self.data ^ other.data
                    if isinstance(xorval, torch.Tensor):
                        return xorval
                    if isinstance(xorval, np.ndarray):
                        return torch.as_tensor(xorval, device=self.device)
                    return torch.full(
                        self.batch_size,
                        bool(xorval),
                        device=self.device,
                    )
                return _xor(self, other)

            cls.__xor__ = __xor__

            _or = cls.__or__

            @functools.wraps(_or)
            def __or__(self, other):
                if isinstance(other, NonTensorDataBase):
                    orval = self.data | other.data  # yuppie!
                    if isinstance(orval, torch.Tensor):
                        return orval
                    if isinstance(orval, np.ndarray):
                        return torch.as_tensor(orval, device=self.device)
                    return torch.full(
                        self.batch_size,
                        bool(orval),
                        device=self.device,
                    )
                return _or(self, other)

            cls.__or__ = __or__

    def __call__(self, *args, **kwargs):
        """Calls the value with the arguments and returns the result."""
        return self.data(*args, **kwargs)

    def __getitem__(self, idx: IndexType) -> Self | Tensor | TensorCollection | Any:
        if isinstance(self.data, list):
            new_data = [self.data[i] for i in torch.as_tensor(idx).tolist()]
        else:
            new_data = self.data[idx]
        new_batch_size = (
            torch.Size(torch.Size(self.batch_size)[idx])
            if self.batch_size
            else torch.Size([])
        )
        return NonTensorData(
            data=new_data, batch_size=new_batch_size, device=self.device
        )

    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T | None = None,
        clone: bool = False,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        is_leaf: Callable[[Type], bool] | None = None,
        update_batch_size: bool = False,
        ignore_lock: bool = False,
        **kwargs,
    ) -> T:
        if ignore_lock and ignore_lock is not _UPDATE_IGNORE_LOCK_WARNED:
            ignore_lock = _warn_update_ignore_lock()
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.update() does not accept keyword-argument "
                "entries; pass another NonTensorDataBase instance as the positional "
                "argument."
            )
        if input_dict_or_td is None:
            return self
        return self._update(
            input_dict_or_td=input_dict_or_td,
            clone=clone,
            inplace=inplace,
            keys_to_update=keys_to_update,
            is_leaf=is_leaf,
            update_batch_size=update_batch_size,
            ignore_lock=ignore_lock,
        )

    def _update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        clone: bool = False,
        inplace: bool = False,
        *,
        keys_to_update: Sequence[NestedKey] | None = None,
        break_on_memmap: bool | None = None,
        is_leaf: Callable[[Type], bool] | None = None,
        update_batch_size: bool = False,
        ignore_lock: bool = False,
    ) -> T:
        if isinstance(input_dict_or_td, NonTensorDataBase):
            data = input_dict_or_td.data
            if inplace and self._tensordict._is_shared:
                _update_shared_nontensor(self._non_tensordict["data"], data)
                return self
            elif inplace and self._is_memmap:
                _is_memmaped_from_above = self._is_memmaped_from_above()
                if break_on_memmap is None:
                    global _BREAK_ON_MEMMAP
                    break_on_memmap = _BREAK_ON_MEMMAP
                if _is_memmaped_from_above and break_on_memmap:
                    raise RuntimeError(
                        "Cannot update a leaf NonTensorDataBase from a memmaped parent NonTensorStack. "
                        "To update this leaf node, please update the NonTensorStack with the proper index."
                    )
                share_non_tensor = self._metadata["_share_non_tensor"]
                if share_non_tensor:
                    _update_shared_nontensor(self._non_tensordict["data"], data)
                else:
                    self._non_tensordict["data"] = data
                # Force json update by setting is memmap to False
                if not _is_memmaped_from_above and "memmap_prefix" in self._metadata:
                    self._tensordict._is_memmap = False
                    self._memmap_(
                        prefix=self._metadata["memmap_prefix"],
                        copy_existing=False,
                        executor=None,
                        futures=None,
                        inplace=True,
                        like=False,
                        share_non_tensor=share_non_tensor,
                        robust_key=True,
                    )
                return self
            elif not inplace and self.is_locked:
                raise RuntimeError(_LOCK_ERROR)
            if clone:
                data = deepcopy(data)
            if inplace and self.is_locked:
                # Like an in-place tensor write, this keeps the keys of the
                # locked tensordicts. The consolidated metadata of a parent
                # holds the old value.
                self._non_tensordict["data"] = data
                self._tensordict._mark_consolidated_stale()
                return self
            self.data = data
        elif isinstance(input_dict_or_td, NonTensorStack):
            raise ValueError(
                "Cannot update a NonTensorDataBase object with a NonTensorStack. Call `non_tensor_data.maybe_to_stack()` "
                "before calling update()."
            )
        elif not input_dict_or_td.is_empty():
            raise RuntimeError(f"Unexpected type {type(input_dict_or_td)}")
        return self

    def __getattr__(self, item):
        if item == "data":
            if self.is_memmap:
                # unwrap the shared object
                return _from_shared_nontensor(self._non_tensor["data"])
            return self._non_tensor["data"]
        return _getattr(self, item)

    def maybe_to_stack(self):
        """Returns a :class:`~tensordict.NonTensorStack` with the value in every batch element.

        The elements share the value: it is not copied. If the batch size is
        empty, returns ``self``.

        Examples:
            >>> from tensordict import NonTensorData
            >>> stack = NonTensorData("a", batch_size=[2]).maybe_to_stack()
            >>> type(stack).__name__, stack.tolist()
            ('NonTensorStack', ['a', 'a'])
        """
        datalist = self.data
        if not self.batch_size:
            return self
        for i in reversed(self.batch_size):
            datalist = [datalist] * i
        return NonTensorStack._from_list(datalist, device=self.device, ndim=self.ndim)

    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T | None = None,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        **kwargs,
    ) -> T:
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.update_() does not accept keyword-argument "
                "entries; pass another NonTensorDataBase instance as the positional "
                "argument."
            )
        if input_dict_or_td is None:
            return self
        return self._update_(
            input_dict_or_td=input_dict_or_td,
            clone=clone,
            keys_to_update=keys_to_update,
        )

    def _update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        clone: bool = False,
        *,
        keys_to_update: Sequence[NestedKey] | None = None,
        break_on_memmap: bool | None = None,
    ) -> T:

        if isinstance(input_dict_or_td, NonTensorStack):
            raise RuntimeError(
                "Cannot update a NonTensorDataBase with a NonTensorStack object."
            )
        if not isinstance(input_dict_or_td, NonTensorDataBase):
            raise RuntimeError(
                "NonTensorDataBase.copy_ / update_ requires the source to be a NonTensorDataBase object."
            )
        return self._update(
            input_dict_or_td,
            inplace=True,
            clone=clone,
            keys_to_update=keys_to_update,
            break_on_memmap=break_on_memmap,
        )

    @_update_at_aliases
    def update_at_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorCollection,
        idx: IndexType,
        clone: bool = False,
        *,
        non_blocking: bool = False,
    ) -> T:
        # Check the type first: comparing an ndarray with () or slice(None) raises.
        if not (
            (isinstance(idx, tuple) and not idx)
            or (isinstance(idx, slice) and idx == slice(None))
        ):
            raise RuntimeError("Cannot update a part of a NonTensorDataBase.")
        return self.update_(
            input_dict_or_td=input_dict_or_td, clone=clone, non_blocking=non_blocking
        )

    def empty(self, recurse=False, *, device=NO_DEFAULT, batch_size=None, names=None):
        if names is None and batch_size is None:
            names = self._maybe_names()
        return type(self)(
            data=self.data,
            batch_size=self.batch_size if batch_size is None else batch_size,
            names=names,
            device=self.device if device is NO_DEFAULT else device,
        )

    def is_empty(self) -> bool:
        return False

    def _apply_nest(self, *args, out=None, **kwargs):
        # kwargs["filter_empty"] = False
        if out is not None:
            return out
        return self.empty(
            batch_size=kwargs.get("batch_size"),
            device=kwargs.get("device", NO_DEFAULT),
            names=kwargs.get("names"),
        )

    def to_dict(
        self,
        *,
        retain_none: bool = True,
        convert_tensors: bool | Literal["numpy"] = False,
        tolist_first: bool = False,
    ) -> dict[str, Any]:
        # override to_dict to return just the data
        return self.data

    def to_tensordict(self, *, retain_none: bool | None = None):
        return self

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        # A modified version of __torch_function__ to account for the different behaviour
        # of stack, which should return lazy stacks of data of data does not match.
        if func is torch.cat and all(
            issubclass(t, (NonTensorData, NonTensorStack)) for t in types
        ):
            return NonTensorData._cat_non_tensor(*args, **(kwargs or {}))
        # torch.where is not passed through: it would run on the tensordict
        # of the NonTensorData, which does not hold the data. Use
        # NonTensorData.where instead. The pointwise math functions are not
        # passed through either
        if (
            func not in _TD_PASS_THROUGH
            or func is torch.where
            or func in _POINTWISE_PASS_THROUGH
            or not all(issubclass(t, (Tensor, cls)) for t in types)
        ):
            from torch._ops import HigherOrderOperator

            if isinstance(func, HigherOrderOperator):
                if kwargs is None:
                    kwargs = {}
                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)
            return NotImplemented

        escape_conversion = func in (torch.stack,)

        if kwargs is None:
            kwargs = {}

        # get the output type from the arguments / keyword arguments
        if len(args) > 0:
            tensorclass_instance = args[0]
        else:
            tensorclass_instance = kwargs.get("input", kwargs["tensors"])
        if isinstance(tensorclass_instance, (tuple, list)):
            tensorclass_instance = tensorclass_instance[0]
        if not escape_conversion:
            args = tuple(_arg_to_tensordict(arg) for arg in args)
            kwargs = {key: _arg_to_tensordict(value) for key, value in kwargs.items()}

        result = TD_HANDLED_FUNCTIONS[func](*args, **kwargs)
        if isinstance(result, (list, tuple)):
            return type(result)(
                _from_tensordict_with_copy(tensorclass_instance, tensordict_result)
                for tensordict_result in result
            )
        if not escape_conversion:
            return _from_tensordict_with_copy(tensorclass_instance, result)
        return result

    def _fast_apply(self, *args, **kwargs):
        kwargs["filter_empty"] = False
        return _wrap_method(
            self, "_fast_apply", self._tensordict._fast_apply, nowarn=True
        )(*args, **kwargs)

    def _multithread_rebuild(self, *args, **kwargs):
        kwargs["filter_empty"] = False
        return _wrap_method(
            self,
            "_multithread_rebuild",
            self._tensordict._multithread_rebuild,
            nowarn=True,
        )(*args, **kwargs)

    def tolist(
        self,
        *,
        convert_tensors: bool | Literal["numpy"] = False,
        tolist_first: bool = False,
        as_linked_list: bool = False,
    ):
        """Returns the value in a nested list with one item per batch element.

        If the batch size is empty, returns the value itself.

        Keyword Args:
            convert_tensors (bool or "numpy", optional): accepted for
                compatibility with :meth:`~tensordict.TensorDictBase.tolist`.
                It has no effect: the items are the stored value. Defaults to
                ``False``.
            tolist_first (bool, optional): accepted for compatibility with
                :meth:`~tensordict.TensorDictBase.tolist`. It has no effect.
                Defaults to ``False``.
            as_linked_list (bool, optional): if ``True``, returns a
                ``tensordict.utils.LinkedList`` bound to the entry instead of a
                list. A :class:`NonTensorData` holds one value, so setting an
                item of the list to a non-tensor value, such as a string,
                raises a ``ValueError`` and leaves the list unchanged. Call
                :meth:`~tensordict.NonTensorData.maybe_to_stack` first to get a
                :class:`~tensordict.NonTensorStack`, whose list writes back.
                Defaults to ``False``.

        Examples:
            >>> from tensordict import NonTensorData
            >>> NonTensorData("a", batch_size=[2, 1]).tolist()
            [['a'], ['a']]
            >>> NonTensorData("a").tolist()
            'a'
        """
        if not self.batch_size:
            return self.data
        result = [
            ntd.tolist(
                convert_tensors=convert_tensors,
                tolist_first=tolist_first,
                as_linked_list=as_linked_list,
            )
            for ntd in self.unbind(0)
        ]
        if as_linked_list:
            return LinkedList(result, td=self)
        return result

    def copy_(
        self, src: NonTensorDataBase | NonTensorStack, non_blocking: bool = False
    ):
        return self.update_(src, non_blocking=non_blocking)

    def clone(self, recurse: bool = True):
        if recurse:
            return type(self)(
                data=deepcopy(self.data),
                batch_size=self.batch_size,
                device=self.device,
                names=self.names if self._has_names() else None,
            )
        return type(self)(
            data=self.data,
            batch_size=self.batch_size,
            device=self.device,
            names=self.names if self._has_names() else None,
        )

    def copy(self):
        """Returns a shallow copy: a new entry that holds the same value, as ``clone(recurse=False)`` does.

        A ``list``, ``tuple`` or ``dict`` value is copied, but not its items.
        """
        return self._clone(recurse=False)

    def _clone(self, recurse: bool = True):
        non_tensordict = self._non_tensordict
        td = self._tensordict
        if (
            not recurse
            and not is_compiling()
            and type(self) is NonTensorData
            and type(td) is TensorDict
            and not td._tensordict
            and non_tensordict.keys() == NonTensorData.__expected_keys__
            and NonTensorData.__eq__ is not _eq
        ):
            # The result of TensorClass._clone, built directly: with these
            # keys, _from_tensordict's checks pass and __post_init__ (which
            # has patched __eq__ already) does nothing.
            result = NonTensorData.__new__(NonTensorData)
            result.__dict__["_tensordict"] = td._clone(recurse=False)
            result.__dict__["_non_tensordict"] = {
                key: _copy_pytree_containers(value)
                for key, value in non_tensordict.items()
            }
        elif not recurse and all(map(tree_is_leaf, non_tensordict.values())):
            # TensorClass._clone would copy the values with tree_map, which
            # returns pytree leaves unchanged: skip it.
            result = self._from_tensordict(
                td._clone(recurse=False),
                dict(non_tensordict),
                safe=False,
            )
        else:
            result = TensorClass._clone(self, recurse)
        if not recurse and td._is_shared:
            # The shallow copy holds the same shared wrapper: mark it shared so
            # that reads unwrap the value and in-place updates write through it.
            result._tensordict._is_shared = True
        elif not recurse and td._is_memmap:
            # memmap_(share_non_tensor=True) wraps the value, and the copy is
            # not memory-mapped: give it the value.
            result._non_tensordict["data"] = self.data
        return result

    def share_memory_(self):
        if self._tensordict._is_shared:
            return self
        with self.unlock_():
            self.__dict__["_non_tensordict"]["data"] = _share_memory_nontensor(
                self.data, manager=_mp_manager()
            )
        self._tensordict.share_memory_()
        return self

    def _memmap_(
        self,
        *,
        prefix: str | None = None,
        copy_existing: bool = False,
        executor=None,
        futures=None,
        inplace=True,
        like=False,
        memmaped: bool = False,
        share_non_tensor: bool = False,
        existsok: bool = True,
        robust_key,
    ):
        # For efficiency, we can avoid doing this saving
        #  if the data is already there.
        if self._tensordict._is_memmap and str(
            getattr(self._tensordict, "_memmap_prefix", None)
        ) == str(prefix):
            return self

        _metadata = {}
        if prefix is not None:
            _metadata = dict(self._metadata) if self._metadata is not None else {}
            _metadata["memmap_prefix"] = prefix
            _metadata["memmaped"] = memmaped

        # memmap_prefix, memmaped and _share_non_tensor describe the current
        # memmap, not the data: leave them out of meta.json (memmap_prefix is
        # a Path, which would be pickled into other.pickle).
        non_tensordict = dict(self._non_tensordict)
        if self._metadata is not None:
            non_tensordict["_metadata"] = {
                key: value
                for key, value in self._metadata.items()
                if key not in ("memmap_prefix", "memmaped", "_share_non_tensor")
            } or None

        out = _memmap_(
            self,
            prefix=prefix,
            copy_existing=copy_existing,
            executor=executor,
            futures=futures,
            inplace=inplace,
            like=like,
            memmaped=memmaped,
            share_non_tensor=share_non_tensor,
            existsok=existsok,
            robust_key=robust_key,
            non_tensordict=non_tensordict,
        )
        _metadata["_share_non_tensor"] = share_non_tensor
        out._non_tensordict["_metadata"] = _metadata
        if share_non_tensor:
            out._non_tensordict["data"] = _share_memory_nontensor(
                out.data, manager=_mp_manager()
            )
        return out

    def _is_memmaped_from_above(self):
        _metadata = self._metadata
        if _metadata is None:
            return False
        return _metadata.get("memmaped", False)

    _load_memmap = classmethod(_load_memmap)


class NonTensorData(NonTensorDataBase):
    """A tensordict entry that holds a Python object instead of a tensor.

    Use it to keep strings, numbers that must not become tensors, or any
    other object next to the tensors of a tensordict: a task name, a file
    path, a configuration. Each batch element has the value: indexing a
    tensordict keeps the value in every element, and stacking gives a
    :class:`~tensordict.NonTensorStack` with one value per element. The value
    is not copied by stacking, expanding and similar operations. To keep one
    value for the whole batch through stacking and unsqueezing, use
    :class:`~tensordict.MetaData` instead. :ref:`non-tensor-data` explains
    the non-tensor classes.

    Args:
        data (Any): the value. A :class:`NonTensorData` is unwrapped: its
            value is used.

    Keyword Args:
        batch_size (torch.Size or list of int, optional): the batch size of
            the entry, usually that of the tensordict that holds it. Defaults
            to ``torch.Size([])``.
        device (torch.device, str or int, optional): the device of the entry.
            The value itself is not moved. Defaults to ``None``.
        names (list of str, optional): the names of the batch dimensions, one
            per dimension. Defaults to ``None``.

    You rarely build a :class:`NonTensorData` yourself. Assigning a value
    that is not a tensor to a key creates one, and reading the key returns
    the value. :meth:`~tensordict.TensorDictBase.get` returns the
    :class:`NonTensorData` itself:

        >>> import torch
        >>> from tensordict import NonTensorData, TensorDict
        >>> td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
        >>> td["b"] = "a string!"
        >>> td["b"]
        'a string!'
        >>> td[0]["b"]
        'a string!'
        >>> td.get("b")
        NonTensorData(data=a string!, batch_size=torch.Size([3]), device=None)

    :meth:`~tensordict.TensorDictBase.set_non_tensor` and
    :meth:`~tensordict.TensorDictBase.get_non_tensor` write and read the value
    of a nested key. Assigning a list to a key gives a
    :class:`~tensordict.NonTensorStack` with one item per batch element
    instead. :meth:`~tensordict.TensorDictBase.filter_non_tensor` removes the
    non-tensor entries of a tensordict.

    .. rubric:: Comparisons

    Unlike other tensorclasses, :class:`NonTensorData` supports ``==``,
    ``!=``, ``^`` and ``|`` between two non-tensor entries. The result is a
    boolean tensor with the batch size of the entry. ``==`` and ``!=`` also
    compare with a :class:`~tensordict.NonTensorStack` of the same batch
    size, element by element, unless the value is an array or a tensor. A
    comparison with any other object returns an empty
    :class:`~tensordict.TensorDict`, for compatibility with
    ``<a tensordict> == <float_number>``.

        >>> a = NonTensorData(True, batch_size=[3])
        >>> b = NonTensorData(True, batch_size=[3])
        >>> a == b
        tensor([True, True, True])
        >>> a ^ b
        tensor([False, False, False])

    .. rubric:: Saving

    :meth:`~tensordict.TensorDictBase.memmap` and the related methods save
    a :class:`NonTensorData`, and :meth:`~tensordict.TensorDictBase.load_memmap`
    loads it. A value that JSON can encode goes to the ``meta.json`` file of
    the entry (a tuple is loaded back as a list). Other values, such as a
    ``set`` or an instance of another class, are pickled to an
    ``other.pickle`` file. Pickle can execute arbitrary code when it loads,
    so pass ``allow_pickle=True`` to
    :meth:`~tensordict.TensorDictBase.load_memmap` only for data that you
    trust. Without it, loading a pickled value warns. Starting with
    TensorDict 0.16, the default will change to ``allow_pickle=False``,
    which raises a ``RuntimeError`` for pickled values. The ``dict``,
    ``collections.OrderedDict``, ``list`` and ``tuple`` classes that
    :meth:`~tensordict.TensorDict.from_pytree` records are also saved by
    name, and load without pickle.

        >>> import tempfile
        >>> from fractions import Fraction
        >>> data = TensorDict(batch_size=[3])
        >>> _ = data.set_non_tensor("picklable", Fraction(1, 3))
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _ = data.memmap(tmpdir)
        ...     loaded = TensorDict.load_memmap(tmpdir, allow_pickle=True)
        >>> loaded.get_non_tensor("picklable")
        Fraction(1, 3)

    .. rubric:: Preallocation

    A tensordict can hold one :class:`NonTensorData` with a batch size, then
    receive different values for some batch elements. The entry then
    becomes a :class:`~tensordict.NonTensorStack`, and ``td[key]`` returns
    a ``tensordict.utils.LinkedList`` of the values instead of one value:

        >>> td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
        >>> td["val"]
        0
        >>> td[1::2] = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])
        >>> type(td.get("val")).__name__
        'NonTensorStack'
        >>> td["val"]
        LinkedList([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    .. rubric:: Sharing between processes

    :meth:`~tensordict.TensorDictBase.memmap_` and
    :meth:`~tensordict.TensorDictBase.share_memory_` make a
    :class:`NonTensorData` that other processes can read. The shared
    tensordict is locked, so assigning the key raises an error. To write a
    new value, use :meth:`~tensordict.TensorDictBase.update_`,
    :meth:`~tensordict.TensorDictBase.update_at_` or
    :meth:`~tensordict.TensorDictBase.update` with ``inplace=True``.
    ``share_memory_`` starts a :class:`multiprocessing.Manager` process for
    the values, so these examples need a process that can start child
    processes.

        >>> td = TensorDict({"val": NonTensorData(data=0, batch_size=[])}, [])
        >>> _ = td.share_memory_()  # doctest: +SKIP
        >>> _ = td.update_(TensorDict({"val": NonTensorData(data=1)}, []))  # doctest: +SKIP
        >>> td["val"]  # doctest: +SKIP
        1
        >>> td["val"] = 2  # doctest: +SKIP
        Traceback (most recent call last):
            ...
        RuntimeError: Cannot modify locked TensorDict. For in-place modification, consider using the `set_()` method and make sure the key is present.

    A shared value can be written if it is a ``str``, ``int``, ``float``,
    ``bool``, ``dict`` or ``list``. For a value of another type (a
    dataclass instance, for example), ``update_`` skips the entry and warns.
    A shared :class:`~tensordict.NonTensorStack` can be written too, if the
    entry was a stack before the tensordict was shared:

        >>> td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
        >>> td[1::2] = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])
        >>> _ = td.share_memory_()  # doctest: +SKIP
        >>> td[1::2] = TensorDict({"val": NonTensorData(data=2, batch_size=[5])}, [5])  # doctest: +SKIP
        >>> td["val"]  # doctest: +SKIP
        LinkedList([0, 2, 0, 2, 0, 2, 0, 2, 0, 2])

    If it is still a single :class:`NonTensorData` when the tensordict is
    shared, writing different values to some elements raises a
    ``RuntimeError``.

    A memory-mapped :class:`NonTensorData` writes the new value to its JSON
    file at each update, which takes time. If the data does not need to stay
    on disk, use ``share_memory_`` instead:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     td = TensorDict({"val": NonTensorData(data=0, batch_size=[])}, [])
        ...     _ = td.memmap_(tmpdir)
        ...     _ = td.update_(TensorDict({"val": NonTensorData(data=1, batch_size=[])}, []))
        ...     print(TensorDict.load_memmap(tmpdir, allow_pickle=False)["val"])
        1

    .. rubric:: Callables

    Calling a :class:`NonTensorData` calls its value. This lets
    :meth:`~tensordict.TensorDictBase.apply` take a tensordict of functions:

        >>> x = TensorDict({"a": 1, "b": 1})
        >>> funcs = TensorDict({"a": lambda v: v - 1, "b": lambda v: v + 1})
        >>> out = x.apply(lambda v, func: func(v), funcs)
        >>> out["a"], out["b"]
        (tensor(0), tensor(2))

    """

    _load_memmap = classmethod(_load_memmap)
    _from_dict = classmethod(_from_dict)
    _from_tensordict = classmethod(_from_tensordict)
    __repr__ = NonTensorDataBase.__repr__

    @classmethod
    def _cat_non_tensor(cls, tensors, dim=0, out=None):
        # Concatenate the values, not the empty TensorDicts that carry their
        # batch sizes. Converting to those TensorDicts loses every value but
        # the first and cannot dispatch mixed NonTensorData/NonTensorStack inputs.
        batch_size = tensors[0].batch_size
        if not -len(batch_size) <= dim < len(batch_size):
            raise RuntimeError(
                f"dim must index a batch dimension, got dim={dim} and "
                f"batch_size={batch_size}"
            )
        dim %= len(batch_size)
        other_dims = batch_size[:dim] + batch_size[dim + 1 :]
        uniform = isinstance(tensors[0], NonTensorData)
        for tensor in tensors:
            shape = tensor.batch_size
            if (
                len(shape) != len(batch_size)
                or shape[:dim] + shape[dim + 1 :] != other_dims
            ):
                raise RuntimeError(
                    "Non-tensor batch sizes must match outside the concatenation "
                    f"dimension, got {batch_size} and {shape}."
                )
            uniform = (
                uniform
                and isinstance(tensor, NonTensorData)
                and _check_equal(tensor.data, tensors[0].data)
            )
        if uniform:
            result_batch_size = list(batch_size)
            result_batch_size[dim] = sum(tensor.batch_size[dim] for tensor in tensors)
            result = tensors[0].empty(
                batch_size=result_batch_size, names=tensors[0]._maybe_names()
            )
        else:
            values = [value for tensor in tensors for value in tensor.unbind(dim)]
            result = (
                cls._stack_non_tensor(values, dim=dim) if values else tensors[0].clone()
            )
        if out is not None:
            if out.batch_size != result.batch_size:
                raise RuntimeError("out.batch_size and cat batch size must match.")
            if isinstance(out, NonTensorData) and isinstance(result, NonTensorStack):
                with _set_capture_non_tensor_stack(True):
                    result = cls._stack_non_tensor(values, dim=dim)
            out.update_(result)
            return out
        return result

    def expand(self, *args, **kwargs) -> T:
        """Returns the entry expanded to a new batch size, as :meth:`torch.Tensor.expand` does.

        If the batch size changes, the result is a
        :class:`~tensordict.NonTensorStack` whose elements share the value.
        A shape that :meth:`torch.Tensor.expand` rejects (fewer sizes than
        batch dimensions, or a size that differs from a batch dimension that
        is not 1) warns, and the batch size of the result can differ from the
        shape. Starting with TensorDict 0.17, the default will change to
        raising a RuntimeError for such shapes, as
        :meth:`~tensordict.TensorDictBase.expand` does.

        Args:
            *shape (int or torch.Size): the new batch size. ``-1`` keeps the
                size of a batch dimension.

        Examples:
            >>> from tensordict import NonTensorData
            >>> expanded = NonTensorData("a", batch_size=[2]).expand(3, 2)
            >>> expanded.batch_size, expanded.tolist()
            (torch.Size([3, 2]), [['a', 'a'], ['a', 'a'], ['a', 'a']])
        """
        # tensordict_dims = self.batch_dims
        shape = _get_shape_from_args(*args, **kwargs)
        if -1 in shape:
            shape = _resolve_expand_shape(shape, self.batch_size)
        if not _expands_to(self.batch_size, shape):
            warnings.warn(
                f"Expanding a NonTensorData of batch size {list(self.batch_size)} "
                f"to {list(shape)} does not raise today, although "
                "torch.Tensor.expand rejects this shape. Starting with "
                "TensorDict 0.17, the default will change to raising a "
                "RuntimeError, as TensorDict.expand does.",
                FutureWarning,
                stacklevel=2,
            )

        # Replicate self until we have the appropriate batch size
        out = self
        for i, s in enumerate(reversed(shape)):
            j = -i - 1
            if i < self.ndim and self.batch_size[j] == s:
                continue
            elif i < self.ndim and self.batch_size[j] == 1:
                out = torch.cat([out.copy() for _ in range(s)], j)
            else:
                out = torch.stack([out.copy() for _ in range(s)])
        return out

    def unsqueeze(self, dim: int):
        return torch.stack([self], dim)

    def where(
        self,
        condition: Tensor,
        other: Any,
        *,
        out: T | None = None,
        pad: int | bool | None = None,
        update_batch_size: bool = False,
    ) -> T:
        """Returns the elements of ``self`` where ``condition`` is ``True``, and those of ``other`` elsewhere.

        When ``other`` is a :class:`NonTensorData` with another value, or a
        :class:`NonTensorStack`, the result is a :class:`NonTensorStack` with
        the value that ``condition`` picks for each element, or a
        :class:`NonTensorData` when all the elements have the same value. This
        reads ``condition``, which breaks the graph under :func:`torch.compile`.
        Otherwise, the result has the value of ``self`` everywhere.

        ``out`` receives the result if it can hold it. A :class:`NonTensorData`
        ``out`` cannot hold a :class:`NonTensorStack`: the stack is returned,
        and ``out`` is left as it is. :meth:`TensorDict.where
        <tensordict.TensorDictBase.where>` then writes the stack in place of
        the entry of its own ``out``. See
        :meth:`~tensordict.TensorDictBase.where` for the keyword arguments
        ``out``, ``pad`` and ``update_batch_size``.

        Args:
            condition (torch.Tensor): a boolean tensor with the batch size of
                the entry.
            other (Any): the non-tensor entry or tensor collection used where
                ``condition`` is ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import NonTensorData
            >>> a = NonTensorData("a", batch_size=[3])
            >>> b = NonTensorData("b", batch_size=[3])
            >>> a.where(torch.tensor([True, False, True]), b).tolist()
            ['a', 'b', 'a']
        """
        if not is_non_tensor(other) or (
            isinstance(other, NonTensorDataBase) and _check_equal(self.data, other.data)
        ):
            # every element has the value of self
            return super().where(
                condition,
                other,
                out=out,
                pad=pad,
                update_batch_size=update_batch_size,
            )
        if other.batch_size != self.batch_size:
            if not _expands_to(other.batch_size, self.batch_size):
                raise ValueError(
                    f"other of batch size {list(other.batch_size)} cannot be "
                    f"expanded to the batch size {list(self.batch_size)}."
                )
            other = other.expand(self.batch_size)

        def pick(cond, other_values):
            if isinstance(cond, bool):
                return self.data if cond else other_values
            return [pick(c, o) for c, o in _zip_strict(cond, other_values)]

        values = pick(expand_as_right(condition, self).tolist(), other.tolist())
        flat = [values]
        for _ in range(self.ndim):
            flat = [value for sublist in flat for value in sublist]
        if not flat or all(_check_equal(value, flat[0]) for value in flat[1:]):
            result = type(self)(
                data=flat[0] if flat else self.data,
                batch_size=self.batch_size,
                names=self._maybe_names(),
                device=self.device,
            )
        else:
            result = NonTensorStack._from_list(
                values, device=self.device, ndim=self.ndim
            )
        if out is None or (
            isinstance(result, NonTensorStack) and not isinstance(out, NonTensorStack)
        ):
            return result
        out.update_(result)
        return out

    @classmethod
    def _stack_non_tensor(
        cls,
        list_of_non_tensor: list[NonTensorDataBase | NonTensorStack],
        dim: int = 0,
        raise_if_non_unique=False,
    ):
        # checks have been performed previously, so we're sure the list is non-empty
        first = list_of_non_tensor[0]

        ids = set()
        firstdata = NO_DEFAULT
        return_stack = not capture_non_tensor_stack()
        if return_stack:
            return NonTensorStack(*list_of_non_tensor, stack_dim=dim)
        for data in list_of_non_tensor:
            if not isinstance(data, cls):
                if raise_if_non_unique:
                    cls._stack_non_tensor(data, raise_if_non_unique=raise_if_non_unique)
                else:
                    return_stack = True
                break
            if firstdata is NO_DEFAULT:
                firstdata = data.data
            ids.add(id(data.data))
            if len(ids) > 1:
                if _check_equal(data.data, firstdata):
                    continue
                if raise_if_non_unique:
                    raise ValueError(
                        "More than one unique value has been found in the stack."
                    )
                return_stack = True
                break
        else:
            return_stack = not capture_non_tensor_stack()
        if not return_stack:
            batch_size = list(first.batch_size)
            batch_size.insert(dim, len(list_of_non_tensor))
            return NonTensorData(
                data=first.data,
                batch_size=batch_size,
                names=first._maybe_names(),
                device=first.device,
            )

        return NonTensorStack(*list_of_non_tensor, stack_dim=dim)


def _reconstruct_typed_metadata(item, data, state):
    """Reconstruct a typed MetaData instance during unpickling."""
    from tensordict.tensorclass import MetaData

    instance = MetaData[item](data)
    instance.__dict__.update(state)
    return instance


class _MetaDataMeta(_TensorClassMeta):
    def __new__(
        mcs,
        name,
        bases,
        namespace,
        datatype=None,
        **kwargs,
    ):
        # Create the class using the parent's __new__ method
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if datatype is not None:
            cls._datatype = datatype
        # Initialize cache for typed classes
        if not hasattr(cls, "_typed_class_cache"):
            cls._typed_class_cache = {}
        return cls

    def __getitem__(cls, item):
        """Returns a subclass of ``MetaData`` that only accepts values of type ``item``.

        Its constructor raises a ``TypeError`` for a value of another type.
        """
        if cls.__name__ != "MetaData":
            # Only allow type specification on the base MetaData class
            raise TypeError(f"Cannot specify type for {cls.__name__}")

        # Check cache first
        if item in cls._typed_class_cache:  # type: ignore
            return cls._typed_class_cache[item]  # type: ignore

        # Create a new class that validates the data type
        type_name = getattr(item, "__name__", str(item))
        class_name = f"MetaData[{type_name}]"

        class TypedMetaData(cls):
            _expected_type = item
            # Define all the classes defined by MetaData
            _load_memmap = classmethod(_load_memmap)
            _from_dict = classmethod(_from_dict)
            _from_tensordict = classmethod(_from_tensordict)
            __repr__ = NonTensorDataBase.__repr__

            def __post_init__(self):
                super().__post_init__()
                # Validate the data type
                if not isinstance(self.data, item):
                    expected_name = getattr(item, "__name__", str(item))
                    actual_name = type(self.data).__name__
                    raise TypeError(
                        f"Expected data of type {expected_name}, got {actual_name}"
                    )

        TypedMetaData.__name__ = class_name
        TypedMetaData.__qualname__ = class_name

        # Add pickle support for the dynamically created class
        def __reduce__(self):
            # Return a callable that can reconstruct the object
            return (_reconstruct_typed_metadata, (item, self.data, self.__dict__))

        TypedMetaData.__reduce__ = __reduce__

        # Cache the class
        cls._typed_class_cache[item] = TypedMetaData  # type: ignore

        return TypedMetaData


class MetaData(NonTensorDataBase, metaclass=_MetaDataMeta):
    """A tensordict entry that holds one Python object for the whole batch.

    Use it for information about the batch as a whole, such as a
    configuration dict or the name of a dataset. It behaves as
    :class:`~tensordict.NonTensorData`, except for indexing, stacking,
    squeezing, unsqueezing and similar operations:

    - When stacking, :class:`MetaData` checks whether the items hold the same
      object (with ``is``, not ``==``). If they do, the result is one
      :class:`MetaData` with the stacked batch size. If they do not, the
      result is a :class:`~tensordict.NonTensorStack`.
      :meth:`~tensordict.TensorDictBase.clone` copies the value, so a clone
      and its source hold different objects.
    - :func:`~torch.unsqueeze` and indexing return a :class:`MetaData`, not a
      stack.

    ``MetaData[cls]`` makes a subclass that raises a ``TypeError`` if the
    value is not an instance of ``cls``. See :ref:`non-tensor-data`.

    Args:
        data (Any): the value. A :class:`~tensordict.NonTensorData` is
            unwrapped: its value is used.

    Keyword Args:
        batch_size (torch.Size or list of int, optional): the batch size of
            the entry, usually that of the tensordict that holds it. Defaults
            to ``torch.Size([])``.
        device (torch.device, str or int, optional): the device of the entry.
            The value itself is not moved. Defaults to ``None``.
        names (list of str, optional): the names of the batch dimensions, one
            per dimension. Defaults to ``None``.

    Examples:
        >>> import torch
        >>> from tensordict import MetaData, TensorDict
        >>> config = {"lr": 0.1}
        >>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
        >>> td["config"] = MetaData(config, batch_size=[3])
        >>> td["config"] is config
        True
        >>> stacked = torch.stack([td, td])
        >>> type(stacked.get("config")).__name__
        'MetaData'
        >>> stacked["config"] is config
        True
        >>> MetaData[str](1)
        Traceback (most recent call last):
            ...
        TypeError: Expected data of type str, got int

    """

    # Remove the __class_getitem__ method since the metaclass handles it

    _load_memmap = classmethod(_load_memmap)
    _from_dict = classmethod(_from_dict)
    _from_tensordict = classmethod(_from_tensordict)
    __repr__ = NonTensorDataBase.__repr__

    @classmethod
    def _stack_non_tensor(
        cls,
        list_of_non_tensor: list[NonTensorDataBase | NonTensorStack],
        dim: int = 0,
        raise_if_non_unique=False,
    ):
        # checks have been performed previously, so we're sure the list is non-empty
        first = list_of_non_tensor[0]

        ids = set()
        firstdata = NO_DEFAULT
        return_stack = False
        for data in list_of_non_tensor:
            if not isinstance(data, cls):
                if raise_if_non_unique:
                    cls._stack_non_tensor(data, raise_if_non_unique=raise_if_non_unique)
                else:
                    return_stack = True
                break
            if firstdata is NO_DEFAULT:
                firstdata = data.data
            ids.add(id(data.data))
            if len(ids) > 1:
                if raise_if_non_unique:
                    raise ValueError(
                        "More than one unique value has been found in the stack."
                    )
                return_stack = True
                break
        if not return_stack:
            batch_size = list(first.batch_size)
            batch_size.insert(dim, len(list_of_non_tensor))
            return cls(
                data=first.data,
                batch_size=batch_size,
                names=first._maybe_names(),
                device=first.device,
            )

        return NonTensorStack(*list_of_non_tensor, stack_dim=dim)


class TensorAttrs(TensorClass):
    """A tensorclass that holds the device, dtype and shape of one tensor.

    :meth:`~tensordict.TensorDictBase.attrs` returns a tensordict with one
    :class:`TensorAttrs` per leaf. Pass it to
    :meth:`~tensordict.TensorDictBase.to` to move or cast each leaf of another
    tensordict to the device and dtype of the matching leaf, for example when
    the leaves are on different devices. The names follow the `PyTorch Tensor
    Attributes <https://pytorch.org/docs/stable/tensor_attributes.html>`_.

    The fields start with ``tgt_`` so that they do not hide the ``device``,
    ``dtype`` and ``shape`` attributes of the tensorclass.

    Args:
        tgt_device (torch.device, optional): the device of the tensor.
            Defaults to ``None``.
        tgt_dtype (torch.dtype, optional): the dtype of the tensor. Defaults
            to ``None``.
        tgt_shape (torch.Size, optional): the shape of the tensor. Defaults
            to ``None``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> td = TensorDict({"a": torch.zeros(3, device="cpu"),
        ...                  "b": torch.zeros(3, dtype=torch.int32)}, batch_size=[3])
        >>> attrs = td.attrs()
        >>> attrs["a"].tgt_device, attrs["a"].tgt_dtype
        (device(type='cpu'), torch.float32)

    """

    tgt_device: Any = None
    tgt_dtype: Any = None
    tgt_shape: Any = None

    @classmethod
    def from_tensor(cls, tensor, *, fields=("device", "dtype", "shape")):
        """Returns a :class:`TensorAttrs` with the attributes of ``tensor``.

        Args:
            tensor (torch.Tensor): the tensor to describe.

        Keyword Args:
            fields (tuple of str, optional): the attributes to read, among
                ``"device"``, ``"dtype"`` and ``"shape"``. The other fields
                are ``None``. Defaults to ``("device", "dtype", "shape")``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorAttrs
            >>> attrs = TensorAttrs.from_tensor(torch.zeros(2, 3), fields=("dtype",))
            >>> attrs.tgt_dtype, attrs.tgt_shape
            (torch.float32, None)
        """
        kwargs = {}
        if "device" in fields:
            kwargs["tgt_device"] = tensor.device
        if "dtype" in fields:
            kwargs["tgt_dtype"] = tensor.dtype
        if "shape" in fields:
            kwargs["tgt_shape"] = torch.Size(tensor.shape)
        return cls(batch_size=(), **kwargs)


def _non_tensor_value_differs(dest: NonTensorData, value: Any) -> bool | None:
    """Whether the non-tensor ``value`` holds a value other than the value of ``dest``.

    Returns ``None`` if the values do not compare to a single bool, as arrays
    and tensors do.
    """
    data = dest.data
    try:
        other = value.data
    except AttributeError:
        # a NonTensorStack of different values
        return True
    if other is data:
        return False
    try:
        return bool(other != data)
    except Exception:
        return None


# For __setitem__ and _update_at_ we don't pass a kwarg but use a global variable instead
_BREAK_ON_MEMMAP = True


class NonTensorStack(LazyStackedTensorDict):
    """A lazy stack of :class:`~tensordict.NonTensorData`, one value per batch element.

    You get a :class:`NonTensorStack` when you stack
    :class:`~tensordict.NonTensorData` (or :class:`NonTensorStack`) instances
    with :func:`torch.stack`, or when you assign a list to a key:
    ``td["names"] = ["a", "b", "c"]``. You can also build one with
    ``NonTensorStack("a", "b")`` or ``NonTensorStack.from_list(["a", "b"])``.

    To read the values, call :meth:`tolist`. ``td["names"]`` returns a
    :class:`~tensordict.utils.LinkedList` of the values, and
    ``td.get("names")`` returns the :class:`NonTensorStack` itself. If every
    element holds the same value, :attr:`data` returns it. To keep one shared
    value for the whole batch, use :class:`~tensordict.MetaData`. See
    :ref:`non-tensor-data`.

    .. note:: ``torch.utils._pytree.tree_unflatten`` and ``tree_map`` rebuild a
        :class:`NonTensorStack` entry of a tensordict whose values are all equal
        from the input, as a :class:`~tensordict.NonTensorData`, and drop the value
        that the function returned for it; they warn when that changes the result.
        Starting with TensorDict 0.17, the default will change to keeping the
        returned value, as they do when the values differ.

    Examples:
        >>> import torch
        >>> from tensordict import NonTensorData, NonTensorStack, TensorDict
        >>> NonTensorStack("a", "b").tolist()
        ['a', 'b']
        >>> td = TensorDict(batch_size=[3])
        >>> td["names"] = ["a", "b", "c"]
        >>> td["names"]
        LinkedList(['a', 'b', 'c'])
        >>> td[1]["names"]
        'b'
        >>> type(td.get("names")).__name__
        'NonTensorStack'
        >>> data = torch.stack([
        ...     torch.stack([NonTensorData(data=(i, j)) for i in range(2)])
        ...     for j in range(3)])
        >>> data.batch_size
        torch.Size([3, 2])
        >>> data[2, 1]
        NonTensorData(data=(1, 2), batch_size=torch.Size([]), device=None)

    """

    _is_non_tensor: bool = True

    def __init__(self, *args, **kwargs):
        stack_dim = kwargs.get("stack_dim", 0)
        if (
            args
            and type(stack_dim) is int
            and not stack_dim
            and kwargs.keys() <= _FLAT_KWARGS
            and type(self) is NonTensorStack
            and not is_compiling()
        ):
            # perf: a flat stack skips the checks of the general path
            members = _flat_members(args)
            if len(members) == len(args):
                _init_flat(self, members, kwargs.get("stack_dim_name"))
                return
            # the members built so far are those that the general path builds
            args = (*members, *args[len(members) :])
        args = [
            arg if is_tensor_collection(arg) else NonTensorData(data=arg)
            for arg in args
        ]
        super().__init__(*args, **kwargs)
        if not all(is_non_tensor(item) for item in self.tensordicts):
            raise RuntimeError("All tensordicts must be non-tensors.")

    # Override: lists the stored values, not dicts of the members.
    def tolist(
        self,
        *,
        convert_nodes: bool = True,
        convert_tensors: bool | Literal["numpy"] = False,
        tolist_first: bool = False,
        as_linked_list: bool = False,
    ):
        """Returns the values of the stack in a nested list, with one item per batch element.

        Keyword Args:
            convert_nodes (bool, optional): accepted for compatibility with
                :meth:`~tensordict.TensorDictBase.tolist`. It has no effect:
                the items are the stored values, not tensordicts. Defaults to
                ``True``.
            convert_tensors (bool or "numpy", optional): accepted for
                compatibility with :meth:`~tensordict.TensorDictBase.tolist`.
                It has no effect: a value that is a tensor stays a tensor.
                Defaults to ``False``.
            tolist_first (bool, optional): accepted for compatibility with
                :meth:`~tensordict.TensorDictBase.tolist`. It has no effect.
                Defaults to ``False``.
            as_linked_list (bool, optional): if ``True``, returns a
                ``tensordict.utils.LinkedList`` bound to the stack: assigning
                one of its items writes into the stack. Defaults to ``False``.

        Examples:
            >>> from tensordict import NonTensorData
            >>> import torch
            >>> data = torch.stack([
            ...     torch.stack([NonTensorData(data=(i, j)) for i in range(2)])
            ...     for j in range(3)])
            >>> data.tolist()
            [[(0, 0), (1, 0)], [(0, 1), (1, 1)], [(0, 2), (1, 2)]]

        """
        # perf: the values of the plain members first
        result = _flat_values(self)
        iterator = self.tensordicts if self.stack_dim == 0 else self.unbind(0)
        result += [
            td.tolist(
                convert_tensors=convert_tensors,
                tolist_first=tolist_first,
                as_linked_list=as_linked_list,
            )
            for td in iterator[len(result) :]
        ]
        if as_linked_list:
            return LinkedList(result, td=self)
        return result

    def maybe_to_stack(self):
        """Returns a new stack in which each element with a batch size is a stack too.

        It calls :meth:`NonTensorDataBase.maybe_to_stack
        <tensordict.NonTensorDataBase.maybe_to_stack>` on each element, so that
        ``maybe_to_stack`` can be called on any non-tensor entry.
        """
        return type(self)(
            *[ntd.maybe_to_stack() for ntd in self.tensordicts],
            stack_dim=self.stack_dim,
        )

    # Override: builds the stack from raw values, not from dicts.
    @classmethod
    def from_list(cls, non_tensors: List[Any]):
        # Use local function because refers to cls
        def _maybe_from_list(nontensor):
            if cls is NonTensorStack and type(nontensor) in _FLAT_SCALAR_TYPES:
                # perf: NonTensorStack wraps it as below, on its fast path
                return nontensor
            if isinstance(nontensor, list):
                return cls.from_list(nontensor)
            if is_non_tensor(nontensor):
                return nontensor
            return NonTensorData(data=nontensor)

        return cls(*[_maybe_from_list(nontensor) for nontensor in non_tensors])

    # Override: a NonTensorStack holds a value, so it is never empty.
    def is_empty(self) -> bool:
        return False

    _stack_non_tensor = NonTensorData._stack_non_tensor

    @classmethod
    @deprecated(
        "NonTensorStack.from_nontensordata()",
        removal="0.17",
        replacement="NonTensorData.maybe_to_stack()",
    )
    def from_nontensordata(cls, non_tensor: NonTensorData):
        """Expands a :class:`NonTensorData` into a stack of copies of it.

        .. deprecated:: 0.15
            Use :meth:`NonTensorData.maybe_to_stack` instead. It differs in
            two ways: its elements share the data object of ``non_tensor``,
            whereas this method copies it into every element; and when
            ``non_tensor`` has an empty batch size, it returns ``non_tensor``
            itself, whereas this method returns a copy.
        """
        return cls._from_nontensordata(non_tensor)

    @classmethod
    def _from_nontensordata(cls, non_tensor: NonTensorData):
        data = non_tensor.data
        prev = NonTensorData(data=data, batch_size=[], device=non_tensor.device)
        for dim in reversed(non_tensor.shape):
            prev = cls(*[prev.clone(False) for _ in range(dim)], stack_dim=0)
        return prev

    # Override: shows the data, cut after 50 characters.
    def _repr_parts(self) -> list[str]:
        selfrepr = str(self.tolist())
        if len(selfrepr) > 50:
            selfrepr = f"{selfrepr[:50]}..."
        return [
            indent(selfrepr, prefix=4 * " "),
            indent(f"batch_size={self.batch_size}", prefix=4 * " "),
            indent(f"device={self.device}", prefix=4 * " "),
        ]

    # Override: returns the stacked values as a list, as NonTensorData.to_dict returns its data.
    def to_dict(
        self,
        *,
        retain_none: bool = True,
        convert_tensors: bool | Literal["numpy"] = False,
        tolist_first: bool = False,
        as_linked_list: bool = False,
    ) -> dict[str, Any]:
        return self.tolist(
            convert_tensors=convert_tensors,
            tolist_first=tolist_first,
            as_linked_list=as_linked_list,
        )

    # Override: a NonTensorStack is already the dense form of its data.
    def _to_tensordict(self, *, retain_none: bool | None) -> NonTensorStack:
        return self

    # Override: a NonTensorStack is its own materialized form.
    def contiguous(
        self, *, canonical: bool = False, inplace: bool = False
    ) -> NonTensorStack:
        # A NonTensorStack is already the materialized form of stacked
        # non-tensor data: it has no tensor leaves to make contiguous, and
        # the parent implementation would materialize it into an empty
        # TensorDict, silently dropping the data.
        return self

    # The shape operations of TensorDictBase build their result from the
    # tensor leaves, and a stack of non-tensor entries has none: the result
    # would be an empty TensorDict. The hooks below move the entries instead:
    # they run the operation on a tensor holding the flat position of each
    # entry, then pick the entries at the resulting positions.
    def _positions(self) -> torch.Tensor:
        return torch.arange(self.batch_size.numel()).reshape(self.batch_size)

    def _select_positions(self, positions: torch.Tensor) -> NonTensorStack:
        index = []
        for size in reversed(self.batch_size):
            index.append(positions % size)
            positions = torch.div(positions, size, rounding_mode="floor")
        # advanced indexing copies the selected entries
        return self[tuple(reversed(index))]

    # Override: move the entries, as there are no tensor leaves.
    def _batch_op(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        batch_size: torch.Size,
        names: list | None,
        *,
        index: tuple | None = None,
    ) -> NonTensorStack:
        if index is not None:
            # Let torch check the arguments, then select a slice, which shares
            # the entries the same way narrowing a tensor returns a view.
            fn(self._positions())
            return self[index]
        return self._select_positions(fn(self._positions()))

    # Override: regroup the stack when the new shape allows it, which shares
    # the entries like a view, and move the entries otherwise.
    def _reshape(
        self,
        shape: torch.Size,
        names: list | None,
        leaf_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> NonTensorStack:
        if not self.batch_size.numel():
            if shape == self.batch_size:
                # an identity unflatten
                return self
            # With no entries, regrouping the stack does not give every shape
            # (it keeps [3, 0] for [1, 0]).
            result = self._reshape_empty(shape)
            if self.is_locked:
                result.lock_()
            return result
        result = self._regroup(shape)
        if result is not None:
            return result
        if shape == self.batch_size:
            # an identity unflatten of a multi-dim stack: return self, as
            # reshape and view do
            return self
        return self._batch_op(lambda positions: positions.reshape(shape), shape, names)

    # Override: expanding the members shares the entries and is twice as fast
    # as moving them.
    def _broadcast_to(self, shape: torch.Size) -> NonTensorStack:
        return self.expand(shape)

    # Override: a NonTensorStack is the dense form of stacked non-tensor data.
    def _densify(self, layout: torch.layout) -> NonTensorStack:
        return self

    def _reshape_empty(self, shape: torch.Size) -> NonTensorStack:
        # Stack the dims before the first zero-size one, so that tolist()
        # nests like torch's. From that dim on, a stack without members, as
        # an empty index gives.
        if shape[0]:
            return type(self)(
                *[self._reshape_empty(shape[1:]) for _ in range(shape[0])]
            )
        return self._new_lazy_unsafe(
            stack_dim=0, device=self.device, batch_size=shape[1:]
        )

    def _memmap_(
        self,
        *,
        prefix: str | None = None,
        copy_existing: bool = False,
        executor=None,
        futures=None,
        inplace=True,
        like=False,
        memmaped: bool = False,
        share_non_tensor: bool = False,
        existsok: bool = True,
        robust_key,
    ) -> T:

        memmaped_leaves = memmaped
        if not memmaped and prefix is not None:
            memmaped_leaves = True

            def save_metadata(prefix=prefix, self=self):
                data = self.tolist()
                device = str(self.device) if self.device is not None else None
                if not prefix.exists():
                    os.makedirs(prefix, exist_ok=True)
                jsondict = {
                    "_type": str(type(self)),
                    **_format_metadata(type(self)),
                    "stack_dim": self.stack_dim,
                    "device": device,
                }
                json_serializable = _is_json_serializable(data)
                if json_serializable:
                    jsondict["data"] = data
                    pickled = None
                else:
                    jsondict["data"] = "pickle.pkl"
                    # Serialize before opening any file, so that data that cannot be
                    # pickled leaves the files on disk as they were.
                    pickled = pickle.dumps(data)
                from tensordict._utils_key_json import json_dumps

                json_str = json_dumps(jsondict, separators=(",", ":"))
                # Ensure we write bytes to the binary file
                if isinstance(json_str, str):
                    json_str = json_str.encode("utf-8")
                if pickled is not None:
                    with open(prefix / "pickle.pkl", "wb") as f:
                        f.write(pickled)
                with open(prefix / "meta.json", "wb") as f:
                    f.write(json_str)
                if json_serializable:
                    (prefix / "pickle.pkl").unlink(missing_ok=True)

            if executor is None:
                save_metadata()
            else:
                futures.append(executor.submit(save_metadata))
        # The leaves are all non-tensor or non-tensor stacks, and we already saved this on disk
        # The only thing remaining to do is share the data between processes
        results = []
        for i, td in enumerate(self.tensordicts):
            td: NonTensorData
            results.append(
                td._memmap_(
                    prefix=(prefix / str(i)) if prefix is not None else None,
                    copy_existing=copy_existing,
                    executor=executor,
                    futures=futures,
                    inplace=inplace,
                    like=like,
                    # tell the nested stack / nontensor that
                    # no memmapping should be executed
                    memmaped=memmaped_leaves,
                    share_non_tensor=share_non_tensor,
                    existsok=existsok,
                    robust_key=robust_key,
                )
            )
        if not inplace:
            results = self.lazy_stack(results, dim=self.stack_dim)
        else:
            results = self
        if not memmaped and prefix is not None:
            results.__dict__["_path_to_memmap"] = prefix
        return results

    @classmethod
    def _load_memmap(
        cls,
        prefix: str,
        metadata: dict,
        *,
        out=None,
        robust_key,
        allow_pickle: bool | None = None,
        **kwargs,
    ) -> LazyStackedTensorDict:
        data = metadata.get("data")
        if data is not None:
            if isinstance(data, str):
                if data != "pickle.pkl":
                    raise RuntimeError(
                        "Invalid pickled NonTensorStack filename in memmap "
                        f"metadata: {data!r}."
                    )
                data = _load_pickle_file(
                    prefix / "pickle.pkl",
                    allow_pickle=allow_pickle,
                )
            device = metadata["device"]
            if device is not None:
                device = torch.device(device)
            return cls._from_list(data, device=device)
        return super()._load_memmap(
            prefix=prefix,
            metadata=metadata,
            robust_key=robust_key,
            allow_pickle=allow_pickle,
            **kwargs,
        )

    @classmethod
    def _from_list(cls, datalist: List, device: torch.device, ndim: int | None = None):
        if (
            all(isinstance(item, list) for item in datalist)
            and all(len(item) == len(datalist[0]) for item in datalist)
            and (ndim is None or ndim > 1)
        ):
            ndim = ndim - 1 if ndim is not None else None
            return NonTensorStack(
                *(cls._from_list(item, device=device, ndim=ndim) for item in datalist),
                stack_dim=0,
            )
        if (
            datalist
            and not is_compiling()
            and all(type(item) in _FLAT_SCALAR_TYPES for item in datalist)
        ):
            # perf: build the elements directly, once the constructor has
            # parsed the device for the first one
            first = NonTensorData(
                data=datalist[0], device=device, batch_size=torch.Size([])
            )
            if first._tensordict._device is device:
                # the elements share the device object, as with the general path
                result = NonTensorStack.__new__(NonTensorStack)
                _init_flat(
                    result,
                    [first, *(_flat_ntd(item, device) for item in datalist[1:])],
                )
                return result
        return NonTensorStack(
            *(
                NonTensorData(data=item, device=device, batch_size=torch.Size([]))
                for item in datalist
            ),
            stack_dim=0,
        )

    # Override: the source is a NonTensorData or a NonTensorStack, not a mapping.
    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T | None = None,
        clone: bool = False,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        is_leaf: Callable[[Type], bool] | None = None,
        update_batch_size: bool = False,
        ignore_lock: bool = False,
        **kwargs,
    ) -> T:
        if ignore_lock and ignore_lock is not _UPDATE_IGNORE_LOCK_WARNED:
            ignore_lock = _warn_update_ignore_lock()
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.update() does not accept keyword-argument "
                "entries; pass a compatible tensor collection as the positional argument."
            )
        if input_dict_or_td is None:
            return self
        return self._update(
            input_dict_or_td=input_dict_or_td,
            clone=clone,
            inplace=inplace,
            keys_to_update=keys_to_update,
            is_leaf=is_leaf,
            update_batch_size=update_batch_size,
            ignore_lock=ignore_lock,
        )

    # Override: the source is a NonTensorData or a NonTensorStack, not a mapping.
    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T | None = None,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        **kwargs,
    ) -> T:
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.update_() does not accept keyword-argument "
                "entries; pass a compatible tensor collection as the positional argument."
            )
        if input_dict_or_td is None:
            return self
        return self._update(
            input_dict_or_td=input_dict_or_td,
            clone=clone,
            inplace=True,
            keys_to_update=keys_to_update,
        )

    def _update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        clone: bool = False,
        inplace: bool = False,
        *,
        keys_to_update: Sequence[NestedKey] | None = None,
        break_on_memmap: bool | None = None,
        non_blocking: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        update_batch_size: bool = False,
        ignore_lock: bool = False,
    ) -> T:
        if isinstance(input_dict_or_td, NonTensorData):
            # One NonTensorData per member, with the member's batch size: a
            # member that is a NonTensorData with batch dims cannot be updated
            # with a NonTensorStack
            data = input_dict_or_td.data
            input_dict_or_td = NonTensorStack(
                *[
                    NonTensorData(data, batch_size=td.batch_size, device=td.device)
                    for td in self.tensordicts
                ],
                stack_dim=self.stack_dim,
            )

        memmap = False
        if self._is_memmap and hasattr(self, "_path_to_memmap"):
            if break_on_memmap is None:
                global _BREAK_ON_MEMMAP
                break_on_memmap = _BREAK_ON_MEMMAP
            if not break_on_memmap:
                raise RuntimeError(
                    "Calling _update with break_on_memmap=False is not permitted if the stack has a path."
                )
            # this is the only way break_on_memmap is False
            break_on_memmap = False
            # remove memmap
            if self._path_to_memmap.exists():
                shutil.rmtree(self._path_to_memmap)
            memmap = True
        if is_tensorclass(input_dict_or_td):
            input_dict_or_td = input_dict_or_td._tensordict

        # update content
        if isinstance(input_dict_or_td, NonTensorStack):
            for leaf_dest, leaf_src in _zip_strict(
                self.tensordicts, input_dict_or_td.unbind(self.stack_dim)
            ):
                leaf_dest._update(
                    leaf_src,
                    clone=clone,
                    inplace=inplace,
                    keys_to_update=keys_to_update,
                    break_on_memmap=break_on_memmap,
                    is_leaf=is_leaf,
                    update_batch_size=update_batch_size,
                    ignore_lock=ignore_lock,
                )
            if memmap:
                self._memmap_(
                    prefix=self._path_to_memmap, inplace=True, robust_key=True
                )
        else:
            raise NotImplementedError(
                f"The data type {type(input_dict_or_td)} is not supported within {type(self).__name__}.update"
            )
        return self

    # Override: an advanced index copies the entries, as it does for tensors.
    def __getitem__(self, index: IndexType) -> NonTensorStack | NonTensorDataBase | Any:
        cls = type(index)
        if cls is not int and cls is not slice:
            # perf: a 1-D index tensor, array or list (or a 1-tuple of one,
            # as a TensorDict passes it) builds the copied stack in one go.
            # Ints and slices only pay the first type test.
            if cls is tuple and len(index) == 1:
                cls = type(index[0])
                element = index[0]
            else:
                element = index
            if (
                cls is list
                or (cls is Tensor or cls is np.ndarray)
                and element.ndim == 1
            ):
                result = self._getitem_copies(index)
                if result is not None:
                    return result
        result = super().__getitem__(index)
        if not isinstance(result, NonTensorStack):
            return result
        indices = index if isinstance(index, tuple) else (index,)
        if all(
            idx is None or idx is Ellipsis or isinstance(idx, (int, slice))
            for idx in indices
        ):
            # basic indexing shares the entries, like a view
            return result
        # Advanced indexing (tensors, masks, lists) copies, as it does for
        # tensors. The lazy stack would otherwise hand out the source entries
        # themselves, so a repeated index yields aliased entries and a later
        # in-place assignment on the result rewrites the source and every
        # alias at once.
        return result.clone(recurse=False)

    def _getitem_copies(self, index: IndexType) -> NonTensorStack | None:
        """``self[index]`` for an advanced index of the stack dim only, or ``None``.

        Contract: returns what :meth:`__getitem__` returns, or ``None`` when
        the result could differ, so that :meth:`__getitem__` takes its general
        path. The general path stacks the selected entries, then clones that
        stack, which builds a second stack. When the index selects whole
        entries along one dim, and the entries are ``NonTensorData`` and
        ``NonTensorStack`` instances, this builds the clone at once from
        clones of the selected entries. The index errors are those of the
        general path, which reads the index in the same way.
        """
        tensordicts = self.tensordicts
        if (
            type(self) is not NonTensorStack
            or self.stack_dim != 0
            or not tensordicts
            # e.g. a stack of MetaData, which stacks in another way
            or type(tensordicts[0]) not in (NonTensorData, NonTensorStack)
            # equal values would stack as one NonTensorData
            or capture_non_tensor_stack()
            or is_compiling()
        ):
            return None
        split = self._split_index(index)
        if split.kind != _GATHER or split.block or len(split.shape) != 1:
            return None
        if not split.parts:
            # an empty selection (the shape is (0,)), as the general path
            # builds it
            result = self._empty_getitem_result(index, split.dim)
            if isinstance(result, NonTensorStack):
                return result.clone(recurse=False)
            return result
        items = [None] * split.shape[0]
        for member, _, positions in split.parts:
            td = tensordicts[member]
            if type(td) not in (NonTensorData, NonTensorStack):
                return None
            for position in positions:
                items[position] = td
        copies = _flat_copies(items)
        if len(copies) == len(items):
            result = NonTensorStack.__new__(NonTensorStack)
            _init_flat(result, copies)
            return result
        try:
            result = NonTensorStack(
                *copies, *[td._clone(recurse=False) for td in items[len(copies) :]]
            )
        except Exception:
            # the general path raises its own error
            return None
        # as clone(recurse=False) does
        if result._is_shared or result._is_memmap:
            result.lock_()
        return result

    # Override: a NonTensorStack is a value: what it takes is wrapped in a
    # NonTensorData (a list in a NonTensorStack), and a memory-mapped stack is
    # written again after the write. (Not with _rewrite_memmap: a generator
    # context manager costs about 10% of a write.)
    def __setitem__(self, index: IndexType, value: Any):
        memmap = False
        if self._is_memmap and hasattr(self, "_path_to_memmap"):
            global _BREAK_ON_MEMMAP
            _BREAK_ON_MEMMAP = False
            memmap = True
        try:
            if not is_tensor_collection(value):
                if isinstance(value, list):
                    value = NonTensorStack(*value)
                else:
                    value = NonTensorData(data=value)
            super().__setitem__(index, value)
            if memmap:
                self._memmap_(
                    prefix=self._path_to_memmap, inplace=True, robust_key=True
                )
        finally:
            _BREAK_ON_MEMMAP = True

    def _setitem_member(self, member: int, member_index: IndexType, piece) -> None:
        td = self.tensordicts[member]
        if isinstance(td, NonTensorData):
            differs = _non_tensor_value_differs(td, piece)
            # A NonTensorData holds one value for all its elements: to take
            # other values in a part of it, the member becomes a stack. A
            # memmap stack saves itself again after the write.
            if differs is not False and (not self.is_locked or self._is_memmap):
                td = self.tensordicts[member] = td.maybe_to_stack()
            elif differs:
                raise RuntimeError(_LOCK_ERROR)
        td[member_index] = piece

    @contextlib.contextmanager
    def _rewrite_memmap(self):
        """Lets the members of a memory-mapped stack be written, and writes the stack to its path again at the end."""
        global _BREAK_ON_MEMMAP
        memmap = self._is_memmap and hasattr(self, "_path_to_memmap")
        if memmap:
            _BREAK_ON_MEMMAP = False
        try:
            yield
            if memmap:
                self._memmap_(
                    prefix=self._path_to_memmap, inplace=True, robust_key=True
                )
        finally:
            _BREAK_ON_MEMMAP = True

    # Override: a memory-mapped stack is written again after its members.
    def _update_at_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorCollection,
        idx: IndexType,
        *,
        clone: bool,
        non_blocking: bool,
        keys_to_update: list[tuple[str, ...]] | None,
    ) -> T:
        with self._rewrite_memmap():
            super()._update_at_(
                input_dict_or_td,
                idx,
                clone=clone,
                non_blocking=non_blocking,
                keys_to_update=keys_to_update,
            )
        return self

    # Override: the value that every element holds.
    @property
    def data(self) -> Self:
        """The value that every element of the stack holds.

        Raises:
            AttributeError: if the elements hold different values. Use
                :meth:`tolist` to read them.

        Examples:
            >>> import torch
            >>> from tensordict import NonTensorData
            >>> torch.stack([NonTensorData("x"), NonTensorData("x")]).data
            'x'
        """
        try:
            with _set_capture_non_tensor_stack(True):
                nt = NonTensorData._stack_non_tensor(
                    self.tensordicts, raise_if_non_unique=True
                )
                if not isinstance(nt, NonTensorData):
                    raise ValueError
                return nt.data
        except ValueError:
            raise AttributeError(
                "Cannot get the non-unique data of a NonTensorStack. Use .tolist() instead."
            )


_register_tensor_class(NonTensorStack)

# The fast paths of NonTensorStack. They apply to a flat stack: a 1-D stack of
# NonTensorData elements with an empty batch size, which is what a list of
# values becomes. They build the same objects as the general paths, and the
# elements are shared or copied as there.

# The value types that NonTensorData stores as they are, with no conversion.
_FLAT_SCALAR_TYPES = frozenset(_PYTREE_SCALAR_LEAVES)


def _flat_ntd(value, device):
    """``NonTensorData(data=value, device=device)``, built directly.

    For a ``value`` of a type in ``_FLAT_SCALAR_TYPES`` and a parsed
    ``device``: the result has the attributes that the constructor sets.
    """
    result = NonTensorData.__new__(NonTensorData)
    # object.__setattr__ rather than writes to __dict__: reading __dict__
    # would make every later attribute read of the result slower.
    setattr_ = object.__setattr__
    setattr_(
        result,
        "_tensordict",
        TensorDict._new_unsafe({}, batch_size=torch.Size([]), device=device),
    )
    setattr_(
        result,
        "_non_tensordict",
        {"data": value, "_metadata": None, "_is_non_tensor": True},
    )
    setattr_(result, "_is_initialized", True)
    return result


# The keyword arguments that the fast path of NonTensorStack() takes.
_FLAT_KWARGS = frozenset({"stack_dim", "stack_dim_name"})


def _init_flat(self, members, stack_dim_name=None):
    """``LazyStackedTensorDict.__init__(self, *members, stack_dim_name=...)`` for plain members of one device."""
    # attributes, not writes to __dict__ (see _flat_ntd)
    self._is_locked = None
    self.stack_dim = 0
    self._batch_size = torch.Size([len(members)])
    self.tensordicts = members
    self._hook_out = None
    self._hook_in = None
    if stack_dim_name is not None:
        self._td_dim_name = stack_dim_name


def _is_plain_member(td):
    """Whether ``td`` is a NonTensorData with an empty batch size and nothing else to check."""
    if type(td) is not NonTensorData:
        return False
    inner = td._tensordict
    return (
        type(inner) is TensorDict
        and not inner._batch_size
        and not inner._tensordict
        and not inner._is_shared
        and not inner._is_memmap
        and not inner._is_locked
        and inner._td_dim_names is None
    )


def _flat_members(args):
    """The members of ``NonTensorStack(*args)`` for a flat stack.

    The args are values of a type in ``_FLAT_SCALAR_TYPES``, which are wrapped
    in a NonTensorData as the general path does, and plain members, which the
    stack holds as they are. If an arg is anything else or the devices
    differ, returns the members of the args before it: the general path then
    builds the stack or raises.
    """
    members = []
    append = members.append
    device = _UNSET
    for arg in args:
        if type(arg) in _FLAT_SCALAR_TYPES:
            arg = _flat_ntd(arg, None)
            arg_device = None
        elif _is_plain_member(arg):
            arg_device = arg._tensordict._device
        else:
            break
        if arg_device != device:
            if device is not _UNSET:
                break
            device = arg_device
        append(arg)
    return members


def _flat_copies(members):
    """``[td._clone(recurse=False) for td in members]`` for plain members.

    For members of one device that hold a value of a type in
    ``_FLAT_SCALAR_TYPES`` (which a clone does not copy) and no metadata.
    Stops at the first other member, and returns the copies before it.
    """
    copies = []
    if not _is_plain_member(members[0]):
        return copies
    device = members[0]._tensordict._device
    append = copies.append
    for td in members:
        if not _is_plain_member(td) or td._tensordict._device != device:
            break
        non_tensordict = td._non_tensordict
        data = non_tensordict.get("data", _UNSET)
        if (
            type(data) not in _FLAT_SCALAR_TYPES
            or len(non_tensordict) != 3
            or non_tensordict.get("_metadata", _UNSET) is not None
            or non_tensordict.get("_is_non_tensor") is not True
        ):
            break
        # what NonTensorData._clone(recurse=False) builds: the values are not
        # copied, and the batch size and device objects are shared
        inner = td._tensordict
        copy = NonTensorData.__new__(NonTensorData)
        object.__setattr__(
            copy,
            "_tensordict",
            TensorDict._new_unsafe(
                {}, batch_size=inner._batch_size, device=inner._device
            ),
        )
        object.__setattr__(copy, "_non_tensordict", dict(non_tensordict))
        append(copy)
    return copies


def _flat_values(stack):
    """The first items of ``stack.tolist()``: the values of its first plain members."""
    values = []
    if type(stack) is not NonTensorStack or stack.stack_dim != 0 or is_compiling():
        return values
    append = values.append
    for td in stack.tensordicts:
        if not _is_plain_member(td):
            break
        data = td._non_tensordict.get("data", _UNSET)
        if data is _UNSET:
            break
        append(data)
    return values


def _share_memory_nontensor(data, manager: Manager):
    # bool is a subclass of int: check it first so that it stays a bool.
    if isinstance(data, bool):
        return mp.Value(ctypes.c_bool, data)
    if isinstance(data, int):
        _warn_if_int_wraps(data)
        return mp.Value(ctypes.c_longlong, data)
    if isinstance(data, float):
        return mp.Value(ctypes.c_double, data)
    if isinstance(data, dict):
        result = manager.dict()
        result.update(data)
        return result
    if isinstance(data, str):
        data = data.encode("utf-8")
        result = mp.Array(ctypes.c_char, max(100, len(data)))
        result[: len(data)] = data
        return result
    if isinstance(data, list):
        result = manager.list()
        result.extend(data)
        return result
    # bytes also end up here: a c_char array would be read back as a str.
    # In all other cases, we just return the tensor. It's ok because the content
    # will be passed to the remote process using regular serialization. We will
    # lock the update in _update_shared_nontensor though.
    return data


def _warn_if_int_wraps(val: int) -> None:
    if not -(2**63) <= val < 2**63:
        warnings.warn(
            "An integer outside the signed 64-bit range was written to a shared "
            f"non-tensor entry: it is stored as {ctypes.c_longlong(val).value}.",
            UserWarning,
        )


def _from_shared_nontensor(nontensor):
    if isinstance(nontensor, multiprocessing.managers.ListProxy):
        return list(nontensor)
    if isinstance(nontensor, multiprocessing.managers.DictProxy):
        return dict(nontensor)
    if isinstance(nontensor, multiprocessing.sharedctypes.Synchronized):
        return nontensor.value
    if isinstance(nontensor, multiprocessing.sharedctypes.SynchronizedArray):
        byte_list = []
        for byte in nontensor:
            if byte == b"\x00":
                break
            byte_list.append(byte)
        return b"".join(byte_list).decode("utf-8")
    return nontensor


def _update_shared_nontensor(nontensor, val):
    if isinstance(nontensor, multiprocessing.managers.ListProxy):
        nontensor[:] = []
        nontensor.extend(val)
    elif isinstance(nontensor, multiprocessing.managers.DictProxy):
        nontensor.clear()
        nontensor.update(val)
    elif isinstance(nontensor, multiprocessing.sharedctypes.Synchronized):
        obj = nontensor.get_obj()
        if isinstance(obj, ctypes.c_bool):
            # c_bool stores the truth value of any object: take only what an
            # int slot takes
            val = operator.index(val)
        elif isinstance(obj, ctypes.c_longlong) and isinstance(val, int):
            _warn_if_int_wraps(val)
        nontensor.value = val
    elif isinstance(nontensor, multiprocessing.sharedctypes.SynchronizedArray):
        val = val.encode("utf-8")
        if len(val) > len(nontensor):
            # The array has a fixed size. Cut on a character boundary so that
            # the stored bytes stay valid UTF-8 and can be read back.
            n_bytes = len(val)
            val = val[: len(nontensor)].decode("utf-8", "ignore").encode("utf-8")
            warnings.warn(
                f"A shared non-tensor string holds at most {len(nontensor)} bytes: "
                f"the new {n_bytes}-byte value was cut to its first {len(val)} bytes.",
                UserWarning,
            )
        for i, byte in enumerate(nontensor):
            if i < len(val):
                v = val[i]
                nontensor[i] = v
            elif byte == b"\x00":
                break
            else:
                nontensor[i] = b"\x00"
        # nontensor[0] = val.encode("utf-8")
    else:
        raise NotImplementedError(
            f"Updating {type(nontensor).__name__} within a shared/memmaped structure is not supported."
        )


def _shared_nontensor_value(nontensor, val):
    """The value that :func:`_update_shared_nontensor` writes into ``nontensor`` for ``val``, or ``NO_DEFAULT`` if the write would raise.

    The manager of a shared list or dict pickles the value it is given, so a
    generator cannot be written. These take a list or dict built from a value
    that pickles: the write stores that one, so an iterator is read once.
    """
    try:
        if isinstance(nontensor, multiprocessing.managers.ListProxy):
            pickle.dumps(val)
            # extend takes any iterable
            val = list(val)
        elif isinstance(nontensor, multiprocessing.managers.DictProxy):
            pickle.dumps(val)
            # update takes a mapping or an iterable of pairs
            val = dict(val)
        elif isinstance(nontensor, multiprocessing.sharedctypes.Synchronized):
            # ctypes converts with __index__ in an int or bool slot, and also
            # with __float__ in a float slot
            if isinstance(nontensor.get_obj(), ctypes.c_double) and hasattr(
                type(val), "__float__"
            ):
                float(val)
            else:
                operator.index(val)
        elif isinstance(nontensor, multiprocessing.sharedctypes.SynchronizedArray):
            if not isinstance(val, str):
                return NO_DEFAULT
        else:
            return NO_DEFAULT
    except Exception:
        return NO_DEFAULT
    return val


def _stored_non_tensor(dest, source, break_on_memmap: bool | None = None):
    """The source that ``dest.copy_`` writes into ``dest``, a shared or memory-mapped non-tensor leaf, for ``source``.

    That is ``source``, or a copy holding the list or dict that a shared
    ``dest`` stores, or ``None`` if ``dest`` cannot store the values of
    ``source``. A memory-mapped value that does not share its data takes any
    value that it can write to its files: JSON or pickle. A member of a
    memory-mapped stack takes none, unless the write goes through the stack
    (``break_on_memmap=False``, as in :meth:`NonTensorDataBase._update`). The
    pairs that ``copy_`` rejects whatever the storage are left to ``copy_``.
    """
    if not is_non_tensor(source):
        return source
    if isinstance(dest, NonTensorStack):
        broadcast = not isinstance(source, NonTensorStack)
        if broadcast:
            sources = [source] * len(dest.tensordicts)
        else:
            sources = source.unbind(dest.stack_dim)
            if len(sources) != len(dest.tensordicts):
                return source
        if dest._is_memmap and hasattr(dest, "_path_to_memmap"):
            # the stack lets its members be written, then saves itself again
            break_on_memmap = False
        stored = []
        for member, value in zip(dest.tensordicts, sources):
            if broadcast and stored and isinstance(source.data, Iterator):
                # an iterator is read once: the other members take what the
                # first one read
                value = stored[0]
            value = _stored_non_tensor(member, value, break_on_memmap)
            if value is None:
                return None
            stored.append(value)
        if all(new is old for new, old in zip(stored, sources)):
            return source
        return NonTensorStack(*stored, stack_dim=dest.stack_dim)
    if isinstance(source, NonTensorStack):
        return source
    if break_on_memmap is None:
        break_on_memmap = _BREAK_ON_MEMMAP
    if dest._is_memmap and dest._is_memmaped_from_above() and break_on_memmap:
        # the write raises: only the stack writes its members
        return None
    if dest._tensordict._is_shared or (
        dest._is_memmap and dest._metadata.get("_share_non_tensor")
    ):
        data = _shared_nontensor_value(dest._non_tensordict["data"], source.data)
        if data is NO_DEFAULT:
            return None
        if data is source.data:
            return source
        return NonTensorData(data, batch_size=dest.batch_size, device=dest.device)
    if dest._is_memmap and not _is_json_serializable(source.data):
        # the write pickles the value: try it before any entry is written
        try:
            pickle.dumps(source.data)
        except Exception:
            return None
    return source
