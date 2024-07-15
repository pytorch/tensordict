# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from tensordict._lazy import LazyStackedTensorDict
from tensordict._td import _SubTensorDict, TensorDict, TensorDictBase
from tensordict.base import _NESTED_TENSORS_AS_LISTS
from tensordict.persistent import PersistentTensorDict
from tensordict.utils import _shape, implement_for

try:
    from torch.utils._pytree import Context, MappingKey, register_pytree_node
except ImportError:
    from torch.utils._pytree import (
        _register_pytree_node as register_pytree_node,
        Context,
    )

PYTREE_REGISTERED_TDS = (
    _SubTensorDict,
    TensorDict,
    PersistentTensorDict,
)
PYTREE_REGISTERED_LAZY_TDS = (LazyStackedTensorDict,)


def _str_to_dict(str_spec: str) -> Tuple[List[str], str]:
    if str_spec[1] != "(" or str_spec[-1] != ")":
        raise ValueError(
            f"string must have '(' as a second character and ')' in last position. Got {str_spec}."
        )
    context_and_child_strings = str_spec[2:-1]

    child_strings = []
    context_strings = []
    nested_parentheses = 0
    start_index = 0
    for i, char in enumerate(context_and_child_strings):
        if char == ":":
            if nested_parentheses == 0:
                context_strings.append(context_and_child_strings[start_index:i])
                start_index = i + 1
        elif char == "(":
            nested_parentheses += 1
        elif char == ")":
            nested_parentheses -= 1

        if nested_parentheses == 0 and char == ",":
            child_strings.append(context_and_child_strings[start_index:i])
            start_index = i + 1

    child_strings.append(context_and_child_strings[start_index:])
    return context_strings, ",".join(child_strings)


def _str_to_tensordictdict(str_spec: str) -> Tuple[List[str], str]:
    context_and_child_strings = str_spec[2:-1]

    child_strings = []
    context_strings = []
    nested_parentheses = 0
    start_index = 0
    for i, char in enumerate(context_and_child_strings):
        if char == ":":
            if nested_parentheses == 0:
                context_strings.append(context_and_child_strings[start_index:i])
                start_index = i + 1
        elif char == "(":
            nested_parentheses += 1
        elif char == ")":
            nested_parentheses -= 1

        if nested_parentheses == 0 and char == ",":
            child_strings.append(context_and_child_strings[start_index:i])
            start_index = i + 1

    child_strings.append(context_and_child_strings[start_index:])
    return context_strings, ",".join(child_strings)


def _tensordict_flatten(d: TensorDict) -> Tuple[List[Any], Context]:
    items = tuple(d.items())
    if items:
        keys, values = zip(*items)
        keys = list(keys)
        values = list(values)
    else:
        keys = []
        values = []
    return values, {
        "keys": keys,
        "batch_size": d.batch_size,
        "names": d.names if d._has_names() else None,
        "device": d.device,
        "constructor": _constructor(type(d)),
        "non_tensor_data": d.non_tensor_items(),
        "cls": type(d),
    }


def _lazy_tensordict_flatten(d: LazyStackedTensorDict) -> Tuple[List[Any], Context]:
    return list(d.tensordicts), {
        "stack_dim_name": d._td_dim_name,
        "stack_dim": d.stack_dim,
        "constructor": _lazy_tensordict_constructor,
        "cls": type(d),
    }


def _tensordict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    device = context["device"]
    if device is not None:
        device = (
            device
            if all(val.device == device for val in values if hasattr(val, "device"))
            else None
        )
    batch_size = context["batch_size"]
    names = context["names"]
    keys = context["keys"]
    constructor = context["constructor"]
    non_tensor_items = context["non_tensor_data"]
    cls = context["cls"]
    batch_dims = len(batch_size)
    if any(tensor is None for tensor in values):
        return
    if any(_shape(tensor)[:batch_dims] != batch_size for tensor in values):
        batch_size = torch.Size([])
        names = None
    return constructor(
        cls=cls,
        keys=keys,
        values=values,
        batch_size=batch_size,
        names=names,
        device=device,
        non_tensor_items=non_tensor_items,
    )


def _lazy_tensordict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    stack_dim = context["stack_dim"]
    return cls(*values, stack_dim=stack_dim, stack_dim_name=context["stack_dim_name"])


def _td_flatten_with_keys(
    d: TensorDictBase,
):
    items = tuple(d.items(is_leaf=_NESTED_TENSORS_AS_LISTS))
    if items:
        keys, values = zip(*items)
        keys = list(keys)
        values = list(values)
    else:
        keys = []
        values = []
    return [(MappingKey(k), v) for k, v in zip(keys, values)], {
        "keys": keys,
        "batch_size": d.batch_size,
        "names": d._maybe_names(),
        "device": d.device,
        "constructor": _constructor(type(d)),
        "non_tensor_data": d.non_tensor_items(),
        "cls": type(d),
    }


def _lazy_td_flatten_with_keys(
    d: LazyStackedTensorDict,
):
    raise NotImplementedError


@implement_for("torch", None, "2.3")
def _register_td_node(cls):
    register_pytree_node(
        cls,
        _tensordict_flatten,
        _tensordict_unflatten,
    )


@implement_for("torch", "2.3")
def _register_td_node(cls):  # noqa: F811
    register_pytree_node(
        cls,
        _tensordict_flatten,
        _tensordict_unflatten,
        flatten_with_keys_fn=_td_flatten_with_keys,
    )


@implement_for("torch", None, "2.3")
def _register_lazy_td_node(cls):
    register_pytree_node(
        cls,
        _lazy_tensordict_flatten,
        _lazy_tensordict_unflatten,
    )


@implement_for("torch", "2.3")
def _register_lazy_td_node(cls):  # noqa: F811
    register_pytree_node(
        cls,
        _lazy_tensordict_flatten,
        _lazy_tensordict_unflatten,
        flatten_with_keys_fn=_lazy_td_flatten_with_keys,
    )


def _constructor(cls):
    return _CONSTRUCTORS[cls]


def _tensorclass_constructor(
    *, cls, keys, values, batch_size, names, device, non_tensor_items
):
    result = _tensordict_constructor(
        cls=TensorDict,
        keys=keys,
        values=values,
        batch_size=batch_size,
        names=names,
        device=device,
        non_tensor_items=(),
    )
    result = cls._from_tensordict(result, dict(non_tensor_items))
    return result


def _tensordict_constructor(
    *, cls, keys, values, batch_size, names, device, non_tensor_items
):
    result = cls._new_unsafe(
        dict(zip(keys, values)),
        batch_size=batch_size,
        names=names,
        device=device,
    )
    for key, item in non_tensor_items:
        result.set_non_tensor(key, item)
    return result


def _lazy_tensordict_constructor(
    *, cls, keys, values, batch_size, names, device, non_tensor_items
):

    result = cls._new_unsafe(
        dict(zip(keys, values)),
        batch_size=batch_size,
        names=names,
        device=device,
    )
    for key, item in non_tensor_items:
        result.set_non_tensor(key, item)
    return result


_CONSTRUCTORS = defaultdict(lambda: _tensordict_constructor)
_CONSTRUCTORS[LazyStackedTensorDict] = _lazy_tensordict_constructor


for cls in PYTREE_REGISTERED_TDS:
    _register_td_node(cls)
for cls in PYTREE_REGISTERED_LAZY_TDS:
    _register_lazy_td_node(cls)
