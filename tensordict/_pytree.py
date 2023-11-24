# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Tuple

from tensordict import (
    LazyStackedTensorDict,
    PersistentTensorDict,
    SubTensorDict,
    TensorDict,
)

try:
    from torch.utils._pytree import Context, register_pytree_node
except ImportError:
    from torch.utils._pytree import (
        _register_pytree_node as register_pytree_node,
        Context,
    )

PYTREE_REGISTERED_TDS = (
    LazyStackedTensorDict,
    SubTensorDict,
    TensorDict,
    PersistentTensorDict,
)


def _str_to_dict(str_spec: str) -> Tuple[List[str], str]:
    assert str_spec[1] == "("
    assert str_spec[-1] == ")"
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
    return list(d.values()), {
        "keys": list(d.keys()),
        "batch_size": d.batch_size,
        "names": d.names,
        "device": d.device,
    }


def _tensordictdict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    device = context["device"]
    if device is not None:
        device = (
            device
            if all(val.device == device for val in values if hasattr(val, "device"))
            else None
        )
    return TensorDict(
        dict(zip(context["keys"], values)),
        context["batch_size"],
        names=context["names"],
        device=device,
    )


for cls in PYTREE_REGISTERED_TDS:
    register_pytree_node(
        cls,
        _tensordict_flatten,
        _tensordictdict_unflatten,
    )
