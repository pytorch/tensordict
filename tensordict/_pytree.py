# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional, Tuple

from tensordict import (
    LazyStackedTensorDict,
    PersistentTensorDict,
    SubTensorDict,
    TensorDict,
)

from torch.utils._pytree import _register_pytree_node, Context


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


def _maybe_str_to_tensordict(str_spec: str) -> Optional[Tuple[Any, Context, str]]:
    if not str_spec.startswith("D"):
        return None
    context_strings, child_strings = _str_to_dict(str_spec)
    return TensorDict, context_strings, child_strings


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
    }


def _tensordictdict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    return TensorDict(
        dict(zip(context["keys"], values)),
        context["batch_size"],
        names=context["names"],
    )


def _tensordict_to_str(spec: "TreeSpec", child_strings: List[str]) -> str:  # noqa: F821
    context_child_strings = []
    for key, child_string in zip(spec.context, child_strings):
        context_child_strings.append(f"{key}:{child_string}")
    return f"D({','.join(context_child_strings)})"


for cls in (LazyStackedTensorDict, SubTensorDict, TensorDict, PersistentTensorDict):
    _register_pytree_node(
        cls,
        _tensordict_flatten,
        _tensordictdict_unflatten,
        _tensordict_to_str,
        _maybe_str_to_tensordict,
    )
