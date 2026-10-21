# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import warnings

# defaultdict stays importable from here for the deprecated
# tensordict.defaultdict alias of tensordict/__init__.py, until 0.17.
from collections import defaultdict  # noqa: F401
from typing import Any, Dict, List, Tuple

import torch
import torch.utils._pytree
from pyvers import implement_for  # noqa: F401
from tensordict._lazy import LazyStackedTensorDict
from tensordict._td import _SubTensorDict, TensorDict, TensorDictBase
from tensordict.base import is_tensor_collection
from tensordict.persistent import PersistentTensorDict

# implement_for (imported from pyvers above) and is_compiling stay importable
# from here for the deprecated tensordict.implement_for and
# tensordict.is_compiling aliases of tensordict/__init__.py, until 0.17.
from tensordict.utils import (  # noqa: F401
    _is_tensorclass,
    _shape,
    is_compiling,
    is_non_tensor,
)
from torch.compiler import is_dynamo_compiling
from torch.utils._pytree import Context, MappingKey, register_pytree_node, SequenceKey

PYTREE_REGISTERED_TDS = (
    _SubTensorDict,
    TensorDict,
    PersistentTensorDict,
)
PYTREE_REGISTERED_LAZY_TDS = (LazyStackedTensorDict,)

# The frames that _warn_non_tensor_stack_rebuilt skips to reach the user's call
_SKIPPED_DIRS = (
    os.path.dirname(os.path.abspath(__file__)) + os.sep,
    os.path.dirname(os.path.abspath(torch.utils._pytree.__file__)) + os.sep,
)


class _PytreeBatchSize:
    """The batch size of a TensorDict in its pytree context.

    Two of them compare equal when they have the same number of dims: like the
    shape of a tensor leaf, the batch size is not part of the tree structure.
    torch.export checks that the inputs of an exported module have the
    structure of the example inputs, and a batch dim exported as dynamic can
    take another size. ``batch_size`` is ``None`` when it held SymInts (traced
    with dynamic shapes), which would be stale once the trace is over.
    """

    __slots__ = ("batch_dims", "batch_size")

    def __init__(self, batch_size: torch.Size) -> None:
        self.batch_dims = len(batch_size)
        self.batch_size = (
            None
            if any(isinstance(dim, torch.SymInt) for dim in batch_size)
            else batch_size
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _PytreeBatchSize):
            return NotImplemented
        return self.batch_dims == other.batch_dims

    def __hash__(self) -> int:
        return hash(self.batch_dims)

    def __repr__(self) -> str:
        if self.batch_size is None:
            return f"{type(self).__name__}(batch_dims={self.batch_dims})"
        return f"{type(self).__name__}({list(self.batch_size)})"


class _PytreeNonTensorStack:
    """A NonTensorStack entry in a pytree context, compared by its values."""

    __slots__ = ("stack",)

    def __init__(self, stack: LazyStackedTensorDict) -> None:
        self.stack = stack

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _PytreeNonTensorStack):
            return NotImplemented
        return self.stack.tolist() == other.stack.tolist()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.stack.tolist()})"


def _pytree_non_tensor_data(d: TensorDictBase) -> dict | tuple:
    # The non-tensor fields of a tensorclass, or the non-tensor entries of a
    # tensordict, which _tensordict_constructor rebuilds with set_non_tensor.
    # Storing the entries as they are would make comparing two contexts call
    # NonTensorData.__eq__, which returns a tensor when the entry has a batch
    # size.
    if _is_tensorclass(type(d)):
        # The fields held outside the TensorDict of the tensorclass. A dict:
        # the fields of a NonTensorData rebuilt by set_non_tensor are in
        # another order than those of the original entry.
        non_tensor = dict(d._non_tensordict)
        if not type(d)._is_non_tensor:
            # tensorclass imports this module
            from tensordict.tensorclass import _none_fields

            # A None field is a missing key
            non_tensor.update(dict.fromkeys(_none_fields(d)))
        return non_tensor
    items = []
    for key, value in d.non_tensor_items():
        if isinstance(value, LazyStackedTensorDict):
            # Reading the data of a NonTensorStack compares all its values,
            # and Dynamo cannot trace that: leave it to set_non_tensor.
            value = _PytreeNonTensorStack(value)
        else:
            # set_non_tensor unwraps its value in the same way.
            while is_non_tensor(value):
                value = value.data
        items.append((key, value))
    return tuple(items)


def _tensordict_flatten(d: TensorDict) -> Tuple[List[Any], Context]:
    items = tuple(d.items())
    if items:
        keys, values = zip(*items)
        keys = list(keys)
        values = list(values)
    else:
        keys = []
        values = []
    context = {
        "keys": keys,
        "names": d.names if d._has_names() else None,
        "device": d.device,
        "constructor": _constructor(type(d)),
        "non_tensor_data": _pytree_non_tensor_data(d),
        "cls": type(d),
    }
    if is_dynamo_compiling():
        # During torch.export with dynamic shapes, batch_size may contain SymInts.
        # torch.export cannot serialize SymInts in the output pytree spec
        # (as_python_constant raises). Store batch_dims (int) instead and
        # reconstruct batch_size from tensor shapes in _tensordict_unflatten.
        # See: https://github.com/pytorch/tensordict/issues/1003
        context["batch_dims"] = len(d.batch_size)
    else:
        # Eager path: store batch_size so that torch.func transforms
        # (jacrev, jacfwd, hessian) can detect basis-vector shape mismatches.
        # Non-strict torch.export also takes this path (is_compiling() is True
        # there, is_dynamo_compiling() is not): the exported module checks that
        # the spec of its inputs, flattened in eager mode, equals the spec of
        # the example inputs, flattened while tracing.
        context["batch_size"] = _PytreeBatchSize(d.batch_size)
    return values, context


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
    if any(tensor is None for tensor in values):
        return
    shapes = [_shape(v) for v in values if hasattr(v, "shape")]
    if "batch_dims" in context:
        batch_dims = context["batch_dims"]
        batch_size = None
    else:
        batch_dims = context["batch_size"].batch_dims
        batch_size = context["batch_size"].batch_size
    if not shapes:
        # No tensor to read the batch size from: keep the stored one.
        if batch_size is None:
            batch_size = torch.Size([0] * batch_dims)
    elif (
        batch_size is None
        or is_dynamo_compiling()
        or any(isinstance(dim, torch.SymInt) for dim in shapes[0][:batch_dims])
    ):
        # Compilation path (torch.export): batch_size was not stored because it
        # may contain SymInts which torch.export cannot serialize. Or the values
        # have symbolic shapes: torch.export builds its example inputs with
        # dynamic dims from a spec flattened in eager mode, and comparing them
        # with its batch_size would specialize them (Dynamo shows SymInts as
        # ints, hence the is_dynamo_compiling() check). Reconstruct from the
        # leading batch_dims dimensions of the actual tensor shapes.
        # A nested tensordict without leaves has the batch size [0] * batch_dims,
        # or its stored one, which is static. A non-tensor entry is rebuilt with
        # the batch size [0] * batch_dims too (set_non_tensor replaces it
        # afterwards). Read the batch size from another value when there is one.
        shapes = [
            _shape(v)
            for v in values
            if hasattr(v, "shape")
            and not is_non_tensor(v)
            and not (is_tensor_collection(v) and v.is_empty())
        ] or shapes
        batch_size = shapes[0][:batch_dims]
    else:
        if shapes and any(s[:batch_dims] != batch_size for s in shapes):
            # Values have different leading dims than the original batch_size.
            # This happens when torch.func transforms (jacrev, jacfwd, hessian)
            # create basis vectors with extra leading dimensions. We infer a new
            # batch_size from the common prefix of all value shapes, capped at
            # batch_dims + 1 to include at most one extra (basis) dimension, and
            # at batch_dims when the values do not have the original batch_size
            # after their first dim (e.g. tree_map over tensordicts with different
            # batch sizes).
            #
            # NOTE: when tensors have no feature dimensions (ndim == batch_dims),
            # the basis leading dim can coincidentally equal a batch dim, making
            # it impossible to detect the mismatch here. In that case, the
            # TensorDict should be created with batch_size=[] or the tensors
            # should be given at least one feature dimension (e.g. via unsqueeze).
            has_basis_dim = all(s[1 : batch_dims + 1] == batch_size for s in shapes)
            min_dims = min(len(s) for s in shapes)
            max_prefix_len = min(min_dims, batch_dims + has_basis_dim)
            common_dims = 0
            for i in range(max_prefix_len):
                if all(s[i] == shapes[0][i] for s in shapes):
                    common_dims = i + 1
                else:
                    break
            batch_size = torch.Size(shapes[0][:common_dims])
            context["names"] = None
    names = context["names"]
    keys = context["keys"]
    constructor = context["constructor"]
    non_tensor_items = context["non_tensor_data"]
    cls = context["cls"]
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
    return context["cls"](
        *values, stack_dim=stack_dim, stack_dim_name=context["stack_dim_name"]
    )


def _td_flatten_with_keys(
    d: TensorDictBase,
):
    # Same context as _tensordict_flatten: torch.export compares the spec of
    # tree_flatten_with_path(call inputs) with the spec of tree_flatten(example inputs).
    values, context = _tensordict_flatten(d)
    return [(MappingKey(k), v) for k, v in zip(context["keys"], values)], context


def _lazy_td_flatten_with_keys(
    d: LazyStackedTensorDict,
):
    # Same context as _lazy_tensordict_flatten: torch.export compares the spec of
    # tree_flatten_with_path(call inputs) with the spec of tree_flatten(example inputs).
    values, context = _lazy_tensordict_flatten(d)
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _register_td_node(cls):
    register_pytree_node(
        cls,
        _tensordict_flatten,
        _tensordict_unflatten,
        flatten_with_keys_fn=_td_flatten_with_keys,
    )


def _register_lazy_td_node(cls):
    register_pytree_node(
        cls,
        _lazy_tensordict_flatten,
        _lazy_tensordict_unflatten,
        flatten_with_keys_fn=_lazy_td_flatten_with_keys,
    )


def _constructor(cls):
    # Not a module-level dict: Dynamo guards on all the keys of a global dict
    # read with a non-constant key, so a class added later would recompile.
    if _is_tensorclass(cls):
        return _tensorclass_constructor
    return _tensordict_constructor


def _tensorclass_constructor(
    *, cls, keys, values, batch_size, names, device, non_tensor_items
):
    # The non-tensor fields that the tensorclass holds in its TensorDict are
    # children without tensors, rebuilt with their stored, static batch size.
    # Set them again with the batch size of the tensorclass, as
    # _tensordict_constructor does with the non-tensor entries of a
    # tensordict: otherwise torch.export with a dynamic batch dimension
    # specializes it to the example's size. Setting them again is slow: skip
    # it when the batch sizes are equal and static (Dynamo shows SymInts as
    # ints, hence the is_compiling() check).
    reset = is_compiling() or any(isinstance(dim, torch.SymInt) for dim in batch_size)
    td_non_tensor = tuple(
        (key, value.data)
        for key, value in zip(keys, values)
        if is_non_tensor(value)
        and not isinstance(value, LazyStackedTensorDict)
        and (reset or value.batch_size != batch_size)
    )
    result = _tensordict_constructor(
        cls=TensorDict,
        keys=keys,
        values=values,
        batch_size=batch_size,
        names=names,
        device=device,
        non_tensor_items=td_non_tensor,
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
        if isinstance(item, _PytreeNonTensorStack):
            leaf = result._get_str(key, None)
            try:
                result.set_non_tensor(key, item.stack)
            except AttributeError:
                # The values of the stack are not all equal, so it has no
                # single value to set: keep the leaf that the stack was
                # flattened to, already in result.
                if not (
                    is_non_tensor(leaf)
                    and leaf.batch_size[: result.batch_dims] == result.batch_size
                ):
                    raise
                continue
            if (
                is_non_tensor(leaf)
                and leaf.batch_size[: result.batch_dims] == result.batch_size
            ):
                _warn_non_tensor_stack_rebuilt(key, leaf, result._get_str(key, None))
            continue
        result.set_non_tensor(key, item)
    return result


def _warn_non_tensor_stack_rebuilt(key: str, leaf: Any, rebuilt: Any) -> None:
    """Warns that ``leaf``, the value the pytree function returned for a stack of equal values, was replaced by ``rebuilt``."""
    if (
        not isinstance(leaf, LazyStackedTensorDict)
        and leaf.batch_size == rebuilt.batch_size
        and leaf.device == rebuilt.device
    ):
        # Imported here: tensordict.tensorclass imports this module.
        from tensordict.tensorclass import _check_equal

        if leaf.data is rebuilt.data or _check_equal(leaf.data, rebuilt.data):
            return
    # point to the first frame outside tensordict and torch.utils: the
    # user's tree_map or tree_unflatten call
    frame, stacklevel = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(_SKIPPED_DIRS):
        frame, stacklevel = frame.f_back, stacklevel + 1
    warnings.warn(
        f"The NonTensorStack at key {key!r} holds the same value in every "
        "element, so tree_unflatten (and tree_map) rebuilds it from the input "
        "as a NonTensorData and drops the value that the function returned "
        "for it. Starting with TensorDict 0.17, the default will change to "
        "keeping that value, as when the values differ: the entry then stays "
        "a NonTensorStack.",
        FutureWarning,
        stacklevel=stacklevel,
    )


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


for cls in PYTREE_REGISTERED_TDS:
    _register_td_node(cls)
for cls in PYTREE_REGISTERED_LAZY_TDS:
    _register_lazy_td_node(cls)
