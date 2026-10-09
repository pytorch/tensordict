# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from tensordict._lazy import LazyStackedTensorDict
from tensordict._td import _SubTensorDict, TensorDict, TensorDictBase
from tensordict.persistent import PersistentTensorDict

# implement_for and is_compiling stay importable from here for the deprecated
# tensordict.implement_for and tensordict.is_compiling aliases of
# tensordict/__init__.py, until 0.17.
from tensordict.utils import _shape, implement_for, is_compiling  # noqa: F401
from torch.compiler import is_dynamo_compiling
from torch.utils._pytree import Context, MappingKey, register_pytree_node

PYTREE_REGISTERED_TDS = (
    _SubTensorDict,
    TensorDict,
    PersistentTensorDict,
)
PYTREE_REGISTERED_LAZY_TDS = (LazyStackedTensorDict,)


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
        "non_tensor_data": d.non_tensor_items(),
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
    if (
        batch_size is None
        or is_dynamo_compiling()
        or (
            shapes
            and any(isinstance(dim, torch.SymInt) for dim in shapes[0][:batch_dims])
        )
    ):
        # Compilation path (torch.export): batch_size was not stored because it
        # may contain SymInts which torch.export cannot serialize. Or the values
        # have symbolic shapes: torch.export builds its example inputs with
        # dynamic dims from a spec flattened in eager mode, and comparing them
        # with its batch_size would specialize them (Dynamo shows SymInts as
        # ints, hence the is_dynamo_compiling() check). Reconstruct from the
        # leading batch_dims dimensions of the actual tensor shapes.
        batch_size = shapes[0][:batch_dims] if shapes else torch.Size([0] * batch_dims)
    else:
        if shapes and any(s[:batch_dims] != batch_size for s in shapes):
            # Values have different leading dims than the original batch_size.
            # This happens when torch.func transforms (jacrev, jacfwd, hessian)
            # create basis vectors with extra leading dimensions. We infer a new
            # batch_size from the common prefix of all value shapes, capped at
            # batch_dims + 1 to include at most one extra (basis) dimension.
            #
            # NOTE: when tensors have no feature dimensions (ndim == batch_dims),
            # the basis leading dim can coincidentally equal a batch dim, making
            # it impossible to detect the mismatch here. In that case, the
            # TensorDict should be created with batch_size=[] or the tensors
            # should be given at least one feature dimension (e.g. via unsqueeze).
            min_dims = min(len(s) for s in shapes)
            max_prefix_len = min(min_dims, batch_dims + 1)
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
    raise NotImplementedError


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
