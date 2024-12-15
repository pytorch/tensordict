# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import functools

from typing import Any, Callable, Sequence, TypeVar

import torch
from tensordict._lazy import LazyStackedTensorDict
from tensordict._td import TensorDict
from tensordict.base import (
    _is_leaf_nontensor,
    _is_tensor_collection,
    NO_DEFAULT,
    TensorDictBase,
)
from tensordict.persistent import PersistentTensorDict
from tensordict.utils import (
    _check_keys,
    _ErrorInteceptor,
    _shape,
    _zip_strict,
    DeviceType,
    is_non_tensor,
    is_tensorclass,
    lazy_legacy,
    set_lazy_legacy,
)
from torch import Tensor
from torch.nn.parameter import (
    UninitializedBuffer,
    UninitializedParameter,
    UninitializedTensorMixin,
)

try:
    from torch.compiler import is_compiling
except ImportError:  # torch 2.0
    from torch._dynamo import is_compiling

TD_HANDLED_FUNCTIONS: dict[Callable, Callable] = {}
LAZY_TD_HANDLED_FUNCTIONS: dict[Callable, Callable] = {}
T = TypeVar("T", bound="TensorDictBase")

try:
    from torch.utils._pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_flatten

    def tree_leaves(pytree):
        """Torch 2.0 compatible version of tree_leaves."""
        return tree_flatten(pytree)[0]


def implements_for_td(torch_function: Callable) -> Callable[[Callable], Callable]:
    """Register a torch function override for TensorDict."""

    @functools.wraps(torch_function)
    def decorator(func: Callable) -> Callable:
        TD_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


def implements_for_lazy_td(torch_function: Callable) -> Callable[[Callable], Callable]:
    """Register a torch function override for TensorDict."""

    @functools.wraps(torch_function)
    def decorator(func: Callable) -> Callable:
        LAZY_TD_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements_for_td(torch.unbind)
def _unbind(td: T, *args: Any, **kwargs: Any) -> tuple[T, ...]:
    return td.unbind(*args, **kwargs)


@implements_for_td(torch.gather)
def _gather(
    input: T,
    dim: int,
    index: Tensor,
    *,
    sparse_grad: bool = False,
    out: T | None = None,
) -> T:
    if sparse_grad:
        raise NotImplementedError(
            "sparse_grad=True not implemented for torch.gather(tensordict, ...)"
        )
    # the index must have as many dims as the tensordict
    if not len(index):
        raise RuntimeError("Cannot use torch.gather with an empty index")
    dim_orig = dim
    if dim < 0:
        dim = input.batch_dims + dim
    if dim > input.batch_dims - 1 or dim < 0:
        raise RuntimeError(
            f"Cannot gather tensordict with shape {input.shape} along dim {dim_orig}."
        )

    def _gather_tensor(tensor, dest_container=None, dest_key=None):
        if dest_container is not None:
            dest = dest_container._get_str(dest_key, default=NO_DEFAULT)
        else:
            dest = None
        index_expand = index
        while index_expand.ndim < tensor.ndim:
            index_expand = index_expand.unsqueeze(-1)
        target_shape = list(tensor.shape)
        target_shape[dim] = index_expand.shape[dim]
        index_expand = index_expand.expand(target_shape)
        out = torch.gather(tensor, dim, index_expand, out=dest)
        return out

    if out is None:
        if len(index.shape) == input.ndim and input._has_names():
            names = input.names
        else:
            names = None
        device = input.device
        return TensorDict(
            {
                key: _gather_tensor(value)
                for key, value in input.items(is_leaf=_is_leaf_nontensor)
            },
            batch_size=index.shape,
            names=names,
            device=device,
        )
    for key, value in input.items(is_leaf=_is_leaf_nontensor):
        _gather_tensor(value, out, key)
    return out


@implements_for_td(torch.full_like)
def _full_like(td: T, fill_value: float, *args, **kwargs: Any) -> T:
    def full_like(x):
        return torch.full_like(x, fill_value, *args, **kwargs)

    return td._fast_apply(
        full_like,
        inplace=True,
        propagate_lock=True,
        device=kwargs.get("device", NO_DEFAULT),
    )


@implements_for_td(torch.zeros_like)
def _zeros_like(td: T, *args, **kwargs: Any) -> T:
    def zeros_like(x):
        return torch.zeros_like(x, *args, **kwargs)

    td_clone = td._fast_apply(
        zeros_like,
        propagate_lock=True,
        device=kwargs.get("device", NO_DEFAULT),
    )
    if "dtype" in kwargs:
        raise ValueError("Cannot pass dtype to full_like with TensorDict")
    if "device" in kwargs:
        td_clone = td_clone.to(kwargs.pop("device"))
    if len(kwargs):
        raise RuntimeError(
            f"keyword arguments {list(kwargs.keys())} are not "
            f"supported with full_like with TensorDict"
        )
    return td_clone


@implements_for_td(torch.ones_like)
def _ones_like(td: T, *args, **kwargs: Any) -> T:
    def ones_like(x):
        return torch.ones_like(x, *args, **kwargs)

    td_clone = td._fast_apply(
        ones_like,
        propagate_lock=True,
        device=kwargs.get("device", NO_DEFAULT),
    )
    if "device" in kwargs:
        td_clone = td_clone.to(kwargs.pop("device"))
    if len(kwargs):
        raise RuntimeError(
            f"keyword arguments {list(kwargs.keys())} are not "
            f"supported with full_like with TensorDict"
        )
    return td_clone


@implements_for_td(torch.rand_like)
def _rand_like(td: T, *args, **kwargs: Any) -> T:
    def rand_like(x):
        return torch.rand_like(x, *args, **kwargs)

    td_clone = td._fast_apply(
        rand_like,
        propagate_lock=True,
        device=kwargs.get("device", NO_DEFAULT),
    )
    if "device" in kwargs:
        td_clone = td_clone.to(kwargs.pop("device"))
    if len(kwargs):
        raise RuntimeError(
            f"keyword arguments {list(kwargs.keys())} are not "
            f"supported with full_like with TensorDict"
        )
    return td_clone


@implements_for_td(torch.randn_like)
def _randn_like(td: T, *args, **kwargs: Any) -> T:
    def randn_like(x):
        return torch.randn_like(x, *args, **kwargs)

    td_clone = td._fast_apply(
        randn_like,
        propagate_lock=True,
        device=kwargs.get("device", NO_DEFAULT),
    )
    if "device" in kwargs:
        td_clone = td_clone.to(kwargs.pop("device"))
    if len(kwargs):
        raise RuntimeError(
            f"keyword arguments {list(kwargs.keys())} are not "
            f"supported with full_like with TensorDict"
        )
    return td_clone


@implements_for_td(torch.empty_like)
def _empty_like(td: T, *args, **kwargs) -> T:
    def empty_like(x):
        return torch.empty_like(x, *args, **kwargs)

    return td._fast_apply(
        empty_like,
        propagate_lock=True,
        device=kwargs.get("device", NO_DEFAULT),
    )


@implements_for_td(torch.clone)
def _clone(td: T, *args: Any, **kwargs: Any) -> T:
    return td.clone(*args, **kwargs)


@implements_for_td(torch.squeeze)
def _squeeze(td: T, *args: Any, **kwargs: Any) -> T:
    return td.squeeze(*args, **kwargs)


@implements_for_td(torch.unsqueeze)
def _unsqueeze(td: T, *args: Any, **kwargs: Any) -> T:
    return td.unsqueeze(*args, **kwargs)


@implements_for_td(torch.masked_select)
def _masked_select(td: T, *args: Any, **kwargs: Any) -> T:
    return td.masked_select(*args, **kwargs)


@implements_for_td(torch.permute)
def _permute(td: T, dims: Sequence[int]) -> T:
    return td.permute(*dims)


@implements_for_td(torch.cat)
def _cat(
    list_of_tensordicts: Sequence[T],
    dim: int = 0,
    device: DeviceType | None = None,
    out: T | None = None,
) -> T:
    if not len(list_of_tensordicts):
        raise RuntimeError("list_of_tensordicts cannot be empty")

    batch_size = list(list_of_tensordicts[0].batch_size)
    if dim < 0:
        dim = len(batch_size) + dim
    if dim >= len(batch_size):
        raise RuntimeError(
            f"dim must be in the range 0 <= dim < len(batch_size), got dim"
            f"={dim} and batch_size={batch_size}"
        )
    batch_size[dim] = sum([td.batch_size[dim] for td in list_of_tensordicts])
    batch_size = TensorDict._parse_batch_size(None, batch_size)

    # check that all tensordict match
    keys = _check_keys(list_of_tensordicts, strict=True)
    if out is None:
        out = {}
        for key in keys:
            items = [td._get_str(key, NO_DEFAULT) for td in list_of_tensordicts]
            if not is_compiling():
                with _ErrorInteceptor(
                    key, "Attempted to concatenate tensors on different devices at key"
                ):
                    out[key] = torch.cat(items, dim)
            else:
                out[key] = torch.cat(items, dim)
        if device is None:
            device = list_of_tensordicts[0].device
            for td in list_of_tensordicts[1:]:
                if device == td.device:
                    continue
                else:
                    device = None
                    break
        names = None
        if list_of_tensordicts[0]._has_names():
            names = list_of_tensordicts[0].names
        return TensorDict._new_unsafe(
            out, device=device, batch_size=batch_size, names=names
        )
    else:
        if out.batch_size != batch_size:
            raise RuntimeError(
                "out.batch_size and cat batch size must match, "
                f"got out.batch_size={out.batch_size} and batch_size"
                f"={batch_size}"
            )

        for key in keys:
            with (
                _ErrorInteceptor(
                    key, "Attempted to concatenate tensors on different devices at key"
                )
                if not is_compiling()
                else contextlib.nullcontext()
            ):
                if isinstance(out, TensorDict):
                    torch.cat(
                        [td.get(key) for td in list_of_tensordicts],
                        dim,
                        out=out.get(key),
                    )
                else:
                    out.set_(
                        key, torch.cat([td.get(key) for td in list_of_tensordicts], dim)
                    )
        return out


@implements_for_lazy_td(torch.cat)
def _lazy_cat(
    list_of_tensordicts: Sequence[LazyStackedTensorDict],
    dim: int = 0,
    out: LazyStackedTensorDict | None = None,
) -> LazyStackedTensorDict:
    # why aren't they feeding you?
    if not len(list_of_tensordicts):
        raise RuntimeError("list_of_tensordicts cannot be empty")

    batch_size = list(list_of_tensordicts[0].batch_size)
    if dim < 0:
        dim = len(batch_size) + dim
    if dim >= len(batch_size):
        raise RuntimeError(
            f"dim must be in the range 0 <= dim < len(batch_size), got dim"
            f"={dim} and batch_size={batch_size}"
        )
    stack_dim = list_of_tensordicts[0].stack_dim
    if any((td.stack_dim != stack_dim) for td in list_of_tensordicts):
        raise RuntimeError("cat lazy stacked tds must have same stack dim")

    batch_size[dim] = sum(td.batch_size[dim] for td in list_of_tensordicts)
    batch_size = torch.Size(batch_size)

    new_dim = dim
    if dim > stack_dim:
        new_dim = dim - 1

    if out is None:
        out = []
        if dim == stack_dim:  # if dim is stack, just add all to the same list
            for lazy_td in list_of_tensordicts:
                if lazy_td.batch_size[stack_dim] == 0:
                    continue
                out += lazy_td.tensordicts
        else:
            for i in range(len(list_of_tensordicts[0].tensordicts)):
                out.append(
                    torch.cat(
                        [lazy_td.tensordicts[i] for lazy_td in list_of_tensordicts],
                        new_dim,
                    )
                )
        return type(list_of_tensordicts[0])(*out, stack_dim=stack_dim)
    else:
        if not isinstance(out, LazyStackedTensorDict):
            return _cat(list_of_tensordicts, dim=dim, out=out)

        if out.batch_size != batch_size:
            raise RuntimeError(
                "out.batch_size and cat batch size must match, "
                f"got out.batch_size={out.batch_size} and batch_size"
                f"={batch_size}"
            )
        if out.stack_dim != dim:
            index_base = (slice(None),) * out.stack_dim
            for i, sub_dest in enumerate(out.tensordicts):
                index = index_base + (i,)
                tds_to_cat = [_td[index] for _td in list_of_tensordicts]
                torch.cat(tds_to_cat, dim, out=sub_dest)
        else:
            init_idx = 0
            for td_in in list_of_tensordicts:
                sub_dest = out.tensordicts[init_idx : init_idx + td_in.shape[dim]]
                init_idx += init_idx + td_in.shape[dim]
                LazyStackedTensorDict.maybe_dense_stack(sub_dest, out.stack_dim).update(
                    td_in, inplace=True
                )

        return out


@implements_for_td(torch.stack)
def _stack(
    list_of_tensordicts: Sequence[TensorDictBase],
    dim: int = 0,
    device: DeviceType | None = None,
    out: T | None = None,
    strict: bool = False,
    contiguous: bool = False,
    maybe_dense_stack: bool | None = None,
) -> T:
    if not len(list_of_tensordicts):
        raise RuntimeError("list_of_tensordicts cannot be empty")
    if maybe_dense_stack is None:
        maybe_dense_stack = lazy_legacy()
    is_tc = any(is_tensorclass(td) for td in list_of_tensordicts)
    if all(is_non_tensor(td) for td in list_of_tensordicts):
        from tensordict.tensorclass import NonTensorData

        return NonTensorData._stack_non_tensor(list_of_tensordicts, dim=dim)
    if is_tc:
        tc_type = type(list_of_tensordicts[0])
        list_of_tensordicts = [tc._tensordict for tc in list_of_tensordicts]

    batch_size = list_of_tensordicts[0].batch_size
    if dim < 0:
        dim = len(batch_size) + dim + 1

    for td in list_of_tensordicts[1:]:
        if td.batch_size != list_of_tensordicts[0].batch_size:
            raise RuntimeError(
                "stacking tensordicts requires them to have congruent batch sizes, "
                f"got td1.batch_size={td.batch_size} and td2.batch_size="
                f"{list_of_tensordicts[0].batch_size}"
            )

    # check that all tensordict match
    # Read lazy_legacy
    _lazy_legacy = lazy_legacy()

    if out is None:
        # We need to handle tensordicts with exclusive keys and tensordicts with
        # mismatching shapes.
        # The first case is handled within _check_keys which fails if keys
        # don't match exactly.
        # The second requires a check over the tensor shapes.
        device = list_of_tensordicts[0].device
        if contiguous or not _lazy_legacy:
            try:
                keys = _check_keys(list_of_tensordicts, strict=True)
            except KeyError:
                if not _lazy_legacy and not contiguous:
                    if maybe_dense_stack:
                        with set_lazy_legacy(True):
                            return _stack(
                                list_of_tensordicts,
                                dim=dim,
                                maybe_dense_stack=maybe_dense_stack,
                            )
                    else:
                        raise RuntimeError(
                            "The sets of keys in the tensordicts to stack are exclusive. "
                            "Consider using `LazyStackedTensorDict.maybe_dense_stack` instead."
                        )
                raise

            if all(
                isinstance(_tensordict, LazyStackedTensorDict)
                for _tensordict in list_of_tensordicts
            ):
                # Let's try to see if all tensors have the same shape
                # If so, we can assume that we can densly stack the sub-tds
                leaves = [tree_leaves(td) for td in list_of_tensordicts]
                for x in _zip_strict(*leaves):
                    # TODO: check what happens with non-tensor data here
                    if len(x) == 1 or all(_x.shape == x[0].shape for _x in x[1:]):
                        continue
                    else:
                        break
                else:
                    # make sure we completed the zip_strict, since strict=True is only available for python >= 3.10
                    if len(leaves) == 1 or all(
                        len(_leaves) == len(leaves[0]) for _leaves in leaves[1:]
                    ):
                        lazy_stack_dim = list_of_tensordicts[0].stack_dim
                        if dim <= lazy_stack_dim:
                            lazy_stack_dim += 1
                        else:
                            dim = dim - 1
                        return LazyStackedTensorDict(
                            *[
                                _stack(
                                    list(subtds),
                                    dim=dim,
                                    maybe_dense_stack=maybe_dense_stack,
                                )
                                for subtds in _zip_strict(
                                    *[td.tensordicts for td in list_of_tensordicts]
                                )
                            ],
                            stack_dim=lazy_stack_dim,
                        )
                lazy_stack_dim = list_of_tensordicts[0].stack_dim
                if dim <= lazy_stack_dim:
                    lazy_stack_dim += 1
                else:
                    dim = dim - 1
                return LazyStackedTensorDict(
                    *[
                        _stack(list_of_td, dim, maybe_dense_stack=maybe_dense_stack)
                        for list_of_td in _zip_strict(
                            *[td.tensordicts for td in list_of_tensordicts]
                        )
                    ],
                    stack_dim=lazy_stack_dim,
                )

            out = {}
            for key in keys:
                out[key] = []
                is_not_init = None
                tensor_shape = None
                is_tensor = None
                for _tensordict in list_of_tensordicts:
                    tensor = _tensordict._get_str(key, default=NO_DEFAULT)
                    if is_tensor is None:
                        tensor_cls = type(tensor)
                        # is_tensor = (
                        #     not _is_tensor_collection(tensor_cls)
                        # ) or is_tensorclass(tensor_cls)
                        # TODO: make sense of this, dynamo cannot pass through stack (and it's unsafe)
                        # only tensors should be tensors
                        is_tensor = not _is_tensor_collection(tensor_cls)
                    if is_not_init is None:
                        is_not_init = isinstance(tensor, UninitializedTensorMixin)
                    if not is_not_init:
                        new_tensor_shape = _shape(tensor)
                        if tensor_shape is not None:
                            if len(new_tensor_shape) != len(tensor_shape) or not all(
                                s1 == s2 and s1 != -1
                                for s1, s2 in _zip_strict(_shape(tensor), tensor_shape)
                            ):
                                # Nested tensors will require a lazy stack
                                if maybe_dense_stack:
                                    with set_lazy_legacy(True):
                                        return _stack(
                                            list_of_tensordicts,
                                            dim=dim,
                                            maybe_dense_stack=maybe_dense_stack,
                                        )
                                else:
                                    raise RuntimeError(
                                        f"The shapes of the tensors to stack is incompatible: {new_tensor_shape} vs {tensor_shape} for key {key}."
                                    )
                        else:
                            tensor_shape = new_tensor_shape

                    out[key].append(tensor)
                out[key] = (out[key], is_not_init, is_tensor)

            def stack_fn(key, values, is_not_init, is_tensor):
                if is_not_init:
                    return _stack_uninit_params(values, dim)
                if is_tensor:
                    return torch.stack(values, dim)
                with (
                    _ErrorInteceptor(
                        key, "Attempted to stack tensors on different devices at key"
                    )
                    if not is_compiling()
                    else contextlib.nullcontext()
                ):
                    return _stack(values, dim, maybe_dense_stack=maybe_dense_stack)

            out = {
                key: stack_fn(key, values, is_not_init, is_tensor)
                for key, (values, is_not_init, is_tensor) in out.items()
            }

            result = TensorDict._new_unsafe(
                out,
                batch_size=LazyStackedTensorDict._compute_batch_size(
                    batch_size, dim, len(list_of_tensordicts)
                ),
                device=device,
            )
            if is_tc:
                return tc_type._from_tensordict(result)
            return result
        else:
            out = LazyStackedTensorDict(
                *list_of_tensordicts,
                stack_dim=dim,
            )
    else:
        keys = _check_keys(list_of_tensordicts)
        batch_size = list(batch_size)
        batch_size.insert(dim, len(list_of_tensordicts))
        batch_size = torch.Size(batch_size)

        if out.batch_size != batch_size:
            raise RuntimeError(
                "out.batch_size and stacked batch size must match, "
                f"got out.batch_size={out.batch_size} and batch_size"
                f"={batch_size}"
            )

        out_keys = set(out.keys())
        if strict:
            in_keys = set(keys)
            if len(out_keys - in_keys) > 0:
                raise RuntimeError(
                    "The output tensordict has keys that are missing in the "
                    "tensordict that has to be written: {out_keys - in_keys}. "
                    "As per the call to `stack(..., strict=True)`, this "
                    "is not permitted."
                )
            elif len(in_keys - out_keys) > 0:
                raise RuntimeError(
                    "The resulting tensordict has keys that are missing in "
                    f"its destination: {in_keys - out_keys}. As per the call "
                    "to `stack(..., strict=True)`, this is not permitted."
                )

        try:
            out._stack_onto_(list_of_tensordicts, dim)
        except KeyError as err:
            raise err
    return out


@implements_for_td(torch.split)
def _split(
    td: TensorDict, split_size_or_sections: int | list[int], dim: int = 0
) -> list[TensorDictBase]:
    return td.split(split_size_or_sections, dim)


@implements_for_td(torch.where)
def where(condition, input, other, *, out=None):
    """Return a ``TensorDict`` of elements selected from either input or other, depending on condition.

    Args:
        condition (BoolTensor): When ``True`` (nonzero), yield ``input``, otherwise yield ``other``.
        input (TensorDictBase or Scalar): value (if ``input`` is a scalar) or values selected at indices where condition is ``True``.
        other (TensorDictBase or Scalar): value (if ``other`` is a scalar) or values selected at indices where condition is ``False``.
        out (Tensor, optional): the output ``TensorDictBase`` instance.

    """
    if isinstance(out, PersistentTensorDict):
        raise RuntimeError(
            "Cannot use a persistent tensordict as output of torch.where."
        )
    return input.where(condition, other, out=out)


def _stack_uninit_params(list_of_params, dim=0, out=None):
    if out is not None:
        raise NotImplementedError
    if dim > 0:
        raise NotImplementedError
    from tensordict.utils import (
        _BatchedUninitializedBuffer,
        _BatchedUninitializedParameter,
    )

    if isinstance(list_of_params[0], UninitializedParameter):
        out = _BatchedUninitializedParameter(
            requires_grad=list_of_params[0].requires_grad,
            device=list_of_params[0].device,
            dtype=list_of_params[0].dtype,
        )
    elif isinstance(list_of_params[0], UninitializedBuffer):
        out = _BatchedUninitializedBuffer(
            requires_grad=list_of_params[0].requires_grad,
            device=list_of_params[0].device,
            dtype=list_of_params[0].dtype,
        )
    out.batch_size = torch.Size([len(list_of_params)])
    return out
