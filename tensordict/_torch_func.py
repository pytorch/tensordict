# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import functools
from functools import partial
from typing import Any, Callable, Sequence, Tuple, TypeVar

import torch
from tensordict._lazy import LazyStackedTensorDict
from tensordict._td import TensorDict
from tensordict._unbatched import _unbatched_with_batch_size
from tensordict.base import (
    _is_leaf_nontensor,
    _is_tensor_collection,
    _TD_HANDLED_FUNCTIONS,
    NO_DEFAULT,
    TensorDictBase,
)
from tensordict.persistent import PersistentTensorDict
from tensordict.utils import (
    _check_keys,
    _ErrorInteceptor,
    _is_tensorclass,
    _is_unbatched,
    _jagged_cat,
    _pass_through,
    _shape,
    _warn_dim_past_batch_dims,
    _zip_strict,
    DeviceType,
    is_non_tensor,
    is_tensorclass,
    lazy_legacy,
)
from torch import Tensor
from torch.compiler import is_compiling
from torch.nn.parameter import (
    UninitializedBuffer,
    UninitializedParameter,
    UninitializedTensorMixin,
)
from torch.utils._pytree import tree_leaves

# the dict that TensorDictBase.__torch_function__ reads
TD_HANDLED_FUNCTIONS: dict[Callable, Callable] = _TD_HANDLED_FUNCTIONS
LAZY_TD_HANDLED_FUNCTIONS: dict[Callable, Callable] = {}
T = TypeVar("T", bound="TensorDictBase")


def implements_for_td(torch_function: Callable) -> Callable[[Callable], Callable]:
    """Registers the decorated function as the implementation of ``torch_function`` for tensordicts."""

    @functools.wraps(torch_function)
    def decorator(func: Callable) -> Callable:
        TD_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


def implements_for_lazy_td(torch_function: Callable) -> Callable[[Callable], Callable]:
    """Registers the decorated function as the implementation of ``torch_function`` for lazy stacks."""

    @functools.wraps(torch_function)
    def decorator(func: Callable) -> Callable:
        LAZY_TD_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements_for_td(torch.unbind)
def _unbind(td: T, *args: Any, **kwargs: Any) -> tuple[T, ...]:
    return td.unbind(*args, **kwargs)


@implements_for_td(torch.unflatten)
def _unflatten(td: T, *args: Any, **kwargs: Any) -> tuple[T, ...]:
    return td.unflatten(*args, **kwargs)


@implements_for_td(torch.flatten)
def _flatten(td: T, *args: Any, **kwargs: Any) -> tuple[T, ...]:
    return td.flatten(*args, **kwargs)


@implements_for_td(torch.reshape)
def _reshape(td: T, shape: Sequence[int]) -> T:
    return td.reshape(shape)


@implements_for_td(torch.flip)
def _flip(td: T, dims: Sequence[int]) -> T:
    return td.flip(dims)


@implements_for_td(torch.fliplr)
def _fliplr(td: T) -> T:
    return td.fliplr()


@implements_for_td(torch.flipud)
def _flipud(td: T) -> T:
    return td.flipud()


@implements_for_td(torch.roll)
def _roll(
    td: T, shifts: int | Sequence[int], dims: int | Sequence[int] | None = None
) -> T:
    return td.roll(shifts, dims)


@implements_for_td(torch.rot90)
def _rot90(td: T, k: int = 1, dims: Sequence[int] = (0, 1)) -> T:
    return td.rot90(k, dims)


@implements_for_td(torch.narrow)
def _narrow(td: T, dim: int, start: int, length: int) -> T:
    return td.narrow(dim, start, length)


@implements_for_td(torch.tile)
def _tile(td: T, dims: Sequence[int]) -> T:
    return td.tile(dims)


@implements_for_td(torch.repeat_interleave)
def _repeat_interleave(
    td: T,
    repeats: Tensor | int,
    dim: int | None = None,
    *,
    output_size: int | None = None,
) -> T:
    return td.repeat_interleave(repeats, dim, output_size=output_size)


@implements_for_td(torch.broadcast_to)
def _broadcast_to(td: T, shape: Sequence[int]) -> T:
    return td.broadcast_to(shape)


@implements_for_td(torch.atleast_1d)
def _atleast_1d(td: T) -> T:
    return td.atleast_1d()


@implements_for_td(torch.atleast_2d)
def _atleast_2d(td: T) -> T:
    return td.atleast_2d()


@implements_for_td(torch.atleast_3d)
def _atleast_3d(td: T) -> T:
    return td.atleast_3d()


@implements_for_td(torch.transpose)
def _transpose(td: T, *args: Any, **kwargs: Any) -> tuple[T, ...]:
    return td.transpose(*args, **kwargs)


@implements_for_td(torch.swapaxes)
def _swapaxes(td: T, axis0: int, axis1: int) -> T:
    return td.swapaxes(axis0, axis1)


@implements_for_td(torch.swapdims)
def _swapdims(td: T, dim0: int, dim1: int) -> T:
    return td.swapdims(dim0, dim1)


def _is_non_tensor_stack(value) -> bool:
    return isinstance(value, LazyStackedTensorDict) and is_non_tensor(value)


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

    # as torch.gather does, keep the size of the index on the batch dims where it
    # is smaller than the input; every leaf starts with the batch size
    target_batch_shape = list(map(min, index.shape, input.batch_size))

    def _gather_tensor(tensor, dest_container=None, dest_key=None):
        if dest_container is not None:
            dest = dest_container._get_str(dest_key, default=NO_DEFAULT)
        else:
            dest = None
        index_expand = index
        while index_expand.ndim < tensor.ndim:
            index_expand = index_expand.unsqueeze(-1)
        target_shape = list(tensor.shape)
        target_shape[: len(target_batch_shape)] = target_batch_shape
        target_shape[dim] = index_expand.shape[dim]
        index_expand = index_expand.expand(target_shape)
        out = torch.gather(tensor, dim, index_expand, out=dest)
        return out

    def _gather_non_tensor_stack(stack):
        # A non-tensor stack has no tensor leaves to gather. Gather the position of
        # each entry instead, then select the entries at those positions, which
        # copies the values themselves. Only expand the trailing dims the index
        # lacks, so that a bare stack follows torch.gather.
        positions = stack._positions().to(index.device)
        index_expand = index
        while index_expand.ndim < positions.ndim:
            index_expand = index_expand.unsqueeze(-1)
        target_shape = list(index.shape) + list(positions.shape[index.ndim :])
        index_expand = index_expand.expand(target_shape)
        return stack._select_positions(torch.gather(positions, dim, index_expand))

    def _process_gather_value(value):
        if _is_unbatched(value):
            return _unbatched_with_batch_size(value, index.shape)
        if _is_non_tensor_stack(value):
            return _gather_non_tensor_stack(value)
        return _gather_tensor(value)

    if _is_non_tensor_stack(input):
        result = _gather_non_tensor_stack(input)
        if out is None:
            return result
        out.update_(result)
        return out

    if out is None:
        if len(index.shape) == input.ndim:
            names = input._maybe_names()
        else:
            names = None
        device = input.device
        return type(input)._new_unsafe(
            {
                key: _process_gather_value(value)
                for key, value in input.items(is_leaf=_is_leaf_nontensor)
            },
            batch_size=index.shape,
            names=names,
            device=device,
        )
    for key, value in input.items(is_leaf=_is_leaf_nontensor):
        if _is_unbatched(value):
            out._set_str(
                key,
                _unbatched_with_batch_size(value, out.batch_size),
                validated=True,
                inplace=False,
                non_blocking=False,
            )
        elif is_non_tensor(value):
            gathered = _process_gather_value(value)
            dest = out._get_str(key, default=None)
            if _is_non_tensor_stack(dest) and dest.batch_size == gathered.batch_size:
                # write into the destination stack, as tensors are written into
                # the destination storage
                dest.update_(gathered)
            else:
                out._set_str(
                    key,
                    gathered,
                    validated=True,
                    inplace=False,
                    non_blocking=False,
                )
        else:
            _gather_tensor(value, out, key)
    return out


@implements_for_td(torch.full_like)
def _full_like(td: T, fill_value: float, *args, **kwargs: Any) -> T:
    dtype = kwargs.pop("dtype", None)
    device_nd = kwargs.pop("device", NO_DEFAULT)
    if device_nd is NO_DEFAULT:
        device = None
    else:
        device = device_nd

    def full_like(x):
        return torch.full_like(
            x, fill_value=fill_value, device=device, dtype=dtype, *args, **kwargs
        )

    td_clone = td._fast_apply(
        full_like,
        propagate_lock=True,
        device=device_nd,
    )
    return td_clone


def _like(torch_function: Callable) -> Callable:
    def _like_td(td: T, *args: Any, **kwargs: Any) -> T:
        dtype = kwargs.pop("dtype", None)
        device_nd = kwargs.pop("device", NO_DEFAULT)
        if device_nd is NO_DEFAULT:
            device = None
        else:
            device = device_nd

        def like(x):
            return torch_function(x, device=device, dtype=dtype, *args, **kwargs)

        td_clone = td._fast_apply(
            like,
            propagate_lock=True,
            device=device_nd,
        )
        return td_clone

    _like_td.__name__ = f"_{torch_function.__name__}"
    return _like_td


for _like_function in (
    torch.zeros_like,
    torch.ones_like,
    torch.rand_like,
    torch.randn_like,
    torch.empty_like,
):
    implements_for_td(_like_function)(_like(_like_function))


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


@implements_for_td(torch.movedim)
def _movedim(td: T, source: int | Sequence[int], destination: int | Sequence[int]) -> T:
    return td.movedim(source, destination)


@implements_for_td(torch.moveaxis)
def _moveaxis(
    td: T, source: int | Sequence[int], destination: int | Sequence[int]
) -> T:
    return td.moveaxis(source, destination)


@implements_for_td(torch.cat)
def _cat(
    list_of_tensordicts: Sequence[T],
    dim: int = 0,
    device: DeviceType | None = None,
    out: T | None = None,
) -> T:
    if not len(list_of_tensordicts):
        raise RuntimeError("list_of_tensordicts cannot be empty")
    if any(isinstance(td, LazyStackedTensorDict) for td in list_of_tensordicts) and (
        out is None or isinstance(out, LazyStackedTensorDict)
    ):
        return _lazy_cat(list_of_tensordicts, dim=dim, out=out)

    batch_size = list(list_of_tensordicts[0].batch_size)
    tdtype = type(list_of_tensordicts[0])
    if dim < 0:
        dim = len(batch_size) + dim
    if dim >= len(batch_size):
        raise RuntimeError(
            f"dim must be in the range 0 <= dim < len(batch_size), got dim"
            f"={dim} and batch_size={batch_size}"
        )
    batch_size[dim] = sum([td.batch_size[dim] for td in list_of_tensordicts])
    if dim < 0:
        _warn_dim_past_batch_dims("cat")
    batch_size = TensorDict._parse_batch_size(None, batch_size)

    # check that all tensordict match
    keys = _check_keys(list_of_tensordicts, strict=True)
    if out is None:
        out = {}
        for key in keys:
            items = [td._get_str(key, NO_DEFAULT) for td in list_of_tensordicts]
            if _is_unbatched(items[0]):
                out[key] = _unbatched_with_batch_size(items[0], batch_size)
            elif not is_compiling():
                with _ErrorInteceptor(
                    key, "Attempted to concatenate tensors on different devices at key"
                ):
                    try:
                        out[key] = torch.cat(items, dim)
                    except RuntimeError:
                        # torch does not concatenate jagged nested tensors
                        # along their batch dim
                        result = _jagged_cat(items) if dim == 0 else None
                        if result is None:
                            raise
                        out[key] = result
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
        # if we have a TD subclass, use _new_unsafe bc we know it exists. Otherwise, use
        #  TensorDict's one
        if (
            issubclass(tdtype, TensorDict)
            or _is_tensorclass(tdtype)
            or hasattr(tdtype, "_wrap_td")
        ):
            clz = tdtype
        else:
            clz = TensorDict
        return clz._new_unsafe(out, device=device, batch_size=batch_size, names=names)
    else:
        if out.batch_size != batch_size:
            raise RuntimeError(
                "out.batch_size and cat batch size must match, "
                f"got out.batch_size={out.batch_size} and batch_size"
                f"={batch_size}"
            )

        for key in keys:
            first_item = list_of_tensordicts[0]._get_str(key, NO_DEFAULT)
            if _is_unbatched(first_item):
                out._set_str(
                    key,
                    _unbatched_with_batch_size(first_item, batch_size),
                    validated=True,
                    inplace=False,
                    non_blocking=False,
                )
            else:
                with (
                    _ErrorInteceptor(
                        key,
                        "Attempted to concatenate tensors on different devices at key",
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
                            key,
                            torch.cat([td.get(key) for td in list_of_tensordicts], dim),
                        )
        return out


@implements_for_lazy_td(torch.cat)
def _lazy_cat(
    list_of_tensordicts: Sequence[TensorDictBase],
    dim: int = 0,
    out: LazyStackedTensorDict | None = None,
) -> LazyStackedTensorDict:
    # why aren't they feeding you?
    if not len(list_of_tensordicts):
        raise RuntimeError("list_of_tensordicts cannot be empty")

    if out is not None and not isinstance(out, LazyStackedTensorDict):
        return _cat(list_of_tensordicts, dim=dim, out=out)

    lazy_tds = [
        td for td in list_of_tensordicts if isinstance(td, LazyStackedTensorDict)
    ]
    stack_dim = lazy_tds[0].stack_dim
    if any((td.stack_dim != stack_dim) for td in lazy_tds):
        raise RuntimeError("cat lazy stacked tds must have same stack dim")

    if len(lazy_tds) != len(list_of_tensordicts):
        list_of_tensordicts = [
            (
                td
                if isinstance(td, LazyStackedTensorDict)
                else _to_lazystack_for_lazy_cat(td, stack_dim)
            )
            for td in list_of_tensordicts
        ]

    batch_size = list(list_of_tensordicts[0].batch_size)
    if dim < 0:
        dim = len(batch_size) + dim
    if dim >= len(batch_size):
        raise RuntimeError(
            f"dim must be in the range 0 <= dim < len(batch_size), got dim"
            f"={dim} and batch_size={batch_size}"
        )
    if dim != stack_dim:
        num_stacked_tds = len(list_of_tensordicts[0].tensordicts)
        if any(len(td.tensordicts) != num_stacked_tds for td in list_of_tensordicts):
            raise RuntimeError(
                "cat lazy stacked tds along a non-stack dimension requires "
                "all inputs to have the same number of stacked tensordicts"
            )

    batch_size[dim] = sum(td.batch_size[dim] for td in list_of_tensordicts)
    if dim < 0:
        _warn_dim_past_batch_dims("cat")
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
        if not out:
            return _empty_lazystack_for_lazy_cat(
                list_of_tensordicts[0], batch_size, stack_dim
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
            # the members of out lack the stack dim, so later dims shift by one
            sub_dim = dim - 1 if dim > out.stack_dim else dim
            for i, sub_dest in enumerate(out.tensordicts):
                index = index_base + (i,)
                tds_to_cat = [_td[index] for _td in list_of_tensordicts]
                torch.cat(tds_to_cat, sub_dim, out=sub_dest)
        else:
            init_idx = 0
            for td_in in list_of_tensordicts:
                if td_in.shape[dim] == 0:
                    continue
                sub_dest = out.tensordicts[init_idx : init_idx + td_in.shape[dim]]
                init_idx += td_in.shape[dim]
                # write into the members of out (a dense stack would copy them)
                LazyStackedTensorDict.lazy_stack(sub_dest, out.stack_dim).update(
                    td_in, inplace=True
                )

        return out


def _to_lazystack_for_lazy_cat(
    td: TensorDictBase,
    stack_dim: int,
) -> LazyStackedTensorDict:
    if stack_dim >= td.batch_dims:
        raise RuntimeError(
            "mixed lazy/plain cat requires non-lazy TensorDict inputs to have "
            "the lazy stack dimension in their batch size, got "
            f"stack_dim={stack_dim} and batch_size={td.batch_size}"
        )
    if td.batch_size[stack_dim] == 0:
        return _empty_lazystack_for_lazy_cat(td, td.batch_size, stack_dim)
    return td.to_lazystack(stack_dim)


def _empty_lazystack_for_lazy_cat(
    td: TensorDictBase,
    batch_size: torch.Size,
    stack_dim: int,
) -> LazyStackedTensorDict:
    batch_size = torch.Size(
        tuple(batch_size[:stack_dim]) + tuple(batch_size[stack_dim + 1 :])
    )
    try:
        names = td._maybe_names()
    except IndexError:
        names = None
    if names is not None:
        names = names[:stack_dim] + names[stack_dim + 1 :]
        if not any(name is not None for name in names):
            names = None
    clz = type(td) if isinstance(td, LazyStackedTensorDict) else LazyStackedTensorDict
    return clz._new_lazy_unsafe(
        stack_dim=stack_dim,
        device=td.device,
        names=names,
        batch_size=batch_size,
    )


def _lazy_stack_result(list_of_tensordicts, dim, clz, is_tc):
    """Return the lazy stack that torch.stack gives when it does not stack densely."""
    out = LazyStackedTensorDict(*list_of_tensordicts, stack_dim=dim)
    if is_tc and not is_tensorclass(out):
        return clz._from_tensordict(out)
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
    td_types = [type(td) for td in list_of_tensordicts]
    is_tc = any(_is_tensorclass(td_type) for td_type in set(td_types))
    if all(_pass_through(td) for td in list_of_tensordicts):
        return type(list_of_tensordicts[0])._stack_non_tensor(
            list_of_tensordicts, dim=dim
        )
    list_of_tensordicts_orig = list_of_tensordicts
    if is_tc:
        list_of_tensordicts = [tc._tensordict for tc in list_of_tensordicts]
        clz = type(list_of_tensordicts_orig[0])
    elif issubclass(td_types[0], TensorDict) or hasattr(td_types[0], "_wrap_td"):
        clz = td_types[0]
    else:
        clz = TensorDict

    batch_size = list_of_tensordicts[0].batch_size
    if dim < 0:
        dim = len(batch_size) + dim + 1
    if not 0 <= dim <= len(batch_size):
        _warn_dim_past_batch_dims("stack")

    # check that all tensordict match
    # Read lazy_legacy
    _lazy_legacy = lazy_legacy()

    if not _lazy_legacy:
        for i, td in enumerate(list_of_tensordicts[1:]):
            if td.batch_size != list_of_tensordicts[0].batch_size:
                if not maybe_dense_stack:
                    raise RuntimeError(
                        "stacking tensordicts requires them to have congruent batch sizes, "
                        f"got td[{i + 1}].batch_size={td.batch_size} and td[0].batch_size="
                        f"{list_of_tensordicts[0].batch_size}"
                    )
                elif td.batch_dims == list_of_tensordicts[0].batch_dims:
                    from tensordict import lazy_stack

                    return lazy_stack(
                        list_of_tensordicts_orig,
                        dim=dim,
                        out=out,
                    )
                else:
                    raise RuntimeError(
                        "Lazy stacking of tensordicts requires them to have congruent batch dimensions, "
                        f"got td[{i + 1}].batch_dims={td.batch_dims} and td[0].batch_dims="
                        f"{list_of_tensordicts[0].batch_dims}"
                    )

    if out is None:
        # We need to handle tensordicts with exclusive keys and tensordicts with
        # mismatching shapes.
        # The first case is handled within _check_keys which fails if keys
        # don't match exactly.
        # The second requires a check over the tensor shapes.
        device = list_of_tensordicts[0].device
        if contiguous or not _lazy_legacy:
            # perf: exact TensorDicts compare the key views of their dicts and
            # read their dense leaves from them in the loop below
            dicts = None
            if all(type(td) is TensorDict for td in list_of_tensordicts):
                dicts = [td._tensordict for td in list_of_tensordicts]
            try:
                if dicts is not None and all(
                    d.keys() == dicts[0].keys() for d in dicts[1:]
                ):
                    keys = list(dicts[0])
                else:
                    keys = _check_keys(list_of_tensordicts, strict=True)
            except KeyError:
                if not _lazy_legacy and not contiguous:
                    if maybe_dense_stack:
                        return _lazy_stack_result(list_of_tensordicts, dim, clz, is_tc)
                    else:
                        raise RuntimeError(
                            "The sets of keys in the tensordicts to stack are exclusive. "
                            "Consider using `LazyStackedTensorDict.maybe_dense_stack` instead."
                        )
                raise

            if all(isinstance(td, LazyStackedTensorDict) for td in list_of_tensordicts):
                # Let's try to see if all tensors have the same shape
                # If so, we can assume that we can densly stack the sub-tds
                leaves = [tree_leaves(td) for td in list_of_tensordicts]
                unique_leaves_len = len({len(leaf) for leaf in leaves}) == 1
                unique_leaves_len &= (
                    len({len(td.tensordicts) for td in list_of_tensordicts}) == 1
                )
                if unique_leaves_len:
                    # If the number of sub-tensordicts is the same, we can stack them internally
                    # and lazy stack them on the outside.
                    lazy_stack_dim = list_of_tensordicts[0].stack_dim
                    if dim <= lazy_stack_dim:
                        lazy_stack_dim += 1
                    else:
                        dim = dim - 1
                    result = LazyStackedTensorDict(
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
                    if is_tc:
                        return clz._from_tensordict(result)
                    return result
                else:
                    from tensordict import lazy_stack

                    return lazy_stack(list_of_tensordicts_orig, dim=dim, out=out)

            out = {}
            for key in keys:
                if dicts is not None:
                    # The loop below gives the same entry when every value
                    # is a dense tensor of one shape; it handles any other key.
                    values = [d[key] for d in dicts]
                    v0 = values[0]
                    if type(v0) is Tensor and not v0.is_nested:
                        shape = v0.shape
                        for v in values[1:]:
                            if type(v) is not Tensor or v.is_nested or v.shape != shape:
                                break
                        else:
                            out[key] = (values, False, True)
                            continue
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
                            # _shape gives -1 for the ragged dims of nested tensors
                            if (
                                new_tensor_shape != tensor_shape
                                or -1 in new_tensor_shape
                            ):
                                # Nested tensors will require a lazy stack
                                if maybe_dense_stack:
                                    return _lazy_stack_result(
                                        list_of_tensordicts, dim, clz, is_tc
                                    )
                                else:
                                    raise RuntimeError(
                                        f"The shapes of the tensors to stack is incompatible: {new_tensor_shape} vs {tensor_shape} for key {key}."
                                    )
                        else:
                            tensor_shape = new_tensor_shape

                    out[key].append(tensor)
                out[key] = (out[key], is_not_init, is_tensor)

            result_batch_size = LazyStackedTensorDict._compute_batch_size(
                batch_size, dim, len(list_of_tensordicts)
            )

            def stack_fn(key, values, is_not_init, is_tensor):
                if is_not_init:
                    return _stack_uninit_params(values, dim)
                if is_tensor:
                    if _is_unbatched(values[0]):
                        return _unbatched_with_batch_size(
                            type(values[0])._stack_non_tensor(values, dim),
                            result_batch_size,
                        )
                    try:
                        return torch.stack(values, dim)
                    except RuntimeError as err:
                        if (
                            dim == 0
                            and getattr(values[0], "layout", None) is torch.jagged
                        ):
                            raise RuntimeError(
                                f"Cannot stack the tensordicts: the entry {key!r} is a "
                                "jagged nested tensor, and stacking jagged nested "
                                "tensors along their batch dim would need a second "
                                "ragged dim, which torch.jagged does not have. Use "
                                "torch.cat, or tensordict.lazy_stack to keep the "
                                "tensordicts apart."
                            ) from err
                        raise
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
            # Add names if all tensordicts have the same `names`
            names = list_of_tensordicts[0]._maybe_names()
            if names is not None:
                if all(td._maybe_names() == names for td in list_of_tensordicts[1:]):
                    names.insert(dim, None)
                else:
                    names = None

            result = clz._new_unsafe(
                out,
                batch_size=result_batch_size,
                device=device,
                names=names,
            )
            if is_tc and not is_tensorclass(result):
                return clz._from_tensordict(result)
            return result
        else:
            return _lazy_stack_result(list_of_tensordicts, dim, clz, is_tc)
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


@implements_for_td(torch.tensor_split)
def _tensor_split(
    td: T,
    indices_or_sections: int | Sequence[int] | Tensor | None = None,
    /,
    dim: int = 0,
    *,
    sections: int | None = None,
    indices: Sequence[int] | None = None,
    tensor_indices_or_sections: Tensor | None = None,
) -> tuple[T, ...]:
    # torch.tensor_split names this argument after its type: sections (int),
    # indices (sequence of ints) or tensor_indices_or_sections (tensor)
    if indices_or_sections is None:
        if sections is not None:
            indices_or_sections = sections
        elif indices is not None:
            indices_or_sections = indices
        else:
            indices_or_sections = tensor_indices_or_sections
    return td.tensor_split(indices_or_sections, dim)


@implements_for_td(torch.chunk)
def _chunk(td: T, chunks: int, dim: int = 0) -> tuple[T, ...]:
    return td.chunk(chunks, dim)


@implements_for_td(torch.where)
def where(condition, input, other, *, out=None):
    """Implements ``torch.where(condition, input, other)`` for tensordicts.

    Selects the elements of ``input`` where ``condition`` is ``True`` and those
    of ``other`` elsewhere. One of ``input`` and ``other`` can be a scalar or
    a tensor. See :meth:`~tensordict.TensorDictBase.where`.

    Args:
        condition (torch.Tensor): a boolean mask.
        input (TensorDictBase, torch.Tensor or scalar): the values where
            ``condition`` is ``True``.
        other (TensorDictBase, torch.Tensor or scalar): the values where
            ``condition`` is ``False``.

    Keyword Args:
        out (TensorDictBase, optional): a tensordict to write the result
            into. A :class:`~tensordict.PersistentTensorDict` raises a
            ``RuntimeError``. Defaults to ``None``.
    """
    if isinstance(out, PersistentTensorDict):
        raise RuntimeError(
            "Cannot use a persistent tensordict as output of torch.where."
        )
    if not _is_tensor_collection(type(input)) and _is_tensor_collection(type(other)):
        # torch.where(condition, value, td): pick from td where condition is False.
        # logical_not, not ~, so that a uint8 condition is negated correctly.
        return other.where(condition.logical_not(), input, out=out)
    return input.where(condition, other, out=out)


def _stack_uninit_params(list_of_params, dim: int = 0, out=None):
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


@implements_for_td(torch.var)
def _var(td: T, *args: Any, **kwargs: Any) -> T:
    return td.var(*args, **kwargs)


@implements_for_td(torch.std)
def _std(td: T, *args: Any, **kwargs: Any) -> T:
    return td.std(*args, **kwargs)


@implements_for_td(torch.mean)
def _mean(td: T, *args: Any, **kwargs: Any) -> T:
    return td.mean(*args, **kwargs)


@implements_for_td(torch.sum)
def _sum(td: T, *args: Any, **kwargs: Any) -> T:
    return td.sum(*args, **kwargs)


@implements_for_td(torch.maximum)
def _maximum(td: T, other: T, *args: Any, **kwargs: Any) -> T:
    return td.maximum(other, *args, **kwargs)


def _call_method(name: str) -> Callable:
    def _method(td: T, *args: Any, **kwargs: Any) -> T:
        if not _is_tensor_collection(type(td)):
            # torch.add(tensor, td): a tensor has no method that takes a tensordict
            raise TypeError(
                f"torch.{name} takes a tensordict as its first argument, "
                f"got {type(td).__name__}."
            )
        return getattr(td, name)(*args, **kwargs)

    _method.__name__ = f"_{name}"
    return _method


# Pointwise functions whose TensorDict method has the arguments and results of
# the torch function. Reductions such as torch.max and torch.all are left out:
# their methods return other results.
_POINTWISE_FUNCTIONS = (
    "abs",
    "abs_",
    "acos",
    "acos_",
    "add",
    "addcdiv",
    "addcmul",
    "asin",
    "asin_",
    "atan",
    "atan_",
    "bitwise_and",
    "ceil",
    "ceil_",
    "clamp",
    "clamp_max",
    "clamp_max_",
    "clamp_min",
    "clamp_min_",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "div",
    "erf",
    "erf_",
    "erfc",
    "erfc_",
    "exp",
    "exp_",
    "expm1",
    "expm1_",
    "floor",
    "floor_",
    "frac",
    "frac_",
    "isfinite",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "lerp",
    "lgamma",
    "log",
    "log10",
    "log10_",
    "log1p",
    "log1p_",
    "log2",
    "log2_",
    "log_",
    "logical_and",
    "minimum",
    "mul",
    "neg",
    "neg_",
    "pow",
    "reciprocal",
    "reciprocal_",
    "round",
    "round_",
    "rsub",
    "sigmoid",
    "sigmoid_",
    "sign",
    "sin",
    "sin_",
    "sinh",
    "sinh_",
    "sqrt",
    "sqrt_",
    "sub",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "trunc",
    "trunc_",
)
for _name in _POINTWISE_FUNCTIONS:
    implements_for_td(getattr(torch, _name))(_call_method(_name))


def _loss(loss_function: Callable) -> Callable:
    def _loss_td(input: T, target: T, *args: Any, **kwargs: Any) -> T:
        reduction = kwargs.pop("reduction", "mean")
        if reduction == "none":
            batch_size = None
        elif reduction == "mean":
            batch_size = ()
        elif reduction == "sum":
            batch_size = ()
        else:
            raise ValueError(
                f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', 'sum'."
            )
        # torch also passes size_average, reduce (and beta), which apply ignored:
        # keep ignoring them, without apply's warning about unknown keywords
        return input.apply(
            partial(loss_function, reduction=reduction),
            target,
            *args,
            batch_size=batch_size,
        )

    _loss_td.__name__ = f"_{loss_function.__name__}"
    return _loss_td


for _loss_function in (
    torch.nn.functional.l1_loss,
    torch.nn.functional.mse_loss,
    torch.nn.functional.smooth_l1_loss,
):
    implements_for_td(_loss_function)(_loss(_loss_function))


@implements_for_td(torch.autograd.grad)
def _grad(
    outputs: TensorDictBase | Tuple[TensorDictBase, ...],
    inputs: TensorDictBase | Tuple[TensorDictBase, ...],
    grad_outputs: TensorDictBase | Tuple[TensorDictBase, ...] | None = None,
    **kwargs: Any,
) -> TensorDict:
    from tensordict.base import _NESTED_TENSORS_AS_LISTS

    if isinstance(outputs, tuple) and len(outputs) > 1:
        raise ValueError(
            "torch.autograd.grad for TensorDict only supports a single output"
        )
    elif isinstance(outputs, tuple):
        outputs = outputs[0]
    if not isinstance(outputs, TensorDictBase):
        raise ValueError(
            "torch.autograd.grad for TensorDict only supports TensorDictBase as output"
        )

    if isinstance(inputs, tuple) and len(inputs) > 1:
        raise ValueError(
            "torch.autograd.grad for TensorDict only supports a single input"
        )
    elif isinstance(inputs, tuple):
        inputs = inputs[0]
    if not isinstance(inputs, TensorDictBase):
        raise ValueError(
            "torch.autograd.grad for TensorDict only supports TensorDictBase as input"
        )

    if (
        grad_outputs is not None
        and isinstance(grad_outputs, tuple)
        and len(grad_outputs) > 1
    ):
        raise ValueError(
            "torch.autograd.grad for TensorDict only supports a single grad_output"
        )
    elif isinstance(grad_outputs, tuple):
        grad_outputs = grad_outputs[0]
    if grad_outputs is not None and not isinstance(grad_outputs, TensorDictBase):
        raise ValueError(
            "torch.autograd.grad for TensorDict only supports TensorDictBase as grad_output"
        )

    if grad_outputs is not None:
        tup_grad_outputs = tuple(
            grad_outputs._values_list(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)
        )
    else:
        tup_grad_outputs = None

    tup_outputs = tuple(
        outputs._values_list(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)
    )

    keys, all_inputs_raw = inputs._items_list(
        True, True, is_leaf=_NESTED_TENSORS_AS_LISTS
    )

    all_grads = torch.autograd.grad(
        tup_outputs, all_inputs_raw, tup_grad_outputs, **kwargs
    )

    pairs = {}
    for key, grad in _zip_strict(keys, all_grads):
        pairs[key] = grad

    def pop(name, val):
        return pairs.pop(name, None)

    # rebuild the tensordict
    return inputs._fast_apply(
        pop,
        named=True,
        nested_keys=True,
        propagate_lock=True,
    )
