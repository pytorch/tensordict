# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools

import warnings
from typing import Any, Callable, Sequence, TypeVar

import torch
from tensordict._lazy import LazyStackedTensorDict
from tensordict._td import TensorDict
from tensordict.base import _is_leaf_nontensor, NO_DEFAULT, TensorDictBase
from tensordict.persistent import PersistentTensorDict
from tensordict.utils import (
    _check_keys,
    _ErrorInteceptor,
    DeviceType,
    is_non_tensor,
    lazy_legacy,
    set_lazy_legacy,
)
from torch import Tensor
from torch.nn.parameter import (
    UninitializedBuffer,
    UninitializedParameter,
    UninitializedTensorMixin,
)

TD_HANDLED_FUNCTIONS: dict[Callable, Callable] = {}
LAZY_TD_HANDLED_FUNCTIONS: dict[Callable, Callable] = {}
T = TypeVar("T", bound="TensorDictBase")


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

    def _gather_tensor(tensor, dest=None):
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
        return TensorDict(
            {
                key: _gather_tensor(value)
                for key, value in input.items(is_leaf=_is_leaf_nontensor)
            },
            batch_size=index.shape,
            names=names,
        )
    TensorDict(
        {
            key: _gather_tensor(value, out.get(key))
            for key, value in input.items(is_leaf=_is_leaf_nontensor)
        },
        batch_size=index.shape,
    )
    return out


@implements_for_td(torch.full_like)
def _full_like(td: T, fill_value: float, **kwargs: Any) -> T:
    td_clone = td.clone()
    for key in td_clone.keys():
        td_clone.fill_(key, fill_value)
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


@implements_for_td(torch.zeros_like)
def _zeros_like(td: T, **kwargs: Any) -> T:
    td_clone = td._fast_apply(torch.zeros_like)
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
def _ones_like(td: T, **kwargs: Any) -> T:
    td_clone = td._fast_apply(lambda x: torch.ones_like(x))
    if "device" in kwargs:
        td_clone = td_clone.to(kwargs.pop("device"))
    if len(kwargs):
        raise RuntimeError(
            f"keyword arguments {list(kwargs.keys())} are not "
            f"supported with full_like with TensorDict"
        )
    return td_clone


@implements_for_td(torch.rand_like)
def _rand_like(td: T, **kwargs: Any) -> T:
    td_clone = td._fast_apply(lambda x: torch.rand_like(x))
    if "device" in kwargs:
        td_clone = td_clone.to(kwargs.pop("device"))
    if len(kwargs):
        raise RuntimeError(
            f"keyword arguments {list(kwargs.keys())} are not "
            f"supported with full_like with TensorDict"
        )
    return td_clone


@implements_for_td(torch.randn_like)
def _randn_like(td: T, **kwargs: Any) -> T:
    td_clone = td._fast_apply(lambda x: torch.randn_like(x))
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
    try:
        tdclone = td.clone()
    except Exception as err:
        raise RuntimeError(
            "The tensordict passed to torch.empty_like cannot be "
            "cloned, preventing empty_like to be called. "
            "Consider calling tensordict.to_tensordict() first."
        ) from err
    return tdclone._fast_apply(
        lambda x: torch.empty_like(x, *args, **kwargs), inplace=True
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
    if not list_of_tensordicts:
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
    batch_size = torch.Size(batch_size)

    # check that all tensordict match
    keys = _check_keys(list_of_tensordicts, strict=True)
    if out is None:
        out = {}
        for key in keys:
            with _ErrorInteceptor(
                key, "Attempted to concatenate tensors on different devices at key"
            ):
                out[key] = torch.cat(
                    [td._get_str(key, NO_DEFAULT) for td in list_of_tensordicts], dim
                )
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
        return TensorDict(
            out, device=device, batch_size=batch_size, _run_checks=False, names=names
        )
    else:
        if out.batch_size != batch_size:
            raise RuntimeError(
                "out.batch_size and cat batch size must match, "
                f"got out.batch_size={out.batch_size} and batch_size"
                f"={batch_size}"
            )

        for key in keys:
            with _ErrorInteceptor(
                key, "Attempted to concatenate tensors on different devices at key"
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
    if not list_of_tensordicts:
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
                out += lazy_td.tensordicts
        else:
            for i in range(len(list_of_tensordicts[0].tensordicts)):
                out.append(
                    torch.cat(
                        [lazy_td.tensordicts[i] for lazy_td in list_of_tensordicts],
                        new_dim,
                    )
                )
        return LazyStackedTensorDict(*out, stack_dim=stack_dim)
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
                torch.stack(sub_dest, out.stack_dim).update(td_in, inplace=True)

        return out


@implements_for_td(torch.stack)
def _stack(
    list_of_tensordicts: Sequence[TensorDictBase],
    dim: int = 0,
    device: DeviceType | None = None,
    out: T | None = None,
    strict: bool = False,
    contiguous: bool = False,
) -> T:
    if not list_of_tensordicts:
        raise RuntimeError("list_of_tensordicts cannot be empty")

    if all(is_non_tensor(td) for td in list_of_tensordicts):
        from tensordict.tensorclass import NonTensorData

        return NonTensorData._stack_non_tensor(list_of_tensordicts, dim=dim)

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
    _lazy_legacy = lazy_legacy(allow_none=True)
    if _lazy_legacy is None:
        warnings.warn(
            """You did not define if torch.stack was to return a dense or lazy
stack of tensordicts. Up until v0.3 included, a lazy stack was returned.
From v0.4 onward, a dense stack will be returned and to build a
lazy stack, an explicit call to LazyStackedTensorDict.lazy_stack will be required.
To silence this warning, choose one of the following options:
- set the LAZY_LEGACY_OP environment variable to 'False' (recommended, default) or 'True' depending on
  the behaviour you want to use. Another way to achieve this is to call
  `tensordict.set_lazy_legacy(False).set()` at the beginning of your script.
- set the decorator/context manager `tensordict.set_lazy_legacy(False)` (recommended) around
  the function or code block where stack is used.
- Use `LazyStackedTensorDict.lazy_stack()` if it is a lazy stack that you wish to use.""",
            category=DeprecationWarning,
        )
        # get default
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
                    with set_lazy_legacy(True):
                        return _stack(list_of_tensordicts, dim=dim)
                raise

            out = {}
            for key in keys:
                out[key] = []
                is_lazy = False
                tensor_shape = None
                for _tensordict in list_of_tensordicts:
                    tensor = _tensordict._get_str(key, default=NO_DEFAULT)
                    if isinstance(tensor, UninitializedTensorMixin):
                        is_lazy = True
                    elif tensor_shape is None:
                        tensor_shape = tensor.shape
                    elif tensor.shape != tensor_shape:
                        with set_lazy_legacy(True):
                            return _stack(list_of_tensordicts, dim=dim)
                    out[key].append(tensor)
                out[key] = (out[key], is_lazy)

            def stack_fn(key, values, is_lazy):
                if is_lazy:
                    return _stack_uninit_params(values, dim)
                with _ErrorInteceptor(
                    key, "Attempted to stack tensors on different devices at key"
                ):
                    return torch.stack(values, dim)

            out = {
                key: stack_fn(key, values, is_lazy)
                for key, (values, is_lazy) in out.items()
            }

            return TensorDict(
                out,
                batch_size=LazyStackedTensorDict._compute_batch_size(
                    batch_size, dim, len(list_of_tensordicts)
                ),
                device=device,
                _run_checks=False,
            )
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
