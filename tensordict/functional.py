# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

import torch
from tensordict._lazy import LazyStackedTensorDict
from tensordict._td import TensorDict
from tensordict.base import (
    _is_leaf_nontensor,
    _is_tensor_collection,
    CompatibleType,
    NestedKey,
    T,
    TensorDictBase,
)
from tensordict.tensorclass import NonTensorData, NonTensorStack
from tensordict.utils import (
    _check_keys,
    _is_unbatched,
    _shape,
    _SKIP_TENSORCLASS_POST_INIT,
    DeviceType,
    is_non_tensor,
    is_tensorclass,
    unravel_key,
)
from torch.compiler import is_compiling
from torch.utils.data import default_collate

__all__ = [
    "collate",
    "dense_stack_tds",
    "make_tensordict",
    "merge_tensordicts",
    "pad",
    "pad_sequence",
    "unpad_sequence",
]


def pad(
    tensordict: T,
    pad_size: Sequence[int],
    value: float = 0.0,
    inplace: bool = False,
    safe: bool = True,
) -> T:
    """Pads all tensors in a tensordict along the batch dimensions with a constant value.

    Use it to grow the batch dimensions of one tensordict, for example to
    bring samples to a common size before you stack them. To pad and stack a
    list of tensordicts in one call, use :func:`~tensordict.pad_sequence`
    (see :ref:`stacking`).

    Args:
         tensordict (TensorDict): The tensordict to pad
         pad_size (Sequence[int]): The padding size by which to pad some batch
            dimensions of the tensordict, starting from the first dimension and
            moving forward. [len(pad_size) / 2] dimensions of the batch size will
            be padded. For example to pad only the first dimension, pad has the form
            (padding_left, padding_right). To pad two dimensions,
            (padding_left, padding_right, padding_top, padding_bottom) and so on.
            pad_size must be even and less than or equal to twice the number of batch dimensions.
         value (float, optional): The fill value to pad by, default 0.0.
            Non-tensor entries (:class:`~tensordict.NonTensorData` /
            :class:`~tensordict.NonTensorStack`) keep their values at the
            valid positions and hold ``None`` in the pad slots instead of
            ``value``.
         inplace (bool, optional): If ``True``, the input tensordict's identity
            and key set are preserved, and each leaf's storage is replaced by
            its padded counterpart one at a time. This keeps peak memory close
            to the size of the tensordict itself rather than 2x (the case when
            a fresh tensordict is allocated alongside the original). The leaf
            tensors themselves are still freshly allocated (``pad`` necessarily
            grows shapes), so this is not a same-storage operation. Defaults to
            ``False``.

            On :class:`~tensordict.LazyStackedTensorDict`, ``inplace=True`` pads
            each constituent tensordict along the non-stack dimensions in place
            and grows the stack along the stack dimension by appending or
            prepending zero-filled copies of the edge constituents; the lazy
            stack's identity is preserved.

            .. warning::
                If ``inplace=True`` and the operation fails partway through
                (for example an out-of-memory error during a leaf's padding),
                the tensordict is left in an inconsistent state: some leaves
                will have the new shape and others the old, and the
                ``batch_size`` will not have been updated. Restoring the
                original state would require keeping every old leaf alive
                until the whole pass succeeded, which would defeat the 1x
                memory contract. Use ``safe=True`` (the default) to catch
                the realistic user-error class of failures before any
                mutation happens.
         safe (bool, optional): If ``True``, validate that the operation
            would succeed for every leaf before any mutation occurs. This
            catches errors such as negative pad widths that exceed a leaf's
            dimension size, leaves that are not paddable, etc., raising
            before any in-place rebind. Set to ``False`` to skip the
            pre-flight walk for a small speedup when the inputs are known
            to be valid. Defaults to ``True``.

    Returns:
        The padded tensordict. When ``inplace=True`` this is the same object as
        the input; otherwise a new tensordict, or a new tensorclass of the
        same type for a tensorclass input.

    .. note::
        With ``inplace=False``, a lazy stack gives a
        :class:`~tensordict.TensorDict`. If the stacked tensordicts have no
        names, it has no names either, even when the stack dimension has a
        name (from ``stack_dim_name`` or the ``names`` setter). To keep that
        name, name the dimension of the result again with
        :meth:`~tensordict.TensorDict.refine_names`.

    Examples:
        >>> from tensordict import TensorDict, pad
        >>> import torch
        >>> td = TensorDict({'a': torch.ones(3, 4, 1),
        ...     'b': torch.ones(3, 4, 1, 1)}, batch_size=[3, 4])
        >>> dim0_left, dim0_right, dim1_left, dim1_right = [0, 1, 0, 2]
        >>> padded_td = pad(td, [dim0_left, dim0_right, dim1_left, dim1_right], value=0.0)
        >>> print(padded_td.batch_size)
        torch.Size([4, 6])
        >>> print(padded_td.get("a").shape)
        torch.Size([4, 6, 1])
        >>> print(padded_td.get("b").shape)
        torch.Size([4, 6, 1, 1])

    """
    if len(pad_size) > 2 * len(tensordict.batch_size):
        raise RuntimeError(
            "The length of pad_size must be <= 2 * the number of batch dimensions"
        )

    if len(pad_size) % 2:
        raise RuntimeError("pad_size must have an even number of dimensions")

    # perf: a dense tensordict padded out of place runs the checks of
    # _pad_preflight in the loop below, leaf by leaf and in the same order,
    # rather than in a walk of the tree before it. Nothing is written to the
    # input, so the first failing check raises the same error.
    check = safe and not inplace and type(tensordict) is TensorDict
    if safe and not check:
        _pad_preflight(tensordict, pad_size, inplace=inplace)

    if is_non_tensor(tensordict):
        # NonTensorStack is a LazyStackedTensorDict subclass: dispatch before
        # the lazy-stack branch so non-tensor values are preserved.
        return _pad_non_tensor(tensordict, pad_size, inplace=inplace)

    if not inplace and is_tensorclass(tensordict):
        # Pad the wrapped tensordict and rebuild the tensorclass around the
        # result. The bound ``pad`` method is skipped: a tensorclass may
        # override it with another signature.
        return tensordict._from_tensordict(
            pad(tensordict._tensordict, pad_size, value, safe=False),
            dict(tensordict._non_tensordict),
            safe=False,
        )

    if inplace and isinstance(tensordict, LazyStackedTensorDict):
        return _pad_lazy_stack_inplace(tensordict, pad_size, value)

    new_batch_size = list(tensordict.batch_size)
    for i in range(len(pad_size)):
        new_batch_size[i // 2] += pad_size[i]

    reverse_pad = list(pad_size[::-1])
    for i in range(0, len(reverse_pad), 2):
        reverse_pad[i], reverse_pad[i + 1] = reverse_pad[i + 1], reverse_pad[i]

    if inplace:
        out = tensordict
    else:
        try:
            names = tensordict._maybe_names()
        except (IndexError, ValueError):
            # A lazy stack without members, or whose members have different
            # names: leave the dims unnamed.
            names = None
        out = TensorDict._new_unsafe(
            {},
            torch.Size(new_batch_size),
            device=tensordict.device,
            names=names,
        )

    # Snapshot keys so mid-iteration rebinds on `out is tensordict` don't
    # invalidate the iterator under inplace=True.
    keys = list(tensordict.keys())
    try:
        for key in keys:
            tensor = tensordict._get_str(key, default=None)
            if tensor is None:
                continue

            if _is_unbatched(tensor):
                if not inplace:
                    out._set_str(key, tensor, validated=True, inplace=False)
                continue

            if is_non_tensor(tensor):
                if check:
                    _pad_preflight(tensor, pad_size, prefix=(key,))
                if not tensor.batch_size:
                    # Unbatched non-tensor metadata: not indexed along the batch
                    # dims, so padding leaves it untouched.
                    if not inplace:
                        out._set_str(key, tensor, validated=True, inplace=False)
                    continue
                if inplace and isinstance(tensor, NonTensorStack):
                    _pad_non_tensor(tensor, pad_size, inplace=True)
                    continue
                padded = _pad_non_tensor(tensor, pad_size, inplace=False)
                del tensor
                out._set_str(key, padded, validated=True, inplace=False)
                continue

            if _is_tensor_collection(type(tensor)):
                nested_check = check and type(tensor) is TensorDict
                if check and not nested_check:
                    _pad_preflight(
                        tensor._tensordict if is_tensorclass(tensor) else tensor,
                        pad_size,
                        prefix=(key,),
                    )
                padded = pad(
                    tensor, pad_size, value, inplace=inplace, safe=nested_check
                )
                if not inplace:
                    out._set_str(key, padded, validated=True, inplace=False)
                continue

            shape = _shape(tensor)
            cur_pad = reverse_pad
            if len(pad_size) < len(shape) * 2:
                cur_pad = [0] * (len(shape) * 2 - len(pad_size)) + reverse_pad
            if check:
                _pad_check_leaf(key, shape, pad_size, cur_pad)

            padded = torch.nn.functional.pad(tensor, cur_pad, value=value)
            # Drop the local ref to the old tensor before rebinding the dict
            # entry: with no other refs, the _set_str overwrite drops the old
            # storage's refcount to 0 and the allocator can reuse the block
            # for the next leaf's pad.
            del tensor
            out._set_str(key, padded, validated=True, inplace=False)
    except Exception:
        if check:
            # The loop can fail before any check does, for example in the
            # pad of a nested lazy stack, whose checks skip its stack dim:
            # raise the error of the first failing check, if one fails, as
            # the walk before the loop would have, without the loop's error
            # as its context.
            try:
                _pad_preflight(tensordict, pad_size)
            except Exception as err:
                err.__context__ = None
                raise
        raise

    if inplace:
        # Leaves now all match new_batch_size; flip the dict's advertised
        # size via the low-level path so we skip the redundant shape check
        # in `_batch_size_setter`.
        out._change_batch_size(torch.Size(new_batch_size))

    return out


def _pad_preflight(
    tensordict: TensorDictBase,
    pad_size: Sequence[int],
    inplace: bool = False,
    prefix: tuple = (),
) -> None:
    """Validate that ``pad(tensordict, pad_size, ...)`` would succeed for every leaf without mutating anything.

    Walks the tensordict recursively (including ``LazyStackedTensorDict``
    constituents) and checks the per-leaf preconditions of
    ``torch.nn.functional.pad``: that each leaf has enough dimensions for the
    requested ``pad_size`` and that the resulting size on every padded dim is
    non-negative. Catches the realistic user-error class of failures before
    ``inplace=True`` has rebound any leaf.

    With ``inplace=True``, a tensordict whose batch size would change but
    cannot change in place (a file, a sub-tensordict, a ``TypedTensorDict``
    over a lazy stack) raises too, at the top or nested: padding it, or a
    tensordict that holds it, would raise or break it after the earlier
    leaves were padded. A sub-tensordict at the top raises its own error on
    the first write, and a lazy stack pads in place on its own.
    """
    resizes = inplace and any(
        pad_size[i] + pad_size[i + 1] for i in range(0, len(pad_size), 2)
    )
    if resizes and not tensordict._inplace_batch_size_change:
        raise NotImplementedError(
            "pad(inplace=True) changes the batch size, which a "
            f"{type(tensordict).__name__}"
            + (
                f" at key {prefix[0] if len(prefix) == 1 else prefix!r}"
                if prefix
                else ""
            )
            + " cannot do in place. Use inplace=False instead."
        )
    if is_non_tensor(tensordict):
        shape = tensordict.batch_size
        if not shape:
            return
        if len(pad_size) > 2 * len(shape):
            raise RuntimeError(
                f"pad_size of length {len(pad_size)} is too long for non-tensor "
                f"value with batch size {tuple(shape)}."
            )
        for i in range(len(pad_size) // 2):
            left, right = pad_size[2 * i], pad_size[2 * i + 1]
            if shape[i] + left + right < 0:
                raise RuntimeError(
                    f"Pad ({left}, {right}) on dim {i} of non-tensor value "
                    f"(size {shape[i]}) would produce a negative output size."
                )
        return

    if isinstance(tensordict, LazyStackedTensorDict):
        stack_dim = tensordict.stack_dim
        pairs = [
            (pad_size[2 * i], pad_size[2 * i + 1]) for i in range(len(pad_size) // 2)
        ]
        if stack_dim < len(pairs):
            non_stack_pairs = pairs[:stack_dim] + pairs[stack_dim + 1 :]
        else:
            non_stack_pairs = pairs
        constituent_pad_size: list[int] = [p for pair in non_stack_pairs for p in pair]
        if constituent_pad_size and any(p != 0 for p in constituent_pad_size):
            for td_i in tensordict.tensordicts:
                _pad_preflight(td_i, constituent_pad_size, inplace=inplace)
        return

    reverse_pad = list(pad_size[::-1])
    for i in range(0, len(reverse_pad), 2):
        reverse_pad[i], reverse_pad[i + 1] = reverse_pad[i + 1], reverse_pad[i]

    for key in tensordict.keys():
        tensor = tensordict._get_str(key, default=None)
        if tensor is None or _is_unbatched(tensor):
            continue
        if _is_tensor_collection(type(tensor)):
            if is_tensorclass(tensor) and not is_non_tensor(tensor):
                tensor = tensor._tensordict
            nested_key = prefix + (key,)
            if (
                resizes
                # a sub-tensordict holds sub-tensordicts: it raises on its own
                and not tensordict._lazy
                and tensor._lazy
                and not isinstance(tensor, LazyStackedTensorDict)
            ):
                raise NotImplementedError(
                    f"pad(inplace=True) changes the batch size, which a "
                    f"{type(tensor).__name__} at key "
                    f"{nested_key[0] if len(nested_key) == 1 else nested_key!r} "
                    "cannot do in place. Use inplace=False instead."
                )
            _pad_preflight(tensor, pad_size, inplace=inplace, prefix=nested_key)
            continue
        shape = _shape(tensor)
        cur_pad = reverse_pad
        if len(pad_size) < len(shape) * 2:
            cur_pad = [0] * (len(shape) * 2 - len(pad_size)) + reverse_pad
        _pad_check_leaf(key, shape, pad_size, cur_pad)


def _pad_check_leaf(
    key: str, shape: torch.Size, pad_size: Sequence[int], cur_pad: list[int]
) -> None:
    """Raise if ``F.pad`` of the leaf at ``key`` with widths ``cur_pad`` would fail."""
    if len(pad_size) > 2 * len(shape):
        raise RuntimeError(
            f"pad_size of length {len(pad_size)} is too long for leaf "
            f"{key!r} with shape {tuple(shape)}."
        )
    for i in range(0, len(cur_pad), 2):
        dim = len(shape) - 1 - (i // 2)
        left, right = cur_pad[i], cur_pad[i + 1]
        if shape[dim] + left + right < 0:
            raise RuntimeError(
                f"Pad ({left}, {right}) on dim {dim} of leaf {key!r} "
                f"(size {shape[dim]}) would produce a negative output size."
            )


def _pad_non_tensor(
    tensordict: TensorDictBase,
    pad_size: Sequence[int],
    *,
    inplace: bool = False,
) -> TensorDictBase:
    """Pad a ``NonTensorData`` / ``NonTensorStack`` along its leading batch dims.

    Non-tensor values cannot be filled with a numeric constant: the valid
    positions keep their original values and every pad slot holds a
    ``NonTensorData`` whose data is ``None``. Negative pad sizes crop, like
    :func:`torch.nn.functional.pad`.
    """
    pairs = [(pad_size[2 * i], pad_size[2 * i + 1]) for i in range(len(pad_size) // 2)]
    if all(left == 0 and right == 0 for left, right in pairs):
        return tensordict
    result = _pad_non_tensor_rec(tensordict, pairs)
    try:
        names = tensordict._maybe_names()
        if names is not None:
            # the pad slots are unnamed: give the whole result the input's names
            result.names = names
    except (IndexError, ValueError):
        # A stack without members, or names that differ between its members
        # or are used twice: leave the names as they are.
        pass
    if not inplace:
        return result
    if not isinstance(tensordict, NonTensorStack):
        raise RuntimeError(
            "pad(..., inplace=True) cannot preserve the identity of a "
            "NonTensorData input since the padded result is a NonTensorStack. "
            "Use inplace=False instead."
        )
    if tensordict.stack_dim == 0:
        new_tensordicts = result.tensordicts
    else:
        new_tensordicts = result.unbind(tensordict.stack_dim)
    tensordict.tensordicts[:] = new_tensordicts
    tensordict._change_batch_size(result.batch_size)
    return tensordict


def _pad_non_tensor_rec(
    tensordict: TensorDictBase, pairs: list[tuple[int, int]]
) -> NonTensorStack:
    left, right = pairs[0]
    rest = pairs[1:]
    elements = list(tensordict.unbind(0))
    if rest and any(p != 0 for pair in rest for p in pair):
        elements = [_pad_non_tensor_rec(element, rest) for element in elements]
    if left < 0:
        elements = elements[-left:]
        left = 0
    if right < 0:
        elements = elements[: len(elements) + right]
        right = 0
    inner_shape = list(tensordict.batch_size[1:])
    for dim, (inner_left, inner_right) in enumerate(rest):
        inner_shape[dim] += inner_left + inner_right
    device = tensordict.device
    items = (
        [
            NonTensorData(data=None, batch_size=inner_shape, device=device)
            for _ in range(left)
        ]
        + elements
        + [
            NonTensorData(data=None, batch_size=inner_shape, device=device)
            for _ in range(right)
        ]
    )
    if not items:
        raise RuntimeError(
            "Padding a non-tensor value to a zero-sized batch dimension is not "
            "supported: a NonTensorStack cannot be empty."
        )
    return NonTensorStack(*items, stack_dim=0)


def _pad_filler_like(template: TensorDictBase, value: float) -> TensorDictBase:
    """Build a pad-slot constituent shaped like ``template``.

    Tensor leaves are filled with ``value``; non-tensor leaves hold ``None``
    (a non-tensor value cannot be filled with a numeric constant).
    """
    if is_non_tensor(template):
        return NonTensorData(
            data=None, batch_size=template.batch_size, device=template.device
        )
    filler = template.apply(lambda t: torch.full_like(t, value))
    for key in template.keys(True, True, is_leaf=_is_leaf_nontensor):
        leaf = template.get(key)
        if is_non_tensor(leaf):
            filler.set(
                key,
                NonTensorData(
                    data=None, batch_size=leaf.batch_size, device=leaf.device
                ),
            )
    return filler


def _with_batch_size(
    template: TensorDictBase, batch_size: Sequence[int]
) -> TensorDictBase:
    """Give a tensordict without leaves the batch size ``batch_size``.

    The sub-tensordicts get it too, followed by their own extra dims. The
    number of elements may change, which a reshape would not allow.
    """
    batch_dims = template.batch_dims
    return template._fast_apply(
        lambda nested: _with_batch_size(
            nested, (*batch_size, *nested.batch_size[batch_dims:])
        ),
        batch_size=batch_size,
        call_on_nested=True,
    )


def _pad_lazy_stack_inplace(
    tensordict: LazyStackedTensorDict,
    pad_size: Sequence[int],
    value: float,
) -> LazyStackedTensorDict:
    """In-place pad for a LazyStackedTensorDict.

    Pads each constituent along the non-stack dimensions in place, then grows
    the stack dimension by appending/prepending zero-filled copies of the edge
    constituents. The lazy stack's identity is preserved.
    """
    stack_dim = tensordict.stack_dim
    pairs = [(pad_size[2 * i], pad_size[2 * i + 1]) for i in range(len(pad_size) // 2)]

    if stack_dim < len(pairs):
        left, right = pairs[stack_dim]
        non_stack_pairs = pairs[:stack_dim] + pairs[stack_dim + 1 :]
    else:
        left, right = 0, 0
        non_stack_pairs = pairs

    constituent_pad_size: list[int] = [p for pair in non_stack_pairs for p in pair]

    if constituent_pad_size and any(p != 0 for p in constituent_pad_size):
        for td_i in tensordict.tensordicts:
            pad(td_i, constituent_pad_size, value, inplace=True, safe=False)

    if (left > 0 or right > 0) and not tensordict.tensordicts:
        raise RuntimeError(
            "Cannot pad along the stack dimension of an empty LazyStackedTensorDict: "
            "no template constituent is available to fill the new slots."
        )

    if right > 0:
        template = tensordict.tensordicts[-1]
        new_pads = [_pad_filler_like(template, value) for _ in range(right)]
        tensordict.tensordicts.extend(new_pads)
    if left > 0:
        template = tensordict.tensordicts[0]
        new_pads = [_pad_filler_like(template, value) for _ in range(left)]
        tensordict.tensordicts[:0] = new_pads

    new_batch_size = list(tensordict.batch_size)
    for i, (l, r) in enumerate(pairs):
        new_batch_size[i] += l + r
    tensordict._change_batch_size(torch.Size(new_batch_size))
    return tensordict


def collate(batch: Any) -> Any:
    """Collates a list of samples into a batch, for :class:`~torch.utils.data.DataLoader`.

    Pass it as ``collate_fn`` when the samples are tensordicts or
    tensorclasses, or tuples or dicts that hold them, such as
    ``(td, label)`` pairs. The default ``collate_fn`` of a ``DataLoader``
    cannot read a tensordict (see :ref:`data-loading`).

    The samples are collated by the type of the first one:

    - tensordicts and tensorclasses go through
      :meth:`~tensordict.LazyStackedTensorDict.maybe_dense_stack`: the batch
      is a dense stack if the samples have the same keys and entry shapes,
      and a :class:`~tensordict.LazyStackedTensorDict` otherwise. A
      tensorclass keeps its type.
    - tensors go through :func:`torch.stack`.
    - mappings give a ``dict`` and tuples a tuple (a namedtuple keeps its
      type), with each entry collated across the samples.
    - anything else goes through :func:`torch.utils.data.default_collate`.

    Args:
        batch (list): the samples of one batch. Anything other than a
            non-empty list is returned as is, for example the tensordict
            that a batch sampler reads in one index operation.

    Returns:
        the collated batch.

    Examples:
        >>> import torch
        >>> from torch.utils.data import DataLoader
        >>> from tensordict import TensorDict, collate
        >>> samples = [(TensorDict(x=torch.randn(3)), i) for i in range(10)]
        >>> td, label = next(iter(DataLoader(samples, batch_size=4, collate_fn=collate)))
        >>> td.batch_size, label
        (torch.Size([4]), tensor([0, 1, 2, 3]))
        >>> clouds = [TensorDict(points=torch.randn(n, 3)) for n in (5, 2, 7, 3)]
        >>> batch = next(iter(DataLoader(clouds, batch_size=4, collate_fn=collate)))
        >>> type(batch).__name__, batch.batch_size
        ('LazyStackedTensorDict', torch.Size([4]))
    """
    if not isinstance(batch, list) or not batch:
        return batch
    first = batch[0]
    if _is_tensor_collection(type(first)):
        return LazyStackedTensorDict.maybe_dense_stack(batch)
    if isinstance(first, torch.Tensor):
        return torch.stack(batch)
    if isinstance(first, Mapping):
        return {key: collate([sample[key] for sample in batch]) for key in first}
    if isinstance(first, tuple):
        items = (collate(list(entries)) for entries in zip(*batch))
        if hasattr(first, "_fields"):
            return type(first)(*items)
        return tuple(items)
    return default_collate(batch)


def pad_sequence(
    list_of_tensordicts: Sequence[T],
    pad_dim: int = 0,
    padding_value: float = 0.0,
    out: T | None = None,
    return_mask: bool | NestedKey = False,
    padding_side: str = "right",
) -> T:
    """Pads a list of tensordicts in order for them to be stacked together in a contiguous format.

    Use it to batch samples whose entries differ in size along one
    dimension, such as sequences of different lengths. Every entry is padded
    along ``pad_dim`` to the largest size among the inputs, then the inputs
    are stacked along a new first dimension. The inputs must have the same
    keys, and the entry shapes must match in every dimension other than
    ``pad_dim``. To keep the inputs unpadded, use
    :func:`~tensordict.lazy_stack` instead (see :ref:`stacking`).

    Args:
        list_of_tensordicts (List[TensorDictBase]): the list of instances to pad and stack.
        pad_dim (int, optional): the dimension of the entries to pad, counted
            from the first dimension of each entry (batch dimensions
            included). If it is a batch dimension, the batch size of the
            result is padded too. A negative ``pad_dim`` counts from the last
            dimension of each entry. If entries of different ranks put it on
            different dimensions, at least one of them a batch dimension, the
            samples must have the same batch size. Defaults to ``0``.
        padding_value (number, optional): the padding value. Defaults to ``0.0``.
            Non-tensor entries are not filled with ``padding_value``: when the padded
            dim is a batch dimension of the output, they keep their values at the
            valid positions and hold ``None`` in the pad slots, as with
            :func:`~tensordict.pad`.
        out (TensorDictBase, optional): if provided, the destination where the data will be
            written.
        return_mask (bool or NestedKey, optional): if ``True``, the result has
            a ``"masks"`` entry. If ``return_mask`` is a nested key (a string
            or a tuple of strings), the masks are stored under that key. The
            masks entry is a tensordict with the structure of the stacked
            tensordict. Each of its entries is the mask of the valid values.
            If ``pad_dim`` is a batch dimension, the mask has the output's
            batch size. Otherwise, its shape is the output's batch size
            followed by the padded length, regardless of the feature
            dimension's position. Not available for tensorclass inputs. Use
            :func:`~tensordict.unpad_sequence` to turn the result back into
            the variable-length samples.
        padding_side (str, optional): ``"right"`` puts the padding after the
            data along ``pad_dim``, ``"left"`` puts it before, as
            ``get(key, as_padded_tensor=True, padding_side=...)`` does on a
            lazy stack. The masks mark the data positions on either side.
            Defaults to ``"right"``.

    Returns:
        a :class:`~tensordict.TensorDict` (or the tensorclass of the inputs)
        with batch size ``[len(list_of_tensordicts), *batch_size]``.

    Raises:
        RuntimeError: if ``list_of_tensordicts`` is empty, or if the entry
            shapes differ in a dimension other than ``pad_dim``.
        ValueError: if ``padding_side`` is neither ``"left"`` nor ``"right"``,
            or if a negative ``pad_dim`` lands on different dimensions of
            entries of different ranks and the batch sizes of the samples
            differ.
        IndexError: if ``pad_dim`` is out of range for an entry.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, pad_sequence
        >>> list_td = [
        ...     TensorDict({"a": torch.zeros((3, 8)), "b": torch.zeros((6, 8))}, batch_size=[]),
        ...     TensorDict({"a": torch.zeros((5, 8)), "b": torch.zeros((6, 8))}, batch_size=[]),
        ...     ]
        >>> padded_td = pad_sequence(list_td, return_mask=True)
        >>> print(padded_td)
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([2, 5, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                b: Tensor(shape=torch.Size([2, 6, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                masks: TensorDict(
                    fields={
                        a: Tensor(shape=torch.Size([2, 5]), device=cpu, dtype=torch.bool, is_shared=False),
                        b: Tensor(shape=torch.Size([2, 6]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False)
    """
    if not len(list_of_tensordicts):
        raise RuntimeError("list_of_tensordicts cannot be empty")
    if padding_side not in ("left", "right"):
        raise ValueError(
            f"padding_side must be 'left' or 'right', got {padding_side!r}."
        )

    if return_mask and is_tensorclass(list_of_tensordicts[0]):
        raise RuntimeError(
            "Expected 'return_mask=False' when list_of_tensordicts contains "
            "tensorclasses, but got 'return_mask=True'. If you want masks, "
            "plase convert the tensorclasses to TensorDicts first."
        )

    if not isinstance(return_mask, bool):
        masks_key = unravel_key(return_mask)
        return_mask = True
    else:
        masks_key = "masks"

    # check that all tensordict match
    update_batch_size = True
    max_seq_length = float("-inf")
    keys = _check_keys(list_of_tensordicts, leaves_only=True, include_nested=True)
    list_of_dicts = [{} for _ in range(len(list_of_tensordicts))]
    mask_pad_dims = {}
    # The batch dims that pad_dim lands on, counted in each entry
    leaf_batch_pad_dims = set()
    for i, td in enumerate(list_of_tensordicts):
        if is_tensorclass(td):
            td = td._tensordict

        for key in keys:
            item = td.get(key)
            list_of_dicts[i][key] = item
            if is_non_tensor(item):
                continue
            tensor_shape = item.shape

            if len(tensor_shape) == 0:
                raise RuntimeError("Cannot pad scalars")

            pos_pad_dim = pad_dim if pad_dim >= 0 else len(tensor_shape) + pad_dim
            if pos_pad_dim < 0:
                raise IndexError(
                    f"pad_dim={pad_dim} is out of range for the entry {key!r} "
                    f"of shape {tuple(tensor_shape)}."
                )

            # track the maximum sequence length to update batch_size accordingly
            if tensor_shape[pos_pad_dim] > max_seq_length:
                max_seq_length = tensor_shape[pos_pad_dim]

            # The mask should always contain the batch_size of the TensorDict
            mask_shape = td.shape

            # if the pad_dim is past the batch_size of the TensorDict, we need to add the new dimension to the mask
            if pos_pad_dim >= td.ndim:
                mask_shape += torch.Size([tensor_shape[pos_pad_dim]])
                update_batch_size = False
            else:
                leaf_batch_pad_dims.add(pos_pad_dim)

            if return_mask:
                mask_key = unravel_key((masks_key, key))
                mask_pad_dims[mask_key] = min(pos_pad_dim, td.ndim)
                list_of_dicts[i][mask_key] = torch.ones(mask_shape, dtype=torch.bool)

    # Each mask key once, in the order of its first sample
    keys = list(keys) + list(mask_pad_dims)

    old_batch_size = list(list_of_tensordicts[0].batch_size)
    if leaf_batch_pad_dims and (len(leaf_batch_pad_dims) > 1 or not update_batch_size):
        # A negative pad_dim counts from the last dim of each entry: entries of
        # different ranks put it on different dims, so the batch cannot be
        # padded. That is fine if no batch dim needs padding.
        if any(
            td.batch_size != list_of_tensordicts[0].batch_size
            for td in list_of_tensordicts
        ):
            raise ValueError(
                f"pad_dim={pad_dim} counts from the last dim of each entry, "
                "and the entries of different ranks put it on different "
                f"dims: batch dims {sorted(leaf_batch_pad_dims)}"
                + ("" if update_batch_size else " and feature dims")
                + ". This works only if the samples have the same batch size. "
                "Pass a non-negative pad_dim."
            )
        update_batch_size = False
    # The batch dim set to max_seq_length, if any: non-tensor entries are
    # padded along it too.
    batch_pad_dim = None
    if update_batch_size and len(old_batch_size) > 0:
        if max_seq_length == float("-inf"):
            # No tensor leaf: pad the non-tensor entries to the longest batch
            max_seq_length = max(td.batch_size[pad_dim] for td in list_of_tensordicts)
            batch_pad_dim = pad_dim % len(old_batch_size)
        else:
            (batch_pad_dim,) = leaf_batch_pad_dims
        old_batch_size[batch_pad_dim] = max_seq_length
    shape = [
        len(list_of_tensordicts),
    ] + old_batch_size

    if out is None:
        # The shell is filled in place below: tensorclass __post_init__ does
        # not run on it. Dynamo cannot set a ContextVar, so compiled code
        # still runs it.
        token = None if is_compiling() else _SKIP_TENSORCLASS_POST_INIT.set(True)
        try:
            out = _with_batch_size(
                list_of_tensordicts[0].empty(recurse=True), torch.Size(shape)
            )
        finally:
            if token is not None:
                _SKIP_TENSORCLASS_POST_INIT.reset(token)

    for key in keys:
        try:
            item0 = list_of_dicts[0][key]
            if _is_unbatched(item0):
                out.set(key, item0)
                continue
            if is_non_tensor(item0):
                items = [d[key] for d in list_of_dicts]
                if batch_pad_dim is not None and out.batch_dims > batch_pad_dim + 1:
                    # Pad the entries that are shorter than out along that
                    # dim, as pad does: original values, then None.
                    length = out.batch_size[batch_pad_dim + 1]
                    items = [
                        _pad_non_tensor(
                            item,
                            [0, 0] * batch_pad_dim
                            + (
                                [0, length - item.batch_size[batch_pad_dim]]
                                if padding_side == "right"
                                else [length - item.batch_size[batch_pad_dim], 0]
                            ),
                        )
                        if item.batch_size[batch_pad_dim] < length
                        else item
                        for item in items
                    ]
                out.set(key, TensorDict.lazy_stack(items))
                continue
            tensor_shape = item0.shape
            if key in mask_pad_dims:
                pos_pad_dim = mask_pad_dims[key]
            else:
                pos_pad_dim = pad_dim if pad_dim >= 0 else len(tensor_shape) + pad_dim
            out.set(
                key,
                torch.nn.utils.rnn.pad_sequence(
                    [d[key].transpose(0, pos_pad_dim) for d in list_of_dicts],
                    batch_first=True,
                    padding_value=padding_value if key not in mask_pad_dims else False,
                    padding_side=padding_side,
                ).transpose(1, pos_pad_dim + 1),
                inplace=True,
            )
        except Exception as err:
            raise RuntimeError(f"pad_sequence failed for key {key}") from err
    return out


def unpad_sequence(
    padded: TensorDictBase,
    mask: TensorDictBase | torch.Tensor | None = None,
    *,
    pad_dim: int = 0,
) -> LazyStackedTensorDict:
    """Splits a padded batch back into its variable-length samples.

    The inverse of :func:`~tensordict.pad_sequence` with
    ``return_mask=True``: each row of ``padded`` keeps, in every entry, the
    positions that the mask marks along ``pad_dim``. These must form one
    contiguous run per row, as the masks of :func:`~tensordict.pad_sequence`
    do. Left and right padding are both read from the mask. Non-tensor
    entries are kept as they are, except when ``pad_dim`` is a batch
    dimension: then they lose their pad slots too.

    The lengths are read on the host (``.tolist()``), which synchronizes
    with the device: this is meant for post-processing, not for compiled
    hot paths.

    Args:
        padded (TensorDictBase): the padded batch, with the samples along its
            first dimension.
        mask (TensorDictBase or torch.Tensor, optional): the valid positions.
            A tensordict holds one mask per entry, laid out as the masks of
            :func:`~tensordict.pad_sequence`. A boolean tensor of shape
            ``[len(padded), max_len]`` is used for every entry. If ``None``,
            the masks are read from the ``"masks"`` entry of ``padded``, and
            the result has no ``"masks"`` entry. When ``mask`` is given,
            every entry of ``padded`` is unpadded. Defaults to ``None``.

    Keyword Args:
        pad_dim (int, optional): the ``pad_dim`` given to
            :func:`~tensordict.pad_sequence`, counted from the first
            dimension of each sample's entries. Defaults to ``0``.

    Returns:
        a :class:`~tensordict.LazyStackedTensorDict` that stacks the samples
        along its first dimension. Each sample has the batch size of the
        original sample, and its entries are views of ``padded``.

    Raises:
        ValueError: if ``padded`` has no batch dimension, if ``mask`` is
            ``None`` and ``padded`` has no ``"masks"`` entry, if a tensor
            mask does not have shape ``[len(padded), max_len]``, or if the
            marked positions of a row are not contiguous.
        KeyError: if a mask tensordict has no mask for an entry of
            ``padded``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, pad_sequence, unpad_sequence
        >>> list_td = [
        ...     TensorDict({"a": torch.arange(3), "b": torch.zeros(3, 2)}, batch_size=[3]),
        ...     TensorDict({"a": torch.arange(5), "b": torch.ones(5, 2)}, batch_size=[5]),
        ... ]
        >>> padded = pad_sequence(list_td, return_mask=True, padding_side="left")
        >>> padded["a"]
        tensor([[0, 0, 0, 1, 2],
                [0, 1, 2, 3, 4]])
        >>> samples = unpad_sequence(padded)
        >>> samples[0]["a"]
        tensor([0, 1, 2])
        >>> [sample.batch_size for sample in samples.tensordicts]
        [torch.Size([3]), torch.Size([5])]
    """
    if not padded.batch_dims:
        raise ValueError(
            "unpad_sequence needs the samples along the first batch dimension "
            "of padded, which has none."
        )
    if mask is None:
        if "masks" not in padded.keys():
            raise ValueError(
                "unpad_sequence needs a mask: pass mask=..., or pad with "
                "pad_sequence(..., return_mask=True)."
            )
        mask = padded.get("masks")
        padded = padded.exclude("masks")
    elif isinstance(mask, torch.Tensor) and (
        mask.ndim != 2 or mask.shape[0] != padded.batch_size[0]
    ):
        raise ValueError(
            "A tensor mask must have shape [len(padded), max_len], got "
            f"{tuple(mask.shape)}."
        )
    # Batch dims of each sample
    batch_dims = padded.batch_dims - 1

    def bounds(key, pos_pad_dim, max_len):
        # The starts and lengths of the data along pad_dim, one per sample
        if isinstance(mask, torch.Tensor):
            if mask.shape[-1] != max_len:
                raise ValueError(
                    "A tensor mask must have shape [len(padded), max_len] = "
                    f"[{mask.shape[0]}, {max_len}], got {tuple(mask.shape)}."
                )
            m = mask
        else:
            # The masks of pad_sequence hold the batch dims, and the padded
            # dim after them if it is not a batch dim.
            m = mask.get(key, None)
            if m is None:
                raise KeyError(f"The mask has no entry for key {key!r} of padded.")
            m = m.movedim(min(pos_pad_dim, batch_dims) + 1, -1)
            if m.ndim > 2:
                m = m.flatten(1, -2).any(1)
        if m.shape[-1] == 0:
            # Every sample is empty along pad_dim
            return [0] * m.shape[0], [0] * m.shape[0]
        m = m.to(torch.uint8)
        # First and last marked positions and their count, in one sync
        starts, ends, lengths = torch.stack(
            (m.argmax(-1), m.shape[-1] - 1 - m.flip(-1).argmax(-1), m.sum(-1))
        ).tolist()
        if any(
            n and end - start + 1 != n for start, end, n in zip(starts, ends, lengths)
        ):
            raise ValueError(
                f"The mask{'' if key is None else f' of key {key!r}'} marks "
                "positions that are not "
                "contiguous in some row: unpad_sequence keeps one contiguous "
                "run per row."
            )
        return starts, lengths

    pad_dims = {}
    for key in padded.keys(True, True, is_leaf=_is_leaf_nontensor):
        item = padded.get(key)
        if is_non_tensor(item):
            continue
        pad_dims[key] = pad_dim if pad_dim >= 0 else item.ndim - 1 + pad_dim
    if batch_dims > 0 and all(pos < batch_dims for pos in pad_dims.values()):
        # The batch was padded too: slicing it unpads every entry
        batch_pad_dim = next(iter(pad_dims.values()), pad_dim % batch_dims)
        starts, lengths = bounds(
            next(iter(pad_dims), None),
            batch_pad_dim,
            padded.batch_size[batch_pad_dim + 1],
        )
        samples = [
            padded[i][(slice(None),) * batch_pad_dim + (slice(start, start + n),)]
            for i, (start, n) in enumerate(zip(starts, lengths))
        ]
    else:
        key_bounds = {
            key: bounds(key, pos, padded.get(key).shape[pos + 1])
            for key, pos in pad_dims.items()
        }
        samples = []
        for i in range(padded.batch_size[0]):
            sample = padded[i]
            unpadded = {}
            for key, (starts, lengths) in key_bounds.items():
                item = sample.get(key)
                pos_pad_dim = pad_dims[key]
                if starts[i] != 0 or lengths[i] != item.shape[pos_pad_dim]:
                    unpadded[key] = item.narrow(pos_pad_dim, starts[i], lengths[i])
            if unpadded:
                # A new tensordict: the members of a lazy stack stay as they are
                sample = sample.exclude(*unpadded)
                for key, item in unpadded.items():
                    sample.set(key, item)
            samples.append(sample)
    return LazyStackedTensorDict(*samples, stack_dim=0)


def merge_tensordicts(
    *tensordicts: T,
    callback_exist: (
        Callable[[Any], Any] | Dict[NestedKey, Callable[[Any], Any]] | None
    ) = None,
) -> T:
    """Merges tensordicts together.

    Args:
        *tensordicts (sequence of TensorDict or equivalent): the list of tensordicts to merge together.

    Keyword Args:
        callback_exist (callable or Dict[str, callable], optional): a callable in case an entry exists in each and every tensordict.
            If the entry is present in some but not all tensordicts, or if ``callback_exist`` is not passed,
            `update` is used and the first non-``None`` value in the tensordict sequence will be used.
            If a dictionary of callables is passed, it will contain the associated callback function for some of the
            nested keys in the tensordicts passed to the function.

    Examples:
        >>> import torch
        >>> from tensordict import merge_tensordicts, TensorDict
        >>> td0 = TensorDict({"a": {"b0": 0}, "c": {"d": {"e": 0}}, "common": 0})
        >>> td1 = TensorDict({"a": {"b1": 1}, "f": {"g": {"h": 1}}, "common": 1})
        >>> td2 = TensorDict({"a": {"b2": 2}, "f": {"g": {"h": 2}}, "common": 2})
        >>> td = merge_tensordicts(td0, td1, td2, callback_exist=lambda *v: torch.stack(list(v)))
        >>> print(td)
        TensorDict(
            fields={
                a: TensorDict(
                    fields={
                        b0: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                        b1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                        b2: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                c: TensorDict(
                    fields={
                        d: TensorDict(
                            fields={
                                e: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                common: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.int64, is_shared=False),
                f: TensorDict(
                    fields={
                        g: TensorDict(
                            fields={
                                h: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(td["common"])
        tensor([0, 1, 2])

    """
    if len(tensordicts) < 2:
        raise RuntimeError(
            f"at least 2 tensordicts must be provided, got {len(tensordicts)}"
        )

    out = tensordicts[0].empty(recurse=True)
    key_list = set()
    # The keys of the non-tensor entries of the other tensordicts, which are
    # added at the end, with the tensordicts that hold them
    non_tensor = {}
    current = None

    def func(name, *vals):
        nonlocal key_list
        if type(vals[0]) is not torch.Tensor and is_non_tensor(vals[0]):
            if current is not tensordicts[0]:
                owners = non_tensor.setdefault(name, [])
                if not owners or owners[-1] is not current:
                    owners.append(current)
            return
        if name in key_list:
            return
        key_list.add(name)
        cb = (
            callback_exist
            if not isinstance(callback_exist, Mapping)
            else callback_exist.get(name)
        )
        if cb is not None and all(val is not None for val in vals):
            out.set(name, cb(*vals))
            return
        for val in vals:
            if val is not None:
                out.set(name, val)
                return

    for i in range(len(tensordicts)):
        if i > 0:
            tds = tensordicts[i + 1 :] + tensordicts[:i]
        else:
            tds = tensordicts[1:]
        current = tensordicts[i]
        tensordicts[i]._fast_apply(
            func,
            *tds,
            named=True,
            nested_keys=True,
            filter_empty=True,
            default=None,
            is_leaf=_is_leaf_nontensor,
        )
    # out has the non-tensor entries of tensordicts[0] (from empty()). Add those
    # of the others where nothing is on their path in out yet, so that the first
    # one wins and the entries set above stay. Lazy stacks (no entry for a key
    # that only some members hold) and tensorclasses (only their fields) in out
    # are left as they are.
    for name, owners in non_tensor.items():
        node, first = out, tensordicts[0]
        for subkey in (name,) if isinstance(name, str) else name:
            if is_tensorclass(node) or isinstance(node, LazyStackedTensorDict):
                break
            parent, node = node, node._get_str(subkey, None)
            if _is_tensor_collection(type(first)):
                first = first._get_str(subkey, None)
            else:
                first = None
            if node is not None and node is first:
                # empty() keeps the sub-tensordicts of tensordicts[0] that have
                # no tensors: write into a copy, not into tensordicts[0]
                if type(node) is not TensorDict:
                    break
                node = node._select(*node.keys())
                parent.set(subkey, node)
            if node is None or _is_leaf_nontensor(type(node)):
                break
        if node is None:
            for td in owners:
                # For a lazy stack, get() stacks the values of the members
                val = td.get(name, None)
                if val is not None:
                    try:
                        # As before, leave out a value that out cannot take
                        # (a stack of another batch size, other dim names)
                        val = parent._validate_value(val, check_shape=True)
                    except (RuntimeError, ValueError):
                        break
                    out.set(name, val)
                    break
    return out


def dense_stack_tds(
    td_list: Sequence[TensorDictBase] | LazyStackedTensorDict,
    dim: int | None = None,
) -> T:
    """Stacks a list of tensordicts, or the tensordicts of a lazy stack, densely if possible.

    Use it to turn a :class:`~tensordict.LazyStackedTensorDict` into a dense
    :class:`~tensordict.TensorDict` when its tensordicts have the same keys
    and entry shapes. It is useful when the tensordicts are lazy stacks
    themselves, or hold lazy stacks among their nested entries: then
    ``lazy_stack(td_list).to_tensordict()`` is not possible. This function
    calls :func:`~tensordict.maybe_dense_stack`. So if the tensordicts have
    different keys or entry shapes at the top level, the result is a
    :class:`~tensordict.LazyStackedTensorDict`. See :ref:`stacking`.

    Args:
        td_list (List of TensorDictBase or LazyStackedTensorDict): the
            tensordicts to stack. A :class:`~tensordict.TensorDict` is
            returned unchanged.
        dim (int, optional): the dimension to stack them along. Required if
            ``td_list`` is a list: without it, a ``ValueError`` is raised.
            Ignored if ``td_list`` is a
            :class:`~tensordict.LazyStackedTensorDict`, whose ``stack_dim`` is
            used.

    Returns:
        a :class:`~tensordict.TensorDict` (or the class of the stacked
        tensordicts) if they can be stacked densely, a
        :class:`~tensordict.LazyStackedTensorDict` otherwise.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, lazy_stack
        >>> from tensordict import assert_allclose_td, dense_stack_tds
        >>> td0 = TensorDict({"a": torch.zeros(3)},[])
        >>> td1 = TensorDict({"a": torch.zeros(4), "b": torch.zeros(2)},[])
        >>> td_lazy = lazy_stack([td0, td1], dim=0)
        >>> td_container = TensorDict({"lazy": td_lazy}, [])
        >>> td_container_clone = td_container.clone()
        >>> td_stack = lazy_stack([td_container, td_container_clone], dim=0)
        >>> td_stack
        LazyStackedTensorDict(
            fields={
                lazy: LazyStackedTensorDict(
                    fields={
                        a: Tensor(shape=torch.Size([2, 2, -1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    exclusive_fields={
                    },
                    batch_size=torch.Size([2, 2]),
                    device=None,
                    is_shared=False,
                    stack_dim=0)},
            exclusive_fields={
            },
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False,
            stack_dim=0)
        >>> td_stack = dense_stack_tds(td_stack) # Automatically use the LazyStackedTensorDict stack_dim
        >>> td_stack
        TensorDict(
            fields={
                lazy: LazyStackedTensorDict(
                    fields={
                        a: Tensor(shape=torch.Size([2, 2, -1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    exclusive_fields={
                        1 ->
                            b: Tensor(shape=torch.Size([2, 2]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([2, 2]),
                    device=None,
                    is_shared=False,
                    stack_dim=1)},
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False)
        >>> # Note that
        >>> # (1) td_stack is now a TensorDict
        >>> # (2) this has pushed the stack_dim of "lazy" (0 -> 1)
        >>> # (3) this has revealed the exclusive keys.
        >>> assert assert_allclose_td(td_stack, dense_stack_tds([td_container, td_container_clone], dim=0))
        >>> # This shows it is the same to pass a list or a LazyStackedTensorDict

    """
    if isinstance(td_list, LazyStackedTensorDict):
        dim = td_list.stack_dim
        td_list = td_list.tensordicts
    elif isinstance(td_list, TensorDict):
        # then it is already dense
        return td_list
    elif dim is None:
        raise ValueError(
            "If a list of tensordicts is provided, stack_dim must not be None"
        )
    shape = list(td_list[0].shape)
    shape.insert(dim, len(td_list))

    return TensorDict.maybe_dense_stack(td_list, dim=dim)


def make_tensordict(
    input_dict: dict[str, CompatibleType] | None = None,
    batch_size: Sequence[int] | torch.Size | int | None = None,
    device: DeviceType | None = None,
    auto_batch_size: bool | None = None,
    **kwargs: CompatibleType,  # source
) -> TensorDict:
    """Returns a TensorDict created from the keyword arguments or an input dictionary.

    If ``batch_size`` is not specified and ``auto_batch_size=True``, the maximum batch size possible is used.

    This function works on nested dictionaries too, or can be used to determine the
    batch-size of a nested tensordict.

    Args:
        input_dict (dictionary, optional): a dictionary to use as a data source
            (nested keys compatible).
        **kwargs (TensorDict or torch.Tensor): keyword arguments as data source
            (incompatible with nested keys).
        batch_size (iterable of int, optional): a batch size for the tensordict.
        device (torch.device or compatible type, optional): a device for the TensorDict.
        auto_batch_size (bool, optional): if ``True``, the batch size will be computed automatically.
            Defaults to ``False``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, make_tensordict
        >>> input_dict = {"a": torch.randn(3, 4), "b": torch.randn(3)}
        >>> print(make_tensordict(input_dict, auto_batch_size=True))
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> # alternatively
        >>> td = make_tensordict(**input_dict)
        >>> # nested dict: the nested TensorDict can have a different batch-size
        >>> # as long as its leading dims match.
        >>> input_dict = {"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}
        >>> print(make_tensordict(input_dict, auto_batch_size=True))
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3, 4]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> # we can also use this to work out the batch sie of a tensordict
        >>> input_td = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}, [])
        >>> print(make_tensordict(input_td, auto_batch_size=True))
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3, 4]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
    """
    if input_dict is not None:
        kwargs.update(input_dict)
    return TensorDict.from_dict(
        kwargs, batch_size=batch_size, device=device, auto_batch_size=auto_batch_size
    )
