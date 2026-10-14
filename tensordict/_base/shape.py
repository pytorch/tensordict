# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Operations on the batch shape: views, reshaping, stacking, splitting and padding of :class:`~tensordict.TensorDictBase`.

The methods live on a mixin that ``tensordict.base`` imports before it
defines ``TensorDictBase``, so this module imports the helpers it needs from
``tensordict.base`` without needing the class itself. A method that uses
``TensorDictBase`` at run time imports it locally. The comment above the
mixin imports in ``tensordict/base.py`` gives the other rules.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from collections.abc import Iterable
from typing import Callable, List, overload, Sequence, TYPE_CHECKING

import torch
from tensordict._nestedkey import NestedKey
from tensordict._tensorcollection import TensorCollection
from tensordict.base import (
    _is_leaf_nontensor,
    _is_tensor_collection,
    is_tensor_collection,
    Self,
    T,
)
from tensordict.utils import (
    _as_context_manager,
    _check_unflatten_size,
    _create_segments_from_list,
    _get_shape_from_args,
    _infer_size_impl,
    _is_tensorclass,
    _is_unbatched,
    _maybe_correct_neg_dim,
    _resolve_expand_shape,
    _warn_dim_past_batch_dims,
    _warn_lazy_shape_args,
    _zip_strict,
    is_non_tensor,
    lazy_legacy,
    unravel_key_list,
)
from torch import Tensor

if TYPE_CHECKING:
    from tensordict.base import TensorDictBase


def _squeeze_dim(dim: int | None = None) -> int | None:
    # the arguments of squeeze, which takes *args and **kwargs
    return dim


def _unsqueeze_dim(dim: int) -> int:
    # the arguments of unsqueeze, which takes *args and **kwargs
    return dim


def _repeat_interleave_batch_size(
    batch_size: torch.Size, repeats: torch.Tensor | int, dim: int
) -> torch.Size:
    # the batch size after repeat_interleave(repeats, dim)
    new_batch_size = []
    for i, s in enumerate(batch_size):
        if i == dim:
            if isinstance(repeats, int):
                new_batch_size.append(s * repeats)
            elif (
                isinstance(repeats, torch.Tensor)
                and repeats.ndim <= 1
                and repeats.numel() == 1
                and repeats.dtype in (torch.int32, torch.int64)
            ):
                # torch broadcasts a 0-d or one-element repeats to the size of dim
                new_batch_size.append(s * int(repeats))
            else:
                new_batch_size.append(repeats.sum().item())
        else:
            new_batch_size.append(s)
    return torch.Size(new_batch_size)


@dataclasses.dataclass(frozen=True)
class PackLayout:
    """The layout of a tensordict packed by :meth:`~tensordict.TensorDictBase.pack`.

    It records the column of each leaf in the packed tensor, and the nesting,
    batch dimensions and classes needed to rebuild the tensordict with
    :meth:`unpack`. It is frozen, hashable and compares by value, so a module
    can cache it and ``torch.compile`` can treat it as a constant.

    Attributes:
        keys (tuple of NestedKey): the leaf keys in column order.
        shapes (tuple of torch.Size): the feature shape of each leaf, that is
            its shape after the batch dimensions of the packed tensordict.
        dim (int): the position of the packed dimension in the packed tensor.
        nested (tuple): one ``(key, batch_shape, cls)`` entry per nested tensor
            collection, where ``batch_shape`` is its batch shape after the
            batch dimensions of the root and ``cls`` is its tensorclass, or
            ``None`` for a tensordict.
        device (torch.device or None): the device of the packed tensordict.
        cls (type or None): the tensorclass of the packed tensordict, or ``None``.
        offsets (tuple of int): the first column of each leaf.
        numel (int): the number of columns of the packed tensor.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> td = TensorDict(p=torch.zeros(5), v=torch.ones(5, 3), batch_size=[5])
        >>> flat, layout = td.pack()
        >>> layout.keys, layout.offsets, layout.numel
        (('p', 'v'), (0, 1), 4)
        >>> out = layout.unpack(flat[:2])
        >>> out["v"].shape
        torch.Size([2, 3])

    """

    keys: tuple[NestedKey, ...]
    shapes: tuple[torch.Size, ...]
    dim: int = -1
    nested: tuple[tuple[NestedKey, torch.Size, type | None], ...] = ()
    device: torch.device | None = None
    cls: type | None = None
    offsets: tuple[int, ...] = dataclasses.field(init=False)
    numel: int = dataclasses.field(init=False)

    def __post_init__(self):
        offsets = []
        numel = 0
        for shape in self.shapes:
            offsets.append(numel)
            numel += math.prod(shape)
        object.__setattr__(self, "offsets", tuple(offsets))
        object.__setattr__(self, "numel", numel)

    def unpack(self, flat: torch.Tensor) -> TensorDictBase:
        """Rebuilds a tensordict from a packed tensor.

        The leaves of the result are views of ``flat``. Its batch size is the
        shape of ``flat`` without the packed dimension, so ``flat`` may have
        other batch dimensions than the packed tensordict, for instance after
        an ``index_select``.

        Args:
            flat (torch.Tensor): a tensor with :attr:`numel` entries along
                dimension :attr:`dim`.

        Returns:
            a :class:`~tensordict.TensorDict`, or an instance of :attr:`cls`,
            on the device of ``flat`` if the packed tensordict had a device.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(p=torch.zeros(5), v=torch.ones(5, 3), batch_size=[5])
            >>> flat, layout = td.pack()
            >>> out = layout.unpack(flat)
            >>> assert (out == td).all()
            >>> out["p"].data_ptr() == flat.data_ptr()
            True

        """
        from tensordict._td import TensorDict

        ndim = flat.ndim
        dim = self.dim + ndim if self.dim < 0 else self.dim
        if not 0 <= dim < ndim:
            raise IndexError(
                f"dim {self.dim} is out of range for a tensor with {ndim} dimensions."
            )
        if flat.shape[dim] != self.numel:
            raise ValueError(
                f"Expected {self.numel} entries along dim {self.dim} of the packed "
                f"tensor, got {flat.shape[dim]}."
            )
        flat = flat.movedim(dim, -1)
        batch_size = flat.shape[:-1]
        device = flat.device if self.device is not None else None
        out = TensorDict(batch_size=batch_size, device=device)
        for key, batch_shape, _ in self.nested:
            out.set(key, TensorDict(batch_size=batch_size + batch_shape, device=device))
        for key, shape, offset in zip(self.keys, self.shapes, self.offsets):
            value = flat.narrow(-1, offset, math.prod(shape))
            value = value.unflatten(-1, shape) if shape else value.squeeze(-1)
            out.set(key, value)
        # children come after their parents in self.nested
        for key, _, cls in reversed(self.nested):
            if cls is not None:
                out.set(key, cls._from_tensordict(out.get(key)))
        if self.cls is not None:
            return self.cls._from_tensordict(out)
        return out


def _has_exclusive_keys(data) -> bool:
    # Whether data is a lazy stack whose members have different keys.
    from tensordict._lazy import LazyStackedTensorDict

    if _is_tensorclass(type(data)):
        data = data._tensordict
    return isinstance(data, LazyStackedTensorDict) and data._has_exclusive_keys


class _ShapeOps:
    """Operations on the batch shape: views, reshaping, stacking, splitting and padding."""

    @overload
    def expand(self, *shape: int) -> Self: ...

    @overload
    def expand(self, shape: torch.Size) -> Self: ...

    def expand(self, *args, **kwargs) -> Self:
        """Expands each tensor of the tensordict according to :meth:`torch.Tensor.expand`, ignoring the feature dimensions.

        The new shape is the new batch size. As for :meth:`torch.Tensor.expand`,
        it can add leading dimensions and expand the dimensions of size 1. The
        other dimensions must keep their size. The result holds views of the
        original tensors: no memory is allocated, and the expanded leaves are
        not contiguous. Call :meth:`~.clone` or :meth:`~.contiguous` before
        you write into them.

        Supports iterables to specify the shape.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     'a': torch.zeros(3, 4, 5),
            ...     'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
            >>> td_expand = td.expand(10, 3, 4)
            >>> assert td_expand.shape == torch.Size([10, 3, 4])
            >>> assert td_expand.get("a").shape == torch.Size([10, 3, 4, 5])

        """
        tensordict_dims = self.batch_dims
        shape = _get_shape_from_args(*args, **kwargs)

        # new shape dim check
        if len(shape) < len(self.shape):
            raise RuntimeError(
                f"the number of sizes provided ({len(shape)}) must be greater or equal to the number of "
                f"dimensions in the TensorDict ({tensordict_dims})"
            )
        if -1 in shape:
            shape = _resolve_expand_shape(shape, self.batch_size)

        # new shape compatibility check
        for old_dim, new_dim in zip(self.batch_size, shape[-tensordict_dims:]):
            if old_dim != 1 and new_dim != old_dim:
                raise RuntimeError(
                    "Incompatible expanded shape: The expanded shape length at non-singleton dimension should be same "
                    f"as the original length. target_shape = {shape}, existing_shape = {self.batch_size}"
                )

        if self._has_names():
            names = [None] * (len(shape) - tensordict_dims) + self.names
        else:
            names = None
        return self._expand(shape, names)

    def _expand(self, shape: torch.Size, names: list | None) -> Self:
        """Returns a tensordict whose batch size is ``shape``, by expanding the batch dims.

        Called by :meth:`expand` after it has resolved ``-1`` and checked that
        ``shape`` keeps each batch dim or expands a dim of size 1, and adds
        leading dims only. ``names`` are the names of the result, or ``None``.
        The leaves of the result are expanded views of the leaves of ``self``.

        The default expands each leaf and gives the result the lock of
        ``self``. ``LazyStackedTensorDict`` expands each member and, when the
        stack dim grows from 1, repeats its single member; its result is a
        ``TensorDict`` when the expanded members have the same keys and
        shapes.
        """
        tensordict_dims = self.batch_dims

        def _expand(tensor):
            tensor_shape = tensor.shape
            tensor_dims = len(tensor_shape)
            last_n_dims = tensor_dims - tensordict_dims
            if last_n_dims > 0:
                new_shape = (*shape, *tensor_shape[-last_n_dims:])
            else:
                new_shape = shape
            return tensor.expand(new_shape)

        return self._fast_apply(
            _expand,
            batch_size=shape,
            call_on_nested=True,
            names=names,
            propagate_lock=True,
        )

    def expand_as(self, other: TensorCollection | torch.Tensor) -> Self:
        """Broadcasts the shape of the tensordict to the shape of `other` and expands it accordingly.

        If the input is a tensor collection (tensordict or tensorclass),
        the leaves will be expanded on a one-to-one basis.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> td0 = TensorDict({
            ...     "a": torch.ones(3, 1, 4),
            ...     "b": {"c": torch.ones(3, 2, 1, 4)}},
            ...     batch_size=[3],
            ... )
            >>> td1 = TensorDict({
            ...     "a": torch.zeros(2, 3, 5, 4),
            ...     "b": {"c": torch.zeros(2, 3, 2, 6, 4)}},
            ...     batch_size=[2, 3],
            ... )
            >>> expanded = td0.expand_as(td1)
            >>> assert (expanded==1).all()
            >>> print(expanded)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([2, 3, 5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([2, 3, 2, 6, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([2, 3]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([2, 3]),
                device=None,
                is_shared=False)

        """
        if _is_tensor_collection(type(other)):

            def expand_as(x, y):
                return x.expand_as(y)

            return self.apply(expand_as, other, batch_size=other.batch_size)
        return self.expand(other.shape)

    def unbind(self, dim: int) -> tuple[TensorDictBase, ...]:
        """Returns a tuple of indexed tensordicts, unbound along the indicated dimension.

        The tensordicts of the result hold views of the original tensors.

        Args:
            dim (int): the batch dimension to remove. The result has
                ``self.batch_size[dim]`` tensordicts.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     'x': torch.arange(12).reshape(3, 4),
            ... }, batch_size=[3, 4])
            >>> td0, td1, td2 = td.unbind(0)
            >>> td0['x']
            tensor([0, 1, 2, 3])
            >>> td1['x']
            tensor([4, 5, 6, 7])

        """
        dim = _maybe_correct_neg_dim(dim, self.batch_size)
        results = self._unbind(dim)
        if self._is_memmap or self._is_shared:
            for result in results:
                result.lock_()
        return results

    def tensor_split(
        self,
        indices_or_sections: int | list[int] | tuple[int, ...] | torch.Tensor,
        dim=0,
    ) -> tuple[TensorDictBase, ...]:
        """Splits a TensorDict into multiple sub-tensordicts, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections.

        Args:
            indices_or_sections (int or List(int) or tuple(int) or 1D tensor of ints): If `indices_or_sections` is an integer
                `n` or a zero dimensional long tensordict with value `n`, input is split into `n` sections along dimension `dim`.
                If input is divisible by `n` along dimension `dim`, each section will be of equal size, `input.size(dim) / n`.
                If input is not divisible by `n`, the sizes of the first `int(input.size(dim) % n)` sections will have
                size `int(input.size(dim) / n) + 1`, and the rest will have size `int(input.size(dim) / n)`.
                If `indices_or_sections` is a list or tuple of ints, or a one-dimensional long tensor, then input is split
                along dimension `dim` at each of the indices in the list, tuple or tensor.
                For instance, `indices_or_sections=[2, 3]` and `dim=0` would result in the tensors `input[:2]`, `input[2:3]`, and `input[3:]`.
                As with :func:`torch.tensor_split`, negative indices count from the end, indices past the end are clamped,
                and an index smaller than the one before it gives an empty or an overlapping section.
                If `indices_or_sections` is a tensor, it must be a zero-dimensional or one-dimensional long tensor on the CPU.
            dim (int, optional): dimension along which to split the tensor. Default: 0

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 4, 2),
            ... }, batch_size=[3, 4])
            >>> td0, td1 = td.tensor_split(dim=-1, indices_or_sections=2)
            >>> td0['x']
            tensor([[[ 0,  1],
                     [ 2,  3]],
                    [[ 8,  9],
                     [10, 11]],
                    [[16, 17],
                     [18, 19]]])

        """
        if not isinstance(indices_or_sections, (int, list, tuple, torch.Tensor)):
            raise ValueError(
                "indices_or_sections must be an integer, a list of integers, or a 1D tensor of integers"
            )
        if isinstance(indices_or_sections, torch.Tensor):
            if (
                indices_or_sections.dtype != torch.long
                or indices_or_sections.device.type != "cpu"
                or indices_or_sections.ndim > 1
            ):
                raise ValueError(
                    "tensor_split: a tensor indices_or_sections must be a zero-dimensional or "
                    "one-dimensional long tensor on the CPU, but got a "
                    f"{indices_or_sections.ndim}-dimensional {indices_or_sections.dtype} tensor "
                    f"on {indices_or_sections.device}."
                )
            # A 0-d tensor gives an int (a number of sections), a 1-d one a list of indices.
            indices_or_sections = indices_or_sections.tolist()

        batch_size = self.batch_size
        dim = _maybe_correct_neg_dim(dim, batch_size)

        if self.ndim == 0:
            msg = "tensor_split: received a rank zero tensor, but expected a tensor of rank one or greater!"
            raise ValueError(msg)

        # Case 0 -- indices_or_sections is an integer or a scalar tensor n and a is split along dim into n parts of equal-ish length
        if isinstance(indices_or_sections, int):
            sections: int = indices_or_sections  # type: ignore[assignment]

            if sections <= 0:
                msg = f"tensor_split: number of sections must be greater than 0, but was {sections}"
                raise ValueError(msg)

            dim_size = self.shape[dim]
            min_split_size = math.floor(dim_size / sections)
            num_splits_one_extra = dim_size % sections

            split_sizes = []
            for split_idx in range(sections):
                split_size = (
                    min_split_size + 1
                    if (split_idx < num_splits_one_extra)
                    else min_split_size
                )
                split_sizes.append(split_size)

            return tuple(self.split(split_sizes, dim=dim))
        # Case 1 -- indices_or_sections is a sequence of integers or a 1D tensor describing the splits
        else:
            dim_size = self.shape[dim]
            # Read each index as a slice bound: negative indices count from the end,
            # and indices out of range are clamped, as in torch.tensor_split.
            bounds = [0]
            for index in indices_or_sections:
                index = operator.index(index)
                if index < 0:
                    index += dim_size
                bounds.append(min(max(index, 0), dim_size))
            bounds.append(dim_size)
            split_sizes = [stop - start for start, stop in zip(bounds[:-1], bounds[1:])]
            if all(size >= 0 for size in split_sizes):
                return tuple(self.split(split_sizes, dim=dim))
            # A decreasing index gives an empty section followed by one that overlaps
            # the previous one, which split cannot express.
            return tuple(
                self.narrow(dim, start, max(stop - start, 0))
                for start, stop in zip(bounds[:-1], bounds[1:])
            )

    @overload
    def unsqueeze(self, dim: int) -> Self: ...

    @_as_context_manager()
    def unsqueeze(self, *args, **kwargs):
        """Inserts a batch dimension of size 1 in every tensor, and returns them in a new tensordict.

        The tensors of the result are views of the original tensors.

        Args:
            dim (int): the position of the new dimension, in
                ``[-batch_dims - 1, batch_dims]`` as for :func:`torch.unsqueeze`.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 4, 2),
            ... }, batch_size=[3, 4])
            >>> td = td.unsqueeze(-2)
            >>> td.shape
            torch.Size([3, 1, 4])
            >>> td.get("x").shape
            torch.Size([3, 1, 4, 2])

        This operation can be used as a context manager too. Inside the
        ``with`` block, you work on the unsqueezed tensordict. When the block
        exits, the inverse operation (:meth:`~.squeeze`) runs, and the entries
        of the block are written back to the original tensordict. They replace
        the original tensors instead of being copied into them. The original
        tensordict must not be locked. If the block raises an exception,
        nothing is written back.

            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 4, 2),
            ... }, batch_size=[3, 4])
            >>> with td.unsqueeze(-2) as tds:
            ...     _ = tds.set("y", torch.zeros(3, 1, 4))
            >>> assert td.get("y").shape == (3, 4)

        """
        _lazy_legacy = lazy_legacy()

        if _lazy_legacy:
            return self._legacy_unsqueeze(*args, **kwargs)
        else:
            dim = _unsqueeze_dim(*args, **kwargs)
            batch_dims = self.batch_dims
            newdim = batch_dims + dim + 1 if dim < 0 else dim
            if newdim > batch_dims or newdim < 0:
                raise RuntimeError(
                    f"unsqueezing is allowed for dims comprised between "
                    f"`-td.batch_dims - 1` and `td.batch_dims` only. Got "
                    f"dim={dim} with a batch size of {self.batch_size}."
                )
            result = self._unsqueeze(newdim)
            if result._is_memmap or result._is_shared:
                result.lock_()
            return result

    def _legacy_unsqueeze(self, dim: int) -> Self:
        if dim < 0:
            dim = self.batch_dims + dim + 1

        if (dim > self.batch_dims) or (dim < 0):
            raise RuntimeError(
                f"unsqueezing is allowed for dims comprised between "
                f"`-td.batch_dims` and `td.batch_dims` only. Got "
                f"dim={dim} with a batch size of {self.batch_size}."
            )
        from tensordict._lazy import _UnsqueezedTensorDict

        return _UnsqueezedTensorDict(
            source=self,
            custom_op="unsqueeze",
            inv_op="squeeze",
            custom_op_kwargs={"dim": dim},
            inv_op_kwargs={"dim": dim},
        )

    @overload
    def squeeze(self, dim: int | None = None) -> Self: ...

    @_as_context_manager()
    def squeeze(self, *args, **kwargs):
        """Removes batch dimensions of size 1 from every tensor, and returns them in a new tensordict.

        The tensors of the result are views of the original tensors. Feature
        dimensions of size 1 are kept.

        Args:
            dim (int | None): the batch dimension to remove, in
                ``[-batch_dims, batch_dims - 1]``. If dim is
                ``None``, all singleton batch dimensions will be squeezed.
                Defaults to ``None``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 1, 4, 2),
            ... }, batch_size=[3, 1, 4])
            >>> td = td.squeeze()
            >>> td.shape
            torch.Size([3, 4])
            >>> td.get("x").shape
            torch.Size([3, 4, 2])

        This operation can be used as a context manager too. Inside the
        ``with`` block, you work on the squeezed tensordict. When the block
        exits, the inverse operation (:meth:`~.unsqueeze`) runs, and the
        entries of the block are written back to the original tensordict. They
        replace the original tensors instead of being copied into them. The
        original tensordict must not be locked. If the block raises an
        exception, nothing is written back. Pass ``dim``: ``with td.squeeze():``
        raises ``RuntimeError``, because the inverse operation needs the
        dimension.

            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 1, 4, 2),
            ... }, batch_size=[3, 1, 4])
            >>> with td.squeeze(1) as tds:
            ...     _ = tds.set("y", torch.zeros(3, 4))
            >>> assert td.get("y").shape == (3, 1, 4)

        """
        _lazy_legacy = lazy_legacy()

        if _lazy_legacy:
            return self._legacy_squeeze(*args, **kwargs)
        else:
            dim = _squeeze_dim(*args, **kwargs)
            if dim is not None:
                dim = _maybe_correct_neg_dim(dim, self.batch_size)
            if dim is not None and self.batch_size[dim] != 1:
                result = self
            else:
                result = self._squeeze(dim)
            if result._is_memmap or result._is_shared:
                result.lock_()
            return result

    def _legacy_squeeze(self, dim: int | None = None) -> Self:
        from tensordict._lazy import _SqueezedTensorDict

        if dim is None:
            size = self.size()
            if len(self.size()) == 1 or size.count(1) == 0:
                return self
            first_singleton_dim = size.index(1)

            squeezed_dict = _SqueezedTensorDict(
                source=self,
                custom_op="squeeze",
                inv_op="unsqueeze",
                custom_op_kwargs={"dim": first_singleton_dim},
                inv_op_kwargs={"dim": first_singleton_dim},
            )
            return squeezed_dict.squeeze(dim=None)

        if dim < 0:
            dim = self.batch_dims + dim

        if self.batch_dims and (dim >= self.batch_dims or dim < 0):
            raise RuntimeError(
                f"squeezing is allowed for dims comprised between 0 and "
                f"td.batch_dims only. Got dim={dim} and batch_size"
                f"={self.batch_size}."
            )

        if dim >= self.batch_dims or self.batch_size[dim] != 1:
            return self

        return _SqueezedTensorDict(
            source=self,
            custom_op="squeeze",
            inv_op="unsqueeze",
            custom_op_kwargs={"dim": dim},
            inv_op_kwargs={"dim": dim},
        )

    @overload
    def reshape(self, *shape: int): ...

    @overload
    def reshape(self, shape: list | tuple): ...

    def reshape(
        self,
        *args,
        **kwargs,
    ) -> Self:
        """Returns a tensordict whose batch dimensions are reshaped to ``shape``.

        Each leaf is reshaped with :meth:`torch.Tensor.reshape`, and keeps its
        feature dimensions. As with :meth:`torch.Tensor.reshape`, a leaf of
        the result is a view of the original leaf when the memory layout
        allows it, and a copy otherwise.

        Args:
            *shape (int): new batch size of the resulting tensordict, as
                integers or as one list, tuple or ``torch.Size``. One
                dimension can be ``-1``. The shape is positional only:
                ``reshape(shape=...)`` raises ``TypeError``.

        Keyword Args:
            inplace (bool, optional): If ``True``, this tensordict's identity
                and key set are preserved; each leaf is reshaped one at a
                time. Note that the underlying ``Tensor.reshape`` may return
                a view of the original leaf (when memory layout permits) or
                a copy (otherwise); in the view case ``inplace=True`` keeps
                the leaves sharing storage with the originals and the
                memory benefit does not materialize. Defaults to ``False``.

        Returns:
            A TensorDict with reshaped keys. When ``inplace=True`` this is
            ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     'x': torch.arange(12).reshape(3, 4),
            ... }, batch_size=[3, 4])
            >>> td = td.reshape(12)
            >>> print(td['x'])
            tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

        """
        inplace = kwargs.pop("inplace", False)
        if inplace:
            self._raise_if_inplace_shape_ops_unsupported("reshape")
        shape = _get_shape_from_args(*args, **kwargs)
        shape = torch.Size(_infer_size_impl(shape, self.batch_size.numel()))
        if shape == self.shape:
            return self
        batch_dims = self.batch_dims

        def _reshape(tensor):
            return tensor.reshape((*shape, *tensor.shape[batch_dims:]))

        if inplace:
            self._raise_if_holds_lazy_stack("reshape")

            def nested_fn(nested):
                nested.reshape((*shape, *nested.batch_size[batch_dims:]), inplace=True)

            self._inplace_rebind_leaves(_reshape, nested_fn, torch.Size(shape))
            if self._has_names():
                # as without inplace: the reshaped dims have no name
                self._erase_names()
            return self
        return self._reshape(shape, None)

    def _reshape(
        self,
        shape: torch.Size,
        names: list | None,
        leaf_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> Self:
        """Returns a tensordict whose batch size is ``shape``.

        Called by :meth:`reshape`, :meth:`flatten` and :meth:`unflatten` for
        ``inplace=False``, after they have resolved ``-1`` and checked that
        ``shape`` has as many elements as the batch and differs from the batch
        size. Each leaf keeps its feature dims and is a view of the original
        leaf when the layout allows it, a copy otherwise. ``names`` (one per
        dim of ``shape``, or ``None``) are the names of the result. The result
        has the lock of ``self``.

        The default applies ``leaf_fn`` to each leaf, or reshapes each leaf
        when it is ``None``: :meth:`flatten` and :meth:`unflatten` pass
        :func:`torch.flatten` and :func:`torch.unflatten` with their dims,
        which are faster than a reshape. ``LazyStackedTensorDict`` regroups its
        members when the new shape only flattens or unflattens dims, which
        keeps the stack lazy, and reshapes each leaf otherwise;
        ``NonTensorStack`` moves its entries instead.
        """
        if leaf_fn is None:
            batch_dims = self.batch_dims

            def leaf_fn(tensor):
                return tensor.reshape((*shape, *tensor.shape[batch_dims:]))

        return self._fast_apply(
            leaf_fn,
            batch_size=shape,
            call_on_nested=True,
            propagate_lock=True,
            names=names,
        )

    def _raise_if_inplace_shape_ops_unsupported(self, op_name: str) -> None:
        if self._inplace_shape_ops_unsupported:
            raise NotImplementedError(
                f"{op_name}(inplace=True) is not supported on "
                f"{type(self).__name__}; call .to_tensordict() first or use "
                "inplace=False."
            )

    def repeat_interleave(
        self,
        repeats: torch.Tensor | int,
        dim: int | None = None,
        *,
        output_size: int | None = None,
        inplace: bool = False,
    ) -> Self:
        """Repeat elements of a TensorDict.

        .. warning:: This is different from :meth:`~torch.Tensor.repeat` but similar to :func:`numpy.repeat`.

        Args:
            repeats (torch.Tensor or int): The number of repetitions for each element. `repeats` is broadcast to fit
                the shape of the given axis.
            dim (int, optional): The dimension along which to repeat values. By default, use the flattened input
                array, and return a flat output array. A ``dim`` past the batch dims warns, and is used on the
                dims of the leaves. Starting with TensorDict 0.17, the default will change to raising an
                ``IndexError``, as for a tensor.

        Keyword Args:
            output_size (int, optional): Total output size for the given axis (e.g. sum of repeats). If given, it
                will avoid stream synchronization needed to calculate output shape of the tensordict.
            inplace (bool, optional): If ``True``, this tensordict's identity
                and key set are preserved; each leaf storage is replaced by
                its repeated counterpart one leaf at a time. Not supported on
                :class:`~tensordict.LazyStackedTensorDict` (call
                ``to_tensordict()`` first). Defaults to ``False``.

        Returns:
            Repeated TensorDict which has the same shape as input, except along the given axis.

        Examples:
            >>> import torch
            >>>
            >>> from tensordict import TensorDict
            >>>
            >>> td = TensorDict(
            ...     {
            ...         "a": torch.randn(3, 4, 5),
            ...         "b": TensorDict({
            ...             "c": torch.randn(3, 4, 10, 1),
            ...             "a string": "a string!",
            ...         }, batch_size=[3, 4, 10])
            ...     }, batch_size=[3, 4],
            ... )
            >>> print(td.repeat_interleave(2, dim=0))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([6, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            a string: NonTensorData(data=a string!, batch_size=torch.Size([6, 4, 10]), device=None),
                            c: Tensor(shape=torch.Size([6, 4, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([6, 4, 10]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([6, 4]),
                device=None,
                is_shared=False)

        """
        if inplace:
            self._raise_if_inplace_shape_ops_unsupported("repeat_interleave")
        ndim = self.batch_dims
        if ndim == 0:
            if inplace:
                raise RuntimeError(
                    "repeat_interleave(inplace=True) is not supported on a "
                    "scalar tensordict because the operation changes ndim."
                )
            return self.unsqueeze(0).repeat_interleave(
                repeats=repeats, dim=dim, output_size=output_size
            )
        if dim is None:
            if inplace:
                raise RuntimeError(
                    "repeat_interleave(inplace=True) requires an explicit dim; "
                    "the dim=None path reshapes the tensordict to 1D first, "
                    "which would change ndim."
                )
            if ndim > 1:
                return self.reshape(-1).repeat_interleave(repeats, dim=0)
            return self.repeat_interleave(repeats, dim=0)
        dim_corrected = dim if dim >= 0 else ndim + dim
        if not (dim_corrected >= 0):
            raise ValueError(
                f"dim {dim} is out of range for tensordict with shape {self.shape}."
            )
        if dim_corrected >= ndim:
            _warn_dim_past_batch_dims("repeat_interleave")
        if inplace:
            self._raise_if_holds_lazy_stack("repeat_interleave")

            def rep(leaf):
                return leaf.repeat_interleave(
                    repeats=repeats, dim=dim_corrected, output_size=output_size
                )

            def nested_fn(nested):
                nested.repeat_interleave(
                    repeats=repeats,
                    dim=dim_corrected,
                    output_size=output_size,
                    inplace=True,
                )

            return self._inplace_rebind_leaves(
                rep,
                nested_fn,
                _repeat_interleave_batch_size(self.batch_size, repeats, dim_corrected),
            )
        return self._repeat_interleave(repeats, dim_corrected, output_size)

    def _repeat_interleave(
        self,
        repeats: torch.Tensor | int,
        dim: int,
        output_size: int | None,
    ) -> Self:
        """Repeats the elements of the batch along ``dim``, as :meth:`torch.Tensor.repeat_interleave` does.

        Called by :meth:`repeat_interleave` for ``inplace=False`` after it has
        handled a 0-d tensordict and ``dim=None``, and made ``dim``
        non-negative (warning for a ``dim`` past the batch dims, until 0.17).
        The result keeps the names of ``self``.

        The default repeats each leaf. ``LazyStackedTensorDict`` repeats its
        members along the stack dim, after checking ``repeats`` as torch
        does, and calls each member along the other dims.
        """

        def rep(leaf):
            return leaf.repeat_interleave(
                repeats=repeats, dim=dim, output_size=output_size
            )

        return self._fast_apply(
            rep,
            batch_size=_repeat_interleave_batch_size(self.batch_size, repeats, dim),
            call_on_nested=True,
            propagate_lock=True,
            names=self._maybe_names(),
        )

    @overload
    def repeat(self, repeats: torch.Size, *, inplace: bool = False): ...

    def repeat(self, *repeats: int, inplace: bool = False) -> Self:
        """Repeats this tensor along the specified dimensions.

        Unlike :meth:`~.expand()`, this function copies the tensor's data.

        .. warning:: :meth:`~.repeat` behaves differently from :func:`~numpy.repeat`, but is more similar to
            :func:`numpy.tile`. For the operator similar to :func:`numpy.repeat`, see :meth:`~tensordict.TensorDictBase.repeat_interleave`.

        Args:
            repeats (torch.Size, int..., tuple of int or list of int): The number of times to repeat this tensor along
                each dimension.

        Keyword Args:
            inplace (bool, optional): If ``True``, this tensordict's identity
                and key set are preserved; each leaf storage is replaced by
                its repeated counterpart one leaf at a time, keeping peak
                memory close to ``1x``. ``LazyStackedTensorDict`` is not
                supported in this mode (the stack dim's repeat factor would
                rebuild the stack's constituents list) — call
                ``to_tensordict()`` first. Defaults to ``False``.

        Examples:
            >>> import torch
            >>>
            >>> from tensordict import TensorDict
            >>>
            >>> td = TensorDict(
            ...     {
            ...         "a": torch.randn(3, 4, 5),
            ...         "b": TensorDict({
            ...             "c": torch.randn(3, 4, 10, 1),
            ...             "a string": "a string!",
            ...         }, batch_size=[3, 4, 10])
            ...     }, batch_size=[3, 4],
            ... )
            >>> print(td.repeat(1, 2))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 8, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            a string: NonTensorData(data=a string!, batch_size=torch.Size([3, 8, 10]), device=None),
                            c: Tensor(shape=torch.Size([3, 8, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 8, 10]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 8]),
                device=None,
                is_shared=False)

        """
        if len(repeats) == 1 and not isinstance(repeats[0], int):
            repeats = repeats[0]
            # a torch.Size is a tuple
            if isinstance(repeats, (list, tuple)):
                return self.repeat(*repeats, inplace=inplace)
            if isinstance(repeats, torch.Tensor):
                # This will cause cuda to sync, which may not be desirable
                return self.repeat(*repeats.tolist())
            raise ValueError(
                f"repeats must be a sequence of integers, a tensor or a torch.Size object. Got {type(repeats)} instead."
            )
        if len(repeats) != self.ndimension():
            raise ValueError(
                f"The number of repeat elements must match the number of dimensions of the tensordict. Got {len(repeats)} but ndim={self.ndimension()}."
            )
        if inplace:
            if self._lazy:
                raise NotImplementedError(
                    "repeat(inplace=True) is not supported on LazyStackedTensorDict; "
                    "call .to_tensordict() first or use inplace=False."
                )
            self._raise_if_holds_lazy_stack("repeat")
            new_batch_size = torch.Size(
                [s * r for s, r in zip(self.batch_size, repeats)]
            )
            ndim = self.ndim

            def leaf_fn(leaf):
                return leaf.repeat(*repeats, *((1,) * (leaf.ndim - ndim)))

            def nested_fn(nested):
                nested.repeat(
                    *repeats, *((1,) * (nested.batch_dims - ndim)), inplace=True
                )

            return self._inplace_rebind_leaves(leaf_fn, nested_fn, new_batch_size)
        return self._repeat(*repeats)

    def _repeat(self, *repeats: int) -> TensorCollection:
        new_batch_size = torch.Size([i * r for i, r in zip(self.batch_size, repeats)])

        def rep(leaf):
            return leaf.repeat(*repeats, *((1,) * (leaf.ndim - self.ndim)))

        return self._fast_apply(
            rep,
            batch_size=new_batch_size,
            call_on_nested=True,
            propagate_lock=True,
            names=self._maybe_names(),
        )

    def pad(
        self,
        pad_size: Sequence[int],
        value: float = 0.0,
        inplace: bool = False,
        safe: bool = True,
    ) -> Self:
        """Pads this tensordict along the batch dimensions with a constant value.

        Args:
            pad_size (Sequence[int]): The padding size, applied to the batch
                dimensions starting from the first. For a tensordict of
                ``ndim`` batch dimensions, ``pad_size`` is a sequence of even
                length up to ``2 * ndim`` formatted as
                ``(dim0_left, dim0_right, dim1_left, dim1_right, ...)``.
            value (float, optional): The fill value. Defaults to ``0.0``.
            inplace (bool, optional): If ``True``, this tensordict's object
                identity and key set are preserved; each leaf storage is
                replaced by its padded counterpart one leaf at a time, keeping
                peak memory close to ``1x`` instead of ``2x``. The leaf
                tensors themselves are still freshly allocated because
                padding necessarily grows shapes. Defaults to ``False``.

                .. warning::
                    If ``inplace=True`` and a later leaf's pad fails (e.g.
                    OOM), the tensordict is left in an inconsistent state.
                    ``safe=True`` (the default) catches the user-error class
                    of failures before any mutation occurs.
            safe (bool, optional): If ``True``, validate that the operation
                would succeed for every leaf before any mutation occurs. Set
                to ``False`` to skip the pre-flight walk when the inputs are
                known to be valid. Defaults to ``True``.

        Returns:
            The padded tensordict. When ``inplace=True`` this is ``self``.

        .. note::
            On a lazy stack whose tensordicts have no names, the result of
            ``inplace=False`` has no names, even when the stack dimension has
            a name. See :func:`~tensordict.pad`.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.zeros(3, 4)}, batch_size=[3, 4])
            >>> td.pad([0, 0, 0, 1]).batch_size
            torch.Size([3, 5])
            >>> td.pad([0, 0, 0, 1], inplace=True) is td
            True
            >>> td.batch_size
            torch.Size([3, 5])
        """
        from tensordict.functional import pad as _pad

        return _pad(self, pad_size, value=value, inplace=inplace, safe=safe)

    def cat_tensors(
        self,
        *keys: NestedKey,
        out_key: NestedKey,
        dim: int = 0,
        keep_entries: bool = False,
    ) -> Self:
        """Concatenates entries into a new entry and possibly remove the original values.

        Args:
            keys (sequence of NestedKey): entries to concatenate.

        Keyword Arguments:
            out_key (NestedKey): new key name for the concatenated inputs.
            keep_entries (bool, optional): if ``False``, entries in ``keys`` will be deleted.
                Defaults to ``False``.
            dim (int, optional): the dimension along which the concatenation must occur.
                Defaults to ``0``.

        Returns: self

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(1), b=torch.ones(1))
            >>> _ = td.cat_tensors("a", "b", out_key="c")
            >>> assert "a" not in td
            >>> assert (td["c"] == torch.tensor([0, 1])).all()

        """
        if keep_entries:
            entries = [self.get(key) for key in keys]
        else:
            entries = [self.pop(key) for key in keys]
        return self.set(out_key, torch.cat(entries, dim=dim))

    def stack_tensors(
        self,
        *keys: NestedKey,
        out_key: NestedKey,
        dim: int = 0,
        keep_entries: bool = False,
    ) -> Self:
        """Stacks entries into a new entry and possibly remove the original values.

        Args:
            keys (sequence of NestedKey): entries to stack.

        Keyword Arguments:
            out_key (NestedKey): new key name for the stacked inputs.
            keep_entries (bool, optional): if ``False``, entries in ``keys`` will be deleted.
                Defaults to ``False``.
            dim (int, optional): the dimension along which the stack must occur.
                Defaults to ``0``.

        Returns: self

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(()), b=torch.ones(()))
            >>> _ = td.stack_tensors("a", "b", out_key="c")
            >>> assert "a" not in td
            >>> assert (td["c"] == torch.tensor([0, 1])).all()

        """
        if keep_entries:
            entries = [self.get(key) for key in keys]
        else:
            entries = [self.pop(key) for key in keys]
        return self.set(out_key, torch.stack(entries, dim=dim))

    def cat_from_tensordict(
        self,
        dim: int = 0,
        *,
        sorted: bool | List[NestedKey] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:  # noqa: D417
        """Concatenates all entries of a tensordict in a single tensor.

        Args:
            dim (int, optional): the dimension along which the entries should be concatenated.

        Keyword Args:
            sorted (bool or list of NestedKeys): if ``True``, the entries will be concatenated in alphabetical order.
                If ``False`` (default), the dict order will be used. Alternatively, a list of key names can be provided
                and the tensors will be concatenated accordingly. This incurs some overhead as the list of keys will
                be checked against the list of leaf names in the tensordict.
            out (torch.Tensor, optional): an optional destination tensor for the cat operation.

        """
        if sorted in (None, False):
            tensors = list(self.values(True, True))
        elif sorted in (True,):
            tensors = list(self.values(True, True, sort=True))
        else:
            keys = unravel_key_list(sorted)
            if set(keys) != set(self.keys(True, True)):
                raise RuntimeError(
                    "The provided set of keys differs from the tensordict list of keys."
                )
            tensors = [self.get(key) for key in keys]
        return torch.cat(tensors, dim, out=out)

    def pack(
        self, dim: int = -1, *, layout: PackLayout | None = None
    ) -> tuple[torch.Tensor, PackLayout]:
        """Packs all the leaves of a tensordict into one tensor of features.

        Each leaf is flattened after the batch dimensions, and the leaves are
        concatenated in the order of ``keys(True, True, sort=True)``, so the
        columns depend on the set of keys only. Leaves can have different
        ranks. The result has shape ``(*batch_size, F)`` for ``dim=-1``, where
        ``F`` is the sum of the numbers of features of the leaves, and it is
        contiguous for any ``dim``. :meth:`PackLayout.unpack
        <tensordict.PackLayout.unpack>` inverts it.

        All the leaves must share one dtype and one device. Non-tensor data
        cannot be packed: drop it with :meth:`filter_non_tensor_data` first.
        A lazy stack packs like its dense equivalent if all its members have
        the same keys and feature shapes.

        Args:
            dim (int, optional): the position of the packed dimension in the
                result, in ``[-(batch_dims + 1), batch_dims]`` as for
                :func:`torch.stack`. Defaults to ``-1``.

        Keyword Args:
            layout (PackLayout, optional): a layout to pack into. The columns
                then follow its key order, and a missing, extra or reshaped
                leaf raises. ``dim`` must match the layout's dim.

        Returns:
            the packed tensor and its :class:`~tensordict.PackLayout`.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     p=torch.zeros(5),
            ...     v=torch.ones(5, 3),
            ...     nested={"stress": torch.ones(5, 3, 3)},
            ...     batch_size=[5],
            ... )
            >>> flat, layout = td.pack()
            >>> flat.shape
            torch.Size([5, 13])
            >>> layout.keys
            (('nested', 'stress'), 'p', 'v')
            >>> out = layout.unpack(flat.index_select(0, torch.tensor([0, 2])))
            >>> out.batch_size
            torch.Size([2])
            >>> flat2, _ = td.pack(layout=layout)
            >>> assert (flat2 == flat).all()

        """
        batch_size = self.batch_size
        batch_dims = len(batch_size)
        if not -batch_dims - 1 <= dim <= batch_dims:
            raise IndexError(
                f"dim {dim} is out of range for packing a tensordict with "
                f"{batch_dims} batch dimensions."
            )
        pos = dim + batch_dims + 1 if dim < 0 else dim
        if _has_exclusive_keys(self):
            raise RuntimeError(
                "Cannot pack a lazy stack whose members have different keys."
            )
        leaves = {}
        nested = []
        for key, value in self.items(
            True, False, is_leaf=_is_leaf_nontensor, sort=True
        ):
            if is_non_tensor(value):
                raise TypeError(
                    f"Cannot pack the non-tensor entry {key!r}. Drop the "
                    "non-tensor entries with filter_non_tensor_data() first."
                )
            if _is_tensor_collection(type(value)):
                if _has_exclusive_keys(value):
                    raise RuntimeError(
                        f"Cannot pack the lazy stack at {key!r}: its members "
                        "have different keys."
                    )
                cls = type(value)
                nested.append(
                    (
                        key,
                        value.batch_size[batch_dims:],
                        cls if _is_tensorclass(cls) else None,
                    )
                )
            else:
                leaves[key] = value
        if layout is not None:
            layout_pos = layout.dim + batch_dims + 1 if layout.dim < 0 else layout.dim
            if layout_pos != pos:
                raise ValueError(
                    f"Cannot pack along dim {dim} with a layout packed along "
                    f"dim {layout.dim}."
                )
            if set(leaves) != set(layout.keys):
                raise KeyError(
                    "The leaves of the tensordict differ from the keys of the "
                    f"layout: {sorted(leaves, key=str)} vs "
                    f"{sorted(layout.keys, key=str)}."
                )
            leaves = {key: leaves[key] for key in layout.keys}
        keys = tuple(leaves)
        shapes = tuple(value.shape[batch_dims:] for value in leaves.values())
        if layout is not None and shapes != layout.shapes:
            raise ValueError(
                "The feature shapes of the leaves differ from those of the "
                f"layout: {shapes} vs {layout.shapes}."
            )
        tensors = []
        dtype = device = None
        for key, value in leaves.items():
            if dtype is None:
                dtype, device = value.dtype, value.device
            elif value.dtype != dtype:
                raise TypeError(
                    f"Cannot pack leaves of different dtypes: {dtype} and "
                    f"{value.dtype} at {key!r}. Cast the tensordict first, e.g. "
                    "with float()."
                )
            elif value.device != device:
                raise RuntimeError(
                    f"Cannot pack leaves on different devices: {device} and "
                    f"{value.device} at {key!r}."
                )
            value = (
                value.flatten(batch_dims)
                if value.ndim > batch_dims
                else value[..., None]
            )
            tensors.append(value.movedim(-1, pos))
        if tensors:
            # cat follows the memory format of its inputs, which movedim makes
            # channels-last for some ranks
            flat = torch.cat(tensors, pos).contiguous()
        else:
            flat = torch.empty(
                (*batch_size[:pos], 0, *batch_size[pos:]), device=self.device
            )
        cls = type(self)
        return flat, PackLayout(
            keys=keys,
            shapes=shapes,
            dim=dim,
            nested=tuple(nested),
            device=self.device,
            cls=cls if _is_tensorclass(cls) else None,
        )

    def stack_from_tensordict(
        self,
        dim: int = 0,
        *,
        sorted: bool | List[NestedKey] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:  # noqa: D417
        """Stacks all entries of a tensordict in a single tensor.

        Args:
            dim (int, optional): the dimension along which the entries should be stacked.

        Keyword Args:
            sorted (bool or list of NestedKeys): if ``True``, the entries will be stacked in alphabetical order.
                If ``False`` (default), the dict order will be used. Alternatively, a list of key names can be provided
                and the tensors will be stacked accordingly. This incurs some overhead as the list of keys will
                be checked against the list of leaf names in the tensordict.
            out (torch.Tensor, optional): an optional destination tensor for the stack operation.

        """
        if sorted in (None, False):
            tensors = list(self.values(True, True))
        elif sorted in (True,):
            tensors = list(self.values(True, True, sort=True))
        else:
            keys = unravel_key_list(sorted)
            if set(keys) != set(self.keys(True, True)):
                raise RuntimeError(
                    "The provided set of keys differs from the tensordict list of keys."
                )
            tensors = [self.get(key) for key in keys]
        return torch.stack(tensors, dim, out=out)

    @classmethod
    def stack(cls, input, dim: int = 0, *, out=None):
        """Stacks tensordicts into a single tensordict along the given dimension.

        This call is equivalent to calling :func:`torch.stack` but is compatible with torch.compile.
        The tensordicts must have the same batch size and the same keys.
        Otherwise, the call raises ``RuntimeError``. To stack tensordicts with
        different keys or shapes, use :meth:`~.lazy_stack`. See
        :ref:`stacking`.

        Args:
            input (sequence of TensorDictBase): the tensordicts to stack.
            dim (int, optional): the new batch dimension. Defaults to ``0``.

        Keyword Args:
            out (TensorDictBase, optional): a tensordict to write the result
                into.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> a = TensorDict(x=torch.zeros(3), batch_size=[3])
            >>> b = TensorDict(x=torch.ones(3), batch_size=[3])
            >>> TensorDict.stack([a, b]).batch_size
            torch.Size([2, 3])

        """
        from tensordict._torch_func import _stack

        if not _is_tensor_collection(type(input[0])):
            return torch.stack(input, dim, out=out)
        return _stack(input, dim, out=out)

    @classmethod
    def cat(cls, input, dim: int = 0, *, out=None):
        """Concatenates tensordicts into a single tensordict along the given dimension.

        This call is equivalent to calling :func:`torch.cat` but is compatible with torch.compile.
        The tensordicts must have the same keys, and the same batch size
        except along ``dim``.

        Args:
            input (sequence of TensorDictBase): the tensordicts to concatenate.
            dim (int, optional): the batch dimension to concatenate along.
                Defaults to ``0``.

        Keyword Args:
            out (TensorDictBase, optional): a tensordict to write the result
                into.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> a = TensorDict(x=torch.zeros(3), batch_size=[3])
            >>> b = TensorDict(x=torch.ones(3), batch_size=[3])
            >>> TensorDict.cat([a, b]).batch_size
            torch.Size([6])

        """
        from tensordict._torch_func import _cat

        if not _is_tensor_collection(type(input[0])):
            return torch.cat(input, dim, out=out)
        return _cat(input, dim, out=out)

    @classmethod
    def lazy_stack(cls, input, dim: int = 0, *, out=None, **kwargs):
        """Creates a lazy stack of tensordicts.

        A lazy stack keeps the tensordicts as they are and stacks their values
        only when you read them. The tensordicts can have different keys and
        different feature shapes. See :ref:`stacking`, and
        :func:`tensordict.lazy_stack` for the arguments.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> a = TensorDict(x=torch.zeros(3), batch_size=[3])
            >>> b = TensorDict(y=torch.ones(3), batch_size=[3])
            >>> stack = TensorDict.lazy_stack([a, b])
            >>> type(stack).__name__, stack.batch_size
            ('LazyStackedTensorDict', torch.Size([2, 3]))
        """
        from tensordict._lazy import LazyStackedTensorDict

        return LazyStackedTensorDict.lazy_stack(input, dim=dim, out=out, **kwargs)

    @classmethod
    def maybe_dense_stack(cls, input, dim: int = 0, *, out=None, **kwargs):
        """Attempts to make a dense stack of tensordicts, and falls back on lazy stack when required.

        The result is a :class:`~tensordict.TensorDict` if the tensordicts
        have the same keys and shapes, and a
        :class:`~tensordict.LazyStackedTensorDict` otherwise. See
        :ref:`stacking`, and :func:`tensordict.maybe_dense_stack` for the
        arguments.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> a = TensorDict(x=torch.zeros(3), batch_size=[3])
            >>> b = TensorDict(x=torch.ones(3), batch_size=[3])
            >>> c = TensorDict(y=torch.ones(3), batch_size=[3])
            >>> type(TensorDict.maybe_dense_stack([a, b])).__name__
            'TensorDict'
            >>> type(TensorDict.maybe_dense_stack([a, c])).__name__
            'LazyStackedTensorDict'
        """
        from tensordict._lazy import LazyStackedTensorDict

        return LazyStackedTensorDict.maybe_dense_stack(
            input, dim=dim, out=out, **kwargs
        )

    def chunk(self, chunks: int, dim: int = 0) -> tuple[TensorCollection, ...]:
        """Splits a tensordict into the specified number of chunks, if possible.

        Each chunk is a view of the input tensordict.

        Args:
            chunks (int): number of chunks to return
            dim (int, optional): dimension along which to split the
                tensordict. Default is 0.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 4, 2),
            ... }, batch_size=[3, 4])
            >>> td0, td1 = td.chunk(dim=-1, chunks=2)
            >>> td0['x']
            tensor([[[ 0,  1],
                     [ 2,  3]],
                    [[ 8,  9],
                     [10, 11]],
                    [[16, 17],
                     [18, 19]]])

        """
        if chunks < 1:
            raise ValueError(
                f"chunks must be a strictly positive integer, got {chunks}."
            )
        batch_size = self.batch_size
        dim = _maybe_correct_neg_dim(dim, batch_size)
        if not isinstance(chunks, int):
            # numpy and 0-d tensor integers
            chunks = operator.index(chunks)
        max_size = batch_size[dim]
        if max_size == 0:
            # torch.chunk returns `chunks` empty pieces along a zero-size dim
            return self.split([0] * chunks, dim)
        return self._chunk(chunks, dim)

    def _chunk(self, chunks: int, dim: int) -> tuple[TensorCollection, ...]:
        # chunk with a positive int `chunks` along a non-negative `dim` of
        # non-zero size: torch.chunk splits with the upper-rounded size
        return self.split(-(self.batch_size[dim] // -chunks), dim)

    def split(
        self, split_size: int | list[int], dim: int = 0
    ) -> tuple[TensorDictBase, ...]:
        """Splits each tensor in the TensorDict with the specified size in the given dimension, like `torch.split`.

        Returns a list of ``TensorDict`` instances with the view of split chunks of items.

        On a :class:`~tensordict.LazyStackedTensorDict`, sizes along the stack
        dim that are negative or do not add up to the stack size emit a
        :class:`FutureWarning`, and the pieces miss members or have a wrong
        batch size. Starting with TensorDict 0.17, the default will change to
        raising a ``RuntimeError``, as ``TensorDict.split`` does.

        Args:
            split_size (int or List(int)): size of a single chunk or list of sizes for each chunk.
            dim (int): dimension along which to split the tensor.

        Returns:
            A list of TensorDict with specified size in given dimension.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     'x': torch.arange(12).reshape(3, 4),
            ... }, batch_size=[3, 4])
            >>> td0, td1 = td.split([1, 2], dim=0)
            >>> print(td0['x'])
            tensor([[0, 1, 2, 3]])
        """
        WRONG_TYPE = "split(): argument 'split_size' must be int or list of ints"
        batch_size = self.batch_size
        dim = _maybe_correct_neg_dim(dim, batch_size)
        max_size = batch_size[dim]
        if isinstance(split_size, int):
            if split_size < 0 or (split_size == 0 and max_size != 0):
                raise ValueError(
                    f"TensorDict.split: split_size must be positive, got {split_size}."
                )
            if max_size == 0:
                # torch.split returns one empty piece along a zero-size dim
                split_sizes = [0]
            else:
                split_size = min(split_size, max_size)
                num_full, rest = divmod(max_size, split_size)
                split_sizes = [split_size] * num_full + ([rest] if rest else [])
        elif isinstance(split_size, (list, tuple, Iterable)):
            # LazyStackedTensorDict.split took any iterable of sizes, such as
            # a tensor, a numpy array or a list of 0-d tensors
            try:
                split_sizes = [operator.index(x) for x in split_size]
            except TypeError:
                raise TypeError(WRONG_TYPE) from None
        else:
            raise TypeError(WRONG_TYPE)
        return self._split(split_sizes, dim)

    def _split(self, split_sizes: list[int], dim: int) -> tuple[TensorDictBase, ...]:
        """Splits the batch along ``dim`` into pieces of the sizes ``split_sizes``.

        Called by :meth:`split` (and so by the default :meth:`_chunk`) after it
        has checked the type of ``split_size``, made ``dim`` non-negative and
        turned an int ``split_size`` into the list of sizes. The default
        raises, as :func:`torch.split` does, when the sizes are empty or do
        not add up to ``batch_size[dim]``; a ``LazyStackedTensorDict`` split
        along its stack dim warns instead until TensorDict 0.17. Each piece
        keeps the names, device and lock of ``self``, and its leaves are views
        of the leaves of ``self``.

        The default splits each leaf. ``LazyStackedTensorDict`` slices its
        list of members along the stack dim and splits each member along the
        other dims.
        """
        # we must use slices to keep the storage of the tensors
        batch_size = self.batch_size
        max_size = batch_size[dim]
        if len(split_sizes) == 0:
            if max_size == 0:
                # torch.split returns no pieces along a zero-size dim
                return ()
            raise RuntimeError("Insufficient number of elements in split_size.")
        num_splits = len(split_sizes)
        splits = {
            k: (v,) * num_splits if _is_unbatched(v) else v.split(split_sizes, dim)
            for k, v in self.items()
        }
        segments = _create_segments_from_list(split_sizes, max_size)
        names = self._maybe_names()
        batch_sizes = [
            torch.Size(
                tuple(d if i != dim else end - start for i, d in enumerate(batch_size))
            )
            for start, end in segments
        ]

        splits = [
            {k: v[ss] for k, v in splits.items()} for ss in range(len(batch_sizes))
        ]
        for split, bsz in _zip_strict(splits, batch_sizes):
            for key, value in split.items():
                if _is_unbatched(value):
                    split[key] = value._with_batch_size(bsz)
        device = self.device
        is_shared = self._is_shared
        is_memmap = self._is_memmap
        is_locked = self.is_locked
        result = tuple(
            self._new_unsafe(
                source=split,
                batch_size=bsz,
                names=names,
                device=device,
                lock=is_locked,
                is_shared=is_shared,
                is_memmap=is_memmap,
            )
            for split, bsz in _zip_strict(splits, batch_sizes)
        )
        return result

    def gather(
        self,
        dim: int,
        index: Tensor,
        out: T | None = None,
        *,
        inplace: bool = False,
    ) -> Self:
        """Gathers values along an axis specified by `dim`.

        Args:
            dim (int): the dimension along which collect the elements
            index (torch.Tensor): a long tensor which number of dimension matches
                the one of the tensordict with only one dimension differing between
                the two (the gathering dimension). Its elements refer to the
                index to be gathered along the required dimension.
            out (TensorDictBase, optional): a destination tensordict. It must
                have the same shape as the index.

        Keyword Args:
            inplace (bool, optional): If ``True``, this tensordict's identity
                and key set are preserved; each leaf storage is replaced by
                its gathered counterpart one leaf at a time, keeping peak
                memory close to ``1x``. Requires ``index.ndim ==
                self.batch_dims`` (the result must have the same number of
                batch dims as the input). Mutually exclusive with ``out``.
                Not supported on :class:`~tensordict.LazyStackedTensorDict`.
                Defaults to ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     {"a": torch.randn(3, 4, 5),
            ...      "b": TensorDict({"c": torch.zeros(3, 4, 5)}, [3, 4, 5])},
            ...     [3, 4])
            >>> index = torch.randint(4, (3, 2))
            >>> td_gather = td.gather(dim=1, index=index)
            >>> print(td_gather)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 2, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 2, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 2, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 2]),
                device=None,
                is_shared=False)

        Gather keeps the dimension names.

        Examples:
            >>> td.names = ["a", "b"]
            >>> td_gather = td.gather(dim=1, index=index)
            >>> td_gather.names
            ['a', 'b']
        """
        if inplace:
            if out is not None:
                raise ValueError(
                    "`out` and `inplace=True` are mutually exclusive in gather."
                )
            if self._lazy:
                raise NotImplementedError(
                    "gather(inplace=True) is not supported on LazyStackedTensorDict; "
                    "call .to_tensordict() first or use inplace=False."
                )
            self._raise_if_holds_lazy_stack("gather")
            if index.ndim != self.batch_dims:
                raise NotImplementedError(
                    f"gather(inplace=True) requires index.ndim == self.batch_dims "
                    f"so the result keeps the same number of batch dims; got "
                    f"index.ndim={index.ndim} and batch_dims={self.batch_dims}. "
                    f"Use inplace=False instead."
                )
            dim_corrected = dim if dim >= 0 else self.batch_dims + dim
            if dim_corrected < 0 or dim_corrected >= self.batch_dims:
                raise RuntimeError(
                    f"Cannot gather tensordict with shape {self.shape} along dim {dim}."
                )
            new_batch_size = torch.Size(index.shape)
            # as torch.gather does, keep the size of the index on the batch dims
            # where it is smaller than the tensordict
            target_batch_shape = list(map(min, index.shape, self.batch_size))

            def expand_index(shape):
                index_expand = index
                while index_expand.ndim < len(shape):
                    index_expand = index_expand.unsqueeze(-1)
                target_shape = list(shape)
                target_shape[: len(target_batch_shape)] = target_batch_shape
                target_shape[dim_corrected] = index_expand.shape[dim_corrected]
                return index_expand.expand(target_shape)

            def leaf_fn(leaf):
                return torch.gather(leaf, dim_corrected, expand_index(leaf.shape))

            def nested_fn(nested):
                nested.gather(
                    dim_corrected, expand_index(nested.batch_size), inplace=True
                )

            return self._inplace_rebind_leaves(leaf_fn, nested_fn, new_batch_size)
        return torch.gather(self, dim, index, out=out)

    @overload
    def view(self, *shape: int): ...

    @overload
    def view(self, dtype): ...

    @overload
    def view(self, shape: torch.Size): ...

    @_as_context_manager()
    def view(
        self,
        *shape: int,
        size: list | tuple | torch.Size | None = None,
        batch_size: torch.Size | None = None,
    ):
        """Returns a tensordict with views of the tensors according to a new shape, compatible with the tensordict batch_size.

        Each leaf keeps its feature dimensions, and is a view of the original
        leaf: they share storage. As for :meth:`torch.Tensor.view`, the
        leaves must have a compatible memory layout. Otherwise, use
        :meth:`~.reshape`.

        Alternatively, a dtype can be provided as a first unnamed argument. In that case, all tensors will be viewed
        with the according dtype. Note that this assume that the new shapes will be compatible with the provided dtype.
        See :meth:`torch.Tensor.view` for more information on dtype views.

        Args:
            *shape (int): new batch size of the resulting tensordict, as
                integers or as one ``torch.Size``.
            dtype (torch.dtype): alternatively, a dtype to use to represent the tensor content.

        Keyword Args:
            size (list, tuple or torch.Size, optional): the new batch size,
                as a keyword argument instead of ``*shape``.
            batch_size (torch.Size, optional): if a dtype is provided, the batch-size can be reset using this
                keyword argument. If the ``view`` is called with a shape, this is without effect.

        Returns:
            a new tensordict with the desired batch_size.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(source={'a': torch.zeros(3,4,5),
            ...    'b': torch.zeros(3,4,10,1)}, batch_size=torch.Size([3, 4]))
            >>> td_view = td.view(12)
            >>> print(td_view.get("a").shape)
            torch.Size([12, 5])
            >>> print(td_view.get("b").shape)
            torch.Size([12, 10, 1])
            >>> td_view = td.view(-1, 4, 3)
            >>> print(td_view.get("a").shape)
            torch.Size([1, 4, 3, 5])
            >>> print(td_view.get("b").shape)
            torch.Size([1, 4, 3, 10, 1])

        """
        if len(shape) == 1 and isinstance(shape[0], torch.dtype):
            dtype = shape[0]
            return self._view_dtype(dtype=dtype, batch_size=batch_size)
        _lazy_legacy = lazy_legacy()

        if _lazy_legacy:
            return self._legacy_view(*shape, size=size)
        else:
            shape = torch.Size(
                _get_shape_from_args(size=size)
                if size is not None
                else _get_shape_from_args(*shape)
            )
            result = self if shape == self.batch_size else self._view(shape)
            if result._is_shared or result._is_memmap:
                result.lock_()
            return result

    def _legacy_view(
        self,
        *shape: int,
        size: list | tuple | torch.Size | None = None,
    ) -> Self:
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = _infer_size_impl(shape, self.batch_size.numel())
            shape = torch.Size(shape)
        if shape == self.shape:
            return self
        from tensordict._lazy import _ViewedTensorDict

        return _ViewedTensorDict(
            source=self,
            custom_op="view",
            inv_op="view",
            custom_op_kwargs={"size": shape},
            inv_op_kwargs={"size": self.batch_size},
        )

    @_as_context_manager()
    def transpose(self, dim0, dim1):
        """Returns a tensordict that is a transposed version of input. The given dimensions ``dim0`` and ``dim1`` are swapped.

        In-place modifications of the transposed tensordict will impact the
        original tensordict too as the memory is shared. To map out-place
        modifications (such as new entries) back on the original tensordict,
        use the transposed tensordict as a context manager.

        Args:
            dim0 (int): the first batch dimension to swap.
            dim1 (int): the second batch dimension to swap.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> tensordict = TensorDict({"a": torch.randn(3, 4, 5)}, [3, 4])
            >>> tensordict_transpose = tensordict.transpose(0, 1)
            >>> print(tensordict_transpose.shape)
            torch.Size([4, 3])
            >>> with tensordict.transpose(0, 1) as tensordict_transpose:
            ...     tensordict_transpose["b"] = torch.randn(4, 3)
            >>> print(tensordict.get("b").shape)
            torch.Size([3, 4])
        """
        _lazy_legacy = lazy_legacy()

        if _lazy_legacy:
            return self._legacy_transpose(dim0, dim1)
        else:
            ndim = self.ndim
            if dim0 < 0:
                dim0 = ndim + dim0
            if dim1 < 0:
                dim1 = ndim + dim1
            if dim0 < 0 or dim1 < 0 or dim0 >= ndim or dim1 >= ndim:
                raise ValueError(
                    "dim0 and dim1 must be within the range of the number of dimensions."
                )
            dim0, dim1 = min(dim0, dim1), max(dim0, dim1)
            if dim0 == dim1:
                return self
            result = self._transpose(dim0, dim1)
            if result._is_shared or result._is_memmap:
                result.lock_()
            return result

    def _legacy_transpose(self, dim0, dim1):
        if dim0 < 0:
            dim0 = self.ndim + dim0
        if dim1 < 0:
            dim1 = self.ndim + dim1
        if any((dim0 < 0, dim1 < 0)):
            raise ValueError(
                "The provided dimensions are incompatible with the tensordict batch-size."
            )
        if dim0 == dim1:
            return self
        from tensordict._lazy import _TransposedTensorDict

        return _TransposedTensorDict(
            source=self,
            custom_op="transpose",
            inv_op="transpose",
            custom_op_kwargs={"dim0": dim0, "dim1": dim1},
            inv_op_kwargs={"dim0": dim0, "dim1": dim1},
        )

    @_as_context_manager()
    def swapaxes(self, axis0: int, axis1: int):
        """Interchange two axes of the tensordict.

        This is an alias for :meth:`~.transpose`.

        Args:
            axis0 (int): First axis.
            axis1 (int): Second axis.

        Returns:
            a new tensordict with the axes swapped.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.randn(3, 4, 5)}, batch_size=[3, 4])
            >>> print(td.swapaxes(0, 1).shape)
            torch.Size([4, 3])
        """
        return self.transpose(axis0, axis1)

    swapdims = swapaxes

    @overload
    def permute(self, *dims: int): ...

    @overload
    def permute(self, dims: list | tuple): ...

    @_as_context_manager()
    def permute(self, *args, **kwargs):
        """Returns a view of a tensordict with the batch dimensions permuted according to dims.

        Args:
            *dims (int): the new ordering of the batch dims of the tensordict. Alternatively,
                a single iterable of integers can be provided.
            dims (list of int): alternative way of calling permute(...).

        Returns:
            a new tensordict with the batch dimensions in the desired order.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> tensordict = TensorDict({"a": torch.randn(3, 4, 5)}, [3, 4])
            >>> print(tensordict.permute([1, 0]))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 3, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([4, 3]),
                device=None,
                is_shared=False)
            >>> print(tensordict.permute(1, 0))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 3, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([4, 3]),
                device=None,
                is_shared=False)
            >>> print(tensordict.permute(dims=[1, 0]))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4, 3, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([4, 3]),
                device=None,
                is_shared=False)
        """
        _lazy_legacy = lazy_legacy()

        if _lazy_legacy:
            return self._legacy_permute(*args, **kwargs)
        else:
            dims = _get_shape_from_args(*args, kwarg_name="dims", **kwargs)
            ndim = self.batch_dims
            dims = [dim if dim >= 0 else ndim + dim for dim in dims]
            result = self._permute(dims)
            if result._is_shared or result._is_memmap:
                result.lock_()
            return result

    def _legacy_permute(
        self,
        *dims_list: int,
        dims: list[int] | None = None,
    ) -> Self:
        if len(dims_list) == 0:
            dims_list = dims
        elif len(dims_list) == 1 and not isinstance(dims_list[0], int):
            dims_list = dims_list[0]
        if len(dims_list) != len(self.shape):
            raise RuntimeError(
                f"number of dims don't match in permute (got {len(dims_list)}, expected {len(self.shape)}"
            )

        if not len(dims_list) and not self.batch_dims:
            return self
        if list(dims_list) == list(range(self.batch_dims)):
            return self
        min_dim, max_dim = -self.batch_dims, self.batch_dims - 1
        seen = [False for dim in range(max_dim + 1)]
        for idx in dims_list:
            if idx < min_dim or idx > max_dim:
                raise IndexError(
                    f"dimension out of range (expected to be in range of [{min_dim}, {max_dim}], but got {idx})"
                )
            if seen[idx]:
                raise RuntimeError("repeated dim in permute")
            seen[idx] = True

        from tensordict._lazy import _PermutedTensorDict

        return _PermutedTensorDict(
            source=self,
            custom_op="permute",
            inv_op="permute",
            custom_op_kwargs={"dims": list(map(int, dims_list))},
            inv_op_kwargs={"dims": list(map(int, dims_list))},
        )

    @_as_context_manager()
    def movedim(
        self, source: int | tuple[int, ...], destination: int | tuple[int, ...]
    ):
        """Moves the dimension(s) of input at the position(s) in source to the position(s) in destination.

        Other dimensions of input that are not explicitly moved remain in their
        original order and appear at the positions not specified in destination.

        Args:
            source (int or tuple of ints): Original positions of the dims to move.
                These must be unique.
            destination (int or tuple of ints): Destination positions for each of
                the original dims. These must also be unique.

        Returns:
            a new tensordict with the batch dimensions moved to the desired positions.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.randn(3, 4, 5)}, batch_size=[3, 4])
            >>> print(td.movedim(0, 1).shape)
            torch.Size([4, 3])
            >>> print(td.movedim((0, 1), (1, 0)).shape)
            torch.Size([4, 3])
        """
        ndim = self.ndim

        # Normalize source and destination to tuples
        if isinstance(source, int):
            source = (source,)
        if isinstance(destination, int):
            destination = (destination,)

        if len(source) != len(destination):
            raise ValueError(
                f"movedim: source and destination must have the same number of elements, "
                f"got {len(source)} and {len(destination)}"
            )

        # Normalize negative indices
        source = tuple(s if s >= 0 else ndim + s for s in source)
        destination = tuple(d if d >= 0 else ndim + d for d in destination)

        # Validate indices
        for s in source:
            if s < 0 or s >= ndim:
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [-{ndim}, {ndim - 1}], but got {s})"
                )
        for d in destination:
            if d < 0 or d >= ndim:
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [-{ndim}, {ndim - 1}], but got {d})"
                )

        # Check for duplicates
        if len(set(source)) != len(source):
            raise RuntimeError("movedim: repeated dim in source")
        if len(set(destination)) != len(destination):
            raise RuntimeError("movedim: repeated dim in destination")

        # Fast path: if source == destination, return self
        if source == destination:
            return self

        # Convert movedim to permute dims
        # Build the permutation by:
        # 1. Create list of dims not in source
        # 2. Insert source dims at destination positions
        remaining_dims = [i for i in range(ndim) if i not in source]

        # Sort source by destination to insert in correct order
        sorted_pairs = sorted(zip(destination, source))
        perm = list(remaining_dims)
        for dest, src in sorted_pairs:
            perm.insert(dest, src)

        result = self._permute(perm)
        if result._is_shared or result._is_memmap:
            result.lock_()
        return result

    moveaxis = movedim

    @_as_context_manager()
    def flip(self, dims: int | tuple[int, ...]):
        """Reverse the order of elements in the tensordict along the given dimensions.

        The shape of the tensordict is preserved, but the elements are reordered.

        Args:
            dims (int or tuple of ints): Dimensions to flip.

        Returns:
            a new tensordict with the dimensions flipped.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
            >>> print(td["a"])
            tensor([[0, 1, 2],
                    [3, 4, 5]])
            >>> print(td.flip(0)["a"])
            tensor([[3, 4, 5],
                    [0, 1, 2]])
        """
        if isinstance(dims, int):
            dims = (dims,)

        ndim = self.ndim
        dims = tuple(d if d >= 0 else ndim + d for d in dims)

        # Validate dimensions
        for d in dims:
            if d < 0 or d >= ndim:
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [-{ndim}, {ndim - 1}], but got {d})"
                )

        def _flip(tensor):
            return tensor.flip(dims)

        return self._batch_op(_flip, self.batch_size, self._maybe_names())

    @_as_context_manager()
    def fliplr(self):
        """Flip the tensordict in the left/right direction.

        Flip the entries in each row in the left/right direction.
        Columns are preserved, but appear in a different order than before.

        Requires the tensordict to have at least 2 batch dimensions.

        Returns:
            a new tensordict with the second dimension flipped.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
            >>> print(td["a"])
            tensor([[0, 1, 2],
                    [3, 4, 5]])
            >>> print(td.fliplr()["a"])
            tensor([[2, 1, 0],
                    [5, 4, 3]])
        """
        if self.ndim < 2:
            raise RuntimeError("fliplr requires at least 2 batch dimensions")
        return self.flip(1)

    @_as_context_manager()
    def flipud(self):
        """Flip the tensordict in the up/down direction.

        Flip the entries in each column in the up/down direction.
        Rows are preserved, but appear in a different order than before.

        Requires the tensordict to have at least 1 batch dimension.

        Returns:
            a new tensordict with the first dimension flipped.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
            >>> print(td["a"])
            tensor([[0, 1, 2],
                    [3, 4, 5]])
            >>> print(td.flipud()["a"])
            tensor([[3, 4, 5],
                    [0, 1, 2]])
        """
        if self.ndim < 1:
            raise RuntimeError("flipud requires at least 1 batch dimension")
        return self.flip(0)

    @_as_context_manager()
    def roll(
        self,
        shifts: int | tuple[int, ...],
        dims: int | tuple[int, ...] = None,
        *,
        inplace: bool = False,
    ):
        """Roll the tensordict along the given dimensions.

        Elements that are shifted beyond the last position are re-introduced at
        the first position.

        Args:
            shifts (int or tuple of ints): The number of places by which the elements
                of the tensordict are shifted. If shifts is a tuple, dims must be a
                tuple of the same size, and each dimension will be rolled by the
                corresponding value.
            dims (int or tuple of ints, optional): Axis along which to roll.
                By default, the tensordict is flattened before rolling.

        Keyword Args:
            inplace (bool, optional): If ``True``, this tensordict's identity
                and key set are preserved; each leaf storage is replaced by
                its rolled counterpart one leaf at a time, keeping peak memory
                close to ``1x``. The batch_size is unchanged by ``roll``.
                Defaults to ``False``.

        Returns:
            a new tensordict with elements rolled. When ``inplace=True`` this
            is ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
            >>> print(td["a"])
            tensor([[0, 1, 2],
                    [3, 4, 5]])
            >>> print(td.roll(1, 0)["a"])
            tensor([[3, 4, 5],
                    [0, 1, 2]])
        """
        # dims index the batch dims: resolve negative dims against the batch
        # size, and roll the flattened batch (not the whole leaf) by default,
        # so that the feature dims of the leaves are left in place.
        batch_dims = self.batch_dims
        if dims is not None and not isinstance(dims, int) and not len(dims):
            # torch rolls the flattened tensor for empty dims too
            dims = None
        if dims is None:
            # -1 would be ambiguous for a leaf with an empty feature dim
            numel = self.batch_size.numel()

            def _roll(tensor):
                flat = tensor.reshape(numel, *tensor.shape[batch_dims:])
                rolled = flat.roll(shifts, 0).reshape(tensor.shape)
                if is_tensor_collection(tensor) and tensor._has_names():
                    # reshape drops the dim names of a nested tensordict
                    rolled.rename_(*tensor.names)
                return rolled

        else:
            if isinstance(dims, int):
                dims = _maybe_correct_neg_dim(dims, self.batch_size)
            else:
                dims = tuple(_maybe_correct_neg_dim(d, self.batch_size) for d in dims)

            def _roll(tensor):
                return tensor.roll(shifts, dims)

        if inplace:
            if self._is_non_tensor:
                # a NonTensorStack has no tensor leaves to rebind: write the
                # rolled entries into it
                return self.update_(self._batch_op(_roll, self.batch_size, None))

            def nested_fn(nested):
                if dims is not None:
                    nested.roll(shifts, dims, inplace=True)
                elif is_non_tensor(nested):
                    # non-tensor entries have no tensor leaves to rebind
                    nested.update_(_roll(nested))
                else:
                    # the nested batch dims may extend the batch dims that
                    # _roll flattens, so rebind the nested leaves with it
                    if _is_tensorclass(type(nested)):
                        nested = nested._tensordict
                    nested._inplace_rebind_leaves(_roll, nested_fn, None)

            return self._inplace_rebind_leaves(_roll, nested_fn, None)

        return self._batch_op(_roll, self.batch_size, self._maybe_names())

    @_as_context_manager()
    def rot90(self, k: int = 1, dims: tuple[int, int] = (0, 1)):
        """Rotate the tensordict by 90 degrees in the plane specified by dims.

        Rotation direction is from the first towards the second axis.

        Args:
            k (int): Number of times to rotate. Default: 1.
            dims (tuple of two ints): The plane to rotate in. Default: (0, 1).

        Returns:
            a new tensordict rotated by 90 degrees.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
            >>> print(td["a"])
            tensor([[0, 1, 2],
                    [3, 4, 5]])
            >>> print(td.rot90()["a"])
            tensor([[2, 5],
                    [1, 4],
                    [0, 3]])
        """
        if self.ndim < 2:
            raise RuntimeError("rot90 requires at least 2 batch dimensions")
        if len(dims) != 2:
            raise RuntimeError("rot90 requires exactly 2 dims")

        # Normalize dims
        ndim = self.ndim
        dims = tuple(d if d >= 0 else ndim + d for d in dims)

        # Calculate new batch size
        k = k % 4  # Normalize k to [0, 3]
        if k == 0:
            return self

        batch_size = list(self.batch_size)
        if k == 1 or k == 3:
            batch_size[dims[0]], batch_size[dims[1]] = (
                batch_size[dims[1]],
                batch_size[dims[0]],
            )

        if self._has_names():
            names = list(self.names)
            if k == 1 or k == 3:
                names[dims[0]], names[dims[1]] = names[dims[1]], names[dims[0]]
        else:
            names = None

        def _rot90(tensor):
            return tensor.rot90(k, dims)

        return self._batch_op(_rot90, torch.Size(batch_size), names)

    def narrow(self, dim: int, start: int, length: int):
        """Returns a new tensordict that is a narrowed version of the input.

        The dimension dim is input from start to start + length.

        Args:
            dim (int): The dimension along which to narrow.
            start (int): Starting index.
            length (int): Length of the narrowed dimension.

        Returns:
            a new tensordict narrowed along the specified dimension.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
            >>> print(td["a"])
            tensor([[0, 1, 2],
                    [3, 4, 5]])
            >>> print(td.narrow(1, 1, 2)["a"])
            tensor([[1, 2],
                    [4, 5]])
        """
        ndim = self.ndim
        if dim < 0:
            dim = ndim + dim
        if dim < 0 or dim >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-{ndim}, {ndim - 1}], but got {dim})"
            )

        batch_size = list(self.batch_size)
        batch_size[dim] = length

        def _narrow(tensor):
            return tensor.narrow(dim, start, length)

        if self._is_non_tensor:
            # A NonTensorStack selects self[index], which keeps the names:
            # skip computing them, which walks over the members.
            index_start = start + self.batch_size[dim] if start < 0 else start
            return self._batch_op(
                _narrow,
                torch.Size(batch_size),
                None,
                index=(slice(None),) * dim
                + (slice(index_start, index_start + length),),
            )
        return self._batch_op(_narrow, torch.Size(batch_size), self._maybe_names())

    def tile(self, dims: tuple[int, ...]):
        """Construct a tensordict by repeating the elements.

        The dims argument specifies the number of repetitions in each dimension.

        Args:
            dims (tuple of ints): The number of repetitions per dimension.

        Returns:
            a new tensordict with elements repeated.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
            >>> print(td["a"])
            tensor([[0, 1, 2],
                    [3, 4, 5]])
            >>> print(td.tile((2, 1))["a"])
            tensor([[0, 1, 2],
                    [3, 4, 5],
                    [0, 1, 2],
                    [3, 4, 5]])
        """
        if isinstance(dims, int):
            dims = (dims,)

        # Calculate new batch size
        ndim = self.ndim
        if len(dims) > ndim:
            # If more dims than batch dims, prepend 1s to batch_size
            new_batch_size = [1] * (len(dims) - ndim) + list(self.batch_size)
            for i, d in enumerate(dims):
                new_batch_size[i] *= d
        else:
            # Pad dims with leading 1s
            new_batch_size = list(self.batch_size)
            offset = ndim - len(dims)
            for i, d in enumerate(dims):
                new_batch_size[offset + i] *= d

        # Align dims with the batch dims and leave the feature dims untiled
        reps = (1,) * (ndim - len(dims)) + tuple(dims)

        def _tile(tensor):
            return tensor.tile(reps + (1,) * (tensor.ndim - ndim))

        # tile invalidates names
        return self._batch_op(_tile, torch.Size(new_batch_size), None)

    def _batch_op(
        self,
        fn: Callable[[Tensor], Tensor],
        batch_size: torch.Size,
        names: list | None,
        *,
        index: tuple | None = None,
    ) -> Self:
        """Returns the result of a shape operation that moves the elements of the batch.

        Called by :meth:`flip`, :meth:`roll`, :meth:`rot90`, :meth:`tile`,
        :meth:`narrow` and :meth:`_broadcast_to` after they have checked their
        arguments and computed the ``batch_size`` and ``names`` of the result.
        ``fn`` takes a tensor whose leading dims are the batch dims, moves its
        elements and leaves the feature dims alone. ``index``, which
        :meth:`narrow` gives a ``NonTensorStack``, is a basic index (slices)
        that selects the same elements, so that the class can select them
        instead; ``names`` is then ``None``, as ``self[index]`` keeps the
        names.

        The default applies ``fn`` to every leaf and gives the result the
        lock and the shared and memory-mapped flags of ``self``, and locks a
        shared or memory-mapped result. ``NonTensorStack``, which has no
        tensor leaves, applies ``fn`` to a tensor of the flat positions of its
        entries and selects the entries at the positions ``fn`` returns (a
        copy), or returns ``self[index]`` (shared entries).
        """
        result = self._fast_apply(
            fn,
            batch_size=batch_size,
            call_on_nested=True,
            names=names,
            propagate_lock=True,
        )
        self._maybe_set_shared_attributes(result)
        if result._is_shared or result._is_memmap:
            result.lock_()
        return result

    def broadcast_to(self, shape: tuple[int, ...]):
        """Broadcasts the tensordict to a new shape.

        The new shape must be compatible with the original shape.

        Args:
            shape (tuple of ints): The desired shape.

        Returns:
            a new tensordict with the shape broadcast.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.arange(3)}, batch_size=[3])
            >>> print(td.broadcast_to((2, 3)).shape)
            torch.Size([2, 3])
        """
        shape = torch.Size(shape)
        if -1 in shape:
            shape = _resolve_expand_shape(shape, self.batch_size)
        return self._broadcast_to(shape)

    def _broadcast_to(self, shape: torch.Size) -> Self:
        """Returns the tensordict broadcast to the batch size ``shape``.

        Called by :meth:`broadcast_to` after it has resolved ``-1``. The
        result has no names, and its leaves are broadcast views of the leaves
        of ``self``.

        The default broadcasts each leaf through :meth:`_batch_op`, which
        raises as :func:`torch.broadcast_to` does for a shape that does not
        fit. ``NonTensorStack`` expands its members, which shares the entries.
        """

        def _broadcast_to(tensor):
            return tensor.broadcast_to(shape + tensor.shape[self.ndim :])

        # broadcast invalidates names
        return self._batch_op(_broadcast_to, shape, None)

    @_as_context_manager()
    def atleast_1d(self):
        """Returns the tensordict with at least 1 batch dimension.

        If the tensordict already has 1 or more batch dimensions, it is returned unchanged.
        Otherwise, a dimension of size 1 is prepended.

        Returns:
            a tensordict with at least 1 batch dimension.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.randn(3)}, batch_size=[])
            >>> print(td.atleast_1d().shape)
            torch.Size([1])
        """
        if self.ndim >= 1:
            return self
        return self.unsqueeze(0)

    @_as_context_manager()
    def atleast_2d(self):
        """Returns the tensordict with at least 2 batch dimensions.

        If the tensordict already has 2 or more batch dimensions, it is returned unchanged.
        Otherwise, dimensions of size 1 are prepended to reach 2 dimensions.

        Returns:
            a tensordict with at least 2 batch dimensions.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.randn(3)}, batch_size=[3])
            >>> print(td.atleast_2d().shape)
            torch.Size([1, 3])
        """
        if self.ndim >= 2:
            return self
        elif self.ndim == 1:
            return self.unsqueeze(0)
        else:
            return self.unsqueeze(0).unsqueeze(0)

    @_as_context_manager()
    def atleast_3d(self):
        """Returns the tensordict with at least 3 batch dimensions.

        If the tensordict already has 3 or more batch dimensions, it is returned unchanged.
        Otherwise, dimensions of size 1 are prepended to reach 3 dimensions.

        Returns:
            a tensordict with at least 3 batch dimensions.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.randn(3)}, batch_size=[3])
            >>> print(td.atleast_3d().shape)
            torch.Size([1, 1, 3])
        """
        if self.ndim >= 3:
            return self
        elif self.ndim == 2:
            return self.unsqueeze(0)
        elif self.ndim == 1:
            return self.unsqueeze(0).unsqueeze(0)
        else:
            return self.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    def densify(self, layout: torch.layout = torch.strided):
        """Attempts to represent the lazy stack with contiguous tensors (plain tensors or nested).

        Keyword Args:
            layout (torch.layout): the layout of the nested tensors, if any. Defaults to
                :class:`~torch.strided`.

        """
        return self._densify(layout)

    def _densify(self, layout: torch.layout) -> TensorCollection:
        """Returns the tensordict with its lazy stacks replaced by dense entries.

        Called by :meth:`densify`. The default densifies each nested tensor
        collection and returns ``self`` when nothing changes.
        ``LazyStackedTensorDict`` builds a ``TensorDict`` that stacks the
        entries of its members, with nested tensors of ``layout`` along stack
        dim 0 when their shapes differ. ``NonTensorStack`` returns ``self``:
        it is the dense form of stacked non-tensor data.
        """
        any_set = False
        out_dict = {}
        for key, val in self.items():
            if is_tensor_collection(val):
                val_dense = val.densify(layout=layout)
                any_set = any_set | (val_dense is not val)
                val = val_dense
            out_dict[key] = val
        if any_set:
            result = self.empty()
            for key, val in out_dict.items():
                result._set_str(key, val, validated=True, inplace=False)
            return result
        return self

    @_as_context_manager()
    def flatten(
        self,
        start_dim: int | None = None,
        end_dim: int | None = None,
        *,
        inplace: bool = False,
    ):
        """Flattens all the tensors of a tensordict.

        Args:
            start_dim (int): the first dim to flatten
            end_dim (int): the last dim to flatten

        A ``start_dim`` or ``end_dim`` outside the batch dims warns, and is
        used on the dims of the leaves. Starting with TensorDict 0.17, the
        default will change to raising an ``IndexError``, as for a tensor.

        Keyword Args:
            inplace (bool, optional): If ``True``, this tensordict's identity
                and key set are preserved; each leaf is flattened one at a
                time. ``torch.flatten`` may return a view when the flattened
                range is contiguous in memory; in that case leaves share
                storage with the originals and the memory benefit does not
                materialize. Not supported on
                :class:`~tensordict.LazyStackedTensorDict`. Defaults to
                ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     "a": torch.arange(60).view(3, 4, 5),
            ...     "b": torch.arange(12).view(3, 4)}, batch_size=[3, 4])
            >>> td_flat = td.flatten(0, 1)
            >>> td_flat.batch_size
            torch.Size([12])
            >>> td_flat["a"]
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  6,  7,  8,  9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 21, 22, 23, 24],
                    [25, 26, 27, 28, 29],
                    [30, 31, 32, 33, 34],
                    [35, 36, 37, 38, 39],
                    [40, 41, 42, 43, 44],
                    [45, 46, 47, 48, 49],
                    [50, 51, 52, 53, 54],
                    [55, 56, 57, 58, 59]])
            >>> td_flat["b"]
            tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

        """
        if start_dim in (None, 0) and end_dim in (None, -1, 0) and not self.ndim:
            return self.unsqueeze(0)
        if inplace and self._lazy:
            raise NotImplementedError(
                "flatten(inplace=True) is not supported on "
                "LazyStackedTensorDict; call .to_tensordict() first or use "
                "inplace=False."
            )
        if start_dim is None:
            start_dim = 0
        if end_dim is None:
            end_dim = -1
        if self._flatten_lenient:
            # the members have no dims past the batch dims of the stack
            start_dim = _maybe_correct_neg_dim(start_dim, self.batch_size)
            end_dim = _maybe_correct_neg_dim(end_dim, self.batch_size)
        if start_dim < 0:
            start_dim = self.ndim + start_dim
        if end_dim < 0:
            end_dim = self.ndim + end_dim
            if end_dim < 0:
                raise ValueError(
                    f"Incompatible end_dim {end_dim} for tensordict with shape {self.shape}."
                )
        if end_dim < start_dim:
            if not self._flatten_lenient:
                raise ValueError(
                    "The end dimension must be greater or equal to the start dim."
                )
            # LazyStackedTensorDict.flatten inserted a dim of size 1
            _warn_lazy_shape_args(
                "flatten", f"start_dim={start_dim} > end_dim={end_dim}"
            )
            batch_size = list(self.batch_size)
            batch_size.insert(start_dim, 1)
            return self._reshape(torch.Size(batch_size), None)
        past_batch_dims = start_dim < 0 or end_dim >= self.ndim
        if past_batch_dims:
            _warn_dim_past_batch_dims("flatten")
        elif start_dim == end_dim and self._flatten_lenient:
            # a no-op, which returns the input, as torch.flatten does (the
            # members of a lazy stack may have no dim to flatten)
            return self

        def flatten(tensor):
            return torch.flatten(tensor, start_dim, end_dim)

        nelt = math.prod(self.batch_size[start_dim : end_dim + 1])
        if start_dim > 0:
            batch_size = (
                list(self.batch_size)[:start_dim]
                + [nelt]
                + list(self.batch_size[end_dim + 1 :])
            )
        else:
            batch_size = [nelt] + list(self.batch_size[end_dim + 1 :])
        # TODO: check that this works with nested tds of different batch size
        if self._has_names():
            names = [
                name
                for i, name in enumerate(self.names)
                if (i < start_dim or i > end_dim)
            ]
            names.insert(start_dim, None)
        else:
            names = None
        if inplace:
            self._raise_if_holds_lazy_stack("flatten")

            def nested_fn(nested):
                nested.flatten(start_dim, end_dim, inplace=True)

            self._inplace_rebind_leaves(flatten, nested_fn, torch.Size(batch_size))
            if names is not None:
                self.names = names
            return self
        if not past_batch_dims and start_dim != end_dim:
            return self._reshape(torch.Size(batch_size), names, flatten)
        out = self._fast_apply(
            flatten,
            batch_size=batch_size,
            propagate_lock=True,
            names=names,
            call_on_nested=True,
        )
        return out

    @_as_context_manager()
    def unflatten(self, dim, unflattened_size, *, inplace: bool = False):
        """Unflattens a tensordict dim expanding it to a desired shape.

        Args:
            dim (int): specifies the dimension of the input tensor to be
                unflattened.
            unflattened_size (shape): is the new shape of the unflattened
                dimension of the tensordict.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     "a": torch.arange(60).view(3, 4, 5),
            ...     "b": torch.arange(12).view(3, 4)},
            ...     batch_size=[3, 4])
            >>> td_flat = td.flatten(0, 1)
            >>> td_unflat = td_flat.unflatten(0, [3, 4])
            >>> assert (td == td_unflat).all()
        """
        if inplace and self._lazy:
            raise NotImplementedError(
                "unflatten(inplace=True) is not supported on "
                "LazyStackedTensorDict; call .to_tensordict() first or use "
                "inplace=False."
            )
        dim = _maybe_correct_neg_dim(dim, self.batch_size)
        if -1 in unflattened_size and -1 not in self.batch_size:
            # Infer the -1 as torch.unflatten does, which raises for sizes that
            # don't fit. A batch with a -1 (a ragged dim) is left as it is.
            unflattened_size = (
                torch.empty(self.batch_size, device="meta")
                .unflatten(dim, unflattened_size)
                .shape[dim : dim + len(unflattened_size)]
            )
        _check_unflatten_size(self.batch_size, dim, unflattened_size)

        def unflatten(tensor):
            return torch.unflatten(
                tensor,
                dim,
                unflattened_size,
            )

        if dim > 0:
            batch_size = (
                list(self.batch_size)[:dim]
                + list(unflattened_size)
                + list(self.batch_size[dim + 1 :])
            )
        else:
            batch_size = list(unflattened_size) + list(self.batch_size[1:])
        if self._has_names():
            names = list(self.names)
            for _ in range(len(unflattened_size) - 1):
                names.insert(dim, None)
        else:
            names = None
        # TODO: check that this works with nested tds of different batch size
        if inplace:
            self._raise_if_holds_lazy_stack("unflatten")

            def nested_fn(nested):
                nested.unflatten(dim, unflattened_size, inplace=True)

            self._inplace_rebind_leaves(unflatten, nested_fn, torch.Size(batch_size))
            if names is not None:
                self.names = names
            return self
        return self._reshape(torch.Size(batch_size), names, unflatten)

    def scatter(
        self,
        src: int,
        tensordicts: list | None = None,
        *,
        group: "torch.distributed.ProcessGroup" | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        """Scatters a list of tensordicts from ``src`` to all ranks.

        On the source rank, ``tensordicts`` must be a list of tensordicts with
        one element per rank. Each is consolidated, and its metadata + storage
        are scattered. Non-source ranks receive and reconstruct their tensordict.

        Args:
            src (int): The rank of the source process.
            tensordicts (list[TensorDict] | None): On the source rank, a list
                of tensordicts to scatter (one per rank). Ignored on other ranks.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): The process group
                to use. Defaults to ``None`` (default group).
            device (torch.device or str, optional): The device on which to
                allocate the receive buffer.  If ``None``, the device is
                inferred from the source's consolidated storage (transmitted
                via metadata).  Defaults to ``None``.

        Returns:
            TensorDict: The tensordict assigned to this rank.
        """
        from tensordict._reductions import _rebuild_tensordict_files_consolidated
        from torch import distributed as dist

        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)

        if rank == src:
            consolidated = []
            for td in tensordicts:
                td_c = td if td.is_consolidated() else td.consolidate(metadata=True)
                consolidated.append(td_c)

            all_sizes = [td_c._consolidated["storage"].numel() for td_c in consolidated]
            max_size = max(all_sizes)
            storage_device = consolidated[0]._consolidated["storage"].device
            all_meta = [
                {
                    **td_c._consolidated["metadata"],
                    "_total_bytes": td_c._consolidated["storage"].numel(),
                    "_max_bytes": max_size,
                    "_storage_device": str(storage_device),
                }
                for td_c in consolidated
            ]
            scatter_meta = [None] * world_size
            dist.scatter_object_list(scatter_meta, all_meta, src=src, group=group)

            padded_list = []
            for td_c, sz in zip(consolidated, all_sizes):
                buf = torch.zeros(max_size, dtype=torch.uint8, device=storage_device)
                buf[:sz] = td_c._consolidated["storage"]
                padded_list.append(buf)
            recv_buf = torch.empty(max_size, dtype=torch.uint8, device=storage_device)
            dist.scatter(recv_buf, padded_list, src=src, group=group)

            metadata = scatter_meta[0]
            total_bytes = metadata.pop("_total_bytes")
            metadata.pop("_max_bytes")
            metadata.pop("_storage_device")
            return _rebuild_tensordict_files_consolidated(
                metadata, recv_buf[:total_bytes]
            )
        else:
            scatter_meta = [None]
            dist.scatter_object_list(scatter_meta, None, src=src, group=group)
            metadata = scatter_meta[0]
            total_bytes = metadata.pop("_total_bytes")
            max_size = metadata.pop("_max_bytes")
            storage_device = metadata.pop("_storage_device")
            recv_device = device if device is not None else torch.device(storage_device)

            recv_buf = torch.empty(max_size, dtype=torch.uint8, device=recv_device)
            dist.scatter(recv_buf, None, src=src, group=group)
            return _rebuild_tensordict_files_consolidated(
                metadata, recv_buf[:total_bytes]
            )

    def to_padded_tensor(self, padding=0.0, mask_key: NestedKey | None = None) -> Self:
        """Converts all nested tensors to a padded version and adapts the batch-size accordingly.

        Args:
            padding (float): the padding value for the tensors in the tensordict.
                Defaults to ``0.0``.
            mask_key (NestedKey, optional): if provided, the key where a
                mask for valid values will be written.
                Will result in an error if the heterogeneous dimension
                isn't part of the tensordict batch-size.
                Defaults to ``None``

        """
        batch_size = self.batch_size
        if any(shape == -1 for shape in batch_size):
            new_batch_size = []
        else:
            new_batch_size = None
            if mask_key is not None:
                raise RuntimeError(
                    "mask_key should only be provided if the "
                    "heterogenous dimension is part of the batch-size."
                )
        padded_names = []

        def to_padded(name, x):
            if x.is_nested:
                padded_names.append(name)
                return torch.nested.to_padded_tensor(x, padding=padding)
            return x

        result = self._apply_nest(
            to_padded,
            batch_size=new_batch_size,
            named=True,
            nested_keys=True,
        )
        if new_batch_size is not None:
            result = result.auto_batch_size_(
                batch_dims=self.batch_dims, keep_compliant_size=True
            )

            if mask_key:
                # take the first of the padded keys
                padded_key = padded_names[0]
                # write the mask
                val = self.get(padded_key)
                val = torch.nested.to_padded_tensor(
                    torch.ones_like(val, dtype=torch.bool), padding=False
                )
                if val.ndim > result.ndim:
                    val = val.flatten(result.ndim, -1)[..., -1].clone()
                result.set(mask_key, val)
        return result

    def masked_select(self, mask: Tensor, *, inplace: bool = False) -> Self:
        """Masks all tensors of the TensorDict and return a new TensorDict instance with similar keys pointing to masked values.

        Args:
            mask (torch.Tensor): boolean mask to be used for the tensors.
                Shape must match the TensorDict ``batch_size``.

        Keyword Args:
            inplace (bool, optional): If ``True``, this tensordict's identity
                and key set are preserved; each leaf storage is replaced by
                its masked counterpart one leaf at a time, keeping peak
                memory close to ``1x``. Not supported on
                :class:`~tensordict.LazyStackedTensorDict`.
                Defaults to ``False``.

        Returns:
            a new tensordict with the masked values. When ``inplace=True``
            this is ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...    batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> td_mask = td.masked_select(mask)
            >>> td_mask.get("a")
            tensor([[0., 0., 0., 0.]])
            >>> td.masked_select(mask, inplace=True) is td
            True
            >>> td.batch_size
            torch.Size([1])

        """
        from tensordict._td import TensorDict

        if inplace and (self._lazy or self._masked_select_inplace_unsupported):
            raise NotImplementedError(
                f"masked_select(inplace=True) is not supported on "
                f"{type(self).__name__}; call .to_tensordict() first or use "
                f"inplace=False."
            )
        if not inplace and self._masked_select_on_dense_copy:
            return self.to_tensordict().masked_select(mask)

        d = {}
        mask_expand = mask
        while mask_expand.ndimension() > self.batch_dims:
            mndim = mask_expand.ndimension()
            mask_expand = mask_expand.squeeze(-1)
            if mndim == mask_expand.ndimension():  # no more squeeze
                break
        dim = int(mask.sum().item())
        other_dim = self.shape[mask.ndim :]
        batch_size = torch.Size([dim, *other_dim])
        if inplace:
            self._raise_if_holds_lazy_stack("masked_select")

            def nested_fn(nested):
                nested.masked_select(mask_expand, inplace=True)

            self._inplace_rebind_leaves(
                lambda value: value[mask_expand], nested_fn, batch_size
            )
            if self._has_names():
                # the masked dims become one dim, which has no name
                self._erase_names()
            return self
        for key, value in self.items():
            if _is_unbatched(value):
                d[key] = value._with_batch_size(batch_size)
            else:
                d[key] = value[mask_expand]
        return TensorDict(device=self.device, source=d, batch_size=batch_size)
