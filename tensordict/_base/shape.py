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
import warnings
from collections.abc import Iterable
from typing import Callable, List, overload, Sequence, TYPE_CHECKING

import torch
from tensordict._nestedkey import NestedKey
from tensordict._tensorcollection import TensorCollection
from tensordict._unbatched import _unbatched_with_batch_size
from tensordict.base import (
    _is_leaf_nontensor,
    _is_tensor_collection,
    is_tensor_collection,
    NO_DEFAULT,
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
    _user_stacklevel,
    _warn_dim_past_batch_dims,
    _warn_lazy_shape_args,
    _zip_strict,
    is_compiling,
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

    Only tensorclasses are rebuilt with their class. Other subclasses of
    :class:`~tensordict.TensorDictBase`, such as
    :class:`~tensordict.TypedTensorDict` or
    :class:`~tensordict.nn.TensorDictParams`, come back as a
    :class:`~tensordict.TensorDict`.

    Attributes:
        keys (tuple of NestedKey): the leaf keys in column order.
        shapes (tuple of torch.Size): the feature shape of each leaf, that is
            its shape after the batch dimensions of the packed tensordict.
            Lists and tuples of ints are converted to ``torch.Size``.
        dim (int): the position of the packed dimension in the packed tensor.
            :meth:`~tensordict.TensorDictBase.pack` stores it counted from the
            end (a negative int), so ``td.pack(1)`` and ``td.pack(-1)`` give
            equal layouts when ``td`` has one batch dimension, and ``flat``
            may have extra leading batch dimensions.
        nested (tuple): one ``(key, batch_shape, cls)`` entry per nested tensor
            collection, where ``batch_shape`` is its batch shape after the
            batch dimensions of the root and ``cls`` is its tensorclass, or
            ``None`` for a tensordict. Batch shapes given as lists or tuples
            of ints are converted to ``torch.Size``.
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
        object.__setattr__(
            self, "shapes", tuple(torch.Size(shape) for shape in self.shapes)
        )
        object.__setattr__(
            self,
            "nested",
            tuple(
                (key, torch.Size(batch_shape), cls)
                for key, batch_shape, cls in self.nested
            ),
        )
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
            A :class:`~tensordict.TensorDict`, or an instance of :attr:`cls`,
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
        """Returns a tensordict whose batch dimensions are expanded to ``shape``.

        The new shape is the new batch size, and the leaves keep their feature
        dimensions. As for :meth:`torch.Tensor.expand`, it can add leading
        dimensions and expand the dimensions of size 1, and ``-1`` keeps the
        size of a dimension. The other dimensions must keep their size. The
        new leading dimensions have no name.

        The result holds views of the original tensors: no memory is
        allocated, and the expanded leaves are not contiguous. Call
        :meth:`~.clone` or :meth:`~.contiguous` before you write into them.

        Args:
            *shape (int): the new batch size, as integers or as one
                ``torch.Size``, list or tuple.

        Raises:
            RuntimeError: if ``shape`` has fewer dimensions than the batch
                size, or changes the size of a dimension that is not 1.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     'a': torch.zeros(3, 4, 5),
            ...     'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
            >>> td_expand = td.expand(10, 3, 4)
            >>> td_expand.batch_size, td_expand["a"].shape
            (torch.Size([10, 3, 4]), torch.Size([10, 3, 4, 5]))
            >>> td.expand(2, -1, -1).batch_size
            torch.Size([2, 3, 4])

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
        """Returns the tensordict expanded to the shape of ``other``.

        If ``other`` is a tensor, this is ``expand(other.shape)``: the batch
        size of the result is the shape of ``other``. If ``other`` is a
        tensordict or a tensorclass, each leaf is expanded to the shape of the
        leaf with the same key in ``other``, with
        :meth:`torch.Tensor.expand_as`, and the result has the batch size of
        ``other``. The leaves of the result are views of the original leaves.

        Args:
            other (torch.Tensor or TensorDictBase): the tensor or the
                tensordict whose shape to expand to.

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
            >>> assert (expanded == 1).all()
            >>> expanded.batch_size
            torch.Size([2, 3])
            >>> expanded["a"].shape, expanded["b", "c"].shape
            (torch.Size([2, 3, 5, 4]), torch.Size([2, 3, 2, 6, 4]))
            >>> td0.expand_as(torch.zeros(2, 3)).batch_size
            torch.Size([2, 3])

        """
        if _is_tensor_collection(type(other)):

            def expand_as(x, y):
                return x.expand_as(y)

            return self.apply(expand_as, other, batch_size=other.batch_size)
        return self.expand(other.shape)

    def unbind(self, dim: int) -> tuple[TensorDictBase, ...]:
        """Returns a tuple of the tensordicts along a batch dimension, as :meth:`torch.Tensor.unbind` does.

        The tensordicts of the result hold views of the original tensors. They
        are locked if the tensordict is shared or memory-mapped.

        Args:
            dim (int): the batch dimension to remove. The result has
                ``batch_size[dim]`` tensordicts.

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
        """Splits the tensordict along a batch dimension, as :func:`torch.tensor_split` does.

        The tensordicts of the result hold views of the original tensors.

        Args:
            indices_or_sections (int, list of int, tuple of int or torch.Tensor):
                an integer ``n`` (or a 0-d long tensor) splits the tensordict
                into ``n`` sections. When ``n`` does not divide
                ``batch_size[dim]``, the first ``batch_size[dim] % n``
                sections have one more element than the others. A list, a tuple
                or a 1-d long tensor of indices splits it before each index:
                ``[2, 3]`` with ``dim=0`` gives ``td[:2]``, ``td[2:3]`` and
                ``td[3:]``. As with :func:`torch.tensor_split`, negative
                indices count from the end, indices past the end are clamped,
                and an index smaller than the one before it gives an empty or
                an overlapping section. A tensor must be on the CPU.
            dim (int, optional): the batch dimension to split. Defaults to
                ``0``.

        Returns:
            A tuple of tensordicts.

        Raises:
            IndexError: if ``dim`` is not a batch dimension, for example when
                the tensordict has no batch dimension.
            ValueError: if the number of sections is smaller than 1, or if a
                tensor ``indices_or_sections`` is not a 0-d or 1-d long tensor
                on the CPU.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(x=torch.arange(5), batch_size=[5])
            >>> [t["x"] for t in td.tensor_split(2)]
            [tensor([0, 1, 2]), tensor([3, 4])]
            >>> [t["x"] for t in td.tensor_split([1, 3])]
            [tensor([0]), tensor([1, 2]), tensor([3, 4])]

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
        """Returns a tensordict with a new batch dimension of size 1 at position ``dim``.

        The tensors of the result are views of the original tensors. The new
        dimension has no name.

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
        """Returns a tensordict without the batch dimensions of size 1.

        The tensors of the result are views of the original tensors. Feature
        dimensions of size 1 are kept.

        Args:
            dim (int, optional): the batch dimension to remove, in
                ``[-batch_dims, batch_dims - 1]``. If its size is not 1, the
                tensordict is returned as it is. Defaults to ``None``, which
                removes every batch dimension of size 1.

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
        allows it, and a copy otherwise. The result has no names.

        Args:
            *shape (int): the new batch size, as integers or as one list,
                tuple or ``torch.Size``. One dimension can be ``-1``. The shape
                is positional only: ``reshape(shape=...)`` raises
                ``TypeError``.

        Keyword Args:
            inplace (bool, optional): if ``True``, the tensordict keeps its
                identity and keys, its leaves are replaced one by one, and the
                call returns ``self``. A leaf keeps its storage when the
                reshape can return a view of it. A lazy stack, or a tensordict
                that holds one, raises a ``NotImplementedError``: call
                :meth:`to_tensordict` first. See :ref:`inplace-conventions`.
                Defaults to ``False``.

        Returns:
            A tensordict with the new batch size, or ``self`` when
            ``inplace=True`` or when the batch size does not change.

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
            self._raise_if_holds_lazy_stack("reshape", new_batch_size=shape)

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
        """Repeats each element of the batch along ``dim``, as :meth:`torch.Tensor.repeat_interleave` does.

        Each element is repeated next to itself, as with :func:`numpy.repeat`.
        To repeat the whole batch instead, as with :func:`numpy.tile`, use
        :meth:`repeat`. The data is copied. With a ``dim``, the names are
        kept.

        Args:
            repeats (torch.Tensor or int): the number of repetitions of each
                element, broadcast to the size of ``dim``.
            dim (int, optional): the batch dimension to repeat along. Defaults
                to ``None``, which flattens the batch dimensions first, so the
                result has one batch dimension. A ``dim`` past the batch
                dimensions warns, and is used on the dimensions of the leaves.
                Starting with TensorDict 0.17, the default will change to
                raising an ``IndexError``, as for a tensor.

        Keyword Args:
            output_size (int, optional): the size of the result along ``dim``
                (the sum of ``repeats``). Passing it avoids the device
                synchronization that computes it. Defaults to ``None``, which
                computes it.
            inplace (bool, optional): if ``True``, the tensordict keeps its
                identity and keys, its leaves are replaced one by one, and the
                call returns ``self``. It needs a ``dim`` and at least one
                batch dimension. A lazy stack, or a tensordict that holds one,
                raises a ``NotImplementedError``: call :meth:`to_tensordict`
                first. See :ref:`inplace-conventions`. Defaults to ``False``.

        Returns:
            A tensordict with the same batch size, except along ``dim``, or
            ``self`` with ``inplace=True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.tensor([[0, 1], [2, 3]]), batch_size=[2, 2])
            >>> td.repeat_interleave(2, dim=0)["a"]
            tensor([[0, 1],
                    [0, 1],
                    [2, 3],
                    [2, 3]])
            >>> td.repeat_interleave(torch.tensor([1, 2]), dim=1)["a"]
            tensor([[0, 1, 1],
                    [2, 3, 3]])
            >>> td.repeat_interleave(2).batch_size
            torch.Size([8])

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
            new_batch_size = _repeat_interleave_batch_size(
                self.batch_size, repeats, dim_corrected
            )
            self._raise_if_holds_lazy_stack(
                "repeat_interleave", new_batch_size=new_batch_size
            )

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

            return self._inplace_rebind_leaves(rep, nested_fn, new_batch_size)
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
        """Repeats the batch along each batch dimension, as :meth:`torch.Tensor.repeat` does.

        The whole batch is repeated, as with :func:`numpy.tile`. To repeat
        each element next to itself instead, as with :func:`numpy.repeat`, use
        :meth:`repeat_interleave`. Unlike :meth:`expand`, it copies the data.
        The feature dimensions are not repeated, and the names are kept.

        Args:
            *repeats (int): the number of repetitions along each batch
                dimension, one per batch dimension, as integers or as one
                ``torch.Size``, list, tuple or tensor.

        Keyword Args:
            inplace (bool, optional): if ``True``, the tensordict keeps its
                identity and keys, its leaves are replaced one by one, and the
                call returns ``self``. A lazy stack, or a tensordict that
                holds one, raises a ``NotImplementedError``: call
                :meth:`to_tensordict` first. See :ref:`inplace-conventions`.
                Defaults to ``False``.

        Raises:
            ValueError: if the number of repeats differs from the number of
                batch dimensions.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.tensor([[0, 1]]), b=torch.zeros(1, 2, 3), batch_size=[1, 2])
            >>> out = td.repeat(2, 2)
            >>> out["a"]
            tensor([[0, 1, 0, 1],
                    [0, 1, 0, 1]])
            >>> out["b"].shape
            torch.Size([2, 4, 3])

        """
        if len(repeats) == 1 and not isinstance(repeats[0], int):
            repeats = repeats[0]
            # a torch.Size is a tuple
            if isinstance(repeats, (list, tuple)):
                return self.repeat(*repeats, inplace=inplace)
            if isinstance(repeats, torch.Tensor):
                # This will cause cuda to sync, which may not be desirable.
                # A 0-dim tensor is one count: tolist() returns that int.
                values = repeats.tolist()
                if repeats.ndim == 0:
                    values = (values,)
                return self.repeat(*values, inplace=inplace)
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
            new_batch_size = torch.Size(
                [s * r for s, r in zip(self.batch_size, repeats)]
            )
            self._raise_if_holds_lazy_stack("repeat", new_batch_size=new_batch_size)
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
        """Returns the tensordict padded along its batch dimensions with a constant value.

        Args:
            pad_size (sequence of int): the padding, applied to the batch
                dimensions starting from the first. For a tensordict with
                ``ndim`` batch dimensions, it is a sequence of even length up
                to ``2 * ndim``:
                ``(dim0_left, dim0_right, dim1_left, dim1_right, ...)``.
            value (float, optional): the fill value. Defaults to ``0.0``.
            inplace (bool, optional): if ``True``, the tensordict keeps its
                identity and keys, its leaves are replaced one by one by
                their padded copies, and the call returns ``self``. Peak memory
                stays close to ``1x`` instead of ``2x``. See
                :ref:`inplace-conventions`. Defaults to ``False``.

                .. warning::
                    If ``inplace=True`` and the padding of a later leaf fails
                    (for example, out of memory), the tensordict is left in an
                    inconsistent state. ``safe=True`` catches the user errors
                    before any leaf is changed.
            safe (bool, optional): if ``True``, checks that the padding works
                for every leaf before any leaf is changed. Pass ``False`` to
                skip this check when the inputs are known to be valid.
                Defaults to ``True``.

        Returns:
            The padded tensordict, or ``self`` with ``inplace=True``.

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

    def _cat_or_stack_tensors(
        self,
        *keys: NestedKey,
        out_key: NestedKey,
        dim: int,
        keep_entries: bool,
        op: Callable,
    ) -> Self:
        # Build the result before removing entries. cat/stack raises on a
        # shape mismatch, and set can still reject the written value.
        entries = [self.get(key, NO_DEFAULT) for key in keys]
        result = op(entries, dim=dim)
        if keep_entries:
            return self.set(out_key, result)
        deleted = 0
        try:
            for key in keys:
                self.del_(key)
                deleted += 1
            return self.set(out_key, result)
        except Exception:
            # Last removed first, so a parent is put back before a nested key.
            for i in reversed(range(deleted)):
                self.set(keys[i], entries[i])
            raise

    def cat_tensors(
        self,
        *keys: NestedKey,
        out_key: NestedKey,
        dim: int = 0,
        keep_entries: bool = False,
    ) -> Self:
        """Concatenates entries of the tensordict into a new entry, and removes them by default.

        Args:
            *keys (NestedKey): the entries to concatenate, with
                :func:`torch.cat`.

        Keyword Args:
            out_key (NestedKey): the key of the result.
            dim (int, optional): the dimension to concatenate along, counted
                in the dimensions of the entries. Defaults to ``0``.
            keep_entries (bool, optional): if ``False``, removes the entries
                of ``keys`` once the result is written. If the concatenation or
                the write fails, every entry is kept with its original value.
                Defaults to ``False``.

        Returns:
            ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(1), b=torch.ones(1))
            >>> _ = td.cat_tensors("a", "b", out_key="c")
            >>> assert "a" not in td
            >>> assert (td["c"] == torch.tensor([0, 1])).all()

        """
        return self._cat_or_stack_tensors(
            *keys,
            out_key=out_key,
            dim=dim,
            keep_entries=keep_entries,
            op=torch.cat,
        )

    def stack_tensors(
        self,
        *keys: NestedKey,
        out_key: NestedKey,
        dim: int = 0,
        keep_entries: bool = False,
    ) -> Self:
        """Stacks entries of the tensordict into a new entry, and removes them by default.

        Args:
            *keys (NestedKey): the entries to stack, with :func:`torch.stack`.

        Keyword Args:
            out_key (NestedKey): the key of the result.
            dim (int, optional): the new dimension, counted in the dimensions
                of the result. Defaults to ``0``.
            keep_entries (bool, optional): if ``False``, removes the entries
                of ``keys`` once the result is written. If the stack or the
                write fails, every entry is kept with its original value.
                Defaults to ``False``.

        Returns:
            ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(()), b=torch.ones(()))
            >>> _ = td.stack_tensors("a", "b", out_key="c")
            >>> assert "a" not in td
            >>> assert (td["c"] == torch.tensor([0, 1])).all()

        """
        return self._cat_or_stack_tensors(
            *keys,
            out_key=out_key,
            dim=dim,
            keep_entries=keep_entries,
            op=torch.stack,
        )

    def cat_from_tensordict(
        self,
        dim: int = 0,
        *,
        sorted: bool | List[NestedKey] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:  # noqa: D417
        """Concatenates all the leaves of the tensordict into one tensor, with :func:`torch.cat`.

        Args:
            dim (int, optional): the dimension to concatenate along, counted
                in the dimensions of the leaves. Defaults to ``0``.

        Keyword Args:
            sorted (bool or list of NestedKey, optional): the order of the
                leaves. ``True`` sorts the keys alphabetically, and ``False``
                or ``None`` keeps the order of insertion. A list of keys gives
                the order, and must hold every leaf key, or a
                ``RuntimeError`` is raised. Defaults to ``None``.
            out (torch.Tensor, optional): a tensor to write the result into.
                Defaults to ``None``.

        Returns:
            The concatenated tensor.

        .. seealso:: :meth:`pack`, which flattens leaves of any shape into one
            tensor and returns the layout to rebuild the tensordict.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(2, 1), b=torch.ones(2, 2), batch_size=[2])
            >>> td.cat_from_tensordict(dim=1)
            tensor([[0., 1., 1.],
                    [0., 1., 1.]])
            >>> td.cat_from_tensordict(dim=1, sorted=["b", "a"])
            tensor([[1., 1., 0.],
                    [1., 1., 0.]])

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

        Packing along the last dim copies the data once, with one
        :func:`torch.cat`. Along another dim the result can take a second
        copy: :func:`torch.cat` keeps the channels-last strides that moving
        the feature dim gives a 4-d or 5-d result, and the result is then made
        contiguous.

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
                leaf raises. ``dim`` must match the layout's dim. Defaults to
                ``None``, which builds a new layout in sorted key order.

        Returns:
            A tuple of the packed tensor and its
            :class:`~tensordict.PackLayout`.

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
            dim=pos - batch_dims - 1,
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
        """Stacks all the leaves of the tensordict into one tensor, with :func:`torch.stack`.

        The leaves must have the same shape.

        Args:
            dim (int, optional): the new dimension, counted in the dimensions
                of the result. Defaults to ``0``.

        Keyword Args:
            sorted (bool or list of NestedKey, optional): the order of the
                leaves. ``True`` sorts the keys alphabetically, and ``False``
                or ``None`` keeps the order of insertion. A list of keys gives
                the order, and must hold every leaf key, or a
                ``RuntimeError`` is raised. Defaults to ``None``.
            out (torch.Tensor, optional): a tensor to write the result into.
                Defaults to ``None``.

        Returns:
            The stacked tensor.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(2), b=torch.ones(2), batch_size=[2])
            >>> td.stack_from_tensordict(dim=1)
            tensor([[0., 1.],
                    [0., 1.]])

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
        """Stacks tensordicts into one dense tensordict along a new dimension ``dim``.

        It is equivalent to :func:`torch.stack`, and also works under
        :func:`torch.compile`. The tensordicts must have the same batch size
        and the same keys, or a ``RuntimeError`` is raised. To stack
        tensordicts with different keys or shapes, use :meth:`~.lazy_stack`.
        See :ref:`stacking`.

        Args:
            input (sequence of TensorDictBase): the tensordicts to stack.
            dim (int, optional): the new batch dimension. Defaults to ``0``.

        Keyword Args:
            out (TensorDictBase, optional): a tensordict to write the result
                into. Defaults to ``None``.

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
        """Concatenates tensordicts along an existing batch dimension ``dim``.

        It is equivalent to :func:`torch.cat`, and also works under
        :func:`torch.compile`. The tensordicts must have the same keys, or a
        ``KeyError`` is raised, and the same batch size except along ``dim``.

        Args:
            input (sequence of TensorDictBase): the tensordicts to concatenate.
            dim (int, optional): the batch dimension to concatenate along.
                Defaults to ``0``.

        Keyword Args:
            out (TensorDictBase, optional): a tensordict to write the result
                into. Defaults to ``None``.

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
        """Stacks tensordicts into a lazy stack.

        A lazy stack keeps the tensordicts as they are and stacks their values
        only when you read them. The tensordicts can have different keys and
        different feature shapes. See :ref:`stacking`.

        Args:
            input (sequence of TensorDictBase): the tensordicts to stack.
            dim (int, optional): the new batch dimension. Defaults to ``0``.

        Keyword Args:
            out (TensorDictBase, optional): a lazy stack to write the result
                into. Defaults to ``None``.
            **kwargs: passed to
                :meth:`LazyStackedTensorDict.lazy_stack
                <tensordict.LazyStackedTensorDict.lazy_stack>`, for example
                ``stack_dim_name`` or ``strict_shape``.

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
        """Stacks tensordicts densely if possible, or into a lazy stack otherwise.

        The result is a :class:`~tensordict.TensorDict` if the tensordicts
        have the same keys and shapes, and a
        :class:`~tensordict.LazyStackedTensorDict` otherwise. See
        :ref:`stacking`.

        Args:
            input (sequence of TensorDictBase): the tensordicts to stack.
            dim (int, optional): the new batch dimension. Defaults to ``0``.

        Keyword Args:
            out (TensorDictBase, optional): a tensordict to write the result
                into. Defaults to ``None``.
            **kwargs: passed to
                :meth:`LazyStackedTensorDict.maybe_dense_stack
                <tensordict.LazyStackedTensorDict.maybe_dense_stack>`, for
                example ``strict``.

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
        """Splits the tensordict into ``chunks`` pieces along a batch dimension, as :func:`torch.chunk` does.

        Each piece has ``ceil(batch_size[dim] / chunks)`` elements, except the
        last one, which can be smaller. So there can be fewer than ``chunks``
        pieces. The pieces hold views of the original tensors.

        Args:
            chunks (int): the number of pieces.
            dim (int, optional): the batch dimension to split. Defaults to
                ``0``.

        Returns:
            A tuple of tensordicts.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(x=torch.arange(5), batch_size=[5])
            >>> [t["x"] for t in td.chunk(2)]
            [tensor([0, 1, 2]), tensor([3, 4])]
            >>> [t["x"] for t in td.chunk(4)]
            [tensor([0, 1]), tensor([2, 3]), tensor([4])]

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
        """Splits the tensordict into ``chunks`` pieces along ``dim``, for :meth:`chunk`.

        :meth:`chunk` has checked ``chunks``, turned it into a positive
        ``int``, made ``dim`` non-negative and handled a zero-size ``dim``.
        The hook returns the pieces as ``torch.chunk`` would: pieces of the
        upper-rounded size (the last one can be smaller), and fewer than ``chunks``
        pieces when the size does not allow more. The default calls
        :meth:`split`.
        """
        # torch.chunk splits with the upper-rounded size
        return self.split(-(self.batch_size[dim] // -chunks), dim)

    def split(
        self, split_size: int | list[int], dim: int = 0
    ) -> tuple[TensorDictBase, ...]:
        """Splits the tensordict along a batch dimension, as :func:`torch.split` does.

        The pieces hold views of the original tensors, and keep the names.

        On a :class:`~tensordict.LazyStackedTensorDict`, sizes along the stack
        dim that are negative or do not add up to the stack size emit a
        :class:`FutureWarning`, and the pieces miss members or have a wrong
        batch size. Starting with TensorDict 0.17, the default will change to
        raising a ``RuntimeError``, as ``TensorDict.split`` does.

        Args:
            split_size (int or list of int): the size of each piece (the last
                one can be smaller), or the list of the sizes of the pieces,
                which must add up to ``batch_size[dim]``.
            dim (int, optional): the batch dimension to split. Defaults to
                ``0``.

        Returns:
            A tuple of tensordicts.

        Raises:
            RuntimeError: if the sizes of a list do not add up to
                ``batch_size[dim]``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(x=torch.arange(5), batch_size=[5])
            >>> [t["x"] for t in td.split(2)]
            [tensor([0, 1]), tensor([2, 3]), tensor([4])]
            >>> [t["x"] for t in td.split([1, 4])]
            [tensor([0]), tensor([1, 2, 3, 4])]
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
                num_full, rest = max_size // split_size, max_size % split_size
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
        # The front has checked dim and the sizes, and a sub-tensordict has
        # the same batch dim: call its hook directly. A call through the
        # front adds a Python frame per nesting level, which slows deep
        # tensordicts.
        splits = {
            k: (
                (v,) * num_splits
                if _is_unbatched(v)
                else v.split(split_sizes, dim)
                if type(v) is Tensor
                or _is_tensorclass(type(v))
                or not _is_tensor_collection(type(v))
                else v._split(split_sizes, dim)
            )
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
                    split[key] = _unbatched_with_batch_size(value, bsz)
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
        """Gathers values along a batch dimension, as :func:`torch.gather` does.

        ``index`` has as many dimensions as the batch, and gives the batch size
        of the result; each leaf keeps its feature dimensions. As for
        :func:`torch.gather`, ``index`` can be smaller than the batch along
        the other dimensions. The data is copied, and the names are kept.

        Args:
            dim (int): the batch dimension to gather along.
            index (torch.Tensor): a long tensor of indices along ``dim``, with
                as many dimensions as the batch.
            out (TensorDictBase, optional): a tensordict to write the result
                into, with the batch size ``index.shape``. Exclusive with
                ``inplace``. Defaults to ``None``.

        Keyword Args:
            inplace (bool, optional): if ``True``, the tensordict keeps its
                identity and keys, its leaves are replaced one by one, and the
                call returns ``self``. It needs ``index.ndim == batch_dims``.
                Exclusive with ``out``. A lazy stack, or a tensordict that
                holds one, raises a ``NotImplementedError``: call
                :meth:`to_tensordict` first. See :ref:`inplace-conventions`.
                Defaults to ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     a=torch.arange(12).view(3, 4),
            ...     b=TensorDict(c=torch.arange(12).view(3, 4, 1), batch_size=[3, 4, 1]),
            ...     batch_size=[3, 4],
            ... )
            >>> index = torch.tensor([[0, 3], [1, 2], [3, 3]])
            >>> out = td.gather(dim=1, index=index)
            >>> out["a"]
            tensor([[ 0,  3],
                    [ 5,  6],
                    [11, 11]])
            >>> out.batch_size, out["b", "c"].shape
            (torch.Size([3, 2]), torch.Size([3, 2, 1]))

            The names of the batch dimensions are kept:

            >>> td.names = ["x", "y"]
            >>> td.gather(dim=1, index=index).names
            ['x', 'y']
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

            self._raise_if_holds_lazy_stack("gather", new_batch_size=new_batch_size)
            warns = self._gather_inplace_warns and new_batch_size == self.batch_size
            self._inplace_rebind_leaves(leaf_fn, nested_fn, new_batch_size)
            if warns:
                # after the write, so that a call that raises does not warn
                warnings.warn(
                    f"gather(inplace=True) on a {type(self).__name__} over a "
                    "lazy tensordict works when the index has the batch shape. "
                    "Starting with TensorDict 0.17, the default will change to "
                    "raising a NotImplementedError, as LazyStackedTensorDict does.",
                    FutureWarning,
                    stacklevel=_user_stacklevel(),
                )
            return self
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
        """Returns a tensordict with a new batch size, whose leaves are views of the original leaves.

        Each leaf keeps its feature dimensions and shares storage with the
        original leaf. As for :meth:`torch.Tensor.view`, the leaves must have a
        compatible memory layout: otherwise, use :meth:`~.reshape`. The result
        has no names, and the tensordict itself is returned when the batch
        size does not change.

        ``view(dtype)`` views every leaf with another dtype instead, as
        :meth:`torch.Tensor.view` does. The last dimension of a leaf changes
        size when the dtypes have different sizes, so the leaves then need a
        feature dimension.

        It can be used as a context manager, as :meth:`unsqueeze` can: when
        the ``with`` block exits, the entries of the block are written back to
        the tensordict with the inverse operation.

        Args:
            *shape (int): the new batch size, as integers or as one
                ``torch.Size``, list or tuple. One dimension can be ``-1``.
                Alternatively, a ``torch.dtype``.

        Keyword Args:
            size (list, tuple or torch.Size, optional): the new batch size,
                as a keyword argument instead of ``*shape``. Defaults to
                ``None``.
            batch_size (torch.Size, optional): with a dtype, the batch size of
                the result. It is not used with a shape. Defaults to ``None``,
                which keeps the batch size.

        Returns:
            A tensordict with the new batch size.

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
            >>> td_int = td.view(torch.int32)
            >>> td_int["a"].dtype, td_int.batch_size
            (torch.int32, torch.Size([3, 4]))

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
        """Returns a tensordict with the batch dimensions ``dim0`` and ``dim1`` swapped.

        The leaves of the result are views of the original leaves, so an
        in-place change of the result changes the tensordict too. The names
        are swapped with the dimensions. To write new entries back to the
        tensordict, use the result as a context manager.

        Args:
            dim0 (int): the first batch dimension to swap.
            dim1 (int): the second batch dimension to swap.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> tensordict = TensorDict({"a": torch.zeros(3, 4, 5)}, [3, 4])
            >>> tensordict_transpose = tensordict.transpose(0, 1)
            >>> print(tensordict_transpose.shape)
            torch.Size([4, 3])
            >>> with tensordict.transpose(0, 1) as tensordict_transpose:
            ...     tensordict_transpose["b"] = torch.zeros(4, 3)
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
        """Returns a tensordict with two batch dimensions swapped, as :meth:`~.transpose` does.

        ``swapaxes`` and ``swapdims`` are aliases of :meth:`~.transpose`. It
        can be used as a context manager, as :meth:`~.transpose` can.

        Args:
            axis0 (int): the first batch dimension to swap.
            axis1 (int): the second batch dimension to swap.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.zeros(3, 4, 5)}, batch_size=[3, 4])
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
        """Returns a tensordict with its batch dimensions permuted, as :meth:`torch.Tensor.permute` does.

        The leaves keep their feature dimensions after the batch dimensions,
        and are views of the original leaves. The names are permuted with the
        dimensions.

        It can be used as a context manager, as :meth:`unsqueeze` can: when
        the ``with`` block exits, the entries of the block are written back to
        the tensordict with the inverse operation.

        Args:
            *dims (int): the new order of the batch dimensions, as integers,
                as one list or tuple, or as the keyword argument ``dims``.

        Returns:
            A tensordict with the batch dimensions in the new order.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.zeros(3, 4, 5)}, [3, 4])
            >>> td.permute(1, 0)["a"].shape
            torch.Size([4, 3, 5])
            >>> td.permute([1, 0]).batch_size
            torch.Size([4, 3])
            >>> td.permute(dims=[1, 0]).batch_size
            torch.Size([4, 3])
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
        """Returns a tensordict with batch dimensions moved to new positions, as :func:`torch.movedim` does.

        The batch dimensions that are not moved keep their order, at the
        positions that ``destination`` leaves free. The leaves are views of the
        original leaves, and the names move with the dimensions.

        It can be used as a context manager, as :meth:`unsqueeze` can: when
        the ``with`` block exits, the entries of the block are written back to
        the tensordict with the inverse operation.

        Args:
            source (int or tuple of int): the batch dimensions to move. They
                must be unique.
            destination (int or tuple of int): the new position of each
                dimension of ``source``. They must be unique.

        Returns:
            A tensordict with the batch dimensions moved.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.zeros(3, 4, 5)}, batch_size=[3, 4])
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
        """Returns a tensordict with the elements reversed along batch dimensions, as :func:`torch.flip` does.

        The batch size and the names do not change, and the feature dimensions
        are not flipped. The data is copied.

        It can be used as a context manager, as :meth:`unsqueeze` can: when
        the ``with`` block exits, the entries of the block are written back to
        the tensordict with the inverse operation.

        Args:
            dims (int or tuple of int): the batch dimensions to flip.

        Raises:
            IndexError: if a dimension is not a batch dimension.

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
        """Returns a tensordict with the elements reversed along batch dimension 1.

        It is ``flip(1)``, as for :func:`torch.fliplr`. The tensordict needs at
        least 2 batch dimensions, or a ``RuntimeError`` is raised. The data is
        copied.

        It can be used as a context manager, as :meth:`unsqueeze` can: when
        the ``with`` block exits, the entries of the block are written back to
        the tensordict with the inverse operation.

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
        """Returns a tensordict with the elements reversed along batch dimension 0.

        It is ``flip(0)``, as for :func:`torch.flipud`. The tensordict needs at
        least 1 batch dimension, or a ``RuntimeError`` is raised. The data is
        copied.

        It can be used as a context manager, as :meth:`unsqueeze` can: when
        the ``with`` block exits, the entries of the block are written back to
        the tensordict with the inverse operation.

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
        """Returns a tensordict with the elements rolled along batch dimensions, as :func:`torch.roll` does.

        Elements shifted past the last position come back at the first one.
        The batch size and the names do not change, and the feature dimensions
        are not rolled. The data is copied.

        It can be used as a context manager, as :meth:`unsqueeze` can: when
        the ``with`` block exits, the entries of the block are written back to
        the tensordict with the inverse operation.

        Args:
            shifts (int or tuple of int): the number of places to shift the
                elements by. With a tuple, ``dims`` is a tuple of the same
                size, and each dimension is rolled by its own shift.
            dims (int or tuple of int, optional): the batch dimensions to roll.
                Defaults to ``None``, which rolls the flattened batch and
                restores its shape.

        Keyword Args:
            inplace (bool, optional): if ``True``, the tensordict keeps its
                identity and keys, its leaves are replaced one by one, and the
                call returns ``self``. See :ref:`inplace-conventions`. Defaults
                to ``False``.

        Returns:
            The rolled tensordict, or ``self`` with ``inplace=True``.

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
        """Returns a tensordict rotated by 90 degrees in the plane of two batch dimensions.

        As for :func:`torch.rot90`, the rotation goes from the first dimension
        of ``dims`` towards the second. The tensordict needs at least 2 batch
        dimensions, or a ``RuntimeError`` is raised. The feature dimensions are
        not rotated, the names follow their dimensions, and the data is copied.

        It can be used as a context manager, as :meth:`unsqueeze` can: when
        the ``with`` block exits, the entries of the block are written back to
        the tensordict with the inverse operation.

        Args:
            k (int, optional): the number of rotations. Defaults to ``1``.
            dims (tuple of int, optional): the two batch dimensions of the
                plane. Defaults to ``(0, 1)``.

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
        """Returns a tensordict narrowed along a batch dimension, as :meth:`torch.Tensor.narrow` does.

        The result keeps the elements ``start`` to ``start + length - 1`` of
        ``dim``. Its leaves are views of the original leaves, and it keeps the
        names.

        Args:
            dim (int): the batch dimension to narrow.
            start (int): the first element to keep.
            length (int): the number of elements to keep.

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
        """Returns a tensordict whose batch is repeated, as :func:`torch.tile` does.

        If ``dims`` has fewer entries than the batch dimensions, it is padded
        with leading ones. If it has more, the result gets new leading batch
        dimensions. The feature dimensions are not repeated, the result has no
        names, and the data is copied. See also :meth:`repeat`.

        Args:
            dims (int or tuple of int): the number of repetitions along each
                batch dimension.

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
        """Returns the tensordict broadcast to the batch size ``shape``, as :func:`torch.broadcast_to` does.

        As for :meth:`expand`, the new shape can add leading dimensions and
        expand the dimensions of size 1, and ``-1`` keeps the size of a
        dimension. The leaves of the result are views of the original leaves,
        and the result has no names.

        Args:
            shape (tuple of int or torch.Size): the new batch size.

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

        If the tensordict already has 1 or more batch dimensions, it is
        returned as it is. Otherwise, a dimension of size 1 is prepended. The
        leaves of the result are views of the original leaves.

        It can be used as a context manager, as :meth:`unsqueeze` can.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.zeros(3)}, batch_size=[])
            >>> print(td.atleast_1d().shape)
            torch.Size([1])
        """
        if self.ndim >= 1:
            return self
        return self.unsqueeze(0)

    @_as_context_manager()
    def atleast_2d(self):
        """Returns the tensordict with at least 2 batch dimensions.

        If the tensordict already has 2 or more batch dimensions, it is
        returned as it is. Otherwise, dimensions of size 1 are prepended to
        reach 2 dimensions. The leaves of the result are views of the original
        leaves.

        It can be used as a context manager, as :meth:`unsqueeze` can.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
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

        If the tensordict already has 3 or more batch dimensions, it is
        returned as it is. Otherwise, dimensions of size 1 are prepended to
        reach 3 dimensions. The leaves of the result are views of the original
        leaves. Unlike :func:`torch.atleast_3d`, the new dimensions always come
        first.

        It can be used as a context manager, as :meth:`unsqueeze` can.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
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
        """Returns the tensordict with its lazy stacks replaced by dense tensordicts.

        The entries of a lazy stack are stacked into one tensor when their
        shapes match, and into a nested tensor (:mod:`torch.nested`) along the
        stack dimension otherwise. A tensordict that holds no lazy stack is
        returned as it is.

        Args:
            layout (torch.layout, optional): the layout of the nested tensors,
                ``torch.strided`` or ``torch.jagged``. Defaults to
                ``torch.strided``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> stack = TensorDict.lazy_stack(
            ...     [TensorDict(a=torch.zeros(3)), TensorDict(a=torch.ones(4))]
            ... )
            >>> dense = stack.densify(layout=torch.jagged)
            >>> type(dense).__name__, dense["a"].is_nested
            ('TensorDict', True)

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
        """Returns a tensordict whose batch dimensions ``start_dim`` to ``end_dim`` are flattened into one.

        The leaves keep their feature dimensions. As for :func:`torch.flatten`,
        a leaf of the result is a view of the original leaf when the memory
        layout allows it, and a copy otherwise. The flattened dimension has no
        name. A tensordict without batch dimensions gets one batch dimension
        of size 1.

        A ``start_dim`` or ``end_dim`` outside the batch dimensions warns, and
        is used on the dimensions of the leaves. Starting with TensorDict 0.17,
        the default will change to raising an ``IndexError``, as for a tensor.

        It can be used as a context manager, as :meth:`unsqueeze` can: when
        the ``with`` block exits, the entries of the block are written back to
        the tensordict with the inverse operation.

        Args:
            start_dim (int, optional): the first batch dimension to flatten.
                Defaults to ``None``, which is ``0``.
            end_dim (int, optional): the last batch dimension to flatten.
                Defaults to ``None``, which is ``-1``.

        Keyword Args:
            inplace (bool, optional): if ``True``, the tensordict keeps its
                identity and keys, its leaves are replaced one by one, and the
                call returns ``self``. A leaf keeps its storage when the
                flatten can return a view of it. A lazy stack, or a tensordict
                that holds one, raises a ``NotImplementedError``: call
                :meth:`to_tensordict` first. See :ref:`inplace-conventions`.
                Defaults to ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     "a": torch.arange(60).view(3, 4, 5),
            ...     "b": torch.arange(12).view(3, 4)}, batch_size=[3, 4])
            >>> td_flat = td.flatten(0, 1)
            >>> td_flat.batch_size
            torch.Size([12])
            >>> td_flat["a"].shape
            torch.Size([12, 5])
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
            self._raise_if_holds_lazy_stack(
                "flatten", new_batch_size=torch.Size(batch_size)
            )

            def nested_fn(nested):
                nested.flatten(start_dim, end_dim, inplace=True)

            # the sizes that a with-block restores on exit
            sizes = self.batch_size[start_dim : end_dim + 1]
            self._inplace_rebind_leaves(flatten, nested_fn, torch.Size(batch_size))
            if names is not None:
                self.names = names
            self._inplace_flatten_sizes = (
                None if past_batch_dims else (start_dim, sizes)
            )
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
        """Returns a tensordict whose batch dimension ``dim`` is split into several.

        It works as :func:`torch.unflatten` does on the batch. The leaves keep
        their feature dimensions, and are views of the original leaves when the
        memory layout allows it. The last new dimension keeps the name of
        ``dim``, and the others have no name.

        It can be used as a context manager, as :meth:`unsqueeze` can: when
        the ``with`` block exits, the entries of the block are written back to
        the tensordict with the inverse operation.

        Args:
            dim (int): the batch dimension to unflatten.
            unflattened_size (tuple of int or torch.Size): the sizes of the new
                dimensions. Their product is ``batch_size[dim]``, and one of
                them can be ``-1``.

        Keyword Args:
            inplace (bool, optional): if ``True``, the tensordict keeps its
                identity and keys, its leaves are replaced one by one, and the
                call returns ``self``. A leaf keeps its storage when the
                unflatten can return a view of it. A lazy stack, or a
                tensordict that holds one, raises a ``NotImplementedError``:
                call :meth:`to_tensordict` first. See
                :ref:`inplace-conventions`. Defaults to ``False``.

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
            self._raise_if_holds_lazy_stack(
                "unflatten", new_batch_size=torch.Size(batch_size)
            )

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

        Every rank of the process group calls it, in an initialized
        :mod:`torch.distributed` process group. On the source rank,
        ``tensordicts`` holds one tensordict per rank; each is consolidated
        (see :meth:`consolidate`) and sent as one buffer. The other ranks
        rebuild their tensordict from the buffer they receive. The tensordict
        the method is called on is not used. See :ref:`distributed`.

        Args:
            src (int): the rank of the source process.
            tensordicts (list of TensorDictBase, optional): on the source rank,
                the tensordicts to scatter, one per rank. Ignored on the other
                ranks. Defaults to ``None``.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): the process group.
                Defaults to ``None``, which uses the default group.
            device (torch.device or str, optional): on the receiving ranks, the
                device of the receive buffer. Defaults to ``None``, which uses
                the device of the consolidated storage of the source.

        Returns:
            The tensordict that this rank receives.
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
        """Returns a tensordict whose nested-tensor leaves are converted to padded tensors.

        Calls :func:`torch.nested.to_padded_tensor` on every leaf that is a
        nested tensor, and keeps the other leaves. If the batch size has a
        ``-1`` (a ragged dimension, as in a dense version of a lazy stack
        of tensordicts with different batch sizes), the batch size of the
        result is computed from the padded leaves.

        Args:
            padding (float, optional): the value of the padding. Defaults to
                ``0.0``.
            mask_key (NestedKey, optional): a key where a boolean mask of the
                valid (not padded) elements is written. It needs a ragged batch
                dimension, or a ``RuntimeError`` is raised. Defaults to
                ``None``, which writes no mask.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> stack = TensorDict.lazy_stack([
            ...     TensorDict(a=torch.zeros(3), batch_size=[3]),
            ...     TensorDict(a=torch.ones(4), batch_size=[4]),
            ... ])
            >>> td = stack.densify(layout=torch.jagged)
            >>> td.batch_size
            torch.Size([2, -1])
            >>> padded = td.to_padded_tensor(padding=-1.0, mask_key="mask")
            >>> padded.batch_size
            torch.Size([2, 4])
            >>> padded["a"]
            tensor([[ 0.,  0.,  0., -1.],
                    [ 1.,  1.,  1.,  1.]])
            >>> padded["mask"]
            tensor([[ True,  True,  True, False],
                    [ True,  True,  True,  True]])

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
        """Returns a tensordict with the elements of the batch where ``mask`` is ``True``.

        The batch dimensions that ``mask`` covers are flattened into one, of
        size ``mask.sum()``, and the other batch dimensions are kept, as for
        ``td[mask]``. The data is copied. The result is a
        :class:`~tensordict.TensorDict` without names.

        Args:
            mask (torch.Tensor): a boolean mask whose shape is the batch size,
                or its leading dimensions.

        Keyword Args:
            inplace (bool, optional): if ``True``, the tensordict keeps its
                identity and keys, its leaves are replaced one by one, and the
                call returns ``self``. A lazy stack, or a tensordict that
                holds one, raises a ``NotImplementedError``: call
                :meth:`to_tensordict` first. See :ref:`inplace-conventions`.
                Defaults to ``False``.

        Returns:
            The masked tensordict, or ``self`` with ``inplace=True``.

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
        if is_compiling():
            # count the mask as the leaves do, so that they share one symbol
            # (see _advanced_shape)
            dim = mask_expand.nonzero().shape[0]
        else:
            dim = int(mask.sum().item())
        other_dim = self.shape[mask.ndim :]
        batch_size = torch.Size([dim, *other_dim])
        if inplace:
            self._raise_if_holds_lazy_stack("masked_select", new_batch_size=batch_size)

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
                d[key] = _unbatched_with_batch_size(value, batch_size)
            else:
                d[key] = value[mask_expand]
        return TensorDict(device=self.device, source=d, batch_size=batch_size)
