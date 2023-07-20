# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict._tensordict import _as_shape, _populate_index
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_map


def _lazy_init(func):
    """A caching helper."""
    name = "_" + func.__name__

    def setter(self, value):
        setattr(self, name, value)

    def new_func(self):
        if not hasattr(self, name):
            r = func(self)
            setter(self, r)
            return r
        return getattr(self, name)

    return property(new_func, setter)


def _broadcast_shapes(*shapes):
    """A modified version of torch.broadcast_shapes that accepts -1."""
    max_len = 0
    for shape in shapes:
        if isinstance(shape, int):
            if max_len < 1:
                max_len = 1
        elif isinstance(shape, (tuple, list)):
            s = len(shape)
            if max_len < s:
                max_len = s
    result = [1] * max_len
    for shape in shapes:
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, (tuple, list)):
            for i in range(-1, -1 - len(shape), -1):
                cur_shape = shape[i]
                if cur_shape == -1:
                    cur_shape = None  # in double we use None as placeholder, which equals nothing
                if cur_shape == 1 or cur_shape == result[i]:
                    continue
                if result[i] == -1:
                    # in this case, we consider this as het dim
                    continue
                if result[i] != 1:
                    raise RuntimeError(
                        "Shape mismatch: objects cannot be broadcast to a single shape"
                    )
                result[i] = shape[i]
        else:
            raise RuntimeError(
                "Input shapes should be of type ints, a tuple of ints, or a list of ints, got ",
                shape,
            )
    return torch.Size(result)


class _NestedShape:
    def __new__(cls, shapes):
        # TODO: if a tensor with nume() == tensor.shape[-1], then return a regular tensor
        if isinstance(shapes, _NestedShape):
            return shapes
        return super().__new__(cls)

    def __init__(self, shapes):
        if not isinstance(shapes, torch.Tensor):
            shapes = torch.tensor(shapes)
        self._shapes = shapes

    @_lazy_init
    def _offsets(self):
        common_shape = self.common_shape
        shapes = self._shapes
        if common_shape:
            shapes = shapes[..., : -len(common_shape)]
        return shapes.prod(-1)

    @_lazy_init
    def _offsets_cs(self):
        common_shape = self.common_shape
        shapes = self._shapes
        if common_shape:
            shapes = shapes[..., : -len(common_shape)]
        cs = shapes.prod(-1).view(-1).cumsum(0)
        cs_pad = torch.nn.functional.pad(cs[:-1], [1, 0])
        return torch.stack(
            [
                cs_pad.view(shapes.shape[:-1]),
                cs.view(shapes.shape[:-1]),
            ]
        )

    def unfold(self):
        """Converts the shape to the maximum-indexable format.

        Examples:
            >>> ns = _NestedShape(([11, 2, 3], [11, 5, 3]))
            >>> print(ns.batch_dim)
            torch.Size([2])
            >>> print(ns.unfold().batch_dim)
            torch.Size([2, 11])
        """
        out = _NestedShape(self._shapes.clone())
        is_unique, val = out.is_unique(out._shapes.ndim - 1)
        while is_unique:
            out._shapes = (
                out._shapes[..., 1:]
                .unsqueeze(-2)
                .expand(*out._shapes.shape[:-1], val, -1)
            )
            is_unique, val = out.is_unique(out._shapes.ndim - 1)
        return out

    @_lazy_init
    def ndim(self):
        return self._shapes.ndim - 1 + self._shapes.shape[-1]

    def is_unique(self, dim):
        if dim < 0:
            dim = self.ndim + dim
        if dim < 0 or dim >= self.ndim:
            raise RuntimeError
        if dim < self._shapes.ndim - 1:
            return (True, self._shapes.shape[dim])
        v = self.as_shape[dim - self._shapes.ndim + 1]
        return v != -1, v

    @_lazy_init
    def het_dims(self):
        return [dim for dim in range(self.ndim) if self.as_shape[dim] == -1]

    def numel(self):
        return (
            self._offsets_cs[(1,) + (-1,) * (self._shapes.ndim - 1)]
            * self.common_shape.numel()
        )

    @property
    def batch_dim(self):
        return self._shapes.shape[:-1]

    @_lazy_init
    def common_shape(self):
        shape = []

        for v in reversed(self.as_shape):
            if v != -1:
                shape.append(v)
            else:
                break
        return torch.Size(reversed(shape))

    @classmethod
    def broadcast_shape(cls, shape: torch.Size, nested_shape: _NestedShape):
        broadcast_shape = _broadcast_shapes(shape, nested_shape.as_shape)
        return nested_shape.expand(broadcast_shape)

    def expand(self, *broadcast_shape):
        if len(broadcast_shape) == 1 and isinstance(broadcast_shape[0], (tuple, list)):
            broadcast_shape = broadcast_shape[0]
        as_shape = self.as_shape
        if len(broadcast_shape) == len(as_shape) and all(
            s1 == s2 or s1 == -1 or s2 == -1
            for (s1, s2) in zip(broadcast_shape, as_shape)
        ):
            return self

        # trailing dims, ie dims that are registered
        broadcast_shape_trailing = broadcast_shape[self._shapes.shape[-1] :]
        broadcast_shape_trailing = _broadcast_shapes(
            broadcast_shape_trailing, as_shape[len(self.batch_dim) :]
        )
        # replace trailing dims
        shapes = self._shapes.clone()
        for i in range(-1, len(broadcast_shape_trailing) - 1, -1):
            if as_shape[i] != -1:
                shapes[..., i] = broadcast_shape_trailing[i]

        # leading dims, ie dims that are not explicitely registered
        broadcast_shape_leading = broadcast_shape[: -self.ndim]

        # find first -1 in broadcast_shape
        if not len(broadcast_shape_leading):
            return _NestedShape(shapes)

        return _NestedShape(
            shapes.expand(*broadcast_shape_leading, *self._shapes.shape)
        )

    @_lazy_init
    def is_plain(self):
        return not self.het_dims

    def __getitem__(self, item):
        try:
            return _NestedShape(self._shapes[item])
        except IndexError as err:
            if "too many indices" in str(err):
                raise IndexError(
                    "Cannot index along dimensions on the right of the heterogeneous dimension."
                )

    @_lazy_init
    def as_shape(self):
        shape_cpp = torch.Size(_as_shape(self._shapes))
        return shape_cpp
        # first_shape = self._shapes[(0,) * (self._shapes.ndim - 1)].clone()
        # unique = (self._shapes == first_shape).view(-1, self._shapes.shape[-1]).all(0)
        # first_shape[~unique] = -1
        # shape = list(self._shapes.shape[:-1]) + list(first_shape)
        # return torch.Size(shape)

    def __repr__(self):
        return str(self.as_shape)

    def __eq__(self, other):
        return (self._shapes == other).all()

    def __ne__(self, other):
        return (self._shapes != other).any()


def get_parent_class(f):
    return f.__globals__.get(f.__qualname__.split(".")[0], None)


def _copy_shapes(func):
    def new_func(self, *args, **kwargs):
        out = getattr(torch.Tensor, func.__name__)(self, *args, **kwargs)
        _shapes = self._shapes
        out._shapes = _shapes
        return out

    return new_func


def _broadcast(func):
    def new_func(self, other):
        other_shape = getattr(other, "shape", torch.Size([]))
        shapes = _NestedShape.broadcast_shape(other_shape, self._shapes)
        compact = self._compact
        if isinstance(other, TensorStack):
            other = other._compact
            if shapes != self._shapes:
                raise RuntimeError("broadcast between TensorStack not implemented yet.")
                # other = other.unsqueeze(-2)
        elif isinstance(other, Tensor) and shapes != self._shapes:
            # we need to squash
            other = other.reshape(
                *other.shape[: -self.ndim], -1, *other.shape[-len(compact.shape[1:]) :]
            )
        out = getattr(torch.Tensor, func.__name__)(compact, other)
        return TensorStack(out, shapes=shapes)

    return new_func


class TensorStack(torch.Tensor):
    def __new__(cls, tensor, *, shapes):
        if shapes.is_plain:
            return tensor.reshape(shapes.as_shape)
        return super().__new__(cls, tensor)

    def __init__(self, tensor, *, shapes, unfold=False):
        super(TensorStack, self).__init__()
        if not isinstance(shapes, _NestedShape):
            raise ValueError("shapes must be a _NestedShape instance")
        if unfold:
            shapes = shapes.unfold()
        self._shapes = shapes

    @classmethod
    def from_tensors(cls, tensors):
        if not len(tensors):
            raise RuntimeError
        shapes = _NestedShape(tree_map(lambda x: x.shape, tensors))
        return TensorStack(
            torch.cat([t.view(-1) for t in tree_flatten(tensors)[0]]), shapes=shapes
        )

    def numel(self):
        return self._shapes.numel()

    @property
    def shape(self):
        return self._shapes.as_shape

    def unstack(self):
        raise NotImplementedError

    @property
    def _flat(self):
        # represents the tensor as a flat one
        return super().view(-1)

    @property
    def _compact(self):
        # represents the tensor with a compact structure (rightmost consistent dims-wise)
        return torch.Tensor(super().view(-1, *self._shapes.common_shape))

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        if isinstance(shape, _NestedShape):
            if shape.numel() != self.numel():
                raise ValueError
            out = TensorStack(self, shapes=shape)
            return out
        if len(shape) == 1 and shape[0] == -1:
            return self._flat
        n = self.numel()
        shape = torch.Size(shape)
        common_shape = self._shapes.common_shape
        compact_shape = torch.Size([n // common_shape.numel(), *common_shape])
        if shape in (torch.Size([-1, *common_shape]), compact_shape):
            return self._compact
        raise RuntimeError(shape)

    @property
    def ndim(self):
        return len(self.shape)

    def ndimension(self):
        return self.ndim

    def __getitem__(self, index):
        if isinstance(index, (int,)):
            idx_beg = self._shapes._offsets_cs[0, index]
            idx_end = self._shapes._offsets_cs[1, index]
            shapes = self._shapes[index]
            out = self._compact.__getitem__(slice(idx_beg, idx_end))
            out = TensorStack(out, shapes=shapes)
            return out
        shapes = self._shapes[index]
        # TODO: capture wrong indexing
        elts = _populate_index(
            self._shapes._offsets[index].view(-1),
            self._shapes._offsets_cs[0][index].view(-1),
        )
        tensor = self._compact[elts]
        return TensorStack(tensor, shapes=shapes)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype}, device={self.device})"

    @_copy_shapes
    def to(self, *args, **kwargs):
        ...

    @_copy_shapes
    def cpu(self):
        ...

    @_copy_shapes
    def bool(self):
        ...

    @_copy_shapes
    def float(self):
        ...

    @_copy_shapes
    def double(self):
        ...

    @_copy_shapes
    def int(self):
        ...

    @_copy_shapes
    def cuda(self):
        ...

    @_copy_shapes
    def __neg__(self):
        ...

    @_copy_shapes
    def __abs__(self):
        ...

    @_copy_shapes
    def __inv__(self):
        ...

    @_copy_shapes
    def __invert__(self):
        ...

    @_broadcast
    def add(self, other):
        ...

    @_broadcast
    def div(self, other):
        ...

    @_broadcast
    def rdiv(self, other):
        ...

    @_broadcast
    def __add__(self, other):
        ...

    @_broadcast
    def __mod__(self, other):
        ...

    @_broadcast
    def __pow__(self, other):
        ...

    @_broadcast
    def __sub__(self, other):
        ...

    @_broadcast
    def __truediv__(self, other):
        ...

    @_broadcast
    def __eq__(self, other):
        ...

    @_broadcast
    def __ne__(self, other):
        ...

    @_broadcast
    def __div__(self, other):
        ...

    @_broadcast
    def __floordiv__(self, other):
        ...

    @_broadcast
    def __lt__(self, other):
        ...

    @_broadcast
    def __le__(self, other):
        ...

    @_broadcast
    def __ge__(self, other):
        ...

    @_broadcast
    def __gt__(self, other):
        ...

    @_broadcast
    def __rdiv__(self, other):
        ...

    @_broadcast
    def __mul__(self, other):
        ...
