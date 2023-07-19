# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tensordict._tensordict import _populate_index


class _NestedSize(tuple):
    def numel(self):
        out = 0
        for elt in self:
            if isinstance(elt, (tuple, torch.Size)):
                out += elt.numel()
            else:
                out = max(out * elt, elt)
        return out

    def __getitem__(self, index):
        out = super().__getitem__(index)
        if _NestedSize.is_nested(out):
            return out
        if isinstance(out, tuple):
            return torch.Size(out)
        return out

    @classmethod
    def is_nested(cls, obj):
        return isinstance(obj, tuple) and isinstance(obj[0], tuple)

    @classmethod
    def broadcast_shape(cls, other_shape, shapes):
        if not cls.is_nested(shapes):
            new_shape = _NestedSize(torch.broadcast_shapes(other_shape, shapes))
        else:
            new_shape = _NestedSize(
                cls.broadcast_shape(other_shape, shape) for shape in shapes
            )
        return new_shape

    @classmethod
    def from_list(cls, nested_list):
        if not isinstance(nested_list, list):
            return nested_list
        return cls([_NestedSize.from_list(elt) for elt in nested_list])

    @classmethod
    def refine_shapes(cls, first_neg_right_index, shapes, new_shape):
        shapes = torch.tensor(shapes)
        shapes = shapes.view(new_shape[:first_neg_right_index])
        shapes = shapes.tolist()
        return _NestedSize.from_list(shapes)


def get_parent_class(f):
    return f.__globals__.get(f.__qualname__.split(".")[0], None)


def _lazy_init(func):
    name = "_" + func.__name__

    def new_func(self):
        if not hasattr(self, name):
            self._init()
        return getattr(self, name)

    def setter(self, value):
        setattr(self, name, value)

    return property(new_func, setter)


def _copy_shapes(func):
    def new_func(self, *args, **kwargs):
        out = getattr(torch.Tensor, func.__name__)(self, *args, **kwargs)
        _shapes = self._shapes
        out._shapes = _shapes
        return out

    return new_func


def _broadcast(func):
    def new_func(self, other):
        out = getattr(torch.Tensor, func.__name__)(
            self.view(-1, *self._trailing_dims), other
        )._flat
        other_shape = getattr(other, "shape", torch.Size([]))
        out._shapes = _NestedSize.broadcast_shape(other_shape, self._shapes)
        return out

    return new_func


class TensorStack(torch.Tensor):
    def __new__(cls, tensor, *, shapes=None):
        return super().__new__(cls, tensor)

    def __init__(self, tensor, *, shapes=None):
        super(TensorStack, self).__init__()
        self._shapes = shapes
        self._init()

    def _init(self):
        self._unique_shape, self._common_shape = self._get_common_shape(self._shapes)
        self._get_offsets_()

    @property
    def _trailing_dims(self):
        dims = []
        for i in reversed(self._common_shape):
            if i >= 0:
                dims.append(i)
            else:
                break
        return torch.Size(reversed(dims))

    def _get_offsets_(self):
        offsets = tuple(shape.numel() for shape in self._shapes)
        n = self._trailing_dims.numel()
        offsets = torch.tensor([offset // n for offset in offsets])
        self._offsets = offsets
        self._offsets_cs = torch.cumsum(torch.nn.functional.pad(offsets, [1, 0]), 0)

    def _get_common_shape(self, shapes):
        common_shape = shapes[0]
        if _NestedSize.is_nested(common_shape):
            new_shapes = []
            for shape in shapes:
                _unique_shape, _common_shape = self._get_common_shape(shape)
                new_shapes.append(_common_shape)
            # unique_shape = all(new_unique_shape)
            return self._get_common_shape(new_shapes)

        for _shape in shapes[1:]:
            if len(_shape) != len(common_shape):
                raise RuntimeError
            if _shape != common_shape:
                unique_shape = False
                common_shape = torch.Size(
                    [
                        s if s == s_other else -1
                        for s, s_other in zip(common_shape, _shape)
                    ]
                )
        else:
            unique_shape = all(s >= 0 for s in common_shape)
        return unique_shape, torch.Size([len(shapes), *common_shape])

    @classmethod
    def from_tensors(cls, tensors):
        if not len(tensors):
            raise RuntimeError
        shapes = _NestedSize(
            _NestedSize(tensor.shape)
            if not isinstance(tensor, TensorStack)
            else tensor._shapes
            for tensor in tensors
        )
        return TensorStack(
            torch.cat([tensor.view(-1) for tensor in tensors]), shapes=shapes
        )

    @property
    def shape(self):
        return self._common_shape

    def unstack(self):
        return tuple(
            super(TensorStack, self).__getitem__(slice(idx0, idx1)).view(shape)
            for idx0, idx1, shape in zip(
                self._offsets_cs[:-1], self._offsets_cs[1:], self._shapes
            )
        )

    @property
    def _flat(self):
        # represents the tensor as a flat one
        return super().view(-1)

    @property
    def _compact(self):
        # represents the tensor with a compact structure (rightmost consistent dims-wise)
        return super().view(-1, *self._trailing_dims)

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        if isinstance(shape, _NestedSize):
            out = TensorStack(self.data, shapes=shape)
            return out
        n_trailing = len(self._trailing_dims)
        if (
            self._trailing_dims
            and shape[-n_trailing:] == self._trailing_dims
            and shape[-n_trailing - 1] == -1
        ):
            # eg, (4, 2, -1, 3, 5) -> (2, 2, 2, -1, 3, 5)
            for i in range(-1, -self.ndim - 1, -1):
                if self.shape[i] == -1:
                    break
            first_neg_right_index = i
            shapes = _NestedSize.refine_shapes(
                first_neg_right_index, self._shapes, shape
            )
            out = TensorStack(self.data, shapes=shapes)
            return out
        else:
            return super().view(shape)

    @property
    def ndim(self):
        return len(self.shape)

    def ndimension(self):
        return self.ndim

    def __getitem__(self, index):
        if isinstance(index, (int,)):
            shape = self._shapes[index]
            idx0 = self._offsets_cs[index]
            idx1 = self._offsets_cs[index + 1]
            out = super(TensorStack, self._compact).__getitem__(slice(idx0, idx1))._flat
            if isinstance(shape, torch.Size):
                return torch.Tensor(out).view(shape)
            else:
                out._shapes = shape
                return out
        if isinstance(index, (slice,)):
            index = range(*index.indices(self.shape[0]))
        if isinstance(
            index,
            (
                range,
                list,
            ),
        ):
            index = torch.tensor(index)
        if isinstance(index, torch.Tensor):
            index_view = index.view(-1)
            shape = _NestedSize([self._shapes[idx] for idx in index_view])
            index_view = _populate_index(self._offsets, self._offsets_cs, index_view)
            out = super(TensorStack, self._compact).__getitem__(index_view)._flat
            out._shapes = shape
            if index.ndim > 1:
                out = out.view(*index.shape, *out.shape[1:])
            return out
        else:
            raise NotImplementedError

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

    @_lazy_init
    def _offsets(self):
        ...

    @_lazy_init
    def _offsets_cs(self):
        ...

    @_lazy_init
    def _unique_shape(self):
        ...

    @_lazy_init
    def _common_shape(self):
        ...
