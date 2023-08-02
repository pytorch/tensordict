# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
import operator
from copy import copy
from typing import Sequence

import numpy as np
import torch
from tensordict.tensordict import _broadcast_tensors, _is_number

from tensordict.utils import convert_ellipsis_to_idx
from torch import Tensor


def _get_shape(
    tensor_data,
):
    shape = list(tensor_data[0].shape)
    for t in tensor_data[1:]:
        tshape = t.shape
        for i, (s1, s2) in enumerate(list(zip(shape, tshape))):
            shape[i] = s1 if s1 == s2 else -1
    return shape


def _get_shape_nested(
    tensor_data,
):
    out = []
    for i in range(tensor_data.ndim):
        try:
            s = tensor_data.size(i)
        except Exception:
            s = -1
        out.append(s)
    shape = torch.Size(out)
    return shape


def _elementiwse_broadcast(func):
    func_name = func.__name__

    def new_func(self, other):
        if self._nested:
            if isinstance(other, torch.Tensor) and not other.is_nested:
                shape = torch.broadcast_shapes(other.shape, self._shape_no0)
                if shape != other.shape:
                    other = other.expand(shape)
                if shape != self._shape_no0:
                    self_expand = self.expand(shape).as_nestedtensor()
                else:
                    self_expand = self
                sd = self.stack_dim - self.ndim
                other = other.unbind(sd)
                other = LazyStackedTensors(other, stack_dim=sd).get_nestedtensor()
            else:
                self_expand = self
                # print("op", func_name, "\nt", self.tensors, "\nother", other)
                # print("result", getattr(torch.Tensor, func_name)(self.tensors, other))
            return type(self)(
                getattr(torch.Tensor, func_name)(self_expand.tensors, other),
                stack_dim=self.stack_dim,
            )
        if isinstance(other, (torch.Tensor,)):
            shape = torch.broadcast_shapes(other.shape, self._shape_no0)
            if shape != other.shape:
                other = other.expand(shape)
            if shape != self._shape_no0:
                self_expand = self.expand(shape)
            else:
                self_expand = self
            other = other.unbind(self_expand.stack_dim)
            new_stack_dim = self.stack_dim + len(shape) - self.ndim
        elif isinstance(other, (LazyStackedTensors,)):
            shape = torch.broadcast_shapes(other._shape_no0, self._shape_no0)
            if shape != other._shape_no0:
                other = other.expand(shape)
            if shape != self._shape_no0:
                self_expand = self.expand(shape)
            else:
                self_expand = self
            other = other.unbind(self_expand.stack_dim)
            new_stack_dim = self.stack_dim + len(shape) - self.ndim
        else:
            self_expand = self
            other = (other,) * self.n
            new_stack_dim = self.stack_dim
        return type(self)(
            [
                getattr(torch.Tensor, func_name)(t, _other)
                for t, _other in zip(self_expand.tensors, other)
            ],
            stack_dim=new_stack_dim,
        )

    return new_func


class LazyStackedTensors:
    def __init__(self, tensors, stack_dim=0):
        self.tensors = tensors
        self.stack_dim = stack_dim
        self._nested = isinstance(tensors, torch.Tensor) and tensors.is_nested
        self._shape = self._get_shape()

    @property
    def shape(self):
        return self._shape

    @property
    def _shape_no0(self):
        return torch.Size([s if s >= 0 else 1 for s in self._shape])

    def _get_shape(self):
        tensors = self.tensors
        if self._nested:
            shape = _get_shape_nested(tensors)
            if self.stack_dim < 0:
                self.stack_dim = len(shape) + self.stack_dim
            if self.stack_dim > len(shape) or self.stack_dim < 0:
                raise RuntimeError
            if self.stack_dim != 0:
                n, *shape = list(shape)
                shape.insert(self.stack_dim, n)
        else:
            shape = _get_shape(tensors)
            if self.stack_dim < 0:
                self.stack_dim = len(shape) + self.stack_dim + 1
            if self.stack_dim > len(shape) or self.stack_dim < 0:
                raise RuntimeError
            shape.insert(self.stack_dim, len(tensors))
        return torch.Size(shape)

    def get_nestedtensor(self):
        return torch.nested.nested_tensor(list(self.tensors))

    def as_nestedtensor(self):
        if self._nested:
            return self
        return type(self)(self.get_nestedtensor(), stack_dim=self.stack_dim)

    @classmethod
    def from_nested_tensor(cls, nt, stack_dim=0):
        return cls(nt, stack_dim=stack_dim)

    def __getitem__(self, index):
        split_index = self._split_index(index)
        converted_idx = split_index["index_dict"]
        isinteger = split_index["isinteger"]
        has_bool = split_index["has_bool"]
        is_nd_tensor = split_index["is_nd_tensor"]
        num_single = split_index.get("num_single", 0)
        num_none = split_index.get("num_none", 0)
        num_squash = split_index.get("num_squash", 0)
        if has_bool:
            mask_unbind = split_index["individual_masks"]
            cat_dim = split_index["mask_loc"] - num_single
            out = []
            if mask_unbind[0].ndim == 0:
                # we can return a stack
                for (i, _idx), mask in zip(converted_idx.items(), mask_unbind):
                    if mask.any():
                        if mask.all() and self.tensors[i].ndim == 0:
                            out.append(self.tensors[i])
                        else:
                            out.append(self.tensors[i][_idx])
                            out[-1] = out[-1].squeeze(cat_dim)
                return LazyStackedTensors(out, cat_dim)
            else:
                for (i, _idx) in converted_idx.items():
                    self_idx = (slice(None),) * split_index["mask_loc"] + (i,)
                    out.append(self[self_idx][_idx])
                return torch.cat(out, cat_dim)
        elif is_nd_tensor:
            new_stack_dim = self.stack_dim - num_single + num_none
            return LazyStackedTensors(
                [self[idx] for idx in converted_idx.values()], new_stack_dim
            )
        else:
            if isinteger:
                for (
                    i,
                    _idx,
                ) in (
                    converted_idx.items()
                ):  # for convenience but there's only one element
                    out = self.tensors[i]
                    if _idx is not None and _idx != ():
                        out = out[_idx]
                    return out
            else:
                out = []
                new_stack_dim = self.stack_dim - num_single + num_none - num_squash
                for (i, _idx) in converted_idx.items():
                    out.append(self.tensors[i][_idx])
                out = LazyStackedTensors(out, new_stack_dim)
                return out

    def _split_index(self, index):
        """Given a tuple index, split it in as many indices as the number of tensordicts.

        Returns:
            a dictionary with {index-of-td: index-within-td}
            the number of single dim indices until stack dim
            a boolean indicating if the index along the stack dim is an integer
        """
        if not isinstance(index, tuple):
            index = (index,)
        index = convert_ellipsis_to_idx(index, self.shape)
        index = _broadcast_tensors(index)
        out = []
        num_single = 0
        num_none = 0
        isinteger = False
        is_nd_tensor = False
        cursor = 0  # the dimension cursor
        selected_td_idx = range(self.n)
        has_bool = False
        num_squash = 0
        for i, idx in enumerate(index):  # noqa: B007
            cursor_incr = 1
            if idx is None:
                out.append(None)
                num_none += cursor <= self.stack_dim
                continue
            if cursor == self.stack_dim:
                # we need to check which tds need to be indexed
                if isinstance(idx, slice) or _is_number(idx):
                    selected_td_idx = range(self.n)[idx]
                    if not isinstance(selected_td_idx, range):
                        isinteger = True
                        selected_td_idx = [selected_td_idx]
                elif isinstance(idx, (list, range)):
                    selected_td_idx = idx
                elif isinstance(idx, (torch.Tensor, np.ndarray)):
                    if idx.dtype in (np.dtype("bool"), torch.bool):
                        # we mark that we need to dispatch the indices across stack idx
                        has_bool = True
                        # split mask along dim
                        individual_masks = idx = idx.unbind(0)
                        selected_td_idx = range(self.n)
                        out.append(idx)
                        split_dim = self.stack_dim - num_single
                        mask_loc = i
                    else:
                        if isinstance(idx, np.ndarray):
                            idx = torch.tensor(idx)
                        is_nd_tensor = True
                        selected_td_idx = range(len(idx))
                        out.append(idx.unbind(0))
                else:
                    raise TypeError(f"Invalid index type: {type(idx)}.")
            else:
                if _is_number(idx) and cursor < self.stack_dim:
                    num_single += 1
                if isinstance(
                    idx,
                    (
                        int,
                        slice,
                        list,
                        range,
                    ),
                ):
                    out.append(idx)
                elif isinstance(idx, (np.ndarray, torch.Tensor)):
                    if idx.dtype in (np.dtype("bool"), torch.bool):
                        cursor_incr = idx.ndim
                        if cursor < self.stack_dim:
                            num_squash += cursor_incr - 1
                        if (
                            cursor < self.stack_dim
                            and cursor + cursor_incr > self.stack_dim
                        ):
                            # we mark that we need to dispatch the indices across stack idx
                            has_bool = True
                            # split mask along dim
                            # relative_stack_dim = self.stack_dim - cursor - cursor_incr
                            individual_masks = idx = idx.unbind(0)
                            selected_td_idx = range(self.shape[i])
                            split_dim = cursor - num_single
                            mask_loc = i
                    out.append(idx)
                else:
                    raise TypeError(f"Invalid index type: {type(idx)}.")
            cursor += cursor_incr
        if has_bool:
            out = tuple(
                tuple(idx if not isinstance(idx, tuple) else idx[i] for idx in out)
                for i in selected_td_idx
            )
            return {
                "index_dict": {i: out[i] for i in selected_td_idx},
                "num_single": num_single,
                "isinteger": isinteger,
                "has_bool": has_bool,
                "individual_masks": individual_masks,
                "split_dim": split_dim,
                "mask_loc": mask_loc,
                "is_nd_tensor": is_nd_tensor,
                "num_none": num_none,
                "num_squash": num_squash,
            }
        elif is_nd_tensor:

            def isindexable(idx):
                if isinstance(idx, (torch.Tensor, np.ndarray)):
                    if idx.dtype in (torch.bool, np.dtype("bool")):
                        return False
                    return True
                if isinstance(idx, (tuple, list, range)):
                    return True
                return False

            out = tuple(
                tuple(idx if not isindexable(idx) else idx[i] for idx in out)
                for i in selected_td_idx
            )
            return {
                "index_dict": dict(enumerate(out)),
                "num_single": num_single,
                "isinteger": isinteger,
                "has_bool": has_bool,
                "is_nd_tensor": is_nd_tensor,
                "num_none": num_none,
                "num_squash": num_squash,
            }
        return {
            "index_dict": {i: tuple(out) for i in selected_td_idx},
            "num_single": num_single,
            "isinteger": isinteger,
            "has_bool": has_bool,
            "is_nd_tensor": is_nd_tensor,
            "num_none": num_none,
            "num_squash": num_squash,
        }

    @_elementiwse_broadcast
    def __add__(self, other):
        ...

    @_elementiwse_broadcast
    def __sub__(self, other):
        ...

    @_elementiwse_broadcast
    def __truediv__(self, other):
        ...

    @_elementiwse_broadcast
    def __div__(self, other):
        ...

    @_elementiwse_broadcast
    def __mul__(self, other):
        ...

    @_elementiwse_broadcast
    def __eq__(self, other):
        ...

    @_elementiwse_broadcast
    def __ne__(self, other):
        ...

    @_elementiwse_broadcast
    def __mod__(self, other):
        ...

    @property
    def n(self):
        return self.shape[self.stack_dim]

    def __len__(self):
        return self._shape[0]

    @property
    def ndim(self):
        return len(self.shape)

    def ndimension(self):
        return self.ndim

    def expand(self, *shape: int):
        dims = self.ndim

        if len(shape) == 1 and isinstance(shape[0], Sequence):
            shape = tuple(shape[0])

        # new shape dim check
        if len(shape) < len(self.shape):
            raise RuntimeError(
                "the number of sizes provided ({shape_dim}) must be greater or equal to the number of "
                "dimensions in the tensor ({t_dim})".format(
                    shape_dim=len(shape), t_dim=dims
                )
            )

        # new shape compatability check
        for old_dim, new_dim in zip(self.shape, shape[-dims:]):
            if old_dim not in (1, -1) and new_dim != old_dim:
                raise RuntimeError(
                    "Incompatible expanded shape: The expanded shape length at non-singleton dimension should be same "
                    "as the original length. target_shape = {new_shape}, existing_shape = {old_shape}".format(
                        new_shape=shape, old_shape=self.shape
                    )
                )

        stack_dim = len(shape) + self.stack_dim - self.ndimension()
        new_shape_t = [v for i, v in enumerate(shape) if i != stack_dim]
        tensors = [
            t.expand(*torch.broadcast_shapes(new_shape_t, t.shape))
            for t in self.tensors
        ]
        return type(self)(tensors, stack_dim=stack_dim)

    def unbind(self, dim: int):
        if dim < 0:
            dim = self.shape + dim
        if dim < 0 or dim >= self.ndim:
            raise ValueError(
                f"Cannot unbind along dimension {dim} with shape {self.shape}."
            )
        if dim == self.stack_dim:
            if self._nested:
                return self.tensors.unbind(0)
            return tuple(self.tensors)
        else:
            # return a stack of unbound tensordicts
            out = []
            new_dim = dim if dim < self.stack_dim else dim - 1
            new_stack_dim = (
                self.stack_dim if dim > self.stack_dim else self.stack_dim - 1
            )
            for t in self.tensors:
                out.append(t.unbind(new_dim))
            return tuple(LazyStackedTensors(vals, new_stack_dim) for vals in zip(*out))

    def all(self, dim: int = None):
        if dim is not None and (dim >= self.ndim or dim < -self.ndim):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            if dim < 0:
                dim = self.ndim + dim
            if dim > self.stack_dim:
                dim = dim - 1
                new_stack_dim = self.stack_dim
            elif dim == self.stack_dim:
                if len(self.tensors) == 1:
                    return self.tensors[0].bool()

                val = functools.reduce(operator.and_, [t.bool() for t in self.tensors])
                return val
            else:
                new_stack_dim = self.stack_dim - 1

            out = LazyStackedTensors(
                [t.all(dim) for t in self.tensors], stack_dim=new_stack_dim
            )
            return out
        return all(value.all() for value in self.tensors)

    def any(self, dim: int = None):
        if dim is not None and (dim >= self.ndim or dim < -self.ndim):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            if dim < 0:
                dim = self.ndim + dim
            if dim > self.stack_dim:
                dim = dim - 1
                new_stack_dim = self.stack_dim
            elif dim == self.stack_dim:
                if len(self.tensors) == 1:
                    return self.tensors[0].bool()

                val = functools.reduce(operator.or_, [t.bool() for t in self.tensors])
                return val
            else:
                new_stack_dim = self.stack_dim - 1

            out = LazyStackedTensors(
                [t.any(dim) for t in self.tensors], stack_dim=new_stack_dim
            )
            return out
        return any(value.any() for value in self.tensors)

    def transpose(self, dim0, dim1=None):
        if isinstance(dim0, (list, tuple)) and dim1 is None:
            dim0, dim1 = dim0
        elif isinstance(dim0, (list, tuple)):
            raise ValueError(
                "Expected one of `transpose((dim0, dim1))` or `transpose(dim0, dim1)`."
            )
        if dim0 < 0:
            newdim0 = self.ndim + dim0
        else:
            newdim0 = dim0
        if dim1 < 0:
            newdim1 = self.ndim + dim1
        else:
            newdim1 = dim1
        if newdim0 < 0 or newdim1 < 0 or newdim0 >= self.ndim or newdim1 > self.ndim:
            raise ValueError(
                f"Dimensions {(dim0, dim1)} are incompatible with a tensor of shape {self.shape}."
            )
        if newdim0 == newdim1:
            return self
        if newdim0 == self.stack_dim:
            newdim1 = newdim1 if newdim1 < self.stack_dim else newdim1 - 1
            pdim = [i for i in range(self.ndim - 1) if i != newdim1]
            pdim.insert(newdim0 - 1, newdim1)
            return LazyStackedTensors(
                [t.permute(pdim) for t in self.tensors], stack_dim=newdim1
            )
        elif newdim1 == self.stack_dim:
            newdim0 = newdim0 if newdim0 < self.stack_dim else newdim0 - 1
            pdim = [i for i in range(self.ndim - 1) if i != newdim0]
            pdim.insert(newdim1 - 1, newdim0)
            return LazyStackedTensors(
                [t.permute(pdim) for t in self.tensors], stack_dim=newdim0
            )
        else:
            newdim0 = newdim0 if newdim0 < self.stack_dim else newdim0 - 1
            newdim1 = newdim1 if newdim1 < self.stack_dim else newdim1 - 1
            return LazyStackedTensors(
                [t.transpose(newdim1, newdim0) for t in self.tensors],
                stack_dim=self.stack_dim,
            )

    def permute(self, *permute_dims):
        orig_permute_dims = permute_dims
        if isinstance(permute_dims[0], (tuple, list)):
            if len(permute_dims) == 1:
                permute_dims = permute_dims[0]
            else:
                raise ValueError(
                    f"Got incompatible argument permute_dims: {orig_permute_dims}."
                )
        permute_dims = [p if p >= 0 else self.ndim + p for p in permute_dims]
        if any(p < 0 or p >= self.ndim for p in permute_dims):
            raise ValueError(
                f"Got incompatible argument permute_dims: {orig_permute_dims}."
            )
        if len(permute_dims) != self.ndim:
            raise ValueError(
                f"permute_dims must have the same length as the number of dimensions of the tensor ({self.ndim}): {orig_permute_dims}."
            )
        for i in range(self.ndim):
            if permute_dims[i] == self.stack_dim:
                new_stack_dim = i
                break
        else:
            # unreachable
            raise RuntimeError
        permute_dims = [
            p if p < self.stack_dim else p - 1
            for p in permute_dims
            if p != self.stack_dim
        ]
        return LazyStackedTensors(
            [t.permute(permute_dims) for t in self.tensors],
            stack_dim=new_stack_dim,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.get_nestedtensor()})"
