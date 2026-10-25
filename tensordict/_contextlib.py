# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import warnings

import numpy as np
from tensordict.utils import (
    _parse_to,
    _user_stacklevel,
    is_tensorclass,
    KEY_SEPARATOR,
)
from torch.compiler import is_compiling

# TD cm functions
LAST_OP_MAPS = {}


def _changed_in_place(self, out):
    # With inplace=True, the op changes the tensordict it was called on and
    # returns it: the reversal must undo the op on ``out`` itself.
    return (self._tensordict if is_tensorclass(self) else self) is out


def _warn_kept_in_place(op, result):
    """Warns that a with-block around ``op(..., inplace=True)`` keeps ``result``, until 0.17 (gh-2700)."""
    if is_compiling():
        return
    warnings.warn(
        f"A with-block around {op}(..., inplace=True) leaves the tensordict "
        f"with {result} after the block. Starting with TensorDict 0.17, the "
        "default will change to restoring it on exit, as without "
        f"inplace=True. To keep {result}, call {op}(..., inplace=True) "
        "outside a with-block.",
        FutureWarning,
        stacklevel=_user_stacklevel(),
    )


def _reverse_lock(self, args, kwargs, out):
    return self.unlock_()


LAST_OP_MAPS["lock_"] = _reverse_lock


def _reverse_unlock(self, args, kwargs, out):
    return self.lock_()


LAST_OP_MAPS["unlock_"] = _reverse_unlock


def _reverse_transpose(self, args, kwargs, out):
    dim0, dim1 = (*args, *kwargs.values())
    if not out.is_locked:
        return out.update(self.transpose(dim0, dim1), inplace=False)
    else:
        return out.update_(self.transpose(dim0, dim1))


LAST_OP_MAPS["transpose"] = _reverse_transpose
LAST_OP_MAPS["swapaxes"] = _reverse_transpose
LAST_OP_MAPS["swapdims"] = _reverse_transpose


def _reverse_flatten_keys(self, args, kwargs, out):
    sep = args[0] if args else KEY_SEPARATOR
    if _changed_in_place(self, out):
        separator = kwargs.get("separator", sep)
        # flatten_keys(inplace=True) records whether unflatten_keys undoes it
        exact, out._inplace_flatten_keys_exact = out._inplace_flatten_keys_exact, None
        if separator != sep:
            # the reversal below reads a positional separator only, and
            # leaves these keys flat
            if any(separator in key for key in out.keys()):
                _warn_kept_in_place("flatten_keys", "its keys flat")
            return out
        if exact:
            return out.unflatten_keys(sep, inplace=True)
        # A key held the separator before the block, so unflatten_keys
        # cannot tell it from a flattened nested key: keep main's result.
    if not out.is_locked:
        return out.update(self.unflatten_keys(sep), inplace=False)
    else:
        return out.update_(self.unflatten_keys(sep))


LAST_OP_MAPS["flatten_keys"] = _reverse_flatten_keys


def _reverse_unflatten_keys(self, args, kwargs, out):
    sep = args[0] if args else KEY_SEPARATOR
    if not out.is_locked:
        return out.update(self.flatten_keys(sep), inplace=False)
    else:
        return out.update_(self.flatten_keys(sep))


LAST_OP_MAPS["unflatten_keys"] = _reverse_unflatten_keys


def _reverse_flatten(self, args, kwargs, out):
    if len(args) == 2:
        dim0, dim1 = args
    elif len(args) == 1:
        dim0 = args[0]
        dim1 = kwargs.get("end_dim", -1)
    else:
        dim0 = kwargs.get("start_dim", 0)
        dim1 = kwargs.get("end_dim", -1)
    # A 0-dim flatten unsqueezes to shape [1] instead of calling unflatten.
    # An empty size would make unflatten raise, so squeeze that dim back.
    if not out.ndim and dim0 in (None, 0) and dim1 in (None, -1, 0):
        self = self.squeeze(0)
        if not out.is_locked:
            return out.update(self, inplace=False)
        return out.update_(self)
    if dim0 is None:
        dim0 = 0
    if dim1 is None:
        dim1 = -1
    if kwargs.get("inplace") and _changed_in_place(self, out):
        # flatten(inplace=True) records the sizes that it merged
        record, out._inplace_flatten_sizes = out._inplace_flatten_sizes, None
        if record is not None:
            start_dim, sizes = record
            ndim = out.ndim
            d0 = dim0 + ndim if dim0 < 0 else dim0
            d1 = dim1 + ndim if dim1 < 0 else dim1
            if -ndim <= d0 < ndim and len(out.shape[d0 : d1 + 1]) == 1:
                # The reversal below unflattens d0 into the one size that it
                # has, and keeps the flat shape.
                if len(sizes) > 1:
                    _warn_kept_in_place("flatten", "its flattened shape")
                return out
            return out.unflatten(start_dim, sizes, inplace=True)
    if dim1 < 0:
        dim1 = out.ndim + dim1
    if dim0 < 0:
        dim0 = out.ndim + dim0

    if not out.is_locked:
        return out.update(
            self.unflatten(dim0, out.shape[dim0 : dim1 + 1]), inplace=False
        )
    else:
        return out.update_(self.unflatten(dim0, out.shape[dim0 : dim1 + 1]))


LAST_OP_MAPS["flatten"] = _reverse_flatten


def _reverse_unflatten(self, args, kwargs, out):
    if args:
        dim0 = args[0]
        if len(args) > 1:
            unflattened_size = args[1]
        else:
            unflattened_size = kwargs.get("unflattened_size")
    else:
        dim0 = kwargs.get("dim")
        unflattened_size = kwargs.get("unflattened_size")
    if _changed_in_place(self, out):
        if dim0 < 0:
            dim0 = out.ndim - len(unflattened_size) + 1 + dim0
        dim1 = dim0 + len(unflattened_size) - 1
        return out.flatten(dim0, dim1, inplace=True)
    if dim0 < 0:
        dim0 = out.ndim + dim0
    dim1 = dim0 + len(unflattened_size) - 1
    if not out.is_locked:
        unflattened = self.flatten(dim0, dim1)
        return out.update(unflattened, inplace=False)
    else:
        unflattened = self.flatten(dim0, dim1)
        return out.update_(unflattened)


LAST_OP_MAPS["unflatten"] = _reverse_unflatten


def _reverse_permute(self, args, kwargs, out):
    from tensordict.utils import _get_shape_from_args

    dims_list = _get_shape_from_args(*args, kwarg_name="dims", **kwargs)
    dims_list = [dim if dim >= 0 else self.ndim + dim for dim in dims_list]
    # inverse map
    inv_dims_list = np.argsort(dims_list)
    if not out.is_locked:
        return out.update(self.permute(inv_dims_list), inplace=False)
    else:
        return out.update_(self.permute(inv_dims_list))


LAST_OP_MAPS["permute"] = _reverse_permute


def _reverse_movedim(self, args, kwargs, out):
    # Reverse of movedim(source, destination) is movedim(destination, source)
    if len(args) >= 2:
        source, destination = args[0], args[1]
    else:
        source = args[0] if args else kwargs.get("source")
        destination = kwargs.get("destination")
    if not out.is_locked:
        return out.update(self.movedim(destination, source), inplace=False)
    else:
        return out.update_(self.movedim(destination, source))


LAST_OP_MAPS["movedim"] = _reverse_movedim
LAST_OP_MAPS["moveaxis"] = _reverse_movedim


def _reverse_flip(self, args, kwargs, out):
    # Flip is its own inverse
    dims = args[0] if args else kwargs.get("dims")
    if not out.is_locked:
        return out.update(self.flip(dims), inplace=False)
    else:
        return out.update_(self.flip(dims))


LAST_OP_MAPS["flip"] = _reverse_flip


def _reverse_fliplr(self, args, kwargs, out):
    # fliplr is its own inverse
    if not out.is_locked:
        return out.update(self.fliplr(), inplace=False)
    else:
        return out.update_(self.fliplr())


LAST_OP_MAPS["fliplr"] = _reverse_fliplr


def _reverse_flipud(self, args, kwargs, out):
    # flipud is its own inverse
    if not out.is_locked:
        return out.update(self.flipud(), inplace=False)
    else:
        return out.update_(self.flipud())


LAST_OP_MAPS["flipud"] = _reverse_flipud


def _reverse_roll(self, args, kwargs, out):
    # Reverse of roll(shifts, dims) is roll(-shifts, dims)
    shifts = args[0] if args else kwargs.get("shifts")
    dims = args[1] if len(args) > 1 else kwargs.get("dims")
    if isinstance(shifts, int):
        neg_shifts = -shifts
    else:
        neg_shifts = tuple(-s for s in shifts)
    if not out.is_locked:
        return out.update(self.roll(neg_shifts, dims), inplace=False)
    else:
        return out.update_(self.roll(neg_shifts, dims))


LAST_OP_MAPS["roll"] = _reverse_roll


def _reverse_rot90(self, args, kwargs, out):
    # Reverse of rot90(k, dims) is rot90(-k, dims) or rot90(4-k, dims)
    k = args[0] if args else kwargs.get("k", 1)
    dims = args[1] if len(args) > 1 else kwargs.get("dims", (0, 1))
    if not out.is_locked:
        return out.update(self.rot90(-k, dims), inplace=False)
    else:
        return out.update_(self.rot90(-k, dims))


LAST_OP_MAPS["rot90"] = _reverse_rot90


def _reverse_atleast_1d(self, args, kwargs, out):
    # ``self`` is the expanded result and ``out`` is the original.
    # atleast_* prepends size-1 dims; squeeze only those before write-back.
    out_ndim = out.ndim
    self_ndim = self.ndim
    if self_ndim > out_ndim:
        for _ in range(self_ndim - out_ndim):
            self = self.squeeze(0)
    if not out.is_locked:
        return out.update(self, inplace=False)
    else:
        return out.update_(self)


LAST_OP_MAPS["atleast_1d"] = _reverse_atleast_1d
LAST_OP_MAPS["atleast_2d"] = _reverse_atleast_1d
LAST_OP_MAPS["atleast_3d"] = _reverse_atleast_1d


def _reverse_view(self, args, kwargs, out):
    if not out.is_locked:
        return out.update(self.view(out.shape), inplace=False)
    else:
        return out.update_(self.view(out.shape))


LAST_OP_MAPS["view"] = _reverse_view


def _reverse_unsqueeze(self, args, kwargs, out):
    if args:
        (dim,) = args
    elif kwargs:
        dim = kwargs["dim"]
    else:
        raise RuntimeError(
            "Cannot use td.unsqueeze() as a decorator if the dimension is implicit."
        )
    if not out.is_locked:
        return out.update(self.squeeze(dim), inplace=False)
    else:
        return out.update_(self.squeeze(dim))


LAST_OP_MAPS["unsqueeze"] = _reverse_unsqueeze


def _reverse_squeeze(self, args, kwargs, out):
    if args:
        (dim,) = args
    elif kwargs:
        dim = kwargs["dim"]
    else:
        raise RuntimeError(
            "Cannot use td.squeeze() as a decorator if the dimension is implicit."
        )
    if self.batch_dims == out.batch_dims:
        # The dim was not of size 1: squeeze returned its input, so the block
        # already wrote into out.
        return out
    if not out.is_locked:
        return out.update(self.unsqueeze(dim), inplace=False)
    else:
        return out.update_(self.unsqueeze(dim))


LAST_OP_MAPS["squeeze"] = _reverse_squeeze


def _reverse_to_module(self, args, kwargs, out):
    if out is not None and not is_compiling():
        from tensordict._base.convert import _restore_module

        return _restore_module(self, args, kwargs, out)
    try:
        with (
            out.unlock_()
            if not is_compiling() and out is not None
            else contextlib.nullcontext()
        ):
            return self.to_module(*args, **kwargs, swap_dest=out)
    except AttributeError:
        # This is a bit unsafe but we assume that out won't have an unlock_() if it's not a TD
        raise RuntimeError(
            "to_module cannot be used as a decorator when return_swap=False."
        )


LAST_OP_MAPS["to_module"] = _reverse_to_module


def _reverse_to(self, args, kwargs, out):
    """Reverse the to() operation by restoring the original device and dtype.

    Writes the tensors of the tensordict returned by to() (self) back into the
    original tensordict (out), on the original device of each tensor. If to() cast
    the dtypes, each tensor gets its original dtype back too.
    """
    if out is None:
        return self
    from tensordict.base import _is_tensor_collection, _warn_read_only
    from tensordict.tensorclass import TensorAttrs

    if getattr(out, "_masking_on_dense_copy", False):
        from tensordict.utils import is_tensorclass

        # to() on a tensorclass wraps the result of to() on its tensordict
        result = self._tensordict if is_tensorclass(self) else self
        if result is out or type(result) is type(out):
            # to() returned ``out`` itself or, for a file, a copy on another
            # device that reads and writes the same file: the block has
            # written to it already. This assumes that no other class with
            # _masking_on_dense_copy returns a copy of its own type from to().
            return self
        if out._storage_is_read_only():
            _warn_read_only("to")
            return self
        # a tensorclass forwards the attribute to its tensordict, but is
        # written back by apply() below
        if out._to_writes_back_in_place and not is_tensorclass(out):
            out._write_back_to(result)
            return self

    # to() casts the dtypes if it gets a dtype, a tensor, other= with a single dtype
    # or an attrs tensordict that records dtypes. A device-only to() leaves the
    # dtypes as they are.
    if args and _is_tensor_collection(type(args[0])):
        restore_dtype = any(
            attrs.tgt_dtype is not None
            for attrs in args[0].values(
                True, True, is_leaf=lambda cls: issubclass(cls, TensorAttrs)
            )
        )
    else:
        # to() takes non_blocking out before it parses the other arguments
        kwargs = {key: val for key, val in kwargs.items() if key != "non_blocking"}
        restore_dtype = _parse_to(*args, **kwargs)[1] is not None
    # Restore each tensor to its original device/dtype from the input tensordict
    return self.apply(
        lambda x, y: (
            (x.to(device=y.device, dtype=y.dtype) if restore_dtype else x.to(y.device))
            if y is not None
            else x
        ),
        out,
        default=None,
        out=out,
    )


LAST_OP_MAPS["to"] = _reverse_to
