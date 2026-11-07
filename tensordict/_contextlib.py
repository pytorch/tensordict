# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import warnings

import numpy as np
import torch
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
    separator = kwargs.get("separator", sep)
    if (
        isinstance(separator, str)
        and separator
        and separator != sep
        # unflatten_keys(sep) restores a flat tensordict whose keys hold no
        # sep exactly: keep it there
        and any(not isinstance(key, str) or sep in key for key in out.keys(True))
        and all(
            separator.join(parts).split(separator) == list(parts)
            for parts in (
                (key,) if isinstance(key, str) else key for key in out.keys(True)
            )
        )
    ):
        # unflatten_keys undoes the op exactly: every key splits back into
        # its own parts
        sep = separator
    if not out.is_locked:
        return out.update(self.unflatten_keys(sep), inplace=False)
    else:
        return out.update_(self.unflatten_keys(sep))


LAST_OP_MAPS["flatten_keys"] = _reverse_flatten_keys


def _is_plain_td(td):
    # A genuine circular import: tensordict.base imports this module before
    # tensordict._td defines TensorDict.
    from tensordict._td import TensorDict

    # exact TensorDicts at every depth, so that _get_tuple returns the stored
    # leaf, and no non-tensor leaf
    return type(td) is TensorDict and all(
        isinstance(value, torch.Tensor) or _is_plain_td(value) for value in td.values()
    )


def _is_plain_leaf(tensor):
    if not (
        type(tensor) is torch.Tensor
        and tensor.layout is torch.strided
        and not tensor.is_nested
        and not tensor.is_quantized
        and not tensor.is_inference()
        and not tensor.is_meta
        and not (tensor.requires_grad and torch.is_grad_enabled())
    ):
        return False
    # no expanded or internally overlapping dims
    extent = 1
    for stride, size in sorted(
        (stride, size)
        for stride, size in zip(tensor.stride(), tensor.shape)
        if size > 1
    ):
        if stride < extent:
            return False
        extent += stride * (size - 1)
    return True


def _storage_range(tensor):
    storage = tensor.untyped_storage()
    return storage.data_ptr(), storage.data_ptr() + storage.nbytes()


def _unsplit_keys(self, out, separator):
    """Returns the leaves of ``self`` under the keys of ``out`` they were split from.

    Returns ``None`` unless splitting the keys of ``out`` at ``separator`` gives
    the keys of ``self`` exactly, one for one, and ``out.update_`` takes every
    leaf without raising, skipping or warning.
    """
    if not (_is_plain_td(self) and _is_plain_td(out)):
        return None
    keys = {}
    for key in out.keys(True, True):
        key = (key,) if isinstance(key, str) else key
        # unflatten_keys splits the keys of the root only
        keys[(*key[0].split(separator), *key[1:])] = key
    self_keys = {
        (key,) if isinstance(key, str) else key for key in self.keys(True, True)
    }
    if len(keys) != len(self_keys) or set(keys) != self_keys:
        # the op made two keys of out one, or the block changed the keys
        return None
    pairs = []
    for split_key, key in keys.items():
        value = self._get_tuple(split_key, None)
        dest = out._get_tuple(key, None)
        # write nothing unless every pair is known to be safe for copy_
        if not (
            _is_plain_leaf(value)
            and _is_plain_leaf(dest)
            and value.shape == dest.shape
            and value.device == dest.device
            and not (value.is_complex() and not dest.is_complex())
        ):
            return None
        pairs.append((key, value, dest))
    # update_ copies one leaf at a time, so a value that shares memory with
    # any entry, or an entry that shares memory with another, could be
    # overwritten before it is read. Compare the byte ranges of the storages:
    # from_numpy, from_dlpack or frombuffer can make a storage that starts
    # inside another. Empty tensors can all have data_ptr 0.
    ranges = sorted(_storage_range(dest) for _, _, dest in pairs if dest.numel())
    if any(start < end for (_, end), (start, _) in zip(ranges, ranges[1:])):
        return None
    source = self.empty()
    for key, value, dest in pairs:
        if value is not dest and value.numel():
            start, end = _storage_range(value)
            if any(start < e and s < end for s, e in ranges):
                return None
        source.set(key, value)
    return source


def _reverse_unflatten_keys(self, args, kwargs, out):
    sep = args[0] if args else KEY_SEPARATOR
    separator = kwargs.get("separator", sep)
    if (
        isinstance(separator, str)
        and separator
        and separator != sep
        and not _changed_in_place(self, out)
        and all(isinstance(key, str) for key in out.keys(True))
        # with no key that holds the separator, the op changed nothing and
        # flatten_keys(sep) restores the keys exactly: keep it there
        and any(separator in key for key in out.keys(True))
    ):
        # flatten_keys undoes the op exactly: no key was nested before it
        sep = separator
    if not out.is_locked:
        return out.update(self.flatten_keys(sep), inplace=False)
    flat = self.flatten_keys(sep)
    try:
        return out.update_(flat)
    except KeyError:
        if not (
            isinstance(separator, str)
            and separator
            and separator != sep
            and not _changed_in_place(self, out)
        ):
            raise
        # #3517: out holds nested keys, which flatten_keys cannot tell from
        # the keys that the op split. Where the block kept the keys that the
        # op made, write each leaf back under its key in out.
        source = _unsplit_keys(self, out, separator)
        if source is None:
            raise
    return out.update_(source)


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
    from tensordict._base.convert import _restore_memo, _restore_module

    if out is not None and not is_compiling():
        return _restore_module(self, args, kwargs, out)
    try:
        with (
            out.unlock_()
            if not is_compiling() and out is not None
            else contextlib.nullcontext()
        ):
            memo = _restore_memo(kwargs)
            return self._to_module(*args, **{**kwargs, "memo": memo}, swap_dest=out)
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
