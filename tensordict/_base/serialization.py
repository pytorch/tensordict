# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""TensorDict's own storage formats: memory-mapping, saving, loading, consolidation and state dicts of :class:`~tensordict.TensorDictBase`.

The methods live on a mixin that ``tensordict.base`` imports before it
defines ``TensorDictBase``, so this module imports the helpers it needs from
``tensordict.base`` without needing the class itself. A method that uses
``TensorDictBase`` at run time imports it locally. The comment above the
mixin imports in ``tensordict/base.py`` gives the other rules.
"""

from __future__ import annotations

import collections
import concurrent.futures
import json
import os.path
import shutil
import sys
import tempfile
import warnings
from concurrent.futures import wait
from copy import copy
from pathlib import Path
from typing import Any, OrderedDict, TYPE_CHECKING

import torch
from tensordict._archive import (
    _ArchivePath,
    _make_archive_like,
    _save_as_archive,
    is_memmap_archive,
    TENSORDICT_ARCHIVE_SUFFIX,
)
from tensordict._deprecation import deprecated
from tensordict._nestedkey import NestedKey
from tensordict._td_memmap import _class_id
from tensordict.base import (
    _is_tensor_collection,
    _load_metadata,
    _NESTED_TENSORS_AS_LISTS_NONTENSOR,
    _raise_on_exclusive_keys,
    _sync_cuda_transfer,
    _unflatten_state_dict,
    is_compiling,
    is_tensor_collection,
    NO_DEFAULT,
    Self,
)
from tensordict.memmap import _raise_if_jagged, MemoryMappedTensor
from tensordict.utils import (
    _DTYPE_TO_STR_DTYPE,
    _encode_key_for_filesystem,
    _get_robust_key_setting_with_warning,
    _get_shared_executor,
    _is_non_tensor,
    _is_safe_legacy_key,
    _is_tensorclass,
    _is_unbatched,
    _prefix_last_key,
    _rebuild_njt_from_njt,
    _unravel_key_to_tuple,
    _user_stacklevel,
    _zip_strict,
    TensorDictFuture,
    unravel_key,
)
from torch._utils import _get_available_device_type, _get_device_module

if TYPE_CHECKING:
    from tensordict.base import TensorDictBase


def _is_saved_tensorclass(prefix: Path) -> bool:
    """Returns True if the ``meta.json`` in ``prefix`` is that of a tensorclass."""
    import tensordict

    if not (prefix / "meta.json").exists():
        return False
    type_name = _load_metadata(prefix).get("_type")
    return any(
        str(cls) == type_name and _is_tensorclass(cls)
        for cls in tensordict.base._ACCEPTED_CLASSES
    )


# Windows cannot rename a directory that holds open (memory-mapped) files.
_IS_WINDOWS = sys.platform == "win32"


def _memmap_atomically(prefix: Path | None, write, *, existsok: bool):
    """Calls ``write`` on a new sibling directory of ``prefix``, then moves it to ``prefix``.

    ``write(path)`` writes the tensordict to ``path`` and returns the result.
    An existing directory at ``prefix`` is replaced only once the new one is
    complete, and the result is rebased onto the files at ``prefix``.
    """
    if _IS_WINDOWS:
        raise NotImplementedError("atomic=True is not supported on Windows.")
    if prefix is None:
        raise ValueError("atomic=True needs a prefix to write the directory to.")
    prefix = Path(prefix)
    # Follow a symlink, so that the directory it points to is replaced.
    target = prefix.resolve()
    if target.exists():
        if not target.is_dir():
            raise NotADirectoryError(f"{prefix} exists and is not a directory.")
        if any(target.iterdir()):
            if not existsok:
                raise RuntimeError(f"{prefix} is not empty and existsok=False.")
            if not (target / "meta.json").exists():
                # Do not delete a directory that is not a saved tensordict.
                raise RuntimeError(
                    f"Refusing to replace {prefix} with atomic=True: it is not "
                    f"empty and holds no saved tensordict (no meta.json)."
                )
    target.parent.mkdir(parents=True, exist_ok=True)
    # A sibling is on the same file system, so a rename moves it into place.
    work = Path(tempfile.mkdtemp(prefix=".tdtmp-", dir=target.parent))
    new, old = work / "new", work / "old"
    try:
        new.mkdir()
        result = write(new)
        if target.exists():
            shutil.copymode(target, new)
            os.rename(target, old)
            try:
                os.rename(new, target)
            except BaseException:
                os.rename(old, target)
                raise
        else:
            os.rename(new, target)
    finally:
        # Keep the old directory if it could not be moved back.
        if target.exists() or not old.exists():
            shutil.rmtree(work, ignore_errors=True)
    _rebase_memmap_paths(result, new, prefix)
    return result


def _rebase_memmap_paths(data, old: Path, new: Path) -> None:
    """Points the files and directories of ``data`` under ``old`` to the same paths under ``new``."""
    from tensordict._lazy import LazyStackedTensorDict
    from tensordict._td import TensorDict
    from tensordict.base import _unwrap
    from tensordict.tensorclass import NonTensorDataBase, NonTensorStack

    def rebase(path):
        if path is not None and Path(path).is_relative_to(old):
            return new / Path(path).relative_to(old)
        return path

    nodes = [data]
    while nodes:
        node = nodes.pop()
        if isinstance(node, MemoryMappedTensor):
            filename = node._filename
            if filename is not None and Path(filename).is_relative_to(old):
                node._filename = str(new.absolute() / Path(filename).relative_to(old))
            continue
        if not is_tensor_collection(node):
            continue
        # Non-tensor leaves write their meta.json back to these paths.
        if isinstance(node, NonTensorDataBase) and node._metadata:
            if "memmap_prefix" in node._metadata:
                node._metadata["memmap_prefix"] = rebase(
                    node._metadata["memmap_prefix"]
                )
        elif isinstance(node, NonTensorStack) and "_path_to_memmap" in node.__dict__:
            node.__dict__["_path_to_memmap"] = rebase(node.__dict__["_path_to_memmap"])
        node = _unwrap(node)
        if isinstance(node, LazyStackedTensorDict):
            nodes.extend(node.tensordicts)
        elif isinstance(node, TensorDict):
            node._memmap_prefix = rebase(node._memmap_prefix)
            nodes.extend(node._tensordict.values())


def _memmap_like_stand_in(key: NestedKey, x: torch.Tensor) -> torch.Tensor:
    """Returns a leaf that ``memmap_like`` can pass to ``_memmap_(like=True)`` without copying values."""
    _raise_if_jagged(x, key)
    if x.is_nested:
        # A nested tensor has no ``shape`` to expand to. ``_memmap_(like=True)``
        # reads only its nested sizes and dtype, so the leaf itself works as a
        # stand-in. Dropping the MemoryMappedTensor subclass makes
        # ``MemoryMappedTensor.from_tensor`` allocate a new file instead of
        # reusing an existing memory-mapped one.
        return x.as_subclass(torch.Tensor)
    return torch.empty((), device=x.device, dtype=x.dtype).expand(x.shape)


class _Serialization:
    """TensorDict's own storage formats: memory-mapping, saving, loading, consolidation and state dicts."""

    # Serialization functionality
    def state_dict(
        self,
        destination=None,
        prefix="",
        keep_vars=False,
        flatten=True,
    ) -> OrderedDict[str, Any]:
        """Returns the leaves of the tensordict in an ``OrderedDict``, as :meth:`torch.nn.Module.state_dict` does.

        The state dict is flat by default: the key of a nested leaf joins its
        keys with ``"."`` (``"n.b"`` for ``("n", "b")``), following
        :meth:`torch.nn.Module.state_dict`. With ``flatten=False``, each
        nested tensordict becomes a nested ``OrderedDict``. Read it back with
        :meth:`~.load_state_dict`.

        The batch size and device of each level are stored in a
        ``_metadata`` attribute of the result, as in
        :meth:`torch.nn.Module.state_dict`. In a flat state dict, they are
        stored under the ``"."``-joined key of the level (``""`` for the root,
        ``"n"`` for the nested tensordict ``"n"``). In a nested state dict,
        each nested ``OrderedDict`` has its own ``_metadata``.

        Args:
            destination (OrderedDict, optional): an ``OrderedDict`` to write the
                entries into, and to return. A plain ``dict`` raises an
                ``AttributeError``. Defaults to ``None``, which creates a new
                ``OrderedDict``.
            prefix (str, optional): a string prepended to every key. Defaults
                to ``""``.
            keep_vars (bool, optional): if ``False``, the tensors are detached
                from autograd. If ``True``, the tensors of the tensordict are
                returned unchanged. Defaults to ``False``.
            flatten (bool, optional): if ``True``, nested keys are joined with
                ``"."``. If ``False``, nested tensordicts become nested
                ``OrderedDict`` instances. Defaults to ``True``.

        Returns:
            ``destination``, or a new ``OrderedDict``.

        Examples:
            >>> from tensordict import TensorDict
            >>> data = TensorDict({"1": 1, "2": 2, "3": {"3": 3}}, [])
            >>> sd = data.state_dict()
            >>> print(dict(sd))
            {'1': tensor(1), '2': tensor(2), '3.3': tensor(3)}
            >>> print(dict(sd._metadata))
            {'': {'batch_size': torch.Size([]), 'device': None}, '3': {'batch_size': torch.Size([]), 'device': None}}

        """
        if self._has_exclusive_keys:
            _raise_on_exclusive_keys("state_dict")
        if destination is None:
            destination = collections.OrderedDict()
            destination._metadata = collections.OrderedDict()
        elif not hasattr(destination, "_metadata"):
            destination._metadata = collections.OrderedDict()

        metadata_key = prefix[:-1] if prefix.endswith(".") else prefix
        destination._metadata[metadata_key] = {
            "batch_size": self.batch_size,
            "device": self.device,
        }

        for key, item in self.items():
            if not _is_tensor_collection(type(item)):
                if not keep_vars:
                    destination[prefix + key] = item.detach()
                else:
                    destination[prefix + key] = item
            elif flatten:
                item.state_dict(
                    destination=destination,
                    prefix=prefix + key + ".",
                    keep_vars=keep_vars,
                    flatten=True,
                )
            else:
                destination[prefix + key] = item.state_dict(
                    keep_vars=keep_vars, flatten=False
                )

        return destination

    def load_state_dict(
        self,
        state_dict: OrderedDict[str, Any],
        strict=True,
        assign=False,
        from_flatten=None,
    ) -> Self:
        """Loads a state dict, as :meth:`~.state_dict` returns it, into the tensordict, and returns ``self``.

        Reads the flat format (the default of :meth:`~.state_dict`), the
        nested format (``flatten=False``), and the legacy format, with
        ``"__batch_size"`` and ``"__device"`` keys. The batch size of the
        tensordict is set to the one stored in the state dict, if any.

        Args:
            state_dict (OrderedDict): the state dict to load.
            strict (bool, optional): if ``True``, a state dict whose keys are
                not those of the tensordict raises a ``RuntimeError`` (empty
                nested tensordicts excepted). If ``False``, the keys of the
                state dict are written and the other entries of the
                tensordict are kept. Defaults to ``True``.
            assign (bool, optional): if ``False``, the values are copied in
                place into the existing tensors, which keep their storage,
                dtype and device. If ``True``, the tensors of the state dict
                replace them. Defaults to ``False``.
            from_flatten (bool, optional): if ``True``, the state dict is read
                as flat, with ``"."``-joined keys. If ``False``, as nested.
                Defaults to ``None``, which reads it as flat if it has a
                ``_metadata`` attribute and its keys differ from those of the
                tensordict.

        Returns:
            ``self``.

        Raises:
            RuntimeError: if ``strict=True`` and the keys do not match, or if
                the state dict was saved on another device than that of the
                tensordict.

        Examples:
            >>> from tensordict import TensorDict
            >>> data = TensorDict({"1": 1, "2": 2, "3": {"3": 3}}, [])
            >>> data_zeroed = TensorDict({"1": 0, "2": 0, "3": {"3": 0}}, [])
            >>> sd = data.state_dict()
            >>> _ = data_zeroed.load_state_dict(sd)
            >>> print(data_zeroed["3", "3"])
            tensor(3)

        """
        if from_flatten is None:
            _metadata = getattr(state_dict, "_metadata", None)
            if _metadata is not None:
                sd_keys = set(state_dict.keys())
                self_keys = set(self.keys())
                from_flatten = sd_keys != self_keys
            else:
                from_flatten = False

        if from_flatten:
            nested_sd = _unflatten_state_dict(state_dict)
            return self.load_state_dict(
                nested_sd, strict=strict, assign=assign, from_flatten=False
            )

        # Read _metadata before copy (copy may not preserve custom attributes)
        _metadata = getattr(state_dict, "_metadata", None)

        if is_compiling():
            state_dict = type(state_dict)(state_dict)
        else:
            state_dict = copy(state_dict)

        if _metadata is not None:
            local_metadata = _metadata.get("", {})
            batch_size = local_metadata.get("batch_size", self.batch_size)
            device = local_metadata.get("device")
        elif "__batch_size" in state_dict:
            # Legacy format: metadata stored as sentinel keys
            batch_size = state_dict.pop("__batch_size")
            device = state_dict.pop("__device", None)
        else:
            # No metadata (e.g., plain dict from nn.Module pipeline) — keep current
            batch_size = self.batch_size
            device = self.device

        if strict and set(state_dict.keys()) != set(self.keys()):
            set_sd = set(state_dict.keys())
            set_td = set(self.keys())

            def _is_empty_dict(sd, key=None):
                if key is not None:
                    if not isinstance(sd[key], dict):
                        return False
                    return _is_empty_dict(sd[key])
                for key, item in sd.items():
                    # Skip legacy sentinel keys if present in nested dicts
                    if key in ("__batch_size", "__device"):
                        continue
                    if isinstance(item, dict):
                        if not _is_empty_dict(item):
                            return False
                        continue
                    return False
                else:
                    return True

            def check_is_empty(target, key):
                item = target.get(key)
                if not is_tensor_collection(item) or not item.is_empty():
                    return False
                return True

            if not all(check_is_empty(self, key) for key in set_td - set_sd) or not all(
                _is_empty_dict(state_dict, key) for key in set_sd - set_td
            ):
                raise RuntimeError(
                    "Cannot load state-dict because the key sets don't match: got "
                    f"state_dict extra keys \n{set_sd - set_td}\n and tensordict extra keys\n{set_td - set_sd}\n"
                )

        self.batch_size = batch_size
        if device is not None and self.device is not None and device != self.device:
            raise RuntimeError("Loading data from another device is not yet supported.")

        for key, item in state_dict.items():
            if isinstance(item, dict):
                dest = self.get(key, None)
                if dest is None:
                    dest = self.empty()
                dest.load_state_dict(item, assign=assign, strict=strict)
                self.set(
                    key,
                    dest,
                    inplace=not assign,
                )
            else:
                self.set(key, item, inplace=not assign)
        return self

    def is_memmap(self) -> bool:
        """Whether the tensordict is memory-mapped.

        ``True`` for the tensordict of :meth:`~.memmap_`, and for the results
        of :meth:`~.memmap` and :meth:`~.memmap_like` written to a directory or
        to anonymous memory (not to an archive).

        A memory-mapped tensordict is locked (see :ref:`keys-locking`), and
        unlocking it makes ``is_memmap`` return ``False``. A tensordict built
        from memory-mapped tensors, or returned by :meth:`~.load_memmap`, is
        not memory-mapped in this sense: a new entry would not be.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3))
            >>> td.is_memmap(), td.memmap().is_memmap()
            (False, True)
        """
        return self._is_memmap

    # Generic method to get a class metadata
    def _reduce_get_metadata(self):
        return {
            "device": str(self.device) if self.device is not None else None,
            "names": self.names,
            "batch_size": list(self.batch_size),
            "is_locked": self._is_locked,
        }

    # @cache  # noqa: B019
    def _reduce_vals_and_metadata(self, *, dtype=NO_DEFAULT, requires_metadata):
        """Returns a nested dictionary of metadata, a flat Dict[NestedKey, Tensor] containing tensor data and a list of tensor sizes."""
        from tensordict.base import TensorDictBase

        if dtype is NO_DEFAULT:
            dtype = self.dtype
        need_padding = dtype is None
        # If the dtype is not unique (self.dtype is None) then we need the metadata
        # because we need a custom unpickler
        requires_metadata = requires_metadata | need_padding

        if requires_metadata:
            # metadata is nested
            cls = type(self)
            from tensordict._reductions import CLS_MAP

            if cls.__name__ in CLS_MAP:
                cls = cls.__name__
            else:
                pass
            metadata_dict = {
                "cls": cls,
                "non_tensors": {},
                "leaves": {},
                "cls_metadata": self._reduce_get_metadata(),
            }
        else:
            metadata_dict = None

        # flat_key_values is flat
        flat_key_values = {}

        flat_size = []
        start = 0

        def add_single_value(
            value, key, metadata_dict, dtype, shape, flat_size, ragged_idx=None
        ):
            nonlocal start
            n = value.element_size() * value.numel()
            if need_padding:
                pad = n % 8
                if pad != 0:
                    pad = 8 - pad
            else:
                pad = 0
            flat_size.append(sum([n, pad]))
            # Using sum to tell dynamo to use sym_sum
            stop = sum([start, flat_size[-1]])
            if requires_metadata:
                leaf_metadata = [
                    _DTYPE_TO_STR_DTYPE[dtype],
                    list(shape),
                    # _DEVICE2STRDEVICE[device],
                    start,
                    stop,
                    pad,
                ]
                if ragged_idx is not None:
                    leaf_metadata.append(ragged_idx)
                metadata_dict["leaves"][key] = tuple(leaf_metadata)
            start = stop

        def assign(
            key,
            value,
            track_key=(),
            metadata_dict=metadata_dict,
            flat_size=flat_size,
        ):
            total_key = key if isinstance(key, tuple) else (key,)
            total_key = track_key + total_key
            cls = type(value)
            if issubclass(cls, torch.Tensor):
                pass
            # We want to skip NonTensorStacks
            elif _is_non_tensor(cls) and not issubclass(cls, TensorDictBase):
                if requires_metadata:
                    metadata_dict["non_tensors"][key] = (
                        value.data,
                        list(value.batch_size),
                        str(value.device) if value.device is not None else None,
                    )
                return
            elif _is_tensor_collection(cls):
                metadata_dict_key = None
                if requires_metadata:
                    from tensordict._reductions import CLS_MAP

                    if cls.__name__ in CLS_MAP:
                        cls = cls.__name__
                    else:
                        pass
                    metadata_dict_key = metadata_dict[key] = {
                        "cls": cls,
                        "non_tensors": {},
                        "leaves": {},
                        "cls_metadata": value._reduce_get_metadata(),
                    }

                def local_assign(*t):
                    return assign(
                        *t,
                        track_key=total_key,
                        metadata_dict=metadata_dict_key,
                        flat_size=flat_size,
                    )

                # Visit a tensorclass through its tensordict: the tensorclass
                # wrapper would rebuild the (empty) result as an instance and
                # run __post_init__ on it.
                if _is_tensorclass(type(value)):
                    value = value._tensordict
                value._fast_apply(
                    local_assign,
                    named=True,
                    nested_keys=True,
                    call_on_nested=True,
                    is_leaf=_NESTED_TENSORS_AS_LISTS_NONTENSOR,
                )
                return
            # Tensors: DTensor, nested and then regular
            if hasattr(value, "full_tensor"):
                raise NotImplementedError("DTensor is not supported yet")
            if _is_unbatched(value):
                # the leaf would be rebuilt as a plain tensor
                raise NotImplementedError(
                    f"UnbatchedTensor entries cannot be consolidated yet, but "
                    f"{unravel_key(total_key)!r} is one."
                )
            if getattr(value, "is_nested", False):
                if value.layout is torch.jagged:
                    # Get the values
                    values = value._values
                    shape = [v if isinstance(v, int) else -1 for v in values.shape]
                    # Get the offsets
                    offsets = value._offsets
                    # Get the lengths
                    lengths = value._lengths

                    # Now we're saving the two tensors
                    # We will rely on the fact that the writing order is preserved in python dict
                    # (since python 3.7). Later, we will read the NJT then the NJT offset in that order
                    # to do the allocation.
                    flat_key_values[_prefix_last_key(total_key, "<NJT>")] = value
                    flat_size.append(0)
                    flat_key_values[_prefix_last_key(total_key, "<NJT_VALUES>")] = (
                        values
                    )
                    add_single_value(
                        values,
                        _prefix_last_key(key, "<NJT_VALUES>"),
                        metadata_dict,
                        values.dtype,
                        shape,
                        flat_size,
                    )
                    # Lengths
                    if lengths is not None:
                        flat_key_values[
                            _prefix_last_key(total_key, "<NJT_LENGTHS>")
                        ] = lengths
                        add_single_value(
                            lengths,
                            _prefix_last_key(key, "<NJT_LENGTHS>"),
                            metadata_dict,
                            lengths.dtype,
                            lengths.shape,
                            flat_size,
                        )
                    # Offsets
                    flat_key_values[_prefix_last_key(total_key, "<NJT_OFFSETS>")] = (
                        offsets
                    )
                    add_single_value(
                        offsets,
                        _prefix_last_key(key, "<NJT_OFFSETS>"),
                        metadata_dict,
                        offsets.dtype,
                        offsets.shape,
                        flat_size,
                        ragged_idx=value._ragged_idx,
                    )

                else:
                    raise NotImplementedError(
                        "NST is not supported, please use layout=torch.jagged when building the nested tensor."
                    )
                return
            flat_key_values[total_key] = value
            add_single_value(
                value,
                key,
                metadata_dict,
                value.dtype,
                value.shape,
                # value.device,
                flat_size,
            )

        self._fast_apply(
            assign,
            named=True,
            call_on_nested=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS_NONTENSOR,
            filter_empty=True,
        )
        return metadata_dict, flat_key_values, flat_size, need_padding

    def consolidate(
        self,
        filename: Path | str | None = None,
        *,
        num_threads=0,
        device: torch.device | None = None,
        non_blocking: bool = False,
        inplace: bool = False,
        return_early: bool = False,
        use_buffer: bool = False,
        share_memory: bool = False,
        pin_memory: bool = False,
        metadata: bool = False,
    ) -> None:
        """Copies all the leaves into one contiguous storage, and returns a tensordict of views of it.

        A consolidated tensordict pickles and moves between devices faster,
        because its leaves are one block of memory: :meth:`~.to`,
        :mod:`pickle`, :meth:`~.send` and :meth:`~.broadcast` move that block
        in one copy. The result is locked (see :ref:`keys-locking`): you can
        write into its leaves in place, but you cannot add or remove entries.

        If the tensordict is already consolidated, ``self`` is returned and
        the other arguments are ignored, except ``pin_memory=True``: a
        storage on the CPU that is not pinned is then copied into a pinned
        one, and a new tensordict is returned, unless ``filename``,
        ``inplace=True``, ``return_early=True`` with ``num_threads > 1`` or a
        ``device`` other than the CPU is given, or no accelerator is
        available. A ``filename`` other than its own
        file writes nothing and emits a :class:`UserWarning`:
        ``td.contiguous().consolidate(filename=other)`` writes a new file. A
        ``device`` other than the one of the storage, and
        ``share_memory=True`` with a storage that is not in shared memory,
        are ignored with a :class:`UserWarning`:
        ``td.contiguous().consolidate(...)`` copies the storage into a new one
        and honours them.

        Args:
            filename (str or Path, optional): a file to use as the storage. The
                leaves are views into a memory map of the file, and the
                metadata is written after them, so that
                :meth:`~.from_consolidated` can read the file back. Defaults to
                ``None``, which keeps the storage in memory.

        Keyword Args:
            num_threads (int, optional): the number of threads that copy the
                leaves into the storage, by chunks of about equal size. When
                ``filename`` is given, ``return_early=False`` and all the
                leaves are on the CPU, one copy is faster and ``num_threads``
                is ignored. Defaults to ``0``, which copies in the calling
                thread.
            device (torch.device, str or int, optional): the device of the
                storage. Exclusive with ``filename``, unless it is ``"cpu"``.
                Defaults to ``None``, which uses the device of the tensordict,
                or the CPU if it has none.
            non_blocking (bool, optional): if ``True``, copies between devices
                can run asynchronously with respect to the host. Defaults to
                ``False``.
            inplace (bool, optional): if ``True``, the leaves of the tensordict
                are replaced by the views, and ``self`` is returned. Defaults
                to ``False``.
            return_early (bool, optional): if ``True`` and ``num_threads > 1``,
                returns a :class:`~tensordict.utils.TensorDictFuture` at once,
                while the threads copy the leaves: ``future.result()`` waits
                for them and returns the tensordict. Not supported with
                ``use_buffer=True`` or ``non_blocking=True``. Defaults to
                ``False``.
            use_buffer (bool, optional): if ``True`` and ``filename`` is given,
                the leaves are copied into a buffer in shared memory, which is
                written to the file at the end. This can be faster on a remote
                file system (for example NFS). In-place writes into the leaves
                then do not reach the file. Defaults to ``False``.
            share_memory (bool, optional): if ``True``, the storage is placed
                in shared memory. Defaults to ``False``.
            pin_memory (bool, optional): if ``True``, the storage is placed in
                pinned memory. Defaults to ``False``.
            metadata (bool, optional): if ``True``, the structure of the
                tensordict is stored with the storage, which changes how the
                result is pickled; :meth:`~.send` and :meth:`~.init_remote` use
                it. It is always stored when ``filename`` is given. Defaults to
                ``False``.

        Returns:
            A consolidated tensordict, ``self`` if ``inplace=True``, or a
            :class:`~tensordict.utils.TensorDictFuture` of it if
            ``return_early=True`` and ``num_threads > 1``.

        .. note::
            The file written with ``filename`` holds the batch sizes and names
            at the time of the call. Setting ``batch_size`` or ``names`` on the
            result later does not change the file: call
            ``consolidate(filename=filename)`` again on the result to write
            them. Likewise, a pickle of the result is rebuilt from metadata
            recorded here: setting the batch size of a nested tensordict, as in
            ``td["nested"].batch_size = ...``, may not update it until
            ``consolidate()`` is called again.

        Examples:
            >>> import pickle
            >>> import torch
            >>> from tensordict import TensorDict
            >>> data = TensorDict({"a": torch.zeros(()), "b": {"c": torch.zeros(())}})
            >>> data_consolidated = data.consolidate()
            >>> # All the leaves share one storage.
            >>> assert torch.tensor([
            ...     v.untyped_storage().data_ptr() for v in data_consolidated.values(True, True)
            ... ]).unique().numel() == 1
            >>> data_consolidated.is_consolidated(), data_consolidated.is_locked
            (True, True)
            >>> # A consolidated tensordict pickles as one buffer.
            >>> restored = pickle.loads(pickle.dumps(data_consolidated))
            >>> (restored == data).all()
            True

        """
        if self.is_consolidated() and not (
            # A storage that is not pinned is copied into a pinned one, as for
            # a tensordict that is not consolidated. Without an accelerator
            # there is no pinned memory, and self is returned as before. With
            # return_early, a tensorclass would come back as a TensorDict.
            pin_memory
            and filename is None
            and not inplace
            and not (return_early and (num_threads or 0) > 1)
            and (device is None or torch.device(device).type == "cpu")
            and self._consolidated["storage"].device.type == "cpu"
            and torch.accelerator.is_available()
            and not self._consolidated["storage"].is_pinned()
        ):
            self._update_consolidated_metadata()
            consolidated = self._consolidated
            storage = consolidated["storage"]
            if (
                filename is not None
                and consolidated.get("filename") == Path(filename).resolve()
                and consolidated["file_id"] == _file_id(filename)
            ):
                _write_consolidated_metadata(
                    filename,
                    consolidated["metadata"],
                    data_size=consolidated["storage"].numel(),
                )
                consolidated["file_id"] = _file_id(filename)
            elif filename is not None:
                if consolidated.get("filename") is None:
                    reason = (
                        "it has no file of its own to rewrite (it was "
                        "consolidated in memory or loaded with "
                        "from_consolidated)"
                    )
                else:
                    reason = "only the file it was consolidated into is rewritten"
                warnings.warn(
                    f"consolidate(filename={str(filename)!r}) writes nothing: "
                    f"the tensordict is already consolidated, and {reason}. "
                    "Call td.contiguous().consolidate(filename=...) to write a "
                    "new file.",
                    UserWarning,
                    stacklevel=_user_stacklevel(),
                )
            ignored = []
            if device is not None:
                try:
                    device = torch.device(device)
                except (RuntimeError, TypeError):
                    # An invalid device is ignored as well, as before
                    device = storage.device
                if device.type != storage.device.type or (
                    device.type != "cpu"
                    and device.index is not None
                    and device.index != storage.device.index
                ):
                    ignored.append(f"device={str(device)!r}")
            if (
                share_memory
                and filename is None
                and storage.device.type == "cpu"
                and not storage.is_shared()
            ):
                ignored.append("share_memory=True")
            if ignored:
                warnings.warn(
                    f"consolidate() ignores {' and '.join(ignored)} on a "
                    "tensordict that is already consolidated, and returns it "
                    "as is. Call td.contiguous().consolidate(...) to copy the "
                    "storage into a new one that honours "
                    f"{'them' if len(ignored) > 1 else 'it'}.",
                    UserWarning,
                    stacklevel=_user_stacklevel(),
                )
            return self

        (
            metadata_dict,
            flat_dict,
            flat_size,
            need_padding,
        ) = self._reduce_vals_and_metadata(
            requires_metadata=filename is not None or metadata, dtype=None
        )
        filesize = sum(flat_size)
        device = torch.device(device) if device is not None else None
        if filename is None:
            storage = torch.empty(
                filesize,
                dtype=torch.uint8,
                device=device if device else self.device,
                pin_memory=pin_memory,
            )
            if share_memory and not (
                device is not None and device.type == "cuda"
            ):  # cuda device is always shared
                storage.share_memory_()
        else:
            # Represent the json metadata as a tensor
            metadata_dict_json = torch.as_tensor(
                bytearray(_consolidated_metadata_json(metadata_dict)), dtype=torch.uint8
            )
            len_metadata = torch.tensor(
                [metadata_dict_json.numel()], dtype=torch.int64
            ).view(torch.uint8)

            if device not in (torch.device("cpu"), None):
                raise RuntimeError(
                    "device and filename are mutually exclusive arguments."
                )
            suffix = len_metadata.numel() + metadata_dict_json.numel()
            if not use_buffer:
                total_storage = torch.from_file(
                    str(filename),
                    size=filesize + suffix,
                    dtype=torch.uint8,
                    shared=True,
                    # needed when device ctx differs
                    device=torch.device("cpu"),
                )
            else:
                total_storage = MemoryMappedTensor.empty(
                    shape=(filesize + suffix,),
                    dtype=torch.uint8,
                )

            total_storage[-8:] = len_metadata
            total_storage[-8 - metadata_dict_json.numel() : -8] = metadata_dict_json
            storage = total_storage[:-suffix]
            # assert len(storage.untyped_storage()) == filesize

        offsets = torch.tensor([0] + flat_size).cumsum(0).tolist()

        def view_old_as_new(v, oldv):
            v = v.view(oldv.dtype)
            if v.numel() > oldv.numel():
                return v[: oldv.numel()].view(oldv.shape)
            return v.view(oldv.shape)

        if num_threads is None:
            num_threads = 0

        if return_early and num_threads > 1:
            if use_buffer:
                raise NotImplementedError(
                    "return_early=True is not supported with use_buffer=True in `consolidate`: "
                    "the buffer must be written to the file once the copies are done."
                )
            if non_blocking:
                raise NotImplementedError(
                    "return_early=True is not supported with non_blocking=True in `consolidate`: "
                    "the storage cannot be synchronized once the copies are done."
                )
        use_threads = num_threads > 1 and (return_early or len(flat_dict) > 1)
        if use_threads and filename is not None and not return_early:
            # Concurrent writes into a freshly created memory-mapped file are
            # dominated by page-fault and writeback contention and measure
            # 2x-4x slower than a single fused copy on local filesystems.
            # Threads only pay off there when they can overlap device
            # transfers. With return_early=True threading is about freeing the
            # main thread rather than raw speed, so it is kept in that case.
            use_threads = any(v.device != storage.device for v in flat_dict.values())
        consolidate_futures = None
        if use_threads:
            values = list(flat_dict.values())

            if all(v.device == storage.device for v in values):
                # Prepare the flat uint8 views on the main thread: this work
                # is cheap but GIL-bound, so running it inside the workers
                # only adds contention. The workers then execute one fused,
                # GIL-releasing copy per chunk.
                flat_views = []
                for idx, v in enumerate(values):
                    if v.is_nested:
                        flat_views.append(None)
                        continue
                    stride = v.stride()
                    # reshape(-1) may return a view that is not stride-1,
                    # e.g. for x[:, :1], which cannot be viewed as bytes
                    if (
                        (stride and stride[-1] != 1)
                        or v.storage_offset()
                        or not v.is_contiguous()
                    ):
                        v = v.clone(memory_format=torch.contiguous_format)
                    flat_view = v.reshape(-1).view(torch.uint8)
                    pad = offsets[idx + 1] - offsets[idx] - flat_view.numel()
                    if pad:
                        flat_view = torch.cat([flat_view, flat_view.new_zeros(pad)])
                    flat_views.append(flat_view)

                def _copy_chunk(start_idx, stop_idx):
                    """Copies values[start_idx:stop_idx] into their storage slices."""
                    items = [
                        flat_view
                        for flat_view in flat_views[start_idx:stop_idx]
                        if flat_view is not None
                    ]
                    if items:
                        torch.cat(
                            items, out=storage[offsets[start_idx] : offsets[stop_idx]]
                        )

            else:

                def _copy_chunk(start_idx, stop_idx):
                    """Copies values[start_idx:stop_idx] into their storage slices.

                    Each leaf is copied individually so that the device
                    transfers run from this worker thread.
                    """
                    for idx in range(start_idx, stop_idx):
                        v = values[idx]
                        if v.is_nested:
                            continue
                        flat_view = v.contiguous().view(-1)
                        if flat_view.stride(-1) != 1:
                            # a one-element leaf such as x[:, 0] of a (1, 5)
                            # tensor counts as contiguous and keeps its stride
                            flat_view = flat_view.clone(
                                memory_format=torch.contiguous_format
                            )
                        flat_view = flat_view.view(torch.uint8)
                        start, stop = offsets[idx], offsets[idx + 1]
                        pad = stop - start - flat_view.numel()
                        storage[start : stop - pad].copy_(
                            flat_view, non_blocking=non_blocking
                        )
                        if pad:
                            storage[stop - pad : stop].zero_()

            # split the leaves in contiguous chunks of roughly equal byte size
            # and run one fused copy per chunk: per-leaf tasks are dominated
            # by task and per-copy overhead when the leaves are small
            target_bytes = max(1, -(-filesize // num_threads))
            chunks = []
            chunk_start = 0
            for idx in range(1, len(values) + 1):
                if (
                    idx == len(values)
                    or offsets[idx] - offsets[chunk_start] >= target_bytes
                ):
                    if idx > chunk_start:
                        chunks.append((chunk_start, idx))
                    chunk_start = idx
            executor = _get_shared_executor(num_threads)
            futures = [
                executor.submit(_copy_chunk, start_idx, stop_idx)
                for start_idx, stop_idx in chunks
            ]
            if return_early:
                # the result construction below only manipulates storage
                # views and metadata, so it can proceed while the copies are
                # still running
                consolidate_futures = futures
            else:
                wait(futures)
                # re-raise the first error of the workers
                for future in futures:
                    future.result()
                if non_blocking and (device is None or device.type != "cuda"):
                    # sync if needed
                    self._sync_all()
        else:

            def _view_and_pad(tensor):
                result = tensor.reshape(-1).view(torch.uint8)
                # result must always have a multiple of 8 elements
                pad = 0
                if need_padding:
                    pad = result.numel() % 8
                    if pad != 0:
                        result = torch.cat([result, result.new_zeros(8 - pad)])
                return result, pad

            items = []
            for v in flat_dict.values():
                if v.is_nested:
                    continue
                if v.device != storage.device:
                    v = v.to(storage.device, non_blocking=non_blocking)
                stride = v.stride()
                if is_compiling():
                    if not v.is_contiguous():
                        v = v.clone(memory_format=torch.contiguous_format)
                elif (
                    (stride and stride[-1] != 1)
                    or v.storage_offset()
                    or not v.is_contiguous()
                ):
                    v = v.clone(memory_format=torch.contiguous_format)
                v, pad = _view_and_pad(v)
                items.append(v)
            if non_blocking and (device is None or device.type != "cuda"):
                # sync if needed
                self._sync_all()
            if items:
                torch.cat(items, out=storage)
        for v, (k, oldv) in _zip_strict(
            storage.split(flat_size), list(flat_dict.items())
        ):
            if not k[-1].startswith("<"):
                flat_dict[k] = view_old_as_new(v, oldv)
            elif k[-1].startswith("<NJT>"):
                # NJT/NT always comes before offsets/shapes
                nt = oldv
                nt_lengths = None
                del flat_dict[k]
            elif k[-1].startswith("<NJT_VALUES>"):
                nt_vaues = view_old_as_new(v, oldv)
                del flat_dict[k]
            elif k[-1].startswith("<NJT_LENGTHS>"):
                nt_lengths = view_old_as_new(v, oldv)
                del flat_dict[k]
            elif k[-1].startswith("<NJT_OFFSETS>"):
                newk = k[:-1] + (k[-1].replace("<NJT_OFFSETS>", ""),)
                nt_offsets = view_old_as_new(v, oldv)
                del flat_dict[k]

                val = _rebuild_njt_from_njt(
                    nt, values=nt_vaues, offsets=nt_offsets, lengths=nt_lengths
                )

                flat_dict[newk] = val

                # delete the nested value to make sure that if there was an
                # ordering mismatch we wouldn't be looking at the value key of
                # another nested tensor.
                del nt, nt_vaues, nt_offsets, nt_lengths
            else:
                flat_dict[k] = view_old_as_new(v, oldv)

        def assign_val(key, val):
            if isinstance(key, str):
                key = (key,)
            if not inplace and _is_non_tensor(type(val)):
                # Locking the result must not lock wrappers owned by the source.
                # Keep the payload shared, as with a shallow TensorDict clone.
                val = val.clone(recurse=False)
            return flat_dict.get(key, val)

        if filename is None:
            device = self.device
        elif not inplace:
            device = torch.device("cpu")
        elif self.device is not None and self.device != torch.device("cpu"):
            self.clear_device_()
            device = None
        else:
            device = None
        result = self._fast_apply(
            assign_val,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS_NONTENSOR,
            out=self if inplace else None,
            device=device,
        )
        result._consolidated = {"storage": storage, "metadata": metadata_dict}
        # Lock the consolidated TensorDict to prevent modifications that could break consolidation
        result.lock_()
        if filename is not None:
            if use_buffer:
                with open(filename, "w+b") as f:
                    f.write(total_storage._handler.buffer)
            result._consolidated["filename"] = Path(filename).resolve()
            result._consolidated["file_id"] = _file_id(filename)
            # with open(Path(filename).with_suffix(".json"), "wb") as f:
            #     metadata_dict["size"] = filesize
            #     f.write(json.dumps(metadata_dict))
        if consolidate_futures is not None:
            return TensorDictFuture(consolidate_futures, result)
        return result

    @classmethod
    def from_consolidated(cls, filename):
        """Reconstructs a tensordict from a file written by ``consolidate(filename=...)``.

        The result is consolidated: its leaves are views into one
        copy-on-write memory map of the file, so in-place writes into them do
        not reach the file. It is not locked. The batch sizes and names are as
        they were when the file was written. Same as
        :func:`~tensordict.from_consolidated`.

        Args:
            filename (str or Path): the file to read.

        Examples:
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3)}, batch_size=[3])
            >>> with tempfile.TemporaryDirectory() as path:
            ...     _ = td.consolidate(filename=f"{path}/td.bin")
            ...     loaded = TensorDict.from_consolidated(f"{path}/td.bin")
            ...     print(loaded["b", "c"], loaded.is_consolidated())
            tensor([1., 1., 1.]) True
        """
        # with open(Path(filename).with_suffix(".json"), "rb") as f:
        #     metadata = json.loads(f.read())
        file = torch.from_file(
            str(filename),
            dtype=torch.uint8,
            size=os.path.getsize(filename),
            # needed when device ctx differs
            device=torch.device("cpu"),
        )
        metadata_size = file[-8:].clone().view(torch.int64)
        metadata = file[-metadata_size - 8 : -8]
        metadata = json.loads(bytes(metadata.tolist()))

        from tensordict._reductions import _rebuild_tensordict_files_consolidated

        return _rebuild_tensordict_files_consolidated(
            metadata, file[: -metadata_size - 8]
        )

    def is_consolidated(self):
        """Whether the leaves of the tensordict are views into one storage made by :meth:`~.consolidate`.

        Becomes ``False`` when the storage no longer matches the content: after
        :meth:`~.memmap_` moves the leaves out of it, after an in-place write
        into a non-tensor entry, or after an entry is replaced, added or
        deleted in an unpickled copy (which is not locked). In-place writes
        into the tensor leaves keep it ``True``. Call :meth:`~.consolidate`
        to build a new consolidated copy.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3))
            >>> td.is_consolidated(), td.consolidate().is_consolidated()
            (False, True)
        """
        return hasattr(self, "_consolidated") and not self._consolidated.get(
            "stale", False
        )

    def _mark_consolidated_stale(self):
        """Marks the consolidated state of self and of its lock parents as stale."""
        for td in (self, *(ref() for ref in self._lock_parents_weakrefs)):
            consolidated = getattr(td, "_consolidated", None)
            # An unpickled copy is not locked: "parent" links each level to
            # the consolidated state of the tensordict that holds it.
            while consolidated is not None:
                consolidated["stale"] = True
                consolidated = consolidated.get("parent")

    def _update_consolidated_metadata(self) -> None:
        # pickle, torch.save and send rebuild a consolidated tensordict from
        # the metadata of consolidate(): write the current batch sizes and
        # names into it
        consolidated = getattr(self, "_consolidated", None)
        if consolidated is None or consolidated.get("metadata") is None:
            return
        _update_consolidated_metadata(self, consolidated["metadata"])

    def memmap_(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        existsok: bool = True,
        robust_key: bool | None = True,
        atomic: bool = False,
    ) -> Self:
        """Writes every leaf to a memory-mapped file in place, and returns ``self``.

        Writes the same files as :meth:`~.memmap`, then replaces each leaf of
        the tensordict with its :class:`~tensordict.MemoryMappedTensor`, so the
        ``data_ptr()`` of every leaf changes. The device of the tensordict
        becomes ``"cpu"``. See :ref:`saving`.

        The tensordict is then locked (see :ref:`keys-locking`): you can write
        into its leaves in place, but you cannot add, remove or replace an
        entry. Once it is unlocked, :meth:`~.is_memmap` returns ``False``.

        Args:
            prefix (str or Path, optional): the directory to write to.
                Defaults to ``None``, which writes into the directory of the
                tensordict if it is already memory-mapped there, and otherwise
                into anonymous shared memory, which has no file name and lasts
                as long as the process.
            copy_existing (bool, optional): if ``False``, a leaf that is
                already a memory-mapped tensor with a file at another path
                raises a ``RuntimeError``, so that data on disk is not
                duplicated by accident. If ``True``, the file is copied to the
                new path. Defaults to ``False``.

        Keyword Args:
            num_threads (int, optional): the number of threads that write the
                files. ``0`` and ``1`` write them one after the other in the
                calling thread. Threads help with a few large leaves and slow
                down many small ones. Defaults to ``0``.
            return_early (bool, optional): if ``True`` and ``num_threads > 1``,
                returns a :class:`~tensordict.utils.TensorDictFuture` at once,
                while the threads write the files: ``future.result()`` waits
                for them and returns the tensordict. Defaults to ``False``.
            share_non_tensor (bool, optional): if ``True``, non-tensor data is
                shared between the processes of the node, so that an in-place
                write of a non-tensor entry in one process is seen by the
                others. Many non-tensor leaves (for example a large stack of
                non-tensor data) can use up the memory. Defaults to ``False``.
            existsok (bool, optional): if ``False``, a file that already exists
                at the path of a leaf raises a ``RuntimeError``. If ``True``,
                it is overwritten. Defaults to ``True``.
            robust_key (bool, optional): if ``True``, keys are percent-encoded
                in file and directory names (``"a/b"`` is written to
                ``a%2Fb.memmap``), so that every file stays inside ``prefix``.
                If ``False``, keys are used as file names unchanged, as in the
                legacy layout: do not use it with keys you do not trust.
                ``None`` is the same as ``True``. Defaults to ``True``.
            atomic (bool, optional): if ``True``, the files are written to a
                new directory next to ``prefix``, which is then renamed to
                ``prefix``, so that an error or a crash during the write
                leaves ``prefix`` as it was. The new directory is removed if
                the write fails (but not after a crash). An existing directory
                at ``prefix`` is replaced as a whole once the new one is
                complete, so files that the tensordict does not write are
                removed: it must be empty or hold a saved tensordict (a
                ``meta.json`` file). Readers see the old or the new directory,
                except for the short time between two renames, during which
                ``prefix`` is missing. The write needs about twice the disk
                space of the tensordict while it runs, and a leaf that is
                already memory-mapped to a file is copied, which needs
                ``copy_existing=True``. The files are not synced to the disk,
                so this guards against a killed process, not against a power
                failure. If the write fails, the leaves already converted map
                deleted files, so a retry needs ``copy_existing=True``. Not
                supported with ``return_early=True``, nor on Windows, where a
                directory that holds open files cannot be renamed. Defaults
                to ``False``.

        Returns:
            ``self``, or a :class:`~tensordict.utils.TensorDictFuture` of it if
            ``return_early=True`` and ``num_threads > 1``.

        Raises:
            RuntimeError: if a leaf already has a file elsewhere and
                ``copy_existing=False``, if a file exists and
                ``existsok=False``, if the tensordict is in shared memory
                (see :meth:`~.share_memory_`), or if it is a
                :class:`~tensordict.PersistentTensorDict`, a
                :class:`~tensordict.store.TensorDictStore`, a
                :class:`~tensordict.store.LazyStackedTensorDictStore` or one of
                their elements. Use :meth:`~.memmap` to make a memory-mapped
                copy of those. With ``atomic=True``, also if ``prefix`` is a
                non-empty directory and ``existsok=False``, or if it holds no
                saved tensordict.
            NotADirectoryError: if ``atomic=True`` and ``prefix`` is a file.
            ValueError: if ``atomic=True`` and there is no directory to write
                to.
            NotImplementedError: if ``return_early=True`` with
                ``atomic=True``, or ``atomic=True`` on Windows.

        Examples:
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(obs=torch.zeros(3, 4), batch_size=[3])
            >>> with tempfile.TemporaryDirectory() as path:
            ...     out = td.memmap_(path)
            ...     print(out is td, td.is_memmap(), td.is_locked)
            ...     print(type(td["obs"]).__name__)
            True True True
            MemoryMappedTensor
        """
        if self._memmap_inplace_unsupported:
            raise RuntimeError(
                "Converting a sub-tensordict values to memmap cannot be done."
            )
        prefix = Path(prefix) if prefix is not None else self._memmap_prefix
        if atomic:
            if return_early:
                raise NotImplementedError(
                    "return_early is not supported with atomic=True."
                )
            return _memmap_atomically(
                prefix,
                lambda path: self.memmap_(
                    path,
                    copy_existing=copy_existing,
                    num_threads=num_threads,
                    share_non_tensor=share_non_tensor,
                    existsok=existsok,
                    robust_key=robust_key,
                ),
                existsok=existsok,
            )
        if num_threads > 1:
            executor = _get_shared_executor(num_threads)
            futures = []
            result = self._memmap_(
                prefix=prefix,
                copy_existing=copy_existing,
                executor=executor,
                futures=futures,
                inplace=True,
                like=False,
                share_non_tensor=share_non_tensor,
                existsok=existsok,
                robust_key=robust_key,
            )
            if not return_early:
                concurrent.futures.wait(futures)
                # re-raise the first error of the workers
                for future in futures:
                    future.result()
                return result
            return TensorDictFuture(futures, result)
        return self._memmap_(
            prefix=prefix,
            copy_existing=copy_existing,
            inplace=True,
            futures=None,
            executor=None,
            like=False,
            share_non_tensor=share_non_tensor,
            existsok=existsok,
            robust_key=robust_key,
        ).lock_()

    def save(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        robust_key: bool | None = True,
        archive: bool | None = None,
        compression: str | int | None = None,
        atomic: bool = False,
    ) -> Self:
        """Saves the tensordict to disk, and returns a memory-mapped copy of it.

        ``save`` (and its alias ``dumps``) calls :meth:`~.memmap` with
        ``existsok=True``. :meth:`~.memmap` documents the arguments in full.
        The tensordict is not changed. Read the files back with :meth:`~.load`.
        See :ref:`saving`.

        Args:
            prefix (str or Path, optional): the directory where the tensordict
                is written, one file per leaf. If ``prefix`` ends with
                ``".tdz"`` or ``archive=True``, a single archive file is
                written instead. Defaults to ``None``, which writes into the
                directory of the tensordict if it is already memory-mapped
                there, and otherwise into anonymous shared memory, which has no
                file name and lasts as long as the process.
            copy_existing (bool, optional): if ``False``, a leaf that is
                already a memory-mapped tensor with a file at another path
                raises a ``RuntimeError``. If ``True``, the file is copied to
                the new path. Defaults to ``False``.

        Keyword Args:
            num_threads (int, optional): the number of threads that write the
                files. Defaults to ``0``, which writes them in the calling
                thread.
            return_early (bool, optional): if ``True`` and ``num_threads > 1``,
                returns a :class:`~tensordict.utils.TensorDictFuture` while
                the threads write the files. Defaults to ``False``.
            share_non_tensor (bool, optional): if ``True``, non-tensor data is
                shared between the processes of the node. Defaults to
                ``False``.
            robust_key (bool, optional): if ``True``, keys are percent-encoded
                in file names, so that every file stays inside ``prefix``.
                Defaults to ``True``.
            archive (bool, optional): if ``True``, writes a single archive
                file. Defaults to ``None``, which writes an archive if
                ``prefix`` ends with ``".tdz"``.
            compression (str or int, optional): the compression of the
                archive entries. Defaults to ``None``, which stores them
                uncompressed.
            atomic (bool, optional): if ``True``, the directory is written
                next to ``prefix`` and then renamed to it, so that an error or
                a crash during the write leaves ``prefix`` as it was. Not
                supported on Windows for a directory. An archive is always
                written this way.
                :meth:`~.memmap` gives the details. Defaults to ``False``.

        Returns:
            A new memory-mapped tensordict, as :meth:`~.memmap` returns.

        Examples:
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), nested=TensorDict(b=torch.ones(3), batch_size=[3]), batch_size=[3])
            >>> with tempfile.TemporaryDirectory() as path:
            ...     _ = td.save(path)
            ...     loaded = TensorDict.load(path)
            ...     print((loaded == td).all())
            True
        """
        return self.memmap(
            prefix=prefix,
            copy_existing=copy_existing,
            num_threads=num_threads,
            return_early=return_early,
            share_non_tensor=share_non_tensor,
            robust_key=robust_key,
            archive=archive,
            compression=compression,
            atomic=atomic,
        )

    dumps = save

    def memmap(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        existsok: bool = True,
        robust_key: bool | None = True,
        archive: bool | None = None,
        compression: str | int | None = None,
        atomic: bool = False,
    ) -> Self:
        """Writes every leaf to a memory-mapped file and returns a new tensordict of them.

        The tensordict is not changed: the result holds new
        :class:`~tensordict.MemoryMappedTensor` leaves. Each leaf is written to
        ``<prefix>/<key>.memmap``, each nested tensordict to a subdirectory,
        and each level has a ``meta.json`` file. Read the files back with
        :meth:`~.load_memmap`. :meth:`~.save` and :meth:`~.dumps` call this
        method, and :meth:`~.memmap_` writes in place. See :ref:`saving`.

        The result is locked (see :ref:`keys-locking`): you can write into its
        leaves in place, but you cannot add, remove or replace an entry. Once
        it is unlocked, :meth:`~.is_memmap` returns ``False``.

        Args:
            prefix (str or Path, optional): the directory to write to. If it
                ends with ``".tdz"``, a single archive file is written instead
                (see ``archive``). Defaults to ``None``, which writes into the
                directory of the tensordict if it is already memory-mapped
                there, and otherwise into anonymous shared memory, which has no
                file name and lasts as long as the process.
            copy_existing (bool, optional): if ``False``, a leaf that is
                already a memory-mapped tensor with a file at another path
                raises a ``RuntimeError``, so that data on disk is not
                duplicated by accident. If ``True``, the file is copied to the
                new path. Defaults to ``False``.

        Keyword Args:
            num_threads (int, optional): the number of threads that write the
                files. ``0`` and ``1`` write them one after the other in the
                calling thread. Threads help with a few large leaves and slow
                down many small ones. Defaults to ``0``.
            return_early (bool, optional): if ``True`` and ``num_threads > 1``,
                returns a :class:`~tensordict.utils.TensorDictFuture` at once,
                while the threads write the files: ``future.result()`` waits
                for them and returns the tensordict. Not supported for an
                archive. Defaults to ``False``.
            share_non_tensor (bool, optional): if ``True``, non-tensor data is
                shared between the processes of the node, so that an in-place
                write of a non-tensor entry in one process is seen by the
                others. Many non-tensor leaves (for example a large stack of
                non-tensor data) can use up the memory. Defaults to ``False``.
            existsok (bool, optional): if ``False``, a file that already exists
                at the path of a leaf (or at the archive path) raises a
                ``RuntimeError``. If ``True``, it is overwritten. Defaults to
                ``True``.
            robust_key (bool, optional): if ``True``, keys are percent-encoded
                in file and directory names (``"a/b"`` is written to
                ``a%2Fb.memmap``), so that every file stays inside ``prefix``.
                If ``False``, keys are used as file names unchanged, as in the
                legacy layout: do not use it with keys you do not trust.
                ``None`` is the same as ``True``. Defaults to ``True``.
            archive (bool, optional): if ``True``, ``prefix`` is a file, and
                the tensordict is written to it as an archive: a zip file
                whose entries are the files of the directory layout, with
                uncompressed and aligned tensor payloads. Defaults to ``None``,
                which writes an archive if ``prefix`` ends with ``".tdz"``.
            compression (str or int, optional): the compression of the archive
                entries: ``"stored"``, ``"deflate"``, ``"bzip2"``, ``"lzma"``,
                or a :mod:`zipfile` constant. A compressed leaf is
                decompressed in memory when it is loaded, instead of being
                memory-mapped. Only valid for an archive. Defaults to ``None``,
                which stores the entries uncompressed.
            atomic (bool, optional): if ``True``, the files are written to a
                new directory next to ``prefix``, which is then renamed to
                ``prefix``, so that an error or a crash during the write
                leaves ``prefix`` as it was. The new directory is removed if
                the write fails (but not after a crash). An existing directory
                at ``prefix`` is replaced as a whole once the new one is
                complete, so files that the tensordict does not write are
                removed: it must be empty or hold a saved tensordict (a
                ``meta.json`` file). Readers see the old or the new directory,
                except for the short time between two renames, during which
                ``prefix`` is missing. The write needs about twice the disk
                space of the tensordict while it runs, and a leaf that is
                already memory-mapped to a file is copied, which needs
                ``copy_existing=True``. The files are not synced to the disk,
                so this guards against a killed process, not against a power
                failure. Not supported with ``return_early=True``, nor for a
                directory on Windows, where a directory that holds open files
                cannot be renamed. An archive is always written this way.
                Defaults to ``False``.

        Returns:
            A new memory-mapped tensordict, or a
            :class:`~tensordict.utils.TensorDictFuture` of it if
            ``return_early=True`` and ``num_threads > 1``.

        Raises:
            RuntimeError: if a leaf already has a file elsewhere and
                ``copy_existing=False``, if a file exists and
                ``existsok=False``, or with ``atomic=True``, if ``prefix`` is
                a non-empty directory and ``existsok=False``, or if it holds
                no saved tensordict.
            NotADirectoryError: if ``atomic=True`` and ``prefix`` is a file.
            ValueError: if ``compression`` is given without an archive, or if
                an archive or ``atomic=True`` is requested without ``prefix``.
            NotImplementedError: if ``return_early=True`` for an archive or
                with ``atomic=True``, or ``atomic=True`` for a directory on
                Windows.

        .. note::
            For an archive, the result is ``load_memmap(prefix)``: its leaves
            are views into one copy-on-write mapping of the file, so in-place
            writes do not reach the file, and it is not locked. Archives are
            written in one pass, so ``num_threads`` does not speed them up. An
            existing archive is replaced by a new file: a symlink is followed,
            but other hard links keep the old content.
            :func:`~tensordict.pack_memmap` and
            :func:`~tensordict.unpack_memmap` convert between archives and
            directories.

        Examples:
            >>> import os
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(
            ...     obs=torch.zeros(3, 4),
            ...     next=TensorDict(reward=torch.ones(3, 1), batch_size=[3]),
            ...     batch_size=[3],
            ... )
            >>> with tempfile.TemporaryDirectory() as path:
            ...     mm = td.memmap(path)
            ...     print(type(mm["obs"]).__name__, mm.is_locked)
            ...     print(sorted(os.listdir(path)))
            MemoryMappedTensor True
            ['meta.json', 'next', 'obs.memmap']
        """
        from tensordict.base import TensorDictBase

        if archive is None:
            archive = (
                prefix is not None and Path(prefix).suffix == TENSORDICT_ARCHIVE_SUFFIX
            )
        if archive:
            if prefix is None:
                raise ValueError("A path is required to write a memmap archive.")
            if return_early:
                raise NotImplementedError(
                    "return_early is not supported when writing a memmap archive."
                )
            _save_as_archive(
                self,
                prefix,
                num_threads=num_threads,
                compression=compression,
                copy_existing=copy_existing,
                share_non_tensor=share_non_tensor,
                existsok=existsok,
                robust_key=robust_key,
            )
            # dispatch on the class recorded in the archive metadata rather
            # than type(self): some views (e.g. sub-tensordicts) are saved as
            # a different class than the one they are created from.
            # This archive was created by this process from the in-memory
            # object, so reloading its arbitrary non-tensor fields is trusted.
            return TensorDictBase.load_memmap(prefix, allow_pickle=True)
        if compression is not None:
            raise ValueError(
                "compression is only supported when writing a memmap archive "
                "(pass archive=True or use a '.tdz' prefix)."
            )
        prefix = Path(prefix) if prefix is not None else self._memmap_prefix
        if atomic:
            if return_early:
                raise NotImplementedError(
                    "return_early is not supported with atomic=True."
                )
            return _memmap_atomically(
                prefix,
                lambda path: self.memmap(
                    path,
                    copy_existing=copy_existing,
                    num_threads=num_threads,
                    share_non_tensor=share_non_tensor,
                    existsok=existsok,
                    robust_key=robust_key,
                ),
                existsok=existsok,
            )

        if num_threads > 1:
            executor = _get_shared_executor(num_threads)
            futures = []
            result = self._memmap_(
                prefix=prefix,
                copy_existing=copy_existing,
                executor=executor,
                futures=futures,
                inplace=False,
                like=False,
                share_non_tensor=share_non_tensor,
                existsok=existsok,
                robust_key=robust_key,
            )
            if not return_early:
                concurrent.futures.wait(futures)
                # re-raise the first error of the workers
                for future in futures:
                    future.result()
                return result
            return TensorDictFuture(futures, result)

        return self._memmap_(
            prefix=prefix,
            copy_existing=copy_existing,
            inplace=False,
            executor=None,
            like=False,
            futures=None,
            share_non_tensor=share_non_tensor,
            existsok=existsok,
            robust_key=robust_key,
        ).lock_()

    def memmap_like(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        *,
        existsok: bool = True,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        robust_key: bool | None = True,
        archive: bool | None = None,
    ) -> Self:
        """Creates a memory-mapped tensordict with the same keys, shapes and dtypes, without copying the values.

        Use it to preallocate a large buffer on disk, then fill it with
        :meth:`~.update_` or by indexing: :meth:`~.memmap` would write every
        value first. New files are filled with zeros. A file that already
        exists at the path of a leaf (with ``existsok=True``) is mapped as it
        is and keeps its content, which lets you reopen a buffer, or its
        start with a smaller leaf. If the file is smaller than the leaf, it
        is grown with zeros and a ``FutureWarning`` is emitted. Starting with
        TensorDict 0.17, the default will change to raising a
        ``RuntimeError`` in that case. A file written with another dtype or
        shape is not detected otherwise, and its bytes are read as the new
        dtype. An archive (``".tdz"``) is always
        rebuilt with zeros. The tensordict is not changed. See
        :ref:`saving`.

        The result is locked (see :ref:`keys-locking`): you can write into its
        leaves in place, but you cannot add, remove or replace an entry. Once
        it is unlocked, :meth:`~.is_memmap` returns ``False``.

        Args:
            prefix (str or Path, optional): the directory to write to. If it
                ends with ``".tdz"``, a single archive file is written instead
                (see ``archive``). Defaults to ``None``, which allocates the
                leaves in anonymous shared memory, which has no file name and
                lasts as long as the process.
            copy_existing (bool, optional): accepted for symmetry with
                :meth:`~.memmap`. It has no effect, since no value is copied.
                Defaults to ``False``.

        Keyword Args:
            existsok (bool, optional): if ``False``, a file that already exists
                at the path of a leaf (or at the archive path) raises a
                ``RuntimeError``. If ``True``, it is reused. Defaults to
                ``True``.
            num_threads (int, optional): the number of threads that create the
                files. ``0`` and ``1`` create them one after the other in the
                calling thread. Defaults to ``0``.
            return_early (bool, optional): if ``True`` and ``num_threads > 1``,
                returns a :class:`~tensordict.utils.TensorDictFuture` at once,
                while the threads create the files: ``future.result()`` waits
                for them and returns the tensordict. Not supported for an
                archive. Defaults to ``False``.
            share_non_tensor (bool, optional): if ``True``, non-tensor data is
                shared between the processes of the node, so that an in-place
                write of a non-tensor entry in one process is seen by the
                others. Many non-tensor leaves (for example a large stack of
                non-tensor data) can use up the memory. Defaults to ``False``.
            robust_key (bool, optional): if ``True``, keys are percent-encoded
                in file and directory names (``"a/b"`` is written to
                ``a%2Fb.memmap``), so that every file stays inside ``prefix``.
                If ``False``, keys are used as file names unchanged, as in the
                legacy layout: do not use it with keys you do not trust.
                ``None`` is the same as ``True``. Defaults to ``True``.
            archive (bool, optional): if ``True``, ``prefix`` is a file, and a
                zero-filled archive is created there and loaded back with
                ``load_memmap(prefix, mode="r+")``, so that in-place writes into
                the result reach the archive. Nested tensors are not supported.
                Defaults to ``None``, which creates an archive if ``prefix``
                ends with ``".tdz"``.

        Returns:
            A new memory-mapped tensordict, or a
            :class:`~tensordict.utils.TensorDictFuture` of it if
            ``return_early=True`` and ``num_threads > 1``.

        .. note::
            For an archive, the result is not locked, and in-place writes do
            not update the CRC-32 checksums of the zip entries. Call
            :func:`~tensordict.refresh_archive_checksums` before you give the
            archive to a tool that checks them, such as ``unzip``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     "a": torch.zeros((3, 64, 64), dtype=torch.uint8),
            ...     "b": torch.zeros(1, dtype=torch.int64),
            ... }, batch_size=[]).expand(1_000)  # expand does not allocate new memory
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as path:
            ...     buffer = td.memmap_like(path)
            ...     print(buffer.is_memmap(), buffer["a"].shape)
            True torch.Size([1000, 3, 64, 64])

        """
        from tensordict.base import TensorDictBase

        if archive is None:
            archive = (
                prefix is not None and Path(prefix).suffix == TENSORDICT_ARCHIVE_SUFFIX
            )
        if archive:
            if prefix is None:
                raise ValueError("A path is required to write a memmap archive.")
            if return_early:
                raise NotImplementedError(
                    "return_early is not supported when writing a memmap archive."
                )
            _make_archive_like(
                self,
                prefix,
                num_threads=num_threads,
                copy_existing=copy_existing,
                share_non_tensor=share_non_tensor,
                existsok=existsok,
                robust_key=robust_key,
            )
            # This archive was created by this process from the in-memory
            # object, so reloading its arbitrary non-tensor fields is trusted.
            return TensorDictBase.load_memmap(prefix, mode="r+", allow_pickle=True)
        # Without a prefix, the result lives in new anonymous memory: falling
        # back to self._memmap_prefix would reopen the source's files.
        if prefix is not None:
            prefix = Path(prefix)
        if num_threads > 1:
            executor = _get_shared_executor(num_threads)
            futures = []

            # we create an empty copy of self
            # This is because calling MMapTensor.from_tensor(mmap_tensor) does nothing
            # if both are in filesystem
            input = self.apply(_memmap_like_stand_in, named=True, nested_keys=True)
            result = input._memmap_(
                prefix=prefix,
                copy_existing=copy_existing,
                executor=executor,
                futures=futures,
                inplace=False,
                like=True,
                share_non_tensor=share_non_tensor,
                existsok=existsok,
                robust_key=robust_key,
            )
            if not return_early:
                concurrent.futures.wait(futures)
                # re-raise the first error of the workers
                for future in futures:
                    future.result()
                return result
            return TensorDictFuture(futures, result)

        input = self.apply(_memmap_like_stand_in, named=True, nested_keys=True)
        return input._memmap_(
            prefix=prefix,
            copy_existing=copy_existing,
            inplace=False,
            like=True,
            executor=None,
            futures=None,
            share_non_tensor=share_non_tensor,
            existsok=existsok,
            robust_key=robust_key,
        ).lock_()

    @classmethod
    def load(cls, prefix: str | Path, *args, **kwargs) -> Self:
        """Loads a tensordict that :meth:`~.save` or :meth:`~.memmap` wrote to disk.

        ``load`` calls :meth:`~.load_memmap`, which documents the other
        arguments, such as ``device``, ``mode``, ``subpath`` and
        ``allow_pickle``. The leaves of the result read the files lazily. See
        :ref:`saving`.

        Args:
            prefix (str or Path): the directory, or the archive file, to read.
            *args: passed to :meth:`~.load_memmap`.
            **kwargs: passed to :meth:`~.load_memmap`.

        Examples:
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.arange(3), batch_size=[3])
            >>> with tempfile.TemporaryDirectory() as path:
            ...     _ = td.save(path)
            ...     loaded = TensorDict.load(path)
            ...     print(loaded["a"], loaded.batch_size)
            MemoryMappedTensor([0, 1, 2]) torch.Size([3])
        """
        return cls.load_memmap(prefix, *args, **kwargs)

    @deprecated("TensorDictBase.load_()", removal="0.17", replacement="load_memmap_()")
    def load_(self, prefix: str | Path, *args, **kwargs):
        """Loads a saved tensordict into the tensordict.

        Calls :meth:`~.load_memmap_` with the same arguments.

        .. deprecated:: 0.15
            Use :meth:`~.load_memmap_` instead. It will be removed in
            TensorDict 0.17.

        Args:
            prefix (str or Path): the directory, or the archive file, to read.
            *args: passed to :meth:`~.load_memmap_`.
            **kwargs: passed to :meth:`~.load_memmap_`.
        """
        return self.load_memmap_(prefix, *args, **kwargs)

    @classmethod
    def load_memmap(
        cls,
        prefix: str | Path,
        device: torch.device | None = None,
        non_blocking: bool = False,
        *,
        out: TensorDictBase | None = None,
        robust_key: bool | None = True,
        subpath: NestedKey | None = None,
        mode: str | None = None,
        num_threads: int = 0,
        allow_pickle: bool | None = None,
    ) -> Self:
        """Loads a tensordict that :meth:`~.memmap` or :meth:`~.save` wrote, from a directory or an archive.

        The leaves of the result read the files lazily: from a directory, each
        leaf is a :class:`~tensordict.MemoryMappedTensor` of its file, and from
        an archive, each leaf is a view into one memory map of the archive.
        Only the pages of the leaves that you use are read from disk. The
        result has the class that was saved (a tensorclass is loaded as that
        tensorclass), and it is not locked. :meth:`~.load` is this method. See
        :ref:`saving`.

        Args:
            prefix (str or Path): the directory to read, or an archive file
                written by ``save(..., archive=True)``, by a ``".tdz"`` prefix,
                or by :func:`~tensordict.pack_memmap`.
            device (torch.device, str or int, optional): the device of the
                result. The leaves are copied to it, which reads the files,
                unless it is the CPU.
                With ``"meta"``, no file is read: the leaves are meta tensors,
                which show the structure and size of the data. Defaults to
                ``None``, which keeps the leaves on the CPU, memory-mapped.
            non_blocking (bool, optional): if ``True``, does not synchronize
                the device after the copies to ``device``. Defaults to
                ``False``.

        Keyword Args:
            out (TensorDictBase, optional): a tensordict to load the data into.
                Its nested tensordicts are reused, but its leaves are replaced
                by the loaded tensors, so their storage is not reused. Its keys
                that are not in the saved data are removed. If ``out`` has a
                device, the data is loaded on it, and another ``device`` raises
                a ``ValueError``. To write the data into the storage of
                ``out``, use
                ``out.update_(TensorDict.load_memmap(prefix, device=device))``.
                Defaults to ``None``.
            robust_key (bool, optional): if ``True``, reads the
                percent-encoded file names that :meth:`~.memmap` writes by
                default, and falls back to the legacy names of keys that are
                safe file names. If ``False``, reads the legacy layout only.
                ``None`` is the same as ``True``. Defaults to ``True``.
            subpath (NestedKey or str, optional): the nested tensordict to
                load, as a nested key (``("module", "0")``) or as a
                ``"/"``-separated path (``"module/0"``). Only that subtree is
                read, from a directory or an archive. A nested key reaches the
                entries of a saved tensorclass as ``obj.get(key)`` does.
                Defaults to ``None``, which loads the whole tensordict.
            mode (str, optional): how the files are mapped. With ``"r"``, the
                mapping is copy-on-write: in-place writes into the leaves stay
                in memory, and the files do not change. With ``"r+"``, in-place
                writes reach the files, and a file that the process cannot
                write raises a ``PermissionError``. Defaults to ``None``, which
                maps an archive as with ``"r"``, and each file of a directory
                as with ``"r+"`` if the process can write it, and as with
                ``"r"`` otherwise. See the note below.
            num_threads (int, optional): the number of threads that decompress
                the leaves of a compressed archive. It has no effect on an
                uncompressed archive or a directory. Defaults to ``0``, which
                decompresses them one after the other.
            allow_pickle (bool, optional): whether non-tensor data saved with
                :mod:`pickle` may be loaded. Pickle can run arbitrary code: pass
                ``True`` only for data you trust, and ``False`` for other data,
                which raises a ``RuntimeError`` on pickled data. Defaults to
                ``None``, which loads pickled data with a ``FutureWarning``.
                Starting with TensorDict 0.16, the default will change to
                ``False``. Data saved without pickle loads with any value.

        Returns:
            The loaded tensordict, or ``out`` if it is given.

        Raises:
            ValueError: if ``prefix`` is a file that is not an archive, if
                ``subpath`` names no saved tensordict, or if ``mode`` is not
                ``None``, ``"r"`` or ``"r+"``.

        .. note::
            Load with ``mode="r"`` data that you only read: on some network
            file systems (for example Lustre), page faults on a shared
            writable mapping take write locks, so readers on different nodes
            block each other. On Linux, a copy-on-write mapping counts against
            the memory commit limit, so a file larger than RAM plus swap can
            fail to map with ``"r"``. With ``"r"``, :meth:`~.memmap_` without a
            prefix does not write back to ``prefix``.

        .. note::
            An archive opened with ``mode="r+"`` must have uncompressed,
            aligned payloads (as written by tensordict without
            ``compression``) and no nested tensor. In-place writes do not
            update the CRC-32 checksums of its entries: ``load_memmap`` does
            not check them, but call
            :func:`~tensordict.refresh_archive_checksums` before you give the
            archive to a tool that does, such as ``unzip``.

        Examples:
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), nested=TensorDict(b=torch.ones(3)))
            >>> with tempfile.TemporaryDirectory() as path:
            ...     _ = td.memmap(path)
            ...     loaded = TensorDict.load_memmap(path)
            ...     print((loaded == td).all(), type(loaded["a"]).__name__)
            ...     # load one nested tensordict
            ...     print(TensorDict.load_memmap(path, subpath="nested")["b"])
            ...     # read the structure without reading the data
            ...     print(TensorDict.load_memmap(path, device="meta")["a"].device)
            True MemoryMappedTensor
            MemoryMappedTensor([1., 1., 1.])
            meta

            With a ``FakeTensorMode`` active, the leaves are fake tensors and
            no file is read:

            >>> from torch._subclasses import FakeTensorMode
            >>> with tempfile.TemporaryDirectory() as path:
            ...     _ = td.memmap(path)
            ...     with FakeTensorMode():
            ...         print(type(TensorDict.load_memmap(path)["a"]).__name__)
            FakeTensor

        """
        if mode not in (None, "r", "r+"):
            raise ValueError(f"mode must be 'r', 'r+' or None, got {mode!r}.")
        if allow_pickle is not None and not isinstance(allow_pickle, bool):
            raise TypeError("allow_pickle must be a bool or None.")
        metadata = None
        if not isinstance(prefix, _ArchivePath):
            # nested (recursive) calls pass _ArchivePath instances directly
            prefix = Path(prefix)
            if subpath is None:
                # The common case is a directory: read its metadata first and
                # look for an archive file only if that fails. This saves a
                # stat per (nested) tensordict, a metadata-server round trip
                # on network file systems.
                try:
                    metadata = _load_metadata(prefix)
                except OSError:
                    pass
            if metadata is None and prefix.is_file():
                if not is_memmap_archive(prefix):
                    raise ValueError(
                        f"{prefix} is a file but not a memory-mapped tensordict "
                        f"archive (expected a zip file with a top-level meta.json "
                        f"entry)."
                    )
                prefix = _ArchivePath.root(prefix, writable=mode == "r+")
        if subpath is not None:
            if isinstance(subpath, str):
                # "/"-separated path form
                subpath = tuple(part for part in subpath.split("/") if part)
            else:
                # NestedKey form: normalize arbitrary nesting, e.g.
                # ("module", ("0", "sub")) -> ("module", "0", "sub")
                subpath = _unravel_key_to_tuple(subpath)
                if not subpath:
                    raise ValueError(
                        "subpath must be a string path or a (nested) tuple of strings."
                    )
            for part in subpath:
                effective_robust_key = _get_robust_key_setting_with_warning(
                    part, robust_key
                )
                safe_part = _encode_key_for_filesystem(
                    part, robust=effective_robust_key
                )
                if (
                    part != "_tensordict"
                    and not (prefix / safe_part).exists()
                    and (prefix / "_tensordict" / "meta.json").exists()
                    and _is_saved_tensorclass(prefix)
                ):
                    # A tensorclass saves its content under _tensordict:
                    # resolve the nested key as tensorclass.get(key) does.
                    prefix = prefix / "_tensordict"
                candidate = prefix / safe_part
                if (
                    effective_robust_key
                    and not (candidate / "meta.json").exists()
                    and _is_safe_legacy_key(part, is_collection=True)
                ):
                    legacy_candidate = prefix / part
                    if (legacy_candidate / "meta.json").exists():
                        candidate = legacy_candidate
                prefix = candidate
            if not (prefix / "meta.json").exists():
                raise ValueError(
                    f"No tensordict found under subpath {'/'.join(subpath)!r} "
                    f"(missing meta.json in {prefix})."
                )
        if num_threads > 1 and isinstance(prefix, _ArchivePath):
            # Defer decompression until a leaf is actually materialized so
            # meta-device and FakeTensor loads remain metadata-only.
            prefix.reader.schedule_compressed_prefetch(prefix.at, num_threads)

        if metadata is None:
            metadata = _load_metadata(prefix)
        type_name = metadata["_type"]
        # The identifier of the class, written since format version 1. Prefer
        # it to _type, which names the private module of built-in classes.
        class_id = metadata.get("_class")
        if device is not None:
            device = torch.device(device)
        if type_name != str(cls) and class_id != _class_id(cls):
            import tensordict

            accepted_classes = tensordict.base._ACCEPTED_CLASSES
            other_cls = None
            if class_id is not None:
                other_cls = next(
                    (c for c in accepted_classes if _class_id(c) == class_id), None
                )
            if other_cls is None:
                other_cls = next(
                    (c for c in accepted_classes if str(c) == type_name), None
                )
            if other_cls is None:
                # The saved name includes the module, which differs when the
                # class was saved from __main__ (a script or a notebook): use
                # the class load_memmap is called on if its qualname matches.
                # The module and the qualname cannot be told apart in general,
                # but __main__ is never a package and no module path contains
                # "<locals>".
                saved_name = type_name.removeprefix("<class '").removesuffix("'>")
                saved_module = saved_name.removesuffix(f".{cls.__qualname__}")
                if (
                    saved_module == saved_name
                    or saved_module.startswith("__main__.")
                    or "<locals>" in saved_module
                ):
                    raise RuntimeError(
                        f"Could not find the class {saved_name} saved in "
                        f"{prefix}. Import the module that defines it, or call "
                        f"{saved_name.rpartition('.')[2]}.load_memmap() instead "
                        f"of {cls.__qualname__}.load_memmap()."
                    )
                other_cls = cls
        else:
            other_cls = cls
        load_kwargs = {
            "device": device,
            "out": out,
            "robust_key": robust_key,
        }
        # Avoid changing the default call contract of third-party registered
        # tensor collection loaders. They only see the new private keywords
        # when the caller explicitly selects a pickle policy, or a mode for
        # the files of a directory (an archive is mapped once, above).
        if allow_pickle is not None:
            load_kwargs["allow_pickle"] = allow_pickle
        if mode is not None and not isinstance(prefix, _ArchivePath):
            load_kwargs["mode"] = mode
        out = other_cls._load_memmap(prefix, metadata, **load_kwargs)
        if (
            not non_blocking
            and device is not None
            and device.type not in ("meta", "cuda")
        ):
            out._sync_all()
        return out

    @classmethod
    def _load_memmap(
        cls,
        prefix: Path,
        metadata: dict,
        device: torch.device | None = None,
        out=None,
        *,
        robust_key,
        allow_pickle: bool | None = None,
        mode: str | None = None,
    ):
        # A class that has no memmap layout of its own (a store, a
        # PersistentTensorDict, TensorDictParams) loads a TensorDict.
        from tensordict._td import TensorDict

        return TensorDict._load_memmap(
            prefix,
            metadata,
            device,
            out,
            robust_key=robust_key,
            allow_pickle=allow_pickle,
            mode=mode,
        )

    def load_memmap_(
        self,
        prefix: str | Path,
        robust_key: bool | None = True,
        *,
        allow_pickle: bool | None = None,
        mode: str | None = None,
    ):
        """Loads a saved tensordict into the tensordict, and returns ``self``.

        Calls :meth:`~.load_memmap` with ``out=self``. The tensordict and its
        nested tensordicts are reused, but each leaf is replaced by the loaded
        tensor, so its storage is not reused. Keys that are not in the saved
        data are removed. To write the data into the existing storage, use
        ``td.update_(TensorDict.load_memmap(prefix, device=td.device))``
        instead.

        If the tensordict is memory-mapped, it is unlocked for the load, then
        memory-mapped again into the loaded directory, which locks it. If the
        data has no directory (an archive, or ``mode="r"``), the tensordict
        stays unlocked and is no longer memory-mapped.

        Args:
            prefix (str or Path): the directory, or the ``".tdz"`` archive
                file, to read.
            robust_key (bool, optional): the key encoding used when the data
                was saved. See :meth:`~.memmap`. Defaults to ``True``.

        Keyword Args:
            allow_pickle (bool, optional): whether non-tensor data saved with
                :mod:`pickle` may be loaded. Pickle can run arbitrary code: pass
                ``True`` only for data you trust. Defaults to ``None``, which
                loads pickled data with a ``FutureWarning``. Starting with
                TensorDict 0.16, the default will change to ``False``. See
                :meth:`~.load_memmap`.
            mode (str, optional): how the files are mapped: ``"r"``, ``"r+"``
                or ``None``. See :meth:`~.load_memmap`. Defaults to ``None``.

        Examples:
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> with tempfile.TemporaryDirectory() as path:
            ...     _ = TensorDict(a=torch.ones(3)).save(path)
            ...     td = TensorDict(a=torch.zeros(3), b=torch.zeros(3))
            ...     out = td.load_memmap_(path)
            ...     print(out is td, list(td.keys()), td["a"])
            True ['a'] MemoryMappedTensor([1., 1., 1.])
        """
        is_memmap = self.is_memmap()
        if is_memmap:
            self.unlock_()
        self.load_memmap(
            prefix=prefix,
            device=self.device,
            out=self,
            robust_key=robust_key,
            allow_pickle=allow_pickle,
            mode=mode,
        )
        # Without a directory (archive, lazy stack, mode="r"), memmap_() would
        # copy every leaf into memory: leave the tensordict unlocked instead.
        if is_memmap and self._memmap_prefix is not None:
            self.memmap_()
        return self

    def memmap_refresh_(self, *, allow_pickle: bool | None = None):
        """Reloads the tensordict from its :attr:`~.saved_path`, and returns ``self``.

        Use it when another process has changed the files: added or removed
        entries, or written non-tensor data. It calls :meth:`~.load_memmap_`
        on :attr:`~.saved_path`, then memory-maps the tensordict there again.

        Keyword Args:
            allow_pickle (bool, optional): whether non-tensor data saved with
                :mod:`pickle` may be loaded. See :meth:`~.load_memmap`.
                Defaults to ``None``.

        Raises:
            RuntimeError: if the tensordict is not memory-mapped in a
                directory, for example a tensordict returned by
                :meth:`~.load_memmap` (for which :meth:`~.is_memmap` returns
                ``False``) or by ``memmap()`` without a prefix.

        Examples:
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> with tempfile.TemporaryDirectory() as path:
            ...     td = TensorDict(a=torch.zeros(3)).memmap(path)
            ...     _ = TensorDict(a=torch.ones(3), b=torch.ones(3)).memmap(path)
            ...     _ = td.memmap_refresh_()
            ...     print(sorted(td.keys()), td["a"])
            ['a', 'b'] MemoryMappedTensor([1., 1., 1.])
        """
        if not self.is_memmap() or self._memmap_prefix is None:
            raise RuntimeError(
                "Cannot refresh a TensorDict that is not memory mapped or has no path associated."
            )
        return self.load_memmap_(
            prefix=self.saved_path,
            allow_pickle=allow_pickle,
        )

    def _to_consolidated(
        self, *, device, pin_memory, num_threads, non_blocking, inplace
    ):
        if num_threads is None:
            # unspecified num_threads should mean 0
            num_threads = 0
        consolidated = self._consolidated
        storage = consolidated["storage"]
        if pin_memory:
            storage = storage.pin_memory()
        storage_cast = storage.to(device, non_blocking=True)
        untyped_storage = storage_cast.untyped_storage()

        def set_(x):
            if x.is_nested:
                from torch._subclasses.fake_tensor import FakeTensor
                from torch._subclasses.functional_tensor import FunctionalTensor
                from torch.nested._internal.nested_tensor import (
                    _tensor_symint_registry,
                    NestedTensor,
                )
                from torch.nested._internal.ops import extract_kwargs

                if x.layout != torch.jagged:
                    raise RuntimeError(
                        "to(device) with nested tensors that do not have a jagged layout is not implemented yet. "
                        "Please raise an issue on GitHub."
                    )
                kwargs = extract_kwargs(x)
                values = x._values
                lengths = x._lengths
                offsets = x._offsets
                kwargs["offsets"] = set_(offsets)
                if lengths is not None:
                    kwargs["lengths"] = set_(lengths)
                    ragged_source = lengths
                else:
                    ragged_source = offsets
                new_thing = kwargs.get("lengths", kwargs.get("offsets"))
                if isinstance(new_thing, (FakeTensor, FunctionalTensor)):
                    from torch._subclasses.functional_tensor import (
                        mb_unwrap_functional_tensor,
                    )

                    # Temporary hack until we have the union find
                    tgt = mb_unwrap_functional_tensor(new_thing)
                    src = mb_unwrap_functional_tensor(ragged_source)
                    tgt.nested_int_memo = src.nested_int_memo
                elif new_thing is not None:
                    _tensor_symint_registry[new_thing] = _tensor_symint_registry[
                        ragged_source
                    ]

                return NestedTensor(
                    set_(values),
                    **kwargs,
                )
            storage_offset = x.storage_offset()
            stride = x.stride()
            return x.new_empty(0, device=device).set_(
                untyped_storage,
                size=x.shape,
                stride=stride,
                storage_offset=storage_offset,
            )

        if inplace:
            # A consolidated tensordict is locked. Every leaf moves to the new
            # storage, so the keys and the consolidation stay intact: unlock
            # it for the move, and lock it again below.
            self.unlock_()
            out = self
        else:
            out = None

        result = self._fast_apply(
            set_,
            device=torch.device(device),
            num_threads=num_threads,
            out=out,
            checked=True,
        )
        if inplace:
            # A lazy stack returns a new stack of its rebound tensordicts
            result = self
        result._consolidated = {"storage": storage_cast}
        if "metadata" in consolidated:
            # faster than deepcopy
            def copy_dict(d):
                return {
                    k: v if not isinstance(v, dict) else copy_dict(v)
                    for k, v in d.items()
                }

            result._consolidated["metadata"] = copy_dict(consolidated["metadata"])
        # Ensure the result remains locked to maintain consolidated state integrity
        result.lock_()
        if non_blocking in (False, None):
            # meta tensors hold no data, so the meta device needs no sync either
            if device.type not in ("cpu", "meta") and non_blocking is False:
                # sending to non-cpu device force sync
                non_cpu_device = device
            elif storage.device.type not in ("cpu", "meta"):
                # sending from non-cpu device: need sync unless intentionally not asked for
                non_cpu_device = storage.device.type
            else:
                non_cpu_device = None
            if non_cpu_device is not None:
                device_type = _get_available_device_type()
                device_module = _get_device_module(device_type)
                if device_type == "cuda" and hasattr(device_module, "current_stream"):
                    stream = device_module.current_stream(non_cpu_device)
                    _sync_cuda_transfer(stream)
                elif hasattr(device_module, "current_stream"):
                    device_module.current_stream(non_cpu_device).synchronize()
                else:
                    # Some device modules, such as torch.mps, don't have current_stream attr
                    device_module.synchronize()

        return result


def _consolidated_metadata_json(metadata: dict) -> bytes:
    """Encodes the metadata of a consolidated tensordict as the json bytes of its file."""
    try:
        from tensordict._utils_key_json import json_dumps

        metadata_json = json_dumps(_tensorclass_names_in_metadata(metadata))
    except TypeError as e:
        raise RuntimeError(
            "Failed to convert the metadata to json. "
            "This is usually due to a nested class that is unaccounted for by the serializer, "
            "such as custom TensorClass. "
            "If you encounter this error, please file an issue on github."
        ) from e
    if isinstance(metadata_json, str):
        metadata_json = metadata_json.encode("utf-8")
    return metadata_json


def _tensorclass_names_in_metadata(metadata: dict) -> dict:
    """Returns a copy of consolidated ``metadata`` that records each tensorclass type as ``str(cls)``, as ``meta.json`` does."""
    result = dict(metadata)
    cls = result["cls"]
    if isinstance(cls, type) and _is_tensorclass(cls):
        result["cls"] = str(cls)
    for key, value in metadata.items():
        if key not in ("cls_metadata", "non_tensors", "leaves") and isinstance(
            value, dict
        ):
            result[key] = _tensorclass_names_in_metadata(value)
    return result


def _file_id(filename) -> tuple[int, int, int] | None:
    """Identifies the file at ``filename``, or returns ``None`` if there is none.

    A consolidated tensordict only rewrites the metadata of the file it was
    written to: the path may since name another file, or no file at all.
    """
    try:
        stat = os.stat(filename)
    except OSError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size


def _write_consolidated_metadata(filename, metadata: dict, data_size: int) -> None:
    """Rewrites the metadata that follows the ``data_size`` bytes of leaves in a consolidated file."""
    metadata_json = _consolidated_metadata_json(metadata)
    # Pad a shorter json with spaces rather than shrink the file, which the
    # tensordict maps
    metadata_json = metadata_json.ljust(os.path.getsize(filename) - data_size - 8)
    len_metadata = torch.tensor([len(metadata_json)], dtype=torch.int64)
    with open(filename, "r+b") as f:
        f.seek(data_size)
        f.write(metadata_json)
        f.write(len_metadata.view(torch.uint8).numpy().tobytes())


def _update_consolidated_metadata(td, metadata: dict) -> None:
    """Writes the batch sizes and names of ``td``'s tree into its consolidated ``metadata``, in place."""
    cls_metadata = metadata["cls_metadata"]
    if "stack_dim" in cls_metadata:
        # a lazy stack records its members under "0", "1", ...
        cls_metadata["stack_dim_name"] = td._td_dim_name

        def get(key):
            return td.tensordicts[int(key)]

    else:
        cls_metadata["batch_size"] = list(td.batch_size)
        cls_metadata["names"] = td.names

        def get(key):
            return td._get_str(key, NO_DEFAULT)

    non_tensors = metadata["non_tensors"]
    for key, (data, _, device) in list(non_tensors.items()):
        non_tensors[key] = (data, list(get(key).batch_size), device)
    for key, value in metadata.items():
        if key in ("cls_metadata", "non_tensors", "leaves") or not isinstance(
            value, dict
        ):
            continue
        _update_consolidated_metadata(get(key), value)
