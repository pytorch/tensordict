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
from tensordict.base import (
    _is_tensor_collection,
    _load_metadata,
    _NESTED_TENSORS_AS_LISTS_NONTENSOR,
    _sync_cuda_transfer,
    _unflatten_state_dict,
    is_compiling,
    is_tensor_collection,
    NO_DEFAULT,
    Self,
)
from tensordict.memmap import MemoryMappedTensor
from tensordict.utils import (
    _DTYPE_TO_STR_DTYPE,
    _encode_key_for_filesystem,
    _get_robust_key_setting_with_warning,
    _get_shared_executor,
    _is_non_tensor,
    _is_safe_legacy_key,
    _is_unbatched,
    _prefix_last_key,
    _rebuild_njt_from_njt,
    _unravel_key_to_tuple,
    _zip_strict,
    TensorDictFuture,
    unravel_key,
)
from torch._utils import _get_available_device_type, _get_device_module

if TYPE_CHECKING:
    from tensordict.base import TensorDictBase


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
        """Produces a state_dict from the tensordict.

        The state-dict is flat by default (dot-separated keys), following the
        convention of :meth:`torch.nn.Module.state_dict`. Set ``flatten=False``
        for a nested structure.

        Metadata (batch_size, device) is stored in a ``_metadata`` attribute on
        the returned OrderedDict, following the same convention as
        :meth:`torch.nn.Module.state_dict`. In flat mode, metadata for each
        nesting level is stored under its dot-separated prefix (``""`` for root,
        ``"sub"`` for a nested tensordict at key ``"sub"``, etc.). In nested
        mode, each nested OrderedDict carries its own ``_metadata``.

        Args:
            destination (dict, optional): If provided, the state of tensordict will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (str, optional): a prefix added to tensor
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (bool, optional): by default the :class:`torch.Tensor` items
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.
            flatten (bool, optional): whether the structure should be flattened
                with the ``"."`` character or not.
                Defaults to ``True``.

        Examples:
            >>> data = TensorDict({"1": 1, "2": 2, "3": {"3": 3}}, [])
            >>> sd = data.state_dict()
            >>> print(sd)
            OrderedDict([('1', tensor(1)), ('2', tensor(2)), ('3.3', tensor(3))])
            >>> print(sd._metadata)
            OrderedDict([('', {'batch_size': torch.Size([]), 'device': None}), ('3', {'batch_size': torch.Size([]), 'device': None})])

        """
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
        """Loads a state-dict, formatted as in :meth:`~.state_dict`, into the tensordict.

        Supports the flat format (with ``_metadata``, the default output of
        :meth:`state_dict`), the nested format (with per-level ``_metadata``),
        and the legacy format (with ``__batch_size``/``__device`` sentinel
        keys).

        When ``from_flatten`` is ``None`` (the default), the format is
        auto-detected: if ``_metadata`` is present and the state_dict keys
        don't match this tensordict's keys, the state_dict is unflattened
        before loading.

        Args:
            state_dict (OrderedDict): the state_dict of to be copied.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this tensordict's
                :meth:`torch.nn.Module.state_dict` function. Default: ``True``
            assign (bool, optional): whether to assign items in the state
                dictionary to their corresponding keys in the tensordict instead
                of copying them inplace into the tensordict's current tensors.
                When ``False``, the properties of the tensors in the current
                module are preserved while when ``True``, the properties of the
                Tensors in the state dict are preserved.
                Default: ``False``
            from_flatten (bool, optional): if ``True``, the input state_dict is
                assumed to be flattened and will be unflattened before loading.
                If ``None`` (default), auto-detects based on ``_metadata`` and
                key comparison.

        Examples:
            >>> data = TensorDict({"1": 1, "2": 2, "3": {"3": 3}}, [])
            >>> data_zeroed = TensorDict({"1": 0, "2": 0, "3": {"3": 0}}, [])
            >>> sd = data.state_dict()
            >>> data_zeroed.load_state_dict(sd)
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
        """Checks if tensordict is memory-mapped.

        If a TensorDict instance is memory-mapped, it is locked (entries cannot
        be renamed, removed or added). If a ``TensorDict`` is created with
        tensors that are all memory-mapped, this does __not__ mean that ``is_memmap``
        will return ``True`` (as a new tensor may or may not be memory-mapped).
        Only if one calls `tensordict.memmap_()` will the tensordict be
        considered as memory-mapped.

        This is always ``True`` for tensordicts on a CUDA device.

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
        """Consolidates the tensordict content in a single storage for fast serialization.

        Args:
            filename (Path, optional): an optional file path for a memory-mapped tensor
                to use as a storage for the tensordict.

        Keyword Args:
            num_threads (integer, optional): the number of threads to use for populating
                the storage. The leaves are copied by contiguous chunks of
                roughly equal byte size, one chunk per thread. When writing to
                a memory-mapped file with all leaves already on the target
                device, ``num_threads`` is ignored and a single fused copy is
                used instead, as concurrent writes to a fresh file mapping are
                slower than a sequential one on most filesystems.
            device (torch.device, optional): an optional device where the storage must be
                instantiated.
            non_blocking (bool, optional): ``non_blocking`` argument passed to :meth:`~torch.Tensor.copy_`.
            inplace (bool, optional): if ``True``, the resulting tensordict is the same
                as ``self`` with updated values. Defaults to ``False``.
            return_early (bool, optional): if ``True`` and ``num_threads>1``,
                the method will return a :class:`~tensordict.utils.TensorDictFuture`
                while the copies keep running in the background. The consolidated
                tensordict can be queried using ``future.result()``. Incompatible
                with ``use_buffer=True`` and ``non_blocking=True``. Defaults to
                ``False``.
            use_buffer (bool, optional): if ``True`` and a filename is passed, an intermediate
                local buffer will be created in shared memory, and the data will be copied at
                the storage location as a last step. This may be faster than writing directly
                to a distant physical memory (e.g., NFS).
                Defaults to ``False``.
            share_memory (bool, optional): if ``True``, the storage will be placed in shared memory.
                Defaults to ``False``.
            pin_memory (bool, optional): whether the consolidated data should be placed in pinned
                memory. Defaults to ``False``.
            metadata (bool, optional): if ``True``, the metadata will be stored alongisde the
                common storage. If a filename is provided, this is without effect.
                Storing the metadata can be useful when one wants to control how serialization
                is achieved, as TensorDict handles the pickling/unpickling of consolidated TDs
                differently if the metadata is or isn't available.

        .. note::
            If the tensordict is already consolidated, all arguments are ignored and ``self``
            is returned. Call :meth:`~.contiguous` to re-consolidate.

        Examples:
            >>> import pickle
            >>> import tempfile
            >>> import torch
            >>> import tqdm
            >>> from torch.utils.benchmark import Timer
            >>> from tensordict import TensorDict
            >>> data = TensorDict({"a": torch.zeros(()), "b": {"c": torch.zeros(())}})
            >>> data_consolidated = data.consolidate()
            >>> # check that the data has a single data_ptr()
            >>> assert torch.tensor([
            ...     v.untyped_storage().data_ptr() for v in data_consolidated.values(True, True)
            ... ]).unique().numel() == 1
            >>> # Serializing the tensordict will be faster with data_consolidated
            >>> with open("data.pickle", "wb") as f:
            ...    print("regular", Timer("pickle.dump(data, f)", globals=globals()).adaptive_autorange())
            >>> with open("data_c.pickle", "wb") as f:
            ...     print("consolidated", Timer("pickle.dump(data_consolidated, f)", globals=globals()).adaptive_autorange())


        """
        if self.is_consolidated():
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
            # Convert the dict to json
            try:
                from tensordict.utils import json_dumps

                metadata_dict_json = json_dumps(metadata_dict)
            except TypeError as e:
                raise RuntimeError(
                    "Failed to convert the metatdata to json. "
                    "This is usually due to a nested class that is unaccounted for by the serializer, "
                    "such as custom TensorClass. "
                    "If you encounter this error, please file an issue on github."
                ) from e
            # Represent as a tensor
            if isinstance(metadata_dict_json, str):
                metadata_dict_json = metadata_dict_json.encode("utf-8")
            metadata_dict_json = torch.as_tensor(
                bytearray(metadata_dict_json), dtype=torch.uint8
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
                    if (stride and stride[-1] != 1) or v.storage_offset():
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
                        flat_view = v.contiguous().view(-1).view(torch.uint8)
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
                elif (stride and stride[-1] != 1) or v.storage_offset():
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
            # with open(Path(filename).with_suffix(".json"), "wb") as f:
            #     metadata_dict["size"] = filesize
            #     f.write(json.dumps(metadata_dict))
        if consolidate_futures is not None:
            return TensorDictFuture(consolidate_futures, result)
        return result

    @classmethod
    def from_consolidated(cls, filename):
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
        """Checks if a TensorDict has a consolidated storage."""
        return hasattr(self, "_consolidated")

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
    ) -> Self:
        """Writes all tensors onto a corresponding memory-mapped Tensor, in-place.

        Args:
            prefix (str): directory prefix where the memory-mapped tensors will
                be stored. The directory tree structure will mimic the tensordict's.
            copy_existing (bool): If False (default), an exception will be raised if an
                entry in the tensordict is already a tensor stored on disk
                with an associated file, but is not saved in the correct
                location according to prefix.
                If ``True``, any existing Tensor will be copied to the new location.

        Keyword Args:
            num_threads (int, optional): the number of threads used to write the memmap
                tensors. Defaults to `0`.
            return_early (bool, optional): if ``True`` and ``num_threads>0``,
                the method will return a future of the tensordict. The resulting
                tensordict can be queried using `future.result()`.
            share_non_tensor (bool, optional): if ``True``, the non-tensor data will be
                shared between the processes and writing operation (such as inplace update
                or set) on any of the workers within a single node will update the value
                on all other workers. If the number of non-tensor leaves is high (e.g.,
                sharing large stacks of non-tensor data) this may result in OOM or similar
                errors. Defaults to ``False``.
            existsok (bool, optional): if ``False``, an exception will be raised if a tensor already
                exists in the same path. Defaults to ``True``.
            robust_key (bool, optional): if ``True`` (default), uses robust key encoding that safely
                handles keys with path separators and special characters. If ``False``,
                uses legacy behavior (keys used as-is). If ``None``, uses the default
                robust behavior.

        The TensorDict is then locked, meaning that any writing operations that
        isn't in-place will throw an exception (eg, rename, set or remove an
        entry).
        Once the tensordict is unlocked, the memory-mapped attribute is turned to ``False``,
        because cross-process identity is not guaranteed anymore.

        Returns:
            self if ``return_early=False``, otherwise a :class:`~tensordict.utils.TensorDictFuture` instance.

        Note:
            Serialising in this fashion might be slow with deeply nested tensordicts, so
            it is not recommended to call this method inside a training loop.
        """
        prefix = Path(prefix) if prefix is not None else self._memmap_prefix
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
    ) -> Self:
        """Saves the tensordict to disk.

        This function is a proxy to :meth:`~.memmap`.
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
    ) -> Self:
        """Writes all tensors onto a corresponding memory-mapped Tensor in a new tensordict.

        Args:
            prefix (str): directory prefix where the memory-mapped tensors will
                be stored. The directory tree structure will mimic the tensordict's.
                If ``prefix`` ends with ``".tdz"`` (or ``archive=True`` is
                passed), a single-file archive is written instead of a
                directory: a standard zip file whose entries replicate the
                memmap directory layout. See ``archive`` below.
            copy_existing (bool): If False (default), an exception will be raised if an
                entry in the tensordict is already a tensor stored on disk
                with an associated file, but is not saved in the correct
                location according to prefix.
                If ``True``, any existing Tensor will be copied to the new location.

        Keyword Args:
            num_threads (int, optional): the number of threads used to write the memmap
                tensors. Defaults to `0`.
            return_early (bool, optional): if ``True`` and ``num_threads>0``,
                the method will return a future of the tensordict.
            share_non_tensor (bool, optional): if ``True``, the non-tensor data will be
                shared between the processes and writing operation (such as inplace update
                or set) on any of the workers within a single node will update the value
                on all other workers. If the number of non_tensor leaves is high (e.g.,
                sharing large stacks of non-tensor data) this may result in OOM or similar
                errors. Defaults to ``False``.
            existsok (bool, optional): if ``False``, an exception will be raised if a tensor already
                exists in the same path. Defaults to ``True``.
            robust_key (bool, optional): if ``True`` (default), uses robust key encoding that safely
                handles keys with path separators and special characters. If ``False``,
                uses legacy behavior (keys used as-is). If ``None``, uses the default
                robust behavior.
            archive (bool, optional): if ``True``, ``prefix`` designates a
                single file rather than a directory and the tensordict is
                written as a memmap archive: a zip file mirroring the memmap
                directory tree, with tensor payloads stored uncompressed and
                aligned so that :meth:`~.load_memmap` can memory-map the file
                and expose every leaf as a zero-copy view. If ``None``
                (default), archive mode is enabled when ``prefix`` ends with
                ``".tdz"``. The result of :meth:`~.load_memmap` on an archive
                behaves like the result of :meth:`~.from_consolidated`: all
                leaves are views into a single storage, and in-place writes do
                not propagate to the file. Archives and memmap directories are
                mutually convertible with :func:`~tensordict.pack_memmap` /
                :func:`~tensordict.unpack_memmap` (or any zip tool). Note that
                archives are written sequentially (single data pass) and
                ``num_threads`` has no effect on them. An existing archive
                is replaced by a new file rather than rewritten in place: a
                symlink is followed, but other hard links to the old file
                keep the old contents.
            compression (str or int, optional): compression for archive
                entries (``"stored"``, ``"deflate"``, ``"bzip2"``, ``"lzma"``
                or a :mod:`zipfile` constant). Defaults to ``"stored"``
                (uncompressed), which is what enables zero-copy loading.
                Compressed archives load correctly but leaves are
                decompressed in memory on access. Only valid in archive mode.

        The TensorDict is then locked, meaning that any writing operations that
        isn't in-place will throw an exception (eg, rename, set or remove an
        entry).
        Once the tensordict is unlocked, the memory-mapped attribute is turned to ``False``,
        because cross-process identity is not guaranteed anymore.

        Returns:
            A new tensordict with the tensors stored on disk if ``return_early=False``,
            otherwise a :class:`~tensordict.utils.TensorDictFuture` instance.

        Note:
            Serialising in this fashion might be slow with deeply nested tensordicts, so
            it is not recommended to call this method inside a training loop.
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
        """Creates a contentless Memory-mapped tensordict with the same shapes as the original one.

        Args:
            prefix (str): directory prefix where the memory-mapped tensors will
                be stored. The directory tree structure will mimic the tensordict's.
                If ``prefix`` ends with ``".tdz"`` (or ``archive=True`` is
                passed), a preallocated single-file archive is created
                instead. See ``archive`` below.
            copy_existing (bool): If False (default), an exception will be raised if an
                entry in the tensordict is already a tensor stored on disk
                with an associated file, but is not saved in the correct
                location according to prefix.
                If ``True``, any existing Tensor will be copied to the new location.

        Keyword Args:
            num_threads (int, optional): the number of threads used to write the memmap
                tensors. Defaults to `0`.
            return_early (bool, optional): if ``True`` and ``num_threads>0``,
                the method will return a future of the tensordict.
            share_non_tensor (bool, optional): if ``True``, the non-tensor data will be
                shared between the processes and writing operation (such as inplace update
                or set) on any of the workers within a single node will update the value
                on all other workers. If the number of non-tensor leaves is high (e.g.,
                sharing large stacks of non-tensor data) this may result in OOM or similar
                errors. Defaults to ``False``.
            existsok (bool, optional): if ``False``, an exception will be raised if a tensor already
                exists in the same path. Defaults to ``True``.
            robust_key (bool, optional): if ``True`` (default), uses robust key encoding that safely
                handles keys with path separators and special characters. If ``False``,
                uses legacy behavior (keys used as-is). If ``None``, uses the default
                robust behavior.
            archive (bool, optional): if ``True``, ``prefix`` designates a
                single file and a preallocated, zero-filled memmap archive is
                created and loaded back with
                ``load_memmap(prefix, mode="r+")``: the returned tensordict
                writes through to the archive. If ``None`` (default), archive
                mode is enabled when ``prefix`` ends with ``".tdz"``.
                In-place writes leave the zip per-entry checksums stale; call
                :func:`~tensordict.refresh_archive_checksums` before handing
                the archive to tools that verify them. Nested tensors are not
                supported in this mode.

        The TensorDict is then locked, meaning that any writing operations that
        isn't in-place will throw an exception (eg, rename, set or remove an
        entry).
        Once the tensordict is unlocked, the memory-mapped attribute is turned to ``False``,
        because cross-process identity is not guaranteed anymore.

        Returns:
            A new ``TensorDict`` instance with data stored as memory-mapped tensors if ``return_early=False``,
            otherwise a :class:`~tensordict.utils.TensorDictFuture` instance.

        .. note::
            This is the recommended method to write a set of large buffers
            on disk, as :meth:`~.memmap_()` will copy the information, which can
            be slow for large content.

        Examples:
            >>> td = TensorDict({
            ...     "a": torch.zeros((3, 64, 64), dtype=torch.uint8),
            ...     "b": torch.zeros(1, dtype=torch.int64),
            ... }, batch_size=[]).expand(1_000_000)  # expand does not allocate new memory
            >>> buffer = td.memmap_like("/path/to/dataset")

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
        prefix = Path(prefix) if prefix is not None else self._memmap_prefix
        if num_threads > 1:
            executor = _get_shared_executor(num_threads)
            futures = []

            # we create an empty copy of self
            # This is because calling MMapTensor.from_tensor(mmap_tensor) does nothing
            # if both are in filesystem
            def empty(x):
                return torch.empty((), device=x.device, dtype=x.dtype).expand(x.shape)

            input = self.apply(empty)
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
                return result
            return TensorDictFuture(futures, result)

        def empty_expand(x):
            return torch.empty((), device=x.device, dtype=x.dtype).expand(x.shape)

        input = self.apply(empty_expand)
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
        """Loads a tensordict from disk.

        This class method is a proxy to :meth:`~.load_memmap`.
        """
        return cls.load_memmap(prefix, *args, **kwargs)

    @deprecated("TensorDictBase.load_()", removal="0.17", replacement="load_memmap_()")
    def load_(self, prefix: str | Path, *args, **kwargs):
        """Loads a tensordict from disk within the current tensordict.

        This class method is a proxy to :meth:`~.load_memmap_`.

        .. deprecated:: 0.15
            Use :meth:`~.load_memmap_` instead.
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
        """Loads a memory-mapped tensordict from disk.

        Args:
            prefix (str or Path to folder): the path to the folder where the
                saved tensordict should be fetched, or the path to a memmap
                archive file written through
                ``save(..., archive=True)`` / a ``".tdz"`` prefix (or packed
                with :func:`~tensordict.pack_memmap`). Archives are
                memory-mapped once and every leaf is exposed as a zero-copy
                view into the mapping: only the pages of the leaves that are
                actually accessed are read from disk. Unlike directory-backed
                tensordicts, in-place writes to the leaves of an
                archive-loaded tensordict do not propagate to the file by
                default (see ``mode``).
            device (torch.device or equivalent, optional): if provided, the
                data will be asynchronously cast to that device.
                Supports `"meta"` device, in which case the data isn't loaded
                but a set of empty "meta" tensors are created. This is
                useful to get a sense of the total model size and structure
                without actually opening any file.
            non_blocking (bool, optional): if ``True``, synchronize won't be
                called after loading tensors on device. Defaults to ``False``.
            out (TensorDictBase, optional): optional tensordict where the data
                should be loaded. Its nested containers are reused, but its
                leaves are rebound to the loaded tensors, so their storage is
                not reused. Keys of ``out`` that are missing from the saved
                data are removed. If ``out`` has a device, the data is loaded
                on it, and a different ``device`` raises a ``ValueError``.
                To write the data into the preallocated storage of ``out``,
                use ``out.update_(TensorDict.load_memmap(prefix, device=device))``
                instead.
            robust_key (bool, optional): if ``True`` (default), expects robust key encoding was used
                when saving and decodes filenames accordingly. If ``False``, uses legacy
                behavior. If ``None``, uses the default robust behavior.
            subpath (NestedKey or str path, optional): the location of a
                nested tensordict to load, as a nested key (e.g.
                ``("module", "0")``, with arbitrary nesting allowed as usual)
                or as a ``"/"``-separated string path (e.g. ``"module/0"``).
                Only that subtree is loaded. Works both for directories
                (equivalent to appending the path to ``prefix``) and for
                archives.
            mode (str, optional): how the files are memory-mapped. With
                ``"r"``, the mapping is copy-on-write: in-place writes to the
                leaves stay in memory. They reach the files only if the
                tensordict is saved there, and :meth:`~.memmap_` without a
                prefix does not write back to ``prefix``. With ``"r+"``, the
                mapping is shared: in-place writes propagate to the files,
                and a file that is not writable raises a
                :class:`PermissionError`. Defaults to ``None``, which maps
                archives as with ``"r"``, and each file of a directory as with
                ``"r+"`` if the process can write it, as with ``"r"``
                otherwise. Prefer ``"r"`` for data that is only read: on some
                network file systems (e.g. Lustre), page faults on a shared
                writable mapping take write locks, so readers on different
                nodes block each other. On Linux, copy-on-write mappings count
                against the memory commit limit, so a file larger than the
                available RAM and swap can fail to map with ``"r"``. With
                archives, ``"r+"`` requires uncompressed, aligned tensor
                payloads (i.e. archives written by tensordict without
                ``compression``) and is not available for nested-tensor
                leaves. In-place writes do not update the per-entry CRC-32
                stored by the zip format; :meth:`~.load_memmap` ignores
                checksums, but call
                :func:`~tensordict.refresh_archive_checksums` before handing
                a modified archive to tools that verify them (``unzip``,
                :func:`~tensordict.unpack_memmap`, ...).
            num_threads (int, optional): number of threads used to decompress
                the leaves of a compressed archive (deflate entries are
                inflated in parallel, which scales nearly linearly). Without
                compression, loading is a metadata-only operation and this
                argument has no effect. Defaults to ``0`` (sequential).
            allow_pickle (bool, optional): whether pickled non-tensor fields
                may be loaded. Pickle can execute arbitrary code, so pass
                ``True`` only for data from a trusted source and ``False``
                for untrusted data. During the 0.14 compatibility window,
                omitting this option loads pickle with a ``FutureWarning``;
                the default will change to ``False`` in 0.15. Saves without
                a pickle sidecar do not require this option.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict.fromkeys(["a", "b", "c", ("nested", "e")], 0)
            >>> td.memmap("./saved_td")
            >>> td_load = TensorDict.load_memmap("./saved_td")
            >>> assert (td == td_load).all()

        This method also allows loading nested tensordicts.

        Examples:
            >>> nested = TensorDict.load_memmap("./saved_td/nested")
            >>> assert nested["e"] == 0

        A tensordict can also be loaded on "meta" device or, alternatively,
        as a fake tensor.

        Examples:
            >>> import tempfile
            >>> td = TensorDict({"a": torch.zeros(()), "b": {"c": torch.zeros(())}})
            >>> with tempfile.TemporaryDirectory() as path:
            ...     td.save(path)
            ...     td_load = TensorDict.load_memmap(path, device="meta")
            ...     print("meta:", td_load)
            ...     from torch._subclasses import FakeTensorMode
            ...     with FakeTensorMode():
            ...         td_load = TensorDict.load_memmap(path)
            ...         print("fake:", td_load)
            meta: TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=meta, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=meta, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=meta,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=meta,
                is_shared=False)
            fake: TensorDict(
                fields={
                    a: FakeTensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: FakeTensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=cpu,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False)

        """
        if mode not in (None, "r", "r+"):
            raise ValueError(f"mode must be 'r', 'r+' or None, got {mode!r}.")
        if allow_pickle is not None and not isinstance(allow_pickle, bool):
            raise TypeError("allow_pickle must be a bool or None.")
        if not isinstance(prefix, _ArchivePath):
            # nested (recursive) calls pass _ArchivePath instances directly
            prefix = Path(prefix)
            if prefix.is_file():
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

        metadata = _load_metadata(prefix)
        type_name = metadata["_type"]
        if device is not None:
            device = torch.device(device)
        if type_name != str(cls):
            import tensordict

            for other_cls in tensordict.base._ACCEPTED_CLASSES:
                if str(other_cls) == type_name:
                    break
            else:
                raise RuntimeError(
                    f"Could not find name {type_name} in {tensordict.base._ACCEPTED_CLASSES}. "
                    f"Did you call _register_tensor_class(cls) on {type_name}?"
                )
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

    def load_memmap_(
        self,
        prefix: str | Path,
        robust_key: bool | None = True,
        *,
        allow_pickle: bool | None = None,
        mode: str | None = None,
    ):
        """Loads the content of a memory-mapped tensordict within the tensordict where ``load_memmap_`` is called.

        The tensordict and its nested containers are reused, but the leaves
        are rebound to the loaded tensors, so their storage is not reused.
        Keys that are missing from the saved data are removed. To write the
        data into the existing storage, use
        ``td.update_(TensorDict.load_memmap(prefix, device=td.device))``
        instead.

        See :meth:`~tensordict.TensorDictBase.load_memmap` for more info.
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
        """Refreshes the content of the memory-mapped tensordict if it has a :attr:`~tensordict.TensorDict.saved_path`.

        This method will raise an exception if no path is associated with it.

        Args:
            allow_pickle (bool, optional): whether pickled non-tensor fields
                may be loaded. See :meth:`~.load_memmap`.

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
        storage = self._consolidated["storage"]
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
        result._consolidated = {"storage": storage_cast}
        if "metadata" in self._consolidated:
            # faster than deepcopy
            def copy_dict(d):
                return {
                    k: v if not isinstance(v, dict) else copy_dict(v)
                    for k, v in d.items()
                }

            result._consolidated["metadata"] = copy_dict(self._consolidated["metadata"])
        # Ensure the result remains locked to maintain consolidated state integrity
        result.lock_()
        if non_blocking in (False, None):
            if device.type != "cpu" and non_blocking is False:
                # sending to non-cpu device force sync
                non_cpu_device = device
            elif storage.device.type != "cpu":
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
