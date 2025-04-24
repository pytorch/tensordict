# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent tensordicts (H5 and others)."""
from __future__ import annotations

import importlib
import os

import tempfile
import warnings
import weakref
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Tuple, Type

import numpy as np

import torch

from tensordict._td import (
    _TensorDictKeysView,
    _unravel_key_to_tuple,
    CompatibleType,
    NO_DEFAULT,
    TensorDict,
)
from tensordict.base import (
    _default_is_leaf,
    _is_leaf_nontensor,
    is_tensor_collection,
    T,
    TensorDictBase,
)
from tensordict.memmap import MemoryMappedTensor
from tensordict.utils import (
    _as_context_manager,
    _CloudpickleWrapper,
    _KEY_ERROR,
    _LOCK_ERROR,
    _parse_to,
    _proc_init,
    _split_tensordict,
    _zip_strict,
    cache,
    erase_cache,
    expand_right,
    IndexType,
    is_non_tensor,
    lock_blocked,
    NestedKey,
    NUMPY_TO_TORCH_DTYPE_DICT,
    unravel_key,
)
from torch import multiprocessing as mp

try:
    import orjson as json
except ImportError:
    # Fallback for 3.13
    import json

_has_h5 = importlib.util.find_spec("h5py", None) is not None


class _Visitor:
    def __init__(self, fun=None):
        self.elts = []
        self.fun = fun

    def __call__(self, name):
        self.elts.append(name)

    def __iter__(self):
        if self.fun is None:
            yield from self.elts
        else:
            for elt in self.elts:
                yield self.fun(elt)


class _PersistentTDKeysView(_TensorDictKeysView):
    def __iter__(self):
        # For consistency with tensordict where currently a non-tensor is stored in a
        # tensorclass and hence can be seen as a nested tensordict
        # that situation should be clarified
        read_non_tensor = self.is_leaf is _is_leaf_nontensor or not self.leaves_only
        if self.include_nested:
            visitor = _Visitor(lambda key: unravel_key(tuple(key.split("/"))))
            self.tensordict.file.visit(visitor)
        else:
            visitor = self.tensordict.file.keys()
        for key in visitor:
            metadata = self.tensordict._get_metadata(key)
            if metadata.get("non_tensor"):
                if read_non_tensor:
                    yield key
                else:
                    continue
            elif metadata.get("array"):
                yield key
            elif not self.leaves_only and (
                not isinstance(key, tuple) or self.include_nested
            ):
                yield key

    def __contains__(self, key):
        key = unravel_key(key)
        return key in list(self)


class PersistentTensorDict(TensorDictBase):
    """Persistent TensorDict implementation.

    :class:`PersistentTensorDict` instances provide an interface with data stored
    on disk such that access to this data is made easy while still taking advantage
    from the fast access provided by the backend.

    Like other :class:`TensorDictBase` subclasses, :class:`PersistentTensorDict`
    has a ``device`` attribute. This does *not* mean that the data is being stored
    on that device, but rather that when loaded, the data will be cast onto
    the desired device.

    Keyword Args:
        batch_size (torch.Size or compatible): the tensordict batch size.
            Defaults to ``torch.Size(())``.
        filename (str, optional): the path to the h5 file. Exclusive with ``group``.
        group (h5py.Group, optional): a file or a group that contains data. Exclusive with ``filename``.
        mode (str, optional): Reading mode. Defaults to ``"r"``.
        backend (str, optional): storage backend. Currently only ``"h5"`` is supported.
        device (torch.device or compatible, optional): device of the tensordict.
            Defaults to ``None`` (ie. default PyTorch device).
        **kwargs: kwargs to be passed to :meth:`h5py.File.create_dataset`.

    .. note::
        Currently, PersistentTensorDict instances are not closed when getting out-of-scope.
        This means that it is the responsibility of the user to close them if necessary.

    Examples:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     data = PersistentTensorDict(file=f, batch_size=[3], mode="w")
        ...     data["a", "b"] = torch.randn(3, 4)
        ...     print(data)

    """

    _td_dim_names = None
    LOCKING = None

    def __init__(
        self,
        *,
        batch_size=None,
        filename=None,
        group=None,
        mode="r",
        backend="h5",
        device=None,
        **kwargs,
    ):
        if batch_size is None:
            batch_size = torch.Size(())
        self._locked_tensordicts = []
        self._lock_id = set()
        if not _has_h5:
            raise ModuleNotFoundError("Could not load h5py.")
        import h5py

        super().__init__()
        self.filename = filename
        self.mode = mode
        if backend != "h5":
            raise NotImplementedError
        if filename is not None and group is None:
            self.file = h5py.File(filename, mode, locking=self.LOCKING)
        elif group is not None:
            self.file = group
        else:
            raise RuntimeError(
                f"Either group or filename must be provided, and not both. Got group={group} and filename={filename}."
            )
        self._batch_size = torch.Size(batch_size)
        self._device = torch.device(device) if device is not None else None
        self._is_shared = False
        self._is_memmap = False
        self.kwargs = kwargs

        # we use this to allow nested tensordicts to have a different batch-size
        self._nested_tensordicts = {}
        self._pin_mem = False

        # this must be kept last
        self._check_batch_size(self._batch_size)

    @classmethod
    def from_h5(cls, filename, *, mode="r", batch_size: torch.size | None = None):
        """Creates a PersistentTensorDict from a h5 file.

        This function will automatically determine the batch-size for each nested tensordict (unless ``batch_size``
        is provided).

        Args:
            filename (str): The path to the h5 file.

        Keyword Args:
            mode (str, optional): Reading mode. Defaults to ``"r"``.
            batch_size (torch.Size, optional): The batch size of the TensorDict. Defaults to None (batch-size automatically
                determined).

        Returns:
            A PersistentTensorDict representation of the input h5 file.

        Examples:
            >>> ptd = PersistentTensorDict.from_h5("path/to/file.h5")
            >>> print(ptd)
            PersistentTensorDict(
                fields={
                    key1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                    key2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
        """
        out = cls(filename=filename, mode=mode, batch_size=batch_size)
        if batch_size is None:
            # determine batch size
            _set_max_batch_size(out)
        return out

    @classmethod
    def from_dict(
        cls,
        input_dict,
        filename,
        *,
        auto_batch_size: bool = False,
        batch_size=None,
        device=None,
        **kwargs,
    ):
        """Converts a dictionary or a TensorDict to a h5 file.

        Args:
            input_dict (dict, TensorDict or compatible): data to be stored as h5.
            filename (str or path): path to the h5 file.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, the batch size will be computed automatically.
                Defaults to ``False``.
            batch_size (tensordict batch-size, optional): if provided, batch size
                of the tensordict. If not, the batch size will be gathered from the
                input structure (if present) or determined automatically.
            device (torch.device or compatible, optional): the device where to
                expect the tensor once they are returned. Defaults to ``None``
                (on cpu by default).
            **kwargs: kwargs to be passed to :meth:`h5py.File.create_dataset`.

        Returns:
            A :class:`PersitentTensorDict` instance linked to the newly created file.

        """
        import h5py

        file = h5py.File(filename, "w", locking=cls.LOCKING)
        _has_batch_size = True
        if batch_size is None:
            if is_tensor_collection(input_dict):
                batch_size = input_dict.batch_size
            else:
                _has_batch_size = False
                batch_size = torch.Size([])

        # let's make a tensordict first
        out = cls(group=file, batch_size=batch_size, device=device, **kwargs)
        if is_tensor_collection(input_dict):
            out.update(input_dict)
        else:
            out.update(TensorDict(input_dict, batch_size=batch_size))
        if not _has_batch_size:
            _set_max_batch_size(out)
        return out

    def close(self):
        """Closes the persistent tensordict."""
        self.file.close()

    def _process_key(self, key):
        key = _unravel_key_to_tuple(key)
        return "/".join(key)

    def _check_batch_size(self, batch_size) -> None:
        for key in self.keys(include_nested=True, leaves_only=True):
            key = self._process_key(key)
            array = self.file[key]
            if _is_non_tensor_h5(array):
                continue
            size = array.shape
            if torch.Size(size[: len(batch_size)]) != batch_size:
                raise ValueError(
                    f"batch size and array size mismatch: array.shape={size}, batch_size={batch_size}."
                )

    def _get_array(self, key, default=NO_DEFAULT):
        try:
            key = self._process_key(key)
            array = self.file[key]
            return array
        except KeyError:
            if default is not NO_DEFAULT:
                return default
            raise KeyError(f"key {key} not found in PersistentTensorDict {self}")

    def _process_array(self, key, array):
        import h5py

        if isinstance(array, (h5py.Dataset,)):
            if self.device is not None:
                device = self.device
            else:
                device = torch.device("cpu")
            # we convert to an array first to avoid "Creating a tensor from a list of numpy.ndarrays is extremely slow."
            if not _is_non_tensor_h5(array):
                array = array[()]
                out = torch.as_tensor(array, device=device)
                if self._pin_mem:
                    out = out.pin_memory()
            else:
                from tensordict.tensorclass import NonTensorData

                array = array[()]
                out = NonTensorData(
                    data=array, device=device, batch_size=self.batch_size
                )
            return out
        else:
            out = self._nested_tensordicts.get(key)
            if out is None:
                out = self._nested_tensordicts[key] = PersistentTensorDict(
                    group=array,
                    batch_size=self.batch_size,
                    device=self.device,
                )
            return out

    @cache  # noqa: B019
    def _get_str(self, key: NestedKey, default, **kwargs):
        key = _unravel_key_to_tuple(key)
        array = self._get_array(key, default)
        if array is default:
            return array
        return self._process_array(key, array)

    _get_tuple = _get_str

    def get_at(
        self, key: NestedKey, idx: IndexType, default: CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        import h5py

        array = self._get_array(key, default)
        if isinstance(array, (h5py.Dataset,)):
            if self.device is not None:
                device = self.device
            else:
                device = torch.device("cpu")
            # indexing must be done before converting to tensor.
            idx = self._process_index(idx, array)
            # `get_at` is there to save us.
            try:
                out = torch.as_tensor(array[idx], device=device)
            except TypeError as err:
                if "Boolean indexing array has incompatible shape" in str(err):
                    # Known bug in h5py: cannot broadcast boolean mask on the right as
                    # done in np and torch. Therefore we put a performance warning
                    # and convert to torch tensor first.
                    warnings.warn(
                        "Indexing an h5py.Dataset object with a boolean mask "
                        "that needs broadcasting does not work directly. "
                        "tensordict will cast the entire array in memory and index it using the mask. "
                        "This is suboptimal and may lead to performance issue."
                    )
                    out = torch.as_tensor(np.asarray(array), device=device)[idx]
                else:
                    raise err
            if self._pin_mem:
                return out.pin_memory()
            return out
        elif array is not default:
            out = self._nested_tensordicts.get(key)
            if out is None:
                out = self._nested_tensordicts[key] = PersistentTensorDict(
                    group=array,
                    batch_size=self.batch_size,
                    device=self.device,
                )
            return out._get_sub_tensordict(idx)
        else:
            return default

    def _get_metadata(self, key):
        """Gets the metadata for an entry.

        This method avoids creating a tensor from scratch, and just reads the metadata of the array.
        """
        import h5py

        array = self._get_array(key)
        if (
            isinstance(array, (h5py.Dataset,))
            and array.dtype in NUMPY_TO_TORCH_DTYPE_DICT
        ):
            shape = torch.Size(array.shape)
            return {
                "dtype": NUMPY_TO_TORCH_DTYPE_DICT[array.dtype],
                "shape": shape,
                "dim": len(shape),
                "array": True,
            }
        elif (
            isinstance(array, (h5py.Dataset,))
            and array.dtype not in NUMPY_TO_TORCH_DTYPE_DICT
        ):
            return {"non_tensor": True}
        else:
            val = self.get(key)
            shape = val.shape
            return {
                "dtype": None,
                "shape": shape,
                "dim": len(shape),
                "array": False,
            }

    @classmethod
    def _process_index(cls, idx, array=None):
        if isinstance(idx, tuple):
            return tuple(cls._process_index(_idx, array) for _idx in idx)
        if isinstance(idx, torch.Tensor):
            return idx.cpu().detach().numpy()
        if isinstance(idx, (range, list)):
            return np.asarray(idx)
        return idx

    def __getitem__(self, item: IndexType) -> Any:
        if isinstance(item, str) or (
            isinstance(item, tuple) and _unravel_key_to_tuple(item)
        ):
            result = self.get(item, default=NO_DEFAULT)
            if is_non_tensor(result):
                result_data = getattr(result, "data", NO_DEFAULT)
                if result_data is NO_DEFAULT:
                    return result.tolist()
                return result_data
            return result
        if isinstance(item, list):
            # convert to tensor
            item = torch.tensor(item)
        return self._get_sub_tensordict(item)

    __getitems__ = __getitem__

    def __setitem__(self, index: IndexType, value: Any):
        index_unravel = _unravel_key_to_tuple(index)
        if index_unravel:
            return self.set(index_unravel, value, inplace=True)

        if isinstance(index, list):
            # convert to tensor
            index = torch.tensor(index)
        sub_td = self._get_sub_tensordict(index)
        err_set_batch_size = None
        if not isinstance(value, TensorDictBase):
            value = TensorDict.from_dict(value, batch_size=[])
            # try to assign the current shape. If that does not work, we can
            # try to expand
            try:
                value.batch_size = sub_td.batch_size
            except RuntimeError as err0:
                err_set_batch_size = err0
        if value.shape != sub_td.shape:
            try:
                value = value.expand(sub_td.shape)
            except RuntimeError as err:
                if err_set_batch_size is not None:
                    raise err from err_set_batch_size
                raise RuntimeError(
                    f"Cannot broadcast the tensordict {value} to the shape of the indexed persistent tensordict {self}[{index}]."
                ) from err
        sub_td.update(value, inplace=True)

    @cache  # noqa: B019
    def _valid_keys(self):
        keys = []
        for key in self.file.keys():
            metadata = self._get_metadata(key)
            if not metadata.get("non_tensor"):
                keys.append(key)
        return keys

    # @cache  # noqa: B019
    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> _PersistentTDKeysView:
        if is_leaf not in (None, _default_is_leaf, _is_leaf_nontensor):
            raise ValueError(
                f"is_leaf {is_leaf} is not supported within tensordicts of type {type(self).__name__}."
            )
        return _PersistentTDKeysView(
            tensordict=self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
            sort=sort,
        )

    def _items_metadata(self, include_nested=False, leaves_only=False):
        """Iterates over the metadata of the PersistentTensorDict."""
        for key in self.keys(include_nested, leaves_only):
            yield (key, self._get_metadata(key))

    def _values_metadata(self, include_nested=False, leaves_only=False):
        """Iterates over the metadata of the PersistentTensorDict."""
        for key in self.keys(include_nested, leaves_only):
            yield self._get_metadata(key)

    def _change_batch_size(self, value):
        raise NotImplementedError

    def _stack_onto_(
        self, list_item: list[CompatibleType], dim: int
    ) -> PersistentTensorDict:
        for key in self.keys():
            vals = [td._get_str(key, None) for td in list_item]
            if all(v is None for v in vals):
                continue
            stacked = torch.stack(vals, dim=dim)
            self.set_(key, stacked)
        return self

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        _batch_size = self._batch_size
        try:
            self._batch_size = torch.Size(value)
            self._check_batch_size(self._batch_size)
        except ValueError:
            self._batch_size = _batch_size

    _erase_names = TensorDict._erase_names
    names = TensorDict.names
    _has_names = TensorDict._has_names

    def _rename_subtds(self, names):
        if names is None:
            names = [None] * self.ndim
        for item in self._nested_tensordicts.values():
            if is_tensor_collection(item):
                td_names = list(names) + [None] * (item.ndim - self.ndim)
                item.rename_(*td_names)

    def contiguous(self):
        """Materializes a PersistentTensorDict on a regular TensorDict."""
        return self.to_tensordict()

    @lock_blocked
    def del_(self, key):
        key = self._process_key(key)
        del self.file[key]
        return self

    def detach_(self):
        # PersistentTensorDict do not carry gradients. This is a no-op
        return self

    @property
    def device(self):
        return self._device

    def empty(
        self, recurse=False, *, batch_size=None, device=NO_DEFAULT, names=None
    ) -> T:
        if recurse:
            out = self.empty(
                recurse=False, batch_size=batch_size, device=device, names=names
            )
            for key, val in self.items():
                if is_tensor_collection(val):
                    out._set_str(
                        key,
                        val.empty(
                            recurse=True,
                            batch_size=batch_size,
                            device=device,
                            names=names,
                        ),
                        inplace=False,
                        validated=True,
                        non_blocking=False,
                    )
            return out
        return TensorDict(
            {},
            device=self.device if device is NO_DEFAULT else device,
            batch_size=self.batch_size if batch_size is None else batch_size,
            names=self.names if names is None and self._has_names() else names,
        )

    def _propagate_lock(self, lock_parents_weakrefs=None, *, is_compiling):
        """Registers the parent tensordict that handles the lock."""
        self._is_locked = True
        if lock_parents_weakrefs is not None:
            lock_parents_weakrefs = [
                ref
                for ref in lock_parents_weakrefs
                if not any(refref is ref for refref in self._lock_parents_weakrefs)
            ]
        if not is_compiling:
            is_root = lock_parents_weakrefs is None
            if is_root:
                lock_parents_weakrefs = []
            else:
                self._lock_parents_weakrefs = (
                    self._lock_parents_weakrefs + lock_parents_weakrefs
                )
            lock_parents_weakrefs = list(lock_parents_weakrefs)
            lock_parents_weakrefs.append(weakref.ref(self))

        for _td in self._nested_tensordicts.values():
            _td._propagate_lock(lock_parents_weakrefs, is_compiling=is_compiling)

    @erase_cache
    def _propagate_unlock(self):
        # if we end up here, we can clear the graph associated with this td
        self._is_locked = False

        self._is_shared = False
        self._is_memmap = False

        sub_tds = []
        for _td in self._nested_tensordicts.values():
            sub_tds.extend(_td._propagate_unlock())
            sub_tds.append(_td)
        return sub_tds

    def zero_(self) -> T:
        for key in self.keys():
            self.fill_(key, 0)
        return self

    def entry_class(self, key: NestedKey) -> type:
        entry_class = self._get_metadata(key)
        is_array = entry_class.get("array")
        if is_array:
            return torch.Tensor
        elif is_array is False:
            return PersistentTensorDict
        else:
            raise RuntimeError(f"Encountered a non-numeric data {key}.")

    def is_contiguous(self):
        return False

    def masked_fill(self, mask, value):
        return self.to_tensordict().masked_fill(mask, value)

    def where(self, condition, other, *, out=None, pad=None):
        return self.to_tensordict().where(
            condition=condition, other=other, out=out, pad=pad
        )

    def masked_fill_(self, mask, value):
        for key in self.keys(include_nested=True, leaves_only=True):
            array = self._get_array(key)
            array[expand_right(mask, array.shape).cpu().numpy()] = value
        return self

    def make_memmap(
        self,
        key: NestedKey,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't allowed for persistent tensordicts."
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    def make_memmap_from_storage(
        self,
        key: NestedKey,
        storage: torch.UntypedStorage,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't allowed for persistent tensordicts."
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    def make_memmap_from_tensor(
        self, key: NestedKey, tensor: torch.Tensor, *, copy_data: bool = True
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't allowed for persistent tensordicts."
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    def memmap_(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        num_threads: int = 0,
    ) -> PersistentTensorDict:
        raise RuntimeError(
            "Cannot build a memmap TensorDict in-place from a PersistentTensorDict. Use `td.memmap()` instead."
        )

    def _memmap_(
        self,
        *,
        prefix: str | None,
        copy_existing: bool,
        executor,
        futures,
        inplace,
        like,
        share_non_tensor,
        existsok,
    ) -> T:
        if inplace:
            raise RuntimeError("Cannot call memmap inplace in a persistent tensordict.")

        # re-implements this to make it faster using the meta-data
        def save_metadata(data: TensorDictBase, filepath, metadata=None):
            if metadata is None:
                metadata = {}
            metadata.update(
                {
                    "shape": list(data.shape),
                    "device": str(data.device),
                    "_type": str(type(self)),
                }
            )
            with open(filepath, "wb") as json_metadata:
                json_metadata.write(json.dumps(metadata))

        if prefix is not None:
            prefix = Path(prefix)
            if not prefix.exists():
                os.makedirs(prefix, exist_ok=True)
            metadata = {}
        if not self.keys():
            raise Exception(
                "memmap_like() must be called when the TensorDict is (partially) "
                "populated. Set a tensor first."
            )
        dest = TensorDict(
            {},
            batch_size=self.batch_size,
            names=self.names if self._has_names() else None,
            device=torch.device("cpu"),
        )
        dest._is_memmap = True
        for key, value in self._items_metadata():
            if not value["array"]:
                value = self._get_str(key, default=NO_DEFAULT)
                dest._set_str(
                    key,
                    value._memmap_(
                        prefix=prefix / key if prefix is not None else None,
                        executor=executor,
                        like=like,
                        copy_existing=copy_existing,
                        futures=futures,
                        inplace=inplace,
                        share_non_tensor=share_non_tensor,
                        existsok=existsok,
                    ),
                    inplace=False,
                    validated=True,
                    non_blocking=False,
                )
                continue
            else:
                value = self._get_str(key, default=NO_DEFAULT)
                if prefix is not None:
                    metadata[key] = {
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "device": str(value.device),
                    }

                def _populate(
                    tensordict=dest, key=key, value=value, prefix=prefix, like=like
                ):
                    val = MemoryMappedTensor.from_tensor(
                        value,
                        filename=(
                            str(prefix / f"{key}.memmap")
                            if prefix is not None
                            else None
                        ),
                        copy_data=not like,
                        copy_existing=copy_existing,
                        existsok=existsok,
                    )
                    tensordict._set_str(
                        key,
                        val,
                        inplace=False,
                        validated=True,
                        non_blocking=False,
                    )

                if executor is None:
                    _populate()
                else:
                    futures.append(executor.submit(_populate))

        if prefix is not None:
            if executor is None:
                save_metadata(dest, prefix / "meta.json", metadata)
            else:
                futures.append(
                    executor.submit(save_metadata, dest, prefix / "meta.json", metadata)
                )
        return dest

    _load_memmap = TensorDict._load_memmap

    def pin_memory(self, *args, **kwargs):
        raise RuntimeError(
            f"Cannot pin memory of a {type(self).__name__}. Call to_tensordict() before making this call."
        )

    @lock_blocked
    def popitem(self) -> Tuple[NestedKey, CompatibleType]:
        raise NotImplementedError(
            f"popitem not implemented for class {type(self).__name__}."
        )

    def map(
        self,
        fn: Callable,
        dim: int = 0,
        num_workers: int = None,
        *,
        out: TensorDictBase = None,
        chunksize: int = None,
        num_chunks: int = None,
        pool: mp.Pool = None,
        generator: torch.Generator | None = None,
        max_tasks_per_child: int | None = None,
        worker_threads: int = 1,
        index_with_generator: bool = False,
        pbar: bool = False,
        mp_start_method: str | None = None,
    ):
        if pool is None:
            if num_workers is None:
                num_workers = mp.cpu_count()  # Get the number of CPU cores
            if generator is None:
                generator = torch.Generator()
            seed = (
                torch.empty((), dtype=torch.int64).random_(generator=generator).item()
            )
            if mp_start_method is not None:
                ctx = mp.get_context(mp_start_method)
            else:
                ctx = mp.get_context()

            queue = ctx.Queue(maxsize=num_workers)
            for i in range(num_workers):
                queue.put(i)
            with ctx.Pool(
                processes=num_workers,
                initializer=_proc_init,
                initargs=(seed, queue, worker_threads),
                maxtasksperchild=max_tasks_per_child,
            ) as pool:
                return self.map(
                    fn,
                    dim=dim,
                    chunksize=chunksize,
                    pool=pool,
                    index_with_generator=index_with_generator,
                )
        num_workers = pool._processes
        dim_orig = dim
        if dim < 0:
            dim = self.ndim + dim
        if dim < 0 or dim >= self.ndim:
            raise ValueError(f"Got incompatible dimension {dim_orig}")

        self_split = _split_tensordict(
            self,
            chunksize,
            num_chunks,
            num_workers,
            dim,
            use_generator=index_with_generator,
            to_tensordict=True,
        )
        if not index_with_generator:
            length = len(self_split)
            self_split = tuple(split.to_tensordict() for split in self_split)
        else:
            length = None

        if out is not None and (out.is_shared() or out.is_memmap()):

            def wrap_fn_with_out(fn, out):
                @wraps(fn)
                def newfn(item_and_out):
                    item, out = item_and_out
                    result = fn(item)
                    out.update_(result)
                    return

                out_split = _split_tensordict(
                    out,
                    chunksize,
                    num_chunks,
                    num_workers,
                    dim,
                    use_generator=index_with_generator,
                )
                return _CloudpickleWrapper(newfn), _zip_strict(self_split, out_split)

            fn, self_split = wrap_fn_with_out(fn, out)
            out = None

        call_chunksize = 1
        imap = pool.imap(fn, self_split, call_chunksize)

        if pbar and importlib.util.find_spec("tqdm", None) is not None:
            import tqdm

            imap = tqdm.tqdm(imap, total=length)

        imaplist = []
        start = 0
        for item in imap:
            if item is not None:
                if out is not None:
                    if chunksize != 0:
                        end = start + item.shape[dim]
                        chunk = slice(start, end)
                        out[chunk].update_(item)
                        start = end
                    else:
                        out[start].update_(item)
                        start += 1
                else:
                    imaplist.append(item)
        del imap

        # support inplace modif
        if imaplist:
            if chunksize == 0:
                out = torch.stack(imaplist, dim)
            else:
                out = torch.cat(imaplist, dim)
        return out

    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> PersistentTensorDict:
        old_key = self._process_key(old_key)
        new_key = self._process_key(new_key)
        try:
            self.file.move(old_key, new_key)
        except ValueError as err:
            raise KeyError(f"key {new_key} already present in TensorDict.") from err
        return self

    def fill_(self, key: NestedKey, value: float | bool) -> TensorDictBase:
        """Fills a tensor pointed by the key with the a given value.

        Args:
            key (str): key to be remaned
            value (Number, bool): value to use for the filling

        Returns:
            self

        """
        md = self._get_metadata(key)
        if md.get("array"):
            array = self._get_array(key)
            array[:] = value
        else:
            nested = self.get(key)
            for subkey in nested.keys():
                nested.fill_(subkey, value)
        return self

    def _create_nested_str(self, key):
        self.file.create_group(key)
        target_td = self._get_str(key, default=NO_DEFAULT)
        return target_td

    def _select(
        self, *keys: NestedKey, inplace: bool = False, strict: bool = True
    ) -> PersistentTensorDict:
        raise NotImplementedError(
            "Cannot call select on a PersistentTensorDict. "
            "Create a regular tensordict first using the `to_tensordict` method."
        )

    def _exclude(
        self, *keys: NestedKey, inplace: bool = False, set_shared: bool = True
    ) -> PersistentTensorDict:
        raise NotImplementedError(
            "Cannot call exclude on a PersistentTensorDict. "
            "Create a regular tensordict first using the `to_tensordict` method."
        )

    @_as_context_manager()
    def flatten_keys(self, separator: str = ".", inplace: bool = False) -> T:
        if inplace:
            raise ValueError(
                "Cannot call flatten_keys in_place with a PersistentTensorDict."
            )
        return self.to_tensordict().flatten_keys(separator=separator)

    @_as_context_manager()
    def unflatten_keys(self, separator: str = ".", inplace: bool = False) -> T:
        if inplace:
            raise ValueError(
                "Cannot call unflatten_keys in_place with a PersistentTensorDict."
            )
        return self.to_tensordict().unflatten_keys(separator=separator)

    def share_memory_(self):
        raise NotImplementedError(
            "Cannot call share_memory_ on a PersistentTensorDict. "
            "Create a regular tensordict first using the `to_tensordict` method."
        )

    def to(self, *args, **kwargs: Any) -> PersistentTensorDict:
        (
            device,
            dtype,
            non_blocking,
            convert_to_format,
            batch_size,
            non_blocking_pin,
            num_threads,
            inplace,
        ) = _parse_to(*args, **kwargs)
        if inplace:
            raise TypeError(f"Cannot use inplace=True with {type(self).__name__}.to().")

        if non_blocking_pin:
            raise RuntimeError(
                f"Cannot use non_blocking_pin=True {type(self).__name__}.to(). Call "
                f"`to_tensordict()` before executing this code."
            )
        result = self
        if device is not None and dtype is None and device == self.device:
            return result
        if dtype is not None:
            return self.to_tensordict().to(*args, **kwargs)
        result = self
        if device is not None:
            result = result.clone(False)
            result._device = device
            for key, nested in list(result._nested_tensordicts.items()):
                result._nested_tensordicts[key] = nested.to(device)
        if batch_size is not None:
            result.batch_size = batch_size
        return result

    def _to_numpy(self, value):
        if hasattr(value, "requires_grad") and value.requires_grad:
            raise RuntimeError("Cannot set a tensor that has requires_grad=True.")
        if isinstance(value, torch.Tensor):
            out = value.cpu().detach().numpy()
        elif isinstance(value, dict):
            out = TensorDict(value, [])
        elif is_non_tensor(value):
            value = value.data
            if isinstance(value, str):
                return value
            import h5py

            out = np.array(value)
            out = out.astype(h5py.opaque_dtype(out.dtype))
        elif is_tensor_collection(value):
            out = value
        elif isinstance(value, (np.ndarray,)):
            out = value
        else:
            raise NotImplementedError(
                f"Cannot set values of type {value} in a PersistentTensorDict."
            )
        return out

    def _set(
        self,
        key: NestedKey,
        value: Any,
        *,
        inplace: bool = False,
        idx=None,
        validated: bool = False,
        ignore_lock: bool = False,
        non_blocking: bool = False,
    ) -> PersistentTensorDict:
        if not validated:
            value = self._validate_value(value, check_shape=idx is None)
        value = self._to_numpy(value)
        if not inplace:
            if idx is not None:
                raise RuntimeError("Cannot pass an index to _set when inplace=False.")
            elif self.is_locked and not ignore_lock:
                raise RuntimeError(_LOCK_ERROR)
        # shortcut set if we're placing a tensordict
        key = _unravel_key_to_tuple(key)
        first_key, subkey = key[0], key[1:]
        if is_tensor_collection(value):
            target_td = self._get_str(first_key, default=None)
            if target_td is None:
                self.file.create_group(first_key)
                target_td = self._get_str(first_key, default=NO_DEFAULT)
                target_td.batch_size = value.batch_size
            elif not is_tensor_collection(target_td):
                raise RuntimeError(
                    f"cannot set a tensor collection in place of a non-tensor collection in {type(self).__name__}. "
                    f"Got self.get({first_key})={target_td} and value={value}."
                )
            if idx is None:
                if len(subkey):
                    target_td.set(subkey, value, inplace=inplace)
                else:
                    target_td.update(value, inplace=inplace)
            else:
                if len(subkey):
                    target_td.set_at_(subkey, value, idx=idx)
                else:
                    target_td.update_at_(value, idx=idx)

            return self

        if inplace:
            # could be called before but will go under further refactoring of set
            key = self._process_key(key)
            array = self.file[key]
            if idx is None:
                idx = ()
            else:
                idx = self._process_index(idx, array)
            try:
                array[idx] = value
            except TypeError as err:
                if "Boolean indexing array has incompatible shape" in str(err):
                    # Known bug in h5py: cannot broadcast boolean mask on the right as
                    # done in np and torch. Therefore we put a performance warning
                    # and convert to torch tensor first.
                    warnings.warn(
                        "Indexing an h5py.Dataset object with a boolean mask "
                        "that needs broadcasting does not work directly. "
                        "tensordict will cast the entire array in memory and index it using the mask. "
                        "This is suboptimal and may lead to performance issue."
                    )
                    idx = tuple(
                        (
                            expand_right(torch.as_tensor(_idx), array.shape).numpy()
                            if _idx.dtype == np.dtype("bool")
                            else _idx
                        )
                        for _idx in idx
                    )
                    array[idx] = torch.as_tensor(value)
                else:
                    raise err

        else:
            key = self._process_key(key)
            try:
                self.file.create_dataset(key, data=value, **self.kwargs)
            except (ValueError, OSError) as err:
                if "name already exists" in str(err):
                    warnings.warn(
                        "Replacing an array with another one is inefficient. "
                        "Consider using different names or populating in-place using `inplace=True`."
                    )
                    del self.file[key]
                    self.file.create_dataset(key, data=value, **self.kwargs)
            # If we have a nested key, let's make sure we have the corresponding TD registered
            if subkey:
                self._get_tuple((first_key, *subkey[:-1]), default=NO_DEFAULT)
        return self

    def _convert_inplace(self, inplace, key):
        key = self._process_key(key)
        if inplace is not False:
            has_key = key in self.file
            if inplace is True and not has_key:  # inplace could be None
                raise KeyError(
                    _KEY_ERROR.format(key, type(self).__name__, sorted(self.keys()))
                )
            inplace = has_key
        return inplace

    def _set_non_tensor(self, key: NestedKey, value: Any):
        raise NotImplementedError(
            f"set_non_tensor is not compatible with the tensordict type {type(self).__name__}."
        )

    def _set_str(
        self,
        key: str,
        value: Any,
        *,
        inplace: bool,
        validated: bool,
        ignore_lock: bool = False,
        non_blocking: bool = False,
    ):
        inplace = self._convert_inplace(inplace, key)
        return self._set(
            key,
            value,
            inplace=inplace,
            validated=validated,
            ignore_lock=ignore_lock,
            non_blocking=non_blocking,
        )

    def _set_tuple(self, key, value, *, inplace, validated, non_blocking):
        key = _unravel_key_to_tuple(key)
        if len(key) == 1:
            return self._set_str(
                key[0],
                value,
                inplace=inplace,
                validated=validated,
                non_blocking=non_blocking,
            )
        elif key[0] in self.keys():
            return self._get_str(key[0], NO_DEFAULT)._set_tuple(
                key[1:],
                value,
                inplace=inplace,
                validated=validated,
                non_blocking=non_blocking,
            )
        inplace = self._convert_inplace(inplace, key)
        return self._set(
            key, value, inplace=inplace, validated=validated, non_blocking=non_blocking
        )

    def _set_at_str(self, key, value, idx, *, validated, non_blocking):
        return self._set(
            key,
            value,
            inplace=True,
            idx=idx,
            validated=validated,
            non_blocking=non_blocking,
        )

    def _set_at_tuple(self, key, value, idx, *, validated, non_blocking):
        return self._set(
            key,
            value,
            inplace=True,
            idx=idx,
            validated=validated,
            non_blocking=non_blocking,
        )

    def _set_metadata(self, orig_metadata_container: PersistentTensorDict):
        for key, td in orig_metadata_container._nested_tensordicts.items():
            array = self._get_array(key)
            self._nested_tensordicts[key] = PersistentTensorDict(
                group=array,
                batch_size=td.batch_size,
                device=td.device,
            )
            self._nested_tensordicts[key].names = td._td_dim_names
            self._nested_tensordicts[key]._set_metadata(td)

    def _clone(self, recurse: bool = True, newfile=None) -> PersistentTensorDict:
        import h5py

        if recurse:
            # this should clone the h5 to a new location indicated by newfile
            if newfile is None:
                warnings.warn(
                    "A destination should be provided when cloning a "
                    "PersistentTensorDict. A temporary file will be used "
                    "instead. Use `recurse=False` to keep track of the original data "
                    "with a new PersistentTensorDict instance."
                )
                tmpfile = tempfile.NamedTemporaryFile()
                newfile = tmpfile.name
            f_dest = h5py.File(newfile, "w", locking=self.LOCKING)
            f_src = self.file
            for key in self.keys(include_nested=True, leaves_only=True):
                key = self._process_key(key)
                f_dest.create_dataset(key, data=f_src[key], **self.kwargs)
                # f_src.copy(f_src[key],  f_dest[key], "DataSet")
            # create a non-recursive copy and update the file
            # this way, we can keep the batch-size of every nested tensordict
            clone = self.clone(False)
            clone.file = f_dest
            clone.filename = newfile
            clone._pin_mem = False
            clone.names = self._td_dim_names
            clone._nested_tensordicts = {}
            clone._set_metadata(self)
            return clone
        else:
            # we need to keep the batch-size of nested tds, which we do manually
            nested_tds = {
                key: td.clone(False) for key, td in self._nested_tensordicts.items()
            }
            filename = self.filename
            file = self.file if filename is None else None
            clone = PersistentTensorDict(
                filename=filename,
                group=file,
                mode=self.mode,
                backend="h5",
                device=self.device,
                batch_size=self.batch_size,
            )
            clone._nested_tensordicts = nested_tds
            clone._pin_mem = False
            clone.names = self._td_dim_names
            return clone

    def __getstate__(self):
        state = self.__dict__.copy()
        filename = state["file"].file.filename
        group_name = state["file"].name
        state["file"] = None
        state["filename"] = filename
        state["group_name"] = group_name
        state["__lock_parents_weakrefs"] = None
        return state

    def __setstate__(self, state):
        import h5py

        state["file"] = h5py.File(
            state["filename"], mode=state["mode"], locking=self.LOCKING
        )
        if state["group_name"] != "/":
            state["file"] = state["file"][state["group_name"]]
        del state["group_name"]
        self.__dict__.update(state)
        if self._is_locked:
            # this can cause avoidable overhead, as we will be locking the leaves
            # then locking their parent, and the parent of the parent, every
            # time re-locking tensordicts that have already been locked.
            # To avoid this, we should lock only at the root, but it isn't easy
            # to spot what the root is...
            self._is_locked = False
            self.lock_()

    def _add_batch_dim(self, *, in_dim, vmap_level):
        raise RuntimeError("Persistent tensordicts cannot be used with vmap.")

    def _remove_batch_dim(self, vmap_level, batch_size, out_dim): ...

    def _maybe_remove_batch_dim(self, funcname, vmap_level, batch_size, out_dim): ...

    def _view(self, *args, **kwargs):
        raise RuntimeError(
            "Cannot call `view` on a persistent tensordict. Call `reshape` instead."
        )

    def _transpose(self, dim0, dim1):
        raise RuntimeError(
            "Cannot call `transpose` on a persistent tensordict. Make it dense before calling this method by calling `to_tensordict`."
        )

    def _permute(
        self,
        *args,
        **kwargs,
    ):
        raise RuntimeError(
            "Cannot call `permute` on a persistent tensordict. Make it dense before calling this method by calling `to_tensordict`."
        )

    def _squeeze(self, dim=None):
        raise RuntimeError(
            "Cannot call `squeeze` on a persistent tensordict. Make it dense before calling this method by calling `to_tensordict`."
        )

    def _unsqueeze(self, dim):
        raise RuntimeError(
            "Cannot call `unsqueeze` on a persistent tensordict. Make it dense before calling this method by calling `to_tensordict`."
        )

    __eq__ = TensorDict.__eq__
    __ne__ = TensorDict.__ne__
    __xor__ = TensorDict.__xor__
    __or__ = TensorDict.__or__
    __ge__ = TensorDict.__ge__
    __gt__ = TensorDict.__gt__
    __le__ = TensorDict.__le__
    __lt__ = TensorDict.__lt__

    _apply_nest = TensorDict._apply_nest
    _cast_reduction = TensorDict._cast_reduction
    _check_device = TensorDict._check_device
    _check_is_shared = TensorDict._check_is_shared
    _convert_to_tensordict = TensorDict._convert_to_tensordict
    _get_names_idx = TensorDict._get_names_idx
    _index_tensordict = TensorDict._index_tensordict
    _multithread_apply_flat = TensorDict._multithread_apply_flat
    _multithread_rebuild = TensorDict._multithread_rebuild
    _to_module = TensorDict._to_module
    _unbind = TensorDict._unbind
    all = TensorDict.all
    any = TensorDict.any
    expand = TensorDict.expand
    from_dict_instance = TensorDict.from_dict_instance
    masked_select = TensorDict.masked_select
    _repeat = TensorDict._repeat
    _repeat = TensorDict._repeat
    repeat_interleave = TensorDict.repeat_interleave
    reshape = TensorDict.reshape
    split = TensorDict.split


def _set_max_batch_size(source: PersistentTensorDict):
    """Updates a tensordict with its maximium batch size."""
    tensor_data = list(source._items_metadata())
    for key, val in tensor_data:
        if not val.get("non_tensor", None) and not val.get("array", None):
            _set_max_batch_size(source.get(key, None))

    batch_size = []
    if not tensor_data:  # when source is empty
        source.batch_size = batch_size
        return

    curr_dim = 0
    # We need to reload this list because the value have changed
    tensor_data = list(source._items_metadata())
    tensor_keys, tensor_data = zip(*tensor_data)
    # Filter out the non-tensor data
    tensor_data = [data for data in tensor_data if not data.get("non_tensor")]
    while True:
        if tensor_data[0]["dim"] > curr_dim:
            curr_dim_size = tensor_data[0]["shape"][curr_dim]
        else:
            source.batch_size = batch_size
            return
        for tensor in tensor_data[1:]:
            if tensor["dim"] <= curr_dim or tensor["shape"][curr_dim] != curr_dim_size:
                source.batch_size = batch_size
                return
        batch_size.append(curr_dim_size)
        curr_dim += 1


def _is_non_tensor_h5(val):
    import h5py

    dt = val.dtype
    if (
        h5py.check_string_dtype(dt)
        or h5py.check_vlen_dtype(dt)
        or h5py.check_enum_dtype(dt)
        or h5py.check_opaque_dtype(dt)
    ):
        return True
    return False
