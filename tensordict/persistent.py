# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent tensordicts (H5 and others)."""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Any

from tensordict._tensordict import _unravel_key_to_tuple

H5_ERR = None
try:
    import h5py

    _has_h5 = True
except ModuleNotFoundError as err:
    H5_ERR = err
    _has_h5 = False

import numpy as np
import torch

from tensordict import MemmapTensor
from tensordict.tensordict import (
    _TensorDictKeysView,
    CompatibleType,
    is_tensor_collection,
    NO_DEFAULT,
    TensorDict,
    TensorDictBase,
)
from tensordict.utils import (
    cache,
    DeviceType,
    expand_right,
    IndexType,
    NestedKey,
    NUMPY_TO_TORCH_DTYPE_DICT,
)


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
        if self.include_nested:
            visitor = _Visitor(lambda key: tuple(key.split("/")))
            self.tensordict.file.visit(visitor)
            if self.leaves_only:
                for key in visitor:
                    if self.tensordict._get_metadata(key).get("array", None):
                        yield key
            else:
                yield from visitor
        else:
            yield from self.tensordict._valid_keys()

    def __contains__(self, key):
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        for a_key in self:
            if isinstance(a_key, tuple) and len(a_key) == 1:
                a_key = a_key[0]
            if key == a_key:
                return True
        else:
            return False


class PersistentTensorDict(TensorDictBase):
    """Persistent TensorDict implementation.

    :class:`PersistentTensorDict` instances provide an interface with data stored
    on disk such that access to this data is made easy while still taking advantage
    from the fast access provided by the backend.

    Like other :class:`TensorDictBase` subclasses, :class:`PersistentTensorDict`
    has a ``device`` attribute. This does *not* mean that the data is being stored
    on that device, but rather that when loaded, the data will be cast onto
    the desired device.

    Args:
        batch_size (torch.Size or compatible): the tensordict batch size.
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

    def __new__(cls, *args, **kwargs):
        cls._td_dim_names = None
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        *,
        batch_size,
        filename=None,
        group=None,
        mode="r",
        backend="h5",
        device=None,
        **kwargs,
    ):
        self._locked_tensordicts = []
        self._lock_id = set()
        if not _has_h5:
            raise ModuleNotFoundError("Could not load h5py.") from H5_ERR
        super().__init__()
        self.filename = filename
        self.mode = mode
        if backend != "h5":
            raise NotImplementedError
        if filename is not None and group is None:
            self.file = h5py.File(filename, mode)
        elif group is not None:
            self.file = group
        else:
            raise RuntimeError(
                f"Either group or filename must be provided, and not both. Got group={group} and filename={filename}."
            )
        self._batch_size = torch.Size(batch_size)
        self._device = device
        self._is_shared = False
        self._is_memmap = False
        self.kwargs = kwargs

        # we use this to allow nested tensordicts to have a different batch-size
        self._nested_tensordicts = {}
        self._pin_mem = False

        # this must be kept last
        self._check_batch_size(self._batch_size)

    @classmethod
    def from_h5(cls, filename, mode="r"):
        """Creates a PersistentTensorDict from a h5 file.

        This function will automatically determine the batch-size for each nested
        tensordict.

        Args:
            filename (str): the path to the h5 file.
            mode (str, optional): reading mode. Defaults to ``"r"``.
        """
        out = cls(filename=filename, mode=mode, batch_size=[])
        # determine batch size
        _set_max_batch_size(out)
        return out

    @classmethod
    def from_dict(cls, input_dict, filename, batch_size=None, device=None, **kwargs):
        """Converts a dictionary or a TensorDict to a h5 file.

        Args:
            input_dict (dict, TensorDict or compatible): data to be stored as h5.
            filename (str or path): path to the h5 file.
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
        file = h5py.File(filename, "w")
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
            size = self.file[key].shape
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
        if isinstance(array, (h5py.Dataset,)):
            if self.device is not None:
                device = self.device
            else:
                device = torch.device("cpu")
            # we convert to an array first to avoid "Creating a tensor from a list of numpy.ndarrays is extremely slow."
            array = array[()]
            out = torch.as_tensor(array, device=device)
            if self._pin_mem:
                return out.pin_memory()
            return out
        else:
            out = self._nested_tensordicts.get(key, None)
            if out is None:
                out = self._nested_tensordicts[key] = PersistentTensorDict(
                    group=array,
                    batch_size=self.batch_size,
                    device=self.device,
                )
            return out

    @cache  # noqa: B019
    def get(self, key, default=NO_DEFAULT):
        array = self._get_array(key, default)
        if array is default:
            return array
        return self._process_array(key, array)

    _get_str = get
    _get_tuple = get

    def get_at(
        self, key: str, idx: IndexType, default: CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
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
            out = self._nested_tensordicts.get(key, None)
            if out is None:
                out = self._nested_tensordicts[key] = PersistentTensorDict(
                    group=array,
                    batch_size=self.batch_size,
                    device=self.device,
                )
            return out.get_sub_tensordict(idx)
        else:
            return default

    def _get_metadata(self, key):
        """Gets the metadata for an entry.

        This method avoids creating a tensor from scratch, and just reads the metadata of the array.
        """
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
            return {}
        else:
            shape = self.get(key).shape
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

    def __getitem__(self, item):
        if isinstance(item, str) or (
            isinstance(item, tuple) and all(isinstance(val, str) for val in item)
        ):
            return self.get(item)
        if isinstance(item, list):
            # convert to tensor
            item = torch.tensor(item)
        return self.get_sub_tensordict(item)

    __getitems__ = __getitem__

    def __setitem__(self, index, value):
        index_unravel = _unravel_key_to_tuple(index)
        if index_unravel:
            return self.set(index_unravel, value, inplace=True)

        if isinstance(index, list):
            # convert to tensor
            index = torch.tensor(index)
        sub_td = self.get_sub_tensordict(index)
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
            if self._get_metadata(key):
                keys.append(key)
        return keys

    # @cache  # noqa: B019
    def keys(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> _PersistentTDKeysView:
        return _PersistentTDKeysView(
            tensordict=self,
            include_nested=include_nested,
            leaves_only=leaves_only,
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
        self, key: str, list_item: list[CompatibleType], dim: int
    ) -> PersistentTensorDict:
        stacked = torch.stack(list_item, dim=dim)
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

    def entry_class(self, key: NestedKey) -> type:
        entry_class = self._get_metadata(key)
        is_array = entry_class.get("array", None)
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

    def masked_fill_(self, mask, value):
        for key in self.keys(include_nested=True, leaves_only=True):
            array = self._get_array(key)
            array[expand_right(mask, array.shape).cpu().numpy()] = value
        return self

    def memmap_(
        self, prefix: str | None = None, copy_existing: bool = False
    ) -> PersistentTensorDict:
        raise RuntimeError(
            "Cannot build a memmap TensorDict in-place from a PersistentTensorDict. Use `td.memmap()` instead."
        )

    def memmap(
        self,
        prefix: str | None = None,
    ) -> TensorDict:
        """Converts the PersistentTensorDict to a memmap equivalent."""
        mm_like = self.memmap_like(prefix)
        for key in self.keys(include_nested=True, leaves_only=True):
            mm_val = mm_like[key]
            mm_val._memmap_array[:] = self._get_array(key)
        return mm_like

    def memmap_like(self, prefix: str | None = None) -> TensorDictBase:
        # re-implements this to make it faster using the meta-data
        if prefix is not None:
            prefix = Path(prefix)
            if not prefix.exists():
                prefix.mkdir(exist_ok=True)
            torch.save(
                {"batch_size": self.batch_size, "device": self.device},
                prefix / "meta.pt",
            )
        if not self.keys():
            raise Exception(
                "memmap_like() must be called when the TensorDict is (partially) "
                "populated. Set a tensor first."
            )
        tensordict = TensorDict({}, self.batch_size, device=self.device)
        for key, value in self._items_metadata():
            if not value["array"]:
                value = self.get(key)
                if prefix is not None:
                    # ensure subdirectory exists
                    (prefix / key).mkdir(exist_ok=True)
                    tensordict[key] = value.memmap_like(
                        prefix=prefix / key,
                    )
                    torch.save(
                        {"batch_size": value.batch_size, "device": value.device},
                        prefix / key / "meta.pt",
                    )
                else:
                    tensordict[key] = value.memmap_like()
                continue
            else:
                tensordict[key] = MemmapTensor(
                    value["shape"],
                    device="cpu",
                    dtype=value["dtype"],
                    filename=str(prefix / f"{key}.memmap")
                    if prefix is not None
                    else None,
                )
            if prefix is not None:
                torch.save(
                    {
                        "shape": value["shape"],
                        "device": torch.device("cpu"),
                        "dtype": value["dtype"],
                    },
                    prefix / f"{key}.meta.pt",
                )
        tensordict._is_memmap = True
        tensordict.lock_()
        return tensordict

    def pin_memory(self):
        """Returns a new PersistentTensorDict where any given Tensor key returns a tensor with pin_memory=True.

        This will fail with PersistentTensorDict with a ``cuda`` device attribute.

        """
        if self.device.type == "cuda":
            raise RuntimeError("cannot pin memory on a tensordict stored on cuda.")
        out = self.clone(False)
        out._pin_mem = True
        out._nested_tensordicts = {
            key: val.pin_memory() for key, val in out._nested_tensordicts.items()
        }
        return out

    def rename_key_(
        self, old_key: str, new_key: str, safe: bool = False
    ) -> PersistentTensorDict:
        old_key = self._process_key(old_key)
        new_key = self._process_key(new_key)
        try:
            self.file.move(old_key, new_key)
        except ValueError as err:
            raise KeyError(f"key {new_key} already present in TensorDict.") from err
        return self

    def fill_(self, key: str, value: float | bool) -> TensorDictBase:
        """Fills a tensor pointed by the key with the a given value.

        Args:
            key (str): key to be remaned
            value (Number, bool): value to use for the filling

        Returns:
            self

        """
        md = self._get_metadata(key)
        if md.get("array", None):
            array = self._get_array(key)
            array[:] = value
        else:
            nested = self.get(key)
            for subkey in nested.keys():
                nested.fill_(subkey, value)
        return self

    def _create_nested_str(self, key):
        self.file.create_group(key)
        target_td = self._get_str(key)
        return target_td

    def select(
        self, *keys: str, inplace: bool = False, strict: bool = True
    ) -> PersistentTensorDict:
        raise NotImplementedError(
            "Cannot call select on a PersistentTensorDict. "
            "Create a regular tensordict first using the `to_tensordict` method."
        )

    def exclude(self, *keys: str, inplace: bool = False) -> PersistentTensorDict:
        raise NotImplementedError(
            "Cannot call exclude on a PersistentTensorDict. "
            "Create a regular tensordict first using the `to_tensordict` method."
        )

    def share_memory_(self):
        raise NotImplementedError(
            "Cannot call share_memory_ on a PersistentTensorDict. "
            "Create a regular tensordict first using the `to_tensordict` method."
        )

    def to(
        self, dest: DeviceType | torch.Size | type, **kwargs: Any
    ) -> PersistentTensorDict:
        if isinstance(dest, type) and issubclass(dest, TensorDictBase):
            if isinstance(self, dest):
                return self
            td = dest(source=self, **kwargs)
            return td
        elif isinstance(dest, (torch.device, str, int)):
            # must be device
            dest = torch.device(dest)
            if self.device is not None and dest == self.device:
                return self
            out = self.clone(False)
            out._device = dest
            for key, nested in list(out._nested_tensordicts.items()):
                out._nested_tensordicts[key] = nested.to(dest)
            return out
        elif isinstance(dest, torch.Size):
            self.batch_size = dest
            return self
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict "
                f"instance, {dest} not allowed"
            )

    def _to_numpy(self, value):
        if hasattr(value, "requires_grad") and value.requires_grad:
            raise RuntimeError("Cannot set a tensor that has requires_grad=True.")
        if isinstance(value, torch.Tensor):
            out = value.cpu().detach().numpy()
        elif isinstance(value, MemmapTensor):
            out = value._memmap_array
        elif isinstance(value, dict):
            out = TensorDict(value, [])
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
        key: str,
        value,
        inplace: bool = False,
        idx=None,
        validated=False,
    ) -> PersistentTensorDict:
        if not validated:
            value = self._validate_value(value, check_shape=idx is None)
        value = self._to_numpy(value)
        if not inplace:
            if idx is not None:
                raise RuntimeError("Cannot pass an index to _set when inplace=False.")
            elif self.is_locked:
                raise RuntimeError(self.LOCK_ERROR)
        # shortcut set if we're placing a tensordict
        if is_tensor_collection(value):
            if isinstance(key, tuple):
                key, subkey = key[0], key[1:]
            else:
                key, subkey = key, []
            target_td = self._get_str(key, default=None)
            if target_td is None:
                self.file.create_group(key)
                target_td = self._get_str(key)
                target_td.batch_size = value.batch_size
            elif not is_tensor_collection(target_td):
                raise RuntimeError(
                    f"cannot set a tensor collection in place of a non-tensor collection in {self.__class__.__name__}. "
                    f"Got self.get({key})={target_td} and value={value}."
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
                        expand_right(torch.as_tensor(_idx), array.shape).numpy()
                        if _idx.dtype == np.dtype("bool")
                        else _idx
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
        return self

    def _convert_inplace(self, inplace, key):
        key = self._process_key(key)
        if inplace is not False:
            has_key = key in self.file
            if inplace is True and not has_key:  # inplace could be None
                raise KeyError(
                    TensorDictBase.KEY_ERROR.format(
                        key, self.__class__.__name__, sorted(self.keys())
                    )
                )
            inplace = has_key
        return inplace

    def _set_str(self, key, value, *, inplace, validated):
        inplace = self._convert_inplace(inplace, key)
        return self._set(key, value, inplace=inplace, validated=validated)

    def _set_tuple(self, key, value, *, inplace, validated):
        if len(key) == 1:
            return self._set_str(key[0], value, inplace=inplace, validated=validated)
        elif key[0] in self.keys():
            return self._get_str(key[0])._set_tuple(
                key[1:], value, inplace=inplace, validated=validated
            )
        inplace = self._convert_inplace(inplace, key)
        return self._set(key, value, inplace=inplace, validated=validated)

    def _set_at_str(self, key, value, idx, *, validated):
        return self._set(key, value, inplace=True, idx=idx, validated=validated)

    def _set_at_tuple(self, key, value, idx, *, validated):
        return self._set(key, value, inplace=True, idx=idx, validated=validated)

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

    def clone(self, recurse: bool = True, newfile=None) -> PersistentTensorDict:
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
            f_dest = h5py.File(newfile, "w")
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
        return state

    def __setstate__(self, state):
        state["file"] = h5py.File(state["filename"], mode=state["mode"])
        if state["group_name"] != "/":
            state["file"] = state["file"][state["group_name"]]
        del state["group_name"]
        self.__dict__.update(state)

    def _add_batch_dim(self, *, in_dim, vmap_level):
        raise RuntimeError("Persistent tensordicts cannot be used with vmap.")

    def _remove_batch_dim(self, vmap_level, batch_size, out_dim):
        # not accessible
        ...


def _set_max_batch_size(source: PersistentTensorDict):
    """Updates a tensordict with its maximium batch size."""
    tensor_data = list(source._items_metadata())
    for key, val in tensor_data:
        if not val["array"]:
            _set_max_batch_size(source.get(key))

    batch_size = []
    if not tensor_data:  # when source is empty
        source.batch_size = batch_size
        return

    curr_dim = 0
    tensor_data = list(source._values_metadata())
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
