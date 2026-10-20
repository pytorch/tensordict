# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Save and load tensordicts as safetensors files."""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tensordict._lazy import LazyStackedTensorDict
from tensordict._td import TensorDict
from tensordict.base import is_tensor_collection
from tensordict.tensorclass import NonTensorStack
from tensordict.utils import _is_tensorclass, DeviceType, is_non_tensor

if TYPE_CHECKING:
    from tensordict.base import TensorDictBase

_has_safetensors = importlib.util.find_spec("safetensors") is not None

# The key of the tensordict structure in the ``__metadata__`` of the file, and
# the version of the JSON document stored under it.
_METADATA_KEY = "tensordict"
_FORMAT_VERSION = 1


def _check_safetensors() -> None:
    if not _has_safetensors:
        raise ImportError(
            "Saving and loading safetensors files requires the safetensors "
            "package. Install it with `pip install safetensors`."
        )


def _is_json_value(value) -> bool:
    """Whether JSON gives ``value`` back with the same types."""
    if type(value) in (str, int, float, bool, type(None)):
        return True
    if type(value) is list:
        return all(_is_json_value(item) for item in value)
    if type(value) is dict:
        return all(
            type(key) is str and _is_json_value(item) for key, item in value.items()
        )
    return False


def _encode_level(td, prefix: tuple, tensors: dict) -> dict:
    """Returns the JSON description of ``td`` and adds its leaves to ``tensors``."""
    inner = td._tensordict if _is_tensorclass(type(td)) else td
    if isinstance(inner, LazyStackedTensorDict) and inner._has_exclusive_keys:
        raise TypeError(
            f"Cannot save the lazy stack at {prefix or 'the root'} as "
            "safetensors: its entries have different keys. Give every entry "
            "the same keys first."
        )
    level = {
        "batch_size": list(td.batch_size),
        "names": list(inner.names) if inner._has_names() else None,
    }
    if _is_tensorclass(type(td)):
        level["class"] = str(type(td))
    entries = level["keys"] = {}
    for key, value in inner.items():
        nested_key = prefix + (key,)
        if isinstance(value, NonTensorStack):
            raise TypeError(
                f"Cannot save the non-tensor stack at {nested_key} as "
                "safetensors: only single JSON values are supported."
            )
        if is_non_tensor(value):
            if not _is_json_value(value.data):
                raise TypeError(
                    f"Cannot save the non-tensor value at {nested_key} as "
                    f"safetensors: {type(value.data).__name__} is not a JSON "
                    "value (str, int, float, bool, None, list or dict)."
                )
            entries[key] = {"non_tensor": value.data}
        elif is_tensor_collection(value):
            entries[key] = {"td": _encode_level(value, nested_key, tensors)}
        else:
            if value.is_nested or value.layout != torch.strided:
                raise TypeError(
                    f"Cannot save the tensor at {nested_key} as safetensors: "
                    f"only dense tensors are supported, got layout {value.layout}"
                    f"{' (nested)' if value.is_nested else ''}."
                )
            name = ".".join(nested_key)
            if name in tensors:
                raise ValueError(
                    f"Two keys of the tensordict have the safetensors name {name!r}."
                )
            tensors[name] = value.detach().contiguous()
            entries[key] = {"tensor": name}
    return level


def _clone_overlapping(tensors: dict[str, torch.Tensor]) -> None:
    """Clones the tensors that overlap another one in memory, which safetensors refuses."""
    by_storage = defaultdict(list)
    for name, tensor in tensors.items():
        if tensor.numel() and tensor.device.type != "meta":
            storage = tensor.untyped_storage()
            by_storage[(tensor.device, storage.data_ptr())].append(name)
    for names in by_storage.values():
        if len(names) < 2:
            continue
        names.sort(key=lambda name: tensors[name].data_ptr())
        stop = 0
        for name in names:
            tensor = tensors[name]
            if tensor.data_ptr() < stop:
                tensors[name] = tensor.clone()
            else:
                stop = tensor.data_ptr() + tensor.numel() * tensor.element_size()


def to_safetensors(data: TensorDictBase, path: str | Path) -> None:
    """Saves a tensordict or a tensorclass to a safetensors file.

    Each tensor is saved under its nested key joined with ``"."``, the names
    that a model's ``state_dict`` uses. The structure of the tensordict (the
    batch size and dim names of each level, the class of each tensorclass,
    and the non-tensor values) is saved as one JSON string in the metadata of
    the file, so that :func:`~tensordict.from_safetensors` rebuilds it.

    Tensors that share memory are saved as separate copies, and a tensordict
    on another device is saved through the CPU. Requires the ``safetensors``
    package.

    Args:
        data (TensorDictBase): the tensordict or tensorclass to save.
        path (str or Path): the path of the file.

    Raises:
        TypeError: if a value cannot be stored in safetensors: a nested,
            jagged or sparse tensor, a non-tensor value that is not JSON
            (``str``, ``int``, ``float``, ``bool``, ``None``, ``list`` or
            ``dict``), a stack of non-tensor values, a lazy stack whose
            entries have different keys, or a bare non-tensor value. A lazy
            stack of tensordicts whose entries have different shapes cannot
            be saved either.
        ValueError: if two keys have the same name once joined with ``"."``,
            for example ``"a.b"`` and ``("a", "b")``.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> import torch
        >>> from tensordict import TensorDict, from_safetensors, to_safetensors
        >>> td = TensorDict(
        ...     {"weight": torch.ones(2, 3), "sub": {"bias": torch.zeros(2)}},
        ...     batch_size=[2],
        ... )
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     to_safetensors(td, Path(tmp) / "td.safetensors")
        ...     loaded = from_safetensors(Path(tmp) / "td.safetensors")
        >>> loaded.batch_size, loaded["sub", "bias"]
        (torch.Size([2]), tensor([0., 0.]))
    """
    _check_safetensors()
    from safetensors.torch import save_file

    if is_non_tensor(data):
        raise TypeError(
            "Cannot save a non-tensor value as safetensors: put it in a "
            "tensordict first."
        )
    tensors = {}
    root = _encode_level(data, (), tensors)
    _clone_overlapping(tensors)
    metadata = json.dumps({"version": _FORMAT_VERSION, "root": root})
    save_file(tensors, str(path), metadata={_METADATA_KEY: metadata})


def _find_tensorclass(class_name: str, path) -> type:
    import tensordict

    for cls in tensordict.base._ACCEPTED_CLASSES:
        if str(cls) == class_name and _is_tensorclass(cls):
            return cls
    raise RuntimeError(
        f"Could not find the tensorclass {class_name} saved in {path}. Import "
        "the module that defines it before you load the file."
    )


def _decode_level(level: dict, file, device, path, names_used: set):
    td = TensorDict(batch_size=level["batch_size"], names=level["names"], device=device)
    for key, entry in level["keys"].items():
        if "tensor" in entry:
            names_used.add(entry["tensor"])
            td.set(key, file.get_tensor(entry["tensor"]))
        elif "non_tensor" in entry:
            td.set_non_tensor(key, entry["non_tensor"])
        else:
            td.set(key, _decode_level(entry["td"], file, device, path, names_used))
    class_name = level.get("class")
    if class_name is not None:
        return _find_tensorclass(class_name, path)._from_tensordict(td)
    return td


def from_safetensors(
    path: str | Path, *, device: DeviceType | None = None
) -> TensorDictBase:
    """Loads a tensordict from a safetensors file.

    A file written by :func:`~tensordict.to_safetensors` loads with its
    structure: the batch sizes, dim names, tensorclasses and non-tensor
    values that it was saved with. A tensorclass loads only if the module
    that defines it is imported: tensordict never imports a module by the
    name in a file.

    Any other safetensors file, such as a model checkpoint from the Hugging
    Face Hub, loads as a nested :class:`~tensordict.TensorDict` with an empty
    batch size: each name is split on ``"."``. The result has the layout of
    :meth:`~tensordict.TensorDict.from_module`, so that
    :meth:`~tensordict.TensorDictBase.to_module` writes it into the model.
    Requires the ``safetensors`` package.

    Args:
        path (str or Path): the path of the file.

    Keyword Args:
        device (torch.device or compatible, optional): the device to load the
            tensors on, which is also the device of the result. Defaults to
            ``None``: the tensors load on the CPU, and the result has no
            device.

    Returns:
        a :class:`~tensordict.TensorDict`, or the tensorclass that was saved.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> import torch
        >>> from safetensors.torch import save_file
        >>> from tensordict import from_safetensors
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     save_file(
        ...         {"0.weight": torch.ones(4, 3), "0.bias": torch.zeros(4)},
        ...         Path(tmp) / "model.safetensors",
        ...     )
        ...     params = from_safetensors(Path(tmp) / "model.safetensors")
        >>> params["0", "weight"].shape
        torch.Size([4, 3])
        >>> _ = params.to_module(torch.nn.Sequential(torch.nn.Linear(3, 4)))
    """
    _check_safetensors()
    from safetensors import safe_open

    if device is not None:
        device = torch.device(device)
    with safe_open(
        str(path), framework="pt", device="cpu" if device is None else str(device)
    ) as file:
        metadata = (file.metadata() or {}).get(_METADATA_KEY)
        if metadata is None:
            td = TensorDict(device=device)
            for name in file.keys():
                td.set(tuple(name.split(".")), file.get_tensor(name))
            return td
        metadata = json.loads(metadata)
        if metadata.get("version") != _FORMAT_VERSION:
            raise ValueError(
                f"Cannot load {path}: its tensordict metadata has version "
                f"{metadata.get('version')!r}, and this version of tensordict "
                f"reads version {_FORMAT_VERSION}."
            )
        names_used = set()
        out = _decode_level(metadata["root"], file, device, path, names_used)
        missing = set(file.keys()) - names_used
        if missing:
            raise ValueError(
                f"The tensors {sorted(missing)} of {path} are not in its "
                "tensordict metadata."
            )
        return out


__all__ = ["from_safetensors", "to_safetensors"]
