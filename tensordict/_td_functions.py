# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

import torch
from tensordict._nestedkey import NestedKey
from tensordict._tensorcollection import TensorCollection

if TYPE_CHECKING:
    from tensordict.base import T

__all__ = [
    "cat",
    "from_consolidated",
    "from_module",
    "from_modules",
    "from_pytree",
    "fromkeys",
    "lazy_stack",
    "load",
    "load_memmap",
    "maybe_dense_stack",
    "memmap",
    "save",
    "stack",
]


def _tensordict_cls():
    from tensordict._td import TensorDict

    return TensorDict


def from_module(
    module,
    as_module: bool = False,
    lock: bool = True,
    use_state_dict: bool = False,
    filter_empty: bool = True,
):
    """Puts the parameters and buffers of a module in a tensordict.

    Same as :meth:`tensordict.TensorDict.from_module`, except for the default
    of ``lock``: this function locks the result by default, and the method
    does not (see #1954). The entries are the parameters and buffers of the
    module themselves, not copies. Submodules become nested tensordicts.

    Args:
        module (nn.Module): the module to get the parameters from.
        as_module (bool, optional): if ``True``, a
            :class:`~tensordict.nn.TensorDictParams` instance is returned,
            which can be used to store parameters within a
            :class:`torch.nn.Module`. Defaults to ``False``.
        lock (bool, optional): if ``True``, the resulting tensordict is
            locked. Defaults to ``True``.
        use_state_dict (bool, optional): if ``True``, the state-dict from the
            module is used and unflattened into a TensorDict with the tree
            structure of the model. Use it when the module has state-dict
            hooks. Defaults to ``False``.
        filter_empty (bool, optional): if ``True``, submodules without
            parameters or buffers (``nn.ReLU``, for example) get no entry.
            If ``False``, they get an empty tensordict. Defaults to
            ``True``.

    Returns:
        a :class:`~tensordict.TensorDict`, or a
        :class:`~tensordict.nn.TensorDictParams` if ``as_module=True``.

    Examples:
        >>> from torch import nn
        >>> import tensordict
        >>> module = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
        >>> params = tensordict.from_module(module)
        >>> params["0", "weight"] is module[0].weight
        True
        >>> params.is_locked
        True

    """
    return _tensordict_cls().from_module(
        module=module,
        as_module=as_module,
        lock=lock,
        use_state_dict=use_state_dict,
        filter_empty=filter_empty,
    )


def from_modules(
    *modules,
    as_module: bool = False,
    lock: bool = True,
    use_state_dict: bool = False,
    lazy_stack: bool = False,
    expand_identical: bool = False,
):
    """Stacks the parameters and buffers of several modules in one tensordict.

    The result has one batch dimension, with one element per module. Use it to
    run an ensemble of modules that have the same structure with
    :func:`torch.vmap` and :meth:`~tensordict.TensorDictBase.to_module`. Same
    as :meth:`tensordict.TensorDict.from_modules`: see that page for the
    difference between a dense and a lazy result, and for a full ensemble
    example.

    Args:
        *modules (nn.Module): the modules to get the parameters from.

    Keyword Args:
        as_module (bool, optional): if ``True``, a
            :class:`~tensordict.nn.TensorDictParams` instance is returned.
            Defaults to ``False``.
        lock (bool, optional): if ``True``, the resulting tensordict is
            locked. Defaults to ``True``.
        use_state_dict (bool, optional): if ``True``, the state-dict of each
            module is used. Defaults to ``False``.
        lazy_stack (bool, optional): if ``True``, the parameters are stacked
            in a :class:`~tensordict.LazyStackedTensorDict` that keeps the
            original parameters. If ``False``, they are stacked into new
            parameters. Exclusive with ``as_module``. Defaults to ``False``.
        expand_identical (bool, optional): if ``True``, a parameter that is
            the same object in every module is expanded instead of stacked.
            Ignored when ``lazy_stack=True``. Defaults to ``False``.

    Returns:
        a :class:`~tensordict.TensorDict` with batch size ``[len(modules)]``,
        a :class:`~tensordict.LazyStackedTensorDict` if ``lazy_stack=True``,
        or a :class:`~tensordict.nn.TensorDictParams` if ``as_module=True``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> import tensordict
        >>> modules = [nn.Linear(3, 4) for _ in range(2)]
        >>> params = tensordict.from_modules(*modules)
        >>> params.batch_size
        torch.Size([2])
        >>> params["weight"].shape
        torch.Size([2, 4, 3])

    """
    return _tensordict_cls().from_modules(
        *modules,
        lazy_stack=lazy_stack,
        expand_identical=expand_identical,
        lock=lock,
        use_state_dict=use_state_dict,
        as_module=as_module,
    )


def from_pytree(
    pytree,
    *,
    batch_size: torch.Size | None = None,
    auto_batch_size: bool = False,
    batch_dims: int | None = None,
):
    """Converts a pytree to a TensorDict instance.

    Lists, tuples, named tuples and dicts become nested tensordicts, and
    non-tensor entries record their types, so
    :meth:`~tensordict.TensorDictBase.to_pytree` can rebuild the pytree. Same
    as :meth:`tensordict.TensorDict.from_pytree`: see that page for the
    details.

    Args:
        pytree: the pytree to convert.

    Keyword Args:
        batch_size (torch.Size, optional): the batch size of the result.
        auto_batch_size (bool, optional): if ``True``, the batch size is the
            largest leading shape that all leaves share. Defaults to
            ``False``.
        batch_dims (int, optional): used with ``auto_batch_size``: the
            number of batch dimensions to keep.

    Returns:
        a :class:`~tensordict.TensorDict`.

    Examples:
        >>> import torch
        >>> import tensordict
        >>> td = tensordict.from_pytree(
        ...     {"a": torch.zeros(3), "b": (torch.ones(3), torch.ones(3, 2))},
        ...     auto_batch_size=True,
        ... )
        >>> td.batch_size
        torch.Size([3])
        >>> td["b", "1"].shape
        torch.Size([3, 2])
        >>> type(td.to_pytree()["b"]).__name__
        'tuple'

    """
    return _tensordict_cls().from_pytree(
        pytree,
        batch_size=batch_size,
        auto_batch_size=auto_batch_size,
        batch_dims=batch_dims,
    )


def stack(input: Sequence["T"], dim: int = 0, *, out=None) -> "T":
    """Stacks tensordicts into one dense tensordict along a new dimension ``dim``.

    Same as ``torch.stack(input, dim)``. Every input must have the same keys,
    entry shapes and batch size. If they do not, this function raises a
    ``RuntimeError``. To fall back to a lazy stack when the inputs differ, use
    :func:`~tensordict.maybe_dense_stack`. To always get a lazy stack, use
    :func:`~tensordict.lazy_stack`. :ref:`stacking` compares these functions.

    Args:
        input (sequence of TensorDictBase or tensorclass instances): the
            tensordicts to stack.
        dim (int, optional): the new dimension. Defaults to ``0``.

    Keyword Args:
        out (TensorDictBase, optional): a tensordict to write the result
            into. Its batch size must be the stacked batch size.

    Returns:
        a :class:`~tensordict.TensorDict` whose batch size is
        ``batch_size[:dim] + [len(input)] + batch_size[dim:]``. Tensorclass
        inputs give an instance of their class. If every input is a
        :class:`~tensordict.LazyStackedTensorDict`, the result is a
        :class:`~tensordict.LazyStackedTensorDict`.

    Examples:
        >>> import torch
        >>> import tensordict
        >>> from tensordict import TensorDict
        >>> tds = [TensorDict(a=torch.zeros(3), batch_size=[3]) for _ in range(2)]
        >>> td = tensordict.stack(tds, dim=1)
        >>> td.batch_size
        torch.Size([3, 2])
        >>> type(td).__name__
        'TensorDict'

    """
    return torch.stack(input, dim=dim, out=out)


def lazy_stack(input: Sequence["T"], dim: int = 0, *, out=None, **kwargs) -> "T":
    """Stacks tensordicts in a :class:`~tensordict.LazyStackedTensorDict`.

    The result keeps each input as a separate tensordict, so the inputs can
    have different keys, entry shapes and batch sizes. Use it for data whose
    size changes from one sample to the next. :ref:`stacking` compares this
    function with :func:`~tensordict.stack` and
    :func:`~tensordict.maybe_dense_stack`. Same as
    :meth:`tensordict.LazyStackedTensorDict.lazy_stack`, which takes the same
    arguments.

    Args:
        input (sequence of TensorDictBase or tensorclass instances): the
            tensordicts to stack. It must not be empty. The batch sizes must
            have the same number of dimensions. If a dimension differs
            between inputs, it is ``-1`` in the stacked batch size.
        dim (int, optional): the new dimension. Defaults to ``0``.

    Keyword Args:
        out (TensorDictBase, optional): a lazy stack to write the result
            into. Its batch size must be the stacked batch size.
        stack_dim_name (str, optional): a name for the stacked dimension.
        strict_shape (bool, optional): if ``True``, the batch sizes of the
            inputs must be equal, or a ``RuntimeError`` is raised. Defaults
            to ``False``.

    Returns:
        a :class:`~tensordict.LazyStackedTensorDict`. Tensorclass inputs give
        an instance of their class that holds a lazy stack.

    Examples:
        >>> import torch
        >>> import tensordict
        >>> from tensordict import TensorDict
        >>> td = tensordict.lazy_stack(
        ...     [TensorDict(a=torch.zeros(3), batch_size=[3]),
        ...      TensorDict(a=torch.zeros(4), batch_size=[4])]
        ... )
        >>> type(td).__name__
        'LazyStackedTensorDict'
        >>> td.batch_size
        torch.Size([2, -1])
        >>> td[1]["a"].shape
        torch.Size([4])

    """
    return _tensordict_cls().lazy_stack(input, dim=dim, out=out, **kwargs)


def cat(input: Sequence["T"], dim: int = 0, *, out=None) -> "T":
    """Concatenates tensordicts along an existing batch dimension ``dim``.

    Same as ``torch.cat(input, dim)``. Every input must have the same keys,
    and the same batch size and entry shapes except along ``dim``. If the
    keys differ, this function raises a ``KeyError``. If the shapes do not
    match, it raises a ``RuntimeError``. To join tensordicts along a new
    dimension, use :func:`~tensordict.stack` or
    :func:`~tensordict.lazy_stack` (see :ref:`stacking`).

    Args:
        input (sequence of TensorDictBase or tensorclass instances): the
            tensordicts to concatenate.
        dim (int, optional): the batch dimension to concatenate along.
            Defaults to ``0``.

    Keyword Args:
        out (TensorDictBase, optional): a tensordict to write the result
            into.

    Returns:
        a :class:`~tensordict.TensorDict`. Tensorclass inputs give an
        instance of their class. If every input is a
        :class:`~tensordict.LazyStackedTensorDict`, the result is a
        :class:`~tensordict.LazyStackedTensorDict`.

    Examples:
        >>> import torch
        >>> import tensordict
        >>> from tensordict import TensorDict
        >>> td = tensordict.cat(
        ...     [TensorDict(a=torch.zeros(3), batch_size=[3]),
        ...      TensorDict(a=torch.ones(4), batch_size=[4])]
        ... )
        >>> td.batch_size
        torch.Size([7])

    """
    return torch.cat(input, dim=dim, out=out)


def maybe_dense_stack(input: Sequence["T"], dim: int = 0, *, out=None, **kwargs) -> "T":
    """Stacks tensordicts densely if possible, or in a lazy stack otherwise.

    Use it when the inputs usually match, but can differ. If every input has
    the same keys, entry shapes and batch size, the result is the dense
    stack that :func:`~tensordict.stack` returns. Otherwise, the result is the
    lazy stack that :func:`~tensordict.lazy_stack` returns. :ref:`stacking`
    compares these functions. Same as
    :meth:`tensordict.LazyStackedTensorDict.maybe_dense_stack`.

    Args:
        input (sequence of TensorDictBase or tensorclass instances): the
            tensordicts to stack.
        dim (int, optional): the new dimension. Defaults to ``0``.

    Keyword Args:
        out (TensorDictBase, optional): a tensordict to write the result
            into.
        strict (bool, optional): used only with ``out``. If ``True``, ``out``
            must have the same keys as the inputs. Defaults to ``False``.

    Returns:
        a :class:`~tensordict.TensorDict` (or the tensorclass of the inputs)
        if the inputs can be stacked densely, a
        :class:`~tensordict.LazyStackedTensorDict` otherwise.

    Examples:
        >>> import torch
        >>> import tensordict
        >>> from tensordict import TensorDict
        >>> same = [TensorDict(a=torch.zeros(3)), TensorDict(a=torch.ones(3))]
        >>> type(tensordict.maybe_dense_stack(same)).__name__
        'TensorDict'
        >>> different = [TensorDict(a=torch.zeros(3)), TensorDict(a=torch.ones(4))]
        >>> type(tensordict.maybe_dense_stack(different)).__name__
        'LazyStackedTensorDict'

    """
    return _tensordict_cls().maybe_dense_stack(input, dim=dim, out=out, **kwargs)


def fromkeys(keys: list[NestedKey], value: Any = 0):
    """Creates a tensordict from a list of keys and a single value.

    Same as :meth:`tensordict.TensorDict.fromkeys`, and as :meth:`dict.fromkeys`
    for a dict. The batch size of the result is ``[]``.

    Args:
        keys (list of NestedKey): the keys of the new tensordict. A tuple of
            strings creates a nested entry.
        value (compatible type, optional): the value of every entry. Defaults
            to ``0``.

    Returns:
        a :class:`~tensordict.TensorDict`.

    Examples:
        >>> import tensordict
        >>> td = tensordict.fromkeys(["a", ("b", "c")], 1)
        >>> td["b", "c"]
        tensor(1)

    """
    return _tensordict_cls().fromkeys(keys=keys, value=value)


def from_consolidated(filename):
    """Reconstructs a tensordict from a consolidated file.

    A consolidated file is the single file that
    :meth:`~tensordict.TensorDictBase.consolidate` writes when you pass it a
    ``filename``. Same as :meth:`tensordict.TensorDict.from_consolidated`.

    Args:
        filename (str or Path): the path of the file.

    Returns:
        a :class:`~tensordict.TensorDict`. Its entries are views into one
        storage that is read from the file.

    Examples:
        >>> import tempfile
        >>> import torch
        >>> import tensordict
        >>> from tensordict import TensorDict
        >>> td = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3)}, batch_size=[3])
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _ = td.consolidate(filename=f"{tmpdir}/td.bin")
        ...     loaded = tensordict.from_consolidated(f"{tmpdir}/td.bin")
        ...     print(loaded["b", "c"])
        tensor([1., 1., 1.])

    """
    return _tensordict_cls().from_consolidated(filename)


def load(
    prefix: str | Path,
    device: torch.device | None = None,
    non_blocking: bool = False,
    *,
    out: TensorCollection | None = None,
    robust_key: bool | None = True,
    subpath: NestedKey | None = None,
    mode: str | None = None,
    num_threads: int = 0,
    allow_pickle: bool | None = None,
) -> TensorCollection:
    """Loads a tensordict that :func:`~tensordict.save` wrote.

    Same as :func:`~tensordict.load_memmap`, which takes the same arguments.
    See :meth:`tensordict.TensorDict.load_memmap` for every argument, and
    :ref:`saving` for the ways to save and load a tensordict.

    Args:
        prefix (str or Path): the directory, or the ``.tdz`` archive file,
            that holds the saved tensordict.
        device (torch.device, optional): if given, the data is cast to that
            device. With ``"meta"``, no file is read.
        non_blocking (bool, optional): if ``True``, the cast to ``device``
            is not followed by a synchronization. Defaults to ``False``.

    Keyword Args:
        out (TensorDictBase, optional): a tensordict to load the data into.
        robust_key (bool, optional): must match the value used to save.
            Defaults to ``True``.
        subpath (NestedKey or str, optional): the nested tensordict to load,
            as a nested key (``("module", "0")``) or a ``"/"``-separated path
            (``"module/0"``). Only that subtree is read.
        mode (str, optional): ``"r"`` maps the files copy-on-write (in-place
            writes stay in memory), ``"r+"`` maps them shared (in-place writes
            reach the files). See :meth:`tensordict.TensorDict.load_memmap`
            for the default.
        num_threads (int, optional): the number of threads that decompress
            the leaves of a compressed archive. It has no effect on other
            data. Defaults to ``0``.
        allow_pickle (bool, optional): if ``True``, non-tensor entries that
            were pickled are unpickled. Only load pickles from trusted
            sources.

    Returns:
        the saved tensordict, of the class it had when it was saved. Its
        leaves are :class:`~tensordict.MemoryMappedTensor` instances.

    Examples:
        >>> import tempfile
        >>> import torch
        >>> import tensordict
        >>> from tensordict import TensorDict
        >>> td = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3, 2)}, batch_size=[3])
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _ = tensordict.save(td, tmpdir)
        ...     loaded = tensordict.load(tmpdir)
        ...     print(loaded["b", "c"].shape)
        torch.Size([3, 2])

    """
    return load_memmap(
        prefix=prefix,
        device=device,
        non_blocking=non_blocking,
        out=out,
        robust_key=robust_key,
        subpath=subpath,
        mode=mode,
        num_threads=num_threads,
        allow_pickle=allow_pickle,
    )


def load_memmap(
    prefix: str | Path,
    device: torch.device | None = None,
    non_blocking: bool = False,
    *,
    out: TensorCollection | None = None,
    robust_key: bool | None = True,
    subpath: NestedKey | None = None,
    mode: str | None = None,
    num_threads: int = 0,
    allow_pickle: bool | None = None,
) -> TensorCollection:
    """Loads a memory-mapped tensordict from disk.

    Same as :func:`~tensordict.load`, and as
    :meth:`tensordict.TensorDict.load_memmap`, which holds the full text of
    every argument. See :ref:`saving` for the ways to save and load a
    tensordict.

    Args:
        prefix (str or Path): the directory, or the ``.tdz`` archive file,
            that holds the saved tensordict.
        device (torch.device, optional): if given, the data is cast to that
            device. With ``"meta"``, no file is read.
        non_blocking (bool, optional): if ``True``, the cast to ``device``
            is not followed by a synchronization. Defaults to ``False``.

    Keyword Args:
        out (TensorDictBase, optional): a tensordict to load the data into.
        robust_key (bool, optional): must match the value used to save.
            Defaults to ``True``.
        subpath (NestedKey or str, optional): the nested tensordict to load,
            as a nested key (``("module", "0")``) or a ``"/"``-separated path
            (``"module/0"``). Only that subtree is read.
        mode (str, optional): ``"r"`` maps the files copy-on-write (in-place
            writes stay in memory), ``"r+"`` maps them shared (in-place writes
            reach the files). See :meth:`tensordict.TensorDict.load_memmap`
            for the default.
        num_threads (int, optional): the number of threads that decompress
            the leaves of a compressed archive. It has no effect on other
            data. Defaults to ``0``.
        allow_pickle (bool, optional): if ``True``, non-tensor entries that
            were pickled are unpickled. Only load pickles from trusted
            sources.

    Returns:
        the saved tensordict, of the class it had when it was saved. Its
        leaves are :class:`~tensordict.MemoryMappedTensor` instances.

    Examples:
        >>> import tempfile
        >>> import torch
        >>> import tensordict
        >>> from tensordict import TensorDict
        >>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _ = tensordict.memmap(td, tmpdir)
        ...     loaded = tensordict.load_memmap(tmpdir)
        ...     print(type(loaded["a"]).__name__)
        MemoryMappedTensor

    """
    return _tensordict_cls().load_memmap(
        prefix=prefix,
        device=device,
        non_blocking=non_blocking,
        out=out,
        robust_key=robust_key,
        subpath=subpath,
        mode=mode,
        num_threads=num_threads,
        allow_pickle=allow_pickle,
    )


def save(
    data: TensorCollection,
    prefix: str | None = None,
    copy_existing: bool = False,
    *,
    num_threads: int = 0,
    return_early: bool = False,
    share_non_tensor: bool = False,
    robust_key: bool | None = True,
    archive: bool | None = None,
    compression: str | int | None = None,
) -> TensorCollection:
    """Saves a tensordict to disk, as memory-mapped files.

    Same as :func:`~tensordict.memmap`, and as
    :meth:`~tensordict.TensorDictBase.memmap`, which holds the full text of
    every argument. The directory mirrors the key structure of the
    tensordict: one file per tensor and one metadata file per nested
    tensordict. If ``prefix`` ends with ``".tdz"``, a single archive file is
    written instead. Load the result with :func:`~tensordict.load`. See
    :ref:`saving`.

    Args:
        data (TensorDictBase or tensorclass instance): the tensordict to save.
        prefix (str, optional): the directory to write to, or a file name that
            ends with ``".tdz"``. If ``None``, the tensors are written to
            anonymous shared memory and nothing is saved to disk.
        copy_existing (bool, optional): if ``False``, an entry that is already
            memory-mapped to a file outside ``prefix`` raises an error. If
            ``True``, it is copied. Defaults to ``False``.

    Keyword Args:
        num_threads (int, optional): the number of threads that write the
            files. Defaults to ``0``.
        return_early (bool, optional): if ``True`` and ``num_threads > 0``,
            a future of the result is returned. Defaults to ``False``.
        share_non_tensor (bool, optional): if ``True``, non-tensor entries are
            shared between the processes of a node. Defaults to ``False``.
        robust_key (bool, optional): if ``True``, keys that hold path
            separators or special characters are encoded in the file names.
            Defaults to ``True``.
        archive (bool, optional): if ``True``, ``prefix`` is a file and the
            tensordict is written as a single zip archive that
            :func:`~tensordict.load` can memory-map. If ``None``, an archive
            is written when ``prefix`` ends with ``".tdz"``. Defaults to
            ``None``.
        compression (str or int, optional): the compression of the archive
            entries: ``"stored"``, ``"deflate"``, ``"bzip2"``, ``"lzma"`` or a
            :mod:`zipfile` constant. Only valid for an archive. Defaults to
            ``None``, which stores them uncompressed. Only uncompressed
            entries load without a copy.

    Returns:
        a new, locked tensordict whose leaves are
        :class:`~tensordict.MemoryMappedTensor` instances backed by the
        saved files. ``data`` itself is not changed.

    Examples:
        >>> import tempfile
        >>> import torch
        >>> import tensordict
        >>> from tensordict import TensorDict
        >>> td = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3, 2)}, batch_size=[3])
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     saved = tensordict.save(td, tmpdir)
        ...     print(saved.is_locked, type(saved["a"]).__name__)
        ...     loaded = tensordict.load(tmpdir)
        True MemoryMappedTensor

    """
    return data.memmap(
        prefix=prefix,
        copy_existing=copy_existing,
        num_threads=num_threads,
        return_early=return_early,
        share_non_tensor=share_non_tensor,
        robust_key=robust_key,
        archive=archive,
        compression=compression,
    )


def memmap(
    data: TensorCollection,
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
) -> TensorCollection:
    """Writes all tensors onto memory-mapped tensors in a new tensordict.

    Same as :func:`~tensordict.save`, and as
    :meth:`~tensordict.TensorDictBase.memmap`, which holds the full text of
    every argument. Use it to share a tensordict between processes without
    copies, or to keep data that is larger than RAM on disk. To convert the
    tensordict in place, use :meth:`~tensordict.TensorDictBase.memmap_`.
    See :ref:`saving` and :ref:`data-loading`.

    Args:
        data (TensorDictBase or tensorclass instance): the tensordict to write.
        prefix (str, optional): the directory to write to, or a file name that
            ends with ``".tdz"``. If ``None``, the tensors are written to
            anonymous shared memory.
        copy_existing (bool, optional): if ``False``, an entry that is already
            memory-mapped to a file outside ``prefix`` raises an error. If
            ``True``, it is copied. Defaults to ``False``.

    Keyword Args:
        num_threads (int, optional): the number of threads that write the
            files. Defaults to ``0``.
        return_early (bool, optional): if ``True`` and ``num_threads > 0``,
            a future of the result is returned. Defaults to ``False``.
        share_non_tensor (bool, optional): if ``True``, non-tensor entries are
            shared between the processes of a node. Defaults to ``False``.
        existsok (bool, optional): if ``False``, a tensor file that already
            exists under ``prefix`` raises an error. Defaults to ``True``.
        robust_key (bool, optional): if ``True``, keys that hold path
            separators or special characters are encoded in the file names.
            Defaults to ``True``.
        archive (bool, optional): if ``True``, ``prefix`` is a file and the
            tensordict is written as a single zip archive that
            :func:`~tensordict.load` can memory-map. If ``None``, an archive
            is written when ``prefix`` ends with ``".tdz"``. Defaults to
            ``None``.
        compression (str or int, optional): the compression of the archive
            entries: ``"stored"``, ``"deflate"``, ``"bzip2"``, ``"lzma"`` or a
            :mod:`zipfile` constant. Only valid for an archive. Defaults to
            ``None``, which stores them uncompressed. Only uncompressed
            entries load without a copy.

    Returns:
        a new, locked tensordict whose leaves are
        :class:`~tensordict.MemoryMappedTensor` instances.

    Examples:
        >>> import torch
        >>> import tensordict
        >>> from tensordict import TensorDict
        >>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
        >>> td_memmap = tensordict.memmap(td)
        >>> type(td_memmap["a"]).__name__
        'MemoryMappedTensor'
        >>> td_memmap.is_locked
        True

    """
    return data.memmap(
        prefix=prefix,
        copy_existing=copy_existing,
        num_threads=num_threads,
        return_early=return_early,
        share_non_tensor=share_non_tensor,
        existsok=existsok,
        robust_key=robust_key,
        archive=archive,
        compression=compression,
    )


for _name in __all__:
    globals()[_name].__module__ = "tensordict._td"
