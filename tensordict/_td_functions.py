# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

import torch
from tensordict._archive import _tensor_chunks
from tensordict._nestedkey import NestedKey
from tensordict._td_memmap import _memmap_filename
from tensordict._tensorcollection import TensorCollection
from tensordict._utils_key_json import json_dumps
from tensordict.utils import is_non_tensor, is_tensorclass

if TYPE_CHECKING:
    from tensordict.base import T

__all__ = [
    "append_memmap",
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
    """Returns a locked tensordict that holds the parameters and buffers of a module.

    Same as :meth:`tensordict.TensorDict.from_module`, except for the default
    of ``lock``: this function locks the result by default, and the method
    does not. The entries are the parameters and buffers of the
    module themselves, not copies. Submodules become nested tensordicts.

    Args:
        module (torch.nn.Module): the module to get the parameters from.
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
        A :class:`~tensordict.TensorDict`, or a
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
        *modules (torch.nn.Module): the modules to get the parameters from.

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
        A :class:`~tensordict.TensorDict` with batch size ``[len(modules)]``,
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
    """Converts a pytree (nested lists, tuples and dicts of tensors) to a tensordict.

    Lists, tuples, named tuples and dicts become nested tensordicts, and
    non-tensor entries record their types, so
    :meth:`~tensordict.TensorDictBase.to_pytree` can rebuild the pytree. Same
    as :meth:`tensordict.TensorDict.from_pytree`: see that page for the
    details.

    Args:
        pytree (Any): the pytree to convert.

    Keyword Args:
        batch_size (torch.Size or list of int, optional): the batch size of
            the result. Defaults to ``None``, which gives ``[]`` unless
            ``auto_batch_size`` is ``True``.
        auto_batch_size (bool, optional): if ``True``, sets the batch size to
            the largest leading shape that all the leaves share. Defaults to
            ``False``.
        batch_dims (int, optional): with ``auto_batch_size``, the maximum
            number of batch dimensions. Ignored without
            ``auto_batch_size``, with a ``FutureWarning`` when it would
            change the result. Starting with TensorDict 0.17, the default
            will change to turning on ``auto_batch_size`` when
            ``batch_dims`` is given, as ``from_dict`` does. Defaults to
            ``None``.

    Returns:
        A :class:`~tensordict.TensorDict`.

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
        dim (int, optional): the new dimension. Defaults to ``0``. A ``dim`` outside
            ``[-batch_dims - 1, batch_dims]`` warns, and is used on the dimensions
            of the leaves. Starting with TensorDict 0.17, the default will
            change to raising an ``IndexError``, as for a tensor.

    Keyword Args:
        out (TensorDictBase, optional): a tensordict to write the result
            into. Its batch size must be the stacked batch size. Defaults to
            ``None``.

    Returns:
        A :class:`~tensordict.TensorDict` whose batch size is
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
    """Stacks tensordicts into a lazy stack (:class:`~tensordict.LazyStackedTensorDict`).

    The result keeps each input as a separate tensordict, so the inputs can
    have different keys, entry shapes and batch sizes. Use it for data whose
    size changes from one sample to the next. :ref:`stacking` compares this
    function with :func:`~tensordict.stack` and
    :func:`~tensordict.maybe_dense_stack`. Same as
    :meth:`tensordict.LazyStackedTensorDict.lazy_stack`, which takes the same
    arguments and documents the ones for lists of tensors.

    Args:
        input (sequence of TensorDictBase or tensorclass instances): the
            tensordicts to stack. It must not be empty. The batch sizes must
            have the same number of dimensions. If a dimension differs
            between inputs, it is ``-1`` in the stacked batch size.
        dim (int, optional): the new dimension. Defaults to ``0``. A ``dim``
            below ``-batch_dims - 1`` warns and wraps around once more (and
            raises if it is still out of range).
            Starting with TensorDict 0.17, the default will change to raising
            an ``IndexError``, as for a tensor.

    Keyword Args:
        out (TensorDictBase, optional): a lazy stack to write the result
            into. Its batch size must be the stacked batch size. Defaults to
            ``None``.
        stack_dim_name (str, optional): the name of the stack dimension.
            Defaults to ``None``.
        strict_shape (bool, optional): if ``True``, the batch sizes of the
            inputs must be equal, or a ``RuntimeError`` is raised. Defaults
            to ``False``.

    Returns:
        A :class:`~tensordict.LazyStackedTensorDict`. Tensorclass inputs give
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
            Defaults to ``0``. A ``dim`` below ``-batch_dims`` warns, and is
            used on the dims of the leaves. Starting with TensorDict 0.17,
            the default will change to raising an ``IndexError``, as for a
            tensor.

    Keyword Args:
        out (TensorDictBase, optional): a tensordict to write the result
            into. Defaults to ``None``.

    Returns:
        A :class:`~tensordict.TensorDict`. Tensorclass inputs give an
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
    """Stacks tensordicts densely if possible, or into a lazy stack otherwise.

    Use it when the inputs usually match, but can differ. If every input has
    the same keys, entry shapes and batch size, the result is the dense
    stack that :func:`~tensordict.stack` returns. Otherwise, the result is the
    lazy stack that :func:`~tensordict.lazy_stack` returns. :ref:`stacking`
    compares these functions. Same as
    :meth:`tensordict.LazyStackedTensorDict.maybe_dense_stack`.

    Args:
        input (sequence of TensorDictBase or tensorclass instances): the
            tensordicts to stack.
        dim (int, optional): the new dimension. Defaults to ``0``. A ``dim`` outside
            ``[-batch_dims - 1, batch_dims]`` warns, and is used on the dimensions
            of the leaves. Starting with TensorDict 0.17, the default will
            change to raising an ``IndexError``, as for a tensor.

    Keyword Args:
        out (TensorDictBase, optional): a tensordict to write the result
            into. Defaults to ``None``.
        strict (bool, optional): used only with ``out``. If ``True``, ``out``
            must have the same keys as the inputs. Defaults to ``False``.

    Returns:
        A :class:`~tensordict.TensorDict` (or the tensorclass of the inputs)
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
    """Creates a tensordict with the same value for every key, like :meth:`dict.fromkeys`.

    Same as :meth:`tensordict.TensorDict.fromkeys`, and as :meth:`dict.fromkeys`
    for a dict. The batch size of the result is ``[]``.

    Args:
        keys (list of NestedKey): the keys of the new tensordict. A tuple of
            strings creates a nested entry.
        value (torch.Tensor, number or Any, optional): the value of every
            entry. Defaults to ``0``.

    Returns:
        A :class:`~tensordict.TensorDict`.

    Examples:
        >>> import tensordict
        >>> td = tensordict.fromkeys(["a", ("b", "c")], 1)
        >>> td["b", "c"]
        tensor(1)

    """
    return _tensordict_cls().fromkeys(keys=keys, value=value)


def from_consolidated(filename):
    """Reconstructs a tensordict from a file written by ``consolidate(filename=...)``.

    A consolidated file is the single file that
    :meth:`~tensordict.TensorDictBase.consolidate` writes when you pass it a
    ``filename``. Same as :meth:`tensordict.TensorDict.from_consolidated`.

    The batch sizes and names are as they were when the file was written.
    To write later changes into the file, call
    ``consolidate(filename=filename)`` again on the consolidated tensordict.

    Args:
        filename (str or Path): the path of the file.

    Returns:
        A :class:`~tensordict.TensorDict`. Its entries are views into one
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
        device (torch.device, str or int, optional): if given, the data is
            cast to that device. With ``"meta"``, no file is read. Defaults
            to ``None``.
        non_blocking (bool, optional): if ``True``, the cast to ``device``
            is not followed by a synchronization. Defaults to ``False``.

    Keyword Args:
        out (TensorDictBase, optional): a tensordict to load the data into.
            Defaults to ``None``.
        robust_key (bool, optional): if ``True``, reads the percent-encoded
            file names that :func:`~tensordict.save` writes by default, and
            falls back to the legacy names of keys that are safe file names.
            If ``False``, reads the legacy layout only. Defaults to ``True``.
        subpath (NestedKey or str, optional): the nested tensordict to load,
            as a nested key (``("module", "0")``) or a ``"/"``-separated path
            (``"module/0"``). Only that subtree is read. Defaults to
            ``None``, which loads the whole tensordict.
        mode (str, optional): ``"r"`` maps the files copy-on-write (in-place
            writes stay in memory), ``"r+"`` maps them shared (in-place writes
            reach the files). See :meth:`tensordict.TensorDict.load_memmap`
            for the default.
        num_threads (int, optional): the number of threads that decompress
            the leaves of a compressed archive. It has no effect on other
            data. Defaults to ``0``.
        allow_pickle (bool, optional): if ``True``, non-tensor entries that
            were pickled are unpickled. Only load pickles from trusted
            sources. If ``False``, pickled entries raise a ``RuntimeError``. Defaults
            to ``None``, which unpickles them with a ``FutureWarning``.

    Returns:
        The saved tensordict, of the class it had when it was saved. Its
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
    """Loads a tensordict that :func:`~tensordict.memmap` or :func:`~tensordict.save` wrote.

    Same as :func:`~tensordict.load`, and as
    :meth:`tensordict.TensorDict.load_memmap`, which holds the full text of
    every argument. See :ref:`saving` for the ways to save and load a
    tensordict.

    Args:
        prefix (str or Path): the directory, or the ``.tdz`` archive file,
            that holds the saved tensordict.
        device (torch.device, str or int, optional): if given, the data is
            cast to that device. With ``"meta"``, no file is read. Defaults
            to ``None``.
        non_blocking (bool, optional): if ``True``, the cast to ``device``
            is not followed by a synchronization. Defaults to ``False``.

    Keyword Args:
        out (TensorDictBase, optional): a tensordict to load the data into.
            Defaults to ``None``.
        robust_key (bool, optional): if ``True``, reads the percent-encoded
            file names that :func:`~tensordict.save` writes by default, and
            falls back to the legacy names of keys that are safe file names.
            If ``False``, reads the legacy layout only. Defaults to ``True``.
        subpath (NestedKey or str, optional): the nested tensordict to load,
            as a nested key (``("module", "0")``) or a ``"/"``-separated path
            (``"module/0"``). Only that subtree is read. Defaults to
            ``None``, which loads the whole tensordict.
        mode (str, optional): ``"r"`` maps the files copy-on-write (in-place
            writes stay in memory), ``"r+"`` maps them shared (in-place writes
            reach the files). See :meth:`tensordict.TensorDict.load_memmap`
            for the default.
        num_threads (int, optional): the number of threads that decompress
            the leaves of a compressed archive. It has no effect on other
            data. Defaults to ``0``.
        allow_pickle (bool, optional): if ``True``, non-tensor entries that
            were pickled are unpickled. Only load pickles from trusted
            sources. If ``False``, pickled entries raise a ``RuntimeError``. Defaults
            to ``None``, which unpickles them with a ``FutureWarning``.

    Returns:
        The saved tensordict, of the class it had when it was saved. Its
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


def append_memmap(
    prefix: str | Path,
    rows: TensorCollection,
    *,
    allow_pickle: bool | None = None,
) -> None:
    """Appends rows to a tensordict saved in a directory, along its first batch dimension.

    Each leaf file grows by the bytes of the new rows, and the ``meta.json``
    files are rewritten last, each one through a temporary file, so that a
    failed append leaves the saved data as it was. Readers that mapped the
    files before the append keep seeing the old number of rows, and
    :func:`~tensordict.load_memmap` after the append sees the new one. Use one
    writer at a time. See :ref:`saving`.

    ``rows`` must have the keys, the classes, the dtypes and the trailing
    shapes of the saved tensordict, and its batch size may differ from the
    saved one in its first dimension only. A non-tensor entry stays as
    saved, and its value in ``rows`` must be equal to it. A
    :class:`~tensordict.NonTensorStack`, a lazy stack, a nested tensor and an
    archive cannot grow, and raise. Everything is checked before the first
    file is written.

    Args:
        prefix (str or Path): the directory that :func:`~tensordict.save` or
            :func:`~tensordict.memmap` wrote.
        rows (TensorDictBase or tensorclass): the rows to append.

    Keyword Args:
        allow_pickle (bool, optional): whether the pickled non-tensor entries
            of the saved tensordict, which are compared with those of
            ``rows``, may be unpickled, as in
            :func:`~tensordict.load_memmap`. Defaults to ``None``.

    Raises:
        ValueError: if ``prefix`` is not a directory, or if ``rows`` cannot be
            appended to the saved tensordict.

    Examples:
        >>> import tempfile
        >>> import torch
        >>> import tensordict
        >>> from tensordict import TensorDict
        >>> td = TensorDict(a=torch.zeros(3, 2), batch_size=[3])
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _ = tensordict.save(td, tmpdir)
        ...     tensordict.append_memmap(tmpdir, TensorDict(a=torch.ones(2, 2), batch_size=[2]))
        ...     print(tensordict.load_memmap(tmpdir)["a"].sum(-1).tolist())
        [0.0, 0.0, 0.0, 2.0, 2.0]

    """
    from tensordict.base import is_tensor_collection

    prefix = Path(prefix)
    if not prefix.is_dir():
        raise ValueError(
            f"append_memmap appends to a directory written by save or memmap, "
            f"and {prefix} is not one. An archive cannot grow: unpack it with "
            f"unpack_memmap first."
        )
    if not is_tensor_collection(rows):
        raise TypeError(
            f"rows must be a tensordict or a tensorclass, got {type(rows)}."
        )
    saved = load_memmap(prefix, allow_pickle=allow_pickle)
    if not saved.batch_dims or not rows.batch_dims:
        raise ValueError(
            "append_memmap appends along the first batch dimension, which the "
            f"saved tensordict (batch size {saved.batch_size}) and the rows "
            f"(batch size {rows.batch_size}) must both have."
        )
    leaves = []
    metadata_files = []
    _check_append(saved, rows, (), leaves, metadata_files)
    num_rows = saved.batch_size[0] + rows.batch_size[0]
    del saved

    # Grow the leaf files. On an error, they get their old sizes back.
    old_sizes = {}
    replacements = []
    try:
        for filename, old_nbytes, value in leaves:
            if not value.numel():
                continue
            old_sizes[filename] = (
                os.path.getsize(filename) if os.path.exists(filename) else None
            )
            with open(filename, "wb" if old_sizes[filename] is None else "r+b") as f:
                # A file can be longer than its leaf, as after a failed
                # append: write the rows right after the saved ones.
                f.seek(old_nbytes)
                for chunk in _tensor_chunks(value):
                    f.write(chunk)
        # Write every new meta.json before the first one replaces the old.
        for metadata_file in metadata_files:
            with open(metadata_file, "rb") as f:
                metadata = json.loads(f.read())
            metadata["shape"][0] = num_rows
            for entry in metadata.values():
                if isinstance(entry, dict) and "dtype" in entry and "shape" in entry:
                    entry["shape"][0] = num_rows
            json_str = json_dumps(metadata)
            tmp_file = metadata_file.with_name(f".{metadata_file.name}.append.tmp")
            replacements.append((tmp_file, metadata_file))
            with open(tmp_file, "wb") as f:
                f.write(
                    json_str.encode("utf-8") if isinstance(json_str, str) else json_str
                )
    except BaseException:
        for filename, size in old_sizes.items():
            if size is None:
                os.remove(filename)
            else:
                os.truncate(filename, size)
        for tmp_file, _ in replacements:
            if tmp_file.exists():
                os.remove(tmp_file)
        raise
    for tmp_file, metadata_file in replacements:
        os.replace(tmp_file, metadata_file)


def _check_append(saved, rows, key, leaves, metadata_files) -> None:
    """Checks that ``rows`` can be appended to the loaded tensordict ``saved``.

    Appends ``(filename, saved_nbytes, value)`` to ``leaves`` for each leaf,
    and the path of each ``meta.json`` to rewrite to ``metadata_files``.
    """
    from tensordict._lazy import LazyStackedTensorDict
    from tensordict._td import TensorDict
    from tensordict.base import is_tensor_collection
    from tensordict.tensorclass import NonTensorData

    where = (
        f"the entry {key if len(key) > 1 else key[0]!r}"
        if key
        else "the saved tensordict"
    )
    if rows.batch_size[1:] != saved.batch_size[1:]:
        raise ValueError(
            f"Cannot append rows of batch size {rows.batch_size} to {where}, of "
            f"batch size {saved.batch_size}: they differ beyond the first dimension."
        )
    if is_tensorclass(saved):
        if type(rows) is not type(saved):
            raise ValueError(
                f"Cannot append a {type(rows).__name__} to {where}, a "
                f"{type(saved).__name__}."
            )
        saved, rows = saved._tensordict, rows._tensordict
    elif not isinstance(saved, TensorDict) or is_tensorclass(rows):
        raise ValueError(
            f"Cannot append a {type(rows).__name__} to {where}, a "
            f"{type(saved).__name__}."
        )
    if set(saved.keys()) != set(rows.keys()):
        raise ValueError(
            f"Cannot append rows with the keys {sorted(rows.keys())} to {where}, "
            f"which has the keys {sorted(saved.keys())}."
        )
    prefix = Path(saved._memmap_prefix)
    metadata_files.append(prefix / "meta.json")
    for name in saved.keys():
        entry_key = key + (name,)
        where = f"the entry {entry_key if key else name!r}"
        old = saved._get_str(name, None)
        new = rows._get_str(name, None)
        if is_non_tensor(old):
            if not isinstance(old, NonTensorData):
                raise ValueError(
                    f"Cannot append to {where}, a {type(old).__name__}: a stack "
                    "of non-tensor values cannot grow."
                )
            if not isinstance(new, NonTensorData) or not _non_tensor_equal(
                new.data, old.data
            ):
                raise ValueError(
                    f"Cannot append to {where}: the saved non-tensor value "
                    f"{old.data!r} stays as it is, so the rows must hold the "
                    f"same value, not "
                    f"{new.data if isinstance(new, NonTensorData) else new!r}."
                )
        elif isinstance(old, LazyStackedTensorDict):
            raise ValueError(f"Cannot append to {where}: a lazy stack cannot grow.")
        elif is_tensor_collection(old):
            if not is_tensor_collection(new) or is_non_tensor(new):
                raise ValueError(
                    f"Cannot append a {type(new).__name__} to {where}, a "
                    f"{type(old).__name__}."
                )
            _check_append(old, new, entry_key, leaves, metadata_files)
        elif not isinstance(new, torch.Tensor) or is_tensor_collection(new):
            raise ValueError(
                f"Cannot append a {type(new).__name__} to {where}, a tensor."
            )
        elif old.is_nested or new.is_nested:
            raise ValueError(f"Cannot append to {where}: a nested tensor cannot grow.")
        elif new.dtype != old.dtype or new.shape[1:] != old.shape[1:]:
            raise ValueError(
                f"Cannot append rows of dtype {new.dtype} and shape "
                f"{tuple(new.shape)} to {where}, of dtype {old.dtype} and shape "
                f"{tuple(old.shape)}."
            )
        else:
            # A leaf with no elements has no file.
            filename = getattr(old, "filename", None) or _memmap_filename(
                prefix, name, robust_key=True
            )
            leaves.append((filename, old.numel() * old.dtype.itemsize, new))


def _non_tensor_equal(a, b) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return False


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
    """Saves a tensordict to disk, and returns a memory-mapped copy of it.

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
            ends with ``".tdz"``. Defaults to ``None``, which writes the
            tensors to anonymous shared memory and saves nothing to disk.
        copy_existing (bool, optional): if ``False``, an entry that is already
            memory-mapped to a file outside ``prefix`` raises an error. If
            ``True``, it is copied. Defaults to ``False``.

    Keyword Args:
        num_threads (int, optional): the number of threads that write the
            files. Defaults to ``0``.
        return_early (bool, optional): if ``True`` and ``num_threads > 1``,
            returns a :class:`~tensordict.utils.TensorDictFuture` at once, while
            the threads write the files. Defaults to ``False``.
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
        A new, locked tensordict whose leaves are
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
    """Writes every leaf to a memory-mapped file and returns a new tensordict of them.

    Same as :func:`~tensordict.save`, and as
    :meth:`~tensordict.TensorDictBase.memmap`, which holds the full text of
    every argument. Use it to share a tensordict between processes without
    copies, or to keep data that is larger than RAM on disk. To convert the
    tensordict in place, use :meth:`~tensordict.TensorDictBase.memmap_`.
    See :ref:`saving` and :ref:`data-loading`.

    Args:
        data (TensorDictBase or tensorclass instance): the tensordict to write.
        prefix (str, optional): the directory to write to, or a file name that
            ends with ``".tdz"``. Defaults to ``None``, which writes the
            tensors to anonymous shared memory and saves nothing to disk.
        copy_existing (bool, optional): if ``False``, an entry that is already
            memory-mapped to a file outside ``prefix`` raises an error. If
            ``True``, it is copied. Defaults to ``False``.

    Keyword Args:
        num_threads (int, optional): the number of threads that write the
            files. Defaults to ``0``.
        return_early (bool, optional): if ``True`` and ``num_threads > 1``,
            returns a :class:`~tensordict.utils.TensorDictFuture` at once, while
            the threads write the files. Defaults to ``False``.
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
        A new, locked tensordict whose leaves are
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
