# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Device and dtype conversions, memory placement and memory size of :class:`~tensordict.TensorDictBase`.

The methods live on a mixin that ``tensordict.base`` imports before it
defines ``TensorDictBase``, so this module imports the helpers it needs from
``tensordict.base`` without needing the class itself. A method that uses
``TensorDictBase`` at run time imports it locally. The comment above the
mixin imports in ``tensordict/base.py`` gives the other rules.
"""

from __future__ import annotations

import queue
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Callable, overload

import torch
from tensordict._deprecation import deprecated
from tensordict.base import (
    _default_is_leaf,
    _is_tensor_collection,
    _NESTED_TENSORS_AS_LISTS,
    _sync_cuda_transfer,
    Buffer,
    is_compiling,
    Self,
    T,
)
from tensordict.memmap import MemoryMappedTensor
from tensordict.utils import (
    _as_context_manager,
    _cache_while_locked,
    _cast_filtered,
    _is_shared,
    _is_unbatched,
    _make_dtype_promotion,
    _parse_to,
    _pin_mem,
    _PIN_MEM_TIMEOUT,
    _zip_strict,
    DeviceType,
    is_non_tensor,
    is_tensorclass,
)
from torch import Tensor
from torch._utils import _get_available_device_type, _get_device_module
from torch.nn.parameter import Parameter


def _check_nested_batch_size(td) -> None:
    # Checks the leaves of td and of its nested tensordicts against their batch
    # sizes. Lazy stacks take their batch size from their parts.
    if is_tensorclass(td):
        td = td._tensordict
    if td._lazy:
        return
    td._check_new_batch_size(td.batch_size)
    for value in td.values():
        if _is_tensor_collection(type(value)) and not is_non_tensor(value):
            _check_nested_batch_size(value)


def _to_batch_size_will_change(td, batch_size, root=True) -> bool:
    # Whether to(device, batch_size=batch_size) on td gives a different result
    # from to(device) followed by the batch-size setter, as it will in 0.17.
    # Today every nested tensordict gets exactly batch_size, nested lazy stacks
    # become TensorDicts and all names are erased.
    if is_tensorclass(td):
        td = td._tensordict
    if not root:
        if td._lazy:
            return True
        # the setter only changes nested batch sizes with fewer dims
        if len(td.batch_size) >= len(batch_size) and td.batch_size != batch_size:
            return True
    if td._has_names() and any(
        name is not None for name in td.names[: len(batch_size)]
    ):
        return True
    return any(
        _to_batch_size_will_change(value, batch_size, root=False)
        for value in td.values()
        if _is_tensor_collection(type(value))
    )


def _pin_memory(x):
    return x.pin_memory()


def _pin_memory_nodes(td) -> list | None:
    # The nodes whose leaves pin_memory(inplace=True) replaces by their pinned
    # copies, or None if any node is locked or is not exactly a TensorDict or
    # a lazy stack (a tensorclass counts as the tensordict it wraps), or if
    # any leaf is memory-mapped, shared, not strided or an UnbatchedTensor.
    # Such a tree keeps the _fast_apply path, which writes the pinned copies
    # back into its leaves.
    from tensordict._lazy import LazyStackedTensorDict
    from tensordict._td import TensorDict

    def plain_node(node):
        if is_tensorclass(node):
            node = node._tensordict
        if node.is_locked or type(node) not in (TensorDict, LazyStackedTensorDict):
            return None
        return node

    node = plain_node(td)
    if node is None:
        return None
    nodes, todo = [], [node]
    while todo:
        node = todo.pop()
        if node._lazy:
            for child in node.tensordicts:
                child = plain_node(child)
                if child is None:
                    return None
                todo.append(child)
            continue
        nodes.append(node)
        for value in node._tensordict.values():
            if type(value) is not Tensor:
                if _is_tensor_collection(type(value)):
                    if not is_non_tensor(value):
                        value = plain_node(value)
                        if value is None:
                            return None
                        todo.append(value)
                    continue
                if _is_unbatched(value) or isinstance(value, MemoryMappedTensor):
                    return None
            if value.layout is not torch.strided or value.untyped_storage().is_shared():
                return None
    return nodes


class _DeviceOps:
    """Device and dtype conversions, memory placement and memory size."""

    def auto_device_(self) -> Self:
        """Automatically sets the device, if it is unique.

        Returns: self with the edited ``device`` attribute.

        """
        devices = {
            value.device
            for value in self.values(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)
            if value.device is not None
        }
        if len(devices) == 1:
            self.clear_device_()
            self._set_device(list(devices)[0])
        else:
            self.clear_device_()
        return self

    def data_ptr(self, *, storage: bool = False):
        """Returns the data_ptr of the tensordict leaves.

        This can be useful to check if two tensordicts share the same ``data_ptr()``.

        Keyword Args:
            storage (bool, optional): if ``True``, `tensor.untyped_storage().data_ptr()` will be called
                instead. Defaults to ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.randn(2), b=torch.randn(2), batch_size=[2])
            >>> td0 = td.unsqueeze(0)  # the leaves of td0 are views on the leaves of td
            >>> assert (td0.data_ptr() == td.data_ptr()).all()

        .. note:: :class:`~tensordict.LazyStackedTensorDict` instances will be displayed as nested tensordicts to
            reflect the true ``data_ptr()`` of their leaves:

                >>> td0 = TensorDict(a=torch.randn(2), b=torch.randn(2), batch_size=[2])
                >>> td1 = TensorDict(a=torch.randn(2), b=torch.randn(2), batch_size=[2])
                >>> td = TensorDict.lazy_stack([td0, td1])
                >>> td.data_ptr()
                TensorDict(
                    fields={
                        0: TensorDict(
                            fields={
                                a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                                b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=cpu,
                            is_shared=False),
                        1: TensorDict(
                            fields={
                                a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                                b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=cpu,
                            is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False)

        """
        if storage:

            def func(x):
                return x.untyped_storage().data_ptr()

        else:

            def func(x):
                return x.data_ptr()

        from tensordict import TensorDict

        return TensorDict(
            {
                key: func(val)
                for key, val in self.items(True, True, is_leaf=_NESTED_TENSORS_AS_LISTS)
            },
            device=torch.device("cpu"),
        )

    def clear_device_(self) -> Self:
        """Sets the device of the tensordict and its sub-tensordicts to ``None``, and returns it.

        The leaves are not moved. After this call, you can set leaves on any
        device. See :ref:`devices`.

        Returns: self

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), batch_size=[3], device="cpu")
            >>> td = td.clear_device_()
            >>> print(td.device)
            None
            >>> td["a"].device
            device(type='cpu')

        """
        return self._set_device(None)

    def _set_device(self, device: torch.device) -> Self:
        self._device = device
        for value in self.values():
            if _is_tensor_collection(type(value)):
                value._set_device(device=device)
        return self

    @_cache_while_locked  # noqa: B019
    def param_count(self, *, count_duplicates: bool = True) -> int:
        """Counts the number of parameters (total number of indexable items), accounting for tensors only.

        Keyword Args:
            count_duplicates (bool): Whether to count duplicated tensor as independent or not.
                If ``False``, only strictly identical tensors will be discarded (same views but different
                ids from a common base tensor will be counted twice). Defaults to `True` (each tensor is assumed
                to be a single copy).

        """
        vals = self._values_list(True, True)
        total = 0
        if not count_duplicates:
            vals = set(vals)
        for v in vals:
            total += v.numel()
        return total

    @_cache_while_locked  # noqa: B019
    def bytes(self, *, count_duplicates: bool = True) -> int:
        """Counts the number of bytes of the contained tensors.

        Keyword Args:
            count_duplicates (bool): Whether to count duplicated tensor as independent or not.
                If ``False``, only strictly identical tensors will be discarded (same views but different
                ids from a common base tensor will be counted twice). Defaults to `True` (each tensor is assumed
                to be a single copy).

        """
        set_of_tensors = set() if not count_duplicates else []

        def add(tensor):
            if count_duplicates:
                set_of_tensors.append(tensor)
            else:
                set_of_tensors.add(tensor)

        def count_bytes(tensor):
            if tensor.is_nested:
                if not tensor.layout == torch.jagged:
                    raise RuntimeError(
                        "NTs that are not jagged are not supported by the bytes method. Please use the jagged layout instead "
                        "or raise and issue on https://github.com/pytorch/tensordict/issues instead."
                    )
                attrs, ctx = tensor.__tensor_flatten__()
                for attr in attrs:
                    t = getattr(tensor, attr)
                    count_bytes(t)
                return
            if isinstance(tensor, torch.Tensor):
                if isinstance(tensor, MemoryMappedTensor):
                    add(tensor)
                    return
                if type(tensor) in (Tensor, Parameter, Buffer):
                    pass
                elif hasattr(tensor, "__tensor_flatten__"):
                    attrs, ctx = tensor.__tensor_flatten__()
                    for attr in attrs:
                        t = getattr(tensor, attr)
                        count_bytes(t)
                    return
                else:
                    warnings.warn(
                        "The sub-tensor doesn't ot have a __tensor_flatten__ attribute, making it "
                        "impossible to count the bytes it contains. Falling back on regular count.",
                        category=UserWarning,
                    )
                    count_bytes(torch.as_tensor(tensor))
                    return

                grad = getattr(tensor, "grad", None)
                if grad is not None:
                    count_bytes(grad)
                    count_bytes(tensor.data)
                else:
                    add(tensor)
                return

        vals = self._values_list(True, True)
        for v in vals:
            count_bytes(v)
        total = 0
        for tensor in set_of_tensors:
            total += tensor.numel() * tensor.dtype.itemsize
        return total

    def pin_memory(self, num_threads: int | None = None, inplace: bool = False) -> Self:
        """Calls :meth:`~torch.Tensor.pin_memory` on the stored tensors.

        Args:
            num_threads (int, optional): if provided, the number of threads to
                use to call ``pin_memory`` on the leaves, in a
                :class:`~concurrent.futures.ThreadPoolExecutor`. Defaults to
                ``None``, which, like ``0``, calls
                :meth:`~torch.Tensor.pin_memory` on the calling thread.
            inplace (bool, optional): if ``True``, the tensordict is modified
                in-place. When every node is an unlocked
                :class:`~tensordict.TensorDict` or
                :class:`~tensordict.LazyStackedTensorDict` (or a tensorclass
                over one) and no leaf is a :class:`~tensordict.MemoryMappedTensor`, in
                shared memory, sparse, jagged or an
                :class:`~tensordict.UnbatchedTensor`, each leaf is replaced by
                its pinned copy, and the tensordict and every nested node stay
                the same objects; keys that hold the same tensor share its
                pinned copy, and a leaf that is a view of another leaf gets its
                own. Any other tree, such as a locked one or one that holds a
                memory-mapped tensordict, has the pinned copies written back
                into its leaves, as before, so they stay in pageable memory.
                Defaults to ``False``.

        """
        if self._pin_memory_unsupported:
            raise RuntimeError(
                f"Cannot pin memory of a {type(self).__name__}. Call to_tensordict() "
                "before making this call."
            )
        nodes = _pin_memory_nodes(self) if inplace else None
        if nodes is not None:
            # Replace each leaf by its pinned copy. Keys that hold one tensor
            # share its copy: memo maps the id of a leaf to a weak reference
            # to it and its copy (or the future of its copy).
            executor = ThreadPoolExecutor(num_threads) if num_threads else None
            memo, pending = {}, []
            try:
                for node in nodes:
                    for key in list(node._tensordict):
                        value = node._tensordict[key]
                        if _is_tensor_collection(type(value)):
                            continue
                        entry = memo.get(id(value))
                        if entry is None or entry[0]() is not value:
                            if executor is None:
                                pinned = _pin_memory(value)
                            else:
                                pinned = executor.submit(_pin_memory, value)
                            entry = memo[id(value)] = (weakref.ref(value), pinned)
                        # release the old leaf as its replacement is written
                        del value
                        if executor is None:
                            node._set_str(key, entry[1], validated=True, inplace=False)
                        else:
                            pending.append((node, key, entry[1]))
                # with threads, nothing is written until every leaf is pinned
                values = [future.result() for _, _, future in pending]
            finally:
                # as in _multithread_apply_nest, do not wait for the threads
                if executor is not None:
                    executor.shutdown(wait=False)
            for (node, key, _), value in zip(pending, values):
                node._set_str(key, value, validated=True, inplace=False)
            return self
        return self._fast_apply(
            _pin_memory,
            num_threads=num_threads,
            inplace=inplace,
            propagate_lock=True,
        )

    @deprecated(
        "TensorDictBase.pin_memory_()",
        removal="0.17",
        replacement="pin_memory(inplace=True)",
    )
    def pin_memory_(self, num_threads: int = 0) -> Self:
        """Calls :meth:`~torch.Tensor.pin_memory` on the stored tensors and returns the TensorDict modifies in-place.

        .. deprecated:: 0.15
            Use ``pin_memory(inplace=True)`` instead. Pinning allocates new
            storage, so it does not fit the trailing-underscore convention.
            ``pin_memory_()`` pins on the main thread by default
            (``num_threads=0``); pass ``num_threads=0`` to
            :meth:`pin_memory` to keep that behavior.

        Args:
            num_threads (int, optional): if provided, the number of threads to
                use to call ``pin_memory`` on the leaves. Defaults to ``0``.

        """
        return self.pin_memory(num_threads=num_threads, inplace=True)

    def cpu(self, **kwargs) -> Self:
        """Casts a tensordict to CPU.

        This function also supports all the keyword arguments of :meth:`~.to`.
        """
        return self.to("cpu", **kwargs)

    def cuda(self, device: int | str | torch.device | None = None, **kwargs) -> Self:
        """Casts a tensordict to a cuda device (if not already on it).

        Args:
            device (int, str or torch.device, optional): if provided, the cuda
                device on which the tensor should be cast. An integer is a cuda
                device index. Defaults to the current cuda device.

        This function also supports all the keyword arguments of :meth:`~.to`.

        Raises:
            RuntimeError: if ``device`` is not a cuda device.

        """
        if device is None:
            device = torch.device("cuda")
        elif isinstance(device, torch.device) or (
            isinstance(device, str) and not device.isdigit()
        ):
            device = torch.device(device)
            if device.type != "cuda":
                raise RuntimeError(f"Invalid device {device}, must be a cuda device.")
        else:
            device = f"cuda:{device}"
        return self.to(device, **kwargs)

    def is_shared(self) -> bool:
        """Checks if tensordict is in shared memory.

        If a TensorDict instance is in shared memory, it is locked (entries cannot
        be renamed, removed or added). If a ``TensorDict`` is created with
        tensors that are all in shared memory, this does __not__ mean that ``is_shared``
        will return ``True`` (as a new tensor may or may not be in shared memory).
        Only if one calls `tensordict.share_memory_()` or places the tensordict
        on a device where the content is shared by default (eg, ``"cuda"``)
        will the tensordict be considered in shared memory.

        This is always ``True`` for tensordicts on a CUDA device.

        """
        if self.device and not self._is_memmap:
            return self.device.type == "cuda" or self._is_shared
        return self._is_shared

    # Stream
    def record_stream(self, stream: torch.cuda.Stream) -> Self:
        """Marks the tensordict as having been used by this stream.

        When the tensordict is deallocated, ensure the tensor memory is not reused for other tensors until all work
        queued on stream at the time of deallocation is complete.

        See :meth:`~torch.Tensor.record_stream` for more information.`

        """
        if self._stream is not None and self._stream != stream:
            raise RuntimeError(
                "A stream is already associated with this TensorDict instance."
            )
        self._stream = stream

        def record(tensor):
            tensor.record_stream(stream)

        self._fast_apply(record, filter_empty=True)
        return self

    def _check_is_shared(self) -> bool:
        share_list = [_is_shared(value) for value in self.values()]
        if any(share_list) and not all(share_list):
            shared_str = ", ".join(
                [f"{key}: {_is_shared(value)}" for key, value in self.items()]
            )
            raise RuntimeError(
                f"tensors must be either all shared or not, but mixed "
                f"features is not allowed. "
                f"Found: {shared_str}"
            )
        return all(share_list) and len(share_list) > 0

    def _check_device(self, *, raise_exception: bool = True) -> None | bool:
        val = True
        for value in self.values():
            if _is_tensor_collection(type(value)):
                val &= value._check_device(raise_exception=raise_exception)
                if not val:
                    return False
            val &= self.device is None or (self.device == value.device)
            if not val:
                if raise_exception:
                    raise RuntimeError(
                        f"devices are incongruent, got value with device {value.device}, "
                        f"-- expected {self.device}."
                    )
                return False
        return val

    # Conversion (device or dtype)
    @overload
    def to(
        self: T,
        device: DeviceType | None = ...,
        dtype: torch.dtype | None = ...,
        non_blocking: bool = ...,
        inplace: bool = False,
    ) -> Self: ...

    @overload
    def to(self: T, dtype: torch.dtype, non_blocking: bool = ...) -> Self: ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> Self: ...

    @overload
    def to(self: T, *, other: T, non_blocking: bool = ...) -> Self: ...

    @overload
    def to(self: T, *, batch_size: torch.Size) -> Self: ...

    def _to_cuda_with_pin_mem(
        self,
        *,
        num_threads,
        device="cuda",
        non_blocking=None,
        to: Callable,
        inplace: bool = False,
    ):
        if self.is_empty():
            return self.to(device, inplace=inplace)
        keys, vals = self._items_list(
            leaves_only=True, include_nested=True, is_leaf=_NESTED_TENSORS_AS_LISTS
        )
        lkeys = len(keys)
        q_in = queue.SimpleQueue()
        q_out = queue.SimpleQueue()
        threads = []
        items = {}
        for key, val in _zip_strict(keys, vals):
            q_in.put_nowait((key, val))
        for _ in range(min(num_threads, lkeys)):
            thread = Thread(target=_pin_mem, args=(q_in, q_out))
            thread.start()
            threads.append(thread)
        try:
            while len(items) < lkeys:
                try:
                    keyval = q_out.get(timeout=_PIN_MEM_TIMEOUT)
                except queue.Empty:
                    # A large or file-backed leaf can take longer than the
                    # timeout to pin: wait as long as a thread is working.
                    if any(thread.is_alive() for thread in threads):
                        continue
                    # A thread may have put its last leaf after the wait
                    # timed out.
                    try:
                        keyval = q_out.get_nowait()
                    except queue.Empty:
                        raise RuntimeError(
                            "The pin_memory threads of to(device, "
                            "non_blocking_pin=True) exited without returning "
                            f"{lkeys - len(items)} of the {lkeys} leaves."
                        ) from None
                if not isinstance(keyval, tuple):
                    raise keyval
                key, val = keyval
                items[key] = to(val)
        finally:
            for thread in threads:
                thread.join(timeout=_PIN_MEM_TIMEOUT)

        def get(name, val):
            return items.get(name, val)

        result = self._fast_apply(
            get,
            named=True,
            nested_keys=True,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
            propagate_lock=True,
            device=device,
            out=self if inplace else None,
            checked=True,
        )
        return result

    @_as_context_manager()
    def to(self, *args, **kwargs) -> Self:
        """Maps a TensorDictBase subclass either on another device, dtype or to another TensorDictBase subclass (if permitted).

        When a dtype is passed, every tensor leaf is cast to that dtype, whatever its original
        dtype (including integer and boolean leaves), unless ``cast_filter`` selects the
        leaves to cast.

        Args:
            device (torch.device, optional): the desired device of the tensordict.
            dtype (torch.dtype, optional): the desired dtype of all the tensors in
                the tensordict.
            tensor (torch.Tensor, optional): Tensor whose dtype and device are the desired
                dtype and device for all tensors in this TensorDict.

        Keyword Args:
            non_blocking (bool, optional): controls how the leaves are copied.
                If ``None`` (default), the leaves are copied with ``non_blocking=True``
                and, when the target device is not a CUDA device (e.g. device-to-host
                copies), the tensordict synchronizes once after all the copies have been
                issued; for a CPU target, only if a leaf comes from another device.
                Under :func:`torch.compile`, these copies are blocking instead.
                If ``True``, the copies are non-blocking and no synchronization
                is performed: the caller is responsible for synchronizing before reading
                the results. If ``False``, the copies are blocking.
            memory_format (torch.memory_format, optional): the desired memory
                format for the 4D and 5D tensors in this tensordict, as in
                :meth:`torch.nn.Module.to`.
            batch_size (torch.Size, optional): resulting batch-size of the
                output tensordict. When the device or dtype changes, every
                nested tensordict gets exactly this batch size and the names
                are erased. Starting with TensorDict 0.17, the default will
                change to setting the batch size as ``to(batch_size=...)``
                does without a device: nested tensordicts with more batch
                dimensions keep their batch size, nested lazy stacks stay
                lazy stacks (or raise, as there, if they need more batch
                dimensions) and the names are kept. A ``FutureWarning`` is
                emitted when the result will change.
            other (TensorDictBase, optional): TensorDict instance whose dtype
                and device are the desired dtype and device for all tensors
                in this TensorDict.

                .. note::
                    Since :class:`~tensordict.TensorDictBase` instances do not have
                    a dtype, the dtype is gathered from the example leaves.
                    If there are more than one dtype, then no dtype
                    casting is done.

            cast_filter (callable, optional): a predicate ``fn(key, value) -> bool``
                that selects the tensor leaves the dtype cast applies to.
                ``key`` is the nested key of the leaf as a tuple of str, for
                example ``("obs",)`` or ``("next", "done")``. Leaves for which
                it returns ``False`` keep their dtype; a device or memory-format
                change still applies to them. It is not called on non-tensor
                data, nor when no dtype is given or when ``to`` gets an attrs
                tensordict, whose leaves give each leaf its dtype. An
                :class:`~tensordict.UnbatchedTensor` leaf is passed as is, not
                as a :class:`torch.Tensor`. Helpers such as
                :func:`~tensordict.is_inexact` can be passed directly. Defaults
                to the ``cast_filter`` of a tensorclass that declares one, and to
                casting every leaf otherwise; pass ``None`` to cast every leaf.
                The dtype shorthands (:meth:`float`, :meth:`bfloat16`, ...) take
                it too.

            non_blocking_pin (bool, optional): if ``True``, the tensors are pinned before
                being sent to device. This will be done asynchronously but can be
                controlled via the ``num_threads`` argument.

                .. note::
                    Each call starts and joins its pinning threads, which costs
                    the same however little data there is. So
                    ``non_blocking_pin=True`` pays off only for large tensordicts,
                    with leaves of a MiB or more and tens of MiB in total. For
                    smaller tensordicts, ``tensordict.to("cuda")`` is usually the
                    fastest, and ``tensordict.pin_memory().to("cuda",
                    non_blocking=True)`` is also faster than
                    ``non_blocking_pin=True``. Where the crossover lies depends on
                    the machine, so measure it on yours.

            num_threads (int or None, optional): if ``non_blocking_pin=True``, the number
                of threads to be used for ``pin_memory``. By default,
                up to ``max(1, torch.get_num_threads() // 2)`` threads are
                started, and no more than one per leaf.
                ``num_threads=0`` will cancel any
                multithreading for the `pin_memory()` calls.
            inplace (bool, optional): if ``True``, the data will be written in-place in the same tensordict.
                This can be significantly faster whenever building a tensordict is CPU-overhead bound.
                Defaults to ``False``.

        Returns:
            a new tensordict instance if the device differs from the tensordict
            device and/or if the dtype or memory format is passed. The same
            tensordict otherwise.
            ``batch_size`` only modifications are done in-place.

        .. note::
            If the TensorDict is consolidated, the resulting TensorDict will be consolidated too.
            Each new tensor will be a view on the consolidated storage cast to the desired device.

        This operation can be used as a context manager too. Inside the
        ``with`` block, you work on the cast tensordict. When the block exits,
        its entries are written back to the original tensordict, on the
        original device. Use it to run a module that expects its input on a
        specific device. See :ref:`devices`.

        .. note::
            On exit, every entry is moved back to the original device. The
            entries that already existed get their original dtype back, but
            the new entries keep the dtype of the ``with`` block: after
            ``with td.to(torch.float64) as td64``, a new entry of ``td64`` is
            ``float64`` in ``td``. The original tensordict must not be
            locked. If the block raises an exception, nothing is written back.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> data = TensorDict({"a": 1.0}, [], device=None)
            >>> data_int = data.to(torch.int)  # casts to int
            >>> data_int["a"]
            tensor(1, dtype=torch.int32)

            Cast the floating-point leaves only:

            >>> from tensordict import is_inexact
            >>> data = TensorDict(x=torch.zeros(3), done=torch.zeros(3, dtype=torch.bool))
            >>> data_bf16 = data.to(torch.bfloat16, cast_filter=is_inexact)
            >>> data_bf16["x"].dtype, data_bf16["done"].dtype
            (torch.bfloat16, torch.bool)

            The next lines need a CUDA device, so they are not run when the
            documentation is tested.

            >>> data_cuda = data.to("cuda:0")  # doctest: +SKIP
            >>> data_cuda_int = data.to("cuda:0", torch.int)  # doctest: +SKIP
            >>> data_cuda = data.to(torch.randn(3, device="cuda:0"))  # doctest: +SKIP
            >>> data_cuda = data.to(other=TensorDict({}, [], device="cuda:0"))  # doctest: +SKIP

            Use it as a context manager for a temporary device change (CUDA
            needed):

            >>> from tensordict.nn import TensorDictModule
            >>> mod = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
            >>> td = TensorDict(x=torch.zeros(3), batch_size=[3], device="cpu")
            >>> with td.to("cuda") as td_gpu:  # doctest: +SKIP
            ...     _ = td_gpu.update(mod(td_gpu))  # runs on the GPU
            >>> # The entries are written back on the original device
            >>> assert td["y"].device.type == "cpu"  # doctest: +SKIP

            A temporary dtype change runs on CPU. The new entry keeps the dtype
            of the ``with`` block:

            >>> td = TensorDict(x=torch.zeros(3), batch_size=[3])
            >>> with td.to(torch.float64) as td64:
            ...     td64["y"] = td64["x"] + 1
            >>> td["x"].dtype, td["y"].dtype
            (torch.float32, torch.float64)
        """
        # Per-leaf spec: a positional tensordict argument is interpreted as an
        # attrs tensordict whose leaves are `TensorAttrs` — each source leaf
        # is cast to its counterpart's device/dtype.
        if (
            args
            and not isinstance(args[0], Tensor)
            and _is_tensor_collection(type(args[0]))
        ):
            attrs_td = args[0]
            if len(args) > 1:
                raise TypeError(
                    "to(attrs_td) does not accept additional positional arguments."
                )
            # The attrs give each leaf its dtype: cast_filter does not apply
            kwargs.pop("cast_filter", None)
            return self._to_per_leaf(attrs_td, **kwargs)

        non_blocking = kwargs.pop("non_blocking", None)

        (
            device,
            dtype,
            _,
            convert_to_format,
            batch_size,
            non_blocking_pin,
            num_threads,
            inplace,
        ) = _parse_to(*args, **kwargs)
        return self._to(
            device,
            dtype,
            non_blocking,
            convert_to_format,
            batch_size,
            non_blocking_pin,
            num_threads,
            inplace,
            args,
            kwargs,
        )

    def _to(
        self,
        device: torch.device | None,
        dtype: torch.dtype | None,
        non_blocking: bool | None,
        convert_to_format: torch.memory_format | None,
        batch_size: torch.Size | None,
        non_blocking_pin: bool,
        num_threads: int | None,
        inplace: bool,
        args: tuple,
        kwargs: dict,
    ) -> Self:
        """Casts the tensordict as :meth:`to` describes, with the parsed arguments.

        :meth:`to` has handled an attrs tensordict, taken ``non_blocking`` out
        of ``kwargs`` and parsed the other arguments; ``args`` and ``kwargs``
        are the ones it received, without ``non_blocking``; ``cast_filter`` is
        read from ``kwargs``. It wraps the result for use as a context manager.

        Returns ``self`` when the device is already the right one and no dtype
        or memory format is given (the batch size is then set in place, as
        for a batch size alone), and a new tensordict otherwise. With
        ``_masking_on_dense_copy``, a change of device, dtype or memory format
        converts a :meth:`to_tensordict` copy. A class overrides it to reject
        arguments that its storage cannot honor, or to change the device
        without reading the data.
        """
        result = self

        if (
            device is not None
            and dtype is None
            and convert_to_format is None
            and device == self.device
        ):
            if batch_size is not None and tuple(batch_size) != result.batch_size:
                result.batch_size = torch.Size(batch_size)
            return result

        if (
            self._masking_on_dense_copy
            and not inplace
            and (
                device is not None or dtype is not None or convert_to_format is not None
            )
        ):
            return self.to_tensordict().to(*args, non_blocking=non_blocking, **kwargs)

        if (
            self.is_consolidated()
            and dtype is None
            and device is not None
            and convert_to_format is None
        ):
            result = self._to_consolidated(
                device=device,
                pin_memory=non_blocking_pin,
                num_threads=num_threads,
                non_blocking=non_blocking,
                inplace=inplace,
            )
            if batch_size is not None and tuple(batch_size) != result.batch_size:
                result.batch_size = torch.Size(batch_size)
            return result

        if non_blocking is None:
            # Under torch.compile, copies to a non-cuda device are blocking: the
            # _sync_all that non-blocking copies to such a device need is not
            # traceable.
            sub_non_blocking = not (
                is_compiling() and device is not None and device.type != "cuda"
            )
            non_blocking = False
        else:
            sub_non_blocking = non_blocking

        if convert_to_format is not None:

            def to(tensor, dtype=dtype):
                # As in nn.Module.to, the memory format applies to 4D and 5D tensors
                if tensor.dim() in (4, 5):
                    return tensor.to(
                        device,
                        dtype,
                        non_blocking=sub_non_blocking,
                        memory_format=convert_to_format,
                    )
                return tensor.to(
                    device=device, dtype=dtype, non_blocking=sub_non_blocking
                )

        else:

            def to(tensor, dtype=dtype):
                return tensor.to(
                    device=device, dtype=dtype, non_blocking=sub_non_blocking
                )

        # A copy to the CPU needs the sync at the end only if a leaf comes from
        # another device: a CPU leaf is copied synchronously, and the sync would
        # wait for all the work queued on the GPU.
        needs_d2h_sync = True
        if (
            device is not None
            and device.type == "cpu"
            and sub_non_blocking
            and not non_blocking
        ):
            needs_d2h_sync = False

            def to(tensor, dtype=dtype, _to=to):
                nonlocal needs_d2h_sync
                if not tensor.is_cpu:
                    needs_d2h_sync = True
                return _to(tensor, dtype)

        cast_filter = kwargs.get("cast_filter") if dtype is not None else None
        named = cast_filter is not None
        if named:

            def to(key, tensor, _to=to):
                # The leaves that cast_filter leaves out keep their dtype
                if cast_filter(key if type(key) is tuple else (key,), tensor):
                    return _to(tensor)
                return _to(tensor, None)

        apply_kwargs = {}
        if device is not None or dtype is not None or convert_to_format is not None:
            # The pin_memory threads walk the members of lazy stacks with keys
            # that hold the member index: cast_filter takes the sequential path
            if non_blocking_pin and num_threads != 0 and not named:
                if num_threads is None:
                    num_threads = max(1, torch.get_num_threads() // 2)
                result = self._to_cuda_with_pin_mem(
                    num_threads=num_threads, to=to, device=device, inplace=inplace
                )
            else:
                apply_kwargs["device"] = device if device is not None else self.device
                # With inplace=True, the setter below sets the batch size
                apply_kwargs["batch_size"] = None if inplace else batch_size
                apply_kwargs["out"] = self if inplace else None
                apply_kwargs["checked"] = True
                apply_kwargs["call_on_unbatched"] = True
                if non_blocking_pin:
                    if named:

                        def to_pinmem(key, tensor, _to=to):
                            return _to(key, tensor.pin_memory())

                    else:

                        def to_pinmem(tensor, _to=to):
                            return to(tensor.pin_memory())

                    result = result._fast_apply(
                        to_pinmem,
                        propagate_lock=True,
                        named=named,
                        nested_keys=named,
                        **apply_kwargs,
                    )
                elif self.is_locked and batch_size is None and not named:
                    # A locked tensordict caches its leaves, so the first walk
                    # is nearly free: keep the two-pass form, which is faster
                    # there than one walk that calls to() on each leaf. With a
                    # batch_size, _fast_apply renames the leaves of nested lazy
                    # stacks, which the keys of _items_list would then miss.
                    # The keys of _items_list hold the member index of lazy
                    # stacks, which cast_filter must not see.
                    keys, tensors = self._items_list(True, True)
                    tensors = [to(t) for t in tensors]
                    items = dict(zip(keys, tensors))

                    def get(name, val):
                        return items.get(name, val)

                    result = self._fast_apply(
                        get,
                        named=True,
                        nested_keys=True,
                        is_leaf=_NESTED_TENSORS_AS_LISTS,
                        propagate_lock=True,
                        **apply_kwargs,
                    )
                else:
                    # walk the tree once
                    if (
                        device is not None
                        and device.type == "cpu"
                        and dtype is None
                        and convert_to_format is None
                    ):

                        def to(tensor, _to=to):
                            # Tensor.to would return the leaf as it is: skip
                            # the call. is_cpu costs less than a device
                            # comparison, which would slow down moves.
                            if tensor.is_cpu:
                                return tensor
                            return _to(tensor)

                    result = self._fast_apply(
                        to,
                        # _NESTED_TENSORS_AS_LISTS puts the member index of
                        # lazy stacks in the keys, which cast_filter must not see
                        is_leaf=_default_is_leaf if named else _NESTED_TENSORS_AS_LISTS,
                        propagate_lock=True,
                        named=named,
                        nested_keys=named,
                        **apply_kwargs,
                    )
                if batch_size is not None and not inplace:
                    # _fast_apply gives every nested tensordict the new batch
                    # size without checking its leaves
                    _check_nested_batch_size(result)
                    if not is_compiling() and _to_batch_size_will_change(
                        self, torch.Size(batch_size)
                    ):
                        warnings.warn(
                            "to() with a batch_size and a new device or dtype "
                            "gives every nested tensordict exactly that batch "
                            "size, turns nested lazy stacks into TensorDicts "
                            "and erases the names. Starting with TensorDict "
                            "0.17, the default will change to setting the "
                            "batch size as to(batch_size=...) does without a "
                            "device, which keeps the nested batch sizes, lazy "
                            "stacks and names. Call to(...) and then set the "
                            "batch_size to get the new behaviour now.",
                            FutureWarning,
                            # skip to() and the _as_context_manager wrapper
                            stacklevel=4,
                        )

        if batch_size is not None:
            result.batch_size = batch_size
        if (
            device is not None
            and sub_non_blocking
            and not non_blocking
            and device.type != "cuda"
            and needs_d2h_sync
        ):
            self._sync_all()
        return result

    def _sync_all(self):
        device_type = _get_available_device_type()
        if device_type is None:
            return

        if device_type == "cuda":
            # TODO: dynamo doesn't like torch.cuda.is_initialized
            if not is_compiling() and torch.cuda.is_initialized():
                _sync_cuda_transfer()
        else:
            device_module = _get_device_module(device_type)
            device_module.synchronize()

    def double(self, *, cast_filter: Callable | None = None) -> Self:
        r"""Casts all tensors to ``torch.double`` (``torch.float64``).

        Keyword Args:
            cast_filter (callable, optional): a predicate ``fn(key, value) -> bool``
                that selects the tensor leaves to cast, as in :meth:`to`.
        """

        def dble(x):
            return x.double()

        if cast_filter is None:
            return self._fast_apply(dble, propagate_lock=True)
        return _cast_filtered(self, dble, cast_filter, propagate_lock=True)

    def float(self, *, cast_filter: Callable | None = None) -> Self:
        r"""Casts all tensors to ``torch.float``.

        Keyword Args:
            cast_filter (callable, optional): a predicate ``fn(key, value) -> bool``
                that selects the tensor leaves to cast, as in :meth:`to`.
        """

        def tofloat(x):
            return x.float()

        if cast_filter is None:
            return self._fast_apply(tofloat, propagate_lock=True)
        return _cast_filtered(self, tofloat, cast_filter, propagate_lock=True)

    def int(self, *, cast_filter: Callable | None = None) -> Self:
        r"""Casts all tensors to ``torch.int``.

        Keyword Args:
            cast_filter (callable, optional): a predicate ``fn(key, value) -> bool``
                that selects the tensor leaves to cast, as in :meth:`to`.
        """

        def toint(x):
            return x.int()

        if cast_filter is None:
            return self._fast_apply(toint, propagate_lock=True)
        return _cast_filtered(self, toint, cast_filter, propagate_lock=True)

    def bool(self, *, cast_filter: Callable | None = None) -> Self:
        r"""Casts all tensors to ``torch.bool``.

        Keyword Args:
            cast_filter (callable, optional): a predicate ``fn(key, value) -> bool``
                that selects the tensor leaves to cast, as in :meth:`to`.
        """

        def tobool(x):
            return x.bool()

        if cast_filter is None:
            return self._fast_apply(tobool, propagate_lock=True)
        return _cast_filtered(self, tobool, cast_filter, propagate_lock=True)

    def half(self, *, cast_filter: Callable | None = None) -> Self:
        r"""Casts all tensors to ``torch.half``.

        Keyword Args:
            cast_filter (callable, optional): a predicate ``fn(key, value) -> bool``
                that selects the tensor leaves to cast, as in :meth:`to`.
        """

        def tohalf(x):
            return x.half()

        if cast_filter is None:
            return self._fast_apply(tohalf, propagate_lock=True)
        return _cast_filtered(self, tohalf, cast_filter, propagate_lock=True)

    def type(
        self, dst_type: torch.dtype, *, cast_filter: Callable | None = None
    ) -> Self:
        r"""Casts all tensors to :attr:`dst_type`.

        Args:
            dst_type (type or string): the desired type

        Keyword Args:
            cast_filter (callable, optional): a predicate ``fn(key, value) -> bool``
                that selects the tensor leaves to cast, as in :meth:`to`.
        """

        def totype(x):
            return x.type(dst_type)

        if cast_filter is None:
            return self._fast_apply(totype)
        return _cast_filtered(self, totype, cast_filter)

    @_make_dtype_promotion
    def bfloat16(self) -> Self: ...

    @_make_dtype_promotion
    def complex128(self) -> Self: ...

    @_make_dtype_promotion
    def complex32(self) -> Self: ...

    @_make_dtype_promotion
    def complex64(self) -> Self: ...

    @_make_dtype_promotion
    def float16(self) -> Self: ...

    @_make_dtype_promotion
    def float32(self) -> Self: ...

    @_make_dtype_promotion
    def float64(self) -> Self: ...

    @_make_dtype_promotion
    def int16(self) -> Self: ...

    @_make_dtype_promotion
    def int32(self) -> Self: ...

    @_make_dtype_promotion
    def int64(self) -> Self: ...

    @_make_dtype_promotion
    def int8(self) -> Self: ...

    @_make_dtype_promotion
    def qint32(self) -> Self: ...

    @_make_dtype_promotion
    def qint8(self) -> Self: ...

    @_make_dtype_promotion
    def quint4x2(self) -> Self: ...

    @_make_dtype_promotion
    def quint8(self) -> Self: ...

    @_make_dtype_promotion
    def uint16(self) -> Self: ...

    @_make_dtype_promotion
    def uint32(self) -> Self: ...

    @_make_dtype_promotion
    def uint64(self) -> Self: ...

    @_make_dtype_promotion
    def uint8(self) -> Self: ...
