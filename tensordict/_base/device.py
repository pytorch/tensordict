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
    _positional_non_blocking,
    _user_stacklevel,
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
        """Sets the device of the tensordict to the device of its leaves, if they all share one.

        If the leaves are on several devices, the device is set to ``None``.
        The nested tensordicts get the same device. The leaves are not moved.
        See :ref:`devices`.

        Returns:
            ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
            >>> print(td.device)
            None
            >>> td.auto_device_().device
            device(type='cpu')

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
        """Returns the data pointers of the leaves, as a tensordict of integers.

        The result has the keys of the tensordict, an empty batch size and the
        CPU device. Each leaf is replaced by its
        :meth:`~torch.Tensor.data_ptr`, as a 0-d ``int64`` tensor. Compare two
        results to check whether two tensordicts share memory. In a lazy stack,
        each member is a nested entry named by its index (``"0"``, ``"1"``,
        ...), which gives the data pointers of the leaves it really holds.

        Keyword Args:
            storage (bool, optional): if ``True``, returns the data pointer of
                the storage of each leaf
                (``tensor.untyped_storage().data_ptr()``) instead, which a view
                shares with its base. Defaults to ``False``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(2), b=torch.zeros(2), batch_size=[2])
            >>> td0 = td.unsqueeze(0)  # the leaves of td0 are views on the leaves of td
            >>> assert (td0.data_ptr() == td.data_ptr()).all()
            >>> td1 = td[1:]  # views with an offset
            >>> (td1.data_ptr() == td.data_ptr()).all()
            False
            >>> (td1.data_ptr(storage=True) == td.data_ptr(storage=True)).all()
            True
            >>> stack = TensorDict.lazy_stack([td, td])
            >>> list(stack.data_ptr().keys(include_nested=True, leaves_only=True))
            [('0', 'a'), ('0', 'b'), ('1', 'a'), ('1', 'b')]

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

        Returns:
            ``self``.

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
        """Returns the total number of elements of the tensor leaves.

        It sums :meth:`~torch.Tensor.numel` over the tensor leaves, nested
        tensordicts included. Non-tensor data is not counted.

        Keyword Args:
            count_duplicates (bool, optional): if ``True``, a tensor stored
                under several keys is counted once per key. If ``False``, it
                is counted once. Two different views of one tensor are counted
                twice in both cases. Defaults to ``True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> x = torch.zeros(6)
            >>> td = TensorDict(a=x, b=x, c=torch.zeros(2, 3), s="text")
            >>> td.param_count()
            18
            >>> td.param_count(count_duplicates=False)
            12

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
        """Returns the number of bytes of the tensor leaves.

        Each tensor leaf counts ``numel() * dtype.itemsize`` bytes, and a leaf
        with a gradient counts the bytes of its gradient too. Non-tensor data
        is not counted. A view counts the bytes of its own elements, not those
        of the storage it shares.

        Keyword Args:
            count_duplicates (bool, optional): if ``True``, a tensor stored
                under several keys is counted once per key. If ``False``, it
                is counted once. Two different views of one tensor are counted
                twice in both cases. Defaults to ``True``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> x = torch.zeros(6)
            >>> td = TensorDict(a=x, b=x, c=torch.zeros(2, 3))
            >>> td.bytes()
            72
            >>> td.bytes(count_duplicates=False)
            48

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
        """Returns a tensordict whose tensor leaves are in pinned (page-locked) memory.

        Calls :meth:`~torch.Tensor.pin_memory` on every tensor leaf. Copies
        from pinned memory to an accelerator are faster, and can run
        asynchronously with ``to(device, non_blocking=True)``. Pinning needs an
        accelerator, such as CUDA. A :class:`~tensordict.PersistentTensorDict`
        or a :class:`~tensordict.store.TensorDictStore` raises a
        ``RuntimeError``: call :meth:`to_tensordict` first. See
        :ref:`devices`.

        Args:
            num_threads (int, optional): the number of threads that pin the
                leaves, in a :class:`~concurrent.futures.ThreadPoolExecutor`.
                Defaults to ``None``, which, like ``0``, pins them on the
                calling thread.
            inplace (bool, optional): if ``True``, the tensordict and its
                nested tensordicts keep their identity, each leaf is replaced
                by its pinned copy, and the call returns ``self``. A locked
                tensordict, one that holds a node other than a
                :class:`~tensordict.TensorDict`, a lazy stack or a tensorclass,
                or one with memory-mapped, shared, sparse, jagged or unbatched
                leaves gets the pinned copies written back into its existing
                leaves, which stay in pageable memory. Defaults to ``False``.

        Examples:
            The example needs an accelerator, so it is not run when the
            documentation is tested.

            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(obs=torch.zeros(1000, 3), batch_size=[1000])
            >>> pinned = td.pin_memory()  # doctest: +SKIP
            >>> pinned["obs"].is_pinned()  # doctest: +SKIP
            True
            >>> td_cuda = pinned.to("cuda", non_blocking=True)  # doctest: +SKIP

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
        """Pins the tensor leaves of the tensordict and returns ``self``.

        .. deprecated:: 0.15
            Use ``pin_memory(inplace=True)`` instead. Pinning allocates new
            storage, so it does not fit the trailing-underscore convention
            (see :ref:`inplace-conventions`). ``pin_memory_`` will be removed
            in TensorDict 0.17.

        Args:
            num_threads (int, optional): the number of threads that pin the
                leaves, as in :meth:`pin_memory`. Defaults to ``0``, which
                pins them on the calling thread.

        """
        return self.pin_memory(num_threads=num_threads, inplace=True)

    def cpu(self, **kwargs) -> Self:
        """Returns the tensordict on the CPU, as ``to("cpu", **kwargs)`` does.

        A tensordict whose device is already the CPU is returned as it is.

        Args:
            **kwargs: passed to :meth:`to`, for example ``non_blocking``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
            >>> td.cpu().device
            device(type='cpu')
        """
        return self.to("cpu", **kwargs)

    def cuda(self, device: int | str | torch.device | None = None, **kwargs) -> Self:
        """Returns the tensordict on a CUDA device, as :meth:`to` does.

        A tensordict that is already on that device is returned as it is.

        Args:
            device (int, str or torch.device, optional): the CUDA device. An
                integer (or a string of digits) is a device index. Defaults to
                ``None``, which uses the current CUDA device.
            **kwargs: passed to :meth:`to`, for example ``non_blocking``.

        Raises:
            RuntimeError: if ``device`` is not a CUDA device.

        Examples:
            The example needs a CUDA device, so it is not run when the
            documentation is tested.

            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
            >>> td.cuda().device  # doctest: +SKIP
            device(type='cuda', index=0)
            >>> td.cuda(1).device  # doctest: +SKIP
            device(type='cuda', index=1)

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
        """Whether the tensordict is in shared memory.

        It is ``True`` after :meth:`share_memory_`, and always for a
        tensordict on a CUDA device. A tensordict built from tensors that are
        in shared memory is not marked as shared: only
        :meth:`share_memory_` marks it. A tensordict in shared memory is
        locked: entries cannot be added, renamed or removed. See
        :ref:`keys-locking`.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
            >>> td.is_shared()
            False
            >>> td.share_memory_().is_shared()
            True
            >>> td.is_locked
            True

        """
        if self.device and not self._is_memmap:
            return self.device.type == "cuda" or self._is_shared
        return self._is_shared

    # Stream
    def record_stream(self, stream: torch.cuda.Stream) -> Self:
        """Marks the tensor leaves of the tensordict as used by ``stream``.

        Calls :meth:`~torch.Tensor.record_stream` on every tensor leaf: when
        a leaf is freed, its memory is not reused until the work queued on
        ``stream`` at that time is complete. A tensordict can be marked with
        one stream only.

        Args:
            stream (torch.cuda.Stream): the stream that uses the leaves.

        Returns:
            ``self``.

        Raises:
            RuntimeError: if the tensordict is already marked with another
                stream.

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
        """Returns a tensordict cast to another device, dtype or memory format, or with a new batch size.

        It accepts the forms of :meth:`torch.Tensor.to`, and some more:

        - ``to(device, dtype=None)``: moves every leaf to ``device`` and, if
          ``dtype`` is given, casts it.
        - ``to(dtype)``: casts every tensor leaf to ``dtype``, whatever its
          original dtype (integer and boolean leaves too), unless
          ``cast_filter`` selects the leaves to cast.
        - ``to(tensor)``: uses the device and the dtype of ``tensor``.
        - ``to(other=tensordict)``: uses the device of ``other``, and its
          dtype if all its leaves share one. Otherwise the dtypes are kept.
        - ``to(batch_size=...)``: only sets the batch size, in place, and
          returns ``self``.
        - ``to(td.attrs())``: casts each leaf to the device and the dtype that
          :meth:`~tensordict.TensorDict.attrs` recorded for the same key. See
          :ref:`devices`.

        The result is ``self`` when the device is already the device of the
        tensordict and no dtype or memory format is given. Otherwise it is a
        new tensordict, unless ``inplace=True``. When only the device changes,
        a consolidated tensordict gives a consolidated result, whose leaves
        are views of one storage on the new device.

        Args:
            device (torch.device, str or int, optional): the device of the
                result. Its nested tensordicts get it too.
            dtype (torch.dtype, optional): the dtype of the tensor leaves of
                the result.
            tensor (torch.Tensor, optional): a tensor whose device and dtype
                are the device and dtype of the result.

        Keyword Args:
            non_blocking (bool, optional): how the leaves are copied. ``None``
                copies them with ``non_blocking=True`` and, when the target
                is not a CUDA device (for example a copy to the host),
                synchronizes once after all the copies are issued; for a CPU
                target, only if a leaf comes from another device. Under
                :func:`torch.compile`, these copies are blocking instead.
                ``True`` copies without synchronizing: synchronize before you
                read the results. ``False`` makes blocking copies. Defaults to
                ``None``. It can also follow the device, dtype or tensor
                positionally, as in ``to("cpu", torch.float64, False)``. A
                positional ``True`` is ignored for now, with a
                ``FutureWarning``: the copies synchronize as for ``None``.
                Starting with TensorDict 0.17, the default will change to
                honoring it as the keyword ``True`` is.
            memory_format (torch.memory_format, optional): the memory format of
                the 4D and 5D tensors of the result, as in
                :meth:`torch.nn.Module.to`. Defaults to ``None``, which keeps
                the memory format.
            batch_size (torch.Size, optional): the batch size of the result.
                When the device or dtype changes, every nested tensordict gets
                exactly this batch size and the names are erased. Starting with
                TensorDict 0.17, the default will change to setting the batch
                size as ``to(batch_size=...)`` does without a device: nested
                tensordicts with more batch dimensions keep their batch size,
                nested lazy stacks stay lazy stacks (or raise, as there, if
                they need more batch dimensions) and the names are kept. A
                ``FutureWarning`` is emitted when the result will change.
                Defaults to ``None``, which keeps the batch size.
            other (TensorDictBase, optional): a tensordict whose device, and
                dtype if all its leaves share one, are those of the result.
                Defaults to ``None``.
            cast_filter (Callable, optional): a predicate
                ``fn(key, value) -> bool`` that selects the tensor leaves the
                dtype cast applies to. ``key`` is the nested key of the leaf as
                a tuple of str, for example ``("obs",)`` or
                ``("next", "done")``. Leaves for which it returns ``False``
                keep their dtype, but still move to the new device or memory
                format. It is not called on non-tensor data, without a dtype,
                or with an attrs tensordict. An
                :class:`~tensordict.UnbatchedTensor` leaf is passed as is.
                Helpers such as :func:`~tensordict.is_inexact` can be passed
                directly. Defaults to the ``cast_filter`` of a tensorclass that
                declares one, and to casting every leaf otherwise; pass
                ``None`` to cast every leaf. The dtype shorthands
                (:meth:`float`, :meth:`bfloat16`, ...) take it too.
            non_blocking_pin (bool, optional): if ``True``, pins the tensors
                before sending them to the device, in ``num_threads``
                threads. Defaults to ``False``.

                .. note::
                    Each call starts and joins its pinning threads, which costs
                    the same however little data there is. So
                    ``non_blocking_pin=True`` pays off only for large
                    tensordicts, with leaves of a MiB or more and tens of MiB
                    in total. For smaller tensordicts, ``td.to("cuda")`` is
                    usually the fastest, and
                    ``td.pin_memory().to("cuda", non_blocking=True)`` is also
                    faster than ``non_blocking_pin=True``. Measure the
                    crossover on your machine.

            num_threads (int, optional): with ``non_blocking_pin=True``, the
                number of threads that pin the leaves, at most one per leaf.
                ``0`` pins them on the calling thread. Defaults to ``None``,
                which starts up to ``max(1, torch.get_num_threads() // 2)``
                threads.
            inplace (bool, optional): if ``True``, writes the cast leaves into
                the tensordict, which keeps its identity and keys, and returns
                ``self``. This can be faster when building a new tensordict is
                what costs the most. See :ref:`inplace-conventions`. Defaults
                to ``False``.

        Returns:
            ``self`` when nothing changes, with ``inplace=True``, or when only
            ``batch_size`` is given; a new tensordict otherwise.

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

        positional = "non_blocking" not in kwargs
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
        if positional and len(args) > 1:
            non_blocking = _positional_non_blocking(args)
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
        """Returns a tensordict whose tensor leaves are cast to ``torch.float64`` (``torch.double``).

        Keyword Args:
            cast_filter (Callable, optional): a predicate
                ``fn(key, value) -> bool`` that selects the tensor leaves to
                cast, as in :meth:`to`. Defaults to ``None``, which casts every
                tensor leaf (a tensorclass uses its own ``cast_filter``, if it
                declares one).
        """

        def dble(x):
            return x.double()

        if cast_filter is None:
            return self._fast_apply(dble, propagate_lock=True)
        return _cast_filtered(self, dble, cast_filter, propagate_lock=True)

    def float(self, *, cast_filter: Callable | None = None) -> Self:
        """Returns a tensordict whose tensor leaves are cast to ``torch.float32`` (``torch.float``).

        Keyword Args:
            cast_filter (Callable, optional): a predicate
                ``fn(key, value) -> bool`` that selects the tensor leaves to
                cast, as in :meth:`to`. Defaults to ``None``, which casts every
                tensor leaf (a tensorclass uses its own ``cast_filter``, if it
                declares one).
        """

        def tofloat(x):
            return x.float()

        if cast_filter is None:
            return self._fast_apply(tofloat, propagate_lock=True)
        return _cast_filtered(self, tofloat, cast_filter, propagate_lock=True)

    def int(self, *, cast_filter: Callable | None = None) -> Self:
        """Returns a tensordict whose tensor leaves are cast to ``torch.int32`` (``torch.int``).

        Keyword Args:
            cast_filter (Callable, optional): a predicate
                ``fn(key, value) -> bool`` that selects the tensor leaves to
                cast, as in :meth:`to`. Defaults to ``None``, which casts every
                tensor leaf (a tensorclass uses its own ``cast_filter``, if it
                declares one).
        """

        def toint(x):
            return x.int()

        if cast_filter is None:
            return self._fast_apply(toint, propagate_lock=True)
        return _cast_filtered(self, toint, cast_filter, propagate_lock=True)

    def bool(self, *, cast_filter: Callable | None = None) -> Self:
        """Returns a tensordict whose tensor leaves are cast to ``torch.bool``.

        Keyword Args:
            cast_filter (Callable, optional): a predicate
                ``fn(key, value) -> bool`` that selects the tensor leaves to
                cast, as in :meth:`to`. Defaults to ``None``, which casts every
                tensor leaf (a tensorclass uses its own ``cast_filter``, if it
                declares one).
        """

        def tobool(x):
            return x.bool()

        if cast_filter is None:
            return self._fast_apply(tobool, propagate_lock=True)
        return _cast_filtered(self, tobool, cast_filter, propagate_lock=True)

    def half(self, *, cast_filter: Callable | None = None) -> Self:
        """Returns a tensordict whose tensor leaves are cast to ``torch.float16`` (``torch.half``).

        Keyword Args:
            cast_filter (Callable, optional): a predicate
                ``fn(key, value) -> bool`` that selects the tensor leaves to
                cast, as in :meth:`to`. Defaults to ``None``, which casts every
                tensor leaf (a tensorclass uses its own ``cast_filter``, if it
                declares one).
        """

        def tohalf(x):
            return x.half()

        if cast_filter is None:
            return self._fast_apply(tohalf, propagate_lock=True)
        return _cast_filtered(self, tohalf, cast_filter, propagate_lock=True)

    def type(
        self, dst_type: torch.dtype, *, cast_filter: Callable | None = None
    ) -> Self:
        """Returns a tensordict whose tensor leaves are cast to ``dst_type``.

        Each leaf is cast with :meth:`torch.Tensor.type`.

        Unlike :meth:`int`, :meth:`float` and ``to(dtype)``, it returns an
        unlocked tensordict for a locked one, with a ``FutureWarning``.
        Starting with TensorDict 0.17, the default will change to a locked
        result, as these casts return.

        Args:
            dst_type (torch.dtype or str): the dtype, or the name of a tensor
                type such as ``"torch.IntTensor"``.

        Keyword Args:
            cast_filter (Callable, optional): a predicate
                ``fn(key, value) -> bool`` that selects the tensor leaves to
                cast, as in :meth:`to`. Defaults to ``None``, which casts every
                tensor leaf (a tensorclass uses its own ``cast_filter``, if it
                declares one).
        """
        if self.is_locked and not is_compiling():
            warnings.warn(
                "type() returns an unlocked tensordict for a locked one. "
                "Starting with TensorDict 0.17, the default will change to a "
                "locked result, as int(), float() and to(dtype) return. Call "
                "unlock_() on the result if you write into it.",
                FutureWarning,
                stacklevel=_user_stacklevel(),
            )

        def totype(x):
            return x.type(dst_type)

        if cast_filter is None:
            return self._fast_apply(totype)
        return _cast_filtered(self, totype, cast_filter)

    @_make_dtype_promotion
    def bfloat16(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def complex128(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def complex32(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def complex64(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def float16(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def float32(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def float64(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def int16(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def int32(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def int64(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def int8(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def qint32(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def qint8(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def quint4x2(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def quint8(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def uint16(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def uint32(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def uint64(self, *, cast_filter: Callable | None = None) -> Self: ...

    @_make_dtype_promotion
    def uint8(self, *, cast_filter: Callable | None = None) -> Self: ...
