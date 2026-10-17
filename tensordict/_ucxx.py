# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import functools
import hashlib
import importlib.util
import json
import struct
from typing import Any, AsyncIterator, Awaitable, Callable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from tensordict.base import TensorDictBase

_has_ucxx = importlib.util.find_spec("ucxx") is not None
_ucxx = None

_NEW_SCHEMA_FLAG = b"\x00"
_SAME_SCHEMA_FLAG = b"\x01"

# Include all installed torch dtypes, including float8 and dtype aliases.
_WIRE_DTYPES = {
    f"torch.{name}": value
    for name, value in vars(torch).items()
    if isinstance(value, torch.dtype)
}


def _check_ucxx(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _ucxx
        if _ucxx is None:
            if not _has_ucxx:
                raise ImportError(
                    "ucxx is required for TensorDictPipe but is not installed. "
                    "Install it with: conda install -c rapidsai ucxx"
                )
            import ucxx

            _ucxx = ucxx
        return func(*args, **kwargs)

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        global _ucxx
        if _ucxx is None:
            if not _has_ucxx:
                raise ImportError(
                    "ucxx is required for TensorDictPipe but is not installed. "
                    "Install it with: conda install -c rapidsai ucxx"
                )
            import ucxx

            _ucxx = ucxx
        return await func(*args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


def _metadata_hash(metadata: dict) -> bytes:
    """Deterministic hash of consolidation metadata for schema comparison."""
    raw = json.dumps(metadata, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).digest()


def _can_recv_into(
    td: TensorDictBase,
    total_bytes: int,
    device: torch.device,
    schema_hash: bytes,
) -> bool:
    """Whether the consolidated storage of ``td`` matches the incoming schema."""
    if not td.is_consolidated():
        return False
    storage = td._consolidated["storage"]
    if storage.numel() != total_bytes or storage.device != device:
        return False
    try:
        td_schema_hash = _metadata_hash(td._consolidated["metadata"])
    except (TypeError, ValueError):
        # Unserializable metadata cannot match metadata sent over JSON.
        return False
    return td_schema_hash == schema_hash


def _decode_uncollated_key(key) -> tuple[str, ...]:
    """Canonicalize a JSON-encoded NestedKey before checking uniqueness."""
    if isinstance(key, str):
        return (key,)
    if isinstance(key, list) and key:
        return tuple(part for item in key for part in _decode_uncollated_key(item))
    raise ValueError(
        "Invalid uncollated key: expected a string or nonempty nested path."
    )


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """View a CPU tensor as a numpy array without copying."""
    return tensor.detach().numpy()


def _infer_device(td: "TensorDictBase") -> torch.device:
    """Return the device of a TensorDict, falling back to inspecting leaf tensors.

    TensorDict.device can be None even when all leaves live on CUDA
    (e.g. when constructed without an explicit ``device=``).  This helper
    ensures we always get a concrete device for consolidation / allocation.
    """
    device = td.device
    if device is not None:
        return device
    for v in td.values(True, True):
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    return loop


def _run_coroutine_sync(coro):
    """Run an async coroutine synchronously.

    Runs ``coro`` on a new event loop. If an event loop is already running
    in this thread, raises a ``RuntimeError`` instead: blocking that thread
    until ``coro`` finishes would stop the loop from ever running it.
    """
    loop = _get_or_create_event_loop()
    if loop is not None and loop.is_running():
        qualname, name = coro.__qualname__, coro.__name__
        coro.close()
        raise RuntimeError(
            f"{qualname}() cannot run synchronously inside a running event "
            f"loop: waiting for it would block the loop forever. Await it "
            f"instead, e.g. `await pipe.{name}()`."
        )
    return asyncio.run(coro)


# Close tasks scheduled by TensorDictPipe.__del__, kept here so that they
# are not garbage collected before they finish.
_PENDING_CLOSE_TASKS: set[asyncio.Task] = set()


def _on_close_task_done(task: asyncio.Task) -> None:
    _PENDING_CLOSE_TASKS.discard(task)
    if not task.cancelled():
        # Retrieve the exception so that asyncio does not log it:
        # __del__ ignores errors raised while closing.
        task.exception()


@_check_ucxx
async def send_tensordict(
    endpoint,
    td: TensorDictBase,
    *,
    consolidated: bool = True,
) -> None:
    """Send a TensorDict over a UCXX endpoint (stateless, always sends metadata).

    .. warning::
        ``send_tensordict`` is experimental: its API can change in a minor
        release without a deprecation cycle.

    Args:
        endpoint: a UCXX endpoint object.
        td: the TensorDict to send.

    Keyword Args:
        consolidated (bool): if ``True``, consolidates into a single buffer
            before sending. Defaults to ``True``.
    """
    if consolidated:
        if td.is_consolidated():
            td_c = td
        else:
            device = _infer_device(td)
            td_c = td.consolidate(metadata=True, device=device)
        metadata = td_c._consolidated["metadata"]
        storage = td_c._consolidated["storage"]

        meta_bytes = json.dumps(metadata).encode("utf-8")
        meta_len = struct.pack("<Q", len(meta_bytes))

        await endpoint.send(np.frombuffer(meta_len, dtype=np.uint8))
        await endpoint.send(np.frombuffer(meta_bytes, dtype=np.uint8).copy())

        if storage.is_cuda:
            await endpoint.send(storage)
        else:
            await endpoint.send(_tensor_to_numpy(storage))
    else:
        await _send_tensordict_uncollated(endpoint, td)


@_check_ucxx
async def recv_tensordict(
    endpoint,
    td: TensorDictBase | None = None,
    *,
    device: torch.device | str | None = None,
) -> TensorDictBase:
    """Receive a TensorDict from a UCXX endpoint (stateless, always expects metadata).

    .. warning::
        ``recv_tensordict`` is experimental: its API can change in a minor
        release without a deprecation cycle.

    Args:
        endpoint: a UCXX endpoint object.
        td: if provided and consolidated with matching layout, storage size,
            and device, receives directly into its storage buffer (in-place
            update). Otherwise returns a new TensorDict without modifying ``td``.

    Keyword Args:
        device: device on which to receive. If omitted, uses the storage
            device of a consolidated ``td``, even when its layout does not
            match. Otherwise defaults to CPU.

    Returns:
        The received TensorDict.
    """
    from tensordict._reductions import _rebuild_tensordict_files_consolidated

    meta_len_buf = np.empty(8, dtype=np.uint8)
    await endpoint.recv(meta_len_buf)
    meta_len = struct.unpack("<Q", meta_len_buf.tobytes())[0]

    meta_buf = np.empty(meta_len, dtype=np.uint8)
    await endpoint.recv(meta_buf)
    metadata = json.loads(bytes(meta_buf))

    total_bytes = metadata.pop("_total_bytes", None)
    if total_bytes is None:
        total_bytes = _compute_total_bytes(metadata)

    effective_device = (
        torch.device(device) if device is not None else torch.device("cpu")
    )
    if td is not None and td.is_consolidated():
        storage = td._consolidated["storage"]
        if device is None:
            effective_device = storage.device
        if _can_recv_into(td, total_bytes, effective_device, _metadata_hash(metadata)):
            if storage.is_cuda:
                await endpoint.recv(storage)
            else:
                storage_np = _tensor_to_numpy(storage)
                await endpoint.recv(storage_np)
            return td

    storage = torch.empty(total_bytes, dtype=torch.uint8, device=effective_device)

    if storage.is_cuda:
        await endpoint.recv(storage)
    else:
        storage_np = _tensor_to_numpy(storage)
        await endpoint.recv(storage_np)

    return _rebuild_tensordict_files_consolidated(metadata, storage)


async def _send_tensordict_uncollated(endpoint, td: TensorDictBase) -> None:
    """Send a TensorDict leaf-by-leaf without consolidation."""
    header = {
        "keys": [],
        "dtypes": [],
        "shapes": [],
    }
    tensors = []
    for key in td.sorted_keys:
        value = td._get_str(key, None)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            header["keys"].append(key)
            header["dtypes"].append(str(value.dtype))
            header["shapes"].append(list(value.shape))
            tensors.append(value)

    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_bytes))
    await endpoint.send(np.frombuffer(header_len, dtype=np.uint8))
    await endpoint.send(np.frombuffer(header_bytes, dtype=np.uint8).copy())

    for t in tensors:
        t_contig = t.contiguous()
        if t_contig.is_cuda:
            await endpoint.send(t_contig)
        else:
            await endpoint.send(_tensor_to_numpy(t_contig.view(torch.uint8)))


def _compute_total_bytes(metadata: dict) -> int:
    """Compute total storage bytes from consolidation metadata."""
    total = 0
    leaves = metadata.get("leaves", {})
    for leaf in leaves.values():
        total = max(total, leaf[3])
    for _key, val in metadata.items():
        if isinstance(val, dict) and "cls" in val:
            total = max(total, _compute_total_bytes(val))
    return total


class TensorDictPipe:
    """A persistent point-to-point channel for streaming TensorDicts over UCXX.

    .. warning::
        ``TensorDictPipe`` is experimental: its API can change in a minor
        release without a deprecation cycle.

    It needs the optional ``ucxx`` package. Import it from
    ``tensordict._ucxx``: there is no public import path.

    Implements a two-phase protocol:

    - Handshake (first send): transmits consolidation metadata and data.
      The receiver allocates a buffer and builds a TensorDict whose leaves
      are views into the consolidated storage.
    - Steady state (subsequent sends with same schema): transmits only
      the raw storage buffer. The receiver overwrites its pre-allocated
      buffer in place, without a new allocation and without metadata parsing.

    .. warning::
        With ``consolidated=True`` (the default), every receive with the same
        schema writes into the same TensorDict, and returns it. The next
        receive overwrites the values of the previous one. Call ``clone()``
        on a received TensorDict to keep it.

    .. warning::
        ``async for td in pipe`` stops at the first error, without raising
        it. To see the error, call :meth:`arecv` or :meth:`recv` in a loop.

    Examples:
        The sender and the receiver run in two processes, on the same or
        on two machines. This example is not tested: it needs ``ucxx`` and
        two processes.

        .. code-block:: python

            # receiver, on host "node-b"
            import asyncio
            from tensordict._ucxx import TensorDictPipe

            async def main():
                pipe = await TensorDictPipe.listen(13337)
                td = await pipe.arecv()  # first receive: allocates the buffer
                async for td in pipe:  # same schema: overwrites td in place
                    print(td["obs"].sum())

            asyncio.run(main())

        .. code-block:: python

            # sender
            import asyncio
            import torch
            from tensordict import TensorDict
            from tensordict._ucxx import TensorDictPipe

            async def main():
                pipe = await TensorDictPipe.connect("node-b", 13337)
                for _ in range(10):
                    await pipe.asend(TensorDict(obs=torch.randn(64, 84)))
                await pipe.aclose()

            asyncio.run(main())
    """

    def __init__(
        self,
        endpoint,
        *,
        consolidated: bool = True,
        listener: Any | None = None,
    ):
        self._endpoint = endpoint
        self._consolidated = consolidated
        self._listener = listener
        self._send_schema_hash: bytes | None = None
        self._recv_schema_hash: bytes | None = None
        self._recv_td: TensorDictBase | None = None
        # Storage size and device of the last schema received.
        self._recv_storage_spec: tuple[int, torch.device] | None = None
        self._closed = False

    @classmethod
    @_check_ucxx
    async def connect(
        cls,
        host: str,
        port: int,
        *,
        consolidated: bool = True,
    ) -> TensorDictPipe:
        """Connect to a remote TensorDictPipe listener.

        Args:
            host: hostname or IP address of the remote listener.
            port: port number.

        Keyword Args:
            consolidated (bool): whether to use consolidated transfers.
                Defaults to ``True``.

        Returns:
            A connected TensorDictPipe.
        """
        endpoint = await _ucxx.create_endpoint(host, port)
        return cls(endpoint, consolidated=consolidated)

    @classmethod
    @_check_ucxx
    async def listen(
        cls,
        port: int,
        *,
        consolidated: bool = True,
    ) -> TensorDictPipe:
        """Listen for an incoming connection and return a pipe to the first client.

        Args:
            port: port number to listen on.

        Keyword Args:
            consolidated (bool): whether to use consolidated transfers.
                Defaults to ``True``.

        Returns:
            A connected TensorDictPipe for the first accepted client.
        """
        pipe_future: asyncio.Future[TensorDictPipe] = (
            asyncio.get_event_loop().create_future()
        )

        async def _on_connect(ep):
            if not pipe_future.done():
                pipe = cls(ep, consolidated=consolidated, listener=listener)
                pipe_future.set_result(pipe)

        listener = _ucxx.create_listener(_on_connect, port=port)
        return await pipe_future

    async def asend(self, td: TensorDictBase) -> None:
        """Send a TensorDict through this pipe (async).

        On the first call (or when the schema changes), sends metadata + data.
        On subsequent calls with the same schema, sends only the raw buffer.

        Args:
            td: the TensorDict to send.
        """
        if self._closed:
            raise RuntimeError("Cannot send on a closed pipe.")

        if self._consolidated:
            await self._asend_consolidated(td)
        else:
            await self._asend_uncollated(td)

    async def _asend_consolidated(self, td: TensorDictBase) -> None:
        if td.is_consolidated():
            td_c = td
        else:
            device = _infer_device(td)
            td_c = td.consolidate(metadata=True, device=device)
        metadata = td_c._consolidated["metadata"]
        storage = td_c._consolidated["storage"]

        schema_hash = _metadata_hash(metadata)

        if schema_hash != self._send_schema_hash:
            self._send_schema_hash = schema_hash

            await self._endpoint.send(
                np.frombuffer(_NEW_SCHEMA_FLAG, dtype=np.uint8).copy()
            )

            metadata_with_size = dict(metadata)
            metadata_with_size["_total_bytes"] = storage.numel()
            metadata_with_size["_device"] = str(storage.device)
            meta_bytes = json.dumps(metadata_with_size).encode("utf-8")
            meta_len = struct.pack("<Q", len(meta_bytes))

            await self._endpoint.send(np.frombuffer(meta_len, dtype=np.uint8))
            await self._endpoint.send(np.frombuffer(meta_bytes, dtype=np.uint8).copy())
        else:
            await self._endpoint.send(
                np.frombuffer(_SAME_SCHEMA_FLAG, dtype=np.uint8).copy()
            )

        if storage.is_cuda:
            await self._endpoint.send(storage)
        else:
            await self._endpoint.send(_tensor_to_numpy(storage))

    async def _asend_uncollated(self, td: TensorDictBase) -> None:
        await _send_tensordict_uncollated(self._endpoint, td)

    def send(self, td: TensorDictBase) -> None:
        """Send a TensorDict through this pipe (sync).

        Blocking wrapper around :meth:`asend`.

        Args:
            td: the TensorDict to send.
        """
        _run_coroutine_sync(self.asend(td))

    async def arecv(
        self,
        td: TensorDictBase | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> TensorDictBase:
        """Receive a TensorDict from this pipe (async).

        On the first call, receives into ``td`` if its consolidated layout
        matches, otherwise builds a new TensorDict from the received metadata.
        On subsequent calls with the same schema, overwrites the existing
        storage in-place (zero allocation).

        Args:
            td: if provided and consolidated, receives into its storage
                when its layout matches the received data. Otherwise the
                pipe returns a new TensorDict, or on a message with the same
                schema its own TensorDict from the previous receive, so use
                the return value. If ``None``, uses the internally cached
                TensorDict from the previous receive (if any). Not used when
                the pipe has ``consolidated=False``.

        Keyword Args:
            device: device for storage allocation on first receive.

        Returns:
            The received TensorDict.
        """
        if self._closed:
            raise RuntimeError("Cannot recv on a closed pipe.")

        if self._consolidated:
            return await self._arecv_consolidated(td, device=device)
        return await self._arecv_uncollated(td, device=device)

    async def _arecv_consolidated(
        self,
        td: TensorDictBase | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> TensorDictBase:
        from tensordict._reductions import _rebuild_tensordict_files_consolidated

        flag_buf = np.empty(1, dtype=np.uint8)
        await self._endpoint.recv(flag_buf)
        flag = bytes(flag_buf)

        if flag == _NEW_SCHEMA_FLAG:
            meta_len_buf = np.empty(8, dtype=np.uint8)
            await self._endpoint.recv(meta_len_buf)
            meta_len = struct.unpack("<Q", meta_len_buf.tobytes())[0]

            meta_buf = np.empty(meta_len, dtype=np.uint8)
            await self._endpoint.recv(meta_buf)
            metadata = json.loads(bytes(meta_buf))

            total_bytes = metadata.pop("_total_bytes")
            sender_device = metadata.pop("_device", "cpu")
            self._recv_schema_hash = _metadata_hash(metadata)

            if device is not None:
                effective_device = torch.device(device)
            else:
                effective_device = torch.device(sender_device)
            self._recv_storage_spec = (total_bytes, effective_device)

            if td is not None and _can_recv_into(
                td, total_bytes, effective_device, self._recv_schema_hash
            ):
                storage = td._consolidated["storage"]
                if storage.is_cuda:
                    await self._endpoint.recv(storage)
                else:
                    await self._endpoint.recv(_tensor_to_numpy(storage))
                self._recv_td = td
                return td

            storage = torch.empty(
                total_bytes, dtype=torch.uint8, device=effective_device
            )

            if storage.is_cuda:
                await self._endpoint.recv(storage)
            else:
                storage_np = _tensor_to_numpy(storage)
                await self._endpoint.recv(storage_np)

            result = _rebuild_tensordict_files_consolidated(metadata, storage)
            self._recv_td = result
            return result

        if flag != _SAME_SCHEMA_FLAG:
            raise RuntimeError(f"Unknown TensorDictPipe protocol flag: {flag!r}.")

        # SAME_SCHEMA_FLAG: receive into td if it has the layout of the last
        # schema, otherwise into the buffer of the previous receive. When td
        # is that buffer, the check cannot change the target, so skip it.
        recv_target = self._recv_td
        if td is not None and (
            td is self._recv_td
            or self._recv_storage_spec is None
            or _can_recv_into(td, *self._recv_storage_spec, self._recv_schema_hash)
        ):
            recv_target = td
        if recv_target is None or not recv_target.is_consolidated():
            raise RuntimeError(
                "Received same-schema flag but no consolidated TensorDict "
                "is available for in-place receive. The sender and receiver "
                "may be out of sync."
            )

        storage = recv_target._consolidated["storage"]
        if storage.is_cuda:
            await self._endpoint.recv(storage)
        else:
            storage_np = _tensor_to_numpy(storage)
            await self._endpoint.recv(storage_np)

        return recv_target

    async def _arecv_uncollated(
        self,
        td: TensorDictBase | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> TensorDictBase:
        from tensordict._td import TensorDict

        header_len_buf = np.empty(8, dtype=np.uint8)
        await self._endpoint.recv(header_len_buf)
        header_len = struct.unpack("<Q", header_len_buf.tobytes())[0]

        header_buf = np.empty(header_len, dtype=np.uint8)
        await self._endpoint.recv(header_buf)
        header = json.loads(bytes(header_buf))

        if not isinstance(header, dict) or not all(
            isinstance(header.get(field), list)
            for field in ("keys", "dtypes", "shapes")
        ):
            raise ValueError(
                "Invalid uncollated header: expected keys, dtypes, and shapes lists."
            )
        keys, dtype_names, shapes = header["keys"], header["dtypes"], header["shapes"]
        if not len(keys) == len(dtype_names) == len(shapes):
            raise ValueError("Invalid uncollated header: list lengths must match.")
        keys = [_decode_uncollated_key(key) for key in keys]
        if len(set(keys)) != len(keys):
            raise ValueError("Invalid uncollated header: keys must be unique.")
        dtypes = []
        for dtype_name, shape in zip(dtype_names, shapes):
            if not isinstance(dtype_name, str) or dtype_name not in _WIRE_DTYPES:
                raise ValueError("Invalid uncollated header: unknown torch dtype.")
            if not isinstance(shape, list) or any(
                type(dim) is not int or dim < 0 for dim in shape
            ):
                raise ValueError(
                    "Invalid uncollated header: dimensions must be nonnegative integers."
                )
            dtypes.append(_WIRE_DTYPES[dtype_name])

        effective_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        result = {}
        for key, dtype, shape in zip(keys, dtypes, shapes):
            numel = 1
            for s in shape:
                numel *= s
            n_bytes = numel * torch.tensor([], dtype=dtype).element_size()
            buf = torch.empty(n_bytes, dtype=torch.uint8, device=effective_device)
            if buf.is_cuda:
                await self._endpoint.recv(buf)
            else:
                buf_np = _tensor_to_numpy(buf)
                await self._endpoint.recv(buf_np)
            result[key] = buf.view(dtype).view(shape)

        return TensorDict(result, batch_size=[])

    def recv(
        self,
        td: TensorDictBase | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> TensorDictBase:
        """Receive a TensorDict from this pipe (sync).

        Blocking wrapper around :meth:`arecv`.

        Args:
            td: if provided, receives into this TensorDict in-place when
                it is consolidated and its layout matches the received data.
                Otherwise the pipe returns a new TensorDict or its own
                TensorDict from the previous receive, so use the return
                value.

        Keyword Args:
            device: device for storage allocation on first receive.

        Returns:
            The received TensorDict.
        """
        return _run_coroutine_sync(self.arecv(td, device=device))

    async def aclose(self) -> None:
        """Close this pipe (async)."""
        if not self._closed:
            self._closed = True
            if self._listener is not None:
                self._listener.close()
            await self._endpoint.close()

    def close(self) -> None:
        """Close this pipe (sync)."""
        _run_coroutine_sync(self.aclose())

    @property
    def closed(self) -> bool:
        return self._closed

    def __aiter__(self) -> TensorDictPipe:
        return self

    async def __anext__(self) -> TensorDictBase:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self.arecv()
        except Exception:
            raise StopAsyncIteration

    async def __aenter__(self) -> TensorDictPipe:
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    def __enter__(self) -> TensorDictPipe:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self):
        if not self._closed:
            try:
                loop = _get_or_create_event_loop()
                if loop is not None and loop.is_running():
                    # close() cannot block the thread of a running loop:
                    # schedule the close on the loop instead.
                    task = loop.create_task(self.aclose())
                    _PENDING_CLOSE_TASKS.add(task)
                    task.add_done_callback(_on_close_task_done)
                else:
                    self.close()
            except Exception:
                pass


class TensorDictServer:
    """A one-to-many listener that accepts connections and yields TensorDictPipe instances.

    .. warning::
        ``TensorDictServer`` is experimental: its API can change in a minor
        release without a deprecation cycle.

    Wraps ``ucxx.create_listener`` to provide a server that can handle
    multiple clients, each getting their own :class:`TensorDictPipe`.
    It needs the optional ``ucxx`` package. Import it from
    ``tensordict._ucxx``.

    Examples:
        Each client connects with :meth:`TensorDictPipe.connect`. This
        example is not tested: it needs ``ucxx`` and client processes.

        .. code-block:: python

            import asyncio
            from tensordict._ucxx import TensorDictServer

            async def handler(pipe):
                async for td in pipe:
                    print(td["obs"].shape)

            async def main():
                server = TensorDictServer(port=13337)
                await server.serve(handler)  # runs until server.close()

            asyncio.run(main())
    """

    @_check_ucxx
    def __init__(self, port: int, *, consolidated: bool = True):
        self._port = port
        self._consolidated = consolidated
        self._listener = None
        self._closed = False

    async def serve(
        self,
        on_connect: Callable[[TensorDictPipe], Awaitable[None]],
    ) -> None:
        """Accept connections and call ``on_connect`` for each new client.

        This runs indefinitely until :meth:`close` is called.

        Args:
            on_connect: an async callable that receives a :class:`TensorDictPipe`
                for each new client connection.
        """
        queue: asyncio.Queue[TensorDictPipe] = asyncio.Queue()

        async def _on_connect(ep):
            pipe = TensorDictPipe(ep, consolidated=self._consolidated)
            await queue.put(pipe)

        self._listener = _ucxx.create_listener(_on_connect, port=self._port)

        while not self._closed:
            try:
                pipe = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            asyncio.ensure_future(on_connect(pipe))

    async def __aiter__(self) -> AsyncIterator[TensorDictPipe]:
        queue: asyncio.Queue[TensorDictPipe] = asyncio.Queue()

        async def _on_connect(ep):
            pipe = TensorDictPipe(ep, consolidated=self._consolidated)
            await queue.put(pipe)

        self._listener = _ucxx.create_listener(_on_connect, port=self._port)

        while not self._closed:
            try:
                pipe = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            yield pipe

    def close(self) -> None:
        """Close the server and stop accepting connections."""
        self._closed = True
        if self._listener is not None:
            self._listener.close()

    async def aclose(self) -> None:
        """Close the server (async)."""
        self.close()

    async def __aenter__(self) -> TensorDictServer:
        return self

    async def __aexit__(self, *args) -> None:
        self.close()


__all__ = [
    "TensorDictPipe",
    "TensorDictServer",
    "send_tensordict",
    "recv_tensordict",
]
