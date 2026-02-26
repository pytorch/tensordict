# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import hashlib
import json
import struct
from typing import Any, AsyncIterator, Awaitable, Callable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from tensordict.base import TensorDictBase

try:
    import ucxx

    _has_ucxx = True
except ImportError:
    _has_ucxx = False

_NEW_SCHEMA_FLAG = b"\x00"
_SAME_SCHEMA_FLAG = b"\x01"


def _check_ucxx():
    if not _has_ucxx:
        raise ImportError(
            "ucxx is required for TensorDictPipe but is not installed. "
            "Install it with: conda install -c rapidsai ucxx"
        )


def _metadata_hash(metadata: dict) -> bytes:
    """Deterministic hash of consolidation metadata for schema comparison."""
    raw = json.dumps(metadata, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).digest()


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """View a CPU tensor as a numpy array without copying."""
    return tensor.detach().numpy()


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    return loop


def _run_coroutine_sync(coro):
    """Run an async coroutine synchronously.

    If there is already a running event loop, schedules on it via
    run_coroutine_threadsafe. Otherwise creates a new loop.
    """
    loop = _get_or_create_event_loop()
    if loop is not None and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    return asyncio.run(coro)


async def send_tensordict(
    endpoint,
    td: TensorDictBase,
    *,
    consolidated: bool = True,
) -> None:
    """Send a TensorDict over a UCXX endpoint (stateless, always sends metadata).

    Args:
        endpoint: a UCXX endpoint object.
        td: the TensorDict to send.

    Keyword Args:
        consolidated (bool): if ``True``, consolidates into a single buffer
            before sending. Defaults to ``True``.
    """
    _check_ucxx()

    if consolidated:
        td_c = td if td.is_consolidated() else td.consolidate(metadata=True)
        metadata = td_c._consolidated["metadata"]
        storage = td_c._consolidated["storage"]

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        meta_len = struct.pack("<Q", len(meta_bytes))

        await endpoint.send(np.frombuffer(meta_len, dtype=np.uint8))
        await endpoint.send(np.frombuffer(meta_bytes, dtype=np.uint8).copy())

        if storage.is_cuda:
            await endpoint.send(storage)
        else:
            await endpoint.send(_tensor_to_numpy(storage))
    else:
        await _send_tensordict_uncollated(endpoint, td)


async def recv_tensordict(
    endpoint,
    td: TensorDictBase | None = None,
    *,
    device: torch.device | str | None = None,
) -> TensorDictBase:
    """Receive a TensorDict from a UCXX endpoint (stateless, always expects metadata).

    Args:
        endpoint: a UCXX endpoint object.
        td: if provided and already consolidated, receives directly into
            its storage buffer (in-place update). Otherwise a new TensorDict
            is constructed.

    Keyword Args:
        device: device on which to allocate the storage buffer. Defaults to CPU.

    Returns:
        The received TensorDict.
    """
    _check_ucxx()
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

    if td is not None and td.is_consolidated():
        storage = td._consolidated["storage"]
        if storage.is_cuda:
            await endpoint.recv(storage)
        else:
            storage_np = _tensor_to_numpy(storage)
            await endpoint.recv(storage_np)
        return td

    effective_device = torch.device(device) if device is not None else torch.device("cpu")
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
    for _dtype, _shape, start, stop, _pad in leaves.values():
        total = max(total, stop)
    for key, val in metadata.items():
        if isinstance(val, dict) and "cls" in val:
            total = max(total, _compute_total_bytes(val))
    return total


class TensorDictPipe:
    """A persistent point-to-point channel for streaming TensorDicts over UCXX.

    Implements a two-phase protocol:

    - **Handshake** (first send): transmits consolidation metadata and data.
      The receiver allocates a buffer and builds a TensorDict whose leaves
      are views into the consolidated storage.
    - **Steady-state** (subsequent sends with same schema): transmits only
      the raw storage buffer. The receiver overwrites its pre-allocated
      buffer in-place — zero allocation, zero metadata parsing.

    Examples:
        >>> # sender
        >>> pipe = await TensorDictPipe.connect("node-b", 13337)
        >>> await pipe.asend(td)
        >>> # receiver
        >>> pipe = await TensorDictPipe.listen(13337)
        >>> td = await pipe.arecv()
        >>> # steady-state loop (zero-alloc after first iteration)
        >>> async for td in pipe:
        ...     train(td)
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
        self._closed = False

    @classmethod
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
        _check_ucxx()
        endpoint = await ucxx.create_endpoint(host, port)
        return cls(endpoint, consolidated=consolidated)

    @classmethod
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
        _check_ucxx()
        pipe_future: asyncio.Future[TensorDictPipe] = asyncio.get_event_loop().create_future()

        async def _on_connect(ep):
            if not pipe_future.done():
                pipe = cls(ep, consolidated=consolidated, listener=listener)
                pipe_future.set_result(pipe)

        listener = ucxx.create_listener(_on_connect, port=port)
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
        td_c = td if td.is_consolidated() else td.consolidate(metadata=True)
        metadata = td_c._consolidated["metadata"]
        storage = td_c._consolidated["storage"]

        schema_hash = _metadata_hash(metadata)

        if schema_hash != self._send_schema_hash:
            self._send_schema_hash = schema_hash

            await self._endpoint.send(np.frombuffer(_NEW_SCHEMA_FLAG, dtype=np.uint8).copy())

            metadata_with_size = dict(metadata)
            metadata_with_size["_total_bytes"] = storage.numel()
            meta_bytes = json.dumps(metadata_with_size, sort_keys=True).encode("utf-8")
            meta_len = struct.pack("<Q", len(meta_bytes))

            await self._endpoint.send(np.frombuffer(meta_len, dtype=np.uint8))
            await self._endpoint.send(np.frombuffer(meta_bytes, dtype=np.uint8).copy())
        else:
            await self._endpoint.send(np.frombuffer(_SAME_SCHEMA_FLAG, dtype=np.uint8).copy())

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

        On the first call, builds a new TensorDict from received metadata.
        On subsequent calls with the same schema, overwrites the existing
        storage in-place (zero allocation).

        Args:
            td: if provided and consolidated, receives into its storage.
                If ``None``, uses the internally cached TensorDict from
                the previous receive (if any).

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
            self._recv_schema_hash = _metadata_hash(metadata)

            effective_device = torch.device(device) if device is not None else torch.device("cpu")
            storage = torch.empty(total_bytes, dtype=torch.uint8, device=effective_device)

            if storage.is_cuda:
                await self._endpoint.recv(storage)
            else:
                storage_np = _tensor_to_numpy(storage)
                await self._endpoint.recv(storage_np)

            result = _rebuild_tensordict_files_consolidated(metadata, storage)
            self._recv_td = result
            return result

        # SAME_SCHEMA_FLAG: receive into existing buffer
        recv_target = td if td is not None else self._recv_td
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

        effective_device = torch.device(device) if device is not None else torch.device("cpu")
        result = {}
        for key, dtype_str, shape in zip(
            header["keys"], header["dtypes"], header["shapes"]
        ):
            dtype = getattr(torch, dtype_str.replace("torch.", ""))
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
            td: if provided, receives into this TensorDict in-place.

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
                self.close()
            except Exception:
                pass


class TensorDictServer:
    """A one-to-many listener that accepts connections and yields TensorDictPipe instances.

    Wraps ``ucxx.create_listener`` to provide a server that can handle
    multiple clients, each getting their own :class:`TensorDictPipe`.

    Examples:
        >>> server = TensorDictServer(port=13337)
        >>> async def handler(pipe):
        ...     async for td in pipe:
        ...         process(td)
        >>> await server.serve(handler)
    """

    def __init__(self, port: int, *, consolidated: bool = True):
        _check_ucxx()
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

        self._listener = ucxx.create_listener(_on_connect, port=self._port)

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

        self._listener = ucxx.create_listener(_on_connect, port=self._port)

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
