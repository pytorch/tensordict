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
from tensordict._unbatched import UnbatchedTensor
from tensordict.base import _is_leaf_nontensor
from tensordict.tensorclass import NonTensorData

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
    if device.type == "cpu":
        device = torch.device("cpu")
    elif device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
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
    """Sends a tensordict over a UCXX endpoint, with its metadata, in one call.

    .. warning::
        ``send_tensordict`` is experimental: its API can change in a minor
        release without a deprecation cycle.

    Unlike a :class:`TensorDictPipe`, it keeps no state: each call sends the
    metadata (as JSON) and then the consolidated storage. The receiver calls
    :func:`recv_tensordict`. It needs the optional ``ucxx`` package. The
    protocol does not authenticate the peer: use it only on a trusted network.

    Args:
        endpoint (ucxx.Endpoint): the endpoint to send through, for example
            from ``ucxx.create_endpoint``.
        td (TensorDictBase): the tensordict to send.

    Keyword Args:
        consolidated (bool, optional): if ``True``, consolidates the tensordict
            (see :meth:`~tensordict.TensorDictBase.consolidate`) unless it is
            already, and sends its storage as one buffer. If ``False``, sends the
            tensor leaves of the tensordict, nested ones included, one by one, in
            the format of a pipe with ``consolidated=False``: the receiver calls
            :func:`recv_tensordict` with ``consolidated=False``. Defaults to
            ``True``.

    Raises:
        ImportError: if ``ucxx`` is not installed.
        ValueError: with ``consolidated=False``, if the tensordict holds a
            non-tensor entry other than a :class:`~tensordict.NonTensorData`
            whose value is a ``str``, ``int``, ``float``, ``bool``, ``None``,
            or a list or dict (with ``str`` keys) of them. Send it with
            ``consolidated=True``.
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
    consolidated: bool = True,
) -> TensorDictBase:
    """Receives a tensordict that :func:`send_tensordict` sends over a UCXX endpoint.

    .. warning::
        ``recv_tensordict`` is experimental: its API can change in a minor
        release without a deprecation cycle.

    It keeps no state: each call reads the metadata, then the storage. It
    needs the optional ``ucxx`` package. The sender controls the size of the
    allocation: use it only on a trusted network.

    Args:
        endpoint (ucxx.Endpoint): the endpoint to receive from.
        td (TensorDictBase, optional): a consolidated tensordict to receive
            into. If its layout, storage size and device match the data, the
            data is written into its storage, and ``td`` is returned. Otherwise
            ``td`` is not changed, and a new tensordict is returned. Defaults to
            ``None``.

    Keyword Args:
        device (torch.device, str or int, optional): the device to receive on.
            Defaults to ``None``, which uses the device of the storage of ``td``
            if it is consolidated, and the CPU otherwise.
        consolidated (bool, optional): the ``consolidated`` argument of the
            sender's :func:`send_tensordict`. With ``False``, reads the
            leaves one by one, as a pipe with ``consolidated=False`` does, and
            returns a new tensordict: ``td`` is not used. Defaults to ``True``.

    Returns:
        The received tensordict, ``td`` or a new consolidated tensordict.

    Raises:
        ImportError: if ``ucxx`` is not installed.
    """
    from tensordict._reductions import _rebuild_tensordict_files_consolidated

    if not consolidated:
        return await _recv_tensordict_uncollated(endpoint, device=device)

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
        "batch_size": list(td.batch_size),
        "unbatched": [],
        "non_tensors": [],
    }
    tensors = []
    for key, value in td.items(
        include_nested=True, leaves_only=True, is_leaf=_is_leaf_nontensor, sort=True
    ):
        wire_key = list(key) if isinstance(key, tuple) else key
        if isinstance(value, torch.Tensor):
            header["keys"].append(wire_key)
            header["dtypes"].append(str(value.dtype))
            header["shapes"].append(list(value.shape))
            if isinstance(value, UnbatchedTensor):
                header["unbatched"].append(wire_key)
            tensors.append(value)
        elif type(value) is NonTensorData and _is_json_value(value.data):
            header["non_tensors"].append([wire_key, value.data])
        else:
            raise ValueError(
                f"consolidated=False cannot send the {type(value).__name__} entry "
                f"{key!r}: it sends only NonTensorData whose value is a str, int, "
                "float, bool, None, or a list or dict (with str keys) of them. "
                "Send it with consolidated=True."
            )

    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_bytes))
    await endpoint.send(np.frombuffer(header_len, dtype=np.uint8))
    await endpoint.send(np.frombuffer(header_bytes, dtype=np.uint8).copy())

    for t in tensors:
        t_contig = t.contiguous()
        if t_contig.is_cuda:
            await endpoint.send(t_contig)
        else:
            await endpoint.send(_tensor_to_numpy(t_contig.view(-1).view(torch.uint8)))


async def _recv_tensordict_uncollated(
    endpoint, *, device: torch.device | str | None = None
) -> TensorDictBase:
    """Receive a TensorDict that ``_send_tensordict_uncollated`` sends."""
    from tensordict._td import TensorDict

    header_len_buf = np.empty(8, dtype=np.uint8)
    await endpoint.recv(header_len_buf)
    header_len = struct.unpack("<Q", header_len_buf.tobytes())[0]

    header_buf = np.empty(header_len, dtype=np.uint8)
    await endpoint.recv(header_buf)
    header = json.loads(bytes(header_buf))

    if not isinstance(header, dict) or not all(
        isinstance(header.get(field), list) for field in ("keys", "dtypes", "shapes")
    ):
        raise ValueError(
            "Invalid uncollated header: expected keys, dtypes, and shapes lists."
        )
    keys, dtype_names, shapes = header["keys"], header["dtypes"], header["shapes"]
    if not len(keys) == len(dtype_names) == len(shapes):
        raise ValueError("Invalid uncollated header: list lengths must match.")
    # Older senders send no batch size, unbatched keys or non-tensor entries.
    batch_size = header.get("batch_size", [])
    unbatched = header.get("unbatched", [])
    non_tensors = header.get("non_tensors", [])
    if not isinstance(batch_size, list) or any(
        type(dim) is not int or dim < 0 for dim in batch_size
    ):
        raise ValueError(
            "Invalid uncollated header: the batch size must be nonnegative integers."
        )
    if not isinstance(unbatched, list) or not (
        isinstance(non_tensors, list)
        and all(isinstance(entry, list) and len(entry) == 2 for entry in non_tensors)
    ):
        raise ValueError(
            "Invalid uncollated header: expected unbatched and non_tensors lists."
        )
    keys = [_decode_uncollated_key(key) for key in keys]
    non_tensor_keys = [_decode_uncollated_key(key) for key, _ in non_tensors]
    unbatched = {_decode_uncollated_key(key) for key in unbatched}
    key_set = set(keys) | set(non_tensor_keys)
    if len(key_set) != len(keys) + len(non_tensor_keys):
        raise ValueError("Invalid uncollated header: keys must be unique.")
    if any(key[:i] in key_set for key in key_set for i in range(1, len(key))):
        raise ValueError(
            "Invalid uncollated header: a leaf key cannot also have descendants."
        )
    if not unbatched.issubset(keys):
        raise ValueError("Invalid uncollated header: unknown unbatched key.")
    dtypes = []
    for key, dtype_name, shape in zip(keys, dtype_names, shapes):
        if not isinstance(dtype_name, str) or dtype_name not in _WIRE_DTYPES:
            raise ValueError("Invalid uncollated header: unknown torch dtype.")
        if not isinstance(shape, list) or any(
            type(dim) is not int or dim < 0 for dim in shape
        ):
            raise ValueError(
                "Invalid uncollated header: dimensions must be nonnegative integers."
            )
        if key not in unbatched and shape[: len(batch_size)] != batch_size:
            raise ValueError(
                "Invalid uncollated header: a shape does not start with the batch size."
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
            await endpoint.recv(buf)
        else:
            buf_np = _tensor_to_numpy(buf)
            await endpoint.recv(buf_np)
        tensor = buf.view(dtype).view(shape)
        result[key] = UnbatchedTensor(tensor) if key in unbatched else tensor

    result = TensorDict(result, batch_size=batch_size)
    for key, (_, value) in zip(non_tensor_keys, non_tensors):
        result.set_non_tensor(key, value)
    return result


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
    """A channel between two processes that streams tensordicts over UCXX.

    .. warning::
        ``TensorDictPipe`` is experimental: its API can change in a minor
        release without a deprecation cycle.

    It needs the optional ``ucxx`` package. Import it from
    ``tensordict._ucxx``: there is no public import path. Create a pipe with
    :meth:`connect` on one side and :meth:`listen` on the other, rather than
    with the constructor. The protocol does not authenticate the peer, which
    controls the size of the allocations: use it only on a trusted network.
    See :ref:`distributed`.

    With ``consolidated=True``, a pipe sends a tensordict in two phases:

    - The first message, and each message with a new structure (keys,
      shapes or dtypes), carries the metadata and then the consolidated
      storage (see :meth:`~tensordict.TensorDictBase.consolidate`). The
      receiver allocates one buffer, and builds a tensordict whose leaves
      are views into it.
    - The next messages carry the storage only. The receiver writes it into
      the buffer of the first message, without allocating memory or parsing
      metadata.

    With ``consolidated=False``, each message carries a JSON header, with the
    batch size and the non-tensor entries, and then the tensor leaves of the
    tensordict, nested ones included, one by one. The receiver builds a new
    tensordict each time, with the batch size of the sent one; its nested
    tensordicts get the same batch size. A non-tensor entry must be a
    :class:`~tensordict.NonTensorData` whose value is a ``str``, ``int``,
    ``float``, ``bool``, ``None``, or a list or dict (with ``str`` keys) of
    them: a send of any other non-tensor entry raises a ``ValueError``.

    A pipe is a context manager and an async context manager, which close
    it at the end of the block, and an async iterator: ``async for td in
    pipe`` receives until the pipe is closed.

    .. warning::
        With ``consolidated=True``, every receive with the same structure
        writes into the same tensordict, and returns it. The next receive
        overwrites the values of the previous one. Call ``clone()`` on a
        received tensordict to keep it.

    .. warning::
        ``async for td in pipe`` stops at the first error, without raising
        it. To see the error, call :meth:`arecv` or :meth:`recv` in a loop.

    Args:
        endpoint (ucxx.Endpoint): the UCXX endpoint of the connection.

    Keyword Args:
        consolidated (bool, optional): if ``True``, sends and receives
            consolidated tensordicts, in the two phases above. Both ends must
            pass the same value. Defaults to ``True``.
        listener (ucxx.Listener, optional): a listener that the pipe closes
            when it closes, as :meth:`listen` passes. Defaults to ``None``.

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
        """Connects to a pipe that listens on ``host`` and ``port``, and returns the pipe.

        The other process calls :meth:`listen` with the same port.

        Args:
            host (str): the host name or IP address of the listener.
            port (int): the port of the listener.

        Keyword Args:
            consolidated (bool, optional): if ``True``, sends and receives
                consolidated tensordicts (see the class description). Both ends must
                pass the same value. Defaults to ``True``.

        Returns:
            A connected :class:`TensorDictPipe`.

        Raises:
            ImportError: if ``ucxx`` is not installed.
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
        """Listens on ``port``, and returns a pipe to the first client that connects.

        Waits until a client calls :meth:`connect`. The listener stops when the
        pipe is closed. To accept several clients, use :class:`TensorDictServer`.

        Args:
            port (int): the port to listen on.

        Keyword Args:
            consolidated (bool, optional): if ``True``, sends and receives
                consolidated tensordicts (see the class description). Both ends must
                pass the same value. Defaults to ``True``.

        Returns:
            A connected :class:`TensorDictPipe`.

        Raises:
            ImportError: if ``ucxx`` is not installed.
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
        """Sends a tensordict through the pipe, asynchronously.

        With ``consolidated=True``, the first call, and each call where the
        structure of the tensordict changes, sends the metadata and then the
        storage. The other calls send the storage only. A tensordict that is not
        consolidated is consolidated on each call, which copies every leaf:
        consolidate it once and refresh it with
        :meth:`~tensordict.TensorDictBase.update_` to avoid the copy.

        Args:
            td (TensorDictBase): the tensordict to send.

        Raises:
            RuntimeError: if the pipe is closed.
            ValueError: with ``consolidated=False``, if ``td`` holds a non-tensor
                entry that the pipe cannot send (see the class description).
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
        """Sends a tensordict through the pipe, and waits until it is sent.

        Runs :meth:`asend` in a new event loop. In a coroutine, ``await
        pipe.asend(td)`` instead: in a thread where an event loop runs, ``send``
        raises a ``RuntimeError``.

        Args:
            td (TensorDictBase): the tensordict to send.
        """
        _run_coroutine_sync(self.asend(td))

    async def arecv(
        self,
        td: TensorDictBase | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> TensorDictBase:
        """Receives a tensordict from the pipe, asynchronously.

        With ``consolidated=True``, the first message, and each message with a new
        structure, carries the metadata: the pipe receives into ``td`` if its
        consolidated layout matches, and builds a new tensordict otherwise. The
        next messages carry the storage only, and the pipe writes it into the
        same tensordict, without allocating memory. So each receive can overwrite
        the values of the previous one: clone a tensordict to keep it.

        With ``consolidated=False``, each call returns a new tensordict with the
        batch size of the sent one (an empty batch size from an older sender).

        Args:
            td (TensorDictBase, optional): a consolidated tensordict to receive
                into, if its layout matches the data. Otherwise the pipe returns a
                new tensordict, or the tensordict of its previous receive, so use the
                return value. Not used with ``consolidated=False``. Defaults to
                ``None``, which receives into the tensordict of the previous
                receive.

        Keyword Args:
            device (torch.device, str or int, optional): the device of the storage
                that the pipe allocates for a new structure. Defaults to ``None``,
                which uses the device of the sender with ``consolidated=True``, and
                the CPU otherwise.

        Returns:
            The received tensordict.

        Raises:
            RuntimeError: if the pipe is closed, or if the sender and the receiver
                are out of sync.
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
        return await _recv_tensordict_uncollated(self._endpoint, device=device)

    def recv(
        self,
        td: TensorDictBase | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> TensorDictBase:
        """Receives a tensordict from the pipe, and waits until it is received.

        Runs :meth:`arecv` in a new event loop, with the same arguments. In a
        coroutine, ``await pipe.arecv()`` instead: in a thread where an event loop
        runs, ``recv`` raises a ``RuntimeError``.

        Args:
            td (TensorDictBase, optional): a consolidated tensordict to receive
                into, if its layout matches the data. See :meth:`arecv`. Defaults to
                ``None``.

        Keyword Args:
            device (torch.device, str or int, optional): the device of the storage
                that the pipe allocates for a new structure. See :meth:`arecv`.
                Defaults to ``None``.

        Returns:
            The received tensordict.
        """
        return _run_coroutine_sync(self.arecv(td, device=device))

    async def aclose(self) -> None:
        """Closes the pipe, its endpoint and its listener (if any), asynchronously.

        Closing a closed pipe does nothing.
        """
        if not self._closed:
            self._closed = True
            if self._listener is not None:
                self._listener.close()
            await self._endpoint.close()

    def close(self) -> None:
        """Closes the pipe, and waits until it is closed.

        Runs :meth:`aclose` in a new event loop. In a coroutine, ``await
        pipe.aclose()`` instead. A pipe also closes at the end of a ``with`` or
        ``async with`` block, and when it is garbage collected.
        """
        _run_coroutine_sync(self.aclose())

    @property
    def closed(self) -> bool:
        """Whether the pipe is closed."""
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
    """A listener that accepts several clients, and gives a :class:`TensorDictPipe` for each.

    .. warning::
        ``TensorDictServer`` is experimental: its API can change in a minor
        release without a deprecation cycle.

    It needs the optional ``ucxx`` package. Import it from
    ``tensordict._ucxx``: there is no public import path. Get the pipes with
    :meth:`serve`, or with ``async for pipe in server``. It starts
    listening when you do either. The protocol does not authenticate the
    clients: use it only on a trusted network. A server is an async context
    manager, which closes it at the end of the block.

    Args:
        port (int): the port to listen on.

    Keyword Args:
        consolidated (bool, optional): the ``consolidated`` argument of the
            pipes (see :class:`TensorDictPipe`). The clients must pass the
            same value to :meth:`TensorDictPipe.connect`. Defaults to
            ``True``.

    Raises:
        ImportError: if ``ucxx`` is not installed.

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
        """Accepts connections, and runs ``on_connect`` on the pipe of each new client.

        Runs until :meth:`close` is called. Each ``on_connect`` call runs as
        its own task, so that the clients are served at the same time. ``async
        for pipe in server`` is the other way to get the pipes.

        Args:
            on_connect (Callable): a coroutine function that takes the
                :class:`TensorDictPipe` of a client.
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
        """Closes the server: it stops accepting connections, and :meth:`serve` returns.

        The pipes of the clients stay open.
        """
        self._closed = True
        if self._listener is not None:
            self._listener.close()

    async def aclose(self) -> None:
        """Closes the server, as :meth:`close` does, in a coroutine."""
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
