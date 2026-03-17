# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Private module for cross-mesh DTensor transfer utilities.

Everything here is private (underscore-prefixed) and subject to change.
The only public API surface is ``dtensor_send`` / ``dtensor_recv`` on
``TensorDictBase``.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import struct
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable, Sequence

import torch
from torch import Tensor

_has_ucxx = importlib.util.find_spec("ucxx") is not None


# ---------------------------------------------------------------------------
# Placement helpers -- lightweight stand-ins so that compute_transfer_plan
# can be tested without importing torch.distributed.
# ---------------------------------------------------------------------------


def _placement_is_shard(p) -> bool:
    return hasattr(p, "dim") and not _placement_is_partial(p)


def _placement_shard_dim(p) -> int:
    return p.dim


def _placement_is_replicate(p) -> bool:
    return hasattr(p, "is_replicate") and p.is_replicate()


def _placement_is_partial(p) -> bool:
    return hasattr(p, "is_partial") and p.is_partial()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ShardSpec:
    """What a single rank holds, expressed as slices into the global tensor."""

    rank: int
    slices: tuple[slice, ...]

    def __repr__(self) -> str:
        slices_str = ", ".join(f"{s.start}:{s.stop}" for s in self.slices)
        return f"ShardSpec(rank={self.rank}, slices=[{slices_str}])"


@dataclass(frozen=True, slots=True)
class _ChunkTransfer:
    """One point-to-point transfer instruction."""

    src_rank: int
    dst_rank: int
    src_slices: tuple[slice, ...]
    dst_slices: tuple[slice, ...]
    global_slices: tuple[slice, ...]

    @property
    def numel(self) -> int:
        """Number of elements this transfer moves."""
        n = 1
        for s in self.global_slices:
            n *= s.stop - s.start
        return n

    def nbytes(self, itemsize: int = 1) -> int:
        """Number of bytes (numel * itemsize) this transfer moves."""
        return self.numel * itemsize

    def __repr__(self) -> str:
        gl = ", ".join(f"{s.start}:{s.stop}" for s in self.global_slices)
        return (
            f"ChunkTransfer(src={self.src_rank}->dst={self.dst_rank}, "
            f"global=[{gl}], numel={self.numel})"
        )


@dataclass(slots=True)
class _TransferPlan:
    """Complete plan for transferring one tensor between two sharding specs."""

    global_shape: torch.Size
    transfers: list[_ChunkTransfer] = field(default_factory=list)

    def sends_for_rank(self, rank: int) -> list[_ChunkTransfer]:
        return [t for t in self.transfers if t.src_rank == rank]

    def recvs_for_rank(self, rank: int) -> list[_ChunkTransfer]:
        return [t for t in self.transfers if t.dst_rank == rank]

    def __repr__(self) -> str:
        return (
            f"TransferPlan(shape={tuple(self.global_shape)}, "
            f"transfers={len(self.transfers)})"
        )


@dataclass(frozen=True)
class ShardingDescriptor:
    """Describes how a logical tensor is distributed across a device mesh.

    This is the bridge between framework-specific metadata (Megatron's
    partition_dim, vLLM's column/row parallel, FSDP's DTensor placements)
    and tensordict's framework-agnostic transfer plan computation.
    """

    mesh_shape: tuple[int, ...]
    placements: tuple
    logical_shape: torch.Size
    rank_map: dict[tuple[int, ...], int] | None = None

    @classmethod
    def from_dtensor(cls, dtensor) -> ShardingDescriptor:
        """Construct from a ``torch.distributed.tensor.DTensor``."""
        mesh = dtensor.device_mesh
        return cls(
            mesh_shape=tuple(mesh.mesh.shape),
            placements=tuple(dtensor.placements),
            logical_shape=dtensor.shape,
            rank_map=_mesh_to_rank_map(mesh),
        )

    @classmethod
    def from_device_mesh(
        cls,
        mesh,
        placements: Sequence,
        logical_shape: torch.Size,
    ) -> ShardingDescriptor:
        """Construct from a DeviceMesh + placements + shape."""
        return cls(
            mesh_shape=tuple(mesh.mesh.shape),
            placements=tuple(placements),
            logical_shape=logical_shape,
            rank_map=_mesh_to_rank_map(mesh),
        )

    @classmethod
    def replicated(cls, shape: torch.Size, world_size: int) -> ShardingDescriptor:
        """All ranks hold a full copy."""
        from torch.distributed.tensor.placement_types import Replicate

        return cls(
            mesh_shape=(world_size,),
            placements=(Replicate(),),
            logical_shape=shape,
        )

    @classmethod
    def sharded(
        cls,
        shape: torch.Size,
        dim: int,
        world_size: int,
        rank_map: dict | None = None,
    ) -> ShardingDescriptor:
        """Simple 1D shard on a single dimension."""
        from torch.distributed.tensor.placement_types import Shard

        return cls(
            mesh_shape=(world_size,),
            placements=(Shard(dim),),
            logical_shape=shape,
            rank_map=rank_map,
        )


# ---------------------------------------------------------------------------
# Slice arithmetic
# ---------------------------------------------------------------------------


def _chunk_slice(total_size: int, num_chunks: int, chunk_idx: int) -> slice:
    """Return the slice for ``chunk_idx`` when splitting into chunks.

    Uses ``torch.chunk`` semantics (last chunk may be smaller).
    """
    chunk_size = (total_size + num_chunks - 1) // num_chunks
    start = chunk_idx * chunk_size
    stop = min(start + chunk_size, total_size)
    if start >= total_size:
        return slice(0, 0)
    return slice(start, stop)


def _intersect_1d(a: slice, b: slice) -> slice | None:
    """Intersect two *simple* slices (step=1, non-negative start/stop)."""
    lo = max(a.start, b.start)
    hi = min(a.stop, b.stop)
    if lo >= hi:
        return None
    return slice(lo, hi)


def _intersect_slices(
    a: tuple[slice, ...], b: tuple[slice, ...]
) -> tuple[slice, ...] | None:
    """Intersect two tuples of per-dimension slices. Returns ``None`` if empty."""
    result = []
    for sa, sb in zip(a, b):
        inter = _intersect_1d(sa, sb)
        if inter is None:
            return None
        result.append(inter)
    return tuple(result)


def _slice_relative_to(inner: slice, outer: slice) -> slice:
    """Express *inner* (global coords) relative to *outer* (also global)."""
    return slice(inner.start - outer.start, inner.stop - outer.start)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def _compute_local_slices(
    global_shape: Sequence[int],
    mesh_shape: Sequence[int],
    placements: Sequence,
    rank_coords: Sequence[int],
) -> tuple[slice, ...]:
    """Compute the global-coordinate slices held by *rank_coords*.

    Args:
        global_shape: shape of the full logical tensor.
        mesh_shape: shape of the device mesh (one int per mesh dim).
        placements: one Placement per mesh dim.
        rank_coords: coordinate of the rank in the mesh (one int per mesh dim).

    Returns:
        A tuple of slices (one per tensor dimension) in global coordinates.
    """
    ndim = len(global_shape)
    slices = [slice(0, s) for s in global_shape]

    for mesh_dim_idx, placement in enumerate(placements):
        if _placement_is_shard(placement):
            shard_dim = _placement_shard_dim(placement)
            if shard_dim < 0:
                shard_dim += ndim
            num_chunks = mesh_shape[mesh_dim_idx]
            chunk_idx = rank_coords[mesh_dim_idx]
            current_dim_size = slices[shard_dim].stop - slices[shard_dim].start
            chunk_sl = _chunk_slice(current_dim_size, num_chunks, chunk_idx)
            base = slices[shard_dim].start
            slices[shard_dim] = slice(base + chunk_sl.start, base + chunk_sl.stop)
        # Replicate / Partial: slices stay as full range

    return tuple(slices)


def _compute_all_local_slices(
    global_shape: Sequence[int],
    mesh_shape: Sequence[int],
    placements: Sequence,
    rank_map: dict[tuple[int, ...], int] | None = None,
) -> list[_ShardSpec]:
    """Compute :class:`_ShardSpec` for every rank in the mesh.

    Args:
        global_shape: shape of the full logical tensor.
        mesh_shape: shape of the device mesh.
        placements: one Placement per mesh dim.
        rank_map: optional mapping from mesh coordinates to global rank ids.
            If *None*, ranks are numbered in row-major order.
    """
    specs: list[_ShardSpec] = []
    ranges = [range(s) for s in mesh_shape]
    for flat_idx, coords in enumerate(itertools.product(*ranges)):
        rank = rank_map[coords] if rank_map is not None else flat_idx
        slices = _compute_local_slices(global_shape, mesh_shape, placements, coords)
        specs.append(_ShardSpec(rank=rank, slices=slices))
    return specs


def _deduplicate_src_specs(
    src_specs: list[_ShardSpec],
    placements: Sequence,
    mesh_shape: Sequence[int],
) -> list[_ShardSpec]:
    """When src has Replicate or Partial dims, multiple ranks hold the same data.

    Keep only one representative per unique slice (the one with the lowest rank)
    to avoid redundant transfers.
    """
    has_replica = any(
        _placement_is_replicate(p) or _placement_is_partial(p) for p in placements
    )
    if not has_replica:
        return src_specs

    seen: dict[tuple[tuple[int, int], ...], _ShardSpec] = {}
    for spec in src_specs:
        key = tuple((s.start, s.stop) for s in spec.slices)
        if key not in seen or spec.rank < seen[key].rank:
            seen[key] = spec
    return list(seen.values())


def _compute_transfer_plan(
    global_shape: Sequence[int],
    src_mesh_shape: Sequence[int],
    src_placements: Sequence,
    dst_mesh_shape: Sequence[int],
    dst_placements: Sequence,
    src_rank_map: dict[tuple[int, ...], int] | None = None,
    dst_rank_map: dict[tuple[int, ...], int] | None = None,
) -> _TransferPlan:
    """Compute the optimal P2P transfer plan for one tensor.

    This is pure computation -- no GPU, no distributed runtime needed.

    Args:
        global_shape: shape of the full logical tensor.
        src_mesh_shape: shape of the source device mesh.
        src_placements: placement per source mesh dim (Shard/Replicate/Partial).
        dst_mesh_shape: shape of the destination device mesh.
        dst_placements: placement per destination mesh dim.
        src_rank_map: mesh-coords -> global rank for src. Row-major if None.
        dst_rank_map: mesh-coords -> global rank for dst. Row-major if None.

    Returns:
        A :class:`_TransferPlan` describing the minimal set of P2P transfers.
    """
    global_shape = torch.Size(global_shape)

    src_specs = _compute_all_local_slices(
        global_shape, src_mesh_shape, src_placements, src_rank_map
    )
    dst_specs = _compute_all_local_slices(
        global_shape, dst_mesh_shape, dst_placements, dst_rank_map
    )

    # Deduplicate replicated src specs
    src_specs_dedup = _deduplicate_src_specs(src_specs, src_placements, src_mesh_shape)

    plan = _TransferPlan(global_shape=global_shape)

    for dst_spec in dst_specs:
        for src_spec in src_specs_dedup:
            overlap = _intersect_slices(src_spec.slices, dst_spec.slices)
            if overlap is None:
                continue
            # Skip empty slices
            if any(s.start >= s.stop for s in overlap):
                continue

            src_local = tuple(
                _slice_relative_to(o, s) for o, s in zip(overlap, src_spec.slices)
            )
            dst_local = tuple(
                _slice_relative_to(o, d) for o, d in zip(overlap, dst_spec.slices)
            )

            plan.transfers.append(
                _ChunkTransfer(
                    src_rank=src_spec.rank,
                    dst_rank=dst_spec.rank,
                    src_slices=src_local,
                    dst_slices=dst_local,
                    global_slices=overlap,
                )
            )

    return plan


def execute_transfer_plan(
    plan: _TransferPlan,
    src_tensor: Tensor | None,
    dst_buffer: Tensor | None,
    rank: int,
    backend: _TransportBackend,
) -> None:
    """Execute a single-tensor transfer plan on this rank.

    This rank participates as sender, receiver, or both, depending on
    whether it appears in the plan's transfers.

    Args:
        plan: the precomputed TransferPlan.
        src_tensor: this rank's local shard (if rank is a source).
            Can be ``None`` if this rank is destination-only.
        dst_buffer: pre-allocated buffer for received data (if rank is
            a destination). Can be ``None`` if this rank is source-only.
        rank: this rank's global rank ID.
        backend: transport backend to use.
    """
    sends = plan.sends_for_rank(rank)
    recvs = plan.recvs_for_rank(rank)

    recv_bufs: list[tuple[Tensor, _ChunkTransfer]] = []
    if dst_buffer is not None:
        for transfer in recvs:
            chunk_shape = tuple(s.stop - s.start for s in transfer.global_slices)
            buf = torch.empty(
                chunk_shape, dtype=dst_buffer.dtype, device=dst_buffer.device
            )
            backend.recv_tensor(buf, transfer.src_rank)
            recv_bufs.append((buf, transfer))

    if src_tensor is not None:
        for transfer in sends:
            chunk = src_tensor[transfer.src_slices].contiguous()
            backend.send_tensor(chunk, transfer.dst_rank)

    for buf, transfer in recv_bufs:
        dst_buffer[transfer.dst_slices].copy_(buf)


# ---------------------------------------------------------------------------
# Transport abstraction
# ---------------------------------------------------------------------------


@runtime_checkable
class _TransportBackend(Protocol):
    """Protocol for transport backends."""

    def send_tensor(self, tensor: Tensor, dst: int, *, tag: int = 0) -> None: ...
    def recv_tensor(self, tensor: Tensor, src: int, *, tag: int = 0) -> None: ...
    def send_object(self, obj: Any, dst: int) -> None: ...
    def recv_object(self, src: int) -> Any: ...


class _TorchDistributedBackend:
    """Transport backend using ``torch.distributed`` P2P primitives.

    Metadata is serialized to JSON and sent as a CUDA byte tensor so that
    only the NCCL backend is required (no Gloo needed for CPU tensors).
    All operations use tag=0 and rely on FIFO ordering of NCCL P2P ops
    between each (src, dst) pair.
    """

    def __init__(self, group=None):
        self.group = group

    def send_tensor(self, tensor: Tensor, dst: int, *, tag: int = 0) -> None:
        from torch import distributed as dist

        dist.send(tensor.contiguous(), dst=dst, group=self.group)

    def recv_tensor(self, tensor: Tensor, src: int, *, tag: int = 0) -> None:
        from torch import distributed as dist

        dist.recv(tensor, src=src, group=self.group)

    def send_object(self, obj: Any, dst: int) -> None:
        from torch import distributed as dist

        data = json.dumps(obj).encode("utf-8")
        length_t = torch.tensor([len(data)], dtype=torch.int64, device="cuda")
        dist.send(length_t, dst=dst, group=self.group)
        data_t = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
        dist.send(data_t, dst=dst, group=self.group)

    def recv_object(self, src: int) -> Any:
        from torch import distributed as dist

        length_t = torch.empty(1, dtype=torch.int64, device="cuda")
        dist.recv(length_t, src=src, group=self.group)
        length = int(length_t.item())
        data_t = torch.empty(length, dtype=torch.uint8, device="cuda")
        dist.recv(data_t, src=src, group=self.group)
        return json.loads(bytes(data_t.cpu().tolist()))


class _UCXXBackend:
    """Transport backend using UCXX endpoints.

    This backend operates on a single pre-established UCXX endpoint
    (i.e. a single peer connection). The ``dst`` / ``src`` / ``tag``
    parameters on :meth:`send_tensor` and :meth:`recv_tensor` are
    accepted for protocol compatibility but are **not used** — all
    data is routed through ``self._endpoint``.  Multi-peer transfers
    (e.g. Strategy C with many dst ranks) require one ``_UCXXBackend``
    instance per peer.

    .. warning::
        Each public method calls :func:`asyncio.run`, which creates a
        new event loop. This will raise ``RuntimeError`` if called from
        within an already-running event loop (e.g. inside vLLM's async
        engine).
    """

    def __init__(self, endpoint):
        self._endpoint = endpoint

    def _tensor_to_numpy(self, t: Tensor):
        import numpy as np

        return np.frombuffer(t.contiguous().view(torch.uint8).numpy(), dtype=np.uint8)

    def send_tensor(self, tensor: Tensor, dst: int, *, tag: int = 0) -> None:
        import asyncio

        asyncio.run(self._asend_tensor(tensor))

    def recv_tensor(self, tensor: Tensor, src: int, *, tag: int = 0) -> None:
        import asyncio

        asyncio.run(self._arecv_tensor(tensor))

    async def _asend_tensor(self, tensor: Tensor) -> None:
        t = tensor.contiguous()
        if t.is_cuda:
            await self._endpoint.send(t)
        else:
            await self._endpoint.send(self._tensor_to_numpy(t))

    async def _arecv_tensor(self, tensor: Tensor) -> None:
        if tensor.is_cuda:
            await self._endpoint.recv(tensor)
        else:
            buf = self._tensor_to_numpy(tensor)
            await self._endpoint.recv(buf)

    def send_object(self, obj: Any, dst: int) -> None:
        import asyncio

        asyncio.run(self._asend_object(obj))

    def recv_object(self, src: int) -> Any:
        import asyncio

        return asyncio.run(self._arecv_object())

    async def _asend_object(self, obj: Any) -> None:
        import numpy as np

        data = json.dumps(obj).encode("utf-8")
        length = struct.pack("<Q", len(data))
        await self._endpoint.send(np.frombuffer(length, dtype=np.uint8))
        await self._endpoint.send(np.frombuffer(data, dtype=np.uint8).copy())

    async def _arecv_object(self) -> Any:
        import numpy as np

        len_buf = np.empty(8, dtype=np.uint8)
        await self._endpoint.recv(len_buf)
        length = struct.unpack("<Q", len_buf.tobytes())[0]
        data_buf = np.empty(length, dtype=np.uint8)
        await self._endpoint.recv(data_buf)
        return json.loads(bytes(data_buf))


def _get_transport_backend(
    transport: str,
    dst_or_src,
    group=None,
) -> _TransportBackend:
    """Resolve transport string to a backend instance.

    Args:
        transport: ``"torch_distributed"``, ``"ucxx"``, or ``"auto"``.
        dst_or_src: the destination/source argument passed by the user
            (int rank or TensorDictPipe). Used for auto-detection.
        group: process group for torch.distributed backend.
    """
    if transport == "auto":
        if _has_ucxx:
            from tensordict._ucxx import TensorDictPipe

            if isinstance(dst_or_src, TensorDictPipe):
                transport = "ucxx"
            else:
                transport = "torch_distributed"
        else:
            transport = "torch_distributed"

    if transport == "torch_distributed":
        return _TorchDistributedBackend(group=group)
    elif transport == "ucxx":
        if not isinstance(dst_or_src, _UCXXBackend):
            from tensordict._ucxx import TensorDictPipe

            if isinstance(dst_or_src, TensorDictPipe):
                return _UCXXBackend(endpoint=dst_or_src._endpoint)
        return _UCXXBackend(endpoint=dst_or_src)
    else:
        raise ValueError(
            f"Unknown transport {transport!r}. "
            "Expected 'torch_distributed', 'ucxx', or 'auto'."
        )


# ---------------------------------------------------------------------------
# DeviceMesh helpers
# ---------------------------------------------------------------------------


def _mesh_to_rank_map(mesh) -> dict[tuple[int, ...], int]:
    """Convert a DeviceMesh to a {coords: global_rank} dict."""
    mesh_tensor = mesh.mesh
    result = {}
    for idx in itertools.product(*(range(s) for s in mesh_tensor.shape)):
        result[idx] = int(mesh_tensor[idx].item())
    return result


def _mesh_all_ranks(mesh) -> list[int]:
    """Return all global ranks in a DeviceMesh (flat, sorted)."""
    return sorted(mesh.mesh.flatten().tolist())
