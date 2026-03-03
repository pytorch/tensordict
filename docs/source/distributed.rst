.. _distributed:

TensorDict in distributed settings
==================================

TensorDict integrates with :mod:`torch.distributed` to send, receive,
broadcast, reduce, gather and scatter collections of tensors with a single
method call.  It also supports shared-storage workflows where nodes access
the same memory-mapped files on a shared filesystem.

.. contents:: On this page
   :local:
   :depth: 2

Prerequisites
-------------

All collective and point-to-point APIs below require a
:func:`torch.distributed.init_process_group` to be set up.  Refer to the
`PyTorch distributed documentation <https://pytorch.org/docs/stable/distributed.html>`_
for details.

.. code-block:: python

   import torch.distributed as dist
   dist.init_process_group("nccl", rank=rank, world_size=world_size)

``consolidate()`` and fast serialization
----------------------------------------

Several distributed primitives (*broadcast*, *all_gather*, *scatter*,
*init_remote / from_remote_init*, and *consolidated send / recv*) rely on
:meth:`~tensordict.TensorDict.consolidate`, which packs every leaf tensor
into a single contiguous storage buffer.  This turns an arbitrarily nested
TensorDict into **one** message on the wire:

.. code-block:: python

   td = TensorDict(a=torch.randn(3), b={"c": torch.randn(3)}, batch_size=[3])
   td_c = td.consolidate(metadata=True)
   # td_c._consolidated["storage"]  – a single flat uint8 tensor
   # td_c._consolidated["metadata"] – a dict describing how to reconstruct the tree

Pass ``metadata=True`` so that the receiving side can reconstruct the full
tree structure.

Point-to-point: ``send`` / ``recv``
-----------------------------------

The simplest pattern sends a TensorDict from one rank to another.
The receiver must already hold a TensorDict with matching structure so that
incoming tensors can be written in-place:

.. code-block:: python

   # rank 0
   td = TensorDict(a=torch.randn(4), b=torch.randn(4, 3), batch_size=[4])
   td.send(dst=1)

   # rank 1
   td = TensorDict(a=torch.zeros(4), b=torch.zeros(4, 3), batch_size=[4])
   td.recv(src=0)
   assert (td != 0).all()

**Consolidated mode** sends the entire TensorDict as a single tensor message
instead of one message per leaf.  This is more efficient when
the TensorDict has many small tensors:

.. code-block:: python

   # rank 0  – sender
   td.send(dst=1, consolidated=True)

   # rank 1  – receiver (must already be consolidated)
   td.recv(src=0, consolidated=True)

See :meth:`~tensordict.TensorDictBase.send` and
:meth:`~tensordict.TensorDictBase.recv` for the full API.

Async point-to-point: ``isend`` / ``irecv``
-------------------------------------------

Non-blocking variants let you overlap communication with computation:

.. code-block:: python

   # rank 0
   td.isend(dst=1)

   # rank 1
   futures = td.irecv(src=0, return_premature=True)
   # ... do other work ...
   for f in futures:
       f.wait()

See :meth:`~tensordict.TensorDictBase.isend` and
:meth:`~tensordict.TensorDictBase.irecv` for the full API.

Cold-start initialization: ``init_remote`` / ``from_remote_init``
-----------------------------------------------------------------

When the receiving rank does **not** know the structure of the TensorDict in
advance, ``init_remote`` sends both the metadata *and* the content so that
the receiver can reconstruct the full object from scratch:

.. code-block:: python

   # rank 0  – sender
   td = TensorDict(
       {"obs": torch.randn(2, 84), "reward": torch.randn(2, 1)},
       batch_size=[2],
   )
   td.init_remote(dst=1)

   # rank 1  – receiver
   td = TensorDict.from_remote_init(src=0)
   print(td)  # fully reconstructed, including nested structure

This is handy during setup / warm-up: the first call uses ``init_remote``
/ ``from_remote_init`` to establish the schema, and subsequent transfers
use the faster ``send(consolidated=True)`` / ``recv(consolidated=True)``
path.

Two transport modes are available:

- **Point-to-point** (default): only sender and receiver participate.
- **Broadcast** (``use_broadcast=True``): delegates to
  :meth:`~tensordict.TensorDictBase.broadcast`.  All ranks in the group
  must participate.

.. code-block:: python

   # all ranks
   if rank == 0:
       td.init_remote(use_broadcast=True)
   else:
       td = TensorDict.from_remote_init(src=0, use_broadcast=True)

``@tensorclass`` objects are preserved across the wire: the receiving side
will automatically reconstruct the original tensorclass type.

See :meth:`~tensordict.TensorDictBase.init_remote` and
:meth:`~tensordict.TensorDictBase.from_remote_init` for the full API.

Broadcast
---------

:meth:`~tensordict.TensorDictBase.broadcast` sends a consolidated
TensorDict from *one* rank to *all* ranks:

.. code-block:: python

   if rank == src:
       td = TensorDict(a=torch.randn(3), batch_size=[3])
   else:
       td = TensorDict()

   td = td.broadcast(src=src)
   # all ranks now hold the same td

Internally this broadcasts the metadata via
``dist.broadcast_object_list`` followed by the storage via
``dist.broadcast``.

See :meth:`~tensordict.TensorDictBase.broadcast`.

All-reduce
----------

:meth:`~tensordict.TensorDictBase.all_reduce` reduces every leaf tensor
across all ranks in-place:

.. code-block:: python

   import torch.distributed as dist

   td = TensorDict(a=torch.ones(3) * rank, batch_size=[3])
   td.all_reduce(op=dist.ReduceOp.SUM)
   # td["a"] is now the sum across all ranks

Supports ``async_op=True`` to return a list of futures.

See :meth:`~tensordict.TensorDictBase.all_reduce`.

Reduce
------

:meth:`~tensordict.TensorDictBase.reduce` is like ``all_reduce`` but only
the destination rank receives the result:

.. code-block:: python

   td.reduce(dst=0, op=dist.ReduceOp.SUM)

See :meth:`~tensordict.TensorDictBase.reduce`.

All-gather
----------

:meth:`~tensordict.TensorDictBase.all_gather` gathers TensorDict instances
from every rank and returns a list (one per rank):

.. code-block:: python

   local_td = TensorDict(a=torch.randn(3) + rank, batch_size=[3])
   all_tds = local_td.all_gather()
   # len(all_tds) == world_size

Uses consolidated transport: each rank consolidates its TensorDict,
metadata is gathered with ``dist.all_gather_object``, and storage buffers
are gathered with ``dist.all_gather``.

See :meth:`~tensordict.TensorDictBase.all_gather`.

Scatter
-------

:meth:`~tensordict.TensorDictBase.scatter` distributes *different*
TensorDict instances from one rank to all other ranks:

.. code-block:: python

   if rank == src:
       tds = [TensorDict(a=torch.randn(3) + i, batch_size=[3]) for i in range(world_size)]
   else:
       tds = None

   td = TensorDict().scatter(src=src, tensordicts=tds)
   # each rank receives its own td

See :meth:`~tensordict.TensorDictBase.scatter`.

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Method
     - Transport
     - Participants
     - Use case
   * - :meth:`~.send` / :meth:`~.recv`
     - point-to-point
     - 2 ranks
     - Steady-state data transfer (supports ``consolidated=True``)
   * - :meth:`~.isend` / :meth:`~.irecv`
     - point-to-point (async)
     - 2 ranks
     - Overlapping communication and computation
   * - :meth:`~.init_remote` / :meth:`~.from_remote_init`
     - point-to-point or broadcast
     - 2 ranks (or all)
     - Cold-start: receiver doesn't know the schema
   * - :meth:`~.broadcast`
     - collective
     - all ranks
     - One-to-all (same data)
   * - :meth:`~.scatter`
     - collective
     - all ranks
     - One-to-all (different data per rank)
   * - :meth:`~.all_gather`
     - collective
     - all ranks
     - All-to-all gather
   * - :meth:`~.all_reduce`
     - collective
     - all ranks
     - In-place reduce (e.g. gradient averaging)
   * - :meth:`~.reduce`
     - collective
     - all ranks
     - Reduce to single rank

High-performance streaming with UCXX
------------------------------------

The ``torch.distributed`` primitives above operate within a process group
and are ideal for collective communication patterns.  For **persistent
point-to-point streaming** — such as a data-producer on one node
continuously feeding batches to a trainer on another —
:class:`~tensordict._ucxx.TensorDictPipe` provides a higher-performance
alternative built on `UCXX <https://github.com/rapidsai/ucxx>`_, the
Python bindings for `UCX <https://openucx.org/>`_.

Key advantages over ``torch.distributed`` send/recv:

- **No process group required** — connect directly by IP and port.
- **Zero-allocation steady-state** — after the first send establishes
  the schema, subsequent sends transmit only the raw storage buffer
  with no metadata, no allocation, and no parsing on the receiver.
- **InfiniBand / RDMA** — UCX auto-selects the fastest available
  transport (TCP, InfiniBand verbs, RoCE).
- **GPUDirect RDMA** — when tensors live on CUDA, the consolidated
  buffer is sent directly from GPU memory over RDMA, matching NCCL
  throughput without CPU staging.
- **Async-native** — built on :mod:`asyncio`, with synchronous wrappers
  for convenience.

Installing UCXX
~~~~~~~~~~~~~~~

UCXX requires a Linux host with UCX system libraries.  Install via pip
or conda:

.. code-block:: bash

   # pip (CUDA 12)
   pip install ucxx-cu12

   # pip (CUDA 13)
   pip install ucxx-cu13

   # conda
   conda install -c conda-forge -c rapidsai ucxx

For InfiniBand support the host needs ``rdma-core`` (or MOFED 5.0+) and
a Linux kernel ≥ 5.6.  For GPUDirect RDMA, the ``nvidia_peermem`` kernel
module must be loaded.  Verify with:

.. code-block:: bash

   # check IB devices
   ibstat

   # check GPUDirect RDMA
   lsmod | grep nvidia_peermem

UCX picks up transports automatically.  You can control transport
selection with environment variables:

.. code-block:: bash

   # let UCX auto-select (recommended)
   export UCX_TLS=all
   export UCX_NET_DEVICES=all

   # restrict to TCP only (useful for debugging)
   export UCX_TLS=tcp

``TensorDictPipe``: the two-phase protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~tensordict._ucxx.TensorDictPipe` wraps a UCXX endpoint into a
typed channel that understands TensorDict's consolidated layout.  It
implements a **two-phase protocol**:

1. **Handshake (first send)** — the sender consolidates the TensorDict,
   transmits the consolidation metadata (keys, dtypes, shapes, byte
   offsets — serialized as JSON) followed by the raw storage buffer.
   The receiver allocates a single flat tensor and rebuilds a TensorDict
   whose leaves are views into that buffer.

2. **Steady-state (subsequent sends with same schema)** — the sender
   sends a one-byte "same schema" flag followed by just the raw buffer.
   The receiver overwrites its pre-allocated storage in-place.  No
   allocation, no metadata parsing, no deserialization.

When the schema changes (e.g. different keys or shapes), the pipe
automatically falls back to the handshake phase.

Establishing a connection
~~~~~~~~~~~~~~~~~~~~~~~~~

One side listens, the other connects:

.. code-block:: python

   import asyncio
   from tensordict._ucxx import TensorDictPipe

   # --- node A (receiver) ---
   async def receiver():
       pipe = await TensorDictPipe.listen(port=13337)

       # first receive: handshake — allocates buffer, builds TensorDict
       td = await pipe.arecv()
       print(td)  # TensorDict with the sender's keys/shapes/dtypes

       # steady-state: zero-alloc receive into the same buffer
       for _ in range(100):
           td = await pipe.arecv(td)
           train(td)

       await pipe.aclose()

   # --- node B (sender) ---
   async def sender():
       pipe = await TensorDictPipe.connect("10.0.1.1", port=13337)

       for batch in dataloader:
           td = TensorDict({"obs": batch[0], "label": batch[1]}, batch_size=[B])
           await pipe.asend(td)

       await pipe.aclose()

The pipe also supports synchronous usage and context managers:

.. code-block:: python

   # synchronous
   pipe.send(td)
   td = pipe.recv()

   # context manager
   async with await TensorDictPipe.listen(13337) as pipe:
       td = await pipe.arecv()

Async iteration
~~~~~~~~~~~~~~~

``TensorDictPipe`` implements the async iterator protocol, so you can
consume incoming TensorDicts with ``async for``:

.. code-block:: python

   pipe = await TensorDictPipe.listen(13337)
   async for td in pipe:
       train(td)

The iterator yields the received TensorDict on each iteration,
automatically reusing the buffer after the first handshake.

CPU and CUDA transfers
~~~~~~~~~~~~~~~~~~~~~~

The pipe is **device-aware**.  When the sender's TensorDict contains
CUDA tensors, the consolidated buffer lives on GPU and is transferred
directly over RDMA (GPUDirect) without touching host memory.  The
receiver automatically allocates its buffer on the same device:

.. code-block:: python

   # sender — data on GPU
   td = TensorDict({"x": torch.randn(1024, 1024, device="cuda")}, batch_size=[1024])
   await pipe.asend(td)  # sends from GPU memory

   # receiver — automatically receives into GPU memory
   td = await pipe.arecv()
   print(td.device)  # cuda:0

You can override the receiver's device by passing ``device=``:

.. code-block:: python

   td = await pipe.arecv(device="cpu")  # force receive onto CPU

Unified API with ``send`` / ``recv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~tensordict.TensorDictBase.send` and
:meth:`~tensordict.TensorDictBase.recv` methods accept a
``TensorDictPipe`` in place of a rank integer:

.. code-block:: python

   # works with torch.distributed
   td.send(dst=1)
   td.recv(src=0)

   # works with UCXX — same API, different transport
   td.send(dst=pipe)
   td.recv(src=pipe)

   # async variants
   await td.asend(pipe)
   await td.arecv(pipe)

This makes it easy to switch between ``torch.distributed`` and UCXX
without changing application logic.

One-to-many: ``TensorDictServer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~tensordict._ucxx.TensorDictServer` accepts multiple incoming
connections, yielding a separate ``TensorDictPipe`` for each client:

.. code-block:: python

   from tensordict._ucxx import TensorDictServer

   async def handle_client(pipe):
       async for td in pipe:
           result = model(td)
           await pipe.asend(result)

   async with TensorDictServer(port=13337) as server:
       await server.serve(handle_client)

Or using the async iterator interface:

.. code-block:: python

   async with TensorDictServer(port=13337) as server:
       async for pipe in server:
           asyncio.ensure_future(handle_client(pipe))

Performance notes
~~~~~~~~~~~~~~~~~

On a two-node cluster with H200 GPUs connected via 400 Gb/s InfiniBand,
typical steady-state numbers are:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20

   * - Size
     - NCCL ``send``/``recv``
     - UCXX ``TensorDictPipe``
     - UCXX speedup
   * - 1 KB
     - 231 µs
     - 138 µs
     - 1.7×
   * - 1 MB
     - 238 µs (8.8 GB/s)
     - 231 µs (9.1 GB/s)
     - ~1×
   * - 100 MB
     - 7.5 ms (28.0 GB/s)
     - 7.5 ms (28.1 GB/s)
     - ~1×

UCXX matches NCCL throughput for large messages and is faster for small
ones (lower per-message overhead).  On CPU, UCXX over InfiniBand
achieves ~5.8 GB/s for 100 MB — roughly 2.3× faster than ``gloo``.

Shared-storage workflows (memory-mapped)
----------------------------------------

When nodes share a filesystem (e.g. NFS), a memory-mapped TensorDict lets
multiple processes read from and write to the same data without sending
tensors over the network.

Create a memory-mapped TensorDict with
:meth:`~tensordict.TensorDictBase.memmap_` or
:meth:`~tensordict.TensorDictBase.memmap`:

.. code-block:: python

   from tensordict import TensorDict, MemoryMappedTensor

   td = TensorDict(
       {
           "images": torch.zeros(50000, 3, 64, 64, dtype=torch.uint8),
           "labels": torch.zeros(50000, dtype=torch.long),
       },
       batch_size=[50000],
   )
   td.memmap_("/shared/nfs/dataset")

Any process that can read ``/shared/nfs/dataset`` can load this:

.. code-block:: python

   td = TensorDict.load_memmap("/shared/nfs/dataset")
   batch = td[:64].clone()  # only reads the first 64 entries

For more details on memory-mapped serialization, see the
:ref:`saving documentation <saving>`.
