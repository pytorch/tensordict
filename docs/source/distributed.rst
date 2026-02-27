.. _distributed:

TensorDict in distributed settings
===================================

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
-----------------------------------------

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
------------------------------------

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
--------------------------------------------

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
------------------------------------------------------------------

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

Shared-storage workflows (memory-mapped)
-----------------------------------------

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
