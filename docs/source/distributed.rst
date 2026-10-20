.. _distributed:

Distributed tensordicts
=======================

A tensordict sends, receives, and reduces all its tensors with one call:
``td.send(dst=1)`` sends every leaf to rank 1, and ``td.all_reduce()``
reduces every leaf over all ranks of :mod:`torch.distributed`. To stream
tensordicts between two nodes without a process group, use a UCXX pipe. To
share a tensordict between processes on one node, see :ref:`data-loading`.

Choose a method
---------------

These are methods of :class:`~tensordict.TensorDictBase`, so they also work
on tensorclasses.

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Method
     - Ranks that call it
     - Use it to
   * - :meth:`~tensordict.TensorDictBase.send`,
       :meth:`~tensordict.TensorDictBase.recv`
     - sender and receiver
     - Send a tensordict to a rank that has a tensordict of the same
       structure. ``consolidated=True`` sends one message.
   * - :meth:`~tensordict.TensorDictBase.isend`,
       :meth:`~tensordict.TensorDictBase.irecv`
     - sender and receiver
     - Send as ``send`` does, and do other work while the data moves.
   * - :meth:`~tensordict.TensorDictBase.init_remote`,
       :meth:`~tensordict.TensorDictBase.from_remote_init`
     - sender and receiver (or all, with ``use_broadcast=True``)
     - Send a tensordict to a rank that does not know its structure.
   * - :meth:`~tensordict.TensorDictBase.broadcast`
     - all
     - Send one tensordict from one rank to all ranks.
   * - :meth:`~tensordict.TensorDictBase.scatter`
     - all
     - Send a different tensordict from one rank to each rank.
   * - :meth:`~tensordict.TensorDictBase.all_gather`
     - all
     - Get the tensordicts of all ranks on every rank.
   * - :meth:`~tensordict.TensorDictBase.gather_and_stack`
     - all
     - Stack the tensordicts of the other ranks on one rank.
   * - :meth:`~tensordict.TensorDictBase.all_reduce`
     - all
     - Reduce each leaf over all ranks, in place on every rank (for
       example, to average gradients).
   * - :meth:`~tensordict.TensorDictBase.reduce`
     - all
     - Reduce each leaf over all ranks, on one rank.

Run the examples
----------------

Every process must join a process group before it calls these methods (see
:func:`torch.distributed.init_process_group`). The script below starts two
processes on one machine, with the ``gloo`` backend on CPU, and runs one
example in each. Each example on this page is a function of ``rank`` and
``world_size``. To run an example, add it to the script, and pass its name to
``run`` in the last line:

.. code-block:: python

   import os
   import shutil
   import socket
   import tempfile

   import torch
   import torch.distributed as dist
   import torch.multiprocessing as mp

   from tensordict import TensorDict


   def _worker(rank, world_size, port, example):
       os.environ["MASTER_ADDR"] = "127.0.0.1"
       os.environ["MASTER_PORT"] = str(port)
       dist.init_process_group("gloo", rank=rank, world_size=world_size)
       try:
           example(rank, world_size)
       finally:
           dist.destroy_process_group()


   def run(example, world_size=2):
       with socket.socket() as s:  # find a free port
           s.bind(("127.0.0.1", 0))
           port = s.getsockname()[1]
       mp.spawn(_worker, args=(world_size, port, example), nprocs=world_size)


   def send_recv(rank, world_size):
       if rank == 0:
           td = TensorDict({"a": torch.randn(4), "b": torch.randn(4, 3)}, batch_size=[4])
           td.send(dst=1)
       elif rank == 1:
           td = TensorDict({"a": torch.zeros(4), "b": torch.zeros(4, 3)}, batch_size=[4])
           td.recv(src=0)
           print((td["b"] != 0).all())  # tensor(True)


   if __name__ == "__main__":
       run(send_recv)

On several machines, or with GPUs, start one process for each GPU with
``torchrun``. Call ``dist.init_process_group("nccl")`` and
``torch.cuda.set_device(rank)``, and put the tensordicts on the GPU. The
calls on this page do not change.

Send to one rank
----------------

send and recv
~~~~~~~~~~~~~

The ``send_recv`` example above sends each leaf as one message. The receiver
must hold a tensordict with the same keys, shapes, and dtypes. ``recv`` writes
the received values into its tensors.

With ``consolidated=True``, ``send`` sends all the leaves as one message
instead. It first calls :meth:`~tensordict.TensorDictBase.consolidate`, which
copies the leaves into one storage (see :ref:`saving`), unless the tensordict
is consolidated already. This is faster when the tensordict has many small
leaves. In one measurement with 100 leaves of 256 bytes, on one machine with
``gloo``, a consolidated send was 34 times as fast as a send per leaf. The receiver must
hold a consolidated tensordict with the same keys, shapes, and dtypes:

.. code-block:: python

   def send_consolidated(rank, world_size):
       if rank == 0:
           td = TensorDict({"a": torch.randn(4), "b": torch.randn(4, 3)}, batch_size=[4])
           td.send(dst=1, consolidated=True)
       elif rank == 1:
           td = TensorDict({"a": torch.zeros(4), "b": torch.zeros(4, 3)}, batch_size=[4])
           td = td.consolidate(metadata=True)
           td.recv(src=0, consolidated=True)
           print((td["b"] != 0).all())  # tensor(True)

If the receiver is not consolidated, ``recv(consolidated=True)`` raises
``AttributeError: 'TensorDict' object has no attribute '_consolidated'``, and
the sender waits until its timeout.

isend and irecv
~~~~~~~~~~~~~~~

:meth:`~tensordict.TensorDictBase.isend` and
:meth:`~tensordict.TensorDictBase.irecv` start one transfer for each leaf. By
default, both wait until all the transfers are done, as ``send`` and
``recv`` do. To do other work while the data moves, pass
``return_early=True`` to ``isend``, or ``return_premature=True`` to
``irecv``. The call then returns a list of ``Work`` objects. Call ``wait()``
on each one before you use or change the data:

.. code-block:: python

   def isend_irecv(rank, world_size):
       if rank == 0:
           td = TensorDict({"a": torch.randn(4), "b": torch.randn(4, 3)}, batch_size=[4])
           works = td.isend(dst=1, return_early=True)
           # ... do other work, without changing td ...
           for work in works:
               work.wait()
       elif rank == 1:
           td = TensorDict({"a": torch.zeros(4), "b": torch.zeros(4, 3)}, batch_size=[4])
           works = td.irecv(src=0, return_premature=True)
           # ... do other work, without reading td ...
           for work in works:
               work.wait()
           print((td["b"] != 0).all())  # tensor(True)

A receiver that does not know the structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the receiver does not know the keys, shapes, and dtypes of the tensordict,
call :meth:`~tensordict.TensorDictBase.init_remote` on the sender and
:meth:`~tensordict.TensorDictBase.from_remote_init` on the receiver. The
sender sends the metadata, then the data. The receiver builds a new
tensordict, which is consolidated. A tensorclass arrives as the same
tensorclass, if the receiver can import its class.

After this first transfer, send the next batches with
``send(consolidated=True)`` and ``recv(consolidated=True)``, into the
tensordict that ``from_remote_init`` returned. The receiver then does not
allocate memory again:

.. code-block:: python

   def cold_start(rank, world_size):
       if rank == 0:
           td = TensorDict(
               {"obs": torch.randn(2, 84), "reward": torch.randn(2, 1)}, batch_size=[2]
           )
           td.init_remote(dst=1)
           for step in range(3):
               td["obs"].add_(1)
               td.send(dst=1, consolidated=True)
       elif rank == 1:
           td = TensorDict.from_remote_init(src=0)
           print(td.batch_size, td.is_consolidated())  # torch.Size([2]) True
           for step in range(3):
               td.recv(src=0, consolidated=True)

With ``use_broadcast=True``, ``init_remote`` and ``from_remote_init`` call
:meth:`~tensordict.TensorDictBase.broadcast` instead, and all the ranks of
the group must call them:

.. code-block:: python

   def cold_start_broadcast(rank, world_size):
       if rank == 0:
           td = TensorDict({"obs": torch.randn(2, 84)}, batch_size=[2])
           td.init_remote(use_broadcast=True)
       else:
           td = TensorDict.from_remote_init(src=0, use_broadcast=True)
           print(td["obs"].shape)  # torch.Size([2, 84])

.. _distributed-collectives:

Collectives
-----------

Every rank of the group must call a collective. The methods that send the
structure (``broadcast``, ``scatter``, ``all_gather``,
``gather_and_stack``, ``init_remote``) send it with the object collectives of
:mod:`torch.distributed`, which use pickle.

.. warning::
   Use these methods only in a process group whose ranks you trust. Each
   receiver unpickles the metadata that the sender sends.

broadcast
~~~~~~~~~

:meth:`~tensordict.TensorDictBase.broadcast` sends the tensordict of rank
``src`` to all ranks, and returns it on each rank, consolidated. The other
ranks can start with an empty tensordict:

.. code-block:: python

   def broadcast(rank, world_size):
       td = TensorDict({"a": torch.arange(3.0)}, batch_size=[3]) if rank == 0 else TensorDict()
       td = td.broadcast(src=0)
       print(rank, td["a"])  # tensor([0., 1., 2.]) on every rank

all_reduce and reduce
~~~~~~~~~~~~~~~~~~~~~

:meth:`~tensordict.TensorDictBase.all_reduce` reduces each leaf over all
ranks, in place. With ``async_op=True``, it returns a list of ``Work``
objects to wait on:

.. code-block:: python

   def all_reduce(rank, world_size):
       td = TensorDict({"a": torch.ones(3) * rank}, batch_size=[3])
       td.all_reduce(op=dist.ReduceOp.SUM)
       print(rank, td["a"])  # with 2 ranks: tensor([1., 1., 1.]) on every rank
       works = td.all_reduce(async_op=True)
       for work in works:
           work.wait()

:meth:`~tensordict.TensorDictBase.reduce` writes the result on rank ``dst``
only. On the other ranks, the values of the tensordict after the call are
not defined: with ``gloo`` and 3 ranks, rank 1 held partial sums. If you need
the local values after the call, clone the tensordict before it:

.. code-block:: python

   def reduce(rank, world_size):
       td = TensorDict({"a": torch.ones(3) * (rank + 1)}, batch_size=[3])
       td.reduce(dst=0, op=dist.ReduceOp.SUM)
       if rank == 0:
           print(td["a"])  # with 2 ranks: tensor([3., 3., 3.])

Both issue one collective per leaf. With many small leaves, pass
``bucket_bytes`` to pack the leaves of each device and dtype into buckets of
at most that many bytes, and issue one collective per bucket, as DDP does for
gradients: ``td.all_reduce(bucket_bytes=2**20)``. With ``async_op=True``,
the list then holds one entry per bucket, and the results are in the leaves
once you have waited on each entry. Packing copies each leaf into its bucket
and back, which pays off for small leaves only: a leaf larger than
``bucket_bytes`` is reduced in place, without a copy. With ``gloo``, world size
2 to 4 and 100 leaves of 64 floats, ``bucket_bytes=2**20`` made
``all_reduce`` 40 to 70 times as fast. With 10 leaves of 4 MB, it made no
difference, and ``bucket_bytes=25 * 2**20``, which packs them, made the call 4
to 7 times slower.

all_gather, scatter, and gather_and_stack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~tensordict.TensorDictBase.all_gather` returns a list with the
tensordict of each rank, on every rank. The tensordicts can have different
structures:

.. code-block:: python

   def all_gather(rank, world_size):
       local = TensorDict({"a": torch.full((3,), float(rank))}, batch_size=[3])
       tds = local.all_gather()
       print(rank, [td["a"][0].item() for td in tds])  # [0.0, 1.0] on every rank

:meth:`~tensordict.TensorDictBase.scatter` sends one tensordict of a list
on rank ``src`` to each rank. The other ranks pass ``tensordicts=None``:

.. code-block:: python

   def scatter(rank, world_size):
       if rank == 0:
           tds = [
               TensorDict({"a": torch.full((3,), float(i))}, batch_size=[3])
               for i in range(world_size)
           ]
       else:
           tds = None
       td = TensorDict().scatter(src=0, tensordicts=tds)
       # tensor([0., 0., 0.]) on rank 0, tensor([1., 1., 1.]) on rank 1
       print(rank, td["a"])

:meth:`~tensordict.TensorDictBase.gather_and_stack` stacks the tensordicts of
the other ranks into the tensordict of rank ``dst``, in place. The
tensordict of rank ``dst`` must have one row for each other rank: its first
dimension is ``world_size - 1``. Its own data is not in the result:

.. code-block:: python

   def gather_and_stack(rank, world_size):
       if rank == 0:
           out = TensorDict({"a": torch.zeros(world_size - 1, 3)}, batch_size=[world_size - 1])
           out.gather_and_stack(dst=0)
           print(out["a"][:, 0])  # with 2 ranks: tensor([1.])
       else:
           td = TensorDict({"a": torch.full((3,), float(rank))}, batch_size=[3])
           td.gather_and_stack(dst=0)

Several transfers and custom groups
-----------------------------------

``send``, ``recv``, ``isend``, and ``irecv`` mark the message of each leaf with
a tag: ``init_tag``, ``init_tag + 1``, and so on. If two tensordicts go
from one rank to another at the same time, give them tag ranges that do not
overlap. Pass the same ``init_tag`` on both sides. With
``pseudo_rand=True``, the tags are pseudo-random numbers instead. Generating
them costs about 10 µs for each leaf.

.. code-block:: python

   def two_transfers(rank, world_size):
       if rank == 0:
           obs = TensorDict({"a": torch.ones(4), "b": torch.ones(4)}, batch_size=[4])
           reward = TensorDict({"r": torch.full((4,), 2.0)}, batch_size=[4])
           works = obs.isend(dst=1, init_tag=0, return_early=True)
           works += reward.isend(dst=1, init_tag=100, return_early=True)
           for work in works:
               work.wait()
       elif rank == 1:
           obs = TensorDict({"a": torch.zeros(4), "b": torch.zeros(4)}, batch_size=[4])
           reward = TensorDict({"r": torch.zeros(4)}, batch_size=[4])
           reward.irecv(src=0, init_tag=100)
           obs.irecv(src=0, init_tag=0)
           print(obs["a"][0], reward["r"][0])  # tensor(1.) tensor(2.)

Each method takes a ``group``. For ``send`` and ``recv``, ``dst`` and ``src``
are global ranks. To give a rank relative to the group, pass ``group_dst`` or
``group_src`` instead. With them, a standalone ``ProcessGroup`` that you
built against a store also works. All ranks must call
:func:`torch.distributed.new_group`, also the ranks that are not in the
group:

.. code-block:: python

   def custom_group(rank, world_size):
       group = dist.new_group([0, world_size - 1])
       if rank == 0:
           TensorDict({"a": torch.ones(3)}, batch_size=[3]).send(group=group, group_dst=1)
       elif rank == world_size - 1:
           td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
           td.recv(group=group, group_src=0)
           print(td["a"])  # tensor([1., 1., 1.])

A dataset on a shared file system
---------------------------------

If the nodes share a file system (for example NFS or Lustre), they can open
the same memory-mapped tensordict. No tensor then goes over the network:

1. On one rank, preallocate the files with
   :meth:`~tensordict.TensorDictBase.from_schema` (see :ref:`storage`). It
   does not allocate the dataset in memory.
2. On each rank, write only its own rows. tensordict does not order the
   writes between ranks (see :ref:`storage-sharing`).
3. Wait at a barrier before you read the rows of the other ranks. Ranks that
   only read can load the files with ``mode="r"`` (see :ref:`saving`).

The script of `Run the examples`_ imports ``shutil`` and ``tempfile`` for
this example:

.. code-block:: python

   def shared_dataset(rank, world_size):
       # On a cluster, use a directory on the shared file system.
       path = [tempfile.mkdtemp() if rank == 0 else None]
       dist.broadcast_object_list(path, src=0)
       path = path[0]
       if rank == 0:
           TensorDict.from_schema(
               {"image": ([3, 64, 64], torch.uint8), "label": ([], torch.int64)},
               batch_size=[1_000],
               storage="memmap",
               prefix=path,
           )
       dist.barrier()
       data = TensorDict.load_memmap(path)
       rows = torch.arange(rank, 1_000, world_size)
       data["label"][rows] = rows
       dist.barrier()
       reader = TensorDict.load_memmap(path, mode="r")
       print(rank, reader["label"][:4])  # MemoryMappedTensor([0, 1, 2, 3]) on every rank
       dist.barrier()
       if rank == 0:
           shutil.rmtree(path)

Stream between nodes with UCXX
------------------------------

.. warning::

    ``TensorDictPipe``, ``TensorDictServer``, ``send_tensordict``, and
    ``recv_tensordict`` are experimental: their API can change in a minor
    release without a deprecation cycle.

.. warning::

    UCXX helpers, pipes, and servers require trusted peers and a trusted
    network. The TensorDict protocol does not authenticate or authorize peers.
    Peers control allocation sizes and, for pipes, the default receive device.
    Set the receiver's ``device=`` to pin placement. Metadata checks do not
    bound memory use or prevent out-of-memory errors from large valid messages.
    After a protocol error, close the connection rather than retrying a receive.

:class:`~tensordict._ucxx.TensorDictPipe` sends tensordicts from one process
to one other process over `UCXX <https://github.com/rapidsai/ucxx>`_, the
Python bindings of `UCX <https://openucx.org/>`_. Use it when one node
streams batches to another, for example a data producer that feeds a
trainer. Compared with ``send`` and ``recv`` of ``torch.distributed``:

- It needs no process group. The two sides connect by IP address and port.
- After the first message, the receiver writes each new batch into the same
  buffer. It does not allocate memory or parse metadata.
- UCX selects the transport: TCP, InfiniBand verbs, or RoCE.
- If the tensors are on a GPU and GPUDirect RDMA is available, the buffer
  goes from GPU memory to the network, with no copy in CPU memory.
- The API uses :mod:`asyncio`, and has synchronous wrappers.

The classes are in the private module ``tensordict._ucxx``, and they need
the ``ucxx`` package. The examples of this section are not run when this page
is tested: ``ucxx`` is not installed on the machines that test it.

Install UCXX
~~~~~~~~~~~~

UCXX needs Linux, Python 3.11+, and the UCX system libraries. The ``pip``
packages are built for CUDA:

.. code-block:: bash

   pip install ucxx-cu12   # CUDA 12
   pip install ucxx-cu13   # CUDA 13
   conda install -c conda-forge -c rapidsai ucxx

For InfiniBand, the host needs ``rdma-core`` (or MOFED 5.0+) and Linux
5.6+. For GPUDirect RDMA, load the ``nvidia_peermem`` kernel module. To list
the InfiniBand devices and see whether the module is loaded:

.. code-block:: bash

   ibstat
   lsmod | grep nvidia_peermem

UCX selects the transports. To choose them, set environment variables before
the process imports ``ucxx``:

.. code-block:: bash

   export UCX_TLS=all          # let UCX select (the default)
   export UCX_NET_DEVICES=all
   export UCX_TLS=tcp          # TCP only, to debug

How a pipe sends a tensordict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pipe sends a consolidated tensordict (one storage and its metadata) in two
phases:

1. On the first message, the sender sends the metadata (keys, dtypes,
   shapes, byte offsets, as JSON), then the storage. The receiver allocates
   one buffer, and builds a tensordict whose leaves are views into it.
2. On the next messages with the same structure, the sender sends a one-byte
   flag, then the storage only. The receiver writes it into the buffer of
   the first message.

If the structure changes (other keys, shapes, or dtypes), the pipe sends the
metadata again. The sender compares a hash of the metadata at each send. If
the tensordict is not consolidated, the sender consolidates it at each send,
which allocates a new buffer. To avoid that, consolidate once and copy each
batch into the consolidated tensordict.

Connect two processes
~~~~~~~~~~~~~~~~~~~~~

One side listens on a port, and the other connects to it. ``pipe.arecv()``
returns the received tensordict. From the second message on, it returns the
same tensordict, with the new values:

.. code-block:: python

   import asyncio

   import torch

   from tensordict import TensorDict
   from tensordict._ucxx import TensorDictPipe


   async def receiver():  # on node A, with IP address 10.0.1.1
       pipe = await TensorDictPipe.listen(port=13337)
       for _ in range(100):
           td = await pipe.arecv()  # the same tensordict after the first message
           train(td)
       await pipe.aclose()


   async def sender():  # on node B
       pipe = await TensorDictPipe.connect("10.0.1.1", port=13337)
       buffer = None
       for obs, label in dataloader:
           batch = TensorDict({"obs": obs, "label": label}, batch_size=[obs.shape[0]])
           if buffer is None:
               buffer = batch.consolidate(metadata=True)  # allocate once
           else:
               buffer.update_(batch)  # copy into the consolidated storage
           await pipe.asend(buffer)
       await pipe.aclose()

``update_`` needs batches with the same shapes. If the last batch is
smaller, consolidate it on its own.

A pipe is also an async context manager, and an async iterator:

.. code-block:: python

   async with await TensorDictPipe.listen(13337) as pipe:
       async for td in pipe:
           train(td)

The ``async for`` loop ends when the pipe is closed. It also ends when an
error happens: the iterator turns every exception into the end of the loop,
and does not raise it. If you need the error, call ``await pipe.arecv()`` in
a ``while`` loop instead.

``pipe.send(td)``, ``pipe.recv()``, and ``pipe.close()`` are synchronous
wrappers. If no event loop runs in the thread, each call runs the coroutine
with :func:`asyncio.run`. In a coroutine, or in a callback of a running event
loop, use the async methods (``asend``, ``arecv``, ``aclose``). There, a
synchronous wrapper raises a ``RuntimeError`` that names the async method to
await, for example "TensorDictPipe.arecv() cannot run synchronously inside a
running event loop".
``TensorDictPipe.listen`` and ``TensorDictPipe.connect`` are coroutines only.

GPU tensors
~~~~~~~~~~~

If the leaves of the sender are on a GPU, the consolidated buffer is on the
GPU, and the pipe sends it from GPU memory. The receiver allocates its buffer
on the device of the sender, unless you pass ``device``:

.. code-block:: python

   td = TensorDict({"x": torch.randn(1024, 1024, device="cuda")}, batch_size=[1024])
   await pipe.asend(td)  # sender

   td = await pipe.arecv()  # receiver: td is on cuda
   td = await pipe.arecv(device="cpu")  # or receive on the CPU

send and recv with a pipe
~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~tensordict.TensorDictBase.send` and
:meth:`~tensordict.TensorDictBase.recv` also take a ``TensorDictPipe`` instead
of a rank, and :meth:`~tensordict.TensorDictBase.asend` and
:meth:`~tensordict.TensorDictBase.arecv` are their async versions. With a
pipe, ``recv`` does not always write into ``td``. It writes into ``td`` and
returns ``td`` if ``td`` is consolidated and has the layout of the sent
tensordict. Otherwise it leaves ``td`` unchanged and returns a tensordict of
the pipe: a new consolidated tensordict when the message brings a new schema,
and the tensordict that received the previous message when the schema has not
changed. A pipe made with ``consolidated=False`` never writes into ``td``.
Keep the return value, and receive into it:

.. code-block:: python

   td.send(dst=pipe)  # sender

   td = td.recv(src=pipe)  # receiver: use the return value
   td = await td.arecv(pipe)  # the async version

Several clients: TensorDictServer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~tensordict._ucxx.TensorDictServer` accepts several connections, and
gives a ``TensorDictPipe`` for each client. Each pipe can send and receive:

.. code-block:: python

   from tensordict._ucxx import TensorDictServer


   async def handle_client(pipe):
       async for td in pipe:
           await pipe.asend(model(td))


   async with TensorDictServer(port=13337) as server:
       await server.serve(handle_client)

To handle each connection yourself, iterate over the server:

.. code-block:: python

   async with TensorDictServer(port=13337) as server:
       async for pipe in server:
           asyncio.ensure_future(handle_client(pipe))

Benchmarks
~~~~~~~~~~

These numbers come from the pull request that added UCXX
(`#1611 <https://github.com/pytorch/tensordict/pull/1611>`_, March 2026). It
measured them on two nodes with NVIDIA H200 GPUs and 400 Gb/s InfiniBand,
with ``nvidia_peermem`` loaded. This page does not repeat the measurement.
The script is ``benchmarks/distributed/bench_transport.py``. It needs a Ray
cluster of two nodes, or the ``RANK``, ``MASTER_ADDR``, and ``PEER_ADDR``
environment variables.

Each tensordict has two leaves of the size in the "Size" column, so each
message has twice that size. Each time is the mean of 10 rounds, after 2
warm-up rounds. The UCXX rows are the steady state, after the first
message.

.. list-table:: CPU: gloo over TCP, and UCXX over InfiniBand
   :header-rows: 1
   :widths: 10 40 15 15

   * - Size
     - Method
     - Latency
     - Throughput
   * - 1 KB
     - gloo ``send``/``recv``, one message per leaf
     - 900 µs
     -
   * - 1 KB
     - gloo ``send``/``recv``, consolidated
     - 212 µs
     -
   * - 1 KB
     - UCXX pipe
     - 105 µs
     -
   * - 1 MB
     - gloo ``send``/``recv``, one message per leaf
     - 963 µs
     - 2.18 GB/s
   * - 1 MB
     - gloo ``send``/``recv``, consolidated
     - 903 µs
     - 2.32 GB/s
   * - 1 MB
     - UCXX pipe
     - 682 µs
     - 3.08 GB/s
   * - 100 MB
     - gloo ``send``/``recv``, one message per leaf
     - 87 ms
     - 2.40 GB/s
   * - 100 MB
     - gloo ``send``/``recv``, consolidated
     - 87 ms
     - 2.42 GB/s
   * - 100 MB
     - gloo ``broadcast``
     - 159 ms
     - 1.32 GB/s
   * - 100 MB
     - UCXX pipe
     - 37 ms
     - 5.71 GB/s

At 100 MB, the UCXX pipe took 37 ms and the fastest gloo method 87 ms.

.. list-table:: GPU: NCCL, and UCXX with GPUDirect RDMA
   :header-rows: 1
   :widths: 10 40 15 15

   * - Size
     - Method
     - Latency
     - Throughput
   * - 1 KB
     - NCCL ``send``/``recv``, one message per leaf
     - 231 µs
     -
   * - 1 KB
     - UCXX pipe
     - 138 µs
     -
   * - 1 MB
     - NCCL ``send``/``recv``, one message per leaf
     - 238 µs
     - 8.81 GB/s
   * - 1 MB
     - UCXX pipe
     - 231 µs
     - 9.08 GB/s
   * - 100 MB
     - NCCL ``send``/``recv``, one message per leaf
     - 7.50 ms
     - 27.95 GB/s
   * - 100 MB
     - UCXX pipe
     - 7.47 ms
     - 28.08 GB/s

On the GPU, NCCL and the UCXX pipe had the same throughput at 1 MB and
100 MB. At 1 KB, the UCXX pipe had a lower latency (138 µs and 231 µs). Both
used GPUDirect RDMA: the storage went from the GPU to the network without a
copy to the CPU.

See also
--------

- :ref:`data-loading`: share tensordicts between processes on one node.
- :ref:`storage`: memory-mapped datasets, and Redis stores that processes on
  several nodes share.
- :ref:`saving`: consolidated tensordicts.
- The reference of the distributed methods in :ref:`api-methods-distributed`,
  and of :class:`~tensordict._ucxx.TensorDictPipe` and
  :class:`~tensordict._ucxx.TensorDictServer` in :ref:`api-storage`.
- `PyTorch distributed documentation <https://pytorch.org/docs/stable/distributed.html>`_.
