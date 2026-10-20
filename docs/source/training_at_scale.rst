.. _training-at-scale:

Train at scale
==============

This page answers the questions that come up when tensordicts feed a large
training job: how to copy batches to the GPU while the model runs, how to
resume a data loader, how to checkpoint with
:mod:`torch.distributed.checkpoint`, what works on DTensor leaves, how to cast
for mixed precision, how to send tensordicts to other processes, and what each
collective costs. Most answers use methods that you already know from
:ref:`devices`, :ref:`data-loading` and :ref:`distributed`.

The examples that need a GPU, several processes, or a package that tensordict
does not depend on are marked ``+SKIP`` or shown as scripts. The others run
when this page is tested.

Prefetch batches to the GPU
---------------------------

A copy from pinned (page-locked) CPU memory to the GPU can run while the GPU
computes. To overlap the copy of the next batch with the work on the current
one:

1. Call :meth:`~tensordict.TensorDictBase.consolidate` with
   ``pin_memory=True``. It copies all the leaves into one pinned buffer, and
   makes each leaf a view into it.
2. On a second CUDA stream, call ``to(device, non_blocking=True)``. A
   consolidated tensordict goes to the GPU in one copy, and the result is
   consolidated too.
3. Before the model reads the batch, make the main stream wait for the
   second stream with :meth:`torch.cuda.Stream.wait_stream`.
4. Call :meth:`~tensordict.TensorDictBase.record_stream` with the main
   stream. The batch was allocated on the second stream, and this keeps its
   memory from being reused before the main stream is done with it.

Step 1 pins only a batch that is not consolidated yet. A consolidated batch
is returned as it is, and if its buffer is not pinned, the copy of step 2
does not overlap. With this recipe, the ``collate_fn`` of the loader should
return the batch without consolidating it.

This generator yields each batch of ``loader`` on ``device``, one batch
ahead:

  >>> import torch
  >>> from tensordict import TensorDict
  >>> def prefetch(loader, device):
  ...     stream = torch.cuda.Stream(device)
  ...     pending = None
  ...     for batch in loader:
  ...         batch = batch.consolidate(pin_memory=True)
  ...         with torch.cuda.stream(stream):
  ...             batch = batch.to(device, non_blocking=True)
  ...         if pending is not None:
  ...             yield pending
  ...         torch.cuda.current_stream(device).wait_stream(stream)
  ...         batch.record_stream(torch.cuda.current_stream(device))
  ...         pending = batch
  ...     if pending is not None:
  ...         yield pending

Use it on any loader that yields tensordicts (see :ref:`data-loading`). These
lines need a GPU, so they are not run when this page is tested:

  >>> from torch.utils.data import DataLoader
  >>> data = TensorDict(
  ...     {"image": torch.randn(1_000, 3, 32, 32), "label": torch.arange(1_000)},
  ...     batch_size=[1_000],
  ... )
  >>> loader = DataLoader(data, batch_size=100, shuffle=True, collate_fn=lambda x: x)
  >>> for batch in prefetch(loader, "cuda"):  # doctest: +SKIP
  ...     loss = model(batch)

Do not also pass ``pin_memory=True`` to the ``DataLoader``. The loader then
calls :meth:`~tensordict.TensorDictBase.pin_memory`, which pins each leaf on
its own, and step 1 copies the batch again into a new pinned buffer: every
batch is pinned twice. Do not replace step 1 with ``pin_memory()`` either:
its result is not consolidated, and ``to()`` then makes one copy for each
leaf. In one test with three leaves, the consolidated batch went to the GPU in
one copy and the pinned batch in three.

Resume a data loader
--------------------

``StatefulDataLoader``, from the ``torchdata`` package
(``pip install torchdata``), adds ``state_dict()`` and ``load_state_dict()``
to the ``DataLoader``. A new loader that loads the state of an old one goes
on with the batches that the old one had not yet given.

Wrap the tensordict in a small :class:`~torch.utils.data.Dataset` before you
pass it to ``StatefulDataLoader``. A tensordict has ``state_dict()`` and
``load_state_dict()`` methods, so the loader takes it for a dataset with a
state: the state of the loader then holds every tensor of the dataset, and
saving the state saves the dataset. With worker processes and a
memory-mapped tensordict, the workers fail with
``OSError: [Errno 9] Bad file descriptor``. The wrapper below has no
``state_dict()``, and it keeps the batched reads of ``__getitems__``:

  >>> from torch.utils.data import Dataset
  >>> class Rows(Dataset):
  ...     def __init__(self, td):
  ...         self.td = td
  ...
  ...     def __len__(self):
  ...         return len(self.td)
  ...
  ...     def __getitem__(self, index):
  ...         return self.td[index]
  ...
  ...     def __getitems__(self, indices):
  ...         return self.td[indices]

``torchdata`` is not installed where this page is tested, so these lines are
not run. Here, the first loader gives one batch, and the second loader gives
the other four:

  >>> from torchdata.stateful_dataloader import StatefulDataLoader  # doctest: +SKIP
  >>> data = TensorDict({"x": torch.randn(20, 3), "label": torch.arange(20)}, batch_size=[20])
  >>> def make_loader():
  ...     return StatefulDataLoader(
  ...         Rows(data), batch_size=4, shuffle=True, collate_fn=lambda x: x
  ...     )
  >>> loader = make_loader()  # doctest: +SKIP
  >>> batches = iter(loader)  # doctest: +SKIP
  >>> first = next(batches)  # doctest: +SKIP
  >>> state = loader.state_dict()  # save it with the model checkpoint  # doctest: +SKIP
  >>> rest = [batch["label"] for batch in batches]  # doctest: +SKIP
  >>> resumed = make_loader()  # doctest: +SKIP
  >>> resumed.load_state_dict(state)  # doctest: +SKIP
  >>> resumed_rest = [batch["label"] for batch in resumed]  # doctest: +SKIP
  >>> len(rest), len(resumed_rest)  # doctest: +SKIP
  (4, 4)
  >>> all((a == b).all() for a, b in zip(rest, resumed_rest))  # doctest: +SKIP
  True

The same works with ``num_workers`` greater than 0 and with a memory-mapped
tensordict. As for the ``DataLoader``, worker processes need a ``collate_fn``
defined at the top level of a module, not a ``lambda``, with the ``spawn``
and ``forkserver`` start methods.

.. _training-at-scale-checkpoint:

Checkpoint with torch.distributed.checkpoint
--------------------------------------------

:func:`torch.distributed.checkpoint.save` and
:func:`torch.distributed.checkpoint.load` (DCP) accept a
:class:`~tensordict.TensorDict` or a :class:`~tensordict.TensorClass` as a
value of the state dict. DCP calls the ``state_dict()`` method of the value
to save it. To load, it fills the tensordict that you pass, in place, and
calls its ``load_state_dict()``. The tensordict must have the keys and the
shapes of the saved one.

Without a process group, DCP saves from one process. It warns that it
assumes so:

  >>> import tempfile
  >>> import warnings
  >>> import torch.distributed.checkpoint as dcp
  >>> from tensordict import TensorClass
  >>> class EMA(TensorClass):
  ...     step: torch.Tensor
  ...     value: torch.Tensor
  >>> buffers = TensorDict({"mean": torch.arange(4.0), "nested": {"var": torch.ones(4)}})
  >>> ema = EMA(step=torch.tensor(10), value=torch.full((4,), 0.5))
  >>> path = tempfile.mkdtemp()
  >>> with warnings.catch_warnings():
  ...     warnings.simplefilter("ignore")  # "torch.distributed is ... uninitialized"
  ...     _ = dcp.save({"buffers": buffers, "ema": ema, "epoch": 3}, checkpoint_id=path)
  ...     state = {
  ...         "buffers": buffers.apply(torch.zeros_like),
  ...         "ema": EMA(step=torch.tensor(0), value=torch.zeros(4)),
  ...         "epoch": 0,
  ...     }
  ...     dcp.load(state, checkpoint_id=path)
  >>> state["buffers"]["mean"], state["ema"].step, state["epoch"]
  (tensor([0., 1., 2., 3.]), tensor(10), 3)

:func:`torch.distributed.checkpoint.async_save` takes the same state dict. It
returns a future: call its ``result()`` before you change the tensordict or
start the next save.

In a training job, every rank calls these functions. This example saves and
loads on each rank of a process group. To run it, add it to the script of
:ref:`distributed` (section "Run the examples"), with
``import torch.distributed.checkpoint as dcp``, and pass it to ``run``:

.. code-block:: python

   def checkpoint(rank, world_size):
       path = [tempfile.mkdtemp() if rank == 0 else None]
       dist.broadcast_object_list(path, src=0)  # one directory for all ranks
       buffers = TensorDict({"mean": torch.arange(4.0)})
       dcp.async_save({"buffers": buffers}, checkpoint_id=path[0]).result()
       restored = buffers.apply(torch.zeros_like)
       dcp.load({"buffers": restored}, checkpoint_id=path[0])
       print(rank, restored["mean"])  # tensor([0., 1., 2., 3.]) on every rank

With DTensor leaves (see the next section), each rank saves its own shard.
DCP reshards when it loads: a checkpoint saved on 2 ranks loads on 4 ranks,
or into plain tensors in one process. This script saves a sharded
tensordict from two processes, then loads all of it in the main process:

.. code-block:: python

   import os
   import socket
   import tempfile

   import torch
   import torch.distributed as dist
   import torch.distributed.checkpoint as dcp
   import torch.multiprocessing as mp
   from torch.distributed.device_mesh import init_device_mesh
   from torch.distributed.tensor import Shard, distribute_tensor

   from tensordict import TensorDict


   def save_sharded(rank, world_size, port, path):
       os.environ["MASTER_ADDR"] = "127.0.0.1"
       os.environ["MASTER_PORT"] = str(port)
       dist.init_process_group("gloo", rank=rank, world_size=world_size)
       mesh = init_device_mesh("cpu", (world_size,))
       weight = distribute_tensor(torch.arange(16.0).reshape(8, 2), mesh, [Shard(0)])
       params = TensorDict({"weight": weight})
       print(rank, params["weight"].to_local().shape)  # torch.Size([4, 2])
       dcp.save({"params": params}, checkpoint_id=path)
       dist.destroy_process_group()


   if __name__ == "__main__":
       path = tempfile.mkdtemp()
       with socket.socket() as s:
           s.bind(("127.0.0.1", 0))
           port = s.getsockname()[1]
       mp.spawn(save_sharded, args=(2, port, path), nprocs=2)
       params = TensorDict({"weight": torch.zeros(8, 2)})
       dcp.load({"params": params}, checkpoint_id=path)
       print(params["weight"][:, 0])  # tensor([ 0.,  2.,  4.,  6.,  8., 10., 12., 14.])

DCP checkpoints are directories of files that other DCP jobs read. To save a
tensordict for other uses, such as a dataset or a file that you open without
``torch.distributed``, see :ref:`saving`.

DTensor leaves and FSDP2
------------------------

FSDP2 (:func:`torch.distributed.fsdp.fully_shard`) and tensor parallelism
turn parameters into :class:`~torch.distributed.tensor.DTensor` objects: each
rank holds a shard, and the shape of the DTensor is the global shape. A
tensordict can hold DTensor leaves, and its batch size refers to the global
shape. These operations work, and return tensordicts of DTensors where the
input has DTensors:

- Indexing, such as ``td[1:3]`` or ``td[0]``.
- :meth:`~tensordict.TensorDictBase.to`, with a device or a dtype.
- Arithmetic, such as ``td * 2 + 1``.
- :func:`torch.stack` and :func:`torch.cat`.
- :meth:`~tensordict.TensorDictBase.clone` and
  :meth:`~tensordict.TensorDictBase.state_dict`.
- :meth:`~tensordict.TensorDictBase.memmap`. Each rank writes the full
  tensor, not its shard, to its own files. To save shards, use DCP, as
  described in :ref:`training-at-scale-checkpoint`.

These do not work on DTensor leaves:

- :meth:`~tensordict.TensorDictBase.consolidate` raises
  ``NotImplementedError: DTensor is not supported yet``. So do the methods
  that consolidate, such as ``send(consolidated=True)`` and ``broadcast``.
- :meth:`~tensordict.TensorDictBase.all_reduce` raises
  ``NotImplementedError``: DTensor has no rule for ``c10d.allreduce_``.
  Reduce DTensors with the DTensor API, or call the collectives on the local
  shards.

To work on the local shards, apply ``to_local()``. The shards have a shape
other than the global one, so pass a new batch size. This example runs on two
ranks with the script of :ref:`distributed`, with
``from torch.distributed.device_mesh import init_device_mesh`` and
``from torch.distributed.tensor import Shard, distribute_tensor``:

.. code-block:: python

   def dtensor_leaves(rank, world_size):
       mesh = init_device_mesh("cpu", (world_size,))
       weight = distribute_tensor(torch.arange(8.0).reshape(4, 2), mesh, [Shard(0)])
       params = TensorDict({"weight": weight}, batch_size=[4])
       print(params[1:3]["weight"].full_tensor())  # tensor([[2., 3.], [4., 5.]])
       print(torch.stack([params, params]).batch_size)  # torch.Size([2, 4])
       local = params.apply(lambda x: x.to_local(), batch_size=[])
       print(rank, local["weight"].shape)  # torch.Size([2, 2]) on each rank

Cast for mixed precision
------------------------

A dtype given to :meth:`~tensordict.TensorDictBase.to`, or a method such as
``bfloat16()``, casts every leaf, also the labels, token ids and masks. With
``cast_filter=is_floating``, only the floating-point leaves are cast. The
predicates :func:`~tensordict.is_floating` and :func:`~tensordict.is_inexact`
(floating-point or complex) test the dtype of a leaf:

  >>> from tensordict import is_floating
  >>> batch = TensorDict(
  ...     {
  ...         "image": torch.randn(4, 3, 8, 8),
  ...         "label": torch.arange(4),
  ...         "mask": torch.ones(4, dtype=torch.bool),
  ...     },
  ...     batch_size=[4],
  ... )
  >>> bf16 = batch.to(torch.bfloat16, cast_filter=is_floating)
  >>> bf16["image"].dtype, bf16["label"].dtype, bf16["mask"].dtype
  (torch.bfloat16, torch.int64, torch.bool)
  >>> batch.bfloat16(cast_filter=is_floating)["label"].dtype
  torch.int64

A tensorclass can make a filter its default, with
``@tensorclass(cast_filter=is_inexact)``: see :func:`~tensordict.tensorclass`.

Send tensordicts to other processes
-----------------------------------

:mod:`pickle` writes one entry for each tensor. A consolidated tensordict
pickles as one storage and its metadata, which is faster when the
tensordict has many leaves (see :ref:`saving`). Consolidate before you
pickle: before a call to a Ray actor, a :mod:`multiprocessing` queue, or
any other API that pickles its arguments. The tensordict that the other
process gets is consolidated too:

  >>> import pickle
  >>> batch = TensorDict({f"metric{i}": torch.zeros(16) for i in range(100)}, batch_size=[16])
  >>> received = pickle.loads(pickle.dumps(batch.consolidate()))
  >>> received.is_consolidated(), received.batch_size
  (True, torch.Size([16]))

With Ray, pass the consolidated tensordict to the remote call. ``ray`` is not
installed where this page is tested, so these lines are not run:

  >>> import ray  # doctest: +SKIP
  >>> @ray.remote  # doctest: +SKIP
  ... class Trainer:
  ...     def step(self, batch):
  ...         return batch.is_consolidated()
  >>> trainer = Trainer.remote()  # doctest: +SKIP
  >>> ray.get(trainer.step.remote(batch.consolidate()))  # doctest: +SKIP
  True

In one measurement on one machine, with a tensordict of 1,000 leaves of 16
elements, a call to a Ray actor took 46 ms with the tensordict as it was, and
11 ms with the tensordict consolidated. ``consolidate()`` took 6 ms.

The processes of one node can also share a tensordict without a copy, in
shared memory or in memory-mapped files: see :ref:`data-loading`.

Collectives and their cost
--------------------------

:ref:`distributed` describes the collectives of tensordicts. Their cost
depends on how many messages they send:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Messages
   * - :meth:`~tensordict.TensorDictBase.all_reduce`,
       :meth:`~tensordict.TensorDictBase.reduce`
     - One collective for each leaf. With many small leaves, the latency of
       each call adds up.
   * - :meth:`~tensordict.TensorDictBase.broadcast`,
       :meth:`~tensordict.TensorDictBase.all_gather`,
       :meth:`~tensordict.TensorDictBase.scatter`
     - The sender consolidates the tensordict, unless it is consolidated
       already. Then one object collective sends the metadata, which is
       pickled, and one tensor collective sends the storage. ``all_gather``
       pads each storage to the size of the largest one.
   * - :meth:`~tensordict.TensorDictBase.gather_and_stack`
     - One :func:`torch.distributed.gather_object`, which pickles each whole
       tensordict.
   * - :meth:`~tensordict.TensorDictBase.send`,
       :meth:`~tensordict.TensorDictBase.recv`
     - One message for each leaf. With ``consolidated=True``, one message.

Consolidate a tensordict once, and reuse it, if you send it often: each
``consolidate()`` copies all the leaves. To fill a consolidated tensordict
with new values without a new allocation, use
:meth:`~tensordict.TensorDictBase.update_`.

See also
--------

- :ref:`devices`: ``to()``, ``non_blocking``, and pinned memory.
- :ref:`data-loading`: tensordicts as datasets, and shared memory.
- :ref:`distributed`: point-to-point transfers and collectives.
- :ref:`saving`: consolidated tensordicts, and the formats for saving.
- `torch.distributed.checkpoint
  <https://docs.pytorch.org/docs/stable/distributed.checkpoint.html>`_ and
  `torchdata StatefulDataLoader
  <https://meta-pytorch.org/data/beta/torchdata.stateful_dataloader.html>`_.
