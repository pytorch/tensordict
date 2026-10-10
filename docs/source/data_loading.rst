.. _data-loading:

Load data and share it between processes
========================================

This page shows how to read a tensordict in batches with a
:class:`~torch.utils.data.DataLoader`, and how to share a tensordict between
processes on one node: worker processes, ``torch.multiprocessing`` and
:meth:`~tensordict.TensorDictBase.map`. To store a dataset that is larger than
memory, see :ref:`storage`. To send tensordicts between nodes, see
:ref:`distributed`.

Use a tensordict as a dataset
-----------------------------

A tensordict with a batch dimension is a map-style dataset: ``len(td)`` is
the size of its first batch dimension, and ``td[i]`` is one sample (see
:ref:`batch-size`). It also has ``__getitems__``, so the ``DataLoader`` reads
each batch with one index operation, and the batch is a tensordict. Pass
``collate_fn=lambda x: x`` to keep that batch as it is:

  >>> import torch
  >>> from torch.utils.data import DataLoader
  >>> from tensordict import TensorDict
  >>> _ = torch.manual_seed(0)
  >>> data = TensorDict(x=torch.randn(10, 3), y=torch.arange(10), batch_size=[10])
  >>> loader = DataLoader(data, batch_size=4, shuffle=True, collate_fn=lambda x: x)
  >>> batches = list(loader)
  >>> [batch["y"].tolist() for batch in batches]
  [[7, 9, 2, 6], [1, 8, 4, 0], [3, 5]]
  >>> type(batches[0]).__name__, batches[0].batch_size
  ('TensorDict', torch.Size([4]))

The default ``collate_fn`` cannot read a tensordict. It raises
``TypeError: iteration over a 0-d tensordict``:

  >>> next(iter(DataLoader(data, batch_size=4)))
  Traceback (most recent call last):
  ...
  TypeError: iteration over a 0-d tensordict. Use keys(), values() or items() to iterate over its entries.

The same works for a memory-mapped tensordict (see :ref:`saving`). Each batch
then reads only its rows from disk.

Without a ``DataLoader``, index the tensordict with a permutation. With no
worker processes, this gives the same batches:

  >>> for idx in torch.randperm(len(data)).split(4):
  ...     batch = data[idx]
  >>> batch.batch_size
  torch.Size([2])

A dataset that returns tensordicts
----------------------------------

A dataset can return one tensordict for each sample. Examples are a list of
tensordicts, or a :class:`~torch.utils.data.Dataset` whose ``__getitem__``
builds a tensordict. The ``DataLoader`` then gets a list of tensordicts for
each batch. The default ``collate_fn`` raises the same ``TypeError``. Pass
``collate_fn=torch.stack`` to stack the list into one tensordict:

  >>> samples = [TensorDict(x=torch.randn(3), y=i) for i in range(10)]
  >>> batch = next(iter(DataLoader(samples, batch_size=4, collate_fn=torch.stack)))
  >>> batch.batch_size, batch["y"]
  (torch.Size([4]), tensor([0, 1, 2, 3]))

If the samples have different shapes, ``torch.stack`` raises an error. Pass
:func:`~tensordict.lazy_stack` instead. It returns a
:class:`~tensordict.LazyStackedTensorDict`, which keeps each sample as it is
(see :ref:`stacking`):

  >>> from tensordict import lazy_stack
  >>> ragged = [TensorDict(x=torch.randn(i + 1)) for i in range(4)]
  >>> batch = next(iter(DataLoader(ragged, batch_size=4, collate_fn=lazy_stack)))
  >>> type(batch).__name__, batch.batch_size
  ('LazyStackedTensorDict', torch.Size([4]))
  >>> [sample["x"].shape for sample in batch.unbind(0)]
  [torch.Size([1]), torch.Size([2]), torch.Size([3]), torch.Size([4])]

Batch samplers
--------------

A batch sampler gives the indices of a whole batch. Pass it as ``sampler``,
with ``batch_size=None``. The ``DataLoader`` then calls ``data[indices]`` with
the list of indices of each batch, and ``collate_fn`` gets the tensordict:

  >>> from torch.utils.data import BatchSampler, SequentialSampler
  >>> sampler = BatchSampler(SequentialSampler(data), batch_size=4, drop_last=True)
  >>> loader = DataLoader(data, sampler=sampler, batch_size=None, collate_fn=lambda x: x)
  >>> [batch["y"].tolist() for batch in loader]
  [[0, 1, 2, 3], [4, 5, 6, 7]]

Worker processes and pinned memory
----------------------------------

With ``num_workers`` greater than 0, the ``DataLoader`` reads the batches in
worker processes. Each worker gets the dataset when it starts:

- With the ``fork`` start method (the default on Linux up to Python 3.13),
  the worker inherits a copy-on-write view of the memory of the main process.
- With ``spawn`` (the default on macOS and Windows) and ``forkserver`` (the
  default on Linux from Python 3.14), the dataset is pickled to each worker.
  A memory-mapped tensordict is pickled as its file names, so the workers
  open the same files and do not copy the data. A ``lambda`` cannot be
  pickled: define ``collate_fn`` as a function at the top level of a module.

The worker examples start processes, so they are complete scripts. Save one
to a file and run it with ``python``:

.. code-block:: python

   import tempfile

   import torch
   from torch.utils.data import DataLoader

   from tensordict import TensorDict


   def keep_batch(batch):
       return batch


   if __name__ == "__main__":
       with tempfile.TemporaryDirectory() as path:
           data = TensorDict(
               x=torch.randn(1_000, 3), y=torch.arange(1_000), batch_size=[1_000]
           ).memmap_(path)
           loader = DataLoader(
               data, batch_size=100, shuffle=True, num_workers=2, collate_fn=keep_batch
           )
           total = sum(batch["y"].sum().item() for batch in loader)
           print(total)  # 499500

With ``pin_memory=True``, the ``DataLoader`` calls
:meth:`~tensordict.TensorDictBase.pin_memory` on each batch, which pins every
leaf. Pinned memory lets ``to(device, non_blocking=True)`` copy the batch to
the GPU while the CPU goes on. These lines need a GPU, so they are not run
when this page is tested:

  >>> loader = DataLoader(data, batch_size=4, collate_fn=lambda x: x, pin_memory=True)  # doctest: +SKIP
  >>> batch = next(iter(loader)).to("cuda", non_blocking=True)  # doctest: +SKIP

Without a ``DataLoader``, ``td.to("cuda", non_blocking_pin=True)`` pins the
leaves with several threads (``num_threads``) and copies them to the GPU. See
:ref:`devices`.

Share a tensordict between processes
------------------------------------

:meth:`~tensordict.TensorDictBase.share_memory_` moves each tensor of a
tensordict to shared memory. When you then pass the tensordict to another
process, as an argument or through a ``torch.multiprocessing`` queue, that
process uses the same memory. The writes of each process are visible to the
others, without a copy.

``share_memory_`` locks the tensordict: you can write into its tensors in
place (``td[i] = ...``, ``td["x"].add_(1)``, ``td.update_(other)``), but you
cannot add or replace entries. This script starts four processes, and each
one fills one row:

.. code-block:: python

   import torch
   import torch.multiprocessing as mp

   from tensordict import TensorDict


   def fill_row(td, i):
       td[i] = TensorDict(x=torch.full((3,), float(i)))


   if __name__ == "__main__":
       td = TensorDict(x=torch.zeros(4, 3), batch_size=[4]).share_memory_()
       processes = [mp.Process(target=fill_row, args=(td, i)) for i in range(4)]
       for p in processes:
           p.start()
       for p in processes:
           p.join()
       print(td["x"][:, 0])  # tensor([0., 1., 2., 3.])

Call ``share_memory_`` before you start the processes. Without it, the
result depends on the start method. With ``fork``, the child writes into a
copy, and the parent does not see the writes. With ``spawn``, PyTorch moves
the tensors to shared memory when it pickles them, so the parent sees the
writes.

The same works with ``memmap_()`` instead of ``share_memory_()``. The tensors
are then in files: memory-mapped files in a directory if you give a path,
anonymous temporary files if you do not. Use files when the data is larger
than RAM, or when processes that you start later must open the data. Shared
memory is limited by the size of ``/dev/shm``.

A queue works the same way. The process that gets the tensordict uses the
memory of the process that sent it:

.. code-block:: python

   import torch
   import torch.multiprocessing as mp

   from tensordict import TensorDict


   def consume(queue):
       td = queue.get()
       td["x"].add_(100)


   if __name__ == "__main__":
       queue = mp.Queue()
       td = TensorDict(x=torch.zeros(4, 3), batch_size=[4]).share_memory_()
       p = mp.Process(target=consume, args=(queue,))
       p.start()
       queue.put(td)
       p.join()
       print(td["x"][0])  # tensor([100., 100., 100.])

Several processes on one dataset on disk
----------------------------------------

Processes can also open a memory-mapped dataset by its path, with
:meth:`~tensordict.TensorDictBase.load_memmap`. Readers that do not change the
data can pass ``mode="r"`` (see :ref:`saving`). tensordict does not lock the
files: give each writer its own rows. This script preallocates a dataset on
disk, fills it from two processes, and reads it back:

.. code-block:: python

   import tempfile

   import torch
   import torch.multiprocessing as mp

   from tensordict import TensorDict


   def fill(path, start, stop):
       data = TensorDict.load_memmap(path)
       for i in range(start, stop):
           data[i] = TensorDict(image=torch.full((3, 8, 8), i, dtype=torch.uint8))


   if __name__ == "__main__":
       with tempfile.TemporaryDirectory() as path:
           TensorDict.from_schema(
               {"image": ([3, 8, 8], torch.uint8)},
               batch_size=[100],
               storage="memmap",
               prefix=path,
           )
           writers = [
               mp.Process(target=fill, args=(path, 0, 50)),
               mp.Process(target=fill, args=(path, 50, 100)),
           ]
           for p in writers:
               p.start()
           for p in writers:
               p.join()
           data = TensorDict.load_memmap(path, mode="r")
           # tensor([ 0, 49, 50, 99], dtype=torch.uint8)
           print(data["image"][[0, 49, 50, 99], 0, 0, 0])

Process a dataset in parallel with map
--------------------------------------

:meth:`~tensordict.TensorDictBase.map` splits a tensordict into chunks along
one dimension, calls a function on each chunk in a pool of processes, and
concatenates the results. If the tensordict is memory-mapped or in shared
memory, the chunks go to the workers without a copy. The function gets a
locked chunk of a memory-mapped tensordict, so it must return a new
tensordict instead of adding keys to its input. If you pass a preallocated
memory-mapped tensordict as ``out``, the workers write the results into it:

.. code-block:: python

   import tempfile

   import torch

   from tensordict import TensorDict


   def double(chunk):
       return TensorDict(y=chunk["x"] * 2, batch_size=chunk.batch_size)


   if __name__ == "__main__":
       with tempfile.TemporaryDirectory() as path:
           data = TensorDict(x=torch.arange(8.0), batch_size=[8]).memmap_()
           out = TensorDict(y=torch.zeros(8), batch_size=[8]).memmap_like(path)
           data.map(double, num_workers=2, chunksize=2, out=out)
           print(out["y"][:4])  # MemoryMappedTensor([0., 2., 4., 6.])

``map`` also takes ``num_chunks``, a ``pool`` that you create, and a
``generator`` that seeds each worker. :meth:`~tensordict.TensorDictBase.map_iter`
returns the results one chunk at a time.

See also
--------

- :ref:`storage`: memory-mapped, HDF5, zarr, shared-memory and Redis
  backends.
- :ref:`saving`: memory-mapped directories and the ``mode`` of
  ``load_memmap``.
- :ref:`distributed`: send tensordicts between nodes.
- The reference of :meth:`~tensordict.TensorDictBase.share_memory_`,
  :meth:`~tensordict.TensorDictBase.pin_memory`,
  :meth:`~tensordict.TensorDictBase.map` and :func:`~tensordict.lazy_stack`.
- Tutorials: :ref:`sphx_glr_tutorials_data_fashion.py`,
  :ref:`sphx_glr_tutorials_tensorclass_fashion.py` and
  :ref:`sphx_glr_tutorials_tensorclass_imagenet.py`.
