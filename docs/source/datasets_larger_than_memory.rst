.. currentmodule:: tensordict

.. meta::
   :description: To train a PyTorch model on a dataset larger than RAM, save
      it with tensordict as memory-mapped files, one per tensor. A DataLoader
      then reads only the rows of each batch from disk, in one index
      operation, and its workers share the files.

.. _datasets-larger-than-memory:

Train on a dataset larger than memory
=====================================

How do you train a PyTorch model on a dataset that does not fit in RAM?
Common answers are :class:`numpy.memmap`, ``h5py``, zarr, WebDataset and
Mosaic Streaming. With tensordict, you save the dataset, a dict of tensors
such as images, labels and nested metadata, as a directory with one
memory-mapped file for each tensor. :meth:`~tensordict.TensorDictBase.load_memmap`
opens it without reading the data, and an index reads only the rows that it
selects. The result is a :class:`~tensordict.TensorDict`, so a
:class:`~torch.utils.data.DataLoader` reads a whole batch with one index
operation, and the worker processes open the same files instead of copying
the data.

This page writes a dataset to disk, row by row or at once, reads it, grows it,
and trains from it with a ``DataLoader``. It ends with the cases where another
format fits better. The datasets here are small, so that the examples run
in a few seconds. The same code works for a dataset of any size that fits
on the disk. Run the examples in order in one Python session:

  >>> import tempfile
  >>> from pathlib import Path
  >>> import torch
  >>> import tensordict
  >>> from tensordict import MemoryMappedTensor, TensorDict
  >>> _ = torch.manual_seed(0)
  >>> device = "cuda" if torch.cuda.is_available() else "cpu"
  >>> tmp = tempfile.TemporaryDirectory()
  >>> root = Path(tmp.name)

Preallocate the dataset on disk and fill its rows
-------------------------------------------------

If you know the number of samples, create the files first, then write the
samples one at a time or in blocks, as you produce them. Make one sample
with the dtypes that you want, expand it to the number of samples, and call
:meth:`~tensordict.TensorDictBase.memmap_like`. ``expand`` does not allocate
memory, and ``memmap_like`` creates the files without writing the values:

  >>> sample = TensorDict(
  ...     {
  ...         "image": torch.zeros(3, 32, 32, dtype=torch.uint8),
  ...         "label": torch.zeros((), dtype=torch.int64),
  ...         "meta": {"id": torch.zeros((), dtype=torch.int64)},
  ...     }
  ... )
  >>> dataset = sample.expand(1_000).memmap_like(root / "images")
  >>> dataset["image"].shape, dataset["meta", "id"].shape
  (torch.Size([1000, 3, 32, 32]), torch.Size([1000]))

The files are sparse: a file uses disk space where you write to it. Write a
sample to a row, or a block of samples to a slice. The writes go to the
files:

  >>> dataset[0] = TensorDict(
  ...     {"image": torch.ones(3, 32, 32, dtype=torch.uint8), "label": 7, "meta": {"id": 0}}
  ... )
  >>> dataset[1:5] = TensorDict(
  ...     {
  ...         "image": torch.randint(256, (4, 3, 32, 32), dtype=torch.uint8),
  ...         "label": torch.tensor([1, 2, 3, 4]),
  ...         "meta": {"id": torch.arange(1, 5)},
  ...     },
  ...     batch_size=[4],
  ... )
  >>> dataset["label"][:6]
  MemoryMappedTensor([7, 1, 2, 3, 4, 0])

The tensordict that ``memmap_like`` returns is locked: you can write into its
tensors, but you cannot add a key (see :ref:`keys-locking`). Declare every
key in the sample. :meth:`~tensordict.TensorDictBase.from_schema` builds the
same files from a dict of shapes and dtypes, and :ref:`storage` shows how
several processes can fill one dataset.

For a single array, :class:`~tensordict.MemoryMappedTensor` is the tensor
form of :class:`numpy.memmap`. Put several of them in a tensordict to keep
them together:

  >>> scores = MemoryMappedTensor.empty((1_000, 10), dtype=torch.float32, filename=root / "scores.memmap")
  >>> scores[0] = torch.arange(10.0)
  >>> TensorDict({"scores": scores}, batch_size=[1_000])["scores"][0, :3]
  MemoryMappedTensor([0., 1., 2.])

Write a dataset once
--------------------

If the dataset, or a first part of it, is in memory,
:meth:`~tensordict.TensorDictBase.memmap` writes it to a directory in one
call. ``num_threads`` writes several files at the same time, which helps
when the tensordict has a few large tensors (see :ref:`saving`):

  >>> def make_part(start, n):
  ...     return TensorDict(
  ...         {
  ...             "image": torch.randint(256, (n, 3, 32, 32), dtype=torch.uint8),
  ...             "label": torch.arange(start, start + n) % 10,
  ...         },
  ...         batch_size=[n],
  ...     )
  >>> _ = make_part(0, 1_000).memmap(root / "train", num_threads=4)

``memmap`` returns a tensordict whose tensors are in the new files.

Open the dataset: load_memmap reads nothing until you index
-----------------------------------------------------------

:meth:`~tensordict.TensorDictBase.load_memmap` reads the ``meta.json`` files
of the directory, and maps each tensor file into memory. It does not read the
data: the operating system reads a page of a file from disk when you first
use it, and keeps the pages that you read often in its page cache. Each
tensor of the result is a :class:`~tensordict.MemoryMappedTensor`:

  >>> train = TensorDict.load_memmap(root / "train")
  >>> train.batch_size, type(train["image"]).__name__
  (torch.Size([1000]), 'MemoryMappedTensor')

Index with a tensor or a list of rows to read a random batch. The result is
a tensordict of regular tensors in memory, with a copy of these rows only:

  >>> rows = torch.tensor([3, 500, 999])
  >>> batch = train[rows]
  >>> type(batch["image"]).__name__, batch["image"].shape, batch["label"]
  ('Tensor', torch.Size([3, 3, 32, 32]), tensor([3, 0, 9]))

The rows of a tensor are contiguous in its file, so each row reads one
region of each file. An integer or a slice returns a view of the files
instead, which reads nothing until you use it.

To read the keys, shapes and dtypes without mapping any file, load on the
``"meta"`` device:

  >>> TensorDict.load_memmap(root / "train", device="meta")["image"]
  tensor(..., device='meta', size=(1000, 3, 32, 32), dtype=torch.uint8)

If many nodes read the files on a network file system, open them with
``mode="r"``. :ref:`saving` explains why, and what each ``mode`` does.

Grow a saved dataset
--------------------

:func:`~tensordict.append_memmap` writes new rows at the end of each file of
a saved directory. Use it to write a dataset that you produce in parts, such
as the output of a simulation or of a crawl, without holding all of it in
memory:

  >>> for start in (1_000, 2_000):
  ...     tensordict.append_memmap(root / "train", make_part(start, 1_000))
  >>> len(train), len(TensorDict.load_memmap(root / "train"))
  (1000, 3000)
  >>> train = TensorDict.load_memmap(root / "train")

The new rows must have the keys, dtypes and trailing shapes of the saved
ones, and ``append_memmap`` checks this before it writes anything. It
updates the ``meta.json`` files last, so a failed append leaves the dataset
as it was. A tensordict that you loaded before the append keeps its number
of rows, as ``train`` shows: load the directory again to see the new ones.
Append from one process at a time.

Do not leave a half-written directory
-------------------------------------

A job that is killed while ``memmap`` writes leaves some files written and
some not. With ``atomic=True``, ``memmap`` writes to a new directory next to
the path, then renames it to the path. A reader, or the next attempt of the
job, finds the old dataset or the new one:

  >>> _ = make_part(0, 100).memmap(root / "val", atomic=True)
  >>> TensorDict.load_memmap(root / "val").batch_size
  torch.Size([100])

The write needs room on the disk for both directories until it ends.
:ref:`saving` describes the case where the job is killed between the two
renames, and the platforms that support ``atomic=True``.

Train with a DataLoader
-----------------------

A tensordict is a map-style dataset: ``len(train)`` is its number of rows. It
also has ``__getitems__``, so the ``DataLoader`` reads each batch with one
index operation, ``train[indices]``, instead of reading the samples one by
one and stacking them. The default ``collate_fn`` returns that batch as it
is (see :ref:`data-loading`). With ``pin_memory=True``, the ``DataLoader`` pins the batch, and
``to(device, non_blocking=True)`` copies it to the GPU while the CPU goes on:

  >>> from torch.utils.data import DataLoader
  >>> loader = DataLoader(
  ...     train, batch_size=256, shuffle=True, pin_memory=torch.cuda.is_available()
  ... )
  >>> for batch in loader:
  ...     batch = batch.to(device, non_blocking=True)
  ...     images = batch["image"].float() / 255
  >>> batch.batch_size, images.shape
  (torch.Size([184]), torch.Size([184, 3, 32, 32]))

With ``num_workers`` greater than 0, each worker process reads its own
batches. With the ``spawn`` and ``forkserver`` start methods, the
``DataLoader`` pickles the dataset to each worker. A memory-mapped tensordict
pickles as the names of its files, so the pickle stays small, and each
worker maps the same files. The pages that one worker reads are then in the
page cache for the others:

  >>> import pickle
  >>> train["image"].nbytes, len(pickle.dumps(train)) < 10_000
  (9216000, True)

The workers start processes, so this example is a complete script. Save it
to a file and run it with ``python``:

.. code-block:: python

   import tempfile

   import torch
   from torch.utils.data import DataLoader

   from tensordict import TensorDict

   if __name__ == "__main__":
       device = "cuda" if torch.cuda.is_available() else "cpu"
       with tempfile.TemporaryDirectory() as path:
           TensorDict(
               {
                   "image": torch.randint(256, (10_000, 3, 32, 32), dtype=torch.uint8),
                   "label": torch.arange(10_000) % 10,
               },
               batch_size=[10_000],
           ).memmap(path)
           train = TensorDict.load_memmap(path)
           loader = DataLoader(
               train,
               batch_size=256,
               shuffle=True,
               num_workers=2,
               pin_memory=torch.cuda.is_available(),
               persistent_workers=True,
           )
           for epoch in range(2):
               seen = 0
               for batch in loader:
                   batch = batch.to(device, non_blocking=True)
                   seen += len(batch)
               print(epoch, seen)  # 0 10000, then 1 10000

If you pass your own ``collate_fn``, define it at the top level of a module:
with ``spawn`` and ``forkserver``, pickle cannot send a ``lambda`` to the
workers. :ref:`data-loading` covers batch samplers and the start methods, and :ref:`training-at-scale` shows how to copy the next batch
to the GPU while the model runs.

When to use another format
--------------------------

Memory-mapped files do not compress the data, and every process that reads
them must open the files: on one node, or on a shared file system. Other
formats fit other cases, and tensordict reads or writes some of them:

- **HDF5**, to give the data as one file to tools that read HDF5.
  :class:`~tensordict.PersistentTensorDict` reads and writes HDF5 files
  through ``h5py``. It pickles as its file name, so it can go to
  ``DataLoader`` workers, which a ``h5py`` object cannot. Close the file
  that you wrote, and open it again with ``mode="r"`` before you train,
  because ``h5py`` locks a file that is open for writing, and the workers
  then cannot open it. An index returns a lazy view of the rows: to read
  them in the workers, call :meth:`~tensordict.TensorDictBase.to_tensordict`
  in the ``collate_fn``. ``operator.methodcaller`` makes a function that
  pickles. See :ref:`storage`:

    >>> from operator import methodcaller
    >>> from tensordict import PersistentTensorDict
    >>> val = TensorDict.load_memmap(root / "val")
    >>> PersistentTensorDict.from_dict(val, root / "val.h5").close()
    >>> h5 = PersistentTensorDict.from_h5(root / "val.h5", mode="r")
    >>> loader = DataLoader(h5, batch_size=32, collate_fn=methodcaller("to_tensordict"))
    >>> batch = next(iter(loader))
    >>> type(batch).__name__, batch["label"][:4]
    ('TensorDict', tensor([0, 1, 2, 3]))
    >>> h5.close()

- **zarr**, to store the data in chunks, compressed, or on an object store.
  :meth:`~tensordict.TensorDictBase.to_zarr` writes a tensordict.
  ``chunks`` sets the number of rows in each chunk, and ``compressors``
  compresses the chunks: without it, the data is not compressed. zarr does
  not map the files: each read decodes the chunks that hold the rows. To
  read a batch, index the rows first, then the key: ``td["image"]`` reads
  the whole array.
  zarr 3 needs Python 3.11 or later (``pip install "tensordict[zarr]"``).
  See :ref:`storage-zarr`:

  .. code-block:: python

     from zarr.codecs import ZstdCodec

     # 16 rows in each chunk, compressed with Zstandard
     store = val.to_zarr(root / "val.zarr", chunks=(16,), compressors=ZstdCodec())
     batch = store[torch.tensor([3, 57])]  # reads rows 3 and 57
     batch["label"]  # tensor([3, 7])

- **Mosaic Streaming**, to stream shards from an object store, such as S3,
  to the training nodes, with shuffling across shards.
  :meth:`~tensordict.TensorDictBase.to_mds` writes a tensordict as an MDS
  dataset (``pip install mosaicml-streaming``), which
  ``streaming.StreamingDataset`` reads back.
- **WebDataset**, to stream tar shards from an object store or over HTTP.
  tensordict does not write WebDataset shards: build a tensordict from each
  sample that WebDataset returns, and batch them with
  :func:`~tensordict.collate`.

How the formats compare
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 21 21 21 21

   * -
     - tensordict memory-mapped files
     - :class:`numpy.memmap`
     - ``h5py``
     - zarr
   * - What it stores
     - A tree of tensors with a shared batch size, one file each, with
       their shapes and dtypes
     - One array for each file. The raw file stores no shape or dtype;
       a ``.npy`` file does
     - A tree of arrays in one file
     - A tree of arrays in a directory, a zip file, or an object store
   * - Read rows
     - Maps the files: the operating system reads the pages of the rows
     - Maps the file, like tensordict
     - Reads the rows, or decodes their chunks if the dataset is chunked
     - Reads and decodes the chunks of the rows
   * - Chunks and compression
     - No
     - No
     - Yes
     - Yes
   * - Add rows
     - :func:`~tensordict.append_memmap`
     - Open the file again with a larger shape
     - Resize a dataset created with a ``maxshape``
     - ``resize`` or ``append`` the arrays
   * - In ``DataLoader`` workers
     - Pickles as file names
     - Pickles as a copy of the data
     - Does not pickle: open the file in each worker
     - Pickles; each worker opens the store
   * - A batch for the model
     - One index gives a tensordict
     - One index per array, then convert to tensors
     - One index per dataset, then convert to tensors
     - One index per array, then convert to tensors
   * - Object storage
     - No: local disk or a shared file system
     - No
     - Read-only, through ros3 or fsspec
     - Yes

``h5py`` and zarr also read the files of other tools, and tensordict reads
both through :class:`~tensordict.PersistentTensorDict`. Clean up the
temporary directory:

  >>> tmp.cleanup()

See also
--------

- :ref:`storage`: preallocate with ``from_schema``, and the HDF5, zarr,
  shared-memory and Redis backends.
- :ref:`saving`: the options of ``memmap`` and ``load_memmap``, archives,
  and the speed of each format.
- :ref:`data-loading`: worker processes, batch samplers, and processes that
  share one dataset on disk.
- The reference of :meth:`~tensordict.TensorDictBase.memmap`,
  :meth:`~tensordict.TensorDictBase.memmap_like`,
  :meth:`~tensordict.TensorDictBase.load_memmap`,
  :func:`~tensordict.append_memmap`, and
  :class:`~tensordict.MemoryMappedTensor`.
- Tutorials: :ref:`sphx_glr_tutorials_tensordict_memory.py` and
  :ref:`sphx_glr_tutorials_zarr_storage.py`.
