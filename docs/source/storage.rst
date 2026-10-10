.. _storage:

Large datasets and storage backends
===================================

A tensordict can keep its tensors outside the memory of the process: in
memory-mapped files, in an HDF5 file, in a zarr store, in shared memory, or on
a Redis server. This page shows how to choose a backend, preallocate a
dataset on it, fill it, and share it with other processes. To save and load
a tensordict that is already in memory, see :ref:`saving`.

The examples on this page run in order and share one temporary directory,
``root``:

  >>> import tempfile
  >>> from pathlib import Path
  >>> import torch
  >>> from tensordict import TensorDict
  >>> tmp = tempfile.TemporaryDirectory()
  >>> root = Path(tmp.name)

.. contents:: On this page
   :local:
   :depth: 1

Choose a backend
----------------

.. list-table::
   :header-rows: 1
   :widths: 14 14 22 22 28

   * - Backend
     - ``storage=``
     - Result type
     - The data is in
     - Use it when
   * - Regular tensors
     - ``None``
     - :class:`~tensordict.TensorDict`
     - the memory of the process
     - The data fits in memory.
   * - Memory-mapped files
     - ``"memmap"``
     - :class:`~tensordict.TensorDict`
     - one file for each leaf, in a directory
     - The dataset is larger than RAM, and the processes that read it can
       open the same files (one node, or a shared file system).
   * - HDF5
     - ``"h5"``
     - :class:`~tensordict.PersistentTensorDict`
     - one HDF5 file
     - You need one portable file that ``h5py`` or HDFView can open.
   * - zarr
     - ``"zarr"``
     - :class:`~tensordict.PersistentTensorDict`
     - a directory, a zip file or an object store
     - You need chunks, compression or an object store, or you read the
       data with xarray or dask.
   * - Shared memory
     - ``"shared"``
     - :class:`~tensordict.TensorDict`
     - shared memory (RAM) of one node
     - Processes on one node read and write the same tensors, and the data
       fits in RAM.
   * - Redis-compatible server
     - ``"redis"``, ``"dragonfly"``
     - :class:`~tensordict.store.TensorDictStore`
     - the memory of a Redis, Dragonfly or KeyDB server
     - Processes on several nodes share data, without a shared file system.

Only the memory-mapped files, HDF5 and zarr keep the data after the
processes end. A Redis server keeps it while the server runs.

Declare a schema with from_schema
---------------------------------

:meth:`~tensordict.TensorDictBase.from_schema` creates a tensordict filled
with zeros on any backend. Its first argument maps each key to the shape of
one element and a dtype. ``batch_size`` comes before each element shape, so a
scalar reward with ``batch_size=[N]`` is a tensor of shape ``(N,)``:

  >>> schema = {
  ...     "obs": ([84, 84, 3], torch.uint8),
  ...     "action": ([4], torch.float32),
  ...     "reward": ([], torch.float32),
  ... }
  >>> td = TensorDict.from_schema(schema, batch_size=[100])
  >>> td["obs"].shape, td["reward"].shape
  (torch.Size([100, 84, 84, 3]), torch.Size([100]))

The ``storage`` keyword selects the backend. The other keywords go to the
backend:

  >>> on_disk = TensorDict.from_schema(
  ...     schema, batch_size=[100], storage="memmap", prefix=root / "replay"
  ... )
  >>> in_h5 = TensorDict.from_schema(
  ...     schema, batch_size=[100], storage="h5", filename=root / "replay.h5"
  ... )
  >>> in_zarr = TensorDict.from_schema(
  ...     schema, batch_size=[100], storage="zarr", filename=root / "replay.zarr"
  ... )
  >>> shared = TensorDict.from_schema(schema, batch_size=[100], storage="shared")
  >>> type(on_disk).__name__, type(in_h5).__name__, type(in_zarr).__name__
  ('TensorDict', 'PersistentTensorDict', 'PersistentTensorDict')

These are the keywords of each backend:

- ``"memmap"``: ``prefix``, the directory of the files. Without it, the files
  are anonymous temporary files (see :ref:`saving`).
- ``"h5"``: ``filename``, the path of the HDF5 file.
- ``"zarr"``: ``filename``, the path of the store or a
  ``zarr.abc.store.Store``, and any argument of
  :meth:`zarr.Group.create_array` (``chunks``, ``compressors``).
- ``"shared"``: none.
- ``"redis"`` and ``"dragonfly"``: ``host``, ``port``, ``db``,
  ``unix_socket_path``, ``prefix`` (the namespace of the keys on the server)
  and ``td_id`` (see `Share data between nodes with Redis`_).

Preallocate a dataset
---------------------

To build a dataset that is larger than memory, preallocate it, then write
one sample (or one batch) at a time:

  >>> schema = {"image": ([3, 64, 64], torch.uint8), "label": ([], torch.int64)}
  >>> buffer = TensorDict.from_schema(
  ...     schema, batch_size=[1_000], storage="memmap", prefix=root / "buffer"
  ... )
  >>> for i in range(4):
  ...     buffer[i] = TensorDict(
  ...         image=torch.full((3, 64, 64), i, dtype=torch.uint8), label=i
  ...     )
  >>> buffer["label"][:5]
  MemoryMappedTensor([0, 1, 2, 3, 0])

``from_schema`` builds each value as ``torch.zeros(()).expand(shape)``, a
view that uses no memory. The memory that the backend then uses depends on
the backend. These numbers were measured for a 256 MiB ``uint8`` schema:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Backend
     - Peak memory of the process
     - Disk or server
   * - ``None``
     - 256 MiB
     - none
   * - ``"memmap"``
     - 2 MiB
     - sparse files: disk space is used when you write to it
   * - ``"h5"``
     - 270 MiB
     - 256 MiB written to the file
   * - ``"zarr"``
     - 401 MiB
     - empty chunks are not written
   * - ``"shared"``
     - 512 MiB
     - 256 MiB stays in ``/dev/shm``
   * - ``"redis"``
     - 6 MiB
     - 256 MiB in the memory of the server

With ``"h5"`` and ``"zarr"``, ``from_schema`` writes the expanded zeros
through memory, so it needs more RAM than the dataset. With ``"shared"``, it
allocates the tensors, then copies them to shared memory. For a dataset that
is larger than RAM, use ``"memmap"`` or ``"redis"``. To get an HDF5 file
without this cost, create the datasets with ``h5py``, which writes no data.
Then open the file with :meth:`~tensordict.PersistentTensorDict.from_h5`. For
zarr, create the arrays with ``zarr`` and open the store with
:meth:`~tensordict.PersistentTensorDict.from_zarr`:

  >>> import h5py
  >>> from tensordict import PersistentTensorDict
  >>> with h5py.File(root / "large.h5", "w") as f:
  ...     _ = f.create_dataset("image", shape=(1_000, 3, 64, 64), dtype="uint8")
  >>> large = PersistentTensorDict.from_h5(root / "large.h5", mode="r+", batch_size=[1_000])
  >>> large[0] = TensorDict(image=torch.ones(3, 64, 64, dtype=torch.uint8))
  >>> large["image"][0].sum()
  tensor(12288)

For memory-mapped files, :meth:`~tensordict.TensorDictBase.memmap_like` is
another way to preallocate. It takes a tensordict with the shapes and dtypes
of the dataset, and creates zero-filled files of the same structure, without
copying the values. Make one sample and expand it, which does not allocate
memory:

  >>> datum = TensorDict(image=torch.zeros(3, 64, 64, dtype=torch.uint8), label=0)
  >>> dataset = datum.expand(1_000).memmap_like(root / "dataset")
  >>> dataset.batch_size
  torch.Size([1000])

Make sure that the sample has the dtype that you want.
``torch.randint(255, shape)`` gives ``int64``, which takes 8 times the space
of ``uint8``. If the path ends with
``".tdz"``, ``memmap_like`` creates a single-file archive instead (see
:ref:`saving`). An archive cannot be sparse: ``memmap_like`` writes every
byte of zeros to disk.

Read and write
--------------

You read and write a tensordict on any backend with the usual operations.
On ``"h5"``, ``"zarr"`` and ``"redis"``, an index reads only the selected
rows. On ``"memmap"``, the operating system reads only the pages of those
rows from disk, and keeps the pages that you read often in memory:

  >>> batch = buffer[torch.tensor([0, 2, 3])]
  >>> batch["label"]
  tensor([0, 2, 3])

The ``"memmap"`` and ``"shared"`` backends return a locked tensordict. You can
write into its tensors in place, but you cannot add, remove or replace an
entry:

- ``td[i] = sample``, ``td.set_(key, value)`` and ``td.update_(other)`` write
  in place.
- ``td["new"] = value`` raises a ``RuntimeError``.
- To change a tensor, call an in-place method such as
  ``td["x"].add_(1)``. ``td["x"] += 1`` changes the tensor in place, then
  raises a ``RuntimeError``, because it also tries to replace the entry.

For example:

  >>> buffer["label"].add_(10)[:4]
  MemoryMappedTensor([10, 11, 12, 13])
  >>> buffer["extra"] = torch.zeros(1_000)
  Traceback (most recent call last):
  ...
  RuntimeError: Cannot modify locked TensorDict. ...

:class:`~tensordict.PersistentTensorDict` and
:class:`~tensordict.store.TensorDictStore` are not locked: you can add keys
to them.

To read files without changing them, open them read-only:
``TensorDict.load_memmap(path, mode="r")`` for memory-mapped files (see
:ref:`saving`), ``PersistentTensorDict.from_h5(path, mode="r")`` and
``PersistentTensorDict.from_zarr(path, mode="r")`` for HDF5 and zarr.

Share data between processes
----------------------------

A tensordict on the ``"memmap"`` or ``"shared"`` backend can go to other
processes on the same node, as an argument or through a queue. The process
that receives it uses the same files or the same shared memory. It sees the
writes of the other processes. :ref:`data-loading` shows how to do this
with ``torch.multiprocessing`` and a ``DataLoader``.

Processes on other nodes can open memory-mapped files if the files are on a
shared file system (see :ref:`distributed`). tensordict does not lock the
files or order the writes between processes. Give each writer its own range
of indices, and make the readers wait until the writers are done (for
example with :func:`torch.distributed.barrier`).

Without a shared file system, use a Redis server.

Share data between nodes with Redis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~tensordict.store.TensorDictStore` keeps each tensor as raw bytes on a
Redis-compatible server (Redis, Dragonfly or KeyDB). Processes on any node
that can connect to the server share one store. It needs the ``redis``
package (``pip install redis``).

.. warning::
   Connect only to a server that you trust. Non-tensor values that JSON
   cannot represent are stored with pickle, and each reader unpickles them.
   Any client that can write to the server can run code in the readers.

``from_schema`` preallocates the keys on the server with the ``SETRANGE``
command. The server fills them with zeros, and no tensor data goes through
the process. Give the store a ``td_id``, a name that the other processes use
to open it. Without one, the store gets a random UUID, which ``print(td)``
shows.

The examples of this section need a Redis server on ``localhost:6379``, so
they are not run when this page is tested:

  >>> td = TensorDict.from_schema(
  ...     {"obs": ([4], torch.float32), "reward": ([], torch.float32)},
  ...     batch_size=[1_000],
  ...     storage="redis",
  ...     host="localhost",
  ...     port=6379,
  ...     td_id="replay-v1",
  ... )  # doctest: +SKIP
  >>> td[0] = TensorDict(obs=torch.ones(4), reward=1.0)  # doctest: +SKIP

In another process, on any node, open the store with
:meth:`~tensordict.store.TensorDictStore.from_store`. It reads the batch size
and the keys from the server:

  >>> from tensordict.store import TensorDictStore
  >>> other = TensorDictStore.from_store(td_id="replay-v1", host="localhost", port=6379)  # doctest: +SKIP
  >>> other["obs"][0]  # doctest: +SKIP
  tensor([1., 1., 1., 1.])

A store caches its list of keys. Writes to existing keys are visible to all
the processes. A key that another process adds after you open the store is
not visible: reading it raises a ``KeyError``. To see new keys, open the store
again, or open it with
``TensorDictStore(td_id=..., batch_size=..., cache_metadata=False)``.
``from_store`` does not accept ``cache_metadata``.

See :class:`~tensordict.store.TensorDictStore` for the other arguments, and
:class:`~tensordict.store.LazyStackedTensorDictStore` for a stack of
tensordicts with different shapes.

.. _compat-redis-prealloc:

Preallocate on Redis and fill it with a typed wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A shared replay buffer on several nodes is a store that you preallocate on
the server and fill one sample at a time. A
:class:`~tensordict.TypedTensorDict` gives it a fixed set of typed fields (see
:ref:`typed-tensordict-guide`):

  >>> from torch import Tensor
  >>> from tensordict import TypedTensorDict
  >>> class Replay(TypedTensorDict):
  ...     obs: Tensor
  ...     action: Tensor
  ...     reward: Tensor
  >>> store = TensorDictStore.from_schema(
  ...     {"obs": ([84, 84, 3], torch.uint8),
  ...      "action": ([4], torch.float32),
  ...      "reward": ([], torch.float32)},
  ...     batch_size=[1_000],
  ...     host="localhost",
  ...     port=6379,
  ... )  # doctest: +SKIP
  >>> replay = Replay.from_tensordict(store)  # doctest: +SKIP
  >>> replay[0] = Replay(
  ...     obs=torch.zeros(84, 84, 3, dtype=torch.uint8),
  ...     action=torch.ones(4),
  ...     reward=torch.tensor(1.0),
  ...     batch_size=[],
  ... )  # doctest: +SKIP
  >>> replay.action[0]  # doctest: +SKIP
  tensor([1., 1., 1., 1.])

If the store has no keys yet, pass ``check=False`` to ``from_tensordict``.
The first indexed write then creates each key on the server, at the size of
the batch:

  >>> empty = TensorDictStore(batch_size=[1_000], host="localhost", port=6379)  # doctest: +SKIP
  >>> replay = Replay.from_tensordict(empty, check=False)  # doctest: +SKIP
  >>> replay[0] = Replay(
  ...     obs=torch.zeros(84, 84, 3, dtype=torch.uint8),
  ...     action=torch.ones(4),
  ...     reward=torch.tensor(1.0),
  ...     batch_size=[],
  ... )  # doctest: +SKIP
  >>> replay.obs.shape  # doctest: +SKIP
  torch.Size([1000, 84, 84, 3])

Non-tensor values
-----------------

Each backend stores non-tensor values, such as strings (see
:ref:`non-tensor-data`), in its own way:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Backend
     - Stored as
     - A string comes back as
   * - ``"memmap"``
     - JSON in ``meta.json``, or pickle in ``other.pickle`` (see
       :ref:`saving`)
     - ``str``
   * - ``"h5"``
     - an HDF5 string or opaque dataset
     - ``bytes``: call ``.decode()``
   * - ``"zarr"``
     - JSON or pickle in a ``uint8`` array
     - ``str``
   * - ``"redis"``
     - a JSON string or pickle bytes
     - ``str``

A locked memory-mapped tensordict cannot get a new entry, so add the
non-tensor values before you call ``memmap_``:

  >>> from tensordict import NonTensorData
  >>> td = TensorDict(
  ...     obs=torch.randn(4, 3),
  ...     label=NonTensorData("cat", batch_size=[4]),
  ...     batch_size=[4],
  ... )
  >>> _ = td.memmap_(root / "labels")
  >>> TensorDict.load_memmap(root / "labels")["label"]
  'cat'

HDF5 returns the string as ``bytes``:

  >>> td_h5 = PersistentTensorDict(filename=root / "labels.h5", mode="w", batch_size=[4])
  >>> td_h5["label"] = NonTensorData("cat", batch_size=[4])
  >>> td_h5["label"]
  b'cat'

On Redis, a value that JSON can represent works as on the other backends. A
value that needs pickle, such as a ``set`` or a ``datetime.date``, is written,
but reading it back raises a ``UnicodeDecodeError``. Store only strings,
numbers, lists and dicts on Redis.

Backend details
---------------

Memory-mapped files
~~~~~~~~~~~~~~~~~~~

The ``"memmap"`` backend stores each leaf in a ``.memmap`` file, through
:class:`~tensordict.MemoryMappedTensor`. The page cache of the operating
system keeps the parts that you read often in RAM, so the dataset can be much
larger than the memory. :ref:`saving` describes the directory layout, the
``meta.json`` files, single-file archives and the ``mode`` of
``load_memmap``.

HDF5
~~~~

:class:`~tensordict.PersistentTensorDict` stores each tensor as an HDF5
dataset, and each nested tensordict as an HDF5 group. Open an existing file
with :meth:`~tensordict.PersistentTensorDict.from_h5`, or write a tensordict
that is in memory with :meth:`~tensordict.PersistentTensorDict.from_dict`:

  >>> td_mem = TensorDict(obs=torch.randn(500, 4), batch_size=[500])
  >>> td_h5 = PersistentTensorDict.from_dict(td_mem, root / "data.h5")
  >>> PersistentTensorDict.from_h5(root / "data.h5")["obs"].shape
  torch.Size([500, 4])

zarr
~~~~

With ``backend="zarr"``, :class:`~tensordict.PersistentTensorDict` stores each
tensor as a zarr array and each nested tensordict as a zarr group. The store
can be a local directory, a ``zarr.storage.ZipStore`` or any other
``zarr.abc.store.Store``. Separate chunks can be read and written in
parallel. It needs ``zarr>=3.0`` (``pip install "tensordict[zarr]"``).

:meth:`~tensordict.TensorDictBase.to_zarr` writes a tensordict, and
:meth:`~tensordict.TensorDictBase.from_zarr` reads it back. ``to_zarr`` stores
the batch size and the dimension names in the attributes of the store, so
``from_zarr`` restores them. For a store written by another tool (xarray,
zarr), ``from_zarr`` infers the batch size:

  >>> td_zarr = td_mem.to_zarr(root / "data.zarr")
  >>> TensorDict.from_zarr(root / "data.zarr").batch_size
  torch.Size([500])

By default, ``to_zarr`` writes each tensor as one chunk without compression.
To read a few rows without reading the whole array, pass ``chunks``, and read
with :meth:`~tensordict.TensorDictBase.get_at`. To compress the chunks, pass
``compressors``. Both arguments go to :meth:`zarr.Group.create_array`:

  >>> from zarr.codecs import ZstdCodec
  >>> chunked = td_mem.to_zarr(root / "chunked.zarr", chunks=(1, 4), compressors=ZstdCodec())
  >>> chunked.get_at("obs", 3).shape
  torch.Size([4])

zarr writes one array, with its own metadata, for each leaf. With many small
leaves, it is slow: with 2000 leaves and 100 MB of data, saving took 17.6 s
with zarr and 0.33 s with memory-mapped files, and reading took 3.7 s and
0.19 s. Group small leaves into larger tensors, or use memory-mapped files.

Shared memory
~~~~~~~~~~~~~

The ``"shared"`` backend moves each tensor to shared memory (on Linux,
``/dev/shm``) with :meth:`~tensordict.TensorDictBase.share_memory_`.
Processes on the same node that receive the tensordict use the same memory,
without a copy. The data does not persist. ``from_schema`` allocates the full
size when it creates the tensordict:

  >>> td = TensorDict.from_schema({"obs": ([4], torch.float32)}, batch_size=[1_000], storage="shared")
  >>> td.is_shared(), td.is_locked
  (True, True)

Typed wrappers
--------------

:class:`~tensordict.TensorClass` and :class:`~tensordict.TypedTensorDict` can
wrap a tensordict on any of these backends with ``from_tensordict``. Writes
through the wrapper go to the backend. :ref:`compatibility` lists what works
with each backend. :ref:`compat-redis-prealloc` shows a typed replay buffer
on Redis.

See also
--------

- :ref:`saving`: memory-mapped directories, archives and their on-disk
  layout.
- :ref:`data-loading`: read these datasets with a ``DataLoader``, and share
  them between processes.
- :ref:`distributed`: send tensordicts between nodes.
- The reference of :meth:`~tensordict.TensorDictBase.from_schema`,
  :class:`~tensordict.PersistentTensorDict`,
  :class:`~tensordict.store.TensorDictStore` and
  :class:`~tensordict.MemoryMappedTensor`.
- Tutorials: :ref:`sphx_glr_tutorials_tensordict_memory.py` and
  :ref:`sphx_glr_tutorials_zarr_storage.py`.
