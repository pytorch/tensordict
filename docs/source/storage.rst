.. _storage:

Large datasets and storage backends
===================================

To work with a dataset that is larger than memory, or that several processes
share, create the tensordict on a storage backend.
:meth:`~tensordict.TensorDictBase.from_schema` preallocates it on any
backend, and you then read and write it like a tensordict in memory. This
example preallocates 1,000 steps of a robot log on disk, and writes the
first four:

  >>> import tempfile
  >>> from pathlib import Path
  >>> import torch
  >>> from tensordict import TensorDict
  >>> tmp = tempfile.TemporaryDirectory()
  >>> root = Path(tmp.name)
  >>> schema = {
  ...     "camera": ([3, 64, 64], torch.uint8),
  ...     "joints": ([7], torch.float32),
  ...     "step": ([], torch.int64),
  ... }
  >>> log = TensorDict.from_schema(
  ...     schema, batch_size=[1_000], storage="memmap", prefix=root / "robot_log"
  ... )
  >>> for i in range(4):
  ...     log[i] = TensorDict(
  ...         {
  ...             "camera": torch.full((3, 64, 64), i, dtype=torch.uint8),
  ...             "joints": torch.zeros(7),
  ...             "step": i,
  ...         },
  ...     )
  >>> log["step"][:5]
  MemoryMappedTensor([0, 1, 2, 3, 0])

The examples on this page run in order, and share the temporary directory
``root``. To save a tensordict that is already in memory, see :ref:`saving`.

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
     - a directory, a zip file, or an object store
     - You need chunks, compression, or an object store, or you read the
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
     - the memory of a Redis, Dragonfly, or KeyDB server
     - Processes on several nodes share data, without a shared file system.

Only the memory-mapped files, HDF5, and zarr keep the data after the
processes end. A Redis server keeps it while the server runs.

Declare a schema with from_schema
---------------------------------

The first argument of :meth:`~tensordict.TensorDictBase.from_schema` maps
each key to the shape of one element and a dtype. ``from_schema`` puts
``batch_size`` before each element shape, so the scalar ``"step"`` with
``batch_size=[N]`` is a tensor of shape ``(N,)``. Without ``storage``, the
tensors are zeros in memory:

  >>> td = TensorDict.from_schema(schema, batch_size=[100])
  >>> td["camera"].shape, td["step"].shape
  (torch.Size([100, 3, 64, 64]), torch.Size([100]))

The ``storage`` keyword selects the backend. The other keywords go to the
backend:

  >>> on_disk = TensorDict.from_schema(
  ...     schema, batch_size=[100], storage="memmap", prefix=root / "memmap_log"
  ... )
  >>> in_h5 = TensorDict.from_schema(
  ...     schema, batch_size=[100], storage="h5", filename=root / "log.h5"
  ... )
  >>> in_zarr = TensorDict.from_schema(
  ...     schema, batch_size=[100], storage="zarr", filename=root / "log.zarr"
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
  ``unix_socket_path``, ``prefix`` (the namespace of the keys on the server),
  and ``td_id`` (see `Share data between nodes with Redis`_).

Preallocate a dataset that is larger than memory
------------------------------------------------

``from_schema`` builds each value as ``torch.zeros(()).expand(shape)``, a
view that uses no memory. The memory that the backend then uses depends on
the backend. These numbers come from one measurement with a 256 MiB
``uint8`` schema:

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
is larger than RAM, use ``"memmap"`` or ``"redis"``.

To get an HDF5 file without this cost, create the datasets with ``h5py``,
which writes no data. Then open the file with
:meth:`~tensordict.PersistentTensorDict.from_h5`. For zarr, create the arrays
with ``zarr``, and open the store with
:meth:`~tensordict.PersistentTensorDict.from_zarr`:

  >>> import h5py
  >>> from tensordict import PersistentTensorDict
  >>> with h5py.File(root / "large.h5", "w") as f:
  ...     _ = f.create_dataset("image", shape=(1_000, 3, 64, 64), dtype="uint8")
  >>> large = PersistentTensorDict.from_h5(root / "large.h5", mode="r+", batch_size=[1_000])
  >>> large[0] = TensorDict({"image": torch.ones(3, 64, 64, dtype=torch.uint8)})
  >>> large["image"][0].sum()
  tensor(12288)

For memory-mapped files, :meth:`~tensordict.TensorDictBase.memmap_like` is
another way to preallocate. It takes a tensordict with the shapes and dtypes
of the dataset, and creates files with the same structure, without copying
the values. New files are filled with zeros. A file that already exists at
the path of a leaf is mapped as it is and keeps its content, so calling
``memmap_like`` again on the same directory reopens the buffer. Make one
sample, and expand it, which does not allocate memory:

  >>> datum = TensorDict({"image": torch.zeros(3, 64, 64, dtype=torch.uint8), "label": 0})
  >>> dataset = datum.expand(1_000).memmap_like(root / "dataset")
  >>> dataset.batch_size
  torch.Size([1000])

Make sure that the sample has the dtype that you want.
``torch.randint(255, shape)`` gives ``int64``, which takes 8 times the space
of ``uint8``. If the path ends with ``".tdz"``, ``memmap_like`` creates a
single-file archive instead (see :ref:`saving`). An archive cannot be
sparse: ``memmap_like`` writes every byte of zeros to disk.

Read and write
--------------

You read and write a tensordict on any backend with the usual operations.
On ``"h5"``, ``"zarr"``, and ``"redis"``, an index reads only the selected
rows. On ``"memmap"``, the operating system reads only the pages of those
rows from disk, and keeps the pages that you read often in memory:

  >>> batch = log[torch.tensor([0, 2, 3])]
  >>> batch["step"]
  tensor([0, 2, 3])

The ``"memmap"`` and ``"shared"`` backends return a locked tensordict (see
:ref:`keys-locking`). Write into its tensors in place, with ``td[i] = sample``,
``set_``, ``update_``, or an in-place method such as ``td["x"].add_(1)``. Do
not use ``td["x"] += 1``: it changes the tensor, then raises a
``RuntimeError``. To add an entry, add it before you create the files:

  >>> log["step"].add_(10)[:4]
  MemoryMappedTensor([10, 11, 12, 13])
  >>> log["extra"] = torch.zeros(1_000)
  Traceback (most recent call last):
  ...
  RuntimeError: Cannot modify locked TensorDict. ...

:class:`~tensordict.PersistentTensorDict` and
:class:`~tensordict.store.TensorDictStore` are not locked, so you can add
keys to them.

To read files without changing them, open them read-only:
``TensorDict.load_memmap(path, mode="r")`` for memory-mapped files (see
:ref:`saving`), and ``PersistentTensorDict.from_h5(path, mode="r")`` or
``PersistentTensorDict.from_zarr(path, mode="r")`` for HDF5 and zarr.

.. _storage-sharing:

Share data between processes
----------------------------

A tensordict on the ``"memmap"`` or ``"shared"`` backend can go to other
processes on the same node, as an argument or through a queue. The process
that receives it uses the same files or the same shared memory, and sees the
writes of the other processes. :ref:`data-loading` shows how, with
``torch.multiprocessing`` and a ``DataLoader``. Processes on other nodes can
open memory-mapped files on a shared file system (see :ref:`distributed`).
Without a shared file system, use a Redis server.

tensordict does not lock the files or order the writes between processes:

- Give each writer its own range of indices.
- Make the readers wait until the writers are done, for example with
  :func:`torch.distributed.barrier`.

.. _storage-redis:

Share data between nodes with Redis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~tensordict.store.TensorDictStore` keeps each tensor as raw bytes on a
Redis-compatible server (Redis, Dragonfly, or KeyDB). Processes on any node
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
they are not run when this page is tested. In this example, 64 simulations
on several nodes write their results into one store:

  >>> results = TensorDict.from_schema(
  ...     {"pos": ([100, 3], torch.float32), "energy": ([], torch.float32)},
  ...     batch_size=[64],
  ...     storage="redis",
  ...     host="localhost",
  ...     port=6379,
  ...     td_id="sweep-1",
  ... )  # doctest: +SKIP
  >>> results[0] = TensorDict({"pos": torch.ones(100, 3), "energy": -1.5})  # doctest: +SKIP

In another process, on any node, open the store with
:meth:`~tensordict.store.TensorDictStore.from_store`. It reads the batch size
and the keys from the server:

  >>> from tensordict.store import TensorDictStore
  >>> other = TensorDictStore.from_store(td_id="sweep-1", host="localhost", port=6379)  # doctest: +SKIP
  >>> other["energy"][:2]  # doctest: +SKIP
  tensor([-1.5000,  0.0000])

A store caches its list of keys. Writes to existing keys are visible to all
the processes. A key that another process adds after you open the store is
not visible: reading it raises a ``KeyError``. To see new keys, open the store
again, or pass ``cache_metadata=False`` to ``from_store``. The store then
reads the list of keys from the server each time.

See :class:`~tensordict.store.TensorDictStore` for the other arguments, and
:class:`~tensordict.store.LazyStackedTensorDictStore` for a stack of
tensordicts with different shapes. ``LazyStackedTensorDictStore`` is
experimental: its API can change in a minor release without a deprecation
cycle.

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
     - ``bytes``: call ``.decode()``. Starting with TensorDict 0.17, the
       default will change to ``str`` for the strings that tensordict wrote;
       reading one as ``bytes`` warns until then
       (`#2255 <https://github.com/pytorch/tensordict/issues/2255>`_)
   * - ``"zarr"``
     - JSON or pickle in a ``uint8`` array
     - ``str``
   * - ``"redis"``
     - a JSON string or pickle bytes
     - ``str``

Add the non-tensor values before you call ``memmap_``. A locked
memory-mapped tensordict cannot get a new entry:

  >>> from tensordict import NonTensorData
  >>> td = TensorDict(
  ...     {
  ...         "features": torch.randn(4, 3),
  ...         "label": NonTensorData("cat", batch_size=[4]),
  ...     },
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

On Redis, a value that JSON can represent goes to the server as JSON. A
value that needs pickle, such as a ``set`` or a ``datetime.date``, goes to the
server as pickle bytes, and reads back as the same value. Each read
unpickles these bytes: see the warning in :ref:`storage-redis`.

Backend details
---------------

Memory-mapped files
~~~~~~~~~~~~~~~~~~~

The ``"memmap"`` backend stores each leaf in a ``.memmap`` file, through
:class:`~tensordict.MemoryMappedTensor`. :ref:`saving` describes the
directory layout, the ``meta.json`` files, single-file archives, and the
``mode`` of ``load_memmap``.

HDF5
~~~~

:class:`~tensordict.PersistentTensorDict` stores each tensor as an HDF5
dataset, and each nested tensordict as an HDF5 group. Open an existing file
with :meth:`~tensordict.PersistentTensorDict.from_h5`, or write a tensordict
that is in memory with :meth:`~tensordict.PersistentTensorDict.from_dict`.
This example writes 500 readings of 4 sensors:

  >>> td_mem = TensorDict({"sensors": torch.randn(500, 4)}, batch_size=[500])
  >>> td_h5 = PersistentTensorDict.from_dict(td_mem, root / "data.h5")
  >>> PersistentTensorDict.from_h5(root / "data.h5")["sensors"].shape
  torch.Size([500, 4])

.. _storage-zarr:

zarr
~~~~

With ``backend="zarr"``, :class:`~tensordict.PersistentTensorDict` stores each
tensor as a zarr array and each nested tensordict as a zarr group. The store
can be a local directory, a ``zarr.storage.ZipStore``, or any other
``zarr.abc.store.Store``. You can read and write separate chunks in parallel.
The zarr backend needs ``zarr>=3.1.1`` (``pip install "tensordict[zarr]"``).

:meth:`~tensordict.TensorDictBase.to_zarr` writes a tensordict, and
:meth:`~tensordict.TensorDictBase.from_zarr` reads it back. ``to_zarr`` stores
the batch size and the dimension names in the attributes of the store, so
``from_zarr`` restores them. For a store written by another tool (xarray,
zarr), ``from_zarr`` infers the batch size:

  >>> td_zarr = td_mem.to_zarr(root / "data.zarr")
  >>> TensorDict.from_zarr(root / "data.zarr").batch_size
  torch.Size([500])

Non-tensor values use JSON where possible and pickle otherwise. Pass
``allow_pickle=False`` to ``from_zarr`` (or to ``PersistentTensorDict``) to
reject pickled values before decoding them, or ``allow_pickle=True`` only
for data you trust. The current default, ``None``, still reads pickle but
emits a ``FutureWarning`` when a pickled field is read, not when the store
is opened. Starting with TensorDict 0.17, the default will change to
``allow_pickle=False``. Tensor and JSON entries are unaffected, as is HDF5.
Unknown non-tensor encodings raise ``ValueError``.

``to_zarr`` and ``PersistentTensorDict.from_dict(..., backend="zarr")``
create a new store and return a handle that trusts the values they just
wrote, unless you pass ``allow_pickle=False``. This policy is not stored in
the zarr store: reopening with ``from_zarr`` uses that call's policy.
Opening an existing store in append mode does not opt into pickle.

By default, ``to_zarr`` writes each tensor as one chunk without compression.
To read a few rows without reading the whole array, pass ``chunks``, and read
with :meth:`~tensordict.TensorDictBase.get_at`. To compress the chunks, pass
``compressors``. Both arguments go to :meth:`zarr.Group.create_array`:

  >>> from zarr.codecs import ZstdCodec
  >>> chunked = td_mem.to_zarr(root / "chunked.zarr", chunks=(1, 4), compressors=ZstdCodec())
  >>> chunked.get_at("sensors", 3).shape
  torch.Size([4])

zarr opens a store lazily, but it does not memory-map the arrays, so each
read copies the data. It writes one array, with its own metadata, for each
leaf, and each leaf costs one round trip to the store. In one measurement
with 2000 leaves and 100 MB of data, saving took 17.6 s with zarr and 0.33 s
with memory-mapped files, and reading took 3.7 s and 0.19 s. If you have
many small leaves, group them into larger tensors, or use memory-mapped
files.

Shared memory
~~~~~~~~~~~~~

The ``"shared"`` backend moves each tensor to shared memory (on Linux,
``/dev/shm``) with :meth:`~tensordict.TensorDictBase.share_memory_`.
Processes on the same node that receive the tensordict use the same memory,
without a copy. The data does not persist. ``from_schema`` allocates the full
size when it creates the tensordict:

  >>> td = TensorDict.from_schema({"state": ([4], torch.float32)}, batch_size=[1_000], storage="shared")
  >>> td.is_shared(), td.is_locked
  (True, True)

Arrow tables
------------

:func:`~tensordict.from_arrow` converts an Arrow table to a tensordict, and
:func:`~tensordict.to_arrow` converts a tensordict back. Numeric columns
become tensors that view the Arrow memory, list columns become jagged nested
tensors, struct columns become nested tensordicts, and strings become a
:class:`~tensordict.NonTensorStack`:

  >>> import pyarrow as pa
  >>> from tensordict import from_arrow, to_arrow
  >>> table = pa.table(
  ...     {
  ...         "reward": pa.array([0.5, 1.0, 0.0]),
  ...         "tokens": pa.array([[1, 2], [3], [4, 5, 6]]),
  ...         "prompt": pa.array(["a", "b", "c"]),
  ...     }
  ... )
  >>> td = from_arrow(table)
  >>> td["tokens"].unbind(0)[2]
  tensor([4, 5, 6])
  >>> to_arrow(td).column_names
  ['reward', 'tokens', 'prompt']

Hugging Face ``datasets`` and Polars hold their data in Arrow tables. After
``select``, ``shuffle`` or ``filter``, a Hugging Face dataset keeps all the
rows of its table and a list of the rows it selects: call
``flatten_indices()`` first, so that its table holds only those rows:

  >>> import datasets  # doctest: +SKIP
  >>> ds = datasets.load_from_disk("my_dataset")  # doctest: +SKIP
  >>> ds = ds.shuffle(seed=0).flatten_indices()  # doctest: +SKIP
  >>> td = from_arrow(ds.data.table)  # doctest: +SKIP
  >>> import polars as pl  # doctest: +SKIP
  >>> td = from_arrow(pl.DataFrame({"x": [1.0, 2.0]}).to_arrow())  # doctest: +SKIP

A dataset loaded from disk is memory-mapped, and its memory is read-only.
``from_arrow`` copies such columns, because an in-place write into a view of
read-only memory crashes the process. ``copy=False`` views them anyway: then
do not write into the tensors. A column with null values raises an error,
unless you pass ``nulls="fill"`` or ``nulls="mask"``; the latter adds a
``"_mask"`` entry (or the ``mask_key`` you pass), which differs from the
``"masks"`` entry of :func:`~tensordict.pad_sequence` with ``return_mask=True``
so that the two can be combined. A tensordict with
jagged tensors can be indexed with an integer, a slice, an index tensor or a
boolean mask, and split, chunked, unbound and concatenated along its batch
dimension, but not stacked. ``memmap`` does not take jagged tensors: pad them
first, with :func:`torch.nested.to_padded_tensor`.
:meth:`~tensordict.TensorDict.from_parquet` keeps its own conversion, which
reads list and struct columns as Python objects. To convert a Parquet file
as ``from_arrow`` does, read it with ``pyarrow.parquet.read_table``.
With the default ``nulls="raise"``, ``from_arrow`` returns a null inside a
map, a list view or the dictionary values that a dictionary-encoded column
selects as ``None``, with a ``FutureWarning``: starting with TensorDict 0.18,
the default will change to raising an error for these nulls too. Nulls inside
a union or a run-end encoded column are not checked.

Typed wrappers
--------------

:class:`~tensordict.TensorClass` and :class:`~tensordict.TypedTensorDict` can
wrap a tensordict on any of these backends with ``from_tensordict``. Writes
through the wrapper go to the backend. :ref:`compatibility` lists what works
with each backend. :ref:`compat-redis-prealloc` shows a typed replay buffer
on Redis.

See also
--------

- :ref:`saving`: memory-mapped directories, archives, and their on-disk
  layout.
- :ref:`data-loading`: read these datasets with a ``DataLoader``, and share
  them between processes.
- :ref:`distributed`: send tensordicts between nodes.
- The reference of :meth:`~tensordict.TensorDictBase.from_schema`,
  :class:`~tensordict.PersistentTensorDict`,
  :class:`~tensordict.store.TensorDictStore`, and
  :class:`~tensordict.MemoryMappedTensor`.
- Tutorials: :ref:`sphx_glr_tutorials_tensordict_memory.py` and
  :ref:`sphx_glr_tutorials_zarr_storage.py`.
