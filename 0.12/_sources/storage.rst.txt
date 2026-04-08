.. _storage:

Storing large heterogeneous data
================================

TensorDict can back its data by several storage backends so that large
datasets never need to live entirely in process memory.  This page explains
how to choose a backend, declare a schema up-front, read and write data,
store non-tensor values, and combine everything with typed wrappers
(:class:`~tensordict.TensorClass` and :class:`~tensordict.TypedTensorDict`).

.. contents:: On this page
   :local:
   :depth: 2


Quick overview
--------------

.. list-table::
   :header-rows: 1
   :widths: 18 20 20 15 15

   * - Backend
     - ``storage=``
     - Result type
     - Persistence
     - Multi-process
   * - Regular tensors
     - ``None``
     - :class:`~tensordict.TensorDict`
     - No
     - No
   * - Memory-mapped
     - ``"memmap"``
     - :class:`~tensordict.TensorDict`
     - On disk
     - Yes (NFS)
   * - HDF5
     - ``"h5"``
     - :class:`~tensordict.PersistentTensorDict`
     - On disk
     - Limited
   * - Shared memory
     - ``"shared"``
     - :class:`~tensordict.TensorDict`
     - No
     - Yes (same node)
   * - Redis / Dragonfly
     - ``"redis"``
     - :class:`~tensordict.store.TensorDictStore`
     - Server
     - Yes (network)


Declaring a schema with ``from_schema``
---------------------------------------

:meth:`~tensordict.TensorDictBase.from_schema` creates a pre-allocated,
zero-filled :class:`~tensordict.TensorDictBase` from a dictionary that maps
field names to ``(element_shape, dtype)`` pairs.  The ``storage`` keyword
selects which backend is used.

.. code-block:: python

   >>> import torch
   >>> from tensordict import TensorDict

   >>> schema = {
   ...     "obs": ([84, 84, 3], torch.uint8),
   ...     "action": ([4], torch.float32),
   ...     "reward": ([], torch.float32),
   ... }

   >>> td = TensorDict.from_schema(schema, batch_size=[100_000])
   >>> td["obs"].shape
   torch.Size([100000, 84, 84, 3])

Each element shape is prepended by ``batch_size``, so a scalar reward with
``batch_size=[N]`` yields a tensor of shape ``(N,)``.

The ``storage`` keyword selects the backend:

.. code-block:: python

   >>> # Memory-mapped tensors on disk
   >>> td = TensorDict.from_schema(schema, batch_size=[100_000],
   ...                             storage="memmap", prefix="/data/replay")

   >>> # HDF5 file
   >>> td = TensorDict.from_schema(schema, batch_size=[100_000],
   ...                             storage="h5", filename="/data/replay.h5")

   >>> # Shared memory (single-node multi-process)
   >>> td = TensorDict.from_schema(schema, batch_size=[100_000],
   ...                             storage="shared")

   >>> # Redis server (multi-node)
   >>> td = TensorDict.from_schema(schema, batch_size=[100_000],
   ...                             storage="redis", host="redis-node")

Extra keyword arguments are forwarded to the backend constructor.  See each
backend section below for details.


Backend details
---------------

Memory-mapped tensors (``storage="memmap"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory-mapped tensors live in per-key ``.memmap`` files on disk, accessed via
:class:`~tensordict.MemoryMappedTensor`.  The OS page cache keeps frequently
accessed regions in RAM while allowing the dataset to far exceed physical
memory.

**Pre-allocating from a schema** uses the expand trick internally -- each
value starts as ``torch.zeros(()).expand(shape)``, which allocates no memory,
and :meth:`~tensordict.TensorDictBase.memmap_like` then creates the on-disk
files.

.. code-block:: python

   >>> import torch, tempfile
   >>> from tensordict import TensorDict

   >>> with tempfile.TemporaryDirectory() as d:
   ...     td = TensorDict.from_schema(
   ...         {"obs": ([4], torch.float32), "reward": ([], torch.float32)},
   ...         batch_size=[1_000],
   ...         storage="memmap",
   ...         prefix=d,
   ...     )
   ...     assert td.is_memmap()
   ...     # Fill iteratively -- each write goes directly to disk
   ...     td[0] = TensorDict(obs=torch.randn(4), reward=torch.tensor(1.0), batch_size=[])
   ...     assert (td[0]["reward"] == 1.0).all()

Keyword arguments:

- ``prefix`` -- directory where ``.memmap`` files are stored.

.. note::

   Memory-mapped TensorDicts are locked after creation.  Use
   :meth:`~tensordict.TensorDictBase.set_` and
   :meth:`~tensordict.TensorDictBase.update_` for in-place writes, or index
   assignment (``td[i] = ...``) which is always in-place.

For more details on the memory-mapped API (``memmap_``, ``memmap_like``,
``load_memmap``, directory layout, ``meta.json``), see :ref:`saving`.


HDF5 (``storage="h5"``)
~~~~~~~~~~~~~~~~~~~~~~~

HDF5-backed storage is provided by :class:`~tensordict.PersistentTensorDict`.
Each tensor becomes an HDF5 dataset; nested keys become HDF5 groups.  This is
useful for datasets that must be portable and inspectable with standard tools
like ``h5py`` or ``HDFView``.

.. code-block:: python

   >>> import torch, tempfile
   >>> from tensordict import TensorDict

   >>> with tempfile.NamedTemporaryFile(suffix=".h5") as f:
   ...     td = TensorDict.from_schema(
   ...         {"obs": ([4], torch.float32), "label": ([], torch.int64)},
   ...         batch_size=[500],
   ...         storage="h5",
   ...         filename=f.name,
   ...     )
   ...     td[0] = TensorDict(obs=torch.randn(4), label=torch.tensor(0), batch_size=[])

You can also load an existing file:

.. code-block:: python

   >>> from tensordict import PersistentTensorDict
   >>> td = PersistentTensorDict.from_h5("data.h5")

Or convert an in-memory TensorDict:

.. code-block:: python

   >>> td_mem = TensorDict(obs=torch.randn(500, 4), batch_size=[500])
   >>> td_h5 = PersistentTensorDict.from_dict(td_mem, "data.h5")

Keyword arguments forwarded by ``from_schema``:

- ``filename`` (required) -- path to the HDF5 file.


Shared memory (``storage="shared"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared-memory tensors allow zero-copy access across processes on the same
machine.  This is the fastest option for single-node multi-process setups
(e.g. multi-worker dataloading).

.. code-block:: python

   >>> import torch
   >>> from tensordict import TensorDict

   >>> td = TensorDict.from_schema(
   ...     {"obs": ([4], torch.float32)},
   ...     batch_size=[1000],
   ...     storage="shared",
   ... )
   >>> assert td.is_shared()

No additional keyword arguments are required.

.. note::

   Shared-memory TensorDicts are locked.  Use in-place operations for writes
   (``set_()``, ``update_()``, index assignment).


Redis / Dragonfly (``storage="redis"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~tensordict.store.TensorDictStore` stores tensors as raw bytes on a
Redis-compatible server (Redis, Dragonfly, KeyDB).  This enables cross-node
shared data stores and replay buffers without a shared file system.

``from_schema`` on the base class delegates to
:meth:`~tensordict.store.TensorDictStore.from_schema`, which uses
server-side ``SETRANGE`` to pre-allocate storage without sending tensor data
through Python:

.. code-block:: python

   >>> import torch
   >>> from tensordict import TensorDict

   >>> td = TensorDict.from_schema(
   ...     {"obs": ([84, 84, 3], torch.uint8),
   ...      "action": ([4], torch.float32),
   ...      "reward": ([], torch.float32)},
   ...     batch_size=[100_000],
   ...     storage="redis",
   ...     host="redis-node",
   ... )

Keyword arguments:

- ``host`` -- server hostname (default ``"localhost"``).
- ``port`` -- server port (default ``6379``).
- ``db`` -- database number (default ``0``).
- ``unix_socket_path`` -- Unix domain socket (alternative to host/port).
- ``prefix`` -- key namespace (default ``"tensordict"``).

You can also connect to an existing store from another process:

.. code-block:: python

   >>> from tensordict.store import TensorDictStore
   >>> td = TensorDictStore.from_store(td_id="<id>", host="redis-node")


Pre-allocation patterns
-----------------------

Pre-allocating large storage and filling it iteratively avoids allocating the
full dataset in process memory.  All backends support the same pattern via
``from_schema``:

.. code-block:: python

   >>> import torch
   >>> from tensordict import TensorDict

   >>> schema = {
   ...     "image": ([3, 64, 64], torch.uint8),
   ...     "label": ([], torch.int64),
   ... }
   >>> buffer = TensorDict.from_schema(
   ...     schema, batch_size=[1_000_000], storage="memmap", prefix="/data/buffer"
   ... )
   >>> for i, sample in enumerate(data_stream):  # doctest: +SKIP
   ...     buffer[i] = TensorDict(
   ...         image=sample["image"], label=sample["label"], batch_size=[]
   ...     )

The expand trick used internally ensures that no temporary allocation happens
regardless of dataset size.

For memory-mapped storage you can also pre-allocate manually using
:meth:`~tensordict.TensorDictBase.memmap_like`:

.. code-block:: python

   >>> datum = TensorDict(image=torch.zeros(3, 64, 64, dtype=torch.uint8),
   ...                    label=torch.tensor(0), batch_size=[])
   >>> buffer = datum.expand(1_000_000).memmap_like("/data/buffer")


Non-tensor data
---------------

Each backend stores non-tensor values (strings, Python objects) using its
own mechanism:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Backend
     - Serialisation
     - Access pattern
   * - memmap
     - JSON in ``meta.json``; pickle fallback (``other.pickle``)
     - :class:`~tensordict.NonTensorData` wrapper
   * - HDF5
     - HDF5 string/opaque datasets
     - :class:`~tensordict.NonTensorData` wrapper on read
   * - Redis
     - JSON string or pickle bytes in Redis ``SET``
     - Transparent via metadata hash

For **memmap**, non-tensor data is serialised via tensorclass's
:class:`~tensordict.NonTensorData`:

.. code-block:: python

   >>> from tensordict import TensorDict, NonTensorData
   >>> td = TensorDict(
   ...     obs=torch.randn(4, 3),
   ...     label=NonTensorData(data="cat", batch_size=[4]),
   ...     batch_size=[4],
   ... )
   >>> td_mm = td.memmap_("/tmp/example")
   >>> loaded = TensorDict.load_memmap("/tmp/example")  # doctest: +SKIP
   >>> loaded["label"].data  # doctest: +SKIP
   'cat'

For **HDF5**, non-tensor values are stored as HDF5 string or opaque datasets:

.. code-block:: python

   >>> from tensordict import PersistentTensorDict
   >>> td_h5 = PersistentTensorDict(filename="data.h5", mode="w", batch_size=[4])  # doctest: +SKIP
   >>> td_h5["label"] = NonTensorData(data="cat", batch_size=[4])  # doctest: +SKIP

For **Redis**, non-tensor data is transparently serialised as JSON (falling
back to pickle for non-JSON-serialisable objects):

.. code-block:: python

   >>> from tensordict.store import TensorDictStore  # doctest: +SKIP
   >>> store = TensorDictStore(batch_size=[4])  # doctest: +SKIP
   >>> store["label"] = NonTensorData(data="cat", batch_size=[4])  # doctest: +SKIP


Typed wrappers
--------------

Both :class:`~tensordict.TensorClass` and :class:`~tensordict.TypedTensorDict`
can wrap any backend via ``from_tensordict``.  Combined with ``from_schema``,
this gives you typed, pre-allocated, backend-agnostic data stores.

Using TypedTensorDict
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   >>> import torch
   >>> from tensordict import TensorDict, TypedTensorDict
   >>> from torch import Tensor

   >>> class Replay(TypedTensorDict):
   ...     obs: Tensor
   ...     action: Tensor
   ...     reward: Tensor

   >>> # Pre-allocate with any backend, then wrap
   >>> store = TensorDict.from_schema(
   ...     {"obs": ([4], torch.float32),
   ...      "action": ([2], torch.float32),
   ...      "reward": ([], torch.float32)},
   ...     batch_size=[10_000],
   ...     storage="memmap",
   ...     prefix="/data/replay",
   ... )
   >>> replay = Replay.from_tensordict(store)
   >>> replay.obs.shape
   torch.Size([10000, 4])

   >>> # Fill iteratively with full type-safety
   >>> replay[0] = Replay(
   ...     obs=torch.randn(4), action=torch.randn(2), reward=torch.tensor(1.0),
   ...     batch_size=[],
   ... )

Using TensorClass
~~~~~~~~~~~~~~~~~

.. code-block:: python

   >>> import torch
   >>> from tensordict import TensorDict, TensorClass
   >>> from torch import Tensor

   >>> class Transition(TensorClass):
   ...     obs: Tensor
   ...     action: Tensor
   ...     reward: float  # non-tensor field

   >>> store = TensorDict.from_schema(
   ...     {"obs": ([4], torch.float32),
   ...      "action": ([2], torch.float32)},
   ...     batch_size=[10_000],
   ...     storage="shared",
   ... )
   >>> tc = Transition.from_tensordict(store, non_tensordict={"reward": 0.0})  # doctest: +SKIP

Unlike :class:`~tensordict.TypedTensorDict`, :class:`~tensordict.TensorClass`
supports non-tensor fields (strings, numbers, arbitrary Python objects).
See :ref:`the compatibility page <compat-redis-prealloc>` for the full
backend-support matrix.


Choosing a backend
------------------

- **Memory-mapped** -- best for large on-disk datasets where you want
  memory-efficient random access (replay buffers, offline RL, large
  datasets that exceed RAM).  Works across processes via NFS.
- **HDF5** -- best when you need a portable, self-describing file format
  inspectable with standard tools.  Good for archival.
- **Shared memory** -- best for single-node multi-process workloads
  (multi-worker dataloading, parallel envs).  Fastest IPC but data does not
  persist.
- **Redis** -- best for multi-node shared data stores (distributed replay
  buffers, parameter servers).  Requires a running server.
- **Plain tensors** -- best for small datasets that fit in memory.  No
  overhead, full PyTorch API.
