.. currentmodule:: tensordict

Cross-class compatibility
=========================

``TensorDict`` has several typed container wrappers and storage backends that
can be composed together.  This page documents which combinations work, which
have caveats, and how to build these "chimera" objects.

Architecture overview
---------------------

.. code-block:: text

   TensorCollection
   ├── TensorDictBase
   │   ├── TensorDict                 (in-memory)
   │   ├── TypedTensorDict            (typed fields, wraps any TensorDictBase)
   │   ├── PersistentTensorDict       (HDF5-backed)
   │   ├── TensorDictStore            (Redis / Dragonfly / KeyDB)
   │   └── LazyStackedTensorDict      (lazy stack of heterogeneous TDs)
   │
   └── TensorClass                    (typed wrapper, HAS-A TensorDictBase)

Two patterns exist for adding typed field declarations:

- **TensorClass** wraps any ``TensorDictBase`` via ``from_tensordict(td)``.
  It delegates all storage to the wrapped object.
- **TypedTensorDict** wraps any ``TensorDictBase`` via ``from_tensordict(td)``,
  similar to ``TensorClass``.  Direct construction creates a ``TensorDict``
  internally.  Unlike ``TensorClass``, it inherits from ``TensorDictBase``
  directly, supports ``**state`` spreading natively, and uses standard
  Python inheritance for schema composition.

TensorClass + backends
----------------------

``TensorClass.from_tensordict(td)`` accepts any ``TensorDictBase`` subclass.
The table below summarises which operations work on each combination.

.. code-block:: python

   from tensordict import TensorClass
   from torch import Tensor

   class MyTC(TensorClass):
       a: Tensor
       b: Tensor

   tc = MyTC.from_tensordict(some_backend)

.. list-table::
   :header-rows: 1
   :widths: 22 10 10 10 10 10 10 10 10

   * - Backend
     - Build
     - Read
     - Write
     - Index
     - Clone
     - Stack
     - Iter
     - Update
   * - ``TensorDict``
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
   * - ``PersistentTensorDict`` (H5)
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
   * - ``TensorDictStore`` (Redis)
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
   * - ``LazyStackedTensorDict``
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
   * - ``TensorDict`` (memmap)
     - yes
     - yes
     - set\_()
     - yes
     - yes
     - yes
     - yes
     - update\_()
   * - ``TypedTensorDict``
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes

.. note::

   Memory-mapped TensorDicts are locked after ``memmap_()``.  Use
   ``set_()`` and ``update_()`` for in-place writes instead of attribute
   assignment or ``update()``.

Building a TensorClass on each backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**In-memory TensorDict** -- the default and simplest case:

.. code-block:: python

   >>> import torch
   >>> from tensordict import TensorDict, TensorClass
   >>> from torch import Tensor
   >>>
   >>> class MyTC(TensorClass):
   ...     a: Tensor
   ...     b: Tensor
   >>>
   >>> td = TensorDict(a=torch.randn(4, 3), b=torch.randn(4, 5), batch_size=[4])
   >>> tc = MyTC.from_tensordict(td)
   >>> tc.a.shape
   torch.Size([4, 3])

**HDF5 (PersistentTensorDict)**:

.. code-block:: python

   >>> from tensordict import PersistentTensorDict
   >>>
   >>> h5 = PersistentTensorDict.from_dict(td, filename="data.h5")
   >>> tc_h5 = MyTC.from_tensordict(h5)
   >>> tc_h5.a.shape  # reads from HDF5
   torch.Size([4, 3])

**Redis (TensorDictStore)**:

.. code-block:: python

   >>> from tensordict.store import TensorDictStore
   >>>
   >>> store = TensorDictStore.from_tensordict(td, host="localhost")
   >>> tc_redis = MyTC.from_tensordict(store)
   >>> tc_redis.a.shape  # fetched from Redis
   torch.Size([4, 3])

**Lazy stack**:

.. code-block:: python

   >>> from tensordict import lazy_stack
   >>>
   >>> tds = [TensorDict(a=torch.randn(3), b=torch.randn(5)) for _ in range(4)]
   >>> ls = lazy_stack(tds, dim=0)
   >>> tc_lazy = MyTC.from_tensordict(ls)
   >>> tc_lazy[0].a.shape
   torch.Size([3])

**Memory-mapped TensorDict**:

.. code-block:: python

   >>> td_mmap = td.memmap_("/tmp/my_memmap")
   >>> tc_mmap = MyTC.from_tensordict(td_mmap)
   >>> tc_mmap.a.shape
   torch.Size([4, 3])
   >>> # memmap TDs are locked -- use in-place operations:
   >>> tc_mmap.set_("a", torch.ones(4, 3))

**TypedTensorDict as backend** -- both TensorClass and TypedTensorDict
enforce schemas, but they compose without conflict:

.. code-block:: python

   >>> from tensordict import TypedTensorDict
   >>>
   >>> class MyTTD(TypedTensorDict):
   ...     a: Tensor
   ...     b: Tensor
   >>>
   >>> ttd = MyTTD(a=torch.randn(4, 3), b=torch.randn(4, 5), batch_size=[4])
   >>> tc_typed = MyTC.from_tensordict(ttd)
   >>> tc_typed.a.shape
   torch.Size([4, 3])


TypedTensorDict + backends
--------------------------

``TypedTensorDict.from_tensordict(td)`` accepts any ``TensorDictBase`` subclass,
just like ``TensorClass``.  The backend is stored live (no copy) -- mutations
through the ``TypedTensorDict`` go directly to the underlying backend.

.. code-block:: python

   from tensordict import TypedTensorDict
   from torch import Tensor

   class State(TypedTensorDict):
       x: Tensor
       y: Tensor

   state = State.from_tensordict(some_backend)

.. list-table::
   :header-rows: 1
   :widths: 22 10 10 10 10 10 10 10 10

   * - Backend
     - Build
     - Read
     - Write
     - Index
     - Clone
     - Stack
     - Iter
     - Update
   * - ``TensorDict``
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
   * - ``PersistentTensorDict`` (H5)
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
   * - ``TensorDictStore`` (Redis)
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
   * - ``LazyStackedTensorDict``
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
     - yes
   * - ``TensorDict`` (memmap)
     - yes
     - yes
     - set\_()
     - yes
     - yes
     - yes
     - yes
     - update\_()

.. note::

   Memory-mapped TensorDicts are locked after ``memmap_()``.  Use
   ``set_()`` and ``update_()`` for in-place writes instead of attribute
   assignment or ``update()``.

Building a TypedTensorDict on each backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**In-memory TensorDict** -- the default (direct construction creates one
internally):

.. code-block:: python

   >>> import torch
   >>> from tensordict import TensorDict, TypedTensorDict
   >>> from torch import Tensor
   >>>
   >>> class State(TypedTensorDict):
   ...     x: Tensor
   ...     y: Tensor
   >>>
   >>> state = State(x=torch.randn(4, 3), y=torch.randn(4, 5), batch_size=[4])
   >>> state.x.shape
   torch.Size([4, 3])

**Wrapping an existing TensorDict** via ``from_tensordict`` (zero-copy):

.. code-block:: python

   >>> td = TensorDict(x=torch.randn(4, 3), y=torch.randn(4, 5), batch_size=[4])
   >>> state = State.from_tensordict(td)
   >>> state.x.shape  # reads from td
   torch.Size([4, 3])
   >>> state.x = torch.ones(4, 3)  # writes to td
   >>> (td["x"] == 1).all()
   True

**HDF5 (PersistentTensorDict)**:

.. code-block:: python

   >>> from tensordict import PersistentTensorDict
   >>>
   >>> h5 = PersistentTensorDict.from_h5("data.h5")
   >>> state = State.from_tensordict(h5)
   >>> state.x.shape  # reads from HDF5
   torch.Size([4, 3])

**Redis (TensorDictStore)**:

.. code-block:: python

   >>> from tensordict.store import TensorDictStore
   >>>
   >>> store = TensorDictStore.from_tensordict(td, host="localhost")
   >>> state = State.from_tensordict(store)
   >>> state.x.shape  # fetched from Redis
   torch.Size([4, 3])

**Lazy stack**:

.. code-block:: python

   >>> from tensordict import lazy_stack
   >>>
   >>> tds = [TensorDict(x=torch.randn(3), y=torch.randn(5)) for _ in range(4)]
   >>> ls = lazy_stack(tds, dim=0)
   >>> state = State.from_tensordict(ls)
   >>> state[0].x.shape
   torch.Size([3])

**Memory-mapped TensorDict**:

.. code-block:: python

   >>> td_mmap = td.memmap_("/tmp/my_memmap")
   >>> state = State.from_tensordict(td_mmap)
   >>> state.x.shape
   torch.Size([4, 3])
   >>> # memmap TDs are locked -- use in-place operations:
   >>> state.set_("x", torch.ones(4, 3))

Stacking TypedTensorDicts
^^^^^^^^^^^^^^^^^^^^^^^^^

Dense stacking with ``torch.stack`` preserves the ``TypedTensorDict`` subclass
type:

.. code-block:: python

   >>> s1 = State(x=torch.randn(3), y=torch.randn(3), batch_size=[3])
   >>> s2 = State(x=torch.randn(3), y=torch.randn(3), batch_size=[3])
   >>> stacked = torch.stack([s1, s2], dim=0)
   >>> stacked.batch_size
   torch.Size([2, 3])

Lazy stacking also works.  Indexing a ``LazyStackedTensorDict`` of
``TypedTensorDict`` instances preserves the subclass type:

.. code-block:: python

   >>> from tensordict._lazy import LazyStackedTensorDict
   >>>
   >>> ls = LazyStackedTensorDict(s1, s2, stack_dim=0)
   >>> isinstance(ls[0], State)
   True

.. _compat-redis-prealloc:

Pre-allocating on Redis and filling iteratively
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common pattern for shared replay buffers or distributed data stores is to
pre-allocate storage on a remote server (Redis / Dragonfly / KeyDB) and fill
it one sample at a time, without ever loading the full dataset into RAM.

``TensorDictStore.from_schema`` creates keys with known shapes and dtypes
directly on the server using ``SETRANGE`` (zero-filled by the server; no
tensor data passes through Python):

.. code-block:: python

   >>> import torch
   >>> from tensordict import TensorDict, TypedTensorDict
   >>> from tensordict.store import TensorDictStore
   >>> from torch import Tensor
   >>>
   >>> class Replay(TypedTensorDict):
   ...     obs: Tensor
   ...     action: Tensor
   ...     reward: Tensor
   >>>
   >>> # Pre-allocate 100k entries directly on Redis -- no RAM used
   >>> store = TensorDictStore.from_schema(
   ...     {"obs": ([84, 84, 3], torch.uint8),
   ...      "action": ([4], torch.float32),
   ...      "reward": ([], torch.float32)},
   ...     batch_size=[100_000],
   ...     host="redis-node",
   ... )
   >>>
   >>> # Wrap with typed access
   >>> replay = Replay.from_tensordict(store)
   >>>
   >>> # Fill iteratively -- each write goes directly to Redis
   >>> for i, sample in enumerate(data_stream):
   ...     replay[i] = Replay(
   ...         obs=sample.obs, action=sample.action, reward=sample.reward,
   ...         batch_size=[],
   ...     )

If the store is initially empty (no keys registered yet), use ``check=False``
to skip the key-presence validation and fill keys on the fly:

.. code-block:: python

   >>> store = TensorDictStore(batch_size=[100_000], host="redis-node")
   >>> replay = Replay.from_tensordict(store, check=False)
   >>>
   >>> # First indexed write auto-creates each key via SETRANGE
   >>> replay[0] = Replay(obs=obs_0, action=act_0, reward=r_0, batch_size=[])
   >>> # Subsequent writes fill in the pre-allocated storage
   >>> replay[1] = Replay(obs=obs_1, action=act_1, reward=r_1, batch_size=[])


TensorClass vs TypedTensorDict
------------------------------

Both enforce typed schemas and can wrap any ``TensorDictBase`` backend, but
they differ architecturally:

.. list-table::
   :header-rows: 1
   :widths: 35 30 30

   * - Aspect
     - ``TensorClass``
     - ``TypedTensorDict``
   * - Relationship to ``TensorDictBase``
     - Wraps a ``TensorDictBase`` (HAS-A via ``TensorCollection``)
     - Is a ``TensorDictBase`` (IS-A, delegates to ``_source``)
   * - Can wrap non-TensorDict backends
     - Yes (H5, Redis, lazy stack, etc.)
     - Yes (H5, Redis, lazy stack, etc.)
   * - ``**state`` spreading
     - Field-by-field repacking
     - Natively (``MutableMapping``)
   * - Non-tensor fields
     - Supported
     - Not supported (tensor-only)
   * - Backend stays live
     - Yes (writes go to original backend)
     - Yes (writes go to original backend)
   * - Python inheritance
     - Not supported
     - Supported (standard class hierarchy)
   * - Composable with each other
     - Yes (``TC.from_tensordict(ttd)`` works)
     - Yes (``TTD.from_tensordict(tc._tensordict)`` works)

Both wrappers keep the backend alive -- mutations through the typed wrapper go
directly to the underlying storage.  Direct construction (without
``from_tensordict``) creates an in-memory ``TensorDict`` as the backend.

Choose ``TensorClass`` when you need non-tensor fields or want to integrate
with existing tensorclass-based APIs.  Choose ``TypedTensorDict`` when you
want native ``**state`` spreading, standard Python inheritance for schema
composition, and full ``TensorDictBase`` API compatibility.
