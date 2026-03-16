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
   │   │   └── TypedTensorDict        (typed fields, IS-A TensorDict)
   │   ├── PersistentTensorDict       (HDF5-backed)
   │   ├── TensorDictStore            (Redis / Dragonfly / KeyDB)
   │   └── LazyStackedTensorDict      (lazy stack of heterogeneous TDs)
   │
   └── TensorClass                    (typed wrapper, HAS-A TensorDictBase)

Two patterns exist for adding typed field declarations:

- **TensorClass** wraps any ``TensorDictBase`` via ``from_tensordict(td)``.
  It delegates all storage to the wrapped object.
- **TypedTensorDict** *is* a ``TensorDict``.  It stores data in-memory and
  interoperates with other backends through conversion or stacking.

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

``TypedTensorDict`` is a ``TensorDict`` subclass.  It stores data in-memory
but interoperates with other backends through conversion or stacking.

.. list-table::
   :header-rows: 1
   :widths: 30 12 12 12 12 12

   * - Pattern
     - Build
     - Read
     - Write
     - Index
     - Stack
   * - Direct construction
     - yes
     - yes
     - yes
     - yes
     - yes
   * - From H5 (materialise then construct)
     - yes
     - yes
     - yes
     - yes
     - yes
   * - From Redis (materialise then construct)
     - yes
     - yes
     - yes
     - yes
     - yes
   * - From lazy stack (materialise then construct)
     - yes
     - yes
     - yes
     - yes
     - yes
   * - ``torch.stack`` (dense)
     - yes
     - yes
     - yes
     - yes
     - --
   * - ``LazyStackedTensorDict`` of TTDs
     - yes
     - yes
     - yes
     - yes
     - --
   * - ``memmap_()``
     - yes
     - yes
     - set\_()
     - yes
     - yes
   * - To H5 (``PersistentTensorDict.from_dict``)
     - yes
     - yes
     - H5 rules
     - yes
     - --
   * - To Redis (``TensorDictStore.from_tensordict``)
     - yes
     - yes
     - yes
     - yes
     - --

Constructing TypedTensorDict from other backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since ``TypedTensorDict`` is an in-memory ``TensorDict``, loading data from a
remote or persistent backend requires materialising the data first:

.. code-block:: python

   >>> import torch
   >>> from tensordict import TypedTensorDict
   >>> from torch import Tensor
   >>>
   >>> class State(TypedTensorDict):
   ...     x: Tensor
   ...     y: Tensor

**From HDF5**:

.. code-block:: python

   >>> from tensordict import PersistentTensorDict
   >>>
   >>> h5 = PersistentTensorDict.from_h5("data.h5")
   >>> local = h5.to_tensordict()
   >>> state = State(x=local["x"], y=local["y"], batch_size=local.batch_size)

**From a lazy stack**:

.. code-block:: python

   >>> from tensordict import lazy_stack
   >>>
   >>> ls = lazy_stack([td1, td2], dim=0)
   >>> local = ls.to_tensordict()
   >>> state = State(x=local["x"], y=local["y"], batch_size=local.batch_size)

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

Saving TypedTensorDict to persistent backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since ``TypedTensorDict`` is a ``TensorDict``, it can be saved to HDF5, Redis,
or memory-mapped storage directly:

.. code-block:: python

   >>> # To HDF5
   >>> from tensordict import PersistentTensorDict
   >>> h5 = PersistentTensorDict.from_dict(state, filename="state.h5")
   >>>
   >>> # To memmap
   >>> state.memmap_("/tmp/state_mmap")
   >>>
   >>> # To Redis
   >>> from tensordict.store import TensorDictStore
   >>> store = TensorDictStore.from_tensordict(state, host="localhost")


TensorClass vs TypedTensorDict
------------------------------

Both enforce typed schemas but differ architecturally:

.. list-table::
   :header-rows: 1
   :widths: 35 30 30

   * - Aspect
     - ``TensorClass``
     - ``TypedTensorDict``
   * - Relationship to ``TensorDict``
     - Wraps a ``TensorDictBase`` (HAS-A)
     - Is a ``TensorDict`` (IS-A)
   * - Can wrap non-TensorDict backends
     - Yes (H5, Redis, lazy stack, etc.)
     - No (in-memory only; convert first)
   * - ``**state`` spreading
     - Field-by-field repacking
     - Natively (``MutableMapping``)
   * - Non-tensor fields
     - Supported
     - Not supported (tensor-only)
   * - Backend stays live
     - Yes (writes go to original backend)
     - No (data is in-memory after construction)
   * - Composable with each other
     - Yes (``TC.from_tensordict(ttd)`` works)
     - N/A

When a ``TensorClass`` wraps a persistent backend (H5, Redis), writes through
the ``TensorClass`` go directly to that backend.  When a ``TypedTensorDict`` is
constructed from persistent data, the data is copied into memory.

Choose ``TensorClass`` when you need live access to a remote or on-disk backend
with typed field access.  Choose ``TypedTensorDict`` when you want typed,
in-memory state with ``**state`` spreading and standard Python inheritance.
