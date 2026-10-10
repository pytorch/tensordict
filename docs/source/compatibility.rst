.. currentmodule:: tensordict

.. _compatibility:

Containers on storage backends
==============================

:class:`~tensordict.TensorClass` and :class:`~tensordict.TypedTensorDict` add
typed fields to a tensordict. Each of them can wrap a tensordict of any
storage backend: an in-memory :class:`~tensordict.TensorDict`, an HDF5 or
zarr file (:class:`~tensordict.PersistentTensorDict`), a Redis server
(:class:`~tensordict.store.TensorDictStore`), a lazy stack
(:class:`~tensordict.LazyStackedTensorDict`), or a memory-mapped
``TensorDict``. This page shows how to build each combination, and lists the
operations that behave differently on some backends.

To choose between ``TensorClass`` and ``TypedTensorDict``, see
:ref:`choosing-a-container`. For the backends themselves, see :ref:`storage`
and :ref:`saving`.

How the classes relate
----------------------

.. code-block:: text

   TensorDictBase
   ├── TensorDict                 (in memory)
   ├── TypedTensorDict            (typed fields, wraps any TensorDictBase)
   ├── PersistentTensorDict       (HDF5 or zarr file)
   ├── TensorDictStore            (Redis, Dragonfly or KeyDB server)
   └── LazyStackedTensorDict      (lazy stack of tensordicts)

   TensorClass                    (typed fields, has a TensorDictBase)

A ``TypedTensorDict`` is a :class:`~tensordict.TensorDictBase`. A
``TensorClass`` is not: it keeps its fields in a tensordict that it holds.
Both have a ``from_tensordict(td)`` class method that wraps ``td`` without a
copy. Reads and writes through the wrapper go to the storage of ``td``. If you
call the constructor instead, the storage is a new in-memory ``TensorDict``.

Operations on each backend
--------------------------

The following operations work on each backend, directly or through a
``TensorClass`` or a ``TypedTensorDict``, with the exceptions in the next
table:

- Build: ``MyTC.from_tensordict(td)``.
- Read a field: ``w.a``.
- Write a field: ``w.a = value``.
- In-place write: ``w.set_("a", value)`` and ``w.update(d, inplace=True)``.
- Update: ``w.update(d)``.
- Index: ``w[0]`` returns the wrapper class, and ``w[0] = other`` writes
  into the backend.
- Clone: ``w.clone()``.
- Stack: ``torch.stack([w, w])``.
- Iterate: ``list(w)``.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Backend
     - Operation
     - What happens, and what to do
   * - Memory-mapped ``TensorDict``
     - Write a field, ``update(d)``
     - ``memmap_()`` locks the tensordict, so these raise ``RuntimeError:
       Cannot modify locked TensorDict``. Use ``set_()``, ``update_()`` or
       ``update(d, inplace=True)``, which write into the existing files.
   * - ``PersistentTensorDict``
     - Write a field, ``update(d)``
     - Each write replaces the dataset in the file and warns ``UserWarning:
       Replacing an array with another one is inefficient``. To write into the
       existing dataset, use ``set_()`` or ``update(d, inplace=True)``.
   * - ``PersistentTensorDict``
     - ``clone()``
     - Without a destination, it warns ``A destination should be provided
       when cloning a PersistentTensorDict``. An HDF5 clone goes to a
       temporary file that is not deleted. A zarr clone raises
       ``FileExistsError``. Pass a destination: ``w.clone(newfile=path)``.
       To copy the data into memory, use ``to_tensordict()``.
   * - ``PersistentTensorDict``, ``TensorDictStore``
     - ``update_(d)``
     - Writes nothing, and raises no error. ``TypedTensorDict`` over a
       ``LazyStackedTensorDict`` has the same fault. Use ``set_()`` for each
       key, or ``update(d, inplace=True)``.
   * - ``PersistentTensorDict``, ``TensorDictStore``,
       ``LazyStackedTensorDict``
     - In-place operation on a field, such as ``w.a.add_(1)``
     - Reading an entry returns a new tensor with a copy of the data, so the
       operation does not change the storage. Write the result back with
       ``w.a = ...``, ``set_()`` or an indexed assignment.

When a field is annotated with another tensorclass (``a: Pose`` or
``a: Pose | None``) and the input is a ``TensorDict`` whose entry for it is a
tensordict, ``from_tensordict`` builds a ``Pose`` from that entry,
recursively. The ``Pose`` is set in a shallow copy of the input: the leaves
are shared, but the input keeps its tensordict entry and does not see the
entries set later on the tensorclass. An entry with keys that ``Pose`` does
not declare stays a tensordict. Other backends are wrapped as they are, and
the nested entry stays a tensordict, because a tensorclass set in them would
be written to their storage.

The examples below write their files to a temporary directory:

  >>> import tempfile
  >>> tmpdir = tempfile.TemporaryDirectory()
  >>> path = tmpdir.name

A TensorClass on each backend
-----------------------------

The examples use this class:

  >>> import torch
  >>> from torch import Tensor
  >>> from tensordict import TensorDict, TensorClass
  >>> class MyTC(TensorClass):
  ...     a: Tensor
  ...     b: Tensor

In-memory ``TensorDict``
~~~~~~~~~~~~~~~~~~~~~~~~

  >>> td = TensorDict(a=torch.randn(4, 3), b=torch.randn(4, 5), batch_size=[4])
  >>> tc = MyTC.from_tensordict(td)
  >>> tc.a.shape
  torch.Size([4, 3])

HDF5 (``PersistentTensorDict``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  >>> import os
  >>> from tensordict import PersistentTensorDict
  >>> h5 = PersistentTensorDict.from_dict(td, filename=os.path.join(path, "data.h5"))
  >>> tc_h5 = MyTC.from_tensordict(h5)
  >>> tc_h5.a.shape  # reads from the HDF5 file
  torch.Size([4, 3])

Redis (``TensorDictStore``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These examples need a Redis server on ``localhost``, so the docs build does
not run them. :ref:`storage` shows how to start one.

  >>> from tensordict.store import TensorDictStore
  >>> store = TensorDictStore.from_tensordict(td, host="localhost")  # doctest: +SKIP
  >>> tc_redis = MyTC.from_tensordict(store)  # doctest: +SKIP
  >>> tc_redis.a.shape  # fetched from Redis  # doctest: +SKIP
  torch.Size([4, 3])

The data stays on the server after the process ends. To delete it, call
``store.clear_redis()``. To close the connection, call ``store.close()``:

  >>> store.clear_redis()  # doctest: +SKIP
  >>> store.close()  # doctest: +SKIP

Lazy stack
~~~~~~~~~~

  >>> from tensordict import lazy_stack
  >>> tds = [TensorDict(a=torch.randn(3), b=torch.randn(5)) for _ in range(4)]
  >>> ls = lazy_stack(tds, dim=0)
  >>> tc_lazy = MyTC.from_tensordict(ls)
  >>> tc_lazy[0].a.shape
  torch.Size([3])

Memory-mapped ``TensorDict``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``memmap_()`` locks the tensordict, so write with ``set_()``:

  >>> td_mmap = td.clone().memmap_(os.path.join(path, "tc_memmap"))
  >>> tc_mmap = MyTC.from_tensordict(td_mmap)
  >>> tc_mmap.a.shape
  torch.Size([4, 3])
  >>> _ = tc_mmap.set_("a", torch.ones(4, 3))
  >>> bool((td_mmap["a"] == 1).all())
  True

``TypedTensorDict``
~~~~~~~~~~~~~~~~~~~

A ``TypedTensorDict`` can be the backend of a ``TensorClass``. Both declare
fields, and the two declarations do not conflict:

  >>> from tensordict import TypedTensorDict
  >>> class MyTTD(TypedTensorDict):
  ...     a: Tensor
  ...     b: Tensor
  >>> ttd = MyTTD(a=torch.randn(4, 3), b=torch.randn(4, 5), batch_size=[4])
  >>> tc_typed = MyTC.from_tensordict(ttd)
  >>> tc_typed.a.shape
  torch.Size([4, 3])

The other direction is not supported: ``TypedTensorDict.from_tensordict``
accepts a tensorclass, but an attribute write through it raises
``ValueError``.

A TypedTensorDict on each backend
---------------------------------

The examples use this class:

  >>> class State(TypedTensorDict):
  ...     x: Tensor
  ...     y: Tensor

In-memory ``TensorDict``
~~~~~~~~~~~~~~~~~~~~~~~~

The constructor creates an in-memory ``TensorDict``:

  >>> state = State(x=torch.randn(4, 3), y=torch.randn(4, 5), batch_size=[4])
  >>> state.x.shape
  torch.Size([4, 3])

``from_tensordict`` wraps an existing ``TensorDict`` without a copy:

  >>> td = TensorDict(x=torch.randn(4, 3), y=torch.randn(4, 5), batch_size=[4])
  >>> state = State.from_tensordict(td)
  >>> state.x.shape  # reads from td
  torch.Size([4, 3])
  >>> state.x = torch.ones(4, 3)  # writes to td
  >>> (td["x"] == 1).all()
  tensor(True)

HDF5 (``PersistentTensorDict``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  >>> h5_state = PersistentTensorDict.from_dict(td, filename=os.path.join(path, "state.h5"))
  >>> state = State.from_tensordict(h5_state)
  >>> state.x.shape  # reads from the HDF5 file
  torch.Size([4, 3])

Redis (``TensorDictStore``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

As above, these examples need a Redis server on ``localhost``:

  >>> store = TensorDictStore.from_tensordict(td, host="localhost")  # doctest: +SKIP
  >>> state = State.from_tensordict(store)  # doctest: +SKIP
  >>> state.x.shape  # fetched from Redis  # doctest: +SKIP
  torch.Size([4, 3])
  >>> store.clear_redis()  # doctest: +SKIP
  >>> store.close()  # doctest: +SKIP

Lazy stack
~~~~~~~~~~

  >>> tds = [TensorDict(x=torch.randn(3), y=torch.randn(5)) for _ in range(4)]
  >>> ls = lazy_stack(tds, dim=0)
  >>> state = State.from_tensordict(ls)
  >>> state[0].x.shape
  torch.Size([3])

Memory-mapped ``TensorDict``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  >>> td_mmap = td.clone().memmap_(os.path.join(path, "state_memmap"))
  >>> state = State.from_tensordict(td_mmap)
  >>> state.x.shape
  torch.Size([4, 3])
  >>> _ = state.set_("x", torch.zeros(4, 3))
  >>> bool((td_mmap["x"] == 0).all())
  True

Stacking TypedTensorDicts
~~~~~~~~~~~~~~~~~~~~~~~~~

``torch.stack`` returns an instance of the ``TypedTensorDict`` subclass:

  >>> s1 = State(x=torch.randn(3), y=torch.randn(3), batch_size=[3])
  >>> s2 = State(x=torch.randn(3), y=torch.randn(3), batch_size=[3])
  >>> stacked = torch.stack([s1, s2], dim=0)
  >>> type(stacked).__name__, stacked.batch_size
  ('State', torch.Size([2, 3]))

A lazy stack also works. An index of a ``LazyStackedTensorDict`` of
``TypedTensorDict`` instances returns the subclass:

  >>> from tensordict import LazyStackedTensorDict
  >>> ls = LazyStackedTensorDict(s1, s2, stack_dim=0)
  >>> isinstance(ls[0], State)
  True

When you no longer need the files, close the HDF5 files and delete the
directory:

  >>> h5.close()
  >>> h5_state.close()
  >>> tmpdir.cleanup()

To allocate storage on a Redis server first and fill it one sample at a time,
see :ref:`pre-allocating on Redis <compat-redis-prealloc>`.

See also
--------

- :ref:`tensorclass-guide` and :ref:`typed-tensordict-guide`: the two typed
  containers.
- :ref:`choosing-a-container`: which container to use, and the
  ``TensorClass`` and ``TypedTensorDict`` comparison.
- :ref:`storage`: HDF5, zarr, Redis and the other storage backends.
- :ref:`saving`: memory-mapped tensordicts and the other formats on disk.
