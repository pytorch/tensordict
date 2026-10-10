.. currentmodule:: tensordict

.. _choosing-a-container:

Which container should I use?
=============================

tensordict has several containers for groups of tensors. They have the same
operations (indexing, reshaping, ``to(device)``, stacking, saving), and they
differ in how you declare their content and where they keep the data. This
page helps you pick one. It is for new users, and for users who know
:class:`~tensordict.TensorDict` and want to know what the other classes add.

All the containers below except the plain dict have a *batch size*: the
leading dimensions that all their entries share (see :ref:`batch-size`).
Indexing, reshaping and stacking apply to these dimensions.

The containers
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Container
     - Use it for
     - Main constraint
   * - :class:`~tensordict.TensorDict`
     - Tensors that share their batch dimensions, under string keys or nested
       keys that you can add and remove at any time. The data is in memory,
       on any device.
     - Each entry must start with the batch dimensions. Any other value
       raises ``RuntimeError: batch dimension mismatch``.
   * - :class:`~tensordict.LazyStackedTensorDict`
     - A stack of tensordicts whose entries have different shapes, for
       example sequences of different lengths. :func:`~tensordict.lazy_stack`
       makes one. It keeps the stacked tensordicts, so a write to the stack
       reaches them.
     - Each read of an entry stacks it into a new tensor. An entry with
       different shapes raises ``RuntimeError`` unless you ask for a padded
       tensor, a nested tensor or a list (``get(key, as_padded_tensor=True)``).
       ``keys()`` lists only the keys that all the stacked tensordicts have.
   * - :class:`~tensordict.TensorClass`
     - A fixed set of fields that you declare like a dataclass, with
       attribute access, your own methods, and plain Python values for
       fields that are not tensors.
     - It is not a :class:`~tensordict.TensorDictBase` or a ``Mapping``:
       ``tc["key"]`` and ``**tc`` raise ``ValueError``, and a function that
       tests ``isinstance(x, TensorDictBase)`` rejects it.
   * - :class:`~tensordict.TypedTensorDict`
     - Declared fields on a class that is a ``TensorDictBase``, with
       ``**state`` unpacking and the full tensordict API.
     - Attribute access returns a string field as a
       :class:`~tensordict.NonTensorData` wrapper. No custom ``__init__``
       and no ``__post_init__``.
   * - :class:`~tensordict.PersistentTensorDict`
     - Data in an HDF5 or zarr file, read on demand. The file can be larger
       than the memory.
     - Each read loads a new tensor from the file, so an in-place operation
       on it does not change the file. ``update_()`` writes nothing. Needs
       ``h5py`` or ``zarr``.
   * - :class:`~tensordict.store.TensorDictStore`
     - Data on a Redis, Dragonfly or KeyDB server, which several processes
       or machines can read and write.
     - Needs a running server and the ``redis`` package. Each read goes over
       the network. The data stays on the server until you call
       ``clear_redis()``. ``update_()`` writes nothing.
   * - A plain ``dict`` of tensors
     - Any Python values, with no rule on their shapes.
     - No batch dimensions: you write a loop, or a
       ``torch.utils._pytree.tree_map`` call, for each operation on the
       values.

The ``update_()`` fault of ``PersistentTensorDict`` and ``TensorDictStore``
has a workaround: use ``set_()`` for each key, or ``update(d, inplace=True)``.
:ref:`compatibility` lists it with the other limits of each backend.

By need
-------

- If the keys change while your program runs, or you do not know them in
  advance, use a :class:`~tensordict.TensorDict`.
- If you want a fixed set of fields, methods of your own, and completion in
  your IDE, use a :class:`~tensordict.TensorClass`. See
  :ref:`tensorclass-guide`.
- If you want declared fields, and the object must also be a
  ``TensorDictBase`` (for ``state["key"]``, ``**state`` or a function that
  takes a tensordict), use a :class:`~tensordict.TypedTensorDict`. See
  :ref:`typed-tensordict-guide`.
- If you stack items of different shapes, use
  :func:`~tensordict.lazy_stack`, which returns a
  :class:`~tensordict.LazyStackedTensorDict`. See :ref:`stacking`.
- If your dataset is larger than the memory and is on a local disk, save a
  ``TensorDict`` with :meth:`~tensordict.TensorDict.memmap_` and load it with
  :meth:`~tensordict.TensorDict.load_memmap`. Its entries are then
  memory-mapped tensors. See :ref:`saving`.
- If your data is in HDF5 or zarr files, use a
  :class:`~tensordict.PersistentTensorDict`. See :ref:`storage`.
- If several processes or machines must share the data through a server, use
  a :class:`~tensordict.store.TensorDictStore`. See :ref:`storage`.
- If you want typed fields on a file, a server or a lazy stack, wrap it with
  ``from_tensordict`` of a ``TensorClass`` or a ``TypedTensorDict``. See
  :ref:`compatibility`.
- If your values have no common leading dimension, use a plain ``dict``, or
  put them in a tensordict with ``batch_size=[]``.

TensorClass or TypedTensorDict?
-------------------------------

Both declare typed fields with class annotations, and both can wrap a
tensordict of any backend. A :class:`~tensordict.TensorClass` holds a
tensordict, and a :class:`~tensordict.TypedTensorDict` is one. The other
differences follow from this:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Feature
     - ``TensorClass``
     - ``TypedTensorDict``
   * - Declaration
     - ``class Foo(TensorClass)``, or the legacy ``@tensorclass`` decorator
     - ``class Foo(TypedTensorDict)``
   * - ``isinstance(obj, TensorDictBase)``
     - ``False``. Use :func:`~tensordict.is_tensor_collection` to accept
       both.
     - ``True``
   * - ``obj["key"]``
     - Raises ``ValueError``. Use ``obj.key`` or ``obj.get("key")``.
     - Returns the entry.
   * - ``**obj``
     - Raises ``ValueError``. Use ``**obj.to_dict()``, which also turns
       nested tensorclasses into dicts.
     - Works: a ``TypedTensorDict`` is a ``MutableMapping``.
   * - A ``str`` field
     - Attribute access returns the ``str``.
     - Attribute access returns a ``NonTensorData`` wrapper.
       ``obj["key"]`` returns the ``str``.
   * - A Python number
     - Becomes a tensor, unless the class has the ``nocast`` flag.
     - Becomes a tensor.
   * - Optional fields
     - ``x: Tensor | None = None``. A field that is not set is not in
       ``keys()``.
     - ``x: Tensor | None = None``. A field that is not set is not in
       ``keys()``.
   * - Class options
     - ``autocast``, ``nocast``, ``tensor_only``, ``frozen``, ``shadow``
     - ``shadow`` and ``frozen``. The other three have no effect, and are
       deprecated.
   * - ``frozen``
     - Attribute assignment raises ``dataclasses.FrozenInstanceError``.
     - Locks the tensordict: attribute assignment raises ``RuntimeError``.
   * - Methods and properties
     - Supported
     - Supported
   * - Custom ``__init__``, ``__post_init__``
     - Supported
     - Not supported
   * - Wrap another backend
     - ``Foo.from_tensordict(td)``. Writes go to ``td``.
     - ``Foo.from_tensordict(td)``. Writes go to ``td``.
   * - Wrap the other class
     - ``Foo.from_tensordict(ttd)`` works.
     - ``Foo.from_tensordict(tc)`` reads the fields, but an attribute write
       raises ``ValueError``.
   * - Type checkers (pyright, mypy)
     - Report a field value of the wrong type, a missing field and an
       unknown keyword argument in the constructor.
     - Type the fields. Only mypy reports a missing field, and neither
       reports an unknown keyword argument.

In short:

- Use ``TensorClass`` if your fields include strings or other Python objects
  that you read as plain values, or if you need a custom ``__init__``,
  ``__post_init__`` or the ``autocast`` flag.
- Use ``TypedTensorDict`` if the object must be a ``TensorDictBase``, or if
  your pipeline builds its state in stages and passes it on with
  ``**state``.

See also
--------

- :ref:`overview`: what a tensordict is.
- :ref:`tensorclass-guide` and :ref:`typed-tensordict-guide`: the two typed
  containers.
- :ref:`stacking`: lazy stacks and data of variable size.
- :ref:`storage` and :ref:`saving`: the backends on disk and on servers.
- :ref:`compatibility`: the typed containers on each backend.
- :ref:`api-tensordict`, :ref:`api-tensorclass` and
  :ref:`api-typed-tensordict`: the API reference.
