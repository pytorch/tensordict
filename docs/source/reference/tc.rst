.. currentmodule:: tensordict

tensorclass
===========

A *tensorclass* is a dataclass-like container that inherits all of
:class:`~tensordict.TensorDict`'s machinery — indexing, reshaping,
``to(device)``, ``stack``/``cat``, memory-mapped serialization, ``torch.compile``
support — while exposing your fields as typed attributes rather than string
keys. Tensorclasses let you constrain a container to a known set of fields,
attach custom methods, and get IDE/type-checker support out of the box.

There are two equivalent entry points:

* :class:`~tensordict.TensorClass` — the **inheritance-based** API. Recommended
  for new code. Static type-checkers understand it without help, and the
  parametric form ``TensorClass["autocast", ...]`` makes the configuration
  explicit at the class declaration site.
* :func:`~tensordict.tensorclass` — the **decorator** form. Kept for
  backwards-compatibility and for migrating plain ``@dataclass`` code. See
  :ref:`tensorclass-legacy-decorator` below.

Quick start
-----------

.. code-block::

  >>> from __future__ import annotations
  >>> from typing import Optional
  >>> import torch
  >>> from tensordict import TensorClass
  >>>
  >>> class MyData(TensorClass):
  ...     floatdata: torch.Tensor
  ...     intdata: torch.Tensor
  ...     non_tensordata: str
  ...     nested: Optional[MyData] = None
  ...
  ...     def check_nested(self):
  ...         assert self.nested is not None
  >>>
  >>> data = MyData(
  ...   floatdata=torch.randn(3, 4, 5),
  ...   intdata=torch.randint(10, (3, 4, 1)),
  ...   non_tensordata="test",
  ...   batch_size=[3, 4],
  ... )
  >>> print("data:", data)
  data: MyData(
    floatdata=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
    intdata=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
    non_tensordata='test',
    nested=None,
    batch_size=torch.Size([3, 4]),
    device=None,
    is_shared=False)
  >>> data.nested = MyData(
  ...     floatdata=torch.randn(3, 4, 5),
  ...     intdata=torch.randint(10, (3, 4, 1)),
  ...     non_tensordata="nested_test",
  ...     batch_size=[3, 4],
  ... )

If the batch size is omitted it defaults to an empty shape. With a non-empty
batch size, tensor fields are indexed elementwise and non-tensor fields are
preserved:

.. code-block::

  >>> print("indexed:", data[:2])
  indexed: MyData(
     floatdata=Tensor(shape=torch.Size([2, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
     intdata=Tensor(shape=torch.Size([2, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
     non_tensordata='test',
     nested=MyData(
        floatdata=Tensor(shape=torch.Size([2, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
        intdata=Tensor(shape=torch.Size([2, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
        non_tensordata='nested_test',
        nested=None,
        batch_size=torch.Size([2, 4]),
        device=None,
        is_shared=False),
     batch_size=torch.Size([2, 4]),
     device=None,
     is_shared=False)

Tensorclasses support attribute mutation (including on nested instances), the
usual tensor-shape operations (``stack``, ``cat``, ``reshape``, ``to(device)``,
...), and equality. See the :class:`~tensordict.TensorDict` documentation for
the full list of operations.

.. code-block::

  >>> data2 = data.clone()
  >>> cat_tc = torch.cat([data, data2], 0)
  >>> assert cat_tc.batch_size == torch.Size([6, 4])

Flags
-----

The behaviour of a tensorclass is controlled by a handful of mutually
intelligible flags. They can be set in three equivalent forms — pick whichever
reads best:

.. code-block::

  >>> class Foo(TensorClass["autocast"]):     # bracket form
  ...     x: int
  >>> class Foo(TensorClass, autocast=True):  # kwargs form
  ...     x: int
  >>> @tensorclass(autocast=True)             # decorator form
  ... class Foo:
  ...     x: int

Multiple flags can be combined in the brackets:

.. code-block::

  >>> class Foo(TensorClass["nocast", "frozen"]):
  ...     x: int

The bracket form is the one that static type-checkers (mypy, pyright) resolve
correctly via :meth:`~object.__class_getitem__`, so IDE completion on instance
attributes works without further configuration.

The available flags are:

* ``autocast`` — coerce values back to the field's annotated type when reading.
  Useful when a field is conceptually a scalar, string, or enum.
* ``nocast`` — store tensor-compatible scalars (``int``, ``float``,
  ``np.ndarray``, ...) as-is, without wrapping them in a tensor.
* ``tensor_only`` — every field must hold a tensor (or be tensor-castable).
  Skips the non-tensor storage path and yields measurable speed-ups on
  attribute access — recommended for performance-critical containers (RL
  trajectories, model I/O batches).
* ``frozen`` — immutable instances, in the spirit of
  ``@dataclass(frozen=True)``. Plays well with ``torch.compile`` and functional
  code paths.
* ``shadow`` — opt out of the check that forbids field names colliding with
  reserved TensorDict attributes (``batch_size``, ``device``, ``data``, ...).

``autocast``, ``nocast`` and ``tensor_only`` are mutually exclusive. See the
:class:`~tensordict.TensorClass` docstring for per-flag runnable examples.

.. _tensorclass-autocasting:

Auto-casting
~~~~~~~~~~~~

.. warning:: Auto-casting is an experimental feature and subject to changes in
  the future. Compatibility with python<=3.9 is limited.

With ``autocast`` enabled, methods such as ``__setattr__``, ``update``,
``update_`` and ``from_dict`` will attempt to cast type-annotated entries to
the desired TensorDict / tensorclass instance (except in cases detailed below).
For instance, the following code casts the ``td`` dictionary to a
:class:`~tensordict.TensorDict` and the ``tc`` entry to a ``MyClass`` instance:

    >>> class MyClass(TensorClass["autocast"]):
    ...     tensor: torch.Tensor
    ...     td: TensorDict
    ...     tc: MyClass
    ...
    >>> obj = MyClass(
    ...     tensor=torch.randn(()),
    ...     td={"a": torch.randn(())},
    ...     tc={"tensor": torch.randn(()), "td": None, "tc": None})
    >>> assert isinstance(obj.tensor, torch.Tensor)
    >>> assert isinstance(obj.tc, MyClass)
    >>> assert isinstance(obj.td, TensorDict)

.. note:: Type-annotated items that include a ``typing.Optional`` or
  ``typing.Union`` will not be compatible with auto-casting, but other items in
  the tensorclass will:

    >>> class MyClass(TensorClass["autocast"]):
    ...     tensor: torch.Tensor
    ...     tc_autocast: MyClass = None
    ...     tc_not_autocast: Optional[MyClass] = None
    >>> obj = MyClass(
    ...     tensor=torch.randn(()),
    ...     tc_autocast={"tensor": torch.randn(())},
    ...     tc_not_autocast={"tensor": torch.randn(())},
    ... )
    >>> assert isinstance(obj.tc_autocast, MyClass)
    >>> # because the type is Optional or Union, auto-casting is disabled for
    >>> # that variable.
    >>> assert not isinstance(obj.tc_not_autocast, MyClass)

  If at least one item in the class is annotated using the ``type0 | type1``
  semantic, the whole class auto-casting capabilities are deactivated.
  Because tensorclass supports non-tensor leaves, setting a dictionary in
  these cases will lead to setting it as a plain dictionary instead of a
  tensor collection subclass (``TensorDict`` or tensorclass).

.. note:: Auto-casting isn't enabled for leaves (tensors). The reason is that
  this feature isn't compatible with type annotations that contain the
  ``type0 | type1`` type-hint semantics, which are widespread. Allowing
  auto-casting on leaves would result in very similar code having drastically
  different behaviour depending on small annotation differences.

Performance with ``tensor_only``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default tensorclass attribute getter first looks in the tensor storage and
then falls back to the non-tensor storage. ``tensor_only`` skips that fallback,
which makes ``self.field`` a direct dict lookup. For containers that genuinely
hold only tensors (think batched observations, actions, rewards in RL, or model
inputs/outputs) this can save a meaningful share of per-step overhead.

    >>> class TrajectoryBatch(TensorClass["tensor_only"]):
    ...     obs: torch.Tensor
    ...     action: torch.Tensor
    ...     reward: torch.Tensor

Non-tensor inputs that are tensor-castable (Python scalars, numpy arrays) are
still accepted — they are converted to tensors at assignment time.

Immutability with ``frozen``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``frozen=True`` makes instances immutable, mirroring ``@dataclass(frozen=True)``.
Frozen tensorclasses are a good fit for state objects passed through
``torch.compile`` or functional pipelines where in-place mutation would be a
correctness hazard.

    >>> class State(TensorClass["frozen"]):
    ...     params: torch.Tensor
    ...     step: torch.Tensor
    >>> s = State(params=torch.zeros(3), step=torch.zeros((), dtype=torch.long))
    >>> s.step = s.step + 1   # raises dataclasses.FrozenInstanceError

``frozen`` is inherited: a non-frozen subclass cannot inherit from a frozen
base, and vice versa.

Migrating from a plain dataclass
--------------------------------

If you already have a ``@dataclass``, :func:`~tensordict.from_dataclass` will
build a tensorclass type or instance from it without rewriting the class
definition:

.. code-block::

  >>> from dataclasses import dataclass
  >>> from tensordict import from_dataclass
  >>>
  >>> @dataclass
  ... class X:
  ...     a: int
  ...     b: torch.Tensor
  >>>
  >>> XTc = from_dataclass(X, autocast=True)   # convert the type
  >>> x_tc = from_dataclass(X(a=0, b=torch.zeros(())))  # convert an instance

The same configuration flags (``autocast``, ``nocast``, ``frozen``,
``tensor_only``, ``shadow``) are accepted.

Non-tensor data: ``NonTensorData`` vs ``MetaData``
--------------------------------------------------

Tensorclasses transparently support non-tensor fields. Two wrappers carry such
values and they differ in how they react to shape operations:

* :class:`~tensordict.NonTensorData` — the default. Behaves like a regular
  TensorDict entry under shape operations: stacking a list of
  ``NonTensorData`` items along a new dimension keeps every value, expansion
  broadcasts to the requested shape, indexing returns the corresponding entry.
* :class:`~tensordict.MetaData` — broadcasts a single value across the batch.
  Stacking ``MetaData("a")`` and ``MetaData("a")`` returns ``MetaData("a")``;
  stacking with a different value raises. Use this for static, per-class
  metadata (a string label, a configuration dict) that should not multiply
  under batching.

Typed ``nn.Module`` I/O
-----------------------

For modules whose inputs and outputs are tensorclasses, use
:class:`~tensordict.nn.TensorClassModuleBase`. The generic parameters declare
the input and output types, giving IDE completion, refactor safety, and an
``as_td_module()`` adapter for use inside
:class:`~tensordict.nn.TensorDictSequential` pipelines. See the
:doc:`nn` reference for a complete example.

``torch.compile`` and cudagraphs
--------------------------------

Tensorclasses are designed to compile well. A few practical notes:

* Prefer ``frozen=True`` for state objects when you can — the compiler reasons
  about immutable containers more cleanly than about mutable ones.
* ``tensor_only=True`` containers avoid the non-tensor lookup branch and tend
  to produce simpler graphs.
* Avoid data-dependent shapes (``.item()`` on a tensor that controls a Python
  branch) and prefer :func:`torch.where` over Python ``if``/``else`` on tensor
  values.

Serialization
-------------

A tensorclass instance can be saved with the ``memmap`` method. Tensor data is
written as memory-mapped tensors and JSON-serializable non-tensor data is
written as JSON; remaining data is pickled via :func:`~torch.save`.

Loading is done with :meth:`~tensordict.TensorDict.load_memmap`. The instance
recovers its original type provided the tensorclass is importable in the
loading process:

  >>> data.memmap("path/to/saved/directory")
  >>> data_loaded = TensorDict.load_memmap("path/to/saved/directory")
  >>> assert isinstance(data_loaded, type(data))

Edge cases
----------

Tensorclasses support equality and inequality operators, including for nested
instances. Non-tensor / meta data is not validated by these operators; the
returned tensorclass has boolean leaves for tensor fields and ``None`` for
non-tensor fields.

.. code-block::

  >>> print(data == data2)
  MyData(
     floatdata=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.bool, is_shared=False),
     intdata=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
     non_tensordata=None,
     nested=MyData(
         floatdata=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.bool, is_shared=False),
         intdata=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
         non_tensordata=None,
         nested=None,
         batch_size=torch.Size([3, 4]),
         device=None,
         is_shared=False),
     batch_size=torch.Size([3, 4]),
     device=None,
     is_shared=False)

Item assignment performs an *identity* check on non-tensor / meta data rather
than equality, for performance reasons. If the values differ, a
``UserWarning`` is emitted; users are responsible for keeping non-tensor data
in sync.

.. code-block::

  >>> data2.non_tensordata = "test_new"
  >>> data[0] = data2[0]
  UserWarning: Meta data at 'non_tensordata' may or may not be equal, this may result in undefined behaviours

``torch.cat`` / ``torch.stack`` work on tensorclasses but do not validate
non-tensor / meta fields — the operation runs on the tensor leaves and the
non-tensor data of the *first* instance in the list is kept. If the inputs
disagree on a non-tensor field, the output will silently follow the first one:

.. code-block::

  >>> data2.non_tensordata = "test_new"
  >>> stack_tc = torch.cat([data, data2], dim=0)
  >>> assert stack_tc.non_tensordata == "test"  # data's value wins

Pre-allocation
--------------

Fields can be initialised to ``None`` and assigned later. While ``None``, the
attribute is stored as non-tensor / meta data; on assignment the appropriate
storage is selected based on the value's type.

.. code-block::

  >>> class MyClass(TensorClass):
  ...     X: Any
  ...     y: Any
  >>>
  >>> data = MyClass(X=None, y=None, batch_size=[3, 4])
  >>> data.X = torch.ones(3, 4, 5)
  >>> data.y = "testing"
  >>> print(data)
  MyClass(
     X=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
     y='testing',
     batch_size=torch.Size([3, 4]),
     device=None,
     is_shared=False)

.. _tensorclass-legacy-decorator:

Legacy: the ``@tensorclass`` decorator
--------------------------------------

The :func:`~tensordict.tensorclass` decorator is the original way to declare a
tensorclass. It is kept for backwards compatibility and remains the most
convenient option when migrating a body of ``@dataclass`` code. New code is
encouraged to use :class:`~tensordict.TensorClass` instead, which carries
identical semantics with better static-type-checker support.

.. code-block::

  >>> from __future__ import annotations
  >>> from typing import Optional
  >>> from tensordict import tensorclass
  >>> import torch
  >>>
  >>> @tensorclass
  ... class MyData:
  ...     floatdata: torch.Tensor
  ...     intdata: torch.Tensor
  ...     non_tensordata: str
  ...     nested: Optional[MyData] = None

All flags described above are accepted as keyword arguments to the decorator:
``@tensorclass(autocast=True)``, ``@tensorclass(frozen=True, tensor_only=True)``,
etc.

API reference
-------------

.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    TensorClass
    tensorclass
    NonTensorData
    MetaData
    NonTensorStack
    TensorAttrs
    UnbatchedTensor
    from_dataclass
