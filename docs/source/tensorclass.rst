.. currentmodule:: tensordict

.. _tensorclass-guide:

TensorClass
===========

A *tensorclass* is a class that you declare like a :func:`~dataclasses.dataclass`
and that has the operations of a :class:`~tensordict.TensorDict`: indexing,
reshaping, ``to(device)``, ``torch.stack`` and ``torch.cat``, and saving to
disk. Its fields are typed attributes, not string keys. This page is for users
who want a container with a fixed set of fields, methods of their own, and
completion and type checks in their IDE.

You can declare a tensorclass in two ways:

- Subclass :class:`~tensordict.TensorClass`. Use this form in new code. Static
  type checkers understand it without a plugin.
- Decorate a class with :func:`~tensordict.tensorclass`. This form is kept for
  backward compatibility and to convert ``@dataclass`` code. See
  :ref:`tensorclass-legacy-decorator`.

To compare a tensorclass with :class:`~tensordict.TypedTensorDict` and the
other containers, see :ref:`choosing-a-container`.

Quick start
-----------

Declare each field with a type annotation. Then pass the values and the batch
size to the constructor. The *batch size* is the leading dimensions that all
the fields share (see :ref:`batch-size`).

  >>> from typing import Optional
  >>> import torch
  >>> from tensordict import TensorClass
  >>> class MyData(TensorClass):
  ...     floatdata: torch.Tensor
  ...     intdata: torch.Tensor
  ...     non_tensordata: str
  ...     nested: Optional["MyData"] = None
  >>> data = MyData(
  ...     floatdata=torch.randn(3, 4, 5),
  ...     intdata=torch.randint(10, (3, 4, 1)),
  ...     non_tensordata="test",
  ...     batch_size=[3, 4],
  ... )
  >>> print(data)
  MyData(
      floatdata=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
      intdata=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
      non_tensordata=NonTensorData(data=test, batch_size=torch.Size([3, 4]), device=None),
      nested=None,
      batch_size=torch.Size([3, 4]),
      device=None,
      is_shared=False)

The ``nested`` annotation is a string because the class refers to itself.
Another way is ``from __future__ import annotations`` at the top of the
module. If you do not pass ``batch_size``, it is ``torch.Size([])``, and the
tensorclass has no batch dimensions.

Read and assign the fields as attributes. A field that is not a tensor
(``non_tensordata`` here) returns its plain value:

  >>> data.non_tensordata
  'test'
  >>> data.nested = MyData(
  ...     floatdata=torch.randn(3, 4, 5),
  ...     intdata=torch.randint(10, (3, 4, 1)),
  ...     non_tensordata="nested_test",
  ...     batch_size=[3, 4],
  ... )

An index applies to the batch dimensions of every field, also in the nested
tensorclass. Fields that are not tensors keep their value:

  >>> sub = data[:2]
  >>> sub.batch_size
  torch.Size([2, 4])
  >>> sub.nested.floatdata.shape
  torch.Size([2, 4, 5])
  >>> sub.nested.non_tensordata
  'nested_test'

The shape and device operations of :class:`~tensordict.TensorDict` (``stack``,
``cat``, ``reshape``, ``to``, ``clone``, ``apply``, and the others) work on a
tensorclass and return an instance of the same class:

  >>> data2 = data.clone()
  >>> cat_tc = torch.cat([data, data2], 0)
  >>> type(cat_tc).__name__, cat_tc.batch_size
  ('MyData', torch.Size([6, 4]))

A tensorclass is not a :class:`~tensordict.TensorDictBase` and not a
:class:`~collections.abc.Mapping`. A string index raises an error, because an
index always applies to the batch dimensions. To read a field by name, use
``get``:

  >>> data["intdata"]
  Traceback (most recent call last):
  ...
  ValueError: Invalid indexing arguments: 'intdata'. A tensorclass is indexed along its batch dimensions; to read a field, use the attribute or get('intdata').
  >>> data.get("intdata").shape
  torch.Size([3, 4, 1])

To test whether an object is a tensordict or a tensorclass, use
:func:`~tensordict.is_tensor_collection`. ``to_tensordict()`` returns a copy
of the data as a :class:`~tensordict.TensorDict`.

Class options
-------------

Flags change how a tensorclass converts and checks the values of its fields.
You can set them in three forms, which have the same effect:

  >>> from tensordict import tensorclass
  >>> class Foo(TensorClass["autocast"]):  # bracket form
  ...     x: int
  >>> class Foo(TensorClass, autocast=True):  # keyword form
  ...     x: int
  >>> @tensorclass(autocast=True)  # decorator form
  ... class Foo:
  ...     x: int

To set more than one flag, list them in the brackets:

  >>> class Foo(TensorClass["nocast", "frozen"]):
  ...     x: int

mypy rejects the bracket form. See :ref:`tensorclass-static-typing`.

The flags are:

``autocast``
  The constructor and attribute assignment convert each value to the type
  annotation of its field. See :ref:`tensorclass-autocasting`.

``nocast``
  Python numbers and NumPy arrays stay as they are. Without this flag, an
  ``int``, ``float``, ``bool`` or NumPy array becomes a tensor.

``tensor_only``
  Each field must have a tensor annotation: a tensor type, a tensor collection
  (:class:`~tensordict.TensorDict` or a tensorclass), or an ``Optional`` or
  union of these. Any other annotation raises ``TypeError`` when the class is
  defined. Python numbers and NumPy arrays become tensors. A value that cannot
  become a tensor, for example a ``str``, is stored as
  :class:`~tensordict.NonTensorData` with no error, and attribute access
  returns the ``NonTensorData`` wrapper. Attribute access does not look in the
  storage for non-tensor values, so it does less work. On one CPU, reading a
  tensor field took 150 ns with ``tensor_only`` and 230 ns without it.
  ``benchmarks/tensorclass/test_tensorclass_speed.py`` measures it.

``frozen``
  Assigning a field raises :class:`dataclasses.FrozenInstanceError`, as with
  ``@dataclass(frozen=True)``. See :ref:`tensorclass-frozen`.

``shadow``
  Allows field names that are members of a tensorclass, for example
  ``batch_size``, ``device``, ``names``, ``sum``, ``keys``, ``get`` or
  ``to_dict``. Without this flag, such a name raises ``AttributeError`` when
  the class is defined. A field named ``data`` or ``fields`` is allowed
  without the flag, and it replaces the member of that name.

``autocast``, ``nocast`` and ``tensor_only`` are mutually exclusive. A class
with two of them raises ``TypeError``.

.. _tensorclass-autocasting:

Auto-casting
~~~~~~~~~~~~

.. warning:: Auto-casting is experimental. Its behavior can change in a
  future release.

With ``autocast``, the constructor and attribute assignment convert each value
to the annotation of its field:

- A dict becomes a :class:`~tensordict.TensorDict`, or the tensorclass of the
  annotation.
- A value for an ``int``, ``float`` or ``str`` field becomes an ``int``,
  ``float`` or ``str``. For example, ``torch.tensor(3)`` becomes ``3``.
- A value for a ``torch.Tensor`` field goes through :func:`torch.as_tensor`.
  A value that it cannot convert, for example a ``str``, raises ``TypeError``.

In this example, the dict of ``td`` becomes a ``TensorDict``, the dict of
``leaf`` becomes a ``Leaf``, and the tensor of ``count`` becomes an ``int``:

  >>> from tensordict import TensorDict
  >>> class Leaf(TensorClass):
  ...     value: torch.Tensor
  >>> class Node(TensorClass["autocast"]):
  ...     tensor: torch.Tensor
  ...     td: TensorDict
  ...     count: int
  ...     leaf: Leaf
  >>> node = Node(
  ...     tensor=torch.zeros(()),
  ...     td={"a": torch.zeros(())},
  ...     count=torch.tensor(3),
  ...     leaf={"value": torch.ones(())},
  ... )
  >>> type(node.td).__name__, type(node.leaf).__name__, node.count
  ('TensorDict', 'Leaf', 3)

``update()`` and ``from_dict()`` also convert a dict into the
``TensorDict`` or tensorclass of its field. They do not convert values to
``int``, ``float`` or ``str``.

A field annotated with ``Optional[...]``, ``Union[...]`` or ``A | B`` is not
auto-cast. A dict that you assign to it stays a dict, because a tensorclass
can hold values that are not tensors. The other fields of the class are still
auto-cast:

  >>> class Pair(TensorClass["autocast"]):
  ...     left: Leaf = None
  ...     right: Optional[Leaf] = None
  >>> pair = Pair(left={"value": torch.ones(())}, right={"value": torch.ones(())})
  >>> type(pair.left).__name__, type(pair.right).__name__
  ('Leaf', 'dict')

Auto-casting reads the annotations with :func:`typing.get_type_hints`. If one
annotation of the class does not resolve, tensordict warns
``type_hints are none, cannot perform auto-casting`` and casts no field of the
class. An annotation does not resolve when it is a string (or when the module
uses ``from __future__ import annotations``) and names a class that is not
visible from the module. A tensorclass that is defined inside a function and
that another class refers to is the usual case. Define such classes at module
level.

.. _tensorclass-frozen:

Immutability with ``frozen``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``frozen``, assigning a field raises
:class:`dataclasses.FrozenInstanceError`, as with ``@dataclass(frozen=True)``.
The instance is also locked, so you cannot add entries to it (see
:ref:`keys`):

  >>> class State(TensorClass["frozen"]):
  ...     params: torch.Tensor
  ...     step: torch.Tensor
  >>> s = State(params=torch.zeros(3), step=torch.zeros((), dtype=torch.long))
  >>> s.step = s.step + 1
  Traceback (most recent call last):
  ...
  dataclasses.FrozenInstanceError: cannot assign to field 'step'

The tensors of a frozen instance are not read-only. An in-place operation on
a field, or ``set_()``, writes into the existing tensor:

  >>> _ = s.step.add_(1)
  >>> s.step
  tensor(1)

A subclass of a frozen tensorclass is frozen, with or without the flag. A
frozen class cannot subclass a tensorclass that is not frozen: this raises
``TypeError: cannot inherit frozen dataclass from a non-frozen one``. Type
checkers read a subclass without the flag as not frozen and report it, so
repeat ``frozen=True`` on the subclass (for mypy, see
:ref:`tensorclass-static-typing`).

Custom methods
--------------

A tensorclass is a regular Python class. You can add methods, properties,
class methods and class constants to it. Of the dataclass features,
``__post_init__``, field defaults and a custom ``__init__`` work as described
below.

Methods and properties
~~~~~~~~~~~~~~~~~~~~~~

Methods and properties can read and assign the fields. A method that calls a
tensordict operation, such as ``apply``, returns an instance of the class. A
property is not a field: it is not stored and not in ``keys()``. Use
:class:`typing.ClassVar` for a constant of the class. It is not a field
either:

  >>> from typing import ClassVar
  >>> class Logits(TensorClass):
  ...     logits: torch.Tensor
  ...     temperature: ClassVar[float] = 2.0
  ...
  ...     @classmethod
  ...     def uniform(cls, batch, n):
  ...         return cls(logits=torch.zeros(batch, n), batch_size=[batch])
  ...
  ...     @property
  ...     def probs(self):
  ...         return (self.logits / self.temperature).softmax(-1)
  ...
  ...     @probs.setter
  ...     def probs(self, value):
  ...         self.logits = value.log() * self.temperature
  ...
  ...     def detached(self):
  ...         return self.apply(lambda t: t.detach())
  >>> x = Logits.uniform(2, 4)
  >>> x.probs[0]
  tensor([0.2500, 0.2500, 0.2500, 0.2500])
  >>> x.probs = torch.tensor([0.4, 0.2, 0.2, 0.2]).expand(2, 4)
  >>> x.probs[0]
  tensor([0.4000, 0.2000, 0.2000, 0.2000])
  >>> list(x.keys())
  ['logits']
  >>> type(x.detached()).__name__, x[0].probs.shape
  ('Logits', torch.Size([4]))

``__post_init__``
~~~~~~~~~~~~~~~~~

As in a dataclass, ``__post_init__`` runs after the generated ``__init__``.
Use it to compute a default value, or to make sure that the fields are valid:

  >>> class Batch(TensorClass):
  ...     obs: torch.Tensor
  ...     mask: torch.Tensor = None
  ...
  ...     def __post_init__(self):
  ...         if self.mask is None:
  ...             self.mask = torch.ones(self.batch_size, dtype=torch.bool)
  >>> b = Batch(obs=torch.zeros(3, 2), batch_size=[3])
  >>> b.mask
  tensor([True, True, True])

``__post_init__`` also runs each time tensordict makes a new instance from an
existing one: an index, ``clone``, ``torch.stack``, ``apply`` and
``from_tensordict``. Do little work in it, and make sure that a second run
gives the same result. In the example above, the second run finds a mask and keeps it.

A custom ``__init__``
~~~~~~~~~~~~~~~~~~~~~

If the class defines ``__init__``, it replaces the generated one. When it
runs, the instance already exists, with the ``batch_size``, ``device`` and
``names`` that the caller passed as keyword arguments. Assign the fields as
attributes. To set the batch size from the data, assign ``self.batch_size``
after the fields:

  >>> class Points(TensorClass):
  ...     xyz: torch.Tensor
  ...
  ...     def __init__(self, n: int):
  ...         self.xyz = torch.zeros(n, 3)
  ...         self.batch_size = [n]
  >>> Points(4).batch_size
  torch.Size([4])
  >>> Points(4)[:2].batch_size
  torch.Size([2])

These rules apply to a custom ``__init__``:

- If it does not set a required field, the constructor raises
  ``TypeError: Points.__init__() did not set the required field 'xyz'``.
- Indexing, ``clone``, ``torch.stack`` and the other operations make new
  instances without a call to ``__init__``. So ``__init__`` must only fill
  the fields, and the fields must be enough to rebuild the instance.
- As in a dataclass, a custom ``__init__`` does not call ``__post_init__``.
  Call it yourself if you need it.
- ``super().__init__(...)`` with fields works when the parent class is a
  tensorclass with fields: it runs the generated ``__init__`` of the parent. If
  the parent is :class:`~tensordict.TensorClass` itself, a call with fields
  raises ``NotImplementedError``. Assign the fields instead.

Overriding ``__setattr__`` and ``__getitem__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``__setattr__`` or ``__getitem__`` that the class defines replaces the one
of the tensorclass. Call the method of ``super()`` for the cases that you do
not handle. A custom ``__setattr__`` also runs when the constructor sets the
fields:

  >>> class Clamped(TensorClass):
  ...     x: torch.Tensor
  ...
  ...     def __setattr__(self, key, value):
  ...         if key == "x":
  ...             value = value.clamp(0, 1)
  ...         super().__setattr__(key, value)
  >>> c = Clamped(x=torch.tensor([-1.0, 2.0]), batch_size=[2])
  >>> c.x
  tensor([0., 1.])

Non-tensor data: ``NonTensorData`` and ``MetaData``
---------------------------------------------------

A field can hold a value that is not a tensor, such as a string. The
tensorclass stores it in a wrapper. Two wrappers exist, and they differ when
you stack instances:

- :class:`~tensordict.NonTensorData` is the default. It keeps one value per
  batch element. Stacking instances gives a
  :class:`~tensordict.NonTensorStack`, and attribute access returns a list of
  the values, even when all the values are equal.
- :class:`~tensordict.MetaData` holds one value for the whole batch. Stacking
  ``MetaData("cfg")`` with ``MetaData("cfg")`` returns ``MetaData("cfg")``.
  Stacking items with different values returns a ``NonTensorStack``, as for
  ``NonTensorData``. Use ``MetaData`` for a value that is the same for every
  element, such as a label or a configuration dict.

To use ``MetaData``, assign a ``MetaData`` instance to the field:

  >>> from tensordict import MetaData
  >>> class Sample(TensorClass):
  ...     x: torch.Tensor
  ...     label: str
  >>> a = Sample(x=torch.zeros(()), label="cfg")
  >>> torch.stack([a, a]).label
  LinkedList(['cfg', 'cfg'])
  >>> m = Sample(x=torch.zeros(()), label=MetaData("cfg"))
  >>> torch.stack([m, m]).label
  'cfg'

``LinkedList`` is a subclass of :class:`list`. For the full rules of
non-tensor values in tensordicts, see :ref:`non-tensor-data`.

Fields set to ``None``
----------------------

A field can be ``None`` at first and get a value later. The value that you
assign decides how the field is stored: a tensor is stored as a tensor, and a
string as ``NonTensorData``:

  >>> from typing import Any
  >>> class Slots(TensorClass):
  ...     X: Any
  ...     y: Any
  >>> slots = Slots(X=None, y=None, batch_size=[3, 4])
  >>> slots.X = torch.ones(3, 4, 5)
  >>> slots.y = "testing"
  >>> print(slots)
  Slots(
      X=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
      y=NonTensorData(data=testing, batch_size=torch.Size([3, 4]), device=None),
      batch_size=torch.Size([3, 4]),
      device=None,
      is_shared=False)

Comparison, item assignment and concatenation
---------------------------------------------

``==`` and ``!=`` compare two tensorclasses field by field, also the nested
ones. The result is a tensorclass with a boolean tensor for each field. For a
field that is not a tensor, the boolean tensor has the batch shape, and it is
``False`` where the values differ:

  >>> print(data == data2)
  MyData(
      floatdata=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.bool, is_shared=False),
      intdata=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
      nested=MyData(
          floatdata=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.bool, is_shared=False),
          intdata=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
          non_tensordata=Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.bool, is_shared=False),
          nested=None,
          batch_size=torch.Size([3, 4]),
          device=None,
          is_shared=False),
      non_tensordata=Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.bool, is_shared=False),
      batch_size=torch.Size([3, 4]),
      device=None,
      is_shared=False)
  >>> data2.non_tensordata = "test_new"
  >>> (data == data2).non_tensordata.any()
  tensor(False)

An item assignment writes the values that are not tensors element by element,
with no warning when they differ from the rest of the batch. The field then
holds one value per element (a :class:`~tensordict.NonTensorStack`), and
attribute access returns a nested list of those values, with one level per
batch dimension:

  >>> data[0] = data2[0]
  >>> [row[0] for row in data.non_tensordata]
  ['test_new', 'test', 'test']

``torch.cat`` and ``torch.stack`` keep the non-tensor value of each element.
If the inputs differ on a non-tensor field, the field of the result is a
``NonTensorStack``, and attribute access returns the nested list of values.
``torch.stack`` does this even when all the inputs have the same value.
``torch.cat`` of inputs that have one same value keeps that value:

  >>> cat_tc = torch.cat([data, data2], dim=0)
  >>> [row[0] for row in cat_tc.non_tensordata]
  ['test_new', 'test', 'test', 'test_new', 'test_new', 'test_new']

Converting a dataclass
----------------------

If you have a ``@dataclass``, :func:`~tensordict.from_dataclass` makes a
tensorclass type or instance from it. You do not change the class definition:

  >>> from dataclasses import dataclass
  >>> from tensordict import from_dataclass
  >>> @dataclass
  ... class X:
  ...     a: int
  ...     b: torch.Tensor
  >>> XTc = from_dataclass(X, autocast=True)  # convert the type
  >>> x_tc = from_dataclass(X(a=0, b=torch.zeros(())))  # convert an instance
  >>> XTc(a=torch.tensor(1), b=torch.zeros(())).a
  1

It accepts the flags of a tensorclass (``autocast``, ``nocast``, ``frozen``,
``tensor_only``, ``shadow``).

Saving and loading
------------------

The :meth:`~tensordict.TensorClass.memmap` method saves a tensorclass instance
to a directory. Tensors are written as memory-mapped tensors, and non-tensor
values that JSON can encode are written as JSON. Other non-tensor values are
written with Python ``pickle`` (to an ``other.pickle`` file, or
``pickle.pkl`` for stacked values).

:meth:`~tensordict.TensorDict.load_memmap` loads it. The result has the
original class if the loading process can import that class:

  >>> import tempfile
  >>> with tempfile.TemporaryDirectory() as path:
  ...     _ = data.memmap(path)
  ...     data_loaded = TensorDict.load_memmap(path)
  ...     assert isinstance(data_loaded, MyData)
  ...     assert (data_loaded == data).all()

The class is saved by its module and name. Another program cannot find a
class that you defined in ``__main__`` (a script or a notebook) under that
name. To load such data, call ``load_memmap`` on the class itself, for example
``MyData.load_memmap(path)``. ``load_memmap`` uses the class that you call it
on if its name matches the saved name.

If the saved data has pickled non-tensor values and you do not pass
``allow_pickle``, loading emits a ``FutureWarning`` in tensordict 0.14. In
0.15, the default becomes ``allow_pickle=False``, which raises an error
instead of loading them.
Pass ``allow_pickle=True`` only for data that you trust. See :ref:`saving` for
the other formats.

Tensorclasses as module inputs and outputs
------------------------------------------

For a module whose inputs and outputs are tensorclasses, subclass
:class:`~tensordict.nn.TensorClassModuleBase`. Its generic parameters declare
the input and output types, so your IDE completes the fields and a type
checker reports a wrong one. Its ``as_td_module()`` method returns a module
that you can put in a :class:`~tensordict.nn.TensorDictSequential`. See
:ref:`modules-guide` for an example.

``torch.compile``
-----------------

Tensorclasses work with :func:`torch.compile`, :mod:`torch.func` and
:func:`torch.export.export`. :ref:`compile` lists what makes a function
recompile or break its graph, and the workarounds. Two of these limits are
specific to tensorclasses: a string field that the function returns, and a
new tensorclass defined after the first call.

.. _tensorclass-static-typing:

Static type checking
--------------------

Type checkers read the fields of a :class:`~tensordict.TensorClass` subclass
as its constructor signature, as they do for a ``@dataclass``. The
keyword-only ``batch_size``, ``device``, ``names`` and ``lock`` arguments
follow the fields. A missing field, an unknown keyword argument or a value of
the wrong type is an error:

  >>> class Obs(TensorClass):
  ...     a: torch.Tensor
  ...     label: str = "x"
  >>> obs = Obs(torch.zeros(3), batch_size=[3])  # accepted
  >>> obs = Obs(a=[0.0, 1.0, 2.0], batch_size=[3])  # error: a list is not a Tensor

The constructor converts a list, a NumPy array or a Python number that you
pass to a ``torch.Tensor`` field, but a type checker expects a tensor there.
Pass ``torch.as_tensor(value)``. Tensordict methods such as ``reshape`` and
an index return the type of the class.

The three forms of the class options differ for type checkers:

- pyright accepts the bracket form (``TensorClass["autocast"]``) and the
  keyword form (``class Foo(TensorClass, autocast=True)``).
- mypy does not evaluate :meth:`~object.__class_getitem__` in a list of base
  classes and rejects the bracket form. Use the keyword form in code that mypy
  checks.
- Type checkers do not understand the decorator form. They know the fields of
  a ``@tensorclass`` class, but not its tensordict methods or its
  ``batch_size`` and ``device`` arguments. Use a subclass of ``TensorClass``
  in code that a type checker reads.

mypy reports each ``frozen=True`` tensorclass as "Frozen dataclass cannot
inherit from a non-frozen dataclass". These statements hold for pyright
1.1.414 and mypy 2.4.0.

.. _tensorclass-legacy-decorator:

Legacy: the ``@tensorclass`` decorator
--------------------------------------

The :func:`~tensordict.tensorclass` decorator is the original way to declare a
tensorclass. It is kept for backward compatibility, and it is the shortest way
to convert a body of ``@dataclass`` code. In new code, subclass
:class:`~tensordict.TensorClass` instead: it has the same behavior, and type
checkers understand it.

  >>> @tensorclass
  ... class LegacyData:
  ...     floatdata: torch.Tensor
  ...     intdata: torch.Tensor
  ...     non_tensordata: str
  ...     nested: Optional["LegacyData"] = None

The decorator accepts each flag as a keyword argument:
``@tensorclass(autocast=True)``, ``@tensorclass(frozen=True, tensor_only=True)``.

See also
--------

- :ref:`api-tensorclass`: the reference of :class:`~tensordict.TensorClass`,
  :func:`~tensordict.tensorclass`, :func:`~tensordict.from_dataclass` and the
  non-tensor classes.
- :ref:`typed-tensordict-guide`: typed fields on a
  :class:`~tensordict.TensorDictBase` subclass.
- :ref:`choosing-a-container`: which container to use.
- :ref:`compatibility`: tensorclasses on HDF5, Redis, lazy stacks and
  memory-mapped tensordicts.
- The tutorials :doc:`tutorials/tensorclass_fashion` and
  :doc:`tutorials/tensorclass_imagenet`.
