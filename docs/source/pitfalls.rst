.. _pitfalls:

Common surprises
================

A TensorDict has the methods of a ``dict`` and the operations of a tensor.
Where the two models differ, it follows the tensor, and new users then get a
result that they did not expect. This page lists the cases that users
report most often. Each one has a short example and a link to the page that
explains it.

Run the examples in order in one Python session:

  >>> import dataclasses
  >>> import torch
  >>> from tensordict import NonTensorData, TensorClass, TensorDict, is_floating

``len``, ``bool``, and ``==`` do not look at the keys
-----------------------------------------------------

``len(td)`` is the size of the first batch dimension, and 0 for a tensordict
without batch dimensions, whatever its keys. ``len(td.keys())`` counts the
keys. ``bool(td)`` raises. ``td == other`` compares the entries and returns a
tensordict of booleans: call ``.all()`` on it to get a Python ``bool``:

  >>> td = TensorDict({"a": torch.zeros(3, 2)}, batch_size=[3])
  >>> len(td), len(TensorDict({"a": torch.zeros(3, 2)})), len(td.keys())
  (3, 0, 1)
  >>> bool(td)
  Traceback (most recent call last):
  ...
  RuntimeError: Converting a tensordict to boolean value is not permitted. ...
  >>> (td == td.clone()).batch_size
  torch.Size([3])
  >>> (td == td.clone()).all()
  True

:ref:`batch-size-not-a-dict` lists the other operations that differ from a
``dict``.

The batch size is not inferred
------------------------------

If you do not give a batch size, it is ``torch.Size([])``: TensorDict does
not infer it from the entries.
:meth:`~tensordict.TensorDict.auto_batch_size_` infers it:

  >>> td = TensorDict({"x": torch.zeros(3, 4), "name": "run-1"})
  >>> td.batch_size
  torch.Size([])
  >>> td.auto_batch_size_().batch_size
  torch.Size([3, 4])

A Python ``int``, ``float``, or ``bool`` becomes a tensor with no
dimensions, and a string becomes a :class:`~tensordict.NonTensorData`. A
tensor with no dimensions has no batch dimensions to share, so it keeps
``auto_batch_size_`` from inferring any:

  >>> td = TensorDict({"x": torch.zeros(3, 4), "scale": 2.0, "name": "run-1"})
  >>> type(td.get("scale")).__name__, type(td.get("name")).__name__
  ('Tensor', 'NonTensorData')
  >>> td.auto_batch_size_().batch_size
  torch.Size([])

For the same reason, a tensordict with batch dimensions raises for a Python
number. To store one value for the whole batch, wrap it in a
:class:`~tensordict.NonTensorData`:

  >>> td = TensorDict({"x": torch.zeros(3, 4)}, batch_size=[3])
  >>> td["scale"] = 2.0
  Traceback (most recent call last):
  ...
  RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([3]) and value.shape=torch.Size([]) for key 'scale'. ...
  >>> td["scale"] = NonTensorData(2.0)
  >>> td["scale"]
  2.0

:ref:`batch-size` explains the batch size, and :ref:`non-tensor-data`
explains the values that are not tensors.

Broadcasting aligns with the batch dimensions
---------------------------------------------

A tensor operand aligns with the batch dimensions of the tensordict, not
with the last dimensions of each leaf. Here the tensor of shape ``(3,)``
multiplies the rows of a leaf of shape ``(3, 2)``:

  >>> td = TensorDict({"a": torch.ones(3, 2)}, batch_size=[3])
  >>> (td * torch.tensor([0.0, 1.0, 2.0]))["a"]
  tensor([[0., 0.],
          [1., 1.],
          [2., 2.]])

To scale the last dimensions of each leaf, use
``td.apply(lambda x: x * w)``. If two tensordicts have batch sizes with
different numbers of dimensions, the result can depend on the order of the
operands:

  >>> td0 = TensorDict({"x": torch.ones(3, 2)}, batch_size=[])
  >>> td1 = TensorDict({"x": torch.ones(3, 2)}, batch_size=[3])
  >>> (td1 + td0)["x"].shape, (td0 + td1)["x"].shape
  (torch.Size([3, 2]), torch.Size([3, 3, 2]))

To get the same result in both orders, give the two tensordicts the same
batch size first, for example with ``td0.expand(3)``. :ref:`operations`
gives the broadcasting rules.

``td * t`` breaks the graph under ``torch.compile``
---------------------------------------------------

Inside :func:`torch.compile`, an operator between a tensordict and a tensor,
such as ``td * t`` or ``td + t``, breaks the graph. With ``fullgraph=True``,
it raises. Call the method instead, such as
:meth:`~tensordict.TensorDict.mul` or :meth:`~tensordict.TensorDict.add`.
An operator with a Python number or with another tensordict compiles:

  >>> t = torch.tensor([0.0, 1.0, 2.0])
  >>> torch.compile(lambda td, t: td.mul(t), fullgraph=True)(td, t)["a"]
  tensor([[0., 0.],
          [1., 1.],
          [2., 2.]])
  >>> torch.compile(lambda td, t: td * t, fullgraph=True)(td, t)
  Traceback (most recent call last):
  ...
  torch._dynamo.exc.Unsupported: ...

:ref:`compile` lists the other patterns that break the graph.

``set``, ``set_``, ``update_``, and locks
-----------------------------------------

``td[key] = value`` and :meth:`~tensordict.TensorDict.set` replace the
tensor under the key. :meth:`~tensordict.TensorDict.set_` and
:meth:`~tensordict.TensorDict.update_` copy the values into the existing
tensor, and raise a ``KeyError`` for a missing key:

  >>> td = TensorDict({"pos": torch.zeros(3)}, batch_size=[3])
  >>> old = td["pos"]
  >>> td["pos"] = torch.ones(3)
  >>> old  # replaced: the old tensor did not change
  tensor([0., 0., 0.])
  >>> old = td["pos"]
  >>> _ = td.set_("pos", torch.full((3,), 2.0))
  >>> old  # copied into the same tensor
  tensor([2., 2., 2.])
  >>> td.update_({"vel": torch.zeros(3)})
  Traceback (most recent call last):
  ...
  KeyError: 'The key vel was not found in the dest tensordict.'

A locked tensordict raises for a new key, a replaced entry, and a removed
one. It accepts ``set_``, ``update_``, and in-place operations, which write into
the existing tensors. ``memmap`` and ``consolidate`` return a locked
tensordict:

  >>> _ = td.lock_()
  >>> td["pos"] = torch.ones(3)
  Traceback (most recent call last):
  ...
  RuntimeError: Cannot modify locked TensorDict. ...
  >>> _ = td.set_("pos", torch.ones(3))
  >>> td["pos"].add_(1)
  tensor([2., 2., 2.])

``td["pos"] += 1`` raises on a locked tensordict, but only after it changed
the tensor: Python runs it as ``td["pos"] = td["pos"].__iadd__(1)``. Use
``td["pos"].add_(1)`` instead. :ref:`keys-write` compares the ways to write
an entry, and :ref:`keys-locking` lists what a lock blocks.

Views and shared memory
-----------------------

Indexing gives views where :mod:`torch` does. An integer or a slice gives
views, so an in-place change of an entry of the result changes the
original. A mask or a tensor of indices gives copies:

  >>> td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
  >>> _ = td[:2]["a"].add_(1)  # a view: changes td
  >>> _ = td[torch.tensor([0, 1])]["a"].add_(1)  # a copy: does not
  >>> td["a"]
  tensor([1., 1., 0.])

:meth:`~tensordict.TensorDict.expand` allocates no memory. All the elements
along a new dimension are the same memory, so a write into one of them
writes into all. An assignment to an index, such as ``e[0] = 1``, writes
every row. An in-place operation on an entry, such as ``e["a"].add_(1)``,
raises a ``RuntimeError`` instead, as :ref:`batch-size` says. Call
``clone()`` first to get separate copies:

  >>> e = TensorDict({"a": torch.zeros(3)}, batch_size=[3]).expand(2, 3)
  >>> e[0] = 1
  >>> e["a"]
  tensor([[1., 1., 1.],
          [1., 1., 1.]])
  >>> e["a"].add_(1)
  Traceback (most recent call last):
  ...
  RuntimeError: unsupported operation: ...
  >>> e = e.clone()
  >>> e[0] = 0
  >>> e["a"]
  tensor([[0., 0., 0.],
          [1., 1., 1.]])

:meth:`~tensordict.TensorDict.consolidate` copies all the leaves into one
storage, and each entry of the result is a view of it. The result is a copy:
a change to the original does not reach it:

  >>> td = TensorDict({"a": torch.zeros(3), "b": torch.ones(3)}, batch_size=[3])
  >>> c = td.consolidate()
  >>> c["a"].untyped_storage().data_ptr() == c["b"].untyped_storage().data_ptr()
  True
  >>> _ = td["a"].add_(1)
  >>> c["a"]
  tensor([0., 0., 0.])

:ref:`batch-size` explains indexing and the shape operations, and
:ref:`saving` explains consolidation.

``to(dtype)`` casts every leaf
------------------------------

``td.to(dtype)`` also casts the integer and boolean leaves, such as labels
and masks. ``cast_filter`` selects the leaves to cast:
:func:`~tensordict.is_floating` selects the floating-point leaves only:

  >>> td = TensorDict(
  ...     {"x": torch.zeros(2), "label": torch.arange(2), "mask": torch.ones(2, dtype=torch.bool)},
  ...     batch_size=[2],
  ... )
  >>> {k: v.dtype for k, v in td.to(torch.float16).items()}
  {'x': torch.float16, 'label': torch.float16, 'mask': torch.float16}
  >>> {k: v.dtype for k, v in td.to(torch.float16, cast_filter=is_floating).items()}
  {'x': torch.float16, 'label': torch.int64, 'mask': torch.bool}

The dtype methods, such as ``td.half()``, take ``cast_filter`` too.
:ref:`devices` explains devices and dtypes.

A tensor default of a tensorclass field is shared
-------------------------------------------------

Python evaluates the default of a field once, when it creates the class. A
tensor default is therefore one tensor that every instance shares, and an
in-place change through one instance shows in all of them:

  >>> class Counter(TensorClass):
  ...     count: torch.Tensor = torch.zeros(3)
  >>> a, b = Counter(), Counter()
  >>> _ = a.count.add_(1)
  >>> b.count
  tensor([1., 1., 1.])

To give each instance its own tensor, use
:func:`dataclasses.field` with a ``default_factory``:

  >>> class Counter(TensorClass):
  ...     count: torch.Tensor = dataclasses.field(default_factory=lambda: torch.zeros(3))
  >>> a, b = Counter(), Counter()
  >>> _ = a.count.add_(1)
  >>> b.count
  tensor([0., 0., 0.])

A default of ``None`` that ``__post_init__`` replaces works too (see
:ref:`tensorclass-post-init`).
