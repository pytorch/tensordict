.. _operations:

Arithmetic, apply, reductions, and gradients
============================================

Arithmetic, :mod:`torch` functions, and reductions on a tensordict apply to
every leaf in one call, and gradients flow through them. A leaf is a tensor at
the end of a key path (see :ref:`keys`). You write the computation once for
the whole group of tensors, and the result keeps the keys and the batch size
(see :ref:`batch-size`).

A :class:`~tensordict.TensorClass` has the same operations, so a method that
you write with them runs on one element or on a whole batch (see
:ref:`tensorclass-guide`).

Pointwise operations
--------------------

A pointwise operation applies to each leaf and returns a new tensordict with
the same keys. The other operand can be a tensordict, a tensor, or a Python
number. In this example, a weather grid of 3 by 4 cells has a pressure and a
2-D wind velocity for each cell:

    >>> import torch
    >>> from tensordict import TensorDict
    >>> td = TensorDict(
    ...     {
    ...         "pressure": torch.ones(3, 4),
    ...         "wind": {"velocity": torch.ones(3, 4, 2)},
    ...     },
    ...     batch_size=[3, 4],
    ... )
    >>> (td * 2 + td)["wind", "velocity"][0, 0]
    tensor([3., 3.])
    >>> (td - torch.ones(4)).batch_size
    torch.Size([3, 4])
    >>> (td > 0)["pressure"].dtype
    torch.bool

The Python operators work as follows:

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Operators
     - Tensordict on the left
     - Tensordict on the right
   * - ``+``, ``-``, ``*``, ``/``
     - yes
     - yes (``1 - td``, ``2 / td``)
   * - ``**``
     - yes
     - no: ``2 ** td`` raises ``NotImplementedError``
   * - ``%``
     - yes
     - no: ``2 % td`` raises ``TypeError``
   * - ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``
     - yes
     - yes
   * - ``&``, ``|``, ``^``
     - yes
     - yes

Comparisons return a tensordict with ``torch.bool`` leaves. The unary
operators ``-`` and ``~`` and the function :func:`abs` also work.

``//`` and ``@`` are not supported: they raise ``TypeError``. To use them,
call :meth:`~tensordict.TensorDict.apply` (see below).

Most pointwise functions of :mod:`torch` also have a tensordict method:
:meth:`~tensordict.TensorDict.exp`, :meth:`~tensordict.TensorDict.sqrt`,
:meth:`~tensordict.TensorDict.clamp`, :meth:`~tensordict.TensorDict.sigmoid`,
:meth:`~tensordict.TensorDict.lerp`, and many more. The methods with a trailing
underscore, such as :meth:`~tensordict.TensorDict.add_`, write into the
leaves (see :ref:`inplace-conventions`). :ref:`api-methods-pointwise` lists
them all.

    >>> TensorDict({"x": torch.tensor([-2.0, 0.5, 3.0])}).clamp(0, 1)["x"]
    tensor([0.0000, 0.5000, 1.0000])

Arithmetic, the pointwise methods, and the comparisons ``<``, ``<=``, ``>``,
and ``>=`` keep non-tensor entries, such as strings, as they are (see
:ref:`non-tensor-data`). ``==`` and ``!=`` between two tensordicts also
compare the non-tensor values, and give a boolean tensor at their keys.

When it can, tensordict computes a pointwise operation with the fused
``torch._foreach_<op>`` kernels, one call for all the leaves. The
:mod:`torch.func` transforms, such as :func:`torch.vmap`, have no batching
rule for the ``_foreach`` ops. Inside them, tensordict applies the operation
to one leaf at a time. Inside :func:`torch.compile`, ``td * t`` with a tensor ``t`` breaks
the graph: call ``td.mul(t)`` instead. :ref:`compile` gives the details.

Broadcasting with a tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~

A tensor operand is aligned with the batch dimensions of the tensordict, not
with the full shape of each leaf:

1. The batch size and the shape of the tensor broadcast together, aligned on
   the right, as two tensor shapes do. The result has this broadcast batch
   size.
2. For each leaf, the tensor gets trailing dimensions of size 1, so it
   broadcasts over the feature dimensions of that leaf (the dimensions after
   the batch dimensions).

In this example, ``td.batch_size`` is ``(3, 4)`` and ``td["wind", "velocity"]``
has shape ``(3, 4, 2)``. ``td * t`` multiplies
``td["wind", "velocity"][i, j, k]`` by ``t[j]``:

    >>> t = torch.arange(4.0)
    >>> (td * t)["wind", "velocity"][1, 2]
    tensor([2., 2.])

A tensor with more dimensions than the batch size gives a result with a
larger batch size:

    >>> (td * torch.ones(2, 3, 4)).batch_size
    torch.Size([2, 3, 4])

A tensor that matches only the feature dimensions does not broadcast. Its
shape must broadcast with the batch size:

    >>> td * torch.ones(2)
    Traceback (most recent call last):
    ...
    RuntimeError: Attempting to broadcast a dimension of length 2 at -1! ...

To scale the feature dimensions, use ``td.apply(lambda x: x * w)``.
In-place operators, such as ``td *= t``, follow the same rules.

Broadcasting between tensordicts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The batch sizes of two tensordicts must broadcast, aligned on the right.
Batch sizes ``(3,)`` and ``(3, 4)`` do not broadcast, even if the leaves have
the same shape:

    >>> td2 = TensorDict({"a": torch.ones(3, 4)}, batch_size=[3])
    >>> TensorDict({"a": torch.ones(3, 4)}, batch_size=[3, 4]) + td2
    Traceback (most recent call last):
    ...
    RuntimeError: Attempting to broadcast a dimension of length 3 at -1! ...

If the two batch sizes have a different number of dimensions, the shape of
the leaves in the result can depend on the order of the operands:

    >>> td0 = TensorDict({"x": torch.ones(3, 2)}, batch_size=[])
    >>> td1 = TensorDict({"x": torch.ones(3, 2)}, batch_size=[3])
    >>> (td1 + td0)["x"].shape
    torch.Size([3, 2])
    >>> (td0 + td1)["x"].shape
    torch.Size([3, 3, 2])

``td0 + td1`` follows the batch rules: ``td0`` has no batch dimensions, so
its entry is used for each of the 3 elements of ``td1``, as with
``td0.expand(3)``. ``td1 + td0`` broadcasts the two leaves as plain tensors.
To get the same result in both orders, give the two tensordicts the same
batch size first, for example with :meth:`~tensordict.TensorDict.expand`:

    >>> (td1 + td0.expand(3))["x"].shape
    torch.Size([3, 3, 2])

Missing entries
~~~~~~~~~~~~~~~

An operator between two tensordicts needs the same keys on both sides.
Otherwise it raises a ``KeyError``. The methods
:meth:`~tensordict.TensorDict.add`, :meth:`~tensordict.TensorDict.sub`,
:meth:`~tensordict.TensorDict.mul`, :meth:`~tensordict.TensorDict.div`,
:meth:`~tensordict.TensorDict.pow`, :meth:`~tensordict.TensorDict.maximum`,
:meth:`~tensordict.TensorDict.minimum`,
:meth:`~tensordict.TensorDict.clamp_max`, and
:meth:`~tensordict.TensorDict.clamp_min` accept a ``default`` keyword
argument for tensordicts with different keys:

- ``default=None`` (the default): the key sets must be the same.
- ``default="intersection"``: the result has only the keys of both
  tensordicts.
- A tensor, such as ``default=torch.tensor(0.0)``: the tensor replaces each
  missing entry, on both sides. A Python number (``default=0``) raises a
  ``TypeError``.

In this example, ``x`` and ``y`` share only the key ``"a"``:

    >>> x = TensorDict({"a": torch.ones(2), "b": torch.ones(2)}, batch_size=[2])
    >>> y = TensorDict({"a": torch.ones(2), "c": torch.ones(2)}, batch_size=[2])
    >>> x + y
    Traceback (most recent call last):
    ...
    KeyError: "keys in tensordicts mismatch. The first TD has keys {'b'} that the second does not have. The second TD has keys {'c'} that the first does not have."
    >>> list(x.add(y, default="intersection").keys())
    ['a']
    >>> sorted(x.add(y, default=torch.tensor(0.0)).keys())
    ['a', 'b', 'c']

Apply a function to each leaf
-----------------------------

:meth:`~tensordict.TensorDict.apply` calls a function on each leaf and
returns a new tensordict with the results, at the same keys. Use it for an
operation that has no tensordict method:

    >>> td = TensorDict(
    ...     {
    ...         "pressure": torch.ones(3, 4),
    ...         "wind": {"velocity": torch.ones(3, 4, 2)},
    ...     },
    ...     batch_size=[3, 4],
    ... )
    >>> td.apply(lambda x: x // 2)["pressure"][0]
    tensor([0., 0., 0., 0.])

The function gets tensors only. Non-tensor entries go to the result as they
are. These arguments change what ``apply`` does:

- Other tensordicts after the function: ``td.apply(fn, other)`` calls
  ``fn(x, y)`` with the leaves of ``td`` and ``other`` at the same key. If
  ``other`` has no entry at a key, ``apply`` raises a ``KeyError``, unless you
  give ``default=``.
- ``batch_size=``: if the function changes the batch dimensions, give the new
  batch size.
- A function that returns ``None`` for a leaf drops that entry from the
  result.
- ``inplace=True``, or :meth:`~tensordict.TensorDict.apply_`: write the
  results into the existing leaves and return the same tensordict.
- ``out=``: write the results into another tensordict.

The next examples use another tensordict, ``batch_size=``, and a function that
returns ``None``:

    >>> td.apply(lambda x, y: x + y, td)["pressure"][0, 0]
    tensor(2.)
    >>> td.apply(lambda x: x.sum(-1), batch_size=[3]).batch_size
    torch.Size([3])
    >>> list(td.apply(lambda x: x if x.ndim == 2 else None).keys(True, True))
    ['pressure']

:meth:`~tensordict.TensorDict.named_apply` also passes the key to the
function, as its first argument. By default, the key is the last string of the
key path. With ``nested_keys=True``, it is the full key, a tuple for a nested
entry:

    >>> calm = td.named_apply(lambda key, x: x * 0 if key == "velocity" else x)
    >>> calm["wind", "velocity"][0, 0, 0]
    tensor(0.)
    >>> _ = td.named_apply(lambda key, x: print(key), nested_keys=True)
    pressure
    ('wind', 'velocity')

Reductions
----------

The reductions of :mod:`torch` are tensordict methods too:
:meth:`~tensordict.TensorDict.sum`, :meth:`~tensordict.TensorDict.mean`,
:meth:`~tensordict.TensorDict.std`, :meth:`~tensordict.TensorDict.var`,
:meth:`~tensordict.TensorDict.prod`, :meth:`~tensordict.TensorDict.min`,
:meth:`~tensordict.TensorDict.max`, :meth:`~tensordict.TensorDict.amin`,
:meth:`~tensordict.TensorDict.amax`, :meth:`~tensordict.TensorDict.nansum`,
:meth:`~tensordict.TensorDict.nanmean`,
:meth:`~tensordict.TensorDict.quantile`,
:meth:`~tensordict.TensorDict.logsumexp`, :meth:`~tensordict.TensorDict.norm`,
:meth:`~tensordict.TensorDict.all`, and :meth:`~tensordict.TensorDict.any`.
The ``dim`` argument selects what they reduce:

- No ``dim``: each leaf is reduced to one value. The result has batch size
  ``[]``.
- ``dim=0``, ``dim=-1``, or a tuple: these are batch dimensions. Each leaf is
  reduced over them, and they are removed from the batch size of the result.
  A ``dim`` that is not a batch dimension raises an ``IndexError``.
- ``dim="feature"``: each leaf is reduced over its feature dimensions. The
  result keeps the batch size.

With ``reduce=True``, the reduction also combines the leaves, and the result
is one tensor:

    >>> td.sum()["wind", "velocity"]
    tensor(24.)
    >>> td.sum(dim=0)["wind", "velocity"].shape
    torch.Size([4, 2])
    >>> td.sum(dim="feature")["wind", "velocity"].shape
    torch.Size([3, 4])
    >>> td.sum(reduce=True)
    tensor(36.)
    >>> td.sum(dim="feature", reduce=True).shape
    torch.Size([3, 4])

:meth:`~tensordict.TensorDict.min` and :meth:`~tensordict.TensorDict.max`
with a ``dim`` return a named tuple of two tensordicts, ``values`` and
``indices``, as :func:`torch.max` does. :meth:`~tensordict.TensorDict.all`
and :meth:`~tensordict.TensorDict.any` without ``dim`` return a Python
``bool``:

    >>> td.max(dim=0).indices["pressure"].dtype
    torch.int64
    >>> (td == 1).all()
    True

.. _inplace-conventions:

In-place operations
-------------------

tensordict has two kinds of in-place operation:

- A method with a trailing underscore (``method_``), such as
  :meth:`~tensordict.TensorDict.add_`, :meth:`~tensordict.TensorDict.mul_`,
  or :meth:`~tensordict.TensorDict.masked_fill_`, writes into the same tensor
  storage for every leaf. The ``data_ptr`` of each leaf does not change. This
  is the convention of PyTorch. These methods exist only for operations that
  keep the shape, dtype, and layout of each tensor.
- The ``inplace=True`` keyword of a shape-changing operation, such as
  :meth:`~tensordict.TensorDict.pad`, :meth:`~tensordict.TensorDict.repeat`,
  :meth:`~tensordict.TensorDict.gather`, or
  :meth:`~tensordict.TensorDict.repeat_interleave`, keeps the tensordict
  object and its keys. Each leaf can get a new tensor. The operation
  allocates no second tensordict, and it releases each old leaf as soon as it
  writes the new one. Peak memory therefore stays close to ``1x``, not
  ``2x``. A shape-changing operation can offer only this kind of in-place.

In short, ``method_`` keeps the bytes, and ``inplace=True`` on a
shape-changing operation keeps the container.

    >>> td = TensorDict({"a": torch.ones(3, 4)}, batch_size=[3, 4])
    >>> ptr = td["a"].data_ptr()
    >>> _ = td.add_(1)
    >>> td["a"].data_ptr() == ptr
    True
    >>> padded = td.pad([0, 1], inplace=True)
    >>> padded is td, td.batch_size, td["a"].data_ptr() == ptr
    (True, torch.Size([4, 4]), False)

For :meth:`~tensordict.TensorDict.set`, :meth:`~tensordict.TensorDict.update`,
and :meth:`~tensordict.TensorDict.apply`, ``inplace=True`` writes into the
existing leaf when the key exists, so its ``data_ptr`` does not change. A
value of another dtype is cast to the dtype of the leaf. ``set()`` and
``update()`` add a missing key, and :meth:`~tensordict.TensorDict.set_` and
:meth:`~tensordict.TensorDict.update_` raise a ``KeyError`` for it (see
:ref:`keys-write`).

    >>> ptr = td["a"].data_ptr()
    >>> _ = td.set("a", torch.zeros(4, 4), inplace=True)
    >>> td["a"].data_ptr() == ptr
    True
    >>> td.set_("missing", torch.zeros(4, 4))
    Traceback (most recent call last):
    ...
    KeyError: 'key "missing" not found in TensorDict with keys ...

These methods follow neither of the two kinds:

- :meth:`~tensordict.TensorDict.memmap_` and
  :meth:`~tensordict.TensorDict.share_memory_` have a trailing underscore,
  but they move each leaf to new storage: memory-mapped files, or shared
  memory. The ``data_ptr`` of each leaf changes, and the tensordict gets
  locked (see :ref:`keys-locking`). For the same reason,
  :meth:`~tensordict.TensorDict.pin_memory_` is deprecated: use
  ``pin_memory(inplace=True)``.
- :meth:`~tensordict.TensorDict.lock_`, :meth:`~tensordict.TensorDict.unlock_`,
  :meth:`~tensordict.TensorDict.rename_key_`, :meth:`~tensordict.TensorDict.del_`,
  :meth:`~tensordict.TensorDict.auto_batch_size_`,
  :meth:`~tensordict.TensorDict.auto_device_`, and
  :meth:`~tensordict.TensorDict.clear_device_` change the tensordict
  itself: its lock, its keys, its batch size, or its device. They do not
  touch the leaves.
- On the key methods :meth:`~tensordict.TensorDict.select`,
  :meth:`~tensordict.TensorDict.exclude`, and
  :meth:`~tensordict.TensorDict.split_keys`, ``inplace=True`` removes the
  other keys from the tensordict and returns it. On
  :meth:`~tensordict.TensorDict.flatten_keys` and
  :meth:`~tensordict.TensorDict.unflatten_keys`, it renames the keys of the
  tensordict and returns it.
- On :meth:`~tensordict.TensorDict.reshape`,
  :meth:`~tensordict.TensorDict.flatten`,
  :meth:`~tensordict.TensorDict.unflatten`, and
  :meth:`~tensordict.TensorDict.contiguous`, ``inplace=True`` keeps the
  object, as for the other shape-changing operations. A leaf keeps its
  storage when the operation can return a view of it.

For example, ``select(inplace=True)`` returns the same tensordict, with fewer
keys:

    >>> td = TensorDict({"a": torch.ones(2), "b": TensorDict({"c": torch.ones(2)})})
    >>> td.select("a", inplace=True) is td, list(td.keys())
    (True, ['a'])

Gradients
---------

Tensordicts work with :mod:`torch.autograd`: a leaf that requires gradients
keeps its graph through the operations above.

``torch.autograd.grad``
~~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.autograd.grad` accepts tensordicts, also nested ones, for its
``outputs``, ``inputs``, and ``grad_outputs``. tensordict passes the leaves to
PyTorch as tuples, and returns the gradients as a new tensordict with the
keys of ``inputs``:

    >>> inputs = TensorDict({"a": torch.ones(2, 3, requires_grad=True)})
    >>> outputs = inputs * 2.0
    >>> grads = torch.autograd.grad(outputs, inputs, torch.ones_like(outputs))
    >>> grads["a"]
    tensor([[2., 2., 2.],
            [2., 2., 2.]])

``outputs`` and ``inputs`` must each be one tensordict. A tensor output with
tensordict inputs raises ``ValueError: torch.autograd.grad for TensorDict
only supports TensorDictBase as output``. To differentiate a scalar loss,
put it in a tensordict:

    >>> params = TensorDict({"w": torch.ones(3, requires_grad=True)})
    >>> loss = (params["w"] ** 2).sum()
    >>> torch.autograd.grad(TensorDict({"loss": loss}), params)["w"]
    tensor([2., 2., 2.])

This also works for a module that reads and writes tensordicts, such as a
:class:`~tensordict.nn.TensorDictModule` (see :ref:`modules-guide`). Select
the output keys of the module, so that the inputs are not part of the
outputs:

    >>> from tensordict.nn import TensorDictModule
    >>> _ = torch.manual_seed(0)
    >>> module = TensorDictModule(torch.nn.Linear(3, 2), in_keys=["x"], out_keys=["y"])
    >>> td_in = TensorDict({"x": torch.randn(4, 3, requires_grad=True)}, batch_size=[4])
    >>> td_out = module(td_in.clone()).select(*module.out_keys)
    >>> grads = torch.autograd.grad(td_out, td_in, torch.ones_like(td_out))
    >>> grads["x"].shape
    torch.Size([4, 3])

``TensorDict.backward``
~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~tensordict.TensorDict.backward` works like
:meth:`torch.Tensor.backward`. It collects the differentiable leaves of the
tensordict, also nested ones, and backpropagates through all of them in one
:func:`torch.autograd.backward` call. If every leaf is a scalar, you do not
need to give a gradient. The usual case is a tensordict of losses, here a fit term and a
regularization term:

    >>> w = torch.ones(3, requires_grad=True)
    >>> loss_td = TensorDict({"fit_loss": (w ** 2).sum(), "reg_loss": w.sum()})
    >>> loss_td.backward()  # same as loss_td.sum(reduce=True).backward()
    >>> w.grad
    tensor([3., 3., 3.])

For leaves that are not scalars, or to weight each loss, pass a gradient
tensordict whose entries match the differentiable leaves by nested key:

    >>> w.grad = None
    >>> loss_td = TensorDict({"fit_loss": (w ** 2).sum(), "reg_loss": w.sum()})
    >>> weights = TensorDict({"fit_loss": torch.tensor(0.5), "reg_loss": torch.tensor(1.0)})
    >>> loss_td.backward(weights)
    >>> w.grad
    tensor([2., 2., 2.])

Jacobians and Hessians
~~~~~~~~~~~~~~~~~~~~~~

:func:`torch.func.jacrev`, :func:`torch.func.jacfwd`, and
:func:`torch.func.hessian` work on functions that take and return
tensordicts. The result is a tensordict in which the entry
``J["output_key", "input_key"]`` is the Jacobian block for that pair of keys:

    >>> from torch.func import jacrev
    >>> _ = torch.manual_seed(0)
    >>> td = TensorDict({"a": torch.randn(3, 2), "b": torch.randn(3, 4)}, batch_size=[3])
    >>> def f(td):
    ...     return TensorDict(
    ...         {"x": td["a"] ** 2, "y": td["b"] ** 3}, batch_size=td.batch_size
    ...     )
    >>> J = jacrev(f)(td)
    >>> J.batch_size  # preserved from input/output
    torch.Size([3])
    >>> J["x", "a"].shape  # output_shape + input_shape = (3, 2) + (3, 2)
    torch.Size([3, 2, 3, 2])

The three transforms can fail if the batch size is not empty and each leaf
has one element for each batch element. Examples are the shapes ``(4,)`` and
``(4, 1)`` with ``batch_size=[4]``. In this case, :func:`~torch.func.jacrev`
works only if the function returns two or more keys.
:func:`~torch.func.jacfwd` and :func:`~torch.func.hessian` fail for any number
of keys. The errors are ``RuntimeError: Expected pytree structure ...`` and
``RuntimeError: jvp(f, primals, tangents): Expected primals and tangents to
have the same python structure ...``.

    >>> def g(td):
    ...     return TensorDict({"x": td["a"] ** 2}, batch_size=td.batch_size)
    >>> jacrev(g)(TensorDict({"a": torch.randn(4)}, batch_size=[4]))
    Traceback (most recent call last):
    ...
    RuntimeError: Expected pytree structure of cotangents to be the same as pytree structure of outputs to the function...

To avoid the error, do one of these:

- Set ``batch_size=[]``, so that all the dimensions are feature dimensions.
- Give the leaves a feature dimension of size 2 or more. A dimension of
  size 1 (``torch.randn(4, 1)``) does not help.

With ``batch_size=[]``, the same function works:

    >>> jacrev(g)(TensorDict({"a": torch.randn(4)}, batch_size=[]))["x", "a"].shape
    torch.Size([4, 4])

:mod:`torch.func` also has :func:`torch.func.grad` and :func:`torch.vmap`
for tensordicts. :ref:`compile` explains how they work with tensordicts, and
how to compute per-sample gradients and model ensembles.

See also
--------

- :ref:`api-methods-pointwise`, :ref:`api-methods-reductions`, and
  :ref:`api-methods-functional`: the reference entries of the methods on
  this page.
- :ref:`stacking`: combine tensordicts with :func:`torch.stack` and
  :func:`torch.cat`.
- :ref:`devices`: cast the dtype and device of the leaves.
- :ref:`compile`: these operations under :func:`torch.compile` and
  :mod:`torch.func`.
- The :doc:`/tutorials/tensordict_shapes` tutorial: shape operations on
  tensordicts.
