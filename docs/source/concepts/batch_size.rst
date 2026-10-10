.. _batch-size:

Batch size, shapes and indexing
===============================

This page explains the batch size of a :class:`~tensordict.TensorDict`, and how
indexing and shape operations use it. It also lists where a TensorDict acts
like a tensor and not like a :class:`dict`: iteration, ``len``, ``bool``,
``in`` and ``==``. Read it before the other pages of the user guide.

Batch size and batch dimensions
-------------------------------

The batch size of a TensorDict is the shape that the leading dimensions of all
its entries share. These leading dimensions are the batch dimensions.
``batch_dims`` is their number. The other dimensions of an entry, after the
batch dimensions, are its feature dimensions. Each entry can have different
feature dimensions.

  >>> import torch
  >>> from tensordict import TensorDict
  >>> td = TensorDict(
  ...     obs=torch.zeros(2, 3, 4),
  ...     reward=torch.zeros(2, 3),
  ...     batch_size=[2, 3],
  ... )
  >>> td.batch_size
  torch.Size([2, 3])
  >>> td.batch_dims
  2
  >>> td["obs"].shape[td.batch_dims:]  # the feature dimensions of "obs"
  torch.Size([4])
  >>> td["reward"].shape[td.batch_dims:]
  torch.Size([])

``td.shape`` and ``td.size()`` return the batch size. ``td.ndim`` and
``td.dim()`` return ``batch_dims``. ``td.numel()`` is the number of elements
in the batch (here 6), not the number of values in the tensors.

The leading dimensions of each entry must be equal to the batch size. If they
are not, the constructor and each write raise a ``RuntimeError`` that names the
key:

  >>> td["done"] = torch.zeros(3)
  Traceback (most recent call last):
  ...
  RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([2, 3]) and value.shape=torch.Size([3]) for key 'done'. The leading dimensions of a value must match the batch size.

TensorDict stores a Python number as a 0-dim tensor. If the batch size is not
empty, a number therefore gives the same error. To store one number for each
element, expand it to the batch size. To store one number for the whole batch,
see :ref:`non-tensor-data`.

  >>> td["done"] = torch.tensor(False).expand(td.batch_size)
  >>> td["done"].shape
  torch.Size([2, 3])

A nested tensordict is an entry too. Its batch size must start with the batch
size of its parent, and it can have more batch dimensions. When you give a
dictionary as a value, the nested tensordict gets the batch size of the parent:

  >>> td["next"] = {"obs": torch.zeros(2, 3, 4)}
  >>> td["next"].batch_size
  torch.Size([2, 3])

To store an entry that does not have the batch dimensions, wrap it in an
:class:`~tensordict.UnbatchedTensor` (see :ref:`unbatched-tensor`).

Empty batch size
----------------

If you do not give a batch size, it is ``torch.Size([])``: no batch
dimensions. TensorDict does not infer it from the entries. With an empty batch
size, the entries can have any shape, but you cannot index the tensordict or
iterate over it:

  >>> data = TensorDict(obs=torch.zeros(3, 4), reward=torch.zeros(3))
  >>> data.batch_size
  torch.Size([])
  >>> data[0]
  Traceback (most recent call last):
  ...
  RuntimeError: indexing a tensordict with td.batch_dims==0 is not permitted. Got index 0.

``TensorDict()`` creates a tensordict with no entries and an empty batch size.

Infer the batch size
--------------------

To infer the batch size from the entries, do one of these:

- Pass ``auto_batch_size=True`` to :meth:`~tensordict.TensorDict.from_dict`.
- Call :meth:`~tensordict.TensorDict.auto_batch_size_` on an existing
  tensordict. It changes the tensordict and returns it.

Both set the batch size to the longest list of leading dimensions that all the
entries share. Use ``batch_dims`` to set a maximum number of batch dimensions:

  >>> TensorDict.from_dict(
  ...     {"obs": torch.zeros(3, 4), "reward": torch.zeros(3)}, auto_batch_size=True
  ... ).batch_size
  torch.Size([3])
  >>> data = TensorDict(obs=torch.zeros(3, 4, 5), mask=torch.zeros(3, 4))
  >>> data.auto_batch_size_().batch_size
  torch.Size([3, 4])
  >>> data.auto_batch_size_(batch_dims=1).batch_size
  torch.Size([3])

``auto_batch_size_`` also gives each nested tensordict the longest batch size
of its own entries.

Change the batch size
---------------------

You can assign a new batch size if all the entries are compatible with it. An
empty batch size is always compatible:

  >>> data = TensorDict(obs=torch.zeros(3, 4), batch_size=[3])
  >>> data.batch_size = [3, 4]
  >>> data.batch_size = [4]
  Traceback (most recent call last):
  ...
  RuntimeError: the Tensor obs has shape torch.Size([3, 4]) which is incompatible with the batch-size torch.Size([4]).

A :class:`~tensordict.LazyStackedTensorDict` (see :ref:`stacking`) does not
accept a new batch size, not even an empty one. Call
:meth:`~tensordict.LazyStackedTensorDict.to_tensordict` first.

Indexing
--------

An index on a TensorDict follows one rule:

- A string, or a tuple of strings, reads one entry (see :ref:`keys`).
- Any other index applies to the batch dimensions of all the entries at once:
  an integer, a slice, ``...``, ``None``, a list or tensor of integers, or a
  boolean mask.

The result of a batch index is a TensorDict. An index applies to the batch
dimensions, not to the dimensions of each entry. ``td[..., :2]["obs"]`` and
``td["obs"][..., :2]`` are therefore different:

  >>> td = TensorDict(obs=torch.zeros(2, 3, 4), reward=torch.zeros(2, 3), batch_size=[2, 3])
  >>> td[0].batch_size
  torch.Size([3])
  >>> td[..., :2]["obs"].shape  # the last batch dimension
  torch.Size([2, 2, 4])
  >>> td["obs"][..., :2].shape  # the last dimension of "obs"
  torch.Size([2, 3, 2])
  >>> td[:, None].batch_size
  torch.Size([2, 1, 3])

A boolean mask selects elements of the leading batch dimensions, as for a
tensor. Its shape must match those dimensions:

  >>> mask = torch.tensor([[True, False, True], [False, False, True]])
  >>> td[mask].batch_size
  torch.Size([3])
  >>> td[mask]["obs"].shape
  torch.Size([3, 4])
  >>> td[torch.tensor([True, False])].batch_size
  torch.Size([1, 3])

A list of strings is not a key selection. To keep some keys, use
:meth:`~tensordict.TensorDict.select` (see :ref:`keys`).

Indexing follows the rules of :mod:`torch` for storage. An integer or a slice
gives views, so an in-place change of an entry of the result changes the
original. A mask or a tensor of indices gives copies. A new key that you write
into the result does not appear in the original:

  >>> sub = td[0]
  >>> sub["reward"] += 1
  >>> td["reward"]
  tensor([[1., 1., 1.],
          [0., 0., 0.]])
  >>> sub["new"] = torch.zeros(3)
  >>> "new" in td
  False

To write into the batch elements of the original, assign to an index. The
value can be a TensorDict with the batch size of the indexed result, or a
number or tensor that broadcasts to each entry:

  >>> td[1] = TensorDict(obs=torch.ones(3, 4), reward=torch.ones(3), batch_size=[3])
  >>> td[:, 0] = 0
  >>> td["reward"]
  tensor([[0., 1., 1.],
          [0., 1., 1.]])

If you write a TensorDict into an index of an empty tensordict, the first write
creates the entries. See :ref:`stacking-preallocation`.

Shape operations
----------------

The shape operations of a TensorDict act on the batch dimensions. They apply
the same operation to each entry, and they do not change the feature
dimensions. The dimension arguments count batch dimensions only:

  >>> td = TensorDict(obs=torch.zeros(2, 3, 4), reward=torch.zeros(2, 3), batch_size=[2, 3])
  >>> td.reshape(6)["obs"].shape
  torch.Size([6, 4])
  >>> td.permute(1, 0)["obs"].shape
  torch.Size([3, 2, 4])
  >>> td.unsqueeze(-1)["obs"].shape  # a new last batch dimension, before the 4
  torch.Size([2, 3, 1, 4])
  >>> td.flatten(0, 1).unflatten(0, (2, 3)).batch_size
  torch.Size([2, 3])
  >>> [t.batch_size for t in td.unbind(1)]
  [torch.Size([2]), torch.Size([2]), torch.Size([2])]
  >>> [t.batch_size for t in td.split([2, 1], dim=1)]
  [torch.Size([2, 2]), torch.Size([2, 1])]

The other shape operations are :meth:`~tensordict.TensorDict.view`,
:meth:`~tensordict.TensorDict.transpose`,
:meth:`~tensordict.TensorDict.squeeze`,
:meth:`~tensordict.TensorDict.expand`,
:meth:`~tensordict.TensorDict.chunk`, :meth:`~tensordict.TensorDict.repeat`
and :meth:`~tensordict.TensorDict.repeat_interleave`. As for tensors,
``view``, ``permute``, ``unbind`` and ``expand`` return views of the same
storage. ``expand`` repeats the same memory, so an in-place write into an
expanded entry raises a ``RuntimeError``. Call ``clone()`` first to get
separate copies.

To join tensordicts with :func:`torch.stack` and :func:`torch.cat`, see
:ref:`stacking`.

Dimension names
---------------

You can give a name to each batch dimension, as with the named tensors of
PyTorch. ``names`` has one entry for each batch dimension. ``None`` means that
the dimension has no name. Give the names at construction, or set them later:

  >>> td = TensorDict(batch_size=[3, 4], names=["a", None])
  >>> td.names
  ['a', None]
  >>> td = td.refine_names(..., "b")
  >>> td.names
  ['a', 'b']
  >>> td.names = ["z", "y"]
  >>> td = td.rename("m", "n")
  >>> td.names
  ['m', 'n']
  >>> td = td.rename(m="h")
  >>> td.names
  ['h', 'n']

The names follow the shape operations:

  >>> td[0].names  # indexing removes the name of the indexed dimension
  ['n']
  >>> td.permute(1, 0).names
  ['n', 'h']
  >>> td.unsqueeze(0).names  # a new dimension has no name
  [None, 'h', 'n']
  >>> td.reshape(12).names
  [None]

If the number of names is not ``batch_dims``, TensorDict raises a ``ValueError``.
A nested tensordict has the names of its parent for the shared batch
dimensions. The shape operations take dimension numbers, not names.

.. _batch-size-not-a-dict:

A TensorDict is not a dict
--------------------------

A TensorDict has the mapping methods of a :class:`dict`: ``keys()``,
``values()``, ``items()``, ``get()``, ``pop()``, ``update()`` and
``setdefault()``, and it unpacks with ``**td``. The operations below follow
the rules of a tensor instead:

Iteration
  ``for x in td`` iterates over the first batch dimension, like a tensor. Each
  ``x`` is a TensorDict. A tensordict with no batch dimensions raises a
  ``TypeError``. To iterate over the entries, use ``keys()``, ``values()`` or
  ``items()``.

``len``
  ``len(td)`` is the size of the first batch dimension, or 0 if the batch size
  is empty. ``len(td.keys())`` is the number of keys.

``bool``
  ``bool(td)``, ``if td:`` and ``while td:`` raise a ``RuntimeError``
  (`#1939 <https://github.com/pytorch/tensordict/issues/1939>`_ asks for a
  dict-like truth value). To test for no entries, use
  :meth:`~tensordict.TensorDict.is_empty`.

``in``
  ``key in td`` tests the keys, as for a dict. A tuple of strings tests a
  nested key.

``==``
  ``td == other`` compares each entry and returns a TensorDict of booleans,
  like a tensor. If the two tensordicts do not have the same keys, it raises a
  ``KeyError``. Call ``.all()`` on the result to get a Python ``bool``.
  ``td in some_list`` uses ``==``, so it raises unless ``td`` itself is in the
  list. A TensorDict is not hashable.

For example:

  >>> td = TensorDict(a=torch.arange(3), nested={"b": torch.zeros(3, 2)}, batch_size=[3])
  >>> [x.batch_size for x in td]
  [torch.Size([]), torch.Size([]), torch.Size([])]
  >>> len(td), len(td.keys())
  (3, 2)
  >>> bool(td)
  Traceback (most recent call last):
  ...
  RuntimeError: Converting a tensordict to boolean value is not permitted
  >>> td.is_empty()
  False
  >>> "a" in td, ("nested", "b") in td, "b" in td
  (True, True, False)
  >>> (td == td.clone()).all()
  True

If a dataset returns tensordicts with no batch dimensions, the default collate
function of :class:`torch.utils.data.DataLoader` iterates over them and raises
the same ``TypeError``. :ref:`data-loading` gives the ``collate_fn`` to use.

See also
--------

- :attr:`~tensordict.TensorDict.batch_size`,
  :meth:`~tensordict.TensorDict.auto_batch_size_`,
  :meth:`~tensordict.TensorDict.from_dict` and
  :attr:`~tensordict.TensorDict.names` in the :ref:`api-reference`.
- :ref:`api-methods-indexing` and :ref:`api-methods-shape`: the indexing and
  shape methods.
- The tutorials :doc:`/tutorials/tensordict_slicing` and
  :doc:`/tutorials/tensordict_shapes`.
