.. _overview:

What is TensorDict?
===================

This page explains what a :class:`~tensordict.TensorDict` is, how it differs
from a dictionary of tensors, and which page to read next. Read it first if you
are new to tensordict.

A TensorDict is a dictionary of tensors that also behaves like a tensor. All
its entries share their first dimensions, called the batch size. Indexing,
reshaping, device moves and other shape and storage operations apply to all the
entries at once. You can save a TensorDict to disk and send it to other
processes or nodes. The :mod:`tensordict.nn` package lets :class:`torch.nn.Module`
objects read and write TensorDicts. It works with :mod:`torch.func`, which makes
model ensembling and parameter manipulation simpler. TensorDict started as part
of TorchRL and is now a separate library.

  >>> import torch
  >>> from tensordict import TensorDict
  >>> _ = torch.manual_seed(0)
  >>> data = TensorDict(
  ...     obs=torch.randn(4, 3),
  ...     action=torch.randn(4, 2),
  ...     next=TensorDict(obs=torch.randn(4, 3), reward=torch.randn(4, 1), batch_size=[4]),
  ...     batch_size=[4],
  ... )
  >>> data
  TensorDict(
      fields={
          action: Tensor(shape=torch.Size([4, 2]), device=cpu, dtype=torch.float32, is_shared=False),
          next: TensorDict(
              fields={
                  obs: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                  reward: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
              batch_size=torch.Size([4]),
              device=None,
              is_shared=False),
          obs: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
      batch_size=torch.Size([4]),
      device=None,
      is_shared=False)

A string, or a tuple of strings for a nested entry, reads one entry. Any other
index applies to the batch dimension of every entry:

  >>> data["next", "reward"].shape
  torch.Size([4, 1])
  >>> data[:2]
  TensorDict(
      fields={
          action: Tensor(shape=torch.Size([2, 2]), device=cpu, dtype=torch.float32, is_shared=False),
          next: TensorDict(
              fields={
                  obs: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                  reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
              batch_size=torch.Size([2]),
              device=None,
              is_shared=False),
          obs: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
      batch_size=torch.Size([2]),
      device=None,
      is_shared=False)
  >>> data.to(torch.float64)["next", "obs"].dtype
  torch.float64

What a TensorDict is not
------------------------

A TensorDict is not a :class:`dict`. It has the methods of a mapping
(``keys``, ``values``, ``items``, ``get``, ``pop``, ``update``, ``in`` and
``**data``), but ``for x in data``, ``len(data)``, ``bool(data)`` and
``data == other`` follow the rules of a tensor:

  >>> isinstance(data, dict)
  False
  >>> len(data)  # the size of the first batch dimension, not the number of keys
  4
  >>> sorted(data.keys())
  ['action', 'next', 'obs']

:ref:`batch-size-not-a-dict` lists these differences.

A TensorDict is not a :class:`torch.Tensor` either. Each entry keeps its own
dtype, device and the dimensions after the batch dimensions. ``data.shape`` is
the batch size, not the shape of an entry:

  >>> isinstance(data, torch.Tensor)
  False
  >>> data.shape
  torch.Size([4])

Compared with a dictionary and tree_map
---------------------------------------

:func:`torch.utils._pytree.tree_map` applies a function to each tensor of a
nested dictionary. It does some of the work of a TensorDict, but it does not
know about a batch dimension. The examples below use this dictionary, with
three samples:

  >>> from torch.utils._pytree import tree_map
  >>> regular_dict = {
  ...     "obs": torch.randn(3, 11),
  ...     "next": {"obs": torch.randn(3, 11), "reward": torch.randn(3)},
  ... }
  >>> td = TensorDict(regular_dict, batch_size=[3])

To split the samples with tree_map, you get one dictionary of tuples. You must
then build the three dictionaries by hand, and that code grows with each level
of nesting:

  >>> unbound = tree_map(lambda x: x.unbind(0), regular_dict)
  >>> type(unbound["next"]["reward"])
  <class 'tuple'>
  >>> samples = [
  ...     {
  ...         "obs": unbound["obs"][i],
  ...         "next": {"obs": unbound["next"]["obs"][i], "reward": unbound["next"]["reward"][i]},
  ...     }
  ...     for i in range(3)
  ... ]

:meth:`~tensordict.TensorDict.unbind` returns the three tensordicts directly,
at any nesting depth. :meth:`~tensordict.TensorDict.split` and
:meth:`~tensordict.TensorDict.chunk` work the same way:

  >>> td0, td1, td2 = td.unbind(0)
  >>> td0["next", "reward"].shape
  torch.Size([])

tree_map applies an index to each tensor as it is. The result then depends on
the shape of each tensor. A TensorDict applies the index to the batch
dimensions only:

  >>> mapped = tree_map(lambda x: x[..., :2], regular_dict)
  >>> mapped["obs"].shape, mapped["next"]["reward"].shape
  (torch.Size([3, 2]), torch.Size([2]))
  >>> sliced = td[..., :2]
  >>> sliced["obs"].shape, sliced["next", "reward"].shape
  (torch.Size([2, 11]), torch.Size([2]))

A dictionary accepts a tensor of any shape. A TensorDict raises an error when
an entry does not match the batch size:

  >>> regular_dict["next"]["done"] = torch.zeros(5)  # no error
  >>> td["next", "done"] = torch.zeros(5)
  Traceback (most recent call last):
  ...
  RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([3]) and value.shape=torch.Size([5]) for key 'done'. The leading dimensions of a value must match the batch size.

TensorDict is registered as a pytree node, so you can still use tree_map, and
:mod:`torch.func`, on a TensorDict. The result is a TensorDict:

  >>> type(tree_map(lambda x: x + 1, td)).__name__
  'TensorDict'

Where to go next
----------------

The user guide explains one concept on each page:

- :ref:`batch-size`: the batch size, indexing, shape operations and dimension
  names.
- :ref:`keys`: string and nested keys, reading and writing entries, nested
  tensordicts and locking.
- :ref:`stacking`: dense and lazy stacks, and data with different shapes.
- :ref:`operations`: arithmetic, :meth:`~tensordict.TensorDict.apply`,
  reductions, gradients and the in-place conventions.
- :ref:`devices`: devices, dtypes and memory.
- :ref:`non-tensor-data`: strings and other Python objects, and entries outside
  the batch size.
- :ref:`tensorclass-guide` and :ref:`typed-tensordict-guide`: containers with
  typed fields. :ref:`choosing-a-container` compares them.
- :ref:`modules-guide`: :class:`~tensordict.nn.TensorDictModule` and the other
  modules that read and write tensordicts.

For saving, storage, data loading, distributed training and
:func:`torch.compile`, see :ref:`saving`, :ref:`storage`, :ref:`data-loading`,
:ref:`distributed` and :ref:`compile`.

See also
--------

- :class:`~tensordict.TensorDict` in the :ref:`api-reference`.
- The tutorials :doc:`/tutorials/tensordict_keys`,
  :doc:`/tutorials/tensordict_slicing` and :doc:`/tutorials/tensordict_shapes`.
