.. currentmodule:: tensordict

.. meta::
   :description: How to batch PyTorch samples of different sizes, such as
      molecules, meshes, point clouds or sequences, without padding: keep
      them in a TensorDict or TensorClass as jagged nested tensors or a lazy
      stack, and pad only where a model needs dense input.

.. _variable-size-batches:

Batch samples of different sizes without padding
================================================

How do you batch samples that have different sizes in PyTorch, such as
molecules with different numbers of atoms, meshes with different numbers of
vertices, point clouds, or token sequences? The default ``collate_fn`` of a
:class:`~torch.utils.data.DataLoader` calls :func:`torch.stack`, which raises
for them, and padding every sample to the largest one wastes memory and
compute. A :class:`~tensordict.TensorDict` or
:class:`~tensordict.TensorClass` holds such a batch without padding, as
jagged nested tensors (:mod:`torch.nested`) or as a lazy stack of the
samples, and you index, split, concatenate, move and send it as one object.
You pad only the entries that a model needs as dense tensors, with a mask.

This page shows each layout on a batch of molecules, then the
``DataLoader`` code, the packed layout that some kernels need, and a table
to choose between them. Run the examples in order in one Python session:

  >>> import pickle
  >>> import tempfile
  >>> from functools import partial
  >>> import torch
  >>> from torch.utils.data import DataLoader
  >>> from tensordict import TensorClass, TensorDict, collate, lazy_stack, pad_sequence
  >>> _ = torch.manual_seed(0)
  >>> device = "cuda" if torch.cuda.is_available() else "cpu"

Four molecules have 3, 5, 2 and 4 atoms. Each atom has an atomic number and
a position, and each molecule has one energy:

  >>> num_atoms = [3, 5, 2, 4]
  >>> molecules = [
  ...     TensorDict(z=torch.randint(1, 10, (n,)), pos=torch.randn(n, 3), energy=torch.randn(()))
  ...     for n in num_atoms
  ... ]

Jagged nested tensors in a TensorDict
-------------------------------------

A nested tensor with the ``torch.jagged`` layout stores the atoms of all the
molecules one after the other, with the offsets where each molecule starts.
Its first dimension is the batch, and its second one is ragged. A
TensorDict holds it as an entry whose first dimension is a batch dimension.
Build the atomic numbers on the offsets of the positions, so that the two
entries share them:

  >>> pos = torch.nested.nested_tensor([m["pos"] for m in molecules], layout=torch.jagged)
  >>> z = torch.nested.nested_tensor_from_jagged(
  ...     torch.cat([m["z"] for m in molecules]), offsets=pos.offsets()
  ... )
  >>> batch = TensorDict(
  ...     pos=pos, z=z, energy=torch.stack([m["energy"] for m in molecules]), batch_size=[4]
  ... )
  >>> batch["pos"].offsets()
  tensor([ 0,  3,  8, 10, 14])
  >>> batch[1]["pos"].shape
  torch.Size([5, 3])

:meth:`~tensordict.LazyStackedTensorDict.densify` builds the same kind of
TensorDict from a list of samples in one call, with one set of offsets for
each entry: ``lazy_stack(molecules).densify(layout=torch.jagged)``.

The batch stores the 14 atoms and the offsets, and nothing for the padding.
If one molecule had 30 atoms, a padded ``pos`` would hold 4 times 30
positions instead of 40:

  >>> big = torch.nested.nested_tensor([torch.randn(n, 3) for n in (3, 5, 2, 30)], layout=torch.jagged)
  >>> big.values().numel(), big.to_padded_tensor(0.0).numel()
  (120, 360)

Index, split and concatenate along the batch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Indexing with an int, a slice, a boolean mask or a tensor of indices,
``split``, ``chunk`` and :func:`torch.cat` work along the batch dimension.
PyTorch does not slice, split, concatenate, or index with a mask or an
index tensor along the batch dimension of a jagged nested tensor itself: for
example, ``pos[torch.tensor([3, 0])]`` raises a ``NotImplementedError``. The
entries that shared their offsets still share them in the result:

  >>> picked = batch[torch.tensor([3, 0])]
  >>> picked["pos"].offsets(), picked["pos"].offsets() is picked["z"].offsets()
  (tensor([0, 4, 7]), True)
  >>> [part.batch_size for part in batch.split([1, 3])]
  [torch.Size([1]), torch.Size([3])]
  >>> both = torch.cat([batch, batch[:2]])
  >>> both.batch_size, both["pos"].offsets()
  (torch.Size([6]), tensor([ 0,  3,  8, 10, 14, 17, 22]))

:func:`torch.stack` raises, because the result would need a second ragged
dimension, which ``torch.jagged`` does not have. Use
:func:`~tensordict.lazy_stack` for that, as in the next section.

Compute on the atoms
~~~~~~~~~~~~~~~~~~~~

Many PyTorch operations accept a jagged nested tensor: elementwise
operations, :class:`~torch.nn.Linear`, reductions over the ragged
dimension, softmax, and
:func:`~torch.nn.functional.scaled_dot_product_attention`. A mean over the
atoms gives one vector for each molecule:

  >>> embed = torch.nn.Linear(3, 8)
  >>> embed(batch["pos"]).mean(dim=1).shape
  torch.Size([4, 8])

A function that reads such a TensorDict compiles with :func:`torch.compile`.
After the first call, a batch with other lengths recompiles once with a
dynamic ragged dimension, and the next ones reuse that graph:

  >>> def center(batch):
  ...     return batch["pos"] - batch["pos"].mean(dim=1, keepdim=True)
  >>> centered = torch.compile(center, fullgraph=True)(batch)
  >>> centered.offsets()
  tensor([ 0,  3,  8, 10, 14])

Move, consolidate and pickle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``to(device)`` moves the nested tensors with the other entries.
:meth:`~tensordict.TensorDictBase.consolidate` copies all the leaves, the
values and offsets of the nested tensors included, into one storage, so that
``to(device)`` copies one buffer and :mod:`pickle` writes one (see
:ref:`devices`). Both keep the jagged layout:

  >>> batch = batch.to(device)
  >>> consolidated = batch.consolidate()
  >>> consolidated.is_consolidated(), consolidated["pos"].is_nested
  (True, True)
  >>> pickle.loads(pickle.dumps(consolidated))["pos"].offsets().tolist()
  [0, 3, 8, 10, 14]

The same in a TensorClass
~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~tensordict.TensorClass` field can hold a jagged nested tensor
too, and the batch keeps its class through indexing and concatenation:

  >>> class Molecules(TensorClass):
  ...     z: torch.Tensor  # [batch, atoms], jagged
  ...     pos: torch.Tensor  # [batch, atoms, 3], jagged
  ...     energy: torch.Tensor  # [batch]
  >>> mols = Molecules(z=z, pos=pos, energy=batch["energy"].cpu(), batch_size=[4])
  >>> type(mols[:2]).__name__, mols[:2].pos.offsets()
  ('Molecules', tensor([0, 3, 8]))
  >>> torch.cat([mols, mols]).batch_size
  torch.Size([8])

Lazy stacks for samples whose shapes or keys differ
---------------------------------------------------

:func:`~tensordict.lazy_stack` keeps a list of samples as they are, and
acts on them as one tensordict. The samples can have different shapes, and
also different keys: here only the second molecule has partial charges. The
keys that only some samples have do not appear in ``keys()``, and you read
them from those samples:

  >>> charged = molecules[1].clone()
  >>> charged["charge"] = torch.zeros(5)
  >>> stack = lazy_stack([molecules[0], charged, molecules[2]])
  >>> stack.batch_size, sorted(stack.keys())
  (torch.Size([3]), ['energy', 'pos', 'z'])
  >>> stack[1]["charge"].shape
  torch.Size([5])

An entry with different shapes cannot be read as one tensor, so pass
``as_list``, ``as_padded_tensor`` or ``as_nested_tensor`` to
:meth:`~tensordict.TensorDictBase.get`. Writes go to the samples:

  >>> [p.shape[0] for p in stack.get("pos", as_list=True)]
  [3, 5, 2]
  >>> stack.get("pos", as_padded_tensor=True).shape
  torch.Size([3, 5, 3])
  >>> stack[1]["charge"] += 1.0
  >>> charged["charge"]
  tensor([1., 1., 1., 1., 1.])

Each operation on a lazy stack runs once for each sample, in Python. Use it
to gather, route and edit samples, and turn it into nested tensors or a
padded batch before the model. :ref:`stacking-variable-size` describes these
options in more detail.

Batches from a DataLoader
-------------------------

:func:`~tensordict.collate` is a ``collate_fn`` for tensordicts and
tensorclasses. It stacks the samples densely if they have the same shapes,
and lazily otherwise. The default ``collate_fn`` of a ``DataLoader`` does the
same, so you pass :func:`~tensordict.collate` for its keyword arguments.
``nested_keys`` stacks the named entries as jagged nested tensors, and the
other entries must then stack densely:

  >>> loader = DataLoader(molecules, batch_size=4)
  >>> type(next(iter(loader))).__name__
  'LazyStackedTensorDict'
  >>> loader = DataLoader(molecules, batch_size=4, collate_fn=partial(collate, nested_keys=["z", "pos"]))
  >>> jagged = next(iter(loader))
  >>> jagged["pos"].offsets(), jagged["energy"].shape
  (tensor([ 0,  3,  8, 10, 14]), torch.Size([4]))

Pad where the model needs dense input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pad_keys`` pads the named entries along their first dimension to the
longest sample, each with its own value, and ``mask_key`` stores a mask of
the values that are not padding. The other entries, such as the energy,
are stacked. Keys can be nested, here the atoms of each molecule are a
sub-tensordict:

  >>> nested_samples = [
  ...     TensorDict(atoms=m.select("z", "pos"), energy=m["energy"]) for m in molecules
  ... ]
  >>> collate_fn = partial(
  ...     collate, pad_keys={("atoms", "z"): 0, ("atoms", "pos"): 0.0}, mask_key="mask"
  ... )
  >>> padded = next(iter(DataLoader(nested_samples, batch_size=4, collate_fn=collate_fn)))
  >>> padded["atoms", "pos"].shape
  torch.Size([4, 5, 3])
  >>> padded["mask", "atoms", "z"]
  tensor([[ True,  True,  True, False, False],
          [ True,  True,  True,  True,  True],
          [ True,  True, False, False, False],
          [ True,  True,  True,  True, False]])

``pad_keys`` needs samples with an empty batch size. Outside a
``DataLoader``, :func:`~tensordict.pad_sequence` pads a list of tensordicts
in the same way, with a mapping of padding values and ``return_mask``. It
pads every entry, so it raises ``RuntimeError: Cannot pad scalars`` for a
sample with a 0-dim entry, such as the energy here:

  >>> padded = pad_sequence([m["atoms"] for m in nested_samples], return_mask=True)
  >>> padded["pos"].shape, padded["masks", "pos"].sum(-1)
  (torch.Size([4, 5, 3]), tensor([3, 5, 2, 4]))

Packed layout with offsets
--------------------------

Some kernels take flat arrays instead: the atoms of all the molecules one
after the other, and either the offsets where each molecule starts (the
``cu_seqlens`` of variable-length attention, the ``ptr`` of PyG) or the
index of the molecule of each atom (the ``batch`` vector of PyG). The
``values()`` and ``offsets()`` of a jagged nested tensor are that layout,
and :func:`torch.repeat_interleave` gives the molecule index. A sum over
the atoms of each molecule is then one ``index_add_``:

  >>> atoms = TensorDict(z=jagged["z"].values(), pos=jagged["pos"].values(), batch_size=[14])
  >>> lengths = jagged["pos"].offsets().diff()
  >>> mol_index = torch.repeat_interleave(torch.arange(4), lengths)
  >>> mol_index
  tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3])
  >>> torch.zeros(4).index_add_(0, mol_index, atoms["z"].float()).shape
  torch.Size([4])

:ref:`cookbook-packing` shows how to pack from a list of samples, from
nested tensors and from a padded batch, and how to go back.

The packed form is also how you save such a batch, because
:meth:`~tensordict.TensorDictBase.memmap` and
:meth:`~tensordict.TensorDictBase.save` raise a ``NotImplementedError`` for
jagged nested tensors. Save the packed atoms and the offsets, then rebuild
the nested tensors with :func:`torch.nested.nested_tensor_from_jagged`:

  >>> tmp = tempfile.TemporaryDirectory()
  >>> offsets = jagged["pos"].offsets()
  >>> _ = TensorDict(atoms=atoms, offsets=offsets).memmap(tmp.name)
  >>> saved = TensorDict.load_memmap(tmp.name)
  >>> restored = saved["atoms"].apply(
  ...     lambda x: torch.nested.nested_tensor_from_jagged(x.clone(), saved["offsets"].clone()),
  ...     batch_size=[4],
  ... )
  >>> restored[1]["pos"].shape
  torch.Size([5, 3])
  >>> del saved
  >>> tmp.cleanup()

Which layout to use
-------------------

.. list-table::
   :header-rows: 1
   :widths: 16 20 30 34

   * - Layout
     - Memory
     - What works
     - Use it for
   * - Jagged nested tensors in a TensorDict or TensorClass
     - The values and one offsets tensor
     - Indexing, ``split`` and ``cat`` along the batch, ``to``,
       ``consolidate``, pickle, :func:`torch.compile`, and the PyTorch ops
       that support nested tensors. Not ``torch.stack``, ``memmap`` or
       ``save``.
     - The default for a model that reads per-sample sets or sequences,
       such as attention over atoms or tokens.
   * - Lazy stack
     - The samples as they are
     - Different keys and shapes in one batch, writes into each sample.
       Each operation loops over the samples.
     - Gathering, routing and editing samples before the model.
   * - Padded tensors with a mask
     - The longest sample times the batch size
     - Every PyTorch op, static shapes for :func:`torch.compile` and CUDA
       graphs if you pad to a fixed length.
     - Models and kernels that need dense input, or batches whose sizes
       are close.
   * - Packed values with offsets
     - The values and the offsets
     - Plain tensors: every op, ``memmap`` and ``save``; segment
       operations with ``index_add_`` or ``scatter_reduce``.
     - Variable-length attention kernels, scatter-style pooling, and
       saving.
   * - PyG's ``Batch``
     - The values, plus ``batch`` and ``ptr``
     - PyG's message-passing layers. ``Batch.from_data_list`` shifts the
       ``edge_index`` of each graph by the number of nodes before it.
     - Graph neural networks built on PyG layers. ``torch.cat`` on a
       tensordict does not shift edge indices.

Limits of jagged nested tensors
-------------------------------

- One ragged dimension, right after the batch dimension. A TensorDict that
  holds jagged nested tensors has one batch dimension: ``reshape`` and
  ``view`` of the batch raise.
- :func:`torch.stack` raises. Use :func:`~tensordict.lazy_stack`.
- ``memmap`` and ``save`` raise. Save the packed values and offsets, as
  above, or convert to the ``torch.strided`` layout.
- Writing a sample of another length in place, as in
  ``batch[0] = other_molecule``, raises, in a lazy stack too. Build a new
  batch instead, for example with :func:`torch.cat` of slices.
- Not every PyTorch op supports nested tensors. The :mod:`torch.nested`
  documentation lists the supported ones, and an op without support
  raises an error.

See also
--------

- :ref:`stacking-variable-size`: lazy stacks, ``get`` with ``as_list``,
  ``as_padded_tensor`` and ``as_nested_tensor``, ``pad_sequence`` and
  ``unpad_sequence``.
- :ref:`cookbook-packing`: packing, token budgets and balancing.
- :ref:`data-loading`: datasets, ``collate_fn`` and worker processes.
