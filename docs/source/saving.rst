.. _saving:

Saving TensorDict and tensorclass objects
=========================================

While we can just save a tensordict with :func:`~torch.save`, this
will create a single file with the whole content of the data structure.
One can easily imagine situations where this is sub-optimal!

TensorDict serialization API mainly relies on :class:`~tensordict.MemoryMappedTensor`
which is used to write tensors independently on disk with a data structure
that mimics the TensorDict's one.

TensorDict's serialization speed can be an order of magnitude **faster** than
PyTorch's one with :func:`~torch.save`'s pickle reliance. This document explains
how to create and interact with data stored on disk using TensorDict.

Saving memory-mapped TensorDicts
---------------------------------

When a tensordict is dumped as a mmap data structure, each entry corresponds
to a single ``*.memmap`` file, and the directory structure is determined by the
key structure: generally, nested keys correspond to sub-directories.

Saving a data structure as a structured set of memory-mapped tensors has the following
advantages:

- The saved data can be partially loaded. If a large model is saved on disk but
  only parts of its weights need to be loaded onto a module created in a separate
  script, only these weights will be loaded in memory.
- Saving data is safe: using the pickle library for serializing big data structures
  can be unsafe as unpickling can execute any arbitrary code. TensorDict's loading
  API only reads pre-selected fields from saved json files and memory buffers
  saved on disk.
- Saving is fast: because the data is written in several independent files,
  we can amortize the IO overhead by launching several concurrent threads that
  each access a dedicated file on their own.
- The structure of the saved data is apparent: the directory tree is indicative
  of the data content.

However, this approach also has some disadvantages:

- Not every data type can be saved. :obj:`~tensordict.tensorclass` allows to save
  any non-tensor data: if these data can be represented in a json file, a json
  format will be used. Otherwise, non-tensor data will be saved independently
  with :func:`~torch.save` as a fallback.
  The :class:`~tensordict.NonTensorData` class can be used to represent non-tensor
  data in a regular :class:`~tensordict.TensorDict` instance.

tensordict's memory-mapped API relies on four core methods:
:meth:`~tensordict.TensorDictBase.memmap_`, :meth:`~tensordict.TensorDictBase.memmap`,
:meth:`~tensordict.TensorDictBase.memmap_like` and :meth:`~tensordict.TensorDictBase.load_memmap`.

The :meth:`~tensordict.TensorDictBase.memmap_` and :meth:`~tensordict.TensorDictBase.memmap`
methods will write the data on disk with or without modifying the tensordict
instance that contains the data. These methods can be used to serialize a model
on disk (we use multiple threads to speed up serialization):

  >>> model = nn.Transformer()
  >>> weights = TensorDict.from_module(model)
  >>> weights_disk = weights.memmap("/path/to/saved/dir", num_threads=32)
  >>> new_weights = TensorDict.load_memmap("/path/to/saved/dir")
  >>> assert (weights_disk == new_weights).all()

The :meth:`~tensordict.TensorDictBase.memmap_like` is to be used when a dataset
needs to be preallocated on disk, the typical usage being:

  >>> def make_datum(): # used for illustration purposes
  ...    return TensorDict({"image": torch.randint(255, (3, 64, 64)), "label": 0}, batch_size=[])
  >>> dataset_size = 1_000_000
  >>> datum = make_datum() # creates a single instance of a TensorDict datapoint
  >>> data = datum.expand(dataset_size) # does NOT require more memory usage than datum, since it's only a view on datum!
  >>> data_disk = data.memmap_like("/path/to/data")  # creates the two memory-mapped tensors on disk
  >>> del data # data is not needed anymore

As illustrated above, when converting entries of a :class:`~tensordict.TensorDict`
to :class:`~tensordict.MemoryMappedTensor`, it is possible to control where
the memory maps are saved on disk so that they persist and can
be loaded at a later date. On the other hand, the file system can also be used.
To use this, simply discard the ``prefix`` argument in the three serialization
methods above.

When a ``prefix`` is specified, the data structure follows the TensorDict's one:

  >>> import torch
  >>> from tensordict import TensorDict
  >>> td = TensorDict({"a": torch.rand(10), "b": {"c": torch.rand(10)}}, [10])
  >>> td.memmap_(prefix="tensordict")

yields the following directory structure

.. code-block::

  tensordict
  ├── a.memmap
  ├── b
  │   ├── c.memmap
  │   └── meta.json
  └── meta.json

The ``meta.json`` files contain all the relevant information to rebuild the
tensordict, such as device, batch-size, but also the tensordict subtypes.
This means that :meth:`~tensordict.TensorDict.load_memmap` will be able to
reconstruct complex nested structures where sub-tensordicts have different types
than parents:

  >>> from tensordict import TensorDict, tensorclass, TensorDictBase
  >>> from tensordict.utils import print_directory_tree
  >>> import torch
  >>> import tempfile
  >>> td_list = [TensorDict({"item": i}, batch_size=[]) for i in range(4)]
  >>> @tensorclass
  ... class MyClass:
  ...     data: torch.Tensor
  ...     metadata: str
  >>> tc = MyClass(torch.randn(3), metadata="some text", batch_size=[])
  >>> data = TensorDict({"td_list": torch.stack(td_list), "tensorclass": tc}, [])
  >>> with tempfile.TemporaryDirectory() as tempdir:
  ...     data.memmap_(tempdir)
  ...
  ...     loaded_data = TensorDictBase.load_memmap(tempdir)
  ...     assert (loaded_data == data).all()
  ...     print_directory_tree(tempdir)
  tmpzy1jcaoq/
      tensorclass/
          _tensordict/
              data.memmap
              meta.json
          meta.json
      td_list/
          0/
              item.memmap
              meta.json
          1/
              item.memmap
              meta.json
          3/
              item.memmap
              meta.json
          2/
              item.memmap
              meta.json
          meta.json
      meta.json


Handling existing :class:`~tensordict.MemoryMappedTensor`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the :class:`~tensordict.TensorDict` already contains
:class:`~tensordict.MemoryMappedTensor` entries there are a few
possible behaviours.

- If ``prefix`` is not specified and :meth:`~tensordict.TensorDict.memmap` is called
  twice, the resulting `TensorDict` will contain the same data as the original one.

    >>> td = TensorDict({"a": 1}, [])
    >>> td0 = td.memmap()
    >>> td1 = td0.memmap()
    >>> td0["a"] is td1["a"]
    True

- If ``prefix`` is specified and differs from the prefix of the existing
  :class:`~tensordict.MemoryMappedTensor` instances, an exception is raised,
  unless `copy_existing=True` is passed:

    >>> with tempfile.TemporaryDirectory() as tmpdir_0:
    ...     td0 = td.memmap(tmpdir_0)
    ...     td0 = td.memmap(tmpdir_0)  # works, results are just overwritten
    ...     with tempfile.TemporaryDirectory() as tmpdir_1:
    ...         td1 = td0.memmap(tmpdir_1)
    ...         td_load = TensorDict.load_memmap(tmpdir_1)  # works!
    ...     assert (td_load == td).all()
    ...     with tempfile.TemporaryDirectory() as tmpdir_1:
    ...         td_load = TensorDict.load_memmap(tmpdir_1)  # breaks!

  This feature is implemented to prevent users from inadvertently copying memory-mapped
  tensors from one location to another.

Consolidated serialization
--------------------------

For fast transfer (e.g. across the network, or to GPU), you can consolidate all
leaf tensors into a single contiguous buffer using
:meth:`~tensordict.TensorDictBase.consolidate`:

  >>> td = TensorDict(a=torch.randn(1000), b={"c": torch.randn(1000)}, batch_size=[1000])
  >>> td_c = td.consolidate()

A consolidated tensordict can be pickled much faster than a regular one because
it becomes a single storage + metadata dict.  It can also be saved to disk as
a memory-mapped file:

  >>> td_c = td.consolidate("/path/to/storage.memmap")

See :meth:`~tensordict.TensorDictBase.consolidate` for the full API, including
options like ``num_threads``, ``device``, ``pin_memory``, and ``share_memory``.

Legacy: TorchSnapshot compatibility
------------------------------------

.. warning::
  torchsnapshot maintenance has been discontinued.  The section below is kept
  for reference only; we recommend using the memory-mapped API above for new
  projects.

TensorDict is compatible with `torchsnapshot <https://github.com/pytorch/torchsnapshot>`_.
TorchSnapshot saves each tensor independently, with a data structure that
mimics the TensorDict's one.

**In-memory loading**

.. code-block:: Python

  >>> import uuid
  >>> import torchsnapshot
  >>> from tensordict import TensorDict
  >>> import torch
  >>>
  >>> tensordict_source = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3)}}, [])
  >>> state = {"state": tensordict_source}
  >>> path = f"/tmp/{uuid.uuid4()}"
  >>> snapshot = torchsnapshot.Snapshot.take(app_state=state, path=path)
  >>> # later
  >>> snapshot = torchsnapshot.Snapshot(path=path)
  >>> tensordict_target = TensorDict()
  >>> target_state = {"state": tensordict_target}
  >>> snapshot.restore(app_state=target_state)
  >>> assert (tensordict_source == tensordict_target).all()

**Big-dataset loading (memory-mapped)**

.. code-block:: Python

  >>> td = TensorDict({"a": torch.randn(3), "b": TensorDict({"c": torch.randn(3, 1)}, [3, 1])}, [3])
  >>> td.memmap_()
  >>> assert isinstance(td["b", "c"], MemoryMappedTensor)
  >>>
  >>> app_state = {
  ...     "state": torchsnapshot.StateDict(tensordict=td.state_dict(keep_vars=True))
  ... }
  >>> snapshot = torchsnapshot.Snapshot.take(app_state=app_state, path=f"/tmp/{uuid.uuid4()}")
  >>>
  >>> td_dest = TensorDict({"a": torch.zeros(3), "b": TensorDict({"c": torch.zeros(3, 1)}, [3, 1])}, [3])
  >>> td_dest.memmap_()
  >>> assert isinstance(td_dest["b", "c"], MemoryMappedTensor)
  >>> app_state = {
  ...     "state": torchsnapshot.StateDict(tensordict=td_dest.state_dict(keep_vars=True))
  ... }
  >>> snapshot.restore(app_state=app_state)
  >>> assert (td_dest == td).all()
  >>> assert (td_dest["b"].batch_size == td["b"].batch_size)
  >>> assert isinstance(td_dest["b", "c"], MemoryMappedTensor)
