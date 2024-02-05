Saving TensorDict and tensorclass objects
=========================================

While we can just save a tensordict with :func:`~torch.save`, this
will create a single file with the whole content of the data structure.
One can easily imagine situations where this is sub-optimal!

TensorDict serialization API mainly relies on :class:`~tensordict.MemoryMappedTensor`
which is used to write tensors independently on disk with a data structure
that mimics the TensorDict's one.

TensorDict's serialization speed can be an order of magnitude __faster__ than
PyTorch's one with :func:`~torch.save`'s pickle reliance. This document explains
how to create and interact with data stored on disk using TensorDict.

Saving memmory-mapped TensorDicts
---------------------------------

When a tensordict is dumped as a mmap data structure, each entry corresponds
to a single ``*.memmap`` file, and the directory structure is determined by the
key structure: generally, nested keys correspond to sub-directories.

Saving a data structure as a structured set of memory-mapped tensors has the following
advantages:

- The saved data can be partially loaded. If a large model is saved on disk but
  only parts of its weights need to be loaded onto a module created in a separate
  scripts, only these weights will be loaded in memory.
- Saving data is safe: using the pickle library for serializing big data structures
  can be unsafe as unpickling can execute any arbitrary code. TensorDict's loading
  API only reads pre-selected fields from saved json files and memorybuffers
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

tensordict's memory-mapped API relies on four core method:
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

As illustrated above, when converting entries of a :class:`~tensordict.TensorDict``
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

The ``meta.json`` files contain all the releant information to rebuild the
tensordict, such as device, batch-size, but also the tensordict subtypes.
This means that :meth:`~tensordict.TensorDict.load_memmap` will be able to
reconstruct complex nested structure where sub-tensordicts have different types
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

If the :class:`~tensordict.TensorDict`` already contains
:class:`~tensordict.MemoryMappedTensor` entries there are a few
possible behaviours.

- If ``prefix`` is not specified and :meth:`~tensordict.TensorDict.memmap` is called
  twice, the resulting `TensorDict` will contain the same data as the orignal one.

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

  This feature is implemented to prevent users from inadvertently copy memorymapped
  tensors from one location to another.

TorchSnapshot compatibility
---------------------------

.. warning::
  As torchsnapshot maintenance is being discontinued. As such, we won't be implementing
  new features for tensordict compatibility with this library.

TensorDict is compatible with `torchsnapshot <https://github.com/pytorch/torchsnapshot>`_,
a PyTorch checkpointing library.
TorchSnapshot will save each of your tensors independently, with a data structure that
mimics the one of your tensordict or tensorclass. Moreover, TensorDict has naturally
buit-in the tools necessary for saving and loading huge datasets on disk without
loading the full tensors in memory: in other words, the combination tensordict + torchsnapshot
makes it possible to load a tensor big as several hundreds of Gb onto a
pre-allocated :class:`~tensordict.MemmapTensor` without passing it in one chunk on RAM.

There are two main use cases: saving and loading tensordicts that fit in memory,
and saving and loading tensordicts stored on disk using :class:`~tensordict.MemmapTensor`.

General use case: in-memory loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method is suitable if your destination tensordict is not pre-allocated.
This offers flexibility (you can load any tensordict onto your tensordict, you
don't need to know its content in advance) and this method is marginally
easier to code than the other.
However, this may break if your tensors are extremely big and do not fit in memory.
Also, it will not allow you to load directly onto the device of your choice.

The two main commands to remember for the saving operation are:

  >>> state = {"state": tensordict_source}
  >>> snapshot = torchsnapshot.Snapshot.take(app_state=state, path="/path/to/my/snapshot")

To load onto a destination tensordict, you can simply load the snapshot and update the
tensordict. Under the hood, this method will call :obj:`tensordict_target.load_state_dict(state_dict)`,
meaning that the :obj:`state_dict` will first be put in memory entirely, and then loaded onto the
destination tensordict:

  >>> snapshot = Snapshot(path="/path/to/my/snapshot")
  >>> state_target = {"state": tensordict_target}
  >>> snapshot.restore(app_state=state_target)

Here is a full example:

.. code-block:: Python

  >>> import uuid
  >>> import torchsnapshot
  >>> from tensordict import TensorDict
  >>> import torch
  >>>
  >>> tensordict_source = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3)}}, [])
  >>> state = {"state": tensordict}
  >>> path = f"/tmp/{uuid.uuid4()}"
  >>> snapshot = torchsnapshot.Snapshot.take(app_state=state, path=path)
  >>> # later
  >>> snapshot = torchsnapshot.Snapshot(path=path)
  >>> tensordict2 = TensorDict({}, [])
  >>> target_state = {
  >>>     "state": tensordict2
  >>> }
  >>> snapshot.restore(app_state=target_state)
  >>> assert (tensordict == tensordict2).all()


Saving and loading big-datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the dataset is too big to fit in memory, the above method could easily break.
We take advantage of the capabilities of torchsnapshot to load the tensors in small chunks
on their preallocated destination.
This requires you to know what shape, device etc. your destination data will have and live on,
but it's a small price to pay to be able to checkpoint your model or dataloading!

In contrast with the previous example, we will not be using the :func:`load_state_dict()` method
of :obj:`TensorDict` but rather a :obj:`state_dict` obtained from the destination object
that we will re-populate with the saved data.

Again, two lines of code are sufficient to save the data:

  >>> app_state = {
  ...     "state": torchsnapshot.StateDict(tensordict=tensordict_source.state_dict(keep_vars=True))
  ... }
  >>> snapshot = torchsnapshot.Snapshot.take(app_state=app_state, path="/path/to/my/snapshot")

We have been using :obj:`torchsnapshot.StateDict` and we explicitly called
:obj:`my_tensordict_source.state_dict(keep_vars=True)`, unlike the previous example.
Now, to load this onto a destination tensordict:

  >>> snapshot = Snapshot(path="/path/to/my/snapshot")
  >>> app_state = {
  ...     "state": torchsnapshot.StateDict(tensordict=tensordict_target.state_dict(keep_vars=True))
  ... }
  >>> snapshot.restore(app_state=app_state)

In this example, the loading is entirely handled by torchsnapshot, ie. there is
no call to :func:`TensorDict.load_state_dict()`.

.. note::

    This has two important implications:

    1. Since :func:`LazyStackedTensorDict.state_dict()` (and other lazy tensordict classes)
       return a copy of the data after some operation has been executed, loading onto the
       state-dict will not update the original class. However, since the `state_dict()` operation
       is supported, this will not raise an error.
    2. Similarly, since the state-dict is updated in-place but the tensordict is not
       updated using :func:`TensorDict.update()` or :func:`TensorDict.set()`, a missing
       key in the destination tensordict will go unnoticed.

Here is a full example:

.. code-block:: Python

  >>> td = TensorDict({"a": torch.randn(3), "b": TensorDict({"c": torch.randn(3, 1)}, [3, 1])}, [3])
  >>> td.memmap_()
  >>> assert isinstance(td["b", "c"], MemmapTensor)
  >>>
  >>> app_state = {
  ...     "state": torchsnapshot.StateDict(tensordict=td.state_dict(keep_vars=True))
  ... }
  >>> snapshot = torchsnapshot.Snapshot.take(app_state=app_state, path=f"/tmp/{uuid.uuid4()}")
  >>>
  >>>
  >>> td_dest = TensorDict({"a": torch.zeros(3), "b": TensorDict({"c": torch.zeros(3, 1)}, [3, 1])}, [3])
  >>> td_dest.memmap_()
  >>> assert isinstance(td_dest["b", "c"], MemmapTensor)
  >>> app_state = {
  ...     "state": torchsnapshot.StateDict(tensordict=td_dest.state_dict(keep_vars=True))
  ... }
  >>> snapshot.restore(app_state=app_state)
  >>> # sanity check
  >>> assert (td_dest == td).all()
  >>> assert (td_dest["b"].batch_size == td["b"].batch_size)
  >>> assert isinstance(td_dest["b", "c"], MemmapTensor)

Finally, tensorclass also supports this feature. The code is fairly similar to the one above:

.. code-block:: Python

  >>> from __future__ import annotations
  >>> import uuid
  >>> from typing import Union, Optional
  >>>
  >>> import torchsnapshot
  >>> from tensordict import TensorDict, MemmapTensor
  >>> import torch
  >>> from tensordict.prototype import tensorclass
  >>>
  >>> @tensorclass
  >>> class MyClass:
  ...      x: torch.Tensor
  ...      y: Optional[MyClass]=None
  ...
  >>> tc = MyClass(x=torch.randn(3), y=MyClass(x=torch.randn(3), batch_size=[]), batch_size=[])
  >>> tc.memmap_()
  >>> assert isinstance(tc.y.x, MemmapTensor)
  >>>
  >>> app_state = {
  ...     "state": torchsnapshot.StateDict(tensordict=tc.state_dict(keep_vars=True))
  ... }
  >>> snapshot = torchsnapshot.Snapshot.take(app_state=app_state, path=f"/tmp/{uuid.uuid4()}")
  >>>
  >>> tc_dest = MyClass(x=torch.randn(3), y=MyClass(x=torch.randn(3), batch_size=[]), batch_size=[])
  >>> tc_dest.memmap_()
  >>> assert isinstance(tc_dest.y.x, MemmapTensor)
  >>> app_state = {
  ...     "state": torchsnapshot.StateDict(tensordict=tc_dest.state_dict(keep_vars=True))
  ... }
  >>> snapshot.restore(app_state=app_state)
  >>>
  >>> assert (tc_dest == tc).all()
  >>> assert (tc_dest.y.batch_size == tc.y.batch_size)
  >>> assert isinstance(tc_dest.y.x, MemmapTensor)
