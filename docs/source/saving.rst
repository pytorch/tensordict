Saving TensorDict and tensorclass objects
=========================================

While we can just save a tensordict with :obj:`torch.save(my_tensordict)`, this
will create a single file with the whole content of the data structure.
One can easily imagine situations where this is sub-optimal!

In general, we recommend using `torchsnapshot <https://github.com/pytorch/torchsnapshot>`_.
TorchSnapshot will save each of your tensors independently, with a data structure that
mimics the one of your tensordict or tensorclass. Moreover, TensorDict has naturally
buit-in the tools necessary for saving and loading huge datasets on disk without
loading the full tensors in memory: in other words, the combination tensordict + torchsnapshot
makes it possible to load a tensor big as several hundreds of Gb onto a
pre-allocated :obj:`MemmapTensor` without passing it in one chunk on RAM.

There are two main use cases: saving and loading tensordicts that fit in memory,
and saving and loading tensordicts stored on disk using MemmapTensors.

General use case: in-memory loading
-----------------------------------

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
-------------------------------

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
  ...     "state": torchsnapshot.StateDict(tensordict=tensordict_source.state_dict())
  ... }
  >>> snapshot = torchsnapshot.Snapshot.take(app_state=app_state, path="/path/to/my/snapshot")

We have been using :obj:`torchsnapshot.StateDict` and we explicitly called
:obj:`my_tensordict_source.state_dict()`, unlike the previous example.
Now, to load this onto a destination tensordict:

  >>> snapshot = Snapshot(path="/path/to/my/snapshot")
  >>> app_state = {
  ...     "state": torchsnapshot.StateDict(tensordict=tensordict_target.state_dict())
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
  ...     "state": torchsnapshot.StateDict(tensordict=td.state_dict())
  ... }
  >>> snapshot = torchsnapshot.Snapshot.take(app_state=app_state, path=f"/tmp/{uuid.uuid4()}")
  >>>
  >>>
  >>> td_dest = TensorDict({"a": torch.zeros(3), "b": TensorDict({"c": torch.zeros(3, 1)}, [3, 1])}, [3])
  >>> td_dest.memmap_()
  >>> assert isinstance(td_dest["b", "c"], MemmapTensor)
  >>> app_state = {
  ...     "state": torchsnapshot.StateDict(tensordict=td_dest.state_dict())
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
  ...     "state": torchsnapshot.StateDict(tensordict=tc.state_dict())
  ... }
  >>> snapshot = torchsnapshot.Snapshot.take(app_state=app_state, path=f"/tmp/{uuid.uuid4()}")
  >>>
  >>> tc_dest = MyClass(x=torch.randn(3), y=MyClass(x=torch.randn(3), batch_size=[]), batch_size=[])
  >>> tc_dest.memmap_()
  >>> assert isinstance(tc_dest.y.x, MemmapTensor)
  >>> app_state = {
  ...     "state": torchsnapshot.StateDict(tensordict=tc_dest.state_dict())
  ... }
  >>> snapshot.restore(app_state=app_state)
  >>>
  >>> assert (tc_dest == tc).all()
  >>> assert (tc_dest.y.batch_size == tc.y.batch_size)
  >>> assert isinstance(tc_dest.y.x, MemmapTensor)

Saving memmory-mapped TensorDicts
---------------------------------

When converting entries of a ``TensorDict`` to ``MemmapTensor``, it is possible
to control where the memory maps are saved on disk so that they persist and can
be loaded at a later date. Simply specify a ``prefix`` when calling ``TensorDict.memmap_``. For example

.. code-block:: Python

  >>> import torch
  >>> from tensordict import TensorDict
  >>> td = TensorDict({"a": torch.rand(10), "b": {"c": torch.rand(10)}}, [10])
  >>> td.memmap_(prefix="tensordict")

yields the following directory structure

.. code-block::

  tensordict
  ├── a.memmap
  ├── a.meta.pt
  ├── b
  │   ├── c.memmap
  │   ├── c.meta.pt
  │   └── meta.pt
  └── meta.pt

Each key in the ``TensorDict`` corresponds to a single ``*.memmap`` file, with
the directory structure determined by the key structure: nested keys correspond
to sub-directories.

.. note::

  Because we must walk the nested directory structure, and write a file for
  each entry, this is not a fast way to serialize the contents of the
  ``TensorDict``, and hence should not be used for example inside a training
  loop.

To load the ``TensorDict`` from these files we can use
``TensorDict.load_memmap``.

.. code-block:: Python

  >>> td2 = TensorDict.load_memmap(prefix="tensordict")
  >>> td2
  TensorDict(
    fields={
        a: MemmapTensor(shape=torch.Size([10]), device=cpu, dtype=torch.float32, is_shared=False),
        b: TensorDict(
            fields={
                c: MemmapTensor(shape=torch.Size([10]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([10]),
            device=None,
            is_shared=False)},
    batch_size=torch.Size([10]),
    device=None,
    is_shared=False)

Because all of the information to reconstruct nested items is contained in the
corresponding subdirectory, we can also load just the nested ``TensorDict`` by
loading from the sub-directory

.. code-block:: Python

  >>> td3 = TensorDict.load_memmap(prefix="tensordict/b")
  TensorDict(
    fields={
        c: MemmapTensor(shape=torch.Size([10]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([10]),
    device=None,
    is_shared=False)

Handling existing ``MemmapTensors``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the ``TensorDict`` already has ``MemmapTensor`` entries, there are a few
possible behaviours.

- If ``prefix`` is not specified, ``memmap_`` does not modify any existing
  ``MemmapTensors`` in the ``TensorDict``, they will keep their original
  location on disk.
- If ``prefix`` is specified, existing ``MemmapTensor`` entries are not
  modified, and an error will be raised if they are not saved in a location
  consistent with ``prefix`` and their key in the ``TensorDict``.
- If ``prefix`` is specified, and the keyword argument ``copy_existing=True``
  is set, then any existing ``MemmapTensor`` entries are left unmodified if
  they already exist in the correct location, or are copied to the correct
  location if they are not.
