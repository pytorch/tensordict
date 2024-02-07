.. currentmodule:: tensordict

tensordict package
==================

The :class:`~tensordict.TensorDict` class simplifies the process of passing multiple tensors
from module to module by packing them in a dictionary-like object that inherits features from
regular pytorch tensors.


.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    TensorDictBase
    TensorDict
    SubTensorDict
    LazyStackedTensorDict
    PersistentTensorDict
    TensorDictParams

TensorDict as a context manager
-------------------------------

:class:`~tensordict.TensorDict` can be used as a context manager in situations
where an action has to be done and then undone. This include temporarily
locking/unlocking a tensordict

    >>> data.lock_()  # data.set will result in an exception
    >>> with data.unlock_():
    ...     data.set("key", value)
    >>> assert data.is_locked()

or to execute functional calls with a TensorDict instance containing the
parameters and buffers of a model:

    >>> params = TensorDict.from_module(module).clone()
    >>> params.zero_()
    >>> with params.to_module(module):
    ...     y = module(x)

In the first example, we can modify the tensordict `data` because we have
temporarily unlocked it. In the second example, we populate the module with the
parameters and buffers contained in the `params` tensordict instance, and reset
the original parameters after this call is completed.

Memory-mapped tensors
---------------------

`tensordict` offers the :class:`~tensordict.MemoryMappedTensor` primitive which
allows you to work with tensors stored in physical memory in a handy way.
The main advantages of :class:`~tensordict.MemoryMappedTensor`
are its ease of construction (no need to handle the storage of a tensor),
the possibility to work with big contiguous data that would not fit in memory,
an efficient (de)serialization across processes and efficient indexing of
stored tensors.

If all workers have access to the same storage (both in multiprocess and distributed
settings), passing a :class:`~tensordict.MemoryMappedTensor`
will just consist in passing a reference to a file on disk plus a bunch of
extra meta-data for reconstructing it. The same goes with indexed memory-mapped
tensors as long as the data-pointer of their storage is the same as the original
one.

Indexing memory-mapped tensors is much faster than loading several independent files from
the disk and does not require to load the full content of the array in memory.
However, physical storage of PyTorch tensors should not be any different:

.. code-block:: Python

  >>> my_images = MemoryMappedTensor.empty((1_000_000, 3, 480, 480), dtype=torch.unint8)
  >>> mini_batch = my_images[:10]  # just reads the first 10 images of the dataset

.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    MemoryMappedTensor


Utils
-----

.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    utils.expand_as_right
    utils.expand_right
    utils.isin
    utils.remove_duplicates
    is_memmap
    is_batchedtensor
    is_tensor_collection
    make_tensordict
    merge_tensordicts
    pad
    pad_sequence
    dense_stack_tds
    set_lazy_legacy
    lazy_legacy
