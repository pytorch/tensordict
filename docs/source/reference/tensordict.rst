.. currentmodule:: tensordict

tensordict package
==================

The `TensorDict` class simplifies the process of passing multiple tensors
from module to module by packing them in a dictionary-like object that inherits features from
regular pytorch tensors.


.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    TensorDict
    SubTensorDict
    LazyStackedTensorDict

Memory-mapped tensors
---------------------

:obj:`tensordict` offers the :obj:`MemmapTensor` primitive which allows you to work
with tensors stored in physical memory in a handy way. The main advantages of :obj:`MemmapTensor`
are its easiness of construction (no need to handle the storage of a tensor), the possibility to
work with big contiguous data that would not fit in memory, an efficient (de)serialization across processes and
efficient indexing of stored tensors.

If all workers have access to the same storage, passing a :obj:`MemmapTensor` will just consist in passing
a reference to a file on disk plus a bunch of extra meta-data for reconstructing it when
sent across processes or workers on a same machine (both in multiprocess and distributed settings).
The same goes with indexed memory-mapped tensors.

Indexing memory-mapped tensors is much faster than loading several independent files from
the disk and does not require to load the full content of the array in memory.
However, physical storage of PyTorch tensors should not be any different:

.. code-block:: Python

  >>> my_images = MemmapTensor(1_000_000, 3, 480, 480, dtype=torch.unint8)
  >>> mini_batch = my_images[:10]  # just reads the first 10 images of the dataset
  >>> mini_batch = my_images.as_tensor()[:10]  # similar but using pytorch tensors directly

The main difference between the two examples above is that, in the first case, indexing
returns a :obj:`MemmapTensor` instance, whereas in the second a :ob:`torch.Tensor` is returned.

.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    MemmapTensor

Utils
-----

.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    utils.expand_as_right
    utils.expand_right
