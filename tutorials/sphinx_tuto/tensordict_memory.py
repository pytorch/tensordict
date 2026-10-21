"""
Devices and memory-mapped tensors
=================================
**Author**: `Tom Begley <https://github.com/tcbegley>`_

One call moves all the tensors of a :class:`~tensordict.TensorDict` to a
device, and one call stores them in memory-mapped files. This tutorial is for
readers who move groups of tensors between the CPU and a GPU, or whose
datasets do not fit in RAM.

What you will learn:

- How the ``device`` of a tensordict applies to its entries.
- How to move a tensordict to another device, and how to remove its device.
- How to convert a tensordict to memory-mapped tensors with ``memmap_``, and
  how to allocate an empty memory-mapped buffer with ``memmap_like``.
- How to save memory-mapped tensors to a directory and load them again.
"""

##############################################################################
# Devices
# -------
# When you create a :class:`~tensordict.TensorDict`, you can give it a device
# with the ``device`` keyword argument. If you set the ``device``, the
# :class:`~tensordict.TensorDict` moves all its entries to this device. If you
# do not set it, the entries can be on different devices.
#
# The examples of this section need a CUDA device, so this page shows them
# with their output, and does not run them. This example creates a
# :class:`~tensordict.TensorDict` with ``device="cuda:0"``. The printed
# tensordict shows the device once, for all its entries.
#
# .. code-block::
#
#    >>> import torch
#    >>> from tensordict import TensorDict
#    >>> tensordict = TensorDict({"a": torch.rand(10)}, [10], device="cuda:0")
#    >>> print(tensordict)
#    TensorDict(batch_size=[10], device=cuda:0, fields={a: f32[]})
#
# If the device of the :class:`~tensordict.TensorDict` is not ``None``, it also
# moves the new entries to the device.
#
# .. code-block::
#
#    >>> tensordict["b"] = torch.rand(10, 10)
#    >>> print(tensordict)
#    TensorDict(batch_size=[10], device=cuda:0, fields={a: f32[], b: f32[10]})
#
# The ``device`` attribute gives the device of the
# :class:`~tensordict.TensorDict`.
#
# .. code-block::
#
#    >>> print(tensordict.device)
#    cuda:0
#
# To send the contents of the :class:`~tensordict.TensorDict` to a device, use
# :meth:`TensorDict.cuda() <tensordict.TensorDict.cuda>` or
# :meth:`TensorDict.to(device) <tensordict.TensorDict.to>`, as for a PyTorch
# tensor. Like the tensor methods, they return a new tensordict and do not
# change the original one.
#
# .. code-block::
#
#    >>> tensordict = tensordict.to(torch.device("cpu"))
#    >>> print(tensordict)
#    TensorDict(batch_size=[10], device=cpu, fields={a: f32[], b: f32[10]})
#    >>> tensordict = tensordict.cuda()
#    >>> print(tensordict)
#    TensorDict(batch_size=[10], device=cuda:0, fields={a: f32[], b: f32[10]})
#
# :meth:`TensorDict.to <tensordict.TensorDict.to>` needs a device as its
# argument. To remove the device of the :class:`~tensordict.TensorDict`, so
# that its entries can be on different devices, use
# :meth:`TensorDict.clear_device_ <tensordict.TensorDict.clear_device_>`. The
# entries stay where they are, and the printed form shows the device of each
# entry that is not on the CPU.
#
# .. code-block::
#
#    >>> tensordict = tensordict.clear_device_()
#    >>> print(tensordict)
#    TensorDict(batch_size=[10], fields={a: f32[]@cuda:0, b: f32[10]@cuda:0})
#
# Memory-mapped tensors
# ---------------------
# A :class:`~tensordict.MemoryMappedTensor` is a tensor whose values are stored
# in a file. The operating system reads the parts of the file that you index,
# so you can read batches from a dataset that does not fit in RAM. The
# :ref:`sphx_glr_tutorials_tensorclass_imagenet.py` tutorial uses memory-mapped
# tensors to load a dataset.
#
# To convert the entries of a :class:`~tensordict.TensorDict` to memory-mapped
# tensors, use :meth:`TensorDict.memmap_ <tensordict.TensorDict.memmap_>`. It
# copies the values into the memory maps, and locks the tensordict: you can
# write into its tensors, but you cannot add or remove keys.

import tempfile
from pathlib import Path

import torch
from tensordict import TensorDict

torch.manual_seed(0)

tensordict = TensorDict({"a": torch.rand(10), "b": {"c": torch.rand(10)}}, [10])
tensordict.memmap_()

print(tensordict)
assert tensordict.is_locked

##############################################################################
# :meth:`TensorDict.memmap_like <tensordict.TensorDict.memmap_like>` creates a
# new tensordict with the same keys and shapes, with
# :class:`~tensordict.MemoryMappedTensor` values. It does not copy the values:
# the output below shows zeros. The new tensordict is also locked.
#
# - Use ``memmap_`` to move existing data into memory maps.
# - Use ``memmap_like`` to allocate a buffer that you fill later, for example
#   a dataset that does not fit in RAM.

tensordict = TensorDict({"a": torch.rand(10), "b": {"c": torch.rand(10)}}, [10])
mm_tensordict = tensordict.memmap_like()

print(mm_tensordict["a"].contiguous())

##############################################################################
# Write into the buffer with an index, or with an in-place operation:

mm_tensordict[:5] = tensordict[:5]
assert (mm_tensordict["a"][:5] == tensordict["a"][:5]).all()

##############################################################################
# Saving to a directory and loading again
# ---------------------------------------
#
# By default, the memory maps are anonymous and temporary. On Linux, they are
# in shared memory under ``/dev/shm`` when it has room. Otherwise, they are in
# an unlinked temporary file, or in a shared-memory block on Windows. They
# disappear when the process ends, and you cannot load them again. To save
# the memory maps to a directory, pass ``prefix="/path/to/root"``.
#
# The directory has the structure of the :class:`~tensordict.TensorDict`. Each
# tensor is a raw ``.memmap`` file, and the metadata is in ``meta.json``
# files. For example, the :class:`~tensordict.TensorDict` above is saved as
# follows:
#
# ::
#
#    ├── a.memmap
#    ├── b
#    │ ├── c.memmap
#    │ └── meta.json
#    └── meta.json

prefix = Path(tempfile.mkdtemp())
tensordict.memmap_(prefix=prefix)
print(sorted(str(path.relative_to(prefix)) for path in prefix.rglob("*")))

##############################################################################
# :meth:`TensorDict.load_memmap <tensordict.TensorDict.load_memmap>` loads the
# directory again, in this process or in another one. The entries of the
# result are memory-mapped tensors that read from the files:

loaded = TensorDict.load_memmap(prefix)
print(loaded)
assert (loaded == tensordict).all()

##############################################################################
# Conclusion
# ----------
#
# You can now set the device of a :class:`~tensordict.TensorDict`, move it to
# another device, and remove its device. You can convert a tensordict to
# memory-mapped tensors with ``memmap_``, or allocate an empty memory-mapped
# buffer with ``memmap_like``. With ``prefix=``, the memory maps go to a
# directory, and ``load_memmap`` loads them again, in this process or in
# another one.
#
# Further reading
# ---------------
#
# - :ref:`devices` explains devices, dtypes, and memory.
# - :ref:`saving` explains all the ways to save and load tensordicts.
# - :ref:`storage` explains how to work with datasets that do not fit in RAM.
# - :ref:`sphx_glr_tutorials_tensorclass_imagenet.py` loads an image dataset
#   from memory-mapped tensors.
