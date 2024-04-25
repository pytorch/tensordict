# -*- coding: utf-8 -*-
"""
Simplifying PyTorch Memory Management with TensorDict
=====================================================
**Author**: `Tom Begley <https://github.com/tcbegley>`_

In this tutorial you will learn how to control where the contents of a
:class:`TensorDict` are stored in memory, either by sending those contents to a device,
or by utilizing memory maps.
"""

##############################################################################
# Devices
# -------
# When you create a :class:`TensorDict`, you can specify a device with the ``device``
# keyword argument. If the ``device`` is set, then all entries of the
# :class:`TensorDict` will be placed on that device. If the ``device`` is not set, then
# there is no requirement that entries in the :class:`TensorDict` must be on the same
# device.
#
# In this example we instantiate a :class:`TensorDict` with ``device="cuda:0"``. When
# we print the contents we can see that they have been moved onto the device.
#
# .. code-block::
#
#    >>> import torch
#    >>> from tensordict import TensorDict
#    >>> tensordict = TensorDict({"a": torch.rand(10)}, [10], device="cuda:0")
#    >>> print(tensordict)
#    TensorDict(
#        fields={
#            a: Tensor(shape=torch.Size([10]), device=cuda:0, dtype=torch.float32, is_shared=True)},
#        batch_size=torch.Size([10]),
#        device=cuda:0,
#        is_shared=True)
#
# If the device of the :class:`TensorDict` is not ``None``, new entries are also moved
# onto the device.
#
# .. code-block::
#
#    >>> tensordict["b"] = torch.rand(10, 10)
#    >>> print(tensordict)
#    TensorDict(
#        fields={
#            a: Tensor(shape=torch.Size([10]), device=cuda:0, dtype=torch.float32, is_shared=True),
#            b: Tensor(shape=torch.Size([10, 10]), device=cuda:0, dtype=torch.float32, is_shared=True)},
#        batch_size=torch.Size([10]),
#        device=cuda:0,
#        is_shared=True)
#
# You can check the current device of the :class:`TensorDict` with the ``device``
# attribute.
#
# .. code-block::
#
#    >>> print(tensordict.device)
#    cuda:0
#
# The contents of the :class:`TensorDict` can be sent to a device like a PyTorch tensor
# with :meth:`TensorDict.cuda() <tensordict.TensorDict.cuda>` or
# :meth:`TensorDict.device(device) <tensordict.TensorDict.device>` with ``device``
# being the desired device.
#
# .. code-block::
#
#    >>> tensordict.to(torch.device("cpu"))
#    >>> print(tensordict)
#    TensorDict(
#        fields={
#            a: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float32, is_shared=False),
#            b: Tensor(shape=torch.Size([10, 10]), device=cpu, dtype=torch.float32, is_shared=False)},
#        batch_size=torch.Size([10]),
#        device=cpu,
#        is_shared=False)
#    >>> tensordict.cuda()
#    >>> print(tensordict)
#    TensorDict(
#        fields={
#            a: Tensor(shape=torch.Size([10]), device=cuda:0, dtype=torch.float32, is_shared=True),
#            b: Tensor(shape=torch.Size([10, 10]), device=cuda:0, dtype=torch.float32, is_shared=True)},
#        batch_size=torch.Size([10]),
#        device=cuda:0,
#        is_shared=True)
#
# The :meth:`TensorDict.device <tensordict.TensorDict.device>` method requires a valid
# device to be passed as the argument. If you want to remove the device from the
# :class:`TensorDict` to allow values with different devices, you should use the
# :meth:`TensorDict.clear_device <tensordict.TensorDict.clear_device>` method.
#
# .. code-block::
#
#    >>> tensordict.clear_device()
#    >>> print(tensordict)
#    TensorDict(
#        fields={
#            a: Tensor(shape=torch.Size([10]), device=cuda:0, dtype=torch.float32, is_shared=True),
#            b: Tensor(shape=torch.Size([10, 10]), device=cuda:0, dtype=torch.float32, is_shared=True)},
#        batch_size=torch.Size([10]),
#        device=None,
#        is_shared=False)
#
# Memory-mapped Tensors
# ---------------------
# ``tensordict`` provides a class :class:`~tensordict.MemoryMappedTensor`
# which allows us to store the contents of a tensor on disk, while still
# supporting fast indexing and loading of the contents in batches.
# See the `ImageNet Tutorial <./tensorclass_imagenet.html>`_ for an
# example of this in action.
#
# To convert the :class:`TensorDict` to a collection of memory-mapped tensors, use the
# :meth:`TensorDict.memmap_ <tensordict.TensorDict.memmap_>`.

# sphinx_gallery_start_ignore
import warnings

import torch
from tensordict import TensorDict

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

tensordict = TensorDict({"a": torch.rand(10), "b": {"c": torch.rand(10)}}, [10])
tensordict.memmap_()

print(tensordict)

##############################################################################
# Alternatively one can use the
# :meth:`TensorDict.memmap_like <tensordict.TensorDict.memmap_like>` method. This will
# create a new :class:`~.TensorDict` of the same structure with
# :class:`~tensordict.MemoryMappedTensor` values, however it will not copy the
# contents of the original tensors to the
# memory-mapped tensors. This allows you to create the memory-mapped
# :class:`~.TensorDict` and then populate it slowly, and hence should generally be
# preferred to ``memmap_``.

tensordict = TensorDict({"a": torch.rand(10), "b": {"c": torch.rand(10)}}, [10])
mm_tensordict = tensordict.memmap_like()

print(mm_tensordict["a"].contiguous())

##############################################################################
# By default the contents of the :class:`TensorDict` will be saved to a temporary
# location on disk, however if you would like to control where they are saved you can
# use the keyword argument ``prefix="/path/to/root"``.
#
# The contents of the :class:`TensorDict` are saved in a directory structure that mimics
# the structure of the :class:`TensorDict` itself. The contents of the tensor is saved
# in a NumPy memmap, and the metadata in an associated PyTorch save file. For example,
# the above :class:`TensorDict` is saved as follows:
#
# ::
#
#    ├── a.memmap
#    ├── a.meta.pt
#    ├── b
#    │   ├── c.memmap
#    │   ├── c.meta.pt
#    │   └── meta.pt
#    └── meta.pt
