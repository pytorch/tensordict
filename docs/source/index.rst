tensordict: batched, nested dictionaries of PyTorch tensors
===========================================================

A :class:`~tensordict.TensorDict` is a dictionary of tensors that you can use
like one tensor. All its tensors share their leading dimensions, the
``batch_size``, and indexing, shape operations, device moves and arithmetic
apply to every tensor in it.

A TensorDict can contain other TensorDicts. This one holds a batch of 32
transitions, with the next observation in a nested TensorDict:

.. code-block:: text

   TensorDict(batch_size=[32])
   |-- obs:      Tensor[32, 128]
   |-- action:   Tensor[32]
   |-- reward:   Tensor[32]
   `-- next:
       `-- obs:  Tensor[32, 128]

The code below builds it, then slices, moves, scales and stacks all the
tensors with one call each:

  >>> import torch
  >>> from tensordict import TensorDict
  >>> batch = TensorDict(
  ...     {
  ...         "obs": torch.randn(32, 128),
  ...         "action": torch.randint(0, 4, (32,)),
  ...         "reward": torch.randn(32),
  ...         "next": {"obs": torch.randn(32, 128)},
  ...     },
  ...     batch_size=[32],
  ... )
  >>> mini = batch[:8]
  >>> mini.shape
  torch.Size([8])
  >>> mini["next", "obs"].shape
  torch.Size([8, 128])
  >>> device = "cuda" if torch.cuda.is_available() else "cpu"
  >>> on_device = batch.to(device)
  >>> scaled = batch * 0.5
  >>> stacked = torch.stack([batch, batch], 0)
  >>> stacked.shape
  torch.Size([2, 32])

Because a model and a loss can read and write the same TensorDict, one
training loop works for data with different keys: classification,
segmentation or reinforcement learning.

Where to go
-----------

- :ref:`Getting started <getting-started>`: a one-hour tour of the main
  features, with examples that you can run. Start here.
- :ref:`User guide <overview>`: one page for each concept, such as the batch
  size, keys, stacking, devices, non-tensor data and the typed containers.
- :ref:`How-to guides <saving>`: one page for each task, such as saving a
  TensorDict, storing a large dataset, loading data, distributed training and
  :func:`torch.compile`.
- :doc:`Tutorials <tutorials/tensordict_keys>`: scripts that you can run or
  download as notebooks, in four groups (basics, storage, ``tensordict.nn``
  and data loading).
- :ref:`API reference <api-reference>`: every public class, function and
  method.

Install
-------

.. code-block:: bash

   pip install tensordict

tensordict requires Python 3.10 or later and PyTorch 2.13 or later.
:ref:`installation` covers conda-forge, nightly builds, the optional
dependencies, source installs and the TorchRL versions.

.. toctree::
   :hidden:
   :caption: Get started

   installation
   getting_started

.. toctree::
   :hidden:
   :caption: User guide

   overview
   concepts/batch_size
   concepts/keys
   concepts/stacking
   concepts/operations
   concepts/devices
   concepts/non_tensor
   tensorclass
   typed_tensordict
   choosing_a_container
   modules

.. toctree::
   :hidden:
   :caption: How-to guides

   saving
   storage
   data_loading
   distributed
   compile
   printing
   compatibility
   fx

.. toctree::
   :hidden:
   :caption: Tutorials: basics

   tutorials/tensordict_keys
   tutorials/tensordict_slicing
   tutorials/tensordict_shapes
   tutorials/tensordict_preallocation
   tutorials/streamed_tensordict

.. toctree::
   :hidden:
   :caption: Tutorials: storage

   tutorials/tensordict_memory
   tutorials/zarr_storage
   tutorials/serialization_speed

.. toctree::
   :hidden:
   :caption: Tutorials: tensordict.nn

   tutorials/tensordict_module
   tutorials/functional
   tutorials/export

.. toctree::
   :hidden:
   :caption: Tutorials: data loading

   tutorials/data_fashion
   tutorials/tensorclass_fashion
   tutorials/tensorclass_imagenet

.. toctree::
   :hidden:
   :caption: API reference

   reference/index

Indices
-------

* :ref:`genindex`
* :ref:`search`
