.. tensordict documentation master file, created by
   sphinx-quickstart on Mon Mar  7 13:23:20 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the TensorDict Documentation!
=====================================

`TensorDict` is a dictionary-like class that inherits properties from tensors, such as indexing, shape operations, casting to device etc.

The main purpose of TensorDict is to make code-bases more *readable* and *modular* by abstracting away tailored operations:

  >>> for i, tensordict in enumerate(dataset):
  ...     # the model reads and writes tensordicts
  ...     tensordict = model(tensordict)
  ...     loss = loss_module(tensordict)
  ...     loss.backward()
  ...     optimizer.step()
  ...     optimizer.zero_grad()

With this level of abstraction, one can recycle a training loop for highly heterogeneous task.
Each individual step of the training loop (data collection and transform, model prediction, loss computation etc.)
can be tailored to the use case at hand without impacting the others.
For instance, the above example can be easily used across classification and segmentation tasks, among many others.

Tutorials
=========

.. toctree::
   :maxdepth: 2

   tutorials/data_fashion
   tutorials/tensorclass_fashion
   tutorials/tensorclass_imagenet

Contents
========

.. toctree::
   :maxdepth: 3

   overview
   distributed
   reference/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
