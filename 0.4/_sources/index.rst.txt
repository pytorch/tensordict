.. tensordict documentation master file, created by
   sphinx-quickstart on Mon Mar  7 13:23:20 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the TensorDict Documentation!
========================================

`TensorDict` is a dictionary-like class that inherits properties from tensors,
such as indexing, shape operations, casting to device etc.

You can install tensordict directly from PyPI (see more about installation
instructions in the dedicated section below):

.. code-block::

  $ pip install tensordict


The main purpose of TensorDict is to make code-bases more *readable* and *modular*
by abstracting away tailored operations:

  >>> for i, tensordict in enumerate(dataset):
  ...     # the model reads and writes tensordicts
  ...     tensordict = model(tensordict)
  ...     loss = loss_module(tensordict)
  ...     loss.backward()
  ...     optimizer.step()
  ...     optimizer.zero_grad()

With this level of abstraction, one can recycle a training loop for highly heterogeneous task.
Each individual step of the training loop (data collection and transform, model
prediction, loss computation etc.)
can be tailored to the use case at hand without impacting the others.
For instance, the above example can be easily used across classification and segmentation tasks, among many others.


Installation
============

Tensordict releases are synced with PyTorch, so make sure you always enjoy the latest
features of the library with the `most recent version of PyTorch <https://pytorch.org/get-started/locally/>`__ (although core features
are guaranteed to be backward compatible with pytorch>=1.13).
Nightly releases can be installed via

.. code-block::

  $ pip install tensordict-nightly

or via a `git clone` if you're willing to contribute to the library:

.. code-block::

  $ cd path/to/root
  $ git clone https://github.com/pytorch/tensordict
  $ cd tensordict
  $ python setup.py develop

Tutorials
=========

Basics
------

.. toctree::
   :maxdepth: 1

   tutorials/tensordict_basics
   tutorials/tensordict_shapes
   tutorials/tensordict_slicing
   tutorials/tensordict_keys
   tutorials/tensordict_preallocation
   tutorials/tensordict_memory

tensordict.nn
-------------

.. toctree::
   :maxdepth: 1

   tutorials/tensordict_module
   tutorials/tensordict_module_functional

Dataloading
-----------

.. toctree::
   :maxdepth: 1

   tutorials/data_fashion
   tutorials/tensorclass_fashion
   tutorials/tensorclass_imagenet

Contents
========

.. toctree::
   :maxdepth: 3

   overview
   distributed
   fx
   saving
   reference/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
