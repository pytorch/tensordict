TensorDict
==========

A ``dict[str, torch.Tensor]`` that behaves like one tensor.

PyTorch code that handles groups of tensors, such as a training batch, a
particle system, or a batch of graphs, repeats every operation for every
tensor:

.. code-block:: python

   def shuffle_and_split(images, labels, masks, weights):
       perm = torch.randperm(len(images))
       images, labels, masks, weights = images[perm], labels[perm], masks[perm], weights[perm]
       half = len(images) // 2
       first = images[:half], labels[:half], masks[:half], weights[:half]
       second = images[half:], labels[half:], masks[half:], weights[half:]
       return first, second

With a :class:`~tensordict.TensorDict`, the group is one object, and the
function body is two lines:

  >>> import torch
  >>> from tensordict import TensorDict
  >>> def shuffle_and_split(batch):
  ...     batch = batch[torch.randperm(len(batch))]
  ...     return batch.split(len(batch) // 2)

Build a TensorDict from the four tensors, then call the function:

  >>> batch = TensorDict(
  ...     {
  ...         "images": torch.randn(32, 3, 64, 64),
  ...         "labels": torch.randint(10, (32,)),
  ...         "masks": torch.ones(32, 64, 64, dtype=torch.bool),
  ...         "weights": torch.rand(32),
  ...     },
  ...     batch_size=[32],  # the leading dimension that all tensors share
  ... )
  >>> first, second = shuffle_and_split(batch)
  >>> first["images"].shape
  torch.Size([16, 3, 64, 64])

The new function names no tensor, so it does not change when you add one.

TensorDict: keys at run time, like a dict
-----------------------------------------

You add and read keys at run time, as in a ``dict``. A ``dict`` value becomes
a nested TensorDict, and a tuple of keys reads its entries. The tensors share
the batch size, so indexing, reshaping, stacking, and ``.to(device)`` apply to
every tensor at once, nested ones too:

  >>> batch["meta"] = {"id": torch.arange(32)}
  >>> batch[:8]["meta", "id"].shape
  torch.Size([8])
  >>> batch.reshape(4, 8)["images"].shape
  torch.Size([4, 8, 3, 64, 64])
  >>> torch.stack([batch, batch]).batch_size
  torch.Size([2, 32])
  >>> batch = batch.to("cuda" if torch.cuda.is_available() else "cpu")

Every tensor that you add must have the batch dimensions, so the tensors of
a group stay aligned:

  >>> batch["scores"] = torch.ones(31)
  Traceback (most recent call last):
  ...
  RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([32]) and value.shape=torch.Size([31]) for key 'scores'. The leading dimensions of a value must match the batch size.

You can also :ref:`save it to disk <saving>`, :ref:`pass it to modules
<modules-guide>`, and :ref:`compile functions that take it <compile>`.

TensorClass: fields and methods, like a dataclass
-------------------------------------------------

:class:`~tensordict.TensorClass` is the class form of the same idea. You
declare typed fields and write methods, as for a dataclass. A method that you
write once runs on one item or on a batch of any shape, and it compiles with
``fullgraph=True``:

  >>> from tensordict import TensorClass
  >>> class Particles(TensorClass):
  ...     pos: torch.Tensor  # [*batch, 3]
  ...     vel: torch.Tensor  # [*batch, 3]
  ...     mass: torch.Tensor  # [*batch]
  ...
  ...     def step(self, force: torch.Tensor, dt: float) -> "Particles":
  ...         vel = self.vel + dt * force / self.mass.unsqueeze(-1)
  ...         pos = self.pos + dt * vel
  ...         return Particles(pos=pos, vel=vel, mass=self.mass, batch_size=self.batch_size)
  >>> systems = Particles(  # 64 systems of 1,000 particles each
  ...     pos=torch.randn(64, 1000, 3),
  ...     vel=torch.zeros(64, 1000, 3),
  ...     mass=torch.ones(64, 1000),
  ...     batch_size=[64, 1000],
  ... )
  >>> systems = systems.step(torch.randn(64, 1000, 3), dt=0.01)  # all 64 systems
  >>> one = systems[0].step(torch.randn(1000, 3), dt=0.01)  # one system
  >>> type(one).__name__, one.batch_size
  ('Particles', torch.Size([1000]))
  >>> step = torch.compile(Particles.step, fullgraph=True)
  >>> step(systems, torch.randn(64, 1000, 3), 0.01).batch_size
  torch.Size([64, 1000])

The fields are class attributes, so your editor completes ``systems.pos``.
Indexing, stacking, ``.to(device)``, and saving work as for a TensorDict.
:ref:`tensorclass-guide` explains TensorClass, and
:ref:`choosing-a-container` helps you choose between the two forms.

Install
-------

.. code-block:: bash

   pip install tensordict

tensordict needs Python 3.10+ and PyTorch 2.11+. :ref:`installation` covers
nightly builds, the optional dependencies, conda-forge, and source installs.

Where to go
-----------

- :ref:`Getting started <getting-started>`: a one-hour tour, with examples
  that you can run. Start here.
- :ref:`User guide <overview>`: one page for each concept, from the batch
  size to TensorClass and ``tensordict.nn``.
- :ref:`How-to guides <saving>`: one page for each task, such as saving,
  large datasets, data loading, distributed training, and
  :func:`torch.compile`.
- :doc:`Tutorials <tutorials/tensordict_keys>`: scripts that you can run or
  download as notebooks.
- :ref:`API reference <api-reference>`: every public class, function, and
  method. The :ref:`genindex` lists them from A to Z.

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
   saving_format
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
