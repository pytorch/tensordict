.. currentmodule:: tensordict

.. meta::
   :description: Replace a hand-written PyTorch dataclass of tensors, with its
      own to(device), pin_memory, __getitem__ and collate_fn, by a TensorClass
      that has these operations and compiles with torch.compile.

.. _coming-from-dataclass:

Coming from a hand-written dataclass
====================================

Many PyTorch projects keep a batch in a ``@dataclass`` of tensors, and then
write the operations that the batch needs: a ``to(device)`` and a
``pin_memory()`` that move every field, a ``__getitem__`` that slices every
field, and a ``stack`` or ``collate_fn`` helper for the
:class:`~torch.utils.data.DataLoader`. Each new field means an edit to each of
these methods. A :class:`~tensordict.TensorClass` declares its fields in the
same way, and has these operations for all its fields, nested ones too. It is
also a pytree, so :func:`torch.compile`, :func:`torch.vmap` and
:mod:`torch.func` accept it.

This page shows each hand-written method next to what replaces it, then three
worked examples: a rollout batch for RL post-training, a ray bundle for
rendering, and a batch of images with nested targets. Run the examples in
order in one Python session:

  >>> import torch
  >>> from torch.utils.data import DataLoader
  >>> from tensordict import TensorClass, TensorDict, collate
  >>> _ = torch.manual_seed(0)
  >>> device = "cuda" if torch.cuda.is_available() else "cpu"

The mapping
-----------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Hand-written
     - TensorClass
   * - ``@dataclass`` with tensor fields
     - A subclass of :class:`~tensordict.TensorClass` with the same
       annotations, and a ``batch_size``
   * - ``def to(self, device)``, ``def cuda(self)``
     - ``batch.to(device)``, ``batch.to(device, non_blocking=True)``,
       ``batch.cuda()``
   * - ``def pin_memory(self)``
     - ``batch.pin_memory()``
   * - ``def __getitem__(self, idx)`` that slices every field
     - ``batch[idx]``, with an int, a slice, a boolean mask or an index tensor
   * - ``def __len__(self)``
     - ``len(batch)``, the first dimension of ``batch.batch_size``
   * - A ``stack`` or ``cat`` helper
     - ``torch.stack(items)``, ``torch.cat(batches)``
   * - A ``collate_fn`` for the ``DataLoader``
     - ``collate_fn=tensordict.collate`` (see :func:`~tensordict.collate`)
   * - A helper that splits a batch into minibatches
     - ``batch.split(size)``, ``batch.chunk(n)``
   * - A field that holds another dataclass of tensors
     - A field annotated with another ``TensorClass``
   * - ``dataclasses.replace(batch, ...)``
     - ``batch.replace(...)``
   * - ``torch.utils._pytree.register_pytree_node(...)``
     - No call: defining the class registers it
   * - A ``@dataclass`` that you cannot change
     - :func:`~tensordict.from_dataclass`

Declare the fields
------------------

A hand-written batch often starts like this:

.. code-block:: python

   @dataclass
   class Batch:
       images: torch.Tensor  # [B, 3, H, W]
       labels: torch.Tensor  # [B]

A tensorclass has the same annotations. The constructor also takes the
*batch size*: the leading dimensions that all the fields share (see
:ref:`batch-size`). The operations below apply to these dimensions:

  >>> class Batch(TensorClass):
  ...     images: torch.Tensor  # [*batch, 3, H, W]
  ...     labels: torch.Tensor  # [*batch]
  >>> batch = Batch(
  ...     images=torch.randn(16, 3, 8, 8),
  ...     labels=torch.randint(10, (16,)),
  ...     batch_size=[16],
  ... )
  >>> len(batch)
  16

A field with a wrong batch dimension raises an error when you build the
batch, not later in the model:

  >>> Batch(images=torch.randn(16, 3, 8, 8), labels=torch.zeros(15), batch_size=[16])
  Traceback (most recent call last):
  ...
  RuntimeError: batch dimension mismatch, ...

Move to a device and pin memory
-------------------------------

The hand-written methods list every field:

.. code-block:: python

   def to(self, device, non_blocking=False):
       return Batch(
           images=self.images.to(device, non_blocking=non_blocking),
           labels=self.labels.to(device, non_blocking=non_blocking),
       )

   def pin_memory(self):
       return Batch(images=self.images.pin_memory(), labels=self.labels.pin_memory())

A tensorclass has :meth:`~tensordict.TensorDict.to`,
:meth:`~tensordict.TensorDict.cuda` and
:meth:`~tensordict.TensorDict.pin_memory`. They move every field and return
a ``Batch``:

  >>> if device == "cuda":
  ...     batch = batch.pin_memory()
  >>> on_device = batch.to(device, non_blocking=True)
  >>> type(on_device).__name__, on_device.images.device.type == device
  ('Batch', True)

:ref:`devices` covers the other options of ``to``.

Slice every field
-----------------

A ``__getitem__`` that slices the batch repeats the index for each field:

.. code-block:: python

   def __getitem__(self, idx):
       return Batch(images=self.images[idx], labels=self.labels[idx])

An index on a tensorclass applies to the batch dimensions of every field.
An int, a slice, a boolean mask and an index tensor all return a ``Batch``:

  >>> batch[:4].images.shape
  torch.Size([4, 3, 8, 8])
  >>> keep = torch.tensor([True, False] * 8)
  >>> batch[keep].batch_size
  torch.Size([8])
  >>> shuffled = batch[torch.randperm(len(batch))]
  >>> [mb.batch_size for mb in shuffled.split(6)]
  [torch.Size([6]), torch.Size([6]), torch.Size([4])]

``reshape``, ``unsqueeze``, ``flatten`` and the other shape operations also
apply to the batch dimensions (see :ref:`operations`).

Stack and collate
-----------------

A dataset that returns one sample at a time needs a function that stacks the
samples into a batch, often a static method passed as ``collate_fn``:

.. code-block:: python

   @staticmethod
   def collate(samples):
       return Batch(
           images=torch.stack([s.images for s in samples]),
           labels=torch.stack([s.labels for s in samples]),
       )

:func:`torch.stack` and :func:`torch.cat` accept a list of tensorclasses
and return one. For a ``DataLoader``, pass :func:`~tensordict.collate` as
``collate_fn``. A sample here has an empty batch size:

  >>> samples = [Batch(images=torch.randn(3, 8, 8), labels=torch.tensor(i)) for i in range(10)]
  >>> torch.stack(samples[:3]).batch_size
  torch.Size([3])
  >>> loader = DataLoader(samples, batch_size=4, collate_fn=collate)
  >>> first = next(iter(loader))
  >>> type(first).__name__, first.labels
  ('Batch', tensor([0, 1, 2, 3]))

:func:`~tensordict.collate` also pads the fields whose length differs
between samples, as the last worked example shows. :ref:`data-loading`
covers worker processes and pinned memory.

Nested batches
--------------

A batch often holds other dataclasses, and each hand-written method then
calls the same method on them. Annotate a field with another tensorclass
instead. Indexing, ``to`` and stacking reach the nested fields, and the
fields that are not tensors keep their value:

  >>> class Camera(TensorClass):
  ...     intrinsics: torch.Tensor  # [*batch, 3, 3]
  ...     pose: torch.Tensor  # [*batch, 4, 4]
  >>> class Frames(TensorClass):
  ...     images: torch.Tensor  # [*batch, 3, H, W]
  ...     camera: Camera
  ...     scene: str
  >>> frames = Frames(
  ...     images=torch.randn(5, 3, 8, 8),
  ...     camera=Camera(intrinsics=torch.eye(3).expand(5, 3, 3), pose=torch.eye(4).expand(5, 4, 4)),
  ...     scene="kitchen",
  ...     batch_size=[5],
  ... )
  >>> frames[1:3].camera.pose.shape, frames[1:3].scene
  (torch.Size([2, 4, 4]), 'kitchen')
  >>> type(frames.to(device).camera).__name__
  'Camera'

The nested ``Camera`` takes the batch size of ``Frames``, so it needs no
``batch_size`` of its own.

Convert a dataclass that you keep
---------------------------------

If you cannot change the class, :func:`~tensordict.from_dataclass` converts a
dataclass type or instance to a tensorclass (see
:ref:`tensorclass-from-dataclass`):

  >>> from dataclasses import dataclass
  >>> from tensordict import from_dataclass
  >>> @dataclass
  ... class Pair:
  ...     x: torch.Tensor
  ...     y: torch.Tensor
  >>> pair = from_dataclass(Pair(x=torch.zeros(4, 2), y=torch.ones(4)), batch_size=[4])
  >>> pair[:2].x.shape
  torch.Size([2, 2])

Worked example: a rollout batch for RL post-training
----------------------------------------------------

The rollout batch of an RL post-training loop for a language model holds the
token ids of the prompts and of the sampled responses, the log-probabilities
of the response tokens under the policy that sampled them, an advantage for
each response, and an id for each row, such as the id of the prompt. The
trainer moves the batch to the GPU, shuffles it, and splits it into
minibatches. Hand-written, that is a dataclass with ``to_device``,
``pin_memory`` and a split helper that list every field. With a tensorclass,
the loss is a method:

  >>> class Rollout(TensorClass):
  ...     prompt_ids: torch.Tensor  # [*batch, prompt_len]
  ...     response_ids: torch.Tensor  # [*batch, response_len]
  ...     response_mask: torch.Tensor  # [*batch, response_len], False on padding
  ...     logprobs: torch.Tensor  # [*batch, response_len], from the sampling policy
  ...     advantages: torch.Tensor  # [*batch]
  ...     uid: str
  ...
  ...     def policy_loss(self, new_logprobs: torch.Tensor, clip: float = 0.2) -> torch.Tensor:
  ...         ratio = (new_logprobs - self.logprobs).exp()
  ...         adv = self.advantages.unsqueeze(-1)
  ...         loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
  ...         mask = self.response_mask.float()
  ...         return (loss * mask).sum() / mask.sum()

Each row comes from one generation. :func:`torch.stack` makes the batch, and
the ``str`` field becomes one value for each row:

  >>> rows = [
  ...     Rollout(
  ...         prompt_ids=torch.randint(1000, (6,)),
  ...         response_ids=torch.randint(1000, (10,)),
  ...         response_mask=torch.arange(10) < 4 + i % 6,
  ...         logprobs=-torch.rand(10),
  ...         advantages=torch.randn(()),
  ...         uid=f"prompt-{i // 2}",  # two responses for each prompt
  ...     )
  ...     for i in range(8)
  ... ]
  >>> rollout = torch.stack(rows)
  >>> rollout.batch_size
  torch.Size([8])
  >>> list(rollout[:4].uid)
  ['prompt-0', 'prompt-0', 'prompt-1', 'prompt-1']

Normalize the advantages, move the batch, and loop over shuffled
minibatches. Each minibatch is a ``Rollout`` and keeps its ids:

  >>> rollout.advantages = (rollout.advantages - rollout.advantages.mean()) / rollout.advantages.std()
  >>> rollout = rollout.to(device)
  >>> for mb in rollout[torch.randperm(len(rollout))].split(4):
  ...     new_logprobs = mb.logprobs + 0.01 * torch.randn_like(mb.logprobs)  # the policy's output
  ...     loss = mb.policy_loss(new_logprobs)
  >>> type(mb).__name__, mb.batch_size, len(mb.uid)
  ('Rollout', torch.Size([4]), 4)

The method compiles with ``fullgraph=True``:

  >>> policy_loss = torch.compile(Rollout.policy_loss, fullgraph=True)
  >>> torch.testing.assert_close(policy_loss(mb, new_logprobs), mb.policy_loss(new_logprobs))

Worked example: a ray bundle for rendering
------------------------------------------

A renderer such as a NeRF casts one ray per pixel. A ray bundle holds the
origin and the direction of each ray, the near and far bounds of the samples
along it, and other values for each ray, such as the index of the camera.
Renderers keep the bundle in the shape of the image, flatten it to sample
random rays for training, and split it into chunks that fit in memory for
rendering. A ``metadata`` field that is a :class:`~tensordict.TensorDict`
holds the other values, with keys that you choose at run time:

  >>> class RayBundle(TensorClass):
  ...     origins: torch.Tensor  # [*batch, 3]
  ...     directions: torch.Tensor  # [*batch, 3]
  ...     near: torch.Tensor  # [*batch, 1]
  ...     far: torch.Tensor  # [*batch, 1]
  ...     metadata: TensorDict  # [*batch], per-ray values
  ...
  ...     def sample_points(self, num_samples: int) -> torch.Tensor:
  ...         t = torch.linspace(0, 1, num_samples, device=self.origins.device)
  ...         depths = self.near + (self.far - self.near) * t  # [*batch, num_samples]
  ...         return self.origins.unsqueeze(-2) + depths.unsqueeze(-1) * self.directions.unsqueeze(-2)

The rays of one 4 x 6 image from camera 2:

  >>> H, W = 4, 6
  >>> directions = torch.nn.functional.normalize(torch.randn(H, W, 3), dim=-1)
  >>> rays = RayBundle(
  ...     origins=torch.zeros(H, W, 3),
  ...     directions=directions,
  ...     near=torch.full((H, W, 1), 0.1),
  ...     far=torch.full((H, W, 1), 4.0),
  ...     metadata=TensorDict(camera_idx=torch.full((H, W), 2)),
  ...     batch_size=[H, W],
  ... )
  >>> rays.sample_points(8).shape
  torch.Size([4, 6, 8, 3])

The method runs on the whole image, on one row of pixels, or on a random
set of rays for a training step:

  >>> rays[0].sample_points(8).shape
  torch.Size([6, 8, 3])
  >>> train_rays = rays.reshape(-1)[torch.randint(H * W, (10,))]
  >>> train_rays.batch_size, train_rays.metadata["camera_idx"].shape
  (torch.Size([10]), torch.Size([10]))

To render the image in chunks, flatten it, split it, and put the outputs
back in the shape of the image:

  >>> chunks = [chunk.sample_points(8).mean(-2) for chunk in rays.reshape(-1).split(10)]
  >>> torch.cat(chunks).reshape(H, W, 3).shape
  torch.Size([4, 6, 3])

Worked example: images with nested targets
------------------------------------------

A detection or segmentation model takes a batch of images and, for each
image, targets: the boxes and the labels of the objects in it. The number of
objects differs between images, so the hand-written collate function pads
each target field and builds a mask of the valid objects. With tensorclasses,
the targets are a nested class:

  >>> from functools import partial
  >>> from typing import Optional
  >>> class Targets(TensorClass):
  ...     boxes: torch.Tensor  # [*batch, num_objects, 4]
  ...     labels: torch.Tensor  # [*batch, num_objects]
  >>> class Datapoint(TensorClass):
  ...     image: torch.Tensor  # [*batch, 3, H, W]
  ...     targets: Targets
  ...     image_id: str
  ...     mask: Optional[TensorDict] = None  # filled by collate

A dataset returns one ``Datapoint`` for each image, with an empty batch size:

  >>> def load(i):
  ...     num_objects = i % 3 + 1
  ...     return Datapoint(
  ...         image=torch.randn(3, 16, 16),
  ...         targets=Targets(boxes=torch.rand(num_objects, 4), labels=torch.arange(num_objects)),
  ...         image_id=f"img-{i}",
  ...     )
  >>> dataset = [load(i) for i in range(6)]

:func:`~tensordict.collate` with ``pad_keys`` pads the target fields to the
image with the most objects, each with its own padding value. With
``mask_key``, it stores the masks of the valid objects in the ``mask``
field:

  >>> collate_fn = partial(
  ...     collate,
  ...     pad_keys={("targets", "boxes"): 0.0, ("targets", "labels"): -1},
  ...     mask_key="mask",
  ... )
  >>> loader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)
  >>> batch = next(iter(loader))
  >>> batch.targets.boxes.shape
  torch.Size([3, 3, 4])
  >>> batch.targets.labels[1]
  tensor([ 0,  1, -1])
  >>> batch.mask["targets", "labels"].sum(-1)
  tensor([1, 2, 3])

The batch moves to the device and slices as one object, nested targets,
masks and image ids included:

  >>> batch = batch.to(device)
  >>> one = batch[1]
  >>> one.image_id, one.targets.boxes.shape, one.mask["targets", "labels"].tolist()
  ('img-1', torch.Size([3, 4]), [True, True, False])

Without ``pad_keys``, :func:`~tensordict.collate` keeps each image's targets
as they are, in a lazy stack (see :ref:`stacking-variable-size`).

See also
--------

- :ref:`tensorclass-guide`: the fields, methods and options of a
  tensorclass.
- :ref:`data-loading`: datasets, ``collate_fn``, worker processes and pinned
  memory.
- :ref:`coming-from-jax`: the JAX pytree idioms in tensordict.
- :ref:`compile`: :func:`torch.compile` with tensorclasses.
