.. currentmodule:: tensordict

.. _integrations:

Use tensordict with other libraries
===================================

Many libraries accept a tensordict without any glue code. A
:class:`~tensordict.TensorDict` is a pytree (``torch.utils._pytree``) and a
:class:`~collections.abc.Mapping`, and it has a
:meth:`~tensordict.TensorDictBase.to` method. Each section of this page shows
one library, with a short example.

The examples need the other libraries, which tensordict does not install, so
they are not run when this page is tested. They were run with torch 2.14,
torchvision 0.29, Lightning 2.6, Transformers 5.20, Datasets 5.1, Pandera
0.34, JAX 0.12 and xarray 2026.9.

torchvision transforms
----------------------

The transforms of ``torchvision.transforms.v2`` accept any pytree, so they
accept a tensordict or a tensorclass. They return the same type, and the
leaves keep their ``tv_tensors`` types. In this example, the boxes follow the
image through a flip and a resize:

  >>> import torch
  >>> from torchvision import tv_tensors  # doctest: +SKIP
  >>> from torchvision.transforms import v2  # doctest: +SKIP
  >>> from tensordict import TensorDict
  >>> sample = TensorDict(  # doctest: +SKIP
  ...     image=tv_tensors.Image(torch.zeros(3, 32, 48, dtype=torch.uint8)),
  ...     boxes=tv_tensors.BoundingBoxes(
  ...         [[2, 4, 20, 30], [10, 10, 40, 28]], format="XYXY", canvas_size=(32, 48)
  ...     ),
  ...     label=torch.tensor([3, 7]),
  ... )
  >>> transform = v2.Compose([v2.RandomHorizontalFlip(p=1.0), v2.Resize((16, 24))])  # doctest: +SKIP
  >>> out = transform(sample)  # doctest: +SKIP
  >>> type(out).__name__, type(out["image"]).__name__, out["image"].shape  # doctest: +SKIP
  ('TensorDict', 'Image', torch.Size([3, 16, 24]))
  >>> out["boxes"].canvas_size, out["boxes"].tolist()  # doctest: +SKIP
  ((16, 24), [[14, 2, 23, 15], [4, 5, 19, 14]])
  >>> out["label"]  # doctest: +SKIP
  tensor([3, 7])

Wrap the image in ``tv_tensors.Image``. Without it, a transform takes the
first plain tensor of the tensordict as the image, in the order of the keys.
Here it flips the labels:

  >>> plain = TensorDict(label=torch.tensor([3, 7]), image=torch.zeros(3, 32, 48))
  >>> v2.RandomHorizontalFlip(p=1.0)(plain)["label"]  # doctest: +SKIP
  tensor([7, 3])

Keep the image dimensions out of the batch size: give one sample
``batch_size=[]``, and a batch of images ``batch_size=[N]``. A transform
applied to a batch draws one set of random parameters for all its images:

  >>> images = TensorDict(  # doctest: +SKIP
  ...     image=tv_tensors.Image(torch.zeros(4, 3, 32, 48, dtype=torch.uint8)),
  ...     label=torch.arange(4),
  ...     batch_size=[4],
  ... )
  >>> v2.RandomCrop((16, 16))(images)["image"].shape  # doctest: +SKIP
  torch.Size([4, 3, 16, 16])

To give each sample its own random parameters, transform the samples in the
dataset, before the batch is built. Samples for object detection have
different numbers of boxes, so ``torch.stack`` cannot stack them. Pass
:func:`~tensordict.lazy_stack` as the ``collate_fn`` of the
:class:`~torch.utils.data.DataLoader` instead (see :ref:`data-loading`):

  >>> from tensordict import lazy_stack
  >>> one_box = TensorDict(  # doctest: +SKIP
  ...     image=tv_tensors.Image(torch.zeros(3, 32, 48, dtype=torch.uint8)),
  ...     boxes=tv_tensors.BoundingBoxes([[0, 0, 8, 8]], format="XYXY", canvas_size=(32, 48)),
  ...     label=torch.tensor([1]),
  ... )
  >>> batch = lazy_stack([transform(sample), transform(one_box)])  # doctest: +SKIP
  >>> [boxes.shape for boxes in batch.get("boxes", as_list=True)]  # doctest: +SKIP
  [torch.Size([2, 4]), torch.Size([1, 4])]

Lightning
---------

A tensordict or a tensorclass can be the batch of a Lightning ``Trainer``.
Lightning moves the batch to the device with one call to ``batch.to(device)``.
It passes ``non_blocking=True`` only to plain tensors, but you do not need to
override ``transfer_batch_to_device`` for that: by default, ``to()`` copies
each leaf with ``non_blocking=True`` (see :ref:`devices`). For copies from
pinned memory, pass ``pin_memory=True`` to the ``DataLoader`` (see
:ref:`data-loading`).

  >>> import lightning as L  # doctest: +SKIP
  >>> from torch.utils.data import DataLoader
  >>> class Regression(L.LightningModule):  # doctest: +SKIP
  ...     def __init__(self):
  ...         super().__init__()
  ...         self.net = torch.nn.Linear(2, 1)
  ...
  ...     def training_step(self, batch, batch_idx):
  ...         pred = self.net(batch["x"]).squeeze(-1)
  ...         loss = torch.nn.functional.mse_loss(pred, batch["y"])
  ...         self.log("loss", loss, on_epoch=True)
  ...         return loss
  ...
  ...     def configure_optimizers(self):
  ...         return torch.optim.SGD(self.parameters(), lr=0.1)
  >>> data = TensorDict(x=torch.randn(64, 2), y=torch.randn(64), batch_size=[64])
  >>> loader = DataLoader(data, batch_size=16, collate_fn=lambda x: x)
  >>> trainer = L.Trainer(  # doctest: +SKIP
  ...     max_epochs=1,
  ...     logger=False,
  ...     enable_checkpointing=False,
  ...     enable_progress_bar=False,
  ...     enable_model_summary=False,
  ... )
  >>> trainer.fit(Regression(), loader)  # doctest: +SKIP
  >>> sorted(trainer.callback_metrics)  # doctest: +SKIP
  ['loss', 'loss_epoch', 'loss_step']

With a tensorclass batch, pass ``batch_size`` to ``self.log`` when it
averages a value over the epoch (``on_epoch=True``, the default in
``validation_step``). Lightning finds the batch size by iterating over the
batch. It reads a tensordict as a mapping, but it iterates over the rows of a
tensorclass, and that raises a ``TypeError``. Write, for example,
``self.log("loss", loss, on_epoch=True, batch_size=batch.batch_size[0])``.

Hugging Face Transformers and Datasets
--------------------------------------

The output of a tokenizer is a mapping of tensors with one row for each text,
so a tensordict holds it with ``batch_size=[len(texts)]``. You can then index,
split, stack, and move the tokens as one object. ``model(**tokens)`` passes
each entry as a keyword argument, as for a ``dict``. The model in this
example is small and untrained:

  >>> from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel  # doctest: +SKIP
  >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")  # doctest: +SKIP
  >>> tokenizer.pad_token = tokenizer.eos_token  # doctest: +SKIP
  >>> texts = ["Hello world", "A longer sentence than the first"]
  >>> encoded = tokenizer(texts, padding=True, return_tensors="pt")  # doctest: +SKIP
  >>> tokens = TensorDict(encoded, batch_size=[len(texts)])  # doctest: +SKIP
  >>> tokens  # doctest: +SKIP
  TensorDict(batch_size=[2], fields={attention_mask: i64[6], input_ids: i64[6]})
  >>> model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16))  # doctest: +SKIP
  >>> model(**tokens).logits.shape  # doctest: +SKIP
  torch.Size([2, 6, 50257])
  >>> model(**tokens[:1]).logits.shape  # doctest: +SKIP
  torch.Size([1, 6, 50257])

A dataset of the ``datasets`` library with ``with_format("torch")`` returns
a ``dict`` of tensors for a slice of rows. A column of strings becomes a
:class:`~tensordict.NonTensorStack` (see :ref:`non-tensor-data`):

  >>> from datasets import Dataset  # doctest: +SKIP
  >>> ds = Dataset.from_dict(  # doctest: +SKIP
  ...     {"x": [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], "label": [0, 1, 0], "text": ["a", "b", "c"]}
  ... ).with_format("torch")
  >>> batch = TensorDict(ds[:2], batch_size=[2])  # doctest: +SKIP
  >>> batch["x"]  # doctest: +SKIP
  tensor([[0., 1.],
          [2., 3.]])
  >>> batch[1]["text"]  # doctest: +SKIP
  'b'

``TensorDict(ds[:], batch_size=[len(ds)])`` loads the whole dataset in
memory, as one tensordict that a ``DataLoader`` reads (see
:ref:`data-loading`). To read batches from the dataset instead, give the
``DataLoader`` a ``collate_fn`` that stacks the rows:

  >>> loader = DataLoader(  # doctest: +SKIP
  ...     ds, batch_size=2, collate_fn=lambda rows: torch.stack([TensorDict(row) for row in rows])
  ... )
  >>> [batch.batch_size for batch in loader]  # doctest: +SKIP
  [torch.Size([2]), torch.Size([1])]

A column of lists with different lengths comes out as a list of tensors, and
the tensordict stores that list as a ``NonTensorStack`` too. Pad the lists in
the dataset, or put them in a lazy stack (see :ref:`stacking-variable-size`):

  >>> ragged = Dataset.from_dict({"tokens": [[1, 2, 3], [4, 5]]}).with_format("torch")  # doctest: +SKIP
  >>> seqs = lazy_stack([TensorDict(tokens=t) for t in ragged[:]["tokens"]])  # doctest: +SKIP
  >>> seqs.get("tokens", as_padded_tensor=True)  # doctest: +SKIP
  tensor([[1, 2, 3],
          [4, 5, 0]])

Validate with Pandera
---------------------

Pandera (0.34 or later) validates tensordicts and tensorclasses with the
``pandera.tensordict`` module. A schema lists the keys, with the dtype, the
shape and the checks of each, and the batch size. ``None`` accepts any size
for a dimension. ``validate`` returns a copy of the data, or raises a
``SchemaError``. With ``inplace=True``, it returns the data without a copy:

  >>> import pandera.tensordict as pa  # doctest: +SKIP
  >>> from pandera import Check  # doctest: +SKIP
  >>> schema = pa.TensorDictSchema(  # doctest: +SKIP
  ...     keys={
  ...         "image": pa.Tensor(dtype=torch.float32, shape=(None, 3, 8, 8)),
  ...         "label": pa.Tensor(dtype=torch.int64, shape=(None,), checks=Check.ge(0)),
  ...     },
  ...     batch_size=(None,),
  ... )
  >>> batch = TensorDict(image=torch.randn(4, 3, 8, 8), label=torch.arange(4), batch_size=[4])
  >>> schema.validate(batch, inplace=True) is batch  # doctest: +SKIP
  True
  >>> schema.validate(batch.float())  # doctest: +SKIP
  Traceback (most recent call last):
  ...
  pandera.errors.SchemaError: Key 'label': expected dtype torch.int64, got torch.float32

The same schema can be written as a class, as for a dataclass, and it
validates a tensorclass too:

  >>> from tensordict import TensorClass
  >>> class ImageSchema(pa.TensorDictModel):  # doctest: +SKIP
  ...     image: torch.float32 = pa.Field(shape=(None, 3, 8, 8))
  ...     label: torch.int64 = pa.Field(shape=(None,), ge=0)
  ...
  ...     class Config:
  ...         batch_size = (None,)
  >>> class Images(TensorClass):
  ...     image: torch.Tensor
  ...     label: torch.Tensor
  >>> images = Images(image=torch.randn(4, 3, 8, 8), label=torch.tensor([0, 1, -2, 3]), batch_size=[4])
  >>> ImageSchema.validate(images)  # doctest: +SKIP
  Traceback (most recent call last):
  ...
  pandera.errors.SchemaError: Check '<Check greater_than_or_equal_to: greater_than_or_equal_to(0)>' failed for key 'label': check failed

With ``validate(..., lazy=True)``, Pandera reports every error at once, in a
``SchemaErrors`` exception. See the `Pandera documentation
<https://pandera.readthedocs.io/>`_ for the other checks.

Arrays from JAX
---------------

A JAX simulator returns a pytree of JAX arrays. Convert each array with
:func:`torch.from_dlpack`, then build the tensordict. On the same device,
DLPack shares the memory of the array, without a copy. Pass the tensors back
to JAX the same way, with ``jax.numpy.from_dlpack``:

  >>> import jax  # doctest: +SKIP
  >>> import jax.numpy as jnp  # doctest: +SKIP
  >>> state = {"qpos": jnp.zeros((4, 7)), "done": jnp.zeros(4, dtype=bool)}  # doctest: +SKIP
  >>> td = TensorDict(jax.tree.map(torch.from_dlpack, state), batch_size=[4])  # doctest: +SKIP
  >>> td["qpos"].shape, td["done"].dtype  # doctest: +SKIP
  (torch.Size([4, 7]), torch.bool)
  >>> back = jax.tree.map(jnp.from_dlpack, td.to_dict())  # doctest: +SKIP
  >>> back["qpos"].shape, back["done"].dtype  # doctest: +SKIP
  ((4, 7), dtype('bool'))

Convert the arrays before you build the tensordict. A tensordict keeps a JAX
array as non-tensor data, which tensor operations skip:

  >>> type(TensorDict(state, batch_size=[4])["qpos"]).__name__  # doctest: +SKIP
  'ArrayImpl'

JAX arrays are immutable, but a tensor from DLPack is not, and a write into
it changes the JAX array too. Clone the tensor before an in-place write.

xarray
------

Convert an xarray ``Dataset`` one variable at a time. Choose the batch
dimensions, which the variables share, and transpose each variable so that
they come first. Pass their names as ``names`` (see :ref:`batch-size`):

  >>> import numpy as np
  >>> import xarray as xr  # doctest: +SKIP
  >>> ds = xr.Dataset(  # doctest: +SKIP
  ...     {
  ...         "t2m": (("time", "lat", "lon"), np.zeros((8, 4, 6), dtype=np.float32)),
  ...         "u10": (("time", "lat", "lon"), np.zeros((8, 4, 6), dtype=np.float32)),
  ...         "z": (("level", "time", "lat", "lon"), np.zeros((3, 8, 4, 6), dtype=np.float32)),
  ...         "lsm": (("lat", "lon"), np.zeros((4, 6), dtype=np.float32)),
  ...     }
  ... )
  >>> batch_dims = ["time"]
  >>> weather = TensorDict(  # doctest: +SKIP
  ...     {k: torch.as_tensor(ds[k].transpose(*batch_dims, ...).values) for k in ["t2m", "u10", "z"]},
  ...     batch_size=[ds.sizes[d] for d in batch_dims],
  ...     names=batch_dims,
  ... )
  >>> weather["z"].shape, weather.names  # doctest: +SKIP
  (torch.Size([8, 3, 4, 6]), ['time'])

Models often take one tensor for each group of variables, with the variables
as channels. ``to_array("channel")`` stacks them along a new first
dimension, so transpose the batch dimensions in front of it:

  >>> def stack(ds, variables, batch_dims):
  ...     stacked = ds[variables].to_array("channel")
  ...     return torch.as_tensor(stacked.transpose(*batch_dims, "channel", ...).values)
  >>> groups = TensorDict(  # doctest: +SKIP
  ...     surface=stack(ds, ["t2m", "u10"], batch_dims),
  ...     upper=stack(ds, ["z"], batch_dims),
  ...     batch_size=[ds.sizes[d] for d in batch_dims],
  ...     names=batch_dims,
  ... )
  >>> groups["surface"].shape, groups["upper"].shape  # doctest: +SKIP
  (torch.Size([8, 2, 4, 6]), torch.Size([8, 1, 3, 4, 6]))

These rules apply:

- A variable without the batch dimensions, such as ``lsm`` here, cannot go in
  the tensordict, not even in a nested tensordict with ``batch_size=[]``: a
  nested tensordict has the batch dimensions of its parent. Keep it in a
  separate tensordict, or expand it to the batch dimensions with
  ``torch.as_tensor(ds["lsm"].values).expand(8, 4, 6)``. The expanded tensor
  is a view, without a copy, but ``clone()`` and
  :meth:`~tensordict.TensorDictBase.memmap` copy it in full.
- If the ``Dataset`` is backed by dask, ``.values`` loads the data in memory.
- The tensordict keeps the names of the batch dimensions only. The names of
  the other dimensions, such as ``level``, and the coordinates are lost.
