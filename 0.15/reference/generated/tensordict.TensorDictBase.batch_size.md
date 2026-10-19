# TensorDictBase.batch_size

*property*TensorDictBase.batch_size*: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*

The batch size of the tensordict, as a [`torch.Size`](https://docs.pytorch.org/docs/stable/size.html#torch.Size). `shape` is the same attribute.

The batch size is the shape of the leading dimensions that all tensors
in the tensordict share. The other dimensions of each tensor are its
feature dimensions. For example, a batch of videos of shape
`[B, T, C, W, H]` can have batch size `[B, T]` (batch and time)
and feature dimensions `[C, W, H]` (channels and space).

You set the batch size when you create the tensordict. The batch size
is not inferred from the tensors unless you ask for it, with
`TensorDict.from_dict(..., auto_batch_size=True)` or
[`auto_batch_size_()`](tensordict.TensorDictBase.auto_batch_size_.html#tensordict.TensorDictBase.auto_batch_size_).

You can assign a new batch size if every tensor has the new leading
dimensions. Otherwise, the assignment raises `RuntimeError`. An empty
batch size `()` always fits a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict). A
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) raises `RuntimeError` for
any new batch size. A nested tensordict with fewer batch dimensions than
the new batch size gets the new batch size. The other nested tensordicts
keep theirs. See [Batch size, shapes, and indexing](../../concepts/batch_size.html#batch-size).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict({
... "key 0": torch.zeros(3, 4),
... "key 1": torch.zeros(3, 5),
... "nested": TensorDict({"key 0": torch.zeros(3, 4)}, batch_size=[3, 4])},
... batch_size=[3])
>>> data.batch_size
torch.Size([3])
>>> data.batch_size = () # an empty batch size always fits
>>> data.batch_size, data["nested"].batch_size
(torch.Size([]), torch.Size([3, 4]))
>>> data.batch_size = [4]
Traceback (most recent call last):
...
RuntimeError: the Tensor key 0 has shape torch.Size([3, 4]) which is incompatible with the batch-size torch.Size([4]).
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `batch_size`

![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict
![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
![](../../_images/sphx_glr_tensordict_slicing_thumb.png)

[Slicing, indexing, and masking](../../tutorials/tensordict_slicing.html)

Slicing, indexing, and masking
![](../../_images/sphx_glr_zarr_storage_thumb.png)

[Storing TensorDicts in zarr](../../tutorials/zarr_storage.html)

Storing TensorDicts in zarr