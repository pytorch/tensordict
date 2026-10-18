# unpad_sequence

tensordict.unpad_sequence(*padded: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *mask: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *pad_dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*) → [LazyStackedTensorDict](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict)

Splits a padded batch back into its variable-length samples.

The inverse of [`pad_sequence()`](tensordict.pad_sequence.html#tensordict.pad_sequence) with
`return_mask=True`: each row of `padded` keeps, in every entry, the
positions that the mask marks along `pad_dim`. These must form one
contiguous run per row, as the masks of [`pad_sequence()`](tensordict.pad_sequence.html#tensordict.pad_sequence)
do. Left and right padding are both read from the mask. Non-tensor
entries are kept as they are, except when `pad_dim` is a batch
dimension: then they lose their pad slots too.

The lengths are read on the host (`.tolist()`), which synchronizes
with the device: this is meant for post-processing, not for compiled
hot paths.

Parameters:

- **padded** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the padded batch, with the samples along its
first dimension.
- **mask** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - the valid positions.
A tensordict holds one mask per entry, laid out as the masks of
[`pad_sequence()`](tensordict.pad_sequence.html#tensordict.pad_sequence). A boolean tensor of shape
`[len(padded), max_len]` is used for every entry. If `None`,
the masks are read from the `"masks"` entry of `padded`, and
the result has no `"masks"` entry. When `mask` is given,
every entry of `padded` is unpadded. Defaults to `None`.

Keyword Arguments:

**pad_dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the `pad_dim` given to
[`pad_sequence()`](tensordict.pad_sequence.html#tensordict.pad_sequence), counted from the first
dimension of each sample's entries. Defaults to `0`.

Returns:

A [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) that stacks the samples
along its first dimension. Each sample has the batch size of the
original sample, and its entries are views of `padded`.

Raises:

- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `padded` has no batch dimension, if `mask` is
 `None` and `padded` has no `"masks"` entry, if a tensor
 mask does not have shape `[len(padded), max_len]`, or if the
 marked positions of a row are not contiguous.
- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if a mask tensordict has no mask for an entry of
 `padded`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, pad_sequence, unpad_sequence
>>> list_td = [
... TensorDict({"a": torch.arange(3), "b": torch.zeros(3, 2)}, batch_size=[3]),
... TensorDict({"a": torch.arange(5), "b": torch.ones(5, 2)}, batch_size=[5]),
... ]
>>> padded = pad_sequence(list_td, return_mask=True, padding_side="left")
>>> padded["a"]
tensor([[0, 0, 0, 1, 2],
 [0, 1, 2, 3, 4]])
>>> samples = unpad_sequence(padded)
>>> samples[0]["a"]
tensor([0, 1, 2])
>>> [sample.batch_size for sample in samples.tensordicts]
[torch.Size([3]), torch.Size([5])]
```