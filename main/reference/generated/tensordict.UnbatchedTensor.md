# UnbatchedTensor

*class*tensordict.UnbatchedTensor(*data*, *batch_size=None*)

Bases: `_UnbatchedTensorMixin`, [`Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

A tensor entry that has no batch dimensions and that batch operations leave unchanged.

Warning

`UnbatchedTensor` is experimental: its API can change in a minor
release without a deprecation cycle.

Use it for an entry that belongs to the whole batch and does not have
the batch dimensions: a lookup table, the parameters of a model, a
mask shared by every element. A plain tensor entry must have a shape
that starts with the batch size. An `UnbatchedTensor` can have
any shape. See [UnbatchedTensor](../../concepts/non_tensor.html#unbatched-tensor).

In a tensordict, the shape operations of the tensordict (indexing,
`reshape`, `unsqueeze`, `squeeze`, `transpose`...) return the
unbatched entry with its data and shape unchanged. Other operations
(arithmetic, `clone`, `to`...) apply to it as to any tensor, and
return an `UnbatchedTensor`. It is a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
subclass, so `td["a"]` can be used as a tensor.

An `UnbatchedTensor` has these constraints:

- Indexed assignment (`td[:, :2] = other`) leaves an unbatched entry
unchanged if `other` does not have that key. If `other` has it
with a payload of the same shape, the payload is written whole,
whatever the index, because the whole batch shares it.
- [`auto_batch_size_()`](tensordict.TensorDictBase.auto_batch_size_.html#tensordict.TensorDictBase.auto_batch_size_) ignores
unbatched entries: the batch size comes from the other entries.

Parameters:

- **data** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**array-like*) - the payload. Data that is not a
tensor is converted with [`torch.as_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.as_tensor.html#torch.as_tensor).
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size
of the tensordict that holds the tensor, read by
[`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap). A tensordict sets it when the tensor is
stored, so you rarely pass it. It does not change the shape
of the payload. Defaults to `None`, which gives `[]`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, UnbatchedTensor
>>> td = TensorDict(
... a=UnbatchedTensor(torch.zeros(3, 4)), b=torch.zeros(2, 3), batch_size=[2]
... )
>>> td[0]["a"].shape
torch.Size([3, 4])
>>> td_reshaped = td.reshape(1, 2)
>>> td_reshaped["a"].shape
torch.Size([3, 4])
>>> td_reshaped["a"].data_ptr() == td["a"].data_ptr()
True
>>> isinstance(td["a"], torch.Tensor)
True
```

data_ptr()

Returns the address of the first element of the payload, as [`torch.Tensor.data_ptr()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html#torch.Tensor.data_ptr).

untyped_storage()

Returns the storage of the payload, as [`torch.Tensor.untyped_storage()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.untyped_storage.html#torch.Tensor.untyped_storage).