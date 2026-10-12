# UnbatchedTensor

*class*tensordict.UnbatchedTensor(*data*, *batch_size=None*)

Bases: `_UnbatchedTensorMixin`, [`Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

A torch.Tensor subclass whose shape is ignored during batch operations on a TensorDict.

Use it for an entry that belongs to the whole batch and does not have
the batch dimensions: a lookup table, the parameters of a model, a
mask shared by every element. A plain tensor entry must have a shape
that starts with the batch size. An `UnbatchedTensor` can have
any shape. See [UnbatchedTensor](../../concepts/non_tensor.html#unbatched-tensor).

When stored in a TensorDict, shape operations (indexing, reshape, unsqueeze,
squeeze, transpose, etc.) on the parent TensorDict leave the payload of
this tensor unchanged. Data operations (arithmetic, clone, to, etc.) work
through standard PyTorch tensor dispatch.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, UnbatchedTensor
>>> td = TensorDict(a=UnbatchedTensor(torch.randn(3, 4)), b=torch.randn(2, 3), batch_size=(2,))
>>> td_reshaped = td.reshape((1, 2))
>>> td_reshaped["a"].shape
torch.Size([3, 4])
>>> td_reshaped["a"].data_ptr() == td["a"].data_ptr()
True
>>> td[0]["a"].shape
torch.Size([3, 4])
```

Since UnbatchedTensor is a torch.Tensor subclass, retrieving it from a
TensorDict returns a usable tensor directly:

```
>>> val = td["a"]
>>> isinstance(val, torch.Tensor)
True
>>> isinstance(val, UnbatchedTensor)
True
```

An `UnbatchedTensor` has these constraints:

- Indexed assignment (`td[:, :2] = other`) leaves an unbatched entry
unchanged only if `other` does not have that key. If `other`
has it, the batch index is applied to the unbatched payload, which
usually raises a shape error
([#2251](https://github.com/pytorch/tensordict/issues/2251)).
- [`auto_batch_size_()`](tensordict.TensorDictBase.auto_batch_size_.html#tensordict.TensorDictBase.auto_batch_size_) ignores
unbatched entries: the batch size comes from the other entries.

Parameters:

- **data** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**tensor-like*) - payload tensor.
- **batch_size** (*torch.Size compatible**,**optional*) - TensorDict-facing
batch-size metadata used by `torch.vmap`. The payload shape is
left unchanged.

data_ptr() → [int](https://docs.python.org/3/builtins/functions.html#int)

Returns the address of the first element of `self` tensor.

Note

If the tensor is a copy-on-write tensor (e.g. created via
`_lazy_clone()`), calling this method will materialize the
copy. Use `const_data_ptr()` if you only need read-only access
to the data pointer.

untyped_storage() → [UntypedStorage](https://docs.pytorch.org/docs/stable/storage.html#torch.UntypedStorage)

Returns the underlying `UntypedStorage`.