# TensorDictBase.data_ptr

TensorDictBase.data_ptr(***, *storage: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns the data_ptr of the tensordict leaves.

This can be useful to check if two tensordicts share the same `data_ptr()`.

Keyword Arguments:

**storage** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, tensor.untyped_storage().data_ptr() will be called
instead. Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.randn(2), b=torch.randn(2), batch_size=[2])
>>> td0 = td.unsqueeze(0) # the leaves of td0 are views on the leaves of td
>>> assert (td0.data_ptr() == td.data_ptr()).all()
```

Note

[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) instances will be displayed as nested tensordicts to
reflect the true `data_ptr()` of their leaves:

```
>>> td0 = TensorDict(a=torch.randn(2), b=torch.randn(2), batch_size=[2])
>>> td1 = TensorDict(a=torch.randn(2), b=torch.randn(2), batch_size=[2])
>>> td = TensorDict.lazy_stack([td0, td1])
>>> td.data_ptr()
TensorDict(
 fields={
 0: TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False),
 1: TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).