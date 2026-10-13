# TensorDictBase.cummin

TensorDictBase.cummin(*dim: [int](https://docs.python.org/3/builtins/functions.html#int)*, ***, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *return_indices: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the cumulative minimum values of all elements in the input tensordict.

Parameters:

**dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - integer representing the dimension along which to perform the cummin operation.

Keyword Arguments:

- **reduce** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the reduction will occur across all TensorDict values
and a single reduced tensor will be returned.
Defaults to `False`. With feature dims, a negative `dim` counts from the end of
each leaf. Starting with TensorDict 0.17, the default will change to counting it from
the end of the batch dims, as with `reduce=False`.
- **return_indices** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - [`cummin()`](https://docs.pytorch.org/docs/stable/generated/torch.cummin.html#torch.cummin) returns a named tuple with values and indices
when the `dim` argument is passed. The `TensorDict` equivalent of this is to return a named tuple
with `values` and `indices` tensordicts of identical structure. If `False`, only the values
are returned. Defaults to `True`.

Examples

```
>>> from tensordict import TensorDict
>>> import torch
>>> _ = torch.manual_seed(0)
>>> td = TensorDict(
... a=torch.randn(3, 4, 5),
... b=TensorDict(
... c=torch.randn(3, 4, 5, 6),
... d=torch.randn(3, 4, 5),
... batch_size=(3, 4, 5),
... ),
... batch_size=(3, 4)
... )
>>> td.cummin(dim=0)
torch.return_types.cummin(
values=TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([3, 4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
 d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3, 4]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3, 4]),
 device=None,
 is_shared=False),
indices=TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.int64, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([3, 4, 5, 6]), device=cpu, dtype=torch.int64, is_shared=False),
 d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([3, 4]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3, 4]),
 device=None,
 is_shared=False))
>>> td = TensorDict(
... a=torch.randn(3, 4, 5),
... b=TensorDict(
... c=torch.randn(3, 4, 5),
... d=torch.randn(3, 4, 5),
... batch_size=(3, 4, 5),
... ),
... batch_size=(3, 4)
... )
>>> td.cummin(reduce=True, dim=0)
torch.return_types.cummin(...)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).