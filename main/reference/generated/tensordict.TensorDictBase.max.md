# TensorDictBase.max

TensorDictBase.max(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) | NO_DEFAULT = NO_DEFAULT*, *keepdim: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *return_indices: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the maximum values of all elements in the input tensordict.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - if `None`, returns a dimensionless
tensordict containing the max value of all leaves (if this can be computed).
If integer, max is called upon the dimension specified if
and only if this dimension is compatible with the tensordict
shape.
A tuple reduces only its first dim; passing one is deprecated
and will be removed in TensorDict 0.17.
- **keepdim** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - whether the output tensor has dim retained or not.

Keyword Arguments:

- **reduce** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the reduction will occur across all TensorDict values
and a single reduced tensor will be returned.
Defaults to `False`. With feature dims, a negative `dim` counts from the end of
each leaf. Starting with TensorDict 0.17, the default will change to counting it from
the end of the batch dims, as with `reduce=False`.
- **return_indices** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - [`max()`](https://docs.pytorch.org/docs/stable/generated/torch.max.html#torch.max) returns a named tuple with values and indices
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
>>> td.max(dim=0)
torch.return_types.max(
values=TensorDict(
 fields={
 a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
 d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([4]),
 device=None,
 is_shared=False),
indices=TensorDict(
 fields={
 a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.int64, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.int64, is_shared=False),
 d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([4]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([4]),
 device=None,
 is_shared=False))
>>> td.max()
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> td.max(reduce=True)
tensor(4.1015)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).