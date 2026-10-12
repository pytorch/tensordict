# TensorDictBase.min

TensorDictBase.min(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) | NO_DEFAULT = NO_DEFAULT*, *keepdim: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *return_indices: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the minimum values of all elements in the input tensordict.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - if `None`, returns a dimensionless
tensordict containing the min value of all leaves (if this can be computed).
If integer, min is called upon the dimension specified if
and only if this dimension is compatible with the tensordict
shape.
- **keepdim** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - whether the output tensor has dim retained or not.

Keyword Arguments:

- **reduce** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the reduction will occur across all TensorDict values
and a single reduced tensor will be returned.
Defaults to `False`.
- **return_indices** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - [`min()`](https://docs.pytorch.org/docs/stable/generated/torch.min.html#torch.min) returns a named tuple with values and indices
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
>>> td.min(dim=0)
torch.return_types.min(
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
>>> td.min()
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
>>> td.min(reduce=True)
tensor(-3.1537)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).