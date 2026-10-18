# TensorDictBase.cummax

TensorDictBase.cummax(*dim: [int](https://docs.python.org/3/builtins/functions.html#int)*, ***, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *return_indices: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the cumulative maximum of every leaf along a batch dimension.

As with [`torch.cummax()`](https://docs.pytorch.org/docs/stable/generated/torch.cummax.html#torch.cummax), the result is a named tuple of two
tensordicts, `values` and `indices`, with the structure and the batch
size of the tensordict.

Parameters:

**dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the batch dimension along which to compute it. A negative
`dim` counts from the end of the batch dimensions. A tuple uses
only its first dimension; passing one is deprecated and will be
removed in TensorDict 0.17.

Keyword Arguments:

- **reduce** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, concatenates the leaves along
`dim`, which requires them to have the same number of dimensions,
and returns what [`torch.cummax()`](https://docs.pytorch.org/docs/stable/generated/torch.cummax.html#torch.cummax) returns for the result. With
feature dimensions, a negative `dim` counts from the end of each
leaf. Starting with TensorDict 0.17, the default will change to
counting it from the end of the batch dimensions, as with
`reduce=False`. Defaults to `False`.
- **return_indices** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, returns only the values
tensordict. Defaults to `True`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.tensor([1.0, 3.0, 2.0]), batch_size=[3])
>>> td.cummax(dim=0).values["a"]
tensor([1., 3., 3.])
>>> td.cummax(dim=0).indices["a"]
tensor([0, 1, 1])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).