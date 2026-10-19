# TensorDictBase.min

TensorDictBase.min(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) | NO_DEFAULT = NO_DEFAULT*, *keepdim: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *return_indices: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the minimum of every leaf, over all its elements or along a batch dimension.

Without `dim`, the result is a tensordict of scalar leaves. With `dim`,
as with [`torch.min()`](https://docs.pytorch.org/docs/stable/generated/torch.min.html#torch.min), the result is a named tuple of two tensordicts of
the same structure, `values` and `indices`.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch dimension to reduce. A negative `dim`
counts from the end of the batch dimensions. A tuple reduces only
its first dimension; passing one is deprecated and will be removed
in TensorDict 0.17. Defaults to `None`, which reduces every
element of each leaf, and the result has an empty batch size.
- **keepdim** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the reduced batch dimension is
kept with size 1. Needs an int `dim`. Defaults to `False`.

Keyword Arguments:

- **reduce** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, reduces all the leaves together
and returns a single tensor. Without `dim`, the result is a
scalar. With `dim`, the leaves are concatenated along `dim`
first, so they must have the same number of dimensions. With feature
dimensions, a negative `dim` counts from the end of each leaf.
Starting with TensorDict 0.17, the default will change to counting
it from the end of the batch dimensions, as with `reduce=False`.
Defaults to `False`.
- **return_indices** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False` and `dim` is given,
returns only the values tensordict. Defaults to `True`.

Returns:

A new tensordict, or a named tuple `(values, indices)` of
tensordicts when `dim` is given and `return_indices=True`. With
`reduce=True`, a tensor, or what [`torch.min()`](https://docs.pytorch.org/docs/stable/generated/torch.min.html#torch.min) returns when
`dim` is given.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> _ = torch.manual_seed(0)
>>> td = TensorDict(
... a=torch.randn(3, 4, 5),
... b=TensorDict(c=torch.randn(3, 4, 5, 6), batch_size=(3, 4, 5)),
... batch_size=(3, 4),
... )
>>> td.min(dim=0).values.batch_size
torch.Size([4])
>>> td.min(dim=0).indices["b", "c"].dtype
torch.int64
>>> td.min()["a"].shape
torch.Size([])
>>> td.min(reduce=True)
tensor(-2.8891)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).