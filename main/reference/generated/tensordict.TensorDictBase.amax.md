# TensorDictBase.amax

TensorDictBase.amax(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) | NO_DEFAULT = NO_DEFAULT*, *keepdim: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the maximum of every leaf, over all its elements or along a batch dimension.

Like [`max()`](tensordict.TensorDictBase.max.html#tensordict.TensorDictBase.max) with `return_indices=False`: the result holds the
values only.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch dimension to reduce. A negative `dim`
counts from the end of the batch dimensions. A tuple reduces only
its first dimension and is not checked against the batch dimensions.
Starting with TensorDict 0.17, the default will change to reducing
every dimension of the tuple, as [`torch.amax()`](https://docs.pytorch.org/docs/stable/generated/torch.amax.html#torch.amax) does. Defaults
to `None`, which reduces every element of each leaf, and the
result has an empty batch size.
- **keepdim** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the reduced batch dimension is
kept with size 1. Defaults to `False`.

Keyword Arguments:

**reduce** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, reduces all the leaves together
and returns a single tensor. Without `dim`, the result is a
scalar. With `dim`, the leaves are concatenated along `dim`
first, so they must have the same number of dimensions. With feature
dimensions, a negative `dim` counts from the end of each leaf.
Starting with TensorDict 0.17, the default will change to counting
it from the end of the batch dimensions, as with `reduce=False`.
Defaults to `False`.

Returns:

A new tensordict, or a tensor if `reduce=True`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.tensor([[1.0, 5.0], [3.0, 2.0]]), batch_size=[2])
>>> td.amax()["a"]
tensor(5.)
>>> td.amax(dim=0)["a"]
tensor([3., 5.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).