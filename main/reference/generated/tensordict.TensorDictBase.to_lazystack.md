# TensorDictBase.to_lazystack

TensorDictBase.to_lazystack(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*)

Returns a lazy stack of the slices of the tensordict along `dim`.

The tensordict is unbound along `dim` and the slices are stacked in
a [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) (see [Stacking and variable-size data](../../concepts/stacking.html#stacking)).
The slices are views: writing into a tensordict of the stack in place
writes into the original data. The batch size does not change. A lazy
stack along `dim` is returned as it is, and a lazy stack along
another dimension is restacked, which changes its stack dimension.

Parameters:

**dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the dimension to stack along. Defaults to
`0`.

Returns:

A lazy stack with `stack_dim == dim`, or `self` when the
tensordict is already one.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(2, 3), batch_size=(2, 3))
>>> stack = td.to_lazystack()
>>> type(stack).__name__, stack.batch_size, stack.stack_dim
('LazyStackedTensorDict', torch.Size([2, 3]), 0)
>>> stack.to_lazystack(1).stack_dim
1
>>> stack.to_lazystack(0) is stack
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).