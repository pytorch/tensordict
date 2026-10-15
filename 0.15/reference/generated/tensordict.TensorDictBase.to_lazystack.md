# TensorDictBase.to_lazystack

TensorDictBase.to_lazystack(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*)

Converts a TensorDict to a LazyStackedTensorDict or equivalent.

Note

This method can be used to swap the stack dimension of a LazyStackedTensorDict.
For example, if you have a LazyStackedTensorDict with stack_dim=1, you can use this method to swap it to stack_dim=0:

```
>>> import torch
>>> from tensordict import LazyStackedTensorDict, TensorDict
>>> td = TensorDict({"a": torch.zeros(2, 3), "b": torch.ones(2, 3)}, batch_size=(2, 3))
>>> td2 = td.to_lazystack()
>>> td2.batch_size
torch.Size([2, 3])
>>> assert isinstance(td2, LazyStackedTensorDict)
>>> assert td2.stack_dim == 0
>>> td3 = td2.to_lazystack(1)
>>> assert td3.stack_dim == 1
>>> td3.batch_size
torch.Size([2, 3])
```

Parameters:

**dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the dimension along which to stack the tensordict.
Defaults to `0`.

Returns:

A LazyStackedTensorDict instance.

Examples

```
>>> import torch
>>> from tensordict import LazyStackedTensorDict, TensorDict
>>> td = TensorDict({"a": torch.zeros(2, 3), "b": torch.ones(2, 3)}, batch_size=(2, 3))
>>> td2 = td.to_lazystack()
>>> td2.batch_size
torch.Size([2, 3])
>>> assert isinstance(td2, LazyStackedTensorDict)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).