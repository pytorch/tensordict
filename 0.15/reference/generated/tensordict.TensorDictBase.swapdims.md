# TensorDictBase.swapdims

TensorDictBase.swapdims(*axis0: [int](https://docs.python.org/3/builtins/functions.html#int)*, *axis1: [int](https://docs.python.org/3/builtins/functions.html#int)*)

Interchange two axes of the tensordict.

This is an alias for [`transpose()`](tensordict.TensorDictBase.transpose.html#tensordict.TensorDictBase.transpose).

Parameters:

- **axis0** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - First axis.
- **axis1** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - Second axis.

Returns:

a new tensordict with the axes swapped.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.randn(3, 4, 5)}, batch_size=[3, 4])
>>> print(td.swapaxes(0, 1).shape)
torch.Size([4, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).