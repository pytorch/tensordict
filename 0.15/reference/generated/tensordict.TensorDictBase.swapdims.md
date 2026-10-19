# TensorDictBase.swapdims

TensorDictBase.swapdims(*axis0: [int](https://docs.python.org/3/builtins/functions.html#int)*, *axis1: [int](https://docs.python.org/3/builtins/functions.html#int)*)

Returns a tensordict with two batch dimensions swapped, as [`transpose()`](tensordict.TensorDictBase.transpose.html#tensordict.TensorDictBase.transpose) does.

`swapaxes` and `swapdims` are aliases of [`transpose()`](tensordict.TensorDictBase.transpose.html#tensordict.TensorDictBase.transpose). It
can be used as a context manager, as [`transpose()`](tensordict.TensorDictBase.transpose.html#tensordict.TensorDictBase.transpose) can.

Parameters:

- **axis0** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the first batch dimension to swap.
- **axis1** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the second batch dimension to swap.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.zeros(3, 4, 5)}, batch_size=[3, 4])
>>> print(td.swapaxes(0, 1).shape)
torch.Size([4, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).