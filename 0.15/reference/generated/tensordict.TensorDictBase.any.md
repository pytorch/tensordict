# TensorDictBase.any

TensorDictBase.any(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [bool](https://docs.python.org/3/builtins/functions.html#bool) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)

Returns whether any element of any leaf is `True` (nonzero).

Parameters:

**dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - a batch dimension to reduce. If given, returns a
new tensordict of boolean leaves without that dimension. A negative
`dim` counts from the end of the batch dimensions. Defaults to
`None`, which checks every element and returns a `bool`.

Returns:

A `bool`, or a new tensordict if `dim` is given.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if `dim` is not a batch dimension.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.tensor([[False, False], [True, False]]), batch_size=[2, 2])
>>> td.any()
True
>>> td.any(dim=1)["a"]
tensor([False, True])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).