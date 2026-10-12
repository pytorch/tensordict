# TensorDictBase.split

TensorDictBase.split(*split_size: [int](https://docs.python.org/3/builtins/functions.html#int) | [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[int](https://docs.python.org/3/builtins/functions.html#int)]*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase), ...]

Splits each tensor in the TensorDict with the specified size in the given dimension, like torch.split.

Returns a list of `TensorDict` instances with the view of split chunks of items.

Parameters:

- **split_size** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or**List**(*[*int*](https://docs.python.org/3/builtins/functions.html#int)*)*) - size of a single chunk or list of sizes for each chunk.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - dimension along which to split the tensor.

Returns:

A list of TensorDict with specified size in given dimension.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... 'x': torch.arange(12).reshape(3, 4),
... }, batch_size=[3, 4])
>>> td0, td1 = td.split([1, 2], dim=0)
>>> print(td0['x'])
tensor([[0, 1, 2, 3]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `split`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict