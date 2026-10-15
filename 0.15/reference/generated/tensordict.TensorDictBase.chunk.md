# TensorDictBase.chunk

TensorDictBase.chunk(*chunks: [int](https://docs.python.org/3/builtins/functions.html#int)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[TensorCollection, ...]

Splits a tensordict into the specified number of chunks, if possible.

Each chunk is a view of the input tensordict.

Parameters:

- **chunks** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - number of chunks to return
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - dimension along which to split the
tensordict. Default is 0.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... 'x': torch.arange(24).reshape(3, 4, 2),
... }, batch_size=[3, 4])
>>> td0, td1 = td.chunk(dim=-1, chunks=2)
>>> td0['x']
tensor([[[ 0, 1],
 [ 2, 3]],
 [[ 8, 9],
 [10, 11]],
 [[16, 17],
 [18, 19]]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).