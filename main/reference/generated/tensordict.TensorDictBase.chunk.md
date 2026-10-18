# TensorDictBase.chunk

TensorDictBase.chunk(*chunks: [int](https://docs.python.org/3/builtins/functions.html#int)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection), ...]

Splits the tensordict into `chunks` pieces along a batch dimension, as [`torch.chunk()`](https://docs.pytorch.org/docs/stable/generated/torch.chunk.html#torch.chunk) does.

Each piece has `ceil(batch_size[dim] / chunks)` elements, except the
last one, which can be smaller. So there can be fewer than `chunks`
pieces. The pieces hold views of the original tensors.

Parameters:

- **chunks** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the number of pieces.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch dimension to split. Defaults to
`0`.

Returns:

A tuple of tensordicts.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(x=torch.arange(5), batch_size=[5])
>>> [t["x"] for t in td.chunk(2)]
[tensor([0, 1, 2]), tensor([3, 4])]
>>> [t["x"] for t in td.chunk(4)]
[tensor([0, 1]), tensor([2, 3]), tensor([4])]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).