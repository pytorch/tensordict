# TensorDictBase.repeat

TensorDictBase.repeat(**repeats: [int](https://docs.python.org/3/builtins/functions.html#int)*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Repeats the batch along each batch dimension, as [`torch.Tensor.repeat()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.repeat.html#torch.Tensor.repeat) does.

The whole batch is repeated, as with [`numpy.tile()`](https://numpy.org/doc/stable/reference/generated/numpy.tile.html#numpy.tile). To repeat
each element next to itself instead, as with [`numpy.repeat()`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html#numpy.repeat), use
[`repeat_interleave()`](tensordict.TensorDictBase.repeat_interleave.html#tensordict.TensorDictBase.repeat_interleave). Unlike [`expand()`](tensordict.TensorDictBase.expand.html#tensordict.TensorDictBase.expand), it copies the data.
The feature dimensions are not repeated, and the names are kept.

Parameters:

***repeats** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the number of repetitions along each batch
dimension, one per batch dimension, as integers or as one
`torch.Size`, list, tuple or tensor.

Keyword Arguments:

**inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict keeps its
identity and keys, its leaves are replaced one by one, and the
call returns `self`. A lazy stack, or a tensordict that
holds one, raises a `NotImplementedError`: call
[`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) first. See [In-place operations](../../concepts/operations.html#inplace-conventions).
Defaults to `False`.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if the number of repeats differs from the number of
 batch dimensions.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.tensor([[0, 1]]), b=torch.zeros(1, 2, 3), batch_size=[1, 2])
>>> out = td.repeat(2, 2)
>>> out["a"]
tensor([[0, 1, 0, 1],
 [0, 1, 0, 1]])
>>> out["b"].shape
torch.Size([2, 4, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).