# TensorDictBase.split

TensorDictBase.split(*split_size: [int](https://docs.python.org/3/builtins/functions.html#int) | [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[int](https://docs.python.org/3/builtins/functions.html#int)]*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase), ...]

Splits the tensordict along a batch dimension, as [`torch.split()`](https://docs.pytorch.org/docs/stable/generated/torch.split.html#torch.split) does.

The pieces hold views of the original tensors, and keep the names.

On a [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), sizes along the stack
dim that are negative or do not add up to the stack size emit a
[`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning), and the pieces miss members or have a wrong
batch size. Starting with TensorDict 0.17, the default will change to
raising a `RuntimeError`, as `TensorDict.split` does.

Parameters:

- **split_size** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - the size of each piece (the last
one can be smaller), or the list of the sizes of the pieces,
which must add up to `batch_size[dim]`.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch dimension to split. Defaults to
`0`.

Returns:

A tuple of tensordicts.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the sizes of a list do not add up to
 `batch_size[dim]`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(x=torch.arange(5), batch_size=[5])
>>> [t["x"] for t in td.split(2)]
[tensor([0, 1]), tensor([2, 3]), tensor([4])]
>>> [t["x"] for t in td.split([1, 4])]
[tensor([0]), tensor([1, 2, 3, 4])]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `split`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict