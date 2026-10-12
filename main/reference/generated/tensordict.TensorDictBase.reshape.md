# TensorDictBase.reshape

TensorDictBase.reshape(**args*, ***kwargs*) → Self

Returns a tensordict whose batch dimensions are reshaped to `shape`.

Each leaf is reshaped with [`torch.Tensor.reshape()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.reshape.html#torch.Tensor.reshape), and keeps its
feature dimensions. As with [`torch.Tensor.reshape()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.reshape.html#torch.Tensor.reshape), a leaf of
the result is a view of the original leaf when the memory layout
allows it, and a copy otherwise.

Parameters:

***shape** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - new batch size of the resulting tensordict, as
integers or as one list, tuple or `torch.Size`. One
dimension can be `-1`. The shape is positional only:
`reshape(shape=...)` raises `TypeError`.

Keyword Arguments:

**inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, this tensordict's identity
and key set are preserved; each leaf is reshaped one at a
time. Note that the underlying `Tensor.reshape` may return
a view of the original leaf (when memory layout permits) or
a copy (otherwise); in the view case `inplace=True` keeps
the leaves sharing storage with the originals and the
memory benefit does not materialize. Defaults to `False`.

Returns:

A TensorDict with reshaped keys. When `inplace=True` this is
`self`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... 'x': torch.arange(12).reshape(3, 4),
... }, batch_size=[3, 4])
>>> td = td.reshape(12)
>>> print(td['x'])
tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `reshape`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict