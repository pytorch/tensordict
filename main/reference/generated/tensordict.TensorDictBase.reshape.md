# TensorDictBase.reshape

TensorDictBase.reshape(**args*, ***kwargs*) → Self

Returns a tensordict whose batch dimensions are reshaped to `shape`.

Each leaf is reshaped with [`torch.Tensor.reshape()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.reshape.html#torch.Tensor.reshape), and keeps its
feature dimensions. As with [`torch.Tensor.reshape()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.reshape.html#torch.Tensor.reshape), a leaf of
the result is a view of the original leaf when the memory layout
allows it, and a copy otherwise. The result has no names.

Parameters:

***shape** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the new batch size, as integers or as one list,
tuple or `torch.Size`. One dimension can be `-1`. The shape
is positional only: `reshape(shape=...)` raises
`TypeError`.

Keyword Arguments:

**inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict keeps its
identity and keys, its leaves are replaced one by one, and the
call returns `self`. A leaf keeps its storage when the
reshape can return a view of it. A lazy stack or an indexed
view (such as a slice of a
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict)), or a tensordict
that holds one, raises a `NotImplementedError` unless the
batch size stays: call [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) first. The legacy
lazy views (`set_lazy_legacy(True)`) still accept it and
leave their source out of step with its batch size (#3406).
See [In-place operations](../../concepts/operations.html#inplace-conventions). Defaults to `False`.

Returns:

A tensordict with the new batch size, or `self` when
`inplace=True` or when the batch size does not change.

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