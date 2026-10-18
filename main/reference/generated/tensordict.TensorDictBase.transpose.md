# TensorDictBase.transpose

TensorDictBase.transpose(*dim0*, *dim1*)

Returns a tensordict with the batch dimensions `dim0` and `dim1` swapped.

The leaves of the result are views of the original leaves, so an
in-place change of the result changes the tensordict too. The names
are swapped with the dimensions. To write new entries back to the
tensordict, use the result as a context manager.

Parameters:

- **dim0** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the first batch dimension to swap.
- **dim1** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the second batch dimension to swap.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> tensordict = TensorDict({"a": torch.zeros(3, 4, 5)}, [3, 4])
>>> tensordict_transpose = tensordict.transpose(0, 1)
>>> print(tensordict_transpose.shape)
torch.Size([4, 3])
>>> with tensordict.transpose(0, 1) as tensordict_transpose:
... tensordict_transpose["b"] = torch.zeros(4, 3)
>>> print(tensordict.get("b").shape)
torch.Size([3, 4])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `transpose`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict