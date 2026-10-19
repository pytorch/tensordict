# TensorDictBase.masked_fill

TensorDictBase.masked_fill_(*mask: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *value: [float](https://docs.python.org/3/builtins/functions.html#float) | [bool](https://docs.python.org/3/builtins/functions.html#bool)*) → Self

Writes `value` into every leaf where `mask` is `True`, in place.

The leaves keep their storage, and the call works on a locked
tensordict. See [In-place operations](../../concepts/operations.html#inplace-conventions).

Parameters:

- **mask** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a boolean mask over the leading batch
dimensions: its shape is the batch size, or its first
dimensions. It is expanded to the shape of each leaf.
- **value** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*or*[*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - the value to write.

Returns:

`self`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(source={'a': torch.zeros(3, 4)},
... batch_size=[3])
>>> mask = torch.tensor([True, False, False])
>>> _ = td.masked_fill_(mask, 1.0)
>>> td.get("a")
tensor([[1., 1., 1., 1.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).