# TensorDictBase.masked_fill

TensorDictBase.masked_fill(*mask: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *value: [float](https://docs.python.org/3/builtins/functions.html#float) | [bool](https://docs.python.org/3/builtins/functions.html#bool)*) → Self

Returns a copy of the tensordict with `value` written into every leaf where `mask` is `True`.

The out-of-place version of [`masked_fill_()`](tensordict.TensorDictBase.masked_fill_.html#tensordict.TensorDictBase.masked_fill_): the tensordict is
not modified.

Parameters:

- **mask** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a boolean mask over the leading batch
dimensions: its shape is the batch size, or its first
dimensions. It is expanded to the shape of each leaf.
- **value** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*or*[*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - the value to write.

Returns:

A new tensordict.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(source={'a': torch.zeros(3, 4)},
... batch_size=[3])
>>> mask = torch.tensor([True, False, False])
>>> td1 = td.masked_fill(mask, 1.0)
>>> td1.get("a")
tensor([[1., 1., 1., 1.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).