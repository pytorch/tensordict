# TensorDictBase.masked_fill

TensorDictBase.masked_fill_(*mask: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *value: [float](https://docs.python.org/3/builtins/functions.html#float) | [bool](https://docs.python.org/3/builtins/functions.html#bool)*) → Self

Fills the values corresponding to the mask with the desired value.

The values are written into the existing leaves, which keep their
storage. See [In-place operations](../../concepts/operations.html#inplace-conventions).

Parameters:

- **mask** (*boolean torch.Tensor*) - mask of values to be filled. Its
shape must match the leading dims of the batch size.
- **value** - value to used to fill the tensors.

Returns:

self

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