# TensorDictBase.masked_fill

*abstract*TensorDictBase.masked_fill(*mask: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *value: [float](https://docs.python.org/3/builtins/functions.html#float) | [bool](https://docs.python.org/3/builtins/functions.html#bool)*) → Self

Out-of-place version of masked_fill.

Parameters:

- **mask** (*boolean torch.Tensor*) - mask of values to be filled. Shape
must match the tensordict batch-size.
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
>>> td1 = td.masked_fill(mask, 1.0)
>>> td1.get("a")
tensor([[1., 1., 1., 1.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).