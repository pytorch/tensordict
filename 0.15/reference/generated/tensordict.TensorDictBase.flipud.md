# TensorDictBase.flipud

TensorDictBase.flipud()

Flip the tensordict in the up/down direction.

Flip the entries in each column in the up/down direction.
Rows are preserved, but appear in a different order than before.

Requires the tensordict to have at least 1 batch dimension.

Returns:

a new tensordict with the first dimension flipped.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
>>> print(td["a"])
tensor([[0, 1, 2],
 [3, 4, 5]])
>>> print(td.flipud()["a"])
tensor([[3, 4, 5],
 [0, 1, 2]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).