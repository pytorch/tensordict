# TensorDictBase.fliplr

TensorDictBase.fliplr()

Flip the tensordict in the left/right direction.

Flip the entries in each row in the left/right direction.
Columns are preserved, but appear in a different order than before.

Requires the tensordict to have at least 2 batch dimensions.

Returns:

a new tensordict with the second dimension flipped.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
>>> print(td["a"])
tensor([[0, 1, 2],
 [3, 4, 5]])
>>> print(td.fliplr()["a"])
tensor([[2, 1, 0],
 [5, 4, 3]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).