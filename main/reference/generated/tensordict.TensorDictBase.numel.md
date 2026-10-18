# TensorDictBase.numel

TensorDictBase.numel() → [int](https://docs.python.org/3/builtins/functions.html#int)

Returns the number of elements in the batch, which is the product of `batch_size`.

The feature dimensions of the leaves do not count. The minimum is 1: a
tensordict with batch size `[]` or `[0]` returns 1, unlike
[`torch.Tensor.numel()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numel.html#torch.Tensor.numel), which returns 0 for a size of 0.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> TensorDict(a=torch.zeros(2, 3, 10), batch_size=[2, 3]).numel()
6
>>> TensorDict(batch_size=[]).numel()
1
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).