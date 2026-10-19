# TensorDictBase.fliplr

TensorDictBase.fliplr()

Returns a tensordict with the elements reversed along batch dimension 1.

It is `flip(1)`, as for [`torch.fliplr()`](https://docs.pytorch.org/docs/stable/generated/torch.fliplr.html#torch.fliplr). The tensordict needs at
least 2 batch dimensions, or a `RuntimeError` is raised. The data is
copied.

It can be used as a context manager, as [`unsqueeze()`](tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze) can: when
the `with` block exits, the entries of the block are written back to
the tensordict with the inverse operation.

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