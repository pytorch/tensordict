# TensorDictBase.expand_as

TensorDictBase.expand_as(*other: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → Self

Broadcasts the shape of the tensordict to the shape of other and expands it accordingly.

If the input is a tensor collection (tensordict or tensorclass),
the leaves will be expanded on a one-to-one basis.

Examples

```
>>> from tensordict import TensorDict
>>> import torch
>>> td0 = TensorDict({
... "a": torch.ones(3, 1, 4),
... "b": {"c": torch.ones(3, 2, 1, 4)}},
... batch_size=[3],
... )
>>> td1 = TensorDict({
... "a": torch.zeros(2, 3, 5, 4),
... "b": {"c": torch.zeros(2, 3, 2, 6, 4)}},
... batch_size=[2, 3],
... )
>>> expanded = td0.expand_as(td1)
>>> assert (expanded==1).all()
>>> print(expanded)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([2, 3, 5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([2, 3, 2, 6, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2, 3]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([2, 3]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).