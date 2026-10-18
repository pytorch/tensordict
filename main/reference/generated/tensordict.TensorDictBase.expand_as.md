# TensorDictBase.expand_as

TensorDictBase.expand_as(*other: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → Self

Returns the tensordict expanded to the shape of `other`.

If `other` is a tensor, this is `expand(other.shape)`: the batch
size of the result is the shape of `other`. If `other` is a
tensordict or a tensorclass, each leaf is expanded to the shape of the
leaf with the same key in `other`, with
[`torch.Tensor.expand_as()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand_as.html#torch.Tensor.expand_as), and the result has the batch size of
`other`. The leaves of the result are views of the original leaves.

Parameters:

**other** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensor or the
tensordict whose shape to expand to.

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
>>> assert (expanded == 1).all()
>>> expanded.batch_size
torch.Size([2, 3])
>>> expanded["a"].shape, expanded["b", "c"].shape
(torch.Size([2, 3, 5, 4]), torch.Size([2, 3, 2, 6, 4]))
>>> td0.expand_as(torch.zeros(2, 3)).batch_size
torch.Size([2, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).