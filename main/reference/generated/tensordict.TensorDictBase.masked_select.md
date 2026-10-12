# TensorDictBase.masked_select

TensorDictBase.masked_select(*mask: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → Self

Masks all tensors of the TensorDict and return a new TensorDict instance with similar keys pointing to masked values.

Parameters:

**mask** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - boolean mask to be used for the tensors.
Shape must match the TensorDict `batch_size`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(source={'a': torch.zeros(3, 4)},
... batch_size=[3])
>>> mask = torch.tensor([True, False, False])
>>> td_mask = td.masked_select(mask)
>>> td_mask.get("a")
tensor([[0., 0., 0., 0.]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).