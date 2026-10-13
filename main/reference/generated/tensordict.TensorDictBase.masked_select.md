# TensorDictBase.masked_select

TensorDictBase.masked_select(*mask: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Masks all tensors of the TensorDict and return a new TensorDict instance with similar keys pointing to masked values.

Parameters:

**mask** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - boolean mask to be used for the tensors.
Shape must match the TensorDict `batch_size`.

Keyword Arguments:

**inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, this tensordict's identity
and key set are preserved; each leaf storage is replaced by
its masked counterpart one leaf at a time, keeping peak
memory close to `1x`. Not supported on
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict).
Defaults to `False`.

Returns:

a new tensordict with the masked values. When `inplace=True`
this is `self`.

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
>>> td.masked_select(mask, inplace=True) is td
True
>>> td.batch_size
torch.Size([1])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).