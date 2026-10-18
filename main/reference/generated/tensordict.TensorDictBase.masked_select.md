# TensorDictBase.masked_select

TensorDictBase.masked_select(*mask: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Returns a tensordict with the elements of the batch where `mask` is `True`.

The batch dimensions that `mask` covers are flattened into one, of
size `mask.sum()`, and the other batch dimensions are kept, as for
`td[mask]`. The data is copied. The result is a
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) without names.

Parameters:

**mask** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a boolean mask whose shape is the batch size,
or its leading dimensions.

Keyword Arguments:

**inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict keeps its
identity and keys, its leaves are replaced one by one, and the
call returns `self`. A lazy stack, or a tensordict that
holds one, raises a `NotImplementedError`: call
[`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) first. See [In-place operations](../../concepts/operations.html#inplace-conventions).
Defaults to `False`.

Returns:

The masked tensordict, or `self` with `inplace=True`.

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