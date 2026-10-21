# TensorDictBase.where

TensorDictBase.where(*condition: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *other: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *pad: [int](https://docs.python.org/3/builtins/functions.html#int) | [bool](https://docs.python.org/3/builtins/functions.html#bool) = None*, *update_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Returns a tensordict with the leaves where `condition` is `True` and `other` elsewhere.

[`torch.where()`](https://docs.pytorch.org/docs/stable/generated/torch.where.html#torch.where) is applied to every leaf. `condition` is aligned
with the batch dimensions of each leaf and expanded over its feature
dimensions.

Parameters:

- **condition** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a boolean mask over the leading batch
dimensions.
- **other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the values to take
where `condition` is `False`. A tensordict is paired with
the tensordict key by key. On a
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), a tensor `other`
that starts with the batch shape up to the stack dimension is
split along that dimension, one piece per member, so it must
have the stack size there: expand it to the batch shape first.
The pieces line up with the leaves as in
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) only for the leaves with as
many dims as `other`; for the others the lazy stack emits a
[`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning): starting with TensorDict 0.17, the
default will change to broadcasting like TensorDict.where,
which aligns `other` with each leaf from the right.

Keyword Arguments:

- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - the tensordict to write the result
into, which is returned. Defaults to `None`, which returns a
new tensordict.
- **pad** (*Number**or*[*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the value that stands in for an
entry that only one of the tensordict and a tensordict
`other` has. Defaults to `None`, which raises a
`KeyError` for such an entry.
- **update_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - passed to [`update()`](tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update)
when the result is written into `out`. If `True`, a lazy
stack `out` gets as many members as the result. When
`other` is not a tensordict, or `self` is a lazy stack, an
`out` whose batch size is a prefix of the result's
(`[]` for `[2]`) keeps it, with a [`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning):
starting with TensorDict 0.17, the default will change to
giving `out` the batch size of the result, as
[`update()`](tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update) does. Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(x=torch.ones(2, 3), batch_size=[2])
>>> td.where(torch.tensor([True, False]), 0.0)["x"]
tensor([[1., 1., 1.],
 [0., 0., 0.]])
>>> other = TensorDict(y=torch.zeros(2), batch_size=[2])
>>> sorted(td.where(torch.tensor([True, False]), other, pad=-1).keys())
['x', 'y']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).