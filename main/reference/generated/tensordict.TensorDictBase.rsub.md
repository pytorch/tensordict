# TensorDictBase.rsub

TensorDictBase.rsub(*other: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*, ***, *alpha: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *default: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Returns a new tensordict with every leaf, scaled by `alpha`, subtracted from `other`.

`other - td` calls this method.

\[\text{out}_i = \text{other}_i - \text{alpha} \times \text{self}_i\]

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the value from which
the leaves are subtracted. A tensordict is paired with the
tensordict key by key. A tensor is aligned with the batch
dimensions of each leaf and expanded over its feature
dimensions. Batch sizes that differ are broadcast together.

Keyword Arguments:

- **alpha** (*Number**,**optional*) - the multiplier for the leaves. Defaults to
`None`, which subtracts them as they are.
- **default** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - what to do with the keys
that only one of the two tensordicts has. `None` requires the
same keys and raises a `KeyError` otherwise.
`"intersection"` computes the keys that both have and drops
the others. A tensor stands in for the missing entry on either
side, and the result has the keys of both. Defaults to `None`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.full((2,), 10.0), batch_size=[2])
>>> other = TensorDict(a=torch.ones(2), batch_size=[2])
>>> td.rsub(other)["a"]
tensor([-9., -9.])
>>> td.rsub(other, alpha=2)["a"] # 1 - 2 * 10
tensor([-19., -19.])
>>> td.rsub(torch.tensor([1.0, 2.0]))["a"]
tensor([-9., -8.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).