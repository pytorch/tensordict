# TensorDictBase.addcdiv

TensorDictBase.addcdiv(*other1: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *other2: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *value: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = 1*) → Self

Returns a new tensordict with `value * other1 / other2` added to every leaf.

\[\text{out}_i = \text{input}_i + \text{value} \times \frac{\text{other1}_i}{\text{other2}_i}\]

Parameters:

- **other1** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the numerator. A tensordict
is paired with the tensordict key by key and must have all its
keys. A tensor is aligned with the batch dimensions of each leaf
and expanded over its feature dimensions.
- **other2** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the denominator, paired
like `other1`.
- **value** (*Number**,**optional*) - the multiplier for
\(\text{other1} / \text{other2}\). Defaults to `1`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(x=torch.ones(2), batch_size=[2])
>>> td.addcdiv(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 2.0]), value=2)["x"]
tensor([2., 3.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).