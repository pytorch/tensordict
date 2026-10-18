# TensorDictBase.addcmul

TensorDictBase.addcmul(*other1: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *other2: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *value: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = 1*) → Self

Returns a new tensordict with `value * other1 * other2` added to every leaf.

\[\text{out}_i = \text{input}_i + \text{value} \times \text{other1}_i \times \text{other2}_i\]

Parameters:

- **other1** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the first factor. A
tensordict is paired with the tensordict key by key and must
have all its keys. A tensor is aligned with the batch dimensions
of each leaf and expanded over its feature dimensions.
- **other2** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the second factor, paired
like `other1`.

Keyword Arguments:

**value** (*Number**,**optional*) - the multiplier for
\(\text{other1} \times \text{other2}\). Defaults to `1`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(x=torch.ones(2), batch_size=[2])
>>> td.addcmul(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 3.0]), value=2)["x"]
tensor([ 7., 13.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).