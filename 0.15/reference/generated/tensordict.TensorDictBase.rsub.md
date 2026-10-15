# TensorDictBase.rsub

TensorDictBase.rsub(*other: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*, ***, *alpha: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *default: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Subtracts `self`, scaled by `alpha`, from `other`, leaf by leaf.

\[\text{out}_i = \text{other}_i - \text{alpha} \times \text{self}_i\]

Supports broadcasting,
type promotion, and integer, float, and complex inputs.

Parameters:

**other** ([*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the value from which
`self` is subtracted. The dimensions of a tensor match the
leading dimensions of each leaf, and the tensor is expanded
over the other dimensions.

Keyword Arguments:

- **alpha** (*Number*) - the multiplier for `self`.
- **default** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the value to use for an
entry that only one of the two tensordicts has. If `None`
(default), the two tensordicts must have the same keys, and a
missing key raises `KeyError`. If `"intersection"`, only
the keys that both tensordicts have are computed, and the
other keys are dropped. Any other value replaces the missing
entries on both sides of the operation.

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