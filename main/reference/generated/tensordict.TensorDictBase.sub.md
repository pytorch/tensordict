# TensorDictBase.sub

TensorDictBase.sub(*other: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*, ***, *alpha: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *default: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Subtracts `other`, scaled by `alpha`, from `self`.

\[\text{out}_i = \text{input}_i - \text{alpha} \times \text{other}_i\]

Supports broadcasting,
type promotion, and integer, float, and complex inputs.

Parameters:

**other** ([*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)*,**Tensor**or**Number*) - the tensor or number to subtract from `self`.

Keyword Arguments:

- **alpha** (*Number*) - the multiplier for `other`.
- **default** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the value to use for an
entry that only one of the two tensordicts has. If `None`
(default), the two tensordicts must have the same keys, and a
missing key raises `KeyError`. If `"intersection"`, only
the keys that both tensordicts have are computed, and the
other keys are dropped. Any other value replaces the missing
entries on both sides of the operation.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).