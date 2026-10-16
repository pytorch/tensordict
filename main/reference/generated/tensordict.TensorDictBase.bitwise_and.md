# TensorDictBase.bitwise_and

TensorDictBase.bitwise_and(*other: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *default: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Performs a bitwise AND operation between `self` and `other`.

\[\text{out}_i = \text{input}_i \land \text{other}_i\]

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor or TensorDict to perform the bitwise AND with.

Keyword Arguments:

**default** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the value to use for an
entry that only one of the two tensordicts has. If `None`
(default), the two tensordicts must have the same keys, and a
missing key raises `KeyError`. If `"intersection"`, only
the keys that both tensordicts have are computed, and the
other keys are dropped. Any other value replaces the missing
entries on both sides of the operation.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).