# TensorDictBase.pow

TensorDictBase.pow(*other: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *default: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Takes the power of each element in `self` with `other` and returns a tensor with the result.

`other` can be either a single `float` number, a Tensor or a `TensorDict`.

When `other` is a tensor, the shapes of `input`
and `other` must be broadcastable.

Parameters:

**other** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,**tensor**or**tensordict*) - the exponent value

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