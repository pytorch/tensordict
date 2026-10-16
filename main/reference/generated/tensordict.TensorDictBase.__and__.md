# TensorDictBase.__and__

TensorDictBase.__and__(*other: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*) → Self

Returns a new TensorDict instance with all tensors performing a logical or bitwise AND operation with the given value.

Parameters:

**other** - The value to perform the AND operation with.

Returns:

A new TensorDict instance with the same key set as the original,
but with all tensors having performed a AND operation with the given value.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).