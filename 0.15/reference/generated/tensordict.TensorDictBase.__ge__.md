# TensorDictBase.__ge__

TensorDictBase.__ge__(*other: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → Self | [bool](https://docs.python.org/3/builtins/functions.html#bool)

Compares two tensordicts against each other using the "greater or equal" operator, for every key. The two tensordicts must have the same key set.

Returns:

a new TensorDict instance with all tensors are boolean
tensors of the same shape as the original tensors.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).