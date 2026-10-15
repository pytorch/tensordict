# TensorDictBase.__xor__

TensorDictBase.__xor__(*other: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → Self | [bool](https://docs.python.org/3/builtins/functions.html#bool)

XOR operation over two tensordicts, for evey key.

The two tensordicts must have the same key set.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*, or*[*float*](https://docs.python.org/3/builtins/functions.html#float)) - the value to compare against.

Returns:

a new TensorDict instance with all tensors are boolean
tensors of the same shape as the original tensors.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).