# TensorDictBase.sorted_keys

*property*TensorDictBase.sorted_keys*: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*

Returns the keys sorted in alphabetical order.

Does not support extra arguments.

If the TensorDict is locked, the keys are cached until the tensordict
is unlocked for faster execution.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).