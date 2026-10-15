# TensorDictBase.copy

TensorDictBase.copy() → Self

Return a shallow copy of the tensordict (ie, copies the structure but not the data).

Equivalent to TensorDictBase.clone(recurse=False)

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).