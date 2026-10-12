# TensorDictBase.bytes

TensorDictBase.bytes(***, *count_duplicates: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → [int](https://docs.python.org/3/builtins/functions.html#int)

Counts the number of bytes of the contained tensors.

Keyword Arguments:

**count_duplicates** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - Whether to count duplicated tensor as independent or not.
If `False`, only strictly identical tensors will be discarded (same views but different
ids from a common base tensor will be counted twice). Defaults to True (each tensor is assumed
to be a single copy).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).