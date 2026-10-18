# TensorDictBase.is_contiguous

TensorDictBase.is_contiguous() → [bool](https://docs.python.org/3/builtins/functions.html#bool)

`True` if all the leaves of the tensordict, nested ones included, are contiguous.

A lazy stack and a [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict), which
build their leaves when they are read, always return `False`. See
[`contiguous()`](tensordict.TensorDictBase.contiguous.html#tensordict.TensorDictBase.contiguous).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).