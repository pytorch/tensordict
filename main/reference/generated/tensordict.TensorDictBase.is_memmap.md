# TensorDictBase.is_memmap

TensorDictBase.is_memmap() → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Checks if tensordict is memory-mapped.

If a TensorDict instance is memory-mapped, it is locked (entries cannot
be renamed, removed or added). If a `TensorDict` is created with
tensors that are all memory-mapped, this does __not__ mean that `is_memmap`
will return `True` (as a new tensor may or may not be memory-mapped).
Only if one calls tensordict.memmap_() will the tensordict be
considered as memory-mapped.

This is always `True` for tensordicts on a CUDA device.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).