# TensorDictBase.is_shared

TensorDictBase.is_shared() → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Checks if tensordict is in shared memory.

If a TensorDict instance is in shared memory, it is locked (entries cannot
be renamed, removed or added). If a `TensorDict` is created with
tensors that are all in shared memory, this does __not__ mean that `is_shared`
will return `True` (as a new tensor may or may not be in shared memory).
Only if one calls tensordict.share_memory_() or places the tensordict
on a device where the content is shared by default (eg, `"cuda"`)
will the tensordict be considered in shared memory.

This is always `True` for tensordicts on a CUDA device.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).