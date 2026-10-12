# TensorDictBase.share_memory

*abstract*TensorDictBase.share_memory_() → Self

Places all the tensors in shared memory.

The TensorDict is then locked, meaning that any writing operations that
isn't in-place will throw an exception (eg, rename, set or remove an
entry).
Conversely, once the tensordict is unlocked, the share_memory attribute
is turned to `False`, because cross-process identity is not
guaranteed anymore.

Returns:

self

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).