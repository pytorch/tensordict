# TensorDictBase.saved_path

*property*TensorDictBase.saved_path

Returns the path where a memmap saved TensorDict is being stored.

This argument valishes as soon as is_memmap() returns `False` (e.g., when the tensordict is unlocked).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).