# TensorDictBase.is_consolidated

TensorDictBase.is_consolidated()

Checks if a TensorDict has a consolidated storage.

Returns `False` once [`memmap_()`](tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_) moves the leaves out of the
consolidated storage, or once an in-place write changes a non-tensor
entry, as the consolidated storage and metadata no longer match the
content. [`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate) then builds a new consolidated copy.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).