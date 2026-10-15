# TensorDictBase.make_memmap_from_tensor

TensorDictBase.make_memmap_from_tensor(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *tensor: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *copy_data: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*) → [MemoryMappedTensor](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor)

Creates an empty memory-mapped tensor given a tensor.

Warning

This method is not lock-safe by design. A memory-mapped TensorDict instance present on multiple nodes
will need to be updated using the method [`memmap_refresh_()`](tensordict.TensorDictBase.memmap_refresh_.html#tensordict.TensorDictBase.memmap_refresh_).

This method always copies the storage content if `copy_data` is `True` (i.e., the storage is not shared).

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the new entry to write. If the key is already present in the tensordict, an
exception is raised.
- **tensor** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor to replicate on physical memory.

Keyword Arguments:

- **copy_data** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optionaL*) - if `False`, the new tensor will share the metadata of the input such as
shape and dtype, but the content will be empty. Defaults to `True`.
- **robust_key** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` (default), uses robust key encoding that safely
handles keys with path separators and special characters. If `False`,
uses legacy behavior (keys used as-is). If `None`, uses the default
robust behavior.

Returns:

A new memory mapped tensor with the given storage.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).