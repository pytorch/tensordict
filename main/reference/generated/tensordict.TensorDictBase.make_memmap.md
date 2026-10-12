# TensorDictBase.make_memmap

*abstract*TensorDictBase.make_memmap(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*) → [MemoryMappedTensor](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor)

Creates an empty memory-mapped tensor given a shape and possibly a dtype.

Warning

This method is not lock-safe by design. A memory-mapped TensorDict instance present on multiple nodes
will need to be updated using the method [`memmap_refresh_()`](tensordict.TensorDictBase.memmap_refresh_.html#tensordict.TensorDictBase.memmap_refresh_).

Writing an existing entry will result in an error.

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the new entry to write. If the key is already present in the tensordict, an
exception is raised.
- **shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or**equivalent**,**torch.Tensor for nested tensors*) - the shape of the tensor to write.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype of the new tensor.
- **robust_key** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` (default), uses robust key encoding that safely
handles keys with path separators and special characters. If `False`,
uses legacy behavior (keys used as-is). If `None`, uses the default
robust behavior.

Returns:

A new memory mapped tensor.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).