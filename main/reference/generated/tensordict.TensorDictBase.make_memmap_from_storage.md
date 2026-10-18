# TensorDictBase.make_memmap_from_storage

TensorDictBase.make_memmap_from_storage(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *storage: [UntypedStorage](https://docs.pytorch.org/docs/stable/storage.html#torch.UntypedStorage)*, *shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*) → [MemoryMappedTensor](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor)

Adds a memory-mapped tensor at `key` that reads its data from `storage`, and returns it.

The tensordict must be memory-mapped (see [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap)).

The entry is written even though a memory-mapped tensordict is
locked. Other tensordicts that share its files, for example in other
processes, see it only after
[`memmap_refresh_()`](tensordict.TensorDictBase.memmap_refresh_.html#tensordict.TensorDictBase.memmap_refresh_).

If the storage has a file, it must be the file of the new entry in the
directory of the tensordict, or the call raises a `RuntimeError`. A
storage without a file is used as it is: its data is not written to
the directory.

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the new entry. A tuple key creates
the missing nested tensordicts.
- **storage** ([*UntypedStorage*](https://docs.pytorch.org/docs/stable/storage.html#torch.UntypedStorage)) - the storage of the new tensor. It
must be in physical memory.
- **shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the shape of the new tensor,
or a tensor of shapes for a nested tensor.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype of the new tensor.
Defaults to `None`, which uses
[`torch.get_default_dtype()`](https://docs.pytorch.org/docs/stable/generated/torch.get_default_dtype.html#torch.get_default_dtype).
- **robust_key** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, keys are percent-encoded
in file names, so that every file stays inside the directory of
the tensordict. If `False`, keys are used as file names
unchanged. `None` is the same as `True`. Defaults to
`True`.

Returns:

The new [`MemoryMappedTensor`](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor).

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the tensordict is not memory-mapped, if the key
 exists, or if the file of the storage is not the file of the
 new entry.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).