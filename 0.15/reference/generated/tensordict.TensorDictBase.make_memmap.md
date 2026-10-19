# TensorDictBase.make_memmap

TensorDictBase.make_memmap(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*) → [MemoryMappedTensor](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor)

Adds an uninitialized memory-mapped tensor of shape `shape` at `key`, and returns it.

The tensordict must be memory-mapped (see [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap)), and the
file of the new tensor goes in its directory.

The entry is written even though a memory-mapped tensordict is
locked. Other tensordicts that share its files, for example in other
processes, see it only after
[`memmap_refresh_()`](tensordict.TensorDictBase.memmap_refresh_.html#tensordict.TensorDictBase.memmap_refresh_).

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the new entry. A tuple key creates
the missing nested tensordicts.
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

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the tensordict is not memory-mapped, or if the key
 exists.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> with tempfile.TemporaryDirectory() as path:
... td = TensorDict(a=torch.zeros(3), batch_size=[3]).memmap(path)
... x = td.make_memmap(("nested", "b"), (3, 4))
... print(td["nested", "b"].shape, td["nested", "b"] is x)
torch.Size([3, 4]) True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).