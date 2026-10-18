# TensorDictBase.make_memmap_from_tensor

TensorDictBase.make_memmap_from_tensor(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *tensor: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *copy_data: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*) → [MemoryMappedTensor](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor)

Adds a memory-mapped copy of `tensor` at `key`, and returns it.

The tensordict must be memory-mapped (see [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap)), and the
file of the new tensor goes in its directory. The new tensor does not
share storage with `tensor`.

The entry is written even though a memory-mapped tensordict is
locked. Other tensordicts that share its files, for example in other
processes, see it only after
[`memmap_refresh_()`](tensordict.TensorDictBase.memmap_refresh_.html#tensordict.TensorDictBase.memmap_refresh_).

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the new entry. A tuple key creates
the missing nested tensordicts.
- **tensor** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor to copy.

Keyword Arguments:

- **copy_data** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, the new tensor has the
shape and dtype of `tensor` but its content is not
initialized. Defaults to `True`.
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
... x = td.make_memmap_from_tensor("b", torch.ones(3, 2))
... print(td["b"].sum())
tensor(6.)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).