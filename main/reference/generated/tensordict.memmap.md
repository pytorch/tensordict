# memmap

tensordict.memmap(*data: TensorCollection*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *copy_existing: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *return_early: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *share_non_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*) → TensorCollection

Writes all tensors onto memory-mapped tensors in a new tensordict.

Same as [`save()`](tensordict.save.html#tensordict.save), and as
[`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap), which holds the full text of
every argument and also takes `archive` and `compression` (see
#1954). Use it to share a tensordict between processes without
copies, or to keep data that is larger than RAM on disk. To convert the
tensordict in place, use [`memmap_()`](tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_).
See [Save and load tensordicts](../../saving.html#saving) and [Load data and share it between processes](../../data_loading.html#data-loading).

Parameters:

- **data** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**tensorclass instance*) - the tensordict to write.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the directory to write to, or a file name that
ends with `".tdz"`. If `None`, the tensors are written to
anonymous shared memory.
- **copy_existing** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, an entry that is already
memory-mapped to a file outside `prefix` raises an error. If
`True`, it is copied. Defaults to `False`.

Keyword Arguments:

- **num_threads** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the number of threads that write the
files. Defaults to `0`.
- **return_early** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and `num_threads > 0`,
a future of the result is returned. Defaults to `False`.
- **share_non_tensor** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, non-tensor entries are
shared between the processes of a node. Defaults to `False`.
- **robust_key** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, keys that hold path
separators or special characters are encoded in the file names.
Defaults to `True`.

Returns:

a new, locked tensordict whose leaves are
[`MemoryMappedTensor`](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor) instances.

Examples

```
>>> import torch
>>> import tensordict
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
>>> td_memmap = tensordict.memmap(td)
>>> type(td_memmap["a"]).__name__
'MemoryMappedTensor'
>>> td_memmap.is_locked
True
```