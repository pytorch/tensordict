# TensorDictBase.memmap_like

TensorDictBase.memmap_like(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *copy_existing: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *return_early: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *share_non_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, *archive: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Creates a memory-mapped tensordict with the same keys, shapes and dtypes, without copying the values.

The files are filled with zeros. Write the values into the result
afterwards, for example with [`update_()`](tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_) or indexing.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - directory prefix where the memory-mapped tensors will
be stored. The directory tree structure will mimic the tensordict's.
If `prefix` ends with `".tdz"` (or `archive=True` is
passed), a preallocated single-file archive is created
instead. See `archive` below.
- **copy_existing** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - If False (default), an exception will be raised if an
entry in the tensordict is already a tensor stored on disk
with an associated file, but is not saved in the correct
location according to prefix.
If `True`, any existing Tensor will be copied to the new location.

Keyword Arguments:

- **num_threads** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the number of threads used to write the memmap
tensors. Defaults to 0.
- **return_early** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and `num_threads>0`,
the method will return a future of the tensordict.
- **share_non_tensor** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the non-tensor data will be
shared between the processes and writing operation (such as inplace update
or set) on any of the workers within a single node will update the value
on all other workers. If the number of non-tensor leaves is high (e.g.,
sharing large stacks of non-tensor data) this may result in OOM or similar
errors. Defaults to `False`.
- **existsok** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `False`, an exception will be raised if a tensor already
exists in the same path. Defaults to `True`.
- **robust_key** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` (default), uses robust key encoding that safely
handles keys with path separators and special characters. If `False`,
uses legacy behavior (keys used as-is). If `None`, uses the default
robust behavior.
- **archive** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, `prefix` designates a
single file and a preallocated, zero-filled memmap archive is
created and loaded back with
`load_memmap(prefix, mode="r+")`: the returned tensordict
writes through to the archive. If `None` (default), archive
mode is enabled when `prefix` ends with `".tdz"`.
In-place writes leave the zip per-entry checksums stale; call
[`refresh_archive_checksums()`](tensordict.refresh_archive_checksums.html#tensordict.refresh_archive_checksums) before handing
the archive to tools that verify them. Nested tensors are not
supported in this mode.

The TensorDict is then locked, meaning that any writing operations that
isn't in-place will throw an exception (eg, rename, set or remove an
entry).
Once the tensordict is unlocked, the memory-mapped attribute is turned to `False`,
because cross-process identity is not guaranteed anymore.

Returns:

A new `TensorDict` instance with data stored as memory-mapped tensors if `return_early=False`,
otherwise a [`TensorDictFuture`](tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) instance.

Note

This is the recommended method to write a set of large buffers
on disk, as [`memmap_()`](tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_) will copy the information, which can
be slow for large content.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... "a": torch.zeros((3, 64, 64), dtype=torch.uint8),
... "b": torch.zeros(1, dtype=torch.int64),
... }, batch_size=[]).expand(1_000) # expand does not allocate new memory
>>> import tempfile
>>> with tempfile.TemporaryDirectory() as path:
... buffer = td.memmap_like(path)
... print(buffer.is_memmap(), buffer["a"].shape)
True torch.Size([1000, 3, 64, 64])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `memmap_like`

![](../../_images/sphx_glr_tensordict_memory_thumb.png)

[Devices and memory-mapped tensors](../../tutorials/tensordict_memory.html)

Devices and memory-mapped tensors