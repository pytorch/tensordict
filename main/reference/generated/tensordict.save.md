# save

tensordict.save(*data: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *copy_existing: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *return_early: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *share_non_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, *archive: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *compression: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *atomic: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)

Saves a tensordict to disk, and returns a memory-mapped copy of it.

Same as [`memmap()`](tensordict.memmap.html#tensordict.memmap), and as
[`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap), which holds the full text of
every argument. The directory mirrors the key structure of the
tensordict: one file per tensor and one metadata file per nested
tensordict. If `prefix` ends with `".tdz"`, a single archive file is
written instead. Load the result with [`load()`](tensordict.load.html#tensordict.load). See
[Save and load tensordicts](../../saving.html#saving).

Parameters:

- **data** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**tensorclass instance*) - the tensordict to save.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the directory to write to, or a file name that
ends with `".tdz"`. Defaults to `None`, which writes the
tensors to anonymous shared memory and saves nothing to disk.
- **copy_existing** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, an entry that is already
memory-mapped to a file outside `prefix` raises an error. If
`True`, it is copied. Defaults to `False`.

Keyword Arguments:

- **num_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of threads that write the
files. Defaults to `0`.
- **return_early** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and `num_threads > 1`,
returns a [`TensorDictFuture`](tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) at once, while
the threads write the files. Defaults to `False`.
- **share_non_tensor** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, non-tensor entries are
shared between the processes of a node. Defaults to `False`.
- **robust_key** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, keys that hold path
separators or special characters are encoded in the file names.
Defaults to `True`.
- **archive** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, `prefix` is a file and the
tensordict is written as a single zip archive that
[`load()`](tensordict.load.html#tensordict.load) can memory-map. If `None`, an archive
is written when `prefix` ends with `".tdz"`. Defaults to
`None`.
- **compression** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the compression of the archive
entries: `"stored"`, `"deflate"`, `"bzip2"`, `"lzma"` or a
[`zipfile`](https://docs.python.org/3/library/zipfile.html#module-zipfile) constant. Only valid for an archive. Defaults to
`None`, which stores them uncompressed. Only uncompressed
entries load without a copy.
- **atomic** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the directory is written next
to `prefix` and then renamed to it, so that an error or a crash
during the write leaves `prefix` as it was. Not supported on
Windows for a directory. [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap)
gives the details.
Defaults to `False`.

Returns:

A new, locked tensordict whose leaves are
[`MemoryMappedTensor`](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor) instances backed by the
saved files. `data` itself is not changed.

Examples

```
>>> import tempfile
>>> import torch
>>> import tensordict
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3, 2)}, batch_size=[3])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... saved = tensordict.save(td, tmpdir)
... print(saved.is_locked, type(saved["a"]).__name__)
... loaded = tensordict.load(tmpdir)
True MemoryMappedTensor
```