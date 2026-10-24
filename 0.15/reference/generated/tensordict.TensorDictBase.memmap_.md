# TensorDictBase.memmap

TensorDictBase.memmap_(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *copy_existing: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *return_early: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *share_non_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, *atomic: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Writes every leaf to a memory-mapped file in place, and returns `self`.

Writes the same files as [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap), then replaces each leaf of
the tensordict with its [`MemoryMappedTensor`](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor), so the
`data_ptr()` of every leaf changes. The device of the tensordict
becomes `"cpu"`. See [Save and load tensordicts](../../saving.html#saving).

The tensordict is then locked (see [Locking and temporary changes](../../concepts/keys.html#keys-locking)): you can write
into its leaves in place, but you cannot add, remove or replace an
entry. Once it is unlocked, [`is_memmap()`](tensordict.TensorDictBase.is_memmap.html#tensordict.TensorDictBase.is_memmap) returns `False`.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the directory to write to.
Defaults to `None`, which writes into the directory of the
tensordict if it is already memory-mapped there, and otherwise
into anonymous shared memory, which has no file name and lasts
as long as the process.
- **copy_existing** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, a leaf that is
already a memory-mapped tensor with a file at another path
raises a `RuntimeError`, so that data on disk is not
duplicated by accident. If `True`, the file is copied to the
new path. Defaults to `False`.

Keyword Arguments:

- **num_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of threads that write the
files. `0` and `1` write them one after the other in the
calling thread. Threads help with a few large leaves and slow
down many small ones. Defaults to `0`.
- **return_early** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and `num_threads > 1`,
returns a [`TensorDictFuture`](tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) at once,
while the threads write the files: `future.result()` waits
for them and returns the tensordict. Defaults to `False`.
- **share_non_tensor** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, non-tensor data is
shared between the processes of the node, so that an in-place
write of a non-tensor entry in one process is seen by the
others. Many non-tensor leaves (for example a large stack of
non-tensor data) can use up the memory. Defaults to `False`.
- **existsok** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, a file that already exists
at the path of a leaf raises a `RuntimeError`. If `True`,
it is overwritten. Defaults to `True`.
- **robust_key** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, keys are percent-encoded
in file and directory names (`"a/b"` is written to
`a%2Fb.memmap`), so that every file stays inside `prefix`.
If `False`, keys are used as file names unchanged, as in the
legacy layout: do not use it with keys you do not trust.
`None` is the same as `True`. Defaults to `True`.
- **atomic** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the files are written to a
new directory next to `prefix`, which is then renamed to
`prefix`, so that an error or a crash during the write
leaves `prefix` as it was. The new directory is removed if
the write fails (but not after a crash). An existing directory
at `prefix` is replaced as a whole once the new one is
complete, so files that the tensordict does not write are
removed: it must be empty or hold a saved tensordict (a
`meta.json` file). Readers see the old or the new directory,
except for the short time between two renames, during which
`prefix` is missing. The write needs about twice the disk
space of the tensordict while it runs, and a leaf that is
already memory-mapped to a file is copied, which needs
`copy_existing=True`. The files are not synced to the disk,
so this guards against a killed process, not against a power
failure. If the write fails, the leaves already converted map
deleted files, so a retry needs `copy_existing=True`. Not
supported with `return_early=True`, nor on Windows, where a
directory that holds open files cannot be renamed. Defaults
to `False`.

Returns:

`self`, or a [`TensorDictFuture`](tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) of it if
`return_early=True` and `num_threads > 1`.

Raises:

- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if a leaf already has a file elsewhere and
 `copy_existing=False`, if a file exists and
 `existsok=False`, if the tensordict is in shared memory
 (see [`share_memory_()`](tensordict.TensorDictBase.share_memory_.html#tensordict.TensorDictBase.share_memory_)), or if it is a
 [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict), a
 [`TensorDictStore`](tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore), a
 [`LazyStackedTensorDictStore`](tensordict.store.LazyStackedTensorDictStore.html#tensordict.store.LazyStackedTensorDictStore) or one of
 their elements. Use [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) to make a memory-mapped
 copy of those. With `atomic=True`, also if `prefix` is a
 non-empty directory and `existsok=False`, or if it holds no
 saved tensordict.
- [**NotADirectoryError**](https://docs.python.org/3/builtins/exceptions.html#NotADirectoryError) - if `atomic=True` and `prefix` is a file.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `atomic=True` and there is no directory to write
 to.
- [**NotImplementedError**](https://docs.python.org/3/builtins/exceptions.html#NotImplementedError) - if `return_early=True` with
 `atomic=True`, or `atomic=True` on Windows.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(obs=torch.zeros(3, 4), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as path:
... out = td.memmap_(path)
... print(out is td, td.is_memmap(), td.is_locked)
... print(type(td["obs"]).__name__)
True True True
MemoryMappedTensor
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `memmap_`

![](../../_images/sphx_glr_tensordict_memory_thumb.png)

[Devices and memory-mapped tensors](../../tutorials/tensordict_memory.html)

Devices and memory-mapped tensors