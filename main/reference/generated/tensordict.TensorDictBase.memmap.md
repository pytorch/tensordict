# TensorDictBase.memmap

TensorDictBase.memmap(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *copy_existing: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *return_early: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *share_non_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, *archive: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *compression: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *atomic: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Writes every leaf to a memory-mapped file and returns a new tensordict of them.

The tensordict is not changed: the result holds new
[`MemoryMappedTensor`](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor) leaves. Each leaf is written to
`<prefix>/<key>.memmap`, each nested tensordict to a subdirectory,
and each level has a `meta.json` file. Read the files back with
[`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap). [`save()`](tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save) and [`dumps()`](tensordict.TensorDictBase.dumps.html#tensordict.TensorDictBase.dumps) call this
method, and [`memmap_()`](tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_) writes in place. See [Save and load tensordicts](../../saving.html#saving).

The result is locked (see [Locking and temporary changes](../../concepts/keys.html#keys-locking)): you can write into its
leaves in place, but you cannot add, remove or replace an entry. Once
it is unlocked, [`is_memmap()`](tensordict.TensorDictBase.is_memmap.html#tensordict.TensorDictBase.is_memmap) returns `False`.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the directory to write to. If it
ends with `".tdz"`, a single archive file is written instead
(see `archive`). Defaults to `None`, which writes into the
directory of the tensordict if it is already memory-mapped
there, and otherwise into anonymous shared memory, which has no
file name and lasts as long as the process.
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
for them and returns the tensordict. Not supported for an
archive. Defaults to `False`.
- **share_non_tensor** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, non-tensor data is
shared between the processes of the node, so that an in-place
write of a non-tensor entry in one process is seen by the
others. Many non-tensor leaves (for example a large stack of
non-tensor data) can use up the memory. Defaults to `False`.
- **existsok** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, a file that already exists
at the path of a leaf (or at the archive path) raises a
`RuntimeError`. If `True`, it is overwritten. Defaults to
`True`.
- **robust_key** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, keys are percent-encoded
in file and directory names (`"a/b"` is written to
`a%2Fb.memmap`), so that every file stays inside `prefix`.
If `False`, keys are used as file names unchanged, as in the
legacy layout: do not use it with keys you do not trust.
`None` is the same as `True`. Defaults to `True`.
- **archive** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, `prefix` is a file, and
the tensordict is written to it as an archive: a zip file
whose entries are the files of the directory layout, with
uncompressed and aligned tensor payloads. Defaults to `None`,
which writes an archive if `prefix` ends with `".tdz"`.
- **compression** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the compression of the archive
entries: `"stored"`, `"deflate"`, `"bzip2"`, `"lzma"`,
or a [`zipfile`](https://docs.python.org/3/library/zipfile.html#module-zipfile) constant. A compressed leaf is
decompressed in memory when it is loaded, instead of being
memory-mapped. Only valid for an archive. Defaults to `None`,
which stores the entries uncompressed.
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
failure. Not supported with `return_early=True`, nor for a
directory on Windows, where a directory that holds open files
cannot be renamed. An archive is always written this way.
Defaults to `False`.

Returns:

A new memory-mapped tensordict, or a
[`TensorDictFuture`](tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) of it if
`return_early=True` and `num_threads > 1`.

Raises:

- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if a leaf already has a file elsewhere and
 `copy_existing=False`, if a file exists and
 `existsok=False`, or with `atomic=True`, if `prefix` is
 a non-empty directory and `existsok=False`, or if it holds
 no saved tensordict.
- [**NotADirectoryError**](https://docs.python.org/3/builtins/exceptions.html#NotADirectoryError) - if `atomic=True` and `prefix` is a file.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `compression` is given without an archive, or if
 an archive or `atomic=True` is requested without `prefix`.
- [**NotImplementedError**](https://docs.python.org/3/builtins/exceptions.html#NotImplementedError) - if `return_early=True` for an archive or
 with `atomic=True`, or `atomic=True` for a directory on
 Windows.

Note

For an archive, the result is `load_memmap(prefix)`: its leaves
are views into one copy-on-write mapping of the file, so in-place
writes do not reach the file, and it is not locked. Archives are
written in one pass, so `num_threads` does not speed them up. An
existing archive is replaced by a new file: a symlink is followed,
but other hard links keep the old content.
[`pack_memmap()`](tensordict.pack_memmap.html#tensordict.pack_memmap) and
[`unpack_memmap()`](tensordict.unpack_memmap.html#tensordict.unpack_memmap) convert between archives and
directories.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... obs=torch.zeros(3, 4),
... next=TensorDict(reward=torch.ones(3, 1), batch_size=[3]),
... batch_size=[3],
... )
>>> with tempfile.TemporaryDirectory() as path:
... mm = td.memmap(path)
... print(type(mm["obs"]).__name__, mm.is_locked)
... print(sorted(os.listdir(path)))
MemoryMappedTensor True
['meta.json', 'next', 'obs.memmap']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).