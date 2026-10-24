# TensorDictBase.save

TensorDictBase.save(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *copy_existing: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *return_early: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *share_non_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, *archive: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *compression: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *atomic: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Saves the tensordict to disk, and returns a memory-mapped copy of it.

`save` (and its alias `dumps`) calls [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) with
`existsok=True`. [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) documents the arguments in full.
The tensordict is not changed. Read the files back with [`load()`](tensordict.TensorDictBase.load.html#tensordict.TensorDictBase.load).
See [Save and load tensordicts](../../saving.html#saving).

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the directory where the tensordict
is written, one file per leaf. If `prefix` ends with
`".tdz"` or `archive=True`, a single archive file is
written instead. Defaults to `None`, which writes into the
directory of the tensordict if it is already memory-mapped
there, and otherwise into anonymous shared memory, which has no
file name and lasts as long as the process.
- **copy_existing** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, a leaf that is
already a memory-mapped tensor with a file at another path
raises a `RuntimeError`. If `True`, the file is copied to
the new path. Defaults to `False`.

Keyword Arguments:

- **num_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of threads that write the
files. Defaults to `0`, which writes them in the calling
thread.
- **return_early** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and `num_threads > 1`,
returns a [`TensorDictFuture`](tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) while
the threads write the files. Defaults to `False`.
- **share_non_tensor** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, non-tensor data is
shared between the processes of the node. Defaults to
`False`.
- **robust_key** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, keys are percent-encoded
in file names, so that every file stays inside `prefix`.
Defaults to `True`.
- **archive** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, writes a single archive
file. Defaults to `None`, which writes an archive if
`prefix` ends with `".tdz"`.
- **compression** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the compression of the
archive entries. Defaults to `None`, which stores them
uncompressed.
- **atomic** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the directory is written
next to `prefix` and then renamed to it, so that an error or
a crash during the write leaves `prefix` as it was. Not
supported on Windows for a directory. An archive is always
written this way.
[`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) gives the details. Defaults to `False`.

Returns:

A new memory-mapped tensordict, as [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) returns.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), nested=TensorDict(b=torch.ones(3), batch_size=[3]), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.save(path)
... loaded = TensorDict.load(path)
... print((loaded == td).all())
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `save`

![](../../_images/sphx_glr_serialization_speed_thumb.png)

[Benchmarking TensorDict serialization speed](../../tutorials/serialization_speed.html)

Benchmarking TensorDict serialization speed