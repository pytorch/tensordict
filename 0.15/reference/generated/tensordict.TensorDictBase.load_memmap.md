# TensorDictBase.load_memmap

*classmethod*TensorDictBase.load_memmap(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, *subpath: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Loads a tensordict that [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) or [`save()`](tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save) wrote, from a directory or an archive.

The leaves of the result read the files lazily: from a directory, each
leaf is a [`MemoryMappedTensor`](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor) of its file, and from
an archive, each leaf is a view into one memory map of the archive.
Only the pages of the leaves that you use are read from disk. The
result has the class that was saved (a tensorclass is loaded as that
tensorclass), and it is not locked. [`load()`](tensordict.TensorDictBase.load.html#tensordict.TensorDictBase.load) is this method. See
[Save and load tensordicts](../../saving.html#saving).

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the directory to read, or an archive file
written by `save(..., archive=True)`, by a `".tdz"` prefix,
or by [`pack_memmap()`](tensordict.pack_memmap.html#tensordict.pack_memmap).
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. The leaves are copied to it, which reads the files,
unless it is the CPU.
With `"meta"`, no file is read: the leaves are meta tensors,
which show the structure and size of the data. Defaults to
`None`, which keeps the leaves on the CPU, memory-mapped.
- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, does not synchronize
the device after the copies to `device`. Defaults to
`False`.

Keyword Arguments:

- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict to load the data into.
Its nested tensordicts are reused, but its leaves are replaced
by the loaded tensors, so their storage is not reused. Its keys
that are not in the saved data are removed. If `out` has a
device, the data is loaded on it, and another `device` raises
a `ValueError`. To write the data into the storage of
`out`, use
`out.update_(TensorDict.load_memmap(prefix, device=device))`.
Defaults to `None`.
- **robust_key** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, reads the
percent-encoded file names that [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) writes by
default, and falls back to the legacy names of keys that are
safe file names. If `False`, reads the legacy layout only.
`None` is the same as `True`. Defaults to `True`.
- **subpath** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the nested tensordict to
load, as a nested key (`("module", "0")`) or as a
`"/"`-separated path (`"module/0"`). Only that subtree is
read, from a directory or an archive. A nested key reaches the
entries of a saved tensorclass as `obj.get(key)` does.
Defaults to `None`, which loads the whole tensordict.
- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - how the files are mapped. With `"r"`, the
mapping is copy-on-write: in-place writes into the leaves stay
in memory, and the files do not change. With `"r+"`, in-place
writes reach the files, and a file that the process cannot
write raises a `PermissionError`. Defaults to `None`, which
maps an archive as with `"r"`, and each file of a directory
as with `"r+"` if the process can write it, and as with
`"r"` otherwise. See the note below.
- **num_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of threads that decompress
the leaves of a compressed archive. It has no effect on an
uncompressed archive or a directory. Defaults to `0`, which
decompresses them one after the other.
- **allow_pickle** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether non-tensor data saved with
[`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle) may be loaded. Pickle can run arbitrary code: pass
`True` only for data you trust, and `False` for other data,
which raises a `RuntimeError` on pickled data. Defaults to
`None`, which loads pickled data with a `FutureWarning`.
Starting with TensorDict 0.16, the default will change to
`False`. Data saved without pickle loads with any value.

Returns:

The loaded tensordict, or `out` if it is given.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `prefix` is a file that is not an archive, if
 `subpath` names no saved tensordict, or if `mode` is not
 `None`, `"r"` or `"r+"`.

Note

Load with `mode="r"` data that you only read: on some network
file systems (for example Lustre), page faults on a shared
writable mapping take write locks, so readers on different nodes
block each other. On Linux, a copy-on-write mapping counts against
the memory commit limit, so a file larger than RAM plus swap can
fail to map with `"r"`. With `"r"`, [`memmap_()`](tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_) without a
prefix does not write back to `prefix`.

Note

An archive opened with `mode="r+"` must have uncompressed,
aligned payloads (as written by tensordict without
`compression`) and no nested tensor. In-place writes do not
update the CRC-32 checksums of its entries: `load_memmap` does
not check them, but call
[`refresh_archive_checksums()`](tensordict.refresh_archive_checksums.html#tensordict.refresh_archive_checksums) before you give the
archive to a tool that does, such as `unzip`.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), nested=TensorDict(b=torch.ones(3)))
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.memmap(path)
... loaded = TensorDict.load_memmap(path)
... print((loaded == td).all(), type(loaded["a"]).__name__)
... # load one nested tensordict
... print(TensorDict.load_memmap(path, subpath="nested")["b"])
... # read the structure without reading the data
... print(TensorDict.load_memmap(path, device="meta")["a"].device)
True MemoryMappedTensor
MemoryMappedTensor([1., 1., 1.])
meta
```

With a `FakeTensorMode` active, the leaves are fake tensors and
no file is read:

```
>>> from torch._subclasses import FakeTensorMode
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.memmap(path)
... with FakeTensorMode():
... print(type(TensorDict.load_memmap(path)["a"]).__name__)
FakeTensor
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `load_memmap`

![](../../_images/sphx_glr_serialization_speed_thumb.png)

[Benchmarking TensorDict serialization speed](../../tutorials/serialization_speed.html)

Benchmarking TensorDict serialization speed
![](../../_images/sphx_glr_tensordict_memory_thumb.png)

[Devices and memory-mapped tensors](../../tutorials/tensordict_memory.html)

Devices and memory-mapped tensors