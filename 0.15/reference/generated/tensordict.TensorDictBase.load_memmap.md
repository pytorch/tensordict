# TensorDictBase.load_memmap

*classmethod*TensorDictBase.load_memmap(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, *subpath: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Loads a memory-mapped tensordict from disk.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path to folder*) - the path to the folder where the
saved tensordict should be fetched, or the path to a memmap
archive file written through
`save(..., archive=True)` / a `".tdz"` prefix (or packed
with [`pack_memmap()`](tensordict.pack_memmap.html#tensordict.pack_memmap)). Archives are
memory-mapped once and every leaf is exposed as a zero-copy
view into the mapping: only the pages of the leaves that are
actually accessed are read from disk. Unlike directory-backed
tensordicts, in-place writes to the leaves of an
archive-loaded tensordict do not propagate to the file by
default (see `mode`).
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**equivalent**,**optional*) - if provided, the
data will be asynchronously cast to that device.
Supports "meta" device, in which case the data isn't loaded
but a set of empty "meta" tensors are created. This is
useful to get a sense of the total model size and structure
without actually opening any file.
- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, synchronize won't be
called after loading tensors on device. Defaults to `False`.
- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - optional tensordict where the data
should be loaded. Its nested containers are reused, but its
leaves are rebound to the loaded tensors, so their storage is
not reused. Keys of `out` that are missing from the saved
data are removed. If `out` has a device, the data is loaded
on it, and a different `device` raises a `ValueError`.
To write the data into the preallocated storage of `out`,
use `out.update_(TensorDict.load_memmap(prefix, device=device))`
instead.
- **robust_key** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` (default), expects robust key encoding was used
when saving and decodes filenames accordingly. If `False`, uses legacy
behavior. If `None`, uses the default robust behavior.
- **subpath** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*or**str path**,**optional*) - the location of a
nested tensordict to load, as a nested key (e.g.
`("module", "0")`, with arbitrary nesting allowed as usual)
or as a `"/"`-separated string path (e.g. `"module/0"`).
Only that subtree is loaded. Works both for directories
(equivalent to appending the path to `prefix`) and for
archives. A nested key reaches the entries of a saved
tensorclass as `obj.get(key)` does, without its
`_tensordict` level.
- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - how the files are memory-mapped. With
`"r"`, the mapping is copy-on-write: in-place writes to the
leaves stay in memory. They reach the files only if the
tensordict is saved there, and [`memmap_()`](tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_) without a
prefix does not write back to `prefix`. With `"r+"`, the
mapping is shared: in-place writes propagate to the files,
and a file that is not writable raises a
[`PermissionError`](https://docs.python.org/3/builtins/exceptions.html#PermissionError). Defaults to `None`, which maps
archives as with `"r"`, and each file of a directory as with
`"r+"` if the process can write it, as with `"r"`
otherwise. Prefer `"r"` for data that is only read: on some
network file systems (e.g. Lustre), page faults on a shared
writable mapping take write locks, so readers on different
nodes block each other. On Linux, copy-on-write mappings count
against the memory commit limit, so a file larger than the
available RAM and swap can fail to map with `"r"`. With
archives, `"r+"` requires uncompressed, aligned tensor
payloads (i.e. archives written by tensordict without
`compression`) and is not available for nested-tensor
leaves. In-place writes do not update the per-entry CRC-32
stored by the zip format; `load_memmap()` ignores
checksums, but call
[`refresh_archive_checksums()`](tensordict.refresh_archive_checksums.html#tensordict.refresh_archive_checksums) before handing
a modified archive to tools that verify them (`unzip`,
[`unpack_memmap()`](tensordict.unpack_memmap.html#tensordict.unpack_memmap), ...).
- **num_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - number of threads used to decompress
the leaves of a compressed archive (deflate entries are
inflated in parallel, which scales nearly linearly). Without
compression, loading is a metadata-only operation and this
argument has no effect. Defaults to `0` (sequential).
- **allow_pickle** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether pickled non-tensor fields
may be loaded. Pickle can execute arbitrary code, so pass
`True` only for data from a trusted source and `False`
for untrusted data. In TensorDict 0.14 and 0.15, omitting
this option loads pickle with a `FutureWarning`; the
default will change to `False` in 0.16. Saves without
a pickle sidecar do not require this option.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict.fromkeys(["a", "b", "c", ("nested", "e")], 0)
>>> _ = td.memmap("./saved_td")
>>> td_load = TensorDict.load_memmap("./saved_td")
>>> assert (td == td_load).all()
```

This method also allows loading nested tensordicts.

Examples

```
>>> nested = TensorDict.load_memmap("./saved_td/nested")
>>> assert nested["e"] == 0
```

A tensordict can also be loaded on "meta" device or, alternatively,
as a fake tensor.

Examples

```
>>> import tempfile
>>> td = TensorDict({"a": torch.zeros(()), "b": {"c": torch.zeros(())}})
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.save(path)
... td_load = TensorDict.load_memmap(path, device="meta")
... print("meta:", td_load)
... from torch._subclasses import FakeTensorMode
... with FakeTensorMode():
... td_load = TensorDict.load_memmap(path)
... print("fake:", td_load)
meta: TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=meta, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([]), device=meta, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=meta,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=meta,
 is_shared=False)
fake: TensorDict(
 fields={
 a: FakeTensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: FakeTensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)
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