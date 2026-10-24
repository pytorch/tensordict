# TensorDictBase.memmap_like

TensorDictBase.memmap_like(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *copy_existing: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *return_early: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *share_non_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, *archive: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Creates a memory-mapped tensordict with the same keys, shapes and dtypes, without copying the values.

Use it to preallocate a large buffer on disk, then fill it with
[`update_()`](tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_) or by indexing: [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) would write every
value first. New files are filled with zeros. A file that already
exists at the path of a leaf (with `existsok=True`) is mapped as it
is and keeps its content, which lets you reopen a buffer, or its
start with a smaller leaf. If the file is smaller than the leaf, it
is grown with zeros and a `FutureWarning` is emitted. Starting with
TensorDict 0.17, the default will change to raising a
`RuntimeError` in that case. A file written with another dtype or
shape is not detected otherwise, and its bytes are read as the new
dtype. An archive (`".tdz"`) is always
rebuilt with zeros. The tensordict is not changed. See
[Save and load tensordicts](../../saving.html#saving).

The result is locked (see [Locking and temporary changes](../../concepts/keys.html#keys-locking)): you can write into its
leaves in place, but you cannot add, remove or replace an entry. Once
it is unlocked, [`is_memmap()`](tensordict.TensorDictBase.is_memmap.html#tensordict.TensorDictBase.is_memmap) returns `False`.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the directory to write to. If it
ends with `".tdz"`, a single archive file is written instead
(see `archive`). Defaults to `None`, which allocates the
leaves in anonymous shared memory, which has no file name and
lasts as long as the process.
- **copy_existing** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - accepted for symmetry with
[`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap). It has no effect, since no value is copied.
Defaults to `False`.

Keyword Arguments:

- **existsok** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, a file that already exists
at the path of a leaf (or at the archive path) raises a
`RuntimeError`. If `True`, it is reused. Defaults to
`True`.
- **num_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of threads that create the
files. `0` and `1` create them one after the other in the
calling thread. Defaults to `0`.
- **return_early** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and `num_threads > 1`,
returns a [`TensorDictFuture`](tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) at once,
while the threads create the files: `future.result()` waits
for them and returns the tensordict. Not supported for an
archive. Defaults to `False`.
- **share_non_tensor** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, non-tensor data is
shared between the processes of the node, so that an in-place
write of a non-tensor entry in one process is seen by the
others. Many non-tensor leaves (for example a large stack of
non-tensor data) can use up the memory. Defaults to `False`.
- **robust_key** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, keys are percent-encoded
in file and directory names (`"a/b"` is written to
`a%2Fb.memmap`), so that every file stays inside `prefix`.
If `False`, keys are used as file names unchanged, as in the
legacy layout: do not use it with keys you do not trust.
`None` is the same as `True`. Defaults to `True`.
- **archive** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, `prefix` is a file, and a
zero-filled archive is created there and loaded back with
`load_memmap(prefix, mode="r+")`, so that in-place writes into
the result reach the archive. Nested tensors are not supported.
Defaults to `None`, which creates an archive if `prefix`
ends with `".tdz"`.

Returns:

A new memory-mapped tensordict, or a
[`TensorDictFuture`](tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) of it if
`return_early=True` and `num_threads > 1`.

Note

For an archive, the result is not locked, and in-place writes do
not update the CRC-32 checksums of the zip entries. Call
[`refresh_archive_checksums()`](tensordict.refresh_archive_checksums.html#tensordict.refresh_archive_checksums) before you give the
archive to a tool that checks them, such as `unzip`.

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

![](../../_images/sphx_glr_tensorclass_particles_thumb.png)

[Simulating particles with a tensorclass](../../tutorials/tensorclass_particles.html)

Simulating particles with a tensorclass
![](../../_images/sphx_glr_tensordict_memory_thumb.png)

[Devices and memory-mapped tensors](../../tutorials/tensordict_memory.html)

Devices and memory-mapped tensors