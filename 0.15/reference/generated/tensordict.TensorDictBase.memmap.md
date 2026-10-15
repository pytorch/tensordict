# TensorDictBase.memmap

TensorDictBase.memmap(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *copy_existing: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *return_early: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *share_non_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, *archive: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *compression: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Writes all tensors onto a corresponding memory-mapped Tensor in a new tensordict.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - directory prefix where the memory-mapped tensors will
be stored. The directory tree structure will mimic the tensordict's.
If `prefix` ends with `".tdz"` (or `archive=True` is
passed), a single-file archive is written instead of a
directory: a standard zip file whose entries replicate the
memmap directory layout. See `archive` below.
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
on all other workers. If the number of non_tensor leaves is high (e.g.,
sharing large stacks of non-tensor data) this may result in OOM or similar
errors. Defaults to `False`.
- **existsok** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `False`, an exception will be raised if a tensor already
exists in the same path. Defaults to `True`.
- **robust_key** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` (default), uses robust key encoding that safely
handles keys with path separators and special characters. If `False`,
uses legacy behavior (keys used as-is). If `None`, uses the default
robust behavior.
- **archive** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, `prefix` designates a
single file rather than a directory and the tensordict is
written as a memmap archive: a zip file mirroring the memmap
directory tree, with tensor payloads stored uncompressed and
aligned so that [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) can memory-map the file
and expose every leaf as a zero-copy view. If `None`
(default), archive mode is enabled when `prefix` ends with
`".tdz"`. The result of [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) on an archive
behaves like the result of [`from_consolidated()`](tensordict.TensorDictBase.from_consolidated.html#tensordict.TensorDictBase.from_consolidated): all
leaves are views into a single storage, and in-place writes do
not propagate to the file. Archives and memmap directories are
mutually convertible with [`pack_memmap()`](tensordict.pack_memmap.html#tensordict.pack_memmap) /
[`unpack_memmap()`](tensordict.unpack_memmap.html#tensordict.unpack_memmap) (or any zip tool). Note that
archives are written sequentially (single data pass) and
`num_threads` has no effect on them. An existing archive
is replaced by a new file rather than rewritten in place: a
symlink is followed, but other hard links to the old file
keep the old contents.
- **compression** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - compression for archive
entries (`"stored"`, `"deflate"`, `"bzip2"`, `"lzma"`
or a [`zipfile`](https://docs.python.org/3/library/zipfile.html#module-zipfile) constant). Defaults to `"stored"`
(uncompressed), which is what enables zero-copy loading.
Compressed archives load correctly but leaves are
decompressed in memory on access. Only valid in archive mode.

The TensorDict is then locked, meaning that any writing operations that
isn't in-place will throw an exception (eg, rename, set or remove an
entry).
Once the tensordict is unlocked, the memory-mapped attribute is turned to `False`,
because cross-process identity is not guaranteed anymore.

Returns:

A new tensordict with the tensors stored on disk if `return_early=False`,
otherwise a [`TensorDictFuture`](tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) instance.

Note

Serialising in this fashion might be slow with deeply nested tensordicts, so
it is not recommended to call this method inside a training loop.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).