# pack_memmap

tensordict.pack_memmap(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, *archive_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, ***, *compression: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *compresslevel: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

Packs a memory-mapped tensordict directory into a single-file zip archive.

The archive entries replicate the directory layout, so the operation is
the exact inverse of [`unpack_memmap()`](tensordict.unpack_memmap.html#tensordict.unpack_memmap) (and of a plain `unzip`).
Tensor payloads are stored uncompressed by default and aligned to
64 bytes, which lets [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap)
memory-map the archive and expose every leaf as a zero-copy view.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - path to a directory previously produced by
[`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) and similar methods.
- **archive_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - path of the archive to create. Keep it
outside the source directory. It must not refer to a source file.
An existing file is replaced rather than rewritten in place: a
symlink is followed, but other hard links to the old file keep
the old contents. Overwriting requires write access to both the
existing file and its directory.

Keyword Arguments:

- **compression** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - one of `"stored"` (default),
`"deflate"`, `"bzip2"`, `"lzma"` or a
[`zipfile`](https://docs.python.org/3/library/zipfile.html#module-zipfile) compression constant. Any value other than
`"stored"` trades memory-mapped (zero-copy) loading for a
smaller file: compressed leaves are decompressed in memory when
accessed.
- **compresslevel** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - compression level forwarded to
[`zipfile`](https://docs.python.org/3/library/zipfile.html#module-zipfile).

Returns:

the path to the archive.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, pack_memmap
>>> td = TensorDict(a=torch.randn(3), b=TensorDict(c=torch.zeros(3, 2)))
>>> td = td.memmap("./saved_td")
>>> path = pack_memmap("./saved_td", "./saved_td.tdz")
>>> path
PosixPath('saved_td.tdz')
>>> td2 = TensorDict.load_memmap("./saved_td.tdz")
>>> assert (td == td2).all()
```