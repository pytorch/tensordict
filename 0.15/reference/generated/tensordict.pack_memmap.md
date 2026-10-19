# pack_memmap

tensordict.pack_memmap(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, *archive_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, ***, *compression: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *compresslevel: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

Packs a memory-mapped tensordict directory into a single-file zip archive.

The entries of the archive are the files of the directory, so
[`unpack_memmap()`](tensordict.unpack_memmap.html#tensordict.unpack_memmap) (or `unzip`) gives the directory back. Tensor
payloads are stored uncompressed by default and aligned to 64 bytes, so
that [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) can memory-map the
archive and make each leaf a view into the mapping, without a copy.
`td.save("x.tdz")` writes the same archive from a tensordict.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - a directory written by
[`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) or a similar method.
- **archive_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the archive to create, outside
`prefix`. An existing file is replaced by a new file, which
needs write access to the file and to its directory: a symlink is
followed, but other hard links keep the old content.

Keyword Arguments:

- **compression** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the compression of the tensor
payloads: `"stored"`, `"deflate"`, `"bzip2"`, `"lzma"`, or
a [`zipfile`](https://docs.python.org/3/library/zipfile.html#module-zipfile) constant. A compressed archive is smaller, but its
leaves are decompressed in memory when they are loaded, instead of
being memory-mapped. Defaults to `None`, which stores them
uncompressed.
- **compresslevel** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the compression level, passed to
[`zipfile`](https://docs.python.org/3/library/zipfile.html#module-zipfile). Defaults to `None`, which uses the default
level of the compression method.

Returns:

The path of the archive, as a [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path).

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `prefix` has no `meta.json` file, or if
 `archive_path` is inside `prefix` or is one of its files.

Examples

```
>>> import tempfile
>>> from pathlib import Path
>>> import torch
>>> from tensordict import TensorDict, pack_memmap
>>> td = TensorDict(a=torch.zeros(3), b=TensorDict(c=torch.ones(3, 2)))
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.memmap(Path(path) / "saved")
... archive = pack_memmap(Path(path) / "saved", Path(path) / "saved.tdz")
... print(archive.name, (TensorDict.load_memmap(archive) == td).all())
saved.tdz True
```