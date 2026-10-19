# unpack_memmap

tensordict.unpack_memmap(*archive_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*) → [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

Extracts a memory-mapped tensordict archive into a directory.

The inverse of [`pack_memmap()`](tensordict.pack_memmap.html#tensordict.pack_memmap), and the same as `unzip`: the
directory has one file per leaf, which
[`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) reads. Existing files in
`prefix` are overwritten.

Parameters:

- **archive_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - an archive written by [`pack_memmap()`](tensordict.pack_memmap.html#tensordict.pack_memmap)
or by [`save()`](tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save).
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the directory to extract into. It is created if
it does not exist.

Returns:

The path of the directory, as a [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path).

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `archive_path` is not an archive (see
 [`is_memmap_archive()`](tensordict.is_memmap_archive.html#tensordict.is_memmap_archive)).

Examples

```
>>> import tempfile
>>> from pathlib import Path
>>> import torch
>>> from tensordict import TensorDict, unpack_memmap
>>> td = TensorDict(a=torch.zeros(3))
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.save(Path(path) / "td.tdz")
... directory = unpack_memmap(Path(path) / "td.tdz", Path(path) / "td")
... print(sorted(p.name for p in directory.iterdir()))
['a.memmap', 'meta.json']
```