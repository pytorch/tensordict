# refresh_archive_checksums

tensordict.refresh_archive_checksums(*archive_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*) → [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

Recomputes the CRC-32 checksums of a memory-mapped archive after in-place writes.

A tensordict loaded with `TensorDict.load_memmap(path, mode="r+")`, or
created by `memmap_like` with a `".tdz"` path, writes into the
archive without updating the CRC-32 checksum that the zip format stores
for each entry. [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) does not
check the checksums of uncompressed leaves, so call this function only
before you give the archive to a tool that checks them, such as
`unzip`, [`zipfile`](https://docs.python.org/3/library/zipfile.html#module-zipfile) or [`unpack_memmap()`](tensordict.unpack_memmap.html#tensordict.unpack_memmap). The archive is
changed in place.

Parameters:

**archive_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the archive to update.

Returns:

The path of the archive, as a [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path).

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `archive_path` is not an archive (see
 [`is_memmap_archive()`](tensordict.is_memmap_archive.html#tensordict.is_memmap_archive)).

Examples

```
>>> import tempfile
>>> import zipfile
>>> from pathlib import Path
>>> import torch
>>> from tensordict import TensorDict, refresh_archive_checksums
>>> with tempfile.TemporaryDirectory() as path:
... archive = Path(path) / "td.tdz"
... _ = TensorDict(a=torch.zeros(3)).save(archive)
... _ = TensorDict.load_memmap(archive, mode="r+")["a"].add_(1)
... print(zipfile.ZipFile(archive).testzip()) # the first bad entry
... _ = refresh_archive_checksums(archive)
... print(zipfile.ZipFile(archive).testzip())
a.memmap
None
```