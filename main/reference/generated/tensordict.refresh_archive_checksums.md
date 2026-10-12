# refresh_archive_checksums

tensordict.refresh_archive_checksums(*archive_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*) → [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

Recomputes the CRC-32 checksums of a memmap archive after in-place writes.

A tensordict loaded with `TensorDict.load_memmap(path, mode="r+")`
writes through to the archive without updating the CRC-32 that the zip
format stores for each entry. [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap)
itself never verifies checksums, so this is only needed before handing
the modified archive to tools that do (`unzip`, [`zipfile`](https://docs.python.org/3/library/zipfile.html#module-zipfile),
[`unpack_memmap()`](tensordict.unpack_memmap.html#tensordict.unpack_memmap)).

Parameters:

**archive_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - path to the archive to fix up.

Returns:

the path to the archive.