# unpack_memmap

tensordict.unpack_memmap(*archive_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*) → [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

Extracts a memmap tensordict archive into a memmap directory.

This is the inverse of [`pack_memmap()`](tensordict.pack_memmap.html#tensordict.pack_memmap) and is equivalent to
`unzip`: the resulting directory can be passed to
[`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) or memory-mapped into
in-place.

Parameters:

- **archive_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - path to an archive produced by
[`pack_memmap()`](tensordict.pack_memmap.html#tensordict.pack_memmap) or [`save()`](tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save).
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - directory where the content should be
extracted.

Returns:

the path to the directory.