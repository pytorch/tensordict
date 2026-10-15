# is_memmap_archive

tensordict.is_memmap_archive(*path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Checks whether `path` points to a memmap tensordict archive.

Returns `True` if `path` is a zip file containing a top-level
`meta.json` entry (the marker of the memmap format), `False`
otherwise. Directories always return `False`.