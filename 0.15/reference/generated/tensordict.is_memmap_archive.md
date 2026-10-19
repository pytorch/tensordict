# is_memmap_archive

tensordict.is_memmap_archive(*path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Whether `path` is a memory-mapped tensordict archive.

An archive is a zip file with a top-level `meta.json` entry, as
written by [`pack_memmap()`](tensordict.pack_memmap.html#tensordict.pack_memmap) or by
[`save()`](tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save) with a `".tdz"` path. A
directory, a missing path or another file gives `False`.

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the path to check.

Examples

```
>>> import tempfile
>>> from pathlib import Path
>>> import torch
>>> from tensordict import TensorDict, is_memmap_archive
>>> td = TensorDict(a=torch.zeros(3))
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.save(Path(path) / "td.tdz")
... _ = td.save(Path(path) / "td")
... print(is_memmap_archive(Path(path) / "td.tdz"), is_memmap_archive(Path(path) / "td"))
True False
```