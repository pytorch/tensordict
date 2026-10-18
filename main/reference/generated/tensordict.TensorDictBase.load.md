# TensorDictBase.load

*classmethod*TensorDictBase.load(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, **args*, ***kwargs*) → Self

Loads a tensordict that [`save()`](tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save) or [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) wrote to disk.

`load` calls [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap), which documents the other
arguments, such as `device`, `mode`, `subpath` and
`allow_pickle`. The leaves of the result read the files lazily. See
[Save and load tensordicts](../../saving.html#saving).

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the directory, or the archive file, to read.
- ***args** - passed to [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap).
- ****kwargs** - passed to [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap).

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.arange(3), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.save(path)
... loaded = TensorDict.load(path)
... print(loaded["a"], loaded.batch_size)
MemoryMappedTensor([0, 1, 2]) torch.Size([3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).