# TensorDictBase.load_memmap

TensorDictBase.load_memmap_(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, ***, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Loads a saved tensordict into the tensordict, and returns `self`.

Calls [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) with `out=self`. The tensordict and its
nested tensordicts are reused, but each leaf is replaced by the loaded
tensor, so its storage is not reused. Keys that are not in the saved
data are removed. To write the data into the existing storage, use
`td.update_(TensorDict.load_memmap(prefix, device=td.device))`
instead.

If the tensordict is memory-mapped, it is unlocked for the load, then
memory-mapped again into the loaded directory, which locks it. If the
data has no directory (an archive, or `mode="r"`), the tensordict
stays unlocked and is no longer memory-mapped.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the directory, or the `".tdz"` archive
file, to read.
- **robust_key** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the key encoding used when the data
was saved. See [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap). Defaults to `True`.

Keyword Arguments:

- **allow_pickle** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether non-tensor data saved with
[`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle) may be loaded. Pickle can run arbitrary code: pass
`True` only for data you trust. Defaults to `None`, which
loads pickled data with a `FutureWarning`. Starting with
TensorDict 0.16, the default will change to `False`. See
[`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap).
- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - how the files are mapped: `"r"`, `"r+"`
or `None`. See [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap). Defaults to `None`.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> with tempfile.TemporaryDirectory() as path:
... _ = TensorDict(a=torch.ones(3)).save(path)
... td = TensorDict(a=torch.zeros(3), b=torch.zeros(3))
... out = td.load_memmap_(path)
... print(out is td, list(td.keys()), td["a"])
True ['a'] MemoryMappedTensor([1., 1., 1.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).