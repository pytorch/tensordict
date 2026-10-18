# TensorDictBase.memmap_refresh

TensorDictBase.memmap_refresh_(***, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Reloads the tensordict from its [`saved_path`](tensordict.TensorDictBase.saved_path.html#tensordict.TensorDictBase.saved_path), and returns `self`.

Use it when another process has changed the files: added or removed
entries, or written non-tensor data. It calls [`load_memmap_()`](tensordict.TensorDictBase.load_memmap_.html#tensordict.TensorDictBase.load_memmap_)
on [`saved_path`](tensordict.TensorDictBase.saved_path.html#tensordict.TensorDictBase.saved_path), then memory-maps the tensordict there again.

Keyword Arguments:

**allow_pickle** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether non-tensor data saved with
[`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle) may be loaded. See [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap).
Defaults to `None`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the tensordict is not memory-mapped in a
 directory, for example a tensordict returned by
 [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) (for which [`is_memmap()`](tensordict.TensorDictBase.is_memmap.html#tensordict.TensorDictBase.is_memmap) returns
 `False`) or by `memmap()` without a prefix.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> with tempfile.TemporaryDirectory() as path:
... td = TensorDict(a=torch.zeros(3)).memmap(path)
... _ = TensorDict(a=torch.ones(3), b=torch.ones(3)).memmap(path)
... _ = td.memmap_refresh_()
... print(sorted(td.keys()), td["a"])
['a', 'b'] MemoryMappedTensor([1., 1., 1.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).