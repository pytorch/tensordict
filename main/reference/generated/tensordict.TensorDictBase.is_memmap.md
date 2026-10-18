# TensorDictBase.is_memmap

TensorDictBase.is_memmap() → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Whether the tensordict is memory-mapped.

`True` for the tensordict of [`memmap_()`](tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_), and for the results
of [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) and [`memmap_like()`](tensordict.TensorDictBase.memmap_like.html#tensordict.TensorDictBase.memmap_like) written to a directory or
to anonymous memory (not to an archive).

A memory-mapped tensordict is locked (see [Locking and temporary changes](../../concepts/keys.html#keys-locking)), and
unlocking it makes `is_memmap` return `False`. A tensordict built
from memory-mapped tensors, or returned by [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap), is
not memory-mapped in this sense: a new entry would not be.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3))
>>> td.is_memmap(), td.memmap().is_memmap()
(False, True)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).