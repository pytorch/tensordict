# TensorDictBase.load

TensorDictBase.load_(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, **args*, ***kwargs*)

Loads a tensordict from disk within the current tensordict.

This method is a proxy to [`load_memmap_()`](tensordict.TensorDictBase.load_memmap_.html#tensordict.TensorDictBase.load_memmap_).

Deprecated since version 0.15: Use [`load_memmap_()`](tensordict.TensorDictBase.load_memmap_.html#tensordict.TensorDictBase.load_memmap_) instead.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).