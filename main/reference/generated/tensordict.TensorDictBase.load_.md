# TensorDictBase.load

TensorDictBase.load_(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, **args*, ***kwargs*)

Loads a saved tensordict into the tensordict.

Calls [`load_memmap_()`](tensordict.TensorDictBase.load_memmap_.html#tensordict.TensorDictBase.load_memmap_) with the same arguments.

Deprecated since version 0.15: Use [`load_memmap_()`](tensordict.TensorDictBase.load_memmap_.html#tensordict.TensorDictBase.load_memmap_) instead. It will be removed in
TensorDict 0.17.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the directory, or the archive file, to read.
- ***args** - passed to [`load_memmap_()`](tensordict.TensorDictBase.load_memmap_.html#tensordict.TensorDictBase.load_memmap_).
- ****kwargs** - passed to [`load_memmap_()`](tensordict.TensorDictBase.load_memmap_.html#tensordict.TensorDictBase.load_memmap_).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).