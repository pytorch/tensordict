# TensorDictBase.load_memmap

TensorDictBase.load_memmap_(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, ***, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Loads the content of a memory-mapped tensordict within the tensordict where `load_memmap_` is called.

The tensordict and its nested containers are reused, but the leaves
are rebound to the loaded tensors, so their storage is not reused.
Keys that are missing from the saved data are removed. To write the
data into the existing storage, use
`td.update_(TensorDict.load_memmap(prefix, device=td.device))`
instead.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the directory, or the `".tdz"` archive
file, to read.
- **robust_key** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the key encoding used when the data
was saved. See [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap). Defaults to `True`.

Keyword Arguments:

- **allow_pickle** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - whether pickled non-tensor fields
may be loaded. Pickle can execute arbitrary code, so pass
`True` only for data from a trusted source and `False`
for untrusted data. As in [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap), omitting
this option loads pickle with a `FutureWarning` in 0.14
and 0.15, and the default changes to `False` in 0.16.
- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the mode of the memory maps. See
[`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap).

See [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) for more info.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).