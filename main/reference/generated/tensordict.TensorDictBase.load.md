# TensorDictBase.load

*classmethod*TensorDictBase.load(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, **args*, ***kwargs*) → Self

Loads a tensordict that [`save()`](tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save) or [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) wrote to disk.

`load` is [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap): it takes the same arguments,
including `allow_pickle`, `device` and `mode`. The leaves of the
result are memory-mapped tensors that read the files. See
[Save and load tensordicts](../../saving.html#saving) and the example of [`save()`](tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save).

Parameters:

**prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the directory, or the `".tdz"` archive
file, to read.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).