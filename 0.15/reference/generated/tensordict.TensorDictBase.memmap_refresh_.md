# TensorDictBase.memmap_refresh

TensorDictBase.memmap_refresh_(***, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Refreshes the content of the memory-mapped tensordict if it has a [`saved_path`](tensordict.TensorDictBase.saved_path.html#tensordict.TensorDictBase.saved_path).

This method will raise an exception if no path is associated with it.

Parameters:

**allow_pickle** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether pickled non-tensor fields
may be loaded. See [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).