# TensorDictBase.saved_path

*property*TensorDictBase.saved_path

The directory where a memory-mapped tensordict stores its files, as a [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path).

Raises:

[**AttributeError**](https://docs.python.org/3/builtins/exceptions.html#AttributeError) - if [`is_memmap()`](tensordict.TensorDictBase.is_memmap.html#tensordict.TensorDictBase.is_memmap) returns `False`, for
 example after [`unlock_()`](tensordict.TensorDictBase.unlock_.html#tensordict.TensorDictBase.unlock_).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).