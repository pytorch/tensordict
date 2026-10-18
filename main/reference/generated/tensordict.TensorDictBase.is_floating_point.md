# TensorDictBase.is_floating_point

TensorDictBase.is_floating_point() → [bool](https://docs.python.org/3/builtins/functions.html#bool)

`True` if every tensor leaf of the tensordict, nested ones included, has a floating-point dtype.

Non-tensor data is not checked. An empty tensordict returns `True`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).