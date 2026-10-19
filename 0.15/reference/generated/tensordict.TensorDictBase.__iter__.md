# TensorDictBase.__iter__

TensorDictBase.__iter__() → [Iterator](https://docs.python.org/3/library/typing.html#typing.Iterator)[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)]

Iterates over the first batch dimension of the tensordict.

Each item is a tensordict with the remaining batch dimensions, as
returned by `td.unbind(0)`. To iterate over the keys, use
[`keys()`](tensordict.TensorDictBase.keys.html#tensordict.TensorDictBase.keys).

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if the tensordict has no batch dimensions, as `iter()`
 does for a 0-d tensor.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).