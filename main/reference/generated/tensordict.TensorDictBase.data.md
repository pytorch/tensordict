# TensorDictBase.data

*property*TensorDictBase.data*: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

Returns a tensordict containing the .data attributes of the leaf tensors.

The leaves of the result share storage with the leaves of `self`,
and do not require gradients. An in-place change to the result
changes `self`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).