# TensorDictBase.popitem

*abstract*TensorDictBase.popitem() → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection]

Removes the item that was last inserted into the TensorDict.

`popitem` will only return non-nested values.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).