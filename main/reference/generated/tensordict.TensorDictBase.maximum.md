# TensorDictBase.maximum

TensorDictBase.maximum(*other: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, ***, *default: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Returns a new tensordict with the element-wise maximum of every leaf and `other`.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the other operand. A
tensordict is paired with the tensordict key by key. A tensor is
aligned with the batch dimensions of each leaf and expanded over
its feature dimensions. Batch sizes that differ are broadcast
together.

Keyword Arguments:

**default** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - what to do with the keys
that only one of the two tensordicts has. `None` requires the
same keys and raises a `KeyError` otherwise.
`"intersection"` computes the keys that both have and drops
the others. A tensor stands in for the missing entry on either
side, and the result has the keys of both. Defaults to `None`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).