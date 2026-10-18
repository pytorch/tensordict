# TensorDictBase.mod

TensorDictBase.mod(*other: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → Self

Returns a new tensordict with the remainder of every leaf divided by `other`.

`td % other` calls this method. The remainder has the sign of the
divisor, as with the `%` operator of tensors.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the divisor. A
tensordict is paired with the tensordict key by key and must
have the same keys. A tensor is aligned with the batch
dimensions of each leaf and expanded over its feature
dimensions.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).