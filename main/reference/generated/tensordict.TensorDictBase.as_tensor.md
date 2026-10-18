# TensorDictBase.as_tensor

TensorDictBase.as_tensor() → Self

Returns a tensordict whose leaves are converted with their `as_tensor()` method, if they have one.

The leaves without an `as_tensor()` method, [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
and [`MemoryMappedTensor`](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor) included, are kept as they
are. No tensordict leaf type defines `as_tensor()`, so the result
holds the same leaves as the tensordict. A locked tensordict gives a
locked result.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).