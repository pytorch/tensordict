# TensorDictBase.__getitem__

TensorDictBase.__getitem__(*index: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]*) → Any

TensorDictBase.__getitem__(*index: IndexType*) → Self

Indexes all tensors according to the provided index.

The index can be a (nested) key or any valid shape index given the
tensordict batch size.

If the index is a nested key and the result is a [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData)
object, the content of the non-tensor is returned.

Examples

```
>>> td = TensorDict({"root": torch.arange(2), ("nested", "entry"): torch.arange(2)}, [2])
>>> td["root"]
tensor([0, 1])
>>> td["nested", "entry"]
tensor([0, 1])
>>> td[:1]
TensorDict(
 fields={
 nested: TensorDict(
 fields={
 entry: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([1]),
 device=None,
 is_shared=False),
 root: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([1]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).