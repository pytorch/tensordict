# TensorDictBase.__getitem__

TensorDictBase.__getitem__(*index: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]*) → Any

TensorDictBase.__getitem__(*index: IndexType*) → Self

Returns the entry at a key, or the tensordict indexed along its batch dimensions.

A string or a tuple of strings reads an entry, as [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) does,
except that a missing key raises `KeyError`. A non-tensor entry
returns its content (the string, not the
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData)), and a stack of non-tensor data
returns a list. See [Keys, nested keys, and locking](../../concepts/keys.html#keys).

Any other index (an int, a slice, a tensor, a boolean mask, `...`,
`None`, or a tuple of these) applies to the batch dimensions of every
entry, as it would to a tensor of shape [`batch_size`](tensordict.TensorDictBase.batch_size.html#tensordict.TensorDictBase.batch_size), and
returns a tensordict. See [Batch size, shapes, and indexing](../../concepts/batch_size.html#batch-size). An index that runs past
the batch dimensions with ints, lists, tensors or masks indexes the
leaves of a tensordict, and grows the batch size to cover their indexed
dims, with a `FutureWarning`. Starting with TensorDict 0.17, the
default will change to raising an `IndexError`: to index the leaves,
write `td.apply(lambda x: x[:, idx])`, which keeps the batch size.

Raises:

[**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if the key is missing.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"root": torch.arange(2), ("nested", "entry"): torch.arange(2)}, [2])
>>> td["root"]
tensor([0, 1])
>>> td["nested", "entry"]
tensor([0, 1])
>>> td[:1].batch_size, td[:1]["nested", "entry"]
(torch.Size([1]), tensor([0]))
>>> td[torch.tensor([True, False])]["root"]
tensor([0])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).