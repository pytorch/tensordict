# is_leaf_nontensor

tensordict.is_leaf_nontensor(*cls: [Type](https://docs.python.org/3/library/typing.html#typing.Type)*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Returns `True` if a type is a tensor or non-tensor data.

Pass it as `is_leaf` to [`keys()`](tensordict.TensorDictBase.keys.html#tensordict.TensorDictBase.keys) and the
methods that walk the nested entries, to list the non-tensor entries
([`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData), [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack))
with the tensors. A [`UnbatchedTensor`](tensordict.UnbatchedTensor.html#tensordict.UnbatchedTensor) is a leaf too.
Other tensordicts and tensorclasses are not leaves, and nor is a type that
is neither a tensor nor a tensor collection. See [Non-tensor data and unbatched entries](../../concepts/non_tensor.html#non-tensor-data).

Parameters:

**cls** ([*type*](https://docs.python.org/3/builtins/functions.html#type)) - the type of an entry, as returned by
[`entry_class()`](tensordict.TensorDictBase.entry_class.html#tensordict.TensorDictBase.entry_class).

Examples

```
>>> import torch
>>> from tensordict import TensorDict, NonTensorData, is_leaf_nontensor
>>> is_leaf_nontensor(torch.Tensor), is_leaf_nontensor(NonTensorData), is_leaf_nontensor(TensorDict)
(True, True, False)
>>> td = TensorDict(a={}, b="a string!", c=torch.zeros(()))
>>> list(td.keys(leaves_only=True, is_leaf=is_leaf_nontensor))
['b', 'c']
```

See also

[`default_is_leaf()`](tensordict.default_is_leaf.html#tensordict.default_is_leaf).