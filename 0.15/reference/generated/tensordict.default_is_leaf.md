# default_is_leaf

tensordict.default_is_leaf(*cls: [Type](https://docs.python.org/3/library/typing.html#typing.Type)*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Returns `True` if a type is a leaf: anything but a tensordict or a tensorclass.

This is the default `is_leaf` of [`keys()`](tensordict.TensorDictBase.keys.html#tensordict.TensorDictBase.keys)
and the methods that walk the nested entries. A
[`UnbatchedTensor`](tensordict.UnbatchedTensor.html#tensordict.UnbatchedTensor) is a leaf too. A
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) is not, so `leaves_only=True` skips
the non-tensor entries. See [Keys, nested keys, and locking](../../concepts/keys.html#keys).

Parameters:

**cls** ([*type*](https://docs.python.org/3/builtins/functions.html#type)) - the type of an entry, as returned by
[`entry_class()`](tensordict.TensorDictBase.entry_class.html#tensordict.TensorDictBase.entry_class).

Examples

```
>>> import torch
>>> from tensordict import TensorDict, NonTensorData, default_is_leaf
>>> default_is_leaf(torch.Tensor), default_is_leaf(NonTensorData), default_is_leaf(TensorDict)
(True, False, False)
>>> td = TensorDict(a={}, b="a string!", c=torch.zeros(()))
>>> list(td.keys(leaves_only=True, is_leaf=default_is_leaf))
['c']
```

See also

[`is_leaf_nontensor()`](tensordict.is_leaf_nontensor.html#tensordict.is_leaf_nontensor).