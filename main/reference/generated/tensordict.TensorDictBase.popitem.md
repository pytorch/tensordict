# TensorDictBase.popitem

TensorDictBase.popitem() → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)]

Removes the last top-level entry and returns it as a `(key, value)` pair.

The last entry is the last key of [`keys()`](tensordict.TensorDictBase.keys.html#tensordict.TensorDictBase.keys), which is the last
one inserted for a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict). A nested
tensordict is returned whole.

Raises:

- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if the tensordict is empty.
- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the tensordict is locked.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b=torch.ones(3), batch_size=[3])
>>> td.popitem()
('b', tensor([1., 1., 1.]))
>>> list(td.keys())
['a']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).