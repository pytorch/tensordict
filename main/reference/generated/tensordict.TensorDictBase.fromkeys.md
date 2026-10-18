# TensorDictBase.fromkeys

*classmethod*TensorDictBase.fromkeys(*keys: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*, *value: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = 0*)

Creates a tensordict with the same value for every key, like [`dict.fromkeys()`](https://docs.python.org/3/builtins/stdtypes.html#dict.fromkeys).

The result is a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) with an empty batch
size. A nested key creates nested tensordicts. A Python number becomes
a tensor for each key, and a tensor is stored as it is: every key then
holds the same tensor object.

Parameters:

- **keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys of the result.
- **value** (*Any**,**optional*) - the value of every key: a tensor, a number,
or any value that the [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict)
constructor takes. Defaults to `0`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict.fromkeys(["a", ("b", "c")], torch.zeros(3))
>>> td["b", "c"]
tensor([0., 0., 0.])
>>> td["a"] is td["b", "c"]
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).