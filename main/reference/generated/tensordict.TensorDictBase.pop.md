# TensorDictBase.pop

TensorDictBase.pop(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *default: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = NO_DEFAULT*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection

Removes and returns a value from a tensordict.

If the value is not present and no default value is provided, a KeyError
is thrown. A locked tensordict raises `RuntimeError` (see
[`lock_()`](tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_)).

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the entry to look for. A tuple removes a nested
entry, and keeps the sub-tensordict that held it.
- **default** (*Any**,**optional*) - the value to return if the key cannot be found,
including a nested key that runs through a tensor (without a
default, that key raises a `ValueError` that names the tensor).

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict({"1": 1}, [])
>>> one = td.pop("1")
>>> assert one == 1
>>> none = td.pop("1", default=None)
>>> assert none is None
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).