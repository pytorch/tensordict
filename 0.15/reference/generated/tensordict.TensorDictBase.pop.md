# TensorDictBase.pop

TensorDictBase.pop(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *default: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = NO_DEFAULT*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)

Removes an entry from the tensordict and returns its value.

A nested key removes one entry of a nested tensordict and keeps the
nested tensordict. A locked tensordict raises `RuntimeError` when
the key is present (see [`lock_()`](tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_)).

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the entry to remove.
- **default** (*Any**,**optional*) - the value returned when the key is
missing, including a nested key that runs through a tensor.
Without it, a missing key raises `KeyError` and a nested key
that runs through a tensor raises `ValueError`.

Returns:

The value of the removed entry, or `default`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b=TensorDict(c=torch.ones(3)))
>>> td.pop(("b", "c"))
tensor([1., 1., 1.])
>>> sorted(td.keys(include_nested=True))
['a', 'b']
>>> td.pop("missing", None) is None
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).