# TensorDictBase.create_nested

TensorDictBase.create_nested(*key*)

Adds an empty nested tensordict at `key`, and returns `self`.

The new tensordict has the batch size, device and names of the
tensordict. An existing entry at `key` is replaced. For a tuple key,
every level is created new, so an existing entry on the path, such as
`td["a"]` for `("a", "b")`, is replaced too.

Parameters:

**key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the new tensordict.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the tensordict is locked.

Examples

```
>>> from tensordict import TensorDict
>>> data = TensorDict({}, [3, 4, 5])
>>> _ = data.create_nested(("some", "nested", "value"))
>>> data["some", "nested", "value"].batch_size
torch.Size([3, 4, 5])
>>> list(data.keys(include_nested=True))
[('some', 'nested', 'value'), ('some', 'nested'), 'some']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).