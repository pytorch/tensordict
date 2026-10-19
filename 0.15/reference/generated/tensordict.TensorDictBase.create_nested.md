# TensorDictBase.create_nested

TensorDictBase.create_nested(*key*)

Adds an empty nested tensordict at `key`, and returns `self`.

The new tensordict has the batch size, device and names of the
tensordict it is added to. An existing entry at `key` is replaced.
For a tuple key, a plain [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) keeps a
plain, unlocked `TensorDict` already on the path with its entries:
`("a", "b")` adds `"b"` to an existing `td["a"]`. A lazy stack of
plain `TensorDict` instances does this in each member, and a
tensorclass or a `TensorDictParams` follows the
rule of the tensordict it wraps. Any other entry on the path, such as a
tensor, a lazy stack, a tensorclass, a typed tensordict or a locked
tensordict, is replaced by a new empty tensordict, and so is every
level of the path when the tensordict is of any other type, such as a
view, a typed tensordict or a file.

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
>>> _ = data.create_nested(("some", "other"))
>>> list(data.keys(include_nested=True))
[('some', 'nested', 'value'), ('some', 'nested'), ('some', 'other'), 'some']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).