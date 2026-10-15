# TensorDictBase.create_nested

TensorDictBase.create_nested(*key*)

Creates a nested tensordict of the same shape, device and dim names as the current tensordict.

If the value already exists, it will be overwritten by this operation.
This operation is blocked in locked tensordicts.

Examples

```
>>> from tensordict import TensorDict
>>> data = TensorDict({}, [3, 4, 5])
>>> _ = data.create_nested("root")
>>> _ = data.create_nested(("some", "nested", "value"))
>>> print(data)
TensorDict(
 fields={
 root: TensorDict(
 fields={
 },
 batch_size=torch.Size([3, 4, 5]),
 device=None,
 is_shared=False),
 some: TensorDict(
 fields={
 nested: TensorDict(
 fields={
 value: TensorDict(
 fields={
 },
 batch_size=torch.Size([3, 4, 5]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3, 4, 5]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3, 4, 5]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3, 4, 5]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).