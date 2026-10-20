# group_by

tensordict.group_by(*tensordict: T*, *key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)], [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[T, ...]]

Splits a tensordict into groups of rows that share the value of an entry.

The rows are taken along the first batch dimension. `groups[i]` holds
the rows whose entry `key` equals `values[i]`, in their original
order, as `tensordict[tensordict[key] == values[i]]` would for a
one-dimensional entry. An entry with more dimensions than one value per
row is compared row by row, as [`torch.unique()`](https://docs.pytorch.org/docs/stable/generated/torch.unique.html#torch.unique) with `dim=0` does.

The rows are gathered once and split into views of the gathered
tensordict, instead of indexing it with one mask per group.

The number of groups depends on the data, and the group sizes are read
on the host (`.tolist()`), which synchronizes with the device: this
function is meant for eager mode, not for compiled code.

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**tensorclass instance*) - the tensordict to
split. It needs at least one batch dimension.
- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the entry to group the rows by. A tensor entry, or a
non-tensor entry (for example a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of strings) whose values are
hashable and can be sorted.

Returns:

A `(values, groups)` tuple. `values` holds the sorted unique values
of the entry: a tensor for a tensor entry, a list for a non-tensor
entry (of tuples if each row holds several values). `groups` is a
tuple of tensordicts of the type of `tensordict`, one per value, in
the same order.

Raises:

- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `tensordict` has no batch dimension.
- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if `tensordict` has no entry `key`.
- [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if the entry is a tensordict, or a non-tensor entry whose
 values cannot be hashed or sorted.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, group_by
>>> td = TensorDict(
... uid=torch.tensor([2, 0, 2, 1, 0]),
... reward=torch.arange(5.0),
... batch_size=[5],
... )
>>> values, groups = group_by(td, "uid")
>>> values
tensor([0, 1, 2])
>>> [group["reward"] for group in groups]
[tensor([1., 4.]), tensor([3.]), tensor([0., 2.])]
```