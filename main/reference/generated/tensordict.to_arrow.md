# to_arrow

tensordict.to_arrow(*td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → pyarrow.Table

Converts a tensordict with one batch dimension to an Arrow table, with one column per entry.

Requires pyarrow 15 or later. This is the reverse of
[`from_arrow()`](tensordict.from_arrow.html#tensordict.from_arrow). The entries convert as follows:

- 1-D tensors become primitive columns. The columns of contiguous numeric
tensors on the CPU view their data. Boolean tensors are copied.
- tensors with more dimensions become `arrow.fixed_shape_tensor` columns.
- jagged nested tensors (`layout=torch.jagged`) become `large_list`
columns.
- nested tensordicts become `struct` columns.
- non-tensor entries, such as a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of
strings, become columns of the type that `pyarrow.array()` infers
from their values.

A [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) with entries of different
shapes converts after
`densify(layout=torch.jagged)`.

Parameters:

**td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict. Its batch size must have one
dimension: call `td.reshape(-1)` first to flatten a batch with
more dimensions.

Returns:

A `pyarrow.Table` with `td.batch_size[0]` rows, or no rows if
`td` has no entries.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, to_arrow
>>> td = TensorDict(
... x=torch.tensor([1.0, 2.0]),
... image=torch.zeros(2, 3, 4),
... obs=TensorDict(y=torch.tensor([3, 4]), batch_size=[2]),
... batch_size=[2],
... )
>>> to_arrow(td).schema
x: float
image: extension<arrow.fixed_shape_tensor[value_type=float, shape=[3,4], permutation=[0,1]]>
obs: struct<y: int64>
 child 0, y: int64
```