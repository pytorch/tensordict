# from_pytree

tensordict.from_pytree(*pytree*, ***, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Converts a pytree to a TensorDict instance.

Lists, tuples, named tuples and dicts become nested tensordicts, and
non-tensor entries record their types, so
[`to_pytree()`](tensordict.TensorDictBase.to_pytree.html#tensordict.TensorDictBase.to_pytree) can rebuild the pytree. Same
as [`tensordict.TensorDict.from_pytree()`](tensordict.TensorDictBase.from_pytree.html#tensordict.TensorDictBase.from_pytree): see that page for the
details.

Parameters:

**pytree** - the pytree to convert.

Keyword Arguments:

- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the batch size of the result.
- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the batch size is the
largest leading shape that all leaves share. Defaults to
`False`.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - used with `auto_batch_size`: the
number of batch dimensions to keep.

Returns:

a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).

Examples

```
>>> import torch
>>> import tensordict
>>> td = tensordict.from_pytree(
... {"a": torch.zeros(3), "b": (torch.ones(3), torch.ones(3, 2))},
... auto_batch_size=True,
... )
>>> td.batch_size
torch.Size([3])
>>> td["b", "1"].shape
torch.Size([3, 2])
>>> type(td.to_pytree()["b"]).__name__
'tuple'
```