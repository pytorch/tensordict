# from_pytree

tensordict.from_pytree(*pytree*, ***, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Converts a pytree (nested lists, tuples and dicts of tensors) to a tensordict.

Lists, tuples, named tuples and dicts become nested tensordicts, and
non-tensor entries record their types, so
[`to_pytree()`](tensordict.TensorDictBase.to_pytree.html#tensordict.TensorDictBase.to_pytree) can rebuild the pytree. Same
as [`tensordict.TensorDict.from_pytree()`](tensordict.TensorDictBase.from_pytree.html#tensordict.TensorDictBase.from_pytree): see that page for the
details.

Parameters:

**pytree** (*Any*) - the pytree to convert.

Keyword Arguments:

- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result. Defaults to `None`, which gives `[]` unless
`auto_batch_size` is `True`.
- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size to
the largest leading shape that all the leaves share. Defaults to
`False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).

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