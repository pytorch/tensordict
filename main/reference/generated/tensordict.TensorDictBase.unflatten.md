# TensorDictBase.unflatten

TensorDictBase.unflatten(*dim*, *unflattened_size*, ***, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns a tensordict whose batch dimension `dim` is split into several.

It works as [`torch.unflatten()`](https://docs.pytorch.org/docs/stable/generated/torch.unflatten.html#torch.unflatten) does on the batch. The leaves keep
their feature dimensions, and are views of the original leaves when the
memory layout allows it. The last new dimension keeps the name of
`dim`, and the others have no name.

It can be used as a context manager, as [`unsqueeze()`](tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze) can: when
the `with` block exits, the entries of the block are written back to
the tensordict with the inverse operation.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the batch dimension to unflatten.
- **unflattened_size** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)) - the sizes of the new
dimensions. Their product is `batch_size[dim]`, and one of
them can be `-1`.

Keyword Arguments:

**inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict keeps its
identity and keys, its leaves are replaced one by one, and the
call returns `self`. A leaf keeps its storage when the
unflatten can return a view of it. A lazy stack or a lazy view
(such as a slice of a [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict)),
or a tensordict that holds one, raises a
`NotImplementedError`: call [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) first. See
[In-place operations](../../concepts/operations.html#inplace-conventions). Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... "a": torch.arange(60).view(3, 4, 5),
... "b": torch.arange(12).view(3, 4)},
... batch_size=[3, 4])
>>> td_flat = td.flatten(0, 1)
>>> td_unflat = td_flat.unflatten(0, [3, 4])
>>> assert (td == td_unflat).all()
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).