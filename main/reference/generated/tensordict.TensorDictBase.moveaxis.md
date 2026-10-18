# TensorDictBase.moveaxis

TensorDictBase.moveaxis(*source: [int](https://docs.python.org/3/builtins/functions.html#int) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...]*, *destination: [int](https://docs.python.org/3/builtins/functions.html#int) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...]*)

Returns a tensordict with batch dimensions moved to new positions, as [`torch.movedim()`](https://docs.pytorch.org/docs/stable/generated/torch.movedim.html#torch.movedim) does.

The batch dimensions that are not moved keep their order, at the
positions that `destination` leaves free. The leaves are views of the
original leaves, and the names move with the dimensions.

It can be used as a context manager, as [`unsqueeze()`](tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze) can: when
the `with` block exits, the entries of the block are written back to
the tensordict with the inverse operation.

Parameters:

- **source** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - the batch dimensions to move. They
must be unique.
- **destination** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - the new position of each
dimension of `source`. They must be unique.

Returns:

A tensordict with the batch dimensions moved.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.zeros(3, 4, 5)}, batch_size=[3, 4])
>>> print(td.movedim(0, 1).shape)
torch.Size([4, 3])
>>> print(td.movedim((0, 1), (1, 0)).shape)
torch.Size([4, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).