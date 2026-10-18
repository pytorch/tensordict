# TensorDictBase.rot90

TensorDictBase.rot90(*k: [int](https://docs.python.org/3/builtins/functions.html#int) = 1*, *dims: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), [int](https://docs.python.org/3/builtins/functions.html#int)] = (0, 1)*)

Returns a tensordict rotated by 90 degrees in the plane of two batch dimensions.

As for [`torch.rot90()`](https://docs.pytorch.org/docs/stable/generated/torch.rot90.html#torch.rot90), the rotation goes from the first dimension
of `dims` towards the second. The tensordict needs at least 2 batch
dimensions, or a `RuntimeError` is raised. The feature dimensions are
not rotated, the names follow their dimensions, and the data is copied.

It can be used as a context manager, as [`unsqueeze()`](tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze) can: when
the `with` block exits, the entries of the block are written back to
the tensordict with the inverse operation.

Parameters:

- **k** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of rotations. Defaults to `1`.
- **dims** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the two batch dimensions of the
plane. Defaults to `(0, 1)`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
>>> print(td["a"])
tensor([[0, 1, 2],
 [3, 4, 5]])
>>> print(td.rot90()["a"])
tensor([[2, 5],
 [1, 4],
 [0, 3]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).